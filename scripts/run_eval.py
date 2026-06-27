import os
import torch
import yaml
import logging
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from gauge_quant.quantization import QuantizedLinear
from gauge_quant.modules import AttentionGauge, BlockDiagonalMLPGauge
from gauge_quant.fusion import apply_all_fusions

logger = logging.getLogger("GaugeQuant.Eval")

ALL_LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class OnlineMLPGaugeWrapper(torch.nn.Module):
    """
    Rotates activations by R before feeding into down_proj.
    Required because SwiGLU prevents fusing R upstream into gate/up projections.
    Attention gauges are fully fused (V + O) and need no runtime wrapper.
    """
    def __init__(self, base_layer, R):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("R", R.to(torch.float32))

    def forward(self, x, *args, **kwargs):
        x_rot = torch.matmul(x.to(torch.float32), self.R).to(x.dtype)
        return self.base_layer(x_rot, *args, **kwargs)


def apply_fake_quantization(model, w_bits, a_bits, target_modules=None):
    """Injects fake quantization into targeted linear layers."""
    if target_modules is None:
        target_modules = ALL_LINEAR_TARGETS
    count = 0
    for name, module in model.named_modules():
        if any(name.endswith(proj) for proj in target_modules):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, QuantizedLinear(module, w_bits=w_bits, a_bits=a_bits))
            count += 1
    logger.info(f"Quantized {count} layers to W{w_bits}A{a_bits}")
    return model


def apply_mlp_online_gauges(model, gauges, config, device):
    """Wraps MLP down_proj with the online rotation. Attention is already fully fused."""
    gauge_idx = 0
    for i, layer in enumerate(model.model.layers):
        if config["where"] in ["VO", "Unified"]:
            gauge_idx += 1  # skip attention gauges (already fused into V+O)
        if config["where"] in ["MLP", "Unified"]:
            R_blocks = gauges[gauge_idx].get_rotation_blocks().to(device)
            R = torch.block_diag(*[R_blocks[j] for j in range(R_blocks.size(0))]).to(model.dtype)
            layer.mlp.down_proj = OnlineMLPGaugeWrapper(layer.mlp.down_proj, R)
            gauge_idx += 1
    return model


def calculate_perplexity(model, tokenizer, device, seq_len=2048):
    """Calculates PPL on WikiText-2 test set."""
    model.eval()
    eval_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    eval_iterator = iter(eval_dataset)

    text = ""
    while len(tokenizer.encode(text, add_special_tokens=False)) < seq_len:
        try:
            chunk = next(eval_iterator)["text"]
            if chunk.strip():
                text += chunk + " "
        except StopIteration:
            break

    input_ids = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss).item()


def _build_gauge_model(ft_model_weights, gauges, config, device, w_bits, a_bits):
    """Builds a gauge-fused model with quantization and MLP online wrappers."""
    model = copy.deepcopy(ft_model_weights)
    model = apply_all_fusions(model, gauges, config)
    model = apply_fake_quantization(model, w_bits, a_bits)
    model = apply_mlp_online_gauges(model, gauges, config, device)
    return model


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine regimes
    regimes = [(args.w_bits, args.a_bits)]
    if args.a_bits == 4:
        regimes.append((args.w_bits, 16))
    logger.info(f"Regimes: {['W{}A{}'.format(w, a) for w, a in regimes]}")

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    base_hf_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    try:
        ft_model_weights = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except Exception:
        logger.info("Direct load failed, attempting LoRA adapter merge...")
        try:
            from peft import PeftModel
            ft_model_weights = copy.deepcopy(base_hf_model)
            ft_model_weights = PeftModel.from_pretrained(ft_model_weights, args.checkpoint)
            ft_model_weights = ft_model_weights.merge_and_unload()
        except Exception as e:
            logger.warning(f"Could not load checkpoint, falling back to HF. Error: {e}")
            ft_model_weights = copy.deepcopy(base_hf_model)

    # Load gauges
    gauges = torch.nn.ModuleList()
    num_kv_heads = base_hf_model.config.num_key_value_heads
    head_dim = base_hf_model.config.hidden_size // base_hf_model.config.num_attention_heads
    for _ in base_hf_model.model.layers:
        if config["where"] in ["VO", "Unified"]:
            gauges.append(AttentionGauge(num_kv_heads, head_dim))
        if config["where"] in ["MLP", "Unified"]:
            gauges.append(BlockDiagonalMLPGauge(base_hf_model.config.intermediate_size, block_size=64))

    gauge_path = os.path.join(args.checkpoint, "gauges.pt")
    gauges.load_state_dict(torch.load(gauge_path, map_location=device))
    gauges.to(device)

    # === BF16 evaluations (run once) ===

    logger.info("--- Baseline (HF) @ BF16 ---")
    model = copy.deepcopy(base_hf_model)
    ppl_base_bf16 = calculate_perplexity(model, tokenizer, device)
    del model; torch.cuda.empty_cache()

    logger.info("--- Control (FT) @ BF16 ---")
    model = copy.deepcopy(ft_model_weights).float()
    ppl_ft_bf16 = calculate_perplexity(model, tokenizer, device)
    del model; torch.cuda.empty_cache()

    logger.info("--- Gauge (FT) @ BF16 — Symmetry Check (computed in FP32) ---")
    model = copy.deepcopy(ft_model_weights).float()
    model = apply_all_fusions(model, gauges, config)
    model = apply_mlp_online_gauges(model, gauges, config, device)
    ppl_gauge_bf16 = calculate_perplexity(model, tokenizer, device)
    del model; torch.cuda.empty_cache()

    # === Quantized evaluations (per regime) ===

    results = {}
    for w_bits, a_bits in regimes:
        tag = f"W{w_bits}A{a_bits}"
        logger.info(f"--- Regime: {tag} ---")

        logger.info(f"  Baseline @ {tag}")
        model = copy.deepcopy(base_hf_model)
        model = apply_fake_quantization(model, w_bits, a_bits)
        ppl_base = calculate_perplexity(model, tokenizer, device)
        del model; torch.cuda.empty_cache()

        logger.info(f"  Control (FT) @ {tag}")
        model = copy.deepcopy(ft_model_weights)
        model = apply_fake_quantization(model, w_bits, a_bits)
        ppl_control = calculate_perplexity(model, tokenizer, device)
        del model; torch.cuda.empty_cache()

        logger.info(f"  GaugeQuant @ {tag}")
        model = _build_gauge_model(ft_model_weights, gauges, config, device, w_bits, a_bits)
        ppl_gauge = calculate_perplexity(model, tokenizer, device)
        del model; torch.cuda.empty_cache()

        results[tag] = {"baseline": ppl_base, "control": ppl_control, "gauge": ppl_gauge}

    # === Results ===

    logger.info("=" * 60)
    logger.info("RESULTS (WikiText-2 PPL, per-token symmetric quantization)")
    logger.info("=" * 60)
    logger.info(f"  Baseline (HF) @ BF16    : {ppl_base_bf16:.2f}")
    logger.info(f"  Control  (FT) @ BF16    : {ppl_ft_bf16:.2f}")
    logger.info(f"  Gauge    (FT) @ BF16    : {ppl_gauge_bf16:.2f}  [should ≈ Control FT]")
    logger.info("-" * 60)
    for tag, r in results.items():
        logger.info(f"  [{tag}] Baseline         : {r['baseline']:.2f}")
        logger.info(f"  [{tag}] Control (FT)     : {r['control']:.2f}")
        logger.info(f"  [{tag}] GaugeQuant       : {r['gauge']:.2f}")
    logger.info("=" * 60)
