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


class OnlineGaugeWrapper(torch.nn.Module):
    """Dynamically applies the orthogonal basis change in strict FP32 to prevent drift."""

    def __init__(self, base_layer, R):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("R", R.to(torch.float32))

    def forward(self, x, *args, **kwargs):
        x_rot = torch.matmul(x.to(torch.float32), self.R).to(x.dtype)
        return self.base_layer(x_rot, *args, **kwargs)


def apply_fake_quantization(model, w_bits, a_bits, target_modules=["down_proj"]):
    """Recursively injects the hardware quantization simulator into targeted layers."""
    count = 0
    for name, module in model.named_modules():
        if any(proj in name for proj in target_modules):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name)

            quantized_layer = QuantizedLinear(module, w_bits=w_bits, a_bits=a_bits)
            setattr(parent, child_name, quantized_layer)
            count += 1
    return model


def calculate_perplexity(model, tokenizer, device, seq_len=2048):
    """Calculates PPL using real linguistic data from WikiText-2."""
    model.eval()
    logger.info("Fetching real WikiText-2 evaluation data...")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
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
        loss = outputs.loss
    return torch.exp(loss).item()


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting 5-Way Scientific Ablation...")

    # Load Base & Checkpoint Models
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    base_hf_model = AutoModelForCausalLM.from_pretrained(config["model_id"], torch_dtype=torch.bfloat16,
                                                         device_map="auto")

    try:
        ft_model_weights = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16,
                                                                device_map="auto")
    except Exception as e:
        logger.warning(f"Could not load FT base weights, falling back to pure HF. Error: {e}")
        ft_model_weights = copy.deepcopy(base_hf_model)

    # Build the Empty Gauges from Disk once
    empty_gauges = torch.nn.ModuleList()
    for _ in base_hf_model.model.layers:
        if config["where"] in ["VO", "Unified"]:
            empty_gauges.append(AttentionGauge(base_hf_model.config.hidden_size))
        if config["where"] in ["MLP", "Unified"]:
            empty_gauges.append(BlockDiagonalMLPGauge(base_hf_model.config.intermediate_size, block_size=64))

    gauge_path = os.path.join(args.checkpoint, "gauges.pt")
    empty_gauges.load_state_dict(torch.load(gauge_path, map_location=device))
    empty_gauges.to(device)

    # ==========================================
    # 1. BASELINE @ BF16
    # ==========================================
    logger.info("--- 1. BASELINE @ BF16 ---")
    model_1 = copy.deepcopy(base_hf_model)
    ppl_base_bf16 = calculate_perplexity(model_1, tokenizer, device)
    del model_1;
    torch.cuda.empty_cache()

    # ==========================================
    # 2. BASELINE @ W4A4
    # ==========================================
    logger.info("--- 2. BASELINE @ W4A4 ---")
    model_2 = copy.deepcopy(base_hf_model)
    model_2 = apply_fake_quantization(model_2, args.w_bits, args.a_bits)
    ppl_base_w4a4 = calculate_perplexity(model_2, tokenizer, device)
    del model_2;
    torch.cuda.empty_cache()

    # ==========================================
    # 3. CONTROL (Fine-Tuned, No Gauges) @ W4A4
    # ==========================================
    logger.info("--- 3. CONTROL (FT) @ W4A4 ---")
    model_3 = copy.deepcopy(ft_model_weights)
    model_3 = apply_fake_quantization(model_3, args.w_bits, args.a_bits)
    ppl_control_w4a4 = calculate_perplexity(model_3, tokenizer, device)
    del model_3;
    torch.cuda.empty_cache()

    # ==========================================
    # 4. GAUGE FIXED @ BF16 (Symmetry Check)
    # ==========================================
    logger.info("--- 4. GAUGE @ BF16 (Symmetry Check) ---")
    model_4 = copy.deepcopy(ft_model_weights)
    model_4 = apply_all_fusions(model_4, empty_gauges, config)

    # Inject Online Gauges (NO QUANTIZATION)
    gauge_idx = 0
    for i, layer in enumerate(model_4.model.layers):
        if config["where"] in ["VO", "Unified"]: gauge_idx += 1
        if config["where"] in ["MLP", "Unified"]:
            R_blocks = empty_gauges[gauge_idx].get_rotation_blocks().to(device)
            R = torch.block_diag(*[R_blocks[j] for j in range(R_blocks.size(0))]).to(model_4.dtype)
            layer.mlp.down_proj = OnlineGaugeWrapper(layer.mlp.down_proj, R)
            gauge_idx += 1

    ppl_gauge_bf16 = calculate_perplexity(model_4, tokenizer, device)
    del model_4;
    torch.cuda.empty_cache()

    # ==========================================
    # 5. GAUGE @ W4A4
    # ==========================================
    logger.info("--- 5. GAUGE @ W4A4 ---")
    model_5 = copy.deepcopy(ft_model_weights)
    model_5 = apply_all_fusions(model_5, empty_gauges, config)

    # Apply Quantization FIRST
    model_5 = apply_fake_quantization(model_5, args.w_bits, args.a_bits)

    # Then Inject Online Gauges wrapped around the quantized layer
    gauge_idx = 0
    for i, layer in enumerate(model_5.model.layers):
        if config["where"] in ["VO", "Unified"]: gauge_idx += 1
        if config["where"] in ["MLP", "Unified"]:
            R_blocks = empty_gauges[gauge_idx].get_rotation_blocks().to(device)
            R = torch.block_diag(*[R_blocks[j] for j in range(R_blocks.size(0))]).to(model_5.dtype)
            layer.mlp.down_proj = OnlineGaugeWrapper(layer.mlp.down_proj, R)
            gauge_idx += 1

    ppl_gauge_w4a4 = calculate_perplexity(model_5, tokenizer, device)
    del model_5;
    torch.cuda.empty_cache()

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    logger.info(f"====== 5-WAY SCIENTIFIC ABLATION ======")
    logger.info(f"1. Baseline @ BF16        : {ppl_base_bf16:.2f}")
    logger.info(f"2. Baseline @ W4A4        : {ppl_base_w4a4:.2f}")
    logger.info(f"3. Control (FT) @ W4A4    : {ppl_control_w4a4:.2f}")
    logger.info(f"4. Gauge (FT) @ BF16      : {ppl_gauge_bf16:.2f}")
    logger.info(f"5. Gauge (FT) @ W4A4      : {ppl_gauge_w4a4:.2f}")
    logger.info(f"=======================================")