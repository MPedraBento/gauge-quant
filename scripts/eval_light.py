"""
Lightweight evaluation script for GPU-constrained environments.
Loads models on CPU, moves to GPU one at a time.
"""
import os
import gc
import sys
import torch
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

from gauge_quant.quantization import QuantizedLinear
from gauge_quant.modules import AttentionGauge, BlockDiagonalMLPGauge
from gauge_quant.fusion import apply_all_fusions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("GaugeQuant.Eval")

ALL_LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class OnlineMLPGaugeWrapper(torch.nn.Module):
    def __init__(self, base_layer, R):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("R", R.to(torch.float32))

    def forward(self, x, *args, **kwargs):
        x_rot = torch.matmul(x.to(torch.float32), self.R).to(x.dtype)
        return self.base_layer(x_rot, *args, **kwargs)


def apply_fake_quantization(model, w_bits, a_bits, group_size=-1):
    count = 0
    for name, module in model.named_modules():
        if any(name.endswith(proj) for proj in ALL_LINEAR_TARGETS):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, QuantizedLinear(module, w_bits=w_bits, a_bits=a_bits, group_size=group_size))
            count += 1
    return model


def apply_mlp_online_gauges(model, gauges, config, device):
    gauge_idx = 0
    for i, layer in enumerate(model.model.layers):
        if config["where"] in ["VO", "Unified"]:
            gauge_idx += 1
        if config["where"] in ["MLP", "Unified"]:
            R_blocks = gauges[gauge_idx].get_rotation_blocks().to(device)
            R = torch.block_diag(*[R_blocks[j] for j in range(R_blocks.size(0))]).to(model.dtype)
            layer.mlp.down_proj = OnlineMLPGaugeWrapper(layer.mlp.down_proj, R)
            gauge_idx += 1
    return model


def cleanup(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def ppl(model, input_ids):
    model.eval()
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=-1, help="Activation group size (e.g. 128). -1=per-token.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda"
    model_id = config["model_id"]

    # Tokenize eval data
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Loading WikiText-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join([t for t in ds["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).input_ids.to(device)

    # Get model dims from config (no GPU needed)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    num_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    intermediate_size = model_config.intermediate_size
    num_layers = model_config.num_hidden_layers

    # Load gauges (keep on CPU, they're small)
    gauges = torch.nn.ModuleList()
    for _ in range(num_layers):
        if config["where"] in ["VO", "Unified"]:
            gauges.append(AttentionGauge(num_kv_heads, head_dim))
        if config["where"] in ["MLP", "Unified"]:
            gauges.append(BlockDiagonalMLPGauge(intermediate_size, block_size=64))
    gauges.load_state_dict(torch.load(os.path.join(args.checkpoint, "gauges.pt"), map_location="cpu"))

    # Regimes
    regimes = [(args.w_bits, args.a_bits)]
    if args.a_bits == 4:
        regimes.append((args.w_bits, 16))

    # Helpers — load on CPU, move to GPU
    def load_base():
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        return m.to(device)

    def load_ft():
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, args.checkpoint).merge_and_unload()
        return merged.to(device)

    # === Evals ===

    logger.info("=== Baseline @ BF16 ===")
    m = load_base()
    ppl_base_bf16 = ppl(m, input_ids)
    logger.info(f"  {ppl_base_bf16:.2f}")
    cleanup(m)

    logger.info("=== Control (FT) @ BF16 ===")
    m = load_ft()
    ppl_ft_bf16 = ppl(m, input_ids)
    logger.info(f"  {ppl_ft_bf16:.2f}")
    cleanup(m)

    results = {}
    for w_bits, a_bits in regimes:
        tag = f"W{w_bits}A{a_bits}"

        logger.info(f"=== Baseline @ {tag} ===")
        m = load_base()
        m = apply_fake_quantization(m, w_bits, a_bits, group_size=args.group_size)
        r_base = ppl(m, input_ids)
        logger.info(f"  {r_base:.2f}")
        cleanup(m)

        logger.info(f"=== Control (FT) @ {tag} ===")
        m = load_ft()
        m = apply_fake_quantization(m, w_bits, a_bits, group_size=args.group_size)
        r_control = ppl(m, input_ids)
        logger.info(f"  {r_control:.2f}")
        cleanup(m)

        logger.info(f"=== GaugeQuant @ {tag} ===")
        m = load_ft()
        gauges.to(device)
        m = apply_all_fusions(m, gauges, config)
        m = apply_fake_quantization(m, w_bits, a_bits, group_size=args.group_size)
        m = apply_mlp_online_gauges(m, gauges, config, device)
        r_gauge = ppl(m, input_ids)
        logger.info(f"  {r_gauge:.2f}")
        gauges.cpu()
        cleanup(m)

        results[tag] = {"baseline": r_base, "control": r_control, "gauge": r_gauge}

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS (WikiText-2 PPL, per-token symmetric quantization)")
    print("=" * 60)
    print(f"  Baseline (HF) @ BF16    : {ppl_base_bf16:.2f}")
    print(f"  Control  (FT) @ BF16    : {ppl_ft_bf16:.2f}")
    print("-" * 60)
    for tag, r in results.items():
        print(f"  [{tag}] Baseline         : {r['baseline']:.2f}")
        print(f"  [{tag}] Control (FT)     : {r['control']:.2f}")
        print(f"  [{tag}] GaugeQuant       : {r['gauge']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
