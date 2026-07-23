"""
Lightweight evaluation script for GPU-constrained environments.
Loads models on CPU, moves to GPU one at a time.
Runs all regimes (W4A16, W4A4 per-token, W4A4 g128) in a single invocation.
"""
import os
import gc
import sys
import torch
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Rotates activations block-wise before feeding into down_proj.
    Stores only [num_blocks, block_size, block_size] instead of dense [d_ff, d_ff]."""
    def __init__(self, base_layer, R_blocks):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("R_blocks", R_blocks.to(torch.float32))  # [num_blocks, b, b]
        self.num_blocks = R_blocks.size(0)
        self.block_size = R_blocks.size(1)

    def forward(self, x, *args, **kwargs):
        shape = x.shape
        x_blocked = x.view(*shape[:-1], self.num_blocks, self.block_size).float()
        x_rot = torch.einsum("...ni,nio->...no", x_blocked, self.R_blocks)
        x_rot = x_rot.reshape(shape).to(x.dtype)
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
            layer.mlp.down_proj = OnlineMLPGaugeWrapper(layer.mlp.down_proj, R_blocks)
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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda"
    torch.manual_seed(42)
    model_id = config["model_id"]

    # Tokenize eval data
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Loading WikiText-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join([t for t in ds["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).input_ids.to(device)

    # Get model dims
    model_config = AutoConfig.from_pretrained(model_id)
    num_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    intermediate_size = model_config.intermediate_size
    num_layers = model_config.num_hidden_layers

    # Load gauges
    gauges = torch.nn.ModuleList()
    for _ in range(num_layers):
        if config["where"] in ["VO", "Unified"]:
            gauges.append(AttentionGauge(num_kv_heads, head_dim))
        if config["where"] in ["MLP", "Unified"]:
            gauges.append(BlockDiagonalMLPGauge(intermediate_size, block_size=64))
    gauges.load_state_dict(torch.load(os.path.join(args.checkpoint, "gauges.pt"), map_location="cpu"))

    # All three regimes
    w = args.w_bits
    regimes = [
        (w, 16, -1,  f"W{w}A16"),
        (w, 4,  -1,  f"W{w}A4 per-token"),
        (w, 4,  128, f"W{w}A4 g128"),
    ]

    # Helpers
    def load_base():
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        return m.to(device)

    def load_ft():
        if config.get("modality") == "LoRA":
            base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            merged = PeftModel.from_pretrained(base, args.checkpoint).merge_and_unload()
            return merged.to(device)
        else:
            m = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
            return m.to(device)

    # === BF16 ===
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

    # === Quantized regimes ===
    results = {}
    for w_bits, a_bits, group_size, tag in regimes:
        logger.info(f"=== Baseline @ {tag} ===")
        m = load_base()
        m = apply_fake_quantization(m, w_bits, a_bits, group_size=group_size)
        r_base = ppl(m, input_ids)
        logger.info(f"  {r_base:.2f}")
        cleanup(m)

        logger.info(f"=== Control (FT) @ {tag} ===")
        m = load_ft()
        m = apply_fake_quantization(m, w_bits, a_bits, group_size=group_size)
        r_control = ppl(m, input_ids)
        logger.info(f"  {r_control:.2f}")
        cleanup(m)

        logger.info(f"=== GaugeQuant @ {tag} ===")
        m = load_ft()
        gauges.to(device)
        m = apply_all_fusions(m, gauges, config)
        m = apply_fake_quantization(m, w_bits, a_bits, group_size=group_size)
        m = apply_mlp_online_gauges(m, gauges, config, device)
        r_gauge = ppl(m, input_ids)
        logger.info(f"  {r_gauge:.2f}")
        gauges.cpu()
        cleanup(m)

        results[tag] = {"baseline": r_base, "control": r_control, "gauge": r_gauge}

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS — WikiText-2 Perplexity (↓ is better)")
    print("=" * 70)
    print(f"  {'Method':<20} {'BF16':>8}")
    print(f"  {'Baseline':<20} {ppl_base_bf16:>8.2f}")
    print(f"  {'Control (FT)':<20} {ppl_ft_bf16:>8.2f}")
    print("-" * 70)
    header = f"  {'Method':<20}"
    for tag in results:
        header += f" {tag:>15}"
    print(header)
    for method in ["baseline", "control", "gauge"]:
        label = {"baseline": "Baseline", "control": "Control (FT)", "gauge": "GaugeQuant"}[method]
        row = f"  {label:<20}"
        for tag in results:
            row += f" {results[tag][method]:>15.2f}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
