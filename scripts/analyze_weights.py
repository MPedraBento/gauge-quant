"""
Weight statistics analysis: per-row norms, kurtosis, max-abs before/after rotation fusion.

Addresses: "Can you provide an analysis of how weight distributions change after
rotation fusion (e.g., per-row/column norms, kurtosis)?"

Usage:
    uv run python scripts/analyze_weights.py \
        --config configs/qwen_0.5b_unified_wr.yaml \
        --checkpoint checkpoints/run_004_qwen_0.5b_unified_wr \
        --output_dir results/weight_analysis
"""
import os
import sys
import argparse
import yaml
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel

from gauge_quant.modules import AttentionGauge, BlockDiagonalMLPGauge
from gauge_quant.fusion import apply_all_fusions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GaugeQuant.WeightAnalysis")

ALL_LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_gauges(checkpoint, config):
    model_config = AutoConfig.from_pretrained(config["model_id"])
    num_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    intermediate_size = model_config.intermediate_size
    num_layers = model_config.num_hidden_layers

    gauges = torch.nn.ModuleList()
    for _ in range(num_layers):
        if config["where"] in ["VO", "Unified"]:
            gauges.append(AttentionGauge(num_kv_heads, head_dim))
        if config["where"] in ["MLP", "Unified"]:
            gauges.append(BlockDiagonalMLPGauge(intermediate_size, block_size=64))

    gauges.load_state_dict(torch.load(os.path.join(checkpoint, "gauges.pt"), map_location="cpu"))
    return gauges


def compute_weight_stats(model, label=""):
    """Computes per-layer weight statistics for all linear projection layers."""
    stats = {}
    for name, module in model.named_modules():
        if any(name.endswith(proj) for proj in ALL_LINEAR_TARGETS):
            W = module.weight.data.float()
            # Per-row statistics (each row = one output channel, quantized together)
            row_norms = W.norm(dim=1)
            row_max_abs = W.abs().max(dim=1).values
            # Kurtosis: excess kurtosis per row
            # kurtosis = E[(x - mu)^4] / (E[(x - mu)^2])^2 - 3
            mu = W.mean(dim=1, keepdim=True)
            centered = W - mu
            var = (centered ** 2).mean(dim=1)
            kurt = (centered ** 4).mean(dim=1) / (var ** 2 + 1e-10) - 3.0

            stats[name] = {
                "row_norm_mean": row_norms.mean().item(),
                "row_norm_std": row_norms.std().item(),
                "row_norm_max": row_norms.max().item(),
                "row_max_abs_mean": row_max_abs.mean().item(),
                "row_max_abs_max": row_max_abs.max().item(),
                "kurtosis_mean": kurt.mean().item(),
                "kurtosis_std": kurt.std().item(),
                "kurtosis_max": kurt.max().item(),
                "weight_range": (W.max() - W.min()).item(),
                "shape": list(W.shape),
            }
    return stats


def compute_global_summary(stats):
    """Summarizes stats across all layers."""
    all_kurt = [s["kurtosis_mean"] for s in stats.values()]
    all_max_abs = [s["row_max_abs_mean"] for s in stats.values()]
    all_norms = [s["row_norm_mean"] for s in stats.values()]

    return {
        "avg_kurtosis": np.mean(all_kurt),
        "max_kurtosis": np.max(all_kurt),
        "avg_row_max_abs": np.mean(all_max_abs),
        "max_row_max_abs": np.max(all_max_abs),
        "avg_row_norm": np.mean(all_norms),
    }


def print_comparison(stats_before, stats_after, summary_before, summary_after):
    """Prints a comparison table."""
    print("\n" + "=" * 90)
    print("WEIGHT STATISTICS: BEFORE vs AFTER ROTATION FUSION")
    print("=" * 90)

    print(f"\n{'Global Summary':<30} {'Before':>12} {'After':>12} {'Δ':>10}")
    print("-" * 66)
    for key in summary_before:
        b = summary_before[key]
        a = summary_after[key]
        delta = a - b
        pct = (delta / abs(b) * 100) if abs(b) > 1e-8 else 0
        print(f"  {key:<28} {b:>10.4f}   {a:>10.4f}   {pct:>+7.1f}%")

    # Per-layer comparison for key metric (kurtosis)
    print(f"\n{'Layer (down_proj / v_proj)':<50} {'Kurt Before':>12} {'Kurt After':>12} {'Δ%':>8}")
    print("-" * 85)

    # Focus on the layers most affected by rotation (v_proj and down_proj)
    target_layers = [k for k in stats_before if k.endswith("v_proj") or k.endswith("down_proj")]
    for layer in sorted(target_layers)[:20]:  # First 20
        if layer in stats_before and layer in stats_after:
            kb = stats_before[layer]["kurtosis_mean"]
            ka = stats_after[layer]["kurtosis_mean"]
            pct = ((ka - kb) / abs(kb) * 100) if abs(kb) > 1e-8 else 0
            short_name = layer.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".mlp.", ".mlp.")
            print(f"  {short_name:<48} {kb:>10.2f}   {ka:>10.2f}   {pct:>+6.1f}%")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Weight distribution analysis before/after gauge fusion")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint with gauges")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint, "weight_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    model_id = config["model_id"]
    device = "cpu"  # Statistics don't need GPU

    # ─── Load model (before fusion) ─────────────────────────────────────────
    logger.info(f"Loading {model_id}...")
    if config.get("modality") == "LoRA":
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, args.checkpoint).merge_and_unload()
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float32)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    logger.info("Computing weight statistics BEFORE rotation fusion...")
    stats_before = compute_weight_stats(model, "before")
    summary_before = compute_global_summary(stats_before)

    # ─── Apply gauge fusion ──────────────────────────────────────────────────
    logger.info("Fusing gauges...")
    gauges = load_gauges(args.checkpoint, config)
    model = apply_all_fusions(model, gauges, config)

    logger.info("Computing weight statistics AFTER rotation fusion...")
    stats_after = compute_weight_stats(model, "after")
    summary_after = compute_global_summary(stats_after)

    # ─── Print and save ──────────────────────────────────────────────────────
    print_comparison(stats_before, stats_after, summary_before, summary_after)

    # Save detailed results
    import json
    results = {
        "model_id": model_id,
        "config": config,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "per_layer_before": stats_before,
        "per_layer_after": stats_after,
    }

    out_path = os.path.join(args.output_dir, "weight_stats.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {out_path}")

    # ─── Save per-layer CSV for plotting ─────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "weight_stats.csv")
    with open(csv_path, "w") as f:
        f.write("layer,metric,before,after,delta_pct\n")
        for layer in stats_before:
            if layer in stats_after:
                for metric in ["kurtosis_mean", "row_max_abs_mean", "row_norm_mean"]:
                    b = stats_before[layer][metric]
                    a = stats_after[layer][metric]
                    pct = ((a - b) / abs(b) * 100) if abs(b) > 1e-8 else 0
                    f.write(f"{layer},{metric},{b:.6f},{a:.6f},{pct:.2f}\n")

    logger.info(f"Per-layer CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
