"""
Downstream task evaluation using lm-evaluation-harness.
Memory-safe for 32GB VRAM (5090 / V100-32GB).

Fixes applied:
  - Uses HFLM wrapper directly (no deepcopy OOM crash)
  - Buffers use .detach().clone() (no non-leaf tensor deepcopy error)
  - Progressive JSON saves after each model evaluation (no lost results on crash)
  - Robust load_model with fallback if adapter_config.json is missing

Usage:
    uv run python scripts/eval_downstream.py \
        --config configs/qwen_0.5b_unified_wr.yaml \
        --checkpoint checkpoints/run_004_qwen_0.5b_unified_wr \
        --tasks "arc_easy,arc_challenge,piqa,winogrande,hellaswag" \
        --regime w4a16 \
        --baseline

Regimes:
    bf16      — no quantization (sanity check)
    w4a16     — 4-bit weights, 16-bit activations
    w4a4      — 4-bit weights, 4-bit activations (per-token)
    w4a4g128  — 4-bit weights, 4-bit activations (group 128)
"""
import os
import sys
import gc
import json
import argparse
import yaml
import logging
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

from gauge_quant.quantization import QuantizedLinear
from gauge_quant.modules import AttentionGauge, BlockDiagonalMLPGauge
from gauge_quant.fusion import apply_all_fusions

import lm_eval
from lm_eval.models.huggingface import HFLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("GaugeQuant.Downstream")

ALL_LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_TASKS = "arc_easy,arc_challenge,piqa,winogrande,hellaswag"


# ─────────────────────────────────────────────────────────────────────────────
# Model construction helpers
# ─────────────────────────────────────────────────────────────────────────────

class OnlineMLPGaugeWrapper(nn.Module):
    """Rotates activations block-wise before feeding into down_proj.
    Uses .detach().clone() for the buffer to avoid deepcopy issues."""
    def __init__(self, base_layer, R_blocks):
        super().__init__()
        self.base_layer = base_layer
        # detach + clone ensures this is a leaf tensor (safe for deepcopy)
        self.register_buffer("R_blocks", R_blocks.detach().clone().to(torch.float32))
        self.num_blocks = R_blocks.size(0)
        self.block_size = R_blocks.size(1)

    def forward(self, x, *args, **kwargs):
        shape = x.shape
        x_blocked = x.view(*shape[:-1], self.num_blocks, self.block_size).float()
        x_rot = torch.einsum("...ni,nio->...no", x_blocked, self.R_blocks)
        x_rot = x_rot.reshape(shape).to(x.dtype)
        return self.base_layer(x_rot, *args, **kwargs)


def apply_fake_quantization(model, w_bits, a_bits, group_size=-1):
    """Replaces all projection linear layers with fake-quantized versions."""
    count = 0
    for name, module in list(model.named_modules()):
        if any(name.endswith(proj) for proj in ALL_LINEAR_TARGETS):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, QuantizedLinear(module, w_bits=w_bits, a_bits=a_bits, group_size=group_size))
            count += 1
    logger.info(f"Applied W{w_bits}A{a_bits} (g={group_size}) fake quantization to {count} layers.")
    return model


def apply_mlp_online_gauges(model, gauges, config, device):
    """Wraps down_proj with online rotation for MLP gauges."""
    gauge_idx = 0
    for i, layer in enumerate(model.model.layers):
        if config["where"] in ["VO", "Unified"]:
            gauge_idx += 1
        if config["where"] in ["MLP", "Unified"]:
            R_blocks = gauges[gauge_idx].get_rotation_blocks().to(device)
            layer.mlp.down_proj = OnlineMLPGaugeWrapper(layer.mlp.down_proj, R_blocks)
            gauge_idx += 1
    return model


def load_gauges(checkpoint, config):
    """Loads gauge modules from checkpoint."""
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
            gauges.append(BlockDiagonalMLPGauge(intermediate_size, block_size=config.get("block_size", 64)))

    gauges_path = os.path.join(checkpoint, "gauges.pt")
    gauges.load_state_dict(torch.load(gauges_path, map_location="cpu"))
    logger.info(f"Loaded {len(gauges)} gauges from {gauges_path}")
    return gauges


def load_model(config, checkpoint, device):
    """Loads and merges fine-tuned model. Falls back gracefully if adapter missing."""
    model_id = config["model_id"]

    if config.get("modality") == "LoRA":
        # Try LoRA adapter first
        adapter_config_path = os.path.join(checkpoint, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            merged = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
            return merged.to(device)
        else:
            logger.warning(
                f"No adapter_config.json found in {checkpoint}. "
                f"Trying to load as standard HF weights..."
            )
            try:
                m = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
                return m.to(device)
            except Exception:
                logger.warning(f"Failed to load from checkpoint, falling back to base model: {model_id}")
                m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
                return m.to(device)
    else:
        # Full fine-tuning
        try:
            m = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
            return m.to(device)
        except Exception:
            logger.warning(f"Failed to load from checkpoint, falling back to base model: {model_id}")
            m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            return m.to(device)


def load_baseline(config, device):
    """Loads the original pretrained model (no fine-tuning, no gauges)."""
    return AutoModelForCausalLM.from_pretrained(
        config["model_id"], torch_dtype=torch.bfloat16
    ).to(device)


def cleanup(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# lm-eval integration (using HFLM wrapper to avoid deepcopy OOM)
# ─────────────────────────────────────────────────────────────────────────────

def run_lm_eval(model, tokenizer, tasks, batch_size=8, num_fewshot=0):
    """
    Runs lm-evaluation-harness on the given model using HFLM wrapper.
    This avoids the deepcopy crash that happens when passing model via model_args.
    """
    model.eval()

    # Wrap model explicitly — lm_eval won't try to deepcopy it
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=num_fewshot,
        log_samples=False,
    )

    return results


def parse_regime(regime_str):
    """Parses regime string into (w_bits, a_bits, group_size)."""
    regimes = {
        "bf16":     (16, 16, -1),
        "w4a16":    (4, 16, -1),
        "w4a4":     (4, 4, -1),
        "w4a4g128": (4, 4, 128),
        "w8a8":     (8, 8, -1),
        "w8a16":    (8, 16, -1),
    }
    if regime_str not in regimes:
        raise ValueError(f"Unknown regime '{regime_str}'. Choose from: {list(regimes.keys())}")
    return regimes[regime_str]


def save_progress(all_results, out_path):
    """Saves current results to JSON (progressive save to avoid losing work)."""
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict) and "results" in v:
            serializable[k] = v["results"]
        else:
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Progress saved to {out_path}")


def print_results_table(all_results):
    """Pretty-prints a comparison table of all evaluated configurations."""
    print("\n" + "=" * 80)
    print("DOWNSTREAM EVALUATION RESULTS — Zero-Shot Accuracy (↑ is better)")
    print("=" * 80)

    # Collect all task names
    all_tasks = set()
    for results in all_results.values():
        if isinstance(results, dict) and "results" in results:
            all_tasks.update(results["results"].keys())
    all_tasks = sorted(all_tasks)

    # Header
    header = f"  {'Configuration':<30}"
    for task in all_tasks:
        short_name = task.replace("arc_", "ARC-").replace("easy", "E").replace("challenge", "C")
        header += f" {short_name:>10}"
    header += f" {'AVG':>8}"
    print(header)
    print("-" * 80)

    # Rows
    for config_name, results in all_results.items():
        row = f"  {config_name:<30}"
        accs = []
        task_results = results.get("results", {}) if isinstance(results, dict) else {}
        for task in all_tasks:
            if task in task_results:
                task_res = task_results[task]
                # lm-eval stores acc or acc_norm depending on the task
                acc = task_res.get("acc_norm,none", task_res.get("acc,none", None))
                if acc is not None:
                    row += f" {acc*100:>9.2f}%"
                    accs.append(acc)
                else:
                    row += f" {'N/A':>10}"
            else:
                row += f" {'—':>10}"
        if accs:
            avg = sum(accs) / len(accs)
            row += f" {avg*100:>7.2f}%"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation with lm-evaluation-harness")
    parser.add_argument("--config", required=True, help="YAML config used for training")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help=f"Comma-separated lm-eval tasks (default: {DEFAULT_TASKS})")
    parser.add_argument("--regime", type=str, default="w4a16",
                        help="Quantization regime: bf16, w4a16, w4a4, w4a4g128, w8a8, w8a16")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate the unmodified baseline model for comparison")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [t.strip() for t in args.tasks.split(",")]
    w_bits, a_bits, group_size = parse_regime(args.regime)

    logger.info(f"Tasks: {tasks}")
    logger.info(f"Regime: {args.regime} (W{w_bits}A{a_bits}, g={group_size})")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Output path for progressive saves
    os.makedirs(args.checkpoint, exist_ok=True)
    out_path = os.path.join(args.checkpoint, f"downstream_{args.regime}.json")

    # ─── Baseline (optional) ────────────────────────────────────────────────
    if args.baseline:
        logger.info("=== Evaluating Baseline (no gauges) ===")
        model = load_baseline(config, device)
        if w_bits < 16 or a_bits < 16:
            model = apply_fake_quantization(model, w_bits, a_bits, group_size)

        results = run_lm_eval(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
        all_results[f"Baseline @ {args.regime}"] = results
        cleanup(model)

        # Progressive save — don't lose baseline if GaugeQuant eval crashes
        save_progress(all_results, out_path)

        # Control: fine-tuned model WITHOUT gauges (same training budget)
        logger.info("=== Evaluating Control (FT, no gauges) ===")
        model = load_model(config, args.checkpoint, device)
        if w_bits < 16 or a_bits < 16:
            model = apply_fake_quantization(model, w_bits, a_bits, group_size)

        results = run_lm_eval(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
        all_results[f"Control (FT) @ {args.regime}"] = results
        cleanup(model)

        save_progress(all_results, out_path)

    # ─── GaugeQuant ─────────────────────────────────────────────────────────
    logger.info("=== Evaluating GaugeQuant ===")
    model = load_model(config, args.checkpoint, device)
    gauges = load_gauges(args.checkpoint, config)
    gauges.to(device)

    # Fuse attention gauges into v_proj/o_proj weights
    model = apply_all_fusions(model, gauges, config)

    # Apply fake quantization
    if w_bits < 16 or a_bits < 16:
        model = apply_fake_quantization(model, w_bits, a_bits, group_size)

    # Apply online MLP gauges (rotation at runtime, can't be fully fused)
    model = apply_mlp_online_gauges(model, gauges, config, device)

    results = run_lm_eval(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
    all_results[f"GaugeQuant @ {args.regime}"] = results
    gauges.cpu()
    cleanup(model)

    # ─── Final save and print ───────────────────────────────────────────────
    save_progress(all_results, out_path)
    print_results_table(all_results)


if __name__ == "__main__":
    main()
