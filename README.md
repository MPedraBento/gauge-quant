# GaugeQuant: Differentiable Gauge Co-Adaptation for LLMs

Learns continuous, differentiable SO(d) gauge symmetries alongside fine-tuning to suppress activation outliers and recover W4A4 quantization collapse.

## Repository Structure

```
main.py                        CLI entry point (train / eval)
configs/
  qwen_0.5b_unified.yaml      Prototyping config (fits on any GPU)
  llama2_7b_unified.yaml       Paper target (needs ≥32GB VRAM)
gauge_quant/
  modules.py                   AttentionGauge (SO(head_dim)^num_kv_heads)
                               BlockDiagonalMLPGauge (SO(64)^(d_ff/64))
  functional.py                Cayley transform, LogSumExp outlier penalty
  fusion.py                    Fuses R into V+O and down_proj post-training
  quantization.py              Per-token/group symmetric fake quantization
scripts/
  run_train.py                 Dual-graph training loop on streaming C4
  run_eval.py                  Full ablation (needs ≥48GB VRAM for 7B)
  eval_light.py                Memory-efficient eval (one model at a time, ≥32GB)
```

## Setup

```bash
# Requires Python 3.11, uses uv for dependency management
uv sync
```

For RunPod / cloud (no uv):
```bash
pip install transformers accelerate datasets peft einops sentencepiece
```

## Running Experiments

### 1. Train (gauge co-adaptation on C4)

```bash
# Qwen 0.5B — local prototyping (any GPU, ~12 min)
uv run python main.py train --config configs/qwen_0.5b_unified.yaml

# LLaMA-2 7B — paper experiments (≥32GB VRAM, ~45 min)
python main.py train --config configs/llama2_7b_unified.yaml
```

Training streams `allenai/c4` (English) and jointly optimizes:
- **Base weights** (LoRA or full FT) via Cross-Entropy on next-token prediction
- **Gauge matrices** via LogSumExp outlier penalty on detached activations

Artifacts saved to `output_dir`: model weights + `gauges.pt` + `gauge_loss.csv`.

### 2. Evaluate

Two eval scripts available:
- `run_eval.py` — full ablation with symmetry check (needs ≥48GB VRAM for 7B)
- `eval_light.py` — memory-efficient, loads one model at a time (≥32GB VRAM)

#### Qwen 0.5B (local)

```bash
# W4A4 per-token + W4A16 (both tested automatically)
uv run python main.py eval \
  --config configs/qwen_0.5b_unified.yaml \
  --checkpoint checkpoints/run_003_qwen_0.5b_unified

# W4A4 g128 (SpinQuant-comparable regime)
uv run python main.py eval \
  --config configs/qwen_0.5b_unified.yaml \
  --checkpoint checkpoints/run_003_qwen_0.5b_unified \
  --group_size 128
```

#### LLaMA-2 7B (RunPod / cloud)

```bash
# W4A4 per-token + W4A16 (both tested automatically)
python scripts/eval_light.py \
  --config configs/llama2_7b_unified.yaml \
  --checkpoint checkpoints/run_llama2_7b_unified

# W4A16 only (if the above OOMs before W4A16 finishes)
python scripts/eval_light.py \
  --config configs/llama2_7b_unified.yaml \
  --checkpoint checkpoints/run_llama2_7b_unified \
  --a_bits 16

# W4A4 g128 (SpinQuant-comparable regime)
python scripts/eval_light.py \
  --config configs/llama2_7b_unified.yaml \
  --checkpoint checkpoints/run_llama2_7b_unified \
  --group_size 128
```

### Full Experiment Matrix

| Model | Training | Eval runs |
|-------|----------|-----------|
| Qwen 0.5B | `configs/qwen_0.5b_unified.yaml` | default (W4A4+W4A16), `--group_size 128` |
| LLaMA-2 7B | `configs/llama2_7b_unified.yaml` | default (W4A4+W4A16), `--a_bits 16`, `--group_size 128` |

Each eval run produces a table with Baseline / Control (FT) / GaugeQuant for the requested regime.

### CLI Flags (eval)

| Flag | Default | Description |
|------|---------|-------------|
| `--w_bits` | 4 | Weight quantization precision |
| `--a_bits` | 4 | Activation quantization precision |
| `--group_size` | -1 | Activation group size (-1 = per-token, 128 = g128) |

### Quantization Details

- **Weights**: per-channel symmetric (scale per output channel)
- **Activations**: per-token symmetric by default (scale per token across hidden dim)
- **Group quantization**: `--group_size 128` computes a separate scale per 128-element group. Matches SpinQuant's W4A4 regime for fair comparison.
- **All 7 linear layers** quantized: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Config Reference

```yaml
model_id: "meta-llama/Llama-2-7b-hf"  # HuggingFace model ID
modality: "LoRA"                        # "LoRA" or "Full Fine-Tuning"
lora_r: 16                              # LoRA rank (ignored if Full Fine-Tuning)
where: "Unified"                        # "VO" (attention only), "MLP", or "Unified" (both)
learning_rate: 2.0e-4                   # Gauge optimizer LR
steps: 8192                             # Training steps on C4
penalty_weight: 0.1                     # λ for LogSumExp gauge loss
output_dir: "checkpoints/..."           # Where to save artifacts
```
