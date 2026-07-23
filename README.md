# GaugeQuant

Online learning of quantization-optimal bases from LLM symmetries.

## Setup

```bash
uv sync
```

## Train

```bash
# Qwen 0.5B (~12 min, any GPU)
uv run python main.py train --config configs/qwen_0.5b_unified.yaml

# LLaMA-2 7B (~100 min, ≥32GB VRAM)
uv run python main.py train --config configs/llama2_7b_unified.yaml
```

## Evaluate

Runs W4A16, W4A4 per-token, and W4A4 g128 in one go:

```bash
uv run python scripts/eval_light.py \
  --config configs/qwen_0.5b_unified.yaml \
  --checkpoint checkpoints/run_003_qwen_0.5b_unified
```

## Plot convergence

```bash
uv run python scripts/plot_convergence.py \
  --csv checkpoints/run_003_qwen_0.5b_unified/rot_loss.csv \
  --labels "Qwen-2.5 0.5B"
```

## Results (LLaMA-2 7B)

| Method | BF16 | W4A16 | W4A4 g128 | W4A4 per-token |
|--------|------|-------|-----------|----------------|
| Baseline | 4.07 | 11.16 | 8.22 | 2397 |
| Control (FT) | 4.08 | 13.22 | 9.25 | 5099 |
| **GaugeQuant** | 4.08 | **5.45** | **6.73** | 1923 |

## Config

```yaml
model_id: "meta-llama/Llama-2-7b-hf"
modality: "LoRA"
lora_r: 16
where: "Unified"
learning_rate: 2.0e-4
steps: 8192
penalty_weight: 0.1
output_dir: "checkpoints/run_llama2_7b_unified"
```
