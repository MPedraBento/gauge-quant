# GaugeQuant: Engineering Architecture

## Repository Structure

```
main.py                        CLI entry point (train / eval)
configs/
  qwen_0.5b_unified.yaml      Prototyping (any GPU)
  llama2_7b_unified.yaml       Paper target (≥24GB VRAM)
gauge_quant/
  modules.py                   AttentionGauge (SO(head_dim)^num_kv_heads)
                               BlockDiagonalMLPGauge (SO(64)^(d_ff/64))
  functional.py                Cayley transform, LogSumExp penalty
  fusion.py                    Fuses R into V+O (attention) and down_proj (MLP)
  quantization.py              Per-token/group symmetric fake quantization
scripts/
  run_train.py                 Dual-graph training loop on streaming C4
  run_eval.py                  Full ablation (needs ≥48GB VRAM for 7B)
  eval_light.py                Memory-efficient eval (loads one model at a time)
```

## Training Architecture

```
              [ Input: X ]
                   |
      +------------+------------+
      |                         |
      v                         v
[ v_proj(X) ]             [ base forward ]
      |                         |
      v                         v
[ gauge(output.detach()) ]  [ predictions ]
      |                         |
      v                         v
[ LogSumExp penalty ]     [ CE Loss ]
      |                         |
      v                         v
  Updates R only           Updates W only
```

### Attention Gauge (OutputGaugeWrapper on v_proj)
- Wraps `v_proj`: computes penalty on the V output (the boundary being rotated)
- `AttentionGauge(num_kv_heads, head_dim)`: one SO(head_dim) block per KV head
- At eval: fully fused into both `v_proj` and `o_proj` — zero inference cost

### MLP Gauge (InputGaugeWrapper on down_proj)
- Wraps `down_proj`: computes penalty on its input (the SwiGLU intermediate H)
- `BlockDiagonalMLPGauge(d_ff, block_size=64)`: block-diagonal SO(64)
- At eval: fused into `down_proj` weights + online `H @ R` rotation (~1.5% overhead)

## Fusion (Post-Training)

All fusions computed in float32 then cast to model dtype:

**Attention** (zero cost):
```
W_V_new = R_kv^T @ W_V       # rotate V output space
b_V_new = b_V @ R_kv         # rotate bias too
W_O_new = W_O @ R_full       # absorb into O input space
```

**MLP** (requires online H@R):
```
W_down_new = W_down @ R       # fuse into down_proj
# At inference: output = (H @ R) @ W_down_new^T = H @ W_down^T
```

## Quantization Regimes

| Regime | Weights | Activations | Flag |
|--------|---------|-------------|------|
| W4A16 | 4-bit per-channel | Full precision | `--a_bits 16` |
| W4A4 per-token | 4-bit per-channel | 4-bit per-token | default |
| W4A4 g128 | 4-bit per-channel | 4-bit groups of 128 | `--group_size 128` |

## Key Design Decisions

1. **Per-KV-head rotation (not full SO(hidden_size))**: Matches the actual GQA symmetry. Allows full fusion into V+O. Correct for both MHA (num_kv_heads = num_heads) and GQA.

2. **Block-diagonal MLP gauge (SO(64), not full SO(d_ff))**: Avoids O(d_ff³) inversion. Linear cost O(d_ff × block_size²). Trade-off: cannot redistribute outliers across block boundaries.

3. **Detached gauge loss**: Gradient firewall prevents the gauge objective from corrupting language modeling. Base weights evolve only via CE; gauge evolves only via LogSumExp.

4. **Float32 fusion**: bf16 mantissa (8 bits) destroys orthogonality during weight rotation. All R @ W computations happen in float32.
