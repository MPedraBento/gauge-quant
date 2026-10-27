"""
Microbenchmark: inference-time cost of the online MLP-boundary rotation in GaugeQuant.

The meaningful question is: how much does the online rotation H @ R add to the
*full MLP block* it lives in? The MLP block is three matmuls (gate, up, down) plus
SiLU and an elementwise product; our rotation sits once, just before down_proj.

So we measure, at LLaMA-2 7B dimensions:

  T_mlp      : full SwiGLU MLP block, no rotation           (baseline)
  T_mlp_rot  : same block WITH the online H @ R before down_proj
  overhead   : (T_mlp_rot - T_mlp) / T_mlp                  <-- the number to report

For context we also time the rotation and down_proj in isolation.

Pure PyTorch, no model/data download. Runs in seconds on any CUDA GPU.

Usage:
  python bench_mlp_overhead.py --tokens 2048 8192 --dtype float16
  python bench_mlp_overhead.py --d-ff 11008 --d-model 4096 --block 64
"""

import argparse
import time

import torch
import torch.nn.functional as F


def time_op(fn, iters: int, warmup: int, device: str) -> float:
    """Mean seconds per call, handling CUDA async + warmup correctly."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-ff", type=int, default=11008, help="MLP hidden dim (LLaMA-2 7B)")
    p.add_argument("--d-model", type=int, default=4096, help="model dim")
    p.add_argument("--block", type=int, default=64, help="gauge block size b")
    p.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    d_ff, d_model, b = args.d_ff, args.d_model, args.block
    assert d_ff % b == 0, f"d_ff={d_ff} must be divisible by block={b}"
    n_blocks = d_ff // b

    print(f"device={device} dtype={args.dtype} d_ff={d_ff} d_model={d_model} "
          f"block={b} n_blocks={n_blocks}\n")

    # SwiGLU weights + the learned rotation blocks (built once, as at inference).
    W_gate = torch.randn(d_ff, d_model, device=device, dtype=dtype)
    W_up   = torch.randn(d_ff, d_model, device=device, dtype=dtype)
    W_down = torch.randn(d_model, d_ff, device=device, dtype=dtype)
    R_blocks = torch.randn(n_blocks, b, b, device=device, dtype=dtype)  # fused gauge blocks

    def rotate(Hmid):
        # block-diagonal H @ R, exactly as BlockDiagonalMLPGauge.forward (einsum form)
        T = Hmid.shape[0]
        x = Hmid.view(T, n_blocks, b)
        z = torch.einsum("tni,nio->tno", x, R_blocks)
        return z.reshape(T, d_ff)

    header = (f"{'tokens':>8} | {'MLP block':>11} | {'MLP+rot':>11} | "
              f"{'overhead':>9} | {'(rot alone':>11} {'down_proj)':>11}")
    print(header)
    print("-" * len(header))

    for T in args.tokens:
        X = torch.randn(T, d_model, device=device, dtype=dtype)

        def mlp():
            H = F.silu(X @ W_gate.t()) * (X @ W_up.t())   # [T, d_ff]
            return H @ W_down.t()                          # [T, d_model]

        def mlp_rot():
            H = F.silu(X @ W_gate.t()) * (X @ W_up.t())
            H = rotate(H)                                  # online gauge rotation
            return H @ W_down.t()

        # context-only timings
        Hmid = torch.randn(T, d_ff, device=device, dtype=dtype)
        def rot_only():
            return rotate(Hmid)
        def down_only():
            return Hmid @ W_down.t()

        t_mlp  = time_op(mlp,      args.iters, args.warmup, device)
        t_mlpr = time_op(mlp_rot,  args.iters, args.warmup, device)
        t_rot  = time_op(rot_only, args.iters, args.warmup, device)
        t_dp   = time_op(down_only, args.iters, args.warmup, device)

        us = lambda s: s * 1e6
        overhead = 100.0 * (t_mlpr - t_mlp) / t_mlp
        print(f"{T:>8} | {us(t_mlp):>9.1f}us | {us(t_mlpr):>9.1f}us | "
              f"{overhead:>7.1f}% | {us(t_rot):>9.1f}us {us(t_dp):>9.1f}us")

    print("\n'overhead' = extra time the online rotation adds to the FULL SwiGLU MLP block,\n"
          "(MLP+rot - MLP) / MLP. This is the inference overhead to report. Measured standalone,\n"
          "so it is a ceiling: in a fully fused forward the marginal cost can only be lower.\n"
          "SpinQuant's online Hadamard is a different (cheaper) operator; that comparison stays\n"
          "analytical (O(d_ff*log d_ff) fused vs our O(d_ff*b) block-diagonal), not a latency claim.")


if __name__ == "__main__":
    main()
