# GaugeQuant — Paper Submission TODO

## Critical (must fix before submission)

- [ ] **Run experiments and fill Section 4 (Empirical studies)**
  - Baseline BF16 perplexity (WikiText-2)
  - Baseline W4A4 perplexity (the collapse)
  - GaugeQuant W4A4 perplexity (the recovery)
  - Target model: LLaMA-2 7B or LLaMA-3 8B (not just Qwen 0.5B)
  - MMLU benchmark scores

- [ ] **Expand quantization coverage in eval**
  - Currently only quantizing `o_proj` and `down_proj`
  - Real W4A4 should quantize ALL linear layers (`q_proj`, `k_proj`, `v_proj`, `up_proj`, `gate_proj`, `o_proj`, `down_proj`)
  - Gauge only helps layers it wraps, but eval must show realistic network-wide degradation

- [ ] **Add SpinQuant comparison**
  - The paper's pitch is "we can do this during pretraining, SpinQuant can't"
  - Must show results are at least competitive with SpinQuant's published numbers on the same model
  - Without this the contribution is unanchored
  - SpinQuant reports on LLaMA-2 7B: ~6.2 PPL W4A16, ~15.5 PPL W4A4 (WikiText-2)
  - Their W4A4 uses group quantization (g128) for activations — clarify if ours is per-token or grouped
  - If we can't beat SpinQuant on raw PPL, emphasize: (1) no calibration data needed, (2) works during pretraining, (3) no STE/non-differentiable hacks

- [ ] **Decide quantization regime(s)**
  - **Do both W4A16 and W4A4.** Reasons:
    - W4A16 (weight-only) is the standard comparison point — SpinQuant, GPTQ, QuIP all report this. Reviewers expect it.
    - W4A4 is where our method *should* shine most (activation outliers are the hard problem, and our gauge explicitly targets them via the LogSumExp penalty on activations)
    - Showing both tells a complete story: competitive at W4A16, potentially superior at W4A4
  - W4A16 is easy to add: just call `apply_fake_quantization` with `a_bits=16`
  - Present as a 2×2 table: {Baseline, GaugeQuant} × {W4A16, W4A4}

- [ ] **Risk: SpinQuant might still win on W4A16**
  - SpinQuant optimizes R directly against quantization error (MSE) — it's purpose-built for the static case
  - Our proxy (LogSumExp on activations) is indirect — it suppresses outliers but doesn't directly minimize quant error
  - If SpinQuant wins W4A16 handily, our narrative must be: "we match/approach SpinQuant without needing a frozen model or calibration data, and we're strictly better positioned for W4A4 where activation outliers dominate"
  - Worst case: frame as "complementary" — GaugeQuant during pretraining, then optionally SpinQuant post-training on top

## Medium (will get flagged by reviewers)

- [ ] **Validate symmetry check (test 4)**
  - BF16 perplexity with gauge applied should equal baseline within ~0.01 PPL
  - If it doesn't → numerical bug
  - Report this explicitly in the paper as correctness validation

- [ ] **Frame training duration carefully**
  - 1024 steps × batch=1 × seq=512 ≈ 500K tokens
  - LLaMA-2 pretraining = 2T tokens
  - Frame as "continued pretraining" or "gauge calibration phase", not full pretraining
  - Consider increasing steps if compute budget allows

- [ ] **Add convergence plots**
  - Gauge loss (LogSumExp) decreasing over training steps
  - Activation kurtosis before/after rotation
  - These make the empirical section much stronger

- [ ] **Write abstract and title**
  - Currently placeholders

## Minor (nice-to-have)

- [ ] Remove blue `\textcolor{blue}{...}` TODO notes from the tex
- [ ] Fix bibliography — QuIP, ParoQuant, SpinOut are listed without proper `\cite{}` keys
- [ ] Update README.md to match actual repo structure (currently describes non-existent files)
