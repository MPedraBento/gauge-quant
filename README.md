# Gauge-Quant: Differentiable Gauge Co-Adaptation for LLMs

This repository contains a proof-of-concept for training continuous, differentiable $SO(d)$ gauge symmetries alongside LoRA adapters to organically heal Large Language Models against extreme W4A4 quantization collapse.

Unlike Post-Training Quantization (PTQ) heuristics (e.g., SpinQuant, SmoothQuant) that apply static rotations to frozen networks, this method dynamically co-adapts the network's geometric coordinate space during pre-training to geometrically smear $L_\infty$ activation outliers.

## Repository Structure

The codebase is organized into three distinct experimental phases, moving from initial prototyping on small models to full W4A4 quantization recovery on 7B parameter models.

### Phase 1: Prototyping (0.5B Model)
* **`model.py`**: The initial prototyping script. Trains a 0.5B parameter model using full fine-tuning to validate the continuous $SO(d)$ gauge symmetry math on Attention matrices. 
* **`eval_quant.py`**: Evaluates the 0.5B models (baseline vs. gauge-rotated) under quantization, demonstrating the initial proof-of-concept for outlier suppression.

### Phase 2: Attention-Only Scaling (7B Model)
* **`model_7b.py`**: Scales the method to 7-Billion parameter models (LLaMA-2). Bypasses massive VRAM requirements by combining frozen base weights, standard LoRA, and gradient checkpointing to isolate continuous gauge adjustments in the Attention $O$-Proj boundaries.
* **`eval_7b.py`**: W4A4 evaluation script for the Attention-only 7B models. Measures the isolated perplexity recovery when only the Attention geometry is co-adapted.

### Phase 3: Unified Attention + MLP Scaling (7B Model)
* **`model_7b_mlp.py`**: The unified pre-training script. Injects the Attention gauge *and* a novel $SO(64)$ block-diagonal gauge into the SwiGLU MLP boundaries. Uses aggressive spatial striding (`::64` and `::32`) on the Autograd graph to fit the massive intermediate dimension (11,008) onto a single 16GB consumer GPU.
* **`eval_7b_mlp.py`**: The ultimate W4A4 evaluation script. Dynamically applies the extracted block-diagonal MLP matrices during the forward pass just before the 4-bit quantizer, proving the network-wide recovery of outlier collapse.

## Quick Start

1.  **Train the Unified Gauge Model:**
    ```bash
    uv run model_7b_mlp.py
    ```
    *Note: This script employs aggressive spatial striding to fit on a 16GB RTX 5070 Ti. The gauge is intentionally blind to 98% of the weights to prevent OOM errors.*

2.  **Evaluate W4A4 Quantization:**
    ```bash
    uv run eval_7b_mlp.py
    ```

## Preliminary Results

These results reflect a highly constrained local run. The model was trained for just 1,024 steps on the C4 dataset. Due to local VRAM limits, the SwiGLU MLP gauges were optimized using a sparse 1.5% subsample of the weight matrix. 

Despite the gauge flying blind to 98.5% of the network, the geometric rotation successfully rescued the W4A4 quantizer from total baseline collapse.

| Model (LLaMA-2 7B) | Quantization Regime | Wikitext-2 Perplexity |
| :--- | :--- | :--- |
| Vanilla Baseline | W4A4 | 84.04 |
| **Gauge-Rotated (Ours)** | **W4A4** | **53.35** |

*(A massive 30-point perplexity recovery purely through intrinsic geometric coordinate rotation).*