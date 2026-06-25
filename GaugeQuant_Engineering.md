# GaugeQuant: Engineering Architecture & Empirical Validation

## 1. Repository Architecture Map
- `main.py` : The CLI entry point for routing `train` and `eval` commands.
- `configs/` : YAML definitions for model hyperparameters, target layers, and modality.
- `gauge_quant/`
  - `modules.py` : Contains the $SO(d)$ logic, Cayley Transform, and `BlockDiagonalMLPGauge`.
  - `functional.py` : LogSumExp proxy loss definitions.
  - `fusion.py` : Mathematical weight rotation logic ($W_{new} = R^T W$).
  - `quantization.py` : Min-Max W4A4/W8A8 fake-quantization hardware simulator (`QuantizedLinear`).
- `scripts/`
  - `run_train.py` : The dual-graph optimization loop over streaming C4 data.
  - `run_eval.py` : The 5-way scientific ablation script utilizing real WikiText-2 streams.

## 2. The Architectural Diagram (Dual-Graph Topology)
```text
                  [ Input Activations: X ]
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
     ( X.detach() )                    [ Base Layer Pass ]
            |                                 |
    [ Rotate: X * R ]                  [ Predicted Tokens ]
            |                                 |
            v                                 v
    [ LogSumExp Loss ]                [ Cross-Entropy Loss ]
            |                                 |
            v                                 v
      Updates ONLY R                   Updates ONLY W_base
  (Manifold Optimization)             (Linguistic Retention)