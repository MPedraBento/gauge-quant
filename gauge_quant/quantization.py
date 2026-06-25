import torch
import torch.nn as nn


def symmetric_fake_quantize(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """
    Applies strict symmetric Min-Max fake quantization.
    The exact mathematical bottleneck that destroys standard LLMs.

    Args:
        x: The tensor to quantize (weights or activations).
        bits: The target precision (e.g., 4 or 8).
        dim: The dimension to compute the scale over.
             (e.g., -1 for token-wise activation quantization, or channel-wise weights).
    """
    if bits >= 16:
        return x

    # 1. Determine the quantization range for signed integers (e.g., 4-bit -> [-8, 7])
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))

    # 2. Find the absolute L_infinity outlier along the specified dimension
    # Keep dimensions so we can broadcast the scale back across the tensor
    max_val = x.abs().amax(dim=dim, keepdim=True)

    # Clamp max_val to avoid division by zero on dead channels
    max_val = torch.clamp(max_val, min=1e-8)

    # 3. Compute the step scale
    scale = max_val / q_max

    # 4. Quantize: Divide by scale, round to nearest integer, and clamp to bit-width
    # We use the Straight-Through Estimator (STE) trick: x_q = (round(x) - x).detach() + x
    # This allows gradients to safely pass through the rounding operation if we ever train through it.
    x_scaled = x / scale
    x_rounded = torch.clamp(torch.round(x_scaled), q_min, q_max)
    x_q = (x_rounded - x_scaled).detach() + x_scaled

    # 5. Dequantize: Multiply back by scale to simulate the precision loss in BF16 space
    return x_q * scale


class QuantizedLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that simulates W_bits / A_bits quantization
    during the forward pass. This is used exclusively during evaluation to measure
    the Perplexity collapse of the model.
    """

    def __init__(self, base_layer: nn.Linear, w_bits: int = 4, a_bits: int = 4):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.w_bits = w_bits
        self.a_bits = a_bits

        # We hold a reference to the original weights and biases
        self.weight = base_layer.weight
        self.bias = base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamically quantizes both the activations and the weights before
        computing the linear projection.
        """
        # 1. Quantize Activations (Token-wise: scale is computed per token across the hidden dim)
        x_q = symmetric_fake_quantize(x, bits=self.a_bits, dim=-1)

        # 2. Quantize Weights (Channel-wise: scale is computed per output channel across the input dim)
        w_q = symmetric_fake_quantize(self.weight, bits=self.w_bits, dim=1)

        # 3. Compute the linear projection using the aggressively degraded tensors
        return nn.functional.linear(x_q, w_q, self.bias)