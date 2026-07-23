import torch
import torch.nn as nn


def symmetric_fake_quantize(x: torch.Tensor, bits: int, dim: int = -1, group_size: int = -1) -> torch.Tensor:
    """
    Applies symmetric Min-Max fake quantization with optional group quantization.

    Args:
        x: The tensor to quantize (weights or activations).
        bits: The target precision (e.g., 4 or 8).
        dim: The dimension to compute the scale over.
        group_size: If > 0, splits dim into groups of this size and computes
                    a separate scale per group. E.g., group_size=128 (g128).
                    If -1, uses the full dimension (per-token/per-channel).
    """
    if bits >= 16:
        return x

    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))

    if group_size > 0 and x.shape[dim] > group_size:
        # Group quantization: reshape, quantize per group, reshape back
        orig_shape = x.shape
        dim_size = orig_shape[dim]

        # Pad if not evenly divisible
        remainder = dim_size % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            pad_shape = list(orig_shape)
            pad_shape[dim] = pad_size
            x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)

        # Reshape to isolate groups
        new_shape = list(x.shape)
        new_shape[dim] = x.shape[dim] // group_size
        new_shape.insert(dim + 1 if dim >= 0 else len(new_shape), group_size)
        x_grouped = x.reshape(new_shape)

        # Compute scale per group (over the group_size dimension)
        group_dim = dim + 1 if dim >= 0 else -1
        max_val = x_grouped.abs().amax(dim=group_dim, keepdim=True)
        max_val = torch.clamp(max_val, min=1e-8)
        scale = max_val / q_max

        # Quantize
        x_scaled = x_grouped / scale
        x_rounded = torch.clamp(torch.round(x_scaled), q_min, q_max)
        x_q = (x_rounded - x_scaled).detach() + x_scaled
        x_q = x_q * scale

        # Reshape back
        x_q = x_q.reshape(x.shape)

        # Remove padding
        if remainder != 0:
            x_q = x_q.narrow(dim, 0, dim_size)

        return x_q
    else:
        # Per-token / per-channel quantization (original behavior)
        max_val = x.abs().amax(dim=dim, keepdim=True)
        max_val = torch.clamp(max_val, min=1e-8)
        scale = max_val / q_max

        x_scaled = x / scale
        x_rounded = torch.clamp(torch.round(x_scaled), q_min, q_max)
        x_q = (x_rounded - x_scaled).detach() + x_scaled

        return x_q * scale


class QuantizedLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that simulates W_bits / A_bits quantization
    during the forward pass.

    Supports both per-token and group quantization for activations.
    Weights are always quantized per-channel (per output row).
    """

    def __init__(self, base_layer: nn.Linear, w_bits: int = 4, a_bits: int = 4, group_size: int = -1):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.group_size = group_size

        self.weight = base_layer.weight
        self.bias = base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations (per-token or grouped along hidden dim)
        x_q = symmetric_fake_quantize(x, bits=self.a_bits, dim=-1, group_size=self.group_size)

        # Quantize weights (always per-channel, no grouping)
        w_q = symmetric_fake_quantize(self.weight, bits=self.w_bits, dim=1)

        return nn.functional.linear(x_q, w_q, self.bias)
