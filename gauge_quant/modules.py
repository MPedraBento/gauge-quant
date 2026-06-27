import torch
import torch.nn as nn
from .functional import cayley_transform


class AttentionGauge(nn.Module):
    """
    Block-diagonal SO(head_dim)^num_kv_heads gauge for the V→O attention boundary.

    One independent rotation per KV head group. Query heads sharing the same KV head
    share the same rotation, which allows full fusion into both v_proj and o_proj
    with zero runtime cost.
    """

    def __init__(self, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # One skew-symmetric generator per KV head
        self.A = nn.Parameter(torch.randn(num_kv_heads, head_dim, head_dim) * 1e-4)

    def get_rotation_blocks(self) -> torch.Tensor:
        """Returns [num_kv_heads, head_dim, head_dim] orthogonal blocks."""
        return cayley_transform(self.A)

    def get_full_rotation(self, num_heads: int) -> torch.Tensor:
        """
        Builds the full [hidden_dim, hidden_dim] block-diagonal rotation by repeating
        each KV-head rotation for all query heads in its group.

        num_heads: total number of query heads
        """
        R_blocks = self.get_rotation_blocks()  # [num_kv_heads, head_dim, head_dim]
        heads_per_kv = num_heads // self.num_kv_heads
        # Repeat each KV block for the query heads it serves
        blocks = []
        for kv_idx in range(self.num_kv_heads):
            for _ in range(heads_per_kv):
                blocks.append(R_blocks[kv_idx])
        return torch.block_diag(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates activations at the V→O boundary during training.
        Input x: [batch, seq, num_kv_heads * head_dim] (the V output before broadcast)
        """
        shape = x.shape
        # Reshape to [batch, seq, num_kv_heads, head_dim]
        x_blocked = x.view(*shape[:-1], self.num_kv_heads, self.head_dim)
        R_blocks = self.get_rotation_blocks().to(x.dtype)
        rotated = torch.einsum("bsni,nio->bsno", x_blocked, R_blocks)
        return rotated.reshape(*shape)


class BlockDiagonalMLPGauge(nn.Module):
    """
    Block-diagonal SO(block_size)^(d_ff/block_size) gauge for the SwiGLU→down_proj boundary.
    """

    def __init__(self, intermediate_dim: int, block_size: int = 64):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.block_size = block_size

        if intermediate_dim % block_size != 0:
            raise ValueError(
                f"Intermediate dim {intermediate_dim} must be divisible by block size {block_size}")

        self.num_blocks = intermediate_dim // block_size
        self.A = nn.Parameter(torch.randn(self.num_blocks, block_size, block_size) * 1e-4)

    def get_rotation_blocks(self) -> torch.Tensor:
        """Returns [num_blocks, block_size, block_size] orthogonal blocks."""
        return cayley_transform(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates activations at the SwiGLU→down_proj boundary during training.
        Input x: [batch, seq, intermediate_dim]
        """
        shape = x.shape
        x_blocked = x.view(*shape[:-1], self.num_blocks, self.block_size)
        R_blocks = self.get_rotation_blocks().to(x.dtype)
        rotated = torch.einsum("bsni,nio->bsno", x_blocked, R_blocks)
        return rotated.reshape(*shape)
