import torch
import torch.nn as nn
from .functional import cayley_transform


class AttentionGauge(nn.Module):
    """
    Applies a continuous SO(d) gauge transformation to the Attention projection boundary.
    It learns a rotation matrix R that rotates the output of W_V before it hits W_O.

    Invariance check: (X @ W_V) @ R @ R^T @ W_O == X @ W_V @ W_O
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # A is the underlying parameter that generates the orthogonal matrix R.
        # Initializing with small random values ensures it starts near the identity matrix.
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 1e-4)

    def get_rotation(self) -> torch.Tensor:
        """
        Computes the strictly orthogonal matrix R via the Cayley transform.
        """
        return cayley_transform(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates the incoming activation tensor during the training/calibration forward pass.
        Shape: [batch_size, seq_len, hidden_dim]
        """
        R = self.get_rotation()
        # Apply the rotation to the token features
        return torch.matmul(x, R)


class BlockDiagonalMLPGauge(nn.Module):
    """
    Applies an SO(block_size) block-diagonal gauge transformation to the massive
    intermediate dimension of the SwiGLU MLP blocks (e.g., 11,008 scaled down to 64-sized blocks).

    This completely avoids the O(d^3) memory/compute overhead of a dense matrix inversion.
    """

    def __init__(self, intermediate_dim: int, block_size: int = 64):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.block_size = block_size

        # Determine the number of blocks required to span the entire dimension
        if intermediate_dim % block_size != 0:
            raise ValueError(
                f"Intermediate dim {intermediate_dim} must be perfectly divisible by block size {block_size}")

        self.num_blocks = intermediate_dim // block_size

        # Define a batch of skew-symmetric generators: [num_blocks, block_size, block_size]
        self.A = nn.Parameter(torch.randn(self.num_blocks, block_size, block_size) * 1e-4)

    def get_rotation_blocks(self) -> torch.Tensor:
        """
        Computes a batch of orthogonal blocks.
        Returns shape: [num_blocks, block_size, block_size]
        """
        return cayley_transform(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates the activations along the intermediate dimension block by block.
        Input shape: [batch_size, seq_len, intermediate_dim]
        """
        shape = x.shape
        # 1. Reshape to isolate the blocks: [batch_size, seq_len, num_blocks, block_size]
        x_blocked = x.view(*shape[:-1], self.num_blocks, self.block_size)

        # 2. Get the orthogonal blocks
        R_blocks = self.get_rotation_blocks()  # [num_blocks, block_size, block_size]

        # 3. Batch matrix multiply across the blocks using einsum for speed and clarity
        # b: batch, s: seq_len, n: num_blocks, i: input_block_dim, o: output_block_dim
        rotated_blocked = torch.einsum("bsni,nio->bsno", x_blocked, R_blocks)

        # 4. Flatten back to the original intermediate dimension shape
        return rotated_blocked.reshape(*shape)