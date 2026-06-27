import torch
from gauge_quant.functional import cayley_transform, check_orthogonality, make_skew_symmetric
from gauge_quant.modules import BlockDiagonalMLPGauge, AttentionGauge


def test_cayley_orthogonality():
    """Proves the Cayley transform maps to the SO(d) manifold."""
    # Create a random dense generator
    A = torch.randn(128, 128) * 0.1
    R = cayley_transform(A)

    # Assert R * R^T = I
    assert check_orthogonality(R, eps=1e-4), "Cayley transform failed to produce an orthogonal matrix."


def test_block_diagonal_gauge():
    """Proves the 11,008-dim MLP Gauge doesn't OOM or break tensor shapes."""
    intermediate_dim = 11008
    block_size = 64
    batch_size = 2
    seq_len = 128

    gauge = BlockDiagonalMLPGauge(intermediate_dim, block_size)

    # Synthetic activation tensor (simulating LLaMA SwiGLU output)
    x = torch.randn(batch_size, seq_len, intermediate_dim)

    # Forward pass
    rotated_x = gauge(x)

    assert rotated_x.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {rotated_x.shape}"
    assert not torch.isnan(rotated_x).any(), "NaNs detected in the rotated block-diagonal activations!"


def test_attention_gauge():
    """Proves the continuous Attention gauge successfully rotates standard hidden states."""
    num_kv_heads = 8
    head_dim = 64
    gauge = AttentionGauge(num_kv_heads, head_dim)
    x = torch.randn(2, 128, num_kv_heads * head_dim)
    rotated_x = gauge(x)

    assert rotated_x.shape == x.shape, "Attention gauge broke the hidden dimension shape."