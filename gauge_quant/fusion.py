import torch
import logging

logger = logging.getLogger("GaugeQuant.Fusion")


@torch.no_grad()
def fuse_attention_gauge(model, gauge, layer_idx):
    """
    Fully fuses per-KV-head rotation into v_proj and o_proj — zero runtime cost.

    v_proj output: [num_kv_heads * head_dim] → rotate each KV head's output by R_k
    o_proj input:  [num_heads * head_dim]    → absorb R_k^T for each query head group

    W_V_new = R_kv @ W_V  (R_kv is block-diag [num_kv_heads, head_dim, head_dim])
    W_O_new = W_O @ R_full (R_full repeats each R_k for its query head group)
    """
    attn = model.model.layers[layer_idx].self_attn
    num_heads = model.config.num_attention_heads
    R_blocks = gauge.get_rotation_blocks()  # float32 from Cayley

    # Fuse into v_proj: shape [num_kv_heads * head_dim, hidden_size]
    # We want v_proj output rotated by R: output_new = (x @ W_V^T + b) @ R_v
    # W_V_new = R_v^T @ W_V, b_new = b @ R_v (bias is a row vector)
    R_v = torch.block_diag(*[R_blocks[i] for i in range(gauge.num_kv_heads)])
    W_v = attn.v_proj.weight.data.float()
    attn.v_proj.weight.data = torch.matmul(R_v.t(), W_v).to(attn.v_proj.weight.dtype)
    if attn.v_proj.bias is not None:
        b_v = attn.v_proj.bias.data.float()
        attn.v_proj.bias.data = torch.matmul(b_v, R_v).to(attn.v_proj.bias.dtype)

    # Fuse into o_proj: shape [hidden_size, num_heads * head_dim]
    # Right-multiply by full R (each KV rotation repeated for its query head group)
    R_full = gauge.get_full_rotation(num_heads)
    W_o = attn.o_proj.weight.data.float()
    attn.o_proj.weight.data = torch.matmul(W_o, R_full).to(attn.o_proj.weight.dtype)


@torch.no_grad()
def fuse_mlp_gauge(model, gauge, layer_idx):
    """Fuses R into down_proj input space. Online H@R still needed at inference."""
    R_blocks = gauge.get_rotation_blocks()  # already float32 from Cayley
    R = torch.block_diag(*[R_blocks[i] for i in range(R_blocks.size(0))])
    layer = model.model.layers[layer_idx].mlp
    # Compute in float32 to preserve orthogonality, then cast back
    W = layer.down_proj.weight.data.float()
    layer.down_proj.weight.data = torch.matmul(W, R).to(layer.down_proj.weight.dtype)


def apply_all_fusions(model, gauges, config):
    """Fuses all gauges into model weights."""
    logger.info("Fusing orthogonal gauges into model weights...")
    gauge_idx = 0
    for i in range(len(model.model.layers)):
        if config["where"] in ["VO", "Unified"]:
            fuse_attention_gauge(model, gauges[gauge_idx], i)
            gauge_idx += 1
        if config["where"] in ["MLP", "Unified"]:
            fuse_mlp_gauge(model, gauges[gauge_idx], i)
            gauge_idx += 1
    logger.info("Fusion complete.")
    return model
