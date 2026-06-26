import torch
import logging

logger = logging.getLogger("GaugeQuant.Fusion")


@torch.no_grad()
def fuse_attention_gauge(model, gauge, layer_idx):
    """
    Fuses the SO(d) gauge R into the W_V and W_O weights.
    PyTorch Linear weights are shape [out_features, in_features].
    """
    layer = model.model.layers[layer_idx].self_attn
    v_out_dim = layer.v_proj.weight.data.shape[0]
    o_in_dim = layer.o_proj.weight.data.shape[1]

    # GQA SAFETY CHECK
    if v_out_dim != o_in_dim:
        logger.warning(
            f"Layer {layer_idx}: GQA detected (V_out={v_out_dim}, O_in={o_in_dim}). "
            f"Cannot fuse a dense SO(d) gauge through broadcasted KV heads. Skipping Attention fusion."
        )
        return

    R = gauge.get_rotation().to(model.dtype)

    # 1. Rotate the output space of W_V
    layer.v_proj.weight.data = torch.matmul(R.T, layer.v_proj.weight.data)

    # 2. Rotate the input space of W_O
    layer.o_proj.weight.data = torch.matmul(layer.o_proj.weight.data, R)


@torch.no_grad()
def fuse_mlp_gauge(model, gauge, layer_idx):
    """
    Fuses the Block-Diagonal SO(64) gauge strictly into the Down projection.
    We CANNOT fuse into Up/Gate because the SwiGLU non-linearity breaks the symmetry!
    """
    R_blocks = gauge.get_rotation_blocks().to(model.dtype)
    R = torch.block_diag(*[R_blocks[i] for i in range(R_blocks.size(0))])

    layer = model.model.layers[layer_idx].mlp

    # Strictly rotate the input space of the Down projection
    layer.down_proj.weight.data = torch.matmul(layer.down_proj.weight.data, R)


def apply_all_fusions(model, gauges, config):
    """
    Iterates through the model and cleanly fuses all loaded gauges.
    """
    logger.info("Fusing orthogonal gauges into model weights...")
    gauge_idx = 0

    for i, layer in enumerate(model.model.layers):
        if config["where"] in ["VO", "Unified"]:
            fuse_attention_gauge(model, gauges[gauge_idx], i)
            gauge_idx += 1

        if config["where"] in ["MLP", "Unified"]:
            fuse_mlp_gauge(model, gauges[gauge_idx], i)
            gauge_idx += 1

    logger.info("Fusion complete. The model is now permanently rotated and the Gauge wrappers can be discarded.")
    return model