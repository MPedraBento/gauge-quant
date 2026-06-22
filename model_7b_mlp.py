import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os


# =======================================================================
# 1. Native BF16 LoRA + Block Diagonal Gauge
# =======================================================================
class LoRALinear(nn.Module):
    def __init__(self, orig_linear, r=16, alpha=16):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.scaling = alpha / r

        in_dim, out_dim = orig_linear.in_features, orig_linear.out_features
        dtype, device = orig_linear.weight.dtype, orig_linear.weight.device

        self.lora_A = nn.Parameter(torch.randn(r, in_dim, dtype=dtype, device=device) * (1 / in_dim) ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, dtype=dtype, device=device))

    def get_effective_weight(self):
        return self.orig_linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        return self.orig_linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class RotatedLoRAOProj(LoRALinear):
    def __init__(self, orig_linear, lora_v_proj, num_heads, num_kv_heads, head_dim, r=16, alpha=16):
        super().__init__(orig_linear, r, alpha)
        self.lora_v_proj = lora_v_proj
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.A = nn.Parameter(torch.zeros(num_kv_heads, head_dim, head_dim, device=orig_linear.weight.device,
                                          dtype=orig_linear.weight.dtype))
        self.quant_loss = 0.0

    def get_R(self):
        skew = (self.A - self.A.transpose(-1, -2)).to(torch.float32)
        I = torch.eye(self.head_dim, device=skew.device, dtype=torch.float32).unsqueeze(0)
        R_kv = torch.matmul(I - skew, torch.linalg.inv(I + skew)).to(self.A.dtype)
        return R_kv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=0), R_kv

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        x_reshaped = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        R_q, R_kv = self.get_R()

        # Subsample Sequence by 32
        x_sub = x_reshaped.detach()[:, ::32, :, :]
        rotated_act = torch.einsum('bshd,hde->bshe', x_sub, R_q)
        act_loss = rotated_act.abs().amax(dim=-1).to(torch.float32).mean()
        del x_sub, rotated_act

        # Subsample W_o by 64 (saves ~100MB across network)
        w_o = self.orig_linear.weight.detach().view(-1, self.num_heads, self.head_dim)[::64, :, :]
        w_o_loss = torch.einsum('ihd,hde->ihe', w_o, R_q).abs().amax(dim=-1).to(torch.float32).mean()
        del w_o

        # Subsample W_v by 64
        w_v = self.lora_v_proj.orig_linear.weight.detach().view(self.num_kv_heads, self.head_dim, -1)[:, :, ::64]
        w_v_loss = torch.einsum('hde,hdi->hei', R_kv, w_v).abs().amax(dim=-1).to(torch.float32).mean()
        del w_v

        self.quant_loss = 10.0 * (act_loss + 0.5 * (w_o_loss + w_v_loss)).to(self.A.dtype)
        return super().forward(x)


class RotatedLoRADownProj(LoRALinear):
    """Novel Block-Diagonal Symmetry for SwiGLU MLP Outliers"""

    def __init__(self, orig_linear, block_size=64, r=16, alpha=16):
        super().__init__(orig_linear, r, alpha)
        self.hidden_dim = orig_linear.in_features
        self.block_size = block_size
        self.num_blocks = self.hidden_dim // block_size

        dtype, device = orig_linear.weight.dtype, orig_linear.weight.device
        self.A = nn.Parameter(torch.zeros(self.num_blocks, block_size, block_size, device=device, dtype=dtype))
        self.quant_loss = 0.0

    def get_R(self):
        skew = (self.A - self.A.transpose(-1, -2)).to(torch.float32)
        I = torch.eye(self.block_size, device=skew.device, dtype=torch.float32).unsqueeze(0)
        R_blocks = torch.matmul(I - skew, torch.linalg.inv(I + skew)).to(self.A.dtype)
        return R_blocks

    def apply_block_R(self, x, R_blocks):
        *leading_dims, _ = x.shape
        x_reshaped = x.view(*leading_dims, self.num_blocks, self.block_size)
        x_rot = torch.einsum('...nb,nbc->...nc', x_reshaped, R_blocks)
        return x_rot.reshape(*leading_dims, self.hidden_dim)

    def forward(self, x):
        R_blocks = self.get_R()

        # Subsample Sequence by 32
        x_sub = x.detach()[:, ::32, :]
        rotated_act = self.apply_block_R(x_sub, R_blocks)
        act_loss = rotated_act.abs().amax(dim=-1).to(torch.float32).mean()
        del x_sub, rotated_act

        # Subsample W_down by 64 (Saves ~2.8 GB of VRAM across the network)
        w_down = self.orig_linear.weight.detach()[::64, :]
        w_reshaped = w_down.view(w_down.shape[0], self.num_blocks, self.block_size)
        w_loss = torch.einsum('onb,nbc->onc', w_reshaped, R_blocks).abs().amax(dim=-1).to(torch.float32).mean()
        del w_down, w_reshaped

        self.quant_loss = 10.0 * (act_loss + 0.5 * w_loss).to(self.A.dtype)
        return super().forward(x)


def inject_symmetries(model, use_gauge=True):
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads

    for layer in model.model.layers:
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj)
        layer.mlp.down_proj = RotatedLoRADownProj(layer.mlp.down_proj) if use_gauge else LoRALinear(layer.mlp.down_proj)
        layer.self_attn.o_proj = RotatedLoRAOProj(layer.self_attn.o_proj, layer.self_attn.v_proj, num_heads,
                                                  num_kv_heads, head_dim) if use_gauge else LoRALinear(
            layer.self_attn.o_proj)
    return model


@torch.no_grad()
def absorb_and_extract(model, output_dir):
    """Absorbs LoRA and Attention R permanently. Extracts MLP R blocks to disk."""
    os.makedirs(output_dir, exist_ok=True)
    mlp_gauges = []

    for layer in model.model.layers:
        # 1. Merge all LoRA
        for proj in [layer.self_attn.v_proj, layer.self_attn.o_proj, layer.mlp.down_proj]:
            if hasattr(proj, 'get_effective_weight'):
                proj.orig_linear.weight.copy_(proj.get_effective_weight())

        # 2. Absorb Attention Symmetries
        if isinstance(layer.self_attn.o_proj, RotatedLoRAOProj):
            v, o = layer.self_attn.v_proj.orig_linear, layer.self_attn.o_proj.orig_linear
            R_q, R_kv = layer.self_attn.o_proj.get_R()

            W_V = v.weight.view(R_kv.shape[0], R_kv.shape[1], -1)
            v.weight.copy_(torch.einsum('hde,hdi->hei', R_kv, W_V).reshape_as(v.weight))

            W_O = o.weight.view(-1, R_q.shape[0], R_q.shape[1])
            o.weight.copy_(torch.einsum('ihd,hde->ihe', W_O, R_q).reshape_as(o.weight))

        # 3. Handle MLP Symmetries (Rotate W_down, extract R_blocks for Inference)
        if isinstance(layer.mlp.down_proj, RotatedLoRADownProj):
            down = layer.mlp.down_proj.orig_linear
            R_blocks = layer.mlp.down_proj.get_R()

            # W_down' = R^T W_down
            w_reshaped = down.weight.view(down.weight.shape[0], layer.mlp.down_proj.num_blocks,
                                          layer.mlp.down_proj.block_size)
            down.weight.copy_(torch.einsum('onb,nbc->onc', w_reshaped, R_blocks).reshape_as(down.weight))

            # Save R_blocks to apply to activations at inference
            mlp_gauges.append(R_blocks.cpu())

        # Strip wrappers
        layer.self_attn.v_proj = getattr(layer.self_attn.v_proj, 'orig_linear', layer.self_attn.v_proj)
        layer.self_attn.o_proj = getattr(layer.self_attn.o_proj, 'orig_linear', layer.self_attn.o_proj)
        layer.mlp.down_proj = getattr(layer.mlp.down_proj, 'orig_linear', layer.mlp.down_proj)

    if mlp_gauges:
        torch.save(mlp_gauges, os.path.join(output_dir, "mlp_gauges.pt"))
    return model


# (Keep get_c4_dataloader exactly as is...)
def get_c4_dataloader(tokenizer, seq_len=256, num_samples=1024):
    print("Loading generalized pre-training dataset (C4)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(dataset)
    batches = []
    while len(batches) < num_samples:
        text = ""
        while True:
            text += next(iterator)["text"] + " \n"
            if len(tokenizer(text, truncation=False)["input_ids"]) >= seq_len: break
        batches.append(tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids)
    return batches


def run_experiment(model_id, tokenizer, calib_batches, use_gauge=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 50}\nStarting {'Unified Gauge-Rotated' if use_gauge else 'Baseline'}\n{'=' * 50}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": device})
    for p in model.parameters(): p.requires_grad = False

    model = inject_symmetries(model, use_gauge=use_gauge)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

    gauge_params = [p for n, p in model.named_parameters() if '.A' in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]

    optim_groups = [{'params': lora_params, 'lr': 1e-4, 'weight_decay': 0.01}]
    if use_gauge: optim_groups.append({'params': gauge_params, 'lr': 5e-3, 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(optim_groups)

    model.train()
    progress_bar = tqdm(calib_batches)
    for input_ids in progress_bar:
        outputs = model(input_ids.to(model.device), labels=input_ids.to(model.device))
        if use_gauge:
            q_loss = sum(l.self_attn.o_proj.quant_loss + l.mlp.down_proj.quant_loss for l in model.model.layers)
            loss = outputs.loss + (0.1 * q_loss)
            progress_bar.set_postfix({"CE": f"{outputs.loss.item():.3f}", "L_q": f"{q_loss.item():.3f}"})
        else:
            loss = outputs.loss
            progress_bar.set_postfix({"CE": f"{loss.item():.3f}"})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        if use_gauge: torch.nn.utils.clip_grad_norm_(gauge_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

    output_dir = f"./llama-7b-{'gauge-rotated-unified' if use_gauge else 'baseline-lora'}"
    model = absorb_and_extract(model, output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    model_id = "NousResearch/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    calib_batches = get_c4_dataloader(tokenizer)
    # run_experiment(model_id, tokenizer, calib_batches, use_gauge=False) # Done
    torch.cuda.empty_cache()
    run_experiment(model_id, tokenizer, calib_batches, use_gauge=True)