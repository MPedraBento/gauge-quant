import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# =======================================================================
# 1. Native BF16 LoRA + Gauge Architecture
# =======================================================================
class LoRALinear(nn.Module):
    """Custom Native LoRA to bypass PEFT wrapper complications."""

    def __init__(self, orig_linear, r=16, alpha=16):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.scaling = alpha / r

        in_dim = orig_linear.in_features
        out_dim = orig_linear.out_features
        dtype = orig_linear.weight.dtype
        device = orig_linear.weight.device

        # Initialize A with normal distribution, B with zeros
        self.lora_A = nn.Parameter(torch.randn(r, in_dim, dtype=dtype, device=device) * (1 / in_dim) ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, dtype=dtype, device=device))

    def get_effective_weight(self):
        return self.orig_linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        # Base forward pass + LoRA forward pass
        return self.orig_linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class RotatedLoRAOProj(LoRALinear):
    def __init__(self, orig_linear, lora_v_proj, num_heads, num_kv_heads, head_dim, r=16, alpha=16):
        super().__init__(orig_linear, r, alpha)
        self.lora_v_proj = lora_v_proj
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        target_device = orig_linear.weight.device
        target_dtype = orig_linear.weight.dtype

        self.A = nn.Parameter(
            torch.zeros(num_kv_heads, head_dim, head_dim, device=target_device, dtype=target_dtype)
        )
        self.quant_loss = 0.0

    def get_R(self):
        skew = self.A - self.A.transpose(-1, -2)
        skew_f32 = skew.to(torch.float32)

        I = torch.eye(self.head_dim, device=skew.device, dtype=torch.float32).unsqueeze(0)
        R_kv_f32 = torch.matmul(I - skew_f32, torch.linalg.inv(I + skew_f32))
        R_kv = R_kv_f32.to(self.A.dtype)
        R_q = R_kv.repeat_interleave(self.num_groups, dim=0)

        return R_q, R_kv

    def forward(self, x):
        bsz, seq_len, hidden_size = x.shape
        x_reshaped = x.view(bsz, seq_len, self.num_heads, self.head_dim)

        R_q, R_kv = self.get_R()
        beta = 10.0

        # 1. Activation Penalty (Max-Pooling compresses graph by 256x)
        rotated_act = torch.einsum('bshd,hde->bshe', x_reshaped.detach(), R_q)
        # Squash sequence dimension to find the worst outlier per head/channel
        act_max = rotated_act.abs().amax(dim=1)
        act_loss = (1.0 / beta) * torch.logsumexp(beta * act_max.to(torch.float32), dim=-1).mean()
        del rotated_act, act_max

        # 2. Weight Penalty O-Proj (Striding compresses graph by 16x)
        w_o = self.orig_linear.weight.detach().view(-1, self.num_heads, self.head_dim)
        if w_o.shape[0] > 256:
            w_o = w_o[::16]  # Uniformly sample 256 rows to represent the geometry
        w_o_rot = torch.einsum('ihd,hde->ihe', w_o, R_q)
        w_o_loss = (1.0 / beta) * torch.logsumexp(beta * w_o_rot.abs().to(torch.float32), dim=-1).mean()
        del w_o, w_o_rot

        # 3. Weight Penalty V-Proj (Striding compresses graph by 16x)
        w_v = self.lora_v_proj.orig_linear.weight.detach().view(self.num_kv_heads, self.head_dim, -1)
        if w_v.shape[-1] > 256:
            w_v = w_v[:, :, ::16]  # Uniformly sample 256 columns
        w_v_rot = torch.einsum('hde,hdi->hei', R_kv, w_v)
        w_v_loss = (1.0 / beta) * torch.logsumexp(beta * w_v_rot.abs().to(torch.float32), dim=-1).mean()
        del w_v, w_v_rot

        lambda_w = 0.5
        self.quant_loss = (act_loss + lambda_w * (w_o_loss + w_v_loss)).to(self.A.dtype)

        return super().forward(x)


def inject_7b_symmetries(model, use_gauge=True):
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads

    for layer in model.model.layers:
        # V is always a standard LoRA wrapper
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj)

        if use_gauge:
            # Inject Gauge math only for the proposed run
            layer.self_attn.o_proj = RotatedLoRAOProj(
                layer.self_attn.o_proj,
                layer.self_attn.v_proj,
                num_heads, num_kv_heads, head_dim
            )
        else:
            # Pure LoRA for the baseline run
            layer.self_attn.o_proj = LoRALinear(layer.self_attn.o_proj)

    return model


@torch.no_grad()
def absorb_lora_and_gauge(model):
    """Folds the LoRA adapters into W, then conditionally folds R into W_merged."""
    for layer in model.model.layers:
        # 1. Gauge-Rotated Absorption
        if isinstance(layer.self_attn.o_proj, RotatedLoRAOProj):
            v_wrapper = layer.self_attn.v_proj
            o_wrapper = layer.self_attn.o_proj

            # Merge LoRA first
            v_wrapper.orig_linear.weight.copy_(v_wrapper.get_effective_weight())
            o_wrapper.orig_linear.weight.copy_(o_wrapper.get_effective_weight())

            R_q, R_kv = o_wrapper.get_R()
            orig_v = v_wrapper.orig_linear
            orig_o = o_wrapper.orig_linear

            # Rotate merged weights
            W_V = orig_v.weight.view(R_kv.shape[0], R_kv.shape[1], -1)
            W_V_rot = torch.einsum('hde,hdi->hei', R_kv, W_V).reshape(-1, orig_v.weight.shape[1])
            orig_v.weight.copy_(W_V_rot)

            W_O = orig_o.weight.view(-1, R_q.shape[0], R_q.shape[1])
            W_O_rot = torch.einsum('ihd,hde->ihe', W_O, R_q).reshape(orig_o.weight.shape[0], -1)
            orig_o.weight.copy_(W_O_rot)

            if getattr(orig_v, "bias", None) is not None:
                b_V = orig_v.bias.view(R_kv.shape[0], R_kv.shape[1])
                b_V_rot = torch.einsum('hd,hde->he', b_V, R_kv).reshape(-1)
                orig_v.bias.copy_(b_V_rot)

            layer.self_attn.v_proj = orig_v
            layer.self_attn.o_proj = orig_o

        # 2. Baseline-LoRA Absorption
        elif hasattr(layer.self_attn.o_proj, 'get_effective_weight'):
            v_wrapper = layer.self_attn.v_proj
            o_wrapper = layer.self_attn.o_proj

            # Simply merge LoRA and strip wrappers
            v_wrapper.orig_linear.weight.copy_(v_wrapper.get_effective_weight())
            o_wrapper.orig_linear.weight.copy_(o_wrapper.get_effective_weight())

            layer.self_attn.v_proj = v_wrapper.orig_linear
            layer.self_attn.o_proj = o_wrapper.orig_linear

    return model


# =======================================================================
# 2. Pre-Training Harness
# =======================================================================
def get_c4_dataloader(tokenizer, seq_len=512, num_samples=1024):
    print("Loading generalized pre-training dataset (C4)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(dataset)

    batches = []
    while len(batches) < num_samples:
        text = ""
        while True:
            text += next(iterator)["text"] + " \n"
            if len(tokenizer(text, truncation=False)["input_ids"]) >= seq_len:
                break
        input_ids = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids
        batches.append(input_ids)
    return batches


def run_7b_experiment(model_id, tokenizer, calib_batches, use_gauge=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_name = 'Gauge-Rotated 7B' if use_gauge else 'Baseline-LoRA 7B'
    print(f"\n{'=' * 50}\nStarting {mode_name}\n{'=' * 50}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device} if device == "cuda" else None
    )

    # Freeze massive 14GB base model
    for param in model.parameters():
        param.requires_grad = False

    model = inject_7b_symmetries(model, use_gauge=use_gauge)

    # This discards intermediate activations to save ~2GB of VRAM
    model.gradient_checkpointing_enable()
    # Required for custom LoRA layers to play nicely with checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Separate trainable parameter groups
    gauge_params = [p for n, p in model.named_parameters() if 'self_attn.o_proj.A' in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]

    if use_gauge:
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': 1e-4, 'weight_decay': 0.01},  # LoRA optimal LR
            {'params': gauge_params, 'lr': 5e-3, 'weight_decay': 0.0}
        ])
    else:
        optimizer = torch.optim.AdamW([{'params': lora_params, 'lr': 1e-4, 'weight_decay': 0.01}])

    model.train()
    progress_bar = tqdm(calib_batches, desc=f"Training {mode_name}")

    for input_ids in progress_bar:
        input_ids = input_ids.to(model.device)
        outputs = model(input_ids, labels=input_ids)
        ce_loss = outputs.loss

        if use_gauge:
            total_quant_loss = sum(layer.self_attn.o_proj.quant_loss for layer in model.model.layers)
            loss = ce_loss + (0.1 * total_quant_loss)
            progress_bar.set_postfix({"CE": f"{ce_loss.item():.3f}", "L_inf": f"{total_quant_loss.item():.3f}"})
        else:
            loss = ce_loss
            progress_bar.set_postfix({"CE": f"{ce_loss.item():.3f}"})

        loss.backward()

        # Orthogonal Gradient Clipping prevents LoRA from being suppressed by Gauge updates
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        if use_gauge:
            torch.nn.utils.clip_grad_norm_(gauge_params, 1.0)

        optimizer.step()
        optimizer.zero_grad()

    print(f"Absorbing LoRA {'and Gauge ' if use_gauge else ''}into dense weights...")
    model = absorb_lora_and_gauge(model)

    output_dir = f"./llama-7b-{'gauge-rotated' if use_gauge else 'baseline-lora'}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Run complete. Model saved to {output_dir}")


if __name__ == "__main__":
    model_id = "NousResearch/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    calib_batches = get_c4_dataloader(tokenizer, seq_len=256, num_samples=1024)

    #run_7b_experiment(model_id, tokenizer, calib_batches, use_gauge=False)
    torch.cuda.empty_cache()
    run_7b_experiment(model_id, tokenizer, calib_batches, use_gauge=True)