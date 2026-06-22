import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# =======================================================================
# 1. Gauge Surgery Architecture
# =======================================================================
class RotatedOProj(nn.Module):
    def __init__(self, orig_o_proj, orig_v_proj, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.orig_o_proj = orig_o_proj
        self.orig_v_proj = orig_v_proj
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        target_device = orig_o_proj.weight.device
        target_dtype = orig_o_proj.weight.dtype

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

        rotated_act = torch.einsum('bshd,hde->bshe', x_reshaped.detach(), R_q).to(torch.float32)
        beta = 10.0
        act_loss = (1.0 / beta) * torch.logsumexp(beta * rotated_act.abs(), dim=-1).mean()

        R_q_f32 = R_q.to(torch.float32)
        R_kv_f32 = R_kv.to(torch.float32)

        w_o = self.orig_o_proj.weight.detach().float().view(-1, self.num_heads, self.head_dim)
        w_o_rot = torch.einsum('ihd,hde->ihe', w_o, R_q_f32)

        w_v = self.orig_v_proj.weight.detach().float().view(self.num_kv_heads, self.head_dim, -1)
        w_v_rot = torch.einsum('hde,hdi->hei', R_kv_f32, w_v)

        w_loss = (1.0 / beta) * torch.logsumexp(beta * w_o_rot.abs(), dim=-1).mean() + \
                 (1.0 / beta) * torch.logsumexp(beta * w_v_rot.abs(), dim=-1).mean()

        lambda_w = 0.5
        self.quant_loss = (act_loss + lambda_w * w_loss).to(self.A.dtype)

        return self.orig_o_proj(x)


def inject_gauge_symmetries(model):
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads

    for layer in model.model.layers:
        layer.self_attn.o_proj = RotatedOProj(
            layer.self_attn.o_proj,
            layer.self_attn.v_proj,
            num_heads, num_kv_heads, head_dim
        )
    return model


@torch.no_grad()
def absorb_gauge_and_restore(model):
    for layer in model.model.layers:
        if isinstance(layer.self_attn.o_proj, RotatedOProj):
            R_q, R_kv = layer.self_attn.o_proj.get_R()
            orig_v = layer.self_attn.o_proj.orig_v_proj
            orig_o = layer.self_attn.o_proj.orig_o_proj

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

            layer.self_attn.o_proj = orig_o
    return model


# =======================================================================
# 2. General Purpose Pre-Training Data (C4)
# =======================================================================
def get_c4_dataloader(tokenizer, seq_len=512, num_samples=256):
    """Streams generalized web data to prevent in-domain overfitting."""
    print("Loading generalized pre-training dataset (C4)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(dataset)

    batches = []
    while len(batches) < num_samples:
        text = ""
        # Accumulate text until we have enough tokens to fill the context length
        while True:
            example = next(iterator)
            text += example["text"] + " \n"
            if len(tokenizer(text, truncation=False)["input_ids"]) >= seq_len:
                break

        input_ids = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids
        batches.append(input_ids)

    return batches


# =======================================================================
# 3. Ablation Harness
# =======================================================================
def run_experiment(model_id, tokenizer, calib_batches, use_gauge=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 50}")
    print(f"Starting Ablation Run | Mode: {'Gauge-Rotated (Proposed)' if use_gauge else 'Baseline-Tuned (Control)'}")
    print(f"{'=' * 50}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto" if device == "cuda" else None
    )

    if use_gauge:
        model = inject_gauge_symmetries(model)

        gauge_params = [p for n, p in model.named_parameters() if 'self_attn.o_proj.A' in n]
        dense_params = [p for n, p in model.named_parameters() if 'self_attn.o_proj.A' not in n]

        optimizer = torch.optim.AdamW([
            {'params': dense_params, 'lr': 2e-6, 'weight_decay': 0.01},  # Was 2e-5
            {'params': gauge_params, 'lr': 5e-3, 'weight_decay': 0.0}
        ])
        lambda_quant = 0.1
    else:
        # Standard fine-tuning without the quantization proxy
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        lambda_quant = 0.0

    model.train()
    progress_bar = tqdm(calib_batches, desc="Training")

    for input_ids in progress_bar:
        input_ids = input_ids.to(model.device)

        outputs = model(input_ids, labels=input_ids)
        ce_loss = outputs.loss

        if use_gauge:
            total_quant_loss = sum(layer.self_attn.o_proj.quant_loss for layer in model.model.layers)
            loss = ce_loss + (lambda_quant * total_quant_loss)
            progress_bar.set_postfix({"CE": f"{ce_loss.item():.3f}", "L_inf": f"{total_quant_loss.item():.3f}"})
        else:
            loss = ce_loss
            progress_bar.set_postfix({"CE": f"{ce_loss.item():.3f}"})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if use_gauge:
        print("Absorbing optimal rotations into dense weights...")
        model = absorb_gauge_and_restore(model)
        output_dir = "./qwen-0.5b-gauge-rotated"
    else:
        output_dir = "./qwen-0.5b-baseline-tuned"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Run complete. Model saved to {output_dir}")


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Generate the exact same 256 batches of C4 data so both runs are perfectly comparable
    calib_batches = get_c4_dataloader(tokenizer, seq_len=512, num_samples=2*2048)

    # 1. Run the Control (Standard CE Training)
    run_experiment(model_id, tokenizer, calib_batches, use_gauge=False)

    # Clear CUDA memory to prevent OOM on the 5070 Ti before second run
    torch.cuda.empty_cache()

    # 2. Run the Proposed Method (CE + Gauge Optimization)
    run_experiment(model_id, tokenizer, calib_batches, use_gauge=True)