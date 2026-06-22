import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os


class QuantizedLinear(nn.Module):
    def __init__(self, orig_layer, w_bits=4, a_bits=4):
        super().__init__()
        self.orig_layer = orig_layer
        self.w_bits, self.a_bits = w_bits, a_bits

        W = self.orig_layer.weight.data.float()
        max_val = W.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        qmax = (2 ** (self.w_bits - 1)) - 1
        scale = max_val / qmax
        self.orig_layer.weight.data.copy_((torch.round(W / scale).clamp(-qmax, qmax) * scale).to(W.dtype))

    def forward(self, x):
        x_f32 = x.float()
        max_val = x_f32.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        qmax = (2 ** (self.a_bits - 1)) - 1
        scale = max_val / qmax
        x_q = (torch.round(x_f32 / scale).clamp(-qmax, qmax) * scale).to(x.dtype)
        return self.orig_layer(x_q)


class RotatedQuantizedDownProj(nn.Module):
    """Applies the explicit Block-Diagonal R to activations before passing to quantized W_down."""

    def __init__(self, orig_layer, R_blocks, w_bits=4, a_bits=4):
        super().__init__()
        self.quant_linear = QuantizedLinear(orig_layer, w_bits, a_bits)
        self.register_buffer("R_blocks", R_blocks)
        self.num_blocks = R_blocks.shape[0]
        self.block_size = R_blocks.shape[1]
        self.hidden_dim = self.num_blocks * self.block_size

    def forward(self, x):
        # 1. Rotate FP16 Activations using explicit Gauge
        *leading_dims, _ = x.shape
        x_reshaped = x.view(*leading_dims, self.num_blocks, self.block_size)
        x_rot = torch.einsum('...nb,nbc->...nc', x_reshaped, self.R_blocks).reshape(*leading_dims, self.hidden_dim)
        # 2. Pass to standard W4A4 quantizer
        return self.quant_linear(x_rot)


def apply_full_quantization(model, path, w_bits=4, a_bits=4):
    gauge_path = os.path.join(path, "mlp_gauges.pt")
    mlp_gauges = torch.load(gauge_path) if os.path.exists(gauge_path) else None

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.v_proj = QuantizedLinear(layer.self_attn.v_proj, w_bits, a_bits)
        layer.self_attn.o_proj = QuantizedLinear(layer.self_attn.o_proj, w_bits, a_bits)

        if mlp_gauges is not None:
            # Inject explicit R before the quantized linear layer
            layer.mlp.down_proj = RotatedQuantizedDownProj(layer.mlp.down_proj, mlp_gauges[i].to(model.device), w_bits,
                                                           a_bits)
        else:
            layer.mlp.down_proj = QuantizedLinear(layer.mlp.down_proj, w_bits, a_bits)

    return model


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset, ctx_len=512, max_samples=32):
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    end_loc = 0
    for begin_loc in tqdm(range(0, min(seq_len, max_samples * ctx_len), ctx_len)):
        end_loc = min(begin_loc + ctx_len, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        nlls.append(model(input_ids, labels=input_ids).loss * (end_loc - begin_loc))

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    experiments = [
        {"name": "Vanilla LLaMA-2 7B", "path": "NousResearch/Llama-2-7b-hf"},
        {"name": "Unified Gauge-Rotated 7B (Proposed)", "path": "./llama-7b-gauge-rotated-unified"}
    ]

    for exp in experiments:
        print(f"\n--- Evaluating: {exp['name']} (W4A4) ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(exp["path"])
            model = AutoModelForCausalLM.from_pretrained(exp["path"], dtype=torch.bfloat16, device_map={"": device})
            model = apply_full_quantization(model, exp["path"], w_bits=4, a_bits=4)
            ppl = evaluate_perplexity(model, tokenizer, test_dataset)
            print(f">>> {exp['name']} W4A4 PPL: {ppl:.4f}")
        except Exception as e:
            print(f"Failed to evaluate {exp['name']}. Error: {e}")
        finally:
            # CRITICAL MEMORY CLEANUP
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            import gc

            gc.collect()
            torch.cuda.empty_cache()