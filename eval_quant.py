import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc


class QuantizedLinear(nn.Module):
    """Wraps an nn.Linear layer to simulate W_bits / A_bits uniform quantization."""

    def __init__(self, orig_layer, w_bits=4, a_bits=8):
        super().__init__()
        self.orig_layer = orig_layer
        self.w_bits = w_bits
        self.a_bits = a_bits

        # 1. Offline Weight Quantization (Symmetric, Channel-wise)
        if self.w_bits < 16:
            W = self.orig_layer.weight.data.float()
            max_val = W.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
            qmax = (2 ** (self.w_bits - 1)) - 1
            scale = max_val / qmax

            q_W = torch.round(W / scale).clamp(-qmax, qmax) * scale
            self.orig_layer.weight.data.copy_(q_W.to(self.orig_layer.weight.dtype))

    def forward(self, x):
        # 2. Dynamic Activation Quantization (Symmetric, Token-wise)
        if self.a_bits < 16:
            x_f32 = x.float()
            max_val = x_f32.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
            qmax = (2 ** (self.a_bits - 1)) - 1
            scale = max_val / qmax

            q_x = torch.round(x_f32 / scale).clamp(-qmax, qmax) * scale
            x = q_x.to(x.dtype)

        return self.orig_layer(x)


def apply_attention_quantization(model, w_bits=4, a_bits=8):
    """Replaces V and O projections with their quantized wrappers."""
    for layer in model.model.layers:
        layer.self_attn.v_proj = QuantizedLinear(
            layer.self_attn.v_proj, w_bits=w_bits, a_bits=a_bits
        )
        layer.self_attn.o_proj = QuantizedLinear(
            layer.self_attn.o_proj, w_bits=w_bits, a_bits=a_bits
        )
    return model


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset, ctx_len=512, max_samples=64):
    """Computes token-level Perplexity on Wikitext-2."""
    model.eval()
    device = next(model.parameters()).device

    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    end_loc = 0

    for begin_loc in tqdm(range(0, min(seq_len, max_samples * ctx_len), ctx_len), desc="Evaluating PPL"):
        end_loc = min(begin_loc + ctx_len, seq_len)
        trg_len = end_loc - begin_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Wikitext validation set (Zero-Shot Eval)...")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    W_BITS = 4
    A_BITS = 8

    # Define the three ablation targets
    experiments = [
        {"name": "Vanilla Qwen-0.5B (No Training)", "path": "Qwen/Qwen2.5-0.5B"},
        {"name": "Baseline-Tuned Qwen (C4 Control)", "path": "./qwen-0.5b-baseline-tuned"},
        {"name": "Gauge-Rotated Qwen (Proposed)", "path": "./qwen-0.5b-gauge-rotated"}
    ]

    results = {}

    for exp in experiments:
        name = exp["name"]
        path = exp["path"]
        print(f"\n{'=' * 60}")
        print(f"--- Evaluating: {name} ---")
        print(f"{'=' * 60}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, device_map="auto" if device == "cuda" else None
            )

            # Apply W4A8 Quantization
            model = apply_attention_quantization(model, w_bits=W_BITS, a_bits=A_BITS)

            # Evaluate
            ppl = evaluate_perplexity(model, tokenizer, test_dataset)
            results[name] = ppl
            print(f">>> {name} W{W_BITS}A{A_BITS} PPL: {ppl:.4f}")

        except Exception as e:
            print(f"Failed to evaluate {name}. Error: {e}")
            results[name] = "FAILED"

        finally:
            # Ruthlessly clear memory for the next run
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    # Final Results Table Generation
    print("\n" + "=" * 50)
    print(" FINAL ABLATION RESULTS (W4A8 on Wikitext-2)")
    print("=" * 50)
    for name, ppl in results.items():
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, float) else ppl
        print(f"{name:<40} | PPL: {ppl_str}")
    print("=" * 50)