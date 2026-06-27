import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
import gc

# =======================================================================
# 1. Reuse Your Quantization Infrastructure
# =======================================================================
class QuantizedLinear(nn.Module):
    def __init__(self, orig_layer, w_bits=4, a_bits=8):
        super().__init__()
        self.orig_layer = orig_layer
        self.w_bits = w_bits
        self.a_bits = a_bits

        if self.w_bits < 16:
            W = self.orig_layer.weight.data.float()
            max_val = W.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
            qmax = (2 ** (self.w_bits - 1)) - 1
            scale = max_val / qmax
            q_W = torch.round(W / scale).clamp(-qmax, qmax) * scale
            self.orig_layer.weight.data.copy_(q_W.to(self.orig_layer.weight.dtype))

    def forward(self, x):
        if self.a_bits < 16:
            x_f32 = x.float()
            max_val = x_f32.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
            qmax = (2 ** (self.a_bits - 1)) - 1
            scale = max_val / qmax
            q_x = torch.round(x_f32 / scale).clamp(-qmax, qmax) * scale
            x = q_x.to(x.dtype)
        return self.orig_layer(x)


def apply_attention_quantization(model, w_bits=4, a_bits=8):
    for layer in model.model.layers:
        layer.self_attn.v_proj = QuantizedLinear(layer.self_attn.v_proj, w_bits=w_bits, a_bits=a_bits)
        layer.self_attn.o_proj = QuantizedLinear(layer.self_attn.o_proj, w_bits=w_bits, a_bits=a_bits)
    return model

# =======================================================================
# 2. MMLU Accuracy Evaluation Logic
# =======================================================================
@torch.no_grad()
def evaluate_mmlu_accuracy(model, tokenizer, dataset, max_samples=100):
    """Evaluates multiple-choice accuracy on MMLU."""
    model.eval()
    device = next(model.parameters()).device
    
    # Map label integer to string option
    choice_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    # Pre-tokenize token IDs for options 'A', 'B', 'C', 'D'
    # We grab the first token of the output to compare with the model's logits
    choice_tokens = [tokenizer.encode(f" {choice_map[i]}", add_special_tokens=False)[-1] for i in range(4)]
    
    correct = 0
    total = 0

    # Subset the dataset to keep evaluation time reasonable
    eval_subset = dataset.select(range(min(len(dataset), max_samples)))

    for item in tqdm(eval_subset, desc="Evaluating MMLU"):
        question = item["question"]
        choices = item["choices"]
        gold_idx = item["answer"] # index 0 to 3
        
        # Format a clear, standard multiple-choice prompt
        prompt = (
            f"Question: {question}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            f"Answer:"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        # Extract logits for the very last generated token position
        next_token_logits = outputs.logits[0, -1, :]
        
        # Isolate the logit scores specifically for choices A, B, C, D
        choice_logits = next_token_logits[choice_tokens]
        
        # Prediction is the choice with the maximum logit value
        pred_idx = torch.argmax(choice_logits).item()
        
        if pred_idx == gold_idx:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading MMLU test split...")
    # 1. Grab all the individual subject names
    all_subjects = get_dataset_config_names("tasksource/mmlu")
    # 2. Load the test split for every single subject into a list
    all_splits = [load_dataset("tasksource/mmlu", subject, split="test") for subject in all_subjects]
    # 3. Combine them all into one single dataset
    test_dataset = concatenate_datasets(all_splits)

    W_BITS = 4
    A_BITS = 8
    MAX_SAMPLES = 200 # Bump this up to 1000+ for exact benchmark fidelity

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

            # Evaluate Accuracy instead of PPL
            acc = evaluate_mmlu_accuracy(model, tokenizer, test_dataset, max_samples=MAX_SAMPLES)
            results[name] = acc
            print(f">>> {name} W{W_BITS}A{A_BITS} MMLU Accuracy: {acc * 100:.2f}%")

        except Exception as e:
            print(f"Failed to evaluate {name}. Error: {e}")
            results[name] = "FAILED"

        finally:
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    # Final Results Table Generation
    print("\n" + "=" * 50)
    print(" FINAL ABLATION RESULTS (W4A8 on MMLU)")
    print("=" * 50)
    for name, acc in results.items():
        acc_str = f"{acc * 100:.2f}%" if isinstance(acc, float) else acc
        print(f"{name:<40} | Accuracy: {acc_str}")
    print("=" * 50)
