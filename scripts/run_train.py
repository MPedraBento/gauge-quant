import os
import yaml
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

from gauge_quant.modules import AttentionGauge, BlockDiagonalMLPGauge
from gauge_quant.functional import logsumexp_outlier_penalty

logger = logging.getLogger("GaugeQuant.Train")
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_decoder_layers(model):
    """Resolves transformer decoder layers regardless of PEFT wrapping."""
    if hasattr(model, 'peft_config'):  # PeftModel
        return model.base_model.model.model.layers
    return model.model.layers


class OutputGaugeWrapper(torch.nn.Module):
    """Wraps v_proj: gauge penalty computed on the layer's OUTPUT (the V output space)."""
    def __init__(self, base_layer, gauge):
        super().__init__()
        self.base_layer = base_layer
        self.gauge = gauge
        self.current_penalty = torch.tensor(0.0)

    def forward(self, x, *args, **kwargs):
        output = self.base_layer(x, *args, **kwargs)
        rotated = self.gauge(output.detach())
        self.current_penalty = logsumexp_outlier_penalty(rotated)
        return output


class InputGaugeWrapper(torch.nn.Module):
    """Wraps down_proj: gauge penalty computed on the layer's INPUT (the SwiGLU output)."""
    def __init__(self, base_layer, gauge):
        super().__init__()
        self.base_layer = base_layer
        self.gauge = gauge
        self.current_penalty = torch.tensor(0.0)

    def forward(self, x, *args, **kwargs):
        rotated = self.gauge(x.detach())
        self.current_penalty = logsumexp_outlier_penalty(rotated)
        return self.base_layer(x, *args, **kwargs)


def inject_gauges(model, config):
    gauges = torch.nn.ModuleList()
    layers = get_decoder_layers(model)
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    intermediate_dim = model.config.intermediate_size

    for i, layer in enumerate(layers):
        if config["where"] in ["VO", "Unified"]:
            vo_gauge = AttentionGauge(num_kv_heads, head_dim).to(model.device)
            layer.self_attn.v_proj = OutputGaugeWrapper(layer.self_attn.v_proj, vo_gauge)
            gauges.append(vo_gauge)

        if config["where"] in ["MLP", "Unified"]:
            mlp_gauge = BlockDiagonalMLPGauge(intermediate_dim, block_size=64).to(model.device)
            layer.mlp.down_proj = InputGaugeWrapper(layer.mlp.down_proj, mlp_gauge)
            gauges.append(mlp_gauge)

    return model, gauges


def unwrap_gauges(model):
    """Removes the LayerWrapper so Hugging Face saves standard HF weight keys."""
    layers = get_decoder_layers(model)
    for layer in layers:
        if hasattr(layer.self_attn.v_proj, 'base_layer'):
            layer.self_attn.v_proj = layer.self_attn.v_proj.base_layer
        if hasattr(layer.mlp.down_proj, 'base_layer'):
            layer.mlp.down_proj = layer.mlp.down_proj.base_layer
    return model


def get_c4_batch(iterator, tokenizer, device, batch_size=1, seq_len=512):
    """Fetches and tokenizes a batch of real text from the C4 dataset stream."""
    texts = []
    while len(texts) < batch_size:
        try:
            sample = next(iterator)
            if len(sample["text"]) > seq_len * 2:
                texts.append(sample["text"])
        except StopIteration:
            break

    if not texts:
        raise StopIteration("C4 stream exhausted")

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="pt"
    )
    return encoded["input_ids"].to(device)


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Loading {config['model_id']} on {device} (seed={seed})...")

    # 1. Initialize Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 2. Apply LoRA (if modality requires it)
    if config["modality"] == "LoRA":
        lora_config = LoraConfig(
            r=config["lora_r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA adapters injected successfully.")
    else:
        for param in model.parameters():
            param.requires_grad = True

    # 3. Inject Gauges
    model, gauges = inject_gauges(model, config)
    logger.info(f"Injected {len(gauges)} continuous gauges ({config['where']} configuration).")

    # 4. Joint Optimizer Setup (Mutually Exclusive Groups)
    gauge_params_set = set(id(p) for p in gauges.parameters())
    base_params = [p for p in model.parameters() if id(p) not in gauge_params_set and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 2e-5},
        {'params': list(gauges.parameters()), 'lr': float(config["learning_rate"]), 'weight_decay': 0.0}
    ])

    # 5. Load Real C4 Dataset Stream
    logger.info("Connecting to AllenAI C4 Dataset stream...")
    c4_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    c4_iterator = iter(c4_dataset)

    # 6. Joint Training Loop
    logger.info(f"Starting Joint Optimization on C4 for {config['steps']} steps...")
    model.train()

    log_every = config.get("log_every", 50)
    os.makedirs(config["output_dir"], exist_ok=True)
    rot_log_path = os.path.join(config["output_dir"], "rot_loss.csv")
    rot_log_file = open(rot_log_path, "w")
    rot_log_file.write("step,ce_loss,rot_loss,total_loss\n")

    layers = get_decoder_layers(model)

    for step in tqdm(range(config["steps"])):
        input_ids = get_c4_batch(c4_iterator, tokenizer, device, seq_len=512)

        labels = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids, labels=labels)
        ce_loss = outputs.loss

        # Aggregate Gauge penalties
        rot_loss = torch.tensor(0.0, device=device)
        for layer in layers:
            if hasattr(layer.self_attn.v_proj, 'current_penalty'):
                rot_loss = rot_loss + layer.self_attn.v_proj.current_penalty
            if hasattr(layer.mlp.down_proj, 'current_penalty'):
                rot_loss = rot_loss + layer.mlp.down_proj.current_penalty

        total_loss = ce_loss + config["penalty_weight"] * rot_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_every == 0:
            logger.info(f"[Step {step}] CE: {ce_loss.item():.4f} | Rot: {rot_loss.item():.4f} | Total: {total_loss.item():.4f}")
        rot_log_file.write(f"{step},{ce_loss.item():.6f},{rot_loss.item():.6f},{total_loss.item():.6f}\n")

    rot_log_file.close()
    logger.info(f"Loss curve saved to {rot_log_path}")

    # 7. Save Artifacts
    model = unwrap_gauges(model)

    if config["modality"] == "LoRA":
        model.save_pretrained(config["output_dir"])
    else:
        model.save_pretrained(config["output_dir"], safe_serialization=True)

    torch.save(gauges.state_dict(), os.path.join(config["output_dir"], "gauges.pt"))
    logger.info(f"Artifacts successfully saved to {config['output_dir']}.")
