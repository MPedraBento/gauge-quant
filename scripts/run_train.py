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


class LayerWrapper(torch.nn.Module):
    """
    Transparent wrapper enforcing the Dual-Graph Optimization Topology.
    CE Loss updates the base weights. Gauge Loss strictly updates the R matrices.
    """
    def __init__(self, base_layer, gauge, is_mlp=False):
        super().__init__()
        self.base_layer = base_layer
        self.gauge = gauge
        self.is_mlp = is_mlp
        self.current_penalty = torch.tensor(0.0)

    def forward(self, x, *args, **kwargs):
        with torch.autocast(device_type=x.device.type, dtype=x.dtype):
            # ========================================================
            # THE GEOMETER (Gauge Graph - isolated via detach)
            # ========================================================
            x_detached = x.detach()
            rotated_x = self.gauge(x_detached).to(x.dtype)
            self.current_penalty = logsumexp_outlier_penalty(rotated_x)

        # ========================================================
        # THE BRAIN (Cross-Entropy Graph - blind to R)
        # ========================================================
        return self.base_layer(x, *args, **kwargs)


def inject_gauges(model, config):
    gauges = torch.nn.ModuleList()
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size

    for i, layer in enumerate(model.model.layers):
        if config["where"] in ["VO", "Unified"]:
            vo_gauge = AttentionGauge(hidden_dim).to(model.device)
            layer.self_attn.o_proj = LayerWrapper(layer.self_attn.o_proj, vo_gauge)
            gauges.append(vo_gauge)

        if config["where"] in ["MLP", "Unified"]:
            mlp_gauge = BlockDiagonalMLPGauge(intermediate_dim, block_size=64).to(model.device)
            layer.mlp.down_proj = LayerWrapper(layer.mlp.down_proj, mlp_gauge, is_mlp=True)
            gauges.append(mlp_gauge)

    return model, gauges


def unwrap_gauges(model):
    """Removes the LayerWrapper so Hugging Face saves standard HF weight keys."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn.o_proj, 'base_layer'):
            layer.self_attn.o_proj = layer.self_attn.o_proj.base_layer
        if hasattr(layer.mlp.down_proj, 'base_layer'):
            layer.mlp.down_proj = layer.mlp.down_proj.base_layer
    return model


def get_c4_batch(iterator, tokenizer, device, batch_size=1, seq_len=512):
    """Fetches and tokenizes a batch of real text from the C4 dataset stream."""
    texts = []
    while len(texts) < batch_size:
        try:
            sample = next(iterator)
            # Only take samples that are reasonably long
            if len(sample["text"]) > seq_len * 2:
                texts.append(sample["text"])
        except StopIteration:
            break

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
    logger.info(f"Loading {config['model_id']} on {device}...")

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
        # If full fine-tuning, ensure weights require grad
        for param in model.parameters():
            param.requires_grad = True

    # 3. Inject Gauges
    model, gauges = inject_gauges(model, config)
    logger.info(f"Injected {len(gauges)} continuous gauges ({config['where']} configuration).")

    # 4. Joint Optimizer Setup (Mutually Exclusive Groups)
    gauge_params_set = set(gauges.parameters())
    base_params = [p for p in model.parameters() if p not in gauge_params_set]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 2e-5},
        {'params': gauges.parameters(), 'lr': float(config["learning_rate"])}
    ])

    # 5. Load Real C4 Dataset Stream
    logger.info("Connecting to AllenAI C4 Dataset stream...")
    c4_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    c4_iterator = iter(c4_dataset)

    # 6. Joint Training Loop
    logger.info(f"Starting Joint Optimization on C4 for {config['steps']} steps...")
    model.train()

    for step in tqdm(range(config["steps"])):
        # Get real linguistic data
        input_ids = get_c4_batch(c4_iterator, tokenizer, device, seq_len=512)

        # Create labels and mask the padding tokens with -100
        labels = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        # Forward Pass
        outputs = model(input_ids, labels=labels)
        ce_loss = outputs.loss

        # Aggregate the Gauge L_infinity penalties
        gauge_loss = torch.tensor(0.0, device=device)
        for layer in model.model.layers:
            if hasattr(layer.self_attn.o_proj, 'current_penalty'):
                gauge_loss += layer.self_attn.o_proj.current_penalty
            if hasattr(layer.mlp.down_proj, 'current_penalty'):
                gauge_loss += layer.mlp.down_proj.current_penalty

        # Total Objective
        total_loss = ce_loss + (config["penalty_weight"] * gauge_loss)

        # Backward Pass & Optimize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 7. Save Artifacts
    os.makedirs(config["output_dir"], exist_ok=True)

    # UNWRAP THE MODEL to restore standard dictionary keys!
    model = unwrap_gauges(model)

    if config["modality"] == "LoRA":
        model.save_pretrained(config["output_dir"])
    else:
        model.save_pretrained(config["output_dir"], safe_serialization=True)

    torch.save(gauges.state_dict(), os.path.join(config["output_dir"], "gauges.pt"))
    logger.info(f"Artifacts successfully saved to {config['output_dir']}.")