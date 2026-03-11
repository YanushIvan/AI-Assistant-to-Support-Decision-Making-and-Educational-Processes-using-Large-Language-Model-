"""Phi-3 Mini training configuration."""
import os
import torch
from peft import LoraConfig

# ── Paths (relative to project root) ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

MODEL_ID     = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = os.path.join(_ROOT, "data", "energy_data_structured.jsonl")
OUTPUT_DIR   = os.path.join(_ROOT, "phi3_mini_energy_finetune_structured")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS            = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE               = 2e-4
MAX_SEQ_LENGTH              = 1024
TEST_SIZE                   = 0.01

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a highly knowledgeable expert on energy, climate, and financial markets. "
    "Always respond in the following structured format:\n\n"
    "ANSWER: <one-sentence direct answer>\n\n"
    "KEY FACTS:\n• <specific fact with data>\n• <specific fact with data>\n• <specific fact with data>\n\n"
    "RISK LEVEL: Low / Medium / High\n→ <one-sentence explanation>\n\n"
    "CONFIDENCE: High / Medium / Low\n→ <reason, e.g. based on IRENA 2025 data / estimated>\n\n"
    "Be concise, technically accurate, and use specific numbers where available."
)

# ── QLoRA (4-bit) ─────────────────────────────────────────────────────────────
LOAD_IN_4BIT   = True
BNB_QUANT_TYPE = "nf4"
COMPUTE_DTYPE  = torch.bfloat16   # RTX A4500 supports bf16

# ── LoRA (PEFT) ───────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
