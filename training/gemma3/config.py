"""Gemma-3 4B training configuration."""
import os
import torch
from peft import LoraConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
#ad
MODEL_ID     = "google/gemma-3-4b-it"
DATASET_PATH = os.path.join(_ROOT, "data", "energy_data_structured.jsonl")
OUTPUT_DIR   = os.path.join(_ROOT, "gemma3_4b_energy_finetune_structured")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS            = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE               = 2e-4
MAX_SEQ_LENGTH              = 1024
TEST_SIZE                   = 0.05

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
LOAD_IN_4BIT       = True
BNB_QUANT_TYPE     = "nf4"
COMPUTE_DTYPE      = torch.bfloat16
USE_DOUBLE_QUANT   = True

# ── LoRA (PEFT) ───────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
