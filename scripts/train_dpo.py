"""
DPO Training Script for Energy AI
==================================
Trains a language model using Direct Preference Optimization (DPO).

Workflow:
  1. Load the DPO dataset (energy_data_dpo.jsonl) — triplets of
     (prompt, chosen, rejected)
  2. Load the SFT-fine-tuned model as both policy and reference
  3. Apply LoRA on top of the policy model
  4. Train with DPOTrainer from TRL

The DPO objective (Rafailov et al. 2023) directly optimises the policy π_θ
to prefer chosen over rejected without needing an explicit reward model:

    L_DPO = -E[ log σ( β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                      - β · (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

where y_w = chosen, y_l = rejected, β controls the strength of the
regularisation towards the reference model.

Usage:
    python train_dpo.py
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

# ── Settings ──────────────────────────────────────────────────────────────────

# SFT-fine-tuned model as reference (and starting point for policy)
# Change to the adapter that's already trained:
SFT_BASE_MODEL_ID  = "microsoft/Phi-3-mini-4k-instruct"
_ROOT              = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SFT_ADAPTER_PATH   = os.path.join(_ROOT, "adapters", "phi3", "final_adapter")

DPO_DATASET_PATH   = os.path.join(_ROOT, "data", "energy_data_dpo.jsonl")
OUTPUT_DIR         = os.path.join(_ROOT, "adapters", "phi3_dpo")

# DPO hyper-params
BETA               = 0.1     # KL penalty strength (0.05–0.2 typical)
NUM_TRAIN_EPOCHS   = 1       # DPO needs fewer epochs than SFT
LEARNING_RATE      = 5e-5
PER_DEVICE_BATCH   = 1
GRAD_ACCUM_STEPS   = 8
MAX_LENGTH         = 1024    # max tokens for prompt + response
MAX_PROMPT_LENGTH  = 512
COMPUTE_DTYPE      = torch.float16

SYSTEM_PROMPT = (
    "You are a highly knowledgeable and witty expert on energy, climate, and financial markets. "
    "Your answers must be **concise, technically accurate, and highly informative**, "
    "providing the core analysis necessary to fully address the user's prompt in a focused manner. "
    "Avoid unnecessary detail."
)

# ── LoRA config for DPO policy ────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SFT_BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # required for DPO

# ── Load DPO dataset ─────────────────────────────────────────────────────────
print(f"Loading DPO dataset from {DPO_DATASET_PATH}...")
raw_dataset = load_dataset("json", data_files=DPO_DATASET_PATH, split="train")

def format_for_dpo(example):
    """Format to DPOTrainer expected format with system prompt included."""
    system = SYSTEM_PROMPT
    prompt = example["prompt"]

    # Apply chat template to the prompt part only
    messages_prompt = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": prompt},
    ]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = f"<|system|>{system}<|end|><|user|>{prompt}<|end|><|assistant|>"

    return {
        "prompt":   formatted_prompt,
        "chosen":   example["chosen"],
        "rejected": example["rejected"],
    }

dataset = raw_dataset.map(format_for_dpo, remove_columns=raw_dataset.column_names)
split   = dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── Load QLoRA base model ─────────────────────────────────────────────────────
print(f"Loading base model: {SFT_BASE_MODEL_ID}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    SFT_BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False

# Load SFT adapter → this becomes our REFERENCE model
print(f"Loading SFT adapter from {SFT_ADAPTER_PATH}...")
if os.path.isdir(SFT_ADAPTER_PATH):
    ref_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    ref_model = ref_model.merge_and_unload()
    print("SFT adapter merged as reference model.")
else:
    print("SFT adapter not found — using base model as reference.")
    ref_model = base_model

# Policy model = reference model + new LoRA (to be trained)
policy_model = prepare_model_for_kbit_training(ref_model)
policy_model = get_peft_model(policy_model, LORA_CONFIG)
policy_model.print_trainable_parameters()

# ── DPO Training Config ───────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    beta=BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fp16=(COMPUTE_DTYPE == torch.float16),
    bf16=(COMPUTE_DTYPE == torch.bfloat16),
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=50,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    remove_unused_columns=False,
    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("\n=== Starting DPO Training ===")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
os.makedirs(final_dir, exist_ok=True)
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\nDPO adapter saved to: {final_dir}")
