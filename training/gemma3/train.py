"""Gemma-3 4B fine-tuning entry point.

Applies a token_type_ids patch required for text-only training on Gemma-3
(which is a multimodal model that expects those ids to separate text from image tokens).

Run from project root:
    python training/gemma3/train.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

# ── Gemma-3 patch — must happen before model import ───────────────────────────
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration as _Gemma3

_orig_forward = _Gemma3.forward

def _patched_forward(self, *args, **kwargs):
    if kwargs.get("token_type_ids") is None:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            kwargs["token_type_ids"] = torch.zeros_like(input_ids)
    return _orig_forward(self, *args, **kwargs)

_Gemma3.forward = _patched_forward
# ─────────────────────────────────────────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer

from data_loader import load_and_prepare_data
from config import (
    MODEL_ID, DATASET_PATH, OUTPUT_DIR, SYSTEM_PROMPT,
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, MAX_SEQ_LENGTH, TEST_SIZE,
    LOAD_IN_4BIT, BNB_QUANT_TYPE, COMPUTE_DTYPE, USE_DOUBLE_QUANT, LORA_CONFIG,
)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Data (shared masked-labels data loader)
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data(
        model_id=MODEL_ID,
        dataset_path=DATASET_PATH,
        system_prompt=SYSTEM_PROMPT,
        max_seq_length=MAX_SEQ_LENGTH,
        test_size=TEST_SIZE,
    )
    tokenizer.padding_side = "right"

    # 2. Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )

    # 3. Model
    print(f"Loading {MODEL_ID} in QLoRA 4-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=COMPUTE_DTYPE,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        optim="paged_adamw_8bit",
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="constant",
        report_to="none",
    )

    # processing_class=tokenizer bypasses AutoProcessor (broken for text-only Gemma-3)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LORA_CONFIG,
        processing_class=tokenizer,
    )

    print(f"--- Starting fine-tuning: {MODEL_ID} ---")
    trainer.train()

    # 5. Save adapter
    final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
