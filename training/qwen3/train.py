"""Qwen3-4B fine-tuning entry point.

Run from project root:
    python training/qwen3/train.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer

from data_loader import load_and_prepare_data
from config import (
    MODEL_ID, DATASET_PATH, OUTPUT_DIR, SYSTEM_PROMPT,
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, MAX_SEQ_LENGTH, TEST_SIZE,
    LOAD_IN_4BIT, BNB_QUANT_TYPE, COMPUTE_DTYPE, LORA_CONFIG,
)


def main():
    # 1. Data
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data(
        model_id=MODEL_ID,
        dataset_path=DATASET_PATH,
        system_prompt=SYSTEM_PROMPT,
        max_seq_length=MAX_SEQ_LENGTH,
        test_size=TEST_SIZE,
    )

    # 2. Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Model
    print(f"Loading {MODEL_ID} in QLoRA 4-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=COMPUTE_DTYPE,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False
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
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="no",
        fp16=(COMPUTE_DTYPE == torch.float16),
        bf16=(COMPUTE_DTYPE == torch.bfloat16),
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LORA_CONFIG,
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
