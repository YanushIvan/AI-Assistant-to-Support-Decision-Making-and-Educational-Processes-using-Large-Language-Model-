import torch
import os
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import prepare_model_for_kbit_training 
from trl import SFTTrainer

# Импортируем все настройки и функцию загрузки данных
from config import *
from data_loader import load_and_prepare_data

def run_fine_tuning():
    # --- 1. Подготовка Данных и Токенизатора ---
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data()

    # --- 2. Настройка Quantization (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=False,
    )

    # --- 3. Загрузка и Подготовка Модели ---
    print(f"Загрузка модели {MODEL_ID} в режиме QLoRA (4-bit)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # ИСПРАВЛЕНИЕ: Используем 'dtype' вместо 'torch_dtype'
        dtype=COMPUTE_DTYPE,
        # Добавляем revision для принудительной загрузки стабильной версии
        revision="main", 
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 4. Применение оптимизации KBit
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- 5. Настройка Аргументов Обучения ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        fp16=(COMPUTE_DTYPE == torch.float16), 
        bf16=(COMPUTE_DTYPE == torch.bfloat16),
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # --- 6. Создание и Запуск SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        peft_config=LORA_CONFIG,
        #tokenizer=tokenizer, 
        #max_seq_length=MAX_SEQ_LENGTH, 
        #dataset_text_field="text" 
    )

    print(f"--- Начинаем Fine-Tuning {MODEL_ID} ---")
    trainer.train()

    # --- 7. Сохранение Адаптера ---
    final_adapter_output_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    os.makedirs(final_adapter_output_dir, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_output_dir)
    tokenizer.save_pretrained(final_adapter_output_dir)
    print(f"Обученный LoRA адаптер сохранен в: {final_adapter_output_dir}")

if __name__ == "__main__":
    run_fine_tuning()