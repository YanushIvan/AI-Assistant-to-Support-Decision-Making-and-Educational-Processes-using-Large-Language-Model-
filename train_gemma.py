import os
# Указываем использовать только одну видеокарту (GPU 0 - NVIDIA RTX A4500)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration as _Gemma3CLS

# Патч на уровне класса: добавляем нулевые token_type_ids для text-only обучения
# (Gemma 3 требует их, чтобы отличать текстовые токены от image-токенов)
_orig_gemma3_forward = _Gemma3CLS.forward
def _patched_gemma3_forward(self, *args, **kwargs):
    if kwargs.get("token_type_ids") is None:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            kwargs["token_type_ids"] = torch.zeros_like(input_ids)
    return _orig_gemma3_forward(self, *args, **kwargs)
_Gemma3CLS.forward = _patched_gemma3_forward

# --- Настройки ---
MODEL_ID = "google/gemma-3-4b-it" # Укажите точное название модели, если оно отличается (например, google/gemma-3-12b-it)
DATASET_PATH = "energy_data.jsonl" # Имя файла датасета, как вы просили (убедитесь, что он существует, или используйте energy_data.jsonl)
OUTPUT_DIR = "./gemma3_4b_energy_finetune"

# Параметры обучения
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # 1 для 12B модели на 20GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024

def format_prompts(examples):
    """
    Функция для форматирования данных. 
    Адаптируйте под структуру вашего energy_data.json.
    Предполагается, что в JSON есть поля 'prompt' и 'completion'.
    """
    texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        # Формат для Gemma (можно адаптировать под instruct-версию)
        text = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn><eos>"
        texts.append(text)
    return {"text": texts}

def main():
    print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Загрузка датасета
    print(f"Загрузка датасета из {DATASET_PATH}...")
    # Если файл на самом деле .jsonl, измените 'json' на 'json' и укажите правильный путь
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(format_prompts, batched=True)
    dataset = dataset.train_test_split(test_size=0.05)
    train_data = dataset["train"]
    eval_data = dataset["test"]

    # 2. Настройка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Настройка квантования (4-bit) для экономии памяти (20GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Загрузка модели
    print(f"Загрузка модели {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Автоматически распределит на доступный GPU (GPU 0)
        dtype=torch.bfloat16
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 5. Настройка LoRA (PEFT)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Аргументы обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True, # Используем bfloat16, так как RTX A4500 его поддерживает
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="constant",
        report_to="none"
    )

    # 7. Инициализация тренера
    # processing_class=tokenizer — обходим AutoProcessor, который ломается на мультимодальной Gemma 3
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # 8. Запуск обучения
    print("Начало обучения...")
    trainer.train()

    # 9. Сохранение модели
    final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Обучение завершено! Адаптер сохранен в {final_dir}")

if __name__ == "__main__":
    main()
