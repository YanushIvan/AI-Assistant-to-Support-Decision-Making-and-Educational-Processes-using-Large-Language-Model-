import torch
from datasets import load_dataset
from transformers import AutoTokenizer 
from config import MODEL_ID, MAX_SEQ_LENGTH, DATASET_PATH, TEST_SIZE, SYSTEM_PROMPT

def create_masked_labels(example, tokenizer):
    """
    Создает `input_ids` и `labels`, маскируя токены `<|system|>` и `<|user|>`
    (устанавливая их в -100). Модель будет учиться только на ответах ассистента.
    """
    
    # 1. Формируем полную историю чата
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    
    # 2. Получаем текст через apply_chat_template (без токенизации)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Токенизируем с паддингом отдельно
    pad_id = tokenizer.pad_token_id
    encoded = tokenizer(
        full_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].squeeze(0)

    # Create attention mask (all non-padded tokens)
    attention_mask = encoded["attention_mask"].squeeze(0)

    # Initialize labels as copy of input_ids
    labels = input_ids.clone()

    # --- ЛОГИКА МАСКИРОВАНИЯ ---
    
    # Создаем текст только до начала ответа ассистента (промпт)
    prompt_only_messages = messages[:-1] 
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )
    
    # Токенизируем только промпт, чтобы найти его длину
    prompt_tokenized = tokenizer(
        prompt_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
    )

    # Get the length of prompt tokens (it's a list)
    prompt_len = len(prompt_tokenized["input_ids"])
    
    # Устанавливаем labels в -100 для всех токенов, которые являются частью промпта
    if prompt_len < labels.shape[0]: 
        labels[:prompt_len] = -100
        
    # Обрабатываем padding токены (если есть), устанавливая их в -100
    labels[labels == pad_id] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_and_prepare_data():
    """
    Основная функция, которая загружает, форматирует и токенизирует обучающие данные.
    """
    # 1. Загрузка Данных
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 2. Инициализация Токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    # 3. Форматирование, Токенизация и Маскирование (в одном шаге)
    dataset = dataset.map(
        lambda x: create_masked_labels(x, tokenizer), 
        batched=False, 
        remove_columns=["prompt", "completion"] 
    ) 

    split_dataset = dataset.train_test_split(test_size=TEST_SIZE)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Теперь датасеты содержат input_ids, attention_mask и labels
    return train_dataset, eval_dataset, tokenizer