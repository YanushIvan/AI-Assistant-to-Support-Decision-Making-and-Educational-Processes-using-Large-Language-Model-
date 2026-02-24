import torch
from datasets import load_dataset
from transformers import AutoTokenizer 
from config import MODEL_ID, MAX_SEQ_LENGTH, DATASET_PATH, TEST_SIZE, SYSTEM_PROMPT

def create_masked_labels(example, tokenizer):
    """
    Создает `input_ids` и `labels`, маскируя токены системного промпта и пользователя.
    Модель учится только на ответах ассистента.
    """
    
    # 1. Формируем полную историю чата
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    
    # 2. Токенизируем полную последовательность
    full_input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Убираем размерность батча: (1, L) -> (L)
    input_ids = full_input_ids.squeeze(0)
    
    # Маска внимания (все, что не является паддингом)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Инициализируем labels как копию input_ids
    labels = input_ids.clone()

    # --- ЛОГИКА МАСКИРОВАНИЯ ---
    
    # Берем текст до начала ответа ассистента
    prompt_only_messages = messages[:-1] 
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )
    
    # Токенизируем только промпт, чтобы узнать его длину
    prompt_tokenized = tokenizer(
        prompt_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
    )

    prompt_len = len(prompt_tokenized["input_ids"])
    
    # Маскируем (-100) всё, что относится к промпту
    if prompt_len < labels.shape[0]: 
        labels[:prompt_len] = -100
        
    # Маскируем паддинг токены
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_and_prepare_data():
    """
    Основная функция загрузки и подготовки данных.
    """
    # 1. Загрузка
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 2. Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    # Qwen обычно не имеет pad_token по умолчанию, назначаем EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    # 3. Маппинг
    dataset = dataset.map(
        lambda x: create_masked_labels(x, tokenizer), 
        batched=False, 
        remove_columns=["prompt", "completion"] 
    ) 

    split_dataset = dataset.train_test_split(test_size=TEST_SIZE)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return train_dataset, eval_dataset, tokenizer