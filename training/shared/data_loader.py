"""Shared data loading and preparation for all model training.

Usage from a model-specific train.py:
    from training.shared.data_loader import load_and_prepare_data
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def _create_masked_labels(example, tokenizer, system_prompt, max_seq_length):
    """
    Creates input_ids and labels, masking system+user tokens so the model
    only learns from assistant responses (the completion part).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]

    # Full conversation text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    pad_id = tokenizer.pad_token_id
    encoded = tokenizer(
        full_text,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    labels         = input_ids.clone()

    # Find where the assistant response starts (mask everything before)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],          # system + user only
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_len = len(
        tokenizer(prompt_text, max_length=max_seq_length, truncation=True)["input_ids"]
    )

    if prompt_len < labels.shape[0]:
        labels[:prompt_len] = -100   # mask system + user tokens

    labels[labels == pad_id] = -100  # mask padding

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_and_prepare_data(
    model_id: str,
    dataset_path: str,
    system_prompt: str,
    max_seq_length: int = 1024,
    test_size: float = 0.01,
) -> tuple:
    """
    Load, tokenize, and mask a JSONL dataset with {"prompt", "completion"} records.

    Returns:
        (train_dataset, eval_dataset, tokenizer)
    """
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: _create_masked_labels(x, tokenizer, system_prompt, max_seq_length),
        batched=False,
        remove_columns=["prompt", "completion"],
    )

    split = dataset.train_test_split(test_size=test_size)
    return split["train"], split["test"], tokenizer
