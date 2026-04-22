"""Train DPO adapters for Phi-3, Gemma-3, or Qwen3.

Usage examples:
    python scripts/train_dpo.py --model phi3
    python scripts/train_dpo.py --model gemma3
    python scripts/train_dpo.py --model qwen3 --dataset data/energy_data_dpo.jsonl
"""

import argparse
import os


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SYSTEM_PROMPT = (
    "You are a highly knowledgeable expert on energy, climate, and financial markets. "
    "Always respond in the following structured format:\n\n"
    "ANSWER: <one-sentence direct answer>\n\n"
    "KEY FACTS:\n• <specific fact with data>\n• <specific fact with data>\n• <specific fact with data>\n\n"
    "RISK LEVEL: Low / Medium / High\n→ <one-sentence explanation>\n\n"
    "CONFIDENCE: High / Medium / Low\n→ <reason, e.g. based on IRENA 2025 data / estimated>\n\n"
    "Be concise, technically accurate, and use specific numbers where available."
)

MODEL_PRESETS = {
    "phi3": {
        "base_model_id": "microsoft/Phi-3-mini-4k-instruct",
        "sft_adapter_path": os.path.join(_ROOT, "phi3_mini_energy_finetune_structured", "final_adapter"),
        "output_dir": os.path.join(_ROOT, "phi3_mini_energy_dpo"),
        "compute_dtype": "bfloat16",
        "use_double_quant": False,
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "trust_remote_code": True,
    },
    "gemma3": {
        "base_model_id": "google/gemma-3-4b-it",
        "sft_adapter_path": os.path.join(_ROOT, "gemma3_4b_energy_finetune_structured", "final_adapter"),
        "output_dir": os.path.join(_ROOT, "gemma3_4b_energy_dpo"),
        "compute_dtype": "bfloat16",
        "use_double_quant": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "trust_remote_code": False,
    },
    "qwen3": {
        "base_model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "sft_adapter_path": os.path.join(_ROOT, "qwen3_4b_energy_finetune_structured", "final_adapter"),
        "output_dir": os.path.join(_ROOT, "qwen3_4b_energy_dpo"),
        "compute_dtype": "bfloat16",
        "use_double_quant": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "trust_remote_code": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for energy models")
    parser.add_argument("--model", choices=MODEL_PRESETS.keys(), required=True, help="Model family preset")
    parser.add_argument(
        "--dataset",
        default=os.path.join(_ROOT, "data", "energy_data_dpo.jsonl"),
        help="Path to DPO dataset JSONL (prompt/chosen/rejected)",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config and required paths, then exit without loading models",
    )
    return parser.parse_args()


def maybe_apply_gemma_patch(model_key: str):
    if model_key != "gemma3":
        return

    import torch
    # Gemma-3 can require token_type_ids even for text-only batches.
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration as _Gemma3

    original_forward = _Gemma3.forward

    def patched_forward(self, *args, **kwargs):
        if kwargs.get("token_type_ids") is None:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)
        return original_forward(self, *args, **kwargs)

    _Gemma3.forward = patched_forward


def build_bnb_config(compute_dtype_name: str, use_double_quant: bool):
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if compute_dtype_name == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def load_quantized_base(base_model_id: str, bnb_config, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    return model


def merge_sft_adapter(base_model, adapter_path: str):
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    return model


def main():
    args = parse_args()
    cfg = MODEL_PRESETS[args.model]

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"DPO dataset not found: {args.dataset}. Run scripts/create_dpo_dataset.py first."
        )
    if not os.path.isdir(cfg["sft_adapter_path"]):
        raise FileNotFoundError(f"SFT adapter path does not exist: {cfg['sft_adapter_path']}")

    print(f"=== DPO preset: {args.model} ===")
    print(f"Base model:  {cfg['base_model_id']}")
    print(f"SFT adapter: {cfg['sft_adapter_path']}")
    print(f"Output dir:  {cfg['output_dir']}")

    if args.check_only:
        print("Check passed: dataset and SFT adapter paths exist.")
        return

    maybe_apply_gemma_patch(args.model)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_id"],
        trust_remote_code=cfg["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading DPO dataset from {args.dataset}...")
    raw_dataset = load_dataset("json", data_files=args.dataset, split="train")
    if len(raw_dataset) == 0:
        raise ValueError(
            "DPO dataset is empty. Expected JSONL rows with prompt/chosen/rejected. "
            f"Got zero rows from: {args.dataset}"
        )
    expected_cols = {"prompt", "chosen", "rejected"}
    if not expected_cols.issubset(set(raw_dataset.column_names)):
        raise ValueError(
            f"Dataset must contain columns {sorted(expected_cols)}; got {raw_dataset.column_names}"
        )

    def format_for_dpo(example):
        messages_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = (
                f"<|system|>{SYSTEM_PROMPT}<|end|><|user|>{example['prompt']}<|end|><|assistant|>"
            )

        return {
            "prompt": formatted_prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    dataset = raw_dataset.map(format_for_dpo, remove_columns=raw_dataset.column_names)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    bnb_config = build_bnb_config(
        compute_dtype_name=cfg["compute_dtype"],
        use_double_quant=cfg["use_double_quant"],
    )

    print("Loading reference model...")
    ref_base = load_quantized_base(
        base_model_id=cfg["base_model_id"],
        bnb_config=bnb_config,
        trust_remote_code=cfg["trust_remote_code"],
    )
    ref_model = merge_sft_adapter(ref_base, cfg["sft_adapter_path"])
    if args.model == "gemma3":
        # Force text-only preprocessing in TRL DPOTrainer for text-only datasets.
        ref_model.config.model_type = "gemma3_text"
    ref_model.eval()
    ref_model.requires_grad_(False)

    print("Loading policy model...")
    policy_base = load_quantized_base(
        base_model_id=cfg["base_model_id"],
        bnb_config=bnb_config,
        trust_remote_code=cfg["trust_remote_code"],
    )
    policy_merged = merge_sft_adapter(policy_base, cfg["sft_adapter_path"])
    policy_merged = prepare_model_for_kbit_training(policy_merged)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_merged, lora_config)
    if args.model == "gemma3":
        # Gemma-3 is multimodal, but this DPO dataset has no images column.
        policy_model.config.model_type = "gemma3_text"
    policy_model.print_trainable_parameters()

    dpo_config = DPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        fp16=(cfg["compute_dtype"] == "float16"),
        bf16=(cfg["compute_dtype"] == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=50,
        optim="paged_adamw_8bit",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\n=== Starting DPO training ===")
    trainer.train()

    final_dir = os.path.join(cfg["output_dir"], "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDPO adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
