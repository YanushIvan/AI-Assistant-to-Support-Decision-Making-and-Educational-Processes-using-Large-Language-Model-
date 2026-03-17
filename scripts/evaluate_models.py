#!/usr/bin/env python3
"""
Comprehensive evaluation script comparing all SLMs (base vs fine-tuned).

Metrics per model:
  - G-Eval LLM-as-a-Judge: Relevance, Coherence, Fluency, Factual Accuracy (1–10)
  - Tokens Per Second (TPS)
  - Time to First Token (TTFT)
  - VRAM Memory Usage (at load + per context length)
  - Perplexity on held-out test samples

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --n-samples 50 --judge --output-dir eval_results
    python scripts/evaluate_models.py --models phi-3-base phi-3-ft  # evaluate only specific models
"""

import os
import sys
import re
import json
import math
import time
import random
import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import Thread

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "phi-3-base": {
        "name": "Phi-3 Mini 4K (Base)",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_path": None,
        "is_qwen3": False,
    },
    "phi-3-ft": {
        "name": "Phi-3 Mini 4K (Fine-tuned)",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_path": str(
            PROJECT_ROOT / "phi3_mini_energy_finetune_structured" / "final_adapter"
        ),
        "is_qwen3": False,
    },
    "gemma-3-base": {
        "name": "Gemma 3 4B IT (Base)",
        "model_id": "google/gemma-3-4b-it",
        "adapter_path": None,
        "is_qwen3": False,
    },
    "gemma-3-ft": {
        "name": "Gemma 3 4B IT (Fine-tuned)",
        "model_id": "google/gemma-3-4b-it",
        "adapter_path": str(
            PROJECT_ROOT / "gemma3_4b_energy_finetune_structured" / "final_adapter"
        ),
        "is_qwen3": False,
    },
    "qwen-3-base": {
        "name": "Qwen3 4B Instruct (Base)",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter_path": None,
        "is_qwen3": True,
    },
    "qwen-3-ft": {
        "name": "Qwen3 4B Instruct (Fine-tuned)",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter_path": str(
            PROJECT_ROOT / "qwen3_4b_energy_finetune_structured" / "final_adapter"
        ),
        "is_qwen3": True,
    },
}

# Context lengths (tokens) to probe for VRAM scaling
VRAM_PROBE_LENGTHS = [128, 256, 512, 1024]

# Generation settings (no system prompt — raw model comparison)
GEN_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    question: str
    reference: str
    response: str
    ttft_s: float          # Time to First Token (seconds)
    tps: float             # Tokens Per Second
    output_tokens: int
    vram_during_gen_mb: float


@dataclass
class ModelMetrics:
    model_key: str
    model_name: str
    vram_load_mb: float    # VRAM right after model load
    vram_reserved_mb: float
    perplexity: float
    avg_ttft_s: float
    avg_tps: float
    avg_output_tokens: float
    vram_by_ctx_mb: Dict[int, float] = field(default_factory=dict)

    # G-Eval scores (averaged over all samples, -1 = not computed)
    relevance: float = -1.0
    coherence: float = -1.0
    fluency: float = -1.0
    factual_accuracy: float = -1.0

    samples: List[SampleResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def vram_allocated_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def vram_reserved_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0.0


def free_vram() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"WARNING: CUDA cleanup failed ({e}) — continuing")


def load_dataset(path: Path) -> List[Dict[str, str]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_stop_token_ids(tokenizer, model_key: str) -> List[int]:
    ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    k = model_key.lower()
    if "phi" in k:
        ids.append(32007)          # <|end|>
    elif "qwen" in k:
        ids.extend([151645, 151643])  # <|im_end|>, <|endoftext|>
    elif "gemma" in k:
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot and eot != tokenizer.unk_token_id:
            ids.append(eot)
    return list(set(ids))


def clean_response(text: str) -> str:
    """Strip special tokens and Qwen3 thinking blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for tok in ["<|end|>", "<|endoftext|>", "<|im_end|>", "<end_of_turn>", "<eos>"]:
        text = text.replace(tok, "")
    return text.strip()


def format_messages(question: str, tokenizer, is_qwen3: bool) -> str:
    """Apply chat template without system prompt (bare model comparison)."""
    messages = [{"role": "user", "content": question}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if is_qwen3:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading / unloading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    cfg: Dict[str, Any],
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_id = cfg["model_id"]
    adapter_path = cfg["adapter_path"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False

    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        model = base

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    del model
    del tokenizer
    free_vram()


# ─────────────────────────────────────────────────────────────────────────────
# Inference + TPS + TTFT
# ─────────────────────────────────────────────────────────────────────────────

def _generate_in_thread(model, kwargs: dict, errors: list) -> None:
    """Thread target — captures exceptions and unblocks the streamer on failure."""
    try:
        model.generate(**kwargs)
    except Exception as exc:
        errors.append(exc)
        streamer = kwargs.get("streamer")
        if streamer is not None:
            try:
                streamer.end()
            except Exception:
                pass


def generate_with_timing(
    model,
    tokenizer,
    prompt_text: str,
    stop_token_ids: List[int],
) -> Tuple[str, float, float, int]:
    """
    Returns (response_text, ttft_seconds, tps, output_tokens).
    Uses TextIteratorStreamer to capture TTFT independently of total time.
    Raises RuntimeError if the generation thread crashes.
    """
    tokenized = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=False
    )

    gen_kwargs = dict(
        **GEN_CONFIG,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_token_ids,
        streamer=streamer,
        use_cache=False,
    )

    errors: list = []
    thread = Thread(
        target=_generate_in_thread,
        args=(model, gen_kwargs, errors),
        daemon=True,
    )
    t_start = time.perf_counter()
    thread.start()

    ttft: Optional[float] = None
    chunks: List[str] = []
    for chunk in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t_start
        chunks.append(chunk)

    thread.join()
    t_end = time.perf_counter()

    if errors:
        raise RuntimeError(f"Generation thread failed: {errors[0]}") from errors[0]

    raw = "".join(chunks)
    response = clean_response(raw)
    total_time = t_end - t_start

    output_tokens = len(tokenizer.encode(response, add_special_tokens=False))
    tps = output_tokens / total_time if total_time > 0 else 0.0
    ttft = ttft if ttft is not None else total_time

    return response, ttft, tps, output_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    tokenizer,
    samples: List[Dict[str, str]],
    max_length: int = 1024,
) -> float:
    """
    Compute mean perplexity over `samples`.
    Loss is computed only on completion (assistant) tokens.
    """
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sample in samples:
            prompt = sample["prompt"]
            full_text = prompt + "\n" + sample["completion"]

            prompt_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )["input_ids"]
            prompt_len = prompt_ids.shape[-1]

            full_ids = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )["input_ids"].to(model.device)

            labels = full_ids.clone()
            labels[:, :prompt_len] = -100  # mask prompt tokens

            n_completion_tokens = (labels != -100).sum().item()
            if n_completion_tokens == 0:
                continue

            try:
                out = model(full_ids, labels=labels)
                loss = out.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_nll += loss.item() * n_completion_tokens
                total_tokens += n_completion_tokens
            except Exception:
                continue

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# VRAM vs context length profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_vram_by_context(
    model,
    tokenizer,
    probe_lengths: List[int],
) -> Dict[int, float]:
    """
    Run a forward pass with a dummy input of each length and record peak VRAM.
    """
    results: Dict[int, float] = {}
    dummy_text = "energy " * 200  # long enough to truncate at any probe length

    for length in probe_lengths:
        try:
            free_vram()
            torch.cuda.reset_peak_memory_stats()

            ids = tokenizer(
                dummy_text,
                return_tensors="pt",
                truncation=True,
                max_length=length,
                padding="max_length",
            )["input_ids"].to(model.device)

            with torch.no_grad():
                model(ids)

            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            results[length] = round(peak_mb, 1)
            log(f"  VRAM @ ctx={length}: {peak_mb:.1f} MB")
        except Exception as e:
            log(f"  VRAM @ ctx={length}: failed ({e})")
            results[length] = -1.0

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-a-Judge (G-Eval)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert evaluator for an AI assistant that answers questions "
    "about energy markets, climate policy, and financial regulation."
)

JUDGE_TEMPLATE = """\
Evaluate the following AI response to an energy-domain question.

QUESTION:
{question}

AI RESPONSE:
{response}

REFERENCE ANSWER:
{reference}

Score each criterion from 1 (worst) to 10 (best). Think step by step before scoring.

Criteria:
1. RELEVANCE     — Does the response address the question directly and stay on topic?
2. COHERENCE     — Is the response logically structured and internally consistent?
3. FLUENCY       — Is the language natural, grammatically correct, and easy to read?
4. FACTUAL_ACCURACY — Are the facts, statistics, and claims accurate?

Respond ONLY with valid JSON in this exact format:
{{
  "reasoning": {{
    "relevance": "<your reasoning>",
    "coherence": "<your reasoning>",
    "fluency": "<your reasoning>",
    "factual_accuracy": "<your reasoning>"
  }},
  "scores": {{
    "relevance": <1-10>,
    "coherence": <1-10>,
    "fluency": <1-10>,
    "factual_accuracy": <1-10>
  }}
}}"""


def build_judge_client() -> Optional[Any]:
    """Return an Anthropic client if ANTHROPIC_API_KEY is set, else None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        log("WARNING: anthropic package not installed. Skipping G-Eval judge.")
        return None


def judge_response(
    client,
    question: str,
    response: str,
    reference: str,
) -> Optional[Dict[str, Any]]:
    """Call the judge model and parse the JSON result."""
    prompt = JUDGE_TEMPLATE.format(
        question=question, response=response, reference=reference
    )
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Extract JSON block if wrapped in markdown
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        log(f"  Judge error: {e}")
    return None


def run_judge_evaluation(
    client,
    model_metrics: ModelMetrics,
) -> None:
    """Evaluate all samples with the judge and store averaged scores."""
    scores_agg: Dict[str, List[float]] = {
        "relevance": [], "coherence": [], "fluency": [], "factual_accuracy": []
    }

    for i, s in enumerate(model_metrics.samples):
        log(f"  Judging sample {i+1}/{len(model_metrics.samples)} ...")
        result = judge_response(client, s.question, s.response, s.reference)
        if result and "scores" in result:
            for k in scores_agg:
                v = result["scores"].get(k)
                if isinstance(v, (int, float)):
                    scores_agg[k].append(float(v))
        time.sleep(0.3)  # rate-limit courtesy pause

    for k, vals in scores_agg.items():
        avg = sum(vals) / len(vals) if vals else -1.0
        setattr(model_metrics, k, round(avg, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    cfg: Dict[str, Any],
    test_samples: List[Dict[str, str]],
    perplexity_samples: List[Dict[str, str]],
    probe_vram: bool = True,
) -> ModelMetrics:
    log(f"\n{'='*60}")
    log(f"Evaluating: {cfg['name']} ({model_key})")
    log(f"{'='*60}")

    log("Loading model...")
    free_vram()
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_and_tokenizer(cfg)
    vram_load = vram_allocated_mb()
    vram_res = vram_reserved_mb()
    log(f"Model loaded — VRAM allocated: {vram_load:.1f} MB, reserved: {vram_res:.1f} MB")

    stop_ids = build_stop_token_ids(tokenizer, model_key)

    # ── Perplexity ────────────────────────────────────────────────────────────
    log(f"Computing perplexity on {len(perplexity_samples)} samples...")
    ppl = compute_perplexity(model, tokenizer, perplexity_samples)
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            log(f"WARNING: CUDA sync after perplexity failed: {e}")
    log(f"Perplexity: {ppl:.2f}")

    # ── VRAM vs context length ────────────────────────────────────────────────
    vram_by_ctx: Dict[int, float] = {}
    if probe_vram and torch.cuda.is_available():
        log("Profiling VRAM vs context length...")
        vram_by_ctx = profile_vram_by_context(model, tokenizer, VRAM_PROBE_LENGTHS)
        try:
            torch.cuda.synchronize()
        except Exception as e:
            log(f"WARNING: CUDA sync after VRAM profile failed: {e}")

    # ── Inference (TPS + TTFT) ────────────────────────────────────────────────
    sample_results: List[SampleResult] = []
    for i, sample in enumerate(test_samples):
        question = sample["prompt"]
        reference = sample["completion"]
        log(f"  Sample {i+1}/{len(test_samples)}: {question[:60]}...")

        prompt_text = format_messages(question, tokenizer, cfg["is_qwen3"])
        free_vram()

        try:
            response, ttft, tps, n_tokens = generate_with_timing(
                model, tokenizer, prompt_text, stop_ids
            )
        except Exception as e:
            log(f"  WARNING: generation failed for sample {i+1}: {e} — skipping")
            free_vram()
            continue

        vram_gen = vram_allocated_mb()

        log(f"    TTFT={ttft:.3f}s  TPS={tps:.1f}  tokens={n_tokens}  VRAM={vram_gen:.0f}MB")
        sample_results.append(
            SampleResult(
                question=question,
                reference=reference,
                response=response,
                ttft_s=round(ttft, 4),
                tps=round(tps, 2),
                output_tokens=n_tokens,
                vram_during_gen_mb=round(vram_gen, 1),
            )
        )

    if not sample_results:
        log("ERROR: all samples failed for this model — recording zeros")
        sample_results = []

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(sample_results)
    avg_ttft = sum(s.ttft_s for s in sample_results) / n if n else 0.0
    avg_tps = sum(s.tps for s in sample_results) / n if n else 0.0
    avg_tokens = sum(s.output_tokens for s in sample_results) / n if n else 0.0

    metrics = ModelMetrics(
        model_key=model_key,
        model_name=cfg["name"],
        vram_load_mb=round(vram_load, 1),
        vram_reserved_mb=round(vram_res, 1),
        perplexity=round(ppl, 3),
        avg_ttft_s=round(avg_ttft, 4),
        avg_tps=round(avg_tps, 2),
        avg_output_tokens=round(avg_tokens, 1),
        vram_by_ctx_mb=vram_by_ctx,
        samples=sample_results,
    )

    log(f"Done — avg TTFT={avg_ttft:.3f}s  avg TPS={avg_tps:.1f}  PPL={ppl:.2f}")

    unload_model(model, tokenizer)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Output / reporting
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    all_metrics: List[ModelMetrics],
    output_dir: Path,
    run_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON — full detail
    json_path = output_dir / f"results_{run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        data = []
        for m in all_metrics:
            d = asdict(m)
            data.append(d)
        json.dump(data, f, indent=2, ensure_ascii=False)
    log(f"\nFull results saved → {json_path}")

    # CSV — summary table
    csv_path = output_dir / f"summary_{run_id}.csv"
    fieldnames = [
        "model_key", "model_name",
        "perplexity", "avg_ttft_s", "avg_tps", "avg_output_tokens",
        "vram_load_mb", "vram_reserved_mb",
        "relevance", "coherence", "fluency", "factual_accuracy",
    ] + [f"vram_ctx_{l}" for l in VRAM_PROBE_LENGTHS]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            row = {
                "model_key": m.model_key,
                "model_name": m.model_name,
                "perplexity": m.perplexity,
                "avg_ttft_s": m.avg_ttft_s,
                "avg_tps": m.avg_tps,
                "avg_output_tokens": m.avg_output_tokens,
                "vram_load_mb": m.vram_load_mb,
                "vram_reserved_mb": m.vram_reserved_mb,
                "relevance": m.relevance if m.relevance >= 0 else "N/A",
                "coherence": m.coherence if m.coherence >= 0 else "N/A",
                "fluency": m.fluency if m.fluency >= 0 else "N/A",
                "factual_accuracy": m.factual_accuracy if m.factual_accuracy >= 0 else "N/A",
            }
            for l in VRAM_PROBE_LENGTHS:
                row[f"vram_ctx_{l}"] = m.vram_by_ctx_mb.get(l, "N/A")
            writer.writerow(row)
    log(f"Summary CSV saved → {csv_path}")


def print_comparison_table(all_metrics: List[ModelMetrics]) -> None:
    col_w = 30
    num_w = 10

    header = (
        f"{'Model':<{col_w}}"
        f"{'PPL':>{num_w}}"
        f"{'TTFT(s)':>{num_w}}"
        f"{'TPS':>{num_w}}"
        f"{'VRAM(MB)':>{num_w}}"
        f"{'Rel':>{num_w}}"
        f"{'Coh':>{num_w}}"
        f"{'Flu':>{num_w}}"
        f"{'FactAcc':>{num_w}}"
    )

    print("\n" + "=" * (col_w + num_w * 8))
    print("EVALUATION RESULTS — MODEL COMPARISON")
    print("=" * (col_w + num_w * 8))
    print(header)
    print("-" * (col_w + num_w * 8))

    def fmt_score(v: float) -> str:
        return f"{v:.2f}" if v >= 0 else "N/A"

    for m in all_metrics:
        row = (
            f"{m.model_name:<{col_w}}"
            f"{m.perplexity:>{num_w}.2f}"
            f"{m.avg_ttft_s:>{num_w}.3f}"
            f"{m.avg_tps:>{num_w}.1f}"
            f"{m.vram_load_mb:>{num_w}.0f}"
            f"{fmt_score(m.relevance):>{num_w}}"
            f"{fmt_score(m.coherence):>{num_w}}"
            f"{fmt_score(m.fluency):>{num_w}}"
            f"{fmt_score(m.factual_accuracy):>{num_w}}"
        )
        print(row)

    print("=" * (col_w + num_w * 8))
    print()

    # VRAM by context
    print("VRAM (MB) by context length:")
    ctx_header = f"{'Model':<{col_w}}" + "".join(
        f"{f'ctx={l}':>{num_w}}" for l in VRAM_PROBE_LENGTHS
    )
    print(ctx_header)
    print("-" * (col_w + num_w * len(VRAM_PROBE_LENGTHS)))
    for m in all_metrics:
        ctx_row = f"{m.model_name:<{col_w}}"
        for l in VRAM_PROBE_LENGTHS:
            v = m.vram_by_ctx_mb.get(l, None)
            ctx_row += f"{str(round(v, 0)) if v else 'N/A':>{num_w}}"
        print(ctx_row)
    print("=" * (col_w + num_w * len(VRAM_PROBE_LENGTHS)))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare base vs fine-tuned SLMs."
    )
    parser.add_argument(
        "--n-samples", type=int, default=30,
        help="Number of test samples for inference evaluation (default: 30)",
    )
    parser.add_argument(
        "--n-ppl-samples", type=int, default=50,
        help="Number of samples for perplexity computation (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-a-judge G-Eval (requires ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--no-vram-profile", action="store_true",
        help="Skip VRAM-vs-context profiling (saves time)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_results",
        help="Directory to save results (default: eval_results)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS_CONFIG.keys()),
        default=list(MODELS_CONFIG.keys()),
        help="Which models to evaluate (default: all)",
    )
    parser.add_argument(
        "--dataset", type=str,
        default=str(PROJECT_ROOT / "energy_data_structured.jsonl"),
        help="Path to JSONL dataset file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log(f"Run ID: {run_id}")
    log(f"Models to evaluate: {args.models}")
    log(f"Inference samples: {args.n_samples}")
    log(f"Perplexity samples: {args.n_ppl_samples}")
    log(f"G-Eval judge: {'yes' if args.judge else 'no'}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    # Load dataset
    log(f"\nLoading dataset: {args.dataset}")
    all_data = load_dataset(Path(args.dataset))
    log(f"Total samples: {len(all_data)}")

    rng = random.Random(args.seed)
    shuffled = all_data.copy()
    rng.shuffle(shuffled)

    # Separate splits (no overlap between inference and perplexity)
    n_total = args.n_samples + args.n_ppl_samples
    if n_total > len(shuffled):
        n_total = len(shuffled)
        args.n_samples = min(args.n_samples, n_total // 2)
        args.n_ppl_samples = n_total - args.n_samples

    test_samples = shuffled[: args.n_samples]
    ppl_samples = shuffled[args.n_samples : args.n_samples + args.n_ppl_samples]
    log(f"Test samples: {len(test_samples)}, Perplexity samples: {len(ppl_samples)}")

    # Judge client
    judge_client = None
    if args.judge:
        judge_client = build_judge_client()
        if judge_client:
            log("LLM-as-a-judge (G-Eval) enabled with claude-haiku-4-5")
        else:
            log("WARNING: ANTHROPIC_API_KEY not set — skipping G-Eval")

    # Evaluate each model
    all_metrics: List[ModelMetrics] = []
    for model_key in args.models:
        cfg = MODELS_CONFIG[model_key]
        metrics = evaluate_model(
            model_key=model_key,
            cfg=cfg,
            test_samples=test_samples,
            perplexity_samples=ppl_samples,
            probe_vram=not args.no_vram_profile,
        )
        all_metrics.append(metrics)

    # G-Eval judge evaluation (after all models are unloaded — saves VRAM)
    if judge_client:
        log("\n" + "=" * 60)
        log("Running G-Eval LLM-as-a-Judge evaluation...")
        log("=" * 60)
        for metrics in all_metrics:
            log(f"\nJudging model: {metrics.model_name}")
            run_judge_evaluation(judge_client, metrics)

    # Save + display
    output_dir = PROJECT_ROOT / args.output_dir
    save_results(all_metrics, output_dir, run_id)
    print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()
