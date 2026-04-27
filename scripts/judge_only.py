#!/usr/bin/env python3
"""
Run LLM-as-a-Judge (G-Eval) on existing evaluation results JSON.
Loads pre-computed model responses and evaluates them with Ollama (locally).

Usage:
    python scripts/judge_only.py --input eval_results/results_20260427_154336.json --model mistral
"""

import os
import json
import argparse
import re
import time
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-a-Judge (G-Eval) with Ollama
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


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_judge_client(model_name: str = "mistral") -> Optional[str]:
    """Check if Ollama is running with the specified model. Return model name if ready."""
    try:
        # Try to reach Ollama
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            log("WARNING: Ollama not responding, trying to start...")
            return None
        
        tags = response.json().get("models", [])
        model_names = [m["name"].split(":")[0] for m in tags]
        
        if model_name not in model_names:
            log(f"Model '{model_name}' not found in Ollama. Available: {model_names}")
            log(f"Pulling {model_name}... (this may take a few minutes)")
            subprocess.run(["ollama", "pull", model_name], check=False)
        
        return model_name
    except Exception as e:
        log(f"WARNING: Could not connect to Ollama: {e}")
        return None


def judge_response(
    model_name: str,
    question: str,
    response: str,
    reference: str,
) -> Optional[Dict[str, Any]]:
    """Call Ollama for judgment and parse the JSON result."""
    import requests
    
    prompt = JUDGE_TEMPLATE.format(
        question=question, response=response, reference=reference
    )
    
    try:
        api_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "system": JUDGE_SYSTEM,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=60,
        )
        
        if api_response.status_code != 200:
            log(f"  Ollama error: {api_response.status_code}")
            return None
        
        raw = api_response.json().get("response", "").strip()
        
        # Extract JSON block if wrapped in markdown
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except requests.exceptions.Timeout:
        log(f"  Timeout waiting for Ollama response")
    except Exception as e:
        log(f"  Judge error: {e}")
    
    return None


def judge_results_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    ollama_model: str = "mistral",
) -> None:
    """Load results JSON and run judge on all samples using Ollama."""
    log(f"Loading results from: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)
    
    if not isinstance(all_metrics, list):
        all_metrics = [all_metrics]
    
    log(f"Loaded {len(all_metrics)} model(s)")
    
    model_name = build_judge_client(ollama_model)
    if not model_name:
        log("ERROR: Could not connect to Ollama")
        return
    
    log(f"Starting G-Eval judge evaluation with Ollama ({model_name})...")
    
    # Evaluate each model
    for model_data in all_metrics:
        model_name_eval = model_data.get("model_name", "Unknown")
        model_key = model_data.get("model_key", "unknown")
        samples = model_data.get("samples", [])
        
        if not samples:
            log(f"  {model_name_eval}: no samples to judge")
            continue
        
        log(f"\nJudging model: {model_name_eval} ({len(samples)} samples)")
        
        scores_agg: Dict[str, List[float]] = {
            "relevance": [], "coherence": [], "fluency": [], "factual_accuracy": []
        }
        
        for i, sample in enumerate(samples):
            log(f"  Sample {i+1}/{len(samples)} ...")
            result = judge_response(
                model_name,
                sample["question"],
                sample["response"],
                sample["reference"],
            )
            
            if result and "scores" in result:
                for k in scores_agg:
                    v = result["scores"].get(k)
                    if isinstance(v, (int, float)):
                        scores_agg[k].append(float(v))
            
            time.sleep(0.3)  # rate-limit courtesy pause
        
        # Store averaged scores
        for k, vals in scores_agg.items():
            avg = sum(vals) / len(vals) if vals else -1.0
            model_data[k] = round(avg, 2)
            log(f"    {k.upper()}: {avg:.2f}")
    
    # Save updated results
    if output_path is None:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_judged.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    log(f"\nJudged results saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run G-Eval judge on existing evaluation results JSON using Ollama."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input results JSON file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to output results JSON file (default: input_judged.json)",
    )
    parser.add_argument(
        "--model", type=str, default="mistral",
        help="Ollama model name to use (default: mistral)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        log(f"ERROR: Input file not found: {input_path}")
        return
    
    output_path = Path(args.output) if args.output else None
    
    log(f"Using Ollama model: {args.model}")
    judge_results_file(input_path, output_path, args.model)


if __name__ == "__main__":
    main()
