"""
Specialized Format Dataset Creator
=====================================
Converts the existing energy Q&A dataset into a structured format that the model
is trained to always produce. This creates a VISIBLY different output compared
to the base model — the base model never produces this exact structure.

Structured output format:
─────────────────────────
ANSWER: <one-sentence direct answer>

KEY FACTS:
• <specific fact with number/statistic>
• <specific fact with number/statistic>
• <specific fact with number/statistic>

RISK LEVEL: Low / Medium / High
→ <one-sentence explanation of the main risk>

CONFIDENCE: High / Medium / Low
→ <reason — e.g. based on IRENA 2025 data / estimated / uncertain>
─────────────────────────

Why this works:
- Base model NEVER produces this structure (it writes paragraphs)
- The structure forces factual bullet points — no vague generalisations
- Risk / Confidence fields make the model's uncertainty explicit
- Immediately obvious to a user whether they're talking to a trained model

Usage:
    python create_structured_dataset.py
    python train.py   (point DATASET_PATH to energy_data_structured.jsonl)
"""

import json
import re
import os

_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_JSONL  = os.path.join(_ROOT, "data", "energy_data.jsonl")
OUTPUT_JSONL = os.path.join(_ROOT, "data", "energy_data_structured.jsonl")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_sentences(text: str) -> list[str]:
    """Split text into sentences, stripping markdown."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # remove bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)         # remove italic
    text = re.sub(r'#+\s', '', text)                  # remove headers
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_bullet_facts(text: str) -> list[str]:
    """Extract facts from text: prefer lines with numbers, fall back to meaningful sentences."""
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    clean = re.sub(r'\*(.+?)\*', r'\1', clean)

    lines = re.split(r'[\n.!?]', clean)
    lines = [l.strip('•-–— ').strip() for l in lines if len(l.strip()) > 20]

    # Priority 1: lines with numbers/percentages/years
    numeric = [l for l in lines if re.search(r'\d', l)]
    if numeric:
        return numeric[:4]

    # Priority 2: any meaningful sentence (no "See answer above" fallback)
    return lines[:3]


def infer_risk_level(prompt: str, completion: str) -> tuple[str, str]:
    """Heuristic: infer risk level from keywords in the text."""
    combined = (prompt + " " + completion).lower()
    
    high_risk_kw  = ["volatile", "uncertain", "risk", "crisis", "shortage",
                     "collapse", "instability", "geopolit", "sanction", "war"]
    med_risk_kw   = ["depends", "may change", "could", "policy", "regulation",
                     "forecast", "projected", "expected", "estimate"]
    low_risk_kw   = ["stable", "consistent", "reliable", "proven", "guaranteed",
                     "established", "historical", "confirmed", "measured"]

    high_score = sum(1 for kw in high_risk_kw if kw in combined)
    low_score  = sum(1 for kw in low_risk_kw  if kw in combined)

    if high_score >= 2:
        return "High", "Topic involves significant uncertainty or geopolitical/market volatility."
    elif low_score >= 2:
        return "Low", "Topic is supported by stable, measured data with low volatility."
    else:
        return "Medium", "Moderate uncertainty; depends on policy or market conditions."


def infer_confidence(completion: str) -> tuple[str, str]:
    """Heuristic: infer confidence from source indicators in completion."""
    low_conf  = ["estimated", "projected", "may", "could", "forecast", "expected",
                 "assumed", "approximately", "roughly", "unclear"]
    high_conf = ["irena", "iea", "report", "data", "%", "survey",
                 "published", "confirmed", "measured", "according to"]

    c_lower = completion.lower()
    high_score = sum(1 for kw in high_conf if kw in c_lower)
    low_score  = sum(1 for kw in low_conf  if kw in c_lower)

    if high_score >= 2:
        return "High", "Based on specific statistics or named reports."
    elif low_score >= 2:
        return "Low", "Answer relies on estimates or projections."
    else:
        return "Medium", "General expert knowledge; exact figures may vary."


def build_structured_completion(prompt: str, completion: str) -> str:
    """Convert a plain completion into the structured format."""

    sentences = extract_sentences(completion)
    direct_answer = sentences[0] if sentences else completion[:120]

    facts = extract_bullet_facts(completion)
    if not facts:
        # Last resort: split answer into sub-clauses
        facts = [s.strip() for s in re.split(r'[;,]', direct_answer) if len(s.strip()) > 15][:3]
    if not facts:
        facts = [direct_answer]

    risk_level, risk_explanation   = infer_risk_level(prompt, completion)
    confidence, conf_explanation   = infer_confidence(completion)

    bullets = "\n".join(f"• {f}" for f in facts) if facts else "• See answer above."

    structured = (
        f"ANSWER: {direct_answer}\n\n"
        f"KEY FACTS:\n{bullets}\n\n"
        f"RISK LEVEL: {risk_level}\n"
        f"→ {risk_explanation}\n\n"
        f"CONFIDENCE: {confidence}\n"
        f"→ {conf_explanation}"
    )
    return structured


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Reading {INPUT_JSONL}...")
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]

print(f"Converting {len(examples)} examples to structured format...")

written = 0
with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
    for ex in examples:
        prompt     = ex["prompt"]
        completion = ex["completion"]
        structured = build_structured_completion(prompt, completion)

        record = {"prompt": prompt, "completion": structured}
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

print(f"Done. {written} structured examples saved to {OUTPUT_JSONL}")
print("\n── Sample output ──────────────────────────────────────")
sample = examples[0]
print("PROMPT:", sample["prompt"])
print()
print(build_structured_completion(sample["prompt"], sample["completion"]))
print("───────────────────────────────────────────────────────")
print(f"\nNext step: update DATASET_PATH in config.py to '{OUTPUT_JSONL}' and run train.py")
