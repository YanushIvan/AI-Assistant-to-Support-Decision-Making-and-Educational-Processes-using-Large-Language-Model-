#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi

source "$VENV/bin/activate"

# ── Anthropic API key (для G-Eval судьи) ──────────────────────────────────────
# Вставь свой ключ сюда:
# ─────────────────────────────────────────────────────────────────────────────

# Install anthropic if missing (needed for --judge)
pip install -q anthropic 2>/dev/null || true

python "$SCRIPT_DIR/scripts/evaluate_models.py" "$@"
