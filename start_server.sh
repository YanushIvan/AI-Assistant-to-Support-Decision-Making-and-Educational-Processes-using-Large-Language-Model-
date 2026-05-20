#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
	PYTHON="$SCRIPT_DIR/venv/bin/python"
else
	PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" energy_chat/main.py
