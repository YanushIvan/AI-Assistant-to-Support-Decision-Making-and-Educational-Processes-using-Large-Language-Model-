# Energy AI Chat

FastAPI web app for energy-domain chat with multiple local or adapter-augmented LLMs.

## Repository Layout (key parts)

```
energy_chat/                # FastAPI app
   core/                     # Config and logging
   api/                      # API routes
   models/                   # Pydantic schemas
   services/                 # Model loading + inference
   templates/                # HTML UI
   static/                   # JS/CSS assets
scripts/                    # Evaluation and training utilities
data/                       # JSONL datasets
eval_results/               # Evaluation outputs
phi3_mini_energy_dpo/       # Phi-3 DPO adapter
gemma3_4b_energy_dpo/        # Gemma-3 DPO adapter
qwen3_4b_energy_dpo/         # Qwen3 DPO adapter
```

## Setup

```bash
pip install -r requirements.txt
```

The app uses 4-bit quantization (bitsandbytes). A CUDA-capable GPU is strongly recommended.

## Run the App

Use the helper script (prefers ./venv if present):

```bash
./start_server.sh
```

Or run directly:

```bash
python energy_chat/main.py
```

Default host/port are defined in `energy_chat/core/config.py` (currently 0.0.0.0:8002).

## API Endpoints

- GET /                - Chat UI
- GET /api/health       - Health check
- GET /api/models       - List available models
- GET /models           - Same as /api/models (root fallback)
- POST /api/chat        - Generate a response

Request body example:

```json
{
   "prompt": "What is the current outlook for EU electricity prices?",
   "history": [
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello"}
   ],
   "model": "phi-3"
}
```

Response example:

```json
{
   "status": "success",
   "answer": "..."
}
```

## Model Selection

Available models are defined in `energy_chat/core/config.py`:

- phi-3        - Phi-3 Mini with Energy DPO adapter
- phi-3-base   - Phi-3 Mini base (no adapter)
- qwen-3       - Qwen3 4B with Energy DPO adapter
- qwen-3-base  - Qwen3 4B base (no adapter)
- gemma-3      - Gemma 3 4B with Energy DPO adapter

Adapters are expected at:

- phi3_mini_energy_dpo/final_adapter
- qwen3_4b_energy_dpo/final_adapter
- gemma3_4b_energy_dpo/final_adapter

You can extend or change these paths in `Settings.MODELS`.

## Evaluation

Run the full benchmark suite:

```bash
./run_eval.sh --help
./run_eval.sh --n-samples 50 --output-dir eval_results
```

Optional judge scoring uses Anthropic. Set `ANTHROPIC_API_KEY` and add `--judge`.

To judge existing results locally with Ollama:

```bash
python scripts/judge_only.py --input eval_results/results_*.json --model mistral
```

## DPO Training

```bash
python scripts/train_dpo.py --model phi3 --dataset data/energy_data_dpo.jsonl
python scripts/train_dpo.py --model gemma3 --dataset data/energy_data_dpo.jsonl
python scripts/train_dpo.py --model qwen3 --dataset data/energy_data_dpo.jsonl
```

Use `--check-only` to validate paths without loading models.
