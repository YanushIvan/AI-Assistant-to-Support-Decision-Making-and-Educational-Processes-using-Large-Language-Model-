# Energy AI Chat

A modern FastAPI application for AI-powered energy-related conversations using fine-tuned Phi-3 model.

## Project Structure

```
energy_chat/
├── core/                    # Configuration and logging
│   ├── __init__.py
│   ├── config.py           # Settings management
│   └── logger.py           # Logging configuration
├── api/                    # API routes
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       └── chat.py         # Chat endpoints
├── models/                 # Pydantic schemas
│   ├── __init__.py
│   └── schemas.py          # Request/response models
├── services/               # Business logic
│   ├── __init__.py
│   └── model_service.py    # Model loading and inference
├── templates/              # HTML templates
│   └── index.html          # Chat UI
├── static/                 # Static files
│   └── chat.js             # Frontend JavaScript
├── __init__.py
└── main.py                 # Application entry point
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Ensure model files are present:**
   ```
   phi3_mini_energy_finetune/
   └── final_adapter/
       ├── adapter_config.json
       ├── adapter_model.safetensors
       └── ... (other adapter files)
   ```

## Running the Application

### Development Mode
```bash
python -m uvicorn energy_chat.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
python -m uvicorn energy_chat.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Using the main.py entry point
```bash
python energy_chat/main.py
```

## API Endpoints

- **GET `/`** - Chat interface (HTML)
- **GET `/api/health`** - Health check
- **POST `/api/chat`** - Send chat message
  - Request: `{"prompt": "string", "history": [{"role": "string", "content": "string"}]}`
  - Response: `{"status": "success", "answer": "string"}`

## Configuration

Settings are managed in `energy_chat/core/config.py`. You can override them by:

1. Setting environment variables
2. Creating a `.env` file
3. Modifying `Settings` class directly

### Key Settings

- `MODEL_ID`: HuggingFace model identifier
- `LORA_ADAPTER_PATH`: Path to fine-tuned LoRA adapter
- `MAX_SEQ_LENGTH`: Maximum sequence length for the model
- `MAX_NEW_TOKENS`: Maximum tokens to generate
- `TEMPERATURE`: Generation temperature (0.0-1.0)
- `TOP_K` / `TOP_P`: Sampling parameters

## Features

- ✅ Clean, modular architecture
- ✅ Async FastAPI with proper error handling
- ✅ Model loading with LoRA adapter support
- ✅ 4-bit quantization with QLoRA
- ✅ Modern chat UI with real-time messaging
- ✅ Type hints and Pydantic validation
- ✅ Comprehensive logging
- ✅ Health check endpoint

## Performance

- Uses 4-bit quantization to reduce memory footprint
- Supports KV-cache disabling for stability
- Efficient message streaming
- Optimized static file serving

## Troubleshooting

### Models not loading
- Verify adapter path exists: `phi3_mini_energy_finetune/final_adapter/`
- Check system memory requirements (~8GB recommended with 4-bit quantization)
- Review logs for detailed error messages

### Port already in use
```bash
# Use a different port
python -m uvicorn energy_chat.main:app --port 8001
```

### Image not loading
- Ensure `image.png` exists in `energy_chat/static/`
- Check static file mounting in logs

## Development

### Adding new routes
1. Create a new file in `energy_chat/api/routes/`
2. Create a router: `router = APIRouter(prefix="/api/new")`
3. Import and include in `api/__init__.py`
4. Include in `main.py`: `app.include_router(new_router)`

### Adding new services
1. Create a new service file in `energy_chat/services/`
2. Implement service class
3. Export from `services/__init__.py`
4. Use in routes via dependency injection

## License

MIT
