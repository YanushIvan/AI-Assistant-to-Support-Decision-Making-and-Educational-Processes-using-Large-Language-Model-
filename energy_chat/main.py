"""
Energy AI Chat FastAPI Application
Main entry point for the application
"""
import os
import sys
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from energy_chat.core import logger, get_settings
from energy_chat.api import chat_router
from energy_chat.services import get_model_service

# Get settings
settings = get_settings()

# Global state for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("=" * 50)
    logger.info("LIFESPAN: Starting Energy AI Chat application...")
    logger.info("=" * 50)
    model_service = get_model_service()
    logger.info("Model service obtained")
    result = model_service.load_models()
    if result:
        logger.info("✓ Models loaded successfully at startup")
    else:
        logger.error("✗ Failed to load models at startup")
    logger.info("=" * 50)
    yield
    # Shutdown
    logger.info("LIFESPAN: Shutting down Energy AI Chat application")
    logger.info("=" * 50)


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered chat interface for energy-related queries",
    lifespan=lifespan
)

# Add startup and shutdown event handlers as fallback
@app.on_event("startup")
async def startup_event():
    """Startup event handler - loads models"""
    logger.info("ON_EVENT: Application startup event triggered")
    model_service = get_model_service()
    result = model_service.load_models()
    if result:
        logger.info("✓ Models loaded via startup event")
    else:
        logger.error("✗ Failed to load models via startup event")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("ON_EVENT: Application shutdown event triggered")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.warning(f"Static directory not found at {static_dir}")

# Include API routes
app.include_router(chat_router)


@app.get("/models")
async def get_models_root():
    """
    Get available models (root fallback).
    """
    models_list = []
    for key, config in settings.MODELS.items():
        models_list.append({
            "id": key,
            "name": config["name"]
        })
    return {"models": models_list}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Serve the main chat interface HTML
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.error(f"Template not found at {template_path}")
        return "<h1>Template not found</h1>"


@app.get("/api/health")
async def health_check() -> dict:
    """
    Health check endpoint
    """
    model_service = get_model_service()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model_loaded": model_service.is_loaded
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print(f" STARTING ENERGY AI CHAT")
    print(f" Serving from: http://{settings.HOST}:{settings.PORT}")
    print(f" If port {settings.PORT} is busy, change it in energy_chat/core/config.py")
    print("="*60 + "\n")
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    logger.info(f"Using lifespan context manager for model loading")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
