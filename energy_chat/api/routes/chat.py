"""
Chat API routes.
"""
from fastapi import APIRouter, HTTPException
from energy_chat.models import ChatRequest, ChatResponse
from energy_chat.services import get_model_service
from energy_chat.core import logger, get_settings

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/models")
async def get_models():
    """
    Get available models.
    """
    settings = get_settings()
    models_list = []
    for key, config in settings.MODELS.items():
        models_list.append({
            "id": key,
            "name": config["name"]
        })
    return {"models": models_list}


@router.post("/chat", response_model=ChatResponse)
async def answer_prompt(request: ChatRequest) -> ChatResponse:
    """
    Handle chat requests and return AI-generated responses.
    
    Args:
        request: ChatRequest containing prompt and optional history
    
    Returns:
        ChatResponse with generated answer
    
    Raises:
        HTTPException: If model is not loaded or generation fails
    """
    model_service = get_model_service()
    
    # Ensure the requested model is loaded
    requested_model = request.model
    if model_service.current_model_key != requested_model:
        logger.info(f"Switching model to {requested_model}...")
        success = model_service.load_models(requested_model)
        if not success:
             raise HTTPException(
                status_code=400,
                detail=f"Failed to load model: {requested_model}"
            )

    if not model_service.is_loaded:
        logger.error("Model not loaded when processing chat request")
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded or failed to load. Please try again later."
        )
    
    try:
        # Build messages
        messages = [
            {"role": "system", "content": model_service.settings.SYSTEM_PROMPT}
        ]
        
        # Add history if provided
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current prompt
        messages.append({"role": "user", "content": request.prompt})
        
        logger.info(f"Processing chat request with {len(messages)} messages")
        
        # Generate response
        response_text = model_service.generate_response(messages)
        
        if response_text is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )
        
        logger.info("Chat response generated successfully")
        
        return ChatResponse(
            status="success",
            answer=response_text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns:
        dict: Health status and model load status
    """
    model_service = get_model_service()
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded
    }
