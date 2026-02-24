"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    prompt: str = Field(..., description="User's prompt/question", min_length=1, max_length=5000)
    history: List[ChatMessage] = Field(
        default=[],
        description="Chat history"
    )
    model: str = Field(
        default="phi-3",
        description="Model to use for generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "How is solar energy generated?",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ],
                "model": "phi-3"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    status: str = Field(..., description="Response status")
    answer: str = Field(..., description="AI generated response")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "answer": "Solar energy is generated through photovoltaic cells..."
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error detail message")
    status_code: int = Field(..., description="HTTP status code")
