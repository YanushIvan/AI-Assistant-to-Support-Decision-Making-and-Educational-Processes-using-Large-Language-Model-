"""
Core configuration settings for the Energy Chat application.
"""
import os
from functools import lru_cache
from typing import Optional
import torch

class Settings:
    """Application settings."""
    
    # FastAPI
    APP_NAME: str = "Energy AI Chat"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model Configuration
    MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
    
    @property
    def MODELS(self) -> dict:
        """Get available models configuration."""
        # Get the workspace root (parent of energy_chat directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # energy_chat/core
        energy_chat_dir = os.path.dirname(current_dir)  # energy_chat
        workspace_root = os.path.dirname(energy_chat_dir)  # workspace root
        
        return {
            "phi-3": {
                "name": "Phi-3 Mini (Energy)",
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "adapter_path": os.path.join(workspace_root, "phi3_mini_energy_finetune", "final_adapter")
            },
            "phi-3-base": {
                "name": "Phi-3 Mini (Vanilla)",
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "adapter_path": None
            },
            "qwen-2.5": {
                "name": "Qwen 2.5 3B (Energy)",
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "adapter_path": os.path.join(workspace_root, "qwen 2.5 3b", "qwen2.5_3b_energy_finetune", "final_adapter")
            }
        }

    @property
    def LORA_ADAPTER_PATH(self) -> str:
        """Get the full path to the LoRA adapter, relative to workspace root."""
        # Get the workspace root (parent of energy_chat directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # energy_chat/core
        energy_chat_dir = os.path.dirname(current_dir)  # energy_chat
        workspace_root = os.path.dirname(energy_chat_dir)  # workspace root
        return os.path.join(workspace_root, "phi3_mini_energy_finetune", "final_adapter")
    
    # Training Parameters
    MAX_SEQ_LENGTH: int = 1024
    
    # System Prompt
    SYSTEM_PROMPT: str = (
        "You are a highly knowledgeable and witty expert on energy, climate, and financial markets. "
        "Your answers must be **concise, technically accurate, and highly informative**, "
        "providing the core analysis necessary to fully address the user's prompt in a focused manner. "
        "Avoid unnecessary detail."
    )
    
    # Compute Configuration
    COMPUTE_DTYPE: torch.dtype = torch.float16
    
    # Generation Parameters
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    TOP_K: int = 50
    TOP_P: float = 0.95
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    RELOAD: bool = False
    
    # Static Files
    STATIC_DIR: str = "static"
    TEMPLATES_DIR: str = "templates"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
