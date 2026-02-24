"""
Logging configuration for the Energy Chat application.
"""
import logging
import sys
from .config import get_settings

settings = get_settings()

def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("energy_chat")
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    logger.addHandler(console_handler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    return logger

logger = setup_logging()
