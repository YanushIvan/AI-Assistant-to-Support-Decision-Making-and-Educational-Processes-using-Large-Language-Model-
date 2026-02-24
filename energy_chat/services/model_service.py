"""
Model loading and management service.
"""
import os
import sys
import re
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from energy_chat.core import logger, get_settings


class ModelService:
    """Service for loading and managing the LLM model."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._is_loaded = False
        self.current_model_key: Optional[str] = None
    
    def load_models(self, model_key: str = "phi-3") -> bool:
        """
        Load the model and tokenizer.
        
        Args:
            model_key: Key of the model to load (default: "phi-3")
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            if self._is_loaded and self.current_model_key == model_key:
                logger.info(f"Model {model_key} already loaded")
                return True
            
            # Unload current model if different
            if self._is_loaded and self.current_model_key != model_key:
                logger.info(f"Unloading current model {self.current_model_key}...")
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()
                self._is_loaded = False
                self.model = None
                self.tokenizer = None
            
            if model_key not in self.settings.MODELS:
                logger.error(f"Model {model_key} not found in configuration")
                return False
                
            model_config = self.settings.MODELS[model_key]
            model_id = model_config["model_id"]
            adapter_path = model_config["adapter_path"]
            
            logger.info(f"Loading model: {model_config['name']} ({model_key})")
            logger.info(f"Base model: {model_id}")
            
            # Check adapter path if it exists
            if adapter_path:
                logger.info(f"Checking adapter path: {adapter_path}")
                if not os.path.isdir(adapter_path):
                    logger.critical(
                        f"LoRA adapter not found at: {adapter_path}"
                    )
                    return False
                logger.info("Adapter path verified")
            else:
                logger.info("No adapter path provided. Loading vanilla model.")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
            except Exception as tokenizer_err:
                logger.error(f"Failed to load tokenizer from {model_id}: {tokenizer_err}")
                if adapter_path:
                    logger.info(f"Attempting to load tokenizer from adapter path...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            adapter_path,
                            trust_remote_code=True
                        )
                    except Exception as local_err:
                        logger.error(f"Failed to load tokenizer from adapter: {local_err}")
                        raise tokenizer_err
                else:
                    raise tokenizer_err
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
            
            # Configure QLoRA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.settings.COMPUTE_DTYPE,
                bnb_4bit_use_double_quant=False,
            )
            
            # Load base model
            logger.info(f"Loading base model {model_id}...")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.error(f"Failed to load base model: {e}")
                raise
            base_model.config.use_cache = True
            logger.info("Base model loaded successfully")
            
            # Load LoRA adapter if configured
            if adapter_path:
                logger.info(f"Loading LoRA adapter from {adapter_path}...")
                try:
                    self.model = PeftModel.from_pretrained(base_model, adapter_path)
                except Exception as e:
                    logger.error(f"Failed to load LoRA adapter: {e}")
                    raise
                
                logger.info("LoRA adapter loaded successfully")
                
                # Merge and unload
                logger.info("Merging adapter with base model...")
                self.model = self.model.merge_and_unload()
                logger.info("Model merge completed")
            else:
                logger.info("Using base model without adapter")
                self.model = base_model
            
            self.model.eval()
            
            self._is_loaded = True
            self.current_model_key = model_key
            logger.info(f"✓ Model {model_key} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def generate_response(self, messages: list) -> Optional[str]:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Generated response text or None on error
        """
        if not self._is_loaded or self.model is None or self.tokenizer is None:
            logger.error("Models not loaded. Call load_models() first.")
            return None
        
        try:
            # Tokenize
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            tokenized = self.tokenizer(
                input_ids,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.settings.MAX_SEQ_LENGTH,
            )
            
            input_ids = tokenized["input_ids"].to(self.model.device)
            attention_mask = tokenized["attention_mask"].to(self.model.device)
            
            # Determine stop tokens based on model
            stop_token_ids = [self.tokenizer.eos_token_id]
            if "phi-3" in self.current_model_key.lower():
                # Add <|end|> token for Phi-3
                stop_token_ids.append(32007)
            elif "qwen" in self.current_model_key.lower():
                # Add <|im_end|> and <|endoftext|> for Qwen
                stop_token_ids.extend([151645, 151643])
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.settings.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=self.settings.TEMPERATURE,
                    top_k=self.settings.TOP_K,
                    top_p=self.settings.TOP_P,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=stop_token_ids,
                    use_cache=False,
                )
            
            # Decode
            new_tokens = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            # Clean up special tokens
            response = response.replace("<|end|>", "").replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
            
            # Convert markdown headers (### Header) to HTML bold with line break
            # Using <b> because h3 might be too big inside a chat bubble
            response = re.sub(r'###\s*(.+)', r'<br><b>\1</b>', response)
            
            # Convert markdown bold (**text**) to HTML bold (<b>text</b>)
            response = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded


# Global instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
