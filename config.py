import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

# --- Настройки Модели и Файлов ---
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct" 
DATASET_PATH = "energy_data.jsonl" 
OUTPUT_DIR = "./phi3_mini_energy_finetune"

# --- Конфигурация Моделей для Чата ---
MODELS = {
    "phi-3": {
        "name": "Phi-3 Mini (Energy)",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_path": "./phi3_mini_energy_finetune/final_adapter"
    },
    "qwen-2.5": {
        "name": "Qwen 2.5 3B (Energy)",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": "./qwen 2.5 3b/qwen2.5_3b_energy_finetune/final_adapter"
    }
}

# --- Настройки Обучения (Hyperparameters) ---
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 1     
GRADIENT_ACCUMULATION_STEPS = 16    
LEARNING_RATE = 2e-4 
MAX_SEQ_LENGTH = 1024               
TEST_SIZE = 0.01 

# --- Системная Инструкция ---
SYSTEM_PROMPT = (
    "You are a highly knowledgeable and witty expert on energy, climate, and financial markets. "
    "Your answers must be **concise, technically accurate, and highly informative**, "
    "providing the core analysis necessary to fully address the user's prompt in a focused manner. "
    "Avoid unnecessary detail."
)

# --- Настройки QLoRA (4-bit) ---
LOAD_IN_4BIT = True
BNB_QUANT_TYPE = "nf4"
# ИСПРАВЛЕНИЕ: Используем bfloat16 для современных GPU
COMPUTE_DTYPE = torch.float16 

# --- Настройки LoRA (PEFT) ---
LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    # Модули для Phi-3
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"], 
    bias="none",
    task_type="CAUSAL_LM",
)