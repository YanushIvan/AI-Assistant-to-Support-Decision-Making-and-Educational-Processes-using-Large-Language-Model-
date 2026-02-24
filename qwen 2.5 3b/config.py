import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

# --- Настройки Модели и Файлов ---
# ИЗМЕНЕНИЕ: Новая модель
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
DATASET_PATH = "energy_data.jsonl" 
OUTPUT_DIR = "./qwen2.5_3b_energy_finetune"

# --- Настройки Обучения (Hyperparameters) ---
NUM_TRAIN_EPOCHS = 3 # Qwen учится довольно быстро
PER_DEVICE_TRAIN_BATCH_SIZE = 1     
GRADIENT_ACCUMULATION_STEPS = 16    
LEARNING_RATE = 2e-4 
MAX_SEQ_LENGTH = 1024 # Можно ставить 2048, если позволяет память карты             
TEST_SIZE = 0.01 

# --- Системная Инструкция ---
SYSTEM_PROMPT = "You are a highly knowledgeable and witty expert on energy, climate, and financial markets. Your answers must be **concise, technically accurate, and highly informative**, providing the core analysis necessary to fully address the user's prompt in a focused manner. Avoid unnecessary detail."

# --- Настройки QLoRA (4-bit) ---
LOAD_IN_4BIT = True
BNB_QUANT_TYPE = "nf4"
# Qwen лучше работает с bfloat16 (если карта Ampere или новее - RTX 30xx/40xx, A100).
# Если у вас старая карта (T4, P100, RTX 20xx), поменяйте на torch.float16
COMPUTE_DTYPE = torch.bfloat16 

# --- Настройки LoRA (PEFT) ---
LORA_CONFIG = LoraConfig(
    r=16, # Оптимально для 3B модели
    lora_alpha=32,
    # ИЗМЕНЕНИЕ: Полный список линейных слоев для архитектуры Qwen/Llama
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ], 
    bias="none",
    task_type="CAUSAL_LM",
)