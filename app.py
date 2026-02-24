import torch
import os
import sys
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- ИМПОРТЫ: SYSTEM_PROMPT добавлен в config.py ---
from config import MODELS, MAX_SEQ_LENGTH, SYSTEM_PROMPT, COMPUTE_DTYPE

# --- Pydantic Модель для Входящего Запроса ---
class ChatRequest(BaseModel):
    """Модель для запроса чата."""
    prompt: str = Field(..., example="Как добывается солнечная энергия?")
    history: list[dict] = Field(default=[], example=[{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Здравствуйте"}])
    model: str = Field(default="phi-3", example="phi-3")

app = FastAPI()
llm_model = None
llm_tokenizer = None
current_model_key = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mount static files directory (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================================================================
# === УПРАВЛЕНИЕ МОДЕЛЯМИ ===
# =========================================================================

def load_model_by_key(model_key: str):
    """Загружает указанную модель, выгружая предыдущую."""
    global llm_model, llm_tokenizer, current_model_key
    
    if current_model_key == model_key and llm_model is not None:
        return # Модель уже загружена
        
    if model_key not in MODELS:
        raise ValueError(f"Модель {model_key} не найдена в конфигурации")
        
    model_config = MODELS[model_key]
    model_id = model_config["model_id"]
    adapter_path = model_config["adapter_path"]
    
    logging.info(f"Переключение на модель: {model_config['name']} ({model_key})")
    
    # Выгрузка текущей модели для освобождения памяти
    if llm_model is not None:
        logging.info("Выгрузка текущей модели...")
        del llm_model
        del llm_tokenizer
        torch.cuda.empty_cache()
        llm_model = None
        llm_tokenizer = None
    
    if not os.path.isdir(adapter_path):
        logging.critical(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Адаптер LoRA не найден по пути: {adapter_path} !!!")
        # Не выходим, чтобы сервер продолжал работать
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
    try:
        # 1. Токенизатор
        logging.info(f"Загрузка токенизатора {model_id}...")
        llm_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        # 2. Конфигурация QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE, 
            bnb_4bit_use_double_quant=False,
        )

        # 3. Загрузка базовой модели
        logging.info(f"Загрузка базовой модели {model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model.config.use_cache = True

        # 4. Загрузка адаптера LoRA
        logging.info(f"Загрузка адаптера LoRA из {adapter_path}...")
        llm_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Опционально: слияние
        llm_model = llm_model.merge_and_unload() 
        llm_model.eval()
        
        current_model_key = model_key
        logging.info(f"Модель {model_key} успешно загружена.")
        
    except Exception as e:
        logging.critical(f"Ошибка при загрузке модели {model_key}: {e}")
        raise e

@app.on_event("startup")
def startup_event():
    """Загружает модель по умолчанию при старте."""
    try:
        load_model_by_key("phi-3")
    except Exception as e:
        logging.error(f"Не удалось загрузить модель по умолчанию: {e}")

@app.get("/models")
def get_models():
    """Возвращает список доступных моделей."""
    return {
        "models": [
            {"id": key, "name": config["name"]} 
            for key, config in MODELS.items()
        ],
        "current": current_model_key
    }
        
# =========================================================================
# === ЭНДПОИНТ ДЛЯ ЧАТА ===
# =========================================================================

@app.post("/chat")
async def answer_prompt(request: ChatRequest):
    """Отвечает на запрос пользователя, используя выбранную модель."""
    
    # Проверка и загрузка модели
    if current_model_key != request.model:
        try:
            load_model_by_key(request.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model {request.model}: {str(e)}")

    if llm_model is None or llm_tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # 1. Формирование истории чата
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(request.history)
    messages.append({"role": "user", "content": request.prompt})

    try:
        # 2. Применение Chat Template и Токенизация
        tokenized = llm_tokenizer(
            llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            ),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # 3. Перемещаем на устройство модели
        input_ids = input_ids.to(llm_model.device) 
        attention_mask = attention_mask.to(llm_model.device)
        
        # 4. Генерация Ответа с явным attention_mask и без кэша
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=llm_tokenizer.pad_token_id, 
                eos_token_id=llm_tokenizer.eos_token_id,
                use_cache=False,
            )

        # 5. Декодирование
        new_tokens = outputs[0][input_ids.shape[-1]:]
        response_text = llm_tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # 6. Очистка от специальных токенов
        response_text = response_text.replace("<|end|>", "").strip()

        return {"status": "success", "answer": response_text}

    except Exception as e:
        logging.error(f"Ошибка во время генерации ответа: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")


# =========================================================================
# === HTML Интерфейс (Modern Chat UI) =====================================
# =========================================================================
@app.get("/", response_class=HTMLResponse)
def get_html():
    """Возвращает основной HTML файл шаблона."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    else:
        logging.error(f"Template not found at {template_path}")
        return HTMLResponse(content="<h1>Template not found</h1>", status_code=404)