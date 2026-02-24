import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Импортируем настройки из вашего конфига, чтобы не дублировать их
from config import MODEL_ID, OUTPUT_DIR, MAX_SEQ_LENGTH, SYSTEM_PROMPT, COMPUTE_DTYPE

def load_model_and_tokenizer():
    """Загружает модель и адаптер (аналогично startup событию в app.py)."""
    
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    
    if not os.path.exists(adapter_path):
        print(f"Ошибка: Адаптер не найден по пути {adapter_path}")
        print("Сначала запустите train.py!")
        sys.exit(1)

    print(f"1. Загрузка токенизатора: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. Настройка 4-bit квантования...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=False,
    )

    print(f"3. Загрузка базовой модели...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"4. Подключение LoRA адаптера из {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Вливаем адаптер в веса модели для ускорения инференса
    model = model.merge_and_unload()
    model.eval()
    
    return model, tokenizer

def chat_loop(model, tokenizer):
    """Запускает цикл чата в консоли."""
    
    # История диалога
    history = []
    
    # Настраиваем стоп-токены для Qwen
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    print("\n" + "="*50)
    print("🤖 Модель готова! Пишите ваш вопрос (или 'exit' для выхода).")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\033[1;34mUser:\033[0m ").strip()
            
            if user_input.lower() in ["exit", "quit", "выход"]:
                print("Завершение работы.")
                break
            
            if not user_input:
                continue

            # Формируем сообщения с учетом истории
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_input})

            # Применяем Chat Template
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(
                text_input, 
                return_tensors="pt", 
                add_special_tokens=False
            )
            
            model_inputs = inputs.to(model.device)

            print("\033[1;33mAssistant думает...\033[0m", end="\r")

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    top_k=50,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True # Для инференса кэш полезен
                )

            # Декодируем только новые токены
            input_len = model_inputs.input_ids.shape[1]
            generated_ids = generated_ids[:, input_len:]
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

            # Очистка от артефактов
            response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

            print(f"\033[1;32mAssistant:\033[0m {response}\n")
            print("-" * 30)

            # Обновляем историю
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Ограничиваем историю (например, последние 10 сообщений), чтобы не переполнить контекст
            if len(history) > 10:
                history = history[-10:]

        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            break
        except Exception as e:
            print(f"\nОшибка: {e}")

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    chat_loop(model, tokenizer)