import json
import os

config_path = "CLaRa-7B-Instruct/compression-128/config.json"

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("🛠 Исправляем конфигурацию для работы без интернета...")

    # Принудительно отключаем Flash Attention, если он мешает
    config["attn_implementation"] = "eager"
    if "use_flash_attention_2" in config:
        config["use_flash_attention_2"] = False

    # Убираем ссылки на Apple-кластера, если они есть
    # и проверяем названия базовых моделей
    config["compr_base_model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
    config["decoder_model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print("✅ Config.json обновлен. Теперь попробуйте запустить ячейку с моделью снова.")
else:
    print("❌ Файл config.json не найден!")