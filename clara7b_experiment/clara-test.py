import os
import time
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
import logging
from tqdm import tqdm
from ragas.run_config import RunConfig
logging.getLogger("ragas").setLevel(logging.ERROR)
# --- 1. ФИКС БЕЗОПАСНОСТИ PYTORCH (ДЛЯ ЛОКАЛЬНЫХ ВЕСОВ) ---
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "weights_only": False})

# Прямой импорт класса из локального файла modeling_clara.py
from modeling_clara import CLaRa

# Импорты для локального Ragas (Ollama)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_correctness

# Настройки
device = "cuda" if torch.cuda.is_available() else "cpu"
CLARA_PATH = "./CLaRa-7B-Instruct/compression-128" # TODO: Убедитесь, что путь верный

# --- 2. МУЛЬТИ-ХОП ДАТАСЕТ ---
# DATASET = [
#     {
#         "question": "Which country is the birthplace of the director of the 2010 film Inception?",
#         "ground_truth": "The director of Inception is Christopher Nolan, who was born in the United Kingdom.",
#         "docs": [
#             "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
#             "Christopher Nolan is a British-American filmmaker born in London, United Kingdom.",
#             "London is the capital and largest city of England and the United Kingdom.",
#             "Leonardo DiCaprio starred as a professional thief in the 2010 movie Inception."
#         ]
#     },
#     {
#         "question": "Did the author of the novel '1984' die in the same country where the story takes place?",
#         "ground_truth": "Yes, George Orwell died in London, UK, which is the same country (Airstrip One, formerly Britain) where the novel is set.",
#         "docs": [
#             "Nineteen Eighty-Four is a dystopian social science fiction novel by the English novelist George Orwell.",
#             "George Orwell died on January 21, 1950, in London, United Kingdom.",
#             "The story of 1984 takes place in Airstrip One, located in what was formerly known as Great Britain."
#         ]
#     }
# ]

# --- 2. ДИНАМИЧЕСКАЯ ЗАГРУЗКА ДАТАСЕТА SQUAD (КОРОТКИЕ КОНТЕКСТЫ) ---
from datasets import load_dataset

print("Загрузка датасета SQuAD из Hugging Face...")
# SQuAD содержит короткие, емкие параграфы из Википедии
raw_dataset = load_dataset("squad", split="validation")

DATASET = []
for i in range(len(raw_dataset)):
    item = raw_dataset[i]
    
    question = item.get("question")
    context = item.get("context", "")
    
    # В SQuAD ответы лежат во вложенном словаре
    answers_dict = item.get("answers", {})
    answer_texts = answers_dict.get("text", [])
    ground_truth = answer_texts[0] if len(answer_texts) > 0 else None
    
    if question and ground_truth and context:
        # Контекст в SQuAD уже короткий, просто оборачиваем его в список документов
        DATASET.append({
            "question": question,
            "ground_truth": ground_truth,
            "docs": [context]
        })
        
    if len(DATASET) >= 100:  # Берем 50 примеров для надежного бенчмарка
        break

print(f"✅ Успешно подготовлено {len(DATASET)} тест-кейсов из SQuAD.")
mistral_results = []
clara_results = []

# --- 3. ТЕСТИРОВАНИЕ MISTRAL-7B-INSTRUCT ---
print("=== [1/3] Запуск тестов на Mistral-7B-Instruct-v0.2 ===")
mistral_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer_mistral = AutoTokenizer.from_pretrained(mistral_id)

# Загружаем Mistral тоже в 4-бита, чтобы оставить максимум памяти V100 под Ollama
bnb_config_mistral = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_mistral = AutoModelForCausalLM.from_pretrained(
    mistral_id, quantization_config=bnb_config_mistral, device_map="auto"
)


for item in tqdm(DATASET, desc="Mistral-7B"):
    q = item["question"]
    full_context = "\n".join(item["docs"])
    prompt = f"Based on the following context, answer the question briefly and accurately.\n\nContext:\n{full_context}\n\nQuestion: {q}"
    messages = [{"role": "user", "content": prompt}]
    
    # Исправленная токенизация:
    prompt_text = tokenizer_mistral.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_mistral(prompt_text, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1] 
    
    torch.cuda.synchronize()
    start_time = time.time()
        
    with torch.no_grad():
        outputs = model_mistral.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer_mistral.eos_token_id)
        
    torch.cuda.synchronize()
    latency = time.time() - start_time
    
    answer = tokenizer_mistral.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    
    mistral_results.append({
        "question": q, "answer": answer, "ground_truth": item["ground_truth"],
        "tokens": prompt_tokens, "latency": latency
    })

del model_mistral, tokenizer_mistral
torch.cuda.empty_cache()


# --- 4. ТЕСТИРОВАНИЕ APPLE CLARA-7B E2E (Локально) ---
print("\n=== [2/3] Запуск тестов на Apple CLaRa-7B E2E (Локально) ===")

CLARA_PATH = "./CLaRa-7B-E2E/compression-128" # Путь к вашей папке 128x

bnb_config_clara = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True
)

# Загружаем модель с вашим исправленным конфигом
model_clara = CLaRa.from_pretrained(
    CLARA_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_load_kwargs={"weights_only": False},
    quantization_config=bnb_config_clara,
    device_map="auto",
    torch_dtype=torch.float16,
)
model_clara.eval()

for item in tqdm(DATASET, desc="CLaRa-7B"):
    q = item["question"]
    chunks = item["docs"]
    
    with torch.no_grad():
        # Принудительно выставляем адаптер генерации
        if 'decoder_adapter' in model_clara.adapter_keys:
            model_clara.decoder.set_adapter('decoder_adapter')
            
        # 1. ЗАМЕР 1: Время полного цикла (Сжатие + Поиск + Генерация ответа)
        torch.cuda.synchronize()
        start_full = time.time()
        
        outputs = model_clara.generate_from_text(
            questions=[q], 
            documents=[chunks], 
            max_new_tokens=64
        )
        
        torch.cuda.synchronize()
        time_full_cycle = time.time() - start_full
        
        # 2. ЗАМЕР 2: Чистое время только сжатия документов (Имитируем Офлайн-этап)
        torch.cuda.synchronize()
        start_compress = time.time()
        
        _, _ = model_clara.compress_documents(documents=chunks)
        
        torch.cuda.synchronize()
        time_only_compression = time.time() - start_compress
        
        # 3. ВЫЧИСЛЕНИЕ: Чистое время генерации из готового кэша в памяти
        # (Полный цикл минус время, затраченное на сжатие текста)
        latency_from_cache = max(0.0001, time_full_cycle - time_only_compression)
        
        # Считаем токены промпта
        instr = model_clara._blend_prompt_and_memory_tokens(query=q, stage="stage1_2")
        clara_prompt_tokens = len(model_clara.decoder_tokenizer.encode(instr, add_special_tokens=False))
        
    clara_results.append({
        "question": q, 
        "answer": outputs[0].strip(), 
        "ground_truth": item["ground_truth"],
        "tokens": clara_prompt_tokens, 
        "latency": latency_from_cache # Пишем в отчет чистый онлайн-latency
    })

del model_clara
torch.cuda.empty_cache()
# --- 5. РАСЧЕТ МЕТРИК ЧЕРЕЗ RAGAS С ИСПОЛЬЗОВАНИЕМ LANGCHAIN-OLLAMA ---
print("\n=== [3/3] Расчет метрик через Ragas (Локально Ollama) ===")

import pandas as pd
from datasets import Dataset
# ВЕРНУЛИ: Современный и актуальный импорт без DeprecationWarning
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_correctness

# Инициализируем локальные модели через новый официальный пакет langchain-ollama
ragas_local_llm = ChatOllama(model="llama3.1", temperature=0.0, format="json")
ragas_local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Создаем базовые датафреймы, где уже лежат токены и latency наших тестов
df_m = pd.DataFrame(mistral_results)
df_c = pd.DataFrame(clara_results)

dataset_m = Dataset.from_dict({"question": df_m["question"], "answer": df_m["answer"], "ground_truth": df_m["ground_truth"]})
dataset_c = Dataset.from_dict({"question": df_c["question"], "answer": df_c["answer"], "ground_truth": df_c["ground_truth"]})

safe_config = RunConfig(max_workers=1, timeout=600)

print("Оцениваем Mistral...")
# Передаем новые инстансы langchain-ollama напрямую в evaluate
score_m = evaluate(
    dataset_m, 
    metrics=[answer_correctness], 
    llm=ragas_local_llm, 
    embeddings=ragas_local_embeddings,
    run_config=safe_config
).to_pandas()

# Безопасно переносим оценки в наш основной датафрейм
df_m["answer_correctness"] = score_m["answer_correctness"]
df_m.to_csv("debug_ragas_mistral_raw.csv", index=False)
print("✅ Отчет по Mistral успешно сохранен в 'debug_ragas_mistral_raw.csv'")

print("Оцениваем CLaRa...")
score_c = evaluate(
    dataset_c, 
    metrics=[answer_correctness], 
    llm=ragas_local_llm, 
    embeddings=ragas_local_embeddings,
    run_config=safe_config
).to_pandas()

# Безопасно переносим оценки в наш основной датафрейм
df_c["answer_correctness"] = score_c["answer_correctness"]
df_c.to_csv("debug_ragas_clara_compressed.csv", index=False)
print("✅ Отчет по CLaRa успешно сохранен в 'debug_ragas_clara_compressed.csv'")


# --- 6. АГРЕГИРОВАННЫЙ ВЫВОД И СОХРАНЕНИЕ СВОДНОГО ОТЧЕТА ---
print("\n" + "="*75)
print("СРАВНИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ (СРЕДНИЕ ЗНАЧЕНИЯ)")
print("="*75)

# Оставляем ваш красивый вывод в консоль
print(f"{'Метрика':<30} | {'Mistral-7B (Raw)':<20} | {'CLaRa-7B (Compressed)':<20}")
print("-" * 75)
print(f"{'Входные токены (в декодер)':<30} | {df_m['tokens'].mean():<20.1f} | {df_c['tokens'].mean():<20.1f}")
print(f"{'Время генерации ответа (сек)':<30} | {df_m['latency'].mean():<20.4f} | {df_c['latency'].mean():<20.4f}")
print(f"{'Точность ответа (Correctness)':<30} | {df_m['answer_correctness'].mean():<20.4f} | {df_c['answer_correctness'].mean():<20.4f}")
print("="*75)

# Формируем структурированные данные для итогового CSV
summary_data = [
    {
        "Model": "Mistral-7B (Raw)",
        "Avg_Input_Tokens": round(df_m['tokens'].mean(), 1),
        "Avg_Latency_sec": round(df_m['latency'].mean(), 4),
        "Avg_Answer_Correctness": round(df_m['answer_correctness'].mean(), 4),
        "Total_Samples": len(df_m)
    },
    {
        "Model": "CLaRa-7B (Compressed)",
        "Avg_Input_Tokens": round(df_c['tokens'].mean(), 1),
        "Avg_Latency_sec": round(df_c['latency'].mean(), 4),
        "Avg_Answer_Correctness": round(df_c['answer_correctness'].mean(), 4),
        "Total_Samples": len(df_c)
    }
]

# Создаем датафрейм и сохраняем его на диск
df_summary = pd.DataFrame(summary_data)
summary_filename = "benchmark_summary.csv"
df_summary.to_csv(summary_filename, index=False)

print(f"\n✅ Сводная статистика успешно сохранена в '{summary_filename}'")