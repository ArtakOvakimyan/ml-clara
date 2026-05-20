import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerCorrectness 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
os.environ["OPENAI_API_KEY"] = "not-used" 
os.environ["OLLAMA_NUM_PARALLEL"] = "8"
os.environ["OLLAMA_KEEP_ALIVE"] = "60m"
import langchain
from langchain_community.embeddings import OllamaEmbeddings
langchain.debug = True
# 1. Загружаем ваши данные (id, question, gold, prediction)
raw_data = []
with open('results/baseline_results_details.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # Пропускаем пустые строки
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                # На случай, если это все же обычный массив [{}, {}]
                # или файл содержит мусор
                continue

# Теперь можно ограничить выборку
data = raw_data[:50]
print(f"Загружено объектов: {len(data)}")

dataset = Dataset.from_dict({
    "question": [item.get("question", "") for item in data],
    "answer": [item.get("prediction", "") for item in data],
    "ground_truth": [item.get("gold", "") for item in data],
    "retrieved_contexts": [[item.get("gold", "")] for item in data] # Обязательно список списков    # Если есть исходные тексты, добавьте их сюда как список списков:
    # "contexts": [[i["raw_document"]] for i in data] 
})

# 2. Настраиваем Qwen как судью (через Ollama)
evaluator_llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen2.5:7b", # или ваша модель qwen3
    timeout=120,
    temperature=0,
    # # Форсируем JSON-ответ на уровне провайдера, если Ollama это поддерживает
    # model_kwargs={
    #     "seed": 42,
    #     "top_p": 0.1,
    #     "extra_body": {
    #         "options": {
    #             "num_ctx": 4096,  # Зафиксируйте размер
    #             "num_predict": 512 # Ограничьте длину ответа судьи
    #         }
    #     }
    # }

)
config = RunConfig(timeout=180, max_workers=1)
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

ragas_llm = LangchainLLMWrapper(evaluator_llm )
faith_metric = Faithfulness(llm=ragas_llm)
ans_cor = AnswerCorrectness(
    llm=ragas_llm, 
    weights=[0.7, 0.3],
    embeddings=ragas_embeddings
)


# 4. Запуск
results = evaluate(
    dataset,
    metrics=[ans_cor],
    run_config=config,
    llm=ragas_llm,
)
df = results.to_pandas()

# Посмотрите первые строки
print("Доступные колонки в результате:", df.columns.tolist())
print(df.head())
# Сохраните в CSV, чтобы изучить в Excel/Sheets
df.to_csv("debug_ragas_ac_base.csv", index=False)

print(results)
