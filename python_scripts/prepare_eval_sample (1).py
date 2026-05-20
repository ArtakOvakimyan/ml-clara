#!/usr/bin/env python3
"""
prepare_eval_sample.py
======================
Готовит случайную выборку из validation SberQUAD для честного сравнения
baseline RAG и CLaRa. Сохраняет в формате, удобном для обоих пайплайнов.

Фильтры:
  - Только примеры с непустыми ответами
  - Документы не длиннее max_doc_words слов (совместимо с doc_max_length=256)
  - Фиксированный random seed для воспроизводимости

Формат выхода (JSONL, одна запись на строку):
{
    "id": "...",
    "question": "...",
    "answer": "...",         # канонический ответ (первый)
    "answers": ["...", ...], # все варианты ответов (для F1 по max)
    "context": "..."
}

Использование:
    python prepare_eval_sample.py \
        --input ./data/raw/sberquad_validation.json \
        --output ./data/eval/sample_500.jsonl \
        --n_samples 500 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def prepare_sample(
    input_path: str,
    output_path: str,
    n_samples: int,
    seed: int,
    max_doc_words: int,
):
    """Загружает SberQUAD validation и сэмплирует n_samples примеров."""

    print(f"📂 Загрузка: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Разворачиваем SQuAD-структуру в плоский список примеров
    flat_examples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"].strip()
            word_count = len(context.split())

            # Фильтр по длине документа
            if word_count > max_doc_words:
                continue

            for qa in para["qas"]:
                question = qa["question"].strip()
                answers = qa.get("answers", [])
                if not answers:
                    continue

                # Все варианты ответов
                answer_texts = [a["text"].strip() for a in answers if a.get("text", "").strip()]
                if not answer_texts:
                    continue

                flat_examples.append({
                    "id": qa.get("id", f"ex_{len(flat_examples)}"),
                    "question": question,
                    "answer": answer_texts[0],
                    "answers": answer_texts,
                    "context": context,
                    "context_word_count": word_count,
                })

    print(f"  Всего подходящих примеров: {len(flat_examples)}")

    if len(flat_examples) < n_samples:
        print(f"  ⚠ Запрошено {n_samples}, но доступно только {len(flat_examples)}")
        n_samples = len(flat_examples)

    # Фиксированный сэмпл
    random.seed(seed)
    sampled = random.sample(flat_examples, n_samples)

    # Сохраняем
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Статистика
    doc_lens = [ex["context_word_count"] for ex in sampled]
    q_lens = [len(ex["question"].split()) for ex in sampled]
    a_lens = [len(ex["answer"].split()) for ex in sampled]

    print(f"\n✅ Сохранено: {out_path}")
    print(f"  Примеров:              {len(sampled)}")
    print(f"  Длина документа (сл.): {min(doc_lens)}–{max(doc_lens)} "
          f"(среднее {sum(doc_lens)/len(doc_lens):.0f})")
    print(f"  Длина вопроса (сл.):   {min(q_lens)}–{max(q_lens)} "
          f"(среднее {sum(q_lens)/len(q_lens):.0f})")
    print(f"  Длина ответа (сл.):    {min(a_lens)}–{max(a_lens)} "
          f"(среднее {sum(a_lens)/len(a_lens):.0f})")

    # Примеры
    print(f"\n  Пример:")
    ex = sampled[0]
    print(f"    Вопрос:   {ex['question']}")
    print(f"    Ответ:    {ex['answer']}")
    print(f"    Документ: {ex['context'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Подготовка eval sample из SberQUAD")
    parser.add_argument(
        "--input",
        default="./data/raw/sberquad_validation.json",
        help="Путь к validation JSON",
    )
    parser.add_argument(
        "--output",
        default="./data/eval/sample_500.jsonl",
        help="Куда сохранить выборку",
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_doc_words",
        type=int,
        default=300,
        help="Макс. длина документа в словах (≈ 256 токенов)",
    )
    args = parser.parse_args()

    prepare_sample(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        max_doc_words=args.max_doc_words,
    )


if __name__ == "__main__":
    main()
