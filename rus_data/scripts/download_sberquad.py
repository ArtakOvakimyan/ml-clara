#!/usr/bin/env python3
"""
download_sberquad.py
====================
Загрузка датасета SberQUAD из Hugging Face Hub и конвертация
в SQuAD-совместимый JSON-формат для дальнейшей обработки.

Использование:
    python download_sberquad.py [--output_dir ./data/raw]

Выходные файлы:
    data/raw/sberquad_train.json
    data/raw/sberquad_validation.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import os
from datasets import load_dataset


def download_and_convert(output_dir: str):
    """Загружает SberQUAD и сохраняет в SQuAD-формате."""    
    if output_dir is None:
        root_dir = Path(__file__).resolve().parent.parent
        output_dir = os.path.join(root_dir, "raw")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("📥 Загрузка SberQUAD из Hugging Face Hub...")
    ds = load_dataset("kuznetsoffandrey/sberquad")
    print(f"  ✓ Загружено: {', '.join(f'{k}: {len(v)}' for k, v in ds.items())}")

    for split_name, filename in [
        ("train", "sberquad_train.json"),
        ("validation", "sberquad_validation.json"),
    ]:
        if split_name not in ds:
            print(f"  ⚠ Сплит '{split_name}' не найден, пропускаю")
            continue

        split_data = ds[split_name]

        # Группируем QA-пары по контексту (как в SQuAD-формате)
        context_groups = defaultdict(list)
        skipped = 0

        for item in split_data:
            context = item["context"].strip()
            question = item["question"].strip()

            # Извлекаем ответы
            answers_data = item.get("answers", {})
            if isinstance(answers_data, dict):
                answer_texts = answers_data.get("text", [])
                answer_starts = answers_data.get("answer_start", [])
            elif isinstance(answers_data, list):
                answer_texts = [a.get("text", "") for a in answers_data]
                answer_starts = [a.get("answer_start", 0) for a in answers_data]
            else:
                skipped += 1
                continue

            # Фильтруем пустые ответы
            valid_answers = []
            for text, start in zip(answer_texts, answer_starts):
                text = text.strip() if isinstance(text, str) else ""
                if text:
                    valid_answers.append({
                        "text": text,
                        "answer_start": start if isinstance(start, int) else 0,
                    })

            if not valid_answers:
                skipped += 1
                continue

            qa_entry = {
                "question": question,
                "answers": valid_answers,
                "id": item.get("id", f"{hash(context + question)}"),
            }
            context_groups[context].append(qa_entry)

        # Формируем SQuAD-структуру
        data = {"data": []}
        for context, qas in context_groups.items():
            data["data"].append({
                "paragraphs": [{
                    "context": context,
                    "qas": qas,
                }]
            })

        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Статистика
        total_qas = sum(len(p["qas"]) for article in data["data"] for p in article["paragraphs"])
        total_contexts = sum(len(article["paragraphs"]) for article in data["data"])
        avg_ctx_len = 0
        if total_contexts > 0:
            all_lengths = [
                len(p["context"].split())
                for article in data["data"]
                for p in article["paragraphs"]
            ]
            avg_ctx_len = sum(all_lengths) / len(all_lengths)

        print(f"\n📊 {split_name}:")
        print(f"  Файл:                 {filepath}")
        print(f"  Уникальных контекстов: {total_contexts}")
        print(f"  QA-пар:               {total_qas}")
        print(f"  Пропущено (пустые):   {skipped}")
        print(f"  Средняя длина контекста: {avg_ctx_len:.0f} слов")

    print("\n✅ Загрузка завершена!")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка SberQUAD из Hugging Face Hub"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Папка для сохранения файлов (default: ./data/raw)",
    )
    args = parser.parse_args()
    download_and_convert(args.output_dir)


if __name__ == "__main__":
    main()
