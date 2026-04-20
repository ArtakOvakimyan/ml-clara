#!/usr/bin/env python3
"""
prepare_sberquad.py
===================
Конвертирует SberQUAD (SQuAD-формат) → JSONL для CLaRa Stage 1 и Stage 2.
Stage 3 (E2E) не включён — требует отдельного корпуса негативных документов
и значительно больше ресурсов.

Ключевые отличия от наивной реализации:
  1. QA-пары группируются по контексту (как в оригинальном CLaRa).
     Одна строка JSONL = один документ + все его QA-пары.
  2. Для Stage 2 формат идентичен Stage 1, но данные берутся из
     другого сплита (validation), чтобы не было утечки.
  3. Поддержка --max_samples и --max_doc_words для отладки на T4.
  4. Фильтрация документов, которые слишком длинные для CLaRa
     (doc_max_length=256 токенов ≈ 200 слов с запасом).

Формат CLaRa Stage 1/2 (из example/pretrain_data.jsonl):
{
    "data_type": "qa",
    "question": ["Вопрос 1?", "Вопрос 2?"],
    "answers":  ["Ответ 1",   "Ответ 2"],
    "docs":     ["Текст документа"]
}

Использование:
    # Полный прогон
    python prepare_sberquad.py

    # Мини-слепок для отладки на T4
    python prepare_sberquad.py --max_samples 100 --output_dir ./data/debug

    # С другим путём к данным
    python prepare_sberquad.py \
        --train_input ./data/raw/sberquad_train.json \
        --val_input ./data/raw/sberquad_validation.json \
        --output_dir ./data/clara
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import os


def load_sberquad_grouped(path: str) -> list[dict]:
    """
    Загружает SberQUAD и группирует QA-пары по контексту.

    Возвращает список записей:
    {
        "context": str,
        "questions": [str, ...],
        "answers": [str, ...],
    }
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Группируем по контексту
    groups = defaultdict(lambda: {"questions": [], "answers": []})

    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"].strip()
            for qa in para["qas"]:
                question = qa["question"].strip()
                # Берём первый (канонический) ответ
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer = answers[0]["text"].strip()
                if not answer:
                    continue

                groups[context]["questions"].append(question)
                groups[context]["answers"].append(answer)

    result = []
    for context, qa_data in groups.items():
        result.append({
            "context": context,
            "questions": qa_data["questions"],
            "answers": qa_data["answers"],
        })

    return result


def make_clara_data(
    samples: list[dict],
    max_doc_words: int = 300,
) -> list[dict]:
    """
    Конвертирует сгруппированные QA-записи в формат CLaRa Stage 1/2.

    CLaRa ожидает doc_max_length=256 токенов. Для Qwen2 токенизатор
    примерно 1 слово ≈ 1.3 токена (для русского чуть больше), поэтому
    фильтруем документы длиннее max_doc_words слов.

    Args:
        samples: Сгруппированные QA-записи
        max_doc_words: Максимальная длина документа в словах
    """
    result = []
    skipped_long = 0
    skipped_empty = 0

    for s in samples:
        context = s["context"]
        questions = s["questions"]
        answers = s["answers"]

        # Фильтр по длине
        word_count = len(context.split())
        if word_count > max_doc_words:
            skipped_long += 1
            continue

        if not questions:
            skipped_empty += 1
            continue

        result.append({
            "data_type": "qa",
            "question": questions,
            "answers": answers,
            "docs": [context],
        })

    if skipped_long > 0:
        print(f"  ⚠ Пропущено {skipped_long} документов (>{max_doc_words} слов)")
    if skipped_empty > 0:
        print(f"  ⚠ Пропущено {skipped_empty} документов (без QA-пар)")

    return result


def save_jsonl(data: list[dict], path: str) -> None:
    """Сохраняет список словарей в JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Сохранено {len(data)} записей → {path}")


def print_stats(data: list[dict], label: str) -> None:
    """Выводит статистику по подготовленным данным."""
    total_docs = len(data)
    total_qa = sum(len(item["question"]) for item in data)
    avg_qa_per_doc = total_qa / total_docs if total_docs > 0 else 0

    doc_lengths = [len(item["docs"][0].split()) for item in data if item["docs"]]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    max_doc_len = max(doc_lengths) if doc_lengths else 0
    min_doc_len = min(doc_lengths) if doc_lengths else 0

    print(f"\n  📊 Статистика {label}:")
    print(f"     Записей (документов): {total_docs}")
    print(f"     QA-пар всего:         {total_qa}")
    print(f"     QA на документ:       {avg_qa_per_doc:.1f} (среднее)")
    print(f"     Длина документа:      {min_doc_len}-{max_doc_len} слов "
          f"(среднее: {avg_doc_len:.0f})")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация SberQUAD → CLaRa формат (Stage 1 и Stage 2)"
    )
    parser.add_argument(
        "--train_input",
        help="Путь к sberquad_train.json",
    )
    parser.add_argument(
        "--val_input",
        help="Путь к sberquad_validation.json",
    )
    parser.add_argument(
        "--output_dir",
        help="Папка для выходных JSONL файлов",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Лимит записей (документов) для отладки. None = все.",
    )
    parser.add_argument(
        "--max_doc_words",
        type=int,
        default=300,
        help="Макс. длина документа в словах (≈ 256 токенов Qwen2). "
             "Default: 300.",
    )
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    if args.output_dir is None:
        args.output_dir = os.path.join(root_dir, "clara")
    if args.train_input is None:
        args.train_input = os.path.join(root_dir, "raw", "sberquad_train.json")
    if args.val_input is None:
        args.val_input = os.path.join(root_dir, "raw", "sberquad_validation.json")
    out = Path(args.output_dir)

    # ─── Stage 1: Compression Pretraining ───────────────────
    print("\n" + "=" * 60)
    print("  Stage 1: Compression Pretraining (SCP)")
    print("=" * 60)
    print(f"\n  📂 Загрузка: {args.train_input}")

    train_samples = load_sberquad_grouped(args.train_input)
    print(f"  Загружено {len(train_samples)} уникальных контекстов")

    if args.max_samples:
        train_samples = train_samples[: args.max_samples]
        print(f"  Ограничено до {len(train_samples)} (--max_samples)")

    s1 = make_clara_data(train_samples, max_doc_words=args.max_doc_words)
    save_jsonl(s1, out / "stage1_pretrain.jsonl")
    print_stats(s1, "Stage 1")

    # ─── Stage 2: Instruction Tuning ────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 2: Compression Instruction Tuning")
    print("=" * 60)

    # Stage 2 берёт данные из validation (или из train, если val нет)
    if not Path(args.val_input).exists():
        print(f"  ⚠ Файл {args.val_input} не найден, используем train для Stage 2")
        args.val_input = args.train_input

    print(f"\n  📂 Загрузка: {args.val_input}")
    val_samples = load_sberquad_grouped(args.val_input)
    print(f"  Загружено {len(val_samples)} уникальных контекстов")

    if args.max_samples:
        val_samples = val_samples[: args.max_samples]
        print(f"  Ограничено до {len(val_samples)} (--max_samples)")

    s2 = make_clara_data(val_samples, max_doc_words=args.max_doc_words)
    save_jsonl(s2, out / "stage2_instruction.jsonl")
    print_stats(s2, "Stage 2")

    # ─── Примеры вывода ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Примеры данных")
    print("=" * 60)

    if s1:
        example = s1[0]
        print("\n  Stage 1 пример:")
        print(f"    data_type:  {example['data_type']}")
        print(f"    doc (начало): {example['docs'][0][:100]}...")
        print(f"    questions ({len(example['question'])}): {example['question'][:2]}")
        print(f"    answers ({len(example['answers'])}):   {example['answers'][:2]}")

    # ─── Итог ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Данные подготовлены!")
    print("=" * 60)
    print(f"\n  {out}/stage1_pretrain.jsonl    — Stage 1 (SCP)")
    print(f"  {out}/stage2_instruction.jsonl — Stage 2 (Instruction Tuning)")
    print(f"\n  Следующий шаг: адаптация modeling_clara.py под Qwen2-0.5B")
    print()


if __name__ == "__main__":
    main()