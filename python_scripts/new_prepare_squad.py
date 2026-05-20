#!/usr/bin/env python3
"""
prepare_sberquad.py
===================
Конвертирует SberQUAD (SQuAD-формат) → JSONL для CLaRa Stage 1 и Stage 2.

Выходные файлы:
    stage1_pretrain.jsonl     — Stage 1 train (из sberquad_train, ~80%)
    stage1_val.jsonl          — Stage 1 eval  (из sberquad_train, последние N=200)
    stage2_instruction.jsonl  — Stage 2 train (из sberquad_validation, ~80%)
    stage2_val.jsonl          — Stage 2 eval  (из sberquad_validation, последние N=200)

Использование:
    python prepare_sberquad.py
    python prepare_sberquad.py --max_samples 100 --output_dir ./data/debug
    python prepare_sberquad.py --n_val 100  # меньше val примеров
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_sberquad_grouped(path: str) -> list[dict]:
    """Загружает SberQUAD и группирует QA-пары по контексту."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    groups = defaultdict(lambda: {"questions": [], "answers": []})

    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"].strip()
            for qa in para["qas"]:
                question = qa["question"].strip()
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer = answers[0]["text"].strip()
                if not answer:
                    continue
                groups[context]["questions"].append(question)
                groups[context]["answers"].append(answer)

    return [
        {"context": ctx, "questions": d["questions"], "answers": d["answers"]}
        for ctx, d in groups.items()
    ]


def make_clara_data(samples: list[dict], max_doc_words: int = 300) -> list[dict]:
    """Конвертирует сгруппированные QA-записи в формат CLaRa."""
    result = []
    skipped_long = 0
    skipped_empty = 0

    for s in samples:
        context = s["context"]
        questions = s["questions"]
        answers = s["answers"]

        if len(context.split()) > max_doc_words:
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Сохранено {len(data)} записей → {path}")


def print_stats(data: list[dict], label: str) -> None:
    total_docs = len(data)
    if total_docs == 0:
        print(f"  📊 {label}: 0 записей")
        return
    total_qa = sum(len(item["question"]) for item in data)
    avg_qa = total_qa / total_docs
    doc_lengths = [len(item["docs"][0].split()) for item in data]
    avg_len = sum(doc_lengths) / len(doc_lengths)
    print(f"  📊 {label}: {total_docs} документов, "
          f"{total_qa} QA-пар ({avg_qa:.1f}/doc), "
          f"ср. длина документа {avg_len:.0f} слов")


def split_train_val(samples: list[dict], n_val: int) -> tuple[list, list]:
    """
    Отделяет последние n_val примеров как val set.
    Не более 20% датасета чтобы не резать train сильно.
    """
    n_val = min(n_val, max(1, len(samples) // 5))
    return samples[:-n_val], samples[-n_val:]


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация SberQUAD → CLaRa формат (Stage 1 и Stage 2)"
    )
    parser.add_argument(
        "--train_input",
        default="./rus_data/raw/sberquad_train.json",
        help="Путь к sberquad_train.json",
    )
    parser.add_argument(
        "--val_input",
        default="./rus_data/raw/sberquad_validation.json",
        help="Путь к sberquad_validation.json",
    )
    parser.add_argument(
        "--output_dir",
        default="./rus_data/clara",
        help="Папка для выходных JSONL файлов",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Лимит обучающих записей (документов). None = все.",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=200,
        help="Количество записей в val set для логирования метрик. Default: 200.",
    )
    parser.add_argument(
        "--max_doc_words",
        type=int,
        default=300,
        help="Макс. длина документа в словах (≈ 256 токенов Qwen2). Default: 300.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)

    # ─── Stage 1 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 1: Compression Pretraining (SCP)")
    print("=" * 60)
    print(f"\n  📂 Загрузка: {args.train_input}")

    train_samples = load_sberquad_grouped(args.train_input)
    print(f"  Загружено {len(train_samples)} уникальных контекстов")

    # Сначала отделяем val, потом применяем max_samples только к train
    s1_train_full, s1_val_raw = split_train_val(train_samples, args.n_val)

    if args.max_samples:
        s1_train_full = s1_train_full[:args.max_samples]
        print(f"  Ограничено до {len(s1_train_full)} обучающих (--max_samples)")

    s1_train = make_clara_data(s1_train_full, max_doc_words=args.max_doc_words)
    s1_val   = make_clara_data(s1_val_raw,   max_doc_words=args.max_doc_words)

    save_jsonl(s1_train, out / "stage1_pretrain.jsonl")
    save_jsonl(s1_val,   out / "stage1_val.jsonl")
    print_stats(s1_train, "Stage 1 train")
    print_stats(s1_val,   "Stage 1 val")

    # ─── Stage 2 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 2: Compression Instruction Tuning")
    print("=" * 60)

    val_input = args.val_input
    if not Path(val_input).exists():
        print(f"  ⚠ {val_input} не найден, используем train для Stage 2")
        val_input = args.train_input

    print(f"\n  📂 Загрузка: {val_input}")
    val_samples = load_sberquad_grouped(val_input)
    print(f"  Загружено {len(val_samples)} уникальных контекстов")

    s2_train_full, s2_val_raw = split_train_val(val_samples, args.n_val)

    if args.max_samples:
        s2_train_full = s2_train_full[:args.max_samples]
        print(f"  Ограничено до {len(s2_train_full)} обучающих (--max_samples)")

    s2_train = make_clara_data(s2_train_full, max_doc_words=args.max_doc_words)
    s2_val   = make_clara_data(s2_val_raw,   max_doc_words=args.max_doc_words)

    save_jsonl(s2_train, out / "stage2_instruction.jsonl")
    save_jsonl(s2_val,   out / "stage2_val.jsonl")
    print_stats(s2_train, "Stage 2 train")
    print_stats(s2_val,   "Stage 2 val")

    # ─── Пример вывода ────────────────────────────────────────
    if s1_train:
        print("\n" + "=" * 60)
        print("  Пример (Stage 1 train)")
        print("=" * 60)
        ex = s1_train[0]
        print(f"  doc:       {ex['docs'][0][:100]}...")
        print(f"  questions: {ex['question'][:2]}")
        print(f"  answers:   {ex['answers'][:2]}")

    print("\n" + "=" * 60)
    print("  ✅ Данные подготовлены!")
    print("=" * 60)
    print(f"\n  Используй в run_training.sh:")
    print(f"    Stage 1: --dataset {out}/stage1_pretrain.jsonl")
    print(f"             --eval_dataset {out}/stage1_val.jsonl")
    print(f"    Stage 2: --dataset {out}/stage2_instruction.jsonl")
    print(f"             --eval_dataset {out}/stage2_val.jsonl")
    print()


if __name__ == "__main__":
    main()