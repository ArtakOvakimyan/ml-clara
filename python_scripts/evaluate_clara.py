#!/usr/bin/env python3
"""
evaluate_clara.py
=================
Оценка обученной CLaRa Stage 2 модели на том же eval sample,
что и baseline — для честного сравнения.

Использование:
    python evaluate_clara.py \
        --checkpoint ./checkpoints/stage2 \
        --eval_file ./data/eval/sample_500.jsonl \
        --output ./results/clara_results.json \
        --max_new_tokens 64
"""

import argparse
import json
import time
from pathlib import Path
from openrlhf.models.modeling_clara import CLaRa
import torch
from transformers import AutoModel

from metrics import compute_metrics, extract_answer_from_generation


def load_eval_examples(eval_file: str) -> list:
    examples = []
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def count_memory_tokens(context_length_words: int, compress_rate: int, doc_max_length: int) -> int:
    """Сколько memory-токенов использует CLaRa для документа."""
    # CLaRa обрезает документ до doc_max_length токенов, затем сжимает в
    # doc_max_length / compress_rate memory-токенов
    return doc_max_length // compress_rate


def evaluate(
    checkpoint: str,
    eval_file: str,
    output_file: str,
    max_new_tokens: int,
    device: str,
):
    print(f"📦 Загрузка CLaRa из {checkpoint}...")
    # model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    model = CLaRa.from_pretrained(checkpoint, pure_inference=True)
    model = model.to(device)
    model.eval()
    print(f"  ✓ Модель на {device}")

    # Из config узнаём параметры компрессии
    import json as json_module
    with open(f"{checkpoint}/config.json") as f:
        cfg = json_module.load(f)
    compress_rate = cfg.get("compr_rate", 16)
    doc_max_length = cfg.get("doc_max_length", 256)
    n_mem_tokens = doc_max_length // compress_rate
    print(f"  compr_rate={compress_rate}, doc_max_length={doc_max_length}")
    print(f"  → {n_mem_tokens} memory-токенов на документ")

    examples = load_eval_examples(eval_file)
    print(f"\n📂 Загружено {len(examples)} eval примеров")

    predictions = []
    references = []
    per_example_times = []

    print(f"\n🔄 Генерация (max_new_tokens={max_new_tokens})...")
    print(f"  (CLaRa обрабатывает примеры по одному — batch API не тривиален)")
    start_total = time.time()

    for i, ex in enumerate(examples):
        # CLaRa API: documents — это List[List[str]], questions — List[str]
        documents = [[ex["context"]]]
        questions = [ex["question"]]

        t0 = time.time()
        try:
            with torch.no_grad():
                output = model.generate_from_text(
                    questions=questions,
                    documents=documents,
                    max_new_tokens=max_new_tokens,
                )
            elapsed = time.time() - t0
            per_example_times.append(elapsed)

            raw_pred = output[0] if isinstance(output, list) else output
            pred = extract_answer_from_generation(raw_pred)
        except Exception as e:
            print(f"  ⚠ Ошибка на примере {i}: {e}")
            pred = ""
            per_example_times.append(0.0)

        predictions.append(pred)
        references.append(ex["answer"])

        if (i + 1) % 25 == 0:
            pct = 100.0 * (i + 1) / len(examples)
            avg_t = sum(per_example_times) / len(per_example_times)
            print(f"  [{i+1}/{len(examples)}] {pct:.0f}% — среднее время {avg_t*1000:.0f} мс/пример")

    total_time = time.time() - start_total

    # Метрики
    metrics = compute_metrics(predictions, references)
    avg_time = sum(per_example_times) / len(per_example_times)

    # Длина "промпта" для CLaRa = memory-токены + системный промпт + вопрос
    # Точное число зависит от chat template, оценим примерно:
    # sys_prompt + "Background:\n" + mem_tokens + "\n\nQuestion:" + question
    # mem_tokens × 1 вектор каждый = n_mem_tokens позиций
    # Плюс ~50 токенов служебных + длина вопроса
    approx_prompt_tokens_compressed = n_mem_tokens + 60  # очень грубая оценка

    results = {
        "method": "clara_stage2_compressed",
        "checkpoint": checkpoint,
        "eval_file": eval_file,
        "num_examples": len(examples),
        "metrics": {
            "exact_match": round(metrics["exact_match"], 2),
            "f1": round(metrics["f1"], 2),
            "cover_em": round(metrics["cover_em"], 2),

        },
        "efficiency": {
            "memory_tokens_per_doc": n_mem_tokens,
            "compression_rate": compress_rate,
            "approx_prompt_tokens": approx_prompt_tokens_compressed,
            "avg_time_per_example_sec": round(avg_time, 3),
            "total_time_sec": round(total_time, 1),
        },
        "config": {
            "max_new_tokens": max_new_tokens,
            "compr_rate": compress_rate,
            "doc_max_length": doc_max_length,
        },
    }

    # Сохранение
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    details_file = output_file.replace(".json", "_details.jsonl")
    with open(details_file, "w", encoding="utf-8") as f:
        for ex, pred in zip(examples, predictions):
            f.write(json.dumps({
                "id": ex["id"],
                "question": ex["question"],
                "gold": ex["answer"],
                "prediction": pred,
            }, ensure_ascii=False) + "\n")

    # Вывод
    print(f"\n{'=' * 50}")
    print(f"  Результаты CLaRa (Qwen2-0.5B compressed)")
    print(f"{'=' * 50}")
    print(f"  Exact Match:          {metrics['exact_match']:.2f}%")
    print(f"  F1:                   {metrics['f1']:.2f}%")
    print(f"  Memory tokens/doc:    {n_mem_tokens} (сжатие {compress_rate}×)")
    print(f"  Время на пример:      {avg_time*1000:.0f} мс")
    print(f"  Общее время:          {total_time:.1f} сек")
    print(f"\n  Результаты: {output_file}")
    print(f"  Детали:     {details_file}")


def main():
    parser = argparse.ArgumentParser(description="CLaRa evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval_file", default="./data/eval/sample_500.jsonl",
    )
    parser.add_argument(
        "--output", default="./results/clara_results.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        eval_file=args.eval_file,
        output_file=args.output,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
