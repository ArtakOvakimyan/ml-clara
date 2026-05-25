#!/usr/bin/env python3
"""
evaluate_baseline.py
====================
Baseline RAG для сравнения с CLaRa:
  - Модель: Qwen2-0.5B-Instruct (без дообучения, без компрессии)
  - Промпт: идентичен CLaRa _blend_standard_prompt (для честного сравнения)
  - Вход: полный текст документа + вопрос
  - Выход: ответ

Использует тот же eval sample, что и evaluate_clara.py.

Использование:
    python evaluate_baseline.py \
        --eval_file ./data/eval/sample_500.jsonl \
        --output ./results/baseline_results.json \
        --max_new_tokens 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import compute_metrics, extract_answer_from_generation,compute_cover_em


# Тот же системный промпт, что в CLaRa _blend_standard_prompt
SYSTEM_PROMPT = (
        "Ты — полезный ассистент. Твоя задача — извлечь нужную информацию "
        "из предоставленных документов и ответить на вопрос максимально кратко."
)


def build_prompt(tokenizer, context: str, question: str) -> str:
    """Строит prompt через chat template — идентично CLaRa."""
    user_content = f"Контекст:\n{context}\n\n\Вопрос:{question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return prompt


def load_eval_examples(eval_file: str) -> list:
    """Загружает JSONL-файл с eval-примерами."""
    examples = []
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate(
    model_name: str,
    eval_file: str,
    output_file: str,
    max_new_tokens: int,
    batch_size: int,
    device: str,
):
    print(f"📦 Загрузка модели {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, enable_thinking=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # важно для generate

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"  ✓ Модель на {device}, dtype={model.dtype}")

    examples = load_eval_examples(eval_file)
    print(f"\n📂 Загружено {len(examples)} eval примеров")

    predictions = []
    references = []
    prompt_token_lengths = []
    per_example_times = []

    print(f"\n🔄 Генерация (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
    start_total = time.time()

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [build_prompt(tokenizer, ex["context"], ex["question"]) for ex in batch]

        # Токенизация
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        prompt_token_lengths.extend(prompt_lens)

        # Генерация
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        per_example_times.append(elapsed / len(batch))

        # Декодирование только сгенерированной части
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for ex, raw_pred in zip(batch, decoded):
            pred = extract_answer_from_generation(raw_pred)
            predictions.append(pred)
            references.append(ex["answer"])

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(examples))
            pct = 100.0 * done / len(examples)
            print(f"  [{done}/{len(examples)}] {pct:.0f}%")

    total_time = time.time() - start_total

    # Метрики
    metrics = compute_metrics(predictions, references)

    avg_prompt_tokens = sum(prompt_token_lengths) / len(prompt_token_lengths)
    avg_time_per_example = sum(per_example_times) / len(per_example_times)

    results = {
        "method": "baseline_full_context",
        "model": model_name,
        "eval_file": eval_file,
        "num_examples": len(examples),
        "metrics": {
            "exact_match": round(metrics["exact_match"], 2),
            "f1": round(metrics["f1"], 2),
            "cover_em": round(metrics["cover_em"], 2),
        },
        "efficiency": {
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "max_prompt_tokens": max(prompt_token_lengths),
            "avg_time_per_example_sec": round(avg_time_per_example, 3),
            "total_time_sec": round(total_time, 1),
        },
        "config": {
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "dtype": str(model.dtype),
        },
    }

    # Сохранение
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Сохраняем также pred/ref для ручной проверки
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
    print(f"  Результаты Baseline (Qwen2-0.5B full-context)")
    print(f"{'=' * 50}")
    print(f"  Exact Match: {metrics['exact_match']:.2f}%")
    print(f"  F1:          {metrics['f1']:.2f}%")
    print(f"  Средняя длина промпта: {avg_prompt_tokens:.0f} токенов")
    print(f"  Время на пример:       {avg_time_per_example*1000:.0f} мс")
    print(f"  Общее время:           {total_time:.1f} сек")
    print(f"\n  Результаты: {output_file}")
    print(f"  Детали:     {details_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline RAG evaluation")
    parser.add_argument(
        "--model_name", default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--eval_file", default="./data/eval/sample_500.jsonl",
    )
    parser.add_argument(
        "--output", default="./results/baseline_results.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evaluate(
        model_name=args.model_name,
        eval_file=args.eval_file,
        output_file=args.output,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
