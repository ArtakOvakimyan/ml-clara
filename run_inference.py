#!/usr/bin/env python3
"""
run_inference.py
================
Проверка обученной CLaRa + Qwen2 на русских вопросах.

Использование:
    # CLaRa Stage 2 — ответ через сжатый документ
    python "run_inference (1).py" \
        --checkpoint ./checkpoints/stage2 \
        --stage 2

    # Baseline — обычный Qwen2 с полным документом (без сжатия)
    python "run_inference (1).py" \
        --baseline \
        --model_name t-tech/T-lite-it-1.0
"""

import argparse
import sys


def run_baseline(model_name: str, question: str, document: str, max_tokens: int):
    """Запускает baseline инференс: обычный Qwen2 с полным документом в контексте."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n📦 Загрузка базовой модели {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  ✓ Модель загружена на {device}")

    print(f"\n📄 Документ:  {document[:200]}{'...' if len(document) > 200 else ''}")
    print(f"❓ Вопрос:   {question}")
    print(f"\n🔄 Baseline генерация (max_tokens={max_tokens})...")

    # Тот же chat template что и CLaRa использует для stage1_2
    prompt_system = (
        "Ты — полезный ассистент. Твоя задача — извлечь нужную информацию "
        "из предоставленных документов и ответить на вопрос максимально кратко."
    )
    prompt_user = f"Контекст:\n{document}\n\nВопрос:{question}"

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user",   "content": prompt_user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback если system role не поддерживается
        combined = prompt_system + "\n" + prompt_user
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": combined}],
            tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Декодируем только новые токены (без prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"\n💡 Baseline ответ:\n   {answer}")
    return answer


def run_inference(checkpoint: str, stage: int, question: str, document: str, max_tokens: int):
    """Запускает инференс CLaRa."""
    import torch
    import os

    print(f"\n📦 Загрузка модели из {checkpoint}...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from openrlhf.models.modeling_clara import CLaRa

    model = CLaRa.from_pretrained(checkpoint, pure_inference=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠ GPU не найден, используем CPU (будет медленно)")
    model = model.to(device)
    model.eval()
    print(f"  ✓ Модель загружена на {device}")

    documents = [[document]]
    questions = [question]

    print(f"\n📄 Документ:  {document[:200]}{'...' if len(document) > 200 else ''}")
    print(f"❓ Вопрос:   {question}")
    print(f"\n🔄 Генерация (stage={stage}, max_tokens={max_tokens})...")

    with torch.no_grad():
        if stage == 1:
            output = model.generate_from_paraphrase(
                questions=[""],
                documents=documents,
                max_new_tokens=max_tokens,
            )
            print(f"\n📝 Парафраз:\n   {output[0]}")
        elif stage == 2:
            output = model.generate_from_text(
                questions=questions,
                documents=documents,
                max_new_tokens=max_tokens,
            )
            print(f"\n💡 Ответ:\n   {output[0]}")
        else:
            print(f"✗ Неизвестная стадия: {stage}")
            sys.exit(1)

    return output[0]


def main():
    parser = argparse.ArgumentParser(description="Инференс CLaRa / Baseline Qwen2")
    parser.add_argument("--checkpoint", help="Путь к чекпоинту CLaRa (не нужен для --baseline)")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2],
                        help="Стадия CLaRa: 1=парафраз, 2=QA (default: 2)")
    parser.add_argument("--baseline", action="store_true",
                        help="Запустить baseline: обычный Qwen2 без CLaRa компрессии")
    parser.add_argument("--model_name", default="t-tech/T-lite-it-1.0",
                        help="HF model name для baseline (default: t-tech/T-lite-it-1.0)")
    parser.add_argument("--question", default="Какой город является столицей России?")
    parser.add_argument("--document", default=(
        "Москва — столица Российской Федерации, город федерального значения, "
        "административный центр Центрального федерального округа и центр Московской области."
    ))
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    if args.baseline:
        run_baseline(
            model_name=args.model_name,
            question=args.question,
            document=args.document,
            max_tokens=args.max_tokens,
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint обязателен без --baseline")
        run_inference(
            checkpoint=args.checkpoint,
            stage=args.stage,
            question=args.question,
            document=args.document,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
