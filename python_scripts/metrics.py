"""
metrics.py
==========
EM (Exact Match) и F1 для SQuAD-подобных задач.
Адаптировано для русского языка: нормализация пунктуации, артиклей не нужно
(русский без артиклей), lowercase, удаление лишних пробелов.

Основано на официальном SQuAD evaluation script.
"""

import re
import string
from collections import Counter

def compute_cover_em(prediction: str, ground_truth: str) -> float:
    """Проверяет содержится ли gold answer как подмножество токенов в prediction."""
    pred_tokens = set(normalize_answer(prediction).split())
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return 1.0
    return float(all(t in pred_tokens for t in gold_tokens))

def normalize_answer(s: str) -> str:
    """Lowercase + убрать пунктуацию + схлопнуть пробелы."""

    def remove_punc(text):
        # Включает русскую и английскую пунктуацию
        exclude = set(string.punctuation + "«»„""''–—…")
        return "".join(ch for ch in text if ch not in exclude)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 если нормализованные строки совпадают, иначе 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 между prediction и ground_truth (после нормализации)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # Если обе пустые — совпадение, иначе 0
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_metrics(predictions: list, references: list) -> dict:
    """
    Вычисляет средние EM и F1 по батчу.

    Args:
        predictions: список строк-предсказаний
        references: список строк-эталонных ответов
    """
    assert len(predictions) == len(references)

    em_scores = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
    cover_em_scores = [compute_cover_em(p, r) for p, r in zip(predictions, references)]
    return {
        "exact_match": 100.0 * sum(em_scores) / len(em_scores),
        "f1": 100.0 * sum(f1_scores) / len(f1_scores),
        "cover_em": 100.0 * sum(cover_em_scores) / len(cover_em_scores),
        "num_examples": len(predictions),
    }

def extract_answer_from_generation(text: str, max_sentences: int = 2) -> str:
    """
    Извлекает короткий ответ из сгенерированного текста.
    LLM часто отвечает полным предложением, а gold-ответ — это короткий span.
    Для честного F1 берём первое предложение или первые N слов.

    Простая эвристика: обрезаем до первой точки/новой строки.
    """
    text = text.strip()

    # Убираем стандартные префиксы
    prefixes_to_remove = [
        "Ответ:", "ответ:", "Answer:", "answer:",
        "A:", "а:",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Берём первые max_sentences предложений
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = " ".join(sentences[:max_sentences]).strip()

    return result if result else text


if __name__ == "__main__":
    # Быстрый тест
    preds = [
        "Москва",
        "Москва — столица России.",
        "Столицей является Москва",
        "Санкт-Петербург",
    ]
    refs = [
        "Москва",
        "Москва",
        "Москва",
        "Москва",
    ]

    for p, r in zip(preds, refs):
        em = compute_exact_match(p, r)
        f1 = compute_f1(p, r)
        print(f"  pred='{p}' | gold='{r}' | EM={em:.2f} F1={f1:.2f}")

    print()
    print("Aggregate:", compute_metrics(preds, refs))
