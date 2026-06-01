# CLaRa + Qwen3-0.6B + SberQUAD

Адаптация фреймворка [Apple CLaRa](https://github.com/apple/ml-clara) (Continuous Latent Reasoning) для русскоязычного вопросно-ответного задания. Модель сжимает документ из ~256 токенов в 16 memory-токенов (16× компрессия) и генерирует ответ из сжатого представления.

Обучение проведено на датасете [SberQUAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) с моделью [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Реализованы Stage 1 (Salient Compressor Pretraining) и Stage 2 (Compression Instruction Tuning).

---

## Результаты

Оценка на 500 примерах из validation SberQUAD:

| Метрика | Baseline (full-context) | CLaRa (16× сжатие) | Δ |
|---|---|---|---|
| Exact Match | 6.00% | 39.20% | +33.20 |
| F1 | 39.87% | 54.35% | +14.48 |
| cover_EM | 61.60% | 40.20% | −21.40 |
| Длина промпта (токены) | 343 | ~76 | 4.5× короче |
| Время на пример | 566 мс | 1057 мс | 1.9× медленнее |

Разница EM/F1 vs cover_EM объясняется стилем генерации: CLaRa обучена на span-ответах SberQUAD и генерирует короткие точные фрагменты, baseline — полные предложения, содержащие ответ как подстроку.

Верификация компрессора (тест с контрфактическим документом):

| Документ | Вопрос | Ответ CLaRa |
|---|---|---|
| «Клубника — зелёная» | Какого цвета клубника? | Зелёного цвета |
| «Арбуз — голубой» | Какого цвета арбуз? | Голубой |

Модель отвечает из сжатого контекста, а не из parametric knowledge.

---

## Порядок запуска

### 1. Окружение (один раз)

```bash
bash scripts/setup_env.sh
conda activate clara-qwen2
```

Устанавливает conda-окружение с Python 3.10, PyTorch 2.4.1+cu118, Transformers, DeepSpeed, PEFT и остальные зависимости.

### 2. Подготовка данных

```bash
bash scripts/setup_data.sh
```

Скачивает SberQUAD из HuggingFace и конвертирует в формат CLaRa. Генерирует четыре файла:

```
rus_data/clara/
├── stage1_pretrain.jsonl       ← Stage 1 train (~9K контекстов)
├── stage1_val.jsonl            ← Stage 1 eval (20 контекстов)
├── stage2_instruction.jsonl    ← Stage 2 train (~3.9K контекстов)
└── stage2_val.jsonl            ← Stage 2 eval (20 контекстов)
```

Для отладки на малом количестве данных:

```bash
python rus_data/scripts/prepare_sberquad.py --max_samples 100 --n_val 20
```

### 3. Обучение

```bash
bash scripts/run_training.sh
```

Последовательно запускает Stage 1, Stage 2 и fix_config. Параметры настраиваются в начале скрипта.

Поддерживаются отдельные запуски:

```bash
bash scripts/run_training.sh --stage1_only
bash scripts/run_training.sh --stage2_only
```

### 4. Инференс

```bash
python run_inference.py \
    --checkpoint checkpoints/stage2 \
    --stage 2 \
    --question "Какого цвета клубника?" \
    --document "Клубника — зелёная"
```

### 5. Оценка

```bash
# Подготовить eval-выборку (500 примеров из validation)
python prepare_eval_sample.py

# Оценить CLaRa
python evaluate_clara.py --checkpoint checkpoints/stage2

# Оценить baseline (Qwen3-0.6B full-context)
python evaluate_baseline.py --model_name Qwen/Qwen3-0.6B

# Сравнить
python compare_results.py
```

---

## Что изменено в apple/ml-clara

Все модификации хранятся как `.patch` файлы в `patches/`. Подробное описание — в [patches/README.md](patches/README.md).

**Замена Flash Attention на SDPA.** V100 (Volta) не поддерживает Flash Attention 2 (требует Ampere). Заменён на Scaled Dot-Product Attention, встроенный в PyTorch. На A100/H100 можно вернуть Flash Attention.

**Адаптация размерностей компрессора.** `compr_mlp_hidden_dim` пересчитан как `hidden_size × 2`: 8096 (Mistral-7B) → 2048 (Qwen3-0.6B).

**Удаление `enable_thinking`.** Параметр специфичен для Qwen3 thinking mode и не нужен при обучении CLaRa.

**Инициализация bos/eos токенов.** Для Qwen3 явно установлены `<|im_start|>` и `<|im_end|>` вместо индексации через `additional_special_tokens`, которая указывала на memory-токены.

**Перевод системных промптов на русский.** Все инструкции компрессора и генератора переведены.

**Конвертация list→str.** SberQUAD группирует QA-пары по контексту — поля приходят как списки, оригинальный код ожидает строки.

**Добавление eval/f1 в мониторинг.** В `sft_trainer.py` добавлено вычисление F1 при eval_gen и логирование в Weights & Biases для отслеживания качества компрессии по эпохам.

**Совместимость зависимостей.** Планировщик `cosine_with_min_lr` из оригинального кода использовал параметр `min_lr`, отсутствующий в публичной версии Transformers. Также зафиксированы версии PEFT и NumPy для устранения конфликтов.

---

## Параметры обучения

| Параметр | Stage 1 | Stage 2 |
|---|---|---|
| Данные | ~9K контекстов (train) | ~3.9K контекстов (validation) |
| Эпох | 10 | 4–8 (оптимум ~4 эпохи) |
| Learning rate | 1e-3 | 3e-4 |
| Scheduler | cosine_with_min_lr (min = lr × 0.1) | cosine_with_min_lr |
| Micro batch size | 4 | 4 |
| Effective batch size | 28 | 28 |
| compress_rate | 16 | 16 |
| doc_max_length | 256 | 256 |
| LoRA rank | 16 | 16 |
| Weight decay | 0 | 0 |

Оптимальный чекпоинт Stage 2 определён по пику eval/f1 на валидационных данных: `global_step600`.

---

## Методика экспериментов

**Sanity check.** Перед обучением на полном датасете проведена верификация пайплайна: намеренное переобучение на 100 примерах (30 эпох Stage 1, 20 эпох Stage 2, LR=1e-3). eval/f1 достиг 1.0 — подтверждена корректность реализации.

**Определение числа эпох.** Количество эпох определено по кривой eval/f1 на валидационном подмножестве через мониторинг в W&B. Stage 1: eval/f1 растёт монотонно до 0.12 за 10 эпох. Stage 2: eval/f1 достигает пика 0.29 на шаге ~600 (~4 эпохи), затем снижается.

**Предварительные эксперименты.** До Qwen3-0.6B тестировались Qwen2-0.5B (F1=6.65%) и T-lite 2.7B (F1=16.3%). Низкие результаты объясняются ограниченной ёмкостью скрытого пространства при малом объёме данных.

---

## Структура проекта

```
.
├── openrlhf/                          # Запатченный код apple/ml-clara
│   ├── cli/train_sft.py
│   ├── models/modeling_clara.py
│   ├── models/ring_attn_utils.py
│   ├── models/actor.py
│   ├── trainer/sft_trainer.py
│   └── utils/deepspeed/deepspeed.py
├── patches/                           # .patch файлы с описанием
│   └── README.md
├── rus_data/
│   ├── scripts/
│   │   ├── download_sberquad.py
│   │   └── prepare_sberquad.py
│   └── clara/                         # Подготовленные JSONL
├── scripts/
│   ├── run_training.sh                # Stage 1 → Stage 2 → fix_config
│   ├── setup_env.sh
│   └── setup_data.sh
├── checkpoints/
│   ├── stage1/
│   └── stage2/
├── evaluate_baseline.py
├── evaluate_clara.py
├── compare_results.py
├── prepare_eval_sample.py
├── metrics.py                         # EM, F1, cover_EM
├── run_inference.py
├── evaluate_ragas.py                  # RAGAS (faithfulness, answer_relevancy)
└── requirements.txt
```

---

## Ограничения

- **Оракульный режим.** Документ подаётся на вход напрямую, без ретривера. Для production нужен внешний поиск (BM25, FAISS) или Stage 3 CLaRa.
- **Скорость инференса.** CLaRa медленнее baseline в 1.9× из-за онлайн-компрессии. Для ускорения нужно предвычисление (precompute) memory-токенов документов корпуса.
- **Объём данных.** SberQUAD (~9K контекстов) значительно меньше оригинального корпуса CLaRa (~2M документов Wikipedia).

## Перспективы

- Предвычисление сжатых представлений документов для ускорения инференса
- Stage 3 (дифференцируемый retrieval) с датасетами Mr. TyDi / MIRACL (русский)
- Масштабирование на T-lite 1.0 (7B) на A100
- Расширение обучающего корпуса за счёт русской Wikipedia

---

## Лицензия

Патчи применяются к [apple/ml-clara](https://github.com/apple/ml-clara) — оригинальный код © Apple Inc. Скрипты, данные и evaluation-код в этом репозитории распространяются под лицензией MIT.
