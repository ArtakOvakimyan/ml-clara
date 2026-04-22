# CLaRa + T-lite + SberQUAD

Адаптация фреймворка [Apple CLaRa](https://github.com/apple/ml-clara) для обучения на русскоязычном датасете [SberQUAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) с использованием модели T-lite (Qwen2-based) вместо оригинального Mistral-7B.

---

## Что изменено в ml-clara и зачем

Основные изменения хранятся в `.patch`-файле в папке `patches/`. Подробное описание — в [patches/README.md](patches/README.md).

**Кратко:**

- **`ring_attn_utils.py`** — flash_attn не поддерживается на T4 (Turing/sm_75), поэтому все импорты обёрнуты в `try/except`. Ring attention используется только при multi-GPU обучении — на одной карте не вызывается.

- **`modeling_clara.py`** — четыре изменения: замена `flash_attention_2` на `sdpa` (работаем на одной GPU), пересчёт `compr_mlp_hidden_dim` под размерность T-lite, удаление параметра `enable_thinking=False` (специфичен для Qwen3, T-lite основан на Qwen2), добавление конвертации списков в строки в `_blend_standard_prompt` для совместимости с форматом SberQUAD, добавление T-lite в условия инициализации специальных токенов.

- **`train_sft.py`** — заменён `attn_implementation` на `sdpa`, удалён параметр `min_lr` из вызова планировщика (не поддерживается в transformers ≥ 4.46).

- **`actor.py`** и **`deepspeed.py`** — аналогичные замены flash_attn на sdpa и защитные `try/except` для импортов.

---

## Добавленные скрипты

| Скрипт | Назначение |
|---|---|
| `setup_env.sh` | Создание conda-окружения и установка зависимостей |
| `setup_data.sh` | Загрузка SberQUAD и подготовка данных для CLaRa |
| `train_stage1.sh` | Обучение Stage 1: Salient Compressor Pretraining |
| `train_stage2.sh` | Обучение Stage 2: Compression Instruction Tuning |
| `fix_config.sh` | Подготовка чекпоинта Stage 2 к инференсу |
| `run_training.sh` | **Главный скрипт** — запускает Stage 1 → Stage 2 → fix_config последовательно |

---

## Зависимости

Изменения в `requirements.txt` по сравнению со стандартным CLaRa:

- `torch==2.4.1` устанавливается **отдельно** через PyTorch CDN
- `numpy>=1.26,<2` — numpy 2.x несовместим со скомпилированными модулями (deepspeed, PyTorch 2.1)
- `peft==0.15.2` — зафиксирована точная версия: 0.13 падает на `target_modules='all-linear'`, 0.18 — на отсутствующем `transformers.integrations.tensor_parallel`
- `torchdata` — требуется `StatefulDataLoader` из deepspeed-окружения CLaRa
- `einops` — нужен для заглушек ring_attn_utils
- `bitsandbytes` — для квантизации (int4/int8) при обучении на GPU с ограниченной VRAM

---

## Порядок запуска

### Шаг 1. Настройка окружения

> Выполняется **один раз** при первой установке.

```bash
bash setup_env.sh
conda activate clara
```

`setup_env.sh` создаёт conda-окружение `clara` с Python 3.10, устанавливает PyTorch 2.4.1+cu118 и все зависимости из `requirements.txt`.

---

### Шаг 2. Подготовка данных

> Данные уже загружены в папку `rus_data/`. Этот шаг можно пропустить если данные есть.

```bash
bash setup_data.sh
```

Скрипт скачивает SberQUAD из HuggingFace Hub и конвертирует его в формат CLaRa (JSONL) для Stage 1 и Stage 2. Результат сохраняется в `rus_data/clara/`.

Если нужно ограничить количество обучающих примеров (для отладки или экономии времени):

```bash
python rus_data/scripts/prepare_sberquad.py \
    --output_dir ./rus_data/clara \
    --max_samples 500
```

---

### Шаг 3. Обучение

```bash
conda activate clara
bash run_training.sh
```

`run_training.sh` последовательно запускает три этапа:

1. **Stage 1** — обучает компрессор сжимать документы в memory-токены через QA-пары (5 эпох, ~30 мин на T4 для 500 примеров)
2. **Stage 2** — дообучает компрессор и генератор в instruction-following режиме (3 эпохи, ~20 мин)
3. **fix_config** — копирует `modeling_clara.py` в папку чекпоинта и обновляет `config.json` для корректной загрузки через `AutoModel`

Чекпоинты сохраняются в `checkpoints/stage1/` и `checkpoints/stage2/`.

**Запуск отдельных стадий:**

```bash
# Только Stage 1
bash run_training.sh --stage1_only

# Только Stage 2 (когда Stage 1 уже готов)
bash run_training.sh --stage2_only
```

---

### Шаг 4. Инференс

```bash
python run_inference.py \
    --checkpoint checkpoints/stage2 \
    --stage 2 \
    --question "Ваш вопрос" \
    --document "Текст документа"
```

---

## Параметры по умолчанию

| Параметр | Значение | Комментарий |
|---|---|---|
| Модель | `MilyaShams/T-lite-it-1.0_Q4_0` | 2.7B параметров, совместима с HF, квантизированная верси T-Lite 1.0 |
| `compress_rate` | 16 | 256 токенов документа → 16 memory-токенов |
| `doc_max_length` | 256 | Макс. длина документа в токенах |
| `max_len` | 512 | Макс. длина обучающей последовательности |
| `batch_size` | 1 micro / 8 effective |
| `zero_stage` | 2 | DeepSpeed ZeRO Stage 2 |
| `compr_n_layers` | 4 | Слои компрессора из 28 |
| `compr_mlp_hidden_dim` | 7168 | hidden_size x 2  |

Для изменения параметров отредактируй переменные в начале `run_training.sh`.

---

## Структура проекта

```
.
├── openrlhf/                  # Запатченный код apple/ml-clara
│   ├── cli/train_sft.py
│   ├── models/modeling_clara.py
│   ├── models/ring_attn_utils.py
│   ├── models/actor.py
│   └── utils/deepspeed/deepspeed.py
├── patches/                   # .patch файлы с описанием изменений
│   └── README.md
├── rus_data/                  # Данные SberQUAD
│   ├── scripts/
│   │   ├── download_sberquad.py
│   │   └── prepare_sberquad.py
│   └── clara/                 # Подготовленные JSONL для CLaRa
├── checkpoints/               # Чекпоинты (создаются при обучении)
│   ├── stage1/
│   └── stage2/
├── scripts/
│   ├── setup_env.sh
│   ├── setup_data.sh
│   ├── run_training.sh
│   ├── train_stage1.sh
│   └── train_stage2.sh
└── requirements.txt
```