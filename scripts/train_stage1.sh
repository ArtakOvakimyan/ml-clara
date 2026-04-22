#!/bin/bash
# Stage 1: Salient Compressor Pretraining (SCP)

set -e

# ─── Пути ────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLARA_DIR="${PROJECT_DIR}"
DATA_DIR="${PROJECT_DIR}/rus_data/clara"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/stage1"

# ─── Модель ──────────────────────────────────────────────────
MODEL_NAME="MilyaShams/T-lite-it-1.0_Q4_0"

# ─── Параметры обучения ─────────────────────────────────────
BATCH_SIZE_PER_GPU=2
TRAIN_BATCH_SIZE=16
LEARNING_RATE=1e-4
NUM_EPOCHS=5
MAX_LEN=2048
COMPRESS_RATE=16
DOC_MAX_LENGTH=256

# ─── DeepSpeed ───────────────────────────────────────────────
ZERO_STAGE=2

# ─── Проверки ────────────────────────────────────────────────
echo "============================================"
echo "  Stage 1: Salient Compressor Pretraining"
echo "  Model:  ${MODEL_NAME}"
echo "  Data:   ${DATA_DIR}/stage1_pretrain.jsonl"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

if [ ! -f "${DATA_DIR}/stage1_pretrain.jsonl" ]; then
    echo "✗ Данные не найдены: ${DATA_DIR}/stage1_pretrain.jsonl"
    exit 1
fi

if [ ! -f "${CLARA_DIR}/openrlhf/cli/train_sft.py" ]; then
    echo "✗ Репозиторий CLaRa не найден: ${CLARA_DIR}"
    exit 1
fi

export PYTHONPATH="${CLARA_DIR}:${PYTHONPATH}"
mkdir -p "${OUTPUT_DIR}"

NUM_SAMPLES=$(wc -l < "${DATA_DIR}/stage1_pretrain.jsonl")
echo ""
echo "  Примеров:          ${NUM_SAMPLES}"
echo "  Batch (effective): ${TRAIN_BATCH_SIZE}"
echo "  Micro batch:       ${BATCH_SIZE_PER_GPU}"
echo "  Эпох:              ${NUM_EPOCHS}"
echo "  Compress rate:     ${COMPRESS_RATE}"
echo ""

# ─── Запуск ──────────────────────────────────────────────────
deepspeed --num_gpus 1 \
    "${CLARA_DIR}/openrlhf/cli/train_sft.py" \
    --pretrain "${MODEL_NAME}" \
    --dataset "${DATA_DIR}/stage1_pretrain.jsonl" \
    --save_path "${OUTPUT_DIR}" \
    --max_len ${MAX_LEN} \
    --micro_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --max_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler cosine \
    --lr_warmup_ratio 0.05 \
    --l2 0.01 \
    --adam_betas 0.9 0.95 \
    --zero_stage ${ZERO_STAGE} \
    --bf16 \
    --gradient_checkpointing \
    --save_steps 500 \
    --eval_steps 100 \
    --logging_steps 10 \
    --stage stage1 \
    --compress_rate ${COMPRESS_RATE} \
    --doc_max_length ${DOC_MAX_LENGTH} \
    --mse_loss \
    --qa_loss \
    2>&1 | tee "${OUTPUT_DIR}/train_stage1.log"

echo ""
echo "✅ Stage 1 завершён!"
echo "   Чекпоинт: ${OUTPUT_DIR}"
echo "   Лог:      ${OUTPUT_DIR}/train_stage1.log"
echo ""
echo "Следующий шаг: bash train_stage2_t4.sh"