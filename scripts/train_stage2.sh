#!/bin/bash
# Stage 2: Compression Instruction Tuning
# Оригинал: scripts/train_instruction_tuning.sh из apple/ml-clara

set -e

# ─── Пути ────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLARA_DIR="${PROJECT_DIR}"
DATA_DIR="${PROJECT_DIR}/rus_data/clara"
STAGE1_CHECKPOINT="${1:-${PROJECT_DIR}/checkpoints/stage1}"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/stage2"

# ─── Модель ──────────────────────────────────────────────────
MODEL_NAME="MilyaShams/T-lite-it-1.0_Q4_0"

# ─── Параметры CLaRa ────────────────────────────────────────
COMPRESS_RATE=16
DOC_MAX_LENGTH=256
COMPR_N_LAYERS=4
GENERATION_TOP_K=1

# ─── Гиперпараметры обучения (T4-optimized) ─────────────────
BATCH_SIZE_PER_GPU=2
GRADIENT_ACCUMULATION=8
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_LEN=2048
LORA_R=16
LORA_ALPHA=32

# ─── DeepSpeed ───────────────────────────────────────────────
ZERO_STAGE=2

# ─── Проверки ────────────────────────────────────────────────
echo "============================================"
echo "  Stage 2: Compression Instruction Tuning"
echo "  Model:        ${MODEL_NAME}"
echo "  Checkpoint:   ${STAGE1_CHECKPOINT}"
echo "  Data:         ${DATA_DIR}/stage2_instruction.jsonl"
echo "  Output:       ${OUTPUT_DIR}"
echo "============================================"

if [ ! -f "${DATA_DIR}/stage2_instruction.jsonl" ]; then
    echo "✗ Данные не найдены: ${DATA_DIR}/stage2_instruction.jsonl"
    echo "  Запустите: python prepare_sberquad.py"
    exit 1
fi

if [ ! -d "${STAGE1_CHECKPOINT}" ]; then
    echo "✗ Чекпоинт Stage 1 не найден: ${STAGE1_CHECKPOINT}"
    echo "  Сначала выполните: bash train_stage1_t4.sh"
    exit 1
fi

if [ ! -f "${CLARA_DIR}/openrlhf/cli/train_sft.py" ]; then
    echo "✗ Репозиторий CLaRa не найден: ${CLARA_DIR}"
    exit 1
fi

export PYTHONPATH="${CLARA_DIR}:${PYTHONPATH}"
mkdir -p "${OUTPUT_DIR}"

NUM_SAMPLES=$(wc -l < "${DATA_DIR}/stage2_instruction.jsonl")
echo ""
echo "  Примеров:          ${NUM_SAMPLES}"
echo "  Batch (effective): $((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION))"
echo "  Эпох:              ${NUM_EPOCHS}"
echo "  generation_top_k:  ${GENERATION_TOP_K}"
echo ""

# ─── Запуск ──────────────────────────────────────────────────
deepspeed --num_gpus 1 \
    "${CLARA_DIR}/openrlhf/cli/train_sft.py" \
    --pretrain "${MODEL_NAME}" \
    --dataset "${DATA_DIR}/stage2_instruction.jsonl" \
    --save_path "${OUTPUT_DIR}" \
    --pretrain_checkpoint "${STAGE1_CHECKPOINT}" \
    --max_len ${MAX_LEN} \
    --micro_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --train_batch_size $((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION)) \
    --max_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler cosine \
    --adam_betas 0.9 0.95 \
    --zero_stage ${ZERO_STAGE} \
    --gradient_checkpointing \
    --save_steps 200 \
    --eval_steps 50 \
    --logging_steps 10 \
    --stage stage1_2 \
    --compress_rate ${COMPRESS_RATE} \
    --doc_max_length ${DOC_MAX_LENGTH} \
    --generation_top_k ${GENERATION_TOP_K} \
    --mse_loss \
    --do_eval_gen \
    2>&1 | tee "${OUTPUT_DIR}/train_stage2.log"

echo ""
echo "✅ Stage 2 завершён!"
echo "   Чекпоинт: ${OUTPUT_DIR}"
echo "   Лог:      ${OUTPUT_DIR}/train_stage2.log"
echo ""
echo "Следующий шаг:"
echo "   python run_inference.py --checkpoint ${OUTPUT_DIR}"
