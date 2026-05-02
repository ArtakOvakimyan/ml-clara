#!/bin/bash
# =============================================================================
# run_training.sh
# =============================================================================
# Запускает Stage 1 → Stage 2 → fix_config последовательно.
# Останавливается при любой ошибке.
#
# Использование:
#   bash run_training.sh
#   bash run_training.sh --stage1_only     # только Stage 1
#   bash run_training.sh --stage2_only     # только Stage 2 (нужен чекпоинт Stage 1)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLARA_DIR="${PROJECT_DIR}"

# ─── Модель ──────────────────────────────────────────────────
MODEL_NAME="tech/T-lite-it-2.1-FP8"

# ─── Данные ──────────────────────────────────────────────────
DATA_DIR="${PROJECT_DIR}/rus_data/clara"
STAGE1_DATA="${DATA_DIR}/stage1_pretrain.jsonl"
STAGE2_DATA="${DATA_DIR}/stage2_instruction.jsonl"

# ─── Чекпоинты ───────────────────────────────────────────────
STAGE1_CHECKPOINT="${PROJECT_DIR}/checkpoints/stage1"
STAGE2_CHECKPOINT="${PROJECT_DIR}/checkpoints/stage2"

# ─── Параметры CLaRa ─────────────────────────────────────────
COMPRESS_RATE=16
DOC_MAX_LENGTH=256

# ─── Параметры обучения ──────────────────────────────────────
BATCH_SIZE_PER_GPU=2      # 1?
TRAIN_BATCH_SIZE=16       # 8?
LEARNING_RATE=1e-4
MAX_LEN=2048              # 512?
ZERO_STAGE=2

# ─── Флаги из аргументов ─────────────────────────────────────
RUN_STAGE1=true
RUN_STAGE2=true

for arg in "$@"; do
    case $arg in
        --stage1_only) RUN_STAGE2=false ;;
        --stage2_only) RUN_STAGE1=false ;;
    esac
done

# ─── Утилиты ─────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
separator() { echo ""; echo "============================================"; echo "  $*"; echo "============================================"; echo ""; }

export PYTHONPATH="${CLARA_DIR}:${PYTHONPATH}"

# ─────────────────────────────────────────────────────────────
# Предварительные проверки
# ─────────────────────────────────────────────────────────────
separator "Проверки"

[ -f "${CLARA_DIR}/openrlhf/cli/train_sft.py" ] || {
    echo "✗ Репозиторий CLaRa не найден: ${CLARA_DIR}"
    exit 1
}
log "✓ CLaRa найден"

if [[ "$CONDA_DEFAULT_ENV" != "clara" ]]; then
    echo "⚠ Активируй окружение: conda activate clara"
    exit 1
fi

if $RUN_STAGE1; then
    [ -f "${STAGE1_DATA}" ] || {
        echo "✗ Данные Stage 1 не найдены: ${STAGE1_DATA}"
        echo "  Запустите: python prepare_sberquad.py"
        exit 1
    }
    log "✓ Данные Stage 1: $(wc -l < "${STAGE1_DATA}") примеров"
fi

if $RUN_STAGE2; then
    [ -f "${STAGE2_DATA}" ] || {
        echo "✗ Данные Stage 2 не найдены: ${STAGE2_DATA}"
        exit 1
    }
    log "✓ Данные Stage 2: $(wc -l < "${STAGE2_DATA}") примеров"

    if ! $RUN_STAGE1; then
        [ -d "${STAGE1_CHECKPOINT}" ] || {
            echo "✗ Чекпоинт Stage 1 не найден: ${STAGE1_CHECKPOINT}"
            echo "  Запустите Stage 1 сначала: bash run_training.sh --stage1_only"
            exit 1
        }
        log "✓ Чекпоинт Stage 1 найден"
    fi
fi

log "  Модель: ${MODEL_NAME}"
log "  batch_size: ${BATCH_SIZE_PER_GPU} micro / ${TRAIN_BATCH_SIZE} effective"
log "  max_len: ${MAX_LEN}, zero_stage: ${ZERO_STAGE}"

# ─────────────────────────────────────────────────────────────
# Stage 1: Salient Compressor Pretraining
# ─────────────────────────────────────────────────────────────
if $RUN_STAGE1; then
    separator "Stage 1: Salient Compressor Pretraining"
    mkdir -p "${STAGE1_CHECKPOINT}"

    STAGE1_START=$(date +%s)

    deepspeed --num_gpus 1 \
        "${CLARA_DIR}/openrlhf/cli/train_sft.py" \
        --pretrain "${MODEL_NAME}" \
        --dataset "${STAGE1_DATA}" \
        --save_path "${STAGE1_CHECKPOINT}" \
        --max_len ${MAX_LEN} \
        --micro_train_batch_size ${BATCH_SIZE_PER_GPU} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --max_epochs 5 \
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
        2>&1 | tee "${STAGE1_CHECKPOINT}/train_stage1.log"

    STAGE1_END=$(date +%s)
    log "✅ Stage 1 завершён за $(( (STAGE1_END - STAGE1_START) / 60 )) мин"
fi

# ─────────────────────────────────────────────────────────────
# Stage 2: Compression Instruction Tuning
# ─────────────────────────────────────────────────────────────
if $RUN_STAGE2; then
    separator "Stage 2: Compression Instruction Tuning"
    mkdir -p "${STAGE2_CHECKPOINT}"

    STAGE2_START=$(date +%s)

    deepspeed --num_gpus 1 \
        "${CLARA_DIR}/openrlhf/cli/train_sft.py" \
        --pretrain "${MODEL_NAME}" \
        --pretrain_checkpoint "${STAGE1_CHECKPOINT}" \
        --dataset "${STAGE2_DATA}" \
        --save_path "${STAGE2_CHECKPOINT}" \
        --max_len ${MAX_LEN} \
        --micro_train_batch_size ${BATCH_SIZE_PER_GPU} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --max_epochs 3 \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler cosine \
        --lr_warmup_ratio 0.05 \
        --l2 0.01 \
        --adam_betas 0.9 0.95 \
        --zero_stage ${ZERO_STAGE} \
        --bf16 \
        --gradient_checkpointing \
        --save_steps 200 \
        --eval_steps 50 \
        --logging_steps 10 \
        --stage stage1_2 \
        --compress_rate ${COMPRESS_RATE} \
        --doc_max_length ${DOC_MAX_LENGTH} \
        --generation_top_k 1 \
        --mse_loss \
        --do_eval_gen \
        2>&1 | tee "${STAGE2_CHECKPOINT}/train_stage2.log"

    STAGE2_END=$(date +%s)
    log "✅ Stage 2 завершён за $(( (STAGE2_END - STAGE2_START) / 60 )) мин"
fi

# ─────────────────────────────────────────────────────────────
# Fix config: подготовка чекпоинта Stage 2 к инференсу
# ─────────────────────────────────────────────────────────────
if $RUN_STAGE2 && [ -d "${STAGE2_CHECKPOINT}" ]; then
    separator "Fix config: подготовка к инференсу"

    cp "${CLARA_DIR}/openrlhf/models/modeling_clara.py" "${STAGE2_CHECKPOINT}/"
    log "✓ modeling_clara.py скопирован"

    python -c "
import json
cfg = json.load(open('${STAGE2_CHECKPOINT}/config.json'))
cfg['compr_base_model_name'] = '${MODEL_NAME}'
json.dump(cfg, open('${STAGE2_CHECKPOINT}/config.json', 'w'), indent=2)
print('✓ compr_base_model_name обновлён:', '${MODEL_NAME}')
"
fi

# ─────────────────────────────────────────────────────────────
# Итог
# ─────────────────────────────────────────────────────────────
separator "Готово"

if $RUN_STAGE1; then
    log "Stage 1 чекпоинт: ${STAGE1_CHECKPOINT}"
fi
if $RUN_STAGE2; then
    log "Stage 2 чекпоинт: ${STAGE2_CHECKPOINT}"
    echo ""
    echo "Для инференса:"
    echo "  python run_inference.py --checkpoint ${STAGE2_CHECKPOINT}"
fi