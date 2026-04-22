#!/bin/bash

set -e

ENV_NAME="clara"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLARA_DIR="$PROJECT_DIR"

[ -f "${PROJECT_DIR}/openrlhf/cli/train_sft.py" ] || {
    echo "Код CLaRa не найден"
    exit 1
}

# 1. Conda-окружение
echo "Создание conda-окружения..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
echo "Python: $(python --version)"

# 2. Зависимости
# Сначала PyTorch с правильным индексом
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Потом остальное
pip install -r requirements.txt
echo "Зависимости установлены"

# 3. Настройка PYTHONPATH
export PYTHONPATH="$CLARA_DIR:$PYTHONPATH"