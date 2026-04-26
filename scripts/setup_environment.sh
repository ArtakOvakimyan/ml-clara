#!/bin/bash

set -e

ENV_NAME="clara"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

conda clean --all -y

# Создайте окружение с явным указанием канала
conda create -n myenv python=3.10 -c conda-forge -y
conda activate myenv

# Зафиксируйте приоритет каналов
conda config --env --set channel_priority strict

# Для CUDA 12.1 (совместимо с 12.2)
conda install -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -y

# Проверка
python -c "import torch; print(f'✅ {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"

# Зависимости для сборки
pip install --upgrade pip setuptools wheel ninja psutil packaging

# flash-attn
pip install flash-attn --no-build-isolation --no-cache-dir

# Остальное
pip install -r "${PROJECT_DIR}/requirements.txt"
echo "Зависимости установлены"

# 3. Настройка PYTHONPATH
export PYTHONPATH="$CLARA_DIR:$PYTHONPATH"