#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "============================================================"
echo "UPTA-LLM - H200 STACK INSTALLER"
echo "============================================================"

echo
echo "Project:"
echo "${PROJECT_DIR}"

echo
echo "Checking Python..."

python3 --version

PYTHON_MAJOR_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [[ "${PYTHON_MAJOR_MINOR}" != "3.12" ]]; then
    echo
    echo "ERROR: Se requiere Python 3.12."
    echo "Detectado: ${PYTHON_MAJOR_MINOR}"
    exit 1
fi

echo
echo "Creating virtual environment..."

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists."
fi

source "${VENV_DIR}/bin/activate"

echo
echo "Python:"
python --version

echo
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing PyTorch..."

python -m pip install \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo
echo "Installing ML stack..."

python -m pip install \
    transformers==5.15.0 \
    peft==0.20.0 \
    trl==1.10.0 \
    bitsandbytes==0.50.1 \
    accelerate \
    datasets \
    safetensors \
    pyyaml

echo
echo "Installing project in editable mode..."

python -m pip install -e "${PROJECT_DIR}"

echo
echo "============================================================"
echo "INSTALLATION COMPLETED"
echo "============================================================"

echo
echo "Python:"
python --version

echo
echo "PyTorch:"
python -c "import torch; print(torch.__version__)"

echo
echo "CUDA:"
python -c "import torch; print(torch.version.cuda)"

echo
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo
echo "GPU:"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"