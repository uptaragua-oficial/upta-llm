#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "UPTA-LLM - H200 SETUP"
echo "============================================================"

echo
echo "Project directory:"
echo "${PROJECT_DIR}"

cd "${PROJECT_DIR}"

echo
echo "============================================================"
echo "1. SYSTEM"
echo "============================================================"

if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a || true
fi

echo
python3 --version

echo
echo "============================================================"
echo "2. NVIDIA GPU"
echo "============================================================"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi no está disponible."
    exit 1
fi

nvidia-smi

echo
echo "============================================================"
echo "3. PYTHON VIRTUAL ENVIRONMENT"
echo "============================================================"

if [[ ! -d ".venv" ]]; then
    echo "Creating .venv..."
    python3 -m venv .venv
else
    echo ".venv already exists."
fi

source .venv/bin/activate

echo
echo "Python:"
python --version

echo
echo "============================================================"
echo "4. INSTALL STACK"
echo "============================================================"

bash scripts/install_stack.sh

echo
echo "============================================================"
echo "5. ENVIRONMENT CHECK"
echo "============================================================"

python scripts/check_environment.py

echo
echo "============================================================"
echo "6. GPU CHECK"
echo "============================================================"

python scripts/check_gpu.py

echo
echo "============================================================"
echo "H200 SETUP COMPLETED"
echo "============================================================"

echo
echo "Virtual environment:"
echo "${PROJECT_DIR}/.venv"

echo
echo "To activate it manually:"
echo "source ${PROJECT_DIR}/.venv/bin/activate"