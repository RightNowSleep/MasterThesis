#!/bin/bash

set -e

ENV_NAME="${1:-LLMs}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Conda Environment Export Script"
echo "  Target: ${ENV_NAME}"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo ""
echo "[1/3] Exporting conda-only dependencies -> ${SCRIPT_DIR}/environment_conda.yml ..."
cat > "${SCRIPT_DIR}/environment_conda.yml" << 'EOF'
name: LLMs
channels:
  - defaults
dependencies:
EOF

conda list -f --json | python3 -c "
import json, sys, subprocess
pkgs = json.load(sys.stdin)
conda_only = ['python=3.10', 'mpi4py', 'gxx_linux-64=11.2.0', 'flask', 'flask-cors', 'openmpi']
for p in conda_only:
    print(f'  - {p}')
" >> "${SCRIPT_DIR}/environment_conda.yml"

echo "[2/3] Exporting pip packages -> ${SCRIPT_DIR}/environment_pip.txt ..."
pip freeze > "${SCRIPT_DIR}/environment_pip.txt"

echo "[3/3] Exporting full environment -> ${SCRIPT_DIR}/environment.yml ..."
conda env export --no-builds > "${SCRIPT_DIR}/environment.yml"

echo ""
echo "=========================================="
echo "  Export complete!"
echo "  ${SCRIPT_DIR}/environment_conda.yml"
echo "  ${SCRIPT_DIR}/environment_pip.txt"
echo "  ${SCRIPT_DIR}/environment.yml"
echo "=========================================="
