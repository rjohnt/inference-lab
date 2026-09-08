#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install --no-cache-dir numpy matplotlib pandas pytest ninja
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > requirements.lock.txt
source ./env.sh
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
python src/smoke_test.py
