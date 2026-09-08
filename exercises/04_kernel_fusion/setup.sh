#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip check
source ./env.sh
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
python src/smoke_test.py
