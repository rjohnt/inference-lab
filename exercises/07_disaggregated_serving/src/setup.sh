#!/usr/bin/env bash
set -euo pipefail
DISAGG_ROOT=${DISAGG_ROOT:-/workspace/disagg}
export DISAGG_ROOT
export PATH="/usr/local/cuda/bin:$PATH"
export UV_CACHE_DIR=/opt/disagg/uv-cache
mkdir -p "$DISAGG_ROOT/raw" /opt/disagg
apt-get update -qq
apt-get install -y -qq libnuma1 libibverbs1 ninja-build
uv venv /opt/disagg/venv
uv pip install --python /opt/disagg/venv/bin/python 'vllm==0.24.0' 'nixl==1.2.0' huggingface-hub
/opt/disagg/venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218',
    local_dir=os.environ['DISAGG_ROOT']+'/model',
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=8)
PY
uv pip freeze --python /opt/disagg/venv/bin/python > "$DISAGG_ROOT/raw/packages.txt"
/opt/disagg/venv/bin/python "$DISAGG_ROOT/hardware.py" --output "$DISAGG_ROOT/raw/hardware.json"
touch "$DISAGG_ROOT/SETUP_COMPLETE"
