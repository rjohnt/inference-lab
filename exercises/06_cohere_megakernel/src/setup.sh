#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/workspace/cohere-benchmark}
export TASK_ROOT
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p "$TASK_ROOT/raw"
export HF_HOME="$TASK_ROOT/hf-cache"
export UV_CACHE_DIR=/opt/cohere-envs/uv-cache
mkdir -p /opt/cohere-envs
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libssl-dev ninja-build cmake python3-dev
if [[ ! -d "$TASK_ROOT/megakernel/.git" ]]; then
  git clone --recurse-submodules https://github.com/cohere-ai/cohere-megakernel.git "$TASK_ROOT/megakernel"
fi
cd "$TASK_ROOT/megakernel"
git checkout 67d0b9ca22ea3652796b715d1d1863459e0e2c3c
git submodule update --init --recursive
[[ -x /opt/cohere-envs/mk-venv/bin/python ]] || uv venv --system-site-packages /opt/cohere-envs/mk-venv
[[ -L "$TASK_ROOT/mk-venv" ]] || ln -s /opt/cohere-envs/mk-venv "$TASK_ROOT/mk-venv"
if [[ -f "$TASK_ROOT/configs/mk-requirements.txt" ]]; then
  uv pip install --python /opt/cohere-envs/mk-venv/bin/python -r "$TASK_ROOT/configs/mk-requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
else
  uv pip install --python /opt/cohere-envs/mk-venv/bin/python -r requirements.txt huggingface-hub 'torch==2.11.0'
  uv pip install --python /opt/cohere-envs/mk-venv/bin/python flash-attn-3 --index-url https://download.pytorch.org/whl/cu130
fi
/opt/cohere-envs/mk-venv/bin/python - <<'PY'
import torch, sys
print(sys.version, torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_properties(0))
assert torch.cuda.get_device_capability() == (9, 0)
assert torch.cuda.get_device_properties(0).multi_processor_count == 132
PY
cmake -S . -B build -G Ninja -DPython_EXECUTABLE=/opt/cohere-envs/mk-venv/bin/python -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 12
"$TASK_ROOT/mk-venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download('CohereLabs/North-Mini-Code-1.0', revision='d11e61a842617a22dc328552fa5bb86231ee4f37', local_dir=os.environ.get('TASK_ROOT','/workspace/cohere-benchmark')+'/checkpoint', max_workers=8)
PY
[[ -x /opt/cohere-envs/vllm-venv/bin/python ]] || uv venv /opt/cohere-envs/vllm-venv
[[ -L "$TASK_ROOT/vllm-venv" ]] || ln -s /opt/cohere-envs/vllm-venv "$TASK_ROOT/vllm-venv"
if [[ -f "$TASK_ROOT/configs/vllm-requirements.txt" ]]; then
  uv pip install --python /opt/cohere-envs/vllm-venv/bin/python -r "$TASK_ROOT/configs/vllm-requirements.txt"
else
  uv pip install --python /opt/cohere-envs/vllm-venv/bin/python 'vllm==0.24.0'
fi
uv pip freeze --python "$TASK_ROOT/mk-venv/bin/python" > "$TASK_ROOT/raw/mk-packages.txt"
uv pip freeze --python "$TASK_ROOT/vllm-venv/bin/python" > "$TASK_ROOT/raw/vllm-packages.txt"
touch "$TASK_ROOT/SETUP_COMPLETE"
