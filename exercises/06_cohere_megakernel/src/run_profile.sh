#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/workspace/cohere-benchmark}
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
trap 'status=$?; printf "%s\n" "$status" > "$TASK_ROOT/profile.exit"' EXIT
while [[ ! -f "$TASK_ROOT/remaining.exit" ]]; do sleep 3; done
[[ $(cat "$TASK_ROOT/remaining.exit") == 0 ]]
timeout 600 /opt/cohere-envs/mk-venv/bin/python "$TASK_ROOT/profile_mk.py" --root "$TASK_ROOT" > "$TASK_ROOT/raw/profile.log" 2>&1
touch "$TASK_ROOT/PROFILE_COMPLETE"
