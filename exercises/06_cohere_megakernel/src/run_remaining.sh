#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/workspace/cohere-benchmark}
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
trap 'status=$?; printf "%s\n" "$status" > "$TASK_ROOT/remaining.exit"' EXIT
while [[ ! -f "$TASK_ROOT/mk.exit" ]]; do sleep 3; done
[[ $(cat "$TASK_ROOT/mk.exit") == 0 ]]
python3 "$TASK_ROOT/run_vllm.py" --root "$TASK_ROOT" --phase "${COHERE_BASELINE_PHASE:-both}" > "$TASK_ROOT/run-vllm.log" 2>&1
python3 "$TASK_ROOT/run_mk_api.py" --root "$TASK_ROOT" > "$TASK_ROOT/run-mk-api.log" 2>&1
touch "$TASK_ROOT/ALL_COMPLETE"
