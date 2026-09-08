#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/workspace/cohere-benchmark}
MK_ENV=$(readlink -f "$TASK_ROOT/mk-venv")
export PATH="/usr/local/cuda/bin:$MK_ENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
cd "$TASK_ROOT/megakernel"
mkdir -p "$TASK_ROOT/raw"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv,nounits --loop=1 > "$TASK_ROOT/raw/mk-decode-gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
python src/tests/test_token_callback.py > "$TASK_ROOT/raw/test_token_callback.log" 2>&1
python src/tests/test_kv_bindings.py > "$TASK_ROOT/raw/test_kv_bindings.log" 2>&1
python src/tests/test_decode_layers.py --checkpoint "$TASK_ROOT/checkpoint" --device cuda:0 --frac-vram-utilization 0.90 > "$TASK_ROOT/raw/test_decode_layers.log" 2>&1
touch "$TASK_ROOT/TESTS_COMPLETE"
for context in 8192 1024 32768; do
  for batch in 1 2 4 8; do
    for rep in 0 1 2 3; do
      # Rep 0 primes the compiler/disk caches and is excluded from aggregate results.
      python "$TASK_ROOT/bench_mk.py" --root "$TASK_ROOT" --context "$context" --batch "$batch" --seed 42 \
        --output "$TASK_ROOT/raw/mk-c${context}-b${batch}-r${rep}.json" \
        > "$TASK_ROOT/raw/mk-c${context}-b${batch}-r${rep}.log" 2>&1
    done
  done
done
touch "$TASK_ROOT/MK_COMPLETE"
