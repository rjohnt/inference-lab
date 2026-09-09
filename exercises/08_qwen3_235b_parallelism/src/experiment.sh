#!/usr/bin/env bash
set -euo pipefail
cd "${QWEN235_ROOT:-/workspace/qwen235b}"
PYTHON=/opt/disagg/venv/bin/python
mkdir -p raw
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits -l 1 > raw/gpu-monitor.csv &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
"$PYTHON" hardware.py --output raw/hardware.json
uv pip freeze --python "$PYTHON" > raw/packages.txt
"$PYTHON" download.py > download.log 2>&1
"$PYTHON" prepare_inputs.py > inputs.log 2>&1
"$PYTHON" run.py --mode tp --pilot-only > pilot.log 2>&1
touch PILOT_COMPLETE
# Pilot startup, output lengths and benchmark success are checked by run.py.
# Each full phase owns and cleans up its server, even if a configuration fails.
failed=0
for mode in tp pp ep; do
  if "$PYTHON" run.py --mode "$mode" > "$mode.log" 2>&1; then
    echo 0 > "$mode.exit"
  else
    code=$?
    echo "$code" > "$mode.exit"
    failed=1
  fi
done
touch FULL_PHASES_FINISHED
if [[ "$failed" == 0 ]]; then touch EXPERIMENT_COMPLETE; fi
exit "$failed"
