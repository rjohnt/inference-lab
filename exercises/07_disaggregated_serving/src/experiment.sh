#!/usr/bin/env bash
set -euo pipefail
cd "${DISAGG_ROOT:-/workspace/disagg}"
python3 run.py --mode replicas > replicas.log 2>&1
python3 run.py --mode pd > pd.log 2>&1
touch EXPERIMENT_COMPLETE
