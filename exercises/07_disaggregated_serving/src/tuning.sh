#!/usr/bin/env bash
set -euo pipefail
cd "${DISAGG_ROOT:-/workspace/disagg}"
test -f EXPERIMENT_COMPLETE
python3 run.py --mode pd --diagnose-only > diagnostic.log 2>&1
python3 run.py --mode pd --variant kv128 > tuning-kv128.log 2>&1
python3 run.py --mode replicas --variant kv128replicas > tuning-kv128replicas.log 2>&1
python3 run.py --mode pd --variant prefill16k > tuning-prefill16k.log 2>&1
python3 run.py --mode replicas --variant batch16k > tuning-batch16k.log 2>&1
python3 run.py --mode replicas --variant batch1k > tuning-batch1k.log 2>&1
python3 run.py --mode replicas --variant balanced > tuning-balanced.log 2>&1
touch TUNING_COMPLETE
