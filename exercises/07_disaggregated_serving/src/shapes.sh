#!/usr/bin/env bash
set -euo pipefail
cd /workspace/disagg
while [[ ! -f TUNING_COMPLETE ]]; do
  if [[ -f tuning.exit ]] && [[ $(cat tuning.exit) != 0 ]]; then exit 1; fi
  sleep 5
done
mkdir -p source-shapes
cp benchmark.py router.py run.py worker.py shapes.sh source-shapes/
python3 run.py --mode pd --variant shapes > shapes-pd.log 2>&1
python3 run.py --mode replicas --variant shapes > shapes-replicas.log 2>&1
touch SHAPES_COMPLETE
