#!/usr/bin/env bash
set -euo pipefail
# Activate the measured Python environment before invoking this script.
workload=${1:-matmul}
variant=${2:-overlapped}
out=${3:?Usage: profile.sh workload variant new-output-directory}
python_bin=$(command -v python)
nsys_bin=${NSYS_BIN:-$(command -v nsys)}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$out" || -e "$out.nsys-rep" ]]; then
  echo 'Choose a new output path.' >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$out")"
"$nsys_bin" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output="$out" "$python_bin" "$script_dir/profile_workload.py" \
  --variant "$variant" --workload "$workload" --out "$out"
test -s "$out.nsys-rep"
"$nsys_bin" export --type=sqlite --output="$out.sqlite" "$out.nsys-rep"
"$python_bin" -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type=\"table\"")}; assert "CUPTI_ACTIVITY_KIND_KERNEL" in tables and "CUPTI_ACTIVITY_KIND_MEMCPY" in tables, "Capture lacks GPU activities; inspect diagnostics and WSL timestamp configuration"' "$out.sqlite"
