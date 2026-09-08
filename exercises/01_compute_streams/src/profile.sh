#!/usr/bin/env bash
set -euo pipefail
strategy=${1:?Usage: profile.sh strategy rows new-output-path}
rows=${2:?Specify rows per request}
out=${3:?Specify a new output path}
python_bin=$(command -v python)
nsys_bin=${NSYS_BIN:-$(command -v nsys)}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$out" || -e "$out.nsys-rep" ]]; then echo 'Choose a new output path.' >&2; exit 1; fi
mkdir -p -- "$(dirname -- "$out")"
"$nsys_bin" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output="$out" \
  "$python_bin" "$script_dir/benchmark.py" --profile "$strategy" --rows "$rows" --out "$out"
test -s "$out.nsys-rep"
"$nsys_bin" export --type=sqlite --output="$out.sqlite" "$out.nsys-rep"
"$python_bin" -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); assert c.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]>0' "$out.sqlite"
