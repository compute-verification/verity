#!/bin/bash
# usage: run_tests4.sh ROOT LIST OUT
ROOT=$1; LIST=$2; OUT=$3
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/adapters/vllm:$ROOT/benchmarks/harness:$ROOT/tools/config"
while read -r t; do
  [ -f "$t" ] || { echo "MISSING $t"; continue; }
  timeout 900 /agent/repos/verity/.venv/bin/python -m pytest -q -p no:cacheprovider -m "not slow" -rfE "$t" 2>&1 | grep -E "^(FAILED|ERROR) |^[0-9]+ (passed|failed)|^=+ .*(passed|failed|error).* =+$|no tests ran|Interrupted" | sed "s#^#$t :: #"
done < "$LIST" > "$OUT"
echo DONE >> "$OUT"
