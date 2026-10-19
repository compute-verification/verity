#!/usr/bin/env bash
# rows_floor.py on one GPU (CUDA_VISIBLE_DEVICES), its SM clock and power sampled beside it. Measurement only.
set -uo pipefail
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "${CUDA_VISIBLE_DEVICES:-0}" \
  > "$OUT/smi.csv" 2>&1 &
smi=$!
"$VENV/bin/python" benchmarks/served_commit/rows_floor.py 2>&1 | tee "$OUT/rows_floor.log"
echo "rows_floor rc ${PIPESTATUS[0]}"
kill $smi
