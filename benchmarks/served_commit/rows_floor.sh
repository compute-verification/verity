#!/usr/bin/env bash
# rows_floor.py on one GPU (CUDA_VISIBLE_DEVICES), its SM clock and power sampled beside it; exit 2 where no GPU is visible.
set -uo pipefail
nvidia-smi -L 2>/dev/null | grep -q '^GPU ' || { echo "rows_floor: refused: needs a CUDA GPU, and none is visible" >&2; exit 2; }
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "${CUDA_VISIBLE_DEVICES:-0}" \
  > "$OUT/smi.csv" 2>&1 &
smi=$!
"$VENV/bin/python" benchmarks/served_commit/rows_floor.py 2>&1 | tee "$OUT/rows_floor.log"
rc=${PIPESTATUS[0]}
echo "rows_floor rc $rc"
kill $smi
exit "$rc"
