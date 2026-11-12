#!/usr/bin/env bash
# Scratch, never lands: part3.py in the three units, each to $OUT/part3-UNIT.json.
set -euo pipefail
cd benchmarks/training_slowdown
for u in 0 1 2; do TS_ROWS=$u PYTHONPATH=. python ../../scratch_rows/part3.py "$OUT/part3-$u.json" > /dev/null; done
