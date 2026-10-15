#!/bin/bash
# Exact Dreyfus-Wagner over F2 for the matmul prefix targets (tasks 0, 1, 5): validation, the brute-force cross-check,
# then each shape and stride in $CASES ("a,b,c,L" items). Run from the clone's root; results in $OUT.
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cd benchmarks/pouw/steiner
gcc -O3 -march=native -fopenmp -o "$d/dw_f2" dw_f2.c
n=$(nproc)
export STEINER_BIN="$d" OMP_NUM_THREADS=$((n < 64 ? n : 64))
echo "threads $OMP_NUM_THREADS on $(taskset -pc $$)" >"$OUT/threads.txt"
python3 steiner.py --out "$OUT/validate.jsonl" validate
python3 steiner.py --out "$OUT/selftest.jsonl" selftest --cases 300 --seed 11
for c in ${CASES:-2,2,2,2 2,2,2,1 2,2,3,2 2,2,3,1}; do
    IFS=, read -r a b cc L <<<"$c"
    python3 steiner.py --out "$OUT/exact_f2.jsonl" matmul --a "$a" --b "$b" --c "$cc" --L "$L"
done
