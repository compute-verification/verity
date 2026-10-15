#!/bin/bash
# The Conjecture A searches after run_exact_f2.sh, by phase ($PHASES, space-separated; default all but subspace):
#   f3        exact <2,2,2>/F3 with L=1 (8 terminals, 3^16 states)
#   general   task 4: the hub example, random (i)+(ii) families, 2x2 block grids (exhaustive) and chained ones
#   grid      2x3 block grids of invertible 2x2 blocks (2^24 states each)
#   exact     more exact F2 shapes ($EXACT, "a,b,c,L" items)
#   subspace  task 3: subspace-restricted DW for <2,4,2>/L=2, then <2,3,2> with L=3 and L=1 ($SUB_BUDGET_S each)
# Run from the clone's root; results in $OUT.
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cd benchmarks/pouw/steiner
gcc -O3 -march=native -fopenmp -o "$d/dw_f2" dw_f2.c
gcc -O3 -march=native -fopenmp -o "$d/dw_f3" dw_f3.c
n=$(nproc)
export STEINER_BIN="$d" OMP_NUM_THREADS=$((n < 64 ? n : 64))
echo "threads $OMP_NUM_THREADS on $(taskset -pc $$)" >>"$OUT/threads.txt"
for phase in ${PHASES:-f3 general grid exact}; do
    echo "phase $phase start $(date -u +%FT%TZ)" >&2
    case $phase in
    f3) python3 steiner.py --out "$OUT/exact_f3.jsonl" matmul3 --a 2 --b 2 --c 2 --L 1 ;;
    general)
        python3 steiner.py --out "$OUT/general.jsonl" general --families 1000 --grid-chains 2000 \
            --sizes 4x4 3x5 2x6 4x5 --max-terminals 8 ;;
    grid) python3 steiner.py --out "$OUT/grid.jsonl" grid --families 40 ;;
    exact)
        for c in ${EXACT:-1,2,3,1 1,2,4,1 1,2,5,1 1,3,2,1 1,3,3,3 1,3,3,1 1,2,6,1 1,4,2,2}; do
            IFS=, read -r a b cc L <<<"$c"
            python3 steiner.py --out "$OUT/exact_f2.jsonl" matmul --a "$a" --b "$b" --c "$cc" --L "$L"
        done ;;
    subspace)
        python3 steiner.py --out "$OUT/subspace_242_L2.jsonl" subspace --a 2 --b 4 --c 2 --L 2 --samples 400 \
            --budget-s "${SUB_BUDGET_S:-4500}"
        python3 steiner.py --out "$OUT/subspace_232_L3.jsonl" subspace --a 2 --b 3 --c 2 --L 3 --samples 200 \
            --budget-s "${SUB232_BUDGET_S:-900}"
        python3 steiner.py --out "$OUT/subspace_232_L1.jsonl" subspace --a 2 --b 3 --c 2 --L 1 --samples 100 \
            --budget-s "${SUB232_BUDGET_S:-900}" --batch-k 16 ;;
    esac
    echo "phase $phase end $(date -u +%FT%TZ)" >&2
done
