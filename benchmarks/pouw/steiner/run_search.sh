#!/bin/bash
# The Conjecture A searches after run_exact_f2.sh, by phase ($PHASES, space-separated; default all but subspace):
#   f3        exact <2,2,2>/F3 for each L in $F3_L (default 1: 8 terminals, 3^16 states)
#   general   task 4: the hub example, random (i)+(ii) families, 2x2 block grids (exhaustive) and chained ones
#   aligned   task 4: (i)+(ii) families whose column and row spaces are coordinate subspaces
#   grid      2x3 block grids of invertible 2x2 blocks (2^24 states each)
#   exact     more exact F2 shapes ($EXACT, "a,b,c,L" items)
#   subspace  task 3: subspace-restricted DW, for each "a,b,c,L,samples,budget_s,batch_k" item in $SUB (default <2,4,2>/L=2,
#             then <2,3,2> with L=3 and L=1), subspaces of dimension at most $SUB_MAX_K (24)
#   sa        annealing over Steiner points on the whole space: calibration against exact DW, then $SA_SHAPES
#   full2x2   exact <2,b,2> with L=b for each b in $FULL_B (default 2 3; b=3 is 2^36 states, 3 arrays of 64 GiB) by
#             dw_2b2, after its relaxation, symmetry and dw_f2 cross-checks
# $THREADS overrides the thread count (default nproc, at most 64); $FAMILIES, $SIZES and $SEED the families'
# count, sizes and seed.
# Run from the clone's root; results in $OUT.
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cd benchmarks/pouw/steiner
gcc -O3 -march=native -fopenmp -o "$d/dw_f2" dw_f2.c
gcc -O3 -march=native -fopenmp -o "$d/dw_f3" dw_f3.c
n=$(nproc)
export STEINER_BIN="$d" OMP_NUM_THREADS=${THREADS:-$((n < 64 ? n : 64))}
echo "threads $OMP_NUM_THREADS on $(taskset -pc $$)" >>"$OUT/threads.txt"
for phase in ${PHASES:-f3 general grid exact}; do
    echo "phase $phase start $(date -u +%FT%TZ)" >&2
    case $phase in
    f3)
        for L in ${F3_L:-1}; do
            python3 steiner.py --out "$OUT/exact_f3.jsonl" matmul3 --a 2 --b 2 --c 2 --L "$L"
        done ;;
    general)
        python3 steiner.py --out "$OUT/general.jsonl" general --seed "${SEED:-1}" --families "${FAMILIES:-1000}" \
            --grid-chains "${GRID_CHAINS:-2000}" --sizes ${SIZES:-4x4 3x5 2x6 4x5} --max-terminals 8 ;;
    aligned)
        python3 steiner.py --out "$OUT/aligned.jsonl" aligned --seed "${SEED:-1}" --families "${FAMILIES:-2000}" \
            --max-positions 6 --sizes ${SIZES:-4x4 3x5 2x6 3x6} ;;
    grid) python3 steiner.py --out "$OUT/grid.jsonl" grid --families 40 ;;
    exact)
        for c in ${EXACT:-1,2,3,1 1,2,4,1 1,2,5,1 1,3,2,1 1,3,3,3 1,3,3,1 1,2,6,1 1,4,2,2}; do
            IFS=, read -r a b cc L <<<"$c"
            python3 steiner.py --out "$OUT/exact_f2.jsonl" matmul --a "$a" --b "$b" --c "$cc" --L "$L"
        done ;;
    subspace)
        for c in ${SUB:-2,4,2,2,400,4500,18 2,3,2,3,200,900,18 2,3,2,1,100,900,16}; do
            IFS=, read -r a b cc L samples budget bk <<<"$c"
            python3 steiner.py --out "$OUT/subspace_$a$b${cc}_L$L.jsonl" subspace --a "$a" --b "$b" --c "$cc" --L "$L" \
                --samples "$samples" --budget-s "$budget" --batch-k "$bk" --max-k "${SUB_MAX_K:-24}"
        done ;;
    full2x2)
        gcc -O3 -march=native -fopenmp -o "$d/dw_2b2" dw_2b2.c
        python3 steiner.py --out "$OUT/full2x2.jsonl" full2x2 --b ${FULL_B:-2 3} --cross 200 ;;
    sa)
        gcc -O3 -march=native -fopenmp -o "$d/sa" sa.c -lm
        python3 steiner.py --out "$OUT/sa_calibrate.jsonl" sa-calibrate --per-size 10 --iters 200000 --restarts 46
        for c in ${SA_SHAPES:-2,2,2,1 2,2,3,1 2,3,2,1 2,3,2,3 2,4,2,1 2,4,2,2 2,4,2,4 3,2,3,1 3,2,3,2 2,3,3,1 2,3,3,3 \
            3,3,3,1 3,3,3,3 4,2,4,1 4,2,4,2 3,4,3,2 4,4,4,2 4,4,4,4}; do
            IFS=, read -r a b cc L <<<"$c"
            python3 steiner.py --out "$OUT/sa.jsonl" sa --a "$a" --b "$b" --c "$cc" --L "$L" \
                --iters "${SA_ITERS:-1000000}" --restarts "${SA_RESTARTS:-920}"
        done ;;
    esac
    echo "phase $phase end $(date -u +%FT%TZ)" >&2
done
