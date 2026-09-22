#!/usr/bin/env bash
# Recorded measurement driver for the c-vole lane (vy-cpu).  Usage:  run_all.sh STAGE [ARGS]
#   STAGE = small     : units (tu-k16, all in-domain instances as one relation of independent units), VU B=1, VU B=64
#           vu B      : one relation with B chained VUs (vu-k1536[0:B]); B=4096 is the decision batch
#           threads B T : VU batch B with T sVOLE threads (dietmc multithreaded; bytes not relayed)
#           field F B : VU batch B over field F (f61p | f127p | f128p | f384p)
#           variants [B] : lpn large, threads 4/8, F128p, F384p(B=8) on B VUs;  profile [B] : callgrind of the prover
# The swanky checkout is outside the repo: it is pinned here by commit and checked before anything runs.
set -euo pipefail
STAGE=${1:?stage}
SWANKY=/workspace/swanky
SWANKY_COMMIT=1b05bddab3e02c2cc4917d08c80a629286a4a4fc
DIETMC=${DIETMC:-/workspace/target-swanky/release/dietmc}   # profile stage: DIETMC=/workspace/target-swanky-prof/release/dietmc (no AVX-512, +debuginfo, for valgrind)
FIX=/workspace/research/runs/r20260922-024439-915c/v1   # bench-instances/v1 arrays, digest-verified by that run
OUT=${RESEARCH_RUN_DIR:-/workspace/vole-out}
mkdir -p "$OUT"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
UV=$(command -v uv)
PY="$UV run --no-project --python 3.12 python"

got=$(git -C $SWANKY rev-parse HEAD)
if [ "$got" != "$SWANKY_COMMIT" ]; then echo "swanky at $got, expected $SWANKY_COMMIT" >&2; exit 3; fi
test -x $DIETMC || { echo "build dietmc first: cargo build --release -j12 -p diet-mac-and-cheese --bin dietmc" >&2; exit 3; }
echo "swanky $SWANKY_COMMIT; dietmc $(md5sum $DIETMC | cut -c1-12); $(grep -m1 'model name' /proc/cpuinfo); nproc=$(nproc)"

bench() {  # dir label [extra bench args]
  local dir=$1 label=$2; shift 2
  $PY backends/vole/bench.py "$dir" --dietmc $DIETMC --label "$label" --swanky-commit $SWANKY_COMMIT --out "$OUT/bench_$label.json" "$@"
}

case "$STAGE" in
  small)
    $PY backends/vole/export_ir.py census > "$OUT/census.json"
    $PY backends/vole/export_ir.py negatives --root $FIX > "$OUT/negatives.txt"
    tail -1 "$OUT/negatives.txt"
    $PY backends/vole/export_ir.py tu "$OUT/units_tu4096" --root $FIX --n 4096
    bench "$OUT/units_tu4096" units-tu4096 --plaintext-too
    $PY backends/vole/export_ir.py unit "$OUT/unit1" --n 1
    bench "$OUT/unit1" unit-1 --plaintext-too --repeat 3
    $PY backends/vole/export_ir.py vu "$OUT/vu1" --root $FIX --B 1
    bench "$OUT/vu1" vu-B1 --plaintext-too --repeat 3
    $PY backends/vole/export_ir.py vu "$OUT/vu64" --root $FIX --B 64 --procs 15
    bench "$OUT/vu64" vu-B64 --plaintext-too
    ;;
  vu)
    B=${2:?B}
    $PY backends/vole/export_ir.py vu "$OUT/vu$B" --root $FIX --B "$B" --procs 15
    ls -la "$OUT/vu$B"
    bench "$OUT/vu$B" "vu-B$B" --plaintext-too
    ;;
  threads)
    B=${2:?B}; T=${3:?threads}
    $PY backends/vole/export_ir.py vu "$OUT/vu$B" --root $FIX --B "$B" --procs 15
    bench "$OUT/vu$B" "vu-B$B-t$T" --threads "$T"
    ;;
  field)
    F=${2:?field}; B=${3:?B}
    $PY backends/vole/export_ir.py vu "$OUT/vu${B}_$F" --root $FIX --B "$B" --procs 15 --field "$F"
    bench "$OUT/vu${B}_$F" "vu-B$B-$F" --plaintext-too
    ;;
  variants)
    # B=64 VUs: sVOLE threads 4 and 8, lpn=large, F128p (2^-97 class), F384p on B=8 (2^-128 class), then the
    # gdb sampling profile of the single-threaded F61p prover (perf is unavailable in the container)
    B=${2:-64}
    $PY backends/vole/export_ir.py vu "$OUT/vu$B" --root $FIX --B "$B" --procs 15
    bench "$OUT/vu$B" "vu-B$B-lpnlarge" --lpn large
    for T in 4 8; do bench "$OUT/vu$B" "vu-B$B-t$T" --threads "$T"; done
    $PY backends/vole/export_ir.py vu "$OUT/vu${B}_f128p" --root $FIX --B "$B" --procs 15 --field f128p
    bench "$OUT/vu${B}_f128p" "vu-B$B-f128p" --plaintext-too
    $PY backends/vole/export_ir.py vu "$OUT/vu8_f384p" --root $FIX --B 8 --procs 8 --field f384p
    bench "$OUT/vu8_f384p" "vu-B8-f384p" --plaintext-too
    ;;
  profile)
    # callgrind instruction profile of the single-threaded F61p prover on B VUs (perf/ptrace unavailable)
    B=${2:-16}
    $PY backends/vole/export_ir.py vu "$OUT/vu$B" --root $FIX --B "$B" --procs 8
    DIETMC=$DIETMC OUT="$OUT/profile_vu$B" bash backends/vole/profile.sh "$OUT/vu$B" F61p 1 | tee "$OUT/profile_vu$B.txt"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2;;
esac
# keep the run directory small: the witness files are reproducible from the exporter
find "$OUT" -name 'witness.txt' -size +50M -delete || true
