#!/usr/bin/env bash
# One kernel-level bench on one dc node, built outside the lease and timed under one gpu-lease:
#   kbench   kb.so (and kb_pg.so on sm_100a): kbench.py (the honest GEMMs gated, then timed against the plain int8 GEMMs),
#            then verify.py after the lease (CPU, CA's numpy twin)
#   h1       h1noise.so: h1bench.py (H1's per-unit noise gated, then timed against the plain int8 GEMMs)
#
#   research run --on vy-dc-h200 --project verity --campaign dither-loop --source . --cwd source --timeout 40m \
#     -- bash benchmarks/hash/dither/job.sh sm_90a [kbench|h1]
#
# Env: BENCH_ARGS (extra flags for the bench, e.g. "--ms 1,32 --linears o"), LEASE (gpu-lease flags, default
# "--max-min 14"), VERITY_BENCH_VENV (default /workspace/cp/venv312), NVCC (default CUDA 13.0's).
# Exit: the bench's (4 a gate failed), 5 verify.py's twin disagreed, 2 a build failed.
set -uo pipefail
ARCH=${1:?usage: job.sh sm_90a|sm_100a [kbench|h1]}
BENCH=${2:-kbench}
HERE=$(cd "$(dirname "$0")" && pwd)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
WORK=$(mktemp -d)
mkdir -p "$OUT/env"
"$NVCC" --version > "$OUT/env/nvcc.txt" 2>&1
LIB64=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)
NV=("$NVCC" -O3 -std=c++17 -shared -Xcompiler -fPIC --expt-relaxed-constexpr -gencode "arch=compute_${ARCH#sm_},code=$ARCH")
build() {  # build NAME SRC [flags...]
  local name=$1 src=$2
  shift 2
  "${NV[@]}" "$HERE/$src" "$@" -o "$WORK/$name.so" 2> "$OUT/env/build-$name.log" || { cat "$OUT/env/build-$name.log"; exit 2; }
  "$(dirname "$NVCC")/cuobjdump" --dump-resource-usage "$WORK/$name.so" > "$OUT/env/$name.res.txt" 2>&1
}
case $BENCH in
  kbench)
    build kb kb.cu -lcublasLt -Xlinker "-rpath=$LIB64"
    PG=""
    if [ "$ARCH" = sm_100a ]; then build kb_pg kb_pg.cu; PG="--pg-lib $WORK/kb_pg.so"; fi
    CMD="$PY $HERE/kbench.py --lib $WORK/kb.so $PG --out $OUT ${BENCH_ARGS:-}" ;;
  h1)
    build h1noise h1noise.cu
    "$(dirname "$NVCC")/cuobjdump" -sass "$WORK/h1noise.so" | gzip > "$OUT/env/h1noise.sass.gz"
    CMD="$PY $HERE/h1bench.py --lib $WORK/h1noise.so --out $OUT ${BENCH_ARGS:-}" ;;
  *) echo "job.sh: no bench $BENCH" >&2; exit 2 ;;
esac
export PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH} CMD
bench='
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  $CMD 2>&1 | tee "$OUT/bench.out"
  rc=${PIPESTATUS[0]}
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash -c "$bench"
else
  read -r -a lease <<< "${LEASE:---max-min 14}"
  GPU_LEASE_WHO=kernels-dither-hb gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "lease rc $rc"
vrc=0
if [ "$BENCH" = kbench ]; then
  "$PY" "$HERE/verify.py" "$OUT" 2>&1 | tail -3
  vrc=${PIPESTATUS[0]}
fi
exit $(( rc ? rc : vrc ))
