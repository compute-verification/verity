#!/usr/bin/env bash
# exactlin's served standard number on node 2 (RTX PRO 6000), vy-dc-h200 or vy-dc-b200: dither's sbench.py (the arms stock,
# stock-cutlass on sm_90, exact and exact-fused, which need none of Dither's libraries) on one leased GPU for at most
# SERVED_MAX_MIN (25) minutes, with the clocks read before and after.  Extra arguments go to sbench.py; stock first, so its
# overheads are against stock.  DITHER_NSYS=1 runs it under nsys with the prof op's per-step kernels (steps.py).
#   research run --on vy-dc-h200 --project verity --campaign exactlin --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-exactlin -- bash benchmarks/hash/exactlin/served.sh \
#     --arms stock-cutlass,stock,exact,exact-fused --gate --ops decode --reps 3 --t 64 --gpu-mem 0.2
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT/env"
TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/exactlin-served.XXXXXX"); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case $CC in
  12.0) export FLASHINFER_CUDA_ARCH_LIST=12.0f ;;
  9.0) export FLASHINFER_CUDA_ARCH_LIST=9.0a VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  10.0) export FLASHINFER_CUDA_ARCH_LIST=10.0a VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  *) echo "served.sh: compute capability $CC" >&2; exit 2 ;;
esac
echo "compute_cap $CC python $PY" > "$OUT/env/node.txt"
export VLLM_CACHE_ROOT=$S/vllm-cache TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache TRITON_CACHE_DIR=$S/triton
export VLLM_DISABLE_COMPILE_CACHE=1 HF_HOME=${HF_HOME:-/workspace/hf}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/benchmarks/hash/dither
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
fi
cd integrations/vllm
MAX_MIN=${SERVED_MAX_MIN:-25}
echo "served: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --preemptible --max-min "$MAX_MIN" -- bash -c '
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.before.csv"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -l 2 > "$0/smi.csv" 2>/dev/null &
  smi=$!
  timeout $((60 * '"$MAX_MIN"' - 60)) "$@"
  r=$?
  kill "$smi" 2>/dev/null
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.after.csv"
  exit $r' "$OUT" "${NSYS[@]}" "$PY" "$ROOT/benchmarks/hash/dither/sbench.py" --out "$OUT" --scratch "$S" "$@"
rc=$?
echo "served: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" export --type sqlite --force-overwrite true -o "$S/gate.sqlite" "$S/gate.nsys-rep" > "$OUT/nsys-export.log" 2>&1
  "$PY" "$ROOT/benchmarks/hash/exactlin/steps.py" "$S/gate.sqlite" > "$OUT/steps.json" || echo "served: steps.py rc $?"
  [ "$(stat -c %s "$S/gate.nsys-rep")" -lt 209715200 ] && cp "$S/gate.nsys-rep" "$OUT/"
fi
exit "$rc"
