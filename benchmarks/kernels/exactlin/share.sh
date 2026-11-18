#!/usr/bin/env bash
# exactlin's uncovered share of a served step (share.py) on one leased GPU for at most SHARE_MAX_MIN (40) minutes: each arm
# of SHARE_ARMS (exact-fused,exact) in its own process under nsys, its capture exported to SQLite and split by
# `share.py --analyse`, then `share.py --summary` over the arms; the clocks read before and after.  The leased half runs on
# the leased GPU's NUMA node, as dither's sbench.sh runs (benchmarks/kernels/numabind/numabind.py, DITHER_NUMA=0 unbound;
# env/numa.json says which).  CUDA 13.0 (and its nsys, unless NSYS) is CUDA where given, else the machine's record's
# (benchmarks/kernels/machine.py: MACHINE, MACHINES_D), refused before the lease; FlashInfer's architecture list is the
# leased card's record's `arch`.
# On an RTX PRO 6000 under a GPU lease:
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO MACHINE=NAME bash benchmarks/kernels/exactlin/share.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT/env"
ROOT=${SHARE_ROOT:-$PWD}
if [ "${1:-}" != --leased ]; then
  TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
  S=$(mktemp -d "$TMP/exactlin-share.XXXXXX"); trap 'rm -rf "$S"' EXIT
  git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
  TOOLKIT=13.0   # the CUDA the pinned-vLLM environment runs with, whose nsys traces it
  CUDA=$(python3 "$ROOT/benchmarks/kernels/machine.py" toolkit "$TOOLKIT") || exit 2
  [ -f "$ROOT/benchmarks/hash/dither/sbench.py" ] || { echo "share.sh: no benchmarks/hash/dither/sbench.py under $ROOT (SHARE_ROOT)" >&2; exit 2; }
  NSYS=${NSYS:-$CUDA/bin/nsys}
  [ -x "$NSYS" ] || { echo "share.sh: no nsys at $NSYS (NSYS, else the CUDA toolkit $TOOLKIT's bin)" >&2; exit 3; }
  echo "share: nsys $NSYS; asking for one GPU at $(date -u +%H:%M:%SZ)"
  pre=(); [ "${SHARE_PREEMPTIBLE:-0}" = 1 ] && pre=(--preemptible)   # a preempted trace is lost whole
  numa=(python3 "$(cd "$(dirname "$0")/../numabind" && pwd)/numabind.py" "$OUT/env/numa.json")
  CUDA=$CUDA NSYS=$NSYS SHARE_ROOT=$ROOT SHARE_S=$S gpu-lease 1 --wait "${pre[@]}" --max-min "${SHARE_MAX_MIN:-40}" -- "${numa[@]}" \
    bash "$ROOT/benchmarks/kernels/exactlin/share.sh" --leased
  rc=$?
  echo "share: exited $rc at $(date -u +%H:%M:%SZ)"
  exit "$rc"
fi
S=${SHARE_S:?}; NSYS=${NSYS:?}; CUDA=${CUDA:?}
PY=${PY:-$(. "$ROOT/benchmarks/harness/env.sh" && echo "$VENV/bin/python3")} || exit 2
ARCH=$(python3 "$ROOT/benchmarks/kernels/machine.py" arch "$(nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=name --format=csv,noheader | head -1)") || exit 2
case $ARCH in
  sm_120a) export FLASHINFER_CUDA_ARCH_LIST=12.0f ;;
  sm_90a) export FLASHINFER_CUDA_ARCH_LIST=9.0a VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  sm_100a) export FLASHINFER_CUDA_ARCH_LIST=10.0a VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  *) echo "share.sh: no FlashInfer architecture list for the card's arch $ARCH" >&2; exit 2 ;;
esac
echo "arch $ARCH cuda $CUDA python $PY" > "$OUT/env/node.txt"
export VLLM_CACHE_ROOT=$S/vllm-cache TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache TRITON_CACHE_DIR=$S/triton
export VLLM_DISABLE_COMPILE_CACHE=1 HF_HOME=${HF_HOME:-/workspace/hf}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config:$ROOT/benchmarks/hash/dither
SHARE=$ROOT/benchmarks/kernels/exactlin/share.py
q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -l 2 > "$OUT/smi.csv" 2>/dev/null &
smi=$!
cd "$ROOT/adapters/vllm"
rc=0
for arm in ${SHARE_ARMS:-exact-fused exact}; do
  mkdir -p "$OUT/$arm" "$S/$arm"
  timeout 1200 "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/$arm/trace" --force-overwrite true \
    "$PY" "$SHARE" --arm "$arm" --out "$OUT/$arm" --scratch "$S/$arm" > "$OUT/$arm/run.log" 2>&1
  r=$?
  tail -2 "$OUT/$arm/run.log"
  if [ $r -ne 0 ] || [ ! -f "$S/$arm/trace.nsys-rep" ]; then echo "share: $arm rc $r"; rc=5; continue; fi
  "$NSYS" export --type sqlite --force-overwrite true -o "$S/$arm/trace.sqlite" "$S/$arm/trace.nsys-rep" \
    > "$OUT/$arm/export.log" 2>&1
  "$PY" "$SHARE" --analyse "$S/$arm/trace.sqlite" --meta "$OUT/$arm/share.json" > "$OUT/$arm/analysis.json" \
    || { echo "share: $arm analyse rc $?"; rc=6; }
  gzip -c "$S/$arm/trace.sqlite" > "$S/$arm/trace.sqlite.gz"
  [ "$(stat -c %s "$S/$arm/trace.sqlite.gz")" -lt 209715200 ] && cp "$S/$arm/trace.sqlite.gz" "$OUT/$arm/"
done
"$PY" "$SHARE" --summary "$OUT" > "$OUT/summary.json" || rc=7
kill "$smi" 2>/dev/null
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
exit "$rc"
