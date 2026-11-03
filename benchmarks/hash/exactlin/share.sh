#!/usr/bin/env bash
# exactlin's uncovered share of a served step (share.py) on one leased GPU for at most SHARE_MAX_MIN (40) minutes: each arm
# of SHARE_ARMS (exact-fused,exact) in its own process under nsys, its capture exported to SQLite and split by
# `share.py --analyse`, then `share.py --summary` over the arms; the clocks read before and after.  The leased half runs on
# the leased GPU's NUMA node, as dither's sbench.sh runs (benchmarks/hash/dither/numabind.py, DITHER_NUMA=0 unbound;
# env/numa.json says which).
# On an RTX PRO 6000 under a GPU lease:
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/exactlin/share.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT/env"
ROOT=${SHARE_ROOT:-$PWD}
if [ "${1:-}" != --leased ]; then
  TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
  S=$(mktemp -d "$TMP/exactlin-share.XXXXXX"); trap 'rm -rf "$S"' EXIT
  git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
  [ -f "$ROOT/benchmarks/hash/dither/sbench.py" ] || { echo "share.sh: no benchmarks/hash/dither/sbench.py under $ROOT (SHARE_ROOT)" >&2; exit 2; }
  NSYS=${NSYS:-/usr/local/cuda-13.0/bin/nsys}
  [ -x "$NSYS" ] || NSYS=$(command -v nsys || ls -d /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda*/bin/nsys 2>/dev/null | tail -1)
  [ -x "$NSYS" ] || { echo "share.sh: no nsys on $(hostname) (looked for NSYS, \$CUDA/bin, PATH, /opt/nvidia/nsight-systems)" >&2; exit 3; }
  echo "share: nsys $NSYS; asking for one GPU at $(date -u +%H:%M:%SZ)"
  pre=(); [ "${SHARE_PREEMPTIBLE:-0}" = 1 ] && pre=(--preemptible)   # a preempted trace is lost whole
  numa=(python3 "$(cd "$(dirname "$0")/../dither" && pwd)/numabind.py" "$OUT/env/numa.json")
  NSYS=$NSYS SHARE_ROOT=$ROOT SHARE_S=$S gpu-lease 1 --wait "${pre[@]}" --max-min "${SHARE_MAX_MIN:-40}" -- "${numa[@]}" \
    bash "$ROOT/benchmarks/hash/exactlin/share.sh" --leased
  rc=$?
  echo "share: exited $rc at $(date -u +%H:%M:%SZ)"
  exit "$rc"
fi
S=${SHARE_S:?}; NSYS=${NSYS:?}
CUDA=${CUDA:-/usr/local/cuda-13.0}
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
PY=${PY:-$(. "$ROOT/benchmarks/harness/env.sh" && echo "$VENV/bin/python3")} || exit 2
CC=$(nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=compute_cap --format=csv,noheader | head -1)
case $CC in
  12.0) export FLASHINFER_CUDA_ARCH_LIST=12.0f ;;
  9.0) export FLASHINFER_CUDA_ARCH_LIST=9.0a VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  10.0) export FLASHINFER_CUDA_ARCH_LIST=10.0a VLLM_USE_FLASHINFER_SAMPLER=0 ;;
  *) echo "share.sh: compute capability $CC" >&2; exit 2 ;;
esac
echo "compute_cap $CC python $PY" > "$OUT/env/node.txt"
export VLLM_CACHE_ROOT=$S/vllm-cache TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache TRITON_CACHE_DIR=$S/triton
export VLLM_DISABLE_COMPILE_CACHE=1 HF_HOME=${HF_HOME:-/workspace/hf}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config:$ROOT/benchmarks/hash/dither
SHARE=$ROOT/benchmarks/hash/exactlin/share.py
q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -l 2 > "$OUT/smi.csv" 2>/dev/null &
smi=$!
cd "$ROOT/integrations/vllm"
rc=0
for arm in ${SHARE_ARMS:-exact-fused exact}; do
  mkdir -p "$OUT/$arm" "$S/$arm"
  timeout 1200 "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/$arm/cap" --force-overwrite true \
    "$PY" "$SHARE" --arm "$arm" --out "$OUT/$arm" --scratch "$S/$arm" > "$OUT/$arm/run.log" 2>&1
  r=$?
  tail -2 "$OUT/$arm/run.log"
  if [ $r -ne 0 ] || [ ! -f "$S/$arm/cap.nsys-rep" ]; then echo "share: $arm rc $r"; rc=5; continue; fi
  "$NSYS" export --type sqlite --force-overwrite true -o "$S/$arm/cap.sqlite" "$S/$arm/cap.nsys-rep" \
    > "$OUT/$arm/export.log" 2>&1
  "$PY" "$SHARE" --analyse "$S/$arm/cap.sqlite" --meta "$OUT/$arm/share.json" > "$OUT/$arm/analysis.json" \
    || { echo "share: $arm analyse rc $?"; rc=6; }
  gzip -c "$S/$arm/cap.sqlite" > "$S/$arm/cap.sqlite.gz"
  [ "$(stat -c %s "$S/$arm/cap.sqlite.gz")" -lt 209715200 ] && cp "$S/$arm/cap.sqlite.gz" "$OUT/$arm/"
done
"$PY" "$SHARE" --summary "$OUT" > "$OUT/summary.json" || rc=7
kill "$smi" 2>/dev/null
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
exit "$rc"
