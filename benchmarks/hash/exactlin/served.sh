#!/usr/bin/env bash
# exactlin's served standard number on an RTX PRO 6000, an H200 or a B200: dither's sbench.py (the arms stock,
# stock-percall, stock-cutlass on sm_90, exact and exact-fused, which need none of Dither's libraries) on one leased GPU for
# at most SERVED_MAX_MIN (25) minutes, with the clocks read before and after, and with an exact-fused arm only after kgate.py's
# bitwise gate (every config row, both models) passed in the same lease.  Extra arguments go to sbench.py; stock first, so its
# overheads are against stock.  DITHER_NSYS=1 runs it under nsys with the prof op's per-step kernels (steps.py).
# SERVED_SCRIPT (a path from the repository root) runs another script in sbench.py's place, e.g. values.py; SERVED_KGATE=1|0
# forces the gate on or off.  Inside the lease everything runs on the leased GPU's NUMA node, as dither's sbench.sh runs it
# (benchmarks/hash/dither/numabind.py, DITHER_NUMA=0 unbound; env/numa.json and summary.json's "numa" say which).
# On an H200 under a GPU lease:
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/exactlin/served.sh \
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
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config
DITHER=$ROOT/benchmarks/hash/dither; [ -d "$DITHER" ] && PYTHONPATH=$PYTHONPATH:$DITHER
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
fi
cd integrations/vllm
MAX_MIN=${SERVED_MAX_MIN:-25}
# an exact-fused arm is timed only after kgate.py's bitwise gate passes on this GPU, in the same lease (OUT/kgate)
case " $* " in *exact-fused*) KGATE=1 ;; *) KGATE=0 ;; esac
KGATE=${SERVED_KGATE:-$KGATE}
SCRIPT=$ROOT/${SERVED_SCRIPT:-benchmarks/hash/dither/sbench.py}
[ -f "$SCRIPT" ] || { echo "served.sh: no $SCRIPT (set SERVED_SCRIPT)" >&2; exit 2; }
export KGATE KGATE_PY=$PY KGATE_SCRIPT=$ROOT/benchmarks/hash/exactlin/kgate.py
echo "served: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --preemptible --max-min "$MAX_MIN" -- "$PY" "$DITHER/numabind.py" "$OUT/env/numa.json" bash -c '
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.before.csv"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -l 2 > "$0/smi.csv" 2>/dev/null &
  smi=$!
  if [ "$KGATE" = 1 ]; then
    timeout 900 "$KGATE_PY" "$KGATE_SCRIPT" --out "$0/kgate" --shapes all --models llama-3.1-8b,qwen3-8b --no-stock \
      > "$0/kgate.log" 2>&1
    g=$?
    tail -1 "$0/kgate.log"
    if [ $g -ne 0 ]; then echo "served: kgate rc $g, nothing timed"; kill "$smi" 2>/dev/null; exit 4; fi
  fi
  timeout $((60 * '"$MAX_MIN"' - 60 - SECONDS)) "$@"
  r=$?
  kill "$smi" 2>/dev/null
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.after.csv"
  exit $r' "$OUT" "${NSYS[@]}" "$PY" "$SCRIPT" --out "$OUT" --scratch "$S" "$@"
rc=$?
echo "served: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" export --type sqlite --force-overwrite true -o "$S/gate.sqlite" "$S/gate.nsys-rep" > "$OUT/nsys-export.log" 2>&1
  "$PY" "$ROOT/benchmarks/hash/exactlin/steps.py" "$S/gate.sqlite" > "$OUT/steps.json" || echo "served: steps.py rc $?"
  [ "$(stat -c %s "$S/gate.nsys-rep")" -lt 209715200 ] && cp "$S/gate.nsys-rep" "$OUT/"
fi
exit "$rc"
