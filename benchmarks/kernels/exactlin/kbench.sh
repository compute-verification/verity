#!/usr/bin/env bash
# exactlin's gate, then its kernel-level timing (kbench.py), on one leased GPU for at most KBENCH_MAX_MIN (30)
# minutes; KSCRIPT=kgate.py runs the gate alone, KSCRIPT=ksweep.py the GEMM config sweep.  Extra arguments go to the
# script (e.g. --reps 15 --shapes decode).  Under `research run --gpus 1` (GPU_LEASE_UUID set) the run already holds its
# GPU, and the script runs on it without a second lease; with no `gpu-lease` on PATH it runs on the visible card.
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/kernels/exactlin/kbench.sh
# Runs on an RTX PRO 6000, an H200 or a B200, in benchmarks/harness/env.sh's Python and PYTHONPATH.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/tmp-exactlin.XXXXXX"); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
. benchmarks/harness/env.sh || exit 2
PY=$VENV/bin/python3
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$CUDA/bin:$PATH
export TRITON_CACHE_DIR=$S/triton
nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
MAX_MIN=${KBENCH_MAX_MIN:-30}
LEASE=(gpu-lease 1 --wait --preemptible --max-min "$MAX_MIN" --)
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  LEASE=()
  echo "kbench: on the run's GPU $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)"
elif ! command -v gpu-lease >/dev/null; then
  LEASE=()
  echo "kbench: no gpu-lease on PATH, so on the visible card (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}) at $(date -u +%H:%M:%SZ)"
else
  echo "kbench: asking for one GPU at $(date -u +%H:%M:%SZ)"
fi
"${LEASE[@]}" timeout $((60 * MAX_MIN - 30)) "$PY" "$ROOT/benchmarks/kernels/exactlin/${KSCRIPT:-kbench.py}" --out "$OUT" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "kbench: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
