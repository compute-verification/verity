#!/usr/bin/env bash
# exactlin's gate, then its kernel-level timing (kbench.py), on one leased GPU for at most KBENCH_MAX_MIN (30)
# minutes; KSCRIPT=kgate.py runs the gate alone, KSCRIPT=ksweep.py the GEMM config sweep.  Extra arguments go to the
# script (e.g. --reps 15 --shapes decode).
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/exactlin/kbench.sh
# Runs on an RTX PRO 6000, an H200 or a B200 (PY, else the first of /workspace/jobs/venv312 and /workspace/cp/venv312).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/tmp-exactlin.XXXXXX"); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH
export PYTHONPATH=$ROOT/integrations/vllm:$ROOT
export TRITON_CACHE_DIR=$S/triton
nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
MAX_MIN=${KBENCH_MAX_MIN:-30}
echo "kbench: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --preemptible --max-min "$MAX_MIN" -- timeout $((60 * MAX_MIN - 30)) \
  "$PY" "$ROOT/benchmarks/hash/exactlin/${KSCRIPT:-kbench.py}" --out "$OUT" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "kbench: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
