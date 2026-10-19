#!/usr/bin/env bash
# exactlin's gate, then its kernel-level timing (kbench.py), on one leased GPU of node 2 for at most KBENCH_MAX_MIN (30)
# minutes; KSCRIPT=kgate.py runs the gate alone, KSCRIPT=ksweep.py the GEMM config sweep.  Extra arguments go to the
# script (e.g. --reps 15 --shapes decode).
#   research run --on vy-nebius-2 --project verity --campaign exactlin --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-exactlin -- bash benchmarks/hash/exactlin/kbench.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-exactlin.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
PY=${PY:-/workspace/jobs/venv312/bin/python3}
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
