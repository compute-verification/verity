#!/usr/bin/env bash
# One of this directory's scripts on one leased GPU of node 2 (or 3) for at most S8_MAX_MIN (30) minutes, from the shipped tree:
# S8_SCRIPT=run.py (the gate, default), tune.py, bench.py or commit.py; extra arguments go to it. What needs no GPU (the
# commitment kernels' nvcc build) runs before the lease.
#   research run --on vy-nebius-2 --project verity --campaign e2e-demo --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-kp2s8 -- bash benchmarks/torch_gpu/s8_linear/run.sh [ARGS...]
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
TMP=/workspace/pouw; mkdir -p "$TMP" 2>/dev/null && [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/tmp-s8-linear.XXXXXX"); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH
export PYTHONPATH=$ROOT/integrations/vllm:$ROOT
export TRITON_CACHE_DIR=$S/triton
SCRIPT=${S8_SCRIPT:-run.py}
if [ "$SCRIPT" = commit.py ]; then
  "$PY" -c "from verity_vllm.commit import row_tables_device as D; D.device_lib('120'); D.host_lib()" || exit 3
fi
nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
  > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
MAX_MIN=${S8_MAX_MIN:-30}
# a timed script pauses the fill queue and isn't preempted (kernel-engineering skill); the gate times nothing
LEASE=(--preemptible); [ "$SCRIPT" != run.py ] && LEASE=(--timed)
echo "s8_linear: $SCRIPT asking for one GPU (${LEASE[*]}) at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait "${LEASE[@]}" --max-min "$MAX_MIN" -- bash -c '
  [ -n "${GPU_LEASE_UUID:-}" ] && nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$0/clocks.txt" 2>&1
  exec timeout $((60 * $1 - 30)) "$2" "$3" --out "$0" "${@:4}"' "$OUT" "$MAX_MIN" "$PY" "$ROOT/benchmarks/torch_gpu/s8_linear/$SCRIPT" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "s8_linear: $SCRIPT exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
