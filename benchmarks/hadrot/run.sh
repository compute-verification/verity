#!/usr/bin/env bash
# bench.py on one GPU of node 1 (cwd = the source tree); results in $RESEARCH_RUN_DIR/out.  Node 1's GPUs are Kueue's, so
# the queue leases the GPU around the whole job (QUEUED=1) and gpu-lease is skipped; elsewhere one gpu-lease 1 --wait.
#   research run --queue --on vy-nebius-1 --gpus 1 --max-min 20 --mem-gb 32 --project verity --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-hadrot --env QUEUED=1 -- bash benchmarks/hadrot/run.sh
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$OUT"
ROOT=$PWD
S=$(mktemp -d /tmp/hadrot.XXXXXX); trap 'rm -rf "$S"' EXIT
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
for v in /workspace/jobs/venv312 /workspace/cp/venv312 /workspace/pouw/mvp-e2e/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3
done
[ -n "${PY:-}" ] || { echo "no venv312"; exit 3; }
export PATH=$(dirname "$PY"):/usr/local/cuda-13.0/bin:$PATH
export PYTHONPATH=$ROOT/integrations/vllm:$ROOT
export TRITON_CACHE_DIR=$S/triton
nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "$OUT/gpu.txt" 2>&1
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-20}" --)
[ "${QUEUED:-0}" = 1 ] && LEASE=()
"${LEASE[@]}" timeout $((60 * ${MAXMIN:-20} - 30)) "$PY" "$ROOT/benchmarks/hadrot/bench.py" --out "$OUT" "$@" \
  2>&1 | tee "$OUT/bench.log"
rc=${PIPESTATUS[0]}
nvidia-smi --query-gpu=index,uuid,clocks.sm,power.draw,temperature.gpu --format=csv >> "$OUT/gpu.txt" 2>&1
exit "$rc"
