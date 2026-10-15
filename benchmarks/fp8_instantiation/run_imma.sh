#!/usr/bin/env bash
# imma.py on one leased sm_120 GPU of vy-nebius-2 (cwd = the clone's root): int8 x int8 -> s32 products and the pair
# design's peel on the device against the exact product.  Nothing to build.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... --send benchmarks/fp8_instantiation/run_imma.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_imma.sh"'
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PYTHONPATH=$PWD/benchmarks/fp8_instantiation OMP_NUM_THREADS=8
export PY=/workspace/jobs/venv312/bin/python
LEASE=(gpu-lease 1 --wait --max-min 30 --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$PY" benchmarks/fp8_instantiation/imma.py --json "$O/imma.json" > "$O/imma_stdout.txt" 2>&1'
rc=$?
echo "imma $rc" | tee "$O/status.txt"
tail -5 "$O/imma_stdout.txt"
exit $rc
