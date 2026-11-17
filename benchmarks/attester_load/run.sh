#!/usr/bin/env bash
# `verity bench attester_load`'s two arms (run.py's docstring), from the source tree's root: the stand-in arm on the
# CPUs and the model arm on the card CARD, pinned to the CPUs CORES, under one gpu-lease of at most LEASE_MIN (50)
# minutes, with vLLM from benchmarks/harness/env.sh's VENV:
#   research run --on NODE --project verity --source . --cwd source --as network-accounting --timeout 60m -- \
#     bash benchmarks/attester_load/run.sh --on CARD --cores CORES
# Under a run that already holds its card (GPU_LEASE_UUID set) both arms run on it without a second lease. Other
# arguments go to run.py.
set -euo pipefail
CARD=; CORES=; ARGS=()
while [ $# -gt 0 ]; do
  case $1 in --on) CARD=${2:?--on CARD}; shift 2 ;; --cores) CORES=${2:?--cores CORES}; shift 2 ;; *) ARGS+=("$1"); shift ;; esac
done
V=$(. benchmarks/harness/env.sh > /dev/null && echo "$VENV")
test -x "$V/bin/python"
RUN=(uv run python benchmarks/attester_load/run.py --vllm-python "$V/bin/python" "${ARGS[@]}")
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  exec "${RUN[@]}"
fi
exec env GPU_LEASE_WHO="${GPU_LEASE_WHO:-network-accounting}" gpu-lease 1 --wait ${CARD:+--on "$CARD"} \
  ${CORES:+--cores "$CORES"} --max-min "${LEASE_MIN:-50}" -- "${RUN[@]}"
