#!/usr/bin/env bash
# The timing-loop measurement on vy-nebius-2, run by
#   research run --on vy-nebius-2 --cwd source --declared-output 'out/*' -- bash benchmarks/pous/timing_loop/node.sh [bench args]
# It builds `timing-loop` from a copy of lean/ in a scratch directory, then takes one GPU of $TIMING_GPUS (NUMA node 1, beside
# the quiet cores) for at most $TIMING_MAX_MIN minutes and runs bench.py in `vy-pous-quiet`'s scope on cores $TIMING_CPUS: the
# responder's thread, the verifier and the worker on the quiet cores 124-127, the harness and its load on 120-123.
# Results go to $RESEARCH_RUN_DIR/out; the store, the leaves and the build are scratch.
set -euo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out
PY=${TIMING_PY:-/workspace/jobs/venv312/bin/python}
GPUS=${TIMING_GPUS:-4,5,6}
export PATH=$HOME/.elan/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/workspace/jobs/cuda-12.9}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting:timing-loop}
SRC=$(pwd)
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp -r "$SRC/benchmarks/pous/timing_loop/lean" "$d/lean"
(cd "$d/lean" && rm -rf .lake && lake build)
mkdir -p "$OUT"
gpu-lease 1 --wait --on "$GPUS" --max-min "${TIMING_MAX_MIN:-30}" -- \
    env VY_POUS_QUIET_CPUS="${TIMING_CPUS:-120-127}" vy-pous-quiet "$PY" "$SRC/benchmarks/pous/timing_loop/bench.py" \
    --out "$OUT" --scratch "$d/scratch" --lean-exe "$d/lean/.lake/build/bin/timing-loop" \
    --build-dir "${TIMING_BUILD_DIR:-/workspace/jobs/cache/jit/pous_native}" \
    --verifier-cpu 126 --helper-cpu 127 --responder-cpu 124 "$@"
