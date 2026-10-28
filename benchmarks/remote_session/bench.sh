#!/usr/bin/env bash
# The pass's τ on one GPU (`pass_bench.py`), under one gpu-lease 1 --wait: build session.cu, then time the read, the
# plain pass and the pipelined pass at each chunk. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 32 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-first-session \
#     -- taskset -c 48-63 bash benchmarks/remote_session/bench.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
export PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
B=$(mktemp -d /tmp/remote-session.XXXXXX); trap 'rm -rf "$B"' EXIT
export B RS=benchmarks/remote_session GIB=${GIB:-16} CHUNKS=${CHUNKS:-14,15,16,17,18} CTAS=${CTAS:-0}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v \
    -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 || { tail -40 "$O/session.ptxas"; exit 4; }
  "$PY" $RS/pass_bench.py --lib "$B/session.so" --gib "$GIB" --chunks "$CHUNKS" --ctas "$CTAS" --out "$O" ${BENCH_ARGS:-} 2>&1 \
    | tee "$O/pass_bench.log"'
