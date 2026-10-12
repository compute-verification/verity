#!/usr/bin/env bash
# The filler climb's job on one GPU (cwd = the source tree): builds `readers.cu` (and `fill.cu` once it exists) outside
# the lease, then `fill.py CMD` under one gpu-lease 1 --wait. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-1 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-fill-bandwidth --env CMD=read -- bash benchmarks/kv_filler/fill_run.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
[ -x "$PY" ] || { echo "no python at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
KF=benchmarks/kv_filler
B=$(mktemp -d /tmp/fill.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
{ nvcc --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
nvcc "${FLAGS[@]}" -o "$B/libreaders.so" $KF/readers.cu > "$O/readers.ptxas" 2>&1 || { tail -30 "$O/readers.ptxas"; exit 4; }
if [ -f $KF/fill.cu ]; then
  nvcc "${FLAGS[@]}" -o "$B/libfill.so" $KF/fill.cu > "$O/fill.ptxas" 2>&1 || { tail -30 "$O/fill.ptxas"; exit 4; }
fi
if grep -E "[1-9][0-9]* bytes spill" "$O"/*.ptxas; then echo "REFUSED: a kernel spills"; exit 4; fi
sha256sum $KF/*.cu "$B"/*.so >> "$O/build.txt"
# shellcheck disable=SC2086
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-20}" -- \
  "$PY" $KF/fill.py "${CMD:-read}" --readers-lib "$B/libreaders.so" ${ARGS:-} --out "$O"
