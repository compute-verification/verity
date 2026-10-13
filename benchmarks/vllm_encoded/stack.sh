#!/usr/bin/env bash
# The encoded stack on one GPU of node 1 (cwd = the source tree): builds paged.cu (kv.cu's paged attention, append and
# gather) at the stack's codec (KV_FIELD, default 1024, masked, FOLD6 on, 32 query heads) outside the lease, then under
# one gpu-lease 1 --wait --max-min MAXMIN (default 20) MODE's job. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-1 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-vllm-encoded-stack --env MODE=paged \
#     -- bash benchmarks/vllm_encoded/stack.sh
#
# MODE=paged runs paged_gate.py (ARGS passes through).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
[ -x "$PY" ] || { echo "no python at $PY"; exit 3; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$WORK/venv312/bin:$PATH
VE=benchmarks/vllm_encoded
B=$(mktemp -d /tmp/vllm-stack.XXXXXX); trap 'rm -rf "$B"' EXIT
CU=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
F=${KV_FIELD:-1024}
KVDEFS=(-DKV_FIELD="$F" -DKV_MASK="${KV_MASK:-1}" -DKV_FOLD6="${KV_FOLD6:-1}" -DKV_QHEADS=32)
{ nvcc --version; echo "flags: ${CU[*]} ${KVDEFS[*]}"; } > "$O/build.txt"
nvcc "${CU[@]}" "${KVDEFS[@]}" -o "$B/paged.so" $VE/paged.cu > "$O/paged.ptxas" 2>&1 || { tail -30 "$O/paged.ptxas"; exit 4; }
sha256sum $VE/paged.cu benchmarks/word_codec/kv.cu "$B"/*.so >> "$O/build.txt"
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-20}" --)
case ${MODE:-paged} in
  paged)
    # shellcheck disable=SC2086
    "${LEASE[@]}" "$PY" $VE/paged_gate.py --lib "p$F=$B/paged.so" --out "$O" ${ARGS:-} ;;
  *) echo "unknown MODE ${MODE}"; exit 2 ;;
esac
