#!/usr/bin/env bash
# The encoded stack on one GPU of node 1 (cwd = the source tree): builds paged.cu (kv.cu's paged attention, append and
# gather) at the stack's codec (KV_FIELD, default 1024, masked, FOLD6 on, 32 query heads) outside the lease, then under
# one gpu-lease 1 --wait --max-min MAXMIN (default 20) MODE's job. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --queue --on vy-nebius-1 --gpus 1 --max-min 30 --project verity --source . --cwd source \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-vllm-encoded-stack \
#     --env QUEUED=1 --env MODE=paged -- bash benchmarks/vllm_encoded/stack.sh
#
# Under the queue (QUEUED=1) the GPU is leased around the whole job and gpu-lease is skipped.
# MODE=paged runs paged_gate.py (ARGS passes through). MODE=serve also builds wt.cu (WT_RECEIPT at
# KV_FIELD, masked, WT_FOLD6 default 1) and reader.cu, then runs serve.py's ARMS (default "stock pstack stack") in order,
# each with SERVE_ARGS (default one gate-sized point), into $O/arm-<arm>[-<i>].json, and stack_compare.py over them.
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
WTDEFS=(-DWT_FIELD="$F" -DWT_MASK="${KV_MASK:-1}" -DWT_FOLD6="${WT_FOLD6:-1}" -DWT_RECEIPT=1)
{ nvcc --version; echo "flags: ${CU[*]} paged ${KVDEFS[*]} wt ${WTDEFS[*]}"; } > "$O/build.txt"
if [ "${MODE:-paged}" = serve ]; then
  CT=benchmarks/codec_tensor
  nvcc "${CU[@]}" "${WTDEFS[@]}" -o "$B/wt.so" $CT/wt.cu > "$O/wt.ptxas" 2>&1 &
  nvcc "${CU[@]}" -o "$B/reader.so" $CT/reader.cu > "$O/reader.ptxas" 2>&1 &
fi
nvcc "${CU[@]}" "${KVDEFS[@]}" -o "$B/paged.so" $VE/paged.cu > "$O/paged.ptxas" 2>&1 || { tail -30 "$O/paged.ptxas"; exit 4; }
wait
for so in "$O"/*.ptxas; do lib=$(basename "$so" .ptxas); [ -e "$B/$lib.so" ] || { tail -40 "$so"; exit 4; }; done
sha256sum $VE/paged.cu benchmarks/word_codec/kv.cu "$B"/*.so >> "$O/build.txt"
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-20}" --)
[ "${QUEUED:-0}" = 1 ] && LEASE=()
case ${MODE:-paged} in
  paged)
    # shellcheck disable=SC2086
    "${LEASE[@]}" "$PY" $VE/paged_gate.py --lib "p$F=$B/paged.so" --out "$O" ${ARGS:-} ;;
  serve)
    MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
    export PATH=$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf}
    . benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
    export PYTHONPATH=$PWD:$PWD/integrations/vllm CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
    LIBS=(--wt-lib "$B/wt.so" --reader-lib "$B/reader.so" --paged-lib "$B/paged.so")
    declare -A SEEN=()
    rc=0
    for arm in ${ARMS:-stock pstack stack}; do
      i=${SEEN[$arm]:-0}; SEEN[$arm]=$((i + 1))
      tag=$arm; [ "$i" -gt 0 ] && tag=$arm-$i
      # shellcheck disable=SC2086
      "${LEASE[@]}" timeout "${ARM_MIN:-12}m" "$PY" $VE/serve.py --arm "$arm" --k "${K:-3}" "${LIBS[@]}" --model "$MODEL" \
        --out "$O/arm-$tag.json" ${SERVE_ARGS:---batches 1,4,32 --contexts 512 --steps 32} > "$O/arm-$tag.log" 2>&1
      r=$?; echo "arm $tag exit $r"; tail -5 "$O/arm-$tag.log"; [ $r = 0 ] || rc=$r
    done
    "$PY" $VE/stack_compare.py "$O" || rc=$?
    exit $rc ;;
  *) echo "unknown MODE ${MODE}"; exit 2 ;;
esac
