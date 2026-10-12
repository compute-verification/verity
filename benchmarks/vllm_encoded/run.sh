#!/usr/bin/env bash
# Llama-3.1-8B decode in vLLM with its linear weights stored only encoded, against the same engine with plain FP8 weights
# (cwd = the source tree).  Builds ve.cu outside the lease, runs the layout tests, then under one gpu-lease 1 --wait
# --max-min MAXMIN (default 25) one serve.py process per arm, and compare.py's gates and points.  Results go to
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-vllm-encoded \
#     -- taskset -c 48-95 bash benchmarks/vllm_encoded/run.sh
#
# WORK is the e2e's vLLM environment on node 2 (benchmarks/pouw/pearl_c_vllm/setup.sh).  ARMS (default
# "stock plain enc:2 enc:3 stock") are serve.py's arms in order, an encoded arm as enc:K or enc:K:LIB (LIB pb56 or pb60);
# SERVE_ARGS passes through to every arm; ARM_MIN bounds one arm (default 7 minutes).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || { echo "no $WORK/ready.json: run benchmarks/pouw/pearl_c_vllm/setup.sh"; exit 3; }
export PATH=/usr/local/cuda-13.0/bin:$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf}
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export PYTHONPATH=$PWD:$PWD/integrations/vllm CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
VE=benchmarks/vllm_encoded
B=$(mktemp -d /tmp/vllm-encoded.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -I benchmarks/word_codec
       -DWC_KMAX="${WC_KMAX:-3}")
declare -A DEFS=([pb56]="-DWC_PB=56" [pb60]="-DWC_PB=60")
for lib in ${LIBSET:-pb56 pb60}; do
  # shellcheck disable=SC2086
  nvcc "${FLAGS[@]}" ${DEFS[$lib]} -o "$B/ve-$lib.so" $VE/ve.cu > "$O/ve-$lib.ptxas" 2>&1 &
done
wait || true
for lib in ${LIBSET:-pb56 pb60}; do [ -e "$B/ve-$lib.so" ] || { tail -30 "$O/ve-$lib.ptxas"; exit 4; }; done
sha256sum "$B"/*.so > "$O/libs.sha256"
(cd $VE && "$PY" -m pytest -q -p no:cacheprovider tests) > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ("model", "revision", "vllm", "torch")}))' \
  "$WORK/ready.json" > "$O/ready.json"
export B PY MODEL VE ARMS=${ARMS:-stock plain enc:2 enc:3 stock} SERVE_ARGS=${SERVE_ARGS:-} ARM_MIN=${ARM_MIN:-7}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  UUID=${GPU_LEASE_UUID%%,*}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,memory.used \
          --format=csv,noheader -i "$UUID" | sed "s/^/$1, /" >> "$O/gpu.txt" 2>&1; }
  i=0
  for arm in $ARMS; do
    i=$((i + 1)); IFS=: read -r name k lib <<< "$arm"
    case $name in
      enc) A=(--arm enc --k "$k" --lib "$B/ve-${lib:-pb56}.so"); tag=enc$k${lib:+-$lib} ;;
      plain) A=(--arm plain --lib "$B/ve-${k:-pb56}.so"); tag=plain${k:+-$k} ;;
      *) A=(--arm "$name"); tag=$name ;;
    esac
    tag=$i-$tag; q "$tag start"
    # shellcheck disable=SC2086
    timeout "$((ARM_MIN * 60))" "$PY" $VE/serve.py "${A[@]}" --model "$MODEL" --out "$O/arm-$tag.json" $SERVE_ARGS \
      > "$O/arm-$tag.log" 2>&1
    echo "$tag $?" >> "$O/arms.rc"; q "$tag end"
  done'
"$PY" $VE/compare.py "$O" > "$O/compare.log" 2>&1; rc=$?
cat "$O/arms.rc" "$O/compare.log"
exit $rc
