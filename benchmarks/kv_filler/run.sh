#!/usr/bin/env bash
# The KV filler's benchmark on one GPU (cwd = the source tree): builds the top-up (topup.cu, which includes the
# bandwidth audit's tagger) and the KV codec at p512 (../word_codec/kv.cu) outside the lease, runs the tests, then
# SCRIPT (default bench.py) under one gpu-lease 1 --wait --max-min MAXMIN (default 25) with ARGS. Results go to
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-kv-filler \
#     -- taskset -c 48-95 bash benchmarks/kv_filler/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). SCRIPT=serve.py and SCRIPT=step_profile.py run
# in the e2e's vLLM venv (WORK, default /workspace/pouw/mvp-e2e); serve.py once for each pool sizing of POOLS (default
# "large sized"), each its own process, under the one lease.
set -uxo pipefail
export POOLS=${POOLS:-large sized}
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || PY=${WORK:-/workspace/pouw/mvp-e2e}/venv312/bin/python
SCRIPT=${SCRIPT:-bench.py}
if [ "$SCRIPT" = serve.py ] || [ "$SCRIPT" = step_profile.py ]; then
  WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
  export PYTHONPATH=$PWD:$PWD/integrations/vllm FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf} \
    CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
  . benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
  cat "$WORK"/models/llama-3.1-8b-instruct/*.safetensors > /dev/null
fi
[ -x "$PY" ] || { echo "no python at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
KF=benchmarks/kv_filler
B=$(mktemp -d /tmp/kv-filler.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
{ nvcc --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
nvcc "${FLAGS[@]}" -o "$B/libtopup.so" $KF/topup.cu > "$O/topup.ptxas" 2>&1 &
P1=$!
nvcc "${FLAGS[@]}" -DKV_FIELD=512 -o "$B/libkv-p512.so" benchmarks/word_codec/kv.cu > "$O/kv-p512.ptxas" 2>&1 &
P2=$!
wait $P1 || { tail -30 "$O/topup.ptxas"; exit 4; }
wait $P2 || { tail -30 "$O/kv-p512.ptxas"; exit 4; }
if grep -E "[1-9][0-9]* bytes spill" "$O/topup.ptxas"; then echo "REFUSED: a top-up kernel spills"; exit 4; fi
sha256sum $KF/topup.cu $KF/fill.cu $KF/readers.cu benchmarks/bandwidth/tagger.cu benchmarks/word_codec/kv.cu "$B"/*.so >> "$O/build.txt"
if [ -d $KF/tests ] && "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $KF/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
# Under `research run --queue --gpus 1` the queue leases the GPU around the whole job: QUEUED=1 skips gpu-lease.
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ "${QUEUED:-0}" = 1 ] && LEASE=()
if [ "$SCRIPT" = serve.py ]; then
  # shellcheck disable=SC2086,SC2016
  "${LEASE[@]}" bash -c '
    rc=0
    for pool in $POOLS; do "$@" --pool "$pool" || rc=$?; done
    exit $rc' _ "$PY" $KF/serve.py --topup-lib "$B/libtopup.so" --kv-lib "$B/libkv-p512.so" --out "$O" ${ARGS:-}
  exit $?
fi
if [ "$SCRIPT" = step_profile.py ] && [ -n "${NCU_SHAPES:-}" ]; then
  # One ncu report a shape (step_profile-ncu.json holds one), then the torch profiler's pass over them all for the
  # idle time between kernels.
  export NCUB=/usr/local/cuda/bin/ncu PY KF
  # shellcheck disable=SC2086,SC2016
  "${LEASE[@]}" bash -c '
    set -eo pipefail
    for shape in $NCU_SHAPES; do
      d=$O/ncu-$shape; mkdir -p "$d"
      $NCUB --target-processes all --profile-from-start off --cache-control none --clock-control none \
        --graph-profiling node --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__cycles_active.sum,dram__cycles_elapsed.sum \
        -f -o "$d/step" "$PY" $KF/step_profile.py --out "$d" --ncu --shapes "$shape" --steps "${NCU_STEPS:-3}" > "$d/ncu.log" 2>&1 \
        || { tail -40 "$d/ncu.log"; exit 6; }
      $NCUB --import "$d/step.ncu-rep" --csv --page raw --print-units base > "$d/ncu.csv"
      "$PY" $KF/step_profile.py --out "$d" --from-ncu "$d/ncu.csv" | tee "$d/split.txt"
    done
    "$PY" $KF/step_profile.py --out "$O" --shapes "$(echo $NCU_SHAPES | tr " " ,)" ${ARGS:-}' 
  exit $?
fi
if [ "$SCRIPT" = step_profile.py ]; then
  # shellcheck disable=SC2086
  "${LEASE[@]}" "$PY" $KF/step_profile.py --out "$O" ${ARGS:-}
  exit $?
fi
# shellcheck disable=SC2086
"${LEASE[@]}" "$PY" $KF/$SCRIPT --topup-lib "$B/libtopup.so" --kv-lib "$B/libkv-p512.so" --out "$O" ${ARGS:-}
