#!/usr/bin/env bash
# One decode step at HBM exhaustion (step.py) on one GPU (cwd = the source tree). Everything runs under one
# gpu-lease 1 --wait --max-min MAXMIN (default 28), which also waits out node 2's booked windows: the builds of wt.cu
# and step.cu at p512 and the codecs' tests, the step at a tiny model under compute-sanitizer (memcheck: every kernel
# the step launches, at the batch's MMA width), then the step itself. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-codec-tensor \
#     --env MODEL=8b --env BATCH=32 -- taskset -c 48-95 bash benchmarks/codec_tensor/step.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). MODEL (8b, 70b) and BATCH name the point (kv.cu
# is built at the model's query heads, KV_QHEADS), GEMM_M the linears' M past a batch of 32 (step.py --gemm-m),
# STEP_ARGS passes through to step.py; SANITIZE=0 skips the smoke run. RECEIPT=1 builds wt.cu's sketch warps with
# sketch-sha512 (WT_RECEIPT) and reader.cu for its commit. Each library's `cuobjdump -res-usage` goes in OUT/*.res.
set -uxo pipefail
MODEL=${MODEL:-8b}
case $MODEL in 70b) QHEADS=64 ;; *) QHEADS=32 ;; esac
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
CT=benchmarks/codec_tensor
B=$(mktemp -d /tmp/hbm-step.XXXXXX); trap 'rm -rf "$B"' EXIT
export B PY CT MODEL QHEADS BATCH=${BATCH:-32} GEMM_M=${GEMM_M:-0} STEP_ARGS=${STEP_ARGS:-} SANITIZE=${SANITIZE:-1} \
  NVFLAGS=${NVFLAGS:-} COMPILE_ONLY=${COMPILE_ONLY:-} RECEIPT=${RECEIPT:-0}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-28}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  # shellcheck disable=SC2206
  FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v $NVFLAGS)
  nvcc "${FLAGS[@]}" -DWT_FIELD=512 -DWT_RECEIPT=$RECEIPT -o "$B/wt-512.so" $CT/wt.cu > "$O/wt-512.ptxas" 2>&1 & P1=$!
  nvcc "${FLAGS[@]}" -DKV_FIELD=512 -DKV_QHEADS=$QHEADS -o "$B/step-512.so" $CT/step.cu > "$O/step-512.ptxas" 2>&1 & P2=$!
  LIBS=(--wt-lib "$B/wt-512.so" --kv-lib "$B/step-512.so")
  if [ "$RECEIPT" = 1 ]; then
    nvcc "${FLAGS[@]}" -o "$B/reader.so" $CT/reader.cu > "$O/reader.ptxas" 2>&1 || { tail -40 "$O/reader.ptxas"; exit 4; }
    LIBS+=(--reader-lib "$B/reader.so")
  fi
  wait $P1 || { tail -40 "$O/wt-512.ptxas"; exit 4; }
  wait $P2 || { tail -40 "$O/step-512.ptxas"; exit 4; }
  sha256sum "$B"/*.so > "$O/libs.sha256"
  for so in "$B"/*.so; do cuobjdump -res-usage "$so" > "$O/$(basename "$so" .so).res" 2>&1; done
  if "$PY" -c "import pytest" 2> /dev/null; then
    "$PY" -m pytest -q -p no:cacheprovider $CT/tests benchmarks/word_codec/tests/test_kv.py > "$O/tests.log" 2>&1 \
      || { tail -30 "$O/tests.log"; exit 5; }
  fi
  [ -n "$COMPILE_ONLY" ] && exit 0
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  if [ "$SANITIZE" = 1 ]; then
    env -u RESEARCH_RUN_ID compute-sanitizer --tool memcheck --error-exitcode 9 --print-limit 20 \
      "$PY" $CT/step.py "${LIBS[@]}" --out "$O/smoke" --model tiny --batch "$BATCH" --gemm-m "$GEMM_M" --ctx 256 --reps 1 \
      --gemm-reps 1 --gate-words 4096 > "$O/smoke.log" 2>&1
    rc=$?; q
    [ $rc = 0 ] || { tail -60 "$O/smoke.log"; exit 9; }
  fi
  # shellcheck disable=SC2086
  "$PY" $CT/step.py "${LIBS[@]}" --out "$O" --model "$MODEL" --batch "$BATCH" --gemm-m "$GEMM_M" $STEP_ARGS; rc=$?
  q
  exit $rc'
