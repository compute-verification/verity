#!/usr/bin/env bash
# One decode step at HBM exhaustion (step.py) on one GPU (cwd = the source tree): builds wt.cu and step.cu at p512
# outside the lease and runs the codecs' tests, then under one gpu-lease 1 --wait --max-min MAXMIN (default 28) the
# step at a tiny model under compute-sanitizer (memcheck: every kernel the step launches, at the batch's MMA width),
# then the step itself. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-codec-tensor \
#     --env MODEL=8b --env BATCH=32 -- taskset -c 48-95 bash benchmarks/codec_tensor/step.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). MODEL (8b, 70b) and BATCH name the point,
# STEP_ARGS passes through to step.py; SANITIZE=0 skips the smoke run.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
CT=benchmarks/codec_tensor
B=$(mktemp -d /tmp/hbm-step.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v ${NVFLAGS:-})
nvcc "${FLAGS[@]}" -DWT_FIELD=512 -o "$B/wt-512.so" $CT/wt.cu > "$O/wt-512.ptxas" 2>&1 & P1=$!
nvcc "${FLAGS[@]}" -DKV_FIELD=512 -o "$B/step-512.so" $CT/step.cu > "$O/step-512.ptxas" 2>&1 & P2=$!
wait $P1 || { tail -40 "$O/wt-512.ptxas"; exit 4; }
wait $P2 || { tail -40 "$O/step-512.ptxas"; exit 4; }
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $CT/tests benchmarks/word_codec/tests/test_kv.py > "$O/tests.log" 2>&1 \
    || { tail -30 "$O/tests.log"; exit 5; }
fi
[ -n "${COMPILE_ONLY:-}" ] && exit 0
export B PY CT MODEL=${MODEL:-8b} BATCH=${BATCH:-32} STEP_ARGS=${STEP_ARGS:-} SANITIZE=${SANITIZE:-1}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-28}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  LIBS=(--wt-lib "$B/wt-512.so" --kv-lib "$B/step-512.so")
  if [ "$SANITIZE" = 1 ]; then
    compute-sanitizer --tool memcheck --error-exitcode 9 --print-limit 20 \
      "$PY" $CT/step.py "${LIBS[@]}" --out "$O/smoke" --model tiny --batch "$BATCH" --ctx 256 --reps 1 \
      --gemm-reps 1 --gate-words 4096 > "$O/smoke.log" 2>&1
    rc=$?; q
    [ $rc = 0 ] || { tail -60 "$O/smoke.log"; exit 9; }
  fi
  # shellcheck disable=SC2086
  "$PY" $CT/step.py "${LIBS[@]}" --out "$O" --model "$MODEL" --batch "$BATCH" $STEP_ARGS; rc=$?
  q
  exit $rc'
