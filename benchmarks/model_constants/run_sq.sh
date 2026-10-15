#!/usr/bin/env bash
# Part A of the model constants (cwd = the source tree): imma_sq.cu built outside the lease at p1024 in kv.cu's default
# squaring form and in FOLD 6, then under one gpu-lease 1 --wait --max-min MAXMIN (default 25) an nvidia-smi sampler
# of the leased GPU's clocks, bench_sq.py's gate and rates, and (NCU=1, the default) Nsight Compute on one launch of each
# squaring chain. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 32 --campaign cmp-12 \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=W12 -- bash benchmarks/model_constants/run_sq.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK); BENCH_ARGS passes through to bench_sq.py.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
MC=benchmarks/model_constants
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
B=$(mktemp -d /tmp/model-constants.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DKV_FIELD=1024)
nvcc "${FLAGS[@]}" -o "$B/mc-default.so" $MC/imma_sq.cu > "$O/mc-default.ptxas" 2>&1 & P1=$!
nvcc "${FLAGS[@]}" -DKV_FOLD6=1 -o "$B/mc-fold6.so" $MC/imma_sq.cu > "$O/mc-fold6.ptxas" 2>&1 & P2=$!
wait $P1 || { tail -30 "$O/mc-default.ptxas"; exit 4; }
wait $P2 || { tail -30 "$O/mc-fold6.ptxas"; exit 4; }
for lib in default fold6; do
  cuobjdump -sass "$B/mc-$lib.so" | awk '/Function : /{p = ($0 ~ /chain_kernel|gate_kernel|mix_kernel|mma_kernel/)} p' \
    > "$O/mc-$lib.sass" 2>&1
done
sha256sum "$B"/*.so > "$O/libs.sha256"
export B PY NCU_BIN MC BENCH_ARGS=${BENCH_ARGS:-} NCU=${NCU:-1}
# shellcheck disable=SC2016
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- bash -c '
  nvidia-smi -q -i "$GPU_LEASE_UUID" > "$O/gpu-q.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,index,uuid,clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/clocks.csv" 2>&1 & SMI=$!
  # shellcheck disable=SC2086
  "$PY" $MC/bench_sq.py --lib default="$B/mc-default.so" --lib fold6="$B/mc-fold6.so" --out "$O" $BENCH_ARGS; rc=$?
  kill $SMI
  if [ "$NCU" = 1 ] && [ $rc = 0 ]; then
    "$NCU_BIN" --clock-control none --section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats \
      --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
      --section SourceCounters -k "regex:chain_kernel" -f -o "$O/ncu-chain" \
      "$PY" $MC/bench_sq.py --lib default="$B/mc-default.so" --out "$O/ncu" --only ncu > "$O/ncu-chain.log" 2>&1
    "$NCU_BIN" --import "$O/ncu-chain.ncu-rep" --csv --page raw > "$O/ncu-chain.csv" 2>&1
  fi
  exit $rc'
