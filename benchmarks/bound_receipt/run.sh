#!/usr/bin/env bash
# The bound receipt's benchmark on one GPU (cwd = the source tree): builds bound.cu at p512 in ra-codec-wide's fastest
# p512 configuration (p512s1r3cuek), one library for each k in KS (default "2 3"), outside the lease, runs
# bound_receipt's tests, then, under one gpu-lease 1 --wait --max-min MAXMIN (default 25), bound_bench.py's gates and
# timings over every library. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-bound-receipt \
#     -- taskset -c 48-95 bash benchmarks/bound_receipt/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). BENCH_ARGS passes through to bound_bench.py;
# WW_DEFS adds -D flags to every build.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
BR=benchmarks/bound_receipt
KS=${KS:-2 3}
B=$(mktemp -d /tmp/bound-receipt.XXXXXX); trap 'rm -rf "$B"' EXIT
declare -A PIDS=()
for k in $KS; do
  # shellcheck disable=SC2086
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v \
    -I benchmarks/bandwidth -DWW_BITS=512 -DWW_SQR=1 -DWW_KMAX="$k" -DWW_ROLL=1 -DWW_OCC=3 -DWW_CC=1 -DWW_B3U=1 \
    -DWW_SKEND=1 -DWW_SKU=1 ${WW_DEFS:-} -o "$B/bound-k$k.so" $BR/bound.cu > "$O/bound-k$k.ptxas" 2>&1 & PIDS[$k]=$!
done
for k in "${!PIDS[@]}"; do
  wait "${PIDS[$k]}" || { tail -30 "$O/bound-k$k.ptxas"; exit 4; }
done
LIBS=""
for k in $KS; do
  cuobjdump -sass "$B/bound-k$k.so" | awk '/Function : /{p = ($0 ~ /bound_kernel/)} p' > "$O/bound-k$k.sass" 2>&1
  LIBS+=" --lib $B/bound-k$k.so"
done
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $BR/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
export B PY BR LIBS BENCH_ARGS=${BENCH_ARGS:-}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
          -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  # shellcheck disable=SC2086
  "$PY" $BR/bound_bench.py $LIBS --out "$O" $BENCH_ARGS; rc=$?
  q
  exit $rc'
