#!/usr/bin/env bash
# The wide-word codec's benchmark on one GPU (cwd = the source tree): builds wt.cu for each field outside the lease, then,
# under one gpu-lease 1 --wait --max-min MAXMIN (default 25), bench.py's gates and timings. Results go to
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-codec-tensor \
#     -- taskset -c 48-95 bash benchmarks/codec_tensor/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). FIELDS (default "255 511 512 521 767 768 1023
# 1279") names the builds; NVFLAGS adds nvcc flags (a variant's -D); DIAGS ("1 3 7") adds a WT_SK_DIAG build of each
# field per value; RECEIPT=1 adds -DWT_RECEIPT=1 and builds reader.cu, the receipts' commit; BENCH_ARGS passes through
# to bench.py. PROFILE, if set, runs bench.py under ncu with those options
# (out/prof.ncu-rep) and prints its details and SASS source pages.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
CT=benchmarks/codec_tensor
FIELDS=${FIELDS:-255 511 512 521 767 768 1023 1279}
B=$(mktemp -d /tmp/codec-tensor.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v ${NVFLAGS:-})
READER=()
if [ "${RECEIPT:-0}" = 1 ]; then
  nvcc "${FLAGS[@]}" -o "$B/reader.so" $CT/reader.cu > "$O/reader.ptxas" 2>&1 || { tail -40 "$O/reader.ptxas"; exit 4; }
  READER=(--reader-lib "$B/reader.so")
  FLAGS+=(-DWT_RECEIPT=1)
fi
declare -A PIDS=()
BUILDS=()
for n in $FIELDS; do
  nvcc "${FLAGS[@]}" -DWT_FIELD="$n" -o "$B/wt-$n.so" $CT/wt.cu > "$O/wt-$n.ptxas" 2>&1 & PIDS[$n]=$!
  BUILDS+=("$n")
  for d in ${DIAGS:-}; do
    nvcc "${FLAGS[@]}" -DWT_FIELD="$n" -DWT_SK_DIAG="$d" -o "$B/wt-$n-diag$d.so" $CT/wt.cu > "$O/wt-$n-diag$d.ptxas" 2>&1 &
    PIDS[$n-diag$d]=$!
    BUILDS+=("$n-diag$d")
  done
done
for n in "${BUILDS[@]}"; do
  wait "${PIDS[$n]}" || { tail -40 "$O/wt-$n.ptxas"; exit 4; }
done
SASS_KEEP='chain_kernelILi0ELi1ELi1E|tc_chain_kernelILi1E|fused_kernelILi1ELi1ELb0E'
for n in $FIELDS; do
  cuobjdump -sass "$B/wt-$n.so" | awk -v keep="$SASS_KEEP" '/Function : /{p = ($0 ~ keep)} p' > "$O/wt-$n.sass" 2>&1
done
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $CT/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
[ -n "${COMPILE_ONLY:-}" ] && exit 0
export B PY CT FIELDS BUILD_NAMES="${BUILDS[*]}" BENCH_ARGS="${BENCH_ARGS:-} ${READER[*]}" PROFILE=${PROFILE:-}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
          -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  LIBS=(); for n in $BUILD_NAMES; do LIBS+=(--lib "$B/wt-$n.so"); done
  NCU=(); [ -n "$PROFILE" ] && NCU=(ncu $PROFILE -f -o "$O/prof")
  # shellcheck disable=SC2086
  "${NCU[@]}" "$PY" $CT/bench.py "${LIBS[@]}" --out "$O" $BENCH_ARGS; rc=$?
  if [ -n "$PROFILE" ]; then
    ncu -i "$O/prof.ncu-rep" --page details > "$O/prof-details.txt" 2>&1
    ncu -i "$O/prof.ncu-rep" --page source --csv --print-source sass > "$O/prof-source.csv" 2>&1
  fi
  q
  exit $rc'
