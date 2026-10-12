#!/usr/bin/env bash
# The word codec's benchmark on one GPU (cwd = the source tree): builds wc.cu outside the lease in both squaring forms
# (split, wide) and with 60 payload bits a word (dense), then, under one gpu-lease 1 --wait --max-min MAXMIN (default
# 25), bench.py's gates and timings and Nsight Compute on the standalone decode and the fused GEMM. Results go to
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-word-codec \
#     -- taskset -c 48-95 bash benchmarks/word_codec/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). BENCH_ARGS passes through to bench.py; NCU=0 skips
# the profiles; LIBSET (default "split wide dense") names the builds and WC_KMAX (default 8) the largest k they hold.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
WC=benchmarks/word_codec
export LIBSET=${LIBSET:-split wide dense}
B=$(mktemp -d /tmp/word-codec.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DWC_KMAX="${WC_KMAX:-8}")
declare -A DEFS=([split]="-DWC_SQR=0" [wide]="-DWC_SQR=1" [dense]="-DWC_SQR=0 -DWC_PB=60")
declare -A PIDS=()
for lib in $LIBSET; do
  # shellcheck disable=SC2086
  nvcc "${FLAGS[@]}" ${DEFS[$lib]} -o "$B/wc-$lib.so" $WC/wc.cu > "$O/wc-$lib.ptxas" 2>&1 & PIDS[$lib]=$!
done
for lib in $LIBSET; do
  wait "${PIDS[$lib]}" || { tail -30 "$O/wc-$lib.ptxas"; exit 4; }
done
SASS_KEEP='chain_squareILi1E|chain_encodeILi1E|decode_kernelILi[12]ELi0E|gemm_pipe_kernelILi[02]ELi1ELb0E|gemm_ws_kernelILi[03]ELi1ELb0E'
for lib in $LIBSET; do
  cuobjdump -sass "$B/wc-$lib.so" | awk -v keep="$SASS_KEEP" '/Function : /{p = ($0 ~ keep)} p' > "$O/wc-$lib.sass" 2>&1
done
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $WC/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
export B PY NCU_BIN WC BENCH_ARGS=${BENCH_ARGS:-} NCU=${NCU:-1} NCU_WS=${NCU_WS:-}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
          -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  LIBS=(); for lib in $LIBSET; do LIBS+=(--lib "$lib=$B/wc-$lib.so"); done
  # shellcheck disable=SC2086
  "$PY" $WC/bench.py "${LIBS[@]}" --out "$O" $BENCH_ARGS; rc=$?
  q
  if [ "$NCU" = 1 ] && [ $rc = 0 ]; then
    S=(--clock-control none --cache-control all --section SpeedOfLight --section ComputeWorkloadAnalysis
       --section InstructionStats --section Occupancy --section MemoryWorkloadAnalysis)
    for lib in split wide; do
      [ -e "$B/wc-$lib.so" ] || continue
      "$NCU_BIN" "${S[@]}" -k "regex:decode_kernel|read_kernel|tag_kernel" -f -o "$O/ncu-decode-$lib" \
        "$PY" $WC/bench.py --lib $lib="$B/wc-$lib.so" --out "$O/ncu-$lib" --mode ncu-decode --store-gib 1 \
        > "$O/ncu-decode-$lib.log" 2>&1
      "$NCU_BIN" --import "$O/ncu-decode-$lib.ncu-rep" --csv --page raw > "$O/ncu-decode-$lib.csv" 2>&1
    done
    for lib in split dense; do
      [ -e "$B/wc-$lib.so" ] || continue
      "$NCU_BIN" "${S[@]}" -k "regex:gemm_kernel|gemm_pipe_kernel|gemm_ws_kernel" -f -o "$O/ncu-gemm-$lib" \
        "$PY" $WC/bench.py --lib $lib="$B/wc-$lib.so" --out "$O/ncu-gemm-$lib" --mode ncu-gemm --shapes gate_up \
        --batches 1,32 --ks 2,3 ${NCU_WS:+--ws "$NCU_WS"} > "$O/ncu-gemm-$lib.log" 2>&1
      "$NCU_BIN" --import "$O/ncu-gemm-$lib.ncu-rep" --csv --page raw > "$O/ncu-gemm-$lib.csv" 2>&1
    done
    q
  fi
  exit $rc'
