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
# the profiles (NCU_KS, default 2,3, the GEMM's k there; NCU_WS and NCU_TAG_WS its configurations); LIBSET (default
# "split wide dense") names the builds (split8, dense8: words decoded 8 at a time, not 4; densev0: the tags in BLAKE3's
# plain form, V0, not V1's IMAD adds)
# and WC_KMAX (default 8) the largest k they hold.
# KV=1 runs the KV cache's codec instead: kv.cu built at each width of KV_FIELDS (default 512 768; a width tagged k1
# squares by one level of Karatsuba, o1 builds at ptxas -O1, h32 at 32 query heads (8B's attention), t16 with the
# tensor-core attention's tiles of 16 tokens, p0 without the occupancy form's L2 prefetch, as in 1536k1o1 or 512h32),
# then KV_SCRIPT (default kv_bench.py) with
# KV_ARGS; SANITIZE=1 runs KV_SCRIPT's --sanitize smoke under compute-sanitizer's memcheck, racecheck and synccheck,
# and SANITIZE=2 runs KV_SCRIPT after it if every tool passed. KV_NCU=1 profiles KV_SCRIPT --ncu instead of timing it.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
WC=benchmarks/word_codec
# the venv doesn't install verity; kv_receipt.py and the tag tests import it (stdlib only) from the source tree
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export LIBSET=${LIBSET:-split wide dense}
B=$(mktemp -d /tmp/word-codec.XXXXXX); trap 'rm -rf "$B"' EXIT
if [ "${KV:-0}" = 1 ]; then
  LIBS=()
  for f in ${KV_FIELDS:-512 768}; do
    n=${f%%[a-z]*} tag=${f#"${f%%[a-z]*}"} flags=()
    lib=p$n${tag:+-$tag}
    [[ $tag == *k1* ]] && flags+=(-DKV_KARATSUBA=1)
    [[ $tag == *o1* ]] && flags+=(-Xptxas -O1)
    [[ $tag == *h32* ]] && flags+=(-DKV_QHEADS=32)
    [[ $tag == *t16* ]] && flags+=(-DKV_MT=16)
    [[ $tag == *p0* ]] && flags+=(-DKV_OCC_PF=0)
    nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DKV_FIELD=$n \
      "${flags[@]}" -o "$B/kv-$lib.so" $WC/kv.cu > "$O/kv-$lib.ptxas" 2>&1 || { tail -30 "$O/kv-$lib.ptxas"; exit 4; }
    LIBS+=(--lib "$lib=$B/kv-$lib.so")
  done
  sha256sum "$B"/*.so > "$O/libs.sha256"
  if "$PY" -c 'import pytest' 2> /dev/null; then
    "$PY" -m pytest -q -p no:cacheprovider $WC/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
  fi
  KV_SCRIPT=${KV_SCRIPT:-kv_bench.py}
  if [ "${SANITIZE:-0}" != 0 ]; then
    SAN=/usr/local/cuda-13.0/bin/compute-sanitizer
    # shellcheck disable=SC2086
    gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- bash -c '
      rc=0
      for tool in memcheck racecheck synccheck; do
        "$0" --tool $tool --error-exitcode 9 "$1" "$2" "${@:4}" --out "$3/san-$tool" --sanitize > "$3/san-$tool.log" 2>&1
        r=$?; echo "$tool rc=$r"; tail -4 "$3/san-$tool.log"; [ $r = 0 ] || rc=$r
      done
      exit $rc' "$SAN" "$PY" $WC/$KV_SCRIPT "$O" "${LIBS[@]}" ${KV_ARGS:-} || exit 6
    # SANITIZE=2: the timed run follows a clean smoke
    [ "$SANITIZE" = 2 ] || exit 0
  fi
  # KV_NCU=1 profiles KV_SCRIPT --ncu under Nsight Compute instead of timing it (the kernels matching KV_NCU_K) and
  # exports the raw metrics and the per-instruction SASS samples
  if [ "${KV_NCU:-0}" = 1 ]; then
    # shellcheck disable=SC2086,SC2016
    gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- bash -c '
      ncu=$1 o=$2; shift 2
      "$ncu" --clock-control none --section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats \
        --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
        --section SourceCounters -k "regex:${KV_NCU_K:-attn_mma_ws|rc_tag}" -f -o "$o/ncu-kv" "$@" --out "$o" --ncu \
        > "$o/ncu-kv.log" 2>&1 || { tail -20 "$o/ncu-kv.log"; exit 7; }
      "$ncu" --import "$o/ncu-kv.ncu-rep" --csv --page raw > "$o/ncu-kv.csv" 2>&1
      "$ncu" --import "$o/ncu-kv.ncu-rep" --csv --page source --print-source sass > "$o/ncu-kv-sass.csv" 2>&1
      exit 0' "$NCU_BIN" "$O" "$PY" $WC/$KV_SCRIPT "${LIBS[@]}" ${KV_ARGS:-}
    exit $?
  fi
  # KV_REPEAT=N (default 1) runs KV_SCRIPT N times in one lease, each its own process, writing under $O/repI
  if [ "${KV_REPEAT:-1}" = 1 ]; then
    # shellcheck disable=SC2086
    gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- \
      "$PY" $WC/$KV_SCRIPT "${LIBS[@]}" --out "$O" ${KV_ARGS:-}
    exit $?
  fi
  # shellcheck disable=SC2086,SC2016
  gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- bash -c '
    n=$1 o=$2; shift 2
    for i in $(seq 1 "$n"); do "$@" --out "$o/rep$i" || exit $?; done' _ "$KV_REPEAT" "$O" \
    "$PY" $WC/$KV_SCRIPT "${LIBS[@]}" ${KV_ARGS:-}
  exit $?
fi
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DWC_KMAX="${WC_KMAX:-8}")
declare -A DEFS=([split]="-DWC_SQR=0" [wide]="-DWC_SQR=1" [dense]="-DWC_SQR=0 -DWC_PB=60"
                [split8]="-DWC_SQR=0 -DWC_G=8" [dense8]="-DWC_SQR=0 -DWC_PB=60 -DWC_G=8"
                [densev0]="-DWC_SQR=0 -DWC_PB=60 -DWC_B3V=0")
declare -A PIDS=()
for lib in $LIBSET; do
  # shellcheck disable=SC2086
  nvcc "${FLAGS[@]}" ${DEFS[$lib]} -o "$B/wc-$lib.so" $WC/wc.cu > "$O/wc-$lib.ptxas" 2>&1 & PIDS[$lib]=$!
done
for lib in $LIBSET; do
  wait "${PIDS[$lib]}" || { tail -30 "$O/wc-$lib.ptxas"; exit 4; }
done
SASS_KEEP='chain_squareILi1E|chain_encodeILi1E|chain_b3ILi1E|decode_kernelILi[12]ELi0E|gemm_pipe_kernelILi[02]ELi1ELb0E|gemm_ws_kernelILi[03]ELi1ELb0ELi[237]E'
for lib in $LIBSET; do
  cuobjdump -sass "$B/wc-$lib.so" | awk -v keep="$SASS_KEEP" '/Function : /{p = ($0 ~ keep)} p' > "$O/wc-$lib.sass" 2>&1
done
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $WC/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
export B PY NCU_BIN WC BENCH_ARGS=${BENCH_ARGS:-} NCU=${NCU:-1} NCU_WS=${NCU_WS:-} NCU_KS=${NCU_KS:-2,3}
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
       --section InstructionStats --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats
       --section SchedulerStats)
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
        --batches 1,32 --ks "${NCU_KS:-2,3}" ${NCU_WS:+--ws "$NCU_WS"} ${NCU_TAG_WS+--tag-ws "$NCU_TAG_WS"} \
        > "$O/ncu-gemm-$lib.log" 2>&1
      "$NCU_BIN" --import "$O/ncu-gemm-$lib.ncu-rep" --csv --page raw > "$O/ncu-gemm-$lib.csv" 2>&1
    done
    q
  fi
  exit $rc'
