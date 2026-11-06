#!/usr/bin/env bash
# The KV cache's codec on one GPU (cwd = the source tree): kv.cu built at each width of KV_FIELDS (default 512 768; a
# width tagged k1 squares by one level of Karatsuba, o1 builds at ptxas -O1, h32 at 32 query heads (8B's attention), t16
# with the tensor-core attention's tiles of 16 tokens, p0 without the occupancy form's L2 prefetch, m with σ_K between
# rounds (kv.py's masked word), f6 squaring by the squaring ceiling's FOLD 6, as in 1536k1o1, 512h32, 512m or
# 1024h32mf6), the tests, then, under one gpu-lease 1 --wait --max-min MAXMIN (default 25), KV_SCRIPT (default
# kv_bench.py) with KV_ARGS. Results go to $RESEARCH_RUN_DIR/out.
#
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=ra-word-codec taskset -c 48-95 bash benchmarks/word_codec/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). SANITIZE=1 runs KV_SCRIPT's --sanitize smoke
# under compute-sanitizer's memcheck, racecheck and synccheck, and SANITIZE=2 runs KV_SCRIPT after it if every tool
# passed. KV_NCU=1 profiles KV_SCRIPT --ncu instead of timing it.
# Under the queue, kv.cu's gates alone (the bound key's among them):
#   RESEARCH_RUN_DIR=DIR QUEUED=1 KV_FIELDS='512 1024m' KV_ARGS='--gates-only --ks 2,3' bash \
#     benchmarks/word_codec/run.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
WC=benchmarks/word_codec
# the venv doesn't install verity; the tests import it (stdlib only) from the source tree
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
B=$(mktemp -d /tmp/word-codec.XXXXXX); trap 'rm -rf "$B"' EXIT
LIBS=()
for f in ${KV_FIELDS:-512 768}; do
  n=${f%%[a-z]*} tag=${f#"${f%%[a-z]*}"} flags=()
  lib=p$n${tag:+-$tag}
  [[ $tag == *k1* ]] && flags+=(-DKV_KARATSUBA=1)
  [[ $tag == *o1* ]] && flags+=(-Xptxas -O1)
  [[ $tag == *h32* ]] && flags+=(-DKV_QHEADS=32)
  [[ $tag == *t16* ]] && flags+=(-DKV_MT=16)
  [[ $tag == *p0* ]] && flags+=(-DKV_OCC_PF=0)
  [[ $tag == *m* ]] && flags+=(-DKV_MASK=1)
  [[ $tag == *f6* ]] && flags+=(-DKV_FOLD6=1)
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
      --section SourceCounters -k "regex:${KV_NCU_K:-attn_mma_ws}" -f -o "$o/ncu-kv" "$@" --out "$o" --ncu \
      > "$o/ncu-kv.log" 2>&1 || { tail -20 "$o/ncu-kv.log"; exit 7; }
    "$ncu" --import "$o/ncu-kv.ncu-rep" --csv --page raw > "$o/ncu-kv.csv" 2>&1
    "$ncu" --import "$o/ncu-kv.ncu-rep" --csv --page source --print-source sass > "$o/ncu-kv-sass.csv" 2>&1
    exit 0' _ "$NCU_BIN" "$O" "$PY" $WC/$KV_SCRIPT "${LIBS[@]}" ${KV_ARGS:-}
  exit $?
fi
# KV_REPEAT=N (default 1) runs KV_SCRIPT N times in one lease, each its own process, writing under $O/repI; with
# QUEUED=1 (a queue that leases the GPU around the whole job) gpu-lease is skipped
if [ "${KV_REPEAT:-1}" = 1 ]; then
  LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
  [ "${QUEUED:-0}" = 1 ] && LEASE=()
  # shellcheck disable=SC2086
  "${LEASE[@]}" "$PY" $WC/$KV_SCRIPT "${LIBS[@]}" --out "$O" ${KV_ARGS:-}
  exit $?
fi
# shellcheck disable=SC2086,SC2016
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- bash -c '
  n=$1 o=$2; shift 2
  for i in $(seq 1 "$n"); do "$@" --out "$o/rep$i" || exit $?; done' _ "$KV_REPEAT" "$O" \
  "$PY" $WC/$KV_SCRIPT "${LIBS[@]}" ${KV_ARGS:-}
exit $?
