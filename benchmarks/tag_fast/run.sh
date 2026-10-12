#!/usr/bin/env bash
# The certified tag fused into the FP8 decode GEMM on one GPU of a node, as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-tag-fast \
#     -- taskset -c 48-95 bash benchmarks/tag_fast/run.sh
#
# Outside the lease: build tf.cu (its ptxas report and the hot kernels' SASS kept), the CPU tests, and reference.py's
# expected tags, leaves and roots (W and the tags in scratch, expect.json in out; kept on the node under REF_CACHE, keyed
# by the reference's sources, and reused by later runs, whose reference.log names the run that wrote it). Under one
# gpu-lease 1 --wait --max-min
# MAXMIN (default 25): bench.py's gates and timings, then (NCU=1, the default) Nsight Compute on a few fused forms.
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK); BENCH_ARGS passes through to bench.py, NCU_ARGS to
# its profiling pass (default: gate_up and o at batches 1 and 32).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
TF=benchmarks/tag_fast
B=$(mktemp -d /tmp/tag-fast.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
{ nvcc --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
nvcc "${FLAGS[@]}" -o "$B/libtf.so" $TF/tf.cu > "$O/tf.ptxas" 2>&1 || { tail -30 "$O/tf.ptxas"; exit 4; }
cuobjdump -sass "$B/libtf.so" \
  | awk '/Function : /{p = ($0 ~ /chain_b3|chain_keccak|fused_kernelILi[013]ELi32ELi1ELi1ELi1ELb0|fused2_kernelILi1ELi4ELi1ELi1ELi4ELi2ELb0/)} p' > "$O/tf.sass" 2>&1
sha256sum $TF/tf.cu "$B/libtf.so" >> "$O/build.txt"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $TF/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
KEY=$(cat $TF/reference.py $TF/layout.py verity/core/protocols/bandwidth/tags.py verity/ml/crypto/blake3.py | sha256sum \
  | cut -c1-16)
REF=${REF_CACHE:-$HOME/.cache/tag-fast-ref}/$KEY
if [ -s "$REF/expect.json" ]; then
  echo "reference $KEY from the node's cache, written by run $(cat "$REF/run")" > "$O/reference.log"
else
  "$PY" $TF/reference.py --out "$B/ref" --workers "${WORKERS:-44}" > "$O/reference.log" 2>&1 \
    || { tail -30 "$O/reference.log"; exit 6; }
  echo "${RESEARCH_RUN_ID:-?}" > "$B/ref/run"
  mkdir -p "$(dirname "$REF")" && cp -r "$B/ref" "$REF.$$" && mv -T "$REF.$$" "$REF" 2> /dev/null || rm -rf "$REF.$$"
  REF=$B/ref
fi
cp "$REF/expect.json" "$O/expect.json"

export B REF PY NCU_BIN TF BENCH_ARGS=${BENCH_ARGS:-} NCU=${NCU:-1} NCU_ARGS=${NCU_ARGS:---shapes gate_up,o --batches 1,32}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  # shellcheck disable=SC2086
  "$PY" $TF/bench.py --lib "$B/libtf.so" --ref "$REF" --out "$O" $BENCH_ARGS; rc=$?
  q
  if [ "$NCU" = 1 ] && [ $rc = 0 ]; then
    S=(--clock-control none --cache-control all --section SpeedOfLight --section ComputeWorkloadAnalysis
       --section InstructionStats --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats)
    # shellcheck disable=SC2086
    "$NCU_BIN" "${S[@]}" -k "regex:fused_kernel|fused2_kernel|plain_kernel" -f -o "$O/ncu-fused" \
      "$PY" $TF/bench.py --lib "$B/libtf.so" --ref "$REF" --out "$O/ncu" --mode ncu $NCU_ARGS > "$O/ncu.log" 2>&1
    "$NCU_BIN" --import "$O/ncu-fused.ncu-rep" --csv --page raw > "$O/ncu-fused.csv" 2>&1
    q
  fi
  exit $rc'
