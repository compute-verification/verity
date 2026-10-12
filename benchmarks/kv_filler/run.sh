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
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK); VLLM=1 runs SCRIPT in the vLLM venv
# (VLLM_PY) instead, with integrations/vllm on the path.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
if [ "${VLLM:-0}" = 1 ]; then
  PY=${VLLM_PY:-/workspace/pouw/mvp-e2e/venv312/bin/python}
  export PYTHONPATH=$PWD:$PWD/integrations/vllm${PYTHONPATH:+:$PYTHONPATH} HF_HOME=${HF_HOME:-/workspace/hf}
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
sha256sum $KF/topup.cu benchmarks/bandwidth/tagger.cu benchmarks/word_codec/kv.cu "$B"/*.so >> "$O/build.txt"
if [ -d $KF/tests ] && "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $KF/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
# shellcheck disable=SC2086
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- \
  "$PY" $KF/${SCRIPT:-bench.py} --topup-lib "$B/libtopup.so" --kv-lib "$B/libkv-p512.so" --out "$O" ${ARGS:-}
