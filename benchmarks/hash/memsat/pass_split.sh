#!/usr/bin/env bash
# The receipts pass, part by part, on one idle GPU of the lease (cwd = the source tree): memory accounting's
# benchmarks/remote_session/pass_bench.py over a PASS_GIB GiB store (default 9, one int8 Llama-3.1-8B arena) at node.py's
# pass (2^18-block chunks, 128 receipt CTAs): the read alone, then the sketch, its SHA-512 receipts, the leaves and the
# tree kernel by kernel, the tensor-core receipts alone, and the pipelined pass. Writes $RESEARCH_RUN_DIR/out/pass_bench.json.
set -euo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
PY=${PY:-/workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$PATH
B=$(mktemp -d /tmp/memsat-pass.XXXXXX); trap 'rm -rf "$B"' EXIT
nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader -i "${GPU_LEASE_UUID:-${CUDA_VISIBLE_DEVICES:?}}" >> "$O/pass_bench.gpu.txt" 2>&1
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared \
  -o "$B/session.so" benchmarks/remote_session/session.cu
"$PY" benchmarks/remote_session/pass_bench.py --lib "$B/session.so" --gib "${PASS_GIB:-9}" --out "$O" --chunks 18 \
  --ctas 128 --passes 30 --open-passes 50 --openings 2773 --contend '' > "$O/pass_bench.log" 2>&1
