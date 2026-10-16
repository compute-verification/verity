#!/usr/bin/env bash
# HC-PREFILL: the launch resources ncu reads for the prefill GEMM's kernels and dfill_k beside them (shared memory config,
# dynamic and static shared memory, registers, the shared-memory occupancy limit), one dscore prefill co point under ncu.
# NVCC_APPEND_FLAGS picks the filler's build. Run from the repository root (research run --cwd source).
set -uo pipefail
O=${OUT:?}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
export PATH="$(dirname "$NVCC"):$PATH"
B=$(mktemp -d /tmp/hcpre-smem.XXXXXX)
"$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dscore" benchmarks/pouw/filler/dscore.cu \
  -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { echo "build failed"; exit 4; }
git rev-parse HEAD > "$O/commit.txt"
M=launch__shared_mem_config_size,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__shared_mem_per_block_driver,launch__registers_per_thread,launch__block_size,launch__grid_size,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,device__attribute_max_shared_memory_per_multiprocessor
cat > "$B/inner.sh" <<EOF
set -uo pipefail
export DFILL_COINS=0123456789abcdef0123456789abcdef DFILL_CHUNK_LOG2=${CHUNK:-20}
ncu --clock-control none -k regex:"${KREGEX:-cutlass|dfill_k}" --launch-count ${COUNT:-6} --metrics $M --csv \
  "$B/dscore" dscore prefill 8192 0 co 0 ${SHAPE:-4 4 256} 2 1 0 > "$O/ncu.csv" 2> "$O/errors.txt"
echo "ncu rc \$?" >> "$O/rc.txt"
EOF
gpu-lease 1 --wait --max-min "${MAXMIN:-8}" -- bash "$B/inner.sh"
echo "lease rc $?" >> "$O/rc.txt"
rm -rf "$B"
