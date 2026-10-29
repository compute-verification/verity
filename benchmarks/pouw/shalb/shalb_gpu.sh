#!/usr/bin/env bash
# The minimal compression on one GPU (sha512_min_gpu.cu): its device results against the host's, the constant-chaining-
# value, padding-block and chain forms too, and its time a compression beside the committer's (SHA_ROLL 1, as served,
# and 0).
# Runs under gpu-lease on its GPU node: research run --on NODE -- gpu-lease 1 --wait -- bash benchmarks/pouw/shalb/shalb_gpu.sh
# The node's CUDA toolkit if it has one, else the CUDA wheels in a scratch venv outside the run dir. cwd: the source
# tree's root. Output: $OUT/{gpu_roll1.txt, gpu_roll0.txt, probe.txt, probe_counts.json, tools.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH=/usr/local/cuda/bin:$PATH
  INC=() LIB=()
else
  python3 -m venv "$d/v"
  "$d/v/bin/pip" install -q nvidia-cuda-nvcc nvidia-cuda-crt nvidia-cuda-runtime
  CU=$(ls -d "$d"/v/lib/python3*/site-packages/nvidia/cu13)
  export PATH="$CU/bin:$CU/nvvm/bin:$PATH"
  INC=(-I"$CU/include") LIB=(-L"$CU/lib")
fi
{ nvcc --version; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv; } > "$OUT/tools.txt"
SRC=integrations/vllm/verity_vllm/commit/committer
HERE=benchmarks/pouw/shalb
for chain in 0 1; do
  nvcc -O3 --use_fast_math -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a "${INC[@]}" -DSHA_CHAIN=$chain \
    "$HERE/sha512_min_probe.cu" -o "$d/probe$chain.cubin"
  cuobjdump -sass "$d/probe$chain.cubin" > "$d/probe$chain.sass"
done
python3 "$HERE/shalb_sass.py" "$OUT/probe_counts.json" "min_variable=$d/probe0.sass" "min_chain=$d/probe1.sass:2" |
  cut -c1-200 | tee "$OUT/probe.txt"
for roll in 1 0; do
  nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90a -DSHA_ROLL=$roll -I"$SRC" "${INC[@]}" "${LIB[@]}" \
    "$HERE/sha512_min_gpu.cu" -o "$d/gpu$roll"
  "$d/gpu$roll" | tee "$OUT/gpu_roll$roll.txt"
done
