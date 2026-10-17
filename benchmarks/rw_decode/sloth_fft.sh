#!/usr/bin/env bash
# sloth_fft.cu on one GPU (cwd = the source tree): the FFT part of a squaring mod 2^1257787 - 1 in FP64 on cuFFT, at
# 2^17 and 2^18 points and several batch sizes. Results in $OUT/sloth.jsonl.
#
#   research run --queue --preemptible --gpus 1 --cpus 4 --mem-gb 16 --max-min 20 --gpu-model rtx-pro-6000 \
#     --project verity --source . --cwd source --campaign stacking-17oct -- bash benchmarks/rw_decode/sloth_fft.sh
set -uxo pipefail
export O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/rw_decode
B=$(mktemp -d /tmp/sloth-fft.XXXXXX); trap 'rm -rf "$B"' EXIT
nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$O/gpu.txt" 2>&1
for l in 17 18; do
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSL_LOG_N=$l -o "$B/sloth$l" $D/sloth_fft.cu -lcufft \
    > "$O/build-$l.log" 2>&1 || { echo "build $l rc $?" >> "$O/errors.txt"; continue; }
  for w in 16 64 256; do
    "$B/sloth$l" "$w" 20 >> "$O/sloth.jsonl" 2>> "$O/errors.txt" || echo "run $l $w rc $?" >> "$O/errors.txt"
  done
done
touch "$O/errors.txt"
