#!/usr/bin/env bash
# The words committer (sha512_words.cuh) on one GPU: its registers a compression from SASS (sha512_words_probe.cu at
# WORDS_NB 8, 16 and 32, extrapolated in pairs to the 16 KiB chunk's 128), and sha512_words_gpu.cu's leaves and nodes
# held to verity.ml.commitments bit for bit (shalb_words_check.py) with its time beside the served committer's
# (SHA_ROLL 1, as served, and 0).
# Runs under gpu-lease on its GPU node: research run --on NODE -- gpu-lease 1 --wait -- bash benchmarks/pouw/shalb/shalb_words_gpu.sh
# The node's CUDA toolkit if it has one, else the CUDA wheels in a scratch venv outside the run dir. cwd: the source
# tree's root. Output: $OUT/{words*.sass, counts.json, counts.txt, gpu_roll{1,0}.txt, check_roll{1,0}.txt,
# words_roll{1,0}.bin, tools.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH=/usr/local/cuda/bin:$PATH
  INC=() LIB=()
else
  python3 -m venv "$d/v"
  "$d/v/bin/pip" install -q nvidia-cuda-nvcc nvidia-cuda-crt nvidia-cuda-runtime nvidia-cuda-cuobjdump \
    nvidia-cuda-nvdisasm
  CU=$(ls -d "$d"/v/lib/python3*/site-packages/nvidia/cu13)
  export PATH="$CU/bin:$CU/nvvm/bin:$PATH"
  INC=(-I"$CU/include") LIB=(-L"$CU/lib")
fi
{ nvcc --version; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv; } > "$OUT/tools.txt"
SRC=integrations/vllm/verity_vllm/commit/committer
HERE=benchmarks/pouw/shalb
counts=()
for nb in 8 16 32; do
  nvcc -O3 --use_fast_math -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a "${INC[@]}" -diag-suppress 177 \
    -DWORDS_NB=$nb "$HERE/sha512_words_probe.cu" -o "$d/words$nb.cubin"
  cuobjdump -sass "$d/words$nb.cubin" > "$OUT/words$nb.sass"
  counts+=("words$nb=$OUT/words$nb.sass:$((nb + 5))")
done
python3 "$HERE/shalb_sass.py" "$OUT/counts.json" "${counts[@]}" | cut -c1-200
python3 - "$OUT/counts.json" <<'EOF' | tee "$OUT/counts.txt"
import json, sys
c = {k: v["sha512_sass"]["registers_written"] for k, v in json.load(open(sys.argv[1])).items()}
pair = (c["words32"] - c["words16"]) / 8
item = c["words32"] + pair * (128 - 32) / 2
print(f"registers a trip: {c}; a pair of data blocks, 16 to 32: {pair:g}, 8 to 16: {(c['words16'] - c['words8']) / 4:g}")
print(f"16 KiB item (128 data blocks, padding, leaf 2, node 2 = 133 compressions): {item:.0f}, {item / 133:.2f} a compression")
EOF
python3 "$HERE/shalb_words_check.py" domain "$d/did"
for roll in 1 0; do
  nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90a -DSHA_ROLL=$roll -I"$SRC" "${INC[@]}" "${LIB[@]}" \
    "$HERE/sha512_words_gpu.cu" -o "$d/gpu$roll"
  "$d/gpu$roll" "$d/did" "$OUT/words_roll$roll.bin" | tee "$OUT/gpu_roll$roll.txt"
  python3 "$HERE/shalb_words_check.py" check "$d/did" "$OUT/words_roll$roll.bin" | tee "$OUT/check_roll$roll.txt"
done
