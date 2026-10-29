#!/usr/bin/env bash
# SHA-512's instructions a compression on sm_90a, from the committer's own sha512.cuh: builds sha512_probe.cu (SHA_ROLL
# 0 and 1, SHA_ADD 0 and 1) and native_rows.cu as the committer builds it, disassembles them and prices them
# (sha512_sass.py). CPU only: nothing is launched. The CUDA wheels go into a scratch venv outside the run dir.
# cwd: the source tree's root. Output: $OUT/{probe_r*_a*.json, probe_r*_a*.sass, native_rows.json, native_rows.sass.gz, tools.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
python3 -m venv "$d/v"
"$d/v/bin/pip" install -q nvidia-cuda-nvcc nvidia-cuda-crt nvidia-cuda-runtime nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm
CU=$(ls -d "$d"/v/lib/python3*/site-packages/nvidia/cu13)
export PATH="$CU/bin:$CU/nvvm/bin:$PATH"
{ nvcc --version; cuobjdump --version; } > "$OUT/tools.txt"
SRC=integrations/vllm/verity_vllm/commit/committer
HERE=benchmarks/served_commit
for roll in 0 1; do
  for add in 0 1; do
    nvcc -O3 --use_fast_math -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a -Xptxas -v \
      -DSHA_ROLL=$roll -DSHA_ADD=$add -I"$SRC" -I"$CU/include" "$HERE/sha512_probe.cu" -o "$d/p.cubin" 2> "$OUT/probe_r${roll}_a${add}.ptxas"
    cuobjdump -sass "$d/p.cubin" > "$OUT/probe_r${roll}_a${add}.sass"
  done
done
nvcc -O3 --use_fast_math -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -DVERITY_ROWS_MIN_BLOCKS=2 \
  -I"$SRC" -I"$CU/include" "$SRC/native_rows.cu" -o "$d/rows.cubin" 2> "$OUT/native_rows.ptxas"
cuobjdump -sass "$d/rows.cubin" > "$d/rows.sass"
for roll in 0 1; do
  for add in 0 1; do
    python3 "$HERE/sha512_sass.py" "$OUT/probe_r${roll}_a${add}.sass" "$OUT/probe_r${roll}_a${add}.json" | tee -a "$OUT/summary.txt"
  done
done
python3 "$HERE/sha512_sass.py" "$OUT/probe_r1_a0.sass" "$d/rows.sass" "$OUT/native_rows.json" > /dev/null
gzip -c "$d/rows.sass" > "$OUT/native_rows.sass.gz"
