#!/usr/bin/env bash
# M_key·y's instructions a salt on sm_90a, from the committer's own native_rows.cu: builds mask_probe.cu (MASK_UNROLL 0 and 1),
# disassembles it and prices it (mask_sass.py). CPU only: nothing is launched. The CUDA wheels go into a scratch venv outside
# the run dir. cwd: the source tree's root. Output: $OUT/{mask_u*.sass, mask_u*.ptxas, mask_sass.json, summary.txt, tools.txt}.
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
for u in 0 1; do
  nvcc -O3 --use_fast_math -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -DMASK_UNROLL=$u \
    -I"$SRC" -I"$CU/include" "$HERE/mask_probe.cu" -o "$d/m.cubin" 2> "$OUT/mask_u$u.ptxas"
  cuobjdump -sass "$d/m.cubin" > "$OUT/mask_u$u.sass"
done
python3 "$HERE/mask_sass.py" "$OUT/mask_u0.sass" "$OUT/mask_u1.sass" "$OUT/mask_sass.json" | tee "$OUT/summary.txt"
