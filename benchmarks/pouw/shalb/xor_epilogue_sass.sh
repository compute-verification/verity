#!/usr/bin/env bash
# Honest's xor-m and xorrot-m epilogue on sm_90a: xor_epilogue_probe.cu (fresh-output IMMA) and xor_wgmma_probe.cu
# (the served form's in-place IGMMA) built to SASS and counted by
# xor_epilogue_sass.py against the count hash_sha.py prices. CPU only: nothing is launched. The CUDA wheels go into a
# scratch venv outside the run dir. cwd: the source tree's root. Output: $OUT/{probe.sass, epilogue.json,
# epilogue.txt, tools.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
python3 -m venv "$d/v"
"$d/v/bin/pip" install -q nvidia-cuda-nvcc nvidia-cuda-crt nvidia-cuda-runtime nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm
CU=$(ls -d "$d"/v/lib/python3*/site-packages/nvidia/cu13)
export PATH="$CU/bin:$CU/nvvm/bin:$PATH"
{ nvcc --version; cuobjdump --version; } > "$OUT/tools.txt"
HERE=benchmarks/pouw/shalb
nvcc -O3 -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a -I"$CU/include" -o "$d/probe.cubin" \
  "$HERE/xor_epilogue_probe.cu"
nvcc -O3 -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a -I"$CU/include" -o "$d/wprobe.cubin" \
  "$HERE/xor_wgmma_probe.cu"
{ cuobjdump -sass "$d/probe.cubin"; cuobjdump -sass "$d/wprobe.cubin"; } > "$OUT/probe.sass"
python3 "$HERE/xor_epilogue_sass.py" "$OUT/probe.sass" --json "$OUT/epilogue.json" | tee "$OUT/epilogue.txt"
