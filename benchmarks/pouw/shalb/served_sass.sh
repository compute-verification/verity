#!/usr/bin/env bash
# The served GEMM (np_kernel's ssn.cu) built to sm_90a SASS and its tensor-core instructions read by served_sass.py:
# in place (D == C) or fresh. CPU only: nothing is launched. cwd: a source tree holding benchmarks/pouw/np_kernel at
# the served kernel's commit; SASS_PY (default: this directory's served_sass.py) is the reader. The CUDA wheels go into
# a scratch venv outside the run dir. Output: $OUT/{ssn.sass, served.json, served.txt, tools.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
SASS_PY=${SASS_PY:-$(dirname "$0")/served_sass.py}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
python3 -m venv "$d/v"
"$d/v/bin/pip" install -q nvidia-cuda-nvcc nvidia-cuda-crt nvidia-cuda-runtime nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm \
  nvidia-cuda-cccl
CU=$(ls -d "$d"/v/lib/python3*/site-packages/nvidia/cu13)
export PATH="$CU/bin:$CU/nvvm/bin:$PATH"
{ nvcc --version; cuobjdump --version; git rev-parse HEAD 2>/dev/null || true; } > "$OUT/tools.txt"
NK=benchmarks/pouw/np_kernel
INC=(-I"$CU/include"); [ -d "$CU/include/cccl" ] && INC+=(-I"$CU/include/cccl")
sha256sum "$NK/ssn.cu" >> "$OUT/tools.txt"
nvcc -O3 -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a "${INC[@]}" \
  -I integrations/vllm/verity_vllm/commit/committer -o "$d/ssn.cubin" "$NK/ssn.cu"
cuobjdump -sass "$d/ssn.cubin" > "$OUT/ssn.sass"
python3 "$SASS_PY" "$OUT/ssn.sass" --json "$OUT/served.json" | tee "$OUT/served.txt"
