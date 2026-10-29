#!/usr/bin/env bash
# SHA-512's registers written a compression on sm_90a: the committer's sha512.cuh (sha512_probe.cu, SHA_ROLL 0 and 1)
# against the minimal compression (sha512_min_probe.cu: a variable chaining value, the IV as a constant, a constant
# padding block, and a chain in chain form, two blocks a trip), an L1 leaf (sha512_l1_probe.cu: ten blocks a trip),
# what an L1 call computes once for its leaves (sha512_l1_call_probe.cu), a bound row-seg segment
# (sha512_rowseg_probe.cu: ten blocks a trip, and its block 1 alone) and a 1 KiB row/v2 chunk after its hoisted prefix
# (sha512_rowv2_probe.cu: nine blocks a trip), each built, disassembled and counted by shalb_sass.py; then the minimal
# compression's, the L1 leaf's, the segment's and the chunk's host builds checked against hashlib. CPU only: nothing is launched. The CUDA wheels go into a scratch venv outside the run dir.
# cwd: the source tree's root. Output: $OUT/{*.sass, *.ptxas, counts.json, summary.txt, check_*.txt, tools.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
python3 -m venv "$d/v"
"$d/v/bin/pip" install -q nvidia-cuda-nvcc nvidia-cuda-crt nvidia-cuda-runtime nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm
CU=$(ls -d "$d"/v/lib/python3*/site-packages/nvidia/cu13)
export PATH="$CU/bin:$CU/nvvm/bin:$PATH"
{ nvcc --version; cuobjdump --version; g++ --version | head -1; } > "$OUT/tools.txt"
SRC=integrations/vllm/verity_vllm/commit/committer
HERE=benchmarks/pouw/shalb
NVCC=(nvcc -O3 --use_fast_math -std=c++17 -cubin -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -I"$CU/include")
build() {  # NAME SOURCE FLAGS...
  local name=$1 src=$2; shift 2
  "${NVCC[@]}" "$@" "$src" -o "$d/$name.cubin" 2> "$OUT/$name.ptxas"
  cuobjdump -sass "$d/$name.cubin" > "$OUT/$name.sass"
}
build committer_straight "$HERE/sha512_probe.cu" -DSHA_ROLL=0 -DSHA_ADD=0 -I"$SRC"
build committer_rolled "$HERE/sha512_probe.cu" -DSHA_ROLL=1 -DSHA_ADD=0 -I"$SRC"
build min_variable "$HERE/sha512_min_probe.cu" -DSHA_CVC=0
build min_constant_cv "$HERE/sha512_min_probe.cu" -DSHA_CVC=1
build min_padding_block "$HERE/sha512_min_probe.cu" -DSHA_PAD=2048
build min_chain "$HERE/sha512_min_probe.cu" -DSHA_CHAIN=1
build l1_leaf "$HERE/sha512_l1_probe.cu"
build l1_call "$HERE/sha512_l1_call_probe.cu"
build rowseg_segment "$HERE/sha512_rowseg_probe.cu"
build rowseg_block1 "$HERE/sha512_rowseg_probe.cu" -DSEG_BLOCK1=1
build rowv2_chunk "$HERE/sha512_rowv2_probe.cu"
args=()
for n in committer_straight committer_rolled min_variable min_constant_cv min_padding_block l1_call rowseg_block1; do
  args+=("$n=$OUT/$n.sass")
done
args+=("min_chain=$OUT/min_chain.sass:2" "l1_leaf=$OUT/l1_leaf.sass:10" "rowseg_segment=$OUT/rowseg_segment.sass:10"
      "rowv2_chunk=$OUT/rowv2_chunk.sass:9")
python3 "$HERE/shalb_sass.py" "$OUT/counts.json" "${args[@]}" | tee "$OUT/summary.txt"
for c in min l1 rowseg rowv2; do
  g++ -O2 -std=c++17 -x c++ -D'__host__=' -D'__device__=' -D'__forceinline__=inline' "$HERE/sha512_${c}_check.cu" -o "$d/$c"
  "$d/$c" > "$OUT/check_$c.txt"
  python3 "$HERE/sha512_check.py" < "$OUT/check_$c.txt" | tee -a "$OUT/summary.txt"
done
