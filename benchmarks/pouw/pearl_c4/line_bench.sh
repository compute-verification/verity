#!/usr/bin/env bash
# line_bench.py's run on one leased GPU of a node: pearl_c4.cu built at LINE_ARM 0, 1 and 2 with CUDA 12.9 (build.sh's flags,
# against pouw_hash.cuh at build.sh's pinned commit, given as $RESEARCH_RUN_DIR/inputs/pouw_hash.cuh or $HASH_H and checked
# against build.sh's sha256), each held to sass.py, then line_bench.py under `gpu-lease 1`.  Outputs in $OUT.
#   research run --on NODE --source TREE --cwd source --send pouw_hash.cuh -- bash benchmarks/pouw/pearl_c4/line_bench.sh [ARGS]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
C9=${C9:-/workspace/jobs/cuda-12.9}; OUT=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}; mkdir -p "$OUT"
HASH_H=${HASH_H:-$RESEARCH_RUN_DIR/inputs/pouw_hash.cuh}
SHA=$(grep '^HASH_SHA256=' "$HERE/build.sh" | cut -d= -f2)
B=$(mktemp -d); trap 'rm -rf "$B"' EXIT
mkdir -p "$B/inc"; cp "$HASH_H" "$B/inc/pouw_hash.cuh"
echo "$SHA  $B/inc/pouw_hash.cuh" | sha256sum -c --quiet
FLAGS="-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin"
ARMS=()
for a in 0 1 2; do
  name=$([ $a = 0 ] && echo ts128 || ([ $a = 1 ] && echo sha512-1 || echo sha512-4))
  "$C9/bin/nvcc" $FLAGS -DLINE_ARM=$a -I "$B/inc" -o "$B/arm$a.cubin" "$HERE/pearl_c4.cu" 2> "$OUT/ptxas-$name.txt"
  "$C9/bin/cuobjdump" -sass "$B/arm$a.cubin" > "$B/arm$a.sass"
  python3 "$HERE/sass.py" "$OUT/ptxas-$name.txt" "$B/arm$a.sass" "$OUT/sass-$name.json"
  sha256sum "$B/arm$a.cubin" | sed "s|$B/||" >> "$OUT/cubins.sha256"
  ARMS+=(--arm "$name=$B/arm$a.cubin")
done
{ "$C9/bin/nvcc" --version | tail -2; echo "nvcc $FLAGS -DLINE_ARM=0|1|2"; } > "$OUT/toolchain.txt"
PY=${PY:-python3}
if ! $PY -c "import numpy" 2> /dev/null; then PY="uv run --no-project --with numpy python3"; fi
GPU_LEASE_WHO=${GPU_LEASE_WHO:-pouw-c4-line-sha512} gpu-lease 1 --wait --max-min 25 -- \
  bash -c "$PY $HERE/line_bench.py $OUT/line_bench.json ${ARMS[*]} --expect-uuid \$GPU_LEASE_UUID $*"
