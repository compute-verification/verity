#!/usr/bin/env bash
# The CPU job: uni64.cu built for sm_90a, its SASS and the count of each uniform-datapath opcode. No GPU:
#
#   research run --on auto --project verity --campaign eps1 --cwd source --wait -- bash benchmarks/pouw/rt_hopper/uni64.sh
#
# Outputs in $OUT: nvcc.txt, uni64.sass, uniform.txt (count, opcode). Exit: 0 done, 1 build failed, 2 no nvcc.
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-$(ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1)}
[ -x "${NVCC:-}" ] || NVCC=$(command -v nvcc) || { echo "uni64.sh: no nvcc" >&2; exit 2; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" --version >"$OUT/nvcc.txt"
"$NVCC" -arch=sm_90a -cubin -O3 -o "$WORK/uni64.cubin" "$HERE/uni64.cu" || exit 1
"$(dirname "$NVCC")/cuobjdump" -sass "$WORK/uni64.cubin" >"$OUT/uni64.sass"
grep -oE '\bU[A-Z0-9]+(\.[A-Z0-9]+)*' "$OUT/uni64.sass" | sort | uniq -c | sort -rn >"$OUT/uniform.txt"
cat "$OUT/uniform.txt"
