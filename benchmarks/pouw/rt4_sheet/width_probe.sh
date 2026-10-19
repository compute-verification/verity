#!/usr/bin/env bash
# RT4-SHEET job 3: which global memory widths sm_120a emits in one warp instruction (width_probe.cu), build and SASS
# only, no device run and so no gpu-lease. Two builds: the default (a 256-bit transfer as two 128-bit ops) and RT4_V8 (a
# true v8.b32, 256-bit). Records whether ptxas accepts v8 and each probe kernel's global store/load opcodes.
#
#   research run --on vy-nebius-2 --project verity --campaign eps1 --source <tree> --cwd source --mem-gb 4 \
#     --timeout 900 -- bash benchmarks/pouw/rt4_sheet/width_probe.sh
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "width_probe.sh: no nvcc" >&2; exit 2; }
"$NVCC" --version >"$OUT/nvcc.txt" 2>&1
CUOBJ="$(dirname "$NVCC")/cuobjdump"
for v in default v8; do
  flag=""; [ "$v" = v8 ] && flag="-DRT4_V8"
  bin="$OUT/wp_$v"
  if "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v $flag "$HERE/width_probe.cu" \
      -o "$bin" >"$OUT/build_$v.log" 2>&1; then
    echo "build_$v=ok"
    "$CUOBJ" -sass "$bin" >"$OUT/sass_$v.txt" 2>&1
    # the global stores/loads each probe issues, by opcode
    grep -oE "\b(STG|LDG|ST|LD)\.[A-Z0-9_.]+" "$OUT/sass_$v.txt" | sort | uniq -c >"$OUT/mem_ops_$v.txt"
    echo "--- $v mem ops"; cat "$OUT/mem_ops_$v.txt"
  else
    echo "build_$v=FAILED"; tail -8 "$OUT/build_$v.log"
  fi
done
exit 0
