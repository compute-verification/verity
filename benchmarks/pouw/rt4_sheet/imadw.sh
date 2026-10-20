#!/usr/bin/env bash
# imadw.cu for the H200's, B200's and RTX PRO 6000's arch (sm_90a, sm_100a, sm_120a): SASS of each arch
# (sass/imadw_<arch>.sass) and, in each li_loop body, the multiply-adds (IMAD.WIDE, IMAD), MOVs and reductions it
# issues (ops.txt), for red_price.py; with RT4_DEVICE=1, the SASS gate and, under one gpu-lease, li_exact<CP, GEN, W> on
# the node's device (device.txt). From a pushed commit, on a node with CUDA 13.0:
#
#   research run --on vy-nebius-2 --project verity --campaign eps1 --source <tree> --cwd source --mem-gb 4 \
#     --timeout 1500 [--env RT4_DEVICE=1 --env GPU_LEASE_WHO=<you>] -- bash benchmarks/pouw/rt4_sheet/imadw.sh
#
# Exit: 0 done, 1 a build failed, 2 no nvcc, 3 not bit-exact, 4 the SASS gate refused the binary.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "imadw.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
git -C "$HERE" rev-parse HEAD >"$OUT/tree.sha" 2>/dev/null
for arch in sm_90a sm_100a sm_120a; do
  "$NVCC" -O3 -std=c++17 -cubin -arch="$arch" -Xptxas -v "$HERE/imadw.cu" -o "$OUT/imadw_$arch.cubin" \
    >"$OUT/build_$arch.log" 2>&1 || { tail -20 "$OUT/build_$arch.log" >&2; exit 1; }
  "$(dirname "$NVCC")/cuobjdump" -sass "$OUT/imadw_$arch.cubin" >"$OUT/sass/imadw_$arch.sass" 2>&1
  awk -v arch="$arch" '
    /Function :/ { name = $3; loop = (name ~ /li_loop/) }
    loop && match($0, /\*\/[ \t]+(@!?U?P[T0-9]+[ \t]+)?[A-Z][A-Z0-9_.]*/) {
      op = substr($0, RSTART, RLENGTH); sub(/^\*\/[ \t]+(@!?U?P[T0-9]+[ \t]+)?/, "", op)
      if (op ~ /^(IMAD|MOV|RED|REDG|ATOM|LOP3)/) n[name " " op]++
    }
    END { for (k in n) print arch, k, n[k] }' "$OUT/sass/imadw_$arch.sass" | sort
done | tee "$OUT/ops.txt"
[ "${RT4_DEVICE:-0}" = 1 ] || exit 0
BIN="$OUT/bin/imadw"
"$NVCC" -O3 -std=c++17 --no-compress -gencode arch=compute_90a,code=sm_90a -gencode arch=compute_100a,code=sm_100a \
  -gencode arch=compute_120a,code=sm_120a "$HERE/imadw.cu" -o "$BIN" >"$OUT/build.log" 2>&1 \
  || { cat "$OUT/build.log" >&2; exit 1; }
sha256sum "$BIN" >"$OUT/bin/imadw.sha256"
PATH="$(dirname "$NVCC"):$PATH" python3 "$HERE/../harness/sass_gate.py" "$BIN" --json "$OUT/sass_gate.json" \
  >"$OUT/sass_gate.txt" 2>&1 || { cat "$OUT/sass_gate.txt" >&2; exit 4; }
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
gpu-lease 1 --wait --max-min 10 -- bash -c 'nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv -i "${GPU_LEASE_UUID:-0}" >"$0/env/leased.csv" 2>&1; "$1" | tee "$0/device.txt"; exit "${PIPESTATUS[0]}"' "$OUT" "$BIN"
