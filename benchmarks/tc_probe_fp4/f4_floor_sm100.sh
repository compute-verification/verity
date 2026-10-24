#!/usr/bin/env bash
# Lane B200-FP8: the B200's (sm_100a) kind::mxf4nvf4 .scale_vec::2X (UE8M0) adder floor.  probe.py scores the pinned model
# (grid_floor -174) beside floors -170, -173, -175 and -178 on the families whose sums sit below the subnormal grid; nvf4's UE4M3
# unit weights are >= 2^-20, so its floor never binds and is not probed.  The build runs outside the lease (probe.py's nvcc
# line), the capture and its scoring under one gpu-lease.
#
#   research run --on vy-dc-b200 --project verity --campaign eps1 --source <tree> --cwd source --env GPU_LEASE_WHO=b200-fp8 \
#     -- bash benchmarks/tc_probe_fp4/f4_floor_sm100.sh [N_RANDOM]
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(cd "$HERE/../.." && pwd)" PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "f4_floor_sm100.sh: no nvcc" >&2; exit 2; }
export PATH="$(dirname "$NVCC"):$PATH"
N=${1:-16384}
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
"$NVCC" -O2 -gencode=arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -o "$BUILD/libtcgen05_fp4_sm_100a.so" \
  "$HERE/tcgen05_fp4.cu" >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$BUILD/libtcgen05_fp4_sm_100a.so" | grep -oE "UTC[A-Z]*MMA[A-Z0-9_.]*" | sort | uniq -c | tee "$OUT/mma-opcodes.txt"
lease=()
command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
"${lease[@]}" python3 "$HERE/probe.py" --instruction tcgen05_mxf4 --arch sm_100a --out "$OUT" \
  --prebuilt "$BUILD/libtcgen05_fp4_sm_100a.so" --sweep --n-random "$N" \
  --families acc_subnormal,ue8m0_scale_extremes,ue8m0_underflow,random_random_scales
