#!/usr/bin/env bash
# CP-W's checkpoint routes on the sm_120 unit (cpw_routes.cu), and mode 8's fold with its XOR in memory: each route's
# SASS on sm_120a, its timed loop's opcodes counted (counts.json), then on the device: every route bit-exact against the
# host's reference, and its rate at 1, 2, 4 warps a SMSP and full occupancy. From a pushed commit:
#
#   research run --on MACHINE --project verity --source <tree> --cwd source --mem-gb 4 \
#     --timeout 40m --env GPU_LEASE_WHO=<you> -- bash benchmarks/pouw/theory/cpw_routes.sh
#
# The build and the count run outside the lease; the device part runs under one gpu-lease (device.txt, env/gpu.csv).
# Exit: 0 done, 1 build failed, 2 no nvcc, 3 a route not bit-exact.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "cpw_routes.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
BIN="$OUT/bin/cpw_routes"
"$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v "$HERE/cpw_routes.cu" -o "$BIN" \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
sha256sum "$BIN" >"$OUT/bin/cpw_routes.sha256"
"$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/cpw_routes.sass" 2>&1
# each mode's timed loop (the backward branch between its two clock reads): its opcodes, guard predicates dropped
python3 - "$OUT/sass/cpw_routes.sass" >"$OUT/sass/counts.json" <<'EOF'
import collections, json, re, sys
parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + open(sys.argv[1]).read())
ins_re = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")
out = {}
for name, body in zip(parts[1::2], parts[2::2]):
    m = re.search(r"cpw[mbx]?ILi(\d+)E", name)
    if not m:
        continue
    ins = [(int(a, 16), op, rest) for a, op, rest in ins_re.findall(body)]
    loops = [(int(t, 16), a) for a, op, rest in ins if op.startswith("BRA")
             for t in re.findall(r"0x([0-9a-f]+)", rest) if int(t, 16) < a]
    head, tail = max(loops, key=lambda l: l[1] - l[0])
    loop = collections.Counter(op for a, op, _ in ins if head <= a <= tail)
    out[f"mode{m.group(1)}"] = {"instructions": sum(loop.values()), "counts": dict(loop.most_common())}
print(json.dumps(out, indent=1, sort_keys=True))
EOF
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
gpu-lease 1 --wait --max-min 15 -- bash -c 'nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv -i "${GPU_LEASE_UUID:-0}" >"$0/env/leased.csv" 2>&1; "$1" | tee "$0/device.txt"; exit "${PIPESTATUS[0]}"' "$OUT" "$BIN"
