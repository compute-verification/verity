#!/usr/bin/env bash
# RT4-SHEET-1: fold_bulk.cu's variants on sm_120a, priced from their own SASS at π* = DS_issue against FOLD-OPT's
# N_w φ* = 16 (price.json), then, with RT4_DEVICE=1, on the device under one gpu-lease: every variant bit-exact against
# the host's reference, and its SMSP-cycles a unit (device.txt). From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign eps1 --source <tree> --cwd source --mem-gb 4 \
#     --timeout 1500 [--env RT4_DEVICE=1 --env GPU_LEASE_WHO=<you>] -- bash benchmarks/pouw/rt4_sheet/fold_bulk.sh
#
# The price splits each variant's timed loop (its longest backward branch) as cpw_price.py does: what every warp issues,
# and the bodies of forward predicated branches, thread 0's, issued once a block of W = BT / 32 warps. A unit's price is
# (outside + bodies / W) / (8 K); IMMA and NOP are dropped. The device part runs after the SASS gate passes the binary.
# Exit: 0 done, 1 build failed, 2 no nvcc, 3 not bit-exact, 4 the SASS gate refused the binary.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "fold_bulk.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
BIN="$OUT/bin/fold_bulk"
"$NVCC" -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v "$HERE/fold_bulk.cu" \
  -o "$BIN" >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
sha256sum "$BIN" >"$OUT/bin/fold_bulk.sha256"
"$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/fold_bulk.sass" 2>&1
grep -E "Compiling entry|registers|spill" "$OUT/build.log" >"$OUT/registers.txt"
python3 - "$OUT/sass/fold_bulk.sass" >"$OUT/price.json" <<'EOF'
import collections, json, re, sys
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?)([A-Z][A-Z0-9_.]*)([^;]*);")
FLOOR = 16.0
def targets(rest):
    return [int(t, 16) for t in re.findall(r"0x([0-9a-f]+)", rest)]
parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + open(sys.argv[1]).read())
out = {}
for name, body in zip(parts[1::2], parts[2::2]):
    m = re.search(r"(fb[ku])ILi(\d+)ELi(\d+)E(?:Li(\d+)E)?", name)
    if not m:
        continue
    kern, K, BT = m[1], int(m[2]), int(m[3])
    WIDE = int(m[4]) if kern == "fbk" else 1
    if kern == "fbu" and m[4] == "1":
        kern = "fbv"                  # fbu<K, BT, 1>: the bulk body under lane 0 of warp 0
    ins = [(int(a, 16), p.strip(), op, rest) for a, p, op, rest in INS.findall(body)]
    loops = [(t, a) for a, p, op, rest in ins if op.startswith("BRA") for t in targets(rest) if t < a]
    head, tail = max(loops, key=lambda lp: lp[1] - lp[0])
    loop = [x for x in ins if head <= x[0] <= tail and not x[2].startswith("IMMA") and x[2] != "NOP"]
    bodies = [(a, targets(rest)[0]) for a, p, op, rest in loop
              if op in ("BRA", "BRA.U") and p and targets(rest) and targets(rest)[0] > a]
    inside = [op for a, p, op, r in loop if any(b < a < t for b, t in bodies)]
    outside = [op for a, p, op, r in loop if not any(b < a < t for b, t in bodies)]
    W, units = BT // 32, 8 * K
    unit = (len(outside) + len(inside) / W) / units
    out[f"{kern}_K{K}_BT{BT}_W{'256' if WIDE else '128'}"] = {
        "K": K, "BT": BT, "wide": WIDE, "warps": W, "units_a_set": units, "loop": [hex(head), hex(tail)],
        "every_warp_a_set": len(outside), "thread0_a_set": len(inside), "branch_bodies": len(bodies),
        "every_warp_a_unit": len(outside) / units, "ds_issue_a_unit": unit, "floor": FLOOR,
        "below_floor": unit < FLOOR, "spills_in_loop": sum(op.startswith(("LDL", "STL")) for op in outside + inside),
        "every_warp": dict(collections.Counter(outside).most_common()),
        "thread0": dict(collections.Counter(inside).most_common())}
print(json.dumps(out, indent=1, sort_keys=True))
EOF
python3 -c "
import json, sys
for k, v in json.load(open('$OUT/price.json')).items():
    print(f\"{k}: {v['ds_issue_a_unit']:.4f} a unit at DS_issue (every warp {v['every_warp_a_unit']:.4f}, thread 0 \"
          f\"{v['thread0_a_set']} a set over {v['warps']} warps), spills {v['spills_in_loop']}, \"
          f\"{'BELOW' if v['below_floor'] else 'at or above'} {v['floor']}\")
" | tee "$OUT/price.txt"
cat "$OUT/registers.txt"
[ "${RT4_DEVICE:-0}" = 1 ] || exit 0
PATH="$(dirname "$NVCC"):$PATH" python3 "$HERE/../harness/sass_gate.py" "$BIN" --json "$OUT/sass_gate.json" \
  >"$OUT/sass_gate.txt" 2>&1 || { cat "$OUT/sass_gate.txt" >&2; exit 4; }
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
gpu-lease 1 --wait --max-min 15 -- bash -c 'nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv -i "${GPU_LEASE_UUID:-0}" >"$0/env/leased.csv" 2>&1; "$1" | tee "$0/device.txt"; exit "${PIPESTATUS[0]}"' "$OUT" "$BIN"
