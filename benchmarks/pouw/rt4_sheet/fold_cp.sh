#!/usr/bin/env bash
# RT4-SHEET-1 (b): fold_cp.cu (fbw's whole-block fold + CP-W's per-set checkpoint, an accepted unit) and fold_bulk.cu
# (fbw alone) on sm_120a, priced from one build's SASS at π* = DS_issue (price.json), then, with RT4_DEVICE=1, on the
# device under one gpu-lease: every fcp variant bit-exact against the host (final states and every set's
# checkpoint), and its SMSP-cycles a unit (device.txt). From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign eps1 --source <tree> --cwd source --mem-gb 4 \
#     --timeout 1500 [--env RT4_DEVICE=1 --env GPU_LEASE_WHO=<you>] -- bash benchmarks/pouw/rt4_sheet/fold_cp.sh
#
# The price splits each timed loop as fold_bulk.sh does (what every warp issues, and thread 0's forward-branch bodies
# once a block of W = BT / 32 warps; a unit is (outside + bodies / W) / (8 K); IMMA and NOP dropped). The checkpoint's
# cost a unit is fcp's price minus fbw's for the same K, BT, KS. An accepted unit is IMMA 1 + ♭'s operand floor
# 0.025641 + fcp's price, against #2791 Headline.lean's break line 16.99993 a unit; the fold alone (fbw) is against
# FOLD-OPT's N_w φ* = 16. measured_price.py on #2791's tree prices the same SASS on every row of both readings.
# Exit: 0 done, 1 build failed, 2 no nvcc, 3 not bit-exact, 4 the SASS gate refused the binary.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "fold_cp.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
for src in fold_cp fold_bulk; do
  "$NVCC" -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v "$HERE/$src.cu" \
    -o "$OUT/bin/$src" >"$OUT/build_$src.log" 2>&1 || { cat "$OUT/build_$src.log" >&2; exit 1; }
  sha256sum "$OUT/bin/$src" >"$OUT/bin/$src.sha256"
  "$(dirname "$NVCC")/cuobjdump" -sass "$OUT/bin/$src" >"$OUT/sass/$src.sass" 2>&1
  grep -E "Compiling entry|registers|spill" "$OUT/build_$src.log" >"$OUT/registers_$src.txt"
done
cat "$OUT/sass/fold_cp.sass" "$OUT/sass/fold_bulk.sass" >"$OUT/sass/combined.sass"
python3 - "$OUT/sass/combined.sass" >"$OUT/price.json" <<'EOF'
import collections, json, re, sys
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?)([A-Z][A-Z0-9_.]*)([^;]*);")
FOLD_OPT, BREAK, IMMA, OPERANDS = 16.0, 16.99993, 1.0, 0.025641
def targets(rest):
    return [int(t, 16) for t in re.findall(r"0x([0-9a-f]+)", rest)]
parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + open(sys.argv[1]).read())
out = {}
for name, body in zip(parts[1::2], parts[2::2]):
    m = re.search(r"(fbw|fcp)ILi(\d+)ELi(\d+)ELi(\d+)E", name)
    if not m:
        continue
    kern, K, BT, KS = m[1], int(m[2]), int(m[3]), int(m[4])
    ins = [(int(a, 16), p.strip(), op, rest) for a, p, op, rest in INS.findall(body)]
    loops = [(t, a) for a, p, op, rest in ins if op.startswith("BRA") for t in targets(rest) if t < a]
    head, tail = max(loops, key=lambda lp: lp[1] - lp[0])
    loop = [x for x in ins if head <= x[0] <= tail and not x[2].startswith("IMMA") and x[2] != "NOP"]
    bodies = [(a, targets(rest)[0]) for a, p, op, rest in loop
              if op in ("BRA", "BRA.U") and p and targets(rest) and targets(rest)[0] > a]
    inside = [op for a, p, op, r in loop if any(b < a < t for b, t in bodies)]
    outside = [op for a, p, op, r in loop if not any(b < a < t for b, t in bodies)]
    W, units = BT // 32, 8 * K
    out[f"{kern}{KS}_K{K}_BT{BT}"] = {
        "kernel": kern, "K": K, "BT": BT, "KS": KS, "warps": W, "units_a_set": units, "loop": [hex(head), hex(tail)],
        "every_warp_a_set": len(outside), "thread0_a_set": len(inside),
        "ds_issue_a_unit": (len(outside) + len(inside) / W) / units,
        "spills_in_loop": sum(op.startswith(("LDL", "STL")) for op in outside + inside),
        "every_warp": dict(collections.Counter(outside).most_common()),
        "thread0": dict(collections.Counter(inside).most_common())}
for key, v in out.items():
    if v["kernel"] != "fcp":
        continue
    fold = out.get(key.replace("fcp", "fbw"))
    v["fold_a_unit"] = fold["ds_issue_a_unit"] if fold else None
    v["checkpoint_a_unit"] = v["ds_issue_a_unit"] - fold["ds_issue_a_unit"] if fold else None
    if fold:
        ev, fe = collections.Counter(v["every_warp"]), collections.Counter(fold["every_warp"])
        v["checkpoint_every_warp_a_set"] = {k: ev[k] - fe[k] for k in sorted({*ev, *fe}) if ev[k] != fe[k]}
    v["accepted_unit"] = IMMA + OPERANDS + v["ds_issue_a_unit"]
    v["break_line"] = BREAK
    v["accepted_minus_break"] = v["accepted_unit"] - BREAK
    v["fold_under_fold_opt"] = fold["ds_issue_a_unit"] < FOLD_OPT if fold else None
    v["breaks_headline_at_pi_star"] = v["accepted_unit"] < BREAK
print(json.dumps(out, indent=1, sort_keys=True))
EOF
python3 -c "
import json
d = json.load(open('$OUT/price.json'))
for k, v in sorted(d.items()):
    if v['kernel'] == 'fbw':
        print(f\"{k}: fold {v['ds_issue_a_unit']:.4f} a unit at DS_issue ({'UNDER' if v['ds_issue_a_unit'] < 16 else 'at or above'} N_w phi* 16), spills {v['spills_in_loop']}\")
for k, v in sorted(d.items()):
    if v['kernel'] == 'fcp':
        print(f\"{k}: fold+CP-W {v['ds_issue_a_unit']:.4f} a unit (fold {v['fold_a_unit']:.4f}, checkpoint {v['checkpoint_a_unit']:.4f}: {v['checkpoint_every_warp_a_set']} a set); \"
              f\"accepted unit {v['accepted_unit']:.4f} vs break 16.99993: {v['accepted_minus_break']:+.4f} \"
              f\"({'BREAKS' if v['breaks_headline_at_pi_star'] else 'holds'}), spills {v['spills_in_loop']}\")
" | tee "$OUT/price.txt"
cat "$OUT"/registers_*.txt
[ "${RT4_DEVICE:-0}" = 1 ] || exit 0
PATH="$(dirname "$NVCC"):$PATH" python3 "$HERE/../harness/sass_gate.py" "$OUT/bin/fold_cp" --json "$OUT/sass_gate.json" \
  >"$OUT/sass_gate.txt" 2>&1 || { cat "$OUT/sass_gate.txt" >&2; exit 4; }
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
gpu-lease 1 --wait --max-min 15 -- bash -c 'nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv -i "${GPU_LEASE_UUID:-0}" >"$0/env/leased.csv" 2>&1; "$1" | tee "$0/device.txt"; exit "${PIPESTATUS[0]}"' "$OUT" "$OUT/bin/fold_cp"
