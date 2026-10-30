#!/usr/bin/env bash
# The census's unpriced classes timed on one GPU: unpriced.cu on a B200 (ARCH=sm_100a, the default) or unpriced90.cu on
# an H200 (ARCH=sm_90a); SRC=unpriced_fp4.cu times the B200's fp4 tensor kinds. From a pushed commit:
#
#   research run --on vy-dc-b200 --project verity --campaign eps1 --source <tree> --cwd source --timeout 30m \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/sass_census/rates.sh
#   research run --on vy-dc-h200 ... --env ARCH=sm_90a --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/sass_census/rates.sh
#
# Outside the lease: build for ARCH and SASS-gate every probe (gate-build.txt; a refusal stops the run before any GPU).
# Under one lease: probe.py run (rows.jsonl, gate.json, times.jsonl), then rates.json, each probe's warp instructions
# and results (FP32 results or MACs) a SM-clock, and the ratios the census needs: FFMA2 / FFMA and FMUL2 / FMUL in FP32
# results, DMMA / IMMA and DMMA / the device's fastest s8 tensor form in MACs, DMMA beside a saturated tcgen05 or wgmma
# (t + s at each joint point: above 1, DMMA's words add to the tensor pipe's), and FFMA2's issue slots (the pairs'
# summed issue rate beside LOP3 and IMAD: near 4 a SM-clock, one slot). BUILD_ONLY=1 stops after the gate (no GPU).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
ARCH=${ARCH:-sm_100a}
case $ARCH in
  sm_100a) SRC=${SRC:-unpriced.cu} CC=10 ;;
  sm_90a) SRC=${SRC:-unpriced90.cu} CC=9 ;;
  *) echo "rates.sh: ARCH is sm_100a or sm_90a, not $ARCH" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "rates.sh: no nvcc (CUDA 13.0 has $ARCH)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/env"
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1

python3 "$PROBE" build --arch "$ARCH" --out "$WORK" "$HERE/$SRC" >"$WORK/built" 2>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
BIN=$(head -1 "$WORK/built")
cuobjdump -sass "$BIN" >"$OUT/${SRC%.cu}.sass" 2>&1
python3 "$PROBE" gate "$BIN" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit 1
[ "${BUILD_ONLY:-}" = 1 ] && exit 0

lease=()
command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
"${lease[@]}" bash -c '
  set -uo pipefail
  uuid=${GPU_LEASE_UUID:-}; uuid=${uuid%%,*}
  sel=(); [ -n "$uuid" ] && sel=(-i "$uuid")
  nvidia-smi "${sel[@]}" -q >"$1/env/nvidia-smi.txt" 2>&1
  cap=$(nvidia-smi "${sel[@]}" --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "${cap%%.*}" = "$4" ] || { echo "rates.sh: $5 needs compute capability $4.x, the device reports $cap" >&2; exit 2; }
  python3 "$2" run "$3" --out "$1" --reps 10' _ "$OUT" "$PROBE" "$BIN" "$CC" "$ARCH" || exit $?

python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
# results a warp instruction: FP32 results (lanes x results a lane) or MACs (m n k, a wgmma's over its 4 warps)
TC, WG, WG256 = 128 * 256 * 32, 64 * 128 * 32 // 4, 64 * 256 * 32 // 4
PER = {"s_ffma": 32, "s_ffma2": 64, "s_ffma2c": 64, "s_fmul": 32, "s_fmul2": 64, "s_dmma": 256, "s_imma": 4096,
       "s_dmma_w8": 256, "s_dmma16": 2048, "s_dmma16_w8": 2048, "tc_i8": TC, "wg_i8": WG, "wg256": WG256,
       "k_i8": TC, "k_f8e4m3": TC, "k_f8e2m1": TC, "k_mxf4": 2 * TC, "k_nvf4": 2 * TC}
# each joint: the tensor form's alone probe, its pattern and MACs, and the DMMA shape's alone probe and MACs
JOINTS = {"dmmaU_tc": ("tc_i8", "UTCIMMA*", TC, "s_dmma_w8", 256), "tcU_dmma": ("tc_i8", "UTCIMMA*", TC, "s_dmma_w8", 256),
          "dmmaU_wg": ("wg_i8", "IGMMA*", WG, "s_dmma_w8", 256), "wgU_dmma": ("wg_i8", "IGMMA*", WG, "s_dmma_w8", 256),
          "dmma16U_wg": ("wg_i8", "IGMMA*", WG, "s_dmma16_w8", 2048),
          "wgU_dmma16": ("wg_i8", "IGMMA*", WG, "s_dmma16_w8", 2048)}
rows = {r["probe"]: r for r in map(json.loads, (out / "rows.jsonl").read_text().splitlines())}
per = {p: {"warp_instr_a_sm_clk": r["rate"], "results_a_sm_clk": r["rate"] * PER[p] if p in PER else None,
           "rates": r["rates"], "spread": r["spread"], "sm_mhz": r["sm_mhz"], "tags": r["tags"],
           "stop_gap": r.get("stop_gap"), "sass_sha256": r["sass_sha256"]} for p, r in rows.items()}
res = lambda p: per[p]["results_a_sm_clk"] if p in per else None  # noqa: E731
ratio = lambda a, b: res(a) / res(b) if res(a) and res(b) else None  # noqa: E731
best2 = max((p for p in ("s_ffma2", "s_ffma2c") if p in per), key=res, default=None)
tensor = next((p for p in ("tc_i8", "wg_i8") if p in per), None)
# the joints: t the tensor rate over its alone probe's, s the DMMA rate over its alone probe's; t + s > 1 means DMMA adds
joints = {}
for name, (talone, tpat, tmacs, dalone, dmacs) in JOINTS.items():
    if name in per and talone in per and dalone in per:
        rt = next(v for k, v in per[name]["rates"].items() if k.split(":", 1)[1].startswith(tpat.rstrip("*")))
        rd = next(v for k, v in per[name]["rates"].items() if k.split(":", 1)[1].startswith("DMMA"))
        t, s = rt / per[talone]["warp_instr_a_sm_clk"], rd / per[dalone]["warp_instr_a_sm_clk"]
        joints[name] = {"t": t, "s": s, "t_plus_s": t + s, "tensor_macs_a_sm_clk": rt * tmacs,
                        "dmma_macs_a_sm_clk": rd * dmacs, "dmma_macs_over_tensor_alone": rd * dmacs / res(talone),
                        "stop_gap": per[name]["stop_gap"]}
# the pairs: both loops' warp instructions a SM-clock, their sum (the issue port's 4 is the ceiling) and FP32 results
FP32 = {"FFMA": 32, "FFMA2*": 64}
pairs = {}
for name, p in per.items():
    if name.startswith("p_"):
        by = {k.split(":", 1)[1]: v for k, v in p["rates"].items()}
        pairs[name] = {"rates": by, "issue_sum": sum(by.values()),
                       "fp32_results_a_sm_clk": sum(v * FP32.get(k, 0) for k, v in by.items())}
summary = {"probes": per, "ffma2_over_ffma": ratio(best2, "s_ffma") if best2 else None, "ffma2_form": best2,
           "fmul2_over_fmul": ratio("s_fmul2", "s_fmul"), "dmma_over_imma_macs": ratio("s_dmma", "s_imma"),
           "dmma16_over_imma_macs": ratio("s_dmma16", "s_imma"), "tensor_form": tensor,
           "dmma_over_tensor_macs": ratio("s_dmma", tensor) if tensor else None,
           "dmma16_over_tensor_macs": ratio("s_dmma16", tensor) if tensor else None,
           "wg_i8_over_wg256_macs": ratio("wg_i8", "wg256"), "dmma_beside_tensor": joints, "pairs": pairs,
           "fp4_over_i8_macs": {k: ratio(k, "k_i8") for k in ("k_f8e4m3", "k_f8e2m1", "k_mxf4", "k_nvf4") if k in per}}
(out / "rates.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=1))
EOF
