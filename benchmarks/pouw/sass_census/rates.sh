#!/usr/bin/env bash
# The census's unpriced sm_100a classes timed (unpriced.cu) on one B200. From a pushed commit:
#
#   research run --on vy-dc-b200 --project verity --campaign eps1 --source <tree> --cwd source --timeout 30m \
#     --env GPU_LEASE_WHO=<lane> -- bash benchmarks/pouw/sass_census/rates.sh
#
# Outside the lease: build for sm_100a and SASS-gate every probe (gate-build.txt; a refusal stops the run before any
# GPU). Under one lease: probe.py run (rows.jsonl, gate.json, times.jsonl), then rates.json, each probe's warp
# instructions and results (FP32 results or MACs) a SM-clock, and the ratios the census needs: FFMA2 / FFMA and
# FMUL2 / FMUL in FP32 results, DMMA / IMMA and DMMA / tcgen05 i8 in MACs, and DMMA beside a saturated tcgen05 (t + s
# at both joint points: above 1, DMMA's words add to the tensor pipe's), and FFMA2's issue slots (the pairs' summed
# issue rate beside LOP3 and IMAD: near 4 a SM-clock, one slot). BUILD_ONLY=1 stops after the gate (no GPU).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "rates.sh: no nvcc (CUDA 13.0 has sm_100a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/env"
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1

python3 "$PROBE" build --arch sm_100a --out "$WORK" "$HERE/unpriced.cu" >"$WORK/built" 2>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
BIN=$(head -1 "$WORK/built")
cuobjdump -sass "$BIN" >"$OUT/unpriced.sass" 2>&1
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
  [ "${cap%%.*}" = 10 ] || { echo "rates.sh: sm_100a needs compute capability 10.x, the device reports $cap" >&2; exit 2; }
  python3 "$2" run "$3" --out "$1" --reps 10' _ "$OUT" "$PROBE" "$BIN" || exit $?

python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
# results a warp instruction: FP32 results (lanes x results a lane) or MACs (m n k)
TC = 128 * 256 * 32
PER = {"s_ffma": 32, "s_ffma2": 64, "s_ffma2c": 64, "s_fmul": 32, "s_fmul2": 64, "s_dmma": 256, "s_imma": 4096,
       "s_dmma_w8": 256, "tc_i8": TC}
rows = {r["probe"]: r for r in map(json.loads, (out / "rows.jsonl").read_text().splitlines())}
per = {p: {"warp_instr_a_sm_clk": r["rate"], "results_a_sm_clk": r["rate"] * PER[p] if p in PER else None,
           "rates": r["rates"], "spread": r["spread"], "sm_mhz": r["sm_mhz"], "tags": r["tags"],
           "sass_sha256": r["sass_sha256"]} for p, r in rows.items()}
res = lambda p: per[p]["results_a_sm_clk"] if p in per else None  # noqa: E731
ratio = lambda a, b: res(a) / res(b) if res(a) and res(b) else None  # noqa: E731
best2 = max((p for p in ("s_ffma2", "s_ffma2c") if p in per), key=res, default=None)
# the joints: t the tcgen05 rate over tc_i8's, s the DMMA rate over s_dmma_w8's; t + s > 1 means DMMA adds beside it
joints = {}
for name in ("dmmaU_tc", "tcU_dmma"):
    if name in per and "tc_i8" in per and "s_dmma_w8" in per:
        rt = next(v for k, v in per[name]["rates"].items() if k.endswith(":UTCIMMA*"))
        rd = next(v for k, v in per[name]["rates"].items() if k.endswith(":DMMA*"))
        t, s = rt / per["tc_i8"]["warp_instr_a_sm_clk"], rd / per["s_dmma_w8"]["warp_instr_a_sm_clk"]
        joints[name] = {"t": t, "s": s, "t_plus_s": t + s, "tc_macs_a_sm_clk": rt * TC, "dmma_macs_a_sm_clk": rd * 256,
                        "dmma_macs_over_tc_alone": rd * 256 / res("tc_i8")}
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
           "dmma_over_tc_i8_macs": ratio("s_dmma", "tc_i8"), "dmma_beside_tc": joints, "pairs": pairs}
(out / "rates.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=1))
EOF
