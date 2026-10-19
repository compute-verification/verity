#!/usr/bin/env bash
# The census's unpriced sm_100a classes timed (unpriced.cu) on one B200. From a pushed commit:
#
#   research run --on vy-dc-b200 --project verity --campaign eps1 --source <tree> --cwd source --timeout 30m \
#     --env GPU_LEASE_WHO=<lane> -- bash benchmarks/pouw/sass_census/rates.sh
#
# Outside the lease: build for sm_100a and SASS-gate every probe (gate-build.txt; a refusal stops the run before any
# GPU). Under one lease: probe.py run (rows.jsonl, gate.json, times.jsonl), then rates.json, each probe's warp
# instructions and results (FP32 results or MACs) a SM-clock, and the ratios the census needs: FFMA2 / FFMA and
# FMUL2 / FMUL in FP32 results, DMMA / IMMA in MACs. BUILD_ONLY=1 stops after the gate (no GPU).
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
PER = {"s_ffma": 32, "s_ffma2": 64, "s_ffma2c": 64, "s_fmul": 32, "s_fmul2": 64, "s_dmma": 256, "s_imma": 4096}
rows = {r["probe"]: r for r in map(json.loads, (out / "rows.jsonl").read_text().splitlines())}
per = {p: {"warp_instr_a_sm_clk": r["rate"], "results_a_sm_clk": r["rate"] * PER[p], "spread": r["spread"],
           "sm_mhz": r["sm_mhz"], "tags": r["tags"], "sass_sha256": r["sass_sha256"]} for p, r in rows.items()}
res = lambda p: per[p]["results_a_sm_clk"] if p in per else None  # noqa: E731
ratio = lambda a, b: res(a) / res(b) if res(a) and res(b) else None  # noqa: E731
best2 = max((p for p in ("s_ffma2", "s_ffma2c") if p in per), key=res, default=None)
summary = {"probes": per, "ffma2_over_ffma": ratio(best2, "s_ffma") if best2 else None, "ffma2_form": best2,
           "fmul2_over_fmul": ratio("s_fmul2", "s_fmul"), "dmma_over_imma_macs": ratio("s_dmma", "s_imma")}
(out / "rates.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "probes"}))
EOF
