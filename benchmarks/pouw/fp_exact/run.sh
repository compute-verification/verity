#!/usr/bin/env bash
# fp_exact's job: build and gate outside the lease; under one gpu-lease the exactness sweep (and its no-write control)
# and the MMA rates (../opcode_probe/probes/mma.cu, gated again before timing); the host's recheck after the lease.
#   research run --on vy-nebius-2 ... --cwd source -- bash benchmarks/pouw/fp_exact/run.sh [--quick]
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE=$HERE/../opcode_probe
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}; [ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/exact" "$OUT/rates"
"$NVCC" -std=c++20 -O2 -gencode arch=compute_120a,code=sm_120a -o "$WORK/fp_exact" "$HERE/fp_exact.cu" \
  2>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
python3 "$PROBE/probe.py" build --out "$WORK" "$PROBE/probes/mma.cu" >/dev/null 2>>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/fp_exact" >"$OUT/exact/fp_exact.sass"
cuobjdump -sass "$WORK/mma" >"$OUT/rates/mma.sass"
python3 "$PROBE/probe.py" gate "$WORK/mma" --json "$OUT/rates/gate-build.json" >"$OUT/rates/gate-build.txt" 2>&1
rc=$?; cat "$OUT/rates/gate-build.txt"; [ $rc -eq 0 ] || exit $rc
gpu-lease 1 --wait --max-min 25 -- bash -c '
  python3 "$1/fp_exact.py" run "$2/fp_exact" --out "$3/exact" "${@:5}"; rc=$?
  python3 "$4/probe.py" run "$2/mma" --out "$3/rates" || rc=${rc/#0/7}
  exit $rc' _ "$HERE" "$WORK" "$OUT" "$PROBE" "$@"
rc=$?
python3 "$HERE/fp_exact.py" summarize "$OUT/exact" | tee "$OUT/exact/verdicts.txt"
src=${PIPESTATUS[0]}
python3 "$HERE/fp_exact.py" rates "$OUT/rates/rows.jsonl" | tee "$OUT/rates/rates.txt"
[ $rc -ne 0 ] && exit $rc
exit "$src"
