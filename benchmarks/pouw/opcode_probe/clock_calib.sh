#!/usr/bin/env bash
# The clock ceiling's reading C: the auditor-side measurement uncertainty of timing a known-length reference kernel with
# the auditor's own clock. Builds and gates probes/power.cu outside the lease, then under one gpu-lease
# (clock_calib_lease.sh) runs a reference kernel of known cycle count REPS times at several lengths, timed by the host's
# steady_clock (the auditor's clock, with no trust in the chip), and reports the jitter = u. From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/clock_calib.sh
#
# $OUT gets jitter.jsonl (REPS host-timed rows a length), restored.txt, nvsmi-restored.txt and calib.txt (eps1.py
# jitter). CALIB_REPS (40), CALIB_LENS ("0.002 0.01 0.05 0.2") and CALIB_REF (p_mma0) tune it; CLOCK_RELOCK_MHZ and
# CLOCK_RELOCK_MEM_MHZ (2100, 12481) are put back on exit.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 "$HERE/probe.py" build --out "$WORK" ${CLOCK_ARCH:+--arch "$CLOCK_ARCH"} "$HERE/probes/power.cu" >"$WORK/built" \
  2>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
python3 "$HERE/probe.py" gate "$WORK/power" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc
gpu-lease 1 --wait --max-min 10 -- bash "$HERE/clock_calib_lease.sh" "$WORK/power" "$OUT"
python3 "$HERE/eps1.py" jitter "$OUT" | tee "$OUT/calib.txt"
