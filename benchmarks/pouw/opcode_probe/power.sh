#!/usr/bin/env bash
# The clock and power run (probes/power.cu): build and gate outside the lease, then under one gpu-lease sustain every
# kernel at the node's clock lock, reset the leased GPU's clocks to the defaults (nvidia-smi -rgc: the application clocks
# and the 600 W cap), sustain them again, and lock the GPU back to the node's clock on exit. From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/power.sh
#
# $OUT gets sustain.jsonl (a line a launch: clock64 over %globaltimer, NVML's clock, power and reasons), smi_trace.csv
# (nvidia-smi every 100 ms), phases.jsonl and the nvidia-smi -q snapshots. POWER_SECS (25) is each kernel's time;
# POWER_RELOCK_MHZ (2100, vm_setup.sh's vy-clocks lock) the clock the GPU is locked back to. The clock changes run as
# `sudo -n nvidia-smi`; where sudo refuses them (the vy-lockout's allowlist has no clock rule) the default phase is skipped
# (rgc.txt).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"
python3 "$HERE/probe.py" build --out "$WORK" "$HERE/probes/power.cu" >"$WORK/built" 2>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/power" >"$OUT/sass/power.sass" 2>&1
python3 "$HERE/probe.py" gate "$WORK/power" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc
gpu-lease 1 --wait --max-min 25 -- bash "$HERE/power_lease.sh" "$WORK/power" "$OUT" "${POWER_SECS:-25}"
