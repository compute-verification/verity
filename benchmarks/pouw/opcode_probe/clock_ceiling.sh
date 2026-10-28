#!/usr/bin/env bash
# The highest SM clock a prover with root reaches on one GPU (the P1 premise, "the run's clock <= the data
# sheet's f_max"). Builds and gates probes/power.cu outside the lease, then under one gpu-lease (clock_ceiling_lease.sh)
# tries every clock and power knob the driver offers on the leased GPU only, each followed by host-timed launches
# (power.cu --calib: clock64 cycles over the host's steady_clock, and over %globaltimer), sustains the clock kernels at
# the settings that clock highest, and puts the node's settings back on exit. From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/clock_ceiling.sh
#
# $OUT gets trials.jsonl (each knob's command, rc and output), states.jsonl (clocks.py state after each), calib.jsonl,
# sustain.jsonl, smi_trace.csv, the nvidia-smi -q snapshots, restored.txt and ceiling.txt (eps1.py ceiling and clock).
# CLOCK_SECS (20) is each sustained kernel's time; CLOCK_RELOCK_MHZ and CLOCK_RELOCK_MEM_MHZ (2100, 12481: vm_setup.sh's
# vy-clocks lock) are put back on exit, with the power limit, vboost, CUDA-clock mode and offsets as found.
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
python3 "$HERE/probe.py" build --out "$WORK" ${CLOCK_ARCH:+--arch "$CLOCK_ARCH"} "$HERE/probes/power.cu" >"$WORK/built" \
  2>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/power" >"$OUT/sass/power.sass" 2>&1
python3 "$HERE/probe.py" gate "$WORK/power" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc
gpu-lease 1 --wait --max-min 30 -- bash "$HERE/clock_ceiling_lease.sh" "$WORK/power" "$OUT"
python3 "$HERE/eps1.py" ceiling "$OUT" | tee "$OUT/ceiling.txt"
python3 "$HERE/eps1.py" clock "$OUT/sustain.jsonl" | tee -a "$OUT/ceiling.txt"
