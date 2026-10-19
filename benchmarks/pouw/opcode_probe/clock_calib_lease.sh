#!/usr/bin/env bash
# clock_calib.sh's part inside the lease: clock_calib_lease.sh BINARY OUT. The leased GPU only (GPU_LEASE_UUID).
# Reading C's question: how precisely can an auditor's own clock time a known-length reference kernel? A reference
# kernel of a known cycle count is launched REPS times at several lengths, each timed three ways (the auditor's host
# steady_clock, the chip's %globaltimer, clock64 cycles); the spread of the host-timed clock across reps is the
# auditor-side measurement uncertainty u. Timing lower-bounds the clock (a prover can always run slower), so this
# measures u for the lower-bound reading only. Clocks are restored on exit.
set -uo pipefail
BIN=$1 OUT=$2
UUID=${GPU_LEASE_UUID:?under gpu-lease}
REF=${CALIB_REF:-p_mma0}            # the reference kernel: IMMA on zeros, a known deterministic cycle count a block
REPS=${CALIB_REPS:-40}
LENS=${CALIB_LENS:-0.002 0.01 0.05 0.2}
LOCK=${CLOCK_RELOCK_MHZ:-2100}
MEMLOCK=${CLOCK_RELOCK_MEM_MHZ:-12481}
q() { nvidia-smi -i "$UUID" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }
FOUND_PL=$(q power.limit)
smi() { sudo -n nvidia-smi -i "$UUID" "$@"; }
calib() {  # calib TRIAL LEN: REPS host-timed launches of the reference kernel lasting ~LEN seconds each
  local t=$1 len=$2
  "$BIN" --calib "$REF" "$len" "$REPS" | sed "s/^{/{\"trial\": \"$t\", \"ref\": \"$REF\", \"len_s\": $len, /" \
    >>"$OUT/jitter.jsonl"
}
restore() {
  { smi -rac; smi -pl "$FOUND_PL"; smi -rgc; smi -lgc "$LOCK,$LOCK"; smi -lmc "$MEMLOCK,$MEMLOCK"; } \
    >"$OUT/restored.txt" 2>&1
  nvidia-smi -i "$UUID" -q -d CLOCK,POWER >"$OUT/nvsmi-restored.txt" 2>&1
}
trap restore EXIT
trap 'exit 143' TERM INT

# At the node's lock (honest's operating point) and unlocked (boost): the jitter is a property of the measurement.
smi -rgc >/dev/null 2>&1
smi -lgc "$LOCK,$LOCK" >/dev/null 2>&1
for len in $LENS; do calib "locked" "$len"; done
smi -rgc >/dev/null 2>&1
for len in $LENS; do calib "boost" "$len"; done
exit 0
