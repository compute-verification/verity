#!/usr/bin/env bash
# power.sh's part inside the lease: power_lease.sh BINARY OUT SECONDS. The leased GPU only (GPU_LEASE_UUID).
set -uo pipefail
BIN=$1 OUT=$2 SECS=$3
UUID=${GPU_LEASE_UUID:?under gpu-lease}
LOCK=${POWER_RELOCK_MHZ:-2100}
KERNELS=(p_honest p_half p_mma p_mma0 p_fold p_ffma p_mixed p_honest)
nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE >"$OUT/nvsmi-before.txt" 2>&1
nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 100 >"$OUT/smi_trace.csv" 2>&1 &
SMI=$!

phase() {
  local label=$1
  # a warm-up so every kernel's window starts from the phase's steady temperature
  "$BIN" --sustain p_honest 30 50 | sed "s/^{/{\"phase\": \"$label-warmup\", /" >>"$OUT/sustain.jsonl"
  for k in "${KERNELS[@]}"; do
    echo "{\"phase\": \"$label\", \"probe\": \"$k\", \"start\": \"$(date -u +%FT%T.%3NZ)\"}" >>"$OUT/phases.jsonl"
    "$BIN" --sustain "$k" "$SECS" 50 | sed "s/^{/{\"phase\": \"$label\", /" >>"$OUT/sustain.jsonl"
  done
}

phase locked
if nvidia-smi -i "$UUID" -rgc >"$OUT/rgc.txt" 2>&1; then
  trap 'nvidia-smi -i "$UUID" -lgc "$LOCK,$LOCK" >>"$OUT/relock.txt" 2>&1' EXIT
  nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-default.txt" 2>&1
  phase default
  nvidia-smi -i "$UUID" -lgc "$LOCK,$LOCK" >>"$OUT/relock.txt" 2>&1
  trap - EXIT
else
  echo "nvidia-smi -rgc refused: the default-clock phase didn't run" >&2
fi
nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-after.txt" 2>&1
kill "$SMI" 2>/dev/null
wait "$SMI" 2>/dev/null
exit 0
