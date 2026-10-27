#!/usr/bin/env bash
# run.sh's part inside the lease: run_lease.sh BINARY MMA, MMA the card's tensor role (imma on sm_120, wgmma on sm_90a).
# The leased GPU only (CUDA_VISIBLE_DEVICES and GPU_LEASE_UUID from gpu-lease).
set -uo pipefail
BIN=$1 MMA=$2
UUID=${GPU_LEASE_UUID:?under gpu-lease}
S=${PF_SECS:-15} W=${PF_WARM:-6}
mkdir -p "$OUT/volt"
LINE=$(mktemp)
nvidia-smi -i "$UUID" -q >"$OUT/nvsmi-before.txt" 2>&1
fields=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi -i "$UUID" --query-gpu=power.draw.instant --format=csv >/dev/null 2>&1 && fields=$fields,power.draw.instant
nvidia-smi -i "$UUID" --query-gpu=$fields --format=csv,nounits -lms 50 >"$OUT/smi_trace.csv" 2>&1 &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

# one(LABEL, ARGS...): energy ARGS, its line tagged with LABEL, and a VOLTAGE,CLOCK snapshot halfway through its window
one() {
  local label=$1
  shift
  echo "{\"label\": \"$label\", \"start\": \"$(date -u +%FT%T.%3NZ)\"}" >>"$OUT/phases.jsonl"
  (sleep "$((W + S / 2))" && nvidia-smi -i "$UUID" -q -d VOLTAGE,CLOCK,POWER >"$OUT/volt/$label.txt" 2>&1) &
  local snap=$!
  "$BIN" "$@" >"$LINE" 2>>"$OUT/energy.err"
  local rc=$?
  wait "$snap" 2>/dev/null
  if [ $rc -eq 0 ]; then
    sed "s/^{/{\"label\": \"$label\", /" "$LINE" >>"$OUT/energy.jsonl"
  else
    echo "{\"label\": \"$label\", \"failed\": $rc}" >>"$OUT/energy.jsonl"
  fi
  echo "{\"label\": \"$label\", \"end\": \"$(date -u +%FT%T.%3NZ)\", \"rc\": $rc}" >>"$OUT/phases.jsonl"
}

pipes() {  # pipes PREFIX SECS WARM: the whole table at the GPU's current clocks
  local p=$1 s=$2 w=$3
  one "$p$MMA-1" "$MMA" "$MMA" 1 "$s" "$w"
  one "${p}idle" idle 10
  one "${p}none" none none 0 "$s" "$w"
  one "${p}sleep" sleep sleep 0 "$s" "$w"
  one "$p$MMA-0" "$MMA" "$MMA" 0 "$s" "$w"
  one "$p$MMA-2" "$MMA" "$MMA" 2 "$s" "$w"
  [ "$MMA" = wgmma ] && one "${p}imma-1" imma imma 1 "$s" "$w" && one "${p}imma-0" imma imma 0 "$s" "$w"
  for m in 0 1 2; do
    one "${p}dp4a-$m" dp4a dp4a $m "$s" "$w"
    one "${p}ffma-$m" ffma ffma $m "$s" "$w"
  done
  one "${p}int-0" int int 0 "$s" "$w"
  one "${p}int-1" int int 1 "$s" "$w"
  one "${p}sha-0" sha sha 0 "$s" "$w"
  one "${p}sha-1" sha sha 1 "$s" "$w"
  one "${p}lds-0" lds lds 0 "$s" "$w"
  one "${p}lds-1" lds lds 1 "$s" "$w"
  for m in 0 1 2; do one "${p}ldg-$m" ldg ldg $m "$s" "$w"; done
  for f in 0.25 0.5 0.75; do one "$p$MMA-1-f$f" "$MMA" "$MMA" 1 "$s" "$w" $f; done
  one "${p}ffma-1-f0.5" ffma ffma 1 "$s" "$w" 0.5
  one "$p$MMA+sleep-1" "$MMA" sleep 1 "$s" "$w"
  for b in dp4a ffma int sha lds ldg; do one "$p$MMA+$b-1" "$MMA" $b 1 "$s" "$w"; done
  one "$p$MMA+ffma-0" "$MMA" ffma 0 "$s" "$w"
  one "${p}dp4a+ffma-1" dp4a ffma 1 "$s" "$w"
  one "${p}ffma+int-1" ffma int 1 "$s" "$w"
  one "${p}ffma+sleep-1" ffma sleep 1 "$s" "$w"
  one "${p}gemm-1" gemm 1 "$s" "$w"
  one "${p}gemm-0" gemm 0 "$s" "$w"
  one "$p$MMA-1-again" "$MMA" "$MMA" 1 "$s" "$w"
}

pipes "" "$S" "$W"

# The clock sweep, where clocks may be set: the tensor pipe, FFMA and the resident-idle floor at each locked clock.
if sudo -n nvidia-smi -i "$UUID" -lgc "${PF_CLOCKS_FIRST:-1500},${PF_CLOCKS_FIRST:-1500}" >"$OUT/lgc.txt" 2>&1; then
  if [ -n "${PF_RELOCK_MHZ:-}" ]; then
    trap 'sudo -n nvidia-smi -i "$UUID" -lgc "$PF_RELOCK_MHZ,$PF_RELOCK_MHZ" >>"$OUT/lgc.txt" 2>&1; kill $SMI 2>/dev/null' EXIT
  else
    trap 'sudo -n nvidia-smi -i "$UUID" -rgc >>"$OUT/lgc.txt" 2>&1; kill $SMI 2>/dev/null' EXIT
  fi
  for f in ${PF_CLOCKS:-1980 1755 1500 1200 900}; do
    sudo -n nvidia-smi -i "$UUID" -lgc "$f,$f" >>"$OUT/lgc.txt" 2>&1 || continue
    one "lgc$f-sleep" sleep sleep 0 10 4
    one "lgc$f-$MMA-1" "$MMA" "$MMA" 1 10 4
    one "lgc$f-$MMA-0" "$MMA" "$MMA" 0 10 4
    one "lgc$f-ffma-1" ffma ffma 1 10 4
    one "lgc$f-gemm-1" gemm 1 10 4
  done
else
  echo "sudo -n nvidia-smi -lgc refused: no clock sweep" >>"$OUT/lgc.txt"
fi
nvidia-smi -i "$UUID" -q >"$OUT/nvsmi-after.txt" 2>&1
rm -f "$LINE"
exit 0
