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

# Clocks may be set only where `sudo -n nvidia-smi` is allowed. Every exit puts the GPU back: locked at PF_RELOCK_MHZ
# (the Nebius nodes' vy-clocks lock, 2100), else reset to the defaults.
restore() {
  if [ -n "${PF_RELOCK_MHZ:-}" ]; then
    sudo -n nvidia-smi -i "$UUID" -lgc "$PF_RELOCK_MHZ,$PF_RELOCK_MHZ" >>"$OUT/lgc.txt" 2>&1
  else
    sudo -n nvidia-smi -i "$UUID" -rgc >>"$OUT/lgc.txt" 2>&1
  fi
  nvidia-smi -i "$UUID" -q -d CLOCK >"$OUT/nvsmi-restored.txt" 2>&1
}
armed=0
arm() {
  [ $armed = 1 ] && return 0
  sudo -n nvidia-smi -i "$UUID" -q -d CLOCK >/dev/null 2>&1 || return 1
  trap 'restore; kill $SMI 2>/dev/null' EXIT
  trap 'exit 143' TERM INT
  armed=1
}

# The clock sweep: the tensor pipe, FFMA, the GEMMs and the resident-idle floor at each locked clock.
sweep() {
  for f in ${PF_CLOCKS:-1980 1755 1500 1200 900}; do
    arm && sudo -n nvidia-smi -i "$UUID" -lgc "$f,$f" >>"$OUT/lgc.txt" 2>&1 || {
      echo "sudo -n nvidia-smi -lgc $f refused" >>"$OUT/lgc.txt"
      continue
    }
    one "lgc$f-sleep" sleep sleep 0 10 4
    one "lgc$f-$MMA-1" "$MMA" "$MMA" 1 10 4
    one "lgc$f-$MMA-0" "$MMA" "$MMA" 0 10 4
    one "lgc$f-ffma-1" ffma ffma 1 10 4
    one "lgc$f-gemm-1" gemm 1 10 4
  done
}

# At the default clocks (the power cap holds the board, not a lock): the pipes the public best uses, alone and beside
# the tensor pipe, and the GEMMs.
cap() {
  arm && sudo -n nvidia-smi -i "$UUID" -rgc >>"$OUT/lgc.txt" 2>&1 || {
    echo "sudo -n nvidia-smi -rgc refused: no default-clock phase" >>"$OUT/lgc.txt"
    return
  }
  nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-default.txt" 2>&1
  one "cap-gemm-1" gemm 1 "$S" "$W"
  one "cap-sleep" sleep sleep 0 "$S" "$W"
  one "cap-idle" idle 10
  for m in 1 0 2; do one "cap-$MMA-$m" "$MMA" "$MMA" $m "$S" "$W"; done
  one "cap-gemm-0" gemm 0 "$S" "$W"
  for r in dp4a ffma int lds; do one "cap-$r-1" $r $r 1 "$S" "$W"; done
  one "cap-dp4a-0" dp4a dp4a 0 "$S" "$W"
  one "cap-$MMA+sleep-1" "$MMA" sleep 1 "$S" "$W"
  for b in dp4a ffma int lds ldg sha; do one "cap-$MMA+$b-1" "$MMA" $b 1 "$S" "$W"; done
  one "cap-$MMA+dp4a-0" "$MMA" dp4a 0 "$S" "$W"
  one "cap-dp4a+ffma-1" dp4a ffma 1 "$S" "$W"
  one "cap-gemm-1-again" gemm 1 "$S" "$W"
}

case ${PF_PLAN:-full} in
  full)
    pipes "" "$S" "$W"
    sweep
    ;;
  cap)
    cap
    PF_CLOCKS=${PF_CAP_CLOCKS:-2430 2250 2100} sweep
    ;;
  gemm)  # the honest GEMMs alone, at the default clocks and then locked at PF_CLOCKS
    arm && sudo -n nvidia-smi -i "$UUID" -rgc >>"$OUT/lgc.txt" 2>&1 || echo "sudo -n nvidia-smi -rgc refused" >>"$OUT/lgc.txt"
    one "cap-sleep" sleep sleep 0 "$S" "$W"
    one "cap-gemm-1" gemm 1 "$S" "$W"
    one "cap-gemm-0" gemm 0 "$S" "$W"
    one "cap-gemm-1-again" gemm 1 "$S" "$W"
    for f in ${PF_CLOCKS:-}; do
      arm && sudo -n nvidia-smi -i "$UUID" -lgc "$f,$f" >>"$OUT/lgc.txt" 2>&1 || continue
      one "lgc$f-sleep" sleep sleep 0 10 4
      one "lgc$f-gemm-1" gemm 1 10 4
    done
    ;;
esac
nvidia-smi -i "$UUID" -q >"$OUT/nvsmi-after.txt" 2>&1
rm -f "$LINE"
exit 0
