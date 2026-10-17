#!/usr/bin/env bash
# clock_ceiling.sh's part inside the lease: clock_ceiling_lease.sh BINARY OUT. The leased GPU only (GPU_LEASE_UUID).
set -uo pipefail
BIN=$1 OUT=$2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UUID=${GPU_LEASE_UUID:?under gpu-lease}
IDX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2 == u {print $1}')
SECS=${CLOCK_SECS:-20}
LOCK=${CLOCK_RELOCK_MHZ:-2100}
MEMLOCK=${CLOCK_RELOCK_MEM_MHZ:-12481}
q() { nvidia-smi -i "$UUID" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }
MAXMHZ=$(q clocks.max.sm)
MAXMEM=$(q clocks.max.mem)
FOUND_PL=$(q power.limit)
FOUND_VBOOST=$(nvidia-smi boost-slider -l -i "$IDX" 2>/dev/null | awk '$3 == "vboost" {print $5}')
MAX_VBOOST=$(nvidia-smi boost-slider -l -i "$IDX" 2>/dev/null | awk '$3 == "vboost" {print $4}')
echo "{\"uuid\": \"$UUID\", \"index\": \"$IDX\", \"max_mhz\": \"$MAXMHZ\", \"max_mem_mhz\": \"$MAXMEM\", \"found_pl_w\": \"$FOUND_PL\"," \
  "\"found_vboost\": \"$FOUND_VBOOST\", \"max_vboost\": \"$MAX_VBOOST\", \"relock_mhz\": \"$LOCK\"}" >"$OUT/found.json"
nvidia-smi -i "$UUID" -q >"$OUT/nvsmi-before.txt" 2>&1
nvidia-smi -i "$UUID" -q -d SUPPORTED_CLOCKS >"$OUT/supported-clocks.txt" 2>&1
nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
  --format=csv,nounits -lms 100 >"$OUT/smi_trace.csv" 2>&1 &
SMI_PID=$!

smi() { sudo -n nvidia-smi -i "$UUID" "$@"; }
nv() { sudo -n python3 "$HERE/clocks.py" "$1" "$UUID" "${@:2}"; }
vboost() { sudo -n nvidia-smi boost-slider -i "$IDX" --vboost "$1"; }
state() { nv state | sed "s/^{/{\"trial\": \"$1\", /" >>"$OUT/states.jsonl"; }
calib() {  # calib TRIAL KERNEL...: two host-timed 3 s launches a kernel
  local t=$1 k
  shift
  for k in "$@"; do "$BIN" --calib "$k" 3 2 | sed "s/^{/{\"trial\": \"$t\", /" >>"$OUT/calib.jsonl"; done
}
# do TRIAL CMD...: run one knob, record its rc and output, then the state it left
do_() {
  local t=$1 o rc
  shift
  o=$("$@" 2>&1)
  rc=$?
  python3 -c 'import json,sys; print(json.dumps({"trial": sys.argv[1], "cmd": sys.argv[2], "rc": int(sys.argv[3]), "out": sys.argv[4]}))' \
    "$t" "$*" "$rc" "$o" >>"$OUT/trials.jsonl"
  state "$t"
  return $rc
}
baseline() { smi -rac >/dev/null 2>&1; smi -cc 0 >/dev/null 2>&1; nv offset 0 >/dev/null 2>&1; smi -rgc >/dev/null 2>&1; }
restore() {
  {
    smi -rac
    smi -cc 0
    [ -n "$FOUND_VBOOST" ] && vboost "$FOUND_VBOOST"
    nv offset 0
    nv offset 0 --mem
    smi -pl "$FOUND_PL"
    smi -rgc
    smi -lgc "$LOCK,$LOCK"
    smi -lmc "$MEMLOCK,$MEMLOCK"
  } >"$OUT/restored.txt" 2>&1
  state restored
  nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-restored.txt" 2>&1
  nvidia-smi boost-slider -l -i "$IDX" >>"$OUT/nvsmi-restored.txt" 2>&1
  kill "$SMI_PID" 2>/dev/null
}
trap restore EXIT
trap 'exit 143' TERM INT

# The knobs, each from the default clocks (baseline), each followed by host-timed launches of the lightest kernel
# (p_light, one warp an SM), IMMA on zeros (p_mma0) and the honest step (p_honest: IMMA plus the fold's lane work).
state found
calib locked "p_light" "p_honest"
do_ rgc smi -rgc
calib rgc p_light p_mma0 p_honest
do_ lgc-max smi -lgc "$MAXMHZ,$MAXMHZ"
calib lgc-max p_light p_mma0 p_honest
do_ lgc-max-m1 smi -lgc "$MAXMHZ,$MAXMHZ" -m 1
calib lgc-max-m1 p_light p_honest
for f in $((MAXMHZ + 8)) $((MAXMHZ + 90)) $((MAXMHZ + 500)); do
  baseline
  do_ "lgc-$f" smi -lgc "$f,$f"
  calib "lgc-$f" p_light
done
baseline
do_ nvml-lock-unlimited nv lock tdp unlimited
calib nvml-lock-unlimited p_light
do_ nvml-lock-max-unlimited nv lock "$MAXMHZ" unlimited
calib nvml-lock-max-unlimited p_light
baseline
do_ ac-max smi -ac "$MAXMEM,$MAXMHZ"
calib ac-max p_light
do_ ac-above smi -ac "$MAXMEM,$((MAXMHZ + 90))"
calib ac-above p_light
baseline
do_ cc-override smi -cc 1
calib cc-override p_light p_honest
baseline
do_ pl-above smi -pl $((FOUND_PL + 50))
for o in 15 105 500; do
  baseline
  do_ "gpc-offset+$o" nv offset "$o"
  calib "gpc-offset+$o" p_light
  do_ "gpc-offset+$o-lgc-max" smi -lgc "$MAXMHZ,$MAXMHZ"
  calib "gpc-offset+$o-lgc-max" p_light
done
baseline
do_ mem-offset+500 nv offset 500 --mem
do_ lmc-above smi -lmc $((MAXMEM + 500)),$((MAXMEM + 500))
smi -lmc "$MEMLOCK,$MEMLOCK" >/dev/null 2>&1
baseline
BEST_VBOOST=""
if [ -n "$MAX_VBOOST" ]; then
  for v in $(seq 1 "$MAX_VBOOST"); do
    do_ "vboost-$v" vboost "$v" && BEST_VBOOST=$v
    calib "vboost-$v" p_light p_honest
  done
  vboost "${FOUND_VBOOST:-0}" >/dev/null 2>&1
fi

# Sustained: every clock kernel at the default clocks, locked at the maximum, and at the highest vboost accepted.
KERNELS=(p_light p_mma0 p_honest p_mma p_half p_mixed p_ffma p_fold p_light)
phase() {
  local label=$1 k
  "$BIN" --sustain p_honest 30 50 | sed "s/^{/{\"phase\": \"$label-warmup\", /" >>"$OUT/sustain.jsonl"
  for k in "${KERNELS[@]}"; do
    "$BIN" --sustain "$k" "$SECS" 50 | sed "s/^{/{\"phase\": \"$label\", /" >>"$OUT/sustain.jsonl"
  done
  calib "$label-after" p_honest p_light
}
baseline
state sustain-default
phase default
smi -lgc "$MAXMHZ,$MAXMHZ" >/dev/null 2>&1
state sustain-lgc-max
phase lgc-max
if [ -n "$BEST_VBOOST" ]; then
  baseline
  vboost "$BEST_VBOOST" >/dev/null 2>&1
  state "sustain-vboost-$BEST_VBOOST"
  phase "vboost-$BEST_VBOOST"
fi
exit 0
