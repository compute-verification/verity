#!/usr/bin/env bash
# The clock and power run on the L40S (sm_89): probes/power.cu built for sm_89 and gated, then under one gpu-lease
# the GPU's clocks reset to the defaults (sudo nvidia-smi -rgc) and every kernel sustained at the power limit as found,
# then at the board's default power limit (sudo nvidia-smi -pl, the found limit restored on exit). From a pushed commit:
#
#   research run --on vy-dc-l40s --project verity --campaign cmp-13 --source <tree> --cwd source --custody-ttl 12h \
#     --timeout 40m --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/sm89/clock.sh
#
# $OUT gets sustain.jsonl (a line a launch, `phase` found or default-pl), smi_trace.csv (nvidia-smi every 100 ms),
# phases.jsonl, the nvidia-smi -q snapshots and clock.txt (eps1.py clock). POWER_SECS (20) is each kernel's time.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
mkdir -p "$OUT/sass"
BIN="$OUT/power"
"$NVCC" -std=c++20 -O3 -gencode arch=compute_89,code=sm_89 -ldl -o "$BIN" "$HERE/../probes/power.cu" >"$OUT/build.log" 2>&1 \
  || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$BIN" >"$OUT/sass/power.sass" 2>&1
python3 "$HERE/../probe.py" gate "$BIN" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc

cat >"$OUT/lease.sh" <<'EOF'
set -uo pipefail
BIN=$1 OUT=$2 SECS=$3
UUID=${GPU_LEASE_UUID:?under gpu-lease}
KERNELS=(p_light p_honest16 p_honest p_half p_mma16 p_mma p_mma0 p_mma16z p_fold p_ffma p_mixed p_light p_honest16)
# the enforced limit: on the L40S `power.limit` reads 350 W while 325 W is enforced
FOUND_PL=$(nvidia-smi -i "$UUID" --query-gpu=enforced.power.limit --format=csv,noheader,nounits | tr -d ' ')
DEFAULT_PL=$(nvidia-smi -i "$UUID" --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
echo "{\"found_pl_w\": \"$FOUND_PL\", \"default_pl_w\": \"$DEFAULT_PL\"}" >"$OUT/limits.json"
nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE >"$OUT/nvsmi-before.txt" 2>&1
nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,clocks_event_reasons.active \
  --format=csv,nounits -lms 100 >"$OUT/smi_trace.csv" 2>&1 &
SMI=$!

phase() {
  local label=$1
  # a warm-up so every kernel's window starts from the phase's steady temperature
  "$BIN" --sustain p_honest16 30 50 | sed "s/^{/{\"phase\": \"$label-warmup\", /" >>"$OUT/sustain.jsonl"
  for k in "${KERNELS[@]}"; do
    echo "{\"phase\": \"$label\", \"probe\": \"$k\", \"start\": \"$(date -u +%FT%T.%3NZ)\"}" >>"$OUT/phases.jsonl"
    "$BIN" --sustain "$k" "$SECS" 50 | sed "s/^{/{\"phase\": \"$label\", /" >>"$OUT/sustain.jsonl"
  done
}

sudo nvidia-smi -i "$UUID" -rgc >"$OUT/rgc.txt" 2>&1 || echo "nvidia-smi -rgc refused (rgc.txt)" >&2
nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-found.txt" 2>&1
phase found
if [ "$FOUND_PL" != "$DEFAULT_PL" ] && sudo nvidia-smi -i "$UUID" -pl "$DEFAULT_PL" >"$OUT/pl.txt" 2>&1; then
  trap 'sudo nvidia-smi -i "$UUID" -pl "$FOUND_PL" >>"$OUT/pl-restore.txt" 2>&1' EXIT
  nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-default-pl.txt" 2>&1
  phase default-pl
  sudo nvidia-smi -i "$UUID" -pl "$FOUND_PL" >>"$OUT/pl-restore.txt" 2>&1
  trap - EXIT
fi
nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE >"$OUT/nvsmi-after.txt" 2>&1
kill "$SMI" 2>/dev/null
wait "$SMI" 2>/dev/null
exit 0
EOF
gpu-lease 1 --wait --max-min 30 -- bash "$OUT/lease.sh" "$BIN" "$OUT" "${POWER_SECS:-20}"
python3 "$HERE/../eps1.py" clock "$OUT/sustain.jsonl" | tee "$OUT/clock.txt"
