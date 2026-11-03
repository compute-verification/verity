#!/usr/bin/env bash
# One-stage's commitment of each Pearl+ call's A (hm96_rows.py time) on one H100 or H200, a session of its own: timing only,
# outside h200.sh's sessions and every verdict's path.  From hm96_rows.py build's library (cwd = the source tree's root):
#
#   research run --on MACHINE --project verity --source . --cwd source --timeout 40m \
#     --env GPU_LEASE_WHO=LANE --send BUILD_RUN/out/lib/hm96_rows.so -- bash benchmarks/pouw/pearl_c_sm90/hm96.sh
#
# Under one lease of at most LEASE_MIN minutes (default 20): the gate and the timings at MODEL's linears (default
# llama32_1b) and each M of MS (default "8 8192": decode's batch and prefill's step), into $OUT/hm96.json, with the card's SM
# clock locked at LOCK_MHZ where given and the node allows it (h200.sh's lock: $OUT/clock.txt) and sampled every 200 ms
# ($OUT/smi.csv).  hm96_rows.py report then reads hm96.json beside an h200.sh session's timing.json and whole.json.
# VERITY_BENCH_VENV names the Python with torch (default /workspace/cp/venv312).  Every step's exit code is in $OUT/rc.txt.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/pouw/pearl_c_sm90
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/integrations/vllm:$PWD/benchmarks/pouw${PYTHONPATH:+:$PYTHONPATH}
export O D PY LIB=$IN/hm96_rows.so MODEL=${MODEL:-llama32_1b} MS=${MS:-8 8192} LOCK_MHZ=${LOCK_MHZ:-}
: > "$O/rc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
sha256sum "$LIB" > "$O/lib.sha256"
echo "lib rc $?" >> "$O/rc.txt"
grep -q "lib rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- bash -c '
  U=$GPU_LEASE_UUID
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$U" > "$O/gpu.txt" 2>&1
  if [ -n "$LOCK_MHZ" ]; then
    trap "sudo -n nvidia-smi -i $U -rgc >> $O/lock.txt 2>&1" EXIT
    sudo -n nvidia-smi -i "$U" -lgc "$LOCK_MHZ,$LOCK_MHZ" > "$O/lock.txt" 2>&1
    echo "lock $LOCK_MHZ MHz: nvidia-smi -lgc exit $?" > "$O/clock.txt"
  else
    echo "unlocked" > "$O/clock.txt"
  fi
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader \
    -lms 200 -i "$U" > "$O/smi.csv" 2>&1 &
  smi=$!
  # shellcheck disable=SC2086
  "$PY" "$D/hm96_rows.py" time --lib "$LIB" --model "$MODEL" --m $MS --out "$O" > "$O/hm96.log" 2>&1
  echo "hm96 rc $?" >> "$O/rc.txt"
  kill "$smi"'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -20 "$O/hm96.log" 2>/dev/null
! grep -qv " rc 0$" "$O/rc.txt"
