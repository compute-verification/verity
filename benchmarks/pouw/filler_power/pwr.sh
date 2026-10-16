#!/usr/bin/env bash
# pwr.cu on one leased GPU (cwd = the source tree): the filler variants' receipts against the host first, under the same
# lease, and nothing is measured unless they pass; then the idle board, the filler alone per (SMs, kind, pattern), and the
# decode step beside it per suite point, every arm sustained (PW_WARM s unmeasured, then PW_MEAS s) with NVML's energy
# counter, instant power, SM clock and clock-event reasons sampled every 5 ms. The round's coins are drawn once per run
# (coins.txt). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=HC-POWER \
#     --send benchmarks/pouw/filler_power/pwr.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/pwr.sh"'
#
# PW_ALONE ("FILL_SMS:KIND:PAT:W:NC[:SLEEP_NS] ..."), PW_POINTS ("B:L:FILL_SMS:KIND:PAT:W:NC[:SLEEP_NS] ..."), PW_CTL (a
# control pattern run beside the step at each point, e.g. 3; empty for none), PW_WARM, PW_MEAS, PW_ROUNDS; PW_PREFILL
# ("M:B:L:FILL_SMS:KIND:PAT:W:NC ...": prefill after heat, pwr.cu's pprefill) with PW_HEAT, PW_PRE_MEAS, PW_PRE_ROUNDS
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler_power
export B; B=$(mktemp -d /tmp/pwr.XXXXXX); trap 'rm -rf "$B"' EXIT
export PW_ALONE=${PW_ALONE:-"188:0:0:8:4 188:1:0:8:4 188:2:0:8:4 188:0:1:8:4 188:0:2:8:4 188:0:4:8:4 188:0:3:8:4 108:0:0:8:4"}
export PW_POINTS=${PW_POINTS:-"1:2048:108:0:0:8:4 32:8192:124:0:0:8:4 256:2048:60:0:0:8:4"}
export PW_CTL=${PW_CTL:-3} PW_WARM=${PW_WARM:-1.5} PW_MEAS=${PW_MEAS:-3} PW_ROUNDS=${PW_ROUNDS:-2}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/pwr" $D/pwr.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
grep -A2 "pfill_k" "$O/build.log" | grep -E "spill|registers" | sort | uniq -c | head -12
cuobjdump -sass "$B/pwr" | awk '/Function : /{p = /pfill_k/} p' > "$O/sass-pfill.txt" 2>&1
echo "pfill_k SASS: $(grep -cE 'IMMA|QMMA|OMMA' "$O/sass-pfill.txt") MMA, $(grep -cE 'LDG|LDS|LDL|STL' "$O/sass-pfill.txt") loads/local"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  nvidia-smi -q -d POWER,CLOCK,PERFORMANCE -i "$GPU_LEASE_UUID" > "$O/smi-q.txt" 2>&1
  "$B/pwr" pgate > "$O/pgate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "pgate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  "$B/pwr" pidle 3 >> "$O/arms.jsonl" 2>> "$O/errors.txt"
  for a in $PW_ALONE; do
    IFS=: read -r f k p w nc sl <<< "$a"
    PF_SLEEP_NS=${sl:-0} "$B/pwr" palone "$f" "$k" "$p" "$w" "$nc" "$PW_WARM" "$PW_MEAS" >> "$O/arms.jsonl" 2>> "$O/errors.txt" ||
      echo "rc $? for palone $a" >> "$O/errors.txt"
  done
  q
  for a in $PW_POINTS; do
    IFS=: read -r b l f k p w nc sl <<< "$a"
    PF_SLEEP_NS=${sl:-0} "$B/pwr" pfill "$b" "$l" "$f" "$k" "$p" "$w" "$nc" "$PW_WARM" "$PW_MEAS" "$PW_ROUNDS" ${PW_CTL} \
      >> "$O/arms.jsonl" 2>> "$O/errors.txt" || echo "rc $? for pfill $a" >> "$O/errors.txt"
    q
  done
  for a in ${PW_PREFILL:-}; do
    IFS=: read -r m b l f k p w nc <<< "$a"
    "$B/pwr" pprefill "$m" "$b" "$l" "$f" "$k" "$p" "$w" "$nc" "${PW_HEAT:-8}" "${PW_PRE_MEAS:-2}" "${PW_PRE_ROUNDS:-2}" \
      >> "$O/arms.jsonl" 2>> "$O/errors.txt" || echo "rc $? for pprefill $a" >> "$O/errors.txt"
    q
  done
  "$B/pwr" pidle 3 >> "$O/arms.jsonl" 2>> "$O/errors.txt"
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/pgate.jsonl" "$O/rc.txt" 2>/dev/null | cut -c1-200
grep -v "^device" "$O/errors.txt" 2>/dev/null | tail -5
exit "$r"
