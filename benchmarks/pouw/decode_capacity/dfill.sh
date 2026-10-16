#!/usr/bin/env bash
# dcap.cu's DRAM-free filler on one leased GPU (cwd = the source tree): its receipts gated bit for bit against the host and
# CAP's gate first, under the same lease, and nothing is timed unless both pass; then its rate alone over (W, NC), ncu's DRAM
# and tensor-pipe counters for one chunk, and the decode step beside it on a green split and co-resident at the lowest
# stream priority. The round's coins are drawn once per run (coins.txt). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=FILL \
#     --send benchmarks/pouw/decode_capacity/dfill.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/dfill.sh"'
#
# DF_ALONE ("W:NC ..."), DF_ARMS ("split|co:FILL_SMS:W:NC ..."), DF_BLS ("B:L ..."), DF_S (steps a tile), DF_REPS, NCU (0|1)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/decode_capacity
export B; B=$(mktemp -d /tmp/dfill.XXXXXX); trap 'rm -rf "$B"' EXIT
export DF_ALONE=${DF_ALONE:-"4:1 4:2 4:4 8:1 8:2 8:4"}
export DF_ARMS=${DF_ARMS:-"split:92:4:2 split:60:4:2 co:0:4:1 co:0:4:2"}
export DF_BLS=${DF_BLS:-"1:2048 32:2048 32:8192 128:2048 256:2048"}
export DF_S=${DF_S:-2048} DF_REPS=${DF_REPS:-15} NCU=${NCU:-1}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
grep -B1 -E "spill" "$O/build.log" | grep -A1 dfill | head -12
cuobjdump -sass "$B/dcap" | awk '/Function : /{p = /dfill_k/} p' > "$O/sass-dfill.txt" 2>&1
echo "dfill_k SASS (NC 1, 2, 4): $(grep -c IMMA "$O/sass-dfill.txt") IMMA, $(grep -cE 'LDG|STG|LDS|STS|LDL|STL' "$O/sass-dfill.txt") memory ops"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  "$B/dcap" dgate > "$O/dgate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "dgate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  "$B/dcap" gate > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "gate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  for a in $DF_ALONE; do
    "$B/dcap" dalone "${a%:*}" "${a#*:}" "$DF_S" >> "$O/dalone.jsonl" 2>> "$O/errors.txt" || echo "rc $? for dalone $a" >> "$O/errors.txt"
  done
  q
  if [ "$NCU" = 1 ]; then
    M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active
    for a in 4:2 8:4; do
      ncu --clock-control none -k regex:dfill_k --launch-count 3 --metrics $M --csv "$B/dcap" done "${a%:*}" "${a#*:}" "$DF_S" \
        > "$O/ncu-dfill-w${a%:*}-nc${a#*:}.csv" 2>&1; echo "ncu $a rc $?" >> "$O/rc.txt"
    done
  fi
  for arm in $DF_ARMS; do
    IFS=: read -r place f w nc <<< "$arm"
    for bl in $DF_BLS; do
      "$B/dcap" dfill "${bl%:*}" "${bl#*:}" "$place" "$f" "$w" "$nc" "$DF_S" "$DF_REPS" >> "$O/dfill.jsonl" 2>> "$O/errors.txt" ||
        echo "rc $? for dfill $bl $arm" >> "$O/errors.txt"
    done
    q
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/dgate.jsonl" "$O/rc.txt" 2>/dev/null
cut -c1-300 "$O/dalone.jsonl" 2>/dev/null
grep -v "^device" "$O/errors.txt" 2>/dev/null | tail -5
exit "$r"
