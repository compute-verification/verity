#!/usr/bin/env bash
# dcap.cu's DRAM-free filler on one leased GPU (cwd = the source tree): its receipts gated bit for bit against the host and
# CAP's gate first, under the same lease, and nothing is timed unless both pass; then its rate alone over (W, NC), ncu's DRAM
# and tensor-pipe counters for one chunk, and the decode step beside it on a green split and co-resident at the lowest
# stream priority. The round's coins are drawn once per run (coins.txt). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --send benchmarks/pouw/filler/dfill.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/dfill.sh"'
#
# DF_ALONE ("W:NC ..."), DF_ARMS ("split|co:FILL_SMS:W:NC[:CHUNK_LOG2[:KIND]] ...", KIND as DFILL_KIND), DF_BLS and
# DF_CO_BLS ("B:L ...", for split and co arms), DF_ARMS2 with DF_BLS2 (a second set), DF_S (steps a tile), DF_REPS, NCU
# (0|1), DF_NCU ("W:NC ...", the configs ncu reads), DF_T2 ("FILL_SMS ...": T2's chain, dcap fill ... big, at DF_BLS in
# the same lease, into fill.jsonl), DF_NVCC (more nvcc flags: pfill.cuh's filler is the default; -DDFILL_V1 builds FILL
# v1's). An empty DF_ARMS runs no arms.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/dfill.XXXXXX); trap 'rm -rf "$B"' EXIT
export DF_ALONE=${DF_ALONE:-"4:1 4:2 4:4 8:1 8:2 8:4"}
export DF_ARMS=${DF_ARMS-"split:92:8:4 split:60:8:4 split:44:8:4 split:28:8:4 co:0:4:4 co:0:2:4 co:0:4:4:15"}
export DF_BLS=${DF_BLS:-"1:2048 32:2048 32:8192 128:2048 256:2048"}
export DF_CO_BLS=${DF_CO_BLS:-"32:2048 256:2048"}
export DF_S=${DF_S:-2048} DF_REPS=${DF_REPS:-15} NCU=${NCU:-1} DF_NCU=${DF_NCU:-"4:2 8:4"} DF_NVCC=${DF_NVCC:-}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 $DF_NVCC -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
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
    ncu --query-metrics 2>&1 | grep -iE "^(dram|fbpa)" > "$O/ncu-dram-metrics.txt"
    M=gpu__time_duration.sum,lts__t_bytes.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active
    for a in $DF_NCU; do
      for m in $M dram__throughput.avg.pct_of_peak_sustained_elapsed dram__bytes.sum dram__sectors_read.sum,dram__sectors_write.sum; do
        ncu --clock-control none -k regex:dfill_k --launch-count 3 --metrics $m --csv "$B/dcap" done "${a%:*}" "${a#*:}" "$DF_S" \
          >> "$O/ncu-dfill-w${a%:*}-nc${a#*:}.csv" 2>&1; echo "ncu $a $m rc $?" >> "$O/rc.txt"
      done
    done
    # positive control for the DRAM counters: bw_read streams 1 GiB, so dram__bytes.sum should read ≈ 1.07e9 a launch
    DCAP_NOGRAPH=1 ncu --clock-control none -k regex:bw_read --launch-count 3 \
      --metrics gpu__time_duration.sum,dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed --csv \
      "$B/dcap" bw 1 3 > "$O/ncu-bw-control.csv" 2>&1; echo "ncu bw control rc $?" >> "$O/rc.txt"
    L=launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks
    for bl in 32:2048 256:2048; do
      DCAP_NOGRAPH=1 ncu --clock-control none --nvtx --nvtx-include "timed/" --launch-count 200 --metrics $L --csv "$B/dcap" step "${bl%:*}" "${bl#*:}" 1 \
        > "$O/ncu-step-${bl%:*}-${bl#*:}.csv" 2>&1; echo "ncu step $bl rc $?" >> "$O/rc.txt"
    done
  fi
  arms() {  # $1: arms, $2: split batches
    for arm in $1; do
      IFS=: read -r place f w nc lg kind <<< "$arm"
      bls=$2; [ "$place" = co ] && bls=$DF_CO_BLS
      for bl in $bls; do
        DFILL_CHUNK_LOG2=${lg:-19} DFILL_KIND=${kind:-2} "$B/dcap" dfill "${bl%:*}" "${bl#*:}" "$place" "$f" "$w" "$nc" "$DF_S" \
          "$DF_REPS" >> "$O/dfill.jsonl" 2>> "$O/errors.txt" || echo "rc $? for dfill $bl $arm" >> "$O/errors.txt"
      done
      q
    done
  }
  arms "$DF_ARMS" "$DF_BLS"
  arms "${DF_ARMS2:-}" "${DF_BLS2:-}"
  for f in ${DF_T2:-}; do
    for bl in $DF_BLS; do
      "$B/dcap" fill "${bl%:*}" "${bl#*:}" "$f" big "$DF_REPS" >> "$O/fill.jsonl" 2>> "$O/errors.txt" ||
        echo "rc $? for fill $bl $f big" >> "$O/errors.txt"
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
