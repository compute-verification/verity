#!/usr/bin/env bash
# The filler kernel's arms (HC-KERN) on one leased GPU (cwd = the source tree): dcap's d* modes with DFILL_MAP 0 (FILL v1)
# and 1 (the accumulator quad in register order). The receipt gate (both maps) and CAP's gate run first under the same
# lease, and nothing is timed unless both pass. Then each arm's rate alone, a run of seconds per arm for board power, ncu's
# tensor-pipe and issue counters, and the decode step beside each map on a green split. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=HC-KERN \
#     --send benchmarks/pouw/decode_capacity/hckern.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/hckern.sh"'
#
# HK_ALONE ("MAP:W:NC[:S[:CHUNK_LOG2[:PIPE]]] ..."), HK_POWER (the same, DFILL_ALONE_N=HK_POWER_N chunks each), HK_NCU (the same),
# HK_ARMS ("MAP:FILL_SMS:W:NC[:S[:CHUNK_LOG2[:PIPE]]] ...") at HK_BLS ("B:L ..."), HK_S (the default steps a tile), HK_REPS,
# HK_ALONE_N (chunks a dalone times; an empty list is " ").
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/decode_capacity
export B; B=$(mktemp -d /tmp/hckern.XXXXXX); trap 'rm -rf "$B"' EXIT
export HK_ALONE=${HK_ALONE:-"0:4:1 0:4:2 0:4:4 0:8:2 0:8:4 1:4:1 1:4:2 1:4:4 1:4:8 1:8:1 1:8:2 1:8:4 1:8:8 1:2:4 1:2:8 1:1:8 1:16:2 1:16:4"}
export HK_POWER=${HK_POWER:-"0:8:4 1:8:4 1:4:4"} HK_POWER_N=${HK_POWER_N:-2500}
export HK_NCU=${HK_NCU:-"0:8:4 1:8:4 1:4:2 1:4:4"}
export HK_ARMS=${HK_ARMS:-"0:124:8:4 1:124:8:4 0:60:8:4 1:60:8:4"}
export HK_BLS=${HK_BLS:-"32:8192 256:2048"}
export HK_S=${HK_S:-2048} HK_REPS=${HK_REPS:-15}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
cuobjdump -sass "$B/dcap" | awk '/Function : /{p = /dfill_k/} p' > "$B/sass.txt" 2>&1
awk '/Function : /{f=$3} /IMMA/{i[f]++} /[^U]MOV /{m[f]++} /LDL|STL/{l[f]++} END{for (k in i) printf "%s IMMA %d MOV %d local %d\n", k, i[k], m[k], l[k]}' \
  "$B/sass.txt" | sort > "$O/sass-counts.txt"
grep -A2 "Compiling entry function '_Z7dfill" "$O/build.log" | grep -E "Compiling|Used|spill" > "$O/regs.txt"
cat "$O/sass-counts.txt"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  "$B/dcap" dgate > "$O/dgate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "dgate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  "$B/dcap" gate > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "gate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  for a in $HK_ALONE; do
    IFS=: read -r m w nc st lg pipe <<< "$a"
    DFILL_MAP=$m DFILL_PIPE=${pipe:-0} DFILL_CHUNK_LOG2=${lg:-19} DFILL_ALONE_N=${HK_ALONE_N:-40} "$B/dcap" dalone "$w" "$nc" "${st:-$HK_S}" >> "$O/dalone.jsonl" 2>> "$O/errors.txt" ||
      echo "rc $? for dalone $a" >> "$O/errors.txt"
  done
  q
  for a in $HK_POWER; do
    IFS=: read -r m w nc st lg pipe <<< "$a"
    ( sleep 1.5; q; sleep 0.5; q ) &
    DFILL_MAP=$m DFILL_PIPE=${pipe:-0} DFILL_CHUNK_LOG2=${lg:-19} DFILL_ALONE_N=$HK_POWER_N "$B/dcap" dalone "$w" "$nc" "${st:-$HK_S}" >> "$O/power.jsonl" 2>> "$O/errors.txt" ||
      echo "rc $? for power $a" >> "$O/errors.txt"
    wait
    echo "power $a done" >> "$O/gpu.txt"
  done
  M=gpu__time_duration.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,sm__cycles_elapsed.avg,lts__t_bytes.sum,launch__registers_per_thread
  for a in $HK_NCU; do
    IFS=: read -r m w nc st lg pipe <<< "$a"
    DFILL_MAP=$m DFILL_PIPE=${pipe:-0} DFILL_CHUNK_LOG2=${lg:-19} ncu --clock-control none -k regex:dfill_k --launch-count 2 --metrics $M --section WarpStateStats \
      --csv "$B/dcap" done "$w" "$nc" "${st:-$HK_S}" > "$O/ncu-m$m-w$w-nc$nc-s${st:-$HK_S}-p${pipe:-0}.csv" 2>&1; echo "ncu $a rc $?" >> "$O/rc.txt"
  done
  q
  for arm in $HK_ARMS; do
    IFS=: read -r m f w nc st lg pipe <<< "$arm"
    for bl in $HK_BLS; do
      DFILL_MAP=$m DFILL_PIPE=${pipe:-0} DFILL_CHUNK_LOG2=${lg:-19} "$B/dcap" dfill "${bl%:*}" "${bl#*:}" split "$f" "$w" "$nc" "${st:-$HK_S}" "$HK_REPS" \
        >> "$O/dfill.jsonl" 2>> "$O/errors.txt" ||
        echo "rc $? for dfill $bl $arm" >> "$O/errors.txt"
    done
    q
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/dgate.jsonl" "$O/rc.txt" 2>/dev/null
cut -c1-260 "$O/dalone.jsonl" "$O/power.jsonl" 2>/dev/null
grep -v "^device" "$O/errors.txt" 2>/dev/null | tail -5
exit "$r"
