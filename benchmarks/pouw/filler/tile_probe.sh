#!/usr/bin/env bash
# Issue slots an MMA, on one leased GPU (cwd = the source tree): for each pfill build in TP_BUILDS, dgate, then the pump's
# rate alone at each of its TP_NC folds (dcap dalone 8 NC S) and dfill_k_warp's alone (dcap walone S), then ncu over one
# launch of each (dcap done's second chunk, walone's first timed launch) with the instruction, scheduler and warp-state
# sections, kept as reports and exported (details, and the raw page with sass__inst_executed_per_opcode's instances), so
# slots an MMA = smsp__inst_executed.sum over the launch's IMMAs. Clocks are read, never set.
#   TP_BUILDS ("name:flags ...", a build's flags joined by commas), TP_NC ("name:NC ..."), TP_W ("name ...": walone),
#   TP_S (the pump's fragments a warp a tile, a list: dalone runs each, ncu the first), TP_WS (walone's steps a warp;
#   ncu takes TP_WS_NCU), TP_NCU (0: no ncu).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/tprobe.XXXXXX); trap 'rm -rf "$B"' EXIT
export TP_BUILDS=${TP_BUILDS:-"bits:-DPFILL_BITS bu:-DPFILL_BU,-Xcompiler,-mpopcnt li:-DPFILL_BU,-DPFILL_LI,-Xcompiler,-mpopcnt xin13:-DPFILL_BU,-DPFILL_XIN13,-Xcompiler,-mpopcnt"}
export TP_NC=${TP_NC:-"bits:4 bu:4 bu:2 li:4 xin13:4"} TP_W=${TP_W-"bits"} TP_S=${TP_S:-256} TP_WS=${TP_WS:-2048}
export TP_WS_NCU=${TP_WS_NCU:-64} TP_NCU=${TP_NCU:-1}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for b in $TP_BUILDS; do
  n=${b%%:*} f=${b#*:}
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 ${f//,/ } -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap-$n" $D/dcap.cu \
    -lcublasLt -lnvidia-ml -ldl > "$O/build-$n.log" 2>&1 &
done
wait
for b in $TP_BUILDS; do
  n=${b%%:*}
  [ -x "$B/dcap-$n" ] || { grep -E "error" "$O/build-$n.log" | head -20; exit 4; }
  cuobjdump -sass "$B/dcap-$n" | awk '/Function : /{p = /dfill_k/} p' > "$O/sass-$n.txt" 2>&1
done
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for b in $TP_BUILDS; do
    n=${b%%:*}
    "$B/dcap-$n" dgate > "$O/dgate-$n.jsonl" 2>> "$O/errors.txt" || { echo "dgate $n rc $?" >> "$O/rc.txt"; exit 5; }
  done
  for a in $TP_NC; do
    n=${a%%:*} nc=${a#*:}
    for s in $TP_S; do
      "$B/dcap-$n" dalone 8 "$nc" "$s" 2>> "$O/errors.txt" | sed "s/^{/{\"build\": \"$n\", /" >> "$O/dalone.jsonl"
    done
  done
  for n in $TP_W; do
    "$B/dcap-$n" walone "$TP_WS" 2>> "$O/errors.txt" | sed "s/^{/{\"build\": \"$n\", /" >> "$O/walone.jsonl"
  done
  [ "$TP_NCU" = 1 ] || exit 0
  s0=${TP_S%% *}
  sec=(--section InstructionStats --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis)
  report() {
    ncu --import "$1.ncu-rep" --page details > "$1.txt" 2>&1
    ncu --import "$1.ncu-rep" --page raw --csv --print-metric-instances values > "$1-raw.csv" 2>&1
  }
  for a in $TP_NC; do
    n=${a%%:*} nc=${a#*:} r="$O/ncu-pump-$n-nc$nc"
    ncu --clock-control none -k regex:dfill_k --launch-count 1 --launch-skip 1 "${sec[@]}" -o "$r" \
      "$B/dcap-$n" done 8 "$nc" "$s0" > "$r.log" 2>&1
    echo "ncu pump $a rc $?" >> "$O/rc.txt"; report "$r"
  done
  for n in $TP_W; do
    r="$O/ncu-warp-$n"
    ncu --clock-control none -k regex:kw_alone --launch-count 1 --launch-skip 1 "${sec[@]}" -o "$r" \
      "$B/dcap-$n" walone "$TP_WS_NCU" 1 > "$r.log" 2>&1
    echo "ncu warp $n rc $?" >> "$O/rc.txt"; report "$r"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O"/dgate-*.jsonl "$O/rc.txt" 2>/dev/null | cut -c1-200; cut -c1-260 "$O/dalone.jsonl" "$O/walone.jsonl" 2>/dev/null
exit "$r"
