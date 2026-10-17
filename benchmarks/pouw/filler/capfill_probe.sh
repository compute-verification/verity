#!/usr/bin/env bash
# FILLER (note:20261017T0652Z-draft-filler): the certified share σ beside a real decode step. Builds dcap as the leader
# pumps (pfill-bU, duo, -DPF_PW=4) at each PF_LANE in CF_LANES, with LF_CH = CF_CH for the lane filler; then in one
# gpu-lease: dgate at every PF_LANE (the pump's receipts with its lane chains, bit for bit) and dalone 8 2 256 (the
# pump's MMA rate with the chains), lgate and an lalone sweep (the lane filler's issue rate alone), and lfill at each
# decode point at the leader's split (the step beside the pump, then beside the pump and the lane filler), with ncu's
# scheduler and instruction sections for dfill_k at each PF_LANE. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfill.XXXXXX); trap 'rm -rf "$B"' EXIT
export CF_LANES=${CF_LANES:-"0 4 8 16"} CF_CH=${CF_CH:-8} CF_LFILL_LANES=${CF_LFILL_LANES:-"0 8 16"}
export CF_POINTS=${CF_POINTS:-"1:2048:140 32:2048:140 128:2048:116 256:2048:60"}
export CF_BLK=${CF_BLK:-"1 2 4"} CF_T=${CF_T:-128} CF_LSTEPS=${CF_LSTEPS:-1024}
export CF_ALONE=${CF_ALONE:-"128:1 128:2 128:4 256:4 256:8"} CF_NOPUMP=${CF_NOPUMP:-"1:2048 32:2048"}
export DFILL_CHUNK_LOG2=19
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CF_LANES CF_CH CF_LFILL_LANES CF_POINTS CF_BLK CF_T CF_LSTEPS CF_ALONE CF_NOPUMP; do
  echo "$v=${!v}"
done > "$O/params.txt"

pids=()
for k in $CF_LANES; do
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 -DPF_LANE="$k" -DLF_CH="$CF_CH" -Xcompiler -mpopcnt \
    -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap_$k" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
    > "$O/build-L$k.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for k in $CF_LANES; do
  [ -x "$B/dcap_$k" ] || { echo "build PF_LANE=$k failed"; grep -E "error" "$O/build-L$k.log" | head -20; bad=1; }
  grep -A2 -E "dfill_k|lfill_k" "$O/build-L$k.log" | grep -E "registers|spill" > "$O/regs-L$k.txt" 2>/dev/null
done
[ "$bad" = 0 ] || exit 4
k0=${CF_LANES%% *}

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for k in $CF_LANES; do
    "$B/dcap_$k" dgate > "$O/dgate-L$k.jsonl" 2>> "$O/errors.txt"; echo "dgate L=$k rc $?" >> "$O/rc.txt"
    for rep in 1 2; do
      "$B/dcap_$k" dalone 8 2 256 >> "$O/dalone-L$k.jsonl" 2>> "$O/errors.txt" || echo "dalone L=$k rc $?" >> "$O/rc.txt"
    done
  done
  "$B/dcap_'"$k0"'" lgate > "$O/lgate.jsonl" 2>> "$O/errors.txt"; echo "lgate rc $?" >> "$O/rc.txt"
  for tb in $CF_ALONE; do
    "$B/dcap_'"$k0"'" lalone "${tb%%:*}" "$CF_LSTEPS" "${tb##*:}" >> "$O/lalone.jsonl" 2>> "$O/errors.txt" \
      || echo "lalone $tb rc $?" >> "$O/rc.txt"
  done
  for pt in $CF_POINTS; do
    IFS=: read -r b l fs <<< "$pt"
    for k in $CF_LFILL_LANES; do
      for blk in $CF_BLK; do
        "$B/dcap_$k" lfill "$b" "$l" "$fs" 8 2 256 "$CF_T" "$CF_LSTEPS" "$blk" >> "$O/lfill.jsonl" 2>> "$O/errors.txt"
        echo "lfill $b $l $fs L=$k blk=$blk rc $?" >> "$O/rc.txt"
      done
    done
  done
  for pt in $CF_NOPUMP; do
    IFS=: read -r b l <<< "$pt"
    for blk in $CF_BLK; do
      "$B/dcap_'"$k0"'" lfill "$b" "$l" 0 8 2 256 "$CF_T" "$CF_LSTEPS" "$blk" >> "$O/lfill.jsonl" 2>> "$O/errors.txt"
      echo "lfill $b $l 0 blk=$blk rc $?" >> "$O/rc.txt"
    done
  done
  for k in $CF_LANES; do
    ncu --clock-control none -k regex:dfill_k --launch-count 1 --launch-skip 1 --section SchedulerStats \
      --section InstructionStats --section ComputeWorkloadAnalysis --section WarpStateStats \
      "$B/dcap_$k" done 8 2 256 > "$O/ncu-L$k.txt" 2>&1; echo "ncu L=$k rc $?" >> "$O/rc.txt"
  done
  ncu --clock-control none -k regex:lfill_k --launch-count 1 --launch-skip 2 --section SchedulerStats \
    --section InstructionStats --section ComputeWorkloadAnalysis "$B/dcap_'"$k0"'" lalone "$CF_T" "$CF_LSTEPS" 2 \
    > "$O/ncu-lfill.txt" 2>&1; echo "ncu lfill rc $?" >> "$O/rc.txt"
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== regs ==="; for k in $CF_LANES; do echo "L=$k:"; cat "$O/regs-L$k.txt" 2>/dev/null; done
echo "=== dgate ==="; cut -c1-160 "$O"/dgate-L*.jsonl 2>/dev/null
echo "=== dalone ==="; for k in $CF_LANES; do echo "L=$k:"; cut -c1-220 "$O/dalone-L$k.jsonl" 2>/dev/null; done
echo "=== lalone ==="; cut -c1-260 "$O/lalone.jsonl" 2>/dev/null
echo "=== lfill ==="; cut -c1-300 "$O/lfill.jsonl" 2>/dev/null
exit "$r"
