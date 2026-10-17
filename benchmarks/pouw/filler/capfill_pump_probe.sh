#!/usr/bin/env bash
# FILLER (note:20261017T0652Z-draft-filler): lane warps beside the pump on its own SMs. Lane chains inside the pump's
# warps cost it MMAs one for one (capfill_probe.sh: it is latency-bound, 45% of issue slots idle), so here the pump runs
# PF_CTAS=1 block an SM at the leader's register cap (-DPF_MINB=2, DFILL_PAD_KB keeping a second block off) and lfill's
# blocks take the freed registers and the idle issue slots, away from the step. Builds: c2 (the leader's 2 blocks), c1
# and c1ch4 (1 block, LF_CH 8 and 4). In one gpu-lease: dgate for c1, dalone for each, lalone for c1ch4, then lfill with
# LFILL_ON=pump at each decode point of the leader's split for each lane block count. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfillp.XXXXXX); trap 'rm -rf "$B"' EXIT
export CP_POINTS=${CP_POINTS:-"1:2048:140 32:2048:140 128:2048:116 256:2048:60"} CP_BLK=${CP_BLK:-"1 2 3 4"}
export CP_T=${CP_T:-128} CP_LSTEPS=${CP_LSTEPS:-1024} CP_PAD=${CP_PAD:-24}
export DFILL_CHUNK_LOG2=19
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CP_POINTS CP_BLK CP_T CP_LSTEPS CP_PAD; do echo "$v=${!v}"; done > "$O/params.txt"

declare -A FLAGS=([c2]="" [c1]="-DPF_CTAS=1 -DPF_MINB=2" [c1ch4]="-DPF_CTAS=1 -DPF_MINB=2 -DLF_CH=4")
pids=()
for b in c2 c1 c1ch4; do
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 ${FLAGS[$b]} -Xcompiler -mpopcnt -gencode arch=compute_120a,code=sm_120a \
    -Xptxas -v -o "$B/dcap_$b" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl > "$O/build-$b.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for b in c2 c1 c1ch4; do
  [ -x "$B/dcap_$b" ] || { echo "build $b failed"; grep -E "error" "$O/build-$b.log" | head -20; bad=1; }
  grep -A4 -E "_Z7dfill_kILi2E|lfill_k" "$O/build-$b.log" | grep -E "registers|spill" > "$O/regs-$b.txt" 2>/dev/null
done
[ "$bad" = 0 ] || exit 4

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/dcap_c1" dgate > "$O/dgate-c1.jsonl" 2>> "$O/errors.txt"; echo "dgate c1 rc $?" >> "$O/rc.txt"
  for b in c2 c1; do
    "$B/dcap_$b" dalone 8 2 256 >> "$O/dalone-$b.jsonl" 2>> "$O/errors.txt" || echo "dalone $b rc $?" >> "$O/rc.txt"
  done
  DFILL_PAD_KB=$CP_PAD "$B/dcap_c1" dalone 8 2 256 >> "$O/dalone-c1pad.jsonl" 2>> "$O/errors.txt" \
    || echo "dalone c1pad rc $?" >> "$O/rc.txt"
  "$B/dcap_c1ch4" lgate > "$O/lgate-ch4.jsonl" 2>> "$O/errors.txt"; echo "lgate ch4 rc $?" >> "$O/rc.txt"
  for blk in 1 2 4; do
    "$B/dcap_c1ch4" lalone "$CP_T" "$CP_LSTEPS" "$blk" >> "$O/lalone-ch4.jsonl" 2>> "$O/errors.txt" \
      || echo "lalone ch4 $blk rc $?" >> "$O/rc.txt"
  done
  for pt in $CP_POINTS; do
    IFS=: read -r b l fs <<< "$pt"
    for bld in c1 c1ch4; do
      for blk in $CP_BLK; do
        LFILL_ON=pump DFILL_PAD_KB=$CP_PAD "$B/dcap_$bld" lfill "$b" "$l" "$fs" 8 2 256 "$CP_T" "$CP_LSTEPS" "$blk" 2>> "$O/errors.txt" \
          | sed "s/^{\"mode\": \"lfill\"/{\"mode\": \"lfill\", \"build\": \"$bld\", \"pad_kb\": $CP_PAD/" \
          >> "$O/lfill.jsonl"
        echo "lfill $b $l $fs $bld blk=$blk rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
      done
    done
  done
  for blk in 2 4; do
    LFILL_ON=pump "$B/dcap_c1" lfill 32 2048 140 8 2 256 "$CP_T" "$CP_LSTEPS" "$blk" 2>> "$O/errors.txt" \
      | sed "s/^{\"mode\": \"lfill\"/{\"mode\": \"lfill\", \"build\": \"c1\", \"pad_kb\": 0/" >> "$O/lfill.jsonl"
    echo "lfill 32 2048 140 c1 nopad blk=$blk rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
  done
  LFILL_ON=pump "$B/dcap_c2" lfill 32 2048 140 8 2 256 "$CP_T" "$CP_LSTEPS" 2 2>> "$O/errors.txt" \
    | sed "s/^{\"mode\": \"lfill\"/{\"mode\": \"lfill\", \"build\": \"c2\", \"pad_kb\": 0/" >> "$O/lfill.jsonl"
  echo "lfill 32 2048 140 c2 blk=2 rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== regs ==="; for b in c2 c1 c1ch4; do echo "$b:"; cat "$O/regs-$b.txt" 2>/dev/null; done
echo "=== dalone ==="; cut -c1-220 "$O"/dalone-*.jsonl 2>/dev/null
echo "=== lalone ch4 ==="; cut -c1-260 "$O/lalone-ch4.jsonl" 2>/dev/null
echo "=== lfill ==="; cut -c1-300 "$O/lfill.jsonl" 2>/dev/null
exit "$r"
