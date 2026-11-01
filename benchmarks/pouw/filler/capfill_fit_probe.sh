#!/usr/bin/env bash
# Why a 1-block pump that leaves half of every SM free still holds the step
# off (capfill_co_probe.sh: 7-37x served time, the same at 4 and 24 KB of pad). ncu's launch statistics (registers, shared
# memory, block size) of every kernel of the step at B 1, 32 and 256; the step's linears alone and its attention alone
# (DFILL_KIND 0, 1) beside the 1-block pump co-scheduled on every SM; and the 2-block pump at true short chunks (S 32,
# DFILL_CHUNK_LOG2 16: one 2^16-MMA tile an SM, ~0.2 ms). Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfillt.XXXXXX); trap 'rm -rf "$B"' EXIT
export CP_POINTS=${CP_POINTS:-"1:2048 32:2048 128:2048 256:2048"} CP_NCU=${CP_NCU:-"1:2048 32:2048 256:2048"}
[ "$CP_NCU" = none ] && CP_NCU=""
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CP_POINTS CP_NCU; do echo "$v=${!v}"; done > "$O/params.txt"

declare -A FLAGS=([c2duo]="" [c1duo]="-DPF_CTAS=1 -DPF_MINB=2")
pids=()
for b in c2duo c1duo; do
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 ${FLAGS[$b]} -Xcompiler -mpopcnt -gencode arch=compute_120a,code=sm_120a \
    -Xptxas -v -o "$B/dcap_$b" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl > "$O/build-$b.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for b in c2duo c1duo; do
  [ -x "$B/dcap_$b" ] || { echo "build $b failed"; grep -E "error" "$O/build-$b.log" | head -20; bad=1; }
done
[ "$bad" = 0 ] || exit 4

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for pt in $CP_NCU; do
    IFS=: read -r bb l <<< "$pt"
    DCAP_NOGRAPH=1 ncu --clock-control none --section LaunchStats --launch-count 400 --csv \
      "$B/dcap_c2duo" step "$bb" "$l" 1 > "$O/ncu-step-$bb-$l.csv" 2>> "$O/errors.txt"
    echo "ncu step $bb $l rc $?" >> "$O/rc.txt"
  done
  for pt in 1:2048 32:2048 256:2048; do
    IFS=: read -r bb l <<< "$pt"
    for k in 0 1; do
      DFILL_KIND=$k DFILL_PAD_KB=4 "$B/dcap_c1duo" dfill "$bb" "$l" co 0 8 2 256 2>> "$O/errors.txt" \
        | sed "s/^{\"mode\": \"dfill\"/{\"mode\": \"dfill\", \"build\": \"c1duo\", \"pad_kb\": 4, \"kind\": $k/" >> "$O/co.jsonl"
      echo "co kind $k $bb $l rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
    done
  done
  DFILL_CHUNK_LOG2=16 "$B/dcap_c2duo" dalone 8 2 32 2>> "$O/errors.txt" | sed "s/^{/{\"build\": \"c2duo\", /" >> "$O/dalone.jsonl"
  for pt in $CP_POINTS; do
    IFS=: read -r bb l <<< "$pt"
    DFILL_CHUNK_LOG2=16 "$B/dcap_c2duo" dfill "$bb" "$l" co 0 8 2 32 2>> "$O/errors.txt" \
      | sed "s/^{\"mode\": \"dfill\"/{\"mode\": \"dfill\", \"build\": \"c2duo\", \"pad_kb\": 0, \"chunk_log2\": 16/" >> "$O/co.jsonl"
    echo "co short $bb $l rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== dalone ==="; cut -c1-240 "$O/dalone.jsonl" 2>/dev/null
echo "=== co ==="; cut -c1-400 "$O/co.jsonl" 2>/dev/null
exit "$r"
