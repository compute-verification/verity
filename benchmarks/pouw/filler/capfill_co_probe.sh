#!/usr/bin/env bash
# The pump on every SM beside the step, co-resident instead of split. The
# leader's 2-block pump fills an SM's registers, so at the lowest priority it holds every SM for a whole chunk (2^19
# MMAs an SM, ~1.7 ms) and the step waits behind it (capfill_fold_probe.sh: 8-28x served time). Here: a 1-block pump
# (-DPF_CTAS=1 -DPF_MINB=2: half the registers) with DFILL_PAD_KB keeping a second pump block off the SM, so the step's
# blocks fit beside it on every SM at once; and the 2-block pump at short tiles (S 32, ~0.2 ms chunks). dcap dfill co at
# each decode point, CP_CO entries BUILD:NC:S:PAD_KB. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfillc.XXXXXX); trap 'rm -rf "$B"' EXIT
export CP_POINTS=${CP_POINTS:-"1:2048 32:2048 128:2048 256:2048"}
export CP_CO=${CP_CO:-"c1duo:2:256:4 c1duo:2:256:24 c1duo:2:32:4 c1xin13:4:256:4 c1xin13:4:32:4 c2duo:2:32:0"}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CP_POINTS CP_CO; do echo "$v=${!v}"; done > "$O/params.txt"

declare -A FLAGS=([c2duo]="" [c1duo]="-DPF_CTAS=1 -DPF_MINB=2" [c1xin13]="-DPF_CTAS=1 -DPF_MINB=2 -DPFILL_XIN13")
pids=()
for b in c2duo c1duo c1xin13; do
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 ${FLAGS[$b]} -Xcompiler -mpopcnt -gencode arch=compute_120a,code=sm_120a \
    -Xptxas -v -o "$B/dcap_$b" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl > "$O/build-$b.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for b in c2duo c1duo c1xin13; do
  [ -x "$B/dcap_$b" ] || { echo "build $b failed"; grep -E "error" "$O/build-$b.log" | head -20; bad=1; }
done
[ "$bad" = 0 ] || exit 4

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for b in c1duo c1xin13; do
    "$B/dcap_$b" dgate > "$O/dgate-$b.jsonl" 2>> "$O/errors.txt"; r=$?; echo "dgate $b rc $r" >> "$O/rc.txt"
    [ "$r" = 0 ] || exit 5
  done
  for c in $CP_CO; do
    IFS=: read -r b nc s pad <<< "$c"
    DFILL_PAD_KB=$pad "$B/dcap_$b" dalone 8 "$nc" "$s" 2>> "$O/errors.txt" \
      | sed "s/^{/{\"build\": \"$b\", \"pad_kb\": $pad, /" >> "$O/dalone.jsonl"
  done
  for pt in $CP_POINTS; do
    IFS=: read -r bb l <<< "$pt"
    for c in $CP_CO; do
      IFS=: read -r b nc s pad <<< "$c"
      DFILL_PAD_KB=$pad "$B/dcap_$b" dfill "$bb" "$l" co 0 8 "$nc" "$s" 2>> "$O/errors.txt" \
        | sed "s/^{\"mode\": \"dfill\"/{\"mode\": \"dfill\", \"build\": \"$b\", \"pad_kb\": $pad/" >> "$O/co.jsonl"
      echo "co $bb $l $c rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
    done
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== dalone ==="; cut -c1-240 "$O/dalone.jsonl" 2>/dev/null
echo "=== co ==="; cut -c1-400 "$O/co.jsonl" 2>/dev/null
exit "$r"
