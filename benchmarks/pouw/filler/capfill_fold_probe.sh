#!/usr/bin/env bash
# What the fold and the step's own SMs are worth beside the step, on one GPU.
# (1) The pump at the leader's split with each fold: duo (NC 2), and the r9 folds XIN13 and LI (NC 4, provisional),
# at S 1024, alone (dalone) and beside the step at each decode point (dcap lfill's pump arm; its lane arm is the lane
# filler on the step's SMs). duo at S 256 is the control. (2) The pump on every SM at the
# lowest stream priority beside the step at the highest (dcap dfill co), at chunks short enough (DFILL_CHUNK_LOG2, MMAs an
# SM a chunk) that the step's blocks get SMs between the pump's: certified MMAs on the step's SMs during the step without
# fusing them into its kernels. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfillf.XXXXXX); trap 'rm -rf "$B"' EXIT
export CP_POINTS=${CP_POINTS:-"1:2048:140 32:2048:140 128:2048:116 256:2048:60"}
export CP_SPLIT=${CP_SPLIT:-"duo:2:256 duo:2:1024 xin13:4:1024 li:4:1024"}
export CP_CO=${CP_CO:-"duo:2:256:12 duo:2:256:14 duo:2:256:16 xin13:4:256:13 xin13:4:256:16"}
export CP_T=${CP_T:-128} CP_LSTEPS=${CP_LSTEPS:-1024}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CP_POINTS CP_SPLIT CP_CO CP_T CP_LSTEPS; do echo "$v=${!v}"; done > "$O/params.txt"

declare -A FLAGS=([duo]="" [xin13]="-DPFILL_XIN13" [li]="-DPFILL_LI")
pids=()
for b in duo xin13 li; do
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 ${FLAGS[$b]} -Xcompiler -mpopcnt -gencode arch=compute_120a,code=sm_120a \
    -Xptxas -v -o "$B/dcap_$b" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl > "$O/build-$b.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for b in duo xin13 li; do
  [ -x "$B/dcap_$b" ] || { echo "build $b failed"; grep -E "error" "$O/build-$b.log" | head -20; bad=1; }
done
[ "$bad" = 0 ] || exit 4

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for b in duo xin13 li; do
    "$B/dcap_$b" dgate > "$O/dgate-$b.jsonl" 2>> "$O/errors.txt"; r=$?; echo "dgate $b rc $r" >> "$O/rc.txt"
    [ "$r" = 0 ] || exit 5
  done
  export DFILL_CHUNK_LOG2=19
  for c in $CP_SPLIT; do
    IFS=: read -r b nc s <<< "$c"
    "$B/dcap_$b" dalone 8 "$nc" "$s" 2>> "$O/errors.txt" | sed "s/^{/{\"build\": \"$b\", /" >> "$O/dalone.jsonl"
  done
  for pt in $CP_POINTS; do
    IFS=: read -r bb l fs <<< "$pt"
    for c in $CP_SPLIT; do
      IFS=: read -r b nc s <<< "$c"
      "$B/dcap_$b" lfill "$bb" "$l" "$fs" 8 "$nc" "$s" "$CP_T" "$CP_LSTEPS" 1 2>> "$O/errors.txt" \
        | sed "s/^{\"mode\": \"lfill\"/{\"mode\": \"lfill\", \"build\": \"$b\"/" >> "$O/lfill.jsonl"
      echo "lfill $bb $l $fs $c rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
    done
  done
  for pt in $CP_POINTS; do
    IFS=: read -r bb l fs <<< "$pt"
    for c in $CP_CO; do
      IFS=: read -r b nc s lg <<< "$c"
      DFILL_CHUNK_LOG2=$lg "$B/dcap_$b" dfill "$bb" "$l" co 0 8 "$nc" "$s" 2>> "$O/errors.txt" \
        | sed "s/^{\"mode\": \"dfill\"/{\"mode\": \"dfill\", \"build\": \"$b\", \"chunk_log2\": $lg/" >> "$O/co.jsonl"
      echo "co $bb $l $c rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
    done
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== dgate ==="; cut -c1-200 "$O"/dgate-*.jsonl 2>/dev/null
echo "=== dalone ==="; cut -c1-240 "$O/dalone.jsonl" 2>/dev/null
echo "=== lfill ==="; cut -c1-300 "$O/lfill.jsonl" 2>/dev/null
echo "=== co ==="; cut -c1-400 "$O/co.jsonl" 2>/dev/null
exit "$r"
