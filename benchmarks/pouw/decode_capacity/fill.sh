#!/usr/bin/env bash
# dcap.cu's fill mode on one leased GPU (cwd = the source tree): the decode step beside T2's chained int8 GEMM filler on a
# green-context split, over decode (batch, context) x filler SMs x chain shape (big: m = k = 8192, l2: m = k = 4096). The
# gate runs first, under the same lease, and nothing is timed unless it passes. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=CAP \
#     --send benchmarks/pouw/decode_capacity/fill.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/fill.sh"'
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/decode_capacity
export B; B=$(mktemp -d /tmp/dcap.XXXXXX); trap 'rm -rf "$B"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  "$B/dcap" gate > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "gate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  for shape in big l2; do
    for f in 60 92; do
      for bl in 1:2048 32:2048 32:8192 128:2048 256:2048; do
        "$B/dcap" fill "${bl%:*}" "${bl#*:}" "$f" "$shape" 9 >> "$O/fill.jsonl" 2>> "$O/errors.txt" ||
          echo "rc $? for fill $bl $f $shape" >> "$O/errors.txt"
      done
      q
    done
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/gate.jsonl" "$O/rc.txt"
grep -v "^device" "$O/errors.txt" | tail -5 || true
exit "$r"
