#!/usr/bin/env bash
# dcap.cu on one leased GPU (cwd = the source tree): build outside the lease; under one lease the gate, then DRAM read
# bandwidth, the 32-layer linear step at M = 1 … 8192, and decode attention and the full decode step over a (batch,
# context) grid. Nothing is timed unless the gate passes. Clocks are read, never set. Results in $OUT/dcap.jsonl.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --send benchmarks/pouw/filler/dcap.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/dcap.sh"'
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/dcap.XXXXXX); trap 'rm -rf "$B"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
grep -E "spill|Used" "$O/build.log" | head -20
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  r() { "$B/dcap" "$@" >> "$O/dcap.jsonl" 2>> "$O/errors.txt" ||
        { echo "rc $? for $* (graph); retrying without graphs" >> "$O/errors.txt";
          DCAP_NOGRAPH=1 "$B/dcap" "$@" >> "$O/dcap.jsonl" 2>> "$O/errors.txt" || echo "rc $? for $*" >> "$O/errors.txt"; }; }
  q
  "$B/dcap" gate > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "gate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  r bw 4 9
  r bw 16 5
  for m in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192; do r lin $m 7; done
  q
  for bl in 1:2048 1:8192 1:32768 8:2048 8:8192 8:32768 32:2048 32:8192 64:2048 64:8192 128:2048 128:4096 256:2048 256:3072; do
    r att "${bl%:*}" "${bl#*:}" 7
    r step "${bl%:*}" "${bl#*:}" 7
  done
  q
'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/gate.jsonl" "$O/rc.txt"
tail -5 "$O/errors.txt" 2>/dev/null
