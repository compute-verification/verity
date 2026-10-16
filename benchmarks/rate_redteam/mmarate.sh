#!/usr/bin/env bash
# mmarate.cu on one GPU (cwd = the source tree): the u8 mma.m16n8k32's rate alone over occupancy and chains, then
# beside the 1,536-bit squaring at one block a SM (the most that leaves registers) and at two (full). The squaring is
# gated against Python first. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/mmarate.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
export B; B=$(mktemp -d /tmp/rate-redteam-mma.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -DSQ_BITS=1536 -I$D -o "$B/mmarate" \
  $D/mmarate.cu > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSQ_BITS=1536 -I$D -o "$B/sqoverlap" $D/sqoverlap.cu \
  >> "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSQ_BITS=1536 -o "$B/sqfloor" $D/sqfloor.cu \
  >> "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
cuobjdump -sass "$B/mmarate" | grep -c 'IMMA' > "$O/imma-count.txt" 2>&1
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-20}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/sqfloor" clock > "$O/clock-before.json" 2>&1
  for k in 1 1000; do
    "$B/sqoverlap" gate $k 512 > "$B/gate-$k.txt" 2>> "$O/errors.txt" || echo "gate $k rc $?" >> "$O/errors.txt"
    python3 $D/sqfloor_gate.py < "$B/gate-$k.txt" > "$O/gate-b1536-k$k.json" 2>&1
  done
  for cfg in "4 1 1" "4 1 2" "4 1 4" "4 1 8" "8 1 4" "8 1 8" "4 2 8" "8 2 8" "8 4 4" "8 4 8" "16 2 8"; do
    "$B/mmarate" alone $cfg 5 >> "$O/alone.jsonl" 2>> "$O/errors.txt" || echo "alone $cfg rc $?" >> "$O/errors.txt"
  done
  for cfg in "1 4 1 8" "1 4 1 4" "1 8 1 8" "1 4 2 8" "2 4 1 8"; do
    "$B/mmarate" both $cfg 5 >> "$O/both.jsonl" 2>> "$O/errors.txt" || echo "both $cfg rc $?" >> "$O/errors.txt"
  done
  "$B/sqfloor" clock > "$O/clock-after.json" 2>&1
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'
