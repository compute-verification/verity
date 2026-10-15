#!/usr/bin/env bash
# sqoverlap.cu on one GPU (cwd = the source tree): the 1,536-bit squaring beside a streaming read on two streams, the
# squaring's work fixed at 0.93 of the read's time (the 1,536-bit decode of E against its read), placed one or two
# blocks a SM against one, two or four read blocks a SM; 768 bits at 0.55 (today's decode) as the case that hides.
# The squaring is gated against Python first. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/sqoverlap.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
export B; B=$(mktemp -d /tmp/rate-redteam-ov.XXXXXX); trap 'rm -rf "$B"' EXIT
for b in 1536 768; do
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -DSQ_BITS=$b -I$D -o "$B/sqoverlap-$b" \
    $D/sqoverlap.cu > "$O/build-$b.log" 2>&1 || { tail -40 "$O/build-$b.log"; exit 4; }
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSQ_BITS=$b -o "$B/sqfloor-$b" $D/sqfloor.cu \
    >> "$O/build-$b.log" 2>&1 || { tail -40 "$O/build-$b.log"; exit 4; }
done
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-20}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/sqfloor-1536" clock > "$O/clock-before.json" 2>&1
  for b in 1536 768; do
    for k in 1 777 1000; do
      "$B/sqoverlap-$b" gate $k 512 > "$B/gate-$b-$k.txt" 2>> "$O/errors.txt" || echo "gate $b $k rc $?" >> "$O/errors.txt"
      python3 $D/sqfloor_gate.py < "$B/gate-$b-$k.txt" > "$O/gate-b$b-k$k.json" 2>&1
    done
  done
  for s in 2 1; do
    for r in 1 2 4; do
      "$B/sqoverlap-1536" run $s $r 0.93 5 >> "$O/overlap.jsonl" 2>> "$O/errors.txt" || echo "run 1536 $s $r rc $?" >> "$O/errors.txt"
    done
  done
  for s in 4 2; do
    for r in 1 2; do
      "$B/sqoverlap-768" run $s $r 0.55 5 >> "$O/overlap.jsonl" 2>> "$O/errors.txt" || echo "run 768 $s $r rc $?" >> "$O/errors.txt"
    done
  done
  "$B/sqfloor-1536" clock > "$O/clock-after.json" 2>&1
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'
