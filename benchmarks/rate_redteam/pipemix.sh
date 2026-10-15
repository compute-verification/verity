#!/usr/bin/env bash
# pipemix.cu on one GPU (cwd = the source tree): the 1,536-bit pipe floor's integer mix (IMAD.WIDE : IMAD : IADD3 =
# 12 : 5 : 30) with 0-24 FFMAs an iteration added, and each class alone and in pairs. Gated against the host's replay
# first; the SASS opcode counts of every instantiation are kept. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/pipemix.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
export B; B=$(mktemp -d /tmp/rate-redteam-pipemix.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/pipemix" $D/pipemix.cu \
  > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -DSQ_BITS=1536 -o "$B/sqfloor" $D/sqfloor.cu \
  >> "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
cuobjdump -sass "$B/pipemix" > "$B/sass.txt"
awk '/Function :/ {fn=$3} fn != "" && /^ +\/\*[0-9a-f]+\*\// {
       op=$2; sub(/;$/, "", op); if (op ~ /^@/) op=$3; split(op, p, "."); base=p[1];
       if (op ~ /^IMAD\.WIDE/) base="IMAD.WIDE"; n[fn" "base]++ }
     END {for (k in n) print k, n[k]}' "$B/sass.txt" | sort > "$O/sass-counts.txt"
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-15}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/sqfloor" clock > "$O/clock-before.json" 2>&1
  "$B/pipemix" gate > "$O/gate.jsonl" 2>> "$O/errors.txt" || { echo "gate rc $?" >> "$O/errors.txt"; exit 5; }
  for rep in 1 2 3; do
    for cfg in "12 5 30 0" "12 5 30 4" "12 5 30 8" "12 5 30 12" "12 5 30 16" "12 5 30 24" "0 0 0 16" "0 0 0 32" \
               "16 0 0 0" "16 0 0 16" "0 16 0 0" "0 16 0 16" "0 0 32 0" "0 0 32 16" "12 5 0 0" "0 0 30 0"; do
      "$B/pipemix" time $cfg 200000 >> "$O/time.jsonl" 2>> "$O/errors.txt" || echo "time $cfg rc $?" >> "$O/errors.txt"
    done
  done
  "$B/sqfloor" clock > "$O/clock-after.json" 2>&1
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'
