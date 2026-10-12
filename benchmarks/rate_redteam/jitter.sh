#!/usr/bin/env bash
# The red team's deadline-slack probe on one GPU (cwd = the source tree): jitter.cu built outside the lease, then
# under one gpu-lease 1 --wait --max-min MAXMIN (default 10) an 8B-shaped and a 70B-shaped step replayed as a CUDA
# graph. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/jitter.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export B; B=$(mktemp -d /tmp/rate-redteam-jitter.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -o "$B/jitter" benchmarks/rate_redteam/jitter.cu \
  > "$O/jitter.build" 2>&1 || { tail -40 "$O/jitter.build"; exit 4; }
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-10}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  rc=0
  "$B/jitter" 32 8 3000 "$O/jitter-8b.json" >> "$O/jitter.log" 2>&1 || rc=2
  q
  "$B/jitter" 80 64 1000 "$O/jitter-70b.json" >> "$O/jitter.log" 2>&1 || rc=2
  q
  exit $rc'
