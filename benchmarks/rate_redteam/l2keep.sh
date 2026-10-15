#!/usr/bin/env bash
# l2keep.cu on one GPU (cwd = the source tree): a sketch-sized buffer written, S MiB of weights streamed, the buffer read
# back, timed and under ncu (the read-back's DRAM bytes, caches not flushed by ncu). Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/l2keep.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
export B; B=$(mktemp -d /tmp/rate-redteam-l2.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -o "$B/l2keep" $D/l2keep.cu > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
cuobjdump -sass "$B/l2keep" | grep -E "Function|LDG|STG|CCTL" > "$O/l2keep-ldst.sass" 2>&1
command -v ncu > "$O/ncu-path.txt" 2>&1
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-15}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M=dram__bytes_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,gpu__time_duration.sum
  run() {
    "$B/l2keep" $1 $2 $3 5 >> "$O/timed.log" 2>&1 || echo "timed $1 $2 $3 rc $?" >> "$O/errors.txt"
    ncu --cache-control none --replay-mode application -k regex:readback -c 1 --metrics $M --csv \
      "$B/l2keep" $1 $2 $3 1 > "$O/ncu-b$1-s$2-h$3.csv" 2>&1 || echo "ncu $1 $2 $3 rc $?" >> "$O/errors.txt"
  }
  for b in 8 32; do
    for s in 0 32 64 96 128 192 256 512 1024; do run $b $s 0; done
    for h in 1 2; do for s in 0 128 256 1024; do run $b $s $h; done; done
  done'
