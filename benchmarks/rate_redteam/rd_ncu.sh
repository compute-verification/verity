#!/usr/bin/env bash
# Why rd's mixed splits count reads above the pure-read rate: single launches (rd one) at k = 3 for splits with the
# encode warps on and off, each run plain and under ncu for the DRAM and L2 bytes, plus the read loop's SASS.
# Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/rd_ncu.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
PY=$(command -v python3 || echo /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)
export PY
export B; B=$(mktemp -d /tmp/rate-redteam-rdncu.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -DWC_SQR=1 -o "$B/rd" $D/rd.cu \
  > "$O/rd.ptxas" 2>&1 || { tail -40 "$O/rd.ptxas"; exit 4; }
cuobjdump -sass "$B/rd" > "$B/rd.sass" 2>&1 && awk '/rd_kernelILi3E/{p=1} p' "$B/rd.sass" | head -400 > "$O/rd-k3.sass"
"$PY" $D/gate.py key --k 3 --out "$B/key3.txt" || exit 5
command -v ncu > "$O/ncu-path.txt" 2>&1
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-15}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,gpu__time_duration.sum
  for s in "1 1" "2 1" "3 1" "8 1" "1 0" "2 0" "3 0"; do
    set -- $s
    "$B/rd" one 3 "$B/key3.txt" $1 $2 16384 40000000 >> "$O/one.log" 2>&1
    if command -v ncu > /dev/null; then
      ncu --metrics $M --csv "$B/rd" one 3 "$B/key3.txt" $1 $2 16384 40000000 > "$O/ncu-rw$1-enc$2.csv" 2>&1
    fi
  done'
