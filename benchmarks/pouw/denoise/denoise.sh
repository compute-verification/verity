#!/usr/bin/env bash
# denoise.cu on one leased H200 (sm_90a): the build outside the lease, then every shape under one lease, then table.py.
#
#   research run --on vy-dc-h200-3 --project verity --campaign CAMPAIGN --source . --cwd source --as LANE \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/denoise/denoise.sh
#
# Env: MS (default "64 512 8192"), LINEARS (default every one of Llama-3.2-1B's: qkv o gu down head), REPS (default 5),
# LEASE (default gpu-lease 1 --wait --max-min 40 --; empty when already under a lease).
set -uo pipefail
D=benchmarks/pouw/denoise
O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 \
  -o "$O/denoise" $D/denoise.cu -lcuda -lcublasLt > "$O/build.log" 2>&1 || { tail -30 "$O/build.log"; exit 3; }
declare -A NK=([qkv]=3072:2048 [o]=2048:2048 [gu]=16384:2048 [down]=2048:8192 [head]=128256:2048)
shapes=()
for m in ${MS:-64 512 8192}; do
  for l in ${LINEARS:-qkv o gu down head}; do shapes+=("$l-m$m:$m:${NK[$l]}"); done
done
nvidia-smi --query-gpu=name,uuid,clocks.max.sm,clocks.applications.graphics,power.limit,driver_version --format=csv > "$O/gpu.csv"
${LEASE-gpu-lease 1 --wait --max-min 40 --} "$O/denoise" "${REPS:-5}" "${shapes[@]}" > "$O/denoise.jsonl" 2> "$O/denoise.err"
rc=$?
echo "denoise rc $rc" > "$O/rc.txt"
python3 $D/table.py "$O/denoise.jsonl" > "$O/table.md" 2> "$O/table.err"
echo "table rc $?" >> "$O/rc.txt"
cat "$O/table.md"
tail -5 "$O/denoise.err"
exit $rc
