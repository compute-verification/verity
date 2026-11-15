#!/usr/bin/env bash
# The write ceilings' GPU half (cwd = the source tree): builds write.cu (bwwrite) and runs every arm on one RTX PRO 6000,
# the queue's GPU (GPU_LEASE_UUID) or one it leases, with the card's clocks, PCIe link and ECC mode before and after and
# nvidia-smi dmon beside it. Outputs in $OUT: build.txt, sass.txt, gpu.txt, dmon.txt, write.jsonl, rc.txt. ARGS go to
# bwwrite (default: 32 GiB on the card, 8 GiB pinned on the host, 7 reps).
#
#   research run --queue --gpus 1 --max-min 30 --cpus 8 --mem-gb 32 --project verity --source . --cwd source \
#     -- bash benchmarks/bandwidth/run_write_gpu.sh
set -uxo pipefail
export O=${OUT:?run under research run}; mkdir -p "$O"
BW=$PWD/benchmarks/bandwidth
export B=$(mktemp -d /tmp/bwwrite.XXXXXX); trap 'rm -rf "$B"' EXIT
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
if ! "$NVCC" -O3 -std=c++17 -arch=sm_120 -o "$B/bwwrite" "$BW/write.cu" > "$O/build.txt" 2>&1; then
  cat "$O/build.txt"; echo 4 > "$O/rc.txt"; exit 4
fi
{ "$NVCC" --version; sha256sum "$B/bwwrite" "$BW/write.cu"; } >> "$O/build.txt"
"$(dirname "$NVCC")/cuobjdump" -sass "$B/bwwrite" | grep -oE "\b(STG|UBLKCP|UTMASTG|LDG)[A-Z0-9._]*" | sort | uniq -c \
  > "$O/sass.txt"

measure() {
  local q=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max
  q=$q,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,ecc.mode.current,power.draw,power.limit,memory.total
  { date -u +%FT%TZ; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv; } > "$O/gpu.txt" 2>&1
  nvidia-smi dmon -i "$GPU_LEASE_UUID" -s pucm -d 1 > "$O/dmon.txt" 2>&1 & local mon=$!
  # shellcheck disable=SC2086
  "$B/bwwrite" ${ARGS:-} > "$O/write.jsonl"; local rc=$?
  kill "$mon"
  { date -u +%FT%TZ; nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv; } >> "$O/gpu.txt" 2>&1
  return $rc
}
export -f measure
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  measure; rc=$?
else
  gpu-lease 1 --wait --max-min 15 -- bash -c measure; rc=$?
fi
echo "$rc" > "$O/rc.txt"
exit "$rc"
