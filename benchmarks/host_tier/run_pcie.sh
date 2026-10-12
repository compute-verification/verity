#!/usr/bin/env bash
# PCIe between host DRAM and N leased GPUs at once, and host DRAM read by the CPUs and the GPUs together, as one Attempt
# (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/host_tier/run_pcie.sh
#
# Outside the lease: build pcie.cu and dram.c. Under `gpu-lease $NGPU --wait --max-min 20`: pcie's gates (a failed gate times
# nothing, exit 3), then every mode with each GPU's host buffer on its own NUMA node (REPS times) and on the other one; the
# zero-copy read over grids; 4 KiB pages; and the host's DRAM read by CPU threads alone, beside every GPU's H2D copies, and
# beside every GPU's zero-copy reads, timed together (--start-at).
# Env: NGPU (1), REPS (2), SECONDS_EACH (3), MIB (host buffer per GPU, 1024), GRIDS ("188 376 752 1504 3008"),
#      PCIE_CPUS (the pcie process's CPUs, 48-51), DRAM_CPUS (the CPU readers', 52-95).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, env.txt, pcie.jsonl, dram.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"

if [ "${1:-}" = leased ]; then
  B=$2
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem \
    --format=csv > "$O/gpu.txt" 2>&1
  { nvidia-smi topo -m; for b in $(nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=pci.bus_id --format=csv,noheader); do
      d=$(echo "$b" | tr 'A-F' 'a-f' | sed 's/^0000\(....\)/\1/'); echo "$b numa_node $(cat /sys/bus/pci/devices/$d/numa_node)"
      lspci -vvv -s "$d" 2> /dev/null | grep -E 'MaxPayload|MaxReadReq|LnkSta:'; done; } > "$O/env.txt" 2>&1
  S=${SECONDS_EACH:-3} M=${MIB:-1024} PC=${PCIE_CPUS:-48-51} DC=${DRAM_CPUS:-52-95}
  P() { taskset -c "$PC" "$B/pcie" --seconds "$S" --mib "$M" "$@"; }
  tag() { sed "s/^{/{\"label\": \"$1\", \"cuda_visible\": \"$CUDA_VISIBLE_DEVICES\", /" | tee -a "$O/pcie.jsonl"; }
  P --modes h2d --seconds 0.5 | tag gates; [ "${PIPESTATUS[0]}" = 0 ] || { echo "gates failed"; exit 3; }
  for r in $(seq "${REPS:-2}"); do P --host-node local | tag local; done
  P --host-node remote | tag remote
  for g in ${GRIDS:-188 376 752 1504 3008}; do P --host-node local --modes zcr,zcr_d2h --grid "$g" | tag grid; done
  P --host-node local --modes h2d,zcr --thp 0 | tag pages4k
  # host DRAM read by the CPUs alone and with every GPU reading it too, over the same seconds
  D() { taskset -c "$DC" "$B/dram" --cpus "$DC" --node 0 --kernel multi --thp 1 --mib 256 --seconds "$S" "$@"; }
  at() { python3 -c 'import time; print(time.time() + 8)'; }
  for r in 1 2; do
    D --label cpu-alone | tee -a "$O/dram.jsonl"
    for m in h2d zcr; do
      t=$(at)
      D --label "cpu-with-$m" --start-at "$t" > "$B/d.json" &
      P --host-node 0 --modes "$m" --start-at "$t" | tag "with-cpu-$m"
      wait; tee -a "$O/dram.jsonl" < "$B/d.json"
    done
  done
  exit 0
fi

B=$(mktemp -d /tmp/host-tier-pcie.XXXXXX); trap 'rm -rf "$B"' EXIT
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
{ "$NVCC" --version; "$NVCC" -O3 -std=c++17 -arch=sm_120 -o "$B/pcie" pcie.cu && gcc -O3 -march=native -pthread -Wall -o "$B/dram" dram.c; } \
  > "$O/build.txt" 2>&1 || { cat "$O/build.txt"; exit 4; }
GPU_LEASE_WHO=ra-host-tier gpu-lease "${NGPU:-1}" --wait --max-min 20 -- bash "$PWD/run_pcie.sh" leased "$B"
