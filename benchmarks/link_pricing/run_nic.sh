#!/usr/bin/env bash
# Node 2's NIC paths (mlx5_0, a 400 Gb/s RoCE VF that is also eth0): perftest in loopback on the one device, host and GPU
# memory each way, RDMA write / read / bidirectional, CUDA dma-buf, inline small writes (WQEs written by the CPU), and
# NVSHMEM's IBGDA put (GPU-initiated) between two GPUs of the node forced onto the NIC.  nsys samples the GPU's DRAM and
# PCIe while the NIC moves its memory.  Each test runs 3 s.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/link_pricing/run_nic.sh
#
# GPUS: the indices gpu-lease may take (default 4,5,6,7: NUMA node 1's, beside the NIC).  GID: the RoCE v2 IPv4 GID index (3).
# Outputs under $RESEARCH_RUN_DIR/out: nic.jsonl (one line a test), nic_<test>.log, nsys.jsonl, ibgda_*.log, gpu.txt.
set -uo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
GID=${GID:-3}
D=mlx5_0

if [ "${1:-}" = leased ]; then
  S=$2
  : "${GPU_LEASE_UUID:?run under gpu-lease}"
  for u in ${GPU_LEASE_UUID//,/ }; do
    nvidia-smi -i "$u" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
  done > "$O/gpu.txt"
  cat "$O/gpu.txt"
  port=18600
  # pt NAME TOOL "SERVER ARGS" "CLIENT ARGS" [nsys]: one loopback test; server and client on $D
  pt() {
    local name=$1 tool=$2 sargs=$3 cargs=$4 prof=${5:-}
    port=$((port + 1))
    local common="-d $D -x $GID -F --report_gbits -D 3 -p $port"
    timeout 40 $tool $common $sargs > "$O/nic_${name}.server.log" 2>&1 &
    local sp=$!
    sleep 1.5
    if [ -n "$prof" ]; then
      nsys profile --trace=none --sample=none --cpuctxsw=none --gpu-metrics-devices=cuda-visible --gpu-metrics-set=gb20x \
        --gpu-metrics-frequency=10000 --export=sqlite --force-overwrite=true -o "$S/nic_$name" \
        timeout 40 $tool $common $cargs localhost > "$O/nic_${name}.log" 2>&1
      python3 nsys_metrics.py "$S/nic_$name.sqlite" "nic_$name" >> "$O/nsys.jsonl" 2>> "$O/nsys.err"
    else
      timeout 40 $tool $common $cargs localhost > "$O/nic_${name}.log" 2>&1
    fi
    local rc=$?
    wait $sp
    # perftest's result row: bytes, iterations, BW peak, BW average (Gb/s), MsgRate (Mpps)
    local row
    row=$(grep -E '^ *[0-9]+ +[0-9]+ +[0-9.]+ +[0-9.]+ +[0-9.]+' "$O/nic_${name}.log" | tail -1)
    echo "{\"test\": \"$name\", \"tool\": \"$tool\", \"server\": \"$sargs\", \"client\": \"$cargs\", \"rc\": $rc, \"row\": \"$(echo $row)\"}" |
      tee -a "$O/nic.jsonl"
  }
  big="-s 1048576 -q 8"
  pt host_to_host_write ib_write_bw "$big" "$big"
  pt host_to_host_write_bidir ib_write_bw "$big -b" "$big -b"
  pt host_to_host_write_node1mem "numactl --membind=1 ib_write_bw" "$big" "$big"
  pt gpu_to_host_write ib_write_bw "$big" "$big --use_cuda=0" nsys
  pt host_to_gpu_write ib_write_bw "$big --use_cuda=0" "$big" nsys
  pt gpu_to_gpu_write ib_write_bw "$big --use_cuda=1" "$big --use_cuda=0"
  pt gpu_to_host_write_dmabuf ib_write_bw "$big" "$big --use_cuda=0 --use_cuda_dmabuf"
  pt host_to_gpu_write_dmabuf ib_write_bw "$big --use_cuda=0 --use_cuda_dmabuf" "$big"
  pt gpu_to_host_write_bidir ib_write_bw "$big -b" "$big -b --use_cuda=0"
  pt host_reads_gpu ib_read_bw "$big --use_cuda=0" "$big" nsys
  pt gpu_reads_host ib_read_bw "$big" "$big --use_cuda=0"
  # a small buffer the NIC keeps reading (TX) or writing (RX) in GPU memory: does it reach DRAM?
  pt gpu_to_host_write_64k ib_write_bw "-s 65536 -q 8" "-s 65536 -q 8 --use_cuda=0" nsys
  pt host_to_gpu_write_64k ib_write_bw "-s 65536 -q 8 --use_cuda=0" "-s 65536 -q 8" nsys
  # inline writes: the CPU writes each WQE with its payload to the NIC (no DMA read of the payload)
  for s in 64 128 192; do
    pt "inline_write_$s" ib_write_bw "-s $s -q 8 -I $s -l 16" "-s $s -q 8 -I $s -l 16"
    pt "noinline_write_$s" ib_write_bw "-s $s -q 8 -I 0 -l 16" "-s $s -q 8 -I 0 -l 16"
  done
  pt inline_write_max ib_write_bw "-s 256 -q 8 -I 256 -l 16" "-s 256 -q 8 -I 256 -l 16"
  # GPU-initiated: NVSHMEM's IBGDA put between the two leased GPUs with P2P disabled, so every put crosses the NIC
  PT=/usr/bin/nvshmem_13/perftest
  ls -R "$PT" > "$O/nvshmem_perftests.txt" 2>&1
  BIN=$(find "$PT" -name 'shmem_put_bw*' -type f | head -1)
  echo "IBGDA binary: ${BIN:-none}" | tee -a "$O/nic.jsonl"
  if [ -n "$BIN" ]; then
    for transport in ibgda ibrc; do
      env NVSHMEM_DISABLE_P2P=1 NVSHMEM_HCA_LIST=$D:1 NVSHMEM_IB_GID_INDEX=$GID NVSHMEM_BOOTSTRAP=PMI \
        NVSHMEM_IB_ENABLE_IBGDA=$([ $transport = ibgda ] && echo 1 || echo 0) \
        NVSHMEM_DEBUG=INFO timeout 120 /usr/bin/nvshmem_13/nvshmrun -n 2 -ppn 2 "$BIN" -b 4096 -e 4194304 -f 4 \
        > "$O/ibgda_$transport.log" 2>&1
      echo "{\"test\": \"nvshmem_put_bw_$transport\", \"rc\": $?}" | tee -a "$O/nic.jsonl"
      grep -i -E 'transport|ibgda|error|size|bw' "$O/ibgda_$transport.log" | tail -40
    done
  fi
  cat /sys/class/infiniband/$D/ports/1/counters/port_xmit_data /sys/class/infiniband/$D/ports/1/counters/port_rcv_data > "$O/port_counters.txt" 2>&1
  exit 0
fi

S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
sha256sum run_nic.sh nsys_metrics.py > "$O/build.txt"
{ ibv_devinfo -d $D | grep -E 'fw_ver|state|active_mtu|link_layer'; cat /sys/class/infiniband/$D/ports/1/counters/port_xmit_data; } >> "$O/build.txt" 2>&1
GPU_LEASE_WHO=ra-link-pricing gpu-lease 2 --wait --on "${GPUS:-4,5,6,7}" --max-min 20 -- bash "$PWD/run_nic.sh" leased "$S"
