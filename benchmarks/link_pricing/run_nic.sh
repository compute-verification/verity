#!/usr/bin/env bash
# Node 2's NIC paths (mlx5_0, a 400 Gb/s RoCE VF that is also eth0), in loopback on the one device: RDMA write and read
# between host and GPU memory, gated byte for byte (`rdma_gate.cu`), small rings the NIC keeps reading or writing in GPU
# memory (does the GPU's DRAM see them?), inline small writes (the CPU writes each payload into the work request), a
# perftest cross-check, and NVSHMEM's IBGDA put (GPU-initiated) between the two leased GPUs forced onto the NIC.
# nsys samples the GPU's DRAM and PCIe while the NIC moves its memory.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/link_pricing/run_nic.sh
#
# GPUS: the indices gpu-lease may take (default 4,5,6,7: NUMA node 1's, beside the NIC).  GID: the RoCE v2 IPv4 GID index (3).
# PART=ibgda runs only the NVSHMEM put.
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, gate.jsonl (rdma_gate), nic.jsonl (perftest rows),
# nsys.jsonl, nic_*.log, ibgda_*.log, port_counters.txt.
set -uo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
GID=${GID:-3}
export PART=${PART:-all}
D=mlx5_0

if [ "${1:-}" = leased ]; then
  S=$2 G=$2/rdma_gate
  : "${GPU_LEASE_UUID:?run under gpu-lease}"
  for u in ${GPU_LEASE_UUID//,/ }; do
    nvidia-smi -i "$u" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
  done > "$O/gpu.txt"
  cat "$O/gpu.txt"
  cat /sys/class/infiniband/$D/ports/1/counters/port_xmit_data /sys/class/infiniband/$D/ports/1/counters/port_rcv_data > "$O/port_counters.txt" 2>&1
  # gt NAME [nsys] ARGS...: one gated rdma_gate test (exit 3 = mismatch)
  gt() {
    local name=$1; shift
    local prof=; [ "${1:-}" = nsys ] && { prof=1; shift; }
    if [ -n "$prof" ]; then
      nsys profile --trace=none --sample=none --cpuctxsw=none --gpu-metrics-devices=cuda-visible --gpu-metrics-set=gb20x \
        --gpu-metrics-frequency=10000 --export=sqlite --force-overwrite=true -o "$S/g_$name" \
        timeout 60 "$G" --gid $GID "$@" > "$O/g_$name.log" 2>&1
      python3 nsys_metrics.py "$S/g_$name.sqlite" "$name" >> "$O/nsys.jsonl" 2>> "$O/nsys.err"
    else
      timeout 60 "$G" --gid $GID "$@" > "$O/g_$name.log" 2>&1
    fi
    local rc=$?
    grep '^{' "$O/g_$name.log" | sed "s/^{/{\"test\": \"$name\", \"rc\": $rc, /" | tee -a "$O/gate.jsonl"
    [ $rc = 0 ] || tail -3 "$O/g_$name.log"
  }
  if [ -x "$G" ] && [ "$PART" = all ]; then
    gt host_to_host_write --op write --src host --dst host
    gt gpu_to_host_write nsys --op write --src gpu --dst host
    gt host_to_gpu_write nsys --op write --src host --dst gpu
    gt gpu_to_gpu_write --op write --src gpu --dst gpu
    gt host_reads_gpu nsys --op read --src gpu --dst host
    gt gpu_reads_host --op read --src host --dst gpu
    gt host_to_host_read --op read --src host --dst host
    # rings the NIC keeps reading (TX) or writing (RX) in GPU memory, small and larger than L2
    for m in 8 1024; do
      gt "gpu_to_host_write_ring_$m" nsys --op write --src gpu --dst host --mib $m
      gt "host_to_gpu_write_ring_$m" nsys --op write --src host --dst gpu --mib $m
    done
    # small writes: inline (the CPU stores the payload into the work request) against DMA-read payloads
    for s in 64 128 192 256; do
      gt "inline_write_$s" --op write --src host --dst host --mib 3 --inline $s
      gt "dma_write_$s" --op write --src host --dst host --mib 3 --chunk $s
      gt "dma_write_gpu_src_$s" --op write --src gpu --dst host --mib 3 --chunk $s
    done
  fi
  # perftest cross-check: pt NAME TOOL "SERVER ARGS" "CLIENT ARGS"
  port=18600
  pt() {
    local name=$1 tool=$2 sargs=$3 cargs=$4
    port=$((port + 1))
    local common="-d $D -x $GID -F --report_gbits -D 3 -p $port"
    timeout 40 $tool $common $sargs > "$O/nic_${name}.server.log" 2>&1 &
    local sp=$!
    sleep 1.5
    timeout 40 $tool $common $cargs localhost > "$O/nic_${name}.log" 2>&1
    local rc=$?
    wait $sp
    local row
    row=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9.]+[[:space:]]+[0-9.]+[[:space:]]+[0-9.]+' "$O/nic_${name}.log" | tail -1)
    echo "{\"test\": \"$name\", \"tool\": \"$tool\", \"server\": \"$sargs\", \"client\": \"$cargs\", \"rc\": $rc, \"row\": \"$(echo $row)\"}" |
      tee -a "$O/nic.jsonl"
  }
  big="-s 1048576 -q 8"
  if [ "$PART" = all ]; then
    pt host_to_host_write ib_write_bw "$big" "$big"
    pt host_to_host_write_bidir ib_write_bw "$big -b" "$big -b"
    pt gpu_to_host_write ib_write_bw "$big" "$big --use_cuda=0"
    pt host_to_gpu_write ib_write_bw "$big --use_cuda=0" "$big"
    pt gpu_to_host_write_dmabuf ib_write_bw "$big" "$big --use_cuda=0 --use_cuda_dmabuf"
    pt host_reads_gpu ib_read_bw "$big --use_cuda=0" "$big"
  fi
  # GPU-initiated: NVSHMEM's IBGDA put between the two leased GPUs with P2P disabled, so every put crosses the NIC
  PT=/usr/bin/nvshmem_13/perftest
  ls -R "$PT" > "$O/nvshmem_perftests.txt" 2>&1
  BIN=$(find "$PT" -name 'shmem_put_bw' -type f | head -1)
  LIB=$(dirname "$(find /usr/lib /usr/lib64 /usr/local /opt -name 'libnvshmem_host.so.3' 2> /dev/null | head -1)")
  echo "{\"nvshmem_binary\": \"${BIN:-none}\", \"lib\": \"$LIB\"}" | tee -a "$O/nic.jsonl"
  if [ -n "$BIN" ]; then
    for transport in ibgda ibrc; do
      env LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" NVSHMEM_DISABLE_P2P=1 NVSHMEM_HCA_LIST=$D:1 NVSHMEM_IB_GID_INDEX=$GID NVSHMEM_BOOTSTRAP=PMI \
        NVSHMEM_IB_ENABLE_IBGDA=$([ $transport = ibgda ] && echo 1 || echo 0) \
        NVSHMEM_DEBUG=INFO timeout 120 /usr/bin/nvshmem_13/nvshmrun -n 2 -ppn 2 "$BIN" -b 4096 -e 4194304 -f 4 \
        > "$O/ibgda_$transport.log" 2>&1
      echo "{\"test\": \"nvshmem_put_bw_$transport\", \"rc\": $?}" | tee -a "$O/nic.jsonl"
      grep -i -E 'transport|ibgda|error|size|bw' "$O/ibgda_$transport.log" | tail -40
    done
  fi
  cat /sys/class/infiniband/$D/ports/1/counters/port_xmit_data /sys/class/infiniband/$D/ports/1/counters/port_rcv_data >> "$O/port_counters.txt" 2>&1
  exit 0
fi

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
{ "$NVCC" --version | tail -2; ls -l /usr/include/infiniband/verbs.h; } > "$O/build.txt" 2>&1
INC=
if [ ! -e /usr/include/infiniband/verbs.h ]; then
  # the headers alone, unpacked into scratch (nothing installed on the node); link against the node's libibverbs.so.1
  (cd "$S" && apt-get download libibverbs-dev && dpkg -x libibverbs-dev_*.deb hdr) >> "$O/build.txt" 2>&1
  INC="-I$S/hdr/usr/include"
  ln -sf /usr/lib/x86_64-linux-gnu/libibverbs.so.1 "$S/libibverbs.so"
fi
"$NVCC" -O3 -std=c++17 $INC rdma_gate.cu -o "$S/rdma_gate" -L"$S" -libverbs -lcuda >> "$O/build.txt" 2>&1 ||
  echo "rdma_gate did not build" | tee -a "$O/build.txt"
sha256sum run_nic.sh rdma_gate.cu nsys_metrics.py >> "$O/build.txt"
[ -x "$S/rdma_gate" ] && sha256sum "$S/rdma_gate" >> "$O/build.txt"
{ ibv_devinfo -d $D | grep -E 'fw_ver|state|active_mtu|link_layer|max_qp_rd_atom|max_res_rd_atom'; } >> "$O/build.txt" 2>&1
GPU_LEASE_WHO=ra-link-pricing gpu-lease 2 --wait --on "${GPUS:-4,5,6,7}" --max-min 20 -- bash "$PWD/run_nic.sh" leased "$S"
