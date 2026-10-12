#!/usr/bin/env bash
# Node 2's egress paths out of a GPU, read-only: the NIC (RDMA device, its BARs, the driver's peer-mapping switches), the
# GPUs' BAR1 and PCIe position, the host CPU, and the tools that could count HBM traffic.  No GPU lease; nothing is timed.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- bash benchmarks/link_pricing/inventory.sh
set -u
O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
sec() { printf '\n===== %s\n' "$1"; }
{
  sec "cpu"; lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|L3|Flags' | sed 's/Flags:.*\(avx512[a-z_]*\).*/Flags: (avx512 present)/'
  grep -o -w -E 'amx_tile|avx512f|ddio|cat_l3|cdp_l3|mba|rdt_a' /proc/cpuinfo | sort | uniq -c
  sec "kernel and driver"; uname -r; cat /proc/driver/nvidia/version 2>&1 | head -2
  sec "nvidia module parameters (peer mapping, stream mem ops)"
  grep -i -E 'RegistryDwords|EnableStreamMemOPs|PeerMapping|EnableResizableBar|DmaRemapPeerMmio' /proc/driver/nvidia/params 2>&1
  sec "modules"; lsmod | grep -E -i 'nvidia|mlx|ib_|rdma|peermem|nv_peer|gdrdrv' 2>&1
  sec "rdma devices"; ls -la /dev/infiniband 2>&1; ls /sys/class/infiniband 2>&1
  for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue; echo "$d:"; cat "$d"/fw_ver "$d"/node_type "$d"/hca_type 2>&1
    for p in "$d"/ports/*; do echo " port $(basename "$p"): state=$(cat "$p"/state 2>&1) rate=$(cat "$p"/rate 2>&1) link=$(cat "$p"/link_layer 2>&1)"; done; done
  command -v ibv_devinfo && ibv_devinfo -v 2>&1 | grep -E -i 'hca_id|fw_ver|vendor_part|max_qp|max_mr|link_layer|active_mtu|state|device_cap|dm|memic|max_inline' | head -40
  command -v rdma && rdma link show 2>&1; command -v rdma && rdma resource show 2>&1 | head
  sec "eth0"; ip -br link 2>&1; ip -br addr 2>&1 | sed -E 's/([0-9]+\.[0-9]+)\.[0-9]+\.[0-9]+/\1.x.x/g'
  ethtool eth0 2>&1 | grep -E 'Speed|Duplex|Link detected'; ethtool -i eth0 2>&1
  ethtool -l eth0 2>&1 | head -12; ethtool -g eth0 2>&1 | head -12
  sec "pci: NIC"
  NIC=$(lspci -D | grep -i mellanox | awk '{print $1}' | head -1); echo "NIC=$NIC"
  [ -n "$NIC" ] && { lspci -vvv -s "$NIC" 2>&1 | grep -E 'Region|Memory at|LnkCap:|LnkSta:|MaxPayload|MaxReadReq|ACSCap|ACSCtl|SR-IOV|NUMA' ;
    cat /sys/bus/pci/devices/$NIC/numa_node; cat /sys/bus/pci/devices/$NIC/resource; ls -la /sys/bus/pci/devices/$NIC/ | grep -E 'resource|iommu'
    readlink -f /sys/bus/pci/devices/$NIC; ls /sys/bus/pci/devices/$NIC/iommu_group/devices 2>&1; }
  sec "pci: GPUs (BAR1, link, place)"
  for g in $(lspci -D | grep -i 'nvidia' | awk '{print $1}'); do
    echo "$g numa=$(cat /sys/bus/pci/devices/$g/numa_node) path=$(readlink -f /sys/bus/pci/devices/$g)"
    lspci -vvv -s "$g" 2>&1 | grep -E 'Region|LnkSta:|MaxPayload|ACSCtl' | head -6; done
  nvidia-smi -q -d MEMORY 2>&1 | grep -E -A3 'BAR1' | head -12
  nvidia-smi topo -m 2>&1 | head -12; nvidia-smi topo -p2p w 2>&1 | head -12; nvidia-smi topo -p2p r 2>&1 | head -12
  sec "iommu"; ls /sys/kernel/iommu_groups 2>&1 | wc -l; cat /proc/cmdline | tr ' ' '\n' | grep -i -E 'iommu|intel_iommu|pci' 2>&1
  dmesg 2>&1 | grep -i -E 'DMAR|IOMMU|peermem|ats' | head -10
  sec "packages and tools"
  dpkg -l 2>/dev/null | grep -E -i 'rdma-core|ibverbs|libmlx5|perftest|mlnx|doca|nccl|nvshmem|gdrcopy|nsight|dcgm|datacenter-gpu' | awk '{print $2, $3}'
  for t in ib_write_bw ib_send_bw ib_read_bw nsys ncu dcgmi nv-hostengine gdrcopy_copybw; do printf '%s: %s\n' "$t" "$(command -v $t || ls /usr/local/cuda/bin/$t 2>/dev/null || echo none)"; done
  ls /opt/nvidia 2>&1 | head; ls -d /usr/local/cuda*/ 2>&1
  ls /usr/include/infiniband 2>&1 | head; ls /usr/lib/x86_64-linux-gnu/ | grep -E 'libibverbs|libmlx5|libnvidia-peer' 2>&1
  sec "profiling permission"; grep -i -E 'RmProfilingAdminOnly' /proc/driver/nvidia/params 2>&1
  cat /proc/sys/kernel/perf_event_paranoid
  sec "gpu-lease"; command -v gpu-lease; gpu-lease --help 2>&1 | head -40
} > "$O/inventory.txt" 2>&1
cat "$O/inventory.txt"
