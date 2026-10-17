#!/usr/bin/env bash
# What node 2 offers for measuring its GPUs' I/O paths, read-only: the GPU-metric sets nsys samples (DRAM and PCIe
# throughput while a DMA runs), perftest's CUDA options, NVSHMEM's IBGDA perftests, the NIC's PCIe link, display outputs,
# and the GPU leases.  No GPU lease; nothing is timed.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/link_pricing/tools.sh
set -u
O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
sec() { printf '\n===== %s\n' "$1"; }
{
  sec "gpu-lease status"; gpu-lease status 2>&1 | head -30
  sec "nsys gpu metrics"; nsys --version; nsys profile --gpu-metrics-devices=help 2>&1 | head -20
  nsys profile --gpu-metrics-set=help 2>&1 | head -40
  sec "ncu dram/pcie metrics (GB202)"; ncu --query-metrics --chip gb202 2>&1 | grep -E -i '^(dram__|pcie__|lts__t_sectors_srcunit|lts__t_sectors_aperture)' | head -40
  sec "perftest cuda options"; ib_write_bw --help 2>&1 | grep -i -E 'cuda|dmabuf|inline|use_|data_direct' | head -20
  sec "nvshmem"; dpkg -L libnvshmem3-cuda-13 2>/dev/null | grep -i -E 'perftest|bin/|nvshmrun|hydra' | head -20
  ls /usr/bin/nvshmem* /usr/lib/x86_64-linux-gnu/nvshmem/13/bin 2>&1 | head; command -v mpirun nvshmrun 2>&1
  sec "gdrcopy"; ls /usr/bin/gdrcopy_* 2>&1; ls -la /dev/gdrdrv 2>&1
  sec "NIC pcie link"; sudo -n lspci -vvv -s d7:00.0 2>&1 | grep -E 'LnkCap:|LnkSta:|MaxPayload|MaxReadReq|DevCap:' | head
  sudo -n lspci -vvv -s d6:00.0 2>&1 | grep -E 'LnkCap:|LnkSta:|MaxPayload|ACSCtl' | head
  sec "GPU0 pcie link"; sudo -n lspci -vvv -s db:00.0 2>&1 | grep -E 'LnkCap:|LnkSta:|MaxPayload|MaxReadReq' | head
  sec "display"; nvidia-smi -q -i 0 2>&1 | grep -i -E 'Product Name|Display|Board Part|Persistence|Compute Mode|MIG Mode' | head
  sec "rdma counters (port)"; ls /sys/class/infiniband/mlx5_0/ports/1/hw_counters 2>&1 | tr '\n' ' '; echo
  for c in port_xmit_data port_rcv_data; do echo "$c: $(cat /sys/class/infiniband/mlx5_0/ports/1/counters/$c 2>&1)"; done
  sec "eth0"; ip -br link show eth0 | awk '{print $1, $2}'; cat /sys/class/net/eth0/mtu; ethtool -S eth0 2>&1 | grep -E -i 'rx_bytes_phy|tx_bytes_phy|^ +rx_bytes:|^ +tx_bytes:' | head
  sec "rdma loopback settings"; cat /sys/class/net/eth0/device/sriov_numvfs 2>&1; rdma dev show 2>&1; rdma system show 2>&1
  ls /sys/class/infiniband/mlx5_0/ports/1/gids/ | head -4; for i in 0 1 2 3; do echo "gid $i: $(cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/$i 2>&1)"; done
} > "$O/tools.txt" 2>&1
cat "$O/tools.txt"
