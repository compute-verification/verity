"""Declarations of real hardware, with every number's source.

`rtx_node` is one of our nodes as `nvidia-smi topo -m` and `lspci -tv` showed it on 18 Oct (vy-nebius-1 and 2): eight
RTX PRO 6000 GPUs, each on its own PCIe root port, in two NUMA groups of four, with host DRAM and one NIC on socket 1.
`with_dpu` adds an NVIDIA BlueField-3 DPU and its BMC with Amodo's measured stores and wipe coverage
(amododesign.com/notes/2026-10-02-dpu-memory-wipes): what a wipe can't reach on them is each device's residual.
`rack` puts nodes behind certifiers on a leaf, with a challenger and the outside world.
"""
from __future__ import annotations

from .physical import Assumption, Block, Cadence, Datacenter, Device, Link

GB = 8e9
TB = 8e12
US = Assumption("hop", 1e-6, "s", "measured")


def _a(name: str, value: float, unit: str, source: str = "datasheet") -> Assumption:
    return Assumption(name, value, unit, source)


def rtx_gpu(name: str) -> Device:
    """An RTX PRO 6000 Blackwell (sm_120): 96 GB of GDDR7 at about 1.6 TB/s, about 1 PFLOP/s of dense FP8 tensor work,
    and, for the walkthrough's unit of work, Boolean gates at its integer rate times 32 lanes."""
    return Device(name, "gpu",
                  rate={"mac": _a(f"{name}.fp8_dense", 1.0e15, "MAC/s"), "gate": _a(f"{name}.gates", 2.6e13 * 32, "gate/s")},
                  memory=_a(f"{name}.gddr7", 96 * GB, "bits"),
                  bandwidth=_a(f"{name}.gddr7_bw", 1.6 * TB, "bit/s"),
                  on_chip=_a(f"{name}.s_chip", 263_163_904 * 8, "bits", "measured"),
                  residual=_a(f"{name}.residual", 1 * GB, "bits", "declared"))


def rtx_node(n: str, gpus: int = 8) -> Block:
    """One node: `gpus` RTX PRO 6000s, 1 TB of host DRAM, and its NIC."""
    devices = [rtx_gpu(f"{n}.gpu{i}") for i in range(gpus)]
    devices.append(Device(f"{n}.dram", "dram", memory=_a(f"{n}.dram", 1000 * GB, "bits", "declared"),
                          bandwidth=_a(f"{n}.dram_bw", 0.4 * TB, "bit/s", "declared"),
                          residual=_a(f"{n}.dram_residual", 8 * GB, "bits", "declared")))
    devices.append(Device(f"{n}.nic", "nic"))
    return Block(n, devices)


def node_links(n: str, gpus: int = 8) -> list[Link]:
    """Each GPU's PCIe 5 x16 link to host DRAM (no shared switch, as `nvidia-smi topo -m` shows), and the NIC on
    socket 1 to that socket's GPUs."""
    pcie = _a("pcie5x16", 64e9 * 8, "bit/s")
    links = [Link(f"{n}.gpu{i}", f"{n}.dram", pcie, US, "pcie") for i in range(gpus)]
    links += [Link(f"{n}.gpu{i}", f"{n}.nic", pcie, US, "pcie") for i in range(gpus // 2, gpus)]
    return links


def with_dpu(node: Block) -> Block:
    """`node` with a BlueField-3 DPU and its BMC. Amodo, 2 Oct 2026: the DPU's 32 GB DDR is at least 85% wipeable and its
    40 GB eMMC and 128 GB NVMe at least 99%; the BMC's 1 GB DDR at least 75%, its 256 MB SPI flash at least 50%, and its
    64 MB secure-firmware flash not at all. Each device's residual is what those leave."""
    n = node.name
    dpu = Device(f"{n}.dpu", "dpu", memory=_a(f"{n}.dpu.stores", (32 + 40 + 128) * GB, "bits"),
                 residual=_a(f"{n}.dpu.unwiped", (0.15 * 32 + 0.01 * 40 + 0.01 * 128) * GB, "bits", "measured"))
    bmc = Device(f"{n}.bmc", "bmc", memory=_a(f"{n}.bmc.stores", (1 + 0.256 + 0.064) * GB, "bits"),
                 residual=_a(f"{n}.bmc.unwiped", (0.25 * 1 + 0.5 * 0.256 + 0.064) * GB, "bits", "measured"))
    return Block(n, [*node.devices, dpu, bmc])


def rack(nodes: int = 2, gpus: int = 8, dpus: bool = True, uplink: float = 400e9, cadence: Cadence | None = None) -> Datacenter:
    """`nodes` nodes, each behind its own certifier on a leaf switch, with an off-node challenger and the outside world.
    A node's NIC (or its DPU, when it has one) is its only way out, through its certifier."""
    up = _a("uplink", uplink, "bit/s", "declared")
    blocks, links = [], []
    for k in range(nodes):
        n = f"n{k}"
        b = rtx_node(n, gpus)
        blocks.append(with_dpu(b) if dpus else b)
        links += node_links(n, gpus)
        if dpus:
            links += [Link(f"{n}.nic", f"{n}.dpu", _a("dpu_link", 400e9, "bit/s"), US, "pcie"),
                      Link(f"{n}.dpu", f"{n}.bmc", _a("bmc_link", 1e9, "bit/s"), US),
                      Link(f"{n}.dpu", f"{n}.cert", up, US)]
        else:
            links.append(Link(f"{n}.nic", f"{n}.cert", up, US))
        links.append(Link(f"{n}.cert", "leaf", up, US))
    fabric = [Device(f"n{k}.cert", "certifier") for k in range(nodes)]
    fabric += [Device("leaf", "switch"), Device("challenger", "challenger"), Device("world", "outside")]
    links += [Link("challenger", "leaf", up, US), Link("world", "leaf", up, _a("wan_latency", 1e-3, "s", "declared"), "wan")]
    return Datacenter("rack", blocks, fabric, links,
                      cadence or Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                                         tick=_a("tick", 1e-4, "s", "declared")))
