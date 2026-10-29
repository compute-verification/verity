"""Declarations of real hardware, with every number's source: the one place a datacenter is declared.

`rtx_node` is one of our two RTX PRO 6000 nodes as both read with `nvidia-smi topo -m`, `lspci -tv` and `lsmem`:
two sockets, each an Intel Xeon 6776P with its host DRAM and a PCIe root complex (a switch); eight RTX PRO 6000 GPUs,
GPUs 0–3 on socket 0 and 4–7 on socket 1, each on a root port of its own; one NIC, on socket 1. Its device model is
`Accounting.Device`'s one Python mirror, `verity.core.protocols.remote_bandwidth.timing`: a GPU's S_chip, read and write
rates, SHA-512 oracle and squaring ceilings, and half the host's on each socket, so the node sums to
`ledger.Node.stated()`. Beside them: each GPU's memory and its rate for each kind of work (`logical.work`: tensor-core
multiply-adds by precision at the dense datasheet peaks `benchmarks/census/hardware.json` names; CUDA-core word
operations, `cuda:op`, and Boolean gates, `cuda:gate`, at its integer rate times 32 lanes), each socket's host DRAM
(half the node's online memory) and each PCIe link at PCIe 5.0 x16's rate. A capacity not measured yet is `math.inf`,
the upper bound that claims nothing; the residuals are declared.

`with_dpu` adds an NVIDIA BlueField-3 DPU between a node's NIC and its way out, and the DPU's BMC behind it, with Amodo's
measured stores and wipe coverage (amododesign.com/notes/2026-10-02-dpu-memory-wipes): what a wipe can't reach on them is
each device's residual, and neither states a write ceiling. `leaf_spine` puts nodes behind a leaf-spine fabric, each
through its own certifier, with hosts outside every block and challengers on the leaves. `rack` is the demo's: nodes with DPUs on one leaf, a challenger, and
the outside world over a WAN. `two_sites` is a fleet: two leaf-spines joined by a WAN between their spines.
"""
from __future__ import annotations

import math

from verity.core.protocols.remote_bandwidth import timing as T

from .physical import Assumption, Block, Cadence, Datacenter, Device, Fleet, Link

GB = 8e9
INF = math.inf
US = Assumption("hop", 1e-6, "s", "measured")

S_CHIP = Assumption("rtx-pro-6000/S_chip", T.GPU_CHIP_BYTES * 8, "bits", "measured")
GPU_MEMORY = Assumption("rtx-pro-6000/memory", 103_079_215_104 * 8, "bits", "datasheet")
GPU_READ = Assumption("rtx-pro-6000/memory-read", T.GPU_READ_BYTES_PER_S * 8, "bit/s", "measured")
GPU_WRITE = Assumption("rtx-pro-6000/memory-write", T.GPU_WRITE_BYTES_PER_S * 8, "bit/s", "datasheet")
GPU_RESIDUAL = Assumption("rtx-pro-6000/residual", 1 * GB, "bits", "declared")
GPU_RATE = {
    "mac:bf16": Assumption("rtx-pro-6000-server/bf16", 250_000_000_000_000, "multiply-adds/s", "datasheet"),
    "mac:fp8": Assumption("rtx-pro-6000-server/e4m3", 500_000_000_000_000, "multiply-adds/s", "datasheet"),
    "mac:fp4": Assumption("rtx-pro-6000-server/e2m1", 1_000_000_000_000_000, "multiply-adds/s", "datasheet"),
    "cuda:op": Assumption("rtx-pro-6000/cuda-ops", INF, "ops/s", "measured"),
    "cuda:gate": Assumption("rtx-pro-6000/gates", 2.6e13 * 32, "gate/s", "datasheet"),
    "oracle": Assumption("rtx-pro-6000/sha512", T.GPU_SHA512_PER_S, "compressions/s", "measured"),
    **{f"square-{w[1:]}": Assumption(f"rtx-pro-6000/squarings-{w}", r, "squarings/s", "declared")
       for w, r in T.GPU_SQUARINGS_PER_S.items()},
}
HOST_CHIP = Assumption("xeon-6776p/caches", T.HOST_CHIP_BYTES // 2 * 8, "bits", "declared")
HOST_DRAM = Assumption("host-dram/socket-capacity", 1_872_605_741_056 // 2 * 8, "bits", "measured")
HOST_READ = Assumption("host-dram/socket-read", T.HOST_READ_BYTES_PER_S / 2 * 8, "bit/s", "measured")
HOST_WRITE = Assumption("host-dram/socket-write", T.HOST_WRITE_BYTES_PER_S / 2 * 8, "bit/s", "datasheet")
HOST_RESIDUAL = Assumption("host-dram/socket-residual", 4 * GB, "bits", "declared")
PCIE5_X16 = 32e9 * 16 * 128 / 130          # 32 GT/s a lane, 16 lanes, 128b/130b coding: one direction's bits a second
PCIE_GPU = Assumption("pcie/gpu", PCIE5_X16, "bit/s", "datasheet")
PCIE_SOCKET = Assumption("pcie/socket-aggregate", INF, "bit/s", "measured")
PCIE_NIC = Assumption("pcie/nic", PCIE5_X16, "bit/s", "datasheet")
SOCKET_LINK = Assumption("socket/inter-socket", INF, "bit/s", "measured")


def rtx_node(name: str = "node") -> Datacenter:
    """One node, a datacenter of one block named `name` and no fabric. A link inside it has no latency until one is
    measured: a delay of 0, the bound that claims nothing for a path's delay."""
    devices, links = [], []
    for s in range(2):
        cpu = Device(f"{name}.cpu{s}", "cpu", memory=HOST_DRAM, bandwidth=HOST_READ, on_chip=HOST_CHIP,
                     residual=HOST_RESIDUAL, write=HOST_WRITE)
        rc = Device(f"{name}.pcie{s}", "switch")
        devices += [cpu, rc]
        links.append(Link(rc.name, cpu.name, PCIE_SOCKET, kind="pcie"))
        for k in range(4 * s, 4 * s + 4):
            gpu = Device(f"{name}.gpu{k}", "gpu", rate=dict(GPU_RATE), memory=GPU_MEMORY, bandwidth=GPU_READ,
                         on_chip=S_CHIP, residual=GPU_RESIDUAL, write=GPU_WRITE)
            devices.append(gpu)
            links.append(Link(gpu.name, rc.name, PCIE_GPU, kind="pcie"))
    nic = Device(f"{name}.nic0", "nic")
    devices.append(nic)
    links += [Link(nic.name, f"{name}.pcie1", PCIE_NIC, kind="pcie"),
              Link(f"{name}.cpu0", f"{name}.cpu1", SOCKET_LINK, kind="upi")]
    return Datacenter(name, [Block(name, devices)], [], links)


def with_dpu(node: Datacenter) -> Datacenter:
    """`node` (one block, as `rtx_node` declares it) with a BlueField-3 DPU after its NIC, which becomes the block's way
    out, and the DPU's BMC behind it. Amodo's measurements: the DPU's 32 GB DDR is at least 85% wipeable and its 40 GB eMMC
    and 128 GB NVMe at least 99%; the BMC's 1 GB DDR at least 75%, its 256 MB SPI flash at least 50%, and its 64 MB
    secure-firmware flash not at all. Each device's residual is what those leave."""
    (block,) = node.blocks
    n = block.name
    dpu = Device(f"{n}.dpu", "dpu", memory=Assumption(f"{n}.dpu.stores", (32 + 40 + 128) * GB, "bits"),
                 residual=Assumption(f"{n}.dpu.unwiped", (0.15 * 32 + 0.01 * 40 + 0.01 * 128) * GB, "bits", "measured"))
    bmc = Device(f"{n}.bmc", "bmc", memory=Assumption(f"{n}.bmc.stores", (1 + 0.256 + 0.064) * GB, "bits"),
                 residual=Assumption(f"{n}.bmc.unwiped", (0.25 * 1 + 0.5 * 0.256 + 0.064) * GB, "bits", "measured"))
    links = [Link(f"{n}.nic0", dpu.name, Assumption("dpu_link", 400e9, "bit/s"), US, "pcie"),
             Link(dpu.name, bmc.name, Assumption("bmc_link", 1e9, "bit/s"), US)]
    return Datacenter(node.name, [Block(n, [*block.devices, dpu, bmc])], list(node.fabric), node.links + links, node.cadence,
                      node.sealed)


def leaf_spine(blocks: int, *, leaves: int = 1, spines: int = 0, outside: tuple[str, ...] = ("ext0",),
               challengers: tuple[str, ...] = (), dpus: bool = False, prefix: str = "b", name: str | None = None,
               uplink_gbps: int = 400, spine_gbps: int = 400, fabric_latency: float = 2e-6, uplink_latency: float = 1e-6,
               outside_latency: float | None = None, cadence: Cadence | None = None) -> Datacenter:
    """`blocks` nodes (`rtx_node`, named `{prefix}0`, `{prefix}1`, …, each `with_dpu` when `dpus`) behind a leaf-spine
    fabric: each node's way out (its NIC, or its DPU) to its own certifier (`{prefix}I.cert`), the certifier to a leaf
    switch (round-robin over `leaves`), every leaf to every spine, and the hosts `outside` every block (kind `outside`)
    and the `challengers` on the leaves too. With one leaf, `spines` may be 0. Latencies are one way, in seconds:
    `uplink_latency` from a node to its certifier, `fabric_latency` a fabric hop, and an outside host's link a WAN of
    `outside_latency` when it is given, else a fabric hop. `cadence` is the certifiers' (`Cadence`'s default when None)."""
    if blocks < 1 or leaves < 1 or (leaves > 1 and spines < 1):
        raise ValueError("at least one block and one leaf, and spines to join more than one leaf")
    up = Assumption("fabric/uplink", uplink_gbps * 1_000_000_000, "bit/s", "declared")
    trunk = Assumption("fabric/spine", spine_gbps * 1_000_000_000, "bit/s", "declared")
    nic_hop = Assumption("fabric/uplink-latency", uplink_latency, "s", "declared")
    hop = Assumption("fabric/hop-latency", fabric_latency, "s", "declared")
    wan = (hop, "ethernet") if outside_latency is None else (Assumption("fabric/wan-latency", outside_latency, "s",
                                                                        "declared"), "wan")
    switches = [Device(f"leaf{j}", "switch") for j in range(leaves)]
    nodes, certs, links = [], [], []
    for i in range(blocks):
        n = f"{prefix}{i}"
        node = with_dpu(rtx_node(n)) if dpus else rtx_node(n)
        cert = Device(f"{n}.cert", "certifier")
        nodes += node.blocks
        certs.append(cert)
        links += node.links
        links += [Link(f"{n}.dpu" if dpus else f"{n}.nic0", cert.name, up, nic_hop, "ethernet"),
                  Link(cert.name, switches[i % leaves].name, up, hop, "ethernet")]
    tops = [Device(f"spine{k}", "switch") for k in range(spines)]
    links += [Link(sw.name, sp.name, trunk, hop, "ethernet") for sp in tops for sw in switches]
    hosts = [Device(x, "outside") for x in outside]
    links += [Link(x.name, switches[c % leaves].name, up, *wan) for c, x in enumerate(hosts)]
    checkers = [Device(x, "challenger") for x in challengers]
    links += [Link(x.name, switches[c % leaves].name, up, hop, "ethernet") for c, x in enumerate(checkers)]
    return Datacenter(name or f"leaf-spine-{blocks}", nodes, switches + certs + tops + hosts + checkers, links,
                      cadence or Cadence())


def rack(nodes: int = 2, dpus: bool = True, cadence: Cadence | None = None) -> Datacenter:
    """The demo's rack: `nodes` nodes (`n0`, `n1`, …), each with a DPU when `dpus`, behind their own certifiers on one
    leaf, with a challenger on it and the outside world (`world`) 1 ms away over a WAN, every other hop 1 µs. Its
    certifiers frame 32 bits, a bucket a tick of 0.1 ms, a record every 4 buckets."""
    return leaf_spine(nodes, prefix="n", name="rack", outside=("world",), challengers=("challenger",), dpus=dpus,
                      fabric_latency=1e-6, outside_latency=1e-3,
                      cadence=cadence or Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                                                 tick=Assumption("tick", 1e-4, "s", "declared")))


def two_sites(blocks: int = 2, *, leaves: int = 2, spines: int = 1, fabric_latency: float = 2e-6,
              wan_latency: float = 20e-3, wan_gbps: int = 100, outside: str = "ext0",
              cadence: Cadence | None = None) -> Fleet:
    """Two sites under one developer, `east` and `west`, each a `leaf_spine` of `blocks` nodes, named `east.b0`, ….
    One WAN link joins the two sites' spines (`east.spine0`, `west.spine0`; their leaves when they have no spine), of
    `wan_gbps` and `wan_latency` one way, both declared. The WAN is fabric: every block is behind its own certifier
    already. The one host outside every block, `outside`, is on the west site, so the east blocks reach it across the
    WAN, through their own certifiers and both sites' fabric."""
    top = "spine0" if spines else "leaf0"
    sites = [leaf_spine(blocks, leaves=leaves, spines=spines, outside=(outside,) if name == "west" else (),
                        fabric_latency=fabric_latency, cadence=cadence).prefixed(name, keep=frozenset({outside}))
             for name in ("east", "west")]
    wan = Link(f"east.{top}", f"west.{top}",
               Assumption("wan/east-west", wan_gbps * 1_000_000_000, "bit/s", "declared"),
               Assumption("wan/east-west-latency", wan_latency, "s", "declared"), "wan")
    return Fleet(sites, [wan])
