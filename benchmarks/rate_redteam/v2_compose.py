"""Composition arithmetic for the v2 design (remote-bandwidth-exhaustion.md 17:40Z).

Not a measurement (ra-link-pricing and ra-host-tier own those); this prices the
*composition* of the two v2 answers against each other, from numbers already in the
design and the red team's F28.

(b) mu_link by binding granularity: a Pearl-C tile ships 4mn bytes of output z for kn
    bytes of weights, so per block |z|/|E| = 4m/k (tile-output binding) or 4m/k_s
    (k-slice binding of width k_s). The relay moves min(|z|, |E|) per block per pass.

(c) host tier: the GPUs stream host DRAM up over PCIe (<= 451 GB/s) and receipt it
    bound to z held on-GPU; DRAM beyond PCIe's reach is receipted by the host CPUs,
    bound to z shipped *down* GPU->host over PCIe. The z shipped down scales with
    |z|/|E| -- the same ratio that (b) wants LARGE to defend the relay. So the binding
    that defends mu_link (k-slice, ratio ~1) is the one that makes host-tier z-shipping
    expensive, and the binding that makes z-shipping cheap (tile-output, ratio ~1/128)
    is the one that breaks mu_link. This prints that tension as numbers.
"""

from __future__ import annotations

import json

K = 4096        # Pearl-C contraction width (weights per output, bytes)
PCIE = 451.0    # GB/s aggregate host->GPU, 8 x 56.4 (F28)
PCIE_DOWN = 451.0   # GB/s aggregate GPU->host (PCIe Gen5 is full duplex; same order)
DDR5_MEAS = 372.0   # GB/s both sockets measured (F28, 2 x 186)
DDR5_SPEC = 819.0   # GB/s spec


def mu_link_table() -> None:
    for binding, width in [("tile-output", K), ("k-slice k_s=32", 32), ("k-slice k_s=16", 16)]:
        for m in (8, 16, 64):
            ratio = 4 * m / width          # |z|/|E|
            cross = min(ratio, 1.0)        # fraction of |E| that must cross per block
            print(json.dumps({
                "part": "b", "binding": binding, "batch_m": m,
                "z_over_e": round(ratio, 4),
                "relay_crosses_frac_of_block": round(cross, 4),
                "mu_link_blowup_x": round(1.0 / cross, 1),  # 1/E read replaced by cross*E on a link
            }))


def host_tier_table() -> None:
    for ddr5, tag in [(DDR5_MEAS, "measured"), (DDR5_SPEC, "spec")]:
        gap = max(0.0, ddr5 - PCIE)        # GB/s DRAM beyond PCIe's reach: host CPUs must receipt it
        for binding, width in [("tile-output", K), ("k-slice k_s=32", 32)]:
            for m in (8, 64):
                ratio = 4 * m / width
                # Host CPUs receipt the gap's blocks: they hold E, need z. Two ways to co-locate:
                #  (i) ship z down in full (host does the hash): gap * ratio GB/s down PCIe;
                #  (ii) ship the host's digest up and let the GPU finish (min crossing, gap * min(ratio,1)),
                #       which puts the final hash back on the GPU and defeats offloading to the host.
                z_down_full = gap * ratio
                crossing_min = gap * min(ratio, 1.0)
                print(json.dumps({
                    "part": "c", "ddr5": tag, "ddr5_gbps": ddr5, "pcie_up_gbps": PCIE,
                    "cpu_receipt_gap_gbps": round(gap, 1),
                    "binding": binding, "batch_m": m, "z_over_e": round(ratio, 4),
                    "z_down_full_gbps": round(z_down_full, 1),
                    "z_down_full_over_pcie_down": round(z_down_full / PCIE_DOWN, 3),
                    "host_receipt_feasible": z_down_full <= PCIE_DOWN,
                    "crossing_min_gbps": round(crossing_min, 1),
                    "min_route_puts_hash_on_gpu": ratio >= 1.0,
                }))


if __name__ == "__main__":
    mu_link_table()
    host_tier_table()
