"""SCOPE-OFFLOAD, mechanically: how much of a credited GPU's int8 work a helper the prover controls can do OFF that GPU's
facets, and what bounds it (RT-SCOPE, the compute-accounting red team).

Where `isa_escape.py` asks whether a unit ON the die can head a committed word more cheaply than the SM menu, this script
asks the same of helpers off the die: the host CPU/AMX, a peer GPU, a machine behind a NIC, or device memory holding
precomputed words. Which of two bounds applies decides the answer:

  - WORD-LEVEL, if the credited GPU must head every committed word itself. A helper can then import only words or
    additive partials (the fixed-word rule lets those in without un-fixing the head, card 0f7721ff). A word is a fresh
    s32 packing 32 int8 products (FOLD-BW), 4 bytes carrying 32 MACs, so the helper is link-bound:

        f_word <= BW_in * (MACs_per_word / bytes_per_word) / MAC_rate = 8 * BW_in / MAC_rate.

  - WHOLE-TILE, which is what the served path allows, since nothing there makes the GPU head its words. The transcript
    is a Merkle root over tile digests (`pouw/audit.py`), and an opening is A's rows, B's rows and the leaf, so a helper
    holding the salt computes whole tiles and ships one 64-byte digest a tile. No link binds then. The helper's own
    rate on the credited arithmetic does, and the credited s8 path is integer (`PouwSegDot_v1`, acc + sum a_k b_k mod
    2^32 on int8), which host AMX reproduces exactly.

So the word-level numbers are floors on each channel's exposure, not caps. The channel classes:

  - NETWORK-H: a NIC. The inter-node network certifiers sign its ingress, so it is in H, counted in bytes.
  - PCIE-UNSIGNED: PCIe. It is in `Device.lean`'s `ι` ("PCIe from NVMe"), but no certifier observes it.
  - UNCOUNTED-NVLINK: intra-node NVLink/NVSwitch, neither in `ι` nor observed. `PouwLocalOff` holds only at H = 0, is
    conditional on the open conjecture A3 (`PouwFloorOff`), and credits "the node ... not which of its GPUs did".
  - COIN-BLOCKED: device memory holding precomputed words. On main every window of a salt shares one interval from the
    salt's release to the deadline (`WorkBetweenSaltAndDeadline`), so early work is counted work. Its word stock is
    also capacity-bound, to a few ms of the GPU's compute.
  - HOST-TILES: the host CPU/AMX computing whole s8 tiles, bounded by its int8 rate shared across the node's GPUs, not
    by PCIe. It is neither credited nor bounded by PoUW today (CMP-49).

The checkable arithmetic: the committed word packs exactly 32 int8*weight products into an s32 with no wrap (verified
over Z/2^32 and the operand alphabets), and a tile digest carries more MACs a byte than a word even at H1's smallest
tile. With `--device-lean PATH` the script also reads a Device.lean and confirms the model names PCIe and the network as
its side channels and neither NVLink nor a host-RAM channel. That check is off by default, so the run reads only this
package.

Run: `python scope_offload.py [--mutations] [--device-lean Security/Definitions/Accounting/Device.lean] [--verdict PATH]`.
Stdlib + exact_kit only.
Sources: MAC_rate = s8 macs/SM/clk * SMs * f_max, and the headline eps (note:20261017T1818Z-draft-axioms secs 2.1, 4.1);
PCIe5 x16 64 GB/s one way; ConnectX-7/BlueField-3 400 Gb/s = 50 GB/s a NIC; NVLink4 450 GB/s and NVLink5 900 GB/s one
way; RTX PRO 6000 has no NVLink; HBM3e 4.8 (H200) / 8 (B200) TB/s, GDDR7 ~1.8 TB/s (sm_120); device memory 96 / 141 /
192 GB; H1's smallest tile 4 x 8 at K = 64 (`h1tile.circuit-check.py`); the ~12% host AMX efficiency training's C7
measured (a relay, note:20261019T0455Z-draft-rt-scope RT-SCOPE-S2).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from exact_kit.checks import Checks

MAC_PER_WORD = 32  # a committed word sums 32 int8 products (FOLD-BW)
BYTES_PER_WORD = 4  # a fresh s32
DIGEST_BYTES = 64  # a tile's SHA-512 digest, the transcript's leaf
H1_SMALLEST_TILE_MACS = 4 * 8 * 64  # one lane (4 x 8) at K = 64
M32 = 0xFFFFFFFF

# The design's operand alphabets (decisions 29, 44, 51), as isa_escape.py uses: int8 activations, weights |w| <= 63.
INT8 = range(-128, 128)
WEIGHT = range(-63, 64)

# Per device: s8 dense MACs a SM clock, SMs, f_max (Hz). Sheet constants, note:20261017T1818Z-draft-axioms sec 2.1.
DEVICES = {
    "sm_120 RTX PRO 6000": (1024, 188, 2430e6),
    "H200 sm_90": (4096, 132, 1980e6),
    "B200 sm_100": (8192, 148, 1965e6),
}

# Per device, the headline eps (note:20261017T1818Z-draft-axioms sec 4.1) and device memory in bytes.
HEADLINE_EPS = {"sm_120 RTX PRO 6000": 0.00585, "H200 sm_90": 0.00716, "B200 sm_100": 0.01032}
MEMORY_BYTES = {"sm_120 RTX PRO 6000": 96e9, "H200 sm_90": 141e9, "B200 sm_100": 192e9}

# Per device, one-way ingress into one chip (bytes/s) and the channel's model class.
GB = 1e9
TB = 1e12
CHANNELS = {
    "sm_120 RTX PRO 6000": [
        ("PCIe5 x16", 64 * GB, "PCIE-UNSIGNED"),
        ("1 NIC (400Gb GDR)", 50 * GB, "NETWORK-H"),
        ("GDDR7 (precomputed)", 1.8 * TB, "COIN-BLOCKED"),
    ],
    "H200 sm_90": [
        ("PCIe5 x16", 64 * GB, "PCIE-UNSIGNED"),
        ("node 8-NIC", 400 * GB, "NETWORK-H"),
        ("NVLink4 in", 450 * GB, "UNCOUNTED-NVLINK"),
        ("HBM3e (precomputed)", 4.8 * TB, "COIN-BLOCKED"),
    ],
    "B200 sm_100": [
        ("PCIe5 x16", 64 * GB, "PCIE-UNSIGNED"),
        ("node 8-NIC", 400 * GB, "NETWORK-H"),
        ("NVLink5 in", 900 * GB, "UNCOUNTED-NVLINK"),
        ("HBM3e (precomputed)", 8 * TB, "COIN-BLOCKED"),
    ],
}

# Host int8 compute: dual-socket Granite Rapids AMX, ~1024 int8 MACs/clk/core, 256 cores, 2 GHz, shared by a node's GPUs.
HOST_MAC_RATE = 256 * 2.0e9 * 1024
GPUS_PER_NODE = 8
AMX_MEASURED_EFFICIENCY = 0.12


def mac_rate(device: str) -> float:
    m, sms, f = DEVICES[device]
    return m * sms * f


def offload_fraction(bw: float, rate: float) -> float:
    """WORD-LEVEL: the most of a GPU's credited MACs a helper can supply over a channel of `bw` bytes/s when the GPU must
    head every word, so each word or additive partial crosses the channel."""
    return bw * MAC_PER_WORD / BYTES_PER_WORD / rate


def host_tile_fraction(rate: float, efficiency: float) -> float:
    """WHOLE-TILE: the share of one GPU's credited MACs the host lends when it computes whole tiles at `efficiency` of its
    int8 peak, split evenly across the node's GPUs. No link term: a tile crosses as one digest."""
    return HOST_MAC_RATE * efficiency / GPUS_PER_NODE / rate


def macs_per_byte(macs: int, nbytes: int) -> float:
    return macs / nbytes


def committed_word_packs_32_macs(rng: random.Random, trials: int = 2000) -> bool:
    """A committed word is the exact s32 sum of 32 int8*weight products with no wrap, so it is 4 bytes carrying 32 MACs:
    every such sum fits in s32 (|acc| < 2^31), and its two's-complement s32 round-trips through 4 bytes (Z/2^32)."""
    lo, hi = -(1 << 31), (1 << 31) - 1
    for _ in range(trials):
        acc = sum(rng.choice(INT8) * rng.choice(WEIGHT) for _ in range(MAC_PER_WORD))
        if not (lo <= acc <= hi):
            return False
        packed = acc & M32  # the 4-byte little-endian s32
        unpacked = packed - (1 << 32) if packed >> 31 else packed
        if unpacked != acc:
            return False
    return True


def device_lean_arg(args: list[str]) -> str | None:
    """The `--device-lean PATH` value, if given (off by default so the run reads only this package)."""
    for i, a in enumerate(args):
        if a == "--device-lean" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--device-lean="):
            return a.split("=", 1)[1]
    return None


def main() -> int:
    checks = Checks()
    rng = random.Random(20261019)

    # 1. The word-level compression, and why it is not the served bound: a tile digest carries more.
    checks.check("a committed word packs 32 int8 MACs into one 4-byte s32 (no wrap), so 8 MACs a byte",
                 committed_word_packs_32_macs(rng),
                 f"{MAC_PER_WORD} MACs / {BYTES_PER_WORD} bytes = {MAC_PER_WORD // BYTES_PER_WORD} MACs/byte")
    word_density = macs_per_byte(MAC_PER_WORD, BYTES_PER_WORD)
    tile_density = macs_per_byte(H1_SMALLEST_TILE_MACS, DIGEST_BYTES)
    checks.check("a tile digest carries more MACs a byte than a word, even at H1's smallest tile",
                 tile_density > word_density,
                 f"{H1_SMALLEST_TILE_MACS} MACs / {DIGEST_BYTES} bytes = {tile_density:.0f} vs {word_density:.0f} "
                 "MACs/byte; larger tiles raise it, so the word-level ceilings are floors")

    # 2. With --device-lean PATH, confirm the model's side channels are PCIe and the network, not NVLink or host RAM.
    lean_path = device_lean_arg(checks.args)
    if lean_path is not None:
        lean = Path(lean_path).read_text(encoding="utf-8")
        checks.check("Device.lean names PCIe and the network as its side channels", "PCIe" in lean and "network" in lean,
                     "Ingress docstring")
        checks.check("Device.lean models no NVLink/NVSwitch ingress channel",
                     "NVLink" not in lean and "NVSwitch" not in lean, "so intra-node peer ingress is unmodeled")
        checks.check("Device.lean states there is no host-RAM channel (Daniel 3 Oct)", "no host RAM" in lean,
                     "host offload rides the PCIe channel, which the docstring frames as 'from NVMe'")

    # 3. The per-device, per-channel ceiling, and the headline-relevant verdicts.
    rows = []
    worst_uncounted = {}
    for device in DEVICES:
        rate = mac_rate(device)
        checks.check(f"{device}: host AMX is a large fraction of its int8 rate",
                     0.2 <= HOST_MAC_RATE / rate <= 1.3, f"{HOST_MAC_RATE / rate * 100:.0f}% (compute, pre-delivery)")
        for name, bw, cls in CHANNELS[device]:
            f = offload_fraction(bw, rate)
            rows.append({"device": device, "channel": name, "cls": cls, "gbps": bw / GB, "f_pct": f * 100})
            checks._print(f"      {device:20s} {name:20s} {bw / GB:7.0f} GB/s  f<= {f * 100:7.3f}%  [{cls}]")
            if cls == "UNCOUNTED-NVLINK":
                worst_uncounted[device] = max(worst_uncounted.get(device, 0.0), f)

    # 4. Word-level: a helper that must deliver every word over PCIe stays under 0.12%; NVLink's floor reaches 0.3% on
    #    H200/B200, and sm_120 has no NVLink.
    for device in DEVICES:
        rate = mac_rate(device)
        pcie = offload_fraction(64 * GB, rate)
        checks.check(f"{device}: word-level, a helper delivering every word over PCIe is under 0.12%", pcie < 0.0012,
                     f"{pcie * 100:.3f}% -- a floor: whole tiles cross as 64-byte digests")
    checks.check("sm_120 has no NVLink, so no uncounted intra-node channel", "sm_120 RTX PRO 6000" not in worst_uncounted)
    for device in ("H200 sm_90", "B200 sm_100"):
        f = worst_uncounted[device]
        checks.check(f"{device}: word-level, NVLink lends >=0.3% of credited MACs, unobserved by any certifier",
                     f >= 0.003, f"{f * 100:.3f}%")

    # 5. Whole-tile: the host's exact int8 rate, not PCIe, bounds what it lends. At peak it beats every headline; at
    #    training's measured AMX efficiency it beats sm_120's and H200's.
    for device in DEVICES:
        rate, eps = mac_rate(device), HEADLINE_EPS[device]
        peak = host_tile_fraction(rate, 1.0)
        checks.check(f"{device}: whole-tile, the host at AMX peak lends more than the headline eps", peak > eps,
                     f"{peak * 100:.2f}% vs {eps * 100:.3f}% (HOST-TILES)")
    for device in ("sm_120 RTX PRO 6000", "H200 sm_90"):
        rate, eps = mac_rate(device), HEADLINE_EPS[device]
        measured = host_tile_fraction(rate, AMX_MEASURED_EFFICIENCY)
        checks.check(f"{device}: whole-tile, the host at measured AMX efficiency lends more than the headline eps",
                     measured > eps, f"{measured * 100:.3f}% vs {eps * 100:.3f}% (HOST-TILES)")

    # 6. Device memory: its word stock covers under 0.2% of a 1-s window, and on main early work is counted work.
    for device in DEVICES:
        rate = mac_rate(device)
        stock = MEMORY_BYTES[device] * MAC_PER_WORD / BYTES_PER_WORD / rate
        checks.check(f"{device}: device memory's word stock covers under 0.2% of a 1-s window", stock < 0.002,
                     f"{stock * 1e3:.2f} ms of compute (COIN-BLOCKED: one interval per salt on main)")

    if checks.mutations:
        b200 = mac_rate("B200 sm_100")
        # Mis-stating the word as 2 bytes would halve the compression and double every ceiling: the arithmetic notices.
        inflated = 900 * GB * MAC_PER_WORD / 2 / b200
        checks.mutation("a word mis-sized at 2 bytes doubles the ceiling",
                        inflated > 1.9 * offload_fraction(900 * GB, b200),
                        f"{inflated * 100:.3f}% vs {offload_fraction(900 * GB, b200) * 100:.3f}%")
        # If a Device.lean named NVLink, the "models no NVLink" check (NVLink not in text) would flip to False.
        sample_with_nvlink = "side channel i: PCIe from NVMe, the network, and NVLink"
        checks.mutation("a source that named NVLink would flip the no-NVLink check", "NVLink" in sample_with_nvlink)
        # A value one past the s32 max does not round-trip through 4 bytes: the packing check catches overflow.
        over = 1 << 31
        packed = over & M32
        round_trips = (packed - (1 << 32) if packed >> 31 else packed) == over
        checks.mutation("an s32-overflowing accumulator breaks the 4-byte round-trip", not round_trips)
        # A "digest" the size of one word over one word's MACs carries no more than a word: the density check notices.
        checks.mutation("a word-sized digest does not beat the word's density",
                        not macs_per_byte(MAC_PER_WORD, BYTES_PER_WORD) > macs_per_byte(MAC_PER_WORD, BYTES_PER_WORD))
        # Bounding the host by PCIe at word level, the 07:00Z error, puts it under every headline: it misses the finding.
        word_level_host = [offload_fraction(64 * GB, mac_rate(d)) < HEADLINE_EPS[d] for d in DEVICES]
        checks.mutation("a word-level PCIe bound on the host would hide HOST-TILES on every device", all(word_level_host))

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
