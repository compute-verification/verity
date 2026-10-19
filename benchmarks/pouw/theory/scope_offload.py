"""SCOPE-OFFLOAD, mechanically: the ceiling on credited int8 work a prover can do OFF the credited GPU's facets and feed
in, and which ingress channels the model counts (RT-SCOPE, the compute-accounting red team).

Where `isa_escape.py` asks whether a unit ON the die can head a committed word more cheaply than the SM menu, this script
asks the complementary question: how much of a credited GPU's work can a helper the prover controls -- the host CPU/AMX,
a DPU/SmartNIC, a peer GPU, or device memory holding precomputed words -- supply through an ingress channel, and whether
that channel is charged.

The one fact that bounds every off-facet helper: a committed word is a fresh s32 that packs 32 int8 products (32 MACs a
word; FOLD-BW), so it is 4 bytes carrying 32 MACs = 8 MACs a byte. A helper that feeds computed words (or additive
partial sums, which the fixed-word rule lets be imported without un-fixing the head, card 0f7721ff) into the credited GPU
is therefore bandwidth-limited to

    f_offload <= BW_in * (MACs_per_word / bytes_per_word) / MAC_rate = 8 * BW_in / MAC_rate

of the GPU's credited MACs, whatever the helper's own compute. So the binding constraint is the ingress bandwidth and
whether it is counted as `help` (A5 `PouwHelpBound`, `Security/Models/Accounting/Assumptions/PouwWork.lean`), not the
helper's FLOPS. This script computes the ceiling per device and channel from the sheet's MAC rates, and classifies each
channel by whether the model charges it:

  - COUNTED-H: the channel is a modeled side channel (`Device.lean`'s `ι`: PCIe, network), signed by the inter-node
    network certifiers, so its ingress is in `help` and bounded by H.
  - UNCOUNTED-NVLINK: intra-node NVLink/NVSwitch, which the inter-node certifier does not observe and which is not in
    `Device.lean`'s `ι`; its ingress is uncounted, and `PouwLocalOff` holds only at H = 0 and is itself conditional on the
    open conjecture A3 (`PouwFloorOff`), crediting "the node ... not which of its GPUs did". This is RT-SCOPE's top
    finding: on H200/B200 it reaches ~0.3% of credited MACs.
  - COIN-BLOCKED: device memory (HBM/GDDR) is fast enough (~3%) to stream precomputed words, but the committed words are
    live-coin-dependent, so this line is unreachable UNLESS the fixed-word rule fails.
  - HOST-UNCREDITED: the host CPU/AMX computes a large fraction of a GPU's int8 rate but its delivery is PCIe-capped; it
    is neither credited nor bounded by PoUW today (CMP-49), a datacenter-r finding, not a per-device-eps one.

The checkable arithmetic (verified bit-exact over Z/2^32 and the operand alphabets): the committed word packs exactly 32
int8*weight products into an s32 with no wrap, so the 8-MACs-a-byte compression is exact. With `--device-lean PATH` the
script also reads a Device.lean and confirms the model names PCIe and the network as its side channels and neither NVLink
nor a host-RAM channel (the live check behind UNCOUNTED-NVLINK); it is off by default so the run reads only this package.

Run: `python scope_offload.py [--mutations] [--device-lean Security/Definitions/Accounting/Device.lean] [--verdict PATH]`.
Stdlib + exact_kit only.
Sources: MAC_rate = s8 macs/SM/clk * SMs * f_max (note:20261017T1818Z-draft-axioms sec 2.1); PCIe5 x16 64 GB/s one way;
ConnectX-7/BlueField-3 400 Gb/s = 50 GB/s a NIC; NVLink4 450 GB/s and NVLink5 900 GB/s one way; RTX PRO 6000 has no
NVLink; HBM3e 4.8 (H200) / 8 (B200) TB/s, GDDR7 ~1.8 TB/s (sm_120).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from exact_kit.checks import Checks

MAC_PER_WORD = 32  # a committed word sums 32 int8 products (FOLD-BW)
BYTES_PER_WORD = 4  # a fresh s32
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

# Per device, one-way ingress into one chip (bytes/s) and the channel's model class.
GB = 1e9
TB = 1e12
CHANNELS = {
    "sm_120 RTX PRO 6000": [
        ("PCIe5 x16", 64 * GB, "COUNTED-H"),
        ("1 NIC (400Gb GDR)", 50 * GB, "COUNTED-H"),
        ("host AMX over PCIe", 64 * GB, "HOST-UNCREDITED"),
        ("GDDR7 (precomputed)", 1.8 * TB, "COIN-BLOCKED"),
    ],
    "H200 sm_90": [
        ("PCIe5 x16", 64 * GB, "COUNTED-H"),
        ("node 8-NIC", 400 * GB, "COUNTED-H"),
        ("NVLink4 in", 450 * GB, "UNCOUNTED-NVLINK"),
        ("host AMX over PCIe", 64 * GB, "HOST-UNCREDITED"),
        ("HBM3e (precomputed)", 4.8 * TB, "COIN-BLOCKED"),
    ],
    "B200 sm_100": [
        ("PCIe5 x16", 64 * GB, "COUNTED-H"),
        ("node 8-NIC", 400 * GB, "COUNTED-H"),
        ("NVLink5 in", 900 * GB, "UNCOUNTED-NVLINK"),
        ("host AMX over PCIe", 64 * GB, "HOST-UNCREDITED"),
        ("HBM3e (precomputed)", 8 * TB, "COIN-BLOCKED"),
    ],
}

# Host int8 compute (not delivery): dual-socket Granite Rapids AMX, ~1024 int8 MACs/clk/core, 256 cores, 2 GHz.
HOST_MAC_RATE = 256 * 2.0e9 * 1024


def mac_rate(device: str) -> float:
    m, sms, f = DEVICES[device]
    return m * sms * f


def offload_fraction(bw: float, rate: float) -> float:
    """The most of a GPU's credited MACs a helper can supply over a channel of `bw` bytes/s."""
    return bw * MAC_PER_WORD / BYTES_PER_WORD / rate


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

    # 1. The compression fact that bounds every off-facet helper.
    checks.check("a committed word packs 32 int8 MACs into one 4-byte s32 (no wrap), so 8 MACs a byte",
                 committed_word_packs_32_macs(rng),
                 f"{MAC_PER_WORD} MACs / {BYTES_PER_WORD} bytes = {MAC_PER_WORD // BYTES_PER_WORD} MACs/byte")

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

    # 4. The findings as checks: PCIe-capped host/COUNTED channels stay well under eps on every device; the UNCOUNTED
    #    NVLink channel reaches the eps budget's order on H200/B200 and is the top finding.
    for device in DEVICES:
        rate = mac_rate(device)
        pcie = offload_fraction(64 * GB, rate)
        checks.check(f"{device}: a PCIe-capped helper is bounded under 0.12% of credited MACs", pcie < 0.0012,
                     f"{pcie * 100:.3f}%")
    checks.check("sm_120 has no NVLink, so no uncounted intra-node channel", "sm_120 RTX PRO 6000" not in worst_uncounted)
    for device in ("H200 sm_90", "B200 sm_100"):
        f = worst_uncounted[device]
        checks.check(f"{device}: NVLink lends >=0.3% of credited MACs, uncounted by the inter-node certifier",
                     f >= 0.003, f"{f * 100:.3f}%")

    # 5. Device memory could stream precomputed words at ~3%, so the live coin is load-bearing.
    for device in DEVICES:
        rate = mac_rate(device)
        hbm_bw = dict((n, bw) for n, bw, c in CHANNELS[device] if c == "COIN-BLOCKED")
        f = offload_fraction(max(hbm_bw.values()), rate)
        checks.check(f"{device}: device memory could stream >1% of words if not coin-blocked", f > 0.01,
                     f"{f * 100:.2f}% -- blocked only by the live coin")

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

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
