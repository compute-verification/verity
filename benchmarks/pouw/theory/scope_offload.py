"""SCOPE-OFFLOAD, mechanically: how much of a credited GPU's int8 work a helper the prover controls can do OFF that GPU's
facets, and what bounds it.

Where `isa_escape.py` asks whether a unit ON the die can head a committed word more cheaply than the SM menu, this script
asks the same of helpers off the die: the host CPU/AMX, a peer GPU, a machine behind a NIC, or device memory holding
precomputed words. Which of two bounds applies decides the answer:

  - WORD-LEVEL, if the credited GPU must head every committed word itself. A helper can then import only words or
    additive partials (the fixed-word rule lets those in without un-fixing the head). A word is a fresh
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
  - HOST-TILES: the host CPU computing whole s8 tiles, its words on AMX and their fold and credited fillers on AVX-512,
    bounded by its cores' measured rates shared across the GPUs beside it, not by PCIe. A whole tile frees the GPU of
    each word's credited time (tensor, fold and filler), so the share is the host's words a second times the headline
    design's credited time a word (`setup(...).baseline.T`, eps's own denominator), not a share of the tensor peak,
    which understates it by the design's tensor utilization (printed per device). The machine model's per-GPU eps has
    no host term; LOC deducts the host at node level at its int8 peak, and the data-sheet rows price that
    peak (AMX's 1,024 int8 MACs a cycle a core at Intel ARK's clocks).

Each fraction is read against the 1% target as counted eps, the headline's eps plus the fraction, at the sheet reading
and at the measured one, and printed as a `verdict` line. Those verdicts move with whichever designs
headline.toml names, so the checks assert only what holds for any of them: the relations between the bounds, the
sheet-constant ceilings and the arithmetic. The word-level ceilings stay shares of the tensor peak: an imported word
still costs the GPU its fold, so they bound the time it frees from above.

The checkable arithmetic: the committed word packs exactly 32 int8*weight products into an s32 with no wrap (verified
over Z/2^32 and the operand alphabets), and a tile digest carries more MACs a byte than a word even at H1's smallest
tile. With `--device-lean PATH` the script also reads a Device.lean and confirms the model names PCIe and the network as
its side channels and neither NVLink nor a host-RAM channel. That check is off by default, so the run reads only this
package.

Run: `python scope_offload.py [--mutations] [--device-lean Security/Definitions/Accounting/Device.lean] [--verdict PATH]`.
Stdlib, exact_kit, and verity_machine_model for each headline design's eps and credited time a word.
Sources: MAC_rate = s8 macs/SM/clk * SMs * f_max; the headline eps at both
readings, `headline.row` for `machine_model/data/headline.toml`'s designs; PCIe5 x16 64 GB/s one way;
ConnectX-7/BlueField-3 400 Gb/s = 50 GB/s a NIC; NVLink4 450 GB/s and NVLink5 900 GB/s one way; RTX PRO 6000 has no
NVLink; HBM3e 4.8 (H200) / 8 (B200) TB/s, GDDR7 ~1.8 TB/s (sm_120); device memory 96 / 141 / 192 GB; H1's smallest tile
4 x 8 at K = 64 (`h1tile.circuit-check.py`). Host rates: `benchmarks/host_tiles` runs, cited where they're used; host
core counts from each run's lscpu, and NVIDIA's DGX H200 / B200 data sheets (2 x 56 cores beside 8 GPUs); cores and
clocks of the Xeon 6776P, 8468, 8480C, 8570 and 8580 from Intel ARK.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from exact_kit.checks import Checks
from verity_machine_model.eps import design, machine, setup
from verity_machine_model.headline import CREDIT, OPERANDS, READINGS, row, spec
from verity_machine_model.model import device as family_data

MAC_PER_WORD = 32  # a committed word sums 32 int8 products (FOLD-BW)
BYTES_PER_WORD = 4  # a fresh s32
DIGEST_BYTES = 64  # a tile's SHA-512 digest, the transcript's leaf
H1_SMALLEST_TILE_MACS = 4 * 8 * 64  # one lane (4 x 8) at K = 64
M32 = 0xFFFFFFFF

# The design's operand alphabets, as isa_escape.py uses them: int8 activations, weights |w| <= 63.
INT8 = range(-128, 128)
WEIGHT = range(-63, 64)

# Per device: s8 dense MACs a SM clock, SMs, f_max (Hz). Sheet constants.
DEVICES = {
    "sm_120 RTX PRO 6000": (1024, 188, 2430e6),
    "H200 sm_90": (4096, 132, 1980e6),
    "B200 sm_100": (8192, 148, 1965e6),
}

# Per device, its headline.toml family; the headline eps and design come from it at each reading.
FAMILY = {"sm_120 RTX PRO 6000": "sm_120", "H200 sm_90": "sm_90", "B200 sm_100": "sm_100"}
TARGET = 0.01
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

# Host AMX words, per core: `ncp_amx` with a snapshot every 32 deep (k_s = 32, one word per 32 MACs a cell), best m, on
# 24 Xeon 6776P cores of node 2 (ncp_resident.jsonl and ncp_stream.jsonl, Y stored). Resident is
# operands blocked in L2, as a large-m (prefill) tile allows; streamed reads Y from DRAM.
AMX_PER_CORE = {"resident": 1.3879e13 / 24, "streamed": 5.2886e12 / 24}
# Sapphire and Emerald Rapids, resident, the same bench one thread a core, gated bit-exact, best m: vy-dc-h200's 8 Xeon
# 8468 cores, 4.3% under node 2's a core, for the H200's hosts; vy-dc-b200's 10 Xeon 8580 cores,
# 46% over, for the B200's. Streamed keeps node 2's rate: the slots' cores stream faster
# (2.8486e12 on 8, 4.4766e12 on 10), which a socket's 56 cores would not.
AMX_RESIDENT_BY_HOST = {"vy-dc-h200 slot, Xeon 8468": 4.4278e12 / 8, "DGX H200, 2 x Xeon 8480C": 4.4278e12 / 8,
                        "vy-dc-b200 slot, Xeon 8580": 8.4242e12 / 10, "DGX B200, 2 x Xeon 8570": 8.4242e12 / 10}
# The fold's and fillers' lane work, per core: LOP3+IMAD chains on AVX-512, best of 10 cores
# (Xeon 8580), priced at the design's own lane instructions a word.
LANE_PER_CORE = 5.1349e11 / 10
AMX_MACS_PER_CYCLE = 1024  # TDPBSSD: int8 MACs a cycle a core (Intel's 2,048 int8 ops)

# The hosts beside each device: (name, cores, GPUs sharing them, ARK base and max turbo clocks in Hz). Node 2's VM
# exposes 48 of each 6776P's 64 cores (lscpu); bare metal has all 128.
HOSTS = {
    "sm_120 RTX PRO 6000": [("node 2 VM, 2 x Xeon 6776P", 96, 8, 2.3e9, 3.9e9),
                            ("node 2 bare metal, 2 x Xeon 6776P", 128, 8, 2.3e9, 3.9e9)],
    "H200 sm_90": [("vy-dc-h200 slot, Xeon 8468", 8, 1, 2.1e9, 3.8e9),
                   ("DGX H200, 2 x Xeon 8480C", 112, 8, 2.0e9, 3.8e9)],
    "B200 sm_100": [("vy-dc-b200 slot, Xeon 8580", 10, 1, 2.0e9, 4.0e9),
                    ("DGX B200, 2 x Xeon 8570", 112, 8, 2.1e9, 4.0e9)],
}


def mac_rate(device: str) -> float:
    m, sms, f = DEVICES[device]
    return m * sms * f


def amx_rate(host: str, mode: str) -> float:
    """The host's AMX words a second a core, MACs, for `mode` ("resident" or "streamed")."""
    return AMX_RESIDENT_BY_HOST[host] if mode == "resident" and host in AMX_RESIDENT_BY_HOST else AMX_PER_CORE[mode]


def offload_fraction(bw: float, rate: float) -> float:
    """WORD-LEVEL: the most of a GPU's credited MACs a helper can supply over a channel of `bw` bytes/s when the GPU must
    head every word, so each word or additive partial crosses the channel."""
    return bw * MAC_PER_WORD / BYTES_PER_WORD / rate


def headline_eps(device: str) -> dict[str, float]:
    """The headline eps of `device`'s family at each reading."""
    return {r: row(FAMILY[device], r).core for r in READINGS}


def credited_word(device: str, reading: str) -> tuple[float, int, float]:
    """The headline design's credited word at `reading`: the GPU's seconds a word in the credited program (eps's
    denominator, SMSP-cycles a word over the SMSPs at the declared clock), its MACs (L), and its lane instructions a
    word (the fold and every credited filler)."""
    fam = FAMILY[device]
    f = spec()["family"][fam]
    dd = family_data(fam)
    d = design(f["design"], dd)
    s = setup(d, machine(f[reading]["vector"], dd), dd, OPERANDS, CREDIT)
    _, sms, clock = DEVICES[device]
    lanes = d.fold_cost() + sum(x.per_word for x in d.filler if x.p < 1)
    return s.baseline.T / (4 * sms * clock), d.L, lanes


def host_tile_fraction(word: tuple[float, int, float], cores: int, gpus: int, amx: float, fold: bool = True) -> float:
    """WHOLE-TILE: the share of one GPU's credited time a host frees computing whole tiles, each word's L MACs on AMX at
    `amx` MACs/s a core, then its lane instructions on the same core, split across the `gpus` beside it. No link term:
    a tile crosses as one digest. With fold False it is an upper bound."""
    gpu_seconds, macs, lanes = word
    host_seconds = macs / amx + (lanes / LANE_PER_CORE if fold else 0.0)
    return cores / gpus * gpu_seconds / host_seconds


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
        checks.check("Device.lean states there is no host-RAM channel", "no host RAM" in lean,
                     "host offload rides the PCIe channel, which the docstring frames as 'from NVMe'")

    # 3. The per-device, per-channel ceiling, and the counted eps each bound gives, against 1%.
    eps = {device: headline_eps(device) for device in DEVICES}
    words = {(device, r): credited_word(device, r) for device in DEVICES for r in READINGS}
    sm120, h200, b200 = "sm_120 RTX PRO 6000", "H200 sm_90", "B200 sm_100"

    def counted(device: str, f: float | dict[str, float]) -> tuple[float, float]:
        at = f if isinstance(f, dict) else {r: f for r in READINGS}
        return eps[device]["sheet"] + at["sheet"], eps[device]["measured"] + at["measured"]

    def shown(pair: tuple[float, float]) -> str:
        return f"{pair[0] * 100:.2f}% sheet / {pair[1] * 100:.2f}% measured"

    def against(pair: tuple[float, float]) -> str:
        over = [r for r, v in zip(READINGS, pair) if v > TARGET]
        if len(over) == len(READINGS):
            return "over 1% at both readings"
        return f"over 1% at the {over[0]} reading only" if over else "under 1% at both readings"

    def lent(device: str, host: int, mode: str, fold: bool = True) -> dict[str, float]:
        name, cores, gpus, _, _ = HOSTS[device][host]
        return {r: host_tile_fraction(words[device, r], cores, gpus, amx_rate(name, mode), fold) for r in READINGS}

    def at_peak(device: str, host: int, clock: float) -> dict[str, float]:
        _, cores, gpus, _, _ = HOSTS[device][host]
        return {r: cores / gpus * words[device, r][0] / (words[device, r][1] / (AMX_MACS_PER_CYCLE * clock))
                for r in READINGS}

    def shown_lent(device: str, f: dict[str, float]) -> str:
        return f"+{f['sheet'] * 100:.3f}% / +{f['measured'] * 100:.3f}%: {shown(counted(device, f))}"

    rows = []
    worst_uncounted = {}
    for device in DEVICES:
        rate = mac_rate(device)
        pcie = offload_fraction(64 * GB, rate)
        for i, (host, *_) in enumerate(HOSTS[device]):
            tiles = min(lent(device, i, "resident").values())
            checks.check(f"{device}: {host} doing whole tiles, words and fold, lends more than PCIe's word-level floor",
                         tiles > pcie, f"{tiles * 100:.3f}% vs {pcie * 100:.3f}% (HOST-TILES is not link-bound)")
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
    checks.check("sm_120 has no NVLink, so no uncounted intra-node channel", sm120 not in worst_uncounted)
    for device in (h200, b200):
        f = worst_uncounted[device]
        checks.check(f"{device}: word-level, NVLink lends >=0.3% of credited MACs, unobserved by any certifier",
                     f >= 0.003, f"{f * 100:.3f}%")
        nv = counted(device, f)
        checks._print(f"      verdict {device:20s} word-level NVLink alone: {shown(nv)}: {against(nv)} "
                      "(a peer doing whole tiles is a separate route, not bounded here)")

    # 5. Whole-tile: the host's measured rates, not PCIe, bound what it lends, as a share of the design's credited time,
    #    paying the design's own fold and fillers.
    for device in DEVICES:
        for i, (host, *_) in enumerate(HOSTS[device]):
            for mode in AMX_PER_CORE:
                checks._print(f"      {device:20s} {host:36s} {mode:9s} {shown_lent(device, lent(device, i, mode))}")
    for device in DEVICES:
        cases = [(counted(device, lent(device, i, m)), host, m) for i, (host, *_) in enumerate(HOSTS[device])
                 for m in AMX_PER_CORE]
        most = max(cases, key=lambda c: min(c[0]))
        least = min(cases, key=lambda c: min(c[0]))
        checks._print(f"      verdict {device:20s} host, whole tiles: most {shown(most[0])} ({most[1]}, {most[2]}): "
                      f"{against(most[0])}; least {shown(least[0])} ({least[1]}, {least[2]}): {against(least[0])}")
    for device in DEVICES:
        use = {r: words[device, r][1] / words[device, r][0] / mac_rate(device) for r in READINGS}
        checks.check(f"{device}: the headline design's credited program runs the tensor pipe at or under its peak, so "
                     "a share of the peak understates a whole tile's share by 1/u",
                     all(0 < u <= 1 for u in use.values()),
                     f"u = {use['sheet']:.3f} sheet / {use['measured']:.3f} measured")

    #    LOC's pricing, the host's AMX at its data-sheet peak (ARK cores, base clock to max turbo, no fold), bounds the
    #    measured route from above on every host.
    for d in DEVICES:
        for i, h in enumerate(HOSTS[d]):
            lo, hi, measured = at_peak(d, i, h[3]), at_peak(d, i, h[4]), lent(d, i, "resident")
            checks.check(f"{d}: {h[0]}: AMX at its data-sheet base clock lends at least the measured route",
                         all(lo[r] >= measured[r] for r in READINGS),
                         f"+{lo['sheet'] * 100:.2f}% vs +{measured['sheet'] * 100:.3f}% (sheet)")
            checks._print(f"      {d:20s} {h[0]:36s} AMX peak +{lo['sheet'] * 100:.2f}–{hi['sheet'] * 100:.2f}% "
                          f"(sheet), so {counted(d, lo)[0] * 100:.2f}–{counted(d, hi)[0] * 100:.2f}% counted")
    least_peak = min((counted(d, at_peak(d, i, h[3])), d, h[0]) for d in DEVICES for i, h in enumerate(HOSTS[d]))
    checks._print(f"      verdict AMX at its data-sheet peak (base clock), every device and host: least "
                  f"{shown(least_peak[0])} ({least_peak[1]}, {least_peak[2]}): {against(least_peak[0])}")

    # 6. Device memory: its word stock covers under 0.2% of a 1-s window, and on main early work is counted work.
    for device in DEVICES:
        rate = mac_rate(device)
        stock = MEMORY_BYTES[device] * MAC_PER_WORD / BYTES_PER_WORD / rate
        checks.check(f"{device}: device memory's word stock covers under 0.2% of a 1-s window", stock < 0.002,
                     f"{stock * 1e3:.2f} ms of compute (COIN-BLOCKED: one interval per salt on main)")

    if checks.mutations:
        b200_rate = mac_rate(b200)
        # Mis-stating the word as 2 bytes would halve the compression and double every ceiling: the arithmetic notices.
        inflated = 900 * GB * MAC_PER_WORD / 2 / b200_rate
        checks.mutation("a word mis-sized at 2 bytes doubles the ceiling",
                        inflated > 1.9 * offload_fraction(900 * GB, b200_rate),
                        f"{inflated * 100:.3f}% vs {offload_fraction(900 * GB, b200_rate) * 100:.3f}%")
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
        hosts = [(d, i) for d in DEVICES for i in range(len(HOSTS[d]))]
        # Bounding the host by PCIe at word level understates the whole-tile share on every host.
        ratio = min(min(lent(d, i, "resident").values()) / offload_fraction(64 * GB, mac_rate(d)) for d, i in hosts)
        checks.mutation("a word-level PCIe bound on the host understates its whole-tile share on every host",
                        ratio > 1, f"by {ratio:.1f}x at the least")
        # A fold priced at no lane work turns the folded estimate into the AMX-only bound: the fold term is not vacuous.
        raised = min(lent(d, i, "resident", fold=False)[r] / lent(d, i, "resident")[r]
                     for d, i in hosts for r in READINGS)
        checks.mutation("dropping the fold's lane work raises every host's estimate to the AMX-only bound", raised > 1,
                        f"by {raised:.2f}x at the least")
        # The host as a share of the tensor peak understates it by 1/u on the least-utilized design.
        u, device, reading = min((words[d, r][1] / words[d, r][0] / mac_rate(d), d, r)
                                 for d in DEVICES for r in READINGS)
        name, cores, gpus, _, _ = HOSTS[device][-1]
        _, macs, lanes = words[device, reading]
        host_seconds = macs / amx_rate(name, "resident") + lanes / LANE_PER_CORE
        peak_share = cores / gpus * macs / host_seconds / mac_rate(device)
        share = lent(device, len(HOSTS[device]) - 1, "resident")[reading]
        checks.mutation("a share of the tensor peak, not the credited time, understates the host where a design runs "
                        "under peak", peak_share < share,
                        f"{device} at the {reading} reading, {name}: +{peak_share * 100:.3f}% vs +{share * 100:.3f}% "
                        f"(1/u = {1 / u:.1f}x)")

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
