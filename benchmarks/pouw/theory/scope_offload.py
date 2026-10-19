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
  - HOST-TILES: the host CPU computing whole s8 tiles, its words on AMX and their fold and credited fillers on AVX-512,
    bounded by its cores' measured rates shared across the GPUs beside it, not by PCIe. A whole tile frees the GPU of
    each word's credited time (tensor, fold and filler), so the share is the host's words a second times the headline
    design's credited time a word (`setup(...).baseline.T`, eps's own denominator), not a share of the tensor peak: the
    B200's design runs its tensor pipe under 30% of peak. The machine model's per-GPU eps has no host term; LOC (decision
    37) deducts the host at node level at its int8 peak, and the data-sheet rows price that peak (AMX's 1,024 int8 MACs a
    cycle a core at Intel ARK's clocks).

Each fraction is read against the 1% target as counted eps, the headline's eps plus the fraction, at the sheet reading
and at decision 89's measured one. The word-level ceilings stay shares of the tensor peak: an imported word still costs
the GPU its fold, so they bound the time it frees from above.

The checkable arithmetic: the committed word packs exactly 32 int8*weight products into an s32 with no wrap (verified
over Z/2^32 and the operand alphabets), and a tile digest carries more MACs a byte than a word even at H1's smallest
tile. With `--device-lean PATH` the script also reads a Device.lean and confirms the model names PCIe and the network as
its side channels and neither NVLink nor a host-RAM channel. That check is off by default, so the run reads only this
package.

Run: `python scope_offload.py [--mutations] [--device-lean Security/Definitions/Accounting/Device.lean] [--verdict PATH]`.
Stdlib, exact_kit, and verity_machine_model for each headline design's eps and credited time a word.
Sources: MAC_rate = s8 macs/SM/clk * SMs * f_max (note:20261017T1818Z-draft-axioms sec 2.1); the headline eps at both
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

# The design's operand alphabets (decisions 29, 44, 51), as isa_escape.py uses: int8 activations, weights |w| <= 63.
INT8 = range(-128, 128)
WEIGHT = range(-63, 64)

# Per device: s8 dense MACs a SM clock, SMs, f_max (Hz). Sheet constants, note:20261017T1818Z-draft-axioms sec 2.1.
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
# 24 Xeon 6776P cores of node 2 (r20261012-211331-ae6a, ncp_resident.jsonl and ncp_stream.jsonl, Y stored). Resident is
# operands blocked in L2, as a large-m (prefill) tile allows; streamed reads Y from DRAM.
AMX_PER_CORE = {"resident": 1.3879e13 / 24, "streamed": 5.2886e12 / 24}
# The fold's and fillers' lane work, per core: LOP3+IMAD chains on AVX-512, best of 10 cores (r20261019-064611-062c,
# Xeon 8580), priced at the design's own lane instructions a word.
LANE_PER_CORE = 5.1349e11 / 10
AMX_MACS_PER_CYCLE = 1024  # TDPBSSD: int8 MACs a cycle a core (Intel's 2,048 int8 ops)

# The hosts beside each device: (name, cores, GPUs sharing them, ARK base and max turbo clocks in Hz). Node 2's VM
# exposes 48 of each 6776P's 64 cores (lscpu, r20261012-211331-ae6a); bare metal has all 128.
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
        checks.check("Device.lean states there is no host-RAM channel (Daniel 3 Oct)", "no host RAM" in lean,
                     "host offload rides the PCIe channel, which the docstring frames as 'from NVMe'")

    # 3. The per-device, per-channel ceiling, and the headline-relevant verdicts.
    eps = {device: headline_eps(device) for device in DEVICES}
    words = {(device, r): credited_word(device, r) for device in DEVICES for r in READINGS}
    rows = []
    worst_uncounted = {}
    for device in DEVICES:
        rate = mac_rate(device)
        pcie = offload_fraction(64 * GB, rate)
        for host, cores, gpus, _, _ in HOSTS[device]:
            tiles = host_tile_fraction(words[device, "sheet"], cores, gpus, AMX_PER_CORE["resident"])
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
    checks.check("sm_120 has no NVLink, so no uncounted intra-node channel", "sm_120 RTX PRO 6000" not in worst_uncounted)
    def counted(device: str, f: float | dict[str, float]) -> tuple[float, float]:
        at = f if isinstance(f, dict) else {r: f for r in READINGS}
        return eps[device]["sheet"] + at["sheet"], eps[device]["measured"] + at["measured"]

    def shown(pair: tuple[float, float]) -> str:
        return f"{pair[0] * 100:.2f}% sheet / {pair[1] * 100:.2f}% measured"

    for device in ("H200 sm_90", "B200 sm_100"):
        f = worst_uncounted[device]
        checks.check(f"{device}: word-level, NVLink lends >=0.3% of credited MACs, unobserved by any certifier",
                     f >= 0.003, f"{f * 100:.3f}%")
    nv_h200 = counted("H200 sm_90", worst_uncounted["H200 sm_90"])
    nv_b200 = counted("B200 sm_100", worst_uncounted["B200 sm_100"])
    checks.check("H200 sm_90: word-level NVLink alone keeps counted eps under 1% on both readings", max(nv_h200) < TARGET,
                 shown(nv_h200) + " (a peer doing whole tiles is RT-SCOPE-1)")
    checks.check("B200 sm_100: word-level NVLink alone stays under 1% at the sheet reading, not at the measured one",
                 nv_b200[0] < TARGET < nv_b200[1], shown(nv_b200))

    # 5. Whole-tile: the host's measured rates, not PCIe, bound what it lends, as a share of the design's credited time.
    #    Paying the design's own fold and fillers, it takes sm_120 over 1% counted on node 2 with resident operands, the
    #    H200 and the B200 on a DGX, and the B200 on every host at the measured reading.
    def lent(device: str, host: int, mode: str, fold: bool = True) -> dict[str, float]:
        _, cores, gpus, _, _ = HOSTS[device][host]
        return {r: host_tile_fraction(words[device, r], cores, gpus, AMX_PER_CORE[mode], fold) for r in READINGS}

    def shown_lent(device: str, f: dict[str, float]) -> str:
        return f"+{f['sheet'] * 100:.3f}% / +{f['measured'] * 100:.3f}%: {shown(counted(device, f))}"

    for device in DEVICES:
        for i, (host, *_) in enumerate(HOSTS[device]):
            for mode in AMX_PER_CORE:
                checks._print(f"      {device:20s} {host:36s} {mode:9s} {shown_lent(device, lent(device, i, mode))}")

    sm120, h200, b200 = "sm_120 RTX PRO 6000", "H200 sm_90", "B200 sm_100"
    folded, upper = lent(sm120, 0, "resident"), lent(sm120, 0, "resident", fold=False)
    checks.check("sm_120: node 2's host (the VM's 96 cores), resident operands, words and fold, takes counted eps over 1%",
                 min(counted(sm120, folded)) > TARGET,
                 shown_lent(sm120, folded) + f" (AMX-only bound +{upper['sheet'] * 100:.2f}%)")
    vm, bare = lent(sm120, 0, "streamed"), lent(sm120, 1, "streamed")
    checks.check("sm_120: operands streamed, node 2's host stays under 1% counted on the VM's 96 cores, over on 128",
                 max(counted(sm120, vm)) < TARGET < min(counted(sm120, bare)),
                 f"96: {shown_lent(sm120, vm)}; 128: {shown_lent(sm120, bare)}")
    least = min(v for i in range(len(HOSTS[sm120])) for m in AMX_PER_CORE for v in lent(sm120, i, m).values())
    checks.check("sm_120: node 2's host is over training C7's 0.3% host target in every case", least > 0.003,
                 f"min +{least * 100:.3f}%")
    for device in (h200, b200):
        dgx = lent(device, 1, "resident")
        checks.check(f"{device}: a DGX's host, resident operands, words and fold, takes counted eps over 1%",
                     min(counted(device, dgx)) > TARGET, shown_lent(device, dgx) + " (GNR per-core AMX rate)")
    slot = lent(h200, 0, "resident")
    checks.check("H200 sm_90: vy-dc-h200's host, resident operands, words and fold, is over 1% at the measured reading",
                 counted(h200, slot)[1] > TARGET, shown_lent(h200, slot))
    worst = min((counted(b200, lent(b200, i, m))[1], HOSTS[b200][i][0], m)
                for i in range(len(HOSTS[b200])) for m in AMX_PER_CORE)
    checks.check("B200 sm_100: every host, resident or streamed, words and fold, is over 1% counted at the measured reading",
                 worst[0] > TARGET, f"least {worst[0] * 100:.2f}% ({worst[1]}, {worst[2]})")
    gpu_seconds, macs, _ = words[b200, "sheet"]
    tensor = macs / gpu_seconds / mac_rate(b200)
    checks.check("B200 sm_100: its headline design's credited program runs the tensor pipe under 30% of peak",
                 tensor < 0.3, f"{tensor:.3f}: a share of the tensor peak understates a whole tile {1 / tensor:.1f}x")

    #    LOC's pricing, the host's AMX at its data-sheet peak (ARK cores, base clock), leaves no device under 1% counted.
    def at_peak(device: str, host: int, clock: float) -> dict[str, float]:
        _, cores, gpus, _, _ = HOSTS[device][host]
        return {r: cores / gpus * words[device, r][0] / (words[device, r][1] / (AMX_MACS_PER_CYCLE * clock))
                for r in READINGS}

    least_peak = min((min(counted(d, at_peak(d, i, h[3]))), d, h[0]) for d in DEVICES for i, h in enumerate(HOSTS[d]))
    checks.check("every device: AMX at its data-sheet peak (base clock) takes counted eps over 1% on every host",
                 least_peak[0] > TARGET, f"least {least_peak[0] * 100:.2f}% ({least_peak[1]}, {least_peak[2]})")
    for d in DEVICES:
        for i, h in enumerate(HOSTS[d]):
            lo, hi = at_peak(d, i, h[3]), at_peak(d, i, h[4])
            checks._print(f"      {d:20s} {h[0]:36s} AMX peak +{lo['sheet'] * 100:.2f}–{hi['sheet'] * 100:.2f}% "
                          f"(sheet), so {counted(d, lo)[0] * 100:.2f}–{counted(d, hi)[0] * 100:.2f}% counted")

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
        # Bounding the host by PCIe at word level, the 07:00Z error, keeps sm_120 under 1% counted: it misses the finding.
        pcie_host = offload_fraction(64 * GB, mac_rate(sm120))
        checks.mutation("a word-level PCIe bound on the host would keep sm_120 under 1% counted",
                        eps[sm120]["sheet"] + pcie_host < TARGET, f"{(eps[sm120]['sheet'] + pcie_host) * 100:.3f}%")
        # A fold priced at no lane work turns the folded estimate into the AMX-only bound: the fold term is not vacuous.
        checks.mutation("dropping the fold's lane work raises node 2's estimate to the AMX-only bound",
                        upper["sheet"] > 2 * folded["sheet"],
                        f"{upper['sheet'] * 100:.3f}% vs {folded['sheet'] * 100:.3f}%")
        # The host as a share of the B200's tensor peak, the 08:15Z error, keeps a DGX B200 under 1% at the sheet reading.
        _, cores, gpus, _, _ = HOSTS[b200][1]
        _, macs, lanes = words[b200, "sheet"]
        peak_share = cores / gpus * macs / (macs / AMX_PER_CORE["resident"] + lanes / LANE_PER_CORE) / mac_rate(b200)
        checks.mutation("a share of the tensor peak, not the credited time, would keep a DGX B200 under 1% counted",
                        eps[b200]["sheet"] + peak_share < TARGET, f"{(eps[b200]['sheet'] + peak_share) * 100:.3f}%")

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
