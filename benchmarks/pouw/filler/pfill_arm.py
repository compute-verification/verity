"""HC-PROVED's arms (approach pouw/filler-proved): pfill, a DRAM-free filler in Theorem E's own setting, so its floor is
proved without CH (note:20261016T0456Z-draft-hc-proved). Every MMA's operands are fresh i.i.d. ±1 noise expanded on chip
from the round's coins and the tile id (A_M = B_M = 0), each MMA's output is a whole prefix sum (k32 on IMMA, k64 on the
e2m1 tile's OMMA), and every one is folded at once into the receipt. pfill-k's noise is {0,1} bits instead of ±1,
and pfill-bu's is bits on A and {±1, ..., ±64} on B (FILL-SEC). The kernel is `pfill.cuh` behind dcap.cu's dfill ABI
(dcap.cu's default filler on this branch); an arm picks the tile's kind at build time through NVCC_APPEND_FLAGS
(-DPFILL_E2M1, -DPFILL_BITS, -DPFILL_BU or nothing), which score.py's nvcc inherits, since it builds after loading the
arm. Placement is FILL v1's
per-point split: on a green split the filler adds ≤ 0.4% to the split's own cost, so decode's slowdown is the split's,
whatever runs on the filler's SMs.

  research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=HC-PROVED \\
    -- python3 benchmarks/pouw/filler/score.py benchmarks/pouw/filler/pfill_arm.py:PFillE2m1Ram
"""

# No `from __future__ import annotations`: arms.load runs this file without registering it in sys.modules, and
# dataclasses needs the module to resolve string annotations.
import os
from dataclasses import dataclass, field

from arms import FILL_V1_SPLIT, NONE, Filler, Placement, point_lookup

SOURCES = ("benchmarks/pouw/decode_capacity/dcap.cu", "benchmarks/pouw/decode_capacity/pfill.cuh",
           "benchmarks/pouw/filler/pfill_arm.py")


@dataclass
class PFillRam:
    """pfill with ram receipts (NC 4: y <- (rotl(y, 7) + d0) a0 + d1, then (rotl(y, 19) + d2) a1 + d3, keys from the
    coins and screened), 8 warps a block, two blocks an SM, 256 fragments a warp a tile, on s8 IMMA m16n8k32."""

    name: str = "pfill-ram"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-v1", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-pm1", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "prefill:none"],
        "hypothesis": "fresh ±1 noise for every MMA with a ram receipt on every k32 prefix sum: Theorem E's floor "
                      "(c/m = 1/4) without CH, at 0.79 of the int8 peak alone on its SMs"})
    sources: tuple = SOURCES
    filler: Filler | None = Filler(W=8, NC=4, S=256, chunk_log2=19)
    choice: str = "FILL v1's offline table of 11 (B, L) splits; a lookup and a green split at serving time"
    nvcc_flags: str = ""

    def __post_init__(self):
        os.environ["NVCC_APPEND_FLAGS"] = self.nvcc_flags

    def placement(self, phase, b, l):
        if phase == "prefill":
            return NONE
        return Placement("split", point_lookup(FILL_V1_SPLIT, b, l))


@dataclass
class PFillDuo(PFillRam):
    """pfill with duo receipts (NC 2), the receipt with no weak key under Lemma B, at three ops a word."""

    name: str = "pfill-duo"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-ram", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-pm1", "receipt:duo-k32", "place:split",
                      "split:per-point-table", "prefill:none"],
        "hypothesis": "pfill with duo for ram: the same proved floor with the stronger receipt, at 0.57 of peak alone"})
    filler: Filler | None = Filler(W=8, NC=2, S=256, chunk_log2=19)


#: HC-POWER's POWER_SPLIT (cursor/hc-power-ea24 benchmarks/pouw/filler/hc_power.py): the largest split its split-only sweep
#: with the scorer's gc_split kept at step + filler <= 1.040 at every scored decode point (r20261016-050639-8537,
#: r20261016-052600-c1e9), the split's own cost whatever runs on the filler's SMs.
POWER_SPLIT = {(1, 2048): 108, (1, 8192): 108, (1, 32768): 108, (32, 1024): 92, (32, 2048): 108, (32, 8192): 124,
               (32, 24576): 132, (48, 16384): 132, (128, 2048): 92, (128, 6144): 108, (132, 5120): 60, (144, 5120): 60,
               (256, 1024): 60, (256, 2048): 60, (256, 3072): 60}


@dataclass
class PFillPowerSplit(PFillRam):
    """pfill-ram on HC-POWER's power-split placement: its 15 decode splits, and 4 filler SMs during prefill (the linears
    on 184, gc_split's only prefill split under 5%; prefill is power-capped and pays its SM share, r20261016-051754-e025)."""

    name: str = "pfill-power-split"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-ram", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-pm1", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "split:every-scored-point", "prefill:split"],
        "hypothesis": "power-split's placement (HC-POWER) is the split's own cost, so it carries to pfill: more filler "
                      "SMs at four decode points and 4 during prefill raise the proved SCORE by its filler share"})
    choice: str = ("HC-POWER's offline table of 15 (B, L) decode splits and one prefill split; a lookup and a green split "
                   "at serving time")

    def placement(self, phase, b, l):
        if phase == "prefill":
            return Placement("split", 4)
        return Placement("split", point_lookup(POWER_SPLIT, b, l))


@dataclass
class PFillE2m1Ram(PFillRam):
    """pfill-ram's tile on OMMA m16n8k64 e2m1 (the same ±1 noise as E2M1 codes, every block scale 1): twice the products
    an MMA at the same receipt words an MMA, committed at L = 64. Its rate is reported in int8 slots (products / 2), the
    unit of the scorer's peak and of Theorem E's floor. That floor is ¼ a product, ½ a slot, only with heads priced at
    cL/m_max = 16 slots; at a CUDA word op's ≈ 8 it is 8/L = ⅛ a product, ¼ a slot, the s8 tile's (pfill_proved.py)."""

    name: str = "pfill-e2m1-ram"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-ram", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-omma", "mma:e2m1-omma-k64", "noise:sha512-coins-pm1", "receipt:ram-k64",
                      "place:split", "split:per-point-table", "prefill:none"],
        "hypothesis": "an E2M1 OMMA does two products an int8 slot at s8's receipt cost an MMA, so the same tile "
                      "certifies 1/2 a slot if heads are priced at 16 slots (L = 64); at 8 a head, s8's 1/4"})
    nvcc_flags: str = "-DPFILL_E2M1"


@dataclass
class PFillK(PFillPowerSplit):
    """pfill-power-split's tile on {0,1} bits (-DPFILL_BITS): every A and B entry its SHA-512 bit as the s8 byte 0 or 1,
    the same s8 IMMA m16n8k32 with C = 0, each s32 word popcount(a & b) folded unscaled into the same ram receipt.
    Theorem K's setting (note:20261016T0341Z-draft-fullfloor-s-r5 §3.6): {0,1} data with targets Σ_k a b at coefficient 1
    in the bits, integer operands affine in the bits, s32 reads unscaled; the noise-only case (x = 0) of round 6's
    dither ⌊x/s⌋ + b. Theorem E's c = ½ covers {0,1} as it does ±1."""

    name: str = "pfill-k"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-power-split", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "split:every-scored-point", "prefill:split"],
        "hypothesis": "{0,1} bits on both operands at the ±1 tile's rate meet Theorem K's hypotheses (int8, affine "
                      "bytes, s32 unscaled), so the same filler reads c = 1 (0.9 a slot) once K is refereed, 0.225 now"})
    filler: Filler | None = Filler(W=8, NC=4, S=256, chunk_log2=19, noise="bits01")
    nvcc_flags: str = "-DPFILL_BITS"


@dataclass
class PFillBU(PFillPowerSplit):
    """pfill-power-split's ram tile as pfill-bU (-DPFILL_BU, note:20261016T0710Z-draft-filler-floor-r2 §5): A's entries
    their SHA-512 bits as s8 0 or 1, B's U = (t & 0x7F) - 64 + ((t >> 6) & 1) from one digest byte t an entry (uniform on
    {±1, ..., ±64}), 512 blocks a tile, on the same s8 IMMA m16n8k32 with C = 0, each s32 word in [-2048, 2048] folded
    unscaled into the same ram receipt at K = L = 32. Theorem K's setting as RS-E-REF r4 refereed it
    (note:20261015T1931Z-draft-theorem-e-referee Part IV): straight-line, cell-pure, int MMAs, unscaled s32 reads,
    operands affine in their bits (U's weights 1, 2, ..., 32, 65 on the odd parity bits t_0, t_6), every prefix word
    committed. dgate adds the uniformity check of U and A; -mpopcnt is for the host reference's seven popcounts a word."""

    name: str = "pfill-bu"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-power-split", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-bU", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "split:every-scored-point", "prefill:split"],
        "hypothesis": "pfill-bU: c = 1 under Theorem K (cell-pure, F₂-affine-parity operands, straight-line, full words; "
                      "Lemma W at L = 32); ¼ under Theorem E."})
    filler: Filler | None = Filler(W=8, NC=4, S=256, chunk_log2=19, noise="bitsU")
    nvcc_flags: str = "-DPFILL_BU -Xcompiler -mpopcnt"


def _fold_lineage(parent: str, noise_tag: str, receipt: str, hypothesis: str) -> dict:
    return {"parent": parent, "approach": "pouw/filler-proved",
            "mechanism": ["filler:fresh-noise-imma", noise_tag, receipt, "place:split", "split:per-point-table",
                          "split:every-scored-point", "prefill:split"],
            "hypothesis": hypothesis}


@dataclass
class PFillBUDuo(PFillBU):
    """pfill-bU with duo receipts (NC 2, three ops a word): a fold with no all-state kernel at a pair or an MMA, so the
    receipt reads every word, where ram's reads a pair only through Q = a0 d0 + d1
    (note:20261016T0917Z-draft-fullfloor-s-r8 §1.2)."""

    name: str = "pfill-bu-duo"
    lineage: dict = field(default_factory=lambda: _fold_lineage(
        "pfill-bu", "noise:sha512-coins-bU", "receipt:duo-k32",
        "pfill-bU with duo for ram: every word certified, so K's c = 1 holds where ram's Q-only receipt halves it"))
    filler: Filler | None = Filler(W=8, NC=2, S=256, chunk_log2=19, noise="bitsU")


@dataclass
class PFillBULi(PFillBU):
    """pfill-bU with P2-FF r9's LOP3+IMAD receipts (NC 4 built with -DPFILL_LI, two ops a pair as ram's:
    y <- (y ^ d0 ^ k0) a0 + d1;  y <- (y ^ d2 ^ k1) a1 + d3, keys under key_ok_r9;
    note:20261016T1117Z-draft-fullfloor-s-r9). It reads every word given (HO), no hand-off sharing a row or column,
    which bU's hashed rows give (pfill_ref.ho_ok). Provisional while RS-E-REF r7 referees r9 (`fold_status`)."""

    name: str = "pfill-bu-li"
    lineage: dict = field(default_factory=lambda: _fold_lineage(
        "pfill-bu", "noise:sha512-coins-bU", "receipt:li-k32",
        "pfill-bU with LOP3+IMAD for ram: every word certified at ram's op count, so K's c = 1 holds (r9, provisional)"))
    nvcc_flags: str = "-DPFILL_BU -DPFILL_LI -Xcompiler -mpopcnt"


@dataclass
class PFillBUXin13(PFillBU):
    """pfill-bU with P2-FF r9's XOR-in (13, 19) receipts (NC 4 built with -DPFILL_XIN13, three ops a pair:
    y <- (rotl(y, 13) ^ d0) a0 + d1;  y <- (rotl(y, 19) ^ d2) a1 + d3, keys under key_ok_r9): every word read with no
    layout premise, the served GEMM's fold. Provisional while RS-E-REF r7 referees r9 (`fold_status`)."""

    name: str = "pfill-bu-xin13"
    lineage: dict = field(default_factory=lambda: _fold_lineage(
        "pfill-bu", "noise:sha512-coins-bU", "receipt:xin13-k32",
        "pfill-bU with XOR-in (13, 19) for ram: every word certified at one op a pair more than ram (r9, provisional)"))
    nvcc_flags: str = "-DPFILL_BU -DPFILL_XIN13 -Xcompiler -mpopcnt"


@dataclass
class PFillKDuo(PFillK):
    """pfill-k with duo receipts (NC 2)."""

    name: str = "pfill-k-duo"
    lineage: dict = field(default_factory=lambda: _fold_lineage(
        "pfill-k", "noise:sha512-coins-01", "receipt:duo-k32",
        "pfill-k with duo for ram: every word certified, so K's c = 1 holds where ram's Q-only receipt halves it"))
    filler: Filler | None = Filler(W=8, NC=2, S=256, chunk_log2=19, noise="bits01")


def fold_share(lineage: dict) -> float:
    """The share of committed words the arm's receipt certifies: ½ for ram (it reads a pair only through Q = a0 d0 +
    d1), 1 for duo, LOP3+IMAD and XOR-in (13, 19) (note:20261016T0917Z-draft-fullfloor-s-r8 §1.2,
    note:20261016T1117Z-draft-fullfloor-s-r9)."""
    mech = lineage.get("mechanism") or []
    return 0.5 if any(t.startswith("receipt:ram") for t in mech) else 1.0


R9_RECEIPTS = ("receipt:li-k32", "receipt:xin13-k32")


def fold_status(lineage: dict) -> str | None:
    """'provisional' for a receipt fold whose argument is under referee: P2-FF r9's LOP3+IMAD and XOR-in (13, 19), until
    RS-E-REF r7 rules."""
    return "provisional" if set(lineage.get("mechanism") or []) & set(R9_RECEIPTS) else None


def products_per_slot(lineage: dict) -> int:
    """Products the arm's filler does an int8 slot it reports: 2 for the e2m1 tile, else 1."""
    return 2 if "mma:e2m1-omma-k64" in (lineage.get("mechanism") or []) else 1


def theorem_k_setting(lineage: dict) -> bool:
    """Whether the arm's filler is one Theorem K reads: {0,1} bits on both sides (pfill-k) or pfill-bU."""
    mech = lineage.get("mechanism") or []
    return "noise:sha512-coins-01" in mech or "noise:sha512-coins-bU" in mech
