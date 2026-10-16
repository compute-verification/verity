"""HC-PROVED's arms (approach pouw/filler-proved): pfill, a DRAM-free filler in Theorem E's own setting, so its floor is
proved without CH (note:20261016T0456Z-draft-hc-proved). Every MMA's operands are fresh i.i.d. ±1 noise expanded on chip
from the round's coins and the tile id (A_M = B_M = 0), each MMA's output is a whole prefix sum (k32 on IMMA, k64 on the
e2m1 tile's OMMA), and every one is folded at once into the receipt. The kernel is `pfill.cuh` behind dcap.cu's dfill ABI
(dcap.cu's default filler on this branch); an arm picks the tile's kind through PFILL_MMA, which the scorer's gates and
dscore inherit. Placement is FILL v1's per-point split: on a green split the filler adds ≤ 0.4% to the split's own cost,
so decode's slowdown is the split's, whatever runs on the filler's SMs.

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
    mma: str = "s8"

    def __post_init__(self):
        os.environ["PFILL_MMA"] = self.mma

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


@dataclass
class PFillE2m1Ram(PFillRam):
    """pfill-ram's tile on OMMA m16n8k64 e2m1 (the same ±1 noise as E2M1 codes, every block scale 1): twice the products
    an MMA at the same receipt words an MMA. Its rate is reported in int8 slots (products / 2), the unit of the scorer's
    peak and of Theorem E's floor, which reads it at c = 1/2 a product, 1/2 an int8 slot (pfill_proved.py)."""

    name: str = "pfill-e2m1-ram"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-ram", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-omma", "mma:e2m1-omma-k64", "noise:sha512-coins-pm1", "receipt:ram-k64",
                      "place:split", "split:per-point-table", "prefill:none"],
        "hypothesis": "the floor is in int8 slots and an E2M1 OMMA does two products a slot, so the same tile on OMMA "
                      "certifies 1/2 a slot (proved) against s8's 1/4, at the same receipt cost an MMA"})
    mma: str = "e2m1"


def products_per_slot(lineage: dict) -> int:
    """Products the arm's filler does an int8 slot it reports: 2 for the e2m1 tile, else 1."""
    return 2 if "mma:e2m1-omma-k64" in (lineage.get("mechanism") or []) else 1
