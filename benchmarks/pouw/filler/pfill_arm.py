"""HC-PROVED's arms (approach pouw/filler-proved): pfill, a DRAM-free filler in Theorem E's own setting, so its floor is
proved without CH (note:20261016T0456Z-draft-hc-proved). Every MMA's operands are fresh i.i.d. ±1 noise expanded on chip
from the round's coins and the tile id (A_M = B_M = 0), each MMA's output is a whole prefix sum (k32 on IMMA, k64 on the
e2m1 tile's OMMA), and every one is folded at once into the receipt. pfill-k's noise is {0,1} bits instead of ±1. The
kernel is `pfill.cuh` behind dcap.cu's dfill ABI (dcap.cu's default filler on this branch); an arm picks the tile's kind
at build time through NVCC_APPEND_FLAGS (-DPFILL_E2M1, -DPFILL_BITS or nothing), which score.py's nvcc inherits, since it
builds after loading the arm. Placement is FILL v1's
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
    filler: Filler | None = Filler(W=8, NC=4, S=256, chunk_log2=19, noise="pm1")
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
    filler: Filler | None = Filler(W=8, NC=2, S=256, chunk_log2=19, noise="pm1")


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


@dataclass(frozen=True)
class PhaseFill:
    """pfill-k's phase filler at a point (psched_score.py/pscore.cu's PW PNC PS PT): 4-warp blocks (the build's PF_PW) of
    NC 4 ram tiles, one block a step SM, S fragments a warp a tile (16 W^2 S MMAs a tile) and `tiles` tiles a block a
    layer, a fixed count chosen before the run."""

    W: int = 4
    NC: int = 4
    S: int = 64
    tiles: int = 1


#: (B, L) -> (split SMs, phase filler or None) from pfill_phase_screen.py's s2 and s4 (r20261016-085255-cdd2,
#: r20261016-090209-21c7; 8-warp blocks were 1.08-1.30 everywhere, 2-warp 1.12-2.0): at most ~1.04 at every scored point
#: and at least a tile below each point's cliff (where the phase outlasts attention and o waits on it). The phase pays
#: at B >= 128; at B 32-48 and 256/1K pfill-k's split alone does as well per slowdown, and 1/32K and 32/8K take lighter
#: splits for margin (pfill-k's read 1.046 there; 1/32K at 100 read 1.046 again in r20261016-091121-7bfc, 92 screened
#: 1.027). 192/2K (robust, unscored) read 1.054 there with 4 tiles, so it keeps the split alone.
PHASE_K = {
    (1, 2048): (108, None), (1, 8192): (108, None), (1, 32768): (92, None),
    (8, 16384): (108, None),
    (32, 1024): (92, None), (32, 2048): (108, None), (32, 8192): (108, PhaseFill(tiles=2)), (32, 24576): (132, None),
    (48, 16384): (108, PhaseFill(tiles=8)),
    (64, 4096): (92, None),
    (128, 2048): (76, PhaseFill(tiles=3)), (128, 6144): (92, PhaseFill(tiles=10)),
    (132, 5120): (60, PhaseFill(tiles=6)), (144, 5120): (60, PhaseFill(tiles=8)),
    (192, 2048): (60, None),
    (256, 1024): (60, None), (256, 2048): (60, PhaseFill(tiles=5)), (256, 3072): (60, PhaseFill(tiles=8)),
}


@dataclass
class PFillKPhase(PFillK):
    """pfill-k on the leader's placement (HC-SCHED's fill-v1-phase-att, SCORER's port): its split, re-chosen per point,
    plus pfill-k's own dfill_p at 4 warps (-DPF_PW=4) forked into each decode layer's attention on the step's SMs, at the
    graph's highest node priority, joined before o. Prefill keeps pfill-k's 4-SM split. Run through psched_score.py:

      python3 benchmarks/pouw/filler/psched_score.py benchmarks/pouw/filler/pfill_arm.py:PFillKPhase
    """

    name: str = "pfill-k-phase"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "sched:phase-att", "sched:graph-fork-join", "sched:node-priority",
                      "phase:pfill-tiles", "phase:4-warp-blocks", "prefill:split"],
        "hypothesis": "decode attention leaves the step SMs' tensor pipe idle: pfill-k tiles in 4-warp blocks forked "
                      "into it add ~0.1 coverage at B >= 128 under 1.04, where 8-warp blocks halve attention's occupancy"})
    sources: tuple = SOURCES + ("benchmarks/pouw/filler/pscore.cu", "benchmarks/pouw/filler/psched_score.py")
    choice: str = ("an offline table of 18 (B, L) points (split SMs, phase tiles) from pfill-k's phase screens s2 and s4; "
                   "a lookup, a green split and one graph a step at serving time; run through psched_score.py/pscore.cu")
    nvcc_flags: str = "-DPFILL_BITS -DPF_PW=4"
    table = PHASE_K

    def _row(self, b, l):
        return point_lookup(self.table, b, l)

    def placement(self, phase, b, l):
        if phase == "prefill":
            return Placement("split", 4)
        return Placement("split", self._row(b, l)[0])

    def phase(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[1]


def products_per_slot(lineage: dict) -> int:
    """Products the arm's filler does an int8 slot it reports: 2 for the e2m1 tile, else 1."""
    return 2 if "mma:e2m1-omma-k64" in (lineage.get("mechanism") or []) else 1


def theorem_k_setting(lineage: dict) -> bool:
    """Whether the arm's filler runs on {0,1} bits (pfill-k), the setting Theorem K reads."""
    return "noise:sha512-coins-01" in (lineage.get("mechanism") or [])
