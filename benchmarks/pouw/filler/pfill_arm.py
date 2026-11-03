"""The proved-floor arms: pfill, a DRAM-free filler in Theorem E's own setting, so its floor is
proved without CH. Every MMA's operands are fresh i.i.d. ±1 noise expanded on chip
from the round's coins and the tile id (A_M = B_M = 0), each MMA's output is a whole prefix sum (k32 on IMMA, k64 on the
e2m1 tile's OMMA), and every one is folded at once into the receipt. pfill-k's noise is {0,1} bits instead of ±1,
and pfill-bu's is bits on A and {±1, ..., ±64} on B. The kernel is `pfill.cuh` behind dcap.cu's dfill ABI
(dcap.cu's default filler on this branch); an arm picks the tile's kind at build time through NVCC_APPEND_FLAGS
(-DPFILL_E2M1, -DPFILL_BITS, -DPFILL_BU or nothing), which score.py's nvcc inherits, since it builds after loading the
arm. Placement is FILL v1's
per-point split: on a green split the filler adds ≤ 0.4% to the split's own cost, so decode's slowdown is the split's,
whatever runs on the filler's SMs.

  research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \\
    -- python3 benchmarks/pouw/filler/score.py benchmarks/pouw/filler/pfill_arm.py:PFillE2m1Ram
"""

# No `from __future__ import annotations`: arms.load runs this file without registering it in sys.modules, and
# dataclasses needs the module to resolve string annotations.
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # fused_arm (the fused tables' FusedStep), wherever loaded from
from arms import FILL_V1_SPLIT, NONE, Filler, Placement, point_lookup

SOURCES = ("benchmarks/pouw/filler/dcap.cu", "benchmarks/pouw/filler/pfill.cuh",
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


#: POWER_SPLIT: the largest split its split-only sweep with the scorer's gc_split kept at step + filler <= 1.040 at
#: every scored decode point, the split's own cost whatever runs on the filler's SMs.
POWER_SPLIT = {(1, 2048): 108, (1, 8192): 108, (1, 32768): 108, (32, 1024): 92, (32, 2048): 108, (32, 8192): 124,
               (32, 24576): 132, (48, 16384): 132, (128, 2048): 92, (128, 6144): 108, (132, 5120): 60, (144, 5120): 60,
               (256, 1024): 60, (256, 2048): 60, (256, 3072): 60}


@dataclass
class PFillPowerSplit(PFillRam):
    """pfill-ram on the power-split placement: its 15 decode splits, and 4 filler SMs during prefill (the linears
    on 184, gc_split's only prefill split under 5%; prefill is power-capped and pays its SM share)."""

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
    Theorem K's setting: {0,1} data with targets Σ_k a b at coefficient 1
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


#: (B, L) -> (split SMs, phase filler or None) from pfill_phase_screen.py's s2 and s4 (8-warp blocks were 1.08-1.30
#: everywhere, 2-warp 1.12-2.0): at most ~1.04 at every scored point and at least a tile below each point's cliff (where
#: the phase outlasts attention and o waits on it). The phase pays at B >= 128; at B 32-48 and 256/1K pfill-k's split
#: alone does as well per slowdown, and 1/32K and 32/8K take lighter splits for margin (pfill-k's read 1.046 there;
#: 1/32K at 100 read 1.046 again, 92 screened 1.027). 192/2K (robust, unscored) read 1.054 there with 4 tiles, so it
#: keeps the split alone.
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
    """pfill-k on the leader's placement (psched's fill-v1-phase-att, SCORER's port): its split, re-chosen per point,
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


def _flin(sms, lm=1, aps=1):
    from fused_arm import FusedStep
    return sms, FusedStep(FW=0, AFW=0, APS=aps, ALZ=0, LT=0, LM=lm)


#: (B, L) -> (split SMs, FILL's step with no filler warps or None for cuBLASLt's): FILL's FusedPump rows (fused_arm.py
#: TABLE_PUMP; every gate under v3: 0.92-1.005 at B <= 32 on 48 step SMs, 0.962 at 128/2K on 88,
#: 1.019-1.020 at 132/5K and 144/5K on 112) where they give the pump more SMs than POWER_SPLIT, POWER_SPLIT elsewhere
#: (48/16K 132, 128/6K 108, B 256 60: FusedPump's 116, 92 and 44 are fewer). s7: POWER_SPLIT's
#: B >= 128 rows are at cuBLASLt's cliff (every larger split 1.050-1.121).
FUSED_K = {
    (1, 2048): _flin(140, aps=0), (1, 8192): _flin(140, aps=0), (1, 32768): _flin(140), (8, 16384): _flin(140),
    (32, 1024): _flin(140), (32, 2048): _flin(140), (32, 8192): _flin(140), (32, 24576): _flin(132),
    (48, 16384): (132, None), (64, 4096): _flin(124),
    (128, 2048): _flin(100, lm=0), (128, 6144): (108, None),
    (132, 5120): _flin(76, lm=0), (144, 5120): _flin(76, lm=0), (192, 2048): _flin(60, lm=0),
    (256, 1024): (60, None), (256, 2048): (60, None), (256, 3072): (60, None),
}


@dataclass
class PFillKFused(PFillK):
    """pfill-k's pump beside FILL's decode step with no filler warps (flin's GEMM, persistent attention, lm_head through
    flin at B <= 64; fused_arm.FusedPump's form) where that step leaves the pump more SMs, cuBLASLt's step on POWER_SPLIT
    elsewhere, and pfill-k's 4-SM prefill split. No arm-side fixed work, so filler-scorer/v3 reads its filler at the
    scorer's own build (Theorem K's entry). Run through fused_score.py:

      python3 benchmarks/pouw/filler/fused_score.py benchmarks/pouw/filler/pfill_arm.py:PFillKFused
    """

    name: str = "pfill-k-fused"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FILL's step without filler warps runs B <= 32 decode on 48 SMs at 0.92-1.0 of cuBLASLt's, so "
                      "pfill-k's pump takes 140 SMs there and 76-100 at B 128-144: Theorem K's 0.9 on more SMs, with no "
                      "fixed work"})
    sources: tuple = SOURCES + ("benchmarks/pouw/filler/flin.cuh", "benchmarks/pouw/filler/fscore.cu",
                                "benchmarks/pouw/filler/fused_arm.py", "benchmarks/pouw/filler/fused_score.py")
    choice: str = ("an offline table of 18 (B, L) points (split SMs, FILL's step or cuBLASLt's) from FILL's FusedPump rows "
                   "and POWER_SPLIT; a lookup, a green split and one graph a step at serving time")
    table = FUSED_K

    def _row(self, b, l):
        return point_lookup(self.table, b, l)

    def placement(self, phase, b, l):
        if phase == "prefill":
            return Placement("split", 4)
        return Placement("split", self._row(b, l)[0])

    def fused(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[1]


#: FUSED_K with the pump on 84 SMs at 132/5K and 144/5K and 108 at 128/2K: screens s8 and s9,
#: 1.032-1.033 at 132/5K and 144/5K on 104 step SMs over four rows, 0.998 at 128/2K on 80. The
#: step's SMs come in eights, so the next pump size, 92 at 132/5K (1.041 twice) and 116 at 128/2K (1.034), sits closer
#: to the gate than prefill's split; lm_head through flin (1.032) and the linears on cuBLASLt (1.088) gain nothing there.
FUSED_K2 = {**FUSED_K, (128, 2048): _flin(108, lm=0), (132, 5120): _flin(84, lm=0), (144, 5120): _flin(84, lm=0)}


@dataclass
class PFillKFused2(PFillKFused):
    """pfill-k-fused on FUSED_K2: 8 more pump SMs at the reasoning and RAG mixes' decode points (132/5K, 144/5K) and at
    128/2K."""

    name: str = "pfill-k-fused2"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k-fused", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FILL's step on 104 SMs holds 132/5K and 144/5K at 1.033 (pfill-k-fused's 112: 1.020), so 8 more "
                      "pump SMs there lift the reasoning mix, about half of SCORE's time, by 0.03"})
    choice: str = ("an offline table of 18 (B, L) points (split SMs, FILL's step or cuBLASLt's) from FILL's FusedPump rows, "
                   "POWER_SPLIT and screens s8-s9; a lookup, a green split and one graph a step at serving time")
    table = FUSED_K2


#: FUSED_K2 with the pump on 92 SMs at 132/5K (s9: 1.041 twice, margin 0.009 to the gate against prefill's 0.011)
FUSED_K3 = {**FUSED_K2, (132, 5120): _flin(92, lm=0)}


@dataclass
class PFillKFused3(PFillKFused2):
    """pfill-k-fused2 with 8 more pump SMs at 132/5K, the reasoning mix's decode: more coverage, less margin there."""

    name: str = "pfill-k-fused3"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k-fused2", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FILL's step on 96 SMs holds 132/5K at 1.041, so the pump on 92 lifts the reasoning mix by another "
                      "0.03 inside the gate"})
    table = FUSED_K3


#: FUSED_K3 with the next pump size where its slowdown repeats inside prefill's margin (s8-s10):
#: 128/2K 116 (1.034 x2), 128/6K 116 on FILL's step (1.035, 1.030), 32/24K 140 (1.040 x2), 144/5K 92 (1.039). B 256
#: stays on cuBLASLt's 60: FILL's step there reads 1.08-1.11 with flin's linears and gains nothing with cuBLASLt's.
FUSED_K4 = {**FUSED_K3, (128, 2048): _flin(116, lm=0), (128, 6144): _flin(116, lm=0), (32, 24576): _flin(140),
            (144, 5120): _flin(92, lm=0)}


@dataclass
class PFillKFused4(PFillKFused3):
    """pfill-k-fused3 with 8 more pump SMs at 128/2K, 128/6K, 32/24K and 144/5K."""

    name: str = "pfill-k-fused4"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k-fused3", "approach": "pouw/filler-proved",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FILL's step holds 128/2K and 128/6K on 72 SMs, 32/24K on 48 and 144/5K on 96 at 1.030-1.040, so "
                      "the pump's next 8 SMs there add ~0.035 coverage a point inside the gate"})
    table = FUSED_K4


def _fused_lineage(noise_tag: str, receipt: str, hypothesis: str, parent: str = "pfill-k-fused4", extra=()) -> dict:
    return {"parent": parent, "approach": "pouw/filler-proved",
            "mechanism": ["filler:fresh-noise-imma", noise_tag, receipt, "place:split", "split:per-point-table",
                          "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin", *extra,
                          "prefill:split"],
            "hypothesis": hypothesis}


@dataclass
class PFillBUDuoFused4(PFillKFused4):
    """pfill-k-fused4's placement (FUSED_K4) pumping pfill-bU with duo receipts (PFillBUDuo's tile): the filler of record
    on the fold that's proved now."""

    name: str = "pfill-bu-duo-fused4"
    lineage: dict = field(default_factory=lambda: _fused_lineage(
        "noise:sha512-coins-bU", "receipt:duo-k32",
        "pfill-bU with duo on fused4's pump: every word certified, so K's 0.9 holds unhalved at duo's ~3/4 of ram's rate"))
    filler: Filler | None = Filler(W=8, NC=2, S=256, chunk_log2=19, noise="bitsU")
    nvcc_flags: str = "-DPFILL_BU -Xcompiler -mpopcnt"


#: FUSED_K4 rows as (split SMs, FILL's step or None, phase filler or None), with pfill-k-phase's phase filler at the two
#: B 256 points where it beat fused4's pump: on cuBLASLt's 60-SM
#: split, 256/2K with 5 tiles a block read 1.41e14 MAC/s at 1.038 and 256/3K with 8 read 1.48e14 at 1.023, the split
#: alone 1.00e14.
FUSED_PHASE_K4 = {**{k: (*v, None) for k, v in FUSED_K4.items()},
                  (256, 2048): (60, None, PhaseFill(tiles=5)), (256, 3072): (60, None, PhaseFill(tiles=8))}
PHASE_TAGS = ("sched:phase-att", "sched:graph-fork-join", "sched:node-priority", "phase:pfill-tiles",
              "phase:4-warp-blocks")
FP_SOURCES = PFillKFused4.__dataclass_fields__["sources"].default + (
    "benchmarks/pouw/filler/pscore.cu", "benchmarks/pouw/filler/psched_score.py", "benchmarks/pouw/filler/pfill_fp_score.py")


@dataclass
class PFillBUDuoFPhase(PFillBUDuoFused4):
    """pfill-bu-duo-fused4 with pfill-k-phase's 4-warp phase filler (-DPF_PW=4) on duo tiles (NC 2) in decode attention
    at 256/2K and 256/3K, the points where it beat fused4's pump. Run through pfill_fp_score.py:

      python3 benchmarks/pouw/filler/pfill_fp_score.py benchmarks/pouw/filler/pfill_arm.py:PFillBUDuoFPhase3
    """

    name: str = "pfill-bu-duo-fphase"
    lineage: dict = field(default_factory=lambda: _fused_lineage(
        "noise:sha512-coins-bU", "receipt:duo-k32",
        "pfill-bu-xin-fphase's B 256 phase filler on duo, the fold proved now",
        parent="pfill-bu-duo-fused4", extra=PHASE_TAGS))
    sources: tuple = FP_SOURCES
    choice: str = ("an offline table of 18 (B, L) points (split SMs, FILL's step or cuBLASLt's, phase tiles) from FUSED_K4 "
                   "and pfill-k-phase's B 256 rows; a lookup, a green split and one graph a step at serving time")
    nvcc_flags: str = "-DPFILL_BU -DPF_PW=4 -Xcompiler -mpopcnt"
    table = {k: (s, q, p and PhaseFill(NC=2, tiles=p.tiles)) for k, (s, q, p) in FUSED_PHASE_K4.items()}

    def phase(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[2]


@dataclass
class PFillBUDuoFPhase2(PFillBUDuoFPhase):
    """pfill-bu-duo-fphase with 6 duo tiles at 256/3K: duo's phase tile runs at 0.85 of XOR-in's alone, so its 8 read
    1.052 there, past the cliff XOR-in's 9 reach (1.057, s11); 6 take about XOR-in's 7."""

    name: str = "pfill-bu-duo-fphase2"
    lineage: dict = field(default_factory=lambda: _fused_lineage(
        "noise:sha512-coins-bU", "receipt:duo-k32",
        "pfill-bu-duo-fphase with duo's 256/3K phase sized to its rate: 6 tiles take XOR-in's ~7, inside the cliff",
        parent="pfill-bu-duo-fphase", extra=PHASE_TAGS))
    table = {**PFillBUDuoFPhase.table, (256, 3072): (60, None, PhaseFill(NC=2, tiles=6))}


@dataclass
class PFillBUDuoFPhase3(PFillBUDuoFPhase):
    """pfill-bu-duo-fphase2 with 7 duo tiles at 256/3K: 6 read 1.021, and 7 take about XOR-in's
    8.2 tile-times, between XOR-in's 8 (1.020) and its cliff at 9 (1.057, s11)."""

    name: str = "pfill-bu-duo-fphase3"
    lineage: dict = field(default_factory=lambda: _fused_lineage(
        "noise:sha512-coins-bU", "receipt:duo-k32",
        "pfill-bu-duo-fphase2 with a seventh duo tile at 256/3K: about XOR-in's 8.2 tile-times, short of its cliff at 9",
        parent="pfill-bu-duo-fphase2", extra=PHASE_TAGS))
    table = {**PFillBUDuoFPhase.table, (256, 3072): (60, None, PhaseFill(NC=2, tiles=7))}


@dataclass
class PFillBU(PFillPowerSplit):
    """pfill-power-split's ram tile as pfill-bU (-DPFILL_BU): A's entries their SHA-512 bits as s8 0 or 1, B's U = (t &
    0x7F) - 64 + ((t >> 6) & 1) from one digest byte t an entry (uniform on {±1, ..., ±64}), 512 blocks a tile, on the
    same s8 IMMA m16n8k32 with C = 0, each s32 word in [-2048, 2048] folded unscaled into the same ram receipt at K = L
    = 32. Theorem K's setting as refereed: straight-line, cell-pure, int MMAs, unscaled s32 reads, operands affine in
    their bits (U's weights 1, 2, ..., 32, 65 on the odd parity bits t_0, t_6), every prefix word committed. dgate adds
    the uniformity check of U and A; -mpopcnt is for the host reference's seven popcounts a word."""

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
    """pfill-bU with duo receipts (NC 2, three ops a word): a fold proved to have no all-state kernel at one, two or
    three words (four at key_ok keys, DuoQuadNoKernel; longer runs open), so the receipt reads every word, where ram's
    reads a pair only through Q = a0 d0 + d1."""

    name: str = "pfill-bu-duo"
    lineage: dict = field(default_factory=lambda: _fold_lineage(
        "pfill-bu", "noise:sha512-coins-bU", "receipt:duo-k32",
        "pfill-bU with duo for ram: every word certified, so K's c = 1 holds where ram's Q-only receipt halves it"))
    filler: Filler | None = Filler(W=8, NC=2, S=256, chunk_log2=19, noise="bitsU")


@dataclass
class PFillBULi(PFillBU):
    """pfill-bU with the r9 LOP3+IMAD receipts (NC 4 built with -DPFILL_LI, two ops a pair as ram's:
    y <- (y ^ d0 ^ k0) a0 + d1;  y <- (y ^ d2 ^ k1) a1 + d3, keys under key_ok_r9).
    It reads every word given (HO), no hand-off sharing a row or column,
    which bU's hashed rows give (pfill_ref.ho_ok). Provisional while r9 is refereed (`fold_status`)."""

    name: str = "pfill-bu-li"
    lineage: dict = field(default_factory=lambda: _fold_lineage(
        "pfill-bu", "noise:sha512-coins-bU", "receipt:li-k32",
        "pfill-bU with LOP3+IMAD for ram: every word certified at ram's op count, so K's c = 1 holds (r9, provisional)"))
    nvcc_flags: str = "-DPFILL_BU -DPFILL_LI -Xcompiler -mpopcnt"


@dataclass
class PFillBUXin13(PFillBU):
    """pfill-bU with the r9 XOR-in (13, 19) receipts (NC 4 built with -DPFILL_XIN13, three ops a pair:
    y <- (rotl(y, 13) ^ d0) a0 + d1;  y <- (rotl(y, 19) ^ d2) a1 + d3, keys under key_ok_r9): every word read with no
    layout premise, the served GEMM's fold. Provisional while r9 is refereed (`fold_status`)."""

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
    d1), 1 for duo, LOP3+IMAD and XOR-in (13, 19)."""
    mech = lineage.get("mechanism") or []
    return 0.5 if any(t.startswith("receipt:ram") for t in mech) else 1.0


R9_RECEIPTS = ("receipt:li-k32", "receipt:xin13-k32")


def fold_status(lineage: dict) -> str | None:
    """'provisional' for a receipt fold whose argument is under referee: the r9 LOP3+IMAD and XOR-in (13, 19), until
    its referee rules."""
    return "provisional" if set(lineage.get("mechanism") or []) & set(R9_RECEIPTS) else None


def products_per_slot(lineage: dict) -> int:
    """Products the arm's filler does an int8 slot it reports: 2 for the e2m1 tile, else 1."""
    return 2 if "mma:e2m1-omma-k64" in (lineage.get("mechanism") or []) else 1


def theorem_k_setting(lineage: dict) -> bool:
    """Whether the arm's filler is one Theorem K reads: {0,1} bits on both sides (pfill-k) or pfill-bU."""
    mech = lineage.get("mechanism") or []
    return "noise:sha512-coins-01" in mech or "noise:sha512-coins-bU" in mech
