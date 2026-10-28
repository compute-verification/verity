"""The scheduled arm: FILL v1's split, re-chosen per point, plus the phase filler, dcap.cu's
dfill_p forked into each decode layer's attention on the step's own SMs (attention is dp4a on the CUDA cores and leaves
the tensor pipe idle). filler-arm/v1 can't place a phase filler, so the arm's `placement` is its split, the contract's,
and `phase(phase, B, L)` names the phase filler; psched_score.py runs the pair through pscore.cu and scores the out dir
with the locked `score.py report`.

  python3 benchmarks/pouw/filler/psched_score.py benchmarks/pouw/filler/psched_arm.py:PhaseSched
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A


@dataclass(frozen=True)
class PhaseFill:
    """The phase filler at a point: W warps a block (one block a step SM), NC chains a warp, S steps a tile, and `tiles`
    tiles a warp a layer, a fixed count chosen before the run."""

    W: int = 8
    NC: int = 4
    S: int = 128
    tiles: int = 1


#: (B, L) -> (split SMs, phase filler or None), chosen on two psched runs
#: (dcap pfill: each arm's slowdown against the whole-GPU step, the phase tiles 0.9 of the layer's attention time on the
#: step's SMs), preferring ≤ 1.03 there; B 1 keeps FILL v1's split alone (the phase filler cost 2% there for 0.01–0.05).
TABLE = {
    (1, 2048): (108, None), (1, 8192): (108, None), (1, 32768): (108, None),
    (8, 16384): (108, PhaseFill(W=4, tiles=25)),
    (32, 1024): (92, PhaseFill(W=4, tiles=7)), (32, 2048): (92, PhaseFill(W=4, tiles=13)),
    (32, 8192): (108, PhaseFill(W=4, tiles=41)), (32, 24576): (108, PhaseFill(W=4, tiles=156)),
    (48, 16384): (92, PhaseFill(tiles=93)),
    (64, 4096): (76, PhaseFill(tiles=29)),
    (128, 2048): (76, PhaseFill(tiles=29)), (128, 6144): (92, PhaseFill(tiles=94)),
    (132, 5120): (60, PhaseFill(tiles=79)), (144, 5120): (60, PhaseFill(tiles=86)),
    (192, 2048): (44, PhaseFill(tiles=43)),
    (256, 1024): (60, PhaseFill(W=4, tiles=44)), (256, 2048): (60, PhaseFill(tiles=64)),
    (256, 3072): (60, PhaseFill(tiles=93)),
}


@dataclass
class PhaseSched:
    """FILL v1's filler on a per-point split, and the phase filler in decode's attention on the step's SMs."""

    name: str = "fill-v1-phase-att"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-v1", "approach": "pouw/filler-sched",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "sched:phase-att",
                      "sched:graph-fork-join", "sched:node-priority", "prefill:none"],
        "hypothesis": "decode's attention leaves the tensor pipe idle on the step's SMs: a fixed count of short filler "
                      "tiles forked into it adds 0.05-0.34 coverage a point at <= 5% slowdown"})
    sources: tuple = ("benchmarks/pouw/filler/dcap.cu", "benchmarks/pouw/filler/pscore.cu",
                      "benchmarks/pouw/filler/psched_arm.py", "benchmarks/pouw/filler/psched_score.py")
    filler: A.Filler | None = A.Filler(W=8, NC=4, S=2048, chunk_log2=19)
    choice: str = ("an offline table of 18 (B, L) points (split, phase W and tiles) from two psched runs; a lookup, a green "
                   "split and one graph a step at serving time; the phase filler runs through psched_score.py/pscore.cu")
    #: dcap.cu builds pfill.cuh's filler unless -DDFILL_V1; these arms were chosen and scored on FILL v1's
    nvcc_flags: str = "-DDFILL_V1"

    def __post_init__(self):
        os.environ["NVCC_APPEND_FLAGS"] = self.nvcc_flags

    def _row(self, b, l):
        return A.point_lookup(TABLE, b, l)

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.NONE
        return A.Colocation("split", self._row(b, l)[0])

    def phase(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[1]


#: TABLE with the larger phase fillers that screened ≤ 1.04 through pscore itself; 48/16K keeps
#: TABLE's (108 W8 t90 screened 1.037 for +0.017 there, +0.0003 SCORE)
TABLE_V2 = {**TABLE, (32, 8192): (108, PhaseFill(W=4, tiles=47)), (32, 24576): (108, PhaseFill(tiles=95)),
            (128, 2048): (92, PhaseFill(tiles=28)), (256, 3072): (60, PhaseFill(tiles=98))}
#: FILL's filler in prefill at the lowest stream priority, in blocks short enough (8 tiles a warp of S 128, ~90 µs) that
#: the GEMMs' blocks, at the highest, take each SM back within one: it fills their wave tails (1.027, coverage 0.180
#: against 0.135 alone; 2^16 a chunk 1.083)
CO_PREFILL = A.Filler(W=8, NC=4, S=128, chunk_log2=15)


@dataclass
class PhaseSchedCo(PhaseSched):
    """PhaseSched on TABLE_V2, and in prefill FILL's filler co-scheduled at the lowest priority in short blocks."""

    name: str = "fill-v1-phase-att-co"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-v1-phase-att", "approach": "pouw/filler-sched",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "sched:phase-att",
                      "sched:graph-fork-join", "sched:node-priority", "prefill:co", "sched:prio-short-blocks"],
        "hypothesis": "prefill's GEMMs leave wave tails idle: FILL's filler at the lowest stream priority in ~90 us "
                      "blocks fills them at <= 3% slowdown, adding ~0.045 to prefill and to the mixes' prefill share"})
    choice: str = ("TABLE_V2 (18 decode points: split, phase W and tiles) from three screens, and one co shape for "
                   "prefill; a lookup, a green split and one graph a step at serving time, a low-priority stream in "
                   "prefill; run through psched_score.py/pscore.cu")

    def _row(self, b, l):
        return A.point_lookup(TABLE_V2, b, l)

    def placement(self, phase, b, l):
        return A.Colocation("co") if phase == "prefill" else super().placement(phase, b, l)

    def filler_at(self, phase, b, l):
        return CO_PREFILL if phase == "prefill" else self.filler


#: TABLE_V2 with the phase fillers that screened below their cliff on two screens
TABLE_V3 = {**TABLE_V2, (32, 1024): (92, PhaseFill(tiles=4)), (32, 8192): (108, PhaseFill(W=4, tiles=52)),
            (48, 16384): (92, PhaseFill(tiles=99)), (128, 6144): (92, PhaseFill(tiles=100)),
            (132, 5120): (60, PhaseFill(tiles=84)), (144, 5120): (60, PhaseFill(tiles=90))}
#: the co shape that spends prefill's slowdown budget: 4-warp blocks of 2^16 MMAs, 1.042 / 0.195 on three screens;
#: every co shape screened lies near
#: coverage 0.135 + 1.5 × (slowdown − 1), and W 5–8 at 2^17 fall off a cliff (1.08–1.25)
CO_PREFILL_V3 = A.Filler(W=4, NC=4, S=128, chunk_log2=16)


@dataclass
class PhaseSchedCo2(PhaseSchedCo):
    """PhaseSchedCo on TABLE_V3, with 4-warp co blocks of 2^16 MMAs in prefill."""

    name: str = "fill-v1-phase-att-co2"
    lineage: dict = field(default_factory=lambda: {
        "parent": "art:a1f1ae67595dd8e4fc28eb4d4a1758ae2c5bc7c2cf858e26856663015769a21e", "approach": "pouw/filler-sched",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "sched:phase-att",
                      "sched:graph-fork-join", "sched:node-priority", "prefill:co", "sched:prio-short-blocks",
                      "co:small-blocks"],
        "hypothesis": "4-warp co blocks fit beside the prefill GEMMs' blocks, so twice the chunk costs 1.042 for 0.195 "
                      "(8 warps at 2^15: 1.027 for 0.180); with six phase fillers sized up to their cliff, ~+0.007"})
    choice: str = ("TABLE_V3 (18 decode points: split, phase W and tiles) from five screens, and one co shape for "
                   "prefill (W 4, 2^16); a lookup, a green split and one graph a step at serving time, a low-priority "
                   "stream in prefill; run through psched_score.py/pscore.cu")

    def _row(self, b, l):
        return A.point_lookup(TABLE_V3, b, l)

    def filler_at(self, phase, b, l):
        return CO_PREFILL_V3 if phase == "prefill" else self.filler
