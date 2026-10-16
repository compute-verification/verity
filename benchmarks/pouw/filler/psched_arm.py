"""HC-SCHED's arm (cmp-13, approach pouw/filler-sched): FILL v1's split, re-chosen per point, plus the phase filler, dcap.cu's
dfill_p forked into each decode layer's attention on the step's own SMs (attention is dp4a on the CUDA cores and leaves
the tensor pipe idle). filler-arm/v1 can't place a phase filler, so the arm's `placement` is its split, the contract's,
and `phase(phase, B, L)` names the phase filler; psched_score.py runs the pair through pscore.cu and scores the out dir
with the locked `score.py report`.

  python3 benchmarks/pouw/filler/psched_score.py benchmarks/pouw/filler/psched_arm.py:PhaseSched
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A  # noqa: E402


@dataclass(frozen=True)
class PhaseFill:
    """The phase filler at a point: W warps a block (one block a step SM), NC chains a warp, S steps a tile, and `tiles`
    tiles a warp a layer, a fixed count chosen before the run."""

    W: int = 8
    NC: int = 4
    S: int = 128
    tiles: int = 1


#: (B, L) -> (split SMs, phase filler or None), chosen on psched runs r20261016-044809-fb5f and r20261016-045835-d45c
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
    sources: tuple = ("benchmarks/pouw/decode_capacity/dcap.cu", "benchmarks/pouw/filler/pscore.cu",
                      "benchmarks/pouw/filler/psched_arm.py", "benchmarks/pouw/filler/psched_score.py")
    filler: A.Filler | None = A.Filler(W=8, NC=4, S=2048, chunk_log2=19)
    choice: str = ("an offline table of 18 (B, L) points (split, phase W and tiles) from two psched runs; a lookup, a green "
                   "split and one graph a step at serving time; the phase filler runs through psched_score.py/pscore.cu")

    def _row(self, b, l):
        return A.point_lookup(TABLE, b, l)

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.NONE
        return A.Placement("split", self._row(b, l)[0])

    def phase(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[1]
