"""HC-KERN's phase arm (approach pouw/filler-kernel; note:20261016T0437Z-draft-hc-kern): HC-SCHED's PhaseSchedCo2
(psched_arm.py) with the filler kernel's map 1 in all three of its fillers (dcap.cu's chain with the accumulator quad read
in register order: one in-place IMMA a step, no MOVs), the split filler at W 16, S 8192 in 2^20-MMA chunks (0.995 of
peak beside serving), and the phase filler re-sized for it (TABLE_K). The phase filler shares the step's SMs with
attention, so there its footprint is the whole game: map 0's four MOVs an IMMA and map 2's second hash a word both
contend with attention for issue, and longer tiles (S 256, 512) pay the tile's seed, hash and receipt less often.

DFILL_MAP is process-wide, so the split, phase and co fillers all run map 1; FILL-SEC requires nothing of the map
(note:20261016T0418Z-draft-filler-floor), and map 2's hardening costs the phase filler 0.01-0.04 of slowdown a point.

  python3 benchmarks/pouw/filler/psched_score.py benchmarks/pouw/filler/hc_kern_phase.py:KernPhase
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A
import psched_arm as PS

#: TABLE_V3's splits, with the phase filler re-sized for map 1 beside a W 16 S 8192 split filler on screens
#: r20261016-060932-6357 and r20261016-061429-cef2 (0 bad receipts): each entry is the best row at ≤ 1.043 there, and
#: the next tile count screened falls off the cliff where the phase work outlasts attention (32/24K: t 56 1.023, t 66
#: 1.103), so these sit below it. NC 2 (56 registers against 88) at W 8 where it beat W 4 NC 4 at the same MACs.
TABLE_K = {**PS.TABLE_V3,
           (32, 1024): (92, PS.PhaseFill(S=256, tiles=2)),
           (32, 2048): (92, PS.PhaseFill(NC=2, S=512, tiles=5)),
           (32, 8192): (108, PS.PhaseFill(W=4, S=256, tiles=31)),
           (32, 24576): (108, PS.PhaseFill(S=256, tiles=56)),
           (48, 16384): (92, PS.PhaseFill(S=256, tiles=58)),
           (128, 2048): (92, PS.PhaseFill(S=512, tiles=10)),
           (128, 6144): (92, PS.PhaseFill(S=256, tiles=59)),
           (132, 5120): (60, PS.PhaseFill(S=256, tiles=50)),
           (144, 5120): (60, PS.PhaseFill(S=256, tiles=53)),
           (256, 1024): (60, PS.PhaseFill(NC=2, S=512, tiles=18)),
           (256, 2048): (60, PS.PhaseFill(S=256, tiles=38)),
           (256, 3072): (60, PS.PhaseFill(S=256, tiles=58))}


@dataclass
class KernPhase(PS.PhaseSchedCo2):
    """PhaseSchedCo2 on map 1, a W 16 S 8192 2^20 split filler and TABLE_K."""

    name: str = "kern-m1-phase-co2"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-v1-phase-att-co2", "approach": "pouw/filler-kernel",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:w16-s8192", "chunk:2^20",
                      "place:split", "split:per-point-table", "sched:phase-att", "sched:graph-fork-join",
                      "sched:node-priority", "prefill:co", "sched:prio-short-blocks", "co:small-blocks",
                      "phase:s256-512", "phase:nc2"],
        "hypothesis": "beside attention the phase filler's own issue load sets its cliff: map 1's MOV-free chain in "
                      "S 256-512 tiles fits 1.2-1.4x the MACs below it, +0.08-0.13 a phase point"})
    sources: tuple = PS.PhaseSchedCo2.sources + ("benchmarks/pouw/filler/hc_kern_phase.py",)
    filler: A.Filler | None = A.Filler(W=16, NC=4, S=8192, chunk_log2=20)
    choice: str = ("TABLE_K (TABLE_V3's splits and phase points, re-sized for map 1 on two screens) and PhaseSchedCo2's "
                   "prefill co shape; DFILL_MAP=1 for every process; run through psched_score.py/pscore.cu")
    map: int = 1

    def __post_init__(self):
        os.environ["DFILL_MAP"] = str(self.map)

    def _row(self, b, l):
        return A.point_lookup(TABLE_K, b, l)
