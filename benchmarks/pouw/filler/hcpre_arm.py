"""HC-PREFILL's arm (cmp-13, approach pouw/filler-prefill; note:20261016T0710Z-draft-hc-prefill): HC-KERN's KernPhase
(hc_kern_phase.py: map 1 in every filler, TABLE_K) with the prefill co shape re-sized for map 1.

The scorer's prefill point is the 32 layers of int8 linears at M = 8192: one CUTLASS kernel (128 threads, one warp a
sub-partition, 98 KB smem, so one CTA an SM) at 0.883 of the int8 peak, its idle tensor time per-tile prologue and
epilogue (~6.5%), mainloop bubbles (~4%) and wave tails (~1.4%), no launch gaps (r20261016-071430-ef97). A co block of
4 warps puts one filler warp beside each GEMM warp; a fifth warp on a sub-partition stalls its GEMM warp and the whole
CTA (W 5-6: 1.42-1.45x). With map 1's MOV-free chain, longer tiles (S 512) and chunks (2^18 MMAs an SM) pay the tile's
seed, hash and receipt and the chunk's launch less often: W 4 NC 4 S 512 2^18 read 1.032 / 0.240 and 1.034 / 0.227 on
two screens (r20261016-072222-53dd, r20261016-072626-9f32) against PhaseSchedCo2's W 4 S 128 2^16 at 1.030 / 0.181 on
the same GPU; S 256-2048 and map 2 lie on the same plateau (0.223-0.231 at 1.031-1.036), 2^19 and 2^20 break the gate.

  python3 benchmarks/pouw/filler/psched_score.py benchmarks/pouw/filler/hcpre_arm.py:PrefillCo
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A
import hc_kern_phase as KP

CO_PREFILL_M1 = A.Filler(W=4, NC=4, S=512, chunk_log2=18)


@dataclass
class PrefillCo(KP.KernPhase):
    """KernPhase with 4-warp co blocks of S 512 tiles in 2^18-MMA chunks in prefill."""

    name: str = "hcpre-m1-co-s512"
    lineage: dict = field(default_factory=lambda: {
        "parent": "kern-m1-phase-co2", "approach": "pouw/filler-prefill",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:w16-s8192", "chunk:2^20",
                      "place:split", "split:per-point-table", "sched:phase-att", "sched:graph-fork-join",
                      "sched:node-priority", "prefill:co", "sched:prio-short-blocks", "co:small-blocks",
                      "phase:s256-512", "phase:nc2", "co:s512-2^18"],
        "hypothesis": "beside the prefill GEMMs a co warp a sub-partition is the most that fits, and map 1's lighter "
                      "chain in S 512 tiles of 2^18-MMA chunks fills ~0.1 of peak there at ~1.03: prefill 0.19 -> ~0.23"})
    sources: tuple = KP.KernPhase.sources + ("benchmarks/pouw/filler/hcpre_arm.py",)
    choice: str = ("KernPhase's TABLE_K and split filler, and one prefill co shape (W 4 NC 4 S 512, 2^18) from two "
                   "screens of 30 shapes; a lookup, a green split and one graph a step at serving time, a low-priority "
                   "stream in prefill; DFILL_MAP=1 for every process; run through psched_score.py/pscore.cu")

    def filler_at(self, phase, b, l):
        return CO_PREFILL_M1 if phase == "prefill" else self.filler
