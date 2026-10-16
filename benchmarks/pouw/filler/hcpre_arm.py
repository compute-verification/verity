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

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A
import hc_kern_phase as KP

CO_PREFILL_M1 = A.Filler(W=4, NC=4, S=512, chunk_log2=18)
CO_PREFILL_ONE = A.Filler(W=4, NC=4, S=512, chunk_log2=20)


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


@dataclass
class PrefillCoOne(PrefillCo):
    """PrefillCo with DFILL_ONE: one dfill_k block running an SM at a time (dcap.cu's dtiles_one) in every process.

    The co pump launches a chunk as one block an SM, but three 88-register filler blocks fit beside a prefill GEMM CTA,
    and the dispatcher puts 2-4 of a chunk's blocks on a few SMs (at 2^18 in ~60% of chunks, at 2^20 on 9-11 SMs a chunk);
    there a second filler warp on a sub-partition slows the GEMM's one warp for the whole chunk, and every GEMM's tail
    waits on that SM (r20261016-081635-dcc5, r20261016-082749-0590). With DFILL_ONE a doubled block sleeps in place until
    its SM's filler block is done and warps take `one` tiles at a time from the chunk's counter. The split filler (one
    512-thread block an SM) runs the same tiles through the same path. On dscore's prefill point (r20261016-083326-c645)
    that leaves 2^18 as it was (1.032 / 0.227) and makes longer chunks pass: 2^19 1.036 / 0.229 (static 1.061),
    2^20 1.046 / 0.260; decode 1/2K's split reads 1.034 / 0.565 as before. So prefill runs 2^20-MMA chunks."""

    name: str = "hcpre-m1-co-one"
    lineage: dict = field(default_factory=lambda: {
        "parent": "hcpre-m1-co-s512", "approach": "pouw/filler-prefill",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:w16-s8192", "chunk:2^20",
                      "place:split", "split:per-point-table", "sched:phase-att", "sched:graph-fork-join",
                      "sched:node-priority", "prefill:co", "sched:prio-short-blocks", "co:small-blocks",
                      "phase:s256-512", "phase:nc2", "co:s512-2^20", "kernel:one-block-an-sm"],
        "hypothesis": "the co filler's erratic slowdown beside the prefill GEMMs is SMs where the dispatcher put two or "
                      "more filler blocks; with one running block an SM, 2^20-MMA chunks pass the gate and fill "
                      "~0.13 of peak there: prefill 0.227 -> ~0.26"})
    choice: str = ("PrefillCo with DFILL_ONE=4 (one filler block running an SM, 4-tile units) for every process and "
                   "2^20-MMA prefill chunks, from occupancy probes and a dscore screen of 2^18-2^20; DFILL_MAP=1; run "
                   "through psched_score.py/pscore.cu")
    one: int = 4

    def __post_init__(self):
        super().__post_init__()
        os.environ["DFILL_ONE"] = str(self.one)

    def filler_at(self, phase, b, l):
        return CO_PREFILL_ONE if phase == "prefill" else self.filler
