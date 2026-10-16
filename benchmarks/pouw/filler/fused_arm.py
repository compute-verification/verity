"""FILL's fused arm (cmp-13, approach pouw/filler-fused): the decode step's own kernels carry the filler. Its linears run
through flin (dcap.cu's flin.cuh: a warp-specialized int8 GEMM whose filler warps run FILL's DRAM-free chain while its
consumers wait on memory) and its attention blocks carry filler warps too, on a green split's step SMs, beside FILL v1's
filler pumped on the rest. filler-arm/v1 can't place a filler inside serving's kernels, so the arm's `placement` is its
split, the contract's, and `fused(phase, B, L)` names the step's filler warps; fused_score.py runs the pair through
fscore.cu and scores the out dir with the locked `score.py report`.

  python3 benchmarks/pouw/filler/fused_score.py benchmarks/pouw/filler/fused_arm.py:Fused
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A


@dataclass(frozen=True)
class FusedStep:
    """The fused step at a point: FW filler warps a flin block with FNC chains (0: flin's GEMM alone), AFW a block of the
    attention kernels with ANC chains, those blocks persistent on the step's SMs (APS) with the lazy done test ALZ
    (0, 2 or 4: every 8, 16 or 32 steps a check), LT the linears kept on cuBLASLt (1 qkv, 2 o, 4 gate_up, 8 down), and
    FLZ flin's filler warps' lazy done test."""

    FW: int = 8
    FNC: int = 4
    AFW: int = 8
    ANC: int = 2
    APS: int = 1
    ALZ: int = 4
    LT: int = 0
    FLZ: int = 0


#: (B, L) -> (split SMs, fused step): FILL's round-2 picks (note:20261016T0420Z-draft-filler-in-decode §0), each at
#: slowdown <= 1.04 over runs 8-16 and 20-22 and holdouts 1-3 against the whole-GPU cuBLASLt step, then holdout 4
#: r20261016-072512-ea9f (every pick <= 1.040, fresh coins); suite points between them take `point_lookup`'s row
TABLE = {
    (1, 2048): (108, FusedStep(AFW=4, ANC=4, APS=0, ALZ=0)),
    (8, 2048): (108, FusedStep()), (8, 32768): (108, FusedStep()),
    (32, 1024): (116, FusedStep(AFW=6, ALZ=2)), (32, 2048): (108, FusedStep()), (32, 5120): (108, FusedStep()),
    (32, 8192): (108, FusedStep()), (32, 16384): (100, FusedStep()), (32, 24576): (92, FusedStep()),
    (48, 2048): (92, FusedStep(FW=4)), (48, 16384): (68, FusedStep()),
    (64, 2048): (92, FusedStep(FW=4)), (64, 12288): (68, FusedStep()),
    (96, 8192): (68, FusedStep(FW=4)),
    (128, 2048): (92, FusedStep(FW=4, AFW=6, ALZ=0)), (128, 4096): (76, FusedStep(FW=4)),
    (132, 5120): (68, FusedStep(FW=0)), (144, 5120): (68, FusedStep(FW=0)),
    (192, 2048): (60, FusedStep(FW=0)),
    (256, 1024): (36, FusedStep(FW=0, LT=15)), (256, 2048): (36, FusedStep(FW=0, LT=15)),
    (256, 3072): (36, FusedStep(FW=0, LT=15)),
}
#: prefill is HC-SCHED's (fill-v1-phase-att-co2, psched_arm.py's CO_PREFILL_V3): FILL v1's filler co-scheduled at the
#: lowest stream priority in 4-warp blocks of 2^16 MMAs, 1.042 / 0.195 on its three screens
CO_PREFILL = A.Filler(W=4, NC=4, S=128, chunk_log2=16)


@dataclass
class Fused:
    """The fused step (flin + attention filler warps) on a per-point split beside FILL v1's pump; HC-SCHED's prefill."""

    name: str = "fill-fused"
    lineage: dict = field(default_factory=lambda: {
        "parent": "cursor/fused-filler-ea24@da86371a1", "approach": "pouw/filler-fused",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "fused:flin-filler-warps",
                      "fused:attention-filler-warps", "fused:persistent-attention", "fused:lazy-done",
                      "prefill:co", "sched:prio-short-blocks"],
        "hypothesis": "decode's kernels leave the tensor pipe idle while they wait on memory: filler warps in the "
                      "linears and attention blocks, beside the pump on the rest, cover 0.70-0.82 of int8 peak on "
                      "decode at <= 5% slowdown"})
    sources: tuple = ("benchmarks/pouw/decode_capacity/dcap.cu", "benchmarks/pouw/decode_capacity/flin.cuh",
                      "benchmarks/pouw/filler/fscore.cu", "benchmarks/pouw/filler/fused_arm.py",
                      "benchmarks/pouw/filler/fused_score.py")
    filler: A.Filler | None = A.Filler(W=8, NC=4, S=2048, chunk_log2=19)
    choice: str = ("an offline table of 22 (B, L) points (split SMs, flin and attention filler warps) from FILL's round-2 "
                   "picks and holdouts, HC-SCHED's prefill co shape; at serving time a lookup, a green split and one "
                   "graph a step")

    def _row(self, b, l):
        return A.point_lookup(TABLE, b, l)

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.Placement("co")
        return A.Placement("split", self._row(b, l)[0])

    def fused(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[1]

    def filler_at(self, phase, b, l):
        return CO_PREFILL if phase == "prefill" else self.filler
