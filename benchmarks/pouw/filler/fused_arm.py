"""FILL's fused arm (cmp-13, approach pouw/filler-fused): the decode step's own kernels carry the filler. Its linears run
through flin (dcap.cu's flin.cuh: a warp-specialized int8 GEMM whose filler warps run FILL's DRAM-free chain while its
consumers wait on memory) and its attention blocks carry filler warps too, on a green split's step SMs, beside FILL v1's
filler pumped on the rest. filler-arm/v1 can't place a filler inside serving's kernels, so the arm's `placement` is its
split, the contract's, and `fused(phase, B, L)` names the step's filler warps; fused_score.py runs the pair through
fscore.cu and scores the out dir with the locked `score.py report`. dcap.cu's filler is pfill unless built with
-DDFILL_V1, so these arms pass it through NVCC_APPEND_FLAGS (pfill_arm.py's way), which score.py's and fscore's nvcc read.

  python3 benchmarks/pouw/filler/fused_score.py benchmarks/pouw/filler/fused_arm.py:Fused
"""

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A


@dataclass(frozen=True)
class FusedStep:
    """The fused step at a point: FW filler warps a flin block with FNC chains (0: flin's GEMM alone), AFW a block of the
    attention kernels with ANC chains, those blocks persistent on the step's SMs (APS) with the lazy done test ALZ
    (0, 2 or 4: every 8, 16 or 32 steps a check), LT the linears kept on cuBLASLt (1 qkv, 2 o, 4 gate_up, 8 down),
    and LM 1 lm_head through flin too (with its filler warps)."""

    FW: int = 8
    FNC: int = 4
    AFW: int = 8
    ANC: int = 2
    APS: int = 1
    ALZ: int = 4
    LT: int = 0
    LM: int = 0


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
CO_PREFILL = A.Filler(W=4, NC=4, S=128, chunk_log2=16, noise="chained")


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
    filler: A.Filler | None = A.Filler(W=8, NC=4, S=2048, chunk_log2=19, noise="chained")
    choice: str = ("an offline table of 22 (B, L) points (split SMs, flin and attention filler warps) from FILL's round-2 "
                   "picks and holdouts, HC-SCHED's prefill co shape; at serving time a lookup, a green split and one "
                   "graph a step")
    table: dict = field(default_factory=lambda: TABLE, repr=False)
    nvcc_flags: str = "-DDFILL_V1"

    def __post_init__(self):
        os.environ["NVCC_APPEND_FLAGS"] = self.nvcc_flags

    def _row(self, b, l):
        return A.point_lookup(self.table, b, l)

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.Placement("co")
        return A.Placement("split", self._row(b, l)[0])

    def fused(self, phase, b, l):
        return None if phase == "prefill" else self._row(b, l)[1]

    def filler_at(self, phase, b, l):
        return CO_PREFILL if phase == "prefill" else self.filler


#: round 3's picks over TABLE (screens r20261016-081436-04e3, -082545-6a39 and -083352-fc74): lm_head through flin at
#: B <= 64 (+0.003-0.022 of peak, the step no slower: flin's lm_head beats cuBLASLt's at M = 32) but B 32 L 1K (+0.005
#: slowdown at 1.024-1.030 without it, r20261016-085926-5760, where prefill's 1.043 already sits), not at M = 128 (<= +0.004 for +0.002-0.006 slowdown);
#: B 1 L 32K its own row, persistent attention and the pump on 100 SMs (the 2K row read 1.045 under the scorer there, this
#: one 0.999). Not taken: FLIN_LAZY 2/4 (+2-10% slowdown) and flin's filler warps at B >= 132 (1:4, 2:2, 2:4: +0.008-0.015
#: of peak for +2.3-4.5%, the pump's trade is ~4x better).
TABLE_V2 = {k: (sms, replace(q, LM=1) if k[0] <= 64 and k != (32, 1024) else q) for k, (sms, q) in TABLE.items()}
TABLE_V2[(1, 32768)] = (100, FusedStep(LM=1))


@dataclass
class FusedV2(Fused):
    """Fused with round 3's table."""

    name: str = "fill-fused-v2"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-fused", "approach": "pouw/filler-fused",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "fused:flin-filler-warps",
                      "fused:attention-filler-warps", "fused:persistent-attention", "fused:lazy-done",
                      "fused:lm-head-flin", "prefill:co", "sched:prio-short-blocks"],
        "hypothesis": "lm_head through flin at B <= 64 and persistent attention at B 1 L 32K add 0.01-0.02 of peak to "
                      "fill-fused's decode points (0.624, r20261016-081018-591c) at no more slowdown"})
    choice: str = ("fill-fused's table with lm_head through flin at B <= 64 (but B 32 L 1K) and a B 1 L 32K row, from "
                   "FILL's round-3 screens")
    table: dict = field(default_factory=lambda: TABLE_V2, repr=False)


def _pump(sms, lm=1, aps=1, lt=0):
    return sms, FusedStep(FW=0, AFW=0, APS=aps, ALZ=0, LT=lt, LM=lm)


#: FILL's step with no filler warps (flin's GEMM, persistent attention, lm_head through flin at B <= 64) beside the pump
#: alone: screens r20261016-090707-4181, -091213-6491 and -091824-6c9f, every row <= 1.025 (B <= 32 holds 0.92-0.96
#: on 48 step SMs and breaks to 1.08-1.13 on 40)
TABLE_PUMP = {
    (1, 2048): _pump(140, aps=0), (1, 32768): _pump(140), (8, 2048): _pump(140), (8, 32768): _pump(140),
    (32, 1024): _pump(140), (32, 2048): _pump(140), (32, 5120): _pump(140), (32, 8192): _pump(140),
    (32, 16384): _pump(132), (32, 24576): _pump(132), (48, 2048): _pump(124), (48, 16384): _pump(116, lm=0),
    (64, 2048): _pump(124), (64, 12288): _pump(100), (96, 8192): _pump(84, lm=0),
    (128, 2048): _pump(100, lm=0), (128, 4096): _pump(92, lm=0), (132, 5120): _pump(76, lm=0),
    (144, 5120): _pump(76, lm=0), (192, 2048): _pump(60, lm=0),
    (256, 1024): _pump(44, lm=0, lt=15), (256, 2048): _pump(44, lm=0, lt=15), (256, 3072): _pump(44, lm=0, lt=15),
}


@dataclass
class FusedPump(Fused):
    """FILL's step without filler warps on a narrower split, the pump alone on the rest: no arm-side fixed work, so the
    filler is the pump's alone, the form filler-scorer/v3's theorem entries read (with a Theorem K pump)."""

    name: str = "fill-fused-pump"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-fused-v2", "approach": "pouw/filler-fused",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "fused:flin-serving",
                      "fused:persistent-attention", "fused:lm-head-flin", "prefill:co", "sched:prio-short-blocks"],
        "hypothesis": "FILL's step without filler warps (flin, persistent attention, lm_head through flin) holds <= 1.02 "
                      "on 48-144 SMs, so the pump alone takes 44-140: no fixed work, the shape a Theorem K pump is read "
                      "in under filler-scorer/v3"})
    choice: str = ("an offline table of FILL's pump-only splits (no filler warps) from round 3's screens; at serving time a "
                   "lookup, a green split and one graph a step")
    table: dict = field(default_factory=lambda: TABLE_PUMP, repr=False)


#: pfill-k's chunk (pfill_arm.PFillK's: 8 warps a block, 256 fragments a warp a tile, about 2^19 MMAs an SM a chunk)
K_FILLER = A.Filler(W=8, NC=4, S=256, chunk_log2=19, noise="bits01")


@dataclass
class FusedPumpK(FusedPump):
    """FusedPump's splits with HC-PROVED's pfill-k pumped on the filler SMs (-DPFILL_BITS: fresh {0,1} SHA-512 noise for
    every IMMA, each k32 word into the ram receipt), and pfill-k's prefill (the linears on 184 SMs, the filler on 4):
    a split's slowdown is the split's own, whatever runs on its filler SMs."""

    name: str = "fill-fused-pump-k"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-fused-pump", "approach": "pouw/filler-fused",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FusedPump's splits beside a pfill-k pump read Theorem K's 0.9 under filler-scorer/v3: the pump's "
                      "0.88 of FILL v1's rate an SM, on 44-140 SMs, projects 0.365"})
    filler: A.Filler | None = K_FILLER
    choice: str = ("FusedPump's offline table of pump-only splits, pfill-k's 4-SM prefill split; at serving time a lookup, "
                   "a green split and one graph a step")
    nvcc_flags: str = "-DPFILL_BITS"

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.Placement("split", 4)
        return super().placement(phase, b, l)

    def filler_at(self, phase, b, l):
        return self.filler
