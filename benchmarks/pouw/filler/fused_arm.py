"""FILL's fused arms: the decode step's own kernels on a green split's step SMs, beside pfill pumped on the rest.
Its linears run through flin (dcap.cu's flin.cuh: a warp-specialized int8 GEMM whose filler warps run pfill's step while
its consumers wait on memory). filler-arm/v1 can't place a filler inside serving's kernels, so the arm's `placement` is
its split, the contract's, and `fused(phase, B, L)` names the step's filler warps; fused_score.py runs the pair through
fscore.cu and scores the out dir with the locked `score.py report`. Each arm passes its pfill build's flags through
NVCC_APPEND_FLAGS (pfill_arm.py's way), which score.py's and fscore's nvcc read.

  python3 benchmarks/pouw/filler/fused_score.py benchmarks/pouw/filler/fused_arm.py:FusedPumpK
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A


@dataclass(frozen=True)
class FusedStep:
    """The fused step at a point: FW filler warps a flin block with FNC chains (0: flin's GEMM alone), AFW a block of the
    attention kernels with ANC chains, those blocks persistent on the step's SMs (APS) with the lazy done test ALZ
    (0, 2 or 4: every 8, 16 or 32 steps a check), LT the linears kept on cuBLASLt (1 qkv, 2 o, 4 gate_up, 8 down),
    LM 1 lm_head through flin too (with its filler warps), and ST > 0 stops a flin filler warp's steps once its block's
    first consumer warp has ST units left (FLIN_STOP, for a filler whose step outlasts a unit's)."""

    FW: int = 8
    FNC: int = 4
    AFW: int = 8
    ANC: int = 2
    APS: int = 1
    ALZ: int = 4
    LT: int = 0
    LM: int = 0
    ST: int = 0


def _pump(sms, lm=1, aps=1, lt=0):
    return sms, FusedStep(FW=0, AFW=0, APS=aps, ALZ=0, LT=lt, LM=lm)


#: FILL's step with no filler warps (flin's GEMM, persistent attention, lm_head through flin at B <= 64) beside the pump
#: alone: three screens, every row <= 1.025 (B <= 32 holds 0.92-0.96
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


#: pfill-k's chunk (pfill_arm.PFillK's: 8 warps a block, 256 fragments a warp a tile, about 2^19 MMAs an SM a chunk)
K_FILLER = A.Filler(W=8, NC=4, S=256, chunk_log2=19, noise="bits01", fold="ram")


@dataclass
class FusedPumpK:
    """FILL's step without filler warps (flin, persistent attention, lm_head through flin) on TABLE_PUMP's splits, with
    pfill-k pumped on the filler SMs (-DPFILL_BITS: fresh {0,1} SHA-512 noise for every IMMA, each k32 word into the
    ram receipt), and pfill-k's prefill (the linears on 184 SMs, the filler on 4): a split's slowdown is the split's
    own, whatever runs on its filler SMs."""

    name: str = "fill-fused-pump-k"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-fused-pump", "approach": "pouw/filler-fused",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FusedPump's splits beside a pfill-k pump read Theorem K's 0.9 under filler-scorer/v3: the pump's "
                      "0.88 of FILL v1's rate an SM, on 44-140 SMs, projects 0.365"})
    sources: tuple = ("verity/protocols/resource_accounting/compute/pouw/prover/kernels/filler/dcap.cu", "verity/protocols/resource_accounting/compute/pouw/prover/kernels/filler/flin.cuh",
                      "benchmarks/pouw/filler/fscore.cu", "benchmarks/pouw/filler/fused_arm.py",
                      "benchmarks/pouw/filler/fused_score.py")
    filler: A.Filler | None = K_FILLER
    choice: str = ("FusedPump's offline table of pump-only splits, pfill-k's 4-SM prefill split; at serving time a lookup, "
                   "a green split and one graph a step")
    table: dict = field(default_factory=lambda: TABLE_PUMP, repr=False)
    nvcc_flags: str = "-DPFILL_BITS"

    def __post_init__(self):
        os.environ["NVCC_APPEND_FLAGS"] = self.nvcc_flags

    def _row(self, b, l):
        return A.point_lookup(self.table, b, l)

    def placement(self, phase: str, b: int, l: int) -> A.Colocation:
        return A.Colocation("split", 4 if phase == "prefill" else self._row(b, l)[0])

    def fused(self, phase: str, b: int, l: int) -> FusedStep | None:
        return None if phase == "prefill" else self._row(b, l)[1]

    def filler_at(self, phase: str, b: int, l: int) -> A.Filler | None:
        return self.filler


#: pfill-bU's chunk (pfill_arm.PFillBU's shape) with duo's receipt (NC 2), or one of the r9 at NC 4: XOR-in (13, 19)
#: (-DPFILL_XIN13) or LOP3+IMAD (-DPFILL_LI). Each takes 1024 fragments a warp a tile: a tile's B hashing and its
#: barriers then stall a block a quarter as often, 0.772 of peak alone at S 256, 0.795 at 1024
BU_DUO_FILLER = A.Filler(W=8, NC=2, S=1024, chunk_log2=19, noise="bitsU", fold="duo")
BU_XIN13_FILLER = A.Filler(W=8, NC=4, S=1024, chunk_log2=19, noise="bitsU", fold="xor-in-13")
BU_LI_FILLER = A.Filler(W=8, NC=4, S=1024, chunk_log2=19, noise="bitsU", fold="lop3-imad")


@dataclass
class FusedPumpBU(FusedPumpK):
    """FusedPumpK's splits with pfill-bU pumped (-DPFILL_BU), the tile-reuse filler: a block's tile is the
    GEMM C = A B with A (128 S x 32) on {0,1} and B (32 x 1024) on U = {±1..±64}, from the coins and the public tile
    id; B sits in shared memory for the block's 8 warps and each A fragment, expanded once, meets all 128 B fragments,
    so the hashing is ~1 issue slot an MMA (dfill_k_warp's ~10). Every k32 word goes into duo's receipt (NC 2, three
    ops a word), which reads every word where ram's reads a pair only through Q."""

    name: str = "fill-fused-pump-bu"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-fused-pump-k", "approach": "pouw/filler-fused",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-bU", "receipt:duo-k32", "place:split",
                      "split:per-point-table", "fused:flin-serving", "fused:persistent-attention", "fused:lm-head-flin",
                      "prefill:split"],
        "hypothesis": "FusedPumpK's splits with a pfill-bU pump (Theorem K's served GEMM shape, every k32 word under "
                      "duo's word-binding fold) score at duo's alone rate, 0.59 of peak against ram's 0.77"})
    filler: A.Filler | None = BU_DUO_FILLER
    choice: str = ("FusedPump's offline table of pump-only splits, a pfill-bU duo pump, pfill's 4-SM prefill split; at "
                   "serving time a lookup, a green split and one graph a step")
    #: FSCORE_PUMP_ONLY: fscore built without flin's and attention's filler warps (every row is FW = AFW = 0), so the
    #: binary filler-scorer/v5 reads holds no ram-folding dfill_k_warp clone beside the pump's duo
    nvcc_flags: str = "-DPFILL_BU -Xcompiler -mpopcnt -DFSCORE_PUMP_ONLY"


def _bu_lineage(receipt: str, hypothesis: str) -> dict:
    return {"parent": "fill-fused-pump-bu", "approach": "pouw/filler-fused",
            "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-bU", receipt, "place:split",
                          "split:per-point-table", "fused:flin-serving", "fused:persistent-attention",
                          "fused:lm-head-flin", "prefill:split"],
            "hypothesis": hypothesis}


@dataclass
class FusedPumpBUXin13(FusedPumpBU):
    """FusedPumpBU with the r9 XOR-in (13, 19) for duo (a drop-in, NC 4 built with -DPFILL_XIN13, three ops
    a pair, keys under pf_key_ok_r9): every word read with no layout premise. It replaces XOR-in (7, 19), whose step
    boundary's ±128 is an int8 ratio. Provisional while r9 is
    refereed."""

    name: str = "fill-fused-pump-bu-xin13"
    lineage: dict = field(default_factory=lambda: _bu_lineage(
        "receipt:xin13-k32", "XOR-in (13, 19) for duo in FusedPumpBU's pump keeps XOR-in (7, 19)'s alone rate (its "
        "rotation is free), 0.80 of peak at S 1024, with every word read in full under filler-scorer/v5"))
    filler: A.Filler | None = BU_XIN13_FILLER
    choice: str = ("FusedPump's offline table of pump-only splits, a pfill-bU XOR-in (13, 19) pump, pfill's 4-SM prefill "
                   "split; at serving time a lookup, a green split and one graph a step")
    nvcc_flags: str = "-DPFILL_BU -DPFILL_XIN13 -Xcompiler -mpopcnt -DFSCORE_PUMP_ONLY"


@dataclass
class FusedPumpBULi(FusedPumpBU):
    """FusedPumpBU with the r9 LOP3+IMAD for duo (a drop-in, NC 4 built with -DPFILL_LI, two ops a pair as
    ram's, keys under pf_key_ok_r9), r9's pick for the filler: every word read given (HO), which the pump's hashed rows
    give. Provisional while r9 is under referee."""

    name: str = "fill-fused-pump-bu-li"
    lineage: dict = field(default_factory=lambda: _bu_lineage(
        "receipt:li-k32", "LOP3+IMAD for duo in FusedPumpBU's pump runs at ram's op count, 0.80 of peak alone at S 1024, "
        "with every word read in full under filler-scorer/v5"))
    filler: A.Filler | None = BU_LI_FILLER
    choice: str = ("FusedPump's offline table of pump-only splits, a pfill-bU LOP3+IMAD pump, pfill's 4-SM prefill split; "
                   "at serving time a lookup, a green split and one graph a step")
    nvcc_flags: str = "-DPFILL_BU -DPFILL_LI -Xcompiler -mpopcnt -DFSCORE_PUMP_ONLY"


def _warp(sms, fw, st=32, lm=1, aps=1):
    return sms, FusedStep(FW=fw, AFW=0, APS=aps, ALZ=0, LM=lm, ST=st)


#: pfill-k's filler warps (dfill_k_warp) in flin's blocks with FLIN_STOP 32, beside the pfill-k pump, at every point where
#: they fit (screened <= 1.02, the linears' filler >= 0.005 of peak), else TABLE_PUMP's row: six
#: screens. No attention filler
#: warps (a pfill-k step outlasts an attention launch: 32/2K FW 8 + AFW 8 2.02 against FW 8 alone 1.37);
#: none at 96/8K (stop 32 1.021, stop 64 never steps) or B >= 132 (1.04-1.07 at 2 warps, stops 32-64, on 44-76 SMs).
TABLE_WARP_K = {
    (1, 2048): _warp(100, 4, aps=0), (1, 32768): _warp(100, 4), (8, 2048): _warp(108, 4), (8, 32768): _warp(116, 4),
    (32, 1024): _warp(116, 2, lm=0), (32, 2048): _warp(116, 4), (32, 5120): _warp(116, 4), (32, 8192): _warp(116, 4),
    (32, 16384): _warp(116, 2), (32, 24576): _warp(108, 2), (48, 2048): _warp(92, 2), (48, 16384): _warp(84, 2),
    (64, 2048): _warp(92, 2), (64, 12288): _warp(100, 2), (96, 8192): TABLE_PUMP[(96, 8192)],
    (128, 2048): _warp(68, 2, lm=0), (128, 4096): _warp(76, 2, lm=0),
    **{k: TABLE_PUMP[k] for k in [(132, 5120), (144, 5120), (192, 2048), (256, 1024), (256, 2048), (256, 3072)]},
}


@dataclass
class FusedWarpK(FusedPumpK):
    """In-kernel filler warps on pfill-k's arithmetic: each flin block's filler warps call `dfill_k_warp`
    (flin.cuh, a `__noinline__` step that holds every filler MMA and reads only the coins, the slot's public tile id and
    the step), each k32 word into the ram receipt, beside FusedPumpK's pump and prefill. A step hashes one SHA-512 block a
    lane (~17 issue slots an MMA against FILL v1's ~1), so the warps fit only on narrower splits than the pump alone's,
    and stop with 32 units of their block left (FLIN_STOP) rather than outlast its last unit."""

    name: str = "fill-fused-warp-k"
    lineage: dict = field(default_factory=lambda: {
        "parent": "fill-fused-pump-k", "approach": "pouw/filler-fused",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "place:split",
                      "split:per-point-table", "fused:flin-filler-warps", "fused:filler-stop", "fused:persistent-attention",
                      "fused:lm-head-flin", "prefill:split"],
        "hypothesis": "pfill-k's step as flin's filler warps (dfill_k_warp), stopped 32 units before their block's end, "
                      "fit decode's step at <= 1.02 on 68-116 pump SMs at B <= 128: a Theorem K filler inside serving's "
                      "kernels, below FusedPumpK"})
    choice: str = ("an offline table of splits, flin filler warps and stops from FILL's round-4 screens (FusedPump's pump-only "
                   "rows where no filler warp fits), pfill-k's 4-SM prefill split; at serving time a lookup, a green split "
                   "and one graph a step")
    table: dict = field(default_factory=lambda: TABLE_WARP_K, repr=False)
