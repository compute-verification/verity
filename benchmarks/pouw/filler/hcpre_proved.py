"""HC-PREFILL's proved arm (approach pouw/filler-prefill): pfill-k's filler, {0,1} bits on both operands of s8 IMMA with
every accumulator input zero (Theorem K's setting, read at 0.9 by filler-scorer/v3), built as pfill.cuh's co tile
(-DPFILL_CO): the 4-warp tile's math with B as bits in 2 KB of shared memory and A's SHA blocks in registers, so one block
fits beside the prefill GEMM's CTA (96 KiB + 1 KiB of an SM's 100 KB), where pfill's 8-warp tile (48 KB) can't, and a
chunk over every SM runs one block an SM. -DPFILL_NO_PHASE leaves the phase filler (dfill_p) out of the build, so the
filler's MMAs are dfill_k's alone. The flags reach score.py's nvcc through NVCC_APPEND_FLAGS, set when the arm loads.

  research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=hc-prefill \\
    -- python3 benchmarks/pouw/filler/score.py benchmarks/pouw/filler/hcpre_proved.py:PFillKCo
"""

# No `from __future__ import annotations`: arms.load runs this file without registering it in sys.modules, and
# dataclasses needs the module to resolve string annotations.
from dataclasses import dataclass, field

from arms import Filler, Placement, point_lookup
from pfill_arm import POWER_SPLIT, SOURCES, PFillK


@dataclass
class PFillKCo(PFillK):
    """pfill-k's decode splits (HC-POWER's 15) with the co tile on the split's SMs (4 blocks an SM), and prefill co: the
    linears on every SM at the highest priority, one filler block an SM beside them at the lowest."""

    name: str = "hcpre-pfill-k-co"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k", "approach": "pouw/filler-prefill",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "tile:co-w4-bits-b-regs-a",
                      "place:split", "split:per-point-table", "split:every-scored-point", "prefill:co",
                      "co:one-block-an-sm", "chunk:2^20"],
        "hypothesis": "pfill-k's tile laid out in 2 KB of smem fits beside the prefill GEMM, so prefill takes a co "
                      "filler's share at Theorem K's 0.9 where pfill-k had a split of 4"})
    sources: tuple = SOURCES + ("benchmarks/pouw/filler/hcpre_proved.py",)
    filler: Filler | None = Filler(W=4, NC=4, S=256, chunk_log2=20, noise="bits01")
    choice: str = ("HC-POWER's offline table of 15 (B, L) decode splits, and co for prefill; a lookup and a green split or "
                   "co at serving time")
    nvcc_flags: str = "-DPFILL_BITS -DPFILL_CO -DPFILL_NO_PHASE"

    def placement(self, phase, b, l):
        if phase == "prefill":
            return Placement("co")
        return Placement("split", point_lookup(POWER_SPLIT, b, l))


@dataclass
class PFillKCoDep(PFillKCo):
    """PFillKCo with each co MMA waiting on the fold before it (-DPF_CO_DEP=2), one MMA in flight a warp: the co tile's
    ~7 instructions an MMA take the GEMM's issue slots, so the full-rate tile costs the prefill linears about 1.15 of
    slowdown a unit of share, and one in flight holds it to the gate."""

    name: str = "hcpre-pfill-k-co-dep"
    lineage: dict = field(default_factory=lambda: {
        "parent": "hcpre-pfill-k-co", "approach": "pouw/filler-prefill",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "tile:co-w4-bits-b-regs-a",
                      "place:split", "split:per-point-table", "split:every-scored-point", "prefill:co",
                      "co:one-block-an-sm", "co:one-mma-in-flight", "chunk:2^20"],
        "hypothesis": "the co tile's ALU work takes the GEMM's issue slots, so with one MMA in flight a warp it fills "
                      "prefill at the gate"})
    nvcc_flags: str = "-DPFILL_BITS -DPFILL_CO -DPFILL_NO_PHASE -DPF_CO_DEP=2"


@dataclass
class PFillKDual(PFillKCoDep):
    """pfill-k itself on its decode splits (the 8-warp tile, its B and A moved from shared memory to L1, -DPFILL_DUAL) and
    the co tile at one MMA in flight for prefill co: the kernel's only shared memory is the co tile's 2 KB, so one build
    holds both. A co chunk runs 4-warp co tiles (dfill_make sets the chunk's W, its receipts dfill_ref's 4-warp tile).
    Its L1 tile fails decode's gate where pfill-k's shared-memory tile passes (1/32K split 108: 1.067 with the filler at
    0.83-0.89 of its alone rate, at any carveout; r20261016-101543-2c88, r20261016-103453-84b7): it loads L2 beside
    memory-bound decode."""

    name: str = "hcpre-pfill-k-dual"
    lineage: dict = field(default_factory=lambda: {
        "parent": "pfill-k", "approach": "pouw/filler-prefill",
        "mechanism": ["filler:fresh-noise-imma", "noise:sha512-coins-01", "receipt:ram-k32", "tile:pfill-k-l1",
                      "tile:co-w4-bits-b-regs-a", "place:split", "split:per-point-table", "split:every-scored-point",
                      "prefill:co", "co:one-block-an-sm", "co:one-mma-in-flight", "chunk:2^19"],
        "hypothesis": "pfill-k's tile with B and A in L1 keeps its decode rate, and the co tile adds prefill at the gate "
                      "in the same build"})
    filler: Filler | None = Filler(W=8, NC=4, S=256, chunk_log2=19, noise="bits01")
    nvcc_flags: str = "-DPFILL_BITS -DPFILL_DUAL -DPFILL_NO_PHASE -DPF_CO_DEP=2"
