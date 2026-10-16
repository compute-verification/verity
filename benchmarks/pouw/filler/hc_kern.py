"""HC-KERN's filler arms (approach pouw/filler-kernel; note:20261016T0437Z-draft-hc-kern): FILL v1's register chain with
the accumulator quad read in register order, so a step is one in-place IMMA and no MOVs (dcap.cu's map 1), and with
FILL-SEC's free hardening on top (map 2: a 128-bit tile seed, C in ±2^15; note:20261016T0418Z-draft-filler-floor). Both
keep `fill-v1`'s split table and prefill policy, so the kernel is the only change against the baseline.

filler-arm/v1 has no field for a kernel variant behind dcap.cu's ABI; dcap.cu reads it from DFILL_MAP (`dfill_make`,
and `dfill_ref` when called without a map), so an arm sets DFILL_MAP when it is constructed and every process the scorer
starts after loading the arm inherits it.

  python3 benchmarks/pouw/filler/score.py benchmarks/pouw/filler/hc_kern.py:KernM2

No `from __future__ import annotations`: `arms.load` runs the module without registering it in sys.modules, where
dataclasses looks up string annotations.
"""

import os
from dataclasses import dataclass, field

import arms as A


@dataclass
class KernM2:
    """Map 2 (hardened) at W 16, NC 4, S 8192: 4 warps a sub-partition, one 8192-step tile a warp a chunk, which
    amortizes map 2's costlier expansion and receipt; 0.990 of int8 peak alone (r20261016-045022-c45c)."""

    name: str = "kern-m2"
    lineage: dict = field(default_factory=lambda: {
        "parent": "cursor/dram-free-filler-ea24@e015e0515", "approach": "pouw/filler-kernel",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:seed-128",
                      "kernel:c-16bit", "kernel:w16-s8192", "place:split", "split:per-point-table", "prefill:none"],
        "hypothesis": "FILL v1's 4 MOVs a step serialize its chains; in-place IMMAs lift the filler 0.89 -> 0.99 of "
                      "peak on its SMs at the same split and decode cost"})
    sources: tuple = ("benchmarks/pouw/decode_capacity/dcap.cu", "benchmarks/pouw/filler/hc_kern.py")
    filler: A.Filler | None = A.Filler(W=16, NC=4, S=8192, chunk_log2=19)
    choice: str = "fill-v1's offline table of 11 (B, L) splits; a lookup and a green split at serving time"
    map: int = 2

    def __post_init__(self):
        os.environ["DFILL_MAP"] = str(self.map)

    def placement(self, phase, b, l):
        return A.FillV1().placement(phase, b, l)


@dataclass
class KernM2C18(KernM2):
    """Map 2 at W 8, NC 4, S 8192 in 2^18-MMA chunks (0.51 ms): 0.988 of peak alone against 0.994 at 2^19
    (r20261016-051127-18ab). dscore counts whole chunks in windows of 3 decode steps whose phase against the chunk
    boundaries is the same every block, so a beside-serving rate is off by up to one chunk a window; half-length chunks
    halve that, and two warps a sub-partition (not four, as at S 4096) keep the per-chunk prologue short."""

    name: str = "kern-m2-c18"
    lineage: dict = field(default_factory=lambda: {
        "parent": "kern-m2", "approach": "pouw/filler-kernel",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:seed-128",
                      "kernel:c-16bit", "kernel:w8-s8192", "chunk:2^18", "place:split", "split:per-point-table",
                      "prefill:none"],
        "hypothesis": "kern-m2 in half-length chunks: 0.6% less filler rate alone, half the whole-chunk counting error "
                      "in dscore's shortest windows"})
    filler: A.Filler | None = A.Filler(W=8, NC=4, S=8192, chunk_log2=18)


@dataclass
class KernM2C21(KernM2):
    """kern-m2 in 2^21-MMA chunks (4 tiles a warp, ~4 ms): 0.998 of peak alone against 0.994 at 2^19
    (r20261016-051127-18ab). Beside serving kern-m2 keeps 0.975-0.990 of peak against 0.990 alone, losing most at the
    largest batches, where serving launches the most kernels beside the filler's chunk launches; four times longer
    chunks pay each launch gap a quarter as often. dscore (5e74544b9 on) times a block's rate over its completion span,
    which a chunk's length no longer biases."""

    name: str = "kern-m2-c21"
    lineage: dict = field(default_factory=lambda: {
        "parent": "kern-m2", "approach": "pouw/filler-kernel",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:seed-128",
                      "kernel:c-16bit", "kernel:w16-s8192", "chunk:2^21", "place:split", "split:per-point-table",
                      "prefill:none"],
        "hypothesis": "kern-m2 in 4x longer chunks: the filler pays its chunk-launch gap a quarter as often, 0.99 -> "
                      "0.995 of peak beside serving"})
    filler: A.Filler | None = A.Filler(W=16, NC=4, S=8192, chunk_log2=21)


@dataclass
class KernM2C20(KernM2):
    """kern-m2 in 2^20-MMA chunks (2 tiles a warp, ~2 ms). kern-m2-c21 (r20261016-054503-d1e8) kept 0.992-0.997 of peak
    beside serving and scored 0.425 if gated, but read 1.001 of peak at decode 32/2K, where a window holds 7 chunks and the
    blocks' NVML clock (2092 MHz) sits below the SM clock's 2100: the honest gate's peak bound has no tolerance for either.
    Half the chunk keeps about half the launch-gap gain with twice the completions a window."""

    name: str = "kern-m2-c20"
    lineage: dict = field(default_factory=lambda: {
        "parent": "kern-m2-c21", "approach": "pouw/filler-kernel",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:seed-128",
                      "kernel:c-16bit", "kernel:w16-s8192", "chunk:2^20", "place:split", "split:per-point-table",
                      "prefill:none"],
        "hypothesis": "kern-m2 in 2x longer chunks: half the chunk-launch gaps, ~0.99 of peak beside serving, under the "
                      "peak bound's noise"})
    filler: A.Filler | None = A.Filler(W=16, NC=4, S=8192, chunk_log2=20)


@dataclass
class KernM1(KernM2):
    """Map 1 (in-place, unhardened) at FILL v1's own shape, W 8, NC 4, S 2048: 0.984 of peak alone."""

    name: str = "kern-m1"
    lineage: dict = field(default_factory=lambda: {
        "parent": "cursor/dram-free-filler-ea24@e015e0515", "approach": "pouw/filler-kernel",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "place:split",
                      "split:per-point-table", "prefill:none"],
        "hypothesis": "FILL v1's chain with the accumulator quad read in register order: no MOVs, 0.89 -> 0.98 of "
                      "peak on its SMs at FILL v1's shape, split and decode cost"})
    filler: A.Filler | None = A.Filler(W=8, NC=4, S=2048, chunk_log2=19)
    map: int = 1
