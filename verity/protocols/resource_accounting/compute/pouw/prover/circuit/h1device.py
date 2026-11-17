"""H1's device records: the hardware facts an H1 execution phase (`h1window`, the `eps1-sm120-h1fx1` tile) is built and served on, one a
GPU architecture. `h1window.h1_tile` takes one by its architecture (``arch``), binds its tiles' exact accumulator with the
record's s8 binding and refuses a contraction outside the record's domain.

* **The step.** Each record names the s8 tensor-core step its kernels' MMAs compute (a `silicon.integer.IntegerMma`, PINNED
  in `silicon.instructions`) and binds it with `ops.boolean.gemm.s8_gemm`, which gives the registered
  `BlackwellS8ImmaDot32_v2` for every card whose step is that function and refuses any other. sm_90's step,
  `HOPPER_SM90_S8_K32`, is sm_120's under Hopper's name (the H200's ``wgmma`` m64nNk32 and ``mma.sync`` m16n8k32 agree
  with it word for word on the H200), so an sm_90 execution phase's Program is the sm_120 one's, byte for byte, and a served
  path registers the same trees on either card.
* **The domain** is the contractions the record's kernels serve, beyond what the Definitions refuse themselves (32 | K and
  K < 2^17, `ops.gemm.S8_K_LIMIT`). sm_90's is Dither's h1u units (`verity/protocols/experimental/dither/prover/kernels/h1u.cu`, gated on an H200 at
  GLM-5.3 int8 v3's shapes): 512 | K, as h1u reads each noise block as 16 segments' bits
  (``kh_unit``: ``K % 512`` refused), and K >= 384, `h1zy.LEAST_K`, the least K whose leaf's combine is proved, so the least
  K served is 512. On that domain m = `segment.n_digits`(K / 32) is 2 or 3, the digits h1u's one stream recombines. h1u
  issues ``mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`` without ``.satfinite``; below 2^17 no partial sum of s8 x s8
  products from a zero accumulator leaves s32 (`integer.exact_k_limit`), so it agrees with the clamped step word for word.
  sm_120's record states no domain: its execution phases keep the Definitions' own refusals, as before records existed.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, HOPPER_SM90_S8_K32, IntegerMma

from . import h1zy as Z


@dataclass(frozen=True)
class Device:
    """One architecture's H1 record (module docstring)."""

    arch: str
    gpu: str
    step: IntegerMma                       # the s8 step the kernels' MMAs compute
    instruction: str                       # its `silicon.instructions` entry, the instruction the kernels issue
    domain: tuple[int, int] | None         # (least K, K's divisor) the kernels serve; None: the Definitions' own

    def __post_init__(self) -> None:
        if self.step.arch != self.arch:
            raise ValueError(f"{self.arch}'s H1 record names {self.step.name}, an {self.step.arch} step")

    @property
    def gemm(self) -> dict:
        """The tiles' s8 binding on bits (`BG.s8_gemm` of the record's step)."""
        return BG.s8_gemm(self.step)

    def refuse(self, k: int) -> str | None:
        """Why the record's kernels serve no H1 tile at contraction ``k``, or None."""
        if self.domain is None:
            return None
        least, div = self.domain
        if k < least or k % div:
            return (f"K={k} is outside H1's domain on {self.arch} ({self.gpu}): K >= {least} (h1zy.LEAST_K, the least K "
                    f"whose leaf's combine is proved) and {div} | K (h1u reads each noise block as 16 segments' bits)")
        return None


SM120 = Device(
    arch="sm_120", gpu="RTX PRO 6000 Blackwell Server Edition", step=BLACKWELL_SM120_S8_M16N8K32,
    instruction="sm120.mma.m16n8k32.s8", domain=None,
)

SM90 = Device(
    arch="sm_90", gpu="H200 SXM (sm_90a)", step=HOPPER_SM90_S8_K32,
    instruction="sm90.mma.m16n8k32.s8", domain=(Z.LEAST_K, 512),
)

DEVICES = {d.arch: d for d in (SM120, SM90)}


def device(arch: str) -> Device:
    """The H1 record of architecture ``arch`` (`DEVICES`), else ValueError."""
    try:
        return DEVICES[arch]
    except (KeyError, TypeError):
        raise ValueError(f"H1 has no device record for {arch!r}: its records are {', '.join(DEVICES)}") from None


__all__ = ["DEVICES", "Device", "SM120", "SM90", "device"]
