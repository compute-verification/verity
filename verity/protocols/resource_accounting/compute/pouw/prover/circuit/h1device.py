"""H1's device records: the hardware facts an H1 execution phase (`h1window`, the `eps1-sm120-h1fx1` tile) is built and served on, one a
GPU architecture. `h1window.h1_tile` takes one by its architecture (``arch``) and binds its tiles' exact accumulator with the
record's s8 binding.

* **The step.** Each record names the s8 tensor-core step its kernels' MMAs compute (a `silicon.integer.IntegerMma`, PINNED
  in `silicon.instructions`) and binds it with `ops.boolean.gemm.s8_gemm`, which gives the registered
  `BlackwellS8ImmaDot32_v2` for every card whose step is that function and refuses any other.
* **The domain** is the Definitions' own refusals (32 | K and K < 2^17, `ops.gemm.S8_K_LIMIT`): a record bounds no K
  beyond them.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, IntegerMma


@dataclass(frozen=True)
class Device:
    """One architecture's H1 record (module docstring)."""

    gpu: str
    step: IntegerMma                       # the s8 step the kernels' MMAs compute
    instruction: str                       # its `silicon.instructions` entry, the instruction the kernels issue

    @property
    def arch(self) -> str:
        """The architecture whose record this is: its s8 step's."""
        return self.step.arch

    @property
    def gemm(self) -> dict:
        """The tiles' s8 binding on bits (`BG.s8_gemm` of the record's step)."""
        return BG.s8_gemm(self.step)


SM120 = Device(
    gpu="RTX PRO 6000 Blackwell Server Edition", step=BLACKWELL_SM120_S8_M16N8K32,
    instruction="sm120.mma.m16n8k32.s8",
)

DEVICES = {d.arch: d for d in (SM120,)}


def device(arch: str) -> Device:
    """The H1 record of architecture ``arch`` (`DEVICES`), else ValueError."""
    try:
        return DEVICES[arch]
    except (KeyError, TypeError):
        raise ValueError(f"H1 has no device record for {arch!r}: its records are {', '.join(DEVICES)}") from None


__all__ = ["DEVICES", "Device", "SM120", "device"]
