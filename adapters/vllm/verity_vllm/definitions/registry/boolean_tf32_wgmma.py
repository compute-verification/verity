"""`tf32_wgmma`'s k8 step on bits: `HopperTf32WgmmaDot8_v2`, traced from `fp.tc_dot_tf32` after RNE f32 -> TF32 on both operands,
the candidate's groups, adder width and floor."""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import OR, add, and_reduce, mux, or_reduce
from verity.circuits.boolean.trace import definition
from verity.circuits.census import silicon_step
from verity.circuits.silicon import fp as FP
from verity.circuits.types import Array

from verity_vllm.definitions.registry import tf32_wgmma as W

F32 = Array(32, BL.BIT)


def _tf32_rne(C, w):
    """A 32-bit f32 word rounded RNE to TF32's sign, exponent and ten mantissa bits, LSB first."""
    high = list(w[13:])
    special = and_reduce(C, w[23:31])
    nan = C.AND(special, or_reduce(C, w[:23]))
    inc = C.AND(w[12], OR(C, or_reduce(C, w[:12]), high[0]))
    rounded = add(C, high, [inc], width=19)
    kept = list(high)
    kept[0] = OR(C, kept[0], nan)
    return mux(C, special, kept, rounded)


HopperTf32WgmmaDot8V2 = silicon_step(definition(
    "HopperTf32WgmmaDot8", 2,
    lambda C, acc, *ab: FP.tc_dot_tf32(C, acc, [_tf32_rne(C, x) for x in ab[:8]], [_tf32_rne(C, y) for y in ab[8:16]],
                                       W.TF32_GROUPS, W.TF32_WIDTH, W.TF32_FLOOR),
    (("acc", F32), ("a", Array(8, F32)), ("b", Array(8, F32))), F32, word=W.HopperTf32WgmmaDot8,
    doc="the candidate wgmma.m64nNk8 tf32 step on bits: HopperTf32WgmmaDot8_v1"), W.HopperTf32WgmmaDot8.silicon)
