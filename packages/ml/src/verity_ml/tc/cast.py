"""Word casts between FP32 and BF16 encodings, as pure integer functions.

Promoted from ``veritor`` ``verity_ir/registry/prims.py`` (``bf16_to_f32_bits``, ``f32_to_bf16_rn_bits``, ``f32_is_nan``)
and ``src/veritor/core/tc.py`` (the word-level casts); the numpy array forms were left behind.
"""
from __future__ import annotations

BF16_NAN_SOFTWARE = 0x7FC0   # c10::BFloat16::round_to_nearest_even, torch CPU: the software canonical NaN
BF16_NAN_HARDWARE = 0x7FFF   # SASS F2FP.BF16.F32.PACK_AB / cvt.rn.bf16.f32: measured on A100 and RTX 4090 (L6)
F32_NAN_CANONICAL = 0x7FC00000

__all__ = ["BF16_NAN_HARDWARE", "BF16_NAN_SOFTWARE", "F32_NAN_CANONICAL", "bf16_is_nan", "bf16_to_f32_word", "f32_is_nan",
           "f32_to_bf16_hw_word", "f32_to_bf16_rn_word", "f32_to_bf16_word"]


def f32_is_nan(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def bf16_is_nan(w: int) -> bool:
    return (w & 0x7F80) == 0x7F80 and (w & 0x7F) != 0


def bf16_to_f32_word(w: int) -> int:
    """Exact widening: the BF16 word becomes the high half of the FP32 word."""
    return (w & 0xFFFF) << 16


def f32_to_bf16_rn_word(u: int, nan: int) -> int:
    """Round to nearest, ties to even, by integer arithmetic on the encoding; every NaN maps to ``nan``.
    A finite word that rounds past the largest finite BF16 carries into the exponent and becomes an infinity."""
    u &= 0xFFFFFFFF
    if f32_is_nan(u):
        return nan
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def f32_to_bf16_word(u: int) -> int:
    """The software conversion (torch CPU, ``gpu/gpt2/pinned_ops.cu``): NaN -> ``0x7FC0``."""
    return f32_to_bf16_rn_word(u, BF16_NAN_SOFTWARE)


def f32_to_bf16_hw_word(u: int) -> int:
    """``cvt.rn.bf16.f32`` as the A100 and RTX 4090 compute it: NaN -> ``0x7FFF`` (equal to ``total.cvt_rn_bf16_f32``)."""
    return f32_to_bf16_rn_word(u, BF16_NAN_HARDWARE)
