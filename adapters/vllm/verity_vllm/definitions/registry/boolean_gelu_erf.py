"""aten's erf GELU on one bf16 word as a Boolean Definition, the next version of its word primitive, on bits.

    GeluErfBf16_v2            (x: Array<16, Value<1>>) -> Array<16, Value<1>>                       word view: GeluErfBf16_v1

`GeluErfBf16_v1` is the measured word of every bf16 x (`act`), so `gelu_erf_bf16` is closed forms on x's fields and the
measured words on two bands (`boolean_activation.band`, one table each, since e = 0 and e = 128 share their low four bits):

      2 <= e <= 117                 x / 2: x's exponent field less one
      e <= 1, or 118 <= e <= 129    the table's magnitude, x's sign
      e >= 130, x > 0               x (+inf included)
      e >= 130, x < 0               -0
      e = 255, NaN or -inf          0x7FFF

The positive measured words end at e = 128; the pair (128, 129) holds 129's too, which are x.
"""
from __future__ import annotations

from functools import lru_cache

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, and_reduce, const_bits, or_reduce, sub
from verity.circuits.boolean.trace import definition
from verity.circuits.types import Array

from verity_vllm.definitions.registry import act as A
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.boolean_activation import _set, band

__all__ = ["GELU_ERF_BANDS", "GeluErfBf16", "gelu_erf_bf16", "gelu_erf_table"]

BF16 = Array(16, BL.BIT)
NAN_F2FP = P.BF16_NAN_F2FP
#: the exponent fields each of the two tables covers, both signs, even to odd
GELU_ERF_BANDS = ((0, 1), (118, 129))


@lru_cache(maxsize=None)
def gelu_erf_table(k: int) -> tuple[int, ...]:
    """``GeluErfBf16_v1(x) & 0x7FFF`` for every x on band ``k`` of `GELU_ERF_BANDS`, at index ``m | (e mod 16) << 7 | s << 11``;
    0 off it."""
    out = [0] * (1 << 12)
    lo, hi = GELU_ERF_BANDS[k]
    for sign in (0, 1):
        for e in range(lo, hi + 1):
            for m in range(128):
                out[m | (e & 0xF) << 7 | sign << 11] = A.gelu_erf_bf16_bits(sign << 15 | e << 7 | m) & 0x7FFF
    return tuple(out)


def gelu_erf_bf16(C, x):
    """`GeluErfBf16_v1` for a 16-bit x (LSB first)."""
    m, e, s = list(x[:7]), list(x[7:15]), x[15]
    e255, mnz = and_reduce(C, e), or_reduce(C, m)
    _, lt2 = sub(C, e, const_bits(2, 8), 8)
    _, lt118 = sub(C, e, const_bits(GELU_ERF_BANDS[1][0], 8), 8)
    _, lt130 = sub(C, e, const_bits(GELU_ERF_BANDS[1][1] + 1, 8), 8)
    half = C.AND(NOT(lt2), lt118)
    ident = C.AND(C.AND(NOT(s), NOT(lt130)), NOT(C.AND(e255, mnz)))
    nan = C.AND(e255, NOT(C.AND(NOT(s), NOT(mnz))))
    e1, _ = sub(C, e, const_bits(1, 8), 8)
    mag = [C.AND(half, b) ^ C.AND(ident, a) for a, b in zip(x[:15], m + e1)]
    for k, (lo, hi) in enumerate(GELU_ERF_BANDS):
        rows = [(sign, e0) for sign in (0, 1) for e0 in range(lo, hi + 1, 2)]
        mag = [a ^ b for a, b in zip(mag, band(C, x, gelu_erf_table(k), rows))]
    return _set(mag, NAN_F2FP, nan) + [C.AND(s, NOT(nan))]


GeluErfBf16 = definition("GeluErfBf16", 2, gelu_erf_bf16, (("x", BF16),), BF16, word=A.GeluErfBf16,
                         doc="aten.gelu (erf) on one bf16 word, on bits: GeluErfBf16_v1, its measured words")
