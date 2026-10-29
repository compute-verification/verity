"""A W8A8 coordinate (`ScaledMmS8Coordinate_v2{K, GEMM_SM90_S8}`) split along K into Boolean Definitions that each output at
most 16 bits (a proof block's outputs, the 16-bit fallback forms).

    S8AccumulatorPartHalf_v1{K, HALF}  (x, wrow: Array<K, Array<8, Value<1>>>) -> Array<16, Value<1>>
    I32SumPartHalf_v1{G, HALF}         (s: Array<G, Array<32, Value<1>>>) -> Array<16, Value<1>>
    ScaledMmS8Epilogue_v1{G}           (s: Array<G, Array<32, Value<1>>>, sx, sw: Array<32, Value<1>>) -> Array<16, Value<1>>

Under the 16-bit rule a coordinate reads x as K/2 rows of `QuantS8Pair_v1`'s two codes, so the unsplit coordinate at K = 512
reads 260 rows with sx's two halves, wrow and sw, past the 255 a proof block reads, and at K = 16,384 about 8,200. The split:

- `S8AccumulatorPartHalf` is the low (HALF = 0) or high (HALF = 1) 16 bits of a chunk's s32 accumulator,
  `GemmAccumulatorS8_v1{K, GEMM_SM90_S8}`'s: K/32 of the H200's IMMA k32 steps (`integer.HOPPER_SM90_S8_K32` on bits,
  `step_s8_bits`) in ascending k from 0.
- `I32SumPartHalf` is 16 bits of G s32 words summed from 0 in ascending order, each add `I32Add_v1`'s.
- `ScaledMmS8Epilogue` sums its G s32 words the same way and applies the coordinate's epilogue to the sum:
  ``F2fpBf16_v2((S32ToF32Rn_v2(acc) * sx) * sw)`` with `F32Mul_v4`, the token's scale first.

A later proof block reads an s32 as its two halves' rows. The split computes the unsplit coordinate's function on every input when
its chunks' K sum to less than `S8_K_LIMIT` = 2^17. Every partial sum of fewer than 2^17 s8 products from 0 lies in s32
(`integer.exact_k_limit`), so no step clamps, no add wraps, and every grouping gives the same exact sum. Chunks whose K sum to
2^17 or more don't give the coordinate's function.

Phase 1's K = 128 coordinate reads 64 + 4 rows and needs no split. Its K = 512 becomes 2 chunks of K = 256 (each reading 128
pair rows and its chunk of wrow), 4 halves, and an epilogue at G = 2 (4 + 2 + 1 rows). Phase 2's K = 16,384 becomes 64 chunks,
128 halves and an epilogue at G = 64 (131 rows). Its K = 65,536 becomes 256 chunks, then 4 groups of 64 summed by
`I32SumPartHalf_v1{64}` (8 halves), and an epilogue at G = 4.

Each is traced from its builders (`elementwise.specialized`), so a half's circuit holds only the gates its own bits read. Each
refuses at its signature a K or G it doesn't cover.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.silicon.integer import HOPPER_SM90_S8_K32, step_s8_bits
from verity.ml.circuits.boolean.forms import add, const_bits

from .. import gemm as G
from .elementwise import specialized

__all__ = ["I32SumPartHalf", "S8AccumulatorPartHalf", "ScaledMmS8Epilogue", "accumulator", "i32_sum", "scaled_epilogue"]

CODE = Array(8, BL.BIT)
S32 = F32 = Array(32, BL.BIT)
U16 = Array(16, BL.BIT)
_STEP = HOPPER_SM90_S8_K32.k
#: F2fpBf16_v1's NaN
_NAN_F2FP = 0x7FFF


def accumulator(C, x, w):
    """The s32 accumulator of s8 codes ``x`` and ``w`` on bits: len(x)/32 `HOPPER_SM90_S8_K32` steps in ascending k from 0."""
    acc = const_bits(0, 32)
    for k in range(0, len(x), _STEP):
        acc = step_s8_bits(C, HOPPER_SM90_S8_K32, acc, list(x[k:k + _STEP]), list(w[k:k + _STEP]))
    return acc


def i32_sum(C, words):
    """The s32 words summed from 0 in ascending order, each add wrapping (`I32Add_v1`)."""
    acc = const_bits(0, 32)
    for s in words:
        acc = add(C, acc, s, width=32)
    return acc


def scaled_epilogue(C, acc, sx, sw):
    """``F2fpBf16_v2((S32ToF32Rn_v2(acc) * sx) * sw)``, each multiply `F32Mul_v4` (NaN 0x7FFFFFFF)."""
    y = fp.f32_mul(C, fp.f32_mul(C, fp.s32_to_f32(C, acc), sx, nan_word=fp.F32_NAN_PTX), sw, nan_word=fp.F32_NAN_PTX)
    return fp.f32_to_bf16(C, y, _NAN_F2FP)


def _half(S: Any) -> int:
    if S.HALF not in (0, 1) or isinstance(S.HALF, bool):
        raise ValueError(f"HALF is 0 (the s32's low 16 bits) or 1 (its high 16 bits), not {S.HALF!r}")
    return S.HALF


def _chunk(S: Any) -> int:
    K = S.K
    if isinstance(K, bool) or not isinstance(K, int) or K < _STEP:
        raise ValueError(f"K is the chunk's length, a multiple of {_STEP} and at least {_STEP}, not {K!r}")
    why = G.GemmAccumulatorS8.refuse({"K": K})
    if why:
        raise ValueError(why)
    return K


def _count(S: Any) -> int:
    v = S.G
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"G is the number of s32 words summed, an int at least 1, not {v!r}")
    return v


def _accumulator_half(S: Any):
    k, lo = _chunk(S), 16 * _half(S)
    return lambda C, *e: accumulator(C, e[:k], e[k:])[lo:lo + 16]


def _sum_half(S: Any):
    _count(S)
    lo = 16 * _half(S)
    return lambda C, *s: i32_sum(C, s)[lo:lo + 16]


def _epilogue(S: Any):
    g = _count(S)
    return lambda C, *e: scaled_epilogue(C, i32_sum(C, e[:g]), e[g], e[g + 1])


def _accumulator_signature(S: Any):
    _half(S)
    chunk = Array(_chunk(S), CODE)
    return (("x", chunk), ("wrow", chunk)), U16


def _sum_signature(S: Any):
    _half(S)
    return (("s", Array(_count(S), S32)),), U16


S8AccumulatorPartHalf = specialized(
    "S8AccumulatorPartHalf", 1, ("K", "HALF"), _accumulator_half, _accumulator_signature,
    doc="16 bits (HALF) of a chunk's s32 accumulator of s8 codes on bits: K/32 sm_90 IMMA k32 .satfinite steps "
        "(HOPPER_SM90_S8_K32) in ascending k from 0, GemmAccumulatorS8_v1{K, GEMM_SM90_S8}'s")
I32SumPartHalf = specialized(
    "I32SumPartHalf", 1, ("G", "HALF"), _sum_half, _sum_signature,
    doc="16 bits (HALF) of G s32 words summed from 0 in ascending order on bits, each add wrapping (I32Add_v1)")
ScaledMmS8Epilogue = specialized(
    "ScaledMmS8Epilogue", 1, ("G",), _epilogue,
    lambda S: ((("s", Array(_count(S), S32)), ("sx", F32), ("sw", F32)), U16),
    doc="a W8A8 coordinate's epilogue on G s32 parts on bits: F2fpBf16_v2((S32ToF32Rn_v2(acc) * sx) * sw) with F32Mul_v4, "
        "acc the parts summed from 0 in ascending order (I32Add_v1)")
