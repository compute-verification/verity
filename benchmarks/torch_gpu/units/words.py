"""The word operations every unit's twin is made of, on numpy arrays of unsigned words, each one GPU instruction exactly
(units.cu's helper of the same name), in IEEE round-to-nearest-even with no flush to zero unless named.

  fadd fsub fmul fdiv fsqrt   add.rn / sub.rn / mul.rn / div.rn / sqrt.rn.f32 (__fadd_rn ... __fsqrt_rn): numpy's f32
                              arithmetic with every NaN result the GPU's 0x7FFFFFFF (`fp32.F32AddV2`, `F32MulV2`, `F32DivV2`)
  widen                       bf16 -> f32 (`prims.Bf16ToF32`): the word shifted up 16
  bf16                        cvt.rn.bf16.f32 (F2FP.BF16.F32.PACK_AB, `prims.F2fpBf16`): RNE, every NaN 0x7FFF
                              (`verity.ml.kernels.tc.f2fp_bf16_batch`)
  ex2                         ex2.approx.ftz.f32 (MUFU.EX2, `mufu.MufuEx2Ftz`, its registered kernel): a subnormal operand
                              reads as 0, a subnormal result is +0
  s8                          cvt.rni.sat.s8.f32 (`F32ToS8RniSat_v1`): ties to even, saturated to [-128, 127], NaN 0
  order16, isnan16, relu16    bf16 words' order (+0 = -0), NaN test and ReLU, integers only
  seq_sum                     f32 sums in ascending index from +0

verity is imported only by bf16 and ex2, so the card can import the unit modules without it.
"""
from __future__ import annotations

import numpy as np

F32_NAN = 0x7FFFFFFF
BF16_NAN = 0x7FFF
BF16_NEG_INF = 0xFF80
ONE = 0x3F800000
F127 = 0x42FE0000
#: RN32(log2 e), softmax's exponent scale
LOG2E = 0x3FB8AA3B
#: RMSNorm's epsilon, 1e-5 as the nearest f32
EPS = 0x3727C5AC


def u32(a) -> np.ndarray:
    return np.asarray(a).astype(np.uint32)


def _f(a) -> np.ndarray:
    return u32(a).view(np.float32)


def canon(r) -> np.ndarray:
    """f32 results as words, every NaN 0x7FFFFFFF."""
    r = np.asarray(r, np.float32)
    return np.where(np.isnan(r), np.uint32(F32_NAN), r.view(np.uint32)).astype(np.uint32)


def fadd(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return canon(_f(a) + _f(b))


def fsub(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return canon(_f(a) - _f(b))


def fmul(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return canon(_f(a) * _f(b))


def fdiv(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return canon(_f(a) / _f(b))


def fsqrt(a) -> np.ndarray:
    """sqrt(-0) = -0; a NaN or a negative nonzero operand 0x7FFFFFFF."""
    with np.errstate(all="ignore"):
        return canon(np.sqrt(_f(a)))


def widen(b) -> np.ndarray:
    return np.asarray(b).astype(np.uint32) << np.uint32(16)


def f32word(v: float) -> int:
    return int(np.float32(v).view(np.uint32))


def bf16(a) -> np.ndarray:
    from verity.ml.kernels.tc import f2fp_bf16_batch
    a = u32(a)
    if a.size == 0:
        return a.astype(np.uint16)
    return np.asarray(f2fp_bf16_batch(a.reshape(-1))[0]).astype(np.uint16).reshape(a.shape)


def ex2(a) -> np.ndarray:
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.circuits.ops import mufu
    a = u32(a)
    if a.size == 0:
        return a
    return np.asarray(evaluate_batch(mufu.MufuEx2Ftz, a.reshape(-1)), np.uint32).reshape(a.shape)


def s8(a) -> np.ndarray:
    """s8 words (uint8)."""
    v = _f(a)
    with np.errstate(all="ignore"):
        r = np.where(np.isnan(v), np.float32(0), np.clip(np.rint(v), -128, 127))
    return r.astype(np.int8).view(np.uint8)


def order16(x) -> np.ndarray:
    """An integer whose order is the bf16 values' (+0 and -0 equal); a NaN gets one too, so callers test isnan16."""
    x = np.asarray(x).astype(np.int64)
    m = x & 0x7FFF
    return np.where(x & 0x8000, -m, m)


def isnan16(x) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    return ((x & 0x7F80) == 0x7F80) & ((x & 0x7F) != 0)


def relu16(x, on: bool = True) -> np.ndarray:
    """x < 0 ? +0 : x on bf16 words: -0 and every NaN are themselves."""
    x = np.asarray(x).astype(np.uint16)
    if not on:
        return x
    return np.where(~isnan16(x) & (order16(x) < 0), np.uint16(0), x).astype(np.uint16)


def seq_sum(words, axis: int = -1) -> np.ndarray:
    """f32 words summed along ``axis`` in ascending index from +0, one RN add at a time."""
    w = np.moveaxis(u32(words), axis, -1)
    acc = np.zeros(w.shape[:-1], np.uint32)
    for j in range(w.shape[-1]):
        acc = fadd(acc, w[..., j])
    return acc
