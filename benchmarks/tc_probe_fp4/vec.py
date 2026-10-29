"""Vectorised forms of `verity.ml.circuits.silicon`'s NVF4 atom (``BLACKWELL_SM120_NVF4``) for f2_strassen.py and f3_lut.py,
which run it over whole tiles and chains:

    ue4m3_dyadic   UE4M3 scale codes as ``(mantissa, exponent)`` (``models._scale_dyadic("ue4m3", code)``)
    nvf4_step      one firing per word on its group terms (``BlockScaledAlignAdd.align_add``)

tests/test_tc_probe_fp4.py holds each to silicon's scalar definition bit for bit: every UE4M3 code, and the step on random
firings and on the edges (zero, signed-zero and subnormal accumulators, no participating group, cancelling groups).
"""
from __future__ import annotations

import numpy as np

NEG_INF = -(1 << 40)


def ue4m3_dyadic(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UE4M3 codes as ``(mantissa, exponent)`` with value ``mantissa * 2^exponent``; bit 7 is ignored, as sm_120 ignores it.
    The NaN codes 0x7F and 0xFF, which silicon refuses, decode as if finite."""

    e, m = (code >> 3) & 0xF, code & 7
    return np.where(e == 0, m, 8 + m), np.where(e == 0, -9, e - 10)


def floor_log2(x: np.ndarray) -> np.ndarray:
    """``floor(log2 x)`` for positive int64 x, exact (float64 frexp can round a 54+-bit value up to a power of two)."""

    bl = np.frexp(x.astype(np.float64))[1].astype(np.int64)
    bl = np.where((x >> np.clip(bl - 1, 0, 62)) == 0, bl - 1, bl)
    return bl - 1


def trunc0(mant: np.ndarray, exp: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    """``mant * 2^exp`` truncated toward zero onto the grid ``2^lsb``, as a multiple of ``2^lsb``."""

    d = exp - lsb
    left = mant << np.clip(d, 0, 62)
    right = np.abs(mant) >> np.clip(-d, 0, 62)
    right = np.where(-d > 62, 0, right)
    return np.where(d >= 0, left, np.where(mant < 0, -right, right))


def f32_rz(n: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    """RZ binary32 bits of ``n * 2^lsb`` (the model's ``_to_f32_rz``); zero is +0, underflow to zero is +0."""

    neg = n < 0
    mag = np.abs(n)
    nz = mag != 0
    e_val = floor_log2(np.where(nz, mag, 1)) + lsb
    grid = np.where(e_val >= -126, e_val - 23, -149)
    shift = grid - lsb
    q = np.where(shift > 0, mag >> np.clip(shift, 0, 62), mag << np.clip(-shift, 0, 62))
    q = np.where(shift > 62, 0, q)
    e = floor_log2(np.where(q > 0, q, 1)) + grid
    if np.any(nz & (q > 0) & (e > 127)):
        raise ValueError("sum leaves binary32 (outside the model)")
    bits = np.where(e >= -126, ((e + 127) << 23) | (q & 0x7FFFFF), q)
    bits = np.where(neg, bits | (1 << 31), bits)
    return np.where(nz & (q > 0), bits, 0).astype(np.uint32)


def f32_dyadic(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """binary32 bits as ``(signed mantissa, exponent, exponent field)``."""

    c = c.astype(np.int64)
    ef = (c >> 23) & 0xFF
    frac = c & 0x7FFFFF
    m = np.where(ef == 0, frac, frac | (1 << 23))
    return np.where((c >> 31) & 1, -m, m), np.where(ef == 0, -149, ef - 150), ef


def nvf4_step(c: np.ndarray, mants: list, xs: list, parts: list) -> np.ndarray:
    """One ``mxf4nvf4`` 4X firing per word: the four scaled group dots ``mant * 2^X`` and the accumulator ``c``."""

    acc_m, acc_e, ef = f32_dyadic(c)
    anchor = np.full(c.shape, NEG_INF, np.int64)
    for x, p in zip(xs, parts):
        anchor = np.maximum(anchor, np.where(p, x - 27, NEG_INF))
    anchor = np.maximum(anchor, np.where(acc_m != 0, np.where(ef == 0, 1, ef) - 127 - 35, NEG_INF))
    has = anchor > NEG_INF
    lsb = np.maximum(anchor, -174)
    total = trunc0(acc_m, acc_e, lsb)
    for m, x in zip(mants, xs):
        total = total + trunc0(m, x, lsb)
    return np.where(has, f32_rz(total, lsb), 0).astype(np.uint32)
