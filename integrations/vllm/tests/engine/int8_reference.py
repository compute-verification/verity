"""NumPy reference of the exact-int8 plain arm, written from `benchmarks/pouw/edited_gemm/egemm_ref.py` and not from the torch
code it checks: the plain quantizer is egemm_ref.py's `Xq = np.clip(np.rint(f), -127, 127)` with `f = x * inv_s` in float32,
the product exact in int64, then `engine.int8_exact`'s stated scale and dequantization rule.  Every float operation is a numpy
float32 operation (IEEE, round to nearest even); bf16 values travel as their uint16 bit patterns."""
from __future__ import annotations

import numpy as np


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def scales(x: np.ndarray) -> tuple[np.float32, np.float32]:
    """(inv_s, s_x) of float32 activations: 127 / amax and amax / 127, each one float32 division; 1 and 1 when amax < 2^-120."""
    amax = np.float32(np.abs(x).max())
    if amax < np.float32(2.0 ** -120):
        return np.float32(1.0), np.float32(1.0)
    return np.float32(127.0) / amax, amax / np.float32(127.0)


def quantize(x: np.ndarray, inv_s: np.float32) -> np.ndarray:
    f = x * inv_s
    assert f.dtype == np.float32
    return np.clip(np.rint(f), -127, 127).astype(np.int64)


def plain(x_bits: np.ndarray, w: np.ndarray, s_w: np.ndarray, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The arm on x [M, K] (bf16 bits), W [N, K] (int8) and s_w [N] (float32): every intermediate, y as bf16 bits.  `rows` keeps
    only those rows of xq, y32 and y; the scales are still the whole call's."""
    x = bf16_to_f32(x_bits)
    inv_s, s_x = scales(x)
    xq = quantize(x if rows is None else x[rows], inv_s)
    y = xq @ np.asarray(w, dtype=np.int64).T
    assert np.abs(y).max(initial=0) < 2 ** 31
    c = s_x * np.asarray(s_w, dtype=np.float32)
    out = y.astype(np.float32) * c
    assert c.dtype == np.float32 and out.dtype == np.float32
    return {"inv_s": inv_s, "s_x": s_x, "xq": xq, "y32": y, "y": f32_to_bf16(out)}
