"""Numpy models of `cublas_f32`'s two reductions (exact f32 FMA and add, vectorized over rows and outputs), the probe inputs they are
checked on (a counter-based generator, so a CPU test rebuilds exactly what the GPU computed), and the probe's recorded output
digests. `probe_inputs` + `output_digest` are what the sm_120 probe runs; `tests/program/test_cublas_f32.py` replays them."""
from __future__ import annotations

import hashlib

import numpy as np

F32, F64, U32, U64 = np.float32, np.float64, np.uint32, np.uint64
K_GATE, N_GATE = 2048, 64
W_KINDS = ("bf16", "f32", "neg")


def _round_f32(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """RN to f32 of the exact s + e (s its f64 RN, e the remainder): f64 rounding can move s onto an f32 midpoint, never across
    one, so only a midpoint s with e != 0 needs e's sign to pick the side."""
    r = s.astype(F32)
    r64 = r.astype(F64)
    other = np.nextafter(r, np.where(s > r64, F32(np.inf), F32(-np.inf)).astype(F32))
    on_mid = (s != r64) & (s == (r64 + other.astype(F64)) / 2) & (e != 0)
    return np.where(on_mid, np.where(e > 0, np.maximum(r, other), np.minimum(r, other)), r).astype(F32)


def _two_sum(a: np.ndarray, b: np.ndarray):
    s = a + b
    bp = s - a
    return s, (a - (s - bp)) + (b - bp)


def fma32(a, b, c) -> np.ndarray:
    """RN(a * b + c) on f32 arrays (the product is exact in f64)."""
    return _round_f32(*_two_sum(a.astype(F64) * b.astype(F64), c.astype(F64)))


def add32(a, b) -> np.ndarray:
    return _round_f32(*_two_sum(a.astype(F64), b.astype(F64)))


def gemv_tree(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`CublasGemvDot` over x [R, K] (f32) and w [N, K] -> [R, N]."""
    R, K = x.shape
    shape = (R, w.shape[0])
    y = None
    for t in range(16):
        total = None
        for g in range(K // 256):
            part = np.zeros(shape, F32)
            for i in range(16):
                k = 256 * g + 16 * i + t
                part = fma32(np.broadcast_to(x[:, None, k], shape), np.broadcast_to(w[None, :, k], shape), part)
            total = part if total is None else add32(total, part)
        y = total if y is None else add32(y, total)
    return y


def simt_seq(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`CublasSimtDot` over x [R, K] and w [N, K] -> [R, N]."""
    R, K = x.shape
    shape = (R, w.shape[0])
    acc = np.zeros(shape, F32)
    for k in range(K):
        acc = fma32(np.broadcast_to(x[:, None, k], shape), np.broadcast_to(w[None, :, k], shape), acc)
    return acc


MODELS = {"gemv": gemv_tree, "simt": simt_seq}


def _words(seed: int, n: int) -> np.ndarray:
    """splitmix64 of the counters seed * 2^32 + [0, n): 64 uniform bits each."""
    with np.errstate(over="ignore"):
        z = (U64(seed) << U64(32)) + np.arange(n, dtype=U64) + U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def _floats(seed: int, shape, e_lo: int, e_hi: int, mant_bits: int, sign: int | None = None) -> np.ndarray:
    """f32 values with a uniform biased exponent in [e_lo, e_hi], `mant_bits` random leading mantissa bits and a random sign
    (or `sign`)."""
    z = _words(seed, int(np.prod(shape)))
    e = U64(e_lo) + z % U64(e_hi - e_lo + 1)
    m = (z >> U64(8)) & U64((1 << mant_bits) - 1)
    s = (z >> U64(40)) & U64(1) if sign is None else U64(sign)
    bits = (s << U64(31)) | (e << U64(23)) | (m << U64(23 - mant_bits))
    return bits.astype(U32).view(F32).reshape(shape)


def probe_weights(kind: str) -> np.ndarray:
    """[N_GATE, K_GATE] f32: `bf16` (bf16-exact, as the loaded checkpoint), `f32` (23 mantissa bits), `neg` (bf16, all negative)."""
    seed = {"bf16": 1, "f32": 2, "neg": 3}[kind]
    return _floats(seed, (N_GATE, K_GATE), 118, 122, 23 if kind == "f32" else 7, 1 if kind == "neg" else None)


def probe_rows(M: int) -> list[int]:
    return sorted({0, min(1, M - 1), min(2, M - 1), M // 2, M - 1})


def probe_inputs(M: int) -> np.ndarray:
    """x [M, K_GATE], bf16-exact f32 (the widened hidden): row 0 zeros (signed zeros against `neg`), row 1 magnitudes 2^-20..2^20."""
    x = _floats(1000 + M, (M, K_GATE), 123, 128, 7)
    x[0] = 0
    if M > 1:
        x[1] = _floats(500_000 + M, (K_GATE,), 107, 147, 7)
    return x


def probe_single(j: int) -> np.ndarray:
    """x [1, K_GATE] of the j-th M = 1 probe (`probe_inputs(1)` is its zero row alone): bf16-exact, odd j magnitudes 2^-20..2^20."""
    return _floats(2000 + j, (1, K_GATE), 107, 147, 7) if j % 2 else _floats(2000 + j, (1, K_GATE), 123, 128, 7)


def output_digest(y_rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(y_rows, dtype=F32).view(U32).astype("<u4").tobytes()).hexdigest()[:16]
