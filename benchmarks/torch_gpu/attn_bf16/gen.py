"""attn_bf16's full-scale inputs, made where they're used: every bf16 word is a function of (seed, array, flat index) through
splitmix64's finalizer in wrapping uint64 arithmetic, so fixture.py on a CPU node and run.py beside the card build the same
arrays from numpy alone, and the run carries per-sequence digests instead of the arrays.

Q, K and V words are a random sign, an exponent in 2^-4 .. 2^3 and a random mantissa; P's are positive, below 1.  One word in
4,096, anywhere, is a special (zeros, infinities, NaNs, subnormals, the extremes) instead.  Every sequence's lens is
1 + b mod Lcap and its one query is its last position (a decode step).
"""

from __future__ import annotations

import hashlib

import numpy as np

SPECIALS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFFF, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F,
                     0xFF7F, 0x3F80, 0xBF80], np.uint16)
TAGS = {"Q": 1, "K": 2, "V": 3, "P": 4}
MASK = (1 << 64) - 1
CHUNK = 1 << 22


def _mix(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def words(seed: int, tag: str, start: int, stop: int) -> np.ndarray:
    """The words at flat indices start..stop of array ``tag``."""
    key = np.uint64((seed * 0x9E3779B97F4A7C15 + TAGS[tag] * 0xD1B54A32D192ED03) & MASK)
    h = _mix(np.arange(start, stop, dtype=np.uint64) + key)
    exp = ((h >> np.uint64(32)) & np.uint64(7)) + np.uint64(119 if tag == "P" else 123)
    sign = np.uint64(0) if tag == "P" else (h >> np.uint64(63)) << np.uint64(15)
    w = (sign | exp << np.uint64(7) | (h & np.uint64(0x7F))).astype(np.uint16)
    special = ((h >> np.uint64(40)) & np.uint64(0xFFF)) == np.uint64(0)
    w[special] = SPECIALS[((h[special] >> np.uint64(52)) % np.uint64(len(SPECIALS))).astype(np.int64)]
    return w


def array(seed: int, tag: str, shape: tuple, b0: int = 0, b1: int | None = None) -> np.ndarray:
    """Sequences b0..b1 of the [B, ...] array ``tag`` of ``shape``."""
    b1 = shape[0] if b1 is None else b1
    per = int(np.prod(shape[1:]))
    out = np.empty((b1 - b0) * per, np.uint16)
    for i in range(0, out.size, CHUNK):
        out[i:i + CHUNK] = words(seed, tag, b0 * per + i, b0 * per + min(out.size, i + CHUNK))
    return out.reshape((b1 - b0, *shape[1:]))


def shapes(B: int, H: int, Hkv: int, D: int, Lcap: int) -> dict:
    return {"Q": (B, H, 1, D), "K": (B, Hkv, Lcap, D), "V": (B, Hkv, Lcap, D), "P": (B, H, 1, Lcap)}


def lens_nq(B: int, Lcap: int, b0: int = 0, b1: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    b = np.arange(b0, B if b1 is None else b1)
    return (1 + b % Lcap).astype(np.int32), np.ones(len(b), np.int32)


def digests(a: np.ndarray) -> np.ndarray:
    """Each sequence's (first axis) SHA-512, its first 16 bytes: (B, 16) uint8."""
    return np.frombuffer(b"".join(hashlib.sha512(np.ascontiguousarray(x).tobytes()).digest()[:16] for x in a), np.uint8).reshape(len(a), 16)
