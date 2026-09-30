"""Inputs for the `ncp-v2` circuit tests: a seeded call at a small shape."""
from __future__ import annotations

import random
import struct

from verity_pouw.circuit import reference as R


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16(x: float) -> int:
    return f32(x) >> 16


def call(m: int, k: int, n: int, seed: int = 7, *, rowids=None, colids=None, a=None) -> R.Call:
    rng = random.Random(seed)
    if a is None:
        a = tuple(tuple(rng.randint(-64, 63) for _ in range(k)) for _ in range(m))
    b = tuple(tuple(rng.randint(-64, 63) for _ in range(k)) for _ in range(n))
    return R.Call(a, tuple(f32(rng.uniform(1e-3, 2e-2)) for _ in range(m)), b,
                  tuple(f32(rng.uniform(1e-3, 2e-2)) for _ in range(n)), tuple(bf16(rng.uniform(-1, 1)) for _ in range(n)),
                  bytes(rng.randrange(256) for _ in range(32)), forward=3, layer=5, site=2, weight=11,
                  rowids=tuple(range(1000, 1000 + m)) if rowids is None else tuple(rowids),
                  colids=tuple(range(n)) if colids is None else tuple(colids))
