"""`lut`'s one-word cancellation: two of its tables' differences match under the benchmark key, and the word difference
they give leaves the whole block's receipt unchanged for a word in its 2^-20 set and changes it for a uniform word."""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import lut_trails as LT  # noqa: E402
import twins  # noqa: E402


def test_two_tables_cancel_inside_one_word():
    k = twins.keys(0xC0FFEE11)
    d0, a0, d1, b0, matches = LT.find_match(k.lt)
    assert matches > 256
    dw = d0 | (d1 << 11)
    rng = random.Random(3)
    p = [rng.getrandbits(32) for _ in range(LT.B2)]
    q = list(p)
    q[40] ^= dw
    assert LT.receipt(p, k) != LT.receipt(q, k)
    p[40] = (p[40] & ~0x3FFFFF) | (a0 ^ d0) | (b0 << 11)
    q = list(p)
    q[40] ^= dw
    assert LT.receipt(p, k) == LT.receipt(q, k)
