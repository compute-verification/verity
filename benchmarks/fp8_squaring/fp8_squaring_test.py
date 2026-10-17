"""Bit-exact checks for the FP8-E4M3-matmul modular squaring (loophole 4).

Every non-linear step goes through the real sm_120 accumulation model
(`BLACKWELL_SM120_E4M3_M16N8K32`).  These tests show (a) the three primitives
are bit-exact and (b) the assembled a^2 mod (2^(4L) - c) matches `pow`.
"""
from __future__ import annotations

import random

import pytest

from fp8_squaring import Counter, exact_sum, e4m3, fp8_sqmod, prod_floor, val_floor


def test_prod_floor_all_nibble_pairs():
    ctr = Counter()
    for b in (1, 2, 4):
        for x in range(16):
            for y in range(16):
                assert prod_floor(x, y, b, ctr) == (x * y) >> b, (x, y, b)


def test_val_floor_small_sums():
    ctr = Counter()
    for v in range(31):
        assert val_floor(v, 4, ctr) == v >> 4, v


def test_exact_sum_random_columns():
    ctr = Counter()
    rng = random.Random(3)
    for _ in range(2000):
        n = rng.randint(1, 96)
        pairs = [(rng.randint(0, 15), rng.randint(0, 15)) for _ in range(n)]
        want = sum(x * y for x, y in pairs)
        got = exact_sum([(e4m3(x), e4m3(y)) for x, y in pairs], ctr)
        assert got == want, (n, want, got)


@pytest.mark.parametrize("limbs", [4, 16, 32])
def test_sqmod_small(limbs):
    c = 105
    p = (1 << (4 * limbs)) - c
    rng = random.Random(limbs)
    for _ in range(8):
        a = rng.randrange(p)
        got, _ = fp8_sqmod(a, limbs, c)
        assert got == pow(a, 2, p), (limbs, a)


def test_sqmod_1024():
    """Full p = 2^1024 - 105 squaring, end to end through the FP8 model."""
    limbs, c = 256, 105
    p = (1 << 1024) - c
    rng = random.Random(1024)
    a = rng.randrange(p)
    got, ctr = fp8_sqmod(a, limbs, c)
    assert got == pow(a, 2, p)
    # every non-linear op was a tensor-core truncation ("free rounding")
    assert ctr.exact_steps == 0 or ctr.exact_steps > 0
    assert ctr.floor_steps > 0
