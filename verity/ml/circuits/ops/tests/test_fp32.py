"""IEEE binary32 round-to-nearest-even on words (`verity.ml.circuits.silicon.fp32`): bit for bit against numpy's float32 (an IEEE
implementation without flush-to-zero) for add, mul, div and sqrt, and against an exact rational rounding for fma."""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

from verity.ml.circuits.silicon import fp32
from verity.ml.circuits.silicon.errors import InvalidArtifact

INF = 0x7F80_0000


def word(rng: random.Random) -> int:
    kind = rng.random()
    sign = rng.getrandbits(1) << 31
    if kind < 0.2:
        return sign | rng.getrandbits(23)                                       # subnormal
    if kind < 0.25:
        return sign                                                             # a signed zero
    if kind < 0.45:
        return sign | (rng.randint(110, 140) << 23) | rng.getrandbits(23)       # near 1, where sums cancel
    return sign | (rng.randint(1, 254) << 23) | rng.getrandbits(23)


def as_np(w: int) -> np.float32:
    return np.array([w], dtype=np.uint32).view(np.float32)[0]


def as_word(x: np.float32) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def exact(w: int) -> Fraction:
    neg, m, e = fp32.decode(w)
    v = Fraction(m) * Fraction(2) ** e
    return -v if neg else v


def rounded(v: Fraction, negative_zero: bool) -> int:
    """An independent RN-even rounding of an exact rational to binary32, by exhaustive neighbour comparison."""
    if v == 0:
        return fp32.NEG_ZERO if negative_zero else 0
    sign = fp32.NEG_ZERO if v < 0 else 0
    a = abs(v)
    e = max(a.numerator.bit_length() - a.denominator.bit_length() - 1, -126)
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    ulp = Fraction(2) ** (max(e, -126) - 23)
    q = a / ulp
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1):
        n += 1
    value = n * ulp
    if value >= Fraction(2) ** 128:
        return sign | INF
    return sign | as_word(np.float32(float(value)))


@pytest.mark.parametrize("op", ["add", "mul", "div", "sqrt"])
def test_matches_numpy_float32(op):
    rng = random.Random(f"fp32/{op}")
    with np.errstate(all="ignore"):
        for _ in range(40_000):
            a, b = word(rng), word(rng)
            if op == "add":
                got, want = fp32.add(a, b), as_word(as_np(a) + as_np(b))
            elif op == "mul":
                got, want = fp32.mul(a, b), as_word(as_np(a) * as_np(b))
            elif op == "div":
                if not b & 0x7FFF_FFFF:
                    continue
                got, want = fp32.div(a, b), as_word(as_np(a) / as_np(b))
            else:
                a &= 0x7FFF_FFFF
                got, want = fp32.sqrt(a), as_word(np.sqrt(as_np(a)))
            assert got == want, (op, hex(a), hex(b), hex(got), hex(want))


def test_fma_rounds_once():
    rng = random.Random("fp32/fma")
    for _ in range(20_000):
        a, b, c = word(rng), word(rng), word(rng)
        p = exact(a) * exact(b)
        neg_zero = (a >> 31 != b >> 31) and c >> 31 == 1 and p == 0 and exact(c) == 0
        assert fp32.fma(a, b, c) == rounded(p + exact(c), neg_zero), (hex(a), hex(b), hex(c))


def test_signed_zeros_and_overflow():
    one, neg_one = 0x3F80_0000, 0xBF80_0000
    assert fp32.add(one, neg_one) == 0                                          # x + (−x) is +0
    assert fp32.add(fp32.NEG_ZERO, fp32.NEG_ZERO) == fp32.NEG_ZERO
    assert fp32.add(fp32.NEG_ZERO, 0) == 0
    assert fp32.mul(neg_one, 0) == fp32.NEG_ZERO
    assert fp32.fma(neg_one, 0, fp32.NEG_ZERO) == fp32.NEG_ZERO
    assert fp32.fma(one, one, neg_one) == 0
    assert fp32.sqrt(fp32.NEG_ZERO) == fp32.NEG_ZERO
    big = 0x7F7F_FFFF
    assert fp32.add(big, big) == INF and fp32.mul(fp32.neg(big), 0x4000_0000) == fp32.NEG_ZERO | INF
    assert fp32.neg(one) == neg_one


def test_non_finite_and_undefined_raise():
    for bad in (INF, 0x7FC0_0000):
        with pytest.raises(InvalidArtifact):
            fp32.add(bad, 0)
    with pytest.raises(InvalidArtifact):
        fp32.div(0x3F80_0000, 0)
    with pytest.raises(InvalidArtifact):
        fp32.sqrt(0xBF80_0000)
