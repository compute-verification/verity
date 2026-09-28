"""IEEE binary16 round-to-nearest-even on words (`verity.ml.tc.fp16`): bit for bit against numpy's float16 for add, mul
and max (numpy rounds each float16 op once through float32, which is exact for these), against an exact rational
rounding for fma, and the E4M3 conversions against the FP32 ones."""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

from verity.ml.tc import fp16
from verity.ml.tc.cast import f32_to_e4m3_sat_word
from verity.ml.tc.errors import InvalidArtifact


def word(rng: random.Random) -> int:
    sign = rng.getrandbits(1) << 15
    kind = rng.random()
    if kind < 0.2:
        return sign | rng.getrandbits(10)
    if kind < 0.25:
        return sign
    return sign | (rng.randint(1, 30) << 10) | rng.getrandbits(10)


def as_np(w: int) -> np.float16:
    return np.array([w], dtype=np.uint16).view(np.float16)[0]


def as_word(x) -> int:
    return int(np.array([x], dtype=np.float16).view(np.uint16)[0])


def exact(w: int) -> Fraction:
    neg, m, e = fp16.decode(w)
    v = Fraction(m) * Fraction(2) ** e
    return -v if neg else v


@pytest.mark.parametrize("op", ["add", "mul", "max_"])
def test_matches_numpy_float16(op):
    rng = random.Random(f"fp16/{op}")
    f = {"add": lambda x, y: x + y, "mul": lambda x, y: x * y, "max_": np.maximum}[op]
    with np.errstate(all="ignore"):
        for _ in range(40_000):
            a, b = word(rng), word(rng)
            got, want = getattr(fp16, op)(a, b), as_word(f(as_np(a), as_np(b)))
            if op == "max_" and exact(a) == exact(b) == 0:
                want = a & b                                         # max(+0, −0) is +0
            assert got == want, (op, hex(a), hex(b), hex(got), hex(want))


def test_fma_rounds_once():
    rng = random.Random("fp16/fma")
    for _ in range(20_000):
        a, b, c = word(rng), word(rng), word(rng)
        v = exact(a) * exact(b) + exact(c)
        got = fp16.fma(a, b, c)
        if v == 0:
            continue
        neg, m, e = fp16.decode(got) if (got >> 10) & 0x1F != 0x1F else (v < 0, None, None)
        if m is None:
            assert abs(v) >= Fraction(65520)
            continue
        # got is the nearest binary16 value, ties to even: no representable value lies strictly closer
        g, ulp = exact(got), Fraction(2) ** e
        assert abs(g - v) <= ulp / 2
        if abs(g - v) == ulp / 2:
            assert m % 2 == 0


def test_e4m3_round_trip_and_widening():
    for code in range(256):
        if code & 0x7F == 0x7F:
            with pytest.raises(InvalidArtifact):
                fp16.from_e4m3(code)
            continue
        h = fp16.from_e4m3(code)
        assert fp16.to_e4m3(h) == (code if code & 0x7F else code & 0x80)
    rng = random.Random("fp16/e4m3")
    for _ in range(20_000):
        w = word(rng)
        assert fp16.to_e4m3(w) == f32_to_e4m3_sat_word(fp16.to_f32(w))
        assert as_np(w).astype(np.float32).view(np.uint32) == fp16.to_f32(w)
