"""The scalar primitives of `verity.ml.circuits.ops.scalar`: registered ids, and the comparisons and instruction models against host
IEEE comparisons (Python floats, which hold every f32 value exactly) on special and random encodings."""
from __future__ import annotations

import itertools
import math
import random
import struct

from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits.ops import scalar as S

NAN32, PINF32, NINF32, ONE32, NEG0 = 0x7FC00000, 0x7F800000, 0xFF800000, 0x3F800000, 0x80000000
SPECIAL32 = [0, NEG0, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, ONE32, 0xBF800000, 0x3F7FFFFF,
             0x3F800001, 0x7F7FFFFF, 0xFF7FFFFF, PINF32, NINF32, NAN32, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, 0x4B400001]


def _f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _words(rng: random.Random, n: int) -> list[int]:
    return SPECIAL32 + [rng.getrandbits(32) for _ in range(n)]


def test_registered_ids() -> None:
    names = ["F32Fabs", "F32Neg", "F32Fmaxf", "F32Fminf", "F32Sat", "F32BitsShl23", "F32IsFinite", "F32Eq", "Bf16GtStrict",
             "I32Le", "I32Eq", "I32Add", "BitAnd", "BitNot", "BitOr", "SelectF32", "SelectBf16", "SelectI32"]
    for n in names:
        d = getattr(S, n)
        assert d.id == f"{n}_v1" and REGISTRY[d.id] is d
    assert S.Bf16GtStrict.signature() == ((("a", S.BF16), ("b", S.BF16)), S.BIT)
    assert S.SelectF32.signature() == ((("c", S.BIT), ("a", S.F32), ("b", S.F32)), S.F32)


def test_comparisons_are_ieee() -> None:
    rng = random.Random(2)
    ws = _words(rng, 300)
    for a, b in itertools.product(ws, ws):
        assert S.F32Eq.evaluate(a, b) == int(_f(a) == _f(b)), (hex(a), hex(b))
    for a in range(0, 1 << 16, 97):
        for b in list(range(0, 1 << 16, 1021)) + [0x7FC0, 0xFF80, 0x8000, 0x0000, 0x0001, 0x8001]:
            assert S.Bf16GtStrict.evaluate(a, b) == int(_f(a << 16) > _f(b << 16)), (hex(a), hex(b))


def test_fmaxf_fminf() -> None:
    rng = random.Random(4)
    ws = _words(rng, 300)
    for a, b in itertools.product(ws, ws):
        fa, fb = _f(a), _f(b)
        mx, mn = S.F32Fmaxf.evaluate(a, b), S.F32Fminf.evaluate(a, b)
        if math.isnan(fa) and math.isnan(fb):
            assert mx == mn == 0x7FFFFFFF
        elif math.isnan(fa) or math.isnan(fb):
            assert mx == mn == (b if math.isnan(fa) else a)
        elif fa == fb == 0:
            assert (mx, mn) == ((0 if 0 in (a, b) else NEG0), (NEG0 if NEG0 in (a, b) else 0))
        else:
            assert _f(mx) == max(fa, fb) and _f(mn) == min(fa, fb), (hex(a), hex(b))
            assert mx in (a, b) and mn in (a, b)


def test_bit_level_instructions() -> None:
    rng = random.Random(6)
    for x in _words(rng, 5000):
        assert S.F32Fabs.evaluate(x) == x & 0x7FFFFFFF
        assert S.F32Neg.evaluate(x) == x ^ 0x80000000
        fx = _f(x)
        assert S.F32IsFinite.evaluate(x) == int(math.isfinite(fx))
        sat = S.F32Sat.evaluate(x)
        assert sat == (0 if math.isnan(fx) or fx <= 0 else ONE32 if fx >= 1 else x), hex(x)
        assert S.F32BitsShl23.evaluate(x) == (x << 23) & 0xFFFFFFFF
    for n in range(1, 254):
        assert _f(S.F32BitsShl23.evaluate(0x4B400000 | n)) == 2.0 ** (n - 127)


def test_integer_bit_and_select() -> None:
    rng = random.Random(8)
    ints = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(200)]
    signed = lambda u: u - (1 << 32) if u >> 31 else u  # noqa: E731
    for a, b in itertools.product(ints, ints):
        assert S.I32Le.evaluate(a, b) == int(signed(a) <= signed(b))
        assert S.I32Eq.evaluate(a, b) == int(a == b)
        assert S.I32Add.evaluate(a, b) == (a + b) % (1 << 32)
    for a, b in itertools.product((0, 1), (0, 1)):
        assert (S.BitAnd.evaluate(a, b), S.BitOr.evaluate(a, b), S.BitNot.evaluate(a)) == (a & b, a | b, 1 - a)
    for c in (0, 1):
        for sel, w in ((S.SelectF32, 32), (S.SelectBf16, 16), (S.SelectI32, 32)):
            a, b = rng.getrandbits(w), rng.getrandbits(w)
            assert sel.evaluate(c, a, b) == (a if c else b)
