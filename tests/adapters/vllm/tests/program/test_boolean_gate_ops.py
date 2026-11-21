"""`registry.boolean_gate_ops`' word primitives and their Boolean versions: the low byte of `cvt.rzi.u32.f32` on its edges and
against the exact truncation, the UE8M0 byte as the exponent rounded up, C `int` division, remainder and the low product, and
each Boolean version against its word; MUFU.LG2 of 1.m where measured, and on bits on every mantissa against the measured
words' SHA-256."""
from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits.boolean.trace import definition, word_of
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.evaluation.reference import standalone_call
from verity.circuits.ops.boolean import versions as SW
from verity.circuits.types import Array

from tests.program.test_boolean_moe import _agree
from tests.program.test_dsa import _is_nan
from tests.program.test_fp8_moe import _q
from verity_vllm.definitions.registry import boolean_gate_ops as BG
from verity_vllm.definitions.registry import prims as P

SIGN, INF, M32 = 1 << 31, 0x7F800000, 0xFFFFFFFF
NAN = 0x7FC00000
ONE = 0x3F800000
I32_EDGES = [0, 1, 2, 3, 7, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFF9]

# --- the scale byte --------------------------------------------------------------------------------------------------------------

BYTE_EDGES = [0, SIGN, 1, 0x007FFFFF, 0x3F7FFFFF, 0x3F800000, 0x3FC00000, 0x437F0000, 0x437FFFFF, 0x43800000, 0x43808000,
              0x4B7FFFFF, 0x4B800000, 0x4F7FFFFF, 0x4F800000, 0x7F7FFFFF, INF, SIGN | INF, NAN, 0xFFFFFFFF, 0x7F800001, 0xBF800000,
              0xC3000000, 0x42FE0000, 0x437F0000]


def _byte(w: int) -> int:
    """The low byte of cvt.rzi.u32.f32 from the exact value: NaN and every negative to 0, past 2^32 - 1 to 0xFF."""
    if _is_nan(w) or w & SIGN:
        return 0
    if w == INF:
        return 0xFF
    return min(math.floor(_q(w)), M32) & 0xFF


def _byte_words(rng) -> list[int]:
    """The edges, every (sign, exponent) at four mantissas, and random words."""
    out = list(BYTE_EDGES)
    for se in range(512):
        out += [se << 23 | m for m in (0, 1, 0x400000, 0x7FFFFF)]
    return out + rng.integers(0, 1 << 32, 4000, dtype=np.uint64).astype(int).tolist()


def test_the_scale_byte_on_its_edges():
    assert [BG.F32ToU32RzByte.evaluate(w) for w in BYTE_EDGES[:16]] == \
        [0, 0, 0, 0, 0, 1, 1, 255, 255, 0, 1, 0xFF, 0, 0, 0xFF, 0xFF]
    assert [BG.F32ToU32RzByte.evaluate(w) for w in BYTE_EDGES[16:]] == [0xFF, 0, 0, 0, 0, 0, 0, 127, 255]


def test_the_boolean_scale_byte_is_its_word():
    target = bind(BG.F32ToU32RzByteV2)
    word = word_of(target)
    assert word.id == BG.F32ToU32RzByte.id and SW.boolean_version(word).id == target.id
    x: np.ndarray = np.asarray(_byte_words(np.random.default_rng(14)), dtype=np.uint64)
    got = np.asarray(BS.evaluate_words(target, x), dtype=np.uint64).reshape(-1)
    assert got.tolist() == [_byte(int(w)) for w in x]


# --- the UE8M0 byte --------------------------------------------------------------------------------------------------------------

def _bits(v: float) -> int:
    return int(np.float32(v).view(np.uint32))


def test_the_ue8m0_byte_is_the_exponent_rounded_up():
    """For every positive normal scale s, `Ue8m0CeilByte_v1` is ceil(log2 s) + 127 (powers of two exactly, every other word one
    up), 255 past 2^127 and for +inf; the floor 1e-4 gives 114."""
    rng = np.random.default_rng(0)
    s = np.concatenate([2.0 ** np.arange(-126, 128), np.nextafter(2.0 ** np.arange(-126, 127), np.inf, dtype=np.float32),
                        rng.uniform(0.5, 1.0, 4000) * 2.0 ** rng.integers(-125, 129, 4000)]).astype(np.float32)
    for v in s[np.isfinite(s)]:
        m, e = math.frexp(float(v))
        assert BG.Ue8m0CeilByte.evaluate(_bits(v)) == min((e if m > 0.5 else e - 1) + 127, 255), float(v)
    assert [BG.Ue8m0CeilByte.evaluate(w) for w in (0x38D1B717, ONE, 0x7F7FFFFF, INF)] == [114, 127, 255, 255]


def test_the_boolean_ue8m0_byte_is_its_word():
    rng = np.random.default_rng(7)
    special = [0, SIGN, INF, 0xFF800000, NAN, 0x7FFFFFFF, 0xFFFFFFFF, 1, 0x80000001, 0x007FFFFF, 0x00800000, ONE, 0xBF800000,
               0x38D1B717, 0x7F000000, 0x7F7FFFFF, 0xFF7FFFFF]
    s = np.concatenate([np.asarray(special, np.uint64), rng.integers(0, 1 << 32, 1000, dtype=np.uint64)])
    target = bind(BG.Ue8m0CeilByteV2)
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    got = np.asarray(BS.evaluate_words(target, s), dtype=np.uint64).reshape(len(s), -1)
    want = np.asarray(evaluate_batch(word, s, kernel="reference"), dtype=np.uint64).reshape(len(s), -1)
    assert np.array_equal(got, want)


# --- C int arithmetic ------------------------------------------------------------------------------------------------------------

def test_divrem_is_c_int() -> None:
    for a, b in itertools.product([0, 7, -7, 2**31 - 1, -2**31], [1, 3, -3, 2**31 - 1, -2**31, -1, 0]):
        q, r = BG.i32_divrem(a & BG.M32, b & BG.M32)
        if b == 0:
            assert (q, r) == (BG.M32, a & BG.M32)
            continue
        want_q = int(a / b) if abs(a) < 2**53 else (abs(a) // abs(b)) * (1 if (a < 0) == (b < 0) else -1)
        assert q == want_q & BG.M32 and r == (a - want_q * b) & BG.M32, (a, b)


def test_divider_is_c_int_on_edge_and_random_words() -> None:
    probe = definition("DivRemProbe", 1, lambda C, a, b: list(BG.divrem(C, a, b)), (("a", BG.I32_BITS), ("b", BG.I32_BITS)),
                       Array(2, BG.I32_BITS), register=False)
    rng = np.random.default_rng(7)
    pairs = list(itertools.product(I32_EDGES, I32_EDGES))
    pairs += [(int(a), int(b)) for a, b in zip(rng.integers(0, 1 << 32, 200, dtype=np.uint64), rng.integers(0, 1 << 32, 200, dtype=np.uint64))]
    pairs += [(int(a), int(b)) for a, b in zip(rng.integers(0, 5000, 200), rng.integers(1, 300, 200))]
    a, b = (np.array(c, dtype=np.uint32) for c in zip(*pairs))
    got = np.asarray(BS.evaluate_words(bind(probe), a, b)).reshape(len(pairs), 2)
    assert [tuple(int(v) for v in r) for r in got] == [BG.i32_divrem(x, y) for x, y in pairs]


def test_the_multiplier_is_mul_lo_on_edge_and_random_words() -> None:
    assert SW.boolean_version(BG.I32Mul).defn is BG.I32MulV2
    rng = np.random.default_rng(11)
    pairs = list(itertools.product(I32_EDGES, I32_EDGES))
    pairs += [(int(a), int(b)) for a, b in zip(rng.integers(0, 1 << 32, 500, dtype=np.uint64), rng.integers(0, 1 << 32, 500, dtype=np.uint64))]
    a, b = (np.array(c, dtype=np.uint64).reshape(-1, 1) for c in zip(*pairs))
    got = np.asarray(BS.evaluate_words(bind(BG.I32MulV2), a, b)).reshape(-1)
    assert got.tolist() == [BG.I32Mul.evaluate(x, y) for x, y in pairs]


# --- MUFU.LG2 on [1, 2) ---------------------------------------------------------------------------------------------------------

#: `lg2.approx.ftz.f32(1.m)` as `LG2_MEASURED`'s run read it, at segment boundaries and both ends
MEASURED = {0x1: 0x3495F600, 0x2: 0x34F24A00, 0x3: 0x35274F00, 0x1FFFF: 0x3CB73C76, 0x20000: 0x3CB73CD1, 0x20001: 0x3CB73D2C,
            0xA0000: 0x3DDE4224, 0x3FFFFF: 0x3F15C019, 0x400000: 0x3F15C01B, 0x7E3039: 0x3F7D6088, 0x7FFFFE: 0x3F7FFFFE,
            0x7FFFFF: 0x3F7FFFFF}


def test_lg2_is_the_measured_unit_where_pinned():
    for m, y in MEASURED.items():
        assert BG.MufuLg2Mantissa.evaluate(0x3F800000 | m) == y, hex(m)
    assert BG.MufuLg2Mantissa.evaluate(0x3F800000) == 0


def test_lg2_reads_only_the_mantissa_and_stays_near_log2():
    for m in [*MEASURED, 0x155555, 0x2AAAAA, 0x6DB6DB]:
        y = BG.MufuLg2Mantissa.evaluate(0x3F800000 | m)
        assert all(BG.MufuLg2Mantissa.evaluate(e << 23 | s << 31 | m) == y for e in (0, 1, 0x80, 0xFF) for s in (0, 1))
        assert abs(float(P.f32_of(y)) - math.log2(1 + m / 2 ** 23)) < 2 ** -21


#: (And, Xor, Not gates, standalone program digest)
LG2_PIN = (986, 4749, 92, "661ff9649d6e2fd4720a6168c02df12165d60caf36009433d919ca127311ab0e4ad3c1fb557a319eafe9b72f6c3e216cfbdbbdaa69f75171e7c6af211ad8e7bf")


def test_the_boolean_lg2_is_pinned_by_counts_and_digest() -> None:
    target = bind(BG.MufuLg2MantissaV2)
    ands, xors, nots, digest = LG2_PIN
    assert target.id == "MufuLg2Mantissa_v2" and BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


def test_the_lg2_word_view() -> None:
    assert word_of(bind(BG.MufuLg2MantissaV2)).id == BG.MufuLg2Mantissa.id == "MufuLg2Mantissa_v1"
    assert SW.boolean_version(BG.MufuLg2Mantissa).id == "MufuLg2Mantissa_v2"


def test_lg2_where_measured_and_on_every_exponent_and_sign() -> None:
    rng = np.random.default_rng(0xD54)
    m = np.concatenate([np.array([0, 1, 2, 3, 0x1FFFF, 0x20000, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF], np.uint32),
                        rng.integers(0, 1 << 23, 2000, dtype=np.uint32)])
    x = m | (rng.integers(0, 1 << 9, len(m), dtype=np.uint32) << 23)
    _agree(bind(BG.MufuLg2MantissaV2), x)


def test_lg2_on_every_mantissa_is_the_measured_unit() -> None:
    """All 2^23 words of [1, 2) against `LG2_MEASURED`'s SHA-256 (about 10 s): the Boolean interpolator is the measured unit, not
    only its word view."""
    h = hashlib.sha256()
    for lo in range(0, 1 << 23, 1 << 19):
        m: np.ndarray = np.arange(lo, lo + (1 << 19), dtype=np.uint32)
        y = np.asarray(BS.evaluate_words(bind(BG.MufuLg2MantissaV2), m | np.uint32(0x3F800000))).astype("<u4").reshape(-1)
        assert np.array_equal((y >> 23 & 0xFF) == 0, m == 0)
        h.update(y.tobytes())
    assert h.hexdigest() == BG.LG2_MEASURED["sha256"]
