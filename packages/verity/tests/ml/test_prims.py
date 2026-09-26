"""The IR-registered primitives of `verity.ml.prims` are exactly the `verity.ml.tc` functions, under their registered ids."""
from __future__ import annotations

import random

import pytest

from verity.ir.defs import REGISTRY
from verity.ml import prims as P
from verity.ml.tc.cast import f32_to_bf16_hw_word, f32_to_bf16_word, f32_to_e4m3_sat_word
from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity.ml.tc.total import HOPPER_BF16_WGMMA_K16, tc_dot_total
from verity.ml.tc.total_fp8 import HOPPER_E4M3_WGMMA_K32, tc_dot_total_e4m3

NAN32, PINF32, NINF32, ONE32 = 0x7FC00000, 0x7F800000, 0xFF800000, 0x3F800000
NAN16, PINF16, NINF16, ONE16, MAX16 = 0x7FC0, 0x7F80, 0xFF80, 0x3F80, 0x7F7F


def _rand_finite_bf16(rng: random.Random) -> int:
    while True:
        w = rng.getrandbits(16)
        if (w & 0x7F80) != 0x7F80:
            return w


def _rand_finite_f32(rng: random.Random) -> int:
    while True:
        u = rng.getrandbits(32)
        if (u & 0x7F800000) != 0x7F800000:
            return u


def test_registered_ids_and_versions() -> None:
    ids = {d.id for d in (P.Bf16ToF32, P.F32ToBf16Rn, P.F2fpBf16, P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16)}
    assert ids == {"Bf16ToF32_v1", "F32ToBf16Rn_v1", "F2fpBf16_v1", "AmpereBF16TcDot16_v2", "HopperBF16WgmmaDot16_v1"}
    for i in ids:
        assert REGISTRY[i].conformance.startswith("exact-model-tested")
    assert P.AmpereBF16TcDot16.signature() == ((("acc", P.F32), ("a", P.A16), ("b", P.A16)), P.F32)
    assert P.HopperBF16WgmmaDot16.signature() == P.AmpereBF16TcDot16.signature()


def test_casts_are_the_cast_functions() -> None:
    rng = random.Random(1)
    for x in [NAN32, PINF32, NINF32, ONE32, 0, 0x80000000, 0x7F7FFFFF] + [rng.getrandbits(32) for _ in range(5000)]:
        assert P.F2fpBf16.evaluate(x) == f32_to_bf16_hw_word(x)
        assert P.F32ToBf16Rn.evaluate(x) == f32_to_bf16_word(x)
    for w in range(1 << 16):
        assert P.Bf16ToF32.evaluate(w) == w << 16


@pytest.mark.parametrize("prim, pipe", [(P.AmpereBF16TcDot16, AMPERE_BF16_M16N8K16), (P.HopperBF16WgmmaDot16, HOPPER_BF16_WGMMA_K16)],
                         ids=["ampere", "hopper"])
def test_steps_are_tc_dot_total_on_random_and_special_operands(prim, pipe) -> None:
    rng = random.Random(7)
    cases = []
    for _ in range(400):
        cases.append((_rand_finite_f32(rng), [_rand_finite_bf16(rng) for _ in range(16)], [_rand_finite_bf16(rng) for _ in range(16)]))
    specials16 = [NAN16, PINF16, NINF16, ONE16, MAX16, 0, 0x8000, 0x0001, 0x0080]
    for acc in (0, 0x80000000, ONE32, NAN32, PINF32, NINF32, 0x7F7FFFFF):
        for sa in specials16:
            for sb in specials16:
                a = [ONE16] * 16
                b = [ONE16] * 16
                a[3], b[3] = sa, sb
                a[11] = rng.choice(specials16)
                cases.append((acc, a, b))
    for acc, a, b in cases:
        assert prim.evaluate(acc, *a, *b) == tc_dot_total(pipe, acc, a, b)


def test_ampere_v2_agrees_with_the_finite_model_where_the_finite_model_is_defined() -> None:
    """On finite encodings the total step is the finite-domain step (the `_v1` evaluator) word for word."""
    rng = random.Random(11)
    for _ in range(300):
        acc = _rand_finite_f32(rng)
        a = [_rand_finite_bf16(rng) for _ in range(16)]
        b = [_rand_finite_bf16(rng) for _ in range(16)]
        assert P.AmpereBF16TcDot16.evaluate(acc, *a, *b) == tc_dot(AMPERE_BF16_M16N8K16, acc, a, b)


def test_ampere_v2_pins_the_measured_special_values_v1_got_wrong() -> None:
    """The two rules where veritor's `_v1` hand-written extension disagreed with silicon: the NaN word, and a saturated
    group followed by an opposite-sign infinity (the campaign's R6)."""
    a = [ONE16] * 16
    b = [ONE16] * 16
    a[0] = NAN16
    assert P.AmpereBF16TcDot16.evaluate(0, *a, *b) == 0x7FFFFFFF  # measured canonical NaN, not 0x7FC00000
    assert P.HopperBF16WgmmaDot16.evaluate(0, *a, *b) == 0x7FFFFFFF
    assert P.AmpereBF16TcDot16.evaluate(PINF32, *([ONE16] * 16), *([ONE16] * 16)) == PINF32   # infinities are sticky
    assert P.AmpereBF16TcDot16.evaluate(PINF32, *([NINF16] + [ONE16] * 15), *([ONE16] * 16)) == 0x7FFFFFFF  # +Inf + -Inf


def test_fp8_primitives_are_the_tc_functions() -> None:
    assert (P.F32ToE4m3Sat.id, P.HopperE4m3QgmmaDot32.id) == ("F32ToE4m3Sat_v1", "HopperE4m3QgmmaDot32_v1")
    assert REGISTRY["F32ToE4m3Sat_v1"] is P.F32ToE4m3Sat and REGISTRY["HopperE4m3QgmmaDot32_v1"] is P.HopperE4m3QgmmaDot32
    assert P.F32ToE4m3Sat.signature() == ((("x", P.F32),), P.E4M3)
    assert P.HopperE4m3QgmmaDot32.signature() == ((("acc", P.F32), ("a", P.A32_8), ("b", P.A32_8)), P.F32)
    rng = random.Random(5)
    for x in [NAN32, 0xFFC00000, PINF32, NINF32, ONE32, 0, 0x80000000] + [rng.getrandbits(32) for _ in range(5000)]:
        assert P.F32ToE4m3Sat.evaluate(x) == f32_to_e4m3_sat_word(x)
    specials8 = [0x7F, 0xFF, 0x00, 0x80, 0x01, 0x7E, 0xFE, 0x38]
    cases = [(rng.getrandbits(32), [rng.getrandbits(8) for _ in range(32)], [rng.getrandbits(8) for _ in range(32)])
             for _ in range(300)]
    for acc in (0, 0x80000000, ONE32, NAN32, PINF32, NINF32, 0x7F7FFFFF, 0x00000001):
        for sa in specials8:
            a, b = [0x38] * 32, [0x38] * 32
            a[5], b[5] = sa, rng.choice(specials8)
            cases.append((acc, a, b))
    for acc, a, b in cases:
        assert P.HopperE4m3QgmmaDot32.evaluate(acc, *a, *b) == tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, a, b)
    assert P.HopperE4m3QgmmaDot32.evaluate(0, *([0x38] * 64)) == 0x42000000  # 32 * (1 * 1)


def test_const_primitives_are_memoised_and_decodable() -> None:
    from verity.ir.codec import _lookup_primitive

    c = P.const(32, 0x3F800000)
    assert c is P.const(32, 0x3F800000)
    assert c.id == "Const32[0x3f800000]_v1" and c.evaluate() == 0x3F800000
    assert P.ZERO32.evaluate() == 0 and P.ZERO32.id == "Const32[0x00000000]_v1"
    # a process that never built the program can still materialise the family from the id alone
    assert _lookup_primitive(REGISTRY, "Const16[0x3f80]_v1") is P.const(16, 0x3F80)
