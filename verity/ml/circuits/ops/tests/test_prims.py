"""The registered primitives of `verity.ml.circuits.ops.prims` are exactly the `verity.ml.circuits.silicon` functions, under their registered ids."""
from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.silicon.cast import f32_to_bf16_hw_word, f32_to_bf16_word, f32_to_e4m3_sat_word
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16, tc_dot_total
from verity.ml.circuits.silicon.total_fp8 import HOPPER_E4M3_WGMMA_K32, tc_dot_total_e4m3

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


def test_the_sm120_fp8_step_is_the_total_function_of_its_pinned_model() -> None:
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32 as SM120

    step = P.BlackwellE4m3QmmaDot32
    assert step.id == "BlackwellE4m3QmmaDot32_v1" and REGISTRY["BlackwellE4m3QmmaDot32_v1"] is step
    assert step.signature() == P.HopperE4m3QgmmaDot32.signature()
    rng = random.Random(7)
    specials8 = [0x7F, 0xFF, 0x00, 0x80, 0x01, 0x7E, 0xFE, 0x38]
    cases = [(rng.getrandbits(32), [rng.getrandbits(8) for _ in range(32)], [rng.getrandbits(8) for _ in range(32)])
             for _ in range(300)]
    for acc in (0, 0x80000000, ONE32, NAN32, PINF32, NINF32, 0x7F7FFFFF, 0x00000001):
        for sa in specials8:
            a, b = [0x38] * 32, [0x38] * 32
            a[5], b[5] = sa, rng.choice(specials8)
            cases.append((acc, a, b))
    for acc, a, b in cases:
        assert step.evaluate(acc, *a, *b) == tc_dot_total_e4m3(SM120, acc, a, b)
    assert step.evaluate(0, *([0x38] * 64)) == 0x42000000
    # the 26-bit adder keeps the accumulator whole; Hopper's 14-bit one truncates it (1 + 2^-23 through zero products)
    zeros = [0x00] * 64
    assert step.evaluate(0x3F800001, *zeros) == 0x3F800001 and P.HopperE4m3QgmmaDot32.evaluate(0x3F800001, *zeros) == 0x3F800000


def test_the_triton_mmav2_step_is_two_chained_hopper_hmma_k16_steps() -> None:
    """`HopperE4m3MmaV2Dot32_v1` is, on every encoding, Hopper's mma.sync k16 step (`HOPPER_BF16_M16N8K16`) on the exactly widened
    operands at k mod 4 in {0, 1}, then again on k mod 4 in {2, 3} from that result: the two `HMMA.16816.F32` Triton chains."""
    from verity.ml.circuits.silicon.cast import e4m3_to_f32_word
    from verity.ml.circuits.silicon.models import HOPPER_BF16_M16N8K16
    from verity.ml.circuits.silicon.total_fp8 import MMAV2_E4M3_K_ORDER

    step = P.HopperE4m3MmaV2Dot32
    assert step.id == "HopperE4m3MmaV2Dot32_v1" and REGISTRY[step.id] is step and step.conformance.startswith("exact-model-tested")
    assert step.signature() == P.HopperE4m3QgmmaDot32.signature()
    assert sorted(MMAV2_E4M3_K_ORDER) == list(range(32)) and MMAV2_E4M3_K_ORDER[:4] == (0, 1, 4, 5)
    S0, S1 = [k for k in range(32) if k % 4 < 2], [k for k in range(32) if k % 4 >= 2]

    def wide(e: int) -> int:
        return 0x7FC0 | (e & 0x80) << 8 if (e & 0x7F) == 0x7F else e4m3_to_f32_word(e) >> 16

    def chained(acc, a, b):
        mid = tc_dot_total(HOPPER_BF16_M16N8K16, acc, [wide(a[k]) for k in S0], [wide(b[k]) for k in S0])
        return tc_dot_total(HOPPER_BF16_M16N8K16, mid, [wide(a[k]) for k in S1], [wide(b[k]) for k in S1])

    rng = random.Random(9)
    specials8 = [0x7F, 0xFF, 0x00, 0x80, 0x01, 0x7E, 0xFE, 0x38]
    cases = [(rng.getrandbits(32), [rng.getrandbits(8) for _ in range(32)], [rng.getrandbits(8) for _ in range(32)])
             for _ in range(300)]
    cases += [(_rand_finite_f32(rng), [rng.getrandbits(7) % 0x7F | rng.getrandbits(1) << 7 for _ in range(32)],
               [rng.getrandbits(7) % 0x7F | rng.getrandbits(1) << 7 for _ in range(32)]) for _ in range(300)]
    for acc in (0, 0x80000000, ONE32, NAN32, PINF32, NINF32, 0x7F7FFFFF, 0x00000001):
        for sa in specials8:
            for k in (5, 6):
                a, b = [0x38] * 32, [0x38] * 32
                a[k], b[k] = sa, rng.choice(specials8)
                cases.append((acc, a, b))
    for acc, a, b in cases:
        assert step.evaluate(acc, *a, *b) == chained(acc, a, b)
    assert step.evaluate(0, *([0x38] * 64)) == 0x42000000
    zeros = [0x00] * 64
    assert step.evaluate(0x3F800001, *zeros) == 0x3F800001 and P.HopperE4m3QgmmaDot32.evaluate(0x3F800001, *zeros) == 0x3F800000
    # 64 + 2^-18 + 2^-18 (two products of the least subnormal 0x01): summed in one group it is 64 + 2^-17, an f32 word; the first
    # mma's 64 + 2^-18 truncates to 64 before the second adds its own 2^-18, so the step is 64 when the two are in different halves
    def two_tiny(i: int, j: int) -> int:
        a = [0x00] * 32
        a[i] = a[j] = 0x01
        return step.evaluate(0x42800000, *a, *a)

    assert two_tiny(0, 2) == two_tiny(1, 31) == 0x42800000
    assert two_tiny(0, 1) == two_tiny(4, 29) == two_tiny(2, 3) == 0x42800001
    a = [0x00] * 32
    a[0] = a[2] = 0x01
    assert P.BlackwellE4m3QmmaDot32.evaluate(0x42800000, *a, *a) == 0x42800001


E5M2_PINF, E5M2_NINF, E5M2_NAN, E5M2_ONE, E5M2_MAX = 0x7C, 0xFC, 0x7D, 0x3C, 0x7B


def test_the_sm120_e5m2_step_is_the_e4m3_pipeline_with_e5m2_products_on_every_encoding() -> None:
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E5M2_M16N8K32 as SM120

    step = P.BlackwellE5m2QmmaDot32
    assert step.id == "BlackwellE5m2QmmaDot32_v1" and REGISTRY[step.id] is step
    assert step.signature() == P.BlackwellE4m3QmmaDot32.signature()
    rng = random.Random(11)
    for _ in range(300):
        acc = _rand_finite_f32(rng)
        a = [b for b in (rng.getrandbits(8) for _ in range(64)) if (b >> 2) & 0x1F != 0x1F][:32]
        b = [b for b in (rng.getrandbits(8) for _ in range(64)) if (b >> 2) & 0x1F != 0x1F][:32]
        if len(a) == len(b) == 32:
            assert step.evaluate(acc, *a, *b) == SM120.step(acc, a, b)
    ones, zeros = [E5M2_ONE] * 32, [0x00] * 32
    assert step.evaluate(0, *ones, *ones) == 0x42000000                                   # 32 * (1 * 1)
    assert step.evaluate(0x3F800001, *zeros, *zeros) == 0x3F800001                         # the accumulator is not truncated

    def one_at(k: int, x: int, rest: list[int]) -> list[int]:
        return rest[:k] + [x] + rest[k + 1:]

    assert step.evaluate(0, *one_at(3, E5M2_PINF, ones), *ones) == PINF32
    assert step.evaluate(ONE32, *one_at(3, E5M2_PINF, ones), *one_at(3, 0xBC, ones)) == NINF32   # +inf * -1
    assert step.evaluate(0, *one_at(3, E5M2_PINF, ones), *one_at(3, 0x00, ones)) == 0x7FFFFFFF    # inf * 0
    assert step.evaluate(0, *one_at(3, E5M2_NAN, ones), *ones) == 0x7FFFFFFF
    both = one_at(9, E5M2_NINF, one_at(3, E5M2_PINF, ones))
    assert step.evaluate(0, *both, *ones) == 0x7FFFFFFF                                    # +inf and -inf in the group
    assert step.evaluate(NINF32, *one_at(3, E5M2_PINF, ones), *ones) == 0x7FFFFFFF          # an infinity of each sign
    assert step.evaluate(PINF32, *ones, *ones) == PINF32 and step.evaluate(NAN32, *ones, *ones) == 0x7FFFFFFF
    # 32 of the largest products (57344^2 * 32 < 2^37) are far below the ulp of the largest finite accumulator: truncated away
    big = [E5M2_MAX] * 32
    assert step.evaluate(0xFF7FFFFF, *big, *[E5M2_MAX | 0x80] * 32) == 0xFF7FFFFF


def test_the_sm120_e5m2_step_reproduces_the_pro_6000_capture_on_every_encoding() -> None:
    """Every word of the tc_probe capture's specials and signed-zero tiles (RTX PRO 6000, one QMMA.16832.F32.E5M2.E5M2 per
    tile; fixtures/README.md), including the 4,824 words with a NaN or infinite operand or accumulator, which the run recorded
    but did not assert."""
    import numpy as np

    from verity.ml.kernels.tc import group_sum_total_batch
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E5M2_M16N8K32 as SM120

    root = Path(__file__).parent / "fixtures" / "tc-sm120-e5m2-2026-09-30"
    for name, words in (("tiles_specials", 6144), ("tiles_signed_zero", 32768)):
        z = np.load(root / f"{name}.npz")
        A, Bt, C, D = z["A"], z["Bt"], z["C"], z["D"]
        a = np.repeat(A, Bt.shape[1], axis=1).reshape(-1, 32)
        b = np.tile(Bt, (1, A.shape[1], 1)).reshape(-1, 32)
        acc, want = C.reshape(-1), D.reshape(-1)
        assert acc.size == words
        got, _ = group_sum_total_batch(SM120, acc, a, b)
        assert np.array_equal(got, want), name
        if name == "tiles_specials":
            for i in range(0, words, 7):
                assert P.BlackwellE5m2QmmaDot32.evaluate(int(acc[i]), *a[i].tolist(), *b[i].tolist()) == int(want[i])


def test_the_sm120_nvfp4_step_is_the_pinned_block_scaled_model() -> None:
    from verity.ml.circuits.silicon.errors import InvalidArtifact
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4 as NVF4

    step = P.BlackwellNvf4OmmaDot64
    assert step.id == "BlackwellNvf4OmmaDot64_v1" and REGISTRY[step.id] is step
    assert step.signature() == ((("acc", P.F32), ("a", P.A64_4), ("b", P.A64_4), ("sa", P.A4_8), ("sb", P.A4_8)), P.F32)
    rng = random.Random(9)
    checked = 0
    for _ in range(300):
        acc = rng.choice([0, 0x80000000, ONE32, 0xC2F60000, rng.getrandbits(32) & 0xBF7FFFFF])
        a, b = [rng.getrandbits(4) for _ in range(64)], [rng.getrandbits(4) for _ in range(64)]
        sa, sb = [rng.randrange(0x7F) for _ in range(4)], [rng.randrange(0x7F) for _ in range(4)]
        try:
            want = NVF4.step_scaled(acc, a, b, sa, sb)
        except InvalidArtifact:
            continue
        assert step.evaluate(acc, *a, *b, *sa, *sb) == want
        checked += 1
    assert checked > 250
    assert step.evaluate(0, *([0x2] * 128), *([0x38] * 8)) == 0x42800000      # 64 * (1 * 1) at scales 1 * 1


def test_the_sm120_nvfp4_step_is_total_as_measured_outside_ue4m3() -> None:
    """Measured on the RTX PRO 6000: a NaN-coded scale byte gives NaN over any accumulator; else NaN / ±inf accumulators pass through;
    else bit 7 of a scale byte is ignored."""
    step = P.BlackwellNvf4OmmaDot64
    rng = random.Random(10)
    codes = [rng.getrandbits(4) for _ in range(128)]
    scales = [rng.randrange(0x20, 0x48) for _ in range(8)]
    for bad in (0x7F, 0xFF):
        for acc in (0, ONE32, PINF32, NINF32, NAN32):
            s = list(scales)
            s[rng.randrange(8)] = bad
            assert step.evaluate(acc, *codes, *s) == 0x7FFFFFFF
    assert step.evaluate(NAN32, *codes, *scales) == step.evaluate(0xFFC00001, *codes, *scales) == 0x7FFFFFFF
    assert step.evaluate(PINF32, *codes, *scales) == PINF32 and step.evaluate(NINF32, *codes, *scales) == NINF32
    for acc in (0, ONE32, 0xC1200000):
        assert step.evaluate(acc, *codes, *[s | 0x80 for s in scales]) == step.evaluate(acc, *codes, *scales)
    rnd = [(rng.getrandbits(32), [rng.getrandbits(4) for _ in range(128)], [rng.getrandbits(8) for _ in range(8)]) for _ in range(500)]
    for acc, c, s in rnd:
        step.evaluate(acc, *c, *s)                                                # defined on every encoding


def test_the_sm120_mxfp4_step_is_the_pinned_block_scaled_model() -> None:
    from verity.ml.circuits.silicon.errors import InvalidArtifact
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_MXF4 as MXF4

    step = P.BlackwellMxf4OmmaDot64
    assert step.id == "BlackwellMxf4OmmaDot64_v1" and REGISTRY[step.id] is step
    assert step.signature() == ((("acc", P.F32), ("a", P.A64_4), ("b", P.A64_4), ("sa", P.A2_8), ("sb", P.A2_8)), P.F32)
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        acc = rng.choice([0, 0x80000000, ONE32, 0xC2F60000, rng.getrandbits(32) & 0xBF7FFFFF])
        a, b = [rng.getrandbits(4) for _ in range(64)], [rng.getrandbits(4) for _ in range(64)]
        sa, sb = [rng.randrange(100, 150) for _ in range(2)], [rng.randrange(100, 150) for _ in range(2)]
        try:
            want = MXF4.step_scaled(acc, a, b, sa, sb)
        except InvalidArtifact:
            continue
        assert step.evaluate(acc, *a, *b, *sa, *sb) == want
        checked += 1
    assert checked > 250
    assert step.evaluate(0, *([0x2] * 128), *([127] * 4)) == 0x42800000       # 64 * (1 * 1) at scales 2^0 * 2^0


def test_the_sm120_mxfp4_step_is_total() -> None:
    """A 0xFF scale byte gives NaN over any accumulator; else NaN / ±inf accumulators pass through; else a sum leaving binary32
    is ±inf of its sign."""
    step = P.BlackwellMxf4OmmaDot64
    rng = random.Random(12)
    codes = [rng.getrandbits(4) for _ in range(128)]
    scales = [rng.randrange(110, 140) for _ in range(4)]
    for acc in (0, ONE32, PINF32, NINF32, NAN32):
        s = list(scales)
        s[rng.randrange(4)] = 0xFF
        assert step.evaluate(acc, *codes, *s) == 0x7FFFFFFF
    assert step.evaluate(NAN32, *codes, *scales) == step.evaluate(0xFFC00001, *codes, *scales) == 0x7FFFFFFF
    assert step.evaluate(PINF32, *codes, *scales) == PINF32 and step.evaluate(NINF32, *codes, *scales) == NINF32
    six, neg_six = [0x7] * 64, [0xF] * 64                                        # E2M1 6.0 and -6.0; 0xFE is 2^127
    assert step.evaluate(0, *six, *six, *([0xFE] * 4)) == PINF32
    assert step.evaluate(ONE32, *neg_six, *six, *([0xFE] * 4)) == NINF32
    rnd = [(rng.getrandbits(32), [rng.getrandbits(4) for _ in range(128)], [rng.getrandbits(8) for _ in range(4)]) for _ in range(500)]
    for acc, c, s in rnd:
        step.evaluate(acc, *c, *s)                                                # defined on every encoding


def test_the_sm120_mxfp4_step_reproduces_the_pro_6000_edge_capture() -> None:
    """Every word of the tc_probe_fp4 capture's four record-only MXFP4 edge families (RTX PRO 6000, one
    OMMA.SF.16864.F32.E2M1.E2M1.E8 per tile; fixtures/README.md), which the run recorded but did not assert: 73,728 words, 9,216
    of them under a 0xFF scale byte, the rest overflowing or cancelling blocks and ±inf / NaN accumulators."""
    import numpy as np

    step = P.BlackwellMxf4OmmaDot64
    root = Path(__file__).parent / "fixtures" / "tc-sm120-mxf4-2026-10-01"
    words = nan_scale = 0
    for name in ("tiles_mxf4_nan_scale", "tiles_mxf4_overflow_cancel", "tiles_mxf4_overflow_edge", "tiles_mxf4_inf_acc_overflow"):
        z = np.load(root / f"{name}.npz")
        A, Bt, SFA, SFB, C, D = (z[k].tolist() for k in ("A", "Bt", "SFA", "SFB", "C", "D"))
        for t in range(len(A)):
            for i in range(16):
                for j in range(8):
                    sa, sb = SFA[t][i], SFB[t][j]
                    assert step.evaluate(C[t][i][j] & 0xFFFFFFFF, *A[t][i], *Bt[t][j], *sa, *sb) == D[t][i][j] & 0xFFFFFFFF, (name, t, i, j)
                    words += 1
                    nan_scale += 0xFF in sa or 0xFF in sb
    assert (words, nan_scale) == (73728, 9216)


def test_the_e2m1_cast_rounds_to_nearest_even_and_saturates() -> None:
    f = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]  # noqa: E731
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for code, v in enumerate(grid):
        assert P.F32ToE2m1Sat.evaluate(f(v)) == code and P.F32ToE2m1Sat.evaluate(f(-v)) == 0x8 | code
    # each midpoint (one ulp below, on it, one ulp above): the tie goes to the even code
    ties = {0.25: (0, 0, 1), 0.75: (1, 2, 2), 1.25: (2, 2, 3), 1.75: (3, 4, 4), 2.5: (4, 4, 5), 3.5: (5, 6, 6), 5.0: (6, 6, 7)}
    for v, codes in ties.items():
        assert tuple(P.F32ToE2m1Sat.evaluate(f(v) + d) for d in (-1, 0, 1)) == codes, v
    assert P.F32ToE2m1Sat.evaluate(f(1e30)) == 7 and P.F32ToE2m1Sat.evaluate(PINF32) == 7 and P.F32ToE2m1Sat.evaluate(NINF32) == 0xF
    assert P.F32ToE2m1Sat.evaluate(0x00000001) == 0 and P.F32ToE2m1Sat.evaluate(0x80000000) == 0x8
    assert P.F32ToE2m1Sat.evaluate(NAN32) == P.F32ToE2m1Sat.evaluate(0xFFFF0000) == P.E2M1_NAN_CODE == 0x7   # +6, sign dropped


def test_the_e4m3_widening_is_exact_on_every_finite_byte() -> None:
    from fractions import Fraction

    for b in range(256):
        e, m = (b >> 3) & 0xF, b & 7
        w = P.E4m3ToF32.evaluate(b)
        if e == 0xF and m == 7:
            assert w == 0x7FFFE000
            continue
        want = (Fraction(m, 8) + (1 if e else 0)) * Fraction(2) ** (max(e, 1) - 7) * (-1 if b & 0x80 else 1)
        assert Fraction(struct.unpack("<f", struct.pack("<I", w))[0]) == want and (w >> 31) == (b >> 7), hex(b)


def test_const_primitives_are_memoised_and_decodable() -> None:
    from verity.ml.circuits.codec import _lookup_primitive

    c = P.const(32, 0x3F800000)
    assert c is P.const(32, 0x3F800000)
    assert c.id == "Const32[0x3f800000]_v1" and c.evaluate() == 0x3F800000
    assert P.ZERO32.evaluate() == 0 and P.ZERO32.id == "Const32[0x00000000]_v1"
    # a process that never built the program can still materialise the family from the id alone
    assert _lookup_primitive(REGISTRY, "Const16[0x3f80]_v1") is P.const(16, 0x3F80)


def test_the_s8_step_is_the_pinned_integer_model() -> None:
    from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32 as M

    step = P.BlackwellS8ImmaDot32
    assert step.id == "BlackwellS8ImmaDot32_v1" and REGISTRY[step.id] is step and step.conformance.startswith("exact-model-tested")
    rng = random.Random(9)
    for acc in [0, 0x7FFFFFFF, 0x80000000, 0x7FFFFF00, 0x80000100] + [rng.getrandbits(32) for _ in range(400)]:
        a, b = [rng.getrandbits(8) for _ in range(32)], [rng.getrandbits(8) for _ in range(32)]
        assert step.evaluate(acc, *a, *b) == M.step(acc, a, b)
    assert step.evaluate(0x7FFFFFFF, *[0x7F] * 32, *[0x7F] * 32) == 0x7FFFFFFF
    assert step.evaluate(0x80000000, *[0x80] * 32, *[0x7F] * 32) == 0x80000000
    assert step.evaluate(0, *[0x80] * 32, *[0x80] * 32) == 32 * 128 * 128


def test_the_s32_cast_is_numpy_int32_to_float32() -> None:
    import numpy as np

    rng = random.Random(10)
    words = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF] + [(1 << e) + d for e in range(32) for d in (-1, 0, 1)] + \
            [rng.getrandbits(32) for _ in range(20_000)]
    words = [w & 0xFFFFFFFF for w in words]
    want = np.array(words, np.uint32).view(np.int32).astype(np.float32).view(np.uint32)
    assert [P.S32ToF32Rn.evaluate(w) for w in words] == want.tolist()
