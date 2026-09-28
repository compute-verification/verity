"""The source-close evaluator against ``verity.ml.circuits.silicon.silicon`` and the hardware captures (test plan items a-d of lane WP1).

The exhaustive/pod counterpart is ``exhaustive_driver.py`` (recorded run, stage ``wp1.exhaustive``); these run in seconds.
"""

from __future__ import annotations

import pytest

from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity_numerical.reference import hawkeye_close as H
from wp1_cases import (
    boundary_groups,
    canon,
    canon_term,
    gfloat_of,
    golden_records,
    random_cases,
    special_records,
    structured_cases,
    term_of,
)

PIPE = S.AMPERE_BF16_M16N8K16
ONE = 0x3F80


def test_constants_are_the_pinned_simulators() -> None:
    assert (H.E_ZERO, H.W_WORK, H.W_FP32, H.N_GROUP) == (-132, 25, 24, 9)
    assert (PIPE.width, PIPE.zero_exponent, PIPE.groups) == (H.W_WORK, H.E_ZERO, (8, 8))
    assert 1 + PIPE.groups[0] == H.N_GROUP


# (a) every BF16 encoding ---------------------------------------------------------------------------------------------------------


def test_every_bf16_encoding_decodes_alike_and_non_finite_is_flagged_alike() -> None:
    for word in range(1 << 16):
        finite = H.is_finite_bf16(word)
        if finite:
            for other in (ONE, 0x0080, 0x0001):  # times 1.0, the smallest normal, the smallest subnormal
                got = H.multiply_bfloat16_to_gfloat(word, other)
                assert canon(got) == canon_term(S.bf16_product(word, other)), f"{word:#06x} * {other:#06x}"
        else:
            with pytest.raises(InvalidArtifact):
                S.bf16_product(word, ONE)
            with pytest.raises(H.NonFinite):
                H.tc_dot16(0, [word] + [0] * 15, [ONE] * 16)
            # the faithful routine has no branch: exponent field 0xFF is the exponent 128 with an implicit one
            got = H.multiply_bfloat16_to_gfloat(word, ONE)
            assert got.exponent == 128 and got.significand == (((word & 0x7F) | 0x80) * 0x80) << 9


# (b) every significand product ------------------------------------------------------------------------------------------------------


def test_all_256x256_significand_products_agree() -> None:
    count = 0
    for sa in range(256):
        a = (sa & 0x7F) | ((0x7F if sa & 0x80 else 0) << 7)  # exponent field 127 gives the implicit one; 0 keeps it off
        for sb in range(256):
            b = (sb & 0x7F) | ((0x7F if sb & 0x80 else 0) << 7)
            for sign in (0, 0x8000):
                got = H.multiply_bfloat16_to_gfloat(a ^ sign, b)
                want = S.bf16_product(a ^ sign, b)
                assert canon(got) == canon_term(want)
                if got.significand:
                    assert got.significand == (sa * sb) << 9
                count += 1
    assert count == 2 * 256 * 256


# (c) boundary groups -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,group", boundary_groups(), ids=[n for n, _ in boundary_groups()])
def test_boundary_group_sums_agree_in_all_three_forms(name: str, group: list[H.Gfloat]) -> None:
    close = H.group_sum(group)
    closed = H.group_sum_closed_form(group)
    sil = S.group_sum([term_of(g) for g in group], width=PIPE.width, zero_exponent=PIPE.zero_exponent)
    assert canon(close) == canon(closed) == canon_term(sil), name
    if close.significand:
        assert close.exponent >= -126 and close.significand < 1 << 24
        assert close.exponent > -126 or close.significand < 1 << 23 or close.significand >> 23 == 1


def test_boundary_values_are_the_expected_ones() -> None:
    groups = dict(boundary_groups())
    assert H.group_sum(groups["all zero"]) == H.Gfloat(False, H.E_ZERO, 0)
    assert H.group_sum(groups["truncation to zero, negative"]) == H.Gfloat(True, H.E_ZERO, 0)  # sign kept internally
    assert H.gfloat_to_float_bits(H.group_sum(groups["truncation to zero, negative"])) == 0  # ... but the word is +0
    assert H.group_sum(groups["just above truncation"]) == H.Gfloat(False, -126, 1)  # the smallest subnormal
    assert H.group_sum(groups["exact cancellation"]) == H.Gfloat(False, H.E_ZERO, 0)
    assert H.group_sum(groups["cancellation to one ulp"]) == H.Gfloat(False, -23, 1 << 23)
    assert H.group_sum(groups["shift 24 keeps one bit"]) == H.Gfloat(False, 0, (1 << 23) | 1)
    assert H.group_sum(groups["shift 25 keeps it in the window, the 24-bit truncation drops it"]) == H.Gfloat(False, 0, 1 << 23)
    assert H.group_sum(groups["shift 26 drops the bit"]) == H.Gfloat(False, 0, 1 << 23)
    over = H.group_sum(groups["overflow to 128, NaN word"])
    assert over.exponent == 128 and H.gfloat_to_float_bits(over) == 0x7FFFFFFF  # the masking yields a NaN word
    over = H.group_sum(groups["overflow to 128, infinity word"])
    assert over.exponent == 128 and H.gfloat_to_float_bits(over) == 0x7F800000  # ... or, with a zero mantissa, +inf
    over = H.group_sum(groups["overflow to 129 (masking)"])
    assert over.exponent == 129 and H.gfloat_to_float_bits(over) >> 23 == 0  # wraps into the subnormal field
    assert S.pack_fp32(term_of(over)) == 0x7F800000  # silicon saturates


# (d) the transition unit --------------------------------------------------------------------------------------------------------


def _compare(cases):
    """Returns (compared, saturating, silicon_vs_hw_mismatches, close_vs_silicon_mismatches_on_finite_results, wrapper_mismatches)."""

    saturating = sil_hw = close_sil = wrapper = 0
    for c in cases:
        sil = S.tc_dot(PIPE, c.acc, list(c.a), list(c.b))
        close = H.tc_dot16(c.acc, c.a, c.b)
        if c.d is not None and sil != c.d:
            sil_hw += 1
        if H.is_finite_fp32(sil):
            if close != sil:
                close_sil += 1
        else:
            saturating += 1
        if c.acc & 0x7FFFFFFF == 0 and H.wrapper_element(c.a, c.b) != close:
            wrapper += 1
    return len(cases), saturating, sil_hw, close_sil, wrapper


def test_transition_unit_agrees_with_silicon_on_structured_and_random_cases() -> None:
    cases = structured_cases()[::7] + random_cases(6000, seed=20260921)
    n, saturating, _, close_sil, wrapper = _compare(cases)
    assert n > 14000 and close_sil == 0 and wrapper == 0
    assert 0 < saturating < n  # the mixtures do reach the saturating region and stay mostly finite


@pytest.mark.parametrize("source", ["golden", "a100", "4090"])
def test_transition_unit_reproduces_every_all_finite_hardware_record(source: str) -> None:
    cases = golden_records() if source == "golden" else special_records(source)
    n, saturating, sil_hw, close_sil, wrapper = _compare(cases)
    assert n == (360 if source == "golden" else 7456 + 257)
    assert sil_hw == 0, "silicon disagrees with the hardware"
    assert close_sil == 0, "the source-close evaluator disagrees with silicon on a finite result"
    assert wrapper == 0
    assert saturating == (114 if source == "golden" else 2197)
    # where the result leaves FP32's range the simulator masks and the hardware saturates; the masked word equals the hardware's
    # infinity exactly when the exponent is 128 and the mantissa bits are zero
    coincide = 0
    for c in cases:
        if not H.is_finite_fp32(c.d):
            assert c.d in (0x7F800000, 0xFF800000)
            g2 = _group2(c)
            same = H.tc_dot16(c.acc, c.a, c.b) == c.d
            assert same == (g2.exponent == 128 and g2.significand & 0x7FFFFF == 0)
            coincide += same
    assert coincide == (0 if source == "golden" else 36)


def _group2(c) -> H.Gfloat:
    acc = H.fp32_to_gfloat(c.acc)
    for start in (0, 8):
        acc = H.group_sum([acc] + [H.multiply_bfloat16_to_gfloat(c.a[l], c.b[l]) for l in range(start, start + 8)])
    return acc


@pytest.mark.parametrize("source,to_zero,to_finite", [("a100", 18, 0), ("4090", 18, 0), ("golden", 0, 1)])
def test_saturation_is_where_the_simulator_and_the_hardware_part(source: str, to_zero: int, to_finite: int) -> None:
    """A group-1 result beyond exponent 127 is sticky on the hardware (silicon returns it); the simulator keeps computing and
    group 2 can bring it back into range -- to an exact zero (18 A100/4090 special cases) or to a finite nonzero value (one 4090
    golden record).  The hardware word is the infinity in every such case; the simulator's word is finite."""

    cases = golden_records() if source == "golden" else special_records(source)
    zero = finite = 0
    for c in cases:
        acc = H.fp32_to_gfloat(c.acc)
        g1 = H.group_sum([acc] + [H.multiply_bfloat16_to_gfloat(c.a[l], c.b[l]) for l in range(8)])
        if g1.significand and g1.exponent > 127:
            g2 = H.group_sum([g1] + [H.multiply_bfloat16_to_gfloat(c.a[l], c.b[l]) for l in range(8, 16)])
            if g2.significand == 0:
                zero += 1
            elif g2.exponent <= 127:
                finite += 1
            else:
                continue
            assert c.d in (0x7F800000, 0xFF800000)
            assert H.is_finite_fp32(H.tc_dot16(c.acc, c.a, c.b)) and S.tc_dot(PIPE, c.acc, list(c.a), list(c.b)) == c.d
    assert (zero, finite) == (to_zero, to_finite)


# the adapter (D7) ---------------------------------------------------------------------------------------------------------------


def _reachable_outputs():
    """Every ``group_sum`` output shape: normal significands (bit 23 set) at every exponent, and every subnormal bit length."""

    patterns = [0, 1, 0x7FFFFF, 0x555555, 0x2AAAAA, 0x400000, 0x123456]
    for exponent in range(-126, 128):
        for m in patterns:
            yield H.Gfloat(False, exponent, (1 << 23) | m)
            yield H.Gfloat(True, exponent, (1 << 23) | m)
    for bits in range(1, 24):
        for m in {1 << (bits - 1), (1 << bits) - 1, (1 << (bits - 1)) | ((1 << bits) - 1) >> 1}:
            yield H.Gfloat(False, -126, m)
            yield H.Gfloat(True, -126, m)


def test_adapter_inverts_operator_float_on_every_reachable_output_and_the_wrapper_finalisation_is_a_fixed_point() -> None:
    n = 0
    for g in _reachable_outputs():
        word = H.gfloat_to_float_bits(g)
        assert H.fp32_to_gfloat(word) == g  # left inverse (the zero is not among the outputs enumerated here)
        assert H.group_sum([g]) == g  # the wrapper's final group_sum over the single remaining Gfloat changes nothing
        assert canon(gfloat_of(S.fp32_term(word))) == canon(g)  # and silicon decodes the word to the same term
        n += 1
    assert n > 3500
    for zero in (H.Gfloat(False, H.E_ZERO, 0), H.Gfloat(True, H.E_ZERO, 0)):
        assert H.group_sum([zero]) == H.Gfloat(False, H.E_ZERO, 0)
        assert H.gfloat_to_float_bits(zero) == 0
        assert H.fp32_to_gfloat(0x00000000) == H.Gfloat(False, H.E_ZERO, 0)
        assert H.fp32_to_gfloat(0x80000000) == H.Gfloat(True, H.E_ZERO, 0)


def test_adapter_round_trips_finite_words_and_rejects_the_rest() -> None:
    for exponent_field in range(0, 255):
        for mantissa in (0, 1, 0x7FFFFF, 0x400000, 0x123456):
            for sign in (0, 1 << 31):
                word = sign | (exponent_field << 23) | mantissa
                assert H.gfloat_to_float_bits(H.fp32_to_gfloat(word)) == (0 if word & 0x7FFFFFFF == 0 else word)
    for word in (0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF):
        with pytest.raises(H.NonFinite):
            H.fp32_to_gfloat(word)
        with pytest.raises(H.NonFinite):
            H.tc_dot16(word, [ONE] * 16, [ONE] * 16)
        with pytest.raises(InvalidArtifact):
            S.tc_dot(PIPE, word, [ONE] * 16, [ONE] * 16)


def test_a_zero_accumulator_must_carry_e_zero() -> None:
    """The one design choice in the adapter: a zero term with any other exponent would raise the group's maximum and change the
    truncation of small products.  Shown on the smallest subnormal product with a +0 accumulator."""

    a = [0x0001] + [0] * 15  # the smallest BF16 subnormal (2^-133) times 1.0: exponent -126, significand 0x80 << 9
    b = [ONE] * 16
    products = [H.multiply_bfloat16_to_gfloat(a[l], b[l]) for l in range(8)]
    with_e_zero = H.group_sum([H.Gfloat(False, H.E_ZERO, 0)] + products)
    with_zero_exponent = H.group_sum([H.Gfloat(False, 0, 0)] + products)
    assert with_e_zero == H.Gfloat(False, -126, 1 << 16) and with_zero_exponent == H.Gfloat(False, H.E_ZERO, 0)
    assert H.tc_dot16(0, a, b) == S.tc_dot(PIPE, 0, a, b) == 0x00010000  # the FP32 subnormal 2^-133
