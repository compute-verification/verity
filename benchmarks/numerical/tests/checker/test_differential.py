"""Completeness against ``verity.ml.circuits.silicon.silicon``: gen(x) is accepted and yields y = tc_dot(x) on structured cases,
random finite inputs, every all-finite golden record, and the K=1536 conformance vector through the VU."""
from __future__ import annotations

import random

import pytest
from conftest import bf16, golden_records, k1536_vector, random_cases, structured_cases

from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot, tc_dot_chain
from verity_numerical.checker import REAL, Reject, check_step, check_vu, gen_step, gen_vu, step_output, vu_output
from verity_numerical.checker import reference

PIPE = AMPERE_BF16_M16N8K16


def _agree(c: int, a: list[int], b: list[int]) -> int:
    expected = tc_dot(PIPE, c, a, b)
    h = gen_step(REAL, c, a, b)
    check_step(REAL, c, a, b, expected, h)          # gen(x) accepted with y = F(x)
    assert step_output(REAL, c, a, b) == expected
    assert reference.dot(REAL, c, a, b) == expected  # the parametric reference is silicon at REAL
    return expected


@pytest.mark.parametrize("name,c,a,b", structured_cases(), ids=[n for n, *_ in structured_cases()])
def test_structured_case_agrees_with_silicon(name: str, c: int, a: list[int], b: list[int]) -> None:
    _agree(c, a, b)


def test_structured_cases_cover_the_families() -> None:
    outs = {name: tc_dot(PIPE, c, a, b) for name, c, a, b in structured_cases()}
    assert outs["cancel/pairs"] == 0 and outs["cancel/across-groups"] == 0 and outs["cancel/negative-zero-state"] == 0
    assert outs["zero/truncation"] == 0 and outs["zero/truncation-negative"] == 0
    assert outs["overflow/group1-positive"] == 0x7F800000 and outs["overflow/group1-negative"] == 0xFF800000
    assert outs["overflow/group2-only"] == 0x7F800000
    assert outs["overflow/group1-sticky-through-cancelling-group2"] == 0x7F800000
    assert outs["overflow/state-max-plus-one-product"] == 0x7F800000
    assert (outs["subnormal/result-subnormal"] >> 23) & 0xFF == 0 and outs["subnormal/result-subnormal"] != 0
    assert outs["subnormal/state-passes-through"] == 0x2A
    assert (outs["near/just-below"] >> 23) & 0xFF == 254


def test_random_finite_inputs_agree_with_silicon() -> None:
    cases = random_cases(2000)
    mismatches = [(c, a, b) for c, a, b in cases if step_output(REAL, c, a, b) != tc_dot(PIPE, c, a, b)]
    assert mismatches == []
    for c, a, b in cases[:200]:
        _agree(c, a, b)


def test_every_all_finite_golden_record_reproduces() -> None:
    records = golden_records()
    assert len(records) == 360
    assert {f for f, *_ in records} >= {"randn_zero_acc", "randn_random_acc", "subnormal", "cancellation",
                                        "mixed_magnitude", "near_overflow", "signed_zero", "uniform_bits"}
    mismatches = []
    for family, c, a, b, d in records:
        expected = _agree(c, a, b)
        if expected != d:
            mismatches.append((family, c, a, b, d, expected))
    assert mismatches == []


def test_vu_reproduces_the_k1536_conformance_vector() -> None:
    a, b, y16 = k1536_vector()
    acc = tc_dot_chain(PIPE, 0, a, b)
    assert f32_to_bf16_word(acc) == y16
    h = gen_vu(REAL, a, b)
    assert len(h["steps"]) == 96
    check_vu(REAL, a, b, y16, h)
    assert vu_output(REAL, a, b) == y16


def test_vu_chains_and_casts_like_the_gate_family() -> None:
    rng = random.Random(7)
    a = [bf16(rng.getrandbits(1), rng.randrange(100, 140), rng.getrandbits(7)) for _ in range(12 * 16)]
    b = [bf16(rng.getrandbits(1), rng.randrange(100, 140), rng.getrandbits(7)) for _ in range(12 * 16)]
    expected = f32_to_bf16_word(tc_dot_chain(PIPE, 0, a, b))
    check_vu(REAL, a, b, expected, gen_vu(REAL, a, b))
    assert reference.vu(REAL, a, b) == expected


def test_vu_rejects_an_infinite_intermediate_accumulator_but_casts_a_final_infinity() -> None:
    big = [bf16(0, 254, 0x7F)] * 16
    one = [bf16(0, 127, 0)] * 16
    # the first unit saturates; a second unit would take an infinite accumulator: outside the relation
    with pytest.raises(Reject, match="acc.t"):
        gen_vu(REAL, big + one, big + one)
    with pytest.raises(InvalidArtifact):
        tc_dot_chain(PIPE, 0, big + one, big + one)
    # saturation in the last unit reaches the epilogue and becomes the BF16 infinity, as the gate f32_to_bf16 does
    assert tc_dot_chain(PIPE, 0, one + big, one + big) == 0x7F800000
    assert f32_to_bf16_word(0x7F800000) == 0x7F80
    check_vu(REAL, one + big, one + big, 0x7F80, gen_vu(REAL, one + big, one + big))


def test_epilogue_is_round_to_nearest_even_on_finite_and_infinite_words() -> None:
    for word in (0, 0x3F808000, 0x3F818000, 0x3F80FFFF, 0x3F800001, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
                 0x80000001, 0x00800000, 0x7F7F8000):
        assert reference.to_bf16(REAL, word) == f32_to_bf16_word(word)
