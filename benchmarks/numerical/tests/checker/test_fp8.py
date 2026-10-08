"""The lossy generalisation: ``Params.from_model`` and the checker on the FP8 pipelines.

``reference.dot`` and the gadget relation (``step_output`` through the honest hints) must equal ``model.step`` on
2,000 random finite E4M3 cases per FP8 model -- random operands and accumulators, zero and negative-zero
accumulators, cancellation, subnormal operands -- and the BF16 parameters must be untouched (``from_model`` of the
frozen target is ``REAL``; no ``rescale_state`` in its contracts).
"""
from __future__ import annotations

import random

import pytest

from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon.models import (
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    ADA_E5M2_M16N8K32,
    AMPERE_BF16_M16N8K16,
    BLACKWELL_SM120_NVF4,
    HOPPER_E4M3_K32,
    HOPPER_E5M2_K32,
)
from verity_numerical.checker import REAL, TOY_LOSSY, Params, Reject, check_step, contracts, gen_step, step_output
from verity_numerical.checker import reference
from verity_numerical.checker.gadgets import RescaleState

FP8_MODELS = (HOPPER_E4M3_K32, ADA_E4M3_M16N8K32)


def _random_fp8_cases(model, n: int, seed: int) -> list[tuple[int, list[int], list[int]]]:
    """Finite E4M3 operands (every code but 0x7F/0xFF, subnormals included) and finite FP32 accumulators: random
    words, +-0, small-exponent words, and directed cancellation (a pair of opposite products against the accumulator)."""
    rng = random.Random(seed)
    finite = [w for w in range(256) if (w & 0x7F) != 0x7F]
    subnormal = [w for w in finite if (w & 0x78) == 0 and (w & 7)]
    k = model.k
    cases = []
    while len(cases) < n:
        kind = rng.randrange(6)
        if kind == 0:
            acc = 0
        elif kind == 1:
            acc = 0x8000_0000
        elif kind == 2:
            acc = (rng.randrange(2) << 31) | (rng.randrange(1, 40) << 23) | rng.randrange(1 << 23)   # tiny / subnormal-ish
        else:
            acc = rng.randrange(1 << 32)
            if (acc >> 23) & 0xFF == 0xFF:
                continue
        pool = subnormal if rng.randrange(5) == 0 else finite
        a = [rng.choice(pool) for _ in range(k)]
        b = [rng.choice(pool) for _ in range(k)]
        if rng.randrange(3) == 0:                       # cancellation: pairs of opposite products
            for i in range(0, k - 1, 2):
                a[i + 1], b[i + 1] = a[i] ^ 0x80, b[i]
        if rng.randrange(4) == 0:                       # a zero-product group (D10: the accumulator is still truncated)
            for i in range(k // 2):
                a[i] = rng.choice([0, 0x80])
        cases.append((acc, a, b))
    return cases


def test_from_model_of_the_frozen_target_is_real() -> None:
    assert Params.from_model(AMPERE_BF16_M16N8K16) == REAL
    assert Params.from_model(ADA_BF16_M16N8K16) == REAL
    assert not REAL.lossy and REAL.acc_exp == REAL.exp == 8 and REAL.op_nan_rule == "ieee"
    assert [c.gadget for c in contracts(REAL)] == [
        "fp32_decode", "bf16_decode", "product", "group_max", "align_state", "align_product", "signed_sum",
        "leading_bit", "normalise", "state_out", "pack", "epilogue",
    ]


@pytest.mark.parametrize("model", FP8_MODELS, ids=lambda m: m.name)
def test_from_model_of_the_fp8_pipelines_is_lossy(model) -> None:
    p = Params.from_model(model)
    assert (p.sig, p.exp, p.acc_sig, p.acc_exp, p.width, p.floor, p.groups) == (4, 4, 24, 8, 14, -139, model.groups)
    assert p.op_nan_rule == "e4m3" and p.lossy and p.acc_drop == 10 and p.acc_shift == -10
    assert (p.bias, p.e_min, p.e_max) == (7, -6, 8) and (p.acc_bias, p.acc_e_min, p.acc_e_max) == (127, -126, 127)
    assert p.t_max_finite == 15 and p.word_bits == 8 and p.acc_word_bits == 32
    assert (p.state_bits, p.state_shift, p.acc_bits, p.prod_bits, p.norm_bits, p.norm_up, p.norm_k_offset) == (14, 0, 14, 15, 14, 10, 0)
    names = [c.gadget for c in contracts(p)]
    assert "rescale_state" in names and "epilogue" not in names   # the bare dot: the FP32 word is the public output
    assert names.index("rescale_state") == names.index("align_state") - 1


def test_from_model_refuses_what_it_cannot_decompose() -> None:
    with pytest.raises(ValueError):
        Params.from_model(BLACKWELL_SM120_NVF4)
    with pytest.raises(Reject):
        RescaleState.contract(REAL)
    with pytest.raises(ValueError):
        Params(sig=4, exp=4, acc_sig=24, width=6, floor=-139, groups=(32,), acc_exp=8)   # narrower than the product
    with pytest.raises(ValueError):
        Params(sig=4, exp=4, acc_sig=24, width=14, floor=-100, groups=(32,), acc_exp=8)  # floor above the FP32 e_min


@pytest.mark.parametrize("model", FP8_MODELS, ids=lambda m: m.name)
def test_reference_and_checker_equal_model_step_on_random_fp8_cases(model) -> None:
    p = Params.from_model(model)
    cases = _random_fp8_cases(model, 2000, seed=hash(model.name) & 0xFFFF)
    mismatches = []
    for acc, a, b in cases:
        want = model.step(acc, a, b)
        got_ref = reference.dot(p, acc, a, b)
        got_chk = step_output(p, acc, a, b)
        if got_ref != want or got_chk != want:
            mismatches.append((hex(acc), hex(want), hex(got_ref), hex(got_chk)))
    assert mismatches == []
    for acc, a, b in cases[:100]:
        check_step(p, acc, a, b, model.step(acc, a, b), gen_step(p, acc, a, b))


@pytest.mark.parametrize("model", FP8_MODELS, ids=lambda m: m.name)
def test_zero_product_group_truncates_the_accumulator(model) -> None:
    # D10: a group of all-zero products still passes the accumulator through the 14-bit adder (Ada, Hopper alike)
    p = Params.from_model(model)
    zeros = [0] * model.k
    for acc in (0x3F800001, 0x3F80FFFF, 0xBF8003FF, 0x00800001):
        want = model.step(acc, zeros, zeros)
        assert want == acc & ~0x3FF, hex(acc)                      # the low ten significand bits are gone
        assert step_output(p, acc, zeros, zeros) == want == reference.dot(p, acc, zeros, zeros)
    # ... and the subnormal accumulator keeps its (already short) bits only above the adder's cut
    assert step_output(p, 0x00000001, zeros, zeros) == model.step(0x00000001, zeros, zeros) == 0


def test_e4m3_nan_words_are_rejected_and_448_is_finite() -> None:
    p = Params.from_model(HOPPER_E4M3_K32)
    one = [0x38] * 32
    for bad in (0x7F, 0xFF):
        with pytest.raises(Reject, match="non-finite operand"):
            reference.dot(p, 0, [bad] + [0] * 31, one)
        with pytest.raises(Reject, match=r"\.finite"):
            gen_step(p, 0, [bad] + [0] * 31, one)
        with pytest.raises(InvalidArtifact):
            HOPPER_E4M3_K32.step(0, [bad] + [0] * 31, one)
    big = [0x7E] * 32                                             # 448 * 448 * 32 = 6,422,528: finite, exact in 14 bits? no -- truncated
    assert step_output(p, 0, big, big) == HOPPER_E4M3_K32.step(0, big, big) == reference.dot(p, 0, big, big)


def test_rescale_mutations_are_rejected() -> None:
    import copy

    p = Params.from_model(ADA_E4M3_M16N8K32)
    acc = 0x3F800001
    a = [0x38, 0x38] + [0] * 30
    b = [0x38, 0xB8] + [0] * 30
    y = ADA_E4M3_M16N8K32.step(acc, a, b)
    h = gen_step(p, acc, a, b)
    check_step(p, acc, a, b, y, h)
    r = h["groups"][0]["acc_rescale"]
    assert r["P"] == (acc & 0x7FFFFF | 0x800000) >> 10 and r["r"] == 1
    bad = copy.deepcopy(h)
    bad["groups"][0]["acc_rescale"]["r"] += 1                    # wrong remainder: the division identity fails
    with pytest.raises(Reject, match="g0.rescale_state.div"):
        check_step(p, acc, a, b, y, bad)
    bad = copy.deepcopy(h)
    bad["groups"][0]["acc_rescale"]["P"] -= 1                    # keep the identity by inflating r past 2^drop
    bad["groups"][0]["acc_rescale"]["r"] += 1 << 10
    with pytest.raises(Reject, match="g0.rescale_state.r"):
        check_step(p, acc, a, b, y, bad)
    bad = copy.deepcopy(h)
    del bad["groups"][0]["acc_rescale"]
    with pytest.raises(Reject, match="rescale_state"):
        check_step(p, acc, a, b, y, bad)


def test_e5m2_hypotheses_go_through_the_same_params() -> None:
    for m in (ADA_E5M2_M16N8K32, HOPPER_E5M2_K32):
        p = Params.from_model(m)
        assert (p.sig, p.exp, p.op_nan_rule, p.width, p.lossy) == (3, 5, "ieee", 14, True)
        rng = random.Random(11)
        for _ in range(300):
            a = [rng.randrange(256) for _ in range(m.k)]
            b = [rng.randrange(256) for _ in range(m.k)]
            if any((w & 0x7C) == 0x7C for w in a + b):
                continue
            acc = rng.choice([0, rng.randrange(1 << 32)])
            if (acc >> 23) & 0xFF == 0xFF:
                continue
            want = m.step(acc, a, b)
            assert reference.dot(p, acc, a, b) == want and step_output(p, acc, a, b) == want


@pytest.mark.slow
def test_toy_lossy_reference_and_checker_agree_exhaustively_on_a_slice() -> None:
    # The reduced lossy widths (sig 3 / exp 3 e4m3-rule operands, 7-bit / 4-exp accumulator, 5-bit adder):
    # every accumulator word x a fixed set of operand vectors, reference against gadget relation.
    p = TOY_LOSSY
    assert p.lossy and p.acc_drop == 2 and p.norm_up == 2 and p.e_max == 4
    rng = random.Random(3)
    finite = [w for w in range(1 << p.word_bits) if reference._finite_operand(p, w)]
    assert len(finite) == (1 << p.word_bits) - 2                  # two NaN words, no infinities
    vectors = [([rng.choice(finite) for _ in range(4)], [rng.choice(finite) for _ in range(4)]) for _ in range(12)]
    for c in range(1 << p.acc_word_bits):
        if (c >> p.acc_frac) & (2**p.acc_exp - 1) == p.acc_t_finite:
            continue
        for a, b in vectors:
            assert step_output(p, c, a, b) == reference.dot(p, c, a, b)
