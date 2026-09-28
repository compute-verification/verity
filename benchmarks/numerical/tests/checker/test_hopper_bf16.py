"""``Params.from_model(HOPPER_BF16_M16N8K16)`` and the checker on the Hopper BF16 pipeline (lane b-hopper).

The relation is FIRST's with one group of 16, the 26-bit adder and floor -133 instead of two groups of 8, 25 bits and
-132: not lossy (the adder is wider than the FP32 significand), so the gadget list is ``REAL``'s -- including the
``f32_to_bf16`` epilogue -- with different widths.  ``reference.dot``, the gadget relation and the whole
``GemmCoordinate<1536>`` VU must equal ``model.step`` / ``tc_dot_chain`` + ``f32_to_bf16`` on random finite cases,
the D-family edge cases, and the pinned reference vectors; and where Hopper and Ampere differ the checker at the Hopper
parameters must follow Hopper.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.models import AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, tc_dot_chain
from verity.ml.circuits.targets.target import BF16_HOPPER, TC_VECTORS
from verity_numerical.checker import REAL, Params, Reject, check_step, check_vu, contracts, gen_step, gen_vu, step_output, vu_output
from verity_numerical.checker import reference

ROOT = Path(__file__).resolve().parents[4]
MODEL = HOPPER_BF16_M16N8K16
P = Params.from_model(MODEL)


def random_bf16_cases(n: int, seed: int) -> list[tuple[int, list[int], list[int]]]:
    """Finite BF16 operands and finite FP32 accumulators: random words over a bounded exponent range, +-0 and tiny
    accumulators, subnormal operands, directed cancellation, zero-product halves, large-plus-small (the adder's window)."""
    rng = random.Random(seed)

    def word(lo: int = 90, hi: int = 165) -> int:
        return (rng.randrange(2) << 15) | (rng.randrange(lo, hi) << 7) | rng.randrange(128)

    def subnormal() -> int:
        return (rng.randrange(2) << 15) | rng.randrange(1, 128)

    cases = []
    while len(cases) < n:
        kind = rng.randrange(7)
        if kind == 0:
            acc = 0
        elif kind == 1:
            acc = 0x8000_0000
        elif kind == 2:
            acc = (rng.randrange(2) << 31) | (rng.randrange(0, 40) << 23) | rng.randrange(1 << 23)   # tiny / subnormal
        elif kind == 3:
            acc = (rng.randrange(2) << 31) | (rng.randrange(150, 200) << 23) | rng.randrange(1 << 23)  # large: products fall off the window
        else:
            acc = (rng.randrange(2) << 31) | (rng.randrange(60, 200) << 23) | rng.randrange(1 << 23)
        pool = rng.randrange(6)
        if pool == 0:
            a = [subnormal() for _ in range(16)]
            b = [word(120, 140) for _ in range(16)]
        elif pool == 1:
            a = [word(60, 100) for _ in range(16)]                 # products near / below the floor
            b = [word(60, 100) for _ in range(16)]
        else:
            a = [word() for _ in range(16)]
            b = [word() for _ in range(16)]
        if rng.randrange(3) == 0:                                   # cancellation: pairs of opposite products
            for i in range(0, 15, 2):
                a[i + 1], b[i + 1] = a[i] ^ 0x8000, b[i]
        if rng.randrange(4) == 0:                                   # a zero half
            for i in range(8):
                a[i] = rng.choice([0, 0x8000])
        try:
            MODEL.step(acc, a, b)
        except InvalidArtifact:
            continue                                                # overflowed the finite domain: not a case
        cases.append((acc, a, b))
    return cases


def test_from_model_is_reals_relation_with_the_hopper_widths() -> None:
    assert (P.sig, P.exp, P.acc_sig, P.acc_exp, P.width, P.floor, P.groups) == (8, 8, 24, 8, 26, -133, (16,))
    assert P.op_nan_rule == "ieee" and not P.lossy and P.word_bits == 16 and P.acc_word_bits == 32
    assert (REAL.width, REAL.floor, REAL.groups) == (25, -132, (8, 8)) and P != REAL
    assert [c.gadget for c in contracts(P)] == [c.gadget for c in contracts(REAL)], "same gadgets, different widths"
    assert "epilogue" in [c.gadget for c in contracts(P)]
    assert Params.from_model(AMPERE_BF16_M16N8K16) == REAL


def test_reference_and_checker_equal_model_step_on_random_cases() -> None:
    cases = random_bf16_cases(2000, seed=0xB16)
    mismatches = []
    for acc, a, b in cases:
        want = MODEL.step(acc, a, b)
        got_ref = reference.dot(P, acc, a, b)
        got_chk = step_output(P, acc, a, b)
        if got_ref != want or got_chk != want:
            mismatches.append((hex(acc), hex(want), hex(got_ref), hex(got_chk)))
    assert mismatches == []
    for acc, a, b in cases[:100]:
        check_step(P, acc, a, b, MODEL.step(acc, a, b), gen_step(P, acc, a, b))


def test_the_checker_at_hopper_params_follows_hopper_where_ampere_differs() -> None:
    cases = random_bf16_cases(3000, seed=0xA1)
    differing = [(acc, a, b) for acc, a, b in cases if MODEL.step(acc, a, b) != AMPERE_BF16_M16N8K16.step(acc, a, b)]
    assert len(differing) >= 50, "the cases must exercise the adder width"
    for acc, a, b in differing:
        h_word, a_word = MODEL.step(acc, a, b), AMPERE_BF16_M16N8K16.step(acc, a, b)
        assert step_output(P, acc, a, b) == h_word
        assert step_output(REAL, acc, a, b) == a_word
        with pytest.raises(Reject):
            check_step(P, acc, a, b, a_word, gen_step(P, acc, a, b))       # the Ampere word is a wrong claim on Hopper


def test_d_family_edge_cases() -> None:
    one, zero = 0x3F80, 0
    z16 = [zero] * 16
    ones = [one] * 16
    # zero and negative-zero accumulators, all-zero products (no truncation: the adder is wider than the significand)
    for acc in (0, 0x8000_0000, 0x3F800001, 0xBF8003FF, 0x00800001, 0x00000001):
        assert step_output(P, acc, z16, z16) == MODEL.step(acc, z16, z16)
    assert MODEL.step(0x3F800001, z16, z16) == 0x3F800001, "a pass-through accumulator keeps every bit (26 > 24)"
    # exact cancellation to zero and the sign of the computed zero
    a = [one, one ^ 0x8000] + [zero] * 14
    b = [one, one] + [zero] * 14
    for acc in (0, 0x8000_0000):
        assert step_output(P, acc, a, b) == MODEL.step(acc, a, b)
    # 16 x 1.0 and max-finite-in-range operands
    assert step_output(P, 0, ones, ones) == MODEL.step(0, ones, ones) == 0x41800000
    big = [0x4F80] * 16                                                # 2^32 each: 16 x 2^64 = 2^68, exact
    assert step_output(P, 0, big, big) == MODEL.step(0, big, big) == 0x61800000
    # subnormal products below the floor vanish
    tiny = [0x0001] * 16                                               # 2^-133
    assert step_output(P, 0x3F800000, tiny, ones) == MODEL.step(0x3F800000, tiny, ones) == 0x3F800000
    assert step_output(P, 0, tiny, [0x4F80] * 16) == MODEL.step(0, tiny, [0x4F80] * 16)  # 2^-101 each: above the floor


def test_non_finite_operands_and_accumulators_are_rejected() -> None:
    one = [0x3F80] * 16
    for bad in (0x7F80, 0xFF80, 0x7FC0, 0x7F81):
        with pytest.raises(Reject):
            reference.dot(P, 0, [bad] + [0] * 15, one)
        with pytest.raises(Reject):
            gen_step(P, 0, [bad] + [0] * 15, one)
        with pytest.raises(InvalidArtifact):
            MODEL.step(0, [bad] + [0] * 15, one)
    with pytest.raises(Reject):
        gen_step(P, 0x7F800000, one, one)


def test_the_vu_is_the_pinned_reference_vectors_and_tc_dot_chain() -> None:
    for name in BF16_HOPPER.reference_vectors:
        rec = json.loads((ROOT / TC_VECTORS / f"{name}.json").read_text())
        a = [int(x, 16) for x in rec["a"]]
        b = [int(x, 16) for x in rec["b"]]
        assert vu_output(P, a, b) == int(rec["y"], 16), name
        assert tc_dot_chain(MODEL, 0, a, b) == int(rec["acc"], 16)
        check_vu(P, a, b, int(rec["y"], 16), gen_vu(P, a, b))
    rng = random.Random(7)
    for _ in range(3):
        a = [(rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(128) for _ in range(1536)]
        b = [(rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(128) for _ in range(1536)]
        acc = tc_dot_chain(MODEL, 0, a, b)
        assert vu_output(P, a, b) == f32_to_bf16_word(acc)
        if acc != tc_dot_chain(AMPERE_BF16_M16N8K16, 0, a, b):
            assert vu_output(REAL, a, b) == f32_to_bf16_word(tc_dot_chain(AMPERE_BF16_M16N8K16, 0, a, b))
