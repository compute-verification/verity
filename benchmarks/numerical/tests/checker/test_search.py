"""The adversarial searches, at the sizes that run in seconds: exhaustive enumeration per gadget at ``TINY``,
single-hint mutation of the composed unit, and the z3 queries whose answer is quick.  The full suite (every
gadget, the modular readings, the group step) is ``python -m verity_numerical.checker.search``."""
from __future__ import annotations

import random

import pytest

from verity_numerical.checker import TINY, Params
from verity_numerical.checker.search import GOLDILOCKS, M31, P40, SPECS, exhaustive, single_hint_search, z3_gadget

FAST = ["bf16_decode", "fp32_decode", "product", "group_max", "align_state", "align_product", "leading_bit", "state_out", "pack"]


@pytest.mark.parametrize("name", FAST)
def test_exhaustive_at_tiny_widths_is_sound_and_complete(name: str) -> None:
    r = exhaustive(SPECS[name], TINY)
    assert r.admissible > 0
    assert r.sound and r.complete
    assert r.inputs_with_two_outputs == 0 and r.inputs_off_reference == 0 and r.incomplete == 0
    if name != "group_max":                      # ties admit several selectors, all giving the same E
        assert r.max_accepted_per_input == 1


def test_exhaustive_at_a_second_width() -> None:
    p = Params(sig=2, exp=2, acc_sig=3, width=4, floor=-2, groups=(1, 1))
    for name in ("align_state", "align_product", "leading_bit", "normalise"):
        r = exhaustive(SPECS[name], p)
        assert r.sound and r.complete, name


def test_single_hint_mutation_of_the_unit_finds_no_alternative() -> None:
    rng = random.Random(7)

    def finite(frac_bits: int) -> int:
        return (rng.getrandbits(1) << (TINY.exp + frac_bits)) | (rng.randrange(TINY.t_finite) << frac_bits) \
            | rng.getrandbits(frac_bits)
    k = sum(TINY.groups)
    cases = [(finite(TINY.acc_frac), [finite(TINY.frac) for _ in range(k)], [finite(TINY.frac) for _ in range(k)])
             for _ in range(6)]
    r = single_hint_search(TINY, cases)
    assert r.inputs == 6 and r.alternatives > 0
    assert r.accepted_other_output == 0 and r.accepted_same_output == 0


@pytest.mark.parametrize("name", ["bf16_decode", "fp32_decode", "product", "align_state", "align_product",
                                  "leading_bit", "normalise", "state_out", "pack", "epilogue"])
def test_z3_at_real_widths_finds_no_deviating_hints(name: str) -> None:
    r = z3_gadget(SPECS[name], timeout_ms=120_000)
    assert r.status == "unsat", (r.status, r.model)


@pytest.mark.parametrize("name, drop", [("group_max", ("attained",)), ("group_max", ("onehot",)),
                                        ("align_state", ("rem",)), ("align_product", ("rem",)),
                                        ("leading_bit", ("window",)), ("signed_sum", ("v", "z1")),
                                        ("normalise", ("clamp_proof",)), ("state_out", ("ovf_proof",))])
def test_z3_each_named_bound_is_load_bearing(name: str, drop: tuple[str, ...]) -> None:
    r = z3_gadget(SPECS[name], timeout_ms=120_000, drop=drop)
    assert r.status == "sat", r.status
    assert r.model["outputs"] != r.model["reference"]


def test_z3_modular_findings_reproduce() -> None:
    # F1: the product realisation of a variable bound wraps in a 40-bit field, not with an ideal realisation
    assert z3_gadget(SPECS["leading_bit"], timeout_ms=120_000, modulus=P40).status == "sat"
    assert z3_gadget(SPECS["leading_bit"], timeout_ms=120_000, modulus=P40, ideal_bounds=True).status == "unsat"
    assert z3_gadget(SPECS["leading_bit"], timeout_ms=120_000, modulus=GOLDILOCKS).status == "unsat"
    # F2: without the tight quotient bound a 31-bit field accepts a wrapped quotient (over the integers it is implied)
    assert z3_gadget(SPECS["align_product"], timeout_ms=120_000, modulus=M31, ideal_bounds=True, drop=("quot",)).status == "sat"
    assert z3_gadget(SPECS["align_product"], timeout_ms=120_000, drop=("quot",)).status == "unsat"
    # F3, F4: the normalisation division and the 32-bit words do not fit a 31-bit field
    assert z3_gadget(SPECS["normalise"], timeout_ms=120_000, modulus=M31, ideal_bounds=True).status == "sat"
    assert z3_gadget(SPECS["fp32_decode"], timeout_ms=120_000, modulus=M31).status == "sat"
