"""Soundness by mutation: for every gadget, a witness the checker must reject, and the predicate that rejects it.
Each mutation keeps the public (x, y) and changes only hints -- the excessive exponent, the false selector, the
wrong remainder, the wrong leading bit, the sign flip and the wrapped reconstruction the handoff names, per gadget."""
from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
from conftest import bf16, f32

from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity_numerical.checker import REAL, Reject, check_step, check_vu, gen_step, gen_vu
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.tables import align_table, norm_table, pow_table

P = REAL
ONE = bf16(0, 127, 0)


def case(c: int, pairs: list[tuple[int, int]]) -> tuple[int, list[int], list[int], int, dict]:
    pairs = pairs + [(0, 0)] * (16 - len(pairs))
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    y = tc_dot(AMPERE_BF16_M16N8K16, c, a, b)
    return c, a, b, y, gen_step(P, c, a, b)


def rejects(x: tuple, mutate: Callable[[dict], None], predicate: str) -> None:
    c, a, b, y, h = x
    check_step(P, c, a, b, y, h)                    # the honest witness is accepted ...
    bad = copy.deepcopy(h)
    mutate(bad)
    with pytest.raises(Reject) as info:             # ... and the mutated one is not, at the named predicate
        check_step(P, c, a, b, y, bad)
    assert predicate in str(info.value), str(info.value)


NONZERO = case(f32(0, 130, 0x123456), [(bf16(0, 128, 3), bf16(1, 126, 5))] * 16)
WITH_ZERO_PRODUCT = case(f32(0, 130, 0), [(ONE, ONE), (0, ONE)] + [(ONE, ONE)] * 14)
SUBNORMAL_STATE = case(f32(0, 0, 0x2A), [(ONE, bf16(0, 20, 0))])
CANCEL = case(0, [(ONE, ONE), (ONE, bf16(1, 127, 0))] * 8)
SHIFTED = case(f32(0, 140, 0), [(bf16(0, 121, 0x55), bf16(0, 121, 0x33))] * 16)     # products 25 below the state
WIDE = case(f32(0, 130, 0x7FFFFF), [(bf16(0, 128, 0x7F), bf16(0, 129, 0x7F))] * 16)  # |S| has 30 bits: a right shift
FLUSH_STATE = case(f32(0, 10, 0x7), [(bf16(0, 200, 1), bf16(0, 200, 1))])                # the state is shifted by >= W
OVERFLOW = case(0, [(bf16(0, 254, 0x7F), bf16(0, 254, 0x7F))] * 8)


# ---- bf16 decode ----------------------------------------------------------------------------------------------------

def test_bf16_decode_wrapped_reconstruction_is_rejected() -> None:
    def wrap(h):                                    # same word, f carrying 2^7 out of range
        h["a"][0]["f"] += 128
        h["a"][0]["t"] -= 1
    rejects(NONZERO, wrap, "a[0].f")


def test_bf16_decode_sign_flip_is_rejected() -> None:
    def flip(h):
        h["b"][0]["s"] ^= 1
    rejects(NONZERO, flip, "b[0].word")


def test_bf16_decode_false_nonzero_flag_is_rejected_both_ways() -> None:
    def clear(h):
        h["a"][0]["nz"] = 0                         # t != 0 but nz = 0
    rejects(NONZERO, clear, "a[0].nz0")

    def set_(h):
        h["a"][1]["nz"] = 1                         # t = 0 (a zero word) but nz = 1
    rejects(WITH_ZERO_PRODUCT, set_, "a[1].nz1")


def test_bf16_decode_rejects_non_finite_operands() -> None:
    c, a, b, y, h = NONZERO
    for word in (0x7F80, 0xFF80, 0x7FC0, 0x7F81):
        bad_a = [word] + a[1:]
        with pytest.raises(Reject, match=r"a\[0\]\.t"):
            check_step(P, c, bad_a, b, y, gen_step_unchecked(c, bad_a, b, h))


def gen_step_unchecked(c, a, b, h):
    """The honest hints with the operand decode of a[0] recomputed for a non-finite word (t = 255)."""
    from verity_numerical.checker.gadgets import Bf16Decode
    bad = copy.deepcopy(h)
    bad["a"][0] = Bf16Decode.gen(P, a[0])
    return bad


# ---- fp32 decode ----------------------------------------------------------------------------------------------------

def test_fp32_decode_wrapped_reconstruction_and_flags_are_rejected() -> None:
    def wrap(h):
        h["acc"]["f"] += 1 << 23
        h["acc"]["t"] -= 1
    rejects(NONZERO, wrap, "acc.f")

    def flip_sign(h):
        h["acc"]["s"] ^= 1
    rejects(NONZERO, flip_sign, "acc.word")

    def false_zero(h):
        h["acc"]["z"] = 1
    rejects(NONZERO, false_zero, "acc.z0")

    def false_nonzero(h):
        h["acc"]["z"] = 0
    rejects(CANCEL, false_nonzero, "acc.z1")

    def subnormal_claims_hidden_bit(h):
        h["acc"]["nz"] = 1
    rejects(SUBNORMAL_STATE, subnormal_claims_hidden_bit, "acc.nz1")


def test_fp32_decode_rejects_an_infinite_accumulator() -> None:
    c, a, b, y, h = NONZERO
    with pytest.raises(Reject, match="acc.t"):
        gen_step(P, 0x7F800000, a, b)


# ---- product --------------------------------------------------------------------------------------------------------

def test_product_zero_flag_is_pinned() -> None:
    def claim_zero(h):
        h["prod"][0]["z"] = 1
    rejects(NONZERO, claim_zero, "prod[0].z0")

    def claim_nonzero(h):
        h["prod"][1]["z"] = 0
    rejects(WITH_ZERO_PRODUCT, claim_nonzero, "prod[1].z1")


# ---- group maximum --------------------------------------------------------------------------------------------------

def test_group_max_excessive_exponent_is_rejected() -> None:
    def excessive(h):
        h["groups"][0]["max"]["E"] += 1             # above every candidate: attained by nothing
    rejects(NONZERO, excessive, "g0.max.attained")

    def excessive_with_floor_selector(h):
        g = h["groups"][1]["max"]
        g["E"] += 3
        g["sel"] = [0] * (len(g["sel"]) - 1) + [1]
    rejects(NONZERO, excessive_with_floor_selector, "g1.max.attained")


def test_group_max_false_selector_is_rejected() -> None:
    c, a, b, y, h = SHIFTED                          # products are 25 below the state: the state attains E
    g = h["groups"][0]["max"]
    assert g["sel"][0] == 1

    def select_a_product(hh):
        gg = hh["groups"][0]["max"]
        gg["sel"] = [0, 1] + [0] * (len(gg["sel"]) - 2)
        gg["E"] -= 25                               # consistent with the false selector; the state's d goes negative
    rejects(SHIFTED, select_a_product, "g0.max.d[0]")

    def two_selectors(hh):
        hh["groups"][0]["max"]["sel"][1] = 1
    rejects(SHIFTED, two_selectors, "g0.max.onehot")

    def below_floor(hh):
        gg = hh["groups"][0]["max"]
        gg["E"] = P.floor - 1
    rejects(SHIFTED, below_floor, "g0.max.E")


# ---- alignment ------------------------------------------------------------------------------------------------------

def test_align_state_wrong_remainder_is_rejected() -> None:
    c, a, b, y, h = SHIFTED
    align = h["groups"][0]["acc_align"]
    assert align["r"] == 0                          # d = 0: the state is unshifted, so r must be 0

    def r_plus_one(hh):
        hh["groups"][0]["acc_align"]["r"] += 1
    rejects(SHIFTED, r_plus_one, "g0.align_state.div")

    def wrapped_remainder(hh):                      # q - 1 and r + 2^d: the identity holds, r is out of its window
        al = hh["groups"][0]["acc_align"]
        al["q"] -= 1
        al["r"] += 1
    rejects(SHIFTED, wrapped_remainder, "g0.align_state.rem")


def test_align_state_flushed_state_cannot_claim_a_quotient() -> None:
    c, a, b, y, h = FLUSH_STATE
    d = h["groups"][0]["max"]["E"] - (10 - 127)
    assert d >= P.acc_bits
    assert h["groups"][0]["acc_align"]["q"] == 0

    def claim_one(hh):
        al = hh["groups"][0]["acc_align"]
        al["q"] = 1
        al["r"] -= 2**P.acc_bits                    # keeps q * 2^W + r = P but r < 0
    rejects(FLUSH_STATE, claim_one, "g0.align_state.r")

    def claim_one_keep_r(hh):
        hh["groups"][0]["acc_align"]["q"] = 1
    rejects(FLUSH_STATE, claim_one_keep_r, "g0.align_state.div")


def test_align_product_wrong_remainder_and_wrapped_quotient_are_rejected() -> None:
    c, a, b, y, h = SHIFTED
    al = h["groups"][0]["prod_align"][0]
    d = h["groups"][0]["max"]["E"] - (2 * (121 - 127))
    assert P.prod_shift < d < P.prod_bits           # the middle regime: u is divided by 2^(d - prod_shift)
    sr = align_table(P).row(d)[1]
    assert al["r"] < sr

    def wrong_r(hh):
        hh["groups"][0]["prod_align"][0]["r"] += 1
    rejects(SHIFTED, wrong_r, "g0.align_prod[0].div")

    def wrapped(hh):                                # q + 1, r - sr: identity holds, r negative
        a0 = hh["groups"][0]["prod_align"][0]
        a0["q"] += 1
        a0["r"] -= sr
    rejects(SHIFTED, wrapped, "g0.align_prod[0].r")

    def under_quotient(hh):                         # q - 1, r + sr: identity holds, r beyond its window
        a0 = hh["groups"][0]["prod_align"][0]
        a0["q"] -= 1
        a0["r"] += sr
    rejects(SHIFTED, under_quotient, "g0.align_prod[0].rem")


def test_align_product_unshifted_product_has_no_remainder() -> None:
    c, a, b, y, h = NONZERO                          # d <= prod_shift: sr = 1, r = 0, q = u
    assert h["groups"][0]["prod_align"][0]["r"] == 0

    def steal(hh):
        a0 = hh["groups"][0]["prod_align"][0]
        a0["q"] -= 1
        a0["r"] += 1
    rejects(NONZERO, steal, "g0.align_prod[0].rem")


# ---- signed sum -----------------------------------------------------------------------------------------------------

def test_signed_sum_sign_flip_and_zero_flag_are_rejected() -> None:
    def flip(h):
        h["groups"][0]["sum"]["sgn"] ^= 1
    rejects(NONZERO, flip, "g0.sum.v")

    def claim_zero(h):
        h["groups"][0]["sum"]["z"] = 1
    rejects(NONZERO, claim_zero, "g0.sum.z0")

    def claim_nonzero(h):
        h["groups"][0]["sum"]["z"] = 0
    rejects(CANCEL, claim_nonzero, "g0.sum.z1")

    def negative_zero(h):
        h["groups"][0]["sum"]["sgn"] = 1
    rejects(CANCEL, negative_zero, "g0.sum.zero_sign")


# ---- leading bit ----------------------------------------------------------------------------------------------------

def test_leading_bit_off_by_one_is_rejected() -> None:
    for delta in (-1, 1):
        def shift(h, delta=delta):
            h["groups"][0]["lead"]["l"] += delta
        rejects(NONZERO, shift, "g0.lead.window")

    def zero_claims_a_bit(h):
        h["groups"][0]["lead"]["l"] = 1
    rejects(CANCEL, zero_claims_a_bit, "g0.lead.window")


# ---- normalisation --------------------------------------------------------------------------------------------------

def test_normalise_clamp_flag_and_quotient_are_pinned() -> None:
    def flip_clamp(h):
        h["groups"][0]["norm"]["clamp"] ^= 1
    rejects(NONZERO, flip_clamp, "g0.norm.clamp_proof")
    rejects(SUBNORMAL_STATE, flip_clamp, "g0.norm.clamp_proof")

    c, a, b, y, h = WIDE
    E, l = h["groups"][0]["max"]["E"], h["groups"][0]["lead"]["l"]
    assert l == 30
    k = E - max(E + l - P.width, P.e_min) + P.acc_sig - P.width
    mul, div, _ = norm_table(P).row(k)
    assert div > 1 and mul == 1                     # a right shift: the remainder has a window

    def m_plus_one(hh):
        hh["groups"][0]["norm"]["M"] += 1
    rejects(WIDE, m_plus_one, "g0.norm.div")
    rejects(NONZERO, m_plus_one, "g0.norm.div")     # and for a left shift (k > 0), where r must be 0

    def wrapped(hh):                                # M - 1 and r + div: identity holds, r beyond its window
        nn = hh["groups"][0]["norm"]
        nn["M"] -= 1
        nn["r"] += div
    rejects(WIDE, wrapped, "g0.norm.rem")

    def left_shift_steals_a_remainder(hh):          # k > 0: v * mul = M + r with r < 1
        nn = hh["groups"][0]["norm"]
        nn["M"] -= 1
        nn["r"] += 1
    rejects(NONZERO, left_shift_steals_a_remainder, "g0.norm.rem")


# ---- state out ------------------------------------------------------------------------------------------------------

def test_state_out_flags_are_pinned() -> None:
    def claim_zero(h):
        h["groups"][1]["out"]["z"] = 1
    rejects(NONZERO, claim_zero, "g1.out.z0")

    def claim_nonzero(h):
        h["groups"][1]["out"]["z"] = 0
    rejects(CANCEL, claim_nonzero, "g1.out.z1")

    def claim_overflow(h):
        h["groups"][1]["out"]["ovf"] = 1
    rejects(NONZERO, claim_overflow, "g1.out.ovf_proof")

    def deny_overflow(h):
        h["groups"][0]["out"]["ovf"] = 0
    rejects(OVERFLOW, deny_overflow, "g0.out.ovf_proof")


# ---- pack and the output --------------------------------------------------------------------------------------------

def test_pack_wrapped_reconstruction_and_output_sign_flip_are_rejected() -> None:
    def wrap(h):
        h["pack"]["hidden"] ^= 1
        h["pack"]["low"] += (1 << 23) if h["pack"]["hidden"] == 0 else -(1 << 23)
    rejects(NONZERO, wrap, "pack.low")

    def hidden_only(h):
        h["pack"]["hidden"] ^= 1
    rejects(NONZERO, hidden_only, "pack.M")

    c, a, b, y, h = NONZERO
    with pytest.raises(Reject, match="^y:"):
        check_step(P, c, a, b, y ^ (1 << 31), h)
    with pytest.raises(Reject, match="^y:"):
        check_step(P, c, a, b, y + 1, h)


def test_overflow_output_is_the_infinity_and_nothing_else() -> None:
    c, a, b, y, h = OVERFLOW
    assert y == 0x7F800000
    for other in (0x7F7FFFFF, 0xFF800000, 0):
        with pytest.raises(Reject, match="^y:"):
            check_step(P, c, a, b, other, h)


# ---- epilogue -------------------------------------------------------------------------------------------------------

def test_epilogue_mutations_are_rejected() -> None:
    a = [bf16(0, 127, 0x40)] * 16
    b = [bf16(0, 127, 0x01)] * 16
    h = gen_vu(P, a, b)
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word
    from verity.ml.circuits.silicon.silicon import tc_dot_chain
    y16 = f32_to_bf16_word(tc_dot_chain(AMPERE_BF16_M16N8K16, 0, a, b))
    check_vu(P, a, b, y16, h)

    def run(mutate, predicate):
        bad = copy.deepcopy(h)
        mutate(bad["epilogue"])
        with pytest.raises(Reject) as info:
            check_vu(P, a, b, y16, bad)
        assert predicate in str(info.value), str(info.value)

    run(lambda e: e.update(lo=e["lo"] + (1 << 16), hi=e["hi"] - 1), "cast.lo")       # wrapped split
    run(lambda e: e.update(carry=e["carry"] ^ 1), "cast.round")                        # rounding direction
    run(lambda e: e.update(lsb=e["lsb"] ^ 1), "cast.lsb_split")                        # ties-to-even bit
    run(lambda e: e.update(rem=e["rem"] + 1), "cast.round")
    run(lambda e: e.update(hi=e["hi"] + 1), "cast.split")
    with pytest.raises(Reject, match="^y16:"):
        check_vu(P, a, b, y16 ^ 1, h)


def test_epilogue_hints_are_the_rn_even_decomposition() -> None:
    word = 0x3F80_8001                                          # above the tie: rounds up
    h = Epilogue.gen(P, word)
    assert (h["hi"], h["lo"], h["lsb"], h["carry"]) == (0x3F80, 0x8001, 0, 1)
    word = 0x3F80_8000                                          # exact tie with even hi: stays
    h = Epilogue.gen(P, word)
    assert (h["lsb"], h["carry"]) == (0, 0)
    word = 0x3F81_8000                                          # exact tie with odd hi: rounds to even
    h = Epilogue.gen(P, word)
    assert (h["lsb"], h["carry"]) == (1, 1)


# ---- structure ------------------------------------------------------------------------------------------------------

def test_missing_or_misshapen_witness_is_rejected() -> None:
    c, a, b, y, h = NONZERO
    bad = copy.deepcopy(h)
    bad["groups"].pop()
    with pytest.raises(Reject):
        check_step(P, c, a, b, y, bad)
    bad = copy.deepcopy(h)
    bad["groups"][0]["prod_align"].pop()
    with pytest.raises(Reject):
        check_step(P, c, a, b, y, bad)
    bad = copy.deepcopy(h)
    bad["groups"][0]["max"]["sel"].append(0)
    with pytest.raises(Reject):
        check_step(P, c, a, b, y, bad)
    bad = copy.deepcopy(h)
    bad["acc"]["s"] = True                                      # a bool is not a bit of the integer model
    with pytest.raises(Reject):
        check_step(P, c, a, b, y, bad)


def test_every_hint_is_range_checked_against_the_contract() -> None:
    """Any single hint pushed just outside its contract range is rejected (the checker enforces the contract)."""
    from verity_numerical.checker.gadgets import (
        AlignProduct, AlignState, Bf16Decode, Fp32Decode, GroupMax, LeadingBit, Normalise, Pack, Product,
        SignedSum, StateOut,
    )
    c, a, b, y, h = NONZERO
    places = [
        (Fp32Decode.contract(P), lambda hh: hh["acc"]),
        (Bf16Decode.contract(P), lambda hh: hh["a"][3]),
        (Product.contract(P), lambda hh: hh["prod"][3]),
        (AlignState.contract(P), lambda hh: hh["groups"][1]["acc_align"]),
        (AlignProduct.contract(P), lambda hh: hh["groups"][1]["prod_align"][2]),
        (SignedSum.contract(P), lambda hh: hh["groups"][1]["sum"]),
        (LeadingBit.contract(P), lambda hh: hh["groups"][1]["lead"]),
        (Normalise.contract(P), lambda hh: hh["groups"][1]["norm"]),
        (StateOut.contract(P), lambda hh: hh["groups"][1]["out"]),
        (Pack.contract(P), lambda hh: hh["pack"]),
    ]
    for contract, locate in places:
        for name, (lo, hi) in contract.hints.items():
            for value in (lo - 1, hi):
                bad = copy.deepcopy(h)
                locate(bad)[name] = value
                with pytest.raises(Reject):
                    check_step(P, c, a, b, y, bad)
    gm = GroupMax.contract(P)
    for i in range(len(h["groups"][0]["max"]["sel"])):
        for value in (-1, 2):
            bad = copy.deepcopy(h)
            bad["groups"][0]["max"]["sel"][i] = value
            with pytest.raises(Reject):
                check_step(P, c, a, b, y, bad)
    for value in (gm.hints["E"][0] - 1, gm.hints["E"][1]):
        bad = copy.deepcopy(h)
        bad["groups"][0]["max"]["E"] = value
        with pytest.raises(Reject):
            check_step(P, c, a, b, y, bad)
