"""checker v2 (``verity_numerical.checker.v2``): the same relation as v1 with the lookup-centred witness.
Completeness against silicon (structured, random, golden, the K=1536 vector), the state-chained VU equals the
word-chained one, soundness by mutation of every v2 hint kind, the symbolic tables agree with their rows, and the
census: fewer multiplications, identities and lookups than v1."""
from __future__ import annotations

import copy
import random

import pytest
from conftest import (
    bf16,
    f32,
    golden_records,
    k1536_vector,
    random_cases,
    structured_cases,
)
from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot, tc_dot_chain
from verity_numerical.checker import REAL, Reject
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.step import step_output as step_output_v1
from verity_numerical.checker.v2 import step as v2
from verity_numerical.checker.v2.params import V2, check_widths
from verity_numerical.checker.v2.search import SPECS_V2, table_selfcheck
from verity_numerical.checker.v2.tables import all_tables
from verity_numerical.security.census import (
    census_unit,
    census_unit_v2,
    census_vu,
    census_vu_v2,
)

PIPE = AMPERE_BF16_M16N8K16
P = REAL
ONE = bf16(0, 127, 0)


def _agree(c: int, a: list[int], b: list[int]) -> int:
    expected = tc_dot(PIPE, c, a, b)
    h = v2.gen_step(P, c, a, b)
    v2.check_step(P, c, a, b, expected, h)
    assert v2.step_output(P, c, a, b) == expected == step_output_v1(P, c, a, b)
    return expected


# ---- completeness -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,c,a,b", structured_cases(), ids=[n for n, *_ in structured_cases()])
def test_structured_case_agrees_with_silicon(name, c, a, b) -> None:
    _agree(c, a, b)


def test_random_finite_inputs_agree_with_silicon() -> None:
    cases = random_cases(2000)
    assert [(c, a, b) for c, a, b in cases if v2.step_output(P, c, a, b) != tc_dot(PIPE, c, a, b)] == []
    for c, a, b in cases[:200]:
        _agree(c, a, b)


def test_every_all_finite_golden_record_reproduces() -> None:
    records = golden_records()
    assert len(records) == 360
    assert [f for f, c, a, b, d in records if _agree(c, a, b) != d] == []


def test_vu_reproduces_the_k1536_conformance_vector_state_chained_and_word_chained() -> None:
    a, b, y16 = k1536_vector()
    assert f32_to_bf16_word(tc_dot_chain(PIPE, 0, a, b)) == y16
    h = v2.gen_vu(P, a, b)
    assert len(h["steps"]) == 96 and "acc" not in h["steps"][0] and h["steps"][0]["groups"][0]["acc_split"] is None
    v2.check_vu(P, a, b, y16, h)
    assert v2.vu_output(P, a, b) == y16
    hw = v2.gen_vu_words(P, a, b)
    assert v2.check_vu_words_expr(IntCtx(), P, a, b, hw) == y16


def test_vu_rejects_an_infinite_intermediate_accumulator_but_casts_a_final_infinity() -> None:
    big = [bf16(0, 254, 0x7F)] * 16
    one = [ONE] * 16
    with pytest.raises(Reject, match="saturation before the last unit"):
        v2.gen_vu(P, big + one, big + one)
    # a forged witness for the same instance (hints of some other chain) cannot pass the per-unit finiteness identity
    h = v2.gen_vu(P, one + one, one + one)
    assert not v2.accepts_vu(P, big + one, big + one, 0x7F80, h)
    v2.check_vu(P, one + big, one + big, 0x7F80, v2.gen_vu(P, one + big, one + big))


def test_hint_count_per_unit() -> None:
    c, a, b = f32(0, 130, 0x123456), [bf16(0, 128, 3)] * 16, [bf16(1, 126, 5)] * 16
    h = v2.gen_step(P, c, a, b)
    flat = 3 + sum(1 + 2 + 3 for _ in h["groups"])
    assert set(h) == {"acc", "groups"} and set(h["acc"]) == {"hdr", "f", "z"} and flat == 15


# ---- soundness by mutation ----------------------------------------------------------------------------------------------

def _case(c: int, pairs: list[tuple[int, int]]):
    pairs = pairs + [(0, 0)] * (16 - len(pairs))
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    return c, a, b, tc_dot(PIPE, c, a, b), v2.gen_step(P, c, a, b)


def _rejects(x, mutate, *predicates: str) -> None:
    """The mutated witness is rejected at one of the named predicates (the first violated one in check order)."""
    c, a, b, y, h = x
    v2.check_step(P, c, a, b, y, h)
    bad = copy.deepcopy(h)
    mutate(bad)
    with pytest.raises(Reject) as info:
        v2.check_step(P, c, a, b, y, bad)
    assert any(pred in str(info.value) for pred in predicates), str(info.value)


NONZERO = _case(f32(0, 130, 0x123456), [(bf16(0, 128, 3), bf16(1, 126, 5))] * 16)
CANCEL = _case(0, [(ONE, ONE), (ONE, bf16(1, 127, 0))] * 8)
SHIFTED = _case(f32(0, 140, 0), [(bf16(0, 121, 0x55), bf16(0, 121, 0x33))] * 16)
WIDE = _case(f32(0, 130, 0x7FFFFF), [(bf16(0, 128, 0x7F), bf16(0, 129, 0x7F))] * 16)


def test_state_decode_wrong_header_or_fraction_is_rejected() -> None:
    _rejects(NONZERO, lambda h: h["acc"].__setitem__("hdr", h["acc"]["hdr"] + 1), "acc.word")
    _rejects(NONZERO, lambda h: h["acc"].__setitem__("f", h["acc"]["f"] + 2**23), "acc.f")
    _rejects(NONZERO, lambda h: h["acc"].__setitem__("hdr", 0xFF), "acc.hdr")          # a NaN/Inf header has no row
    _rejects(NONZERO, lambda h: h["acc"].__setitem__("z", 1), "acc.z0")


def test_group_max_excessive_or_deficient_E_is_rejected() -> None:
    # E + 1: every term is halved, so the honest sum split no longer matches (sum.split), and were the sum hints
    # re-derived, `attained` would refuse (no candidate reaches E): the z3 query `without attained` is SAT
    _rejects(NONZERO, lambda h: h["groups"][0].__setitem__("E", h["groups"][0]["E"] + 1), "max.attained", "sum.split")
    _rejects(NONZERO, lambda h: h["groups"][0].__setitem__("E", h["groups"][0]["E"] - 1), "align")    # a negative shift has no row
    _rejects(SHIFTED, lambda h: h["groups"][1].__setitem__("E", h["groups"][1]["E"] + 3), "attained", "sum.split")
    # E too large *with* the sum hints re-derived for the forged E: only `attained` stands in the way
    c, a, b, y, h = NONZERO
    bad = _forge_E(c, a, b, h, 0, +1)
    with pytest.raises(Reject, match="attained"):
        v2.check_step(P, c, a, b, y, bad)


def _forge_E(c, a, b, h, g: int, delta: int) -> dict:
    """The witness with group ``g``'s E shifted by ``delta`` and every downstream hint of that group recomputed
    honestly for the forged E (the aligned terms are what the tables give for it)."""
    from verity_numerical.checker.v2.gadgets import (
        OpDecode,
        Product,
        SignedSum,
        StateDecode,
    )
    from verity_numerical.checker.v2.tables import (
        shift_table,
        sshift_hi_table,
        sshift_lo_table,
    )

    assert g == 0, "the forged group must be the first (its incoming state is the decoded word)"
    bad = copy.deepcopy(h)
    E = bad["groups"][g]["E"] + delta
    bad["groups"][g]["E"] = E
    ctx = IntCtx()
    s_c, e_c, _M_c, _ = StateDecode.check(ctx, P, c, h["acc"])
    R, DT = 2 * V2.dt_rows(P), V2.dt(P)
    sp = bad["groups"][g]["acc_split"]
    d0 = min(E - e_c, DT)
    terms = [sshift_hi_table(P).row(sp["M_hi"] * R + 2 * d0 + s_c)[0] + sshift_lo_table(P).row(sp["M_lo"] * R + 2 * d0 + s_c)[0]]
    for i in range(P.groups[g]):
        u, e, s4 = Product.check(ctx, P, OpDecode.check(ctx, P, a[i]), OpDecode.check(ctx, P, b[i]))
        terms.append(shift_table(P).row(u * R + 2 * min(E - e, DT) + int(s4 in (1, 2)))[0])
    bad["groups"][g]["sum"] = SignedSum.gen(P, terms)
    return bad


def test_state_split_and_sum_split_forgeries_are_rejected() -> None:
    def carry(h):
        s = h["groups"][0]["acc_split"]
        s["M_lo"] += 2 ** V2.mb(P)
        s["M_hi"] -= 1
    _rejects(NONZERO, carry, "sshift.lo")                                # M_lo out of its half: off the SSHIFT_LO table

    def vsplit(h):
        s = h["groups"][0]["sum"]
        s["v_lo"] += 2 ** V2.vb(P)
        s["v_hi"] -= 1
    _rejects(WIDE, vsplit, "lead_lo")                                    # v_lo out of its half: off the LEAD table
    _rejects(WIDE, lambda h: h["groups"][0]["sum"].__setitem__("sgn", 1 - h["groups"][0]["sum"]["sgn"]), "sum.split")
    _rejects(NONZERO, lambda h: h["groups"][0]["sum"].__setitem__("v_lo", h["groups"][0]["sum"]["v_lo"] ^ 1), "sum.split")


def test_sign_of_a_zero_sum_is_free_but_harmless() -> None:
    c, a, b, y, h = CANCEL
    assert y == 0
    alt = copy.deepcopy(h)
    alt["groups"][0]["sum"]["sgn"] ^= 1                                  # S = 0: the sign hint is unconstrained ...
    v2.check_step(P, c, a, b, y, alt)                                    # ... and the output word is the same +0


# ---- tables ----------------------------------------------------------------------------------------------------------

def test_widths_and_tables() -> None:
    check_widths(P)
    assert V2.e_zero(P) + P.e_max < P.floor
    names = [t.name for t in all_tables(P)]
    assert names == ["T_OP", "T_HDR", "ALIGN4", "SHIFT", "SSHIFT_HI", "SSHIFT_LO", "LEAD", "LEADNORM", "TNORM"]
    for r in table_selfcheck(P, full_below=600, sample=64, seed=1):
        assert r["mismatches"] == 0, r


def test_op_table_has_no_row_for_nan_or_inf() -> None:
    from verity_numerical.checker.v2.tables import op_table
    t = op_table(P)
    for w in (0x7F80, 0xFF80, 0x7FC0, 0x7F81):
        with pytest.raises(KeyError):
            t.row(w)
    assert t.row(0) == (0, 0, V2.e_zero(P)) and t.row(0x8000) == (1, 0, V2.e_zero(P))
    assert t.row(ONE) == (0, 128, 0)


def test_specs_generate_accepted_hints_on_random_inputs() -> None:
    rng = random.Random(3)
    for name, spec in SPECS_V2.items():
        c = spec.contract(P)
        for _ in range(20):
            x = {k: rng.randrange(lo, hi) for k, (lo, hi) in c.inputs.items()}
            if spec.inputs_of is not None:
                x = spec.inputs_of(P, x)
            if not spec.admissible(P, x):
                continue
            outs = spec.call(IntCtx(), P, x, spec.gen(P, x))
            assert tuple(outs) == tuple(spec.ref(P, x)), (name, x)


# ---- census ----------------------------------------------------------------------------------------------------------

def test_v2_unit_is_smaller_than_v1_on_every_count() -> None:
    u1, u2 = census_unit(P), census_unit_v2(P)
    assert (u1.mul, u1.lin_total, u1.lookup) == (358, 137, 306)
    assert u2.mul < u1.mul * 0.15 and u2.lin_total < u1.lin_total * 0.15 and u2.lookup < u1.lookup * 0.35
    assert u2.columns_hints == 15 and u2.depth <= 3
    assert set(u2.by_gadget) >= {"op_decode", "group_max", "product_shift", "state_shift", "signed_sum", "normalise"}


def test_v2_vu_is_smaller_than_v1() -> None:
    v1, vv2 = census_vu(1536, P), census_vu_v2(1536, P)
    assert vv2.mul < v1.mul / 10 and vv2.lookup < v1.lookup / 3 and vv2.columns_committed_a < v1.columns_committed_a


def test_field_reading_is_wrap_free_for_the_hint_splitting_gadgets() -> None:
    """Over Goldilocks and M31 no congruence of ``state_shift``, ``signed_sum`` or ``normalise`` can hold with a
    non-zero multiple of p (so the integer UNSATs transfer to the field).  ``state_shift`` needs the halves' own
    range checks for this: keyed through M_hi R + 2dt + s alone it is SAT over M31 (z3, 53 s on vy-cpu2)."""
    from verity_numerical.checker.search import GOLDILOCKS, M31
    from verity_numerical.checker.v2.search import wrapfree_queries_v2

    rows = wrapfree_queries_v2(REAL, timeout_ms=20_000, names=("state_shift", "signed_sum", "normalise"),
                               moduli=((GOLDILOCKS, "F_goldilocks"), (M31, "F_m31")))
    assert [r.status for r in rows] == ["unsat"] * 6, [(r.gadget, r.query, r.status) for r in rows]
