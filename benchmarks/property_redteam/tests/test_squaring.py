"""Squaring mod p1024 in 6b's model (`squaring.py`) and the free codec word (`free_codec.py`): what they cost and that they
compute what they claim. The costs are the round-4 result against `EncodeChainHardness`'s `e = 119` a squaring."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import free_codec as FC  # noqa: E402
import squaring as S  # noqa: E402

# Security/Proofs/Accounting/Gates.lean's `codecRows` (6b, at d6df95f24a): (n, c, cs, K, τ, i, m, encodeWord).
CODEC_ROWS = [
    (8, 5, [17, 200, 99], 248, 3, 5, 219, 234), (8, 5, [92, 7], 253, 18, 54, 195, 147),
    (8, 5, [44, 209], 252, 132, 41, 181, 166), (8, 5, [139, 0], 254, 232, 36, 68, 225),
    (8, 5, [56, 35], 12, 69, 52, 101, 18), (8, 5, [88, 65], 93, 168, 12, 225, 4),
    (8, 5, [103, 10, 98], 252, 50, 51, 47, 10), (8, 5, [156, 202], 253, 47, 41, 11, 126),
    (16, 17, [26622, 53477], 34513, 22959, 218370, 28017, 16283),
    (16, 17, [27420, 1802, 17095], 22236, 6729, 817333, 53283, 48655),
    (16, 17, [60688, 23696, 15300], 349, 10049, 806969, 33473, 64012),
]


def _build(fn, *args):
    b = S.Builder()
    a = [b.inp(i, 0, 255) for i in range(S.N_LIMBS)]
    out, extra = fn(b, a, *args)
    return b, out, extra


def _run(b, out, a):
    vals, window, hold = b.evaluate(dict(enumerate(S.limbs(a))))
    limbs = [vals[o] for o in out]
    return vals, limbs, window, hold


def _restricted_and_correct(b, out, a):
    assert S.proven(b.gates) == [] and S.operand_bits(128, b.gates)
    _, limbs, window, hold = _run(b, out, a)
    assert window and hold and all(0 <= v <= 255 for v in limbs)
    return S.output_value(limbs) % S.P == a * a % S.P


def test_a_one_word_constant_gather_is_a_free_comparator_and_the_fix_charges_it():
    b = S.Builder()
    r = b.inp(0, 0, 1000)
    bit = S.free_bit(b, r, 5)
    assert b.lean_units == 0 and b.fixed_units == 1
    assert S.budgeted(b.gates) == 0 and S.budgeted(b.gates, fixed=True) == 1
    for v in (0, 31, 32, 1000):
        assert b.evaluate({0: v})[0][bit] == (v >= 32)


def test_the_free_squaring_costs_nothing_at_p1024_and_chains():
    rng = random.Random(20261017)
    b = S.Builder()
    a = [b.inp(i, 0, 255) for i in range(S.N_LIMBS)]
    once, c1 = S.free_square(b, a)
    fixed_once = b.fixed_units
    twice, c2 = S.free_square(b, once)
    assert not any(g.op.kind == "budgeted" for g in b.gates)
    assert S.restricted(128, b.gates)
    assert not any(g.op.kind == "input" and g.op.binding == "run" for g in b.gates)
    assert b.lean_units == 0
    assert fixed_once == 6014 and b.fixed_units == 2 * fixed_once
    # Renormalization ⟨1024, 105⟩ d₀ = 4, c = 0, t = 64 at ε = 2^−40 (Guarantees/Gates.lean): its conclusion
    # ε·2^(8·128) ≤ 2^(8·(budgeted + d₀ + c)) fails for this circuit, whose inputs are a's bytes (coin) and outputs u8.
    assert not 2 ** (8 * S.N_LIMBS) <= 2 ** 40 * 2 ** (8 * (b.lean_units + 4))
    for x in (rng.randrange(S.P), S.P - 1, 0):
        vals, window, hold = b.evaluate(dict(enumerate(S.limbs(x))))
        assert window and hold and vals[c1] == 0 and vals[c2] == 0
        for gates, want in ((once, x * x), (twice, pow(x, 4))):
            limbs = [vals[o] for o in gates]
            assert all(0 <= v <= 255 for v in limbs) and S.output_value(limbs) % S.P == want % S.P


def test_ripple_and_lookahead_reproduce_6bs_counts():
    rng = random.Random(20261017)
    a = rng.randrange(S.P // 4)
    for fn, args, units in ((S.ripple_square, (), 512), (S.lookahead_square, (3,), 342)):
        b, out, carry = _build(fn, *args)
        assert b.lean_units == units == b.fixed_units
        assert b.products == S.products(b.gates) == S.N_LIMBS ** 2 and b.nonlinear_units == units + S.N_LIMBS ** 2
        assert b.nonlinear_units == S.nonlinear(b.gates) > S.e_per_squaring(0)
        assert _restricted_and_correct(b, out, a)
        assert _run(b, out, a)[0][carry] == 0


def test_the_counted_constructions_cost_what_the_report_says():
    rng = random.Random(7)
    inputs = [rng.randrange(1 << 1024) for _ in range(4)] + [S.P - 1, 0]
    for fn, units in ((S.merged_square, 297), (S.onepass_square, 168), (S.seam_square, 160)):
        b, out, _ = _build(fn)
        assert b.lean_units == units == b.fixed_units and b.nonlinear_units == units + S.N_LIMBS ** 2
        assert max(sum(b.gates[j].bits for j in g.args) for g in b.gates if g.op.kind == "budgeted") <= 128
        for a in inputs:
            if fn is S.seam_square and S.seam_fails(a):
                continue
            assert _restricted_and_correct(b, out, a)


def test_the_greedy_pass_takes_four_columns_a_group():
    b = S.Builder()
    f = S.column_sums(b, [b.inp(i, 0, 255) for i in range(S.N_LIMBS)])
    assert [S.clog2(b.gates[g].hi + 1) for g in f][:4] == [30] * 4 and S.clog2(b.gates[f[-1]].hi + 1) == 23
    _, _, widths = S._greedy_pass(b, f, 0, S.N_LIMBS)
    assert widths == [4] * 32


def test_the_seam_is_wrong_exactly_where_its_predicate_says():
    b, out, c_top = _build(S.seam_square)
    rng = random.Random(20261018)
    bad = next(a for a in (rng.randrange(1 << 1024) for _ in range(1 << 15)) if S.seam_fails(a))
    good = next(a for a in (rng.randrange(1 << 1024) for _ in range(16)) if not S.seam_fails(a))
    assert not _restricted_and_correct(b, out, bad)
    assert _restricted_and_correct(b, out, good)
    f = S.folded_columns(good)
    vals = _run(b, out, good)[0]
    assert vals[c_top] == sum(v << (8 * m) for m, v in enumerate(f)) >> 1024


def test_hole_b_is_refused_by_the_gather_hull():
    b = S.Builder()
    a = [b.inp(i, 0, 255) for i in range(S.N_LIMBS)]
    once, _ = S.holeb_square(b, a)
    assert b.fixed_units == 128
    with pytest.raises(AssertionError, match="outside s32"):
        S.holeb_square(b, once)  # the next squaring's columns, over the bytes' derived hulls, leave s32
    b._append(S.WordOp("mma", fa=S.U8, fb=S.U8, k=1), (b.const(0), once[0], once[0]), S.S32_MIN, S.S32_MAX)
    errs = S.proven(b.gates)
    for g in once:
        assert b.gates[g].op.kind == "gather" and (b.gates[g].lo, b.gates[g].hi) == (0, 255)
        lo, hi = b.ivs[g]
        assert lo < 0 or hi > 255
        assert any(e.startswith(f"gate {g} (gather): derived") for e in errs)
    assert any(f"(mma): operand {once[0]}" in e and "outside format (0, 255)" in e for e in errs)
    assert not S.restricted(128, b.gates)
    rng = random.Random(20261018)
    x = next(x for x in (rng.randrange(1 << 1024) for _ in range(16)) if not S.seam_fails(x))
    vals, window, hold = b.evaluate(dict(enumerate(S.limbs(x))))
    assert window and hold and S.output_value([vals[o] for o in once]) % S.P == x * x % S.P


def test_a_gather_derives_its_columns_hull_and_a_table_read_its_low_bits():
    b = S.Builder()
    i = b.inp(0, -1000, 1000)
    col = [b.const(c) for c in (5, -3, 9)]
    rng = b.gather(i, col)
    low = b.gather(i, col, low=2)
    assert (b.gates[rng].lo, b.gates[rng].hi) == b.ivs[rng] == (-3, 9)
    assert b.ivs[low] == (-3, 9) and S.proven(b.gates) == []
    for v in (-7, -1, 0, 1, 2, 3, 4, 6, 1000):
        vals = b.evaluate({0: v})[0]
        assert vals[rng] == ([5, -3, 9][max(v, 0)] if v < 3 else 0)  # `toNat` reads a negative index as 0
        assert vals[low] == ([5, -3, 9, 0][v % 4])
    assert b.fixed_units == S.budgeted(b.gates, fixed=True) == 2 + 2  # ⌈log₂ 4⌉ at range, n = 2 at low: two kinds
    again = b.gather(i, [b.const(7)] * 3, low=2)
    wider = b.gather(i, [b.const(1)] * 3, low=3)
    assert S.units(b.gates, again, fixed=True) == 0 and S.units(b.gates, wider, fixed=True) == 1
    assert b.fixed_units == S.budgeted(b.gates, fixed=True) == 5


def test_no_one_pass_plan_beats_32_groups():
    u8, s8 = S.column_intervals(S.U8), S.column_intervals(S.S8)
    assert (S._iv_bits(u8[0]), S._iv_bits(u8[-1]), S._iv_bits(s8[0]), S._iv_bits(s8[-1])) == (30, 23, 29, 22)
    n, plan = S.group_plan(u8)
    assert n == 32 and sorted(p + d for g in plan for p, span in g for d in range(span)) == list(range(S.N_LIMBS))
    assert S.group_plan(s8, True)[0] == 32
    for cols, signed in ((u8, False), (s8, True)):
        for start in range(0, S.N_LIMBS, 4 if signed else 1):
            assert S.group_plan(cols, signed, start=start)[0] >= 32


def test_the_seam_fails_on_about_2_to_the_minus_12_of_inputs():
    rng = random.Random(1)
    n = 1 << 12
    fails = sum(S.seam_fails(rng.randrange(1 << 1024)) for _ in range(n))
    assert fails <= n >> 8  # well below the 2^−8 a 3,063-squaring word at ε = 2^−40 could absorb


def test_the_costs_against_the_conjecture():
    assert (S.SQUARINGS, S.ROUNDS) == (1021, 3)
    assert (S.renorm_floor(40), S.renorm_floor(20)) == (119, 121)
    assert (S.e_per_squaring(40), S.e_per_squaring(20)) == (374, 376)
    assert S.ROUNDS * S.SQUARINGS * S.e_per_squaring(40) == 1_145_562
    assert S.ROUNDS * S.SQUARINGS * S.e_per_squaring(20) == 1_151_688
    assert (S.counting_floor(40), S.counting_floor(20)) == (123, 126)
    assert S.ROUNDS * S.SQUARINGS * 160 == 490_080


def test_the_whole_codec_word_is_free_at_6bs_fields():
    builds = {}
    for n, c, cs, K, tau, i, m, y in CODEC_ROWS:
        p = (1 << n) - c
        assert FC.encode_word_ref(p, cs, K, tau * i % p, m) == y
        key = (n, c, len(cs))
        if key not in builds:
            builds[key] = FC.free_encode_word(n, c, len(cs))
        b, out, slots = builds[key]
        vals, window, hold = b.evaluate(FC.draw_inputs(n, c, cs, K, tau, i, m))
        assert window and hold
        assert [vals[o] for o in out] == [(y // 2 * 2) >> (8 * l) & 255 for l in range(n // 8)]
    for b, out, slots in builds.values():
        assert not any(g.op.kind == "budgeted" for g in b.gates)
        assert not any(g.op.kind == "input" and g.op.binding == "run" for g in b.gates)
        assert S.restricted(128, b.gates)
        assert b.lean_units == 0 and b.fixed_units > 0
        if len(b.gates) < 3000:  # the literal `units` scan is quadratic; it agrees with the builder's running count
            assert S.budgeted(b.gates) == 0 and S.budgeted(b.gates, fixed=True) == b.fixed_units
        assert {g.op.binding for g in b.gates if g.op.kind == "input" and g.op.slot[0] == "payload"} == {"registered"}


def test_the_free_codec_word_is_right_on_every_draw_tried():
    rng = random.Random(3)
    n, c, k = 16, 17, 3
    p = (1 << n) - c
    b, out, _ = FC.free_encode_word(n, c, k)
    for _ in range(12):
        cs = [rng.randrange(p) for _ in range(k)]
        K, tau, i, m = rng.randrange(1 << n), rng.randrange(1, p), rng.randrange(1 << 20), rng.randrange(1 << n)
        y = FC.encode_word_ref(p, cs, K, tau * i % p, m)
        vals, window, hold = b.evaluate(FC.draw_inputs(n, c, cs, K, tau, i, m))
        assert window and hold
        assert [vals[o] for o in out] == [(y // 2 * 2) >> (8 * l) & 255 for l in range(n // 8)]
