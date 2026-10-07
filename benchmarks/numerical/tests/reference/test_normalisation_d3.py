"""Log entry D3: the successive-shift normalisation of ``group_sum`` (the simulator's, ported in ``hawkeye_close.group_sum``, and
``silicon.group_sum``'s) against the closed form ``e' = max(E + l - 25, -126)``, ``M' = floor(v * 2^(E - e' - 1))``.

What is covered, and how:

* ``test_closed_form_equals_successive_shifts_over_the_whole_reachable_domain_by_smt``: an SMT proof (z3, 32-bit bitvectors) that
  the successive-shift form and the closed form agree for **every** ``E in [-132, 254]`` and **every** ``v in [0, 2^30)`` -- the
  whole reachable domain (``E`` is the maximum of ``E_ZERO`` and term exponents in ``[-252, 254]``; ``|S| < 9 * 2^26 < 2^30``).
  Skipped when z3 is not importable.
* ``test_three_forms_agree_on_every_exponent_and_bit_length``: exhaustive over ``E in [-132, 254]`` x ``l in [0, 30]`` with several
  bit patterns per length, through the real ``group_sum`` functions (a two-term group realises any ``(E, v)``), all three forms.
* ``test_three_forms_agree_exhaustively_on_small_sums``: every ``v < 2^12`` at the exponents where a branch changes.
* ``test_three_forms_agree_on_random_nine_term_groups``: random full groups.

Proof sketch (why the closed form is the successive form): the successive form computes ``exp = E + l - 25`` and normalises ``v`` to
25 bits (``v >> (l - 25)`` if ``l > 25``, else ``v << (25 - l)``, both exact or floor), then if ``exp < -126`` shifts right by
``-126 - exp`` and sets ``exp = -126``, then shifts right by 1.  Composing: the exponent is ``e' = max(E + l - 25, -126)`` and
the magnitude is ``v`` shifted by ``(25 - l) - (e' - (E + l - 25)) - 1 = E - e' - 1`` in total, where a left shift is exact and a
composition of floor right shifts (of integers) is the floor right shift by the sum -- so ``M' = floor(v * 2^(E - e' - 1))``.  A
left shift followed by right shifts loses no bits before the right shifts begin, so the order does not matter either.
"""

from __future__ import annotations

import random

import pytest

from verity.ml.circuits.silicon import silicon as S
from verity_numerical.reference import hawkeye_close as H
from wp1_cases import canon, canon_term, realise as _pair, term_of

PIPE = S.AMPERE_BF16_M16N8K16
E_LO, E_HI = H.E_ZERO, 254  # the reachable range of a group's maximum exponent
V_BITS = 30  # |S| < 2^30


def _three_forms(group: list[H.Gfloat]) -> tuple[tuple, tuple, tuple]:
    return (
        canon(H.group_sum(group)),
        canon(H.group_sum_closed_form(group)),
        canon_term(S.group_sum([term_of(g) for g in group], width=PIPE.width, zero_exponent=PIPE.zero_exponent)),
    )


def test_two_term_realisation_reaches_the_intended_sum() -> None:
    for E, v in ((0, 5), (-132, 1), (254, (1 << 30) - 1), (-126, 1 << 24)):
        g = _pair(False, E, v)
        # the closed form on (E, v) directly equals the closed form through the two-term group
        assert canon(H.normalise_closed_form(False, E, v)) == canon(H.group_sum_closed_form(g))


def test_three_forms_agree_on_every_exponent_and_bit_length() -> None:
    rng = random.Random(3)
    n = 0
    for E in range(E_LO, E_HI + 1):
        for l in range(0, V_BITS + 1):
            if l == 0:
                vs = [0]
            else:
                vs = {1 << (l - 1), (1 << l) - 1, (1 << (l - 1)) | 1, rng.getrandbits(l) | (1 << (l - 1))}
            for v in vs:
                for sign in (False, True):
                    a, b, c = _three_forms(_pair(sign, E, v))
                    assert a == b == c, (E, l, v, sign, a, b, c)
                    assert a == canon(H.normalise_closed_form(sign, E, v))
                    n += 1
    assert n > 40000


@pytest.mark.parametrize("E", [-132, -131, -127, -126, -125, -108, -107, -102, -101, -100, 0, 101, 102, 127, 128, 254])
def test_three_forms_agree_exhaustively_on_small_sums(E: int) -> None:
    for v in range(1 << 12):
        a, b, c = _three_forms(_pair(v & 1 == 1, E, v))
        assert a == b == c, (E, v)


def test_three_forms_agree_on_random_nine_term_groups() -> None:
    rng = random.Random(20260921)
    top = 65025 << 9
    for _ in range(3000):
        centre = rng.randint(-132, 254)
        group = [H.Gfloat(rng.random() < 0.5, centre - rng.randint(0, 40), rng.randint(0, top)) for _ in range(9)]
        group[0] = H.Gfloat(group[0].sign, max(-126, min(127, group[0].exponent)), rng.getrandbits(24))
        a, b, c = _three_forms(group)
        assert a == b == c


def test_closed_form_equals_successive_shifts_over_the_whole_reachable_domain_by_smt() -> None:
    z3 = pytest.importorskip("z3")
    BV = lambda x: z3.BitVecVal(x, 32)
    v = z3.BitVec("v", 32)
    E = z3.BitVec("E", 32)
    # l = bit_length(v) for v < 2^30
    l = BV(0)
    for k in range(1, V_BITS + 1):
        l = z3.If(z3.UGE(v, BV(1 << (k - 1))), BV(k), l)
    # successive shifts (group_sum's tail): normalise to 25 bits, denormalise below -126, drop the rescale bit
    exp = E + l - BV(25)
    m1 = z3.If(l > BV(25), z3.LShR(v, l - BV(25)), v << (BV(25) - l))
    denorm = exp < BV(-126)
    m2 = z3.If(denorm, z3.LShR(m1, BV(-126) - exp), m1)
    e2 = z3.If(denorm, BV(-126), exp)
    m3 = z3.LShR(m2, BV(1))
    # closed form
    e_closed = z3.If(exp < BV(-126), BV(-126), exp)
    s = E - e_closed - BV(1)
    m_closed = z3.If(s >= BV(0), v << s, z3.LShR(v, BV(0) - s))
    domain = z3.And(z3.ULT(v, BV(1 << V_BITS)), E >= BV(E_LO), E <= BV(E_HI))
    equal = z3.And(m3 == m_closed, z3.Or(m3 == BV(0), e2 == e_closed))
    solver = z3.Solver()
    solver.add(domain, z3.Not(equal))
    assert solver.check() == z3.unsat
    # sanity: the encoding is satisfiable at all and agrees with the Python on a few points
    check = z3.Solver()
    check.add(domain, v == BV(5), E == BV(-130))
    assert check.check() == z3.sat
    model = check.model()
    got = H.normalise_closed_form(False, -130, 5)
    assert model.eval(m_closed).as_long() == got.significand and (got.significand == 0 or model.eval(e_closed).as_signed_long() == got.exponent)
