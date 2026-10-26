"""theory/accel_bet_fold.py's small cases: Lemma T, the Latin LUTs, NCP at width 4, binding, and the exact LP's ε."""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1] / "theory"
sys.path.insert(0, str(THEORY))

import accel_bet_fold as T


def _no_fail(f, *args):
    T.FAIL.clear()
    out = f(*args)
    assert not T.FAIL, T.FAIL
    return out


def test_lemma_t_and_latin_luts():
    _no_fail(T.tri_steps)
    assert T.latin_luts() == [0x69, 0x96]


def test_ncp_width_4():
    """XC and li fail NCP under both laws; RA fails NCP⁺ at every start state; FE and FA meet both."""
    ones = [(c,) for c in range(16)]
    twos = [(1, 2), (5, 11), (9, 0), (14, 7)]
    for name in ("XC", "li"):
        r = T.ncp(name, 4, 2, ones)
        assert r["xor"]["invariant_d"] > 0 and r["add"]["invariant_d"] > 0
    ra = T.ncp("RA", 4, 2, ones)
    assert len(ra["bad_contexts_constant_derivative"]) == 16
    assert all(T.ra_bad_context(c[0], 4) for c in ra["bad_contexts"])
    for name in ("FE", "FA"):
        r = T.ncp(name, 4, 2, twos)
        assert not r["bad_contexts"] and not r["bad_contexts_constant_derivative"]


def test_binding_and_facts():
    _no_fail(T.binding, True)
    _no_fail(T.facts_32, 200)
    _no_fail(T.conservation, 0xC0DE, 300)


def test_simplex_certificate():
    val, x, y = T.simplex([F(3), F(2)], [[F(1), F(1)], [F(1), F(3)]], [F(4), F(6)])
    assert val == 12 and x == [4, 0] and sum(yi * bi for yi, bi in zip(y, [4, 6])) == val


def test_eps_falls_with_the_premise():
    out = {}
    _no_fail(T.table, out)
    _no_fail(T.checks_lp, out)
    assert F(out["eps"]["li/L32/FOLD-OPT"]["eps"]) == 0
    assert F(out["eps"]["XC/L32/SEM"]["eps"]) > F(1, 100)
