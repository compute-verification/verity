"""lean_headline: the families' headline instances in Lean are the machine model's data, exactly."""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_lean_headline", Path(__file__).parents[1] / "lean_headline.py")
H = importlib.util.module_from_spec(spec)
sys.modules["pouw_lean_headline"] = H
spec.loader.exec_module(H)


def test_the_committed_instances_are_the_exporters():
    """Security/Definitions/Accounting/Headline/Sm120.lean, Sm120Measured.lean and Sm80.lean are what lean_headline.py
    writes from the data, π* included."""
    assert H.main(["--check"]) == 0


def test_the_instance_meets_the_headline():
    """π* is the issue facet alone, ♭'s bound is its issue count, and the bound Sm120SheetHeadline's proof reads off the
    identity is below 0.656%: its folded branch at p = 7/8, the receipt split's (ln 2 and ln(8/7) by their rational
    bounds, above the exact logs' value), is below 0.599%, and the unfolded one, fold skippers at the whole-unit luck ½,
    binds."""
    i = H.instance()
    assert i.pistar == "DS_issue" and i.B == H.load(i, "DS_issue", {k: i.nw.get(k, 0) + i.nf.get(k, 0) + i.ns.get(k, 0)
                                                                   for k in {*i.nw, *i.nf, *i.ns}})
    assert H.P == Fraction(7, 8) and H.LN_P == Fraction(1, 8)
    assert i.folded < Fraction(599, 100000) < i.unfolded == i.bound
    assert H.exact_logs(i) <= float(i.bound) < float(H.EPS) and H.EPS == Fraction(656, 100000)


def test_the_measured_instance_meets_its_reading():
    """On VF3r-16 π* is JI3s_0 alone and ♭'s bound is its load there, w* is honest's IMMA16832 at its 128 words, no
    writer kind is the forms' fallback line or UNI64, and the bound Sm120MeasuredHeadline's proof reads off the identity
    is below 0.591%, its folded branch binding."""
    i = H.instance()
    m = H.measured(i)
    n = {k: i.nw.get(k, 0) + i.nf.get(k, 0) + m.ns.get(k, 0) for k in {*i.nw, *i.nf, *m.ns}}
    assert m.pistar == "JI3s_0" and m.B == H.mload(m, "JI3s_0", n) == max(H.mload(m, r, n) for r in m.rows)
    assert m.wstar == "IMMA16832" and dict(m.writers)["IMMA16832"] == 128 and not set(H.OFF_MENU) & set(dict(m.writers))
    assert m.unfolded < m.folded == m.bound
    assert H.exact_logs(m) < float(m.bound) < float(H.EPS_M) and H.EPS_M == Fraction(591, 100000)


def test_the_sm80_instance_meets_its_headline():
    """On DS-80 π* is the ALU pipe alone and ♭'s bound is its ALU count; the first writer kind issues nothing there, so
    w* = κ = 0, the folded branch of the bound Sm80SheetHeadline's proof reads off the identity, 1 − N_w φ*/B_♭ at
    φ* = 1/8, is ε_skip, and the unfolded one, `1 − 16/B_♭ · (1 − 41/48000)`, binds at most 0.232%."""
    i = H.instance(H.SM80)
    n = {k: i.nw.get(k, 0) + i.nf.get(k, 0) + i.ns.get(k, 0) for k in {*i.nw, *i.nf, *i.ns}}
    assert i.pistar == "DS_alu" and i.B == H.load(i, "DS_alu", n)
    assert H.coef(i, "DS_alu", i.writers[0][0]) == 0 and i.kappa == 0 and i.Nw == 128
    assert [H.load(i, "DS_alu", f) for f in i.folds] == [Fraction(1, 8)]
    assert i.folded == 1 - 16 / i.B == i.eps_skip
    assert i.bound == i.unfolded == 1 - 16 / i.B * (1 - Fraction(41, 48000)) <= i.fam.eps == Fraction(232, 100000)
