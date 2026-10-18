"""lean_headline: sm_120's headline instances in Lean are the machine model's data, exactly."""

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
    """Security/Definitions/Accounting/Headline/Sm120.lean and Sm120Measured.lean are what lean_headline.py writes from
    the data, π* included."""
    assert H.main(["--check"]) == 0


def test_the_instance_meets_the_headline():
    """π* is the issue facet alone, ♭'s bound is its issue count, and the bound Sm120SheetHeadline's proof reads off the
    identity (ln 2 and ln(4/3) by their rational bounds, above the exact logs' value) is below 0.585%."""
    i = H.instance()
    assert i.pistar == "DS_issue" and i.B == H.load(i, "DS_issue", {k: i.nw.get(k, 0) + i.nf.get(k, 0) + i.ns.get(k, 0)
                                                                   for k in {*i.nw, *i.nf, *i.ns}})
    assert H.exact_logs(i) < float(i.bound) < float(H.EPS) and H.EPS == Fraction(117, 20000)


def test_the_measured_instance_meets_its_reading():
    """On VF3r-16 π* is JI3s_0 alone and ♭'s bound is its load there, w* is honest's IMMA16832 at its 128 words, no
    writer kind is the forms' fallback line or UNI64, and the bound Sm120MeasuredHeadline's proof reads off the identity
    is below 0.540%."""
    i = H.instance()
    m = H.measured(i)
    n = {k: i.nw.get(k, 0) + i.nf.get(k, 0) + m.ns.get(k, 0) for k in {*i.nw, *i.nf, *m.ns}}
    assert m.pistar == "JI3s_0" and m.B == H.mload(m, "JI3s_0", n) == max(H.mload(m, r, n) for r in m.rows)
    assert m.wstar == "IMMA16832" and dict(m.writers)["IMMA16832"] == 128 and not set(H.OFF_MENU) & set(dict(m.writers))
    assert H.exact_logs(m) < float(m.bound) < float(H.EPS_M) and H.EPS_M == Fraction(27, 5000)
