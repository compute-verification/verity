"""lean_headline: sm_120's headline instance in Lean is the machine model's data, exactly."""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_lean_headline", Path(__file__).parents[1] / "lean_headline.py")
H = importlib.util.module_from_spec(spec)
sys.modules["pouw_lean_headline"] = H
spec.loader.exec_module(H)


def test_the_committed_instance_is_the_exporters():
    """Security/Definitions/Accounting/Headline/Sm120.lean is what lean_headline.py writes from the data, π* included."""
    assert H.main(["--check"]) == 0


def test_the_instance_meets_the_headline():
    """π* is the issue facet alone, ♭'s bound is its issue count, and the bound Sm120SheetHeadline's proof reads off the
    identity (ln 2 and ln(4/3) by their rational bounds, above the exact logs' value) is below 0.585%."""
    i = H.instance()
    assert i.pistar == "DS_issue" and i.B == H.load(i, "DS_issue", {k: i.nw.get(k, 0) + i.nf.get(k, 0) + i.ns.get(k, 0)
                                                                   for k in {*i.nw, *i.nf, *i.ns}})
    assert H.exact_logs(i) < float(i.bound) < float(H.EPS) and H.EPS == Fraction(117, 20000)
