"""lean_headline_dc: the H200's and B200's headline instances in Lean are the machine model's data, exactly."""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_lean_headline_dc", Path(__file__).parents[1] / "lean_headline_dc.py")
H = importlib.util.module_from_spec(spec)
sys.modules["pouw_lean_headline_dc"] = H
spec.loader.exec_module(H)


def test_the_committed_instances_are_the_exporters():
    """Security/Definitions/Accounting/HeadlineDc/Sm90.lean, Sm90Measured.lean, Sm100.lean and Sm100Measured.lean are
    what lean_headline_dc.py writes from the data, π* included, and every reading's bound is below 1%."""
    assert H.main(["--check"]) == 0


def test_each_certificate_is_one_and_prices_the_bound():
    """Each π* is a certificate (weights ≥ 0 summing to 1 exactly), the bound is the larger branch at its prices, and
    the H200's filler enters Y: χ* > 0 on both its readings, 0 on the B200's."""
    for fam in H.FAMILIES:
        for v in (fam.sheet, fam.measured):
            r = H.reading(fam, v)
            lam = r.data.lam
            assert all(x > 0 for x in lam.values()) and sum(lam.values()) == 1
            assert r.data.bound == max(Fraction(r.exact["bound"]), Fraction(0)) and r.bound <= r.target
            assert (r.chi > 0) == (fam is H.SM90)


def test_the_h200_measured_menu_is_honests_form():
    """On DC-90-ri the writer kinds are honest's IGMMA_S8_N256 and UNI: every other tensor form carries the scalar
    coefficient there, which leaves the full menu's bound far above 1%."""
    r = H.reading(H.SM90, "DC-90-ri")
    assert [k for k, _ in r.data.writers] == ["IGMMA_S8_N256", "UNI"] and r.menu == "honest"
    assert H.reading(H.SM90, "DC-90-ri", full=True).bound > 0.1
