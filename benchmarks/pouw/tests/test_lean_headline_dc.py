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
    what lean_headline_dc.py writes from the data, π* included, and every reading's bound is at most its target."""
    assert H.main(["--check"]) == 0


def test_each_certificate_is_one_and_prices_the_bound():
    """Each π* is a certificate (weights ≥ 0 summing to 1 exactly), the bound is the larger branch at its prices, χ* is
    0 on every reading (the form `headline_of_branches` proves), and a uniform writer heads 2 words (RT-HOPPER-7)."""
    for fam in H.FAMILIES:
        for v in (fam.sheet, fam.measured):
            r = H.reading(fam, v)
            lam = r.data.lam
            assert all(x > 0 for x in lam.values()) and sum(lam.values()) == 1
            assert r.data.bound == max(Fraction(r.exact["bound"]), Fraction(0)) and r.bound <= r.target
            assert r.chi == 0 and not r.data.nx
            assert dict(r.data.writers)["UNI"] == 2


def test_the_credit_is_what_the_checks_read():
    """RT-LOGIC-12: the design's fold is honest's, the one PouwWarpCheck recomputes (♭ folds by the cheapest mix of it
    and its checkpoint routes, which the check can't tell apart), and a filler is credited only at p = 1 while no check
    reads it."""
    for fam in H.FAMILIES:
        d = H.E.design(fam.design, fam.data)
        assert d.fold == H.E.honest(d.honest, fam.data).fold, fam.design
        assert all(f.p == 1 for f in d.filler), fam.design


def test_the_h200_measured_menu_is_honests_form():
    """On DC-90-ri the writer kinds are honest's IGMMA_S8_N256 and UNI: every other tensor form carries the scalar
    coefficient there, which leaves the full menu's bound far above 1%. Its ♭ is `setup`'s floor, which heads a share
    of its words on honest's lane baseline (dp4a) once ffill is out of it, so the measured file states its own word
    production; the reading is over 1% (RT-HB-HEAD-1)."""
    r, s = H.reading(H.SM90, "DC-90-ri"), H.reading(H.SM90, "DS-90")
    assert [k for k, _ in r.data.writers] == ["IGMMA_S8_N256", "UNI"] and r.menu == "honest"
    assert 0.01 < r.machine_core <= r.bound
    assert H.reading(H.SM90, "DC-90-ri", full=True).bound > 0.1
    assert "dp4a" in r.data.words_mix and r.data.nw != s.data.nw and "def flatWords" in H.render_measured(H.SM90, r, s)
