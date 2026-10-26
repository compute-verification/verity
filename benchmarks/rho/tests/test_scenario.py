import math

import pytest
from rho import classes, lean
from rho.attacks import ATTACKS
from rho.classes import CLASSES
from rho.params import Registry
from rho.regimes import KINDS
from rho.scenario import evaluate_named

REG = Registry.load()


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_every_scenario_evaluates_and_its_bracket_is_ordered_unless_a_premise_is_named_false(name):
    r = evaluate_named(REG, name)
    assert r.families
    for f in r.families:
        assert f.lo <= f.hi * (1 + 1e-9) or f.contradictions


def test_unmonitored_is_the_algorithm_alone():
    r = evaluate_named(REG, "unmonitored-h200-kimi")
    assert all(math.isclose(f.lo, f.profile.efficiency) and math.isclose(f.hi, f.profile.efficiency)
               for f in r.families)


def test_onsite_adds_a_cut_so_it_never_slows_less_than_remote():
    assert evaluate_named(REG, "onsite-h200-kimi").lo >= evaluate_named(REG, "remote-h200-kimi").lo * (1 - 1e-9)


def test_the_counts_match_the_lean_docstrings_at_kimi_k2():
    k = 2e7
    # FanOutIntensity: 7.45e11 products per block at c = 1; FanOutReadIntensity: 1.86e11 at f = 4
    assert math.isclose(classes.fanout(k, 2.5, 512, 37248, 1, 163840, 1), 7.45e11, rel_tol=2e-3)
    assert math.isclose(classes.fanout_read(k, 2.5, 4, 512, 37248, 1, 163840, 1), 1.86e11, rel_tol=2e-3)
    assert classes.lookups(k, 1, 163840) == 122
    assert classes.fanout(k, 2.5, 100, 37248, 1, 163840, 1) == math.inf
    assert classes.cover(k, 2.5) == k * k / 2.5 + k


def test_every_lean_citation_is_in_a_lock():
    cites = [c for a in (*CLASSES.values(), *ATTACKS.values()) for c in a.cites]
    cites += [c for kind in KINDS.values() for c in getattr(kind, "CITES", ()) + getattr(kind, "PHYSICAL", ())]
    absent = sorted({c for c in cites if lean.status(c) == "absent"})
    assert not absent, f"no lock lists {absent}"
