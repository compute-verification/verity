import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exposure as X  # noqa: E402
import tiers as T  # noqa: E402

HOLDS = [c for c in X.CHECKS if not c[-1].startswith("breaks")]
BREAKS = [c for c in X.CHECKS if c[-1].startswith("breaks")]


def test_every_parameter_has_a_range_around_its_default_inside_its_slider():
    assert set(X.RANGES) == set(T.DEFAULTS)
    for key, (low, high, source) in X.RANGES.items():
        s = X.SPECS[key]
        assert low <= T.DEFAULTS[key] <= high and source, key
        assert all(s["min"] <= v <= s["max"] or (v == 0 and s.get("zero")) for v in (low, high)), key


def test_the_defaults_give_the_statements_numbers():
    t = X.terms(T.DEFAULTS)
    assert t["R_wipe_gb"] == pytest.approx(166, rel=0.01)
    assert t["mu"] == pytest.approx(278, rel=0.01)
    assert t["nu"] == pytest.approx(1.04e-3, rel=0.01)
    assert t["chi"] == pytest.approx(1.73e-3, rel=0.01)
    assert t["eta_T4"] == pytest.approx(0.329, rel=0.01)


@pytest.mark.parametrize("probe", HOLDS, ids=[f"{c[0]}-{c[2]}-{c[3]}" for c in HOLDS])
def test_closed_forms_match_evaluate(probe):
    name, over, tier, term, _ = probe
    c = X.check(T.DEFAULTS | over, tier)
    got, closed = c[term]
    assert got == pytest.approx(closed, rel=0.02)
    assert c["capped"] == (term == "mu")


@pytest.mark.parametrize("probe", BREAKS, ids=[f"{c[0]}-{c[2]}-{c[3]}" for c in BREAKS])
def test_closed_forms_break_where_their_regimes_end(probe):
    name, over, tier, term, _ = probe
    got, closed = X.check(T.DEFAULTS | over, tier)[term]
    assert abs(got / closed - 1) > 0.2


def test_the_memory_cap_has_unit_elasticities_once_it_binds():
    """Served today at R_wipe 4 GB, eta = mu = 4 R_wipe alpha / (kappa d b_w b_a) at T4."""
    p = T.DEFAULTS | X.SCENARIOS["served today, R_wipe 4 GB"] | X.KEYED
    for key in ("d", "b_w", "b_a", "kappa"):
        assert X.elasticity(p, key, "T4") == pytest.approx(-1, rel=0.02), key
    assert X.elasticity(p, "residual", "T4") == pytest.approx(0, abs=1e-9)


def test_at_the_design_eta_is_the_residual():
    p = T.DEFAULTS | X.DESIGN | X.KEYED
    r = T.residual_share(p)
    assert X.elasticity(p, "host_share", "T1") == pytest.approx(p["host_share"] * (1 - p["eps_w"]) / r, rel=0.01)
    assert X.elasticity(p, "pous_rounds", "T4") == 0


@pytest.mark.parametrize("scenario, key, tier, rising", [
    ("defaults", "residual", "T1", True),
    ("defaults", "unmonitored", "T4", True),
    ("served today", "K", "T4", True),
    ("served today", "b_a", "T4", False),
    ("served today", "d", "T4", False),
])
def test_thresholds_are_monotone(scenario, key, tier, rising):
    """The set where 1/eta < 2 lies inside the set where 1/eta < 10, which lies inside the set where 1/eta < 100; each
    is one interval at the end of the slider that raises eta, so its inner edge moves outward as the level falls."""
    th = X.thresholds(T.DEFAULTS | X.SCENARIOS[scenario], key, tier)
    lo, hi = X.SPECS[key]["min"], X.SPECS[key]["max"]
    edges = []
    for level in X.LEVELS:
        if not th[level]:
            continue
        (a, b), = th[level]
        assert (b == hi) if rising else (a == lo)
        edges.append(a if rising else b)
    assert len(edges) >= 2
    assert edges == sorted(edges, reverse=not rising)


def test_the_residual_threshold_is_the_closed_form():
    """At T1, 1/eta < X once r_gpu > (1/X - eps_W - eps' - nu) / (1 - eps_W) - host_share - clock_share: eta is
    r-hat + eps_W (1 - r-hat) + eps' + nu."""
    p = T.DEFAULTS | X.DESIGN | {"lag_s": 0.0}
    th = X.thresholds(p, "residual", "T1")
    for level in (10.0, 2.0):
        (a, _), = th[level]
        edge = (1 / level - p["eps_w"] - p["eps_agg"] - X.nu(p)) / (1 - p["eps_w"]) - p["host_share"] - p["clock_share"]
        assert a == pytest.approx(edge, rel=1e-3)
