import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiers as T  # noqa: E402


def test_each_tier_gives_the_adversary_no_more_than_the_last():
    totals = [T.evaluate(T.DEFAULTS, T.TIERS[k])["total"] for k in ("T1", "T2", "T3", "T4")]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(totals, totals[1:]))


def test_without_mechanisms_headroom_trains_at_its_compute():
    r = T.evaluate(T.DEFAULTS, T.TIERS["T1"])
    assert r["headroom_compute_bound"]


def test_a_small_wipe_residual_makes_the_region_bound_bind():
    p = T.DEFAULTS | {"residual_gb": 1.0}
    r = T.evaluate(p, T.TIERS["T3"])
    assert not r["headroom_compute_bound"]
    assert r["slowdown"] > 100 * T.evaluate(T.DEFAULTS, T.TIERS["T3"])["slowdown"]


def test_declared_side_is_the_thread_bound():
    p = T.DEFAULTS
    r = T.evaluate(p | {"h": 0.0}, T.TIERS["T4"])
    assert r["declared"] == pytest.approx(p["nodes"] * p["node_flops"] * p["years"] * T.YEAR / r["rho_T"])
    assert r["headroom"] == 0.0
