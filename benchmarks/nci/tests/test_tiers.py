import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import slowdown as S  # noqa: E402
import tiers as T  # noqa: E402


def test_each_tier_gives_the_adversary_no_more_than_the_last():
    totals = [T.evaluate(T.DEFAULTS, T.TIERS[k])["total"] for k in ("T1", "T2", "T3", "T4")]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(totals, totals[1:]))


def test_every_parameter_has_a_slider_and_its_default_is_in_range():
    assert {s["key"] for s in T.SPEC} == set(T.DEFAULTS)
    for s in T.SPEC:
        v = T.DEFAULTS[s["key"]]
        assert s["min"] <= v <= s["max"] or (v == 0 and s.get("zero")), s["key"]


def test_without_mechanisms_headroom_trains_at_its_compute():
    assert T.evaluate(T.DEFAULTS, T.TIERS["T1"])["binding"] == "headroom compute"


def test_headroom_voids_the_partial_sum_law():
    p = T.DEFAULTS
    r = T.evaluate(p, T.TIERS["T4"])
    C = p["nodes"] * p["node_flops"] * p["years"] * T.YEAR
    assert r["declared"] == pytest.approx(C / r["rho_T"])


def test_without_dark_compute_the_declared_side_is_the_chain_bound():
    """At h = u = 0 nothing absorbs partial sums, so the slowdown is the calculator's chain bound for pretrain-full."""
    p = T.DEFAULTS | {"h": 0.0, "u": 0.0}
    r = T.evaluate(p, T.TIERS["T4"])
    alpha = p["K"] / p["Gamma"]
    chain = p["kappa_chain"] * S.read_factor(p["K"], p["b_w"], p["b_a"], p["d_chain"]) / (2 * alpha)
    assert r["binding"] == "partial-sum law"
    assert r["slowdown"] == pytest.approx(chain, rel=1e-3)


def test_absorbed_partial_sums_weaken_the_law_as_ps_prime_says():
    """PS' on Kimi K2's forward chain: G_D b_a / R at zeta = b_a Z / R of 0, 1 and 10 (Tier 1's table)."""
    rates = T.chain_rates(T.DEFAULTS, 2000)
    got = [T.chain_products(rates, 1.0, z) * T.DEFAULTS["b_a"] for z in (0.0, 1.0, 10.0)]
    assert got == pytest.approx([1714, 2482, 6483], rel=1e-3)


def test_the_wipe_floor_is_the_closed_form():
    """With every gate windowed and the wipe period below R / r, the floor is kappa b_a b_w d Phi / (4 R r)."""
    p = T.DEFAULTS | {"h": 1.0, "slack": 1e-9, "unwipeable_gb": 1.0, "sigma": 0.0, "wipe_s": 1.0}
    r = T.evaluate(p, T.TIERS["T3"])
    rate = p["tau"] + p["K"] / p["Gamma"] * p["node_flops"]
    closed = p["kappa"] * p["b_a"] * p["b_w"] * p["d"] * p["node_flops"] / (4 * r["residual_bits"] * rate)
    assert r["binding"] == "server windows"
    assert r["slowdown"] == pytest.approx(closed, rel=1e-3)


def test_a_small_wipe_residual_binds_headroom():
    p = T.DEFAULTS | {"unwipeable_gb": 1.0, "slack": 1e-9}
    r = T.evaluate(p, T.TIERS["T3"])
    assert r["binding"] == "server windows"
    assert r["slowdown"] > 10 * T.evaluate(T.DEFAULTS, T.TIERS["T3"])["slowdown"]


def test_helpers_within_the_read_deadline_refill_the_residual():
    quiet = T.evaluate(T.DEFAULTS | {"sigma": 1e3}, T.TIERS["T3"])["residual_bits"]
    loud = T.evaluate(T.DEFAULTS | {"sigma": 1e6}, T.TIERS["T3"])["residual_bits"]
    mem = (T.DEFAULTS["hbm_gb"] + T.DEFAULTS["host_gb"]) * T.GB
    assert loud - quiet == pytest.approx((1e6 * 1e-3 - 25) / 524288 * mem)


def test_the_cheating_pool_does_not_grow_with_the_fleet():
    small = T.evaluate(T.DEFAULTS | {"nodes": 1e3}, T.TIERS["T5"])
    large = T.evaluate(T.DEFAULTS, T.TIERS["T5"])
    assert small["cheat"] == large["cheat"] > 0
    q = T.DEFAULTS["share"] / (T.DEFAULTS["overhead"] + T.DEFAULTS["replay"])
    assert large["cheat_steps"] == pytest.approx(math.log(1 / T.DEFAULTS["p_evade"]) / q)
