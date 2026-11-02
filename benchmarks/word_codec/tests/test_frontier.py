"""The frontier renderer's point schema and its frontier rule: x ≥ x* = (1 − τ)/μ_budget, references excluded."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frontier as F  # noqa: E402


def point(**kw):
    p = {"schema": F.SCHEMA, "lane": "t", "b": 521, "k": 1, "arithmetic": "a", "batch": 1, "y_total": 1.6,
         "y_decode": 0.3, "y_tag": 0.2, "y_commit": 0.1, "x": 230.0, "tau": 0.6}
    p.update(kw)
    return p


def test_check_fills_base_and_measured():
    (p,) = F.check([point()])
    assert p["y_base"] == pytest.approx(1.0) and p["measured"] is True


def test_check_prices_a_mixed_k_point_at_its_smallest_k():
    (p,) = F.check([point(k=1.75, x=437.5, ks={"a": 2, "b": 1})])
    assert (p["k"], p["k_mean"], p["x"]) == (1, 1.75, pytest.approx(250.0))


def test_check_refuses_other_schemas_and_missing_fields():
    with pytest.raises(ValueError):
        F.check([point(schema="other")])
    bad = point()
    del bad["tau"]
    with pytest.raises(ValueError):
        F.check([bad])


def test_frontier_takes_the_lowest_y_at_x_at_least_x_star():
    budget = 0.005
    assert F.x_star(0.6, budget) == pytest.approx(80.0) and F.x_star(0.6, budget, 1.6) == pytest.approx(200.0)
    pts = F.check([point(y_total=1.5, x=150.0), point(y_total=1.7), point(y_total=1.6), point(batch=32, y_total=1.9),
                   point(batch=32, y_total=2.0, x=300.0), point(y_total=1.0, reference=True)])
    best = F.frontier(pts, budget, idle=0.0, boost=1.0)
    assert best[1]["y_total"] == 1.6 and best[32]["y_total"] == 2.0
    assert F.frontier(pts, budget, d=1.0, idle=0.0, boost=1.0)[1]["y_total"] == 1.5
    assert "| 1 | 521 | 1 |" in F.table(pts, budget, idle=0.0, boost=1.0)
    assert "none at x ≥ x*" in F.table(pts, budget, batches=(8,), idle=0.0, boost=1.0)


def test_frontier_adds_the_idle_share_to_d_and_grants_the_adversary_the_boost():
    budget = 0.005
    # y 1.6 + idle 0.37 − τ 0.6 = 1.37: x* = 274; x 300 is 258.6 at 1.16×, 300 when the point says it is boosted
    p, q = point(y_total=1.6, x=300.0), point(y_total=1.6, x=300.0, x_boosted=True, arithmetic="b")
    assert F.step(p, None) == pytest.approx(1.97) and F.x_eff(p) == pytest.approx(300 / 1.16)
    assert F.frontier(F.check([p]), budget) == {}
    assert F.frontier(F.check([p, q]), budget)[1]["arithmetic"] == "b"
    assert F.frontier(F.check([p]), budget, idle=0.10)[1]["x"] == 300.0


def test_adversary_takes_the_fastest_encode_of_each_width_and_the_cheapest_tag():
    mine = point(lane="wide", k=2, x=554.0, tau=0.82)                    # 238.8 a round at 1.16×
    theirs = point(lane="tensor", k=1, x=216.0, x_boosted=True, tau=0.48)
    other = point(lane="wide", b=255, k=1, x=109.0, tau=0.82)
    a, t, o = F.adversary(F.check([mine, theirs, other]), tau_cap=0.61)
    assert a["x"] == pytest.approx(432.0) and a["x_boosted"] and a["x_own"] == 554.0 and a["tau"] == 0.61
    assert t["x"] == pytest.approx(216.0) and t["tau"] == 0.48
    assert o["x"] == pytest.approx(109.0 / 1.16)


def test_sketch_twins_keep_base_and_decode_and_replace_tag_and_commit():
    pts = F.check([point(), point(arithmetic="a/commit-overlapped"), point(batch=32), point(measured=False),
                   point(batch=8), point(arithmetic="a/sketch"), point(arithmetic="a/sketch/commit-overlapped")])
    twins = F.sketch_twins(pts, {1: 0.07, 32: 0.15})
    assert [(t["batch"], t["arithmetic"]) for t in twins] == [(1, "a/sketch-model"), (32, "a/sketch-model")]
    t = twins[0]
    assert t["y_total"] == pytest.approx(1.0 + 0.3 + 0.07) and t["y_tag"] == 0.07 and t["y_commit"] == 0.0
    assert t["tau"] == 0.0 and t["measured"] is False and t["x"] == 230.0


def test_tensor_points_take_y_from_a_runs_layer_and_x_from_any_runs_analysis(capsys):
    gemm = {"model": {"name": "70b"}, "libs": {
        "p512": {"constants": {"n": 512}, "layer": {"1": {"cublas": 1.05, "plain": 1.0, "fused_k1": 1.051,
                                                          "fused_k1_tagged": 3.0},
                                                    "32": {"plain": 1.017, "fused_k1": 1.445}}},
        "p768": {"layer": {"1": {"plain": 1.0, "fused_k1": 1.159}}}}}
    rates = {"libs": {"p512": {"analysis": {"k": {"1": {"x_cuda": 210.0, "x_both": 203.0, "x_tc": 900.0,
                                                        "x_cuda_measured_encode": 150.0, "delta_cuda": 0.48}}}}}}
    pts = F.check(F.tensor_points([(gemm, "art:g"), (rates, "art:r")]))
    assert "p768 has no analysis" in capsys.readouterr().err
    assert sorted(p["batch"] for p in pts) == [1, 32]
    p1 = next(p for p in pts if p["batch"] == 1)
    assert (p1["b"], p1["k"], p1["model"], p1["x"], p1["delta"], p1["tau"], p1["art"]) == (
        512, 1, "llama-3.1-70b", 203.0, 0.48, 0.0, "art:g")
    assert p1["y_decode"] == pytest.approx(0.051) and p1["y_base"] == 1.0 and p1["y_tag"] == p1["y_commit"] == 0.0
    (twin,) = F.sketch_twins([p1], {1: 0.073})
    assert twin["y_total"] == pytest.approx(1.124)


def test_best_per_config_keeps_the_lowest_y_of_each_configuration():
    pts = F.check([point(y_total=1.7), point(y_total=1.6, arithmetic="b"), point(k=2, y_total=1.9),
                   point(lane="u", y_total=1.8), point(batch=32, y_total=2.0), point(y_total=1.0, measured=False)])
    best = {(p["lane"], p["k"], p["batch"], p["measured"]): p["y_total"] for p in F.best_per_config(pts)}
    assert best == {("t", 1, 1, True): 1.6, ("t", 2, 1, True): 1.9, ("u", 1, 1, True): 1.8, ("t", 1, 32, True): 2.0,
                    ("t", 1, 1, False): 1.0}
