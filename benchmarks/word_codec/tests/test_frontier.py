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


def test_the_bench_stamps_its_model_and_70b_sets_its_linears_and_layers():
    import types

    import bench
    import wide
    import wide_bench

    lib = types.SimpleNamespace(field=wide.field(521), sqr=1, roll=1, occ=1, cc=0, b3u=0, tb=1, kw=1, name="p")
    fq = {1: {"x": 332.0, "delta": 0.9, "tau": 0.82, "mu_rebuild": 0.0005}}
    bd = {1: {1: {"y_total": 2.5, "y_decode": 0.5, "y_tag": 0.6, "y_commit": 0.4, "y_base": 1.0}}}
    saved = dict(bench.SHAPES)
    try:
        wide_bench.use_model("llama-3.1-70b")
        assert wide_bench.LAYERS == 80 and bench.SHAPES["gate_up"] == (57344, 8192) and len(bench.SHAPES) == 4
        (p,) = F.check(wide_bench.points(lib, fq, bd, "r", "gpu"))
        assert F.model_of(p) == "llama-3.1-70b" and F.model_of(point()) == "llama-3.1-8b"
    finally:
        wide_bench.use_model("llama-3.1-8b")
    assert bench.SHAPES == saved and wide_bench.LAYERS == 32


def test_best_per_config_keeps_the_lowest_y_of_each_configuration():
    pts = F.check([point(y_total=1.7), point(y_total=1.6, arithmetic="b"), point(k=2, y_total=1.9),
                   point(lane="u", y_total=1.8), point(batch=32, y_total=2.0), point(y_total=1.0, measured=False)])
    best = {(p["lane"], p["k"], p["batch"], p["measured"]): p["y_total"] for p in F.best_per_config(pts)}
    assert best == {("t", 1, 1, True): 1.6, ("t", 2, 1, True): 1.9, ("u", 1, 1, True): 1.8, ("t", 1, 32, True): 2.0,
                    ("t", 1, 1, False): 1.0}


def test_wide_bench_points_fit_the_schema_and_the_overlapped_commit_replaces_the_serial_one():
    import types

    import wide
    import wide_bench

    lib = types.SimpleNamespace(field=wide.field(521), sqr=1, roll=1, occ=1, cc=0, b3u=0, tb=1, kw=1, name="p521s1r")
    fq = {1: {"x": 332.0, "delta": 0.9, "tau": 0.82, "mu_rebuild": 0.0005}}
    bd = {1: {1: {"y_total": 2.5, "y_decode": 0.5, "y_tag": 0.6, "y_commit": 0.4, "y_commit_linear": 0.9,
                  "y_base": 1.0}}, 32: {1: {"y_total": 2.6, "y_decode": 0.5, "y_tag": 0.7, "y_commit": 0.4,
                                            "y_commit_linear": 0.9, "y_base": 1.0}}}
    po = {1: {"per_batch": {1: {"y_commit_overlap": 0.1}}}}
    pts = F.check(wide_bench.points(lib, fq, bd, "r", "gpu", 2092.0, po))
    assert [(p["batch"], p["arithmetic"].endswith("/commit-overlapped")) for p in pts] == [
        (1, False), (1, True), (32, False)]
    assert pts[0]["y_total"] == pytest.approx(2.5) and pts[1]["y_total"] == pytest.approx(2.2)
    assert pts[1]["y_commit"] == pytest.approx(0.1) and pts[1]["label"].endswith("/commit-overlapped")
    lib.tb, lib.kw, lib.cc, lib.b3u = 2, 2, 1, 1
    assert wide_bench.points(lib, fq, bd, "r", "gpu")[0]["arithmetic"] == (
        "cuda-int32/mersenne/sqr1/rolled/cc/b3u2/tb2")
    lib.tb = 1
    assert wide_bench.points(lib, fq, bd, "r", "gpu")[0]["arithmetic"].endswith("/b3u2/kw2")


def test_mixed_k_keeps_the_assignments_no_other_beats_in_y_and_mean_k():
    import types

    import wide
    import wide_bench

    def case(wb, native, t):
        arms = {"ww0": native, **{f"d{k}": native * (1 + 0.5 * k) for k in (1, 2)},
                **{f"t{k}": native * t[k] for k in (1, 2)}}
        return {"weight_bytes": wb, "times": {1: {"native_s": native, "median_s": arms}}}

    cases = {"a": case(3, 1.0, {1: 2.0, 2: 2.5}), "b": case(1, 1.0, {1: 2.0, 2: 3.5})}
    rows = wide_bench.mixed(cases, [1], [1, 2], commit_s=0.0)[1]
    got = [(r["ks"], r["k_mean"], r["y_total"]) for r in rows]
    assert got == [({"a": 1, "b": 1}, 1.0, 2.0), ({"a": 2, "b": 1}, 1.75, 2.25), ({"a": 2, "b": 2}, 2.0, 3.0)]
    lib = types.SimpleNamespace(field=wide.field(521), sqr=1, roll=1, occ=1, cc=1, b3u=1, tb=1, kw=1, name="p521s1rcu")
    fq = {k: {"x": 250.0 * k, "delta": 0.6 * k, "tau": 0.8, "mu_rebuild": 0.0} for k in (1, 2)}
    pts = F.check(wide_bench.points(lib, fq, {}, "r", "gpu", mx={1: rows}))
    assert len(pts) == 1 and pts[0]["arithmetic"].endswith("/mixed-k") and pts[0]["k"] == 1.75
    assert pts[0]["x"] == pytest.approx(437.5) and pts[0]["delta"] == pytest.approx(0.75 * 1.2 + 0.25 * 0.6)
    assert pts[0]["label"] == "p521 k=a2b1 p521s1rcu/mixed-k"


def test_tau_counts_a_batched_tags_compressions_over_its_records():
    import wide
    import wide_bench

    f = wide.field(521)
    one = wide_bench.frontier_quantities(f, 4, 5, 1600.0, 1e9, {1: 1e9}, 4e10, [1])[1]
    two = wide_bench.frontier_quantities(f, 4, 9 / 2, 1600.0, 1e9, {1: 1e9}, 4e10, [1])[1]
    assert two["tau"] == pytest.approx(one["tau"] * 0.9) and two["x"] == one["x"]
