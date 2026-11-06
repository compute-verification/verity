"""Figure 1's pipeline: its input cards, its world against the census's, the curves' bisection, their shape, and the
JSON and renders the console and the post read."""
import json

import figure1 as f
import pytest
from render import render, stem

SCOPES = ("world year", "datacenter week")


@pytest.fixture(scope="module")
def out() -> dict:
    return f.run(points=3, scope_names=SCOPES)


@pytest.fixture(scope="module")
def world():
    c = f.cards()
    part = f.TokenBlocks.of(c["partition"].value["strata"])
    return c, part, next(s for s in f.scopes(c, part) if s.name == "world year")


def test_every_input_is_a_card_with_its_source(out):
    for name, card in out["inputs"].items():
        assert card["status"] in ("measured", "modeled") and card["source"], name
    assert out["vertical"]["x"] == out["inputs"]["proof_overhead"]["value"]
    assert ("provisional" in out["vertical"]["label"]) == out["inputs"]["proof_overhead"]["provisional"]
    for status, source in (("assumed", "a source"), ("measured", "")):
        with pytest.raises(ValueError, match="measured or modeled, with its source"):
            f.InputCard(1, "", status, source)


def test_the_world_fleet_computes_the_census_world(world):
    """rho's world fleet at serving's MFU over a year computes the census's world, and its requests are that compute at
    GLM-5.3's FLOP a request."""
    c, _, s = world
    flop = s.gpu_seconds * c["mfu_serve"].value * c["peak_flops"].value
    assert flop == pytest.approx(f._entry(f.ADVERSARY, f.CENSUS_WORLD)["total_flop"], rel=1e-3)
    assert s.requests * s.flop_request == pytest.approx(flop)


@pytest.mark.parametrize("panel,target", [("exfiltration", "leak"), ("skipped_work", "work")])
def test_a_point_is_the_least_target_its_budget_buys(world, panel, target):
    c, part, s = world
    m = f.model(c, part, s, c["proof_overhead"].value)
    budget = 0.01 * s.gpu_seconds
    for design in f.DESIGNS[panel]:
        p = f.point(m, design, target, budget)
        assert not p["capped"], design
        assert f.cost(m, design, target, p["bound"])["total"] <= budget, design
        assert f.cost(m, design, target, p["bound"] * (1 - 1e-3))["total"] > budget, design


def test_two_stage_certifies_no_work(world):
    c, part, s = world
    with pytest.raises(ValueError, match="no work target"):
        f.cost(f.model(c, part, s, 1.0), "two-stage", "work", 1e20)


def test_curves_rise_with_overhead_and_fall_with_budget(out):
    for panel in ("exfiltration", "skipped_work"):
        series = out["panels"][panel]["series"]
        assert {(s["scope"], s["design"]) for s in series} == {(x, d) for x in SCOPES for d in f.DESIGNS[panel]}
        for s in series:
            assert all(a <= b * (1 + 1e-3) for a, b in zip(s["y"], s["y"][1:])), (s["scope"], s["design"])
        for scope in SCOPES:
            for design in f.DESIGNS[panel]:
                ys = [s["y"] for s in sorted(series, key=lambda s: s["budget"])
                      if (s["scope"], s["design"]) == (scope, design)]
                for more, less in zip(ys, ys[1:]):
                    assert all(b <= a * (1 + 1e-3) for a, b in zip(more, less)), (panel, scope, design)


def test_the_headline_designs_win_at_the_measured_overhead(out):
    """The figure's claim at today's inputs: at the vertical line, two-stage bounds exfiltration below both comparisons,
    and one-stage bounds skipped work below request-level sampling."""
    rows = f.at_overhead(out)
    for scope in SCOPES:
        for beta in out["budgets"]:
            at = {(r["panel"], r["design"]): r["bound"] for r in rows if (r["scope"], r["budget"]) == (scope, beta)}
            assert at["exfiltration", "two-stage"] < min(at["exfiltration", "one-stage"], at["exfiltration", "request"])
            assert at["skipped_work", "one-stage"] < at["skipped_work", "request"]


def test_compute_and_skipped_work_are_in_and_gates_too(out):
    rate = out["inputs"]["ands_per_mac"]["value"] / 2
    assert out["panels"]["skipped_work"]["and_per_flop"] == rate
    for scope in SCOPES:
        assert out["scopes"][scope]["ands"] == pytest.approx(out["scopes"][scope]["flop"] * rate)
    for r in f.at_overhead(out):
        assert ("ands" in r) == (r["panel"] == "skipped_work")


def test_the_json_is_plain_and_renders(out, tmp_path):
    back = json.loads(json.dumps(out, allow_nan=False, default=str))
    assert back["schema"] == f.SCHEMA and back["panels"]["infiltration"]["status"] == "pending"
    assert set(back["scopes"]) == set(SCOPES)
    for scope in SCOPES:
        assert set(back["scopes"][scope]["captions"]) == {"exfiltration", "infiltration", "skipped_work"}
    paths = render(back, "datacenter week", tmp_path)
    assert [p.name for p in paths] == ["figure1_datacenter_week.png", "figure1_datacenter_week.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert stem("world year") == "figure1"
