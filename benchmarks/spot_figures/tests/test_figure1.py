"""Figure 1's pipeline: its input cards, its world against the census's, the curves' bisection, their shape, the
infiltration panel from its card, and the JSON and renders the console and the post read."""
import json
import shutil

import cards
import figure1 as f
import pytest
from render import _lines, render, stem

SCOPES = ("world year", "datacenter week")


@pytest.fixture(scope="module")
def out() -> dict:
    return f.run("GLM-5.3", 3, SCOPES)


@pytest.fixture
def card_dir(tmp_path, monkeypatch):
    """The tree's cards, copied where a test may add one."""
    shutil.copytree(cards.CARDS, tmp_path / "cards")
    monkeypatch.setattr(cards, "CARDS", tmp_path / "cards")
    return tmp_path / "cards"


def envelope(slot: str, model: str | None, value: dict) -> dict:
    return {"schema": cards.SCHEMA, "slot": slot, "model": model, "status": "modeled", "provisional": True,
            "source": "a test", "runs": [], "owner": "@proofs", "value": value}


@pytest.fixture(scope="module")
def world():
    c = f.cards()
    part = f.TokenBlocks.of(c["partition"].value["strata"])
    fill = f.Fill.of(c["partition"].value["strata"], c["proof_blocks"].value)
    return c, part, fill, next(s for s in f.scopes(c, part) if s.name == "world year")


def test_every_input_is_a_card_with_its_source(out):
    for name, card in out["inputs"].items():
        assert card["status"] in ("measured", "modeled") and card["source"], name
    assert out["vertical"]["x"] == out["inputs"]["proof_overhead"]["value"]
    assert ("provisional" in out["vertical"]["label"]) == out["inputs"]["proof_overhead"]["provisional"]
    assert out["vertical"]["x"] == out["cards"]["proof_overhead"]["value"]["model_overhead"]
    for status, source in (("assumed", "a source"), ("measured", "")):
        with pytest.raises(ValueError, match="measured or modeled, with its source"):
            f.InputCard(1, "", status, source)


def test_a_model_without_a_workload_card_is_refused():
    with pytest.raises(FileNotFoundError, match="no workload card for DeepSeek-V4-Pro"):
        f.cards("DeepSeek-V4-Pro")


def test_a_model_runs_once_its_workload_card_lands(card_dir):
    work = cards.load("workload", "GLM-5.3")
    work["value"].pop("params")
    cards.write(work | {"model": "DeepSeek-V4-Pro"})
    out = f.run("DeepSeek-V4-Pro", 2, ("world year",))
    assert out["model"] == "DeepSeek-V4-Pro"
    assert {out["cards"][s]["model"] for s in ("workload", "proof_overhead", "replay")} == {"DeepSeek-V4-Pro"}
    syncs, weights = out["panels"]["exfiltration"]["references"][:2]
    assert weights["label"] == "DeepSeek-V4-Pro's weights" and "references" not in out["cards"]
    assert (syncs["label"], syncs["value"]) == ("1,000 SparseLoCo syncs of DeepSeek-V4-Pro",
                                                out["inputs"]["ref_sparseloco"]["value"])


def test_sparseloco_goes_as_the_served_models_parameters(out):
    """rho's syncs are its reference model's: the line is scaled to the served model's parameters, and without them
    names the model it is sized for; rho's work references name their model."""
    c = out["inputs"]
    syncs = out["panels"]["exfiltration"]["references"][0]
    assert (syncs["label"], syncs["scaled_by"]) == ("1,000 SparseLoCo syncs of GLM-5.3", "params / ref_params")
    assert syncs["value"] == pytest.approx(c["ref_sparseloco"]["value"] * c["params"]["value"] / c["ref_params"]["value"])
    assert syncs["value"] / 8 == pytest.approx(8.0e12, rel=.01)
    assert [r["label"] for r in out["panels"]["skipped_work"]["references"][:2]] == [
        "DeepSeek-V4-Pro's pretraining run", "One optimizer step of DeepSeek-V4-Pro"]
    bare = {k: v for k, v in f.cards().items() if k != "params"}
    assert f.references(bare, "GLM-5.3")["exfiltration"][0]["label"] == "1,000 SparseLoCo syncs of a 1.6T-parameter model"


def test_every_references_card_adds_its_lines(out):
    arts = [c["art"] for c in out["cards"]["references"]]
    assert len(arts) == len(set(arts)) >= 2
    drawn = {r.get("art") for r in out["panels"]["exfiltration"]["references"]}
    assert set(arts) <= drawn


def test_a_line_at_zero_is_captioned_and_coincident_lines_drawn_once():
    refs = [{"label": "a", "value": 100.0}, {"label": "b", "value": 101.0}, {"label": "zero", "value": 0.0},
            {"label": "c", "by_scope": {"world year": 1e4}}, {"label": "elsewhere", "by_scope": {"world week": 1.0}}]
    assert _lines(refs, "world year", 1.0) == [(100.0, "a"), (1e4, "c")]
    assert f.zero_lines({"exfiltration": refs}, "world year") == {"exfiltration": ["zero"]}
    assert f.captions(1.0, zeros={"exfiltration": ["zero"]})["exfiltration"].endswith(" At 0, so not drawn: zero.")


def test_the_infiltration_panel_draws_its_card(card_dir, tmp_path):
    xs = [1.0, 1.5e8, 1e14]
    card = envelope("infiltration", "GLM-5.3", {
        "design": "layer_of_token", "form": "write", "unit": "bit", "caption": "Deferred interiors at a layer.",
        "consumer": None, "series": [{"scope": s, "budget": b, "x": xs, "y": [1e3, 1e9, 1e15]}
                                     for s in SCOPES for b in (0.001, 0.01, 0.1)]})
    cards.write(card)
    cards.write(envelope("references", None, {"lines": [
        {"panel": "exfiltration", "label": "no advice", "unit": "bit", "value": 0.0, "source": "a test"},
        {"panel": "infiltration", "label": "a steering vector", "unit": "bit", "value": 98304.0, "source": "a test"},
        {"panel": "skipped_work", "label": "every token", "unit": "AND", "by_scope": {"world year": 1e30},
         "source": "a test"}]}))
    out = f.run("GLM-5.3", 3, SCOPES)
    inf = out["panels"]["infiltration"]
    assert inf["status"] == "card" and inf["headline"] == "layer_of_token" and len(inf["series"]) == 6
    assert out["scopes"]["world year"]["captions"]["infiltration"] == "Deferred interiors at a layer."
    assert out["scopes"]["world year"]["captions"]["exfiltration"].endswith(" At 0, so not drawn: no advice.")
    assert inf["references"] == [{"label": "a steering vector", "input": "references", "value": 98304.0}]
    line = out["panels"]["skipped_work"]["references"][-1]
    assert line["by_scope"]["world year"] == pytest.approx(1e30 * 2 / out["inputs"]["ands_per_mac"]["value"])
    assert all(p.stat().st_size > 0 for p in render(out, "world year", tmp_path))
    assert all(p.stat().st_size > 0 for p in render(out, "datacenter week", tmp_path))


def test_the_world_fleet_computes_the_census_world(world):
    """rho's world fleet at serving's MFU over a year computes the census's world, and its requests are that compute at
    GLM-5.3's FLOP a request."""
    c, _, _, s = world
    flop = s.gpu_seconds * c["mfu_serve"].value * c["peak_flops"].value
    assert flop == pytest.approx(f._entry(f.ADVERSARY, f.CENSUS_WORLD)["total_flop"], rel=1e-3)
    assert s.requests * s.flop_request == pytest.approx(flop)


@pytest.mark.parametrize("panel,target", [("exfiltration", "leak"), ("skipped_work", "work")])
def test_a_point_is_the_least_target_its_budget_buys(world, panel, target):
    c, part, fill, s = world
    m = f.model(c, part, s, c["proof_overhead"].value, fill)
    budget = 0.01 * s.gpu_seconds
    for design in f.DESIGNS[panel]:
        p = f.point(m, design, target, budget)
        assert not p["capped"], design
        assert f.cost(m, design, target, p["bound"])["total"] <= budget, design
        assert f.cost(m, design, target, p["bound"] * (1 - 1e-3))["total"] > budget, design


def test_an_incorrect_block_leaks_the_widest_blocks_output(world):
    """Proof blocks are sampled uniformly, so the adversary makes the widest incorrect first (`harm_bound`'s order):
    the leak grain is as wide as the widest stratum, not the mean, and a sampled block costs the count-weighted MACs."""
    c, part, fill, s = world
    m = f.model(c, part, s, 1.0, fill)
    assert m.dot.out_bits == max(x["out_bits"] for x in c["partition"].value["strata"]) > part.bits
    assert m.dot.prove == pytest.approx(2 * part.macs / part.blocks / c["peak_flops"].value)
    assert f.certified(m, "two-stage", "leak", 1e12) >= m.dot.out_bits * f.leak_count(m.dot, 1e12)


def test_two_stage_certifies_no_work(world):
    c, part, fill, s = world
    with pytest.raises(ValueError, match="no work target"):
        f.cost(f.model(c, part, s, 1.0, fill), "two-stage", "work", 1e20)


def test_a_statement_pays_its_fixed_cost_over_its_fill():
    """A type's price over the marginal is 1 + fixed / (blocks in its statement), at most its cap; a type the card
    measures takes its own fixed cost and cap, the rest the measured blocks' pooled ones."""
    strata = [{"block": "A", "per_token": 3.0, "macs": 1, "out_bits": 16},
              {"block": "B", "per_token": 1.0, "macs": 1, "out_bits": 16}]
    blocks = [{"block": "A", "macs": 1, "seconds_per_block": 1.0, "fixed_seconds": 10.0, "statement_blocks": 4},
              {"block": "S", "macs": 1, "seconds_per_block": 1.0, "fixed_seconds": 30.0, "statement_blocks": 8}]
    fill = f.Fill.of(strata, blocks)
    assert fill.types == ((.75, .75, 10.0, 4), (.25, .25, 20.0, 4))
    assert fill.factor(1) == pytest.approx(.75 * 11 + .25 * 21)
    assert fill.factor(4) == pytest.approx(.75 * (1 + 10 / 3) + .25 * 21)
    assert fill.factor(f.inf) == pytest.approx(.75 * 3.5 + .25 * 6)


def test_each_design_is_priced_at_its_fill(world):
    """One-stage proves one block a statement, request-level a request's blocks, and two-stage the k its planner picks
    at that price."""
    c, part, fill, s = world
    m = f.model(c, part, s, c["proof_overhead"].value, fill)
    budget = 0.01 * s.gpu_seconds
    po = c["proof_blocks"].value[0]
    assert f.cost(m, "one-stage", "leak", 1e12)["fill"] == pytest.approx(1 + po["fixed_seconds"]
                                                                          / po["seconds_per_block"])
    assert f.cost(m, "request", "leak", 1e18)["fill"] == pytest.approx(fill.factor(f.inf), rel=1e-6)
    row = f.point(m, "two-stage", "leak", budget)["row"]
    assert row["fill"] == pytest.approx(fill.factor(row["k"]), rel=1e-6)
    assert fill.factor(f.inf) <= row["fill"] < fill.factor(1)


def test_each_design_states_its_serve_time_hashing(out):
    rows = f.at_overhead(out)
    for r in rows:
        sc = out["scopes"][r["scope"]]
        one = r["design"] == "one-stage"
        assert r["serve_hash_share"] == sc["serve_hash_share" if one else "boundary_hash_share"]
        assert r["serve_hash_in_budget"] is not one
        assert r["fill"] >= 1 if r["fill"] is not None else r["capped"]
    for scope in SCOPES:
        sc = out["scopes"][scope]
        assert sc["boundary_hash_share"] < 1e-6 < 1 < sc["serve_hash_share"]
        for panel in ("exfiltration", "skipped_work"):
            assert f"the token ids, {sc['boundary_hash_share']:.2g} of the scope's compute" in sc["captions"][panel]


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
    back = json.loads(json.dumps(out, allow_nan=False))
    assert back["schema"] == f.SCHEMA and back["panels"]["infiltration"]["status"] == "pending"
    assert set(back["scopes"]) == set(SCOPES)
    for scope in SCOPES:
        assert set(back["scopes"][scope]["captions"]) == {"exfiltration", "infiltration", "skipped_work"}
    paths = render(back, "datacenter week", tmp_path)
    assert [p.name for p in paths] == ["figure1_datacenter_week.png", "figure1_datacenter_week.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert stem("world year") == "figure1"
