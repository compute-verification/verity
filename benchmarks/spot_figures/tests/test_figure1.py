"""Figure 1's pipeline: its input cards, its world against the census's, the curves' bisection, their shape, the
infiltration panel from its card, and the JSON and renders the console and the post read."""
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from math import ceil, log, log1p

import cards
import figure1 as f
import pytest
from render import _fit, _lines, _zeros, appendix_stem, render, render_appendix, stem
from verity.core.protocols.spot.optimize import Cell

SCOPES = ("world year", "datacenter week")
#: the designs every bit charged, which plan in about a second; the capped ones take seconds a point
UNCAPPED = tuple(d for ds in f.DESIGNS.values() for d in ds if f.TWO_STAGE.get(d, (True, None))[1] is None)


@pytest.fixture(scope="module")
def out() -> dict:
    return f.run("GLM-5.3", 3, SCOPES, designs=UNCAPPED, lean=False)


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


def test_a_model_without_a_workload_card_is_refused(card_dir):
    (card_dir / "workload.deepseek-v4-pro.json").unlink()
    with pytest.raises(FileNotFoundError, match="no workload card for DeepSeek-V4-Pro"):
        f.cards("DeepSeek-V4-Pro")


def test_a_model_runs_once_its_workload_card_lands(card_dir):
    (card_dir / "workload.deepseek-v4-pro.json").unlink()
    work = cards.load("workload", "GLM-5.3")
    work["value"].pop("params")
    cards.write(work | {"model": "DeepSeek-V4-Pro"})
    out = f.run("DeepSeek-V4-Pro", 2, ("world year",), designs=UNCAPPED, lean=False)
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
    for card in out["cards"]["references"]:
        for panel in {line["panel"] for line in card["value"]["lines"]}:
            assert card["art"] in {r.get("art") for r in out["panels"][panel]["references"]}, (card["art"], panel)


def test_a_line_at_zero_is_drawn_on_the_floor_and_captioned_and_coincident_lines_plotted_once():
    refs = [{"label": "a", "value": 100.0}, {"label": "b", "value": 101.0}, {"label": "zero", "value": 0.0},
            {"label": "c", "by_scope": {"world year": 1e4}}, {"label": "elsewhere", "by_scope": {"world week": 1.0}}]
    assert _lines(refs, "world year", 1.0) == [(100.0, "a", "dashed"), (1e4, "c", "dashed")]
    assert _lines([refs[0] | {"style": "solid"}], "world year", 1.0) == [(100.0, "a", "solid")]
    assert f.zero_lines({"exfiltration": refs}, "world year") == {"exfiltration": ["zero"]}
    assert f.captions(1.0, zeros={"exfiltration": ["zero"]})["exfiltration"].endswith(
        " At 0, drawn along the axis's floor: zero.")
    assert _zeros(refs, "world year") == [("zero", "dashed")] and _zeros(refs, "world week") == [("zero", "dashed")]


def test_a_label_wider_than_its_axis_wraps_onto_balanced_lines_inside_it():
    """From either side, a label is wrapped onto as few lines as keep it inside its axis, none of them a stray word;
    one that fits stays on one line."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    frame = ax.get_window_extent(fig.canvas.get_renderer())
    long = "Advice once the length rule is checked in-circuit, prefix caching off (pending): 0, on the axis's floor"
    for x, ha in ((.01, "left"), (.99, "right")):
        t = ax.text(x, .5, long, ha=ha, transform=ax.transAxes, fontsize=15)
        _fit(t)
        box, lines = t.get_window_extent(fig.canvas.get_renderer()), t.get_text().split("\n")
        assert len(lines) > 1 and " ".join(lines) == long and frame.x0 <= box.x0 and box.x1 <= frame.x1
        assert min(map(len, lines)) > len(long) / (2 * len(lines))
    short = ax.text(.01, .2, "a short label", transform=ax.transAxes, fontsize=15)
    _fit(short)
    assert short.get_text() == "a short label"
    plt.close(fig)


def test_a_capped_caption_says_what_it_rests_on():
    """The capped sentences name each form's status, the patterns' form only in the appendix, beside every panel's own
    sentences."""
    caps = f.captions(1.0, boundary_hash_share=1e-7, served=(144000.0, 4500, 32, 3.36e11, 1.5e16))
    assert "Capped (dash-dot; card_influenceTwo_le_served_min): two-stage with each request charged at most the " \
           "144,000 bits it serves (4,500 tokens of 32 bits)" in caps["exfiltration"]
    assert "never above the 1.5e+16 bits the scope serves (card_influenceTwo_le_min)" in caps["exfiltration"]
    assert "Patterns" not in caps["exfiltration"]
    assert "Patterns (dash-dot-dot; card_influenceTwo_le_patterns)" in caps["exfiltration_appendix"]
    for panel in ("exfiltration", "skipped_work"):
        assert "the token ids, 1e-07 of the scope's compute" in caps[panel], panel


def test_every_caption_names_spots_roles_and_what_its_bounds_rest_on():
    """Each panel's caption says, after its deployment, that the proof server proves with no ZK server, that the
    network certifier is a layer of its own, and that its bound is soundness alone on the trust list."""
    caps = f.captions(1.0)
    for panel in ("exfiltration", "exfiltration_appendix", "skipped_work"):
        deployment = f.DEPLOYMENT["skipped_work" if panel == "skipped_work" else "exfiltration"]
        assert caps[panel].startswith(f"{deployment} {f.ROLES} {f.TRUST} "), panel
    assert "no ZK server and no recursion" in f.ROLES and "network certifier" in f.ROLES
    assert all(x in f.TRUST for x in ("soundness alone", "SHA-512's collision resistance",
                                       "uniform verifier randomness", "an honest verifier", "certifier's own assumptions"))
    assert "zero knowledge" not in f.ROLES + f.TRUST


def test_a_references_cards_caption_joins_its_panels_captions():
    line = {"label": "x", "unit": "bit", "value": 1.0, "source": "s"}
    notes = f.reference_notes([
        {"value": {"lines": [line | {"panel": "exfiltration"}] * 2, "caption": "Caps."}},
        {"value": {"lines": [line | {"panel": "skipped_work"}]}}])
    assert notes == {"exfiltration": ["Caps."]}
    caps = f.captions(1.0, notes=notes)
    assert caps["exfiltration"].endswith(" Caps.") and "Caps." not in caps["skipped_work"]
    off = {"exfiltration": [{"label": "x", "main": False}], "skipped_work": [{"label": "x", "main": True}]}
    assert f.reference_notes([{"value": {"lines": [line | {"panel": "exfiltration"}], "caption": "Caps."}}],
                             off) == {"exfiltration_appendix": ["Caps."]}


def test_the_design_card_picks_the_main_panels_lines(out):
    """The main exfiltration panel plots the lines the `design` card's `main_lines` names, in its styles, and the
    appendix the rest, at most six among the curves with the designs (a line at 0 is on the floor); the other panels
    plot every line, and the card's notes end the captions."""
    chosen = out["inputs"]["main_lines"]["value"]["exfiltration"]
    refs = out["panels"]["exfiltration"]["references"]
    main = [r for r in refs if r["main"]]
    among = [r for r in main if max([r["value"]] if "value" in r else r["by_scope"].values()) > 0]
    assert {r["label"] for r in main} == {"GLM-5.3's weights", "GLM-5.3's weights at FP4 (376.7 GB)",
                                         "Advice today: the response length, 18 bits a request (no circuit checks "
                                         "the stop rule yet)",
                                         "Advice once the length rule is checked in-circuit, prefix caching off "
                                         "(pending)"}
    assert 0 < len(among) <= 6 - len(f.MAIN["exfiltration"]) and len(refs) > len(main)
    for r in main:
        assert r["style"] == next(v for k, v in chosen.items() if r["label"].startswith(k.format(model="GLM-5.3")))
    assert all(r["main"] for p in ("infiltration", "skipped_work") for r in out["panels"][p]["references"])
    caps = out["scopes"]["world year"]["captions"]
    for panel, sentences in out["inputs"]["notes"]["value"].items():
        assert all(s in caps[panel] for s in sentences)
    assert all(s["main"] == (s["design"] in f.MAIN["exfiltration"]) for s in out["panels"]["exfiltration"]["series"])


def test_the_infiltration_panel_draws_its_card(card_dir, tmp_path):
    vertical = cards.load("proof_overhead", "GLM-5.3")["value"]["model_overhead"]
    xs = [1.0, vertical, 1e14]
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
    out = f.run("GLM-5.3", 3, SCOPES, designs=UNCAPPED, lean=False)
    inf = out["panels"]["infiltration"]
    assert inf["status"] == "card" and inf["headline"] == "layer_of_token" and len(inf["series"]) == 6
    caption = out["scopes"]["world year"]["captions"]["infiltration"]
    head = next(s for s in out["panels"]["exfiltration"]["series"]
                if (s["scope"], s["budget"], s["design"]) == ("world year", f.RATIO_BUDGET, "two-stage"))
    ratio = 1e9 / head["y"][head["x"].index(vertical)]
    assert caption.startswith("Deferred interiors at a layer. At the measured overhead and 1% of compute, this curve "
                              f"is 1e9 bits, {ratio:.2g}× the exfiltration panel's tokens-only two-stage bound")
    assert out["scopes"]["world year"]["captions"]["exfiltration_appendix"].endswith(
        " At 0, drawn along the axis's floor: no advice.")
    assert "no advice" not in out["scopes"]["world year"]["captions"]["exfiltration"]
    assert {"label": "a steering vector", "input": "references", "value": 98304.0, "main": True,
            "style": "dashed"} in inf["references"]
    line = out["panels"]["skipped_work"]["references"][-1]
    assert line["by_scope"]["world year"] == pytest.approx(1e30 * 2 / out["inputs"]["ands_per_mac"]["value"])
    assert all(p.stat().st_size > 0 for p in render(out, "world year", tmp_path))
    assert all(p.stat().st_size > 0 for p in render(out, "datacenter week", tmp_path))
    assert all(p.stat().st_size > 0 for p in render_appendix(out, "world year", tmp_path))


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
    for design in (d for d in f.DESIGNS[panel] if d not in f.TWO_STAGE):
        p = f.point(m, design, target, budget)
        assert not p["capped"], design
        assert f.cost(m, design, target, p["bound"])["total"] <= budget, design
        assert f.cost(m, design, target, p["bound"] * (1 - 1e-3))["total"] > budget, design


def test_a_one_stage_or_request_point_is_what_lean_certifies(world, monkeypatch):
    """Each one-stage and request-level point's design goes to Lean's checker in one run: a leak point becomes the bits
    Lean certifies, a point at the trivial bound has no design, and a design Lean refuses is an error.  Lean's answers
    are stubbed; the sampled-proofs suite runs the checker on the planner's designs (`test_plancheck.py`)."""
    c, part, fill, s = world
    m = f.model(c, part, s, c["proof_overhead"].value, fill)
    pts = [(m, d, t, f.point(m, d, t, 0.01 * s.gpu_seconds)) for d in ("one-stage", "request") for t in ("leak", "work")]
    runs = []

    def check(designs):
        runs.append(designs)
        return [Fraction(10**6, 7) if d["kind"] == "leak" else True for d in designs]

    monkeypatch.setattr(f.plancheck, "check", check)
    got = f.lean_check([*pts, (m, "one-stage", "leak", {"bound": 1e30, "capped": True})])
    lean = f.plancheck.upper(Fraction(10**6, 7))
    assert got == [(True, lean), (True, pts[1][3]["bound"]), (True, lean), (True, pts[3][3]["bound"]), (None, 1e30)]
    (designs,) = runs
    assert [(d["kind"], d.get("units")) for d in designs] == [
        ("leak", str(m.dot.units)), ("work", None), ("leak", str(m.request.units)), ("work", None)]
    assert designs[0]["count"] == str(f.leak_count(m.dot, pts[0][3]["bound"]))
    monkeypatch.setattr(f.plancheck, "check", lambda ds: [None if d["kind"] == "leak" else True for d in ds])
    with pytest.raises(f.plancheck.LeanError, match="refuses the planner's one-stage design"):
        f.lean_check(pts)


def test_lean_two_sends_the_planners_design_in_each_form(monkeypatch):
    """`lean_two` sends the planner's design to Lean's checker (`plancheck.two` of its cells in the row's strata, its
    replay blocks and the planner's θ): every bit charged, or each replay block at most the bits it serves in the
    capped sets' form or the patterns'.  The count is the bits Lean certifies, rounded up to a float, with the
    planner's bits and Lean's θ beside it; a point Lean refuses is an error.  The checker is stubbed here; the
    sampled-proofs suite runs it."""
    cells = (Cell(3, 16), Cell(1, 64, stratum=64))
    row = {"merged": 64, "k": {0: 1}, "q": .5, "theta": 2.0, "bound": 50.0}
    seen = []
    monkeypatch.setattr(f.plancheck, "check", lambda ds: seen.extend(ds) or [(Fraction(10**6, 7), Fraction(3))])
    want = (f.plancheck.upper(Fraction(10**6, 7)), {"lean": True, "planner": 50.0, "theta": 3.0})
    for served, union, form in ((None, False, "sets"), (80.0, False, "capped"), (80.0, True, "patterns")):
        seen.clear()
        assert f.lean_two(cells, 4, row, .01, served, union) == want
        assert seen == [f.plancheck.two([Cell(3, 16), Cell(1, 64)], {0: 1}, 4, .5, .01, served, union, 2.0)]
        assert seen[0]["form"] == form
    monkeypatch.setattr(f.plancheck, "check", lambda ds: [(None, Fraction(0))])
    with pytest.raises(f.plancheck.LeanError, match="refuses the planner's two-stage point"):
        f.lean_two(cells, 4, row, .01, 80.0, True)


#: the bits the stubbed checker certifies for a two-stage point, and the θ it finds
LEAN_BITS, LEAN_THETA = Fraction(10**9, 7), Fraction(3)


def lean_in_range(seen: list):
    """A stub of Lean's checker on two-stage points: it refuses a point out of range, as `TwoStageCheck.Point.valid`
    does (q outside [0, 1] or δ at most 0: test_plancheck.py's out-of-range test runs the checker on them), and
    certifies `LEAN_BITS` at `LEAN_THETA` otherwise."""
    def check(designs):
        seen.extend(designs)
        return [(None, Fraction(d["theta"])) if not (0 <= Fraction(d["q"]) <= 1 and Fraction(d["delta"]) > 0)
                else (LEAN_BITS, LEAN_THETA) for d in designs]
    return check


def test_a_two_stage_point_is_what_lean_certifies_and_a_tampered_design_is_refused(world, monkeypatch):
    """A two-stage point's bound is the bits Lean certifies for the planner's design, not the planner's own bits, at
    most its trivial bound, with `lean`, the planner's bits and Lean's θ beside it; each design goes to Lean in its own
    form, with the bits a request serves under a capped one.  A point at the trivial bound has no design to check.  A
    design tampered out of Lean's range (a first-stage rate above 1, or δ = 0) is refused, and a refusal is an error."""
    c, part, fill, s = world
    m = f.model(c, part, s, c["proof_overhead"].value, fill)
    p = f.point(m, "two-stage", "leak", 0.01 * s.gpu_seconds)
    row, seen = p["row"], []
    monkeypatch.setattr(f.plancheck, "check", lean_in_range(seen))
    lean = f.plancheck.upper(LEAN_BITS)
    assert not p["capped"] and p["bound"] != lean
    served = str(ceil(m.served))
    forms = {None: ("sets", "0"), "sets": ("capped", served), "patterns": ("patterns", served)}
    for design, (_, form) in f.TWO_STAGE.items():
        seen.clear()
        got = f.lean_point(m, design, p)
        top = f.two_stage_top(m, design)
        assert (got["bound"], got["capped"]) == (min(lean, top), lean >= top), design
        assert (got["lean"], got["planner"], got["theta"]) == (True, row["bound"], float(LEAN_THETA)), design
        assert seen == [f.plancheck.two(f.split_cells(m.cells, row["merged"]), row["k"], m.replay.replay_blocks,
                                        row["q"], m.delta, None if form is None else m.served, form == "patterns",
                                        row["theta"])]
        assert (seen[0]["form"], seen[0]["S"]) == forms[form], design
    seen.clear()
    assert f.lean_point(m, "two-stage", p | {"capped": True}) == p | {"capped": True, "lean": None,
                                                                    "planner": row["bound"], "theta": None}
    assert f.lean_point(m, "two-stage", p | {"row": None})["lean"] is None and not seen
    for tampered, at in ((p | {"row": row | {"q": 1.5}}, m), (p, replace(m, delta=0.0))):
        with pytest.raises(f.plancheck.LeanError, match="refuses the planner's two-stage point"):
            f.lean_point(at, "two-stage", tampered)


def test_a_two_stage_series_carries_lean_and_no_lean_plans_without_it(monkeypatch, out):
    """With Lean on, each two-stage series' points are Lean's (`lean_point`): `lean` for each, beside the planner's
    bits and Lean's θ, the trivial bound's unchecked, and the JSON says how Lean searched.  Without it (`--no-lean`),
    a series is the planner's and carries none of them.  The planner is stubbed here, its first point at the trivial
    bound and every other one row, since the planner's own points are the test above's."""
    seen, rows = [], iter([None])
    row = {"merged": 1 << 18, "k": {0: 1}, "q": .5, "theta": 2.0, "bound": 50.0}
    monkeypatch.setattr(f.plancheck, "check", lean_in_range(seen))
    monkeypatch.setattr(f, "two_stage", lambda m, design, budget: next(rows, row))
    checked = f.run("GLM-5.3", 2, ("world year",), designs=("two-stage",), lean=True)
    assert checked["lean"] == {"exe": f.plancheck.EXE, "slack": str(f.plancheck.SLACK), "search": f.plancheck.SEARCH,
                               "precision": f.plancheck.TWO_P}
    series = checked["panels"]["exfiltration"]["series"]
    assert len(series) == len(checked["budgets"]) and not checked["panels"]["skipped_work"]["series"]
    lean = f.plancheck.upper(LEAN_BITS)
    for s in series:
        assert len(s["lean"]) == len(s["planner"]) == len(s["theta"]) == len(s["x"])
        for y, on, planner, theta, capped in zip(s["y"], s["lean"], s["planner"], s["theta"], s["capped"]):
            if on:
                assert (on, y, planner, theta, capped) == (True, lean, row["bound"], float(LEAN_THETA), False)
            else:
                assert (on, planner, theta, capped) == (None, None, None, True)
    assert [s["lean"].count(None) for s in series] == [1] + [0] * (len(series) - 1)
    assert len(seen) == sum(s["lean"].count(True) for s in series) > 0 and {d["form"] for d in seen} == {"sets"}
    assert out["lean"] is None
    for s in out["panels"]["exfiltration"]["series"] + out["panels"]["skipped_work"]["series"]:
        assert not {"lean", "planner", "theta"} & set(s), s["design"]


def test_an_incorrect_block_leaks_the_widest_blocks_output(world):
    """Proof blocks are sampled uniformly, so the adversary makes the widest incorrect first (`harm_bound`'s order):
    the leak grain is as wide as the widest stratum, not the mean, and a sampled block costs the count-weighted MACs."""
    c, part, fill, s = world
    m = f.model(c, part, s, 1.0, fill)
    assert m.dot.out_bits == max(x["out_bits"] for x in c["partition"].value["strata"]) > part.bits
    assert m.dot.prove == pytest.approx(2 * part.macs / part.blocks / c["peak_flops"].value)
    assert f.certified(m, "one-stage", "leak", 1e12) >= m.dot.out_bits * f.leak_count(m.dot, 1e12)


def test_two_stage_certifies_no_work(world):
    c, part, fill, s = world
    m = f.model(c, part, s, 1.0, fill)
    with pytest.raises(ValueError, match="no work target"):
        f.point(m, "two-stage", "work", 1e20)
    with pytest.raises(ValueError, match="the planner's at a budget"):
        f.cost(m, "two-stage-uniform", "leak", 1e20)


@pytest.fixture(scope="module")
def two_stage_points(world):
    """Each two-stage design's point at 1% of the world year's compute, at the measured overhead."""
    c, part, fill, s = world
    m = f.model(c, part, s, c["proof_overhead"].value, fill)
    budget = 0.01 * s.gpu_seconds
    return m, budget, {d: f.point(m, d, "leak", budget) for d in f.TWO_STAGE}


@pytest.mark.slow
def test_a_two_stage_point_is_the_planners_design_at_its_budget(world, two_stage_points):
    """A two-stage point is `two_stage_design`'s on the model's cells, a request a replay block and the token ids its
    boundary, paid from the budget at its statement fill.  It lets ln(1/δ)/q − 1 wholly incorrect requests through,
    each charged every bit it commits or, capped, the bits it serves, so its bound is at least theirs; a capped bound
    is at most the uncapped one, the patterns' form at most the sets', and width-stratified at most uniform."""
    c, part, fill, _ = world
    m, budget, points = two_stage_points
    bits = sum(x.n * x.bits for x in m.cells)
    assert bits == pytest.approx(c["tokens_per_request"].value * part.bits * part.blocks)
    assert m.served == c["served_bits"].value < bits
    rows = {}
    for design, (_, form) in f.TWO_STAGE.items():
        p = points[design]
        row = rows[design] = p["row"]
        assert p["bound"] == row["bound"] and not p["capped"]
        assert design not in f.MAIN["exfiltration"] or row == f.two_stage(m, design, budget)
        assert row["total"] <= budget * (1 + 1e-9) and row["boundary"] == pytest.approx(
            m.replay.replay_blocks * m.replay.boundary_bytes * m.byte_s)
        assert p["bound"] >= (log(1 / m.delta) / -log1p(-row["q"]) - 1) * (bits if form is None else m.served)
        assert 1 + min(r / cap for _, _, r, cap in fill.types) <= row["fill"] <= 1 + max(r for _, _, r, _ in fill.types)
    slack = 1 + 1e-3
    for suffix in ("", "-capped", "-patterns"):
        assert rows[f"two-stage{suffix}"]["bound"] <= rows[f"two-stage{suffix}-uniform"]["bound"] * slack
    for suffix in ("", "-uniform"):
        assert rows[f"two-stage-patterns{suffix}"]["bound"] <= rows[f"two-stage-capped{suffix}"]["bound"] * slack
        assert rows[f"two-stage-capped{suffix}"]["bound"] <= rows[f"two-stage{suffix}"]["bound"] * slack
    assert set(rows["two-stage-uniform"]["k"]) == {0} and len(rows["two-stage"]["k"]) > 1


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
    """One-stage proves one block a statement, each type paying its own statement's fixed cost (an unmeasured type the
    measured ones' pooled cost), MAC-weighted, and request-level a request's blocks (two-stage the k its planner picks,
    at that price: the planner's-design test)."""
    c, part, fill, s = world
    m = f.model(c, part, s, c["proof_overhead"].value, fill)
    strata, blocks = c["partition"].value["strata"], c["proof_blocks"].value
    by = {b["block"]: b["fixed_seconds"] / b["seconds_per_block"] for b in blocks}
    pooled = sum(b["fixed_seconds"] for b in blocks) / sum(b["seconds_per_block"] for b in blocks)
    macs = sum(x["per_token"] * x["macs"] for x in strata)
    one = sum(x["per_token"] * x["macs"] / macs * (1 + by.get(x["block"], pooled)) for x in strata)
    assert f.cost(m, "one-stage", "leak", 1e12)["fill"] == pytest.approx(one)
    assert f.cost(m, "request", "leak", 1e18)["fill"] == pytest.approx(fill.factor(f.inf), rel=1e-6)


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


def rises_and_falls(out: dict) -> None:
    """Every planned series rises with overhead and falls with budget."""
    for panel in ("exfiltration", "skipped_work"):
        series, designs = out["panels"][panel]["series"], out["panels"][panel]["designs"]
        assert {(s["scope"], s["design"]) for s in series} == {(x, d) for x in out["scopes"] for d in designs}
        for s in series:
            assert all(a <= b * (1 + 1e-3) for a, b in zip(s["y"], s["y"][1:])), (s["scope"], s["design"])
        for scope in out["scopes"]:
            for design in designs:
                ys = [s["y"] for s in sorted(series, key=lambda s: s["budget"])
                      if (s["scope"], s["design"]) == (scope, design)]
                for more, less in zip(ys, ys[1:]):
                    assert all(b <= a * (1 + 1e-3) for a, b in zip(more, less)), (panel, scope, design)


def test_curves_rise_with_overhead_and_fall_with_budget(out):
    assert out["panels"]["exfiltration"]["designs"] == [d for d in f.DESIGNS["exfiltration"] if d in UNCAPPED]
    assert not out["panels"]["exfiltration"]["capped"]
    assert "Capped" not in out["scopes"]["world year"]["captions"]["exfiltration"]
    rises_and_falls(out)


@pytest.mark.slow
def test_the_capped_curves_charge_at_most_what_a_request_serves(tmp_path):
    """The capped curves, which plan in seconds a point and no cheaper test plots: each rises with overhead and falls
    with budget, the sets' form is at most the uncapped curve at every point and the patterns' form at most the sets',
    and the panel and captions say what each rests on."""
    out = f.run("GLM-5.3", 2, ("datacenter week",), designs=("two-stage", "two-stage-capped", "two-stage-patterns"),
                lean=False)
    rises_and_falls(out)
    p = out["panels"]["exfiltration"]
    assert p["capped"] == {"two-stage-capped": {"form": "sets", "status": "card_influenceTwo_le_served_min",
                                                "served": "card_influenceTwo_le_min"},
                           "two-stage-patterns": {"form": "patterns", "status": "card_influenceTwo_le_patterns",
                                                  "served": "card_influenceTwo_le_two_pow_outs"}}
    assert p["main"] == ["two-stage", "two-stage-capped"]
    ys = {(s["design"], s["budget"]): s["y"] for s in p["series"]}
    served = out["scopes"]["datacenter week"]["served_bits"]
    for beta in out["budgets"]:
        for patterns, sets, every in zip(*(ys[d, beta] for d in ("two-stage-patterns", "two-stage-capped",
                                                                  "two-stage"))):
            assert patterns <= sets * (1 + 1e-3) and sets <= every * (1 + 1e-3) and max(sets, patterns) <= served
    caps = out["scopes"]["datacenter week"]["captions"]
    assert (f"Capped (dash-dot; card_influenceTwo_le_served_min): two-stage with each request charged at most the "
            f"{out['inputs']['served_bits']['value']:,.0f} bits it serves") in caps["exfiltration"]
    assert "card_influenceTwo_le_patterns" in caps["exfiltration_appendix"] and "Patterns" not in caps["exfiltration"]
    assert all(path.stat().st_size > 0 for path in render(out, "datacenter week", tmp_path)
               + render_appendix(out, "datacenter week", tmp_path))


def test_a_request_serves_each_token_as_the_token_selects_word():
    """A request serves each of its tokens as the token select's 32-bit output word, the registered circuit's output,
    which is wider than the vocabulary's ⌈log2 vocab_size⌉ = 18 bits because an incorrect token select can write any
    word: the census's mean request or, with `request="workload"`, the workload card's; another request is refused."""
    for request, tokens in (("census", f._entry(f.ADVERSARY, f.CENSUS_WORLD)["tokens_per_request"]),
                            ("workload", cards.load("workload", "GLM-5.3")["value"]["tokens"])):
        c = f.cards("GLM-5.3", request)
        assert c["tokens_per_request"].value == tokens
        assert c["served_bits"].value == tokens * 8 * c["token_bytes"].value == tokens * 32
    with pytest.raises(ValueError, match="a request is one of"):
        f.cards(request="mean")


def test_the_headline_designs_win_at_the_measured_overhead(out):
    """The figure's claim at today's inputs: at the vertical line, two-stage bounds exfiltration below both comparisons,
    width-stratified at most uniform, and one-stage bounds skipped work below request-level sampling; where the
    winner is at the trivial bound, everything the scope outputs or computes, every design is there with it."""
    rows = f.at_overhead(out)
    for scope in SCOPES:
        for beta in out["budgets"]:
            here = [r for r in rows if (r["scope"], r["budget"]) == (scope, beta)]
            at = {(r["panel"], r["design"]): r["bound"] for r in here}
            top = {(r["panel"], r["design"]) for r in here if r["capped"]}

            def wins(panel: str, design: str, *others: str) -> bool:
                rest = min(at[panel, d] for d in others)
                return at[panel, design] < rest or ((panel, design) in top and at[panel, design] == rest)

            assert wins("exfiltration", "two-stage", "one-stage", "request")
            assert at["exfiltration", "two-stage"] <= at["exfiltration", "two-stage-uniform"]
            assert wins("skipped_work", "one-stage", "request")


def test_compute_and_skipped_work_are_in_and_gates_too(out):
    rate = out["inputs"]["ands_per_mac"]["value"] / 2
    assert out["panels"]["skipped_work"]["and_per_flop"] == rate
    for scope in SCOPES:
        assert out["scopes"][scope]["ands"] == pytest.approx(out["scopes"][scope]["flop"] * rate)
    for r in f.at_overhead(out):
        assert ("ands" in r) == (r["panel"] == "skipped_work")


def test_the_json_is_plain_and_renders(out, tmp_path):
    back = json.loads(json.dumps(out, allow_nan=False))
    assert back["schema"] == f.SCHEMA and back["panels"]["infiltration"]["status"] == "card"
    assert set(back["scopes"]) == set(SCOPES)
    for scope in SCOPES:
        assert set(back["scopes"][scope]["captions"]) == {"exfiltration", "exfiltration_appendix", "infiltration",
                                                          "skipped_work"}
    for panel in ("exfiltration", "infiltration", "skipped_work"):
        assert back["panels"][panel]["deployment"]
    paths = render(back, "datacenter week", tmp_path) + render_appendix(back, "datacenter week", tmp_path)
    assert [p.name for p in paths] == ["figure1_overhead_datacenter_week.png", "figure1_overhead_datacenter_week.svg",
                                       "figure1_overhead_appendix_datacenter_week.png",
                                       "figure1_overhead_appendix_datacenter_week.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert (stem("world year"), appendix_stem("world year")) == ("figure1_overhead", "figure1_overhead_appendix")
