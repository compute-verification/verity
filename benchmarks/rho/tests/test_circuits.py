import json
import sys

import pytest
from rho import analysis, circuits, instances, lean
from rho.classes import REFERENCE, structure
from rho.envelopes import ROUTES
from rho.families import layered
from rho.params import Registry
from rho.scenario import build, evaluate_named

REG = Registry.load()
STEP = "smollm2-135m-step"
C = circuits.circuit(REG, STEP)
MODEL = instances.model(REG, instances.run(REG, C.run).model)
MEASURED = ("gates", "ands", "products", "products_steps", "padded", "s_x", "s_x_unsplit", "a_out", "a_in")
REFERENCES = ("reference-onsite-h200", "reference-remote-h200", "reference-onsite-h200-today")


@pytest.mark.parametrize("name", REG.names("circuit"))
def test_a_circuit_is_one_step_of_its_run_and_its_structure_is_its_s_x_with_the_runs_that_measured_it(name):
    """Each measured value names the Definition it was counted on and the measurement (the circuits census, or
    circuit_properties' measure); the structure is its `s_x` on the Definition's basis, and cites the runs that
    measured it (evidence, which no status reads) beside the straddling premise."""
    c = circuits.circuit(REG, name)
    run, st = instances.run(REG, c.run), structure(REG, c.structure)
    design = c.definition.split("{")[0]
    assert run.tokens == run.step_tokens == c.batch * c.seq_len
    assert st.s_times == c.s_x and st.basis == f"boolean:{design}"
    assert REG.param(f"structure.{c.structure}.s_times").source == f"circuit.{name}.s_x"
    assert st.evidence and all(lean.status(e) == "evidence" for e in st.evidence)
    assert st.cites() == st.STRADDLE + st.evidence and lean.weakest(st.cites()) == lean.weakest(st.STRADDLE)
    for k in MEASURED:
        p = REG.param(f"circuit.{name}.{k}")
        assert p.status == "measured" and design in p.source
        assert "circuits census" in p.source or "circuit_properties' measure" in p.source


def test_the_census_products_are_the_step_s_gemms_and_its_whole_dots_are_as_many_as_kappa_counts():
    """Every active weight meets each token three times (forward, input and weight gradients; the LM head's too, tied
    to the embedding), attention's six batched GEMMs meet each sequence's S² positions at `d_model`, and the split-K
    routes pad the rest. The products outside the dot steps (the layers' forward and input gradients, the LM head's
    forward) are as many as `κ` counts (`families.layered`, which takes the head's input gradient for its forward).
    `a_out` is the LM head's input, which `fanout` leaves out; `a_in` is attention's 64-wide head."""
    tokens = C.batch * C.seq_len
    attention = 6 * C.batch * C.seq_len ** 2 * MODEL.d_model * MODEL.layers
    assert C.products == 3 * MODEL.active() * tokens + attention + C.padded
    assert C.products - C.products_steps == layered(MODEL, 2)[0] * tokens
    assert C.definition == f"TrainStep_v1{{L={MODEL.layers},B={C.batch},S={C.seq_len}}}"
    assert C.a_out == MODEL.vocab > MODEL.fanout and C.a_in < MODEL.legs
    assert C.s_x < C.s_x_unsplit == REG.param(f"circuit.{STEP}.s_x_unsplit").low


def test_the_flop_model_beside_the_step():
    """At its `6 · n_active · tokens` FLOPs, the step has 3.77 times the FLOP model's G = 250 gates a FLOP (944, near
    the high end of G's range, a bf16 product into an f32 accumulator) and 1.25 times as many ANDs. The reference's
    gates per counted product, `2G / κ` = 831 (`structure.reference`'s `s_×`), is the step's 3165 at its own `κ`, and
    the fewest gates in a counted product, 113 (a dot step's lane, 47.5% of the products), is 0.136 of it: a gate holds
    `u/w` = 27.7 of the reference's weight at the step's G (4.6 at a whole dot's 681)."""
    m = circuits.compare(REG, STEP)
    f = 6 * MODEL.n_active * C.batch * C.seq_len
    assert m.flops == f == REG[f"run.{C.run}.compute"]
    assert (m.gates_per_flop, m.ands_per_flop) == (C.gates / f, C.ands / f)
    assert m.model_gates_per_flop == REG["units.gates_per_flop"] == 250
    assert m.gates_ratio == pytest.approx(3.7746, rel=1e-4) and m.ands_ratio == pytest.approx(1.2497, rel=1e-4)
    g = REG.param("units.gates_per_flop")
    assert g.low < m.gates_per_flop < g.high
    assert m.reference_s_times == 2 * 250 / m.kappa == pytest.approx(830.56, rel=1e-4)
    assert m.kappa_step == pytest.approx(0.59634, rel=1e-4) and m.step_s_times == pytest.approx(3164.8, rel=1e-4)
    assert m.step_s_times == pytest.approx(C.gates / (C.products - C.products_steps), rel=1e-12)
    assert m.gates_per_product == pytest.approx(1661.66, rel=1e-5)
    assert m.ands_per_product == pytest.approx(550.17, rel=1e-5)
    assert m.steps_share == pytest.approx(0.47495, rel=1e-4)
    assert m.s_x / m.reference_s_times == pytest.approx(0.13605, rel=1e-4)
    assert m.u_over_w(C.s_x) == pytest.approx(27.743, rel=1e-4)
    assert m.u_over_w(C.s_x_unsplit) == pytest.approx(4.6035, rel=1e-4)


def test_in_place_the_design_gives_both_s_times_and_gates_per_flop():
    """`in_place` gives every scenario that names no structure the step's (`structure.default`, `reference` as the
    data stands) and its own gates a FLOP, unranged; a scenario that names one keeps it. The step's `s_×` beside the
    FLOP model's G is refused."""
    assert REG["structure.default"] == REFERENCE
    at = circuits.in_place(REG, STEP)
    s = build(at, "reference-onsite-h200")
    assert s.structure == structure(REG, C.structure) and s.context.units == circuits.units(REG, C)
    g = at.param("units.gates_per_flop")
    assert not g.ranged and g.status == "derived" and g.source.startswith(f"circuit.{STEP}")
    assert build(at, "reference-onsite-h200-elementary").structure.name == "elementary"
    with pytest.raises(ValueError, match="one source"):
        build(REG.override({"structure.default": C.structure}), "reference-onsite-h200")


@pytest.mark.parametrize("name", REFERENCES)
def test_the_measured_s_times_moves_the_bound_by_u_over_w_on_the_share_counted_at_u(name):
    """With the design in place `G` no longer cancels: a gate counted at `u` (the dark leaves, the uncertified routes)
    holds `u/w = 2G / (κ s_×)` of the reference's weight, so the lower bound is the default's times `1 / (a u/w + 1 −
    a)` on the share `a` counted at `u`, the rest the same in gates. The bound cites the runs that measured `s_×`
    beside the straddling premise, its status as it was."""
    base, one = evaluate_named(REG, name), evaluate_named(circuits.in_place(REG, STEP), name)
    f = base.families[0]
    uw = 2 * circuits.units(REG, C).gates_per_flop / (f.profile.kappa * C.s_x)
    at_u = sum(t.share for t in analysis.attribution(base) if t.name == "dark" or t.name.split(".")[0] in ROUTES)
    assert 0 < at_u <= 1
    assert one.lo / base.lo == pytest.approx(1 / (at_u * uw + 1 - at_u), rel=1e-9)
    c = one.families[0].combined
    assert set(structure(REG, C.structure).cites()) <= set(c.cites) and c.status == f.combined.status


def test_periods_sets_the_step_beside_the_flop_model_and_evaluates_with_its_design(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "path", list(sys.path))
    import periods

    name = "reference-remote-h200"
    monkeypatch.setattr(sys, "argv", ["periods.py", "--scenario", name, "--measured", STEP, "--out", str(tmp_path)])
    periods.main()
    text = capsys.readouterr().out
    assert text.index("\n## The measured design\n") < text.index("\n## The lower bound\n")
    doc = json.loads((tmp_path / "periods.json").read_text(), parse_constant=lambda c: pytest.fail(f"JSON has {c}"))
    assert doc["measured"]["s_x"] == C.s_x and doc["measured"]["u_over_w"] == circuits.compare(REG, STEP).u_over_w(C.s_x)
    at = circuits.in_place(REG, STEP)
    for x in doc["rows"]:
        assert x["overhead"] == analysis.result(at.override(periods.choices(name, x["fleet"], x["period"])), name).lo
        assert any(r["what"] == "draft:no-straddle" and r["at"][1] < r["at"][0] for r in x["ranking"])
