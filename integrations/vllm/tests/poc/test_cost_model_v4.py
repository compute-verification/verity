"""Cost model v4 (PROPOSED) — vocabulary, metadata and the §38.1 selection/reporting regression (lane COST-D, D4)."""
from __future__ import annotations

import json
from fractions import Fraction

import pytest

from frontier import cost_model_v4 as cm
from frontier import result_schema as rs

DOC = cm.load()
PLANS = list(DOC["plans"])
REQUIRED_META = {"value", "unit", "hardware", "workload", "shape", "denominator_kind", "evidence", "anchor", "substitution", "uncertainty", "used_by", "note"}


def test_every_term_and_used_by_is_a_stage_of_agreement_3():
    for cid, c in DOC["coefficients"].items():
        assert c["used_by"], cid
        for st in c["used_by"]:
            assert st in rs.STAGE_NAMES, (cid, st)
    m = cm.Model(DOC)
    for plan in PLANS:
        for t in m.terms("B0_47pos", plan, "1/16", "1/4096"):
            assert t.stage in rs.STAGE_NAMES, t.term_id


def test_every_coefficient_carries_the_d3_metadata():
    for cid, c in DOC["coefficients"].items():
        assert REQUIRED_META <= set(c), (cid, REQUIRED_META - set(c))
        if c["evidence"] == "MISSING":
            assert c["value"] is None, cid
        else:
            assert c["evidence"] in rs.EVIDENCE, (cid, c["evidence"])
            assert c["anchor"] and c["anchor"]["source"], cid
    for aid, a in DOC["anchors"].items():
        assert a["evidence"] == "MEASURED" and a["anchor"]["source"], aid


@pytest.mark.parametrize("den", ["stock", "profile"])
@pytest.mark.parametrize("plan", PLANS)
def test_point_validates_for_every_plan_and_denominator(plan, den):
    m = cm.Model(DOC, den)
    p = m.point("B0_47pos", plan, "1/16", "1/4096", budget=1e4, M=10**6)
    rep = rs.validate(p)
    assert not rep.errors, rep.errors
    # round trip through JSON (what SEARCH-B will emit) still validates
    p2 = rs.FrontierPoint.from_json(json.loads(json.dumps(p.to_json())))
    assert not rs.validate(p2).errors
    # every objective term states the same denominator (R6: no silent mixing)
    dens = {t.denominator for t in p.cost_by_stage if t.in_objective and t.unit == "ratio"}
    assert len(dens) == 1


def test_38_1_selection_and_reporting_are_the_same_number():
    """The plan chosen under the cost model is reported at exactly the cost it was selected with."""
    m = cm.Model(DOC)
    grid = [(plan, Fraction(1, 2**k), Fraction(1, 2**j)) for plan in PLANS for k in range(0, 24, 3) for j in range(0, 40, 4)]
    budget = 1e4  # stock units; a budget every plan can meet at small q·s
    best, omega_sel, fits = cm.select(m, "B0_47pos", grid, budget, M=10**6)
    assert best is not None and fits > 0
    plan, q, s = best
    p = m.point("B0_47pos", plan, q, s, budget=budget, M=10**6, search=rs.SearchInfo("test grid", len(grid), fits, 0))
    assert p.totals.omega_selection == omega_sel  # bitwise: same function, same terms
    assert p.totals.omega_reported == omega_sel
    recomputed = sum(t.value for t in p.cost_by_stage if t.in_objective and t.priced)
    assert recomputed == omega_sel
    assert not rs.validate(p).errors
    # and for EVERY candidate, not just the winner
    for plan, q, s in grid[::7]:
        pt = m.point("B0_47pos", plan, q, s, budget=budget, M=10**6)
        assert pt.totals.omega_reported == m.omega("B0_47pos", plan, q, s, M=10**6)


def test_repriced_emitter_is_caught_by_the_validator():
    """The earlier witness-pricing inconsistency: a reporter that re-prices a component after selection."""
    m = cm.Model(DOC)
    p = m.point("B0_47pos", "whole", "1/16", "1/4096", budget=1e4, M=10**6)
    wit = next(t for t in p.cost_by_stage if t.stage == "prove.witness")
    wit.value = wit.value * 1.5  # printed components no longer sum to the selection value
    rep = rs.validate(p)
    assert any(e.startswith("R8") for e in rep.errors), rep.errors
    # a reporter that fixes the sum but prints a different headline than it selected with
    p = m.point("B0_47pos", "whole", "1/16", "1/4096", budget=1e4, M=10**6)
    p.totals.omega_reported = p.totals.omega_selection * 1.01
    rep = rs.validate(p)
    assert any(e.startswith("R9") for e in rep.errors), rep.errors


def test_mixed_hardware_proof_rates_are_flagged_as_substitutions():
    m = cm.Model(DOC)
    for t in m.terms("B0_47pos", "whole", "1/16", "1/4096"):
        if t.stage == "prove.argument" and "H100" in t.hardware:
            assert t.substitution is not None and t.evidence == "MODELED", t.term_id
            assert "A100" in t.substitution.stands_in_for


def test_first_boundary_terms_follow_the_plan():
    m = cm.Model(DOC)

    def by(plan):
        return {t.term_id: t for t in m.terms("B0_47pos", plan, "1/16", "1/4096")}

    w, o, l = by("whole"), by("operator"), by("layer")
    assert w["boundary.collect"].value == 0.0 and w["acquire.replay"].value > 0
    assert o["acquire.replay"].value == 0.0 and o["acquire.commit"].value == 0.0
    assert o["boundary.collect"].value > l["boundary.collect"].value > 0
    assert l["boundary.collect"].evidence == "MODELED" and l["boundary.collect"].substitution is not None
    # operator plan's recurring cost is q-independent
    a = m.terms("B0_47pos", "operator", "1/2", "1/4096")
    b = m.terms("B0_47pos", "operator", Fraction(1, 2**20), "1/4096")
    rec = lambda T: sum(t.value for t in T if t.in_objective and t.priced and rs.STAGE_BY_NAME[t.stage].point == "before_J")
    assert rec(a) == rec(b)


def test_prove_checker_is_an_explicit_unpriced_term_and_forces_c2_when_feasible():
    m = cm.Model(DOC)
    p = m.point("B0_47pos", "whole", Fraction(1, 2**20), Fraction(1, 2**36), budget=1e6, M=10**6)
    chk = next(t for t in p.cost_by_stage if t.stage == "prove.checker")
    assert chk.priced is False and chk.value is None
    assert p.classification.cls == "conditional-unpriced", p.classification


def test_reconciliation_table_lists_every_coefficient_and_anchor():
    md = cm.reconciliation_table_markdown(DOC)
    for cid in DOC["coefficients"]:
        assert f"`{cid}`" in md
    for aid in DOC["anchors"]:
        assert aid in md
    assert "MIXED HARDWARE" in md and "SUBSTITUTION" in md and "UNPRICED" in md


def test_agreement_1_reviewer_check_on_proto_a_b0_instance():
    """COST-D as reviewer of agreement 1: which charged stages the B0 historical instance does not name."""
    from pathlib import Path

    from frontier import instance_check as ic

    path = Path(__file__).resolve().parents[1] / "results" / "PROTO-A" / "b0-historical" / "protocol_instance.json"
    if not path.exists():
        pytest.skip("PROTO-A B0 instance not merged")
    doc = json.loads(path.read_text())
    m = cm.Model(DOC)
    p = m.point("B0_47pos", "block", "1/2", "1/4096", budget=None, M=4)
    rep = ic.check_point(p, doc)
    # the instance names the commitments, both challenges, the sampled proofs and the unconditional checks
    for s in ("boundary.commit", "challenge.replay_draw", "acquire.replay", "acquire.commit", "challenge.inner_draw", "prove.argument", "verify.output_checks"):
        assert rep["named"].get(s), s
    # the objections COST-D filed (agreements-review.md, 2026-09-09 17:10 UTC) named four charged-but-unnamed stages; agreement-1 draft 2
    # (PROTO-A, 17:44 UTC) names boundary.collect (partition.first_boundary), prove.witness / prove.checker (the sampled obligation's
    # proof_steps); fixed.setup stays unnamed BY DESIGN -- the shared-once rule (protocol-instance.md §3): per-process / per-checkpoint
    # setup is T_fixed's setup share reported by COST-D outside the instance, so COST-D moves it out of Omega_prover's mandatory set.
    assert set(rep["unnamed_charged"]) == {"fixed.setup"}, rep["unnamed_charged"]
    for s in ("boundary.collect", "prove.witness", "prove.checker"):  # verify.weights_unconditional: partition.weight_input_check.stage names it; the keyword mapper has no rule yet (PROTO-A request)
        assert rep["named"].get(s), s
    # layers 1–2 are never protocol stages and must not be reported as unnamed
    assert not rep["reported_only_unnamed"]
