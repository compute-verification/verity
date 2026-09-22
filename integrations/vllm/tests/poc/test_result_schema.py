"""Agreement 3 (cost/result format): the validator enforces the doc's rules R1–R12 and the §38.1
selection/reporting consistency regression (COST-D, 2026-09-09).  Laptop-safe: no data files, < 1 s."""

from __future__ import annotations

import copy
import json

import pytest

from frontier import result_schema as rs


def _valid() -> dict:
    p = rs.example_point()
    rep = rs.validate(p)
    assert rep.ok, rep.errors
    return p.to_json()


def test_example_round_trips_and_validates():
    doc = _valid()
    doc2 = json.loads(json.dumps(doc))
    p = rs.FrontierPoint.from_json(doc2)
    assert rs.validate(p).ok
    assert p.to_json() == doc


def test_stage_vocabulary_is_closed_and_consistent():
    names = [s.name for s in rs.STAGES]
    assert len(names) == len(set(names))
    for s in rs.STAGES:
        assert s.layer in rs.LAYERS and s.party in rs.PARTIES and s.point in rs.POINTS
        # verifier / harness / non-verification-layer stages never enter Ω_prover
        if s.party != "prover" or s.layer != "verification":
            assert not s.in_omega_prover, s.name
        if s.party == "harness":
            assert not s.mandatory, s.name
    # the six stage groups the brief names all exist
    for prefix in ("acquire.", "boundary.commit", "challenge.", "acquire.replay", "prove.", "verify."):
        assert any(n.startswith(prefix) for n in names), prefix


def test_unknown_stage_rejected():
    doc = _valid()
    doc["cost_by_stage"][0]["stage"] = "recording"  # not in the closed list
    rep = rs.validate(doc)
    assert any("R1" in e and "closed list" in e for e in rep.errors)


def test_profile_tax_cannot_enter_objective():
    doc = _valid()
    t = next(t for t in doc["cost_by_stage"] if t["stage"] == "profile.execution")
    t["in_objective"] = True
    rep = rs.validate(doc)
    assert any("R5" in e for e in rep.errors)


def test_verifier_work_cannot_enter_objective():
    doc = _valid()
    t = next(t for t in doc["cost_by_stage"] if t["stage"] == "verify.output_checks")
    t["in_objective"] = True
    rep = rs.validate(doc)
    assert any("R5" in e and "never part of" in e for e in rep.errors)


def test_38_1_a_printed_cost_equals_sum_of_components():
    doc = _valid()
    doc["totals"]["omega_selection"] *= 1.01
    doc["totals"]["omega_reported"] = doc["totals"]["omega_selection"]
    rep = rs.validate(doc)
    assert any("R8 §38.1(a)" in e for e in rep.errors)


def test_38_1_b_selection_and_reporting_must_agree():
    """The witness-pricing inconsistency (v2_correction): selected under one cost, printed another."""
    doc = _valid()
    doc["totals"]["omega_reported"] = doc["totals"]["omega_selection"] - 1.0
    rep = rs.validate(doc)
    assert any("R9 §38.1(b)" in e for e in rep.errors)


def test_38_1_c_other_fitting_smaller_U_is_an_error():
    doc = _valid()
    doc["search"]["other_fitting_smaller_U"] = 3
    rep = rs.validate(doc)
    assert any("R10 §38.1(c)" in e for e in rep.errors)


def test_measured_needs_anchor_and_labels_are_three():
    doc = _valid()
    t = next(t for t in doc["cost_by_stage"] if t["stage"] == "acquire.replay")
    t["anchor"] = None
    rep = rs.validate(doc)
    assert any("R4" in e for e in rep.errors)
    doc = _valid()
    t = next(t for t in doc["cost_by_stage"] if t["stage"] == "acquire.replay")
    t["evidence"] = "extrapolated"  # legacy five-way label: must be mapped, not used
    rep = rs.validate(doc)
    assert any("R2" in e for e in rep.errors)
    assert rs.LEGACY_EVIDENCE["extrapolated"] == "MODELED"


def test_mixed_denominators_must_be_declared():
    doc = _valid()
    # remove the declaration → error; keep it → warning
    doc["assumptions"] = [a for a in doc["assumptions"] if "mixed-denominator" not in a]
    t = next(t for t in doc["cost_by_stage"] if t["term_id"] == "prove.argument/gemm")
    t["denominator"] = "T_stock_A100_pure"
    rep = rs.validate(doc)
    assert any("R7" in e for e in rep.errors)


def _priced_point(budget: float, U_precap: float | None, W: float, status: str = "converged", at_ceiling: bool = False) -> dict:
    doc = _valid()
    # price the checker so the point is fully priced
    for t in doc["cost_by_stage"]:
        if t["stage"] == "prove.checker":
            t.update(value=0.0, evidence="ANALYTICAL", priced=True, formula="assumed 0 for this test", notes="test")
    obj = [t for t in doc["cost_by_stage"] if t["in_objective"]]
    total = sum(t["value"] for t in obj)
    rec = sum(t["value"] for t in obj if rs.STAGE_BY_NAME[t["stage"]].point == "before_J")
    doc["totals"].update(budget=budget, omega_selection=total, omega_reported=total, omega_recurring=rec)
    cap = None if U_precap is None else min(U_precap, W)
    doc["certificate"].update(U_precap_bits=U_precap, U_capped_bits=cap, W_agg_bits=W, capped=(None if U_precap is None else U_precap >= W),
                              search_status=status, minimiser_at_ceiling=at_ceiling)
    p = rs.FrontierPoint.from_json(doc)
    doc["classification"] = rs.classify(p).__dict__
    return doc


def test_classification_rules():
    d = _priced_point(budget=1e9, U_precap=100.0, W=256.0)
    assert d["classification"]["cls"] == "nontrivial" and rs.validate(d).ok
    d = _priced_point(budget=1e9, U_precap=300.0, W=256.0)
    assert d["classification"]["cls"] == "feasible-trivial" and rs.validate(d).ok
    d = _priced_point(budget=1e-3, U_precap=100.0, W=256.0)  # recurring (before_J) 0.011 > 1e-3
    assert d["classification"]["cls"] == "infeasible" and d["classification"]["rule"].startswith("C1") and rs.validate(d).ok
    d = _priced_point(budget=1e9, U_precap=100.0, W=256.0, status="hit_ceiling")
    assert d["classification"]["cls"] == "conditional-unpriced" and rs.validate(d).ok
    d = _priced_point(budget=1e9, U_precap=None, W=256.0, status="not_run")
    assert d["classification"]["cls"] == "conditional-unpriced"


def test_nontrivial_claim_without_precap_is_rejected():
    d = _priced_point(budget=1e9, U_precap=100.0, W=256.0)
    d["certificate"]["U_precap_bits"] = None
    d["certificate"]["U_capped_bits"] = 100.0
    d["classification"] = {"cls": "nontrivial", "rule": "claimed", "inputs": {}}
    rep = rs.validate(d)
    assert any("R11" in e for e in rep.errors)


def test_stated_classification_must_match_recomputed():
    d = _priced_point(budget=1e9, U_precap=300.0, W=256.0)
    d["classification"]["cls"] = "nontrivial"
    rep = rs.validate(d)
    assert any("R11 classification stated" in e for e in rep.errors)


def test_instance_stage_check_R12():
    p = rs.example_point()
    charged = {t.stage for t in p.cost_by_stage if rs.STAGE_BY_NAME[t.stage].mandatory}
    assert rs.validate(p, instance_stages=charged).ok
    missing = rs.stages_not_in_instance(p, charged - {"acquire.commit"})
    assert missing == ["acquire.commit"]
    rep = rs.validate(p, instance_stages=charged - {"acquire.commit"})
    assert any("R12" in e and "acquire.commit" in e for e in rep.errors)


def test_json_schema_structure_matches_example():
    schema = rs.json_schema()
    doc = _valid()
    assert set(schema["required"]) <= set(doc)
    assert schema["properties"]["cost_by_stage"]["items"]["properties"]["stage"]["enum"] == list(rs.STAGE_NAMES)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(doc, schema)


def test_unpriced_term_forces_conditional():
    doc = _valid()
    assert any(not t["priced"] for t in doc["cost_by_stage"])
    assert doc["classification"]["cls"] == "conditional-unpriced"
    # an unpriced term carrying a value is malformed
    t = next(t for t in doc["cost_by_stage"] if not t["priced"])
    t["value"] = 1.0
    assert any("R2" in e for e in rs.validate(doc).errors)


def test_stage_table_markdown_lists_every_stage():
    md = rs.stage_table_markdown()
    for s in rs.STAGES:
        assert f"`{s.name}`" in md
