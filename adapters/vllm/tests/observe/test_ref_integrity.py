"""Reference integrity: the fail-closed rule R1 (`gates.fold_integrity` / `gate_R1` / `verdict`) and the
fold's recompute token-wiring fix (a preempted request's re-fed generated token wires to the TokenSelect output that produced it,
or the fold fails loudly)."""

from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.observe.fold import fold as fold_mod
from verity_vllm.check import gates as G
from verity_vllm.observe.events import StepLink
from verity_vllm.observe.fold.fold import FoldError, fold, instance_nodes, summarize
from tests.observe import test_fold_synthetic as toy
from tests.observe.test_fold_synthetic import Ids, PROMPT, REQ, SAMPLED, preamble, step_events, toy_profile
from tests.check.test_gates_fixtures import by_id, passing_artifacts, write_run

memory = toy.memory  # the fixture (real memory model when importable, else the in-test stub)

RULE = f"{G.REF_INTEGRITY_RULE['id']}/{G.REF_INTEGRITY_RULE['version']}"


# ---------------------------------------------------------------------------------------------------------
# R1 — the pure rule
# ---------------------------------------------------------------------------------------------------------


def _fs(**over):
    fs = {"schema": G.FOLD_SUMMARY_SCHEMA, "errors": 0, "unresolved": 0, "unsupported": 0, "error_classes": {}, "digest": "ab" * 32,
          "with_program": True, "n_instances": 10, "per_step": [{"step": 0, "errors": 0}, {"step": 1, "errors": 0}]}
    fs.update(over)
    return fs


def test_rule_is_versioned_and_quoted():
    assert G.REF_INTEGRITY_RULE["id"] == "R1" and G.REF_INTEGRITY_RULE["version"] == "2026-09-18"
    assert "errors > 0" in G.REF_INTEGRITY_RULE["statement"] and "unverified" in G.REF_INTEGRITY_RULE["statement"]
    assert fold_mod.REFERENCE_INTEGRITY_RULE == RULE
    assert G.GATE_NAMES["R1"] == "reference integrity (fail-closed)"
    d = G.gates_json([G.GateResult("R1", G.GATE_NAMES["R1"], "pass")])
    assert d["constants"]["REF_INTEGRITY_RULE"] == RULE


def test_pass_requires_every_count_present_and_zero():
    st, problems, detail = G.fold_integrity(_fs(), _fs(), {"totals": {"errors": 0, "unsupported": 0}})
    assert st == "pass" and not problems and detail["unverified"] == [] and detail["rule"]["id"] == "R1"
    assert detail["fold_summary"]["errors"] == 0 and detail["fold_summary_2"]["errors"] == 0 and detail["resolution"] == {"errors": 0, "unsupported": 0}


def test_errors_one_fails():
    # the qwen8 shape: one TokenWiring error, everything else clean, Program built
    fs = _fs(errors=1, error_classes={"TokenWiring": 1}, per_step=[{"step": 0, "errors": 0}, {"step": 1, "errors": 1}])
    st, problems, _ = G.fold_integrity(fs)
    assert st == "fail" and problems == ["fold_summary.errors = 1"]


@pytest.mark.parametrize("key", ["errors", "unresolved", "unsupported"])
def test_each_count_fails_on_its_own(key):
    fs = _fs(**{key: 2}, error_classes={"X": 2} if key == "errors" else {}, per_step=[])
    st, problems, _ = G.fold_integrity(fs)
    assert st == "fail" and problems == [f"fold_summary.{key} = 2"]


def test_second_fold_and_resolution_errors_fail():
    st, problems, _ = G.fold_integrity(_fs(), _fs(errors=3, error_classes={"Unresolved": 3}, per_step=[]))
    assert st == "fail" and problems == ["fold_summary_2.errors = 3"]
    st, problems, _ = G.fold_integrity(_fs(), None, {"totals": {"errors": 1, "unsupported": 0}})
    assert st == "fail" and problems == ["resolution.totals.errors = 1"]
    st, problems, _ = G.fold_integrity(_fs(), None, {"totals": {"errors": 0, "unsupported": 4}})
    assert st == "fail" and problems == ["resolution.totals.unsupported = 4"]


def test_missing_evidence_is_unverified_never_zero():
    st, _, detail = G.fold_integrity(None)
    assert st == "unverified" and detail["unverified"] == ["fold_summary.json missing: no fold error evidence"]
    fs = _fs()
    del fs["errors"]
    st, _, detail = G.fold_integrity(fs)
    assert st == "unverified" and detail["unverified"] == ["fold_summary.errors missing"]
    st, _, detail = G.fold_integrity(_fs(errors=None))
    assert st == "unverified" and "not a non-negative integer" in detail["unverified"][0]
    st, _, detail = G.fold_integrity(_fs(errors="0"))
    assert st == "unverified"
    st, _, detail = G.fold_integrity(_fs(errors=True))
    assert st == "unverified"
    st, _, detail = G.fold_integrity("not a dict")
    assert st == "unverified"
    st, _, detail = G.fold_integrity(_fs(), None, {"no": "totals"})
    assert st == "unverified" and detail["unverified"] == ["resolution.json has no `totals` object"]


def test_inconsistent_evidence_is_unverified():
    # histogram says one error, the count says zero: the evidence disagrees with itself -> unverified (not pass)
    st, _, detail = G.fold_integrity(_fs(error_classes={"TokenWiring": 1}))
    assert st == "unverified" and detail["unverified"] == ["fold_summary.error_classes sums to 1, errors = 0"]
    st, _, detail = G.fold_integrity(_fs(per_step=[{"step": 0, "errors": 1}]))
    assert st == "unverified" and detail["unverified"] == ["fold_summary.per_step errors sum to 1, errors = 0"]
    # per_step without integer error counts (older summaries) is simply not used for the consistency check
    st, _, _ = G.fold_integrity(_fs(per_step=[{"step": 0}]))
    assert st == "pass"


def test_fail_dominates_unverified():
    fs = _fs(errors=1, error_classes={"TokenWiring": 2}, per_step=[])
    st, problems, detail = G.fold_integrity(fs)
    assert st == "fail" and problems == ["fold_summary.errors = 1"] and detail["unverified"]


# ---------------------------------------------------------------------------------------------------------
# R1 — the gate over a run directory and the verdict assembly
# ---------------------------------------------------------------------------------------------------------


def test_gate_R1_pass_on_a_clean_run_and_verdict_accepted(tmp_path):
    write_run(tmp_path, passing_artifacts())
    res = G.evaluate_gates(tmp_path)
    r = by_id(res)["R1"]
    assert r.status == "pass" and r.id == "R1" and r.name == G.GATE_NAMES["R1"] and r.reason is None
    assert r.evidence == ["fold_summary.json", "fold_summary_2.json", "resolution.json"]
    assert G.verdict(res) == "ACCEPTED"


def test_gate_R1_synthetic_fold_record_with_errors_1_rejects(tmp_path):
    """A record whose fold summary carries `errors: 1` (qwen8's shape) must not be ACCEPTED,
    even when every frozen gate that does not look at the count would pass."""
    arts = passing_artifacts()
    fs = arts["fold_summary.json"]
    fs["errors"], fs["error_classes"] = 1, {"TokenWiring": 1}
    fs["error_examples"] = [{"class": "TokenWiring", "step": 37, "row": 32,
                             "message": "step 37 row 32: position 109 of '4524e470…' is past the prompt (109) and has no StepLink"}]
    write_run(tmp_path, arts)
    res = G.evaluate_gates(tmp_path)
    r = by_id(res)["R1"]
    assert r.status == "fail" and "fold_summary.errors = 1" in r.reason and RULE in r.reason
    assert r.detail["fold_summary"]["error_classes"] == {"TokenWiring": 1}
    assert G.verdict(res) == "REJECTED"
    # and R1 decides it even if G2 (which also reads the count) were absent from the result list
    assert G.verdict([x for x in res if x.id != "G2"]) == "REJECTED"


def test_gate_R1_missing_summary_is_unverified_and_verdict_incomplete(tmp_path):
    arts = passing_artifacts()
    del arts["fold_summary.json"]
    write_run(tmp_path, arts)
    res = G.evaluate_gates(tmp_path)
    r = by_id(res)["R1"]
    assert r.status == "unverified" and "no fold error evidence" in r.reason and r.evidence[0] == "fold_summary.json"
    assert G.verdict(res) == "INCOMPLETE"
    # an otherwise fully passing frozen set + R1 unverified is INCOMPLETE, never ACCEPTED
    ok = [G.GateResult(g, G.GATE_NAMES[g], "pass") for g in G.GATES]
    assert G.verdict(ok + [G.GateResult("R1", G.GATE_NAMES["R1"], "unverified")]) == "INCOMPLETE"
    assert G.verdict(ok + [G.GateResult("R1", G.GATE_NAMES["R1"], "fail")]) == "REJECTED"
    assert G.verdict(ok + [G.GateResult("R1", G.GATE_NAMES["R1"], "pass")]) == "ACCEPTED"
    assert G.verdict(ok) == "ACCEPTED"                    # a historical record that predates R1 keeps its verdict


def test_gate_R1_summary_without_error_count_is_unverified(tmp_path):
    arts = passing_artifacts()
    del arts["fold_summary.json"]["errors"]
    write_run(tmp_path, arts)
    res = G.evaluate_gates(tmp_path)
    assert by_id(res)["R1"].status == "unverified" and "fold_summary.errors missing" in by_id(res)["R1"].reason
    assert G.verdict(res) == "INCOMPLETE"


def test_gate_R1_unreadable_summary_is_unverified(tmp_path):
    write_run(tmp_path, passing_artifacts())
    (tmp_path / "fold_summary.json").write_text("{not json")
    r = G.gate_R1(G.RunArtifacts(tmp_path))
    assert r.status == "unverified" and "unreadable" in r.detail and "fold_summary.json" in r.reason


def test_R1_never_changes_the_frozen_set():
    assert G.GATES == ("G1", "G2", "G3", "G4", "G5", "G6", "G7") and "R1" not in G.GATES and "R1" not in G.HOLDOUT_GATES
    assert G.INTEGRITY_GATES == ("R1",) and G.ALL_GATES == G.GATES + ("G8", "R1")
    # R1 pass never rescues a failing / incomplete frozen set
    ok = [G.GateResult(g, G.GATE_NAMES[g], "pass") for g in G.GATES]
    inc = copy.deepcopy(ok)
    inc[6].status = "skipped"
    assert G.verdict(inc + [G.GateResult("R1", G.GATE_NAMES["R1"], "pass")]) == "INCOMPLETE"
    bad = copy.deepcopy(ok)
    bad[1].status = "fail"
    assert G.verdict(bad + [G.GateResult("R1", G.GATE_NAMES["R1"], "pass")]) == "REJECTED"


# ---------------------------------------------------------------------------------------------------------
# the fold: re-fed generated tokens (preemption -> recompute) wire to their producing TokenSelect output
# ---------------------------------------------------------------------------------------------------------


def _strip_links(evs):
    return [e for e in evs if not isinstance(e, StepLink)]


def recompute_log():
    """step 0: prefill positions [0, 1] -> sample 9 (= the token at position 2).  The request is then preempted (nothing observed)
    and at step 1 the engine recomputes the whole sequence in one chunk, positions [0, 1, 2]: rows 0-1 are prompt rows, row 2 is
    the RE-FED generated token 9, and — as the adapter only links the request's most recent pending sample when the row is at
    its position — carries no StepLink.  seq_len 3 >= prompt 2 + 1 served -> the sample (3, position 3) is served."""
    ids = Ids()
    evs = preamble()
    evs += step_events(0, [0, 1], ids, 20, SAMPLED[0])
    evs += _strip_links(step_events(1, [0, 1, 2], ids, 40, SAMPLED[1]))
    return evs


def test_refed_generated_token_wires_to_its_sample(memory):
    r = fold(recompute_log(), toy_profile(), strict=True)
    assert r.errors == [] and r.report.unsupported == 0 and r.reference_eligible
    nodes = r.body.nodes
    ts0 = instance_nodes(r, spec_prefix="TokenSelect_v1", step=0)[0]["node"]
    embs1 = [m["node"] for m in instance_nodes(r, spec_prefix="Embedding_v1", step=1)]
    assert [list(nodes[k].args[0].refs) for k in embs1] == [[("p", 0, 0)], [("p", 0, 1)], [("n", ts0, 0)]]   # row 2 <- step 0's sample
    assert r.refed == [{"step": 1, "row": 2, "request": REQ, "position": 2, "src_step": 0, "token": SAMPLED[0]}]
    assert r.sampled == {REQ: [(0, 9), (1, 3)]} and r.discarded == {}
    s = summarize(r)
    assert s["refed"] == 1 and s["refed_examples"] == r.refed
    assert s["reference_integrity"] == {"rule": RULE, "eligible": True, "errors": 0, "unsupported": 0}
    # the fold is deterministic with the recompute path exercised
    assert fold(recompute_log(), toy_profile(), strict=True).digest == r.digest


def test_refed_token_without_a_producing_sample_fails_loudly(memory):
    """Position 3 was never sampled (the only served sample so far produced position 2): strict raises, development mode records
    a `TokenWiring` error naming the stale slot content, and the fold is not eligible as a reference."""
    ids = Ids()
    evs = preamble() + step_events(0, [0, 1], ids, 20, SAMPLED[0])
    evs += _strip_links(step_events(1, [0, 1, 2, 3], ids, 40, SAMPLED[1]))
    with pytest.raises(FoldError, match="no served sample of 'r0' produced position 3"):
        fold(evs, toy_profile(), strict=True, with_program=False)
    r = fold(evs, toy_profile(), strict=False, with_program=False)
    tw = [e for e in r.errors if e["class"] == "TokenWiring"]
    assert len(tw) == 1 and tw[0]["step"] == 1 and tw[0]["row"] == 3 and tw[0]["position"] == 3 and "stale_refs" in tw[0]
    assert not r.reference_eligible and summarize(r)["reference_integrity"]["eligible"] is False
    assert r.refed and r.refed[0]["position"] == 2        # row 2 was still wired correctly


def test_refed_token_value_mismatch_fails_loudly(memory):
    ids = Ids()
    evs = preamble() + step_events(0, [0, 1], ids, 20, SAMPLED[0])
    step1 = _strip_links(step_events(1, [0, 1, 2], ids, 40, SAMPLED[1]))
    start = step1[0]
    bad = list(start.input_token_ids)
    bad[2] = SAMPLED[0] + 1                                 # the log says the re-fed token is not the one sampled
    step1[0] = type(start)(**{**start.__dict__, "input_token_ids": bad})
    with pytest.raises(FoldError, match="re-fed token 10 at position 2 of 'r0' != token 9 sampled at step 0"):
        fold(evs + step1, toy_profile(), strict=True, with_program=False)


def test_partial_recompute_chunk_sample_is_discarded(memory):
    """vLLM discards the sample of a chunk that ends inside the tokens already generated (`seq_len < prompt + generated`):
    step 1 recomputes positions [0, 1] only (seq_len 2 < 2 + 1) -> its sample is discarded, not served, and never becomes a
    producer; step 2 re-feeds position 2 (no StepLink) and wires to step 0's sample; its own sample (seq_len 3) is served."""
    ids = Ids()
    evs = preamble() + step_events(0, [0, 1], ids, 20, SAMPLED[0])
    evs += _strip_links(step_events(1, [0, 1], ids, 40, 11))            # a chunk ending at seq_len 2: discarded
    evs += _strip_links(step_events(2, [2], ids, 60, SAMPLED[1]))       # the re-fed generated token, no link
    r = fold(evs, toy_profile(), strict=True)
    assert r.errors == []
    assert r.sampled == {REQ: [(0, 9), (2, 3)]} and r.discarded == {REQ: [(1, 11, 1)]}
    assert r.refed == [{"step": 2, "row": 0, "request": REQ, "position": 2, "src_step": 0, "token": 9}]
    ts0 = instance_nodes(r, spec_prefix="TokenSelect_v1", step=0)[0]["node"]
    emb2 = instance_nodes(r, spec_prefix="Embedding_v1", step=2)[0]["node"]
    assert list(r.body.nodes[emb2].args[0].refs) == [("n", ts0, 0)]
    assert any("partial recompute" in n for n in r.notes)
    # the Program returns only the served tokens
    assert r.body.ret_type.leaves == 2


def test_plain_decode_unchanged(memory):
    """The reference toy log (prefill + linked decode) folds to the same digest as before the fix: no preemption, no change."""
    r = fold(toy.toy_log(), toy_profile(), strict=True)
    assert r.errors == [] and r.refed == [] and r.sampled == {REQ: [(0, 9), (1, 3)]}
    ts0 = instance_nodes(r, spec_prefix="TokenSelect_v1", step=0)[0]["node"]
    emb1 = instance_nodes(r, spec_prefix="Embedding_v1", step=1)[0]["node"]
    assert list(r.body.nodes[emb1].args[0].refs) == [("n", ts0, 0)]
    assert summarize(r)["reference_integrity"]["eligible"] is True
