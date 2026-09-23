"""The two Program-side CLIs of task (4) / SPEC A4.4-A6 on a REAL Build (sys's int-3 SmolLM2 B1 earlystop row, in the lane tree):
`lifted_use_document` (the lifted VU-partition use document with the width_summary revb's B1 condition names) and
`reqval_rows_lift_vs_unlifted` (RequiredValues(Lift(P)) == RequiredValues(P) row for row through reqval's traversal engine, the one that
unlifts).  Everything read is the Build's own artefact; nothing is a record fact."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROW = Path("out/gen/sweep/evidence/pad/smollm2-135m__bf16__l40s__tp1__b1__i32__o32__earlystop__greedy__bi-eager/r16_0730_sys_int3c_earlystop")
BR = ROW / "build_request_LP32_T11"
pytestmark = pytest.mark.skipif(not (BR / "descriptor.json.gz").exists(), reason="the R16 SmolLM2 earlystop evidence row is not in this tree")


def test_lifted_use_document_on_the_real_earlystop_build_reports_true_lifted_widths_at_w_max_lifted(tmp_path):
    from verity_vllm.registry import lifted as L
    from tests.ir import lifted_use_document as U
    out = tmp_path / "use"
    # the row's workload_program.json declares no lag (pre-R17-1 Build): a synthetic --lag is the only source, stated as such
    assert U.main(["--build-request", str(BR), "--out", str(out), "--lag", "1", "--eos", "0", "--cap", "12", "--max-sets", "1000"]) == 0
    rep = json.loads((out / "lifted_use_document.json").read_text())
    use = json.loads((out / "query_use_lifted.json").read_text())
    q = json.loads((out / "query_lifted.json").read_text())
    assert rep["label"] == "lifted-padded" and rep["rules_version"] == "lifted-v1" and rep["ok"] is True and rep["verdict_status"] == "ok"
    assert rep["lag"] == 1 and rep["lag_declared_field"] == "--lag" and rep["cap"] == 12 and rep["eos"] == [0]
    assert rep["use_kind"] == use["use"]["kind"] == "vu-partition-lifted" and use["use"]["w_max"] == 64 and use["use"]["rules_version"] == "lifted-v1"
    assert "kind" not in q and "w_max" not in q  # the query document is unchanged in shape (A5)
    ws = rep["width_summary"]
    assert all(k in ws for k in L.WIDTH_SUMMARY_FIELDS)
    # the validator's max_seen is the true lifted width over EVERY set (the sampler's 32-bit leaf -> 33); the bounded exact walk (1000 of
    # the 10^9 sets) visits weight-read sets first, whose escaping leaf is ORDINARY under G1 (v0.7 A11: a weight is never widened) -- 32
    # bits, no tag bit, growth 0 -- so the two differ HERE and the summary says so (`max_seen_equals_w_out_lifted_max` False, `complete` False)
    assert ws["w_max_lifted"] == 64 and ws["max_seen"] == 33 and ws["w_out_lifted_max"] == 32 and ws["w_out_payload_max"] == 32
    assert ws["max_seen_equals_w_out_lifted_max"] is False and ws["complete"] is False
    assert ws["n_out_max"] == 1 and ws["growth_max"] == 0 and ws["readiness_at_32"]["ok"] is False and ws["readiness_at_w_max_lifted"]["ok"] is True
    # the exact walk was bounded (1000 of 10^9 sets): the structural bound covers every set and is complete
    assert ws["complete"] is False and ws["sets_visited"] == 1000
    sb = ws["n_out_structural_bound"]
    assert sb["complete"] is True and sb["sets"] == rep["sets"] and sb["n_out_bound_max"] == 1 and sb["n_out_bound_histogram"] == {"1": rep["sets"]}
    # beside it, information: the active-reference validator on the UNLIFTED P at 32 -- ok there (no lifted bit), never mixed with the verdict
    ref = rep["active_reference_at_32_on_unlifted_P"]
    assert ref["w_max"] == 32 and ref["status"] == "ok" and ref["width"]["max_seen"] == 32
    assert rep["lifted_program_digest"] != rep["program_digest"]


def test_lifted_use_document_refuses_a_lag_that_disagrees_with_the_rows_declaration(tmp_path):
    from tests.ir import lifted_use_document as U
    wp = tmp_path / "workload_program.json"
    wp.write_text(json.dumps({"workload_fn": 'Workload_v1{requests=[],shared="weights",lag=1}', "stop_rule": {"eos_token_ids": [0]}, "requests": [{"cap": 12}]}))
    with pytest.raises(SystemExit, match="disagrees with the row's declared lag 1"):
        U.main(["--build-request", str(BR), "--out", str(tmp_path / "u"), "--lag", "0", "--workload-program", str(wp)])
    with pytest.raises(SystemExit, match="no declared LAG"):
        U.main(["--build-request", str(BR), "--out", str(tmp_path / "u"), "--eos", "0", "--cap", "12"])


@pytest.mark.parametrize("build, cap, n_ident, lifted_rows, guard_rows, is_commits", [
    # guard_rows = L.continuation_gate_count(cap, |E|=1, lag=1): the R17-1 guard + `enable/one` (v0.6 B7) + the R17-32 controller's 3·cap
    ("build_request", 32, 12943, None, 159, True),           # the cap-length Program (32 sampling events): the one the row's Commit manifests (12,943 identities)
    ("build_request_LP32_T11", 12, 5043, 13010, 59, False),  # the served-length specialization (12 events): a different Program of the same row
])
def test_reqval_rows_lift_vs_unlifted_on_the_real_earlystop_build_is_row_for_row_through_the_traversal(tmp_path, build, cap, n_ident, lifted_rows, guard_rows, is_commits):
    from tests.ir import lifted_instances_emit as E
    from tests.ir import reqval_rows_lift_vs_unlifted as RQ
    br = ROW / build
    if not (br / "descriptor.json.gz").exists():
        pytest.skip(f"{build} is not in this tree")
    lifted = tmp_path / "lift"
    assert E.main(["--build-request", str(br), "--out", str(lifted), "--eos", "0", "--cap", str(cap), "--lag", "1"]) == 0  # guard rows 2(cap-1-LAG)+|E|+1
    out = tmp_path / "rows.json"
    assert RQ.main(["--unlifted", str(br), "--lifted", str(lifted), "--out", str(out), "--commit-verdict", str(ROW / "commit" / "verdict.json")]) == 0
    rep = json.loads(out.read_text())
    assert rep["label"] == "lifted-padded" and rep["ok"] is True
    assert rep["identities"] == {"unlifted": n_ident, "lifted": n_ident} and rep["rows_equal_in_order_modulo_program_digest"] is True and rep["set_equal"] is True
    assert rep["by_family_equal"] is True and rep["only_unlifted"] == rep["only_lifted"] == 0
    assert rep["complete"] == {"unlifted": True, "lifted": True} and rep["unmodelled"] == {"unlifted": {}, "lifted": {}}
    assert rep["unlifted_engines_agree"]["rows_equal"] is True and rep["unlifted_engines_agree"]["manifest_digest_equal"] is True
    assert rep["manifest_digest"]["differs_only_by_the_program_digest_column"] is True and rep["program_digest"]["lifted"] != rep["program_digest"]["unlifted"]
    assert rep["lifted_program_digest_agrees"] is True and rep["source_program_digest_agrees"] is True
    assert rep["query_lifted"]["lifted"] is True and rep["lifted_emit"]["guard_rows"] == guard_rows
    if lifted_rows is not None:
        assert rep["query_lifted"]["lifted_rows"] == lifted_rows
    assert rep["boundary_audit_lifted"]["fully_wired"] is True and rep["boundary_audit_lifted"]["rows"] == rep["query_lifted"]["lifted_rows"] + guard_rows
    # the Commit's manifest is compared BY PROGRAM, never by a batch-size guess: it is this traversal's exactly when the Programs are the same
    assert rep["unlifted_traversal_is_the_commits_manifest"] is is_commits
    assert (rep["commit_program_digest"] == rep["program_digest"]["unlifted"]) is is_commits
    assert ("IS the Commit's manifest" in rep["commit_manifest_note"]) is is_commits
    if not is_commits:
        assert "differs by the Program" in rep["commit_manifest_note"]
