"""What a row's records say when a block of a document they read is left out (`verity_vllm.pipeline.row_records`).

The row's summaries and stage lines are read off documents its stages wrote: a workload manifest, a derive's result and artifact, the
compare's and the global Match's documents, the sampler value check, the manifest, a weights record, the Commit's verdict and
summary.  A block of such a document may be absent, null or empty (a stage that did not run, a writer with nothing to state), and
the record then states `None`, nothing or a named default for it.  Each test gives a block absent, null, empty and stated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.row_records as R
from tests.pipeline.test_row import STAGE_START, _summary_row
from verity_vllm.config import CHECKPOINTS, WORKLOADS
from verity_vllm.pipeline import row_json

SAMPLED = WORKLOADS / "llama32-1b__bf16__l40s__tp1__b8__i1024__o128__mixed__stoch-t0.8-p0.95__bi-eager.json"


def _left_out(key: str, empty: Any = None) -> list[dict[str, Any]]:
    """A document's block `key` left out three ways: absent, null and empty (`{}`, or `empty` for a block that is a list)."""
    return [{}, {key: None}, {key: {} if empty is None else empty}]


def _put(path: Path, doc: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return str(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def test_a_block_that_lists_nothing_is_walked_as_nothing():
    """What a block lists is the block's own list or mapping; one it leaves out, or states as null or empty, is nothing to walk."""
    nothing: list[dict[str, Any]] = [{}, {"xs": None}, {"xs": []}, {"xs": {}}, {"xs": 0}, {"xs": ""}]
    for block in nothing:
        assert len(row_json.listed_of(block, "xs")) == 0 and list(row_json.listed_of(block, "xs")) == [], block
    stated: dict[str, Any] = {"xs": [1, 2], "by": {"a": 1}}
    assert row_json.listed_of(stated, "xs") is stated["xs"] and row_json.listed_of(stated, "by") is stated["by"]
    with pytest.raises(AttributeError):
        row_json.listed_of(None, "xs")


def test_a_rows_shape_reads_a_sweep_without_its_sampling_parameters_as_greedy(tmp_path):
    """Without sampling parameters, or with a temperature that is null or zero, the row keeps its envelope and snapshots two steps.  A
    row that samples counts its first request's prompt ids where the request states no prompt length."""
    w = json.loads(SAMPLED.read_text())
    sweep, first = w["sweep"], w["requests"][0]
    envelope = (1024, 127, "0,1")   # the sweep's longest prompt, one step short of its longest answer, and the first two steps
    greedy = [*_left_out("sampling_params"), {"sampling_params": {"temperature": None}}, {"sampling_params": {"temperature": 0}}, {"sampling_params": {"top_p": 0.9}}]
    for params in greedy:
        doc = {**w, "sweep": {**{k: v for k, v in sweep.items() if k != "sampling_params"}, **params}}
        shape = R.row_shape(_put(tmp_path / "w.json", doc), str(CHECKPOINTS), str(tmp_path))
        assert (shape["LP"], shape["T"], shape["snap"]) == envelope, params
    for ids, n in (({}, 0), ({"prompt_token_ids": None}, 0), ({"prompt_token_ids": []}, 0), ({"prompt_token_ids": [5, 6, 7]}, 3)):
        r0 = {**{k: v for k, v in first.items() if k not in ("prompt_len", "prompt_token_ids")}, **ids}
        shape = R.row_shape(_put(tmp_path / "w.json", {**w, "requests": [r0]}), str(CHECKPOINTS), str(tmp_path))
        assert (shape["LP"], shape["T"], shape["snap"]) == (n, first["max_tokens"] - 1, "all"), ids


def test_a_result_without_a_census_or_a_program_has_no_census_digest_and_no_ir():
    for result in _left_out("census"):
        assert R.census_digest(result) is None
    assert R.census_digest({"census": {"census": "c" * 64}}) == "c" * 64 and R.census_digest({"census": {"other": 1}}) is None
    for result in (*_left_out("program"), {"program": {"ir": None}}, {"program": {"ir": {}}}, {"program": {"digest": "d"}}):
        assert R._ir(result) == {}
    assert R._ir({"program": {"ir": {"root": "w"}}}) == {"ir": {"root": "w"}}


def test_a_build_summary_reads_each_derives_result_and_artifact_with_blocks_left_out(tmp_path):
    """A result that states no Program takes the artifact's digests, and a derive with neither has none; an artifact that states no
    target has an empty one.  What a result states is taken before what its artifact states."""
    d = str(tmp_path)
    assert R.build_summary(d) == "FAIL"
    nothing: dict[str, Any] = {"ok": False, "unsupported": None, "exception": None, "message": "no result.json", "program_digest": None, "census_digest": None,
               "identity": None, "correspondence_digest": None, "target": {}, "applicability": None}
    assert _read(tmp_path / "build_summary.json") == {"step": nothing, "request": nothing}
    for program in _left_out("program"):
        for target in _left_out("target_profile"):
            for part in ("build_step", "build_request", "build_request_LP8_T3"):
                _put(tmp_path / part / "result.json", {"ok": True, **program})
            _put(tmp_path / "build_step" / "artifact.json", {"program_digest": "a" * 64, "correspondence_digest": "k" * 64, **target})
            assert R.build_summary(d) == "PASS"
            s = _read(tmp_path / "build_summary.json")
            assert (s["step"]["program_digest"], s["step"]["correspondence_digest"], s["step"]["target"]) == ("a" * 64, "k" * 64, {}), (program, target)
            assert (s["request"]["program_digest"], s["request"]["correspondence_digest"], s["request"]["target"]) == (None, None, {})
            assert s["build_request_LP8_T3"] == {"ok": True, "unsupported": None, "message": "", "program_digest": None, "census_digest": None,
                                                 "correspondence_digest": None}
            assert "ir" not in s["step"] and "ir" not in s["build_request_LP8_T3"]
    stated = {"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": "q" * 64, "ir": {"root": "w"}}, "census": {"census": "c" * 64}}
    for part in ("build_step", "build_request_LP8_T3"):
        _put(tmp_path / part / "result.json", stated)
    _put(tmp_path / "build_step" / "artifact.json", {"program_digest": "a" * 64, "correspondence_digest": "k" * 64, "target_profile": {"protocols": ["pouw"]}})
    assert R.build_summary(d) == "PASS"
    s = _read(tmp_path / "build_summary.json")
    assert (s["step"]["program_digest"], s["step"]["correspondence_digest"], s["step"]["census_digest"]) == ("p" * 64, "q" * 64, "c" * 64)
    assert s["step"]["target"] == {"protocols": ["pouw"]} and s["step"]["ir"] == {"root": "w"}
    assert s["build_request_LP8_T3"] == {"ok": True, "unsupported": None, "message": "", "program_digest": "p" * 64, "census_digest": "c" * 64,
                                         "correspondence_digest": "q" * 64, "ir": {"root": "w"}}


@pytest.mark.parametrize(("workload", "stochastic"), [
    ({"sampling": None, "variants": {"greedy": {"temperature": 0.5}}}, True),
    ({"sampling": {}, "variants": {"greedy": {"temperature": 0.5}}}, True),
    ({"sampling": {"seed": 3}, "variants": {"greedy": {"temperature": 0.5}}}, True),
    ({"sampling": {"variant": "topp"}}, False),
    ({"sampling": {"variant": "topp"}, "variants": None}, False),
    ({"sampling": {"variant": "topp"}, "variants": {}}, False),
    ({"sampling": {"variant": "topp"}, "variants": {"topp": None}}, False),
    ({"sampling": {"variant": "topp"}, "variants": {"topp": {}}}, False),
    ({"sampling": {"variant": "topp"}, "variants": {"topp": {"top_p": 0.9}}}, False),
])
def test_a_workload_without_a_sampling_block_reads_its_greedy_variant_and_one_without_the_variant_does_not_sample(tmp_path, workload, stochastic):
    assert R.is_stochastic(_put(tmp_path / "w.json", workload)) is stochastic


def _match_summary(d: Path, rcs: int = -1) -> dict[str, Any]:
    R.match_summary(str(d), 0, 0, 0, STAGE_START, rcs, False)
    return _read(d / "match_summary.json")


def test_a_batch_rows_match_summary_reads_a_decomposition_with_blocks_left_out(tmp_path):
    """A decomposition that lists no request is not canonically equal and counts nothing; one that gives no reasons fails with none."""
    d = _summary_row(tmp_path, old=False, stage_record=True)
    for requests in _left_out("requests", []):
        for reasons in _left_out("reasons", []):
            for attribution in _left_out("attribution"):
                _put(d / "match_decomp.json", {"verdict": "FAIL", **requests, **reasons, **attribution})
                s = _match_summary(d)
                assert (s["canonical_equal"], s["steps"], s["pass"], s["compare_verdict"]) == (False, [0, 0], False, "FAIL batch_decomp FAIL: ")
                assert {k: s["totals"][k] for k in ("requests", "operands_equal", "operands_different", "unattributed", "shapes")} == {
                    "requests": 0, "operands_equal": 0, "operands_different": 0, "unattributed": None, "shapes": []}
    equal = {"canonical_equal": True}
    _put(d / "match_decomp.json", {"verdict": "FAIL", "reasons": ["a", "b"], "attribution": {"unattributed": 4}, "xreq_total": 1, "requests": [
        {**equal, "steps": 3, "LP": 8, "T": 2, "totals": {"operands_equal": 5, "operands_different": 1}},
        {**equal, "steps": 2, "LP": 4, "T": 1, "totals": None}, {**equal, "totals": {}}, {**equal, "totals": {"operands_equal": None}}]})
    s = _match_summary(d)
    assert (s["canonical_equal"], s["steps"], s["compare_verdict"]) == (True, [5, 5], "FAIL batch_decomp FAIL: a; b")
    assert {k: s["totals"][k] for k in ("requests", "operands_equal", "operands_different", "unattributed", "xreq", "shapes")} == {
        "requests": 4, "operands_equal": 5, "operands_different": 1, "unattributed": 4, "xreq": 1, "shapes": ["LP4_T1", "LP8_T2", "LPNone_TNone"]}


def test_a_single_requests_match_summary_reads_a_comparison_with_blocks_left_out(tmp_path):
    """A comparison document that states no comparison fails with nothing compared; each block of a comparison may be left out alone."""
    d = _summary_row(tmp_path, old=False, stage_record=True)
    (d / "match_decomp.json").unlink()
    for compare in _left_out("compare"):
        for diff in _left_out("histogram_diff_T_normalised", []):
            _put(d / "match_compare.json", {"profile": "p", **compare, **diff})
            s = _match_summary(d)
            assert s["compare_verdict"] == "FAIL canonical_equal=None steps=None/None operands_different=None hist_diff=0", (compare, diff)
            assert (s["canonical_equal"], s["steps"], s["totals"], s["pass"]) == (None, [None, None], {}, False)
    whole = {"totals": {"operands_different": 0, "operands_equal": 9}, "a": {"steps": 3}, "b": {"steps": 3}, "canonical_sha256": {"equal": True}}
    _put(d / "match_compare.json", {"compare": whole})
    s = _match_summary(d)
    assert (s["compare_verdict"], s["canonical_equal"], s["steps"], s["totals"], s["pass"]) == ("PASS", True, [3, 3], whole["totals"], True)
    for totals in _left_out("totals"):
        _put(d / "match_compare.json", {"compare": {**{k: v for k, v in whole.items() if k != "totals"}, **totals}})
        s = _match_summary(d)
        assert (s["compare_verdict"], s["totals"]) == ("PASS", {}), "a comparison that counts no operand has none that differs"
    said = {"a": "FAIL canonical_equal=True steps=None/3 operands_different=1 hist_diff=2", "b": "FAIL canonical_equal=True steps=3/None operands_different=1 hist_diff=2",
            "canonical_sha256": "FAIL canonical_equal=None steps=3/3 operands_different=1 hist_diff=2"}
    differs = {**whole, "totals": {"operands_different": 1}}
    for block, verdict in said.items():
        for left_out in _left_out(block):
            _put(d / "match_compare.json", {"compare": {**{k: v for k, v in differs.items() if k != block}, **left_out}, "histogram_diff_T_normalised": ["x", "y"]})
            s = _match_summary(d)
            assert (s["compare_verdict"], s["totals"]) == (verdict, {"operands_different": 1}), (block, left_out)


def test_a_match_summary_names_the_capture_or_control_stage_whose_record_is_absent_or_failed(tmp_path):
    d = _summary_row(tmp_path, old=False, stage_record=False)
    records: list[tuple[list[dict[str, Any]] | None, int | None, int | None]] = [
        (None, None, None), ([], None, None), ([{"stage": "control", "returncode": 0}], None, 0), ([{"stage": "capture", "returncode": 0}], 0, None),
        ([{"stage": "capture", "returncode": 1}, {"stage": "control", "returncode": 0}], 1, 0)]
    for stages, capture_rc, control_rc in records:
        if stages is not None:
            _put(d / "match" / "stages.json", {"stages": stages})
        s = _match_summary(d)
        reason = f"capture/control stage failed: capture rc={capture_rc} control rc={control_rc}"
        assert (s["record"]["capture_rc"], s["record"]["control_rc"], s["record"]["reason"]) == (capture_rc, control_rc, reason)
        assert (s["record"]["fresh"], s["compare_verdict"], s["pass"]) == (False, f"FAIL record: {reason}", False)
    _put(d / "match" / "stages.json", {"stages": [{"stage": "capture", "returncode": 0}, {"stage": "control", "returncode": 0}]})
    s = _match_summary(d)
    assert (s["record"]["fresh"], s["record"]["reason"], s["compare_verdict"], s["pass"]) == (True, None, "PASS", True)


def test_a_match_summary_reads_a_global_match_with_blocks_left_out(tmp_path):
    """A global Match that names no workload Program, no check or no request has no workload digest and no per-request length."""
    d = _summary_row(tmp_path, old=False, stage_record=True)
    for program in _left_out("global_program"):
        for checks in _left_out("checks", []):
            for requests in _left_out("requests", []):
                _put(d / "global_match.json", {"verdict": "PASS", "eos_truncated_requests": ["r0"], **program, **checks, **requests})
                g = _match_summary(d)["global_match"]
                assert (g["workload_digest"], g["checks"], g["generated_lengths"]) == (None, {}, {}), (program, checks, requests)
                assert (g["eos_verified"], g["served_output_lens"], g["executed_prefix_lens"], g["stop_rule"]) == (None, None, None, None)
    stated = {"verdict": "PASS", "global_program": {"workload_digest": "w" * 64}, "eos_truncated_requests": ["r0"],
              "checks": [{"id": "c1", "status": "pass"}, {"id": "chronology", "status": "pass", "stop_rule": {"set": [2]}}],
              "requests": [{"request_id": "r0", "generated_length": 3, "eos_verified": True, "served_output_len": 3, "executed_prefix_len": 2},
                           {"request_id": "r1", "generated_length": 1}]}
    _put(d / "global_match.json", stated)
    g = _match_summary(d)["global_match"]
    assert (g["workload_digest"], g["checks"], g["generated_lengths"]) == ("w" * 64, {"c1": "pass", "chronology": "pass"}, {"r0": 3, "r1": 1})
    assert (g["eos_verified"], g["served_output_lens"], g["executed_prefix_lens"], g["stop_rule"]) == ({"r0": True}, {"r0": 3}, {"r0": 2}, {"set": [2]})
    _put(d / "global_match.json", {**stated, "eos_truncated_requests": []})
    assert _match_summary(d)["global_match"]["stop_rule"] is None, "the stop rule is stated beside an early stop only"


def test_a_sampler_value_check_without_its_summary_was_not_run_and_one_without_reasons_gives_none(tmp_path):
    d = _summary_row(tmp_path, old=False, stage_record=True)
    for summary in _left_out("summary"):
        for reasons in _left_out("reasons", []):
            _put(d / "match_stoch.json", {"variant": "temperature", **summary, **reasons})
            s = _match_summary(d, rcs=0)
            assert s["stoch_value_check"] == {"rc": 0, "verdict": "NOT-RUN", "rows": None, "device_facts_all_match": None, "masked_rows_all_equal": None,
                                              "reasons": [], "variant": "temperature"}, (summary, reasons)
            assert (s["compare_verdict"], s["pass"]) == ("FAIL stoch value check NOT-RUN: ", False)
    _put(d / "match_stoch.json", {"summary": {"rows": 3, "device_facts_all_match": False, "masked_rows_all_equal": True}, "reasons": [f"r{i}" for i in range(8)]})
    s = _match_summary(d, rcs=0)
    assert s["stoch_value_check"] == {"rc": 0, "verdict": "NOT-RUN", "rows": 3, "device_facts_all_match": False, "masked_rows_all_equal": True,
                                      "reasons": ["r0", "r1", "r2", "r3", "r4", "r5"], "variant": None}
    assert s["compare_verdict"] == "FAIL stoch value check NOT-RUN: r0; r1; r2; r3; r4; r5"


def test_the_match_line_reads_a_summary_without_its_global_match_or_its_concurrency(tmp_path):
    summary = {"compare_verdict": "PASS", "control_vs_instrumented_tokens_equal": True, "fold_ok": None}
    for global_match in _left_out("global_match"):
        _put(tmp_path / "match_summary.json", {**summary, **global_match})
        assert R.match_line(str(tmp_path)) == "verdict PASS global None   concurrency None tokens_equal True fold None", global_match
    for concurrency in _left_out("achieved_concurrency"):
        _put(tmp_path / "match_summary.json", {**summary, "global_match": {"verdict": "FAIL", **concurrency}})
        assert R.match_line(str(tmp_path)) == "verdict PASS global FAIL   concurrency None tokens_equal True fold None", concurrency
    stated = {"verdict": "FAIL", "failed_checks": ["G1"], "within_step_criterion": "dag", "achieved_concurrency": {"observed_max": 3}}
    _put(tmp_path / "match_summary.json", {**summary, "global_match": stated})
    assert R.match_line(str(tmp_path)) == "verdict PASS global FAIL ['G1'] dag concurrency 3 tokens_equal True fold None"


def test_a_tap_applies_when_it_is_on_and_the_manifest_names_an_identity_of_its_family(tmp_path):
    taps = SimpleNamespace(norm_tap="1", router_tap="1", vocab_tap="0")
    assert R.applicable_taps(str(tmp_path / "absent.json"), taps) == []
    for manifest in (*_left_out("identities", []), []):
        assert R.applicable_taps(_put(tmp_path / "manifest.json", manifest), taps) == [], manifest
    (tmp_path / "torn.json").write_text("{")
    assert R.applicable_taps(str(tmp_path / "torn.json"), taps) == []
    identities = [{"family": "norm_scales"}, {"family": "vocab_range"}, {}]
    assert R.applicable_taps(_put(tmp_path / "manifest.json", {"identities": identities}), taps) == ["norm_tap"]
    every = SimpleNamespace(norm_tap="1", router_tap="1", vocab_tap="1")
    assert R.applicable_taps(_put(tmp_path / "manifest.json", {"identities": [*identities, {"family": "router_softmax"}]}), every) == [
        "norm_tap", "router_tap", "vocab_tap"]


def test_a_single_requests_digest_of_record_and_the_snapshot_steps_read_a_block_that_is_left_out(tmp_path):
    d = str(tmp_path)
    for request in (*_left_out("request"), {"request": {"program_digest": None}}):
        _put(tmp_path / "build_summary.json", request)
        assert R.program_digest_of_record(d, 1) == "", request
    for steps in _left_out("snapshot_steps", []):
        _put(tmp_path / "match" / "run.json", steps)
        assert R.snapshot_steps_of_record(d) == "", steps
    _put(tmp_path / "match" / "run.json", {"snapshot_steps": [4]})
    assert R.snapshot_steps_of_record(d) == "4"


def test_the_weights_of_record_come_from_the_manifests_first_component_that_is_derived(tmp_path):
    """A manifest that names no component, or whose components are not on disk, leaves the declared request's Program dir."""
    d = str(tmp_path)
    for manifest in _left_out("components"):
        _put(tmp_path / "manifest.json", manifest)
        assert R.wor_component(d) == os.path.join(d, "build_request"), manifest
    _put(tmp_path / "manifest.json", {"components": {"r10": {"LP": 4, "T": 1}, "r2": None, "r3": {"LP": 8}, "r5": {"LP": 8, "T": 3}}})
    assert R.wor_component(d) == os.path.join(d, "build_request")
    _put(tmp_path / "build_request_LP4_T1" / "result.json", {"ok": True, "LP": 4, "T": 1})
    assert R.wor_component(d) == os.path.join(d, "build_request_LP4_T1")
    _put(tmp_path / "build_request_LP8_T3" / "result.json", {"ok": True, "LP": 8, "T": 3})
    assert R.wor_component(d) == os.path.join(d, "build_request_LP8_T3"), "r5 is before r10: the shorter id first"


def test_a_weights_record_is_stale_by_its_components_digest_wherever_the_result_states_it(tmp_path):
    """A result states its Program digest at its top, in its `artifact` or in its `program`, in that order; a result that states none is
    not compared.  A record with no engine assumptions counts what it left unpinned, and an incomplete one what it left unresolved."""
    whole = {"program_digest": "p" * 64, "engine_assumptions": {}, "schema": "s/v1", "complete": True}
    record, result = _put(tmp_path / "weights.json", whole), str(tmp_path / "result.json")
    assert R.wor_stale(record, result, "s/v1") == ""
    same: list[dict[str, Any]] = [{}, {"artifact": None, "program": None}, {"artifact": {}, "program": {}}, {"program": {"digest": "p" * 64}}]
    for res in same:
        _put(tmp_path / "result.json", res)
        assert R.wor_stale(record, result, "s/v1") == "", res
    other: list[dict[str, Any]] = [{"program_digest": "q" * 64, "artifact": {"program_digest": "z" * 64}},
                                   {"artifact": {"program_digest": "q" * 64}, "program": {"digest": "z" * 64}},
                                   {"artifact": None, "program": {"digest": "q" * 64}}, {"artifact": {}, "program": {"digest": "q" * 64}}]
    for res in other:
        _put(tmp_path / "result.json", res)
        assert R.wor_stale(record, result, "s/v1") == f"built for Program {'p' * 16}, this component is {'q' * 16} (stale)", res
    _put(tmp_path / "result.json", {})
    for unpinned, n in (({}, 0), ({"unpinned": None}, 0), ({"unpinned": []}, 0), ({"unpinned": ["a", "b"]}, 2)):
        _put(tmp_path / "weights.json", {**{k: v for k, v in whole.items() if k != "engine_assumptions"}, **unpinned})
        assert R.wor_stale(record, result, "s/v1").endswith(f"(no engine_assumptions; {n} engine constant(s) unpinned by name)"), unpinned
    for unresolved, n in (({}, 0), ({"unresolved": None}, 0), ({"unresolved": []}, 0), ({"unresolved": ["a", "b", "c"]}, 3)):
        _put(tmp_path / "weights.json", {**whole, "complete": False, **unresolved})
        assert R.wor_stale(record, result, "s/v1") == f"incomplete ({n} unresolved field(s))", unresolved


def test_the_commit_line_and_a_tp_rows_world_read_a_document_without_its_runs_its_checks_or_its_sweep(tmp_path):
    for runs in _left_out("runs"):
        for checks in _left_out("checks", []):
            _put(tmp_path / "verdict.json", {**runs, **checks})
            assert R.commit_line(str(tmp_path)) == "runs None failed None checks ", (runs, checks)
    _put(tmp_path / "verdict.json", {"runs": {"failed": 2}, "checks": [{"name": "C2", "outcome": "pass"}]})
    assert R.commit_line(str(tmp_path)) == "runs None failed 2 checks C2=pass"
    for sweep in _left_out("sweep"):
        assert R.tp_world(_put(tmp_path / "w.json", sweep)) == 2, sweep
    assert R.tp_world(_put(tmp_path / "w.json", {"sweep": {"tp": 4}})) == 4


def test_a_tp_build_summary_reads_each_ranks_results_without_their_program(tmp_path):
    """A rank's derive that states no Program has no digest, and then the summary's step digest, rank 0's, is none."""
    b = tmp_path / "build"
    for program in _left_out("program"):
        for r in (0, 1):
            _put(b / f"rank{r}" / "build_step" / "result.json", {"ok": True, **program})
            _put(b / f"rank{r}" / "build_request_LP8_T3" / "result.json", {"ok": True, "aot": True, **program})
            _put(b / f"rank{r}" / "build_workload" / "workload_program.json", {"workload_digest": "w" * 64})
        assert R.tp_build_summary(str(b), 2) == "PASS"
        s = _read(tmp_path / "build_summary.json")
        assert s["step"] == {"program_digest": None} and s["tp"] == {"world": 2}, program
        for rank in (s["ranks"]["rank0"], s["ranks"]["rank1"]):
            assert rank["build_step"] == {"ok": True, "unsupported": None, "message": "", "program_digest": None, "census_digest": None, "construction": "trace"}
            assert (rank["build_request_LP8_T3"]["program_digest"], rank["build_request_LP8_T3"]["construction"]) == (None, "aot")
    _put(b / "rank0" / "build_step" / "result.json", {"ok": True, "program": {"digest": "p" * 64}, "census": {"census": "c" * 64}})
    assert R.tp_build_summary(str(b), 2) == "PASS"
    s = _read(tmp_path / "build_summary.json")
    assert s["step"] == {"program_digest": "p" * 64} and s["ranks"]["rank0"]["build_step"]["census_digest"] == "c" * 64
    assert R.tp_build_digests(str(tmp_path)) == ("p" * 16, "")
    for step in _left_out("build_step"):
        _put(tmp_path / "build_summary.json", {**s, "ranks": {**s["ranks"], "rank1": {"workload": {}, **step}}})
        assert R.tp_build_digests(str(tmp_path)) == ("p" * 16, ""), step
    _put(tmp_path / "build_summary.json", {**s, "ranks": {**s["ranks"], "rank1": {"build_step": {"program_digest": "s" * 64}}}})
    assert R.tp_build_digests(str(tmp_path)) == ("p" * 16, "s" * 16)
    three = {**s["ranks"], "rank1": {"build_step": {"program_digest": "s" * 64}}, "rank2": {"workload": {}}}
    _put(tmp_path / "build_summary.json", {**s, "tp": {"world": 3}, "ranks": three})
    assert R.tp_build_digests(str(tmp_path)) == ("p" * 16, "s" * 16 + "/")   # a rank without its step joins as an empty digest


def test_a_tp_rows_manifest_and_commit_lines_read_documents_with_blocks_left_out(tmp_path):
    d = str(tmp_path)
    for populations in _left_out("populations"):
        for binding in _left_out("tp_peer_binding"):
            _put(tmp_path / "manifest.json", {"complete": True, **populations, **binding})
            assert R.tp_manifest_line(d) == "complete True identities None tp_peer_binding_n_unbound None unmodelled null digest ", (populations, binding)
    _put(tmp_path / "manifest.json", {"complete": False, "populations": {"identities": 7}, "tp_peer_binding": {"n_unbound": 0}, "unmodelled": ["x"],
                                      "manifest_digest": "m" * 64})
    assert R.tp_manifest_line(d) == f"complete False identities 7 tp_peer_binding_n_unbound 0 unmodelled [\"x\"] digest {'m' * 16}"
    blocks = ("sampled_replay", "boundary_linkage", "cross_rank_collectives", "fold_match_binding", "weights_pin")
    nothing = "pass True pairs None delta None roots null tokens None replay None linkage None xrank None fold_binding None weights_pin None"
    left_out: list[dict[str, Any]] = [{}, dict.fromkeys(blocks), dict.fromkeys(blocks, {})]
    for summary in left_out:
        _put(tmp_path / "commit" / "summary.json", {"pass": True, **summary})
        assert R.tp_commit_line(d) == nothing, summary
    _put(tmp_path / "commit" / "summary.json", {"pass": True, **{b: {"result": i} for i, b in enumerate(blocks)}})
    assert R.tp_commit_line(d) == nothing.replace("replay None linkage None xrank None fold_binding None weights_pin None",
                                                 "replay 0 linkage 1 xrank 2 fold_binding 3 weights_pin 4")
