"""What a row's records say when a block of a document they read is left out (`verity_vllm.pipeline.row_records`).

The row's summaries and stage lines are read off documents its stages wrote: a workload manifest, a derive's result and artifact, the
compare's and the global Match's documents, the sampler value check, the manifest, a weights record, the Commit's verdict and
summary.  A block of such a document may be absent, null or empty (a stage that did not run, a writer with nothing to state), and
the record then states `None`, nothing or a named default for it.  Each test gives a block absent, null, empty and stated.

The Match summary is also held whole, a batch row's and a single request's, with each part of the rule its verdict PASS rests on
taken away alone, the name its global Match gets from its stage's exit where it left no document, and a stale record's reason.

The last tests are of a document that cannot be read at all, the memory and the CPUs a row's derives are sized by on a host
without a cgroup file or an affinity mask, token records that list no requests, a manifest built under other tap policies than
the Commit's, and what the records say whole: why a manifest is not of record, which tap build is absent, a batch row's digest of
record, the dir the weights of record come from, a weights record without its Program or its schema, and a served shape.  A
tensor-parallel row's records are held whole too: its shape, its build summary, its check, fold and Commit lines, its Match summary
and which of its documents passed.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.row_records as R
from tests.pipeline.test_row import PD, STAGE_START, _record_manifest, _summary_row
from tests.pipeline.test_row_match import _match_row
from verity_vllm.config import CHECKPOINTS, WORKLOADS, workload_file
from verity_vllm.pipeline import row_json
from verity_vllm.query.manifest.format import components_fold_digest

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


@pytest.fixture
def shape_names(tmp_path_factory, monkeypatch) -> list[str]:
    """The names ``match_shape.json``'s writer gives LP, the served T and the row's T, in that order: a single request's Match
    (`test_row_match`'s row, its commands stood in) writes the record, and its reader looks each up by that name."""
    r, _ = _match_row(tmp_path_factory.mktemp("writer"), monkeypatch)
    r.compare(60)
    return list(_read(Path(r.d, "match_shape.json")))[:3]


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
    row that samples counts its first request's prompt ids where the request states no prompt length, and takes the length it states."""
    w = json.loads(workload_file(SAMPLED).read_text())
    sweep, first = w["sweep"], w["requests"][0]
    envelope = (1024, 127, "0,1")   # the sweep's longest prompt, one step short of its longest answer, and the first two steps
    greedy = [*_left_out("sampling_params"), {"sampling_params": {"temperature": None}}, {"sampling_params": {"temperature": 0}}, {"sampling_params": {"top_p": 0.9}}]
    for params in greedy:
        doc = {**w, "sweep": {**{k: v for k, v in sweep.items() if k != "sampling_params"}, **params}}
        shape = R.row_shape(_put(tmp_path / "w.json", doc), str(CHECKPOINTS), str(tmp_path))
        assert (shape["LP"], shape["T"], shape["snap"]) == envelope, params
    for ids, n in (({}, 0), ({"prompt_token_ids": None}, 0), ({"prompt_token_ids": []}, 0), ({"prompt_token_ids": [5, 6, 7]}, 3),
                   ({"prompt_len": 9, "prompt_token_ids": [5, 6, 7]}, 9), ({"prompt_len": 0, "prompt_token_ids": [5, 6]}, 2), ({"prompt_len": 9}, 9)):
        r0 = {**{k: v for k, v in first.items() if k not in ("prompt_len", "prompt_token_ids")}, **ids}
        shape = R.row_shape(_put(tmp_path / "w.json", {**w, "requests": [r0]}), str(CHECKPOINTS), str(tmp_path))
        assert (shape["LP"], shape["T"], shape["snap"]) == (n, first["max_tokens"] - 1, "all"), ids


def test_a_logs_last_line_is_read_past_a_byte_that_is_no_text_and_a_derive_is_ok_only_where_its_result_says_true(tmp_path):
    log = tmp_path / "stage.log"
    log.write_bytes(b"first\n\xff last\n")
    assert R.tail(str(log))[1:] == " last" and R.tail(str(log), 3)[1:] == " l"
    assert R.derived_ok(str(tmp_path)) is False
    for result, ok in (({"ok": True}, True), ({"ok": 1}, False), ({"ok": "true"}, False), ({"ok": False}, False), ({}, False), ([], False), (None, False)):
        _put(tmp_path / "result.json", result)
        assert R.derived_ok(str(tmp_path)) is ok, result
    (tmp_path / "result.json").write_text("{")
    assert R.derived_ok(str(tmp_path)) is False


def test_shapes_with_as_many_tokens_start_with_the_longest_prompt_and_a_model_of_one_layer_has_one(tmp_path):
    """Shapes are ordered by their prompt and decode steps together, the longest first; of two with as many, the longer prompt, and
    of two the same, the one that reads no cached row."""
    assert R.longest_first({(4, 6), (8, 1), (6, 4, 2), (5, 5), (6, 4), (2, 9)}) == [(2, 9), (6, 4), (6, 4, 2), (5, 5), (4, 6), (8, 1)]
    for config, layers in (({"num_hidden_layers": 1}, 1), ({"num_hidden_layers": 0, "text_config": {"num_hidden_layers": 1}}, 1),
                           ({"num_hidden_layers": 0}, None), ({"num_hidden_layers": "2"}, None), ({"text_config": {"num_hidden_layers": -1}}, None)):
        _put(tmp_path / "config.json", config)
        assert R.model_layers(str(tmp_path)) == layers, config
    assert R.model_layers("") is None and R.model_layers(str(tmp_path / "absent")) is None


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


def test_a_build_summary_states_what_each_derive_refused_and_what_its_artifact_says_it_is(tmp_path):
    """A derive's refusal, its exception and its message cut at 300 characters are stated under its name, with its artifact's
    identity and applicability; a request shape's derive states its refusal and its message."""
    d = str(tmp_path)
    refused = {"ok": False, "unsupported": "op x", "exception": "ValueError", "message": "m" * 310}
    for part in ("build_step", "build_request", "build_request_LP8_T3"):
        _put(tmp_path / part / "result.json", refused)
    _put(tmp_path / "build_step" / "artifact.json", {"identity": {"model": "m"}, "applicability": {"B": [1]}})
    assert R.build_summary(d) == "FAIL"
    said = {"ok": False, "unsupported": "op x", "message": "m" * 300, "program_digest": None, "census_digest": None, "correspondence_digest": None}
    assert _read(tmp_path / "build_summary.json") == {
        "step": {**said, "exception": "ValueError", "identity": {"model": "m"}, "target": {}, "applicability": {"B": [1]}},
        "request": {**said, "exception": "ValueError", "identity": None, "target": {}, "applicability": None}, "build_request_LP8_T3": said}


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


def _verdict(d: Path, *, rcm: int = 0, rcc: int = 0, rcg: int = 0, resumed: bool = False) -> tuple[str, dict[str, Any]]:
    """What `match_summary` returns for the greedy row at `d`, and the summary it wrote."""
    return R.match_summary(str(d), rcm, rcc, rcg, STAGE_START, -1, resumed), _read(d / "match_summary.json")


def _single_row(tmp_path: Path) -> Path:
    """A single request's row whose Match passes: its comparison is equal step for step with no operand that differs, its tokens
    are the control's, its fold left a Program, its global Match passes and its record is this attempt's."""
    d = _summary_row(tmp_path, old=False, stage_record=True)
    (d / "match_decomp.json").unlink()
    _put(d / "match_compare.json", {"compare": {"totals": {"operands_different": 0, "operands_equal": 9}, "a": {"steps": 3}, "b": {"steps": 3},
                                                 "canonical_sha256": {"equal": True}}, "histogram_diff_T_normalised": ["x"]})
    _put(d / "global_match.json", {"verdict": "PASS"})
    return d


STAGE_START_UTC = "2020-09-13T12:26:40Z"
RAN = {"rc": 0, "status": "ok", "reason": ""}
NO_GLOBAL_MATCH: dict[str, Any] = {
    "comparator": None, "rules_version": None, "rc": 0, "verdict": None, "failed_checks": None, "checks": {}, "within_step_criterion": None,
    "workload_digest": None, "achieved_concurrency": None, "generated_lengths": {}, "eos_truncated_requests": None, "eos_verified": None,
    "served_output_lens": None, "executed_prefix_lens": None, "lag": None, "alerts": None, "stop_rule": None, "seconds": None,
    "artifact": "global_match.json"}


def test_a_batch_rows_match_summary_states_each_document_it_read_and_returns_its_verdict(tmp_path):
    """A batch row that passes, its summary whole.  The steps and the operand counts are its requests' added; a stage is named by
    `name` or `stage` and its exit by `returncode` or `rc`, its reason cut at 200 characters; the log's time and the stage's start
    are each stated; the global Match's document is stated key for key, and the comparator names the decomposition inside it."""
    d = _summary_row(tmp_path, old=False, stage_record=False)
    os.utime(d / "match" / "capture" / "log.jsonl.gz", (STAGE_START + 60, STAGE_START + 60))
    _put(d / "match" / "stages.json", {"stages": [{"stage": "capture", "returncode": 0, "status": "ok"},
                                                   {"name": "control", "rc": 0, "status": "ok", "reason": "r" * 250},
                                                   {"stage": "fold", "returncode": 0, "status": "ok", "reason": None}]})
    equal = {"canonical_equal": True}
    _put(d / "match_decomp.json", {"verdict": "PASS", "reasons": [], "within_step_criterion": "dag", "xreq_total": 0, "attribution": {"unattributed": 0}, "requests": [
        {**equal, "steps": 3, "LP": 8, "T": 2, "totals": {"operands_equal": 5, "operands_different": 0}},
        {**equal, "steps": 2, "LP": 4, "T": 1, "totals": {"operands_equal": 4, "operands_different": 0}}]})
    stated = {"comparator": "cb_a.global_match/v2", "rules_version": "r7", "verdict": "PASS", "failed_checks": [], "within_step_criterion": "dag",
              "achieved_concurrency": {"observed_max": 2}, "eos_truncated_requests": ["r1"], "lag": {"declared": 1}, "alerts": ["a"], "seconds": 1.5}
    _put(d / "global_match.json", {**stated, "global_program": {"workload_digest": "w" * 64},
                                   "checks": [{"id": "G1", "status": "pass"}, {"id": "chronology", "status": "pass", "stop_rule": {"set": [2]}}],
                                   "requests": [{"request_id": "r0", "generated_length": 3, "eos_verified": True, "served_output_len": 3, "executed_prefix_len": 2},
                                                {"request_id": "r1", "generated_length": 1}]})
    said, s = _verdict(d, rcm=7, rcc=8)
    assert said == "PASS"
    assert s == {
        "run_config_rc": 7, "compare_rc": 8, "compare_verdict": "PASS", "canonical_equal": True, "steps": [5, 5],
        "totals": {"requests": 2, "operands_equal": 9, "operands_different": 0, "unattributed": 0, "xreq": 0, "shapes": ["LP4_T1", "LP8_T2"],
                   "within_step_criterion": "dag",
                   "operands_note": "positional comparison counts (sequence); verdict per within_step_criterion, see match_decomp.json per request"},
        "request_shape": None,
        "record": {"fresh": True, "log_mtime_utc": "2020-09-13T12:27:40Z", "stage_start_utc": STAGE_START_UTC, "resumed": False,
                   "resumed_record_complete": None, "stage_record_absent": False, "capture_rc": 0, "control_rc": 0, "reason": None},
        "stoch_value_check": None, "histogram_diff_T_normalised": None, "fold_ok": True,
        "comparator": "cb_a.global_match/v2 (X-09 batch_decomp inside)", "per_request_comparator": "cb_a.batch_decomp (X-09)",
        "global_match": {**stated, "rc": 0, "checks": {"G1": "pass", "chronology": "pass"}, "workload_digest": "w" * 64, "generated_lengths": {"r0": 3, "r1": 1},
                         "eos_verified": {"r0": True}, "served_output_lens": {"r0": 3}, "executed_prefix_lens": {"r0": 2}, "stop_rule": {"set": [2]},
                         "artifact": "global_match.json"},
        "control_vs_instrumented_tokens_equal": True, "stages": {"capture": RAN, "control": {**RAN, "reason": "r" * 200}, "fold": RAN}, "pass": True}


def test_a_single_requests_match_summary_names_the_comparison_inside_it_and_the_shape_it_was_served_at(tmp_path, shape_names):
    """A single request's row that passes, its summary whole: the comparison's own counts and steps, the differing histogram rows it
    lists, the shape the request was served at, and a comparator that is the global Match's first version where its document names
    none."""
    d = _single_row(tmp_path)
    shape = dict(zip(shape_names, (8, 2, 3), strict=True))
    _put(d / "match_shape.json", shape)
    said, s = _verdict(d)
    assert said == "PASS"
    assert s == {
        "run_config_rc": 0, "compare_rc": 0, "compare_verdict": "PASS", "canonical_equal": True, "steps": [3, 3],
        "totals": {"operands_different": 0, "operands_equal": 9}, "request_shape": shape,
        "record": {"fresh": True, "log_mtime_utc": STAGE_START_UTC, "stage_start_utc": STAGE_START_UTC, "resumed": False,
                   "resumed_record_complete": None, "stage_record_absent": False, "capture_rc": 0, "control_rc": 0, "reason": None},
        "stoch_value_check": None, "histogram_diff_T_normalised": ["x"], "fold_ok": True,
        "comparator": "cb_a.global_match/v1 (cb_a.compare inside)", "per_request_comparator": "cb_a.compare",
        "global_match": {**NO_GLOBAL_MATCH, "verdict": "PASS"},
        "control_vs_instrumented_tokens_equal": True, "stages": {"capture": RAN, "control": RAN, "fold": RAN}, "pass": True}


def _tokens(d: Path, **fields: Any) -> None:
    """The capture's token record with its one request's `fields` changed, the control's as it was."""
    doc = _read(d / "match" / "capture" / "tokens.json")
    _put(d / "match" / "capture" / "tokens.json", {**doc, "requests": [{**doc["requests"][0], **fields}]})


def _compare(d: Path, **blocks: Any) -> None:
    doc = _read(d / "match_compare.json")
    _put(d / "match_compare.json", {**doc, "compare": {**doc["compare"], **blocks}})


def test_a_match_passes_with_every_part_of_its_rule_and_fails_without_any_one(tmp_path):
    """Each part of the rule alone makes a FAIL of a row that passes otherwise, and the summary says which: an operand that
    differs, a step count that differs, a token record that is not the control's in a request's id, its prompt or what it
    generated, a fold that left no Program or no instances, a global Match that fails or whose stage did not exit 0, and no
    comparison at all.  A token record that differs only in what is not a request's tokens is the control's."""
    one_wrong: list[tuple[Any, int, tuple[str | None, bool, bool, str]]] = [
        (lambda d: _compare(d, totals={"operands_different": 1}), 0,
         ("FAIL canonical_equal=True steps=3/3 operands_different=1 hist_diff=1", True, True, "PASS")),
        (lambda d: _compare(d, b={"steps": 4}), 0, ("FAIL canonical_equal=True steps=3/4 operands_different=0 hist_diff=1", True, True, "PASS")),
        (lambda d: _tokens(d, generated_token_ids=[4]), 0, ("PASS", False, True, "PASS")),
        (lambda d: _tokens(d, prompt_token_ids=[1]), 0, ("PASS", False, True, "PASS")),
        (lambda d: _tokens(d, request_id="r1"), 0, ("PASS", False, True, "PASS")),
        (lambda d: (d / "match" / "program.json").unlink(), 0, ("PASS", True, False, "PASS")),
        (lambda d: (d / "match" / "instances.jsonl").unlink(), 0, ("PASS", True, False, "PASS")),
        (lambda d: None, 1, ("PASS", True, True, "PASS")),
        (lambda d: _put(d / "global_match.json", {"verdict": "FAIL"}), 0, ("PASS", True, True, "FAIL")),
        (lambda d: (d / "match_compare.json").unlink(), 0, (None, True, True, "PASS"))]
    for k, (wrong, rcg, stated) in enumerate(one_wrong):
        d = _single_row(tmp_path / str(k))
        assert _verdict(d)[0] == "PASS"
        wrong(d)
        said, s = _verdict(d, rcg=rcg)
        assert (said, s["pass"]) == ("FAIL", False), k
        assert (s["compare_verdict"], s["control_vs_instrumented_tokens_equal"], s["fold_ok"], s["global_match"]["verdict"]) == stated, k
        assert s["global_match"]["rc"] == rcg and s["record"]["fresh"] is True, k
    d = _single_row(tmp_path / "same")
    _tokens(d, logprobs=[0.5])
    _put(d / "match" / "capture" / "tokens.json", {**_read(d / "match" / "capture" / "tokens.json"), "timing": 2, "observer": "o"})
    said, s = _verdict(d)
    assert (said, s["pass"], s["control_vs_instrumented_tokens_equal"]) == ("PASS", True, True)


@pytest.mark.parametrize(("rcg", "verdict"), [(97, "NOT RUN"), (98, "NO WORKLOAD PROGRAM"), (99, "NO FOLD"), (124, "TIMEOUT"), (5, "rc=5")])
def test_a_match_without_a_global_match_document_names_why_by_the_stages_exit_and_fails(tmp_path, rcg, verdict):
    """With no document the global Match's verdict is the name of its stage's exit, and the comparator is the first version's.  A
    document that states a verdict keeps it whatever the exit."""
    d = _single_row(tmp_path)
    said, s = _verdict(d, rcg=rcg)
    assert (said, s["pass"], s["global_match"]) == ("FAIL", False, {**NO_GLOBAL_MATCH, "rc": rcg, "verdict": "PASS"})
    (d / "global_match.json").unlink()
    said, s = _verdict(d, rcg=rcg)
    assert (said, s["pass"], s["compare_verdict"]) == ("FAIL", False, "PASS")
    assert s["global_match"] == {**NO_GLOBAL_MATCH, "rc": rcg, "verdict": verdict} and s["comparator"] == "cb_a.global_match/v1 (cb_a.compare inside)"


def test_a_stale_record_and_a_resume_with_one_gpu_stages_record_are_each_named_whole(tmp_path):
    """A record older than the stage fails the Match by name, with the log's own time.  A resume whose stage record names the capture
    or the control but not both is not a resume without a stage record: the stage it does not name has no exit, and the Match fails."""
    stale = "record predates this match stage (stale record from a previous attempt)"
    said, s = _verdict(_summary_row(tmp_path / "old", old=True, stage_record=True))
    assert (said, s["pass"], s["compare_verdict"]) == ("FAIL", False, f"FAIL record: {stale}")
    assert (s["record"]["reason"], s["record"]["log_mtime_utc"], s["record"]["stage_start_utc"]) == (stale, "2020-09-13T11:26:40Z", STAGE_START_UTC)
    for k, (stage, capture_rc, control_rc) in enumerate((("capture", 0, None), ("control", None, 0))):
        d = _summary_row(tmp_path / str(k), old=True, stage_record=False)
        _put(d / "match" / "stages.json", {"stages": [{"stage": stage, "returncode": 0}]})
        said, s = _verdict(d, resumed=True)
        reason = f"capture/control stage failed: capture rc={capture_rc} control rc={control_rc}"
        assert (said, s["pass"], s["compare_verdict"]) == ("FAIL", False, f"FAIL record: {reason}"), stage
        assert s["record"] == {"fresh": False, "log_mtime_utc": "2020-09-13T11:26:40Z", "stage_start_utc": STAGE_START_UTC, "resumed": True,
                               "resumed_record_complete": True, "stage_record_absent": False, "capture_rc": capture_rc, "control_rc": control_rc,
                               "reason": reason}, stage


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


def test_a_record_that_cannot_be_read_gives_its_named_default_and_one_that_can_gives_what_it_states(tmp_path):
    """The workload Program's digest and a failed compose's reason are "" where `build_workload/workload_program.json` is absent or
    no document; the compare's exit is 3 where `match_decomp.json` is no document; a TP row's manifest is complete only where it
    says so."""
    d = str(tmp_path)
    assert (R.workload_digest(d), R.compose_reason(d), R.tp_manifest_complete(d)) == ("", "", False)
    program = tmp_path / "build_workload" / "workload_program.json"
    _put(program, {"reason": "missing_shape", "message": "LP 8 T 3"})
    assert R.compose_reason(d) == "missing_shape LP 8 T 3"
    _put(program, {"reason": "missing_shape", "message": None})
    assert R.compose_reason(d) == "missing_shape "
    program.write_text("not a document")
    assert (R.workload_digest(d), R.compose_reason(d)) == ("", "")
    (tmp_path / "match_decomp.json").write_text("not a document")
    assert R.decomp_rc(d) == 3
    for complete, said in ((True, True), (False, False), ("yes", False), (None, False)):
        _put(tmp_path / "manifest.json", {"complete": complete})
        assert R.tp_manifest_complete(d) is said, complete


def test_the_workload_digest_is_the_composes_workload_digest_else_its_program_digest_else_its_digest(tmp_path):
    program = tmp_path / "build_workload" / "workload_program.json"
    for doc, digest in (({"workload_digest": "w" * 64, "program_digest": "p" * 64, "digest": "g" * 64}, "w" * 64),
                        ({"workload_digest": None, "program_digest": "p" * 64, "digest": "g" * 64}, "p" * 64), ({"program_digest": "p" * 64}, "p" * 64),
                        ({"program_digest": "", "digest": "g" * 64}, "g" * 64), ({"digest": "g" * 64}, "g" * 64), ({"digest": None}, ""), ({}, "")):
        _put(program, doc)
        assert R.workload_digest(str(tmp_path)) == digest, doc


def test_a_workload_program_is_ok_only_where_its_compose_exited_0_with_a_digest_and_its_refusal_is_stated(tmp_path):
    """`add_workload` adds the workload Program to the Build's summary and leaves the rest as it was: ok only for a compose that
    exited 0 and gave a digest, and its refusal, applicability and stop rule as `build_workload/workload_program.json` states them."""
    _put(tmp_path / "build_summary.json", {"step": {"ok": True}})
    refusal = {"reason": "applicability", "message": "B 9 is not declared", "applicability": {"B": [1, 8]}, "stop_rule": {"eos": [2]}}
    _put(tmp_path / "build_workload" / "workload_program.json", refusal)
    for rc, wd, ok in ((0, "w" * 64, True), (0, "", False), (1, "w" * 64, False), (2, "", False)):
        R.add_workload(str(tmp_path), wd, rc)
        assert _read(tmp_path / "build_summary.json") == {"step": {"ok": True}, "workload": {
            "ok": ok, "workload_digest": wd or None, "mechanism": "GP-01 cb_a.global_program", "artifact": "build_workload/workload_program.json",
            **refusal}}, (rc, wd)


def test_the_memory_headroom_without_a_cgroup_file_is_the_roots_and_the_cpus_are_those_the_process_may_run_on(tmp_path, monkeypatch):
    meminfo, root = tmp_path / "meminfo", tmp_path / "cgroup"
    meminfo.write_text("MemTotal:       1000 kB\nMemAvailable:    500 kB\n")
    root.mkdir()

    def head() -> int:
        return R._memory_headroom(str(tmp_path / "no-cgroup-file"), str(root), str(meminfo))
    assert head() == 500 * 1024
    (root / "memory.max").write_text("204800\n")
    (root / "memory.current").write_text("4096\n")
    assert head() == 200704
    monkeypatch.setattr(R.os, "sched_getaffinity", lambda pid: {0, 2, 5}, raising=False)
    assert R._cpus() == 3
    monkeypatch.delattr(R.os, "sched_getaffinity")
    monkeypatch.setattr(R.os, "cpu_count", lambda: None)
    assert R._cpus() == 1


def test_the_memory_headroom_by_default_is_this_processs_cgroup_chain_under_the_mount_and_the_hosts_meminfo(monkeypatch):
    """With no paths given, the headroom reads `/proc/self/cgroup`, the cgroup it names under `/sys/fs/cgroup` and `/proc/meminfo`:
    the files are stood in by path, so no host file is read."""
    files = {"/proc/meminfo": "MemAvailable:    500 kB\n", "/proc/self/cgroup": "0::/q/job\n",
             "/sys/fs/cgroup/q/job/memory.max": "307200\n", "/sys/fs/cgroup/q/job/memory.current": "4096\n", "/sys/fs/cgroup/q/memory.max": "max\n"}

    def stood_in(path: str) -> io.StringIO:
        if path not in files:
            raise FileNotFoundError(path)
        return io.StringIO(files[path])
    monkeypatch.setattr(R, "open", stood_in, raising=False)
    assert R._memory_headroom() == 307200 - 4096


def test_a_match_summary_compares_token_records_that_list_no_requests_by_their_tokens_or_whole(tmp_path):
    """A token record without `requests` is compared by its `tokens`, or whole where it has none or is a list.  Two records with
    nothing in them are not equal tokens, and one that is neither a list nor a table says nothing."""
    d = _summary_row(tmp_path, old=False, stage_record=True)

    def equal(control: Any, capture: Any) -> Any:
        _put(d / "match" / "control" / "tokens.json", control)
        _put(d / "match" / "capture" / "tokens.json", capture)
        return _match_summary(d)["control_vs_instrumented_tokens_equal"]
    assert equal({"tokens": [1, 2, 3], "timing": 1.0}, {"tokens": [1, 2, 3], "timing": 2.0}) is True
    assert equal({"tokens": [1, 2, 3]}, {"tokens": [1, 2, 4]}) is False
    assert equal([1, 2, 3], [1, 2, 3]) is True and equal([1, 2, 3], {"tokens": [1, 2, 3]}) is True
    assert equal({"timing": 1.0}, {"timing": 1.0}) is True and equal({"timing": 1.0}, {"timing": 2.0}) is False
    assert equal([], []) is False and equal({"tokens": []}, {"tokens": []}) is False
    assert equal(7, 7) is None


def test_a_manifest_built_under_other_tap_policies_than_the_commits_is_not_its_manifest_of_record(tmp_path):
    """Either way: a policy the manifest's query states that the Commit does not run, and one the Commit runs that it does not state."""
    p, d = tmp_path / "manifest.json", str(tmp_path)
    _record_manifest(p, R.QUERY_OF_RECORD)
    _put(p, _read(p) | {"complete": True})
    every = SimpleNamespace(norm_tap="1", router_tap="1", vocab_tap="1")
    assert R.manifest_not_of_record(_read(p), PD, every) == ("", "") and R.tp_manifest_not_of_record(d, every) == ""
    no_router = SimpleNamespace(norm_tap="1", router_tap="0", vocab_tap="1")
    why = "disagrees with this Commit's taps on the router_softmax policy"
    assert R.manifest_not_of_record(_read(p), PD, no_router) == ("policy-router-softmax", why)
    assert R.tp_manifest_not_of_record(d, no_router) == why
    doc = _read(p)
    del doc["query"]["vocab_range"]
    _put(p, doc)
    why = "disagrees with this Commit's taps on the router_softmax, vocab_range policy"
    assert R.manifest_not_of_record(doc, PD, no_router) == ("policy-router-softmax-vocab-range", why)
    assert R.tp_manifest_not_of_record(d, no_router) == why


def test_a_component_whose_result_cannot_be_read_is_passed_over_for_the_next(tmp_path):
    d = str(tmp_path)
    _put(tmp_path / "manifest.json", {"components": {"r0": {"LP": 4, "T": 1}, "r1": {"LP": 8, "T": 3}}})
    (tmp_path / "build_request_LP4_T1").mkdir()
    (tmp_path / "build_request_LP4_T1" / "result.json").write_text("not a document")
    assert R.wor_component(d) == os.path.join(d, "build_request")
    _put(tmp_path / "build_request_LP8_T3" / "result.json", {"ok": True, "LP": 8, "T": 3})
    assert R.wor_component(d) == os.path.join(d, "build_request_LP8_T3")


def test_the_first_component_is_the_declared_requests_dir_where_its_own_result_states_that_shape(tmp_path):
    """The first component's shape has no dir of its own and is the shape the declared request's result states: the weights of record
    come from `build_request`, not from a later component that has its dir."""
    d = str(tmp_path)
    _put(tmp_path / "manifest.json", {"components": {"r0": {"LP": 4, "T": 1}, "r1": {"LP": 8, "T": 3}}})
    _put(tmp_path / "build_request_LP8_T3" / "result.json", {"ok": True, "LP": 8, "T": 3})
    assert R.wor_component(d) == os.path.join(d, "build_request_LP8_T3")
    _put(tmp_path / "build_request" / "result.json", {"ok": True, "LP": 4, "T": 1})
    assert R.wor_component(d) == os.path.join(d, "build_request")
    _put(tmp_path / "build_request" / "result.json", {"ok": True, "LP": 4, "T": 2})
    assert R.wor_component(d) == os.path.join(d, "build_request_LP8_T3"), "another shape than the first component's"


def test_a_manifest_that_is_not_of_record_says_which_of_its_statements_is_not_the_commits(tmp_path):
    """Each reason whole, in the order they are looked for: a document with no query block or another query's, one that names no
    partition, and one of another Program than the Build's, or beside a Build that recorded none."""
    every = SimpleNamespace(norm_tap="1", router_tap="1", vocab_tap="1")
    other = f"is not built under the query of record {R.QUERY_OF_RECORD} (query.query_id="
    no_query: list[Any] = [None, [], {}, {"query": None}, {"query": "Q"}, {"query": {}}, {"query": {"query_id": None}}, {"query": {"query_id": ""}}]
    for doc in no_query:
        assert R.manifest_not_of_record(doc, PD, every) == ("other-query", other + "none)"), doc
    assert R.manifest_not_of_record({"query": {"query_id": "Q_module_body_v1"}}, PD, every) == ("other-query", other + "Q_module_body_v1)")
    p = tmp_path / "manifest.json"
    _record_manifest(p, R.QUERY_OF_RECORD, partition=False)
    assert R.manifest_not_of_record(_read(p), PD, every) == ("no-partition", "names no partition (query.partition)")
    _record_manifest(p, R.QUERY_OF_RECORD, program_digest="q" * 64)
    assert R.manifest_not_of_record(_read(p), PD, every) == ("other-program", f"is of Program {'q' * 16}, not the Build's {'p' * 16}")
    _record_manifest(p, R.QUERY_OF_RECORD)
    assert R.manifest_not_of_record(_read(p), "", every) == ("other-program", f"is of Program {'p' * 16}, not the Build's (unknown: no build_summary.json)")
    assert R.manifest_not_of_record(_read(p), PD, every) == ("", "")


def test_a_tap_build_that_is_not_on_disk_is_named_with_the_script_that_builds_it(tmp_path):
    """The first applicable tap whose build is absent, the norm tap before the router's; the vocabulary range has no build."""
    norm, router = tmp_path / "verity_norm_tap.so", tmp_path / "verity_router_tap.so"
    env = SimpleNamespace(norm_tap_so=str(norm), router_tap_so=str(router))
    said = "{}=1 and the manifest names its family, but the tap .so is missing: {} (ops/{}, run by ops/pod_bootstrap.sh)"
    taps = ["norm_tap", "router_tap", "vocab_tap"]
    assert R.missing_tap_build(taps, env) == said.format("NORM_TAP", norm, "pod_norm_tap.sh")
    assert R.missing_tap_build(["router_tap", "vocab_tap"], env) == said.format("ROUTER_TAP", router, "pod_router_tap.sh")
    norm.write_bytes(b"")
    assert R.missing_tap_build(taps, env) == said.format("ROUTER_TAP", router, "pod_router_tap.sh")
    router.write_bytes(b"")
    assert R.missing_tap_build(taps, env) == "" and R.missing_tap_build(["vocab_tap"], SimpleNamespace(norm_tap_so="", router_tap_so="")) == ""


def test_a_batch_rows_digest_of_record_is_the_workload_programs_under_any_of_its_names_or_the_fold_of_its_shapes(tmp_path):
    """The workload Program's digest is read under `workload_digest`, `program_digest` or `digest`, in that order.  Without one the
    digest of record is the fold of the per-shape requests' digests, sorted: the declared request's is not one of them, nor is a
    shape whose entry states no digest; and with no shape either there is none."""
    d, program = str(tmp_path), tmp_path / "build_workload" / "workload_program.json"
    a, b = "a" * 64, "b" * 64
    _put(tmp_path / "build_summary.json", {"ok": True, "request": {"program_digest": "r" * 64}, "build_request_LP8_T3": {"program_digest": b},
                                           "build_request_LP4_T1": {"program_digest": a}, "build_request_LP2_T1": None, "build_request_LP1_T1": {},
                                           "build_request_LP16_T1": {"program_digest": None}, "workload": {"program_digest": "w" * 64}})
    folded = components_fold_digest([a, b])
    assert R.program_digest_of_record(d, 2) == folded and folded not in ("", a, b)
    for doc, digest in (({"workload_digest": "w" * 64, "program_digest": "p" * 64, "digest": "g" * 64}, "w" * 64),
                        ({"workload_digest": None, "program_digest": "p" * 64, "digest": "g" * 64}, "p" * 64),
                        ({"workload_digest": "", "program_digest": "", "digest": "g" * 64}, "g" * 64), ({"digest": "g" * 64}, "g" * 64),
                        ({"digest": None}, folded), ({}, folded)):
        _put(program, doc)
        assert R.program_digest_of_record(d, 2) == digest, doc
    _put(tmp_path / "build_summary.json", {"ok": True, "request": {"program_digest": "r" * 64}, "build_request_LP1_T1": {}})
    assert R.program_digest_of_record(d, 2) == "" and R.program_digest_of_record(d, 1) == "r" * 64
    _put(program, {"digest": "g" * 64})
    assert R.program_digest_of_record(d, 2) == "g" * 64 and R.program_digest_of_record(d, 1) == "r" * 64


def test_a_weights_record_without_its_program_or_its_schema_says_so_with_a_question_mark(tmp_path):
    """A record of another schema than this tree derives is stale by name, and one that states no Program digest or no schema is
    named `?` where the digest or the schema would be."""
    whole = {"program_digest": "p" * 64, "engine_assumptions": {}, "schema": "s/v1", "complete": True}
    record, result = str(tmp_path / "weights.json"), _put(tmp_path / "result.json", {"program_digest": "q" * 64})
    for digest in _left_out("program_digest", ""):
        _put(tmp_path / "weights.json", {**{k: v for k, v in whole.items() if k != "program_digest"}, **digest})
        assert R.wor_stale(record, result, "s/v1") == f"built for Program ?, this component is {'q' * 16} (stale)", digest
    _put(tmp_path / "result.json", {})
    other = ", this tree derives s/v2 (root formula differs: the analytic entry is keyed by its inputs, not the host)"
    _put(tmp_path / "weights.json", whole)
    assert R.wor_stale(record, result, "s/v2") == "schema s/v1" + other and R.wor_stale(record, result, "") == ""
    for schema in _left_out("schema", ""):
        _put(tmp_path / "weights.json", {**{k: v for k, v in whole.items() if k != "schema"}, **schema})
        assert R.wor_stale(record, result, "s/v2") == "schema ?" + other and R.wor_stale(record, result, "") == "", schema


@pytest.mark.parametrize(("t", "said"), [(1, True), (2, True), (3, False), (0, False), (-1, False), (4, True)])
def test_a_served_shape_is_said_for_a_request_that_generated_something_and_did_not_reach_its_token_limit(tmp_path, shape_names, t, said):
    lp, served, row_t = shape_names
    _put(tmp_path / "match_shape.json", {lp: 8, served: t, row_t: 3})
    assert R.served_shape(str(tmp_path)) == (f"LP=8 T_r={t} {row_t}=3" if said else "")


def _tp_workload(tmp_path: Path, args: dict[str, Any], **top: Any) -> str:
    sweep = {"batch": 2, "max_input": 8, "max_output": 4, "input_lengths": [8, 5], "output_caps": [4, 3]}
    return _put(tmp_path / "wl.json", {"sweep": sweep, "schema": "verity-vllm/workload/v2", "model": "m", "engine": {"revision": "0" * 40, "args": args}, **top})


def test_a_tp_rows_shape_reads_its_engine_lengths_where_stated_else_its_envelope_and_counts_its_requests(tmp_path, monkeypatch):
    """The token budget is the workload's where it is under ``B * MML``; the requests are the workload's list, else one per input length."""
    asked = []
    monkeypatch.setattr(R, "checkpoint_dir", lambda *a: asked.append(a) or "/ck")
    shapes = [(5, 2), (8, 3)]
    assert R.tp_shape(_tp_workload(tmp_path, {}), "cks.json", "hf") == {"B": 2, "I": 8, "O": 4, "requests": 2, "shapes": shapes, "MML": 12, "MNBT": 12, "ck": "/ck"}
    assert asked == [("m", "0" * 40, "hf", "cks.json")]
    wl = _tp_workload(tmp_path, {"max_model_len": 20, "max_num_batched_tokens": 10}, requests=[{}, {}, {}])
    assert R.tp_shape(wl, "cks.json", "hf") == {"B": 2, "I": 8, "O": 4, "requests": 3, "shapes": shapes, "MML": 20, "MNBT": 10, "ck": "/ck"}
    monkeypatch.setattr(R, "checkpoint_dir", lambda *a: None)
    assert R.tp_shape(wl, "cks.json", "hf")["ck"] == ""


def test_a_tp_build_summary_fails_on_any_ranks_refused_derive_or_absent_workload_and_states_each_whole(tmp_path):
    b = tmp_path / "build"
    _put(b / "rank0" / "build_step" / "result.json", {"ok": True, "program": {"digest": "p" * 64}})
    _put(b / "rank0" / "build_request_LP8_T3" / "result.json", {"ok": False, "unsupported": "op", "message": "m" * 310})
    _put(b / "rank0" / "build_workload" / "workload_program.json", {"workload_digest": "w" * 64})
    _put(b / "rank1" / "build_workload" / "workload_program.json", {"reason": "refused", "message": "no compose"})
    assert R.tp_build_summary(str(b), 2) == "FAIL"
    absent = {"ok": False, "unsupported": None, "message": "no result.json", "program_digest": None, "census_digest": None, "construction": "trace"}
    assert _read(tmp_path / "build_summary.json") == {"tp": {"world": 2}, "step": {"program_digest": "p" * 64}, "ranks": {
        "rank0": {"build_step": {**absent, "ok": True, "message": "", "program_digest": "p" * 64},
                  "build_request_LP8_T3": {**absent, "unsupported": "op", "message": "m" * 300},
                  "workload": {"ok": True, "workload_digest": "w" * 64, "reason": None, "message": None, "mechanism": "GP-01 cb_a.global_program"}},
        "rank1": {"build_step": absent,
                  "workload": {"ok": False, "workload_digest": None, "reason": "refused", "message": "no compose", "mechanism": "GP-01 cb_a.global_program"}}}}
    _put(b / "rank0" / "build_request_LP8_T3" / "result.json", {"ok": True})
    for r in (0, 1):
        _put(b / f"rank{r}" / "build_step" / "result.json", {"ok": True})
    assert R.tp_build_summary(str(b), 2) == "FAIL"   # rank 1's workload alone
    _put(b / "rank1" / "build_workload" / "workload_program.json", {"workload_digest": "v" * 64})
    assert R.tp_build_summary(str(b), 2) == "PASS"
    _put(b / "rank1" / "build_step" / "result.json", {"ok": False})
    assert R.tp_build_summary(str(b), 2) == "FAIL"   # rank 1's step alone
    assert R.tp_rank_workload_digest(str(b), 1) == "v" * 64
    _put(b / "rank1" / "build_workload" / "workload_program.json", {"workload_digest": None})
    assert R.tp_rank_workload_digest(str(b), 1) == "" and R.tp_rank_workload_digest(str(b), 2) == ""


def test_a_tp_manifest_that_is_absent_or_not_complete_is_not_of_record_for_that_reason(tmp_path):
    d, every = str(tmp_path), SimpleNamespace(norm_tap="1", router_tap="1", vocab_tap="1")
    assert R.tp_manifest_not_of_record(d, every) == "absent or not complete"
    for doc in ([1], {"complete": False, "query": {"query_id": R.QUERY_OF_RECORD}}, {"complete": "True"}):
        _put(tmp_path / "manifest.json", doc)
        assert R.tp_manifest_not_of_record(d, every) == "absent or not complete", doc


def test_a_tp_rows_check_and_fold_lines_state_each_field_of_their_documents(tmp_path):
    m = str(tmp_path)
    assert R.tp_check_line(m) == "" and R.tp_fold_line(m) == "" and R.tp_served_shapes(m) == []
    rank0 = {"n_collectives": 11, "build_signature_len": 12, "n_steps": 13}
    check = {"pass": True, "structure": {"pass": False, "per_rank": {"0": rank0}}, "provenance": {"pass": True, "collectives_checked": 14, "n_mismatches": 15}}
    _put(tmp_path / "tp2_match.json", {**check, "tokens": {"pass": False}})
    line = "pass True structure False n_coll_r0 11 sig_len 12 steps 13 provenance True 14 mismatches 15 tokens False"
    assert R.tp_check_line(m) == line
    _put(tmp_path / "tp2_match.json", {**check, "tokens": {}})
    assert R.tp_check_line(m) == line.replace("tokens False", "tokens None")
    _put(tmp_path / "fold_match" / "tp2_fold_match.json", {"pass": False, "legs": {"a": 1}, "problems": ["p1", "p2", "p3", "p4"]})
    assert R.tp_fold_line(m) == 'pass False legs {"a": 1} problems ["p1", "p2", "p3"]'
    requests = [{"prompt_token_ids": [1, 2, 3], "generated_token_ids": [4, 5]}, {"prompt_token_ids": [1], "generated_token_ids": [7, 8, 9]}]
    _put(tmp_path / "capture" / "tokens.json", {"requests": requests})
    assert R.tp_served_shapes(m) == ["3,1", "1,2"]


@pytest.mark.parametrize("passed", [R.tp_passed, lambda p: R.tp_commit_passed(os.path.dirname(os.path.dirname(p)))])
def test_a_tp_document_passed_only_where_it_states_pass_true(tmp_path, passed):
    p = tmp_path / "commit" / "summary.json"
    assert passed(str(p)) is False
    for doc, said in (({"pass": True}, True), ({"pass": False}, False), ({"pass": "False"}, False), ({}, False)):
        _put(p, doc)
        assert passed(str(p)) is said, doc


def test_a_tp_match_summary_states_its_stages_exits_and_the_blocks_of_each_document_it_read(tmp_path):
    m = tmp_path / "match"
    R.tp_match_summary(str(m), 1, 2, 3, -1, "FAIL", "")
    assert _read(tmp_path / "match_summary.json") == {
        "pass": False, "level": "collective-level (tp.match check; no TP rank profile)", "capture_rc": 1, "control_rc": 2, "check_rc": 3,
        "fold_rc": None, "fold_profile": None, "tp2_match": {}, "tp2_fold_match": None}
    check = {"pass": True, "structure": {"pass": True}, "provenance": {"pass": False}, "tokens": {"pass": True}}
    _put(m / "tp2_match.json", {**check, "other": 1})
    _put(m / "fold_match" / "tp2_fold_match.json", {"pass": True, "legs": [1], "problems": [], "seconds": 9, "other": 1})
    for fold_rc in (0, 4):
        R.tp_match_summary(str(m), 0, 0, 0, fold_rc, "PASS", "prof")
        assert _read(tmp_path / "match_summary.json") == {
            "pass": True, "level": "fold (D90 per-rank fold Match + collective-level)", "capture_rc": 0, "control_rc": 0, "check_rc": 0,
            "fold_rc": fold_rc, "fold_profile": "prof", "tp2_match": check, "tp2_fold_match": {"pass": True, "legs": [1], "problems": [], "seconds": 9}}, fold_rc
    _put(m / "tp2_match.json", {"pass": False, "tokens": {}})
    R.tp_match_summary(str(m), 0, 0, 0, 0, "FAIL", "prof")
    assert _read(tmp_path / "match_summary.json")["tp2_match"] == {"pass": False, "tokens": {}}   # a block it does not state is left out


def test_a_tp_commit_line_states_its_pairs_delta_roots_and_tokens(tmp_path):
    roots = ["r" * 100]
    _put(tmp_path / "commit" / "summary.json", {"pass": False, "pairs": 3, "delta_wall_mean": 0.5, "tp_run_roots": roots, "tokens_equal": True})
    assert R.tp_commit_line(str(tmp_path)) == (f"pass False pairs 3 delta 0.5 roots {json.dumps(roots)[:80]} tokens True replay None linkage None xrank None"
                                               " fold_binding None weights_pin None")
