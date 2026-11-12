"""What a row's stages read back from the files beside the row, for the readers no other test runs: the lines of ``stages.txt``
(`verity_vllm.pipeline.row_records`: `build_line`, `match_line`, `commit_line`), a row's shape when it samples (`row_shape`),
the Match and Build records' fields
(`missing_derived`, `fold_steps`, `served_shape`, `snapshot_steps_of_record`, `program_digest_of_record`, `manifest_verify_ok`),
the workload's (`is_stochastic`, `tp_world`), a
manifest build that fails (`verity_vllm.pipeline.row_build.required_manifest`), the launch contexts a workload's shapes are built
under (`verity_vllm.pipeline.launch_context`: `main`, `batch_dispatch_of`, `bad_chunks`), the card's line for a stage that is not ok
(`verity_vllm.pipeline.report.summarize_md`) and its reading of a log's first record (`log_header`), the config record of a row that
ran nothing (`verity_vllm.pipeline.config_record.record`), the timeline report's window and its command, which report a file's
last attempt (`verity_vllm.pipeline.timeline`: `report`, `main`), the
experiment index's entries for the captured values and the query-use document (`verity_vllm.pipeline.experiment`), the command
a stage's end declares its outputs with (`verity_vllm.pipeline.research_outputs.main`) and the one that writes the rows' digests
(`verity_vllm.pipeline.epoch`: `main` for `digests`).

Each reads the document a stage wrote and gives a word, a line or a default; a document that is absent or does not parse gives the
default its docstring names, never an error.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.config_record as CR
import verity_vllm.pipeline.epoch as EP
import verity_vllm.pipeline.experiment as EXP
import verity_vllm.pipeline.launch_context as LC
import verity_vllm.pipeline.report as CARD
import verity_vllm.pipeline.research_outputs as RO
import verity_vllm.pipeline.row_build as RB
import verity_vllm.pipeline.row_records as R
import verity_vllm.pipeline.timeline as TL
from tests.pipeline.test_row import STAGE_START, _summary_row
from verity_vllm.config import CHECKPOINTS, WORKLOADS
from verity_vllm.program import prefix_plan as PP


def _put(path: Path, doc: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(path)


def test_a_row_that_samples_takes_its_first_requests_shape_and_snapshots_every_step(tmp_path):
    """A greedy row's default shape is its envelope and it snapshots two steps.  A row that samples takes the shape of its first
    declared request and snapshots every step; with no declared request it keeps the envelope and the two steps."""
    wl = WORKLOADS / "llama32-1b__bf16__l40s__tp1__b8__i1024__o128__mixed__stoch-t0.8-p0.95__bi-eager.json"
    w = json.loads(wl.read_text())
    first, longest = w["requests"][0], max(w["sweep"]["input_lengths"])
    shape = R.row_shape(str(wl), str(CHECKPOINTS), str(tmp_path))
    assert (shape["LP"], shape["T"], shape["snap"]) == (first["prompt_len"], first["max_tokens"] - 1, "all") and shape["LP"] != longest
    assert (shape["B"], shape["I"], shape["O"]) == (8, 1024, 128)
    bare = R.row_shape(_put(tmp_path / "bare.json", {**w, "requests": []}), str(CHECKPOINTS), str(tmp_path))
    assert (bare["LP"], bare["snap"]) == (longest, "0,1") and bare["T"] > shape["T"] and bare["B"] == 8
    unmeasured = {**first, "prompt_len": None, "prompt_token_ids": [1, 2, 3]}
    counted = R.row_shape(_put(tmp_path / "counted.json", {**w, "requests": [unmeasured]}), str(CHECKPOINTS), str(tmp_path))
    assert (counted["LP"], counted["T"], counted["snap"]) == (3, shape["T"], "all")


def test_the_build_line_is_the_step_digest_and_the_manifest(tmp_path):
    d = str(tmp_path)
    _put(tmp_path / "build_summary.json", {"step": {"program_digest": "ab" * 32}})
    assert R.build_line(d) == ("ab" * 8, "none")
    _put(tmp_path / "manifest.json", {"manifest_digest": "cd" * 32, "populations": {"identities": 7}})
    assert R.build_line(d) == ("ab" * 8, f"{'cd' * 8} 7")
    _put(tmp_path / "manifest.json", "{not json")
    assert R.build_line(d) == ("ab" * 8, "none")
    _put(tmp_path / "build_summary.json", {"step": {}})
    assert R.build_line(d)[0] == ""


def test_the_shapes_the_compare_names_as_not_derived(tmp_path):
    _put(tmp_path / "match_decomp.json", {"verdict": "MISSING_DERIVED", "missing_derived": [[8, 3], ["9", "4", "2"]]})
    assert R.missing_derived(str(tmp_path)) == [PP.shape(8, 3), PP.shape(9, 4, 2)]
    assert R.decomp_rc(str(tmp_path)) == 4


def test_a_fold_gives_its_decode_steps_or_the_default(tmp_path):
    assert R.fold_steps(str(tmp_path), 6) == 6
    _put(tmp_path / "match" / "fold_summary.json", {"n_steps": 5})
    assert R.fold_steps(str(tmp_path), 6) == 4
    _put(tmp_path / "match" / "fold_summary.json", {})
    assert R.fold_steps(str(tmp_path), 6) == -1
    _put(tmp_path / "match" / "fold_summary.json", "{")
    assert R.fold_steps(str(tmp_path), 6) == 6


@pytest.mark.parametrize(("workload", "stochastic"), [
    ({}, False),
    ({"sampling": {"variant": "greedy"}, "variants": {"greedy": {"temperature": 0.0}}}, False),
    ({"sampling": {"variant": "topp"}, "variants": {"greedy": {"temperature": 0.0}, "topp": {"temperature": 0.7}}}, True),
    ({"sampling": {"variant": "topp"}, "variants": {"greedy": {"temperature": 0.7}}}, False),
])
def test_a_workload_is_stochastic_when_its_variant_has_a_temperature(tmp_path, workload, stochastic):
    assert R.is_stochastic(_put(tmp_path / "w.json", workload)) is stochastic


def test_the_match_line_reads_the_summary(tmp_path):
    summary = {"compare_verdict": "PASS", "control_vs_instrumented_tokens_equal": True, "fold_ok": None,
               "global_match": {"verdict": "FAIL", "failed_checks": ["G1", "G2"], "achieved_concurrency": {"observed_max": 3}}}
    _put(tmp_path / "match_summary.json", summary)
    assert R.match_line(str(tmp_path)) == "verdict PASS global FAIL ['G1', 'G2']  concurrency 3 tokens_equal True fold None"
    _put(tmp_path / "match_summary.json", {**summary, "global_match": None})
    assert R.match_line(str(tmp_path)) == "verdict PASS global None   concurrency None tokens_equal True fold None"


def test_a_stochastic_rows_match_summary_holds_its_sampler_value_check(tmp_path):
    d = _summary_row(tmp_path, old=False, stage_record=True)

    def summary(rcs: int) -> dict[str, Any]:
        R.match_summary(str(d), 0, 0, 0, STAGE_START, rcs, False)
        return json.loads((d / "match_summary.json").read_text())

    assert summary(-1)["stoch_value_check"] is None, "a greedy row has none"
    absent = summary(0)
    assert absent["stoch_value_check"]["verdict"] == "NOT-RUN" and absent["pass"] is False
    assert absent["compare_verdict"] == "FAIL stoch value check NOT-RUN: "
    _put(d / "match_stoch.json", {"summary": {"verdict": "PASS", "rows": 3}, "reasons": [], "variant": "temperature"})
    passed = summary(0)
    assert passed["stoch_value_check"] == {"rc": 0, "verdict": "PASS", "rows": 3, "device_facts_all_match": None,
                                           "masked_rows_all_equal": None, "reasons": [], "variant": "temperature"}
    assert passed["compare_verdict"] == "PASS" and passed["pass"] is True


@pytest.mark.parametrize("shape", [{"LP": 8, "T": 3}, {"LP": 8}, [8, 3], "{"])
def test_no_served_shape_is_said_without_a_whole_record_of_it(tmp_path, shape):
    assert R.served_shape(str(tmp_path)) == ""
    _put(tmp_path / "match_shape.json", shape)
    assert R.served_shape(str(tmp_path)) == ""


def test_a_batch_rows_digest_of_record_is_the_workload_programs(tmp_path):
    d = str(tmp_path)
    _put(tmp_path / "build_summary.json", {"request": {"program_digest": "r" * 64}, "build_request_LP8_T3": {"program_digest": "a" * 64}})
    assert R.program_digest_of_record(d, 1) == "r" * 64
    folded = R.program_digest_of_record(d, 2)
    assert len(folded) == 64 and folded != "a" * 64
    _put(tmp_path / "build_workload" / "workload_program.json", {"program_digest": "p" * 64, "digest": "g" * 64})
    assert R.program_digest_of_record(d, 2) == "p" * 64
    _put(tmp_path / "build_workload" / "workload_program.json", {"workload_digest": "w" * 64, "program_digest": "p" * 64})
    assert R.program_digest_of_record(d, 2) == "w" * 64
    _put(tmp_path / "build_workload" / "workload_program.json", {"reason": "not composed"})
    assert R.program_digest_of_record(d, 2) == folded


def test_the_snapshot_steps_of_record_are_the_match_runs(tmp_path):
    assert R.snapshot_steps_of_record(str(tmp_path)) == ""
    _put(tmp_path / "match" / "run.json", {"snapshot_steps": [0, 1, 5]})
    assert R.snapshot_steps_of_record(str(tmp_path)) == "0,1,5"
    _put(tmp_path / "match" / "run.json", {"snapshot_steps": None})
    assert R.snapshot_steps_of_record(str(tmp_path)) == ""
    _put(tmp_path / "match" / "run.json", [0, 1])
    assert R.snapshot_steps_of_record(str(tmp_path)) == ""


def test_the_commit_line_names_the_runs_and_each_check_that_is_no_rule(tmp_path):
    assert R.commit_line(str(tmp_path)) == ""
    _put(tmp_path / "verdict.json", {"runs": {"instrumented": 3, "failed": 1},
                                    "checks": [{"name": "C2", "outcome": "pass"}, {"name": "rule-7", "outcome": "fail", "rule": "R7"},
                                               {"name": "C5", "outcome": None, "rule": None}]})
    assert R.commit_line(str(tmp_path)) == "runs 3 failed 1 checks C2=pass C5=None"
    _put(tmp_path / "verdict.json", {})
    assert R.commit_line(str(tmp_path)) == "runs None failed None checks "
    _put(tmp_path / "verdict.json", "{")
    assert R.commit_line(str(tmp_path)) == ""


def test_the_manifest_verify_word(tmp_path):
    assert R.manifest_verify_ok(str(tmp_path)) == "not-run"
    _put(tmp_path / "commit" / "manifest_verify.json", {"ok": False})
    assert R.manifest_verify_ok(str(tmp_path)) == "False"
    _put(tmp_path / "commit" / "manifest_verify.json", {})
    assert R.manifest_verify_ok(str(tmp_path)) == "None"


@pytest.mark.parametrize(("workload", "world"), [({"sweep": {"tp": 8}}, 8), ({"sweep": {"tp": "4"}}, 4), ({"sweep": {}}, 2), ({}, 2),
                                                 ([], 2), ("{", 2), ({"sweep": {"tp": "two"}}, 2)])
def test_a_tensor_parallel_rows_world_is_the_workloads_or_two(tmp_path, workload, world):
    assert R.tp_world(str(tmp_path / "absent.json")) == 2
    assert R.tp_world(_put(tmp_path / "w.json", workload)) == world


#: `manifest build` as far as the row sees it: it writes a manifest, says SAYS and ends with RC
_MANIFEST_CHILD = """
import sys
a = sys.argv[1:]
open(a[a.index("--out") + 1], "w").write("{}")
print("built 3 components")
print(sys.argv[-1])
sys.exit(int(sys.argv[-2]))
"""


@pytest.mark.parametrize(("child_rc", "says", "rc", "ends", "logged"), [
    (0, "ok", 0, [("rc", 0)], "required-value manifest of record"),
    (1, "PRECHECK_FAIL_REGISTRY_UNAVAILABLE: no registry at /r", 3, [("rc", 3), ("fail_class", "PRECHECK_FAIL_REGISTRY_UNAVAILABLE")],
     "PRECHECK_FAIL_REGISTRY_UNAVAILABLE: PRECHECK_FAIL_REGISTRY_UNAVAILABLE: no registry at /r"),
    (2, "a rule refused it", 1, [("rc", 1)], "required-value manifest NOT built (rc=2; see manifest.log): a rule refused it"),
])
def test_a_manifest_build_that_fails_keeps_no_manifest_and_says_why(tmp_path, child_rc, says, rc, ends, logged):
    marks: list[tuple[Any, ...]] = []
    lines: list[str] = []
    row = SimpleNamespace(ck_=None, B=1, d=str(tmp_path), wl="w.json", tree=str(tmp_path), child_env=None, log=lines.append,
                          tl=lambda *a: marks.append(a),
                          cli=lambda *a: [sys.executable, "-c", _MANIFEST_CHILD, *a, str(child_rc), says])
    assert RB.required_manifest(row, "build") == rc
    assert (tmp_path / "manifest.json").exists() is (rc == 0)
    assert marks[0] == ("build.required_manifest", "start", ("query", R.QUERY_OF_RECORD))
    assert marks[1:] == [("build.required_manifest", "end", *ends)]
    assert len(lines) == 1 and logged in lines[0]


def test_the_launch_context_command_prints_each_shapes_context(tmp_path, capsys):
    wl = _put(tmp_path / "w.json", {"requests": [{"prompt_len": 8, "max_tokens": 3}, {"prompt_len": 5, "max_tokens": 2}]})
    assert LC.main(LC.Options(workload=wl)) == 0
    assert capsys.readouterr().out == "LP5_T1 [8,1]\nLP8_T2 [8,1,1]\n"
    assert LC.main(LC.Options(workload=wl, LP=8, T=2)) == 0
    assert capsys.readouterr().out == "[8,1,1]\n"
    assert LC.main(LC.Options(workload=wl, LP=7, T=2)) == 0
    assert capsys.readouterr().out == "\n"


def test_the_card_names_each_stage_that_is_not_ok():
    def card(stages: dict[str, dict[str, Any]]) -> str:
        return CARD.summarize_md(CARD.ReportCard(schema="s", utc="u", config_id="c", model={}, C={}, engine_args={}, workload={}, kappa={},
                                                 gates={}, verdict="REJECTED", stages=stages))
    md = card({"build": {"status": "failed", "log": "build.log"}, "match": {"status": "ok"}, "commit": {}, "check": {"status": "skipped"}})
    assert "- stages not ok: build (failed, build.log), check (skipped, None)\n" in md
    assert "stages not ok" not in card({"build": {"status": "ok"}, "commit": {}})


def test_the_config_record_of_a_row_that_ran_nothing_fails_at_its_build(tmp_path):
    rec = CR.record(tmp_path, "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager")
    assert (rec["schema"], rec["outcome"], rec["why"]) == ("verity-vllm/config-record/v1", "FAIL", "build FAIL")


def test_a_row_dispatches_by_batch_when_its_target_is_not_batch_invariant_and_it_has_several_requests(tmp_path):
    several = [{"prompt_len": 8, "max_tokens": 3}, {"prompt_len": 5, "max_tokens": 2}]
    varies = {"engine": {"target": {"batch_invariant": False}}}
    assert LC.batch_dispatch_of(None) is False and LC.batch_dispatch_of("") is False
    assert LC.batch_dispatch_of(_put(tmp_path / "several.json", {**varies, "requests": several})) is True
    assert LC.batch_dispatch_of(_put(tmp_path / "one.json", {**varies, "requests": several[:1]})) is False
    assert LC.batch_dispatch_of(_put(tmp_path / "invariant.json", {"requests": several})) is False


def test_prefill_chunks_are_a_requests_prompt_rows_one_a_step_before_its_decodes():
    launch = {"chunks": [5, 3], "batch_by_step": [0] * 5}
    for none in (None, {}, {"batch_by_step": [1]}, {"chunks": [], "batch_by_step": [1]}):
        assert LC.bad_chunks(none, "request", 8, 3, 0) is None
    assert LC.bad_chunks(launch, "request", 8, 3, 0) is None
    refused = LC.bad_chunks(launch, "request", 9, 3, 0)
    assert refused is not None and refused["reason"] == "launch_context" and refused["phase"] == "applicability"
    assert refused["hint"].startswith("prefill chunks [5, 3] at a launch batch of 5 steps") and refused["hint"].endswith("this is a request with C=0")
    for wrapper, T, C in (("step", 3, 0), ("request", 3, 2), ("request", 4, 0)):
        assert LC.bad_chunks(launch, wrapper, 8, T, C) is not None


def test_the_timeline_reports_its_window_or_none_without_a_timed_event():
    start = {"t": 10.0, "proc": "row", "pid": 1, "stage": "build", "ev": "start"}
    rep = TL.report([start, {**start, "t": 12.5, "ev": "end"}])
    assert (rep["schema"], rep["events"], rep["spans"], rep["elapsed_s"]) == ("verity-sweep/timeline/v1/report", 2, 1, 2.5)
    assert (rep["t_first_utc"], rep["t_last_utc"]) == ("1970-01-01T00:00:10Z", "1970-01-01T00:00:12Z")
    for events in ([], [{"proc": "row", "pid": 1, "stage": "build", "ev": "mark"}]):
        rep = TL.report(events)
        assert (rep["events"], rep["spans"], rep["elapsed_s"], rep["t_first_utc"], rep["t_last_utc"]) == (len(events), 0, None, None, None)


def test_the_timeline_report_command_reports_the_files_last_attempt(tmp_path, capsys):
    def events(t: float, pid: int) -> list[dict[str, Any]]:
        row = {"t": t, "proc": "row_pod", "pid": pid, "stage": "row", "ev": "start"}
        build = {"t": t + 1, "proc": "build", "pid": pid + 1, "stage": "build.derive", "ev": "start"}
        return [row, build, {**build, "t": t + 4, "ev": "end"}]

    path = tmp_path / "timeline.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in [*events(100.0, 1), *events(500.0, 7)]))
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    assert TL.main(TL.ReportOptions(path=str(path), json=str(out), md=str(md), title="a-row")) == 0
    rep = json.loads(out.read_text())
    assert (rep["attempt"], rep["attempts_in_file"], rep["events"], rep["spans"]) == ("last", 2, 3, 1)
    assert (rep["t_first_utc"], rep["t_last_utc"], rep["elapsed_s"]) == ("1970-01-01T00:08:20Z", "1970-01-01T00:08:24Z", 4.0)
    assert capsys.readouterr().out == md.read_text() and "a-row" in md.read_text() and "attempt last of 2 in the file" in md.read_text()


def test_the_experiment_index_counts_the_captured_values_and_their_steps(tmp_path):
    absent = EXP._transcript_values(tmp_path)
    assert absent["snapshot_index"] == {"path": "capture/snapshots.json", "present": False} and "snapshot_steps" not in absent
    entries = [{"step": 2, "path": "a"}, {"step": "0", "skipped": "oversize", "path": "b", "shape": [1]}, {"path": "c"}, "not an entry", {"step": 2}]
    for doc in ({"snapshots": entries}, {"entries": entries}, entries):
        _put(tmp_path / "capture" / "snapshots.json", doc)
        values = EXP._transcript_values(tmp_path)
        assert (values["tensors_captured"], values["tensors_skipped_oversize"], values["snapshot_steps"]) == (4, 1, [0, 2])
        assert values["skipped_examples"] == [{"path": "b", "shape": [1], "reason": "oversize"}]


def test_the_experiment_index_lifts_the_query_use_documents_verdict():
    assert EXP.vu_partition_index(None, None) == {"present": False, "gate": None}
    gates = {"gates": {"G8": {"status": "pass", "reason": "r", "detail": {"binding": {"checked": True}}}}}
    assert EXP.vu_partition_index(None, gates) == {"present": False, "gate": {"status": "pass", "reason": "r"}}
    verdict = {"partition": {"status": "ok", "codes": ["c1"]}, "count": 5, "status": "pass",
               "width": {"status": "ok", "w_max": 32, "max_seen": 16, "violators": [1, 2], "unsupported": None}}
    assert EXP.vu_partition_index({"schema": "s", "binding_id": "b", "use": {"kind": "k", "w_max": 32}, "verdict": verdict}, gates) == {
        "present": True, "schema": "s", "binding_id": "b", "use": {"kind": "k", "w_max": 32}, "partition": {"status": "ok", "codes": ["c1"]},
        "width": {"status": "ok", "w_max": 32, "max_seen": 16, "n_violators": 2, "n_unsupported": 0}, "count": 5, "status": "pass",
        "verdict_status": "pass", "binding": {"checked": True}, "gate": {"status": "pass", "reason": "r"}}
    empty: dict[str, Any] = {
        "present": True, "schema": None, "binding_id": None, "use": None, "partition": {"status": None, "codes": []},
        "width": {"status": None, "w_max": None, "max_seen": None, "n_violators": 0, "n_unsupported": 0}, "count": None, "status": None,
        "verdict_status": None, "binding": {"checked": False}, "gate": None}
    assert EXP.vu_partition_index({"verdict": "not a verdict"}, {}) == empty
    assert EXP.vu_partition_index({"verdict": {"partition": [], "width": None, "count": 1}}, None) == {**empty, "count": 1}


def test_the_card_reads_a_logs_first_record_when_it_is_the_run_header(tmp_path):
    header = {"$e": "RunHeader", "model": "m"}
    lines = json.dumps(header) + "\n" + json.dumps({"$e": "Step"}) + "\n"
    plain = _put(tmp_path / "log.jsonl", lines)
    packed = tmp_path / "log.jsonl.gz"
    packed.write_bytes(gzip.compress(lines.encode()))
    assert CARD.log_header(plain) == header and CARD.log_header(packed) == header
    assert CARD.log_header(tmp_path / "absent.jsonl.gz") is None
    (tmp_path / "a-directory.jsonl").mkdir()
    assert CARD.log_header(tmp_path / "a-directory.jsonl") is None


@pytest.mark.parametrize("first", [b"", b"\n", b"not json\n", b'["RunHeader"]\n', b'"RunHeader"\n', b"7\n", b'{"$e": "Step"}\n', b"{}\n",
                                   b"\xff\xfe{}\n"])
def test_the_card_has_no_run_header_for_a_first_line_that_is_not_one(tmp_path, first: bytes):
    (tmp_path / "log.jsonl").write_bytes(first)
    (tmp_path / "log.jsonl.gz").write_bytes(gzip.compress(first))
    assert CARD.log_header(tmp_path / "log.jsonl") is None and CARD.log_header(tmp_path / "log.jsonl.gz") is None


def test_the_card_has_no_run_header_for_a_packed_log_that_does_not_unpack(tmp_path):
    whole = gzip.compress((json.dumps({"$e": "RunHeader", "pad": "x" * 4000}) + "\n").encode())
    packed = tmp_path / "log.jsonl.gz"
    for raw in (whole[: len(whole) // 2], b"plain text, no gzip\n", whole[:10] + b"\xff" * 40):    # cut short, not packed, a bad stream
        packed.write_bytes(raw)
        assert CARD.log_header(packed) is None
    packed.write_bytes(whole)
    assert CARD.log_header(packed) == {"$e": "RunHeader", "pad": "x" * 4000}


def test_the_outputs_command_prints_the_digest_of_record_and_declares_a_stages_outputs(tmp_path, capsys):
    """`research-outputs digest` prints the Build's Program digest of record, an empty line where there is no Build; `write` declares
    the stage's outputs in the run dir's outputs.json and writes nothing without a run dir.  A Commit that reported PASS without its
    decision document is refused by name and leaves no outputs.json, and the command still returns 0: the stage's own status stands."""
    row = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    d, run = tmp_path / row, tmp_path / "run"
    _put(d / "build_summary.json", {"request": {"program_digest": "a" * 64}, "step": {"program_digest": "b" * 64}})
    _put(d / "stages.txt", "build PASS 2026-09-23T00:00:00Z rc=0\n")
    assert RO.main(RO.DigestOptions(evidence_dir=str(d), row=row)) == 0
    assert capsys.readouterr().out == "a" * 64 + "\n"
    assert RO.main(RO.DigestOptions(evidence_dir=str(tmp_path / "absent"), row=row)) == 0
    assert capsys.readouterr().out == "\n"
    assert RO.main(RO.WriteOptions(evidence_dir=str(d), row=row, stage="build")) == 0
    assert "RESEARCH_RUN_DIR not set" in capsys.readouterr().err and not run.exists()
    _put(run / "job.json", {"store": {"inputs": {"match": "art:" + "1" * 64, "weights": "/a/path"}}})
    given = ["build=art:" + "2" * 64, "plan=art:short", "=art:" + "3" * 64]
    assert RO.main(RO.WriteOptions(evidence_dir=str(d), row=row, stage="build", run_dir=str(run), input=given)) == 0
    doc = json.loads((run / "outputs.json").read_text())
    assert doc["inputs"] == {"build": "art:" + "2" * 64, "match": "art:" + "1" * 64}
    assert [(e["name"], e["kind"]) for e in doc["outputs"]] == [("build", "vllm-build/v1"), ("evidence", "execution-evidence/v1")]
    assert (run / "outputs" / "build" / "build_summary.json").is_file() and not (run / "outputs.json.tmp").exists()
    assert "build=vllm-build/v1(1 files), evidence=execution-evidence/v1(1 files)" in capsys.readouterr().out
    (run / "outputs.json").unlink()
    _put(d / "stages.txt", "commit PASS 2026-09-23T00:00:00Z rc=0\n")
    assert RO.main(RO.WriteOptions(evidence_dir=str(d), row=row, stage="commit", run_dir=str(run))) == 0
    assert "FAILED to write outputs.json: ValueError: commit stage PASS but its decision document verdict.json" in capsys.readouterr().err
    assert not (run / "outputs.json").exists()


def test_the_digests_command_writes_each_rows_digests_and_their_table(tmp_path, capsys):
    """`epoch digests` prints the document of the rows' digests; with `--out` the document goes to that file and the table is
    printed, and `--markdown` keeps the table too.  A row with no Program or no manifest digest is in the document with what it
    has, and the command returns 1."""
    d = tmp_path / "row-a"
    _put(d / "build_summary.json", {"step": {"program_digest": "s0"}, "request": {"program_digest": "p0"}, "workload": {"workload_digest": "w0"}})
    _put(d / "manifest.json", {"manifest_digest": "m0", "program_digest": "g0", "query": {
        "query_id": "q0", "partition": {"by_rank": {"1": {"object": "p", "digest": "pd1"}, "0": {"object": "p", "digest": "pd0"}}}}})
    _put(d / "verdict.json", {"outcome": "PASS", "run_roots": ["r" * 20, "t" * 4]})
    whole = {"row": "row-a", "programs": {"0": {"step": "s0", "workload": "w0", "requests": {"request": "p0"}}}, "manifest_digest": "m0",
             "program_digest": "g0", "query_id": "q0", "partition_digests": ["pd0", "pd1"], "run_roots": ["r" * 20, "t" * 4], "verdict": "PASS"}
    assert EP.main(EP.DigestsOptions(row_dirs=[str(d)])) == 0
    assert json.loads(capsys.readouterr().out) == {"schema": "verity-vllm/rebaseline-digests/v1", "rows": [whole]}
    empty: dict[str, Any] = {"row": "none", "programs": {}, "manifest_digest": None, "program_digest": None, "query_id": None,
                             "partition_digests": [], "run_roots": [], "verdict": None}
    out, md = tmp_path / "digests.json", tmp_path / "digests.md"
    assert EP.main(EP.DigestsOptions(row_dirs=[str(d), str(tmp_path / "none")], out=str(out), markdown=str(md))) == 1
    assert json.loads(out.read_text()) == {"schema": "verity-vllm/rebaseline-digests/v1", "rows": [whole, empty]}
    table = capsys.readouterr().out
    assert table == md.read_text() and table.splitlines()[2:] == [
        "| row-a | PASS | s0 | w0 | 1 | m0 | 2 | " + "r" * 16 + ", tttt |", "| none | - | - | - | 0 | - | 0 | - |"]
    (d / "manifest.json").unlink()
    assert EP.main(EP.DigestsOptions(row_dirs=[str(d)], out=str(out))) == 1
    assert json.loads(out.read_text())["rows"][0]["programs"] == whole["programs"] and not capsys.readouterr().err
