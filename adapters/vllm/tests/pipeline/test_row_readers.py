"""What a row's stages read back from the files beside the row, for the readers no other test runs: the lines of ``stages.txt``
(`verity_vllm.pipeline.row_records`: `build_line`, `match_line`, `commit_line`), a row's shape when it samples (`row_shape`),
the Match and Build records' fields
(`missing_derived`, `fold_steps`, `served_shape`, `snapshot_steps_of_record`, `program_digest_of_record`, `manifest_verify_ok`),
the workload's (`is_stochastic`, `tp_world`), a
manifest build that fails (`verity_vllm.pipeline.row_build.required_manifest`), the launch contexts a workload's shapes are built
under (`verity_vllm.pipeline.launch_context`: `main`, `batch_dispatch_of`, `bad_chunks`), the card's line for a stage that is not ok
(`verity_vllm.pipeline.report.summarize_md`), the gate whose detail each of its counts reads (`build_card`) and its reading of a
log's first record (`log_header`), the line `verdict from-record` prints (`verity_vllm.pipeline.verdict_record.main`), the config
record of a row that
ran nothing (`verity_vllm.pipeline.config_record.record`), the timeline report's window and its command, which report a file's
last attempt (`verity_vllm.pipeline.timeline`: `report`, `main`), the
experiment index's entries for the captured values and the query-use document (`verity_vllm.pipeline.experiment`), the command
a stage's end declares its outputs with (`verity_vllm.pipeline.research_outputs.main`) and the one that writes the rows' digests
(`verity_vllm.pipeline.epoch`: `main` for `digests`).  The last tests are of a field its document may leave out, null or empty:
a TP Build's rank blocks, a Commit's decision document, a job's inputs (`research_outputs`), a run's source identity
(`verity_vllm.pipeline.research_result.build_result`), the replay bundles a Commit's verdict names
(`verity_vllm.pipeline.row_config`: `commit_verdict`, `replay`), and what `verity_vllm.pipeline.epoch` reads of a run's status, a
row file, a manifest, a Build's summary and a verdict.

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
import verity_vllm.pipeline.research_result as RES
import verity_vllm.pipeline.row_build as RB
import verity_vllm.pipeline.row_config as RCFG
import verity_vllm.pipeline.row_records as R
import verity_vllm.pipeline.timeline as TL
from tests.check.test_verdict import _base, _row_dir
from tests.pipeline.test_row import STAGE_START, _summary_row
from verity_vllm.check.gates import GateResult
from verity_vllm.config import CHECKPOINTS, WORKLOADS
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.row_driver import Stop
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


def test_the_card_reads_each_gates_detail_under_that_gates_id(tmp_path):
    """The ambiguous count is G2's, tier a and the quarantine are G4's, and the sample is G5's: a detail under another id is not read."""
    sample = {"status": "pass", "k": 2, "strata": 4, "seed": 9}
    detail: dict[str, dict[str, Any]] = {"G2": {"ambiguous": 7}, "G4": {"k": 3, "quarantine": {"specs": 1}}, "G5": {"sample": sample}}
    other = {"ambiguous": 99, "k": 99, "quarantine": {"specs": 99}, "sample": {"status": "fail"}}
    card = CARD.build_card(tmp_path, [GateResult(id=g, name=g, status="pass", detail=detail.get(g, other))
                                      for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")])
    assert card.counts["ambiguous"] == 7 and card.counts["sampled_vus"] == "k=2 x 4 strata, seed 9 PASS"
    assert (card.replay["tier_a"]["k"], card.replay["quarantine"], card.replay["sample"]) == (3, {"specs": 1}, sample)


def test_the_verdict_command_says_the_outcome_first_and_takes_the_commits_exit_code(tmp_path, capsys):
    """The row driver's Commit stage reads the first word of the one line `verdict from-record` prints.  Evidence that passes
    under a Commit that exited nonzero is not a PASS, and without `--out` the record goes beside the row."""
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    for rc, outcome in (("0", "PASS"), ("1", "INSUFFICIENT_EVIDENCE")):
        out = tmp_path / f"rc{rc}.json"
        assert cli.main(["verdict", "from-record", str(d), "--rc", rc, "--out", str(out)]) == 0
        said = capsys.readouterr().out.splitlines()
        assert len(said) == 1 and said[0].startswith(f"{outcome} {out} [") and json.loads(out.read_text())["outcome"] == outcome
    assert cli.main(["verdict", "from-record", str(d)]) == 0 and capsys.readouterr().out.startswith(f"PASS {d / 'verdict.json'} [")


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
    engines: list[dict[str, Any] | None] = [None, {}, {"target": None}, {"target": {}}, {"target": {"batch_invariant": None}},
                                            {"target": {"batch_invariant": True}}]
    for engine in engines:
        assert LC.batch_dispatch({"engine": engine, "requests": several}) is False, engine


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


def test_a_tp_builds_digests_read_a_summary_whose_rank_blocks_are_not_all_there(tmp_path):
    """A rank whose block, or whose `workload`, is absent, null or empty has no workload digest, and then the row has no digest of
    record.  Rank 0's `build_step` may be left out the same way, and then there is no step digest."""
    row = "llama32-1b__bf16__l40s__tp2__b8__i256__o32__mixed__greedy__bi-eager"
    whole = {"build_step": {"program_digest": "s" * 64}, "workload": {"workload_digest": "w" * 64}}

    def digests(ranks: dict[str, Any]) -> dict[str, Any]:
        _put(tmp_path / "build_summary.json", {"tp": {"world": 2}, "ranks": ranks})
        return RO.program_digest(tmp_path, row)
    of_record = digests({"rank0": whole, "rank1": whole})
    assert of_record["program_digest"] and (of_record["step_digest"], of_record["rank_workload_digests"]) == ("s" * 64, ["w" * 64] * 2)
    left_out: list[dict[str, Any]] = [{}, {"rank1": None}, {"rank1": {}}, {"rank1": {"workload": None}}, {"rank1": {"workload": {}}}]
    for rank1 in left_out:
        got = digests({"rank0": whole, **rank1})
        assert (got["program_digest"], got["step_digest"], got["rank_workload_digests"]) == (None, "s" * 64, ["w" * 64, None]), rank1
    left_out = [{}, {"build_step": None}, {"build_step": {}}]
    for step in left_out:
        got = digests({"rank0": {"workload": whole["workload"], **step}, "rank1": whole})
        assert (got["program_digest"], got["step_digest"]) == (of_record["program_digest"], None), step
    left_out = [{}, {"rank0": None}]
    for rank0 in left_out:
        got = digests({"rank1": whole, **rank0})
        assert (got["program_digest"], got["step_digest"], got["rank_workload_digests"]) == (None, None, [None, "w" * 64]), rank0


def test_a_commits_verdict_entry_reads_a_decision_document_without_its_replay_or_its_runs(tmp_path):
    """A config record's `replay` and a verdict's `runs` may be absent, null or empty: the entry then states no replay and no runs."""
    row, run = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager", tmp_path / "run"

    def meta(d: Path, stages: str, rel: str, doc: dict[str, Any]) -> dict[str, Any]:
        _put(d / "stages.txt", stages)
        _put(d / rel, doc)
        return RO.build_outputs(d, row, "commit", run, {})["outputs"][0]["meta"]
    config, verdict = "config PASS 2026-09-23T00:00:00Z\ncommit PASS 2026-09-23T00:00:01Z rc=0\n", "commit PASS 2026-09-23T00:00:00Z rc=0\n"
    left_out: list[dict[str, Any]] = [{}, {"replay": None}, {"replay": {}}]
    for replay in left_out:
        got = meta(tmp_path / "config", config, "config_record.json", {"outcome": "PASS", "why": "w", **replay})
        assert (got["outcome"], got["why"], got["replay_equal"], got["replay_k"]) == ("PASS", "w", None, None), replay
    got = meta(tmp_path / "config", config, "config_record.json", {"outcome": "PASS", "replay": {"equal": True, "k": 64}})
    assert (got["decision_record"], got["replay_equal"], got["replay_k"]) == ("config_record.json", True, 64)
    left_out = [{}, {"runs": None}, {"runs": {}}]
    for runs in left_out:
        got = meta(tmp_path / "tp1", verdict, "verdict.json", {"outcome": "PASS", **runs})
        assert (got["outcome"], got["n_runs"], got["runs_failed"]) == ("PASS", None, None), runs
    got = meta(tmp_path / "tp1", verdict, "verdict.json", {"outcome": "PASS", "runs": {"instrumented": 1, "control": 1, "failed": 0}})
    assert (got["decision_record"], got["n_runs"], got["runs_failed"]) == ("verdict.json", 2, 0)


def test_a_jobs_inputs_are_the_artifacts_its_store_names(tmp_path):
    """A job.json that is absent or no document, or whose `store` or its `inputs` is absent, null or empty, names no input."""
    assert RO.inputs_from_job(tmp_path) == {}
    left_out: list[Any] = [{}, {"store": None}, {"store": {}}, {"store": {"inputs": None}}, {"store": {"inputs": {}}}, ["not a job"], "{"]
    for job in left_out:
        _put(tmp_path / "job.json", job)
        assert RO.inputs_from_job(tmp_path) == {}, job
    _put(tmp_path / "job.json", {"store": {"inputs": {"build": "art:" + "1" * 64, "weights": "/a/path", "n": 3}}})
    assert RO.inputs_from_job(tmp_path) == {"build": "art:" + "1" * 64}


def test_a_results_tree_is_the_first_one_its_source_identity_states(tmp_path):
    """The tree of a run's result is the first `git_sha` its `source_identity.json` states.  One whose `identities` is absent, null or
    empty states no tree, and one that does not parse is said to be unreadable."""
    def fingerprint(doc: Any) -> dict[str, Any]:
        _put(tmp_path / "source_identity.json", doc)
        return RES.build_result(tmp_path, "row", "role", ["build"], "r1", None)["workload_fingerprint"]
    stated = fingerprint({"ok": True, "requested_sha": "ab", "identities": {"a": "not a dict", "b": {"git_sha": ""}, "c": {"git_sha": "cd"}}})
    assert (stated["tree_sha"], stated["source_identity"]) == ("cd", {"ok": True, "requested_sha": "ab"})
    left_out: list[dict[str, Any]] = [{}, {"identities": None}, {"identities": {}}, {"identities": {"a": {"path": "/p"}}}]
    for identities in left_out:
        got = fingerprint({"ok": True, **identities})
        assert (got["tree_sha"], got["source_identity"]) == (None, {"ok": True}), identities
    assert fingerprint("{")["source_identity"] == "unreadable"


@pytest.mark.parametrize("verdict", [None, {}, {"replay_deferred": None}, {"replay_deferred": {}}, {"replay_deferred": {"bundles": None}},
                                     {"replay_deferred": {"bundles": []}}])
def test_a_commit_verdict_that_names_no_replay_bundle_is_not_pending_and_leaves_nothing_to_replay(tmp_path, verdict):
    """`commit/verdict.json`, its `replay_deferred` or the `bundles` under it may be absent, null or empty, and each reads as no bundle:
    a Commit that decided nothing is then a FAIL, not PENDING, and the replay stage refuses by the count."""
    said: list[tuple[Any, ...]] = []
    row = SimpleNamespace(d=str(tmp_path), config_run=True, ck="", verdict=lambda *a: said.append(a), tl=lambda *a: None, log=lambda *a: None)
    _put(tmp_path / "commit" / "summary.json", {"commit_pass": None})
    if verdict is not None:
        _put(tmp_path / "commit" / "verdict.json", verdict)
    with pytest.raises(Stop) as stop:
        RCFG.commit_verdict(row, 0, 60)
    assert stop.value.rc == 12 and said[-1][:2] == ("commit", "FAIL")
    with pytest.raises(Stop):
        RCFG.replay(row)
    assert said[-1][:2] == ("replay", "FAIL") and "0 replay bundles named in commit/verdict.json" in said[-1][2]


def test_a_runs_end_and_a_rows_tp_world_read_a_status_and_a_row_file_without_their_optional_blocks(tmp_path, monkeypatch):
    """A status.json whose `transitions` is absent, null or empty, or holds no running one, has no end.  A row file states its TP world
    under `sweep`; one whose `sweep` is absent, null or empty states it as `tp`, else in its id, else it is 1."""
    left_out: list[dict[str, Any]] = [{}, {"transitions": None}, {"transitions": []}, {"transitions": [{"state": "queued"}]}]
    for status in left_out:
        _put(tmp_path / "run" / "status.json", status)
        assert EP.run_end(tmp_path / "run") is None, status
    _put(tmp_path / "run" / "status.json", {"transitions": [{"state": "running", "started_utc": "2026-09-29T00:00:00Z", "timeout_s": 60}]})
    assert EP.run_end(tmp_path / "run") == 1790640000 + 60
    role = json.loads(CHECKPOINTS.read_text())["checkpoints"][0]["role"]
    monkeypatch.setenv("VERITY_WORKLOADS", str(tmp_path / "rows"))

    def world(row: str, doc: dict[str, Any]) -> int:
        _put(tmp_path / "rows" / f"{row}.json", {"case": role, **doc})
        return EP.resolve(row)[3]
    assert world("a-row-outside__tp2__the-tree", {"sweep": {"tp": 4}, "tp": 3}) == 4
    left_out = [{}, {"sweep": None}, {"sweep": {}}, {"sweep": {"batch": 8}}, {"sweep": {"tp": None}}]
    for sweep in left_out:
        assert world("a-row-outside__tp2__the-tree", {"tp": 3, **sweep}) == 3, sweep
        assert world("a-row-outside__tp2__the-tree", sweep) == 2, sweep
        assert world("a-row-outside-the-tree", sweep) == 1, sweep


def test_the_word_check_and_the_row_state_read_a_manifest_without_its_query_and_a_row_file_without_its_batch(tmp_path, monkeypatch):
    """A manifest whose `query` is absent, null or empty requires no family, is not the query of record and names no partition.  The
    word check wants one Q_word line a component: the row's `sweep.batch` times the world, else its count of `requests`, else one."""
    monkeypatch.setenv("VERITY_WORKLOADS", str(tmp_path / "rows"))
    d = tmp_path / "row"
    line = "Q_word_v1{X=16,W=32,R=no-recompute}: 10 calls -> 20 units over 30 gates\n"
    _put(d / "manifest.log", line * 3)
    assert EP.row_state(d, None) == ([], None, None)
    left_out: list[dict[str, Any]] = [{"manifest_digest": "m"}, {"query": None}, {"query": {}}, {"query": {"required_families": None}},
                                      {"query": {"required_families": []}}]
    for man in left_out:
        _put(d / "manifest.json", man)
        assert EP.row_state(d, None)[0] == [], man
        got = EP.word_check("a-row-outside-the-tree", d, 1, "", "", "")
        assert (got["query_id"], got["partition"], got["ok"], len(got["problems"])) == (None, False, False, 2), man
        assert f"manifest query None is not the query of record {got['query_of_record']}" in got["problems"], man
        assert "the manifest names no partition (query.partition)" in got["problems"], man
    of_record = EP.word_check("a-row-outside-the-tree", d, 1, "", "", "")["query_of_record"]
    _put(d / "manifest.json", {"query": {"query_id": of_record, "partition": {"object": "p", "digest": "pd"}, "required_families": ("f1", "f2")}})
    assert EP.row_state(d, None)[0] == ["f1", "f2"]

    def components(doc: dict[str, Any], world: int = 1) -> tuple[int, bool]:
        _put(tmp_path / "rows" / "a-row-outside-the-tree.json", doc)
        got = EP.word_check("a-row-outside-the-tree", d, world, "", "", "")
        return got["components"], got["ok"]
    three = [{"prompt_len": 4, "max_tokens": 2}] * 3
    assert components({"sweep": {"batch": 2}, "requests": three}) == (2, True) and components({"sweep": {"batch": 2}}, 2) == (4, False)
    left_out = [{}, {"sweep": None}, {"sweep": {}}, {"sweep": {"tp": 2}}, {"sweep": {"batch": None}}]
    for sweep in left_out:
        assert components({"requests": three, **sweep}) == (3, True), sweep
        none: list[dict[str, Any]] = [{}, {"requests": None}, {"requests": []}]
        for requests in none:
            assert components({**sweep, **requests}) == (1, True), (sweep, requests)


def test_a_rows_digests_read_a_build_a_manifest_and_a_verdict_with_blocks_left_out(tmp_path):
    """A Build summary's `ranks` that is null is a TP1 summary and one that is empty has no rank; a rank's step digest is its `step`'s,
    else its `build_step`'s, and a block that is absent, null or empty has no digest.  A partition's `by_component` or `by_rank`, a
    verdict's `run_roots` and a manifest's `query` may each be absent, null or empty; the table's cells for a row with no rank 0 are dashes."""
    d = tmp_path / "row"

    def programs(summary: dict[str, Any]) -> dict[str, Any]:
        _put(d / "build_summary.json", summary)
        return EP.build_programs(d)
    none: dict[str, Any] = {"step": None, "workload": None, "requests": {}}
    assert programs({"ranks": None, "step": {"program_digest": "s"}}) == {"0": {**none, "step": "s"}} and programs({"ranks": {}}) == {}
    left_out: list[dict[str, Any]] = [{"x": 1}, {"step": None, "build_step": None, "workload": None}, {"step": {}, "build_step": {}, "workload": {}}]
    for rank in left_out:
        assert programs({"ranks": {"rank0": rank}}) == {"0": none}, rank
    assert programs({"ranks": {"rank0": {"step": {}, "build_step": {"program_digest": "b"}, "workload": {"workload_digest": "w"}}}}) == {
        "0": {"step": "b", "workload": "w", "requests": {}}}
    _put(d / "build_summary.json", {})
    left_out = [{}, {"query": None}, {"query": {}}, {"query": {"partition": None}}, {"query": {"partition": {}}},
                {"query": {"partition": {"by_component": None, "by_rank": {}}}}]
    for man in left_out:
        _put(d / "manifest.json", man)
        got = EP.row_digests(d)
        assert (got["query_id"], got["partition_digests"], got["programs"]) == (None, [], {}), man
        assert EP.store_meta(d, kind="build", role="R", rownum=1, run_id="r", epoch_sha="x", gpu="", pod="", epoch="e")["query_id"] is None, man
    part = {"by_component": {"c": {"object": "p", "digest": "pc"}}, "by_rank": {"0": {"by_component": {"c": {"object": "p", "digest": "pa"}}}}}
    _put(d / "manifest.json", {"query": {"query_id": "q", "partition": part}})
    assert (EP.row_digests(d)["query_id"], EP.row_digests(d)["partition_digests"]) == ("q", ["pa", "pc"])
    assert EP.store_meta(d, kind="build", role="R", rownum=1, run_id="r", epoch_sha="x", gpu="", pod="", epoch="e")["query_id"] == "q"
    left_out = [{"outcome": "PASS"}, {"outcome": "PASS", "run_roots": None}, {"outcome": "PASS", "run_roots": []}]
    for verdict in left_out:
        _put(d / "verdict.json", verdict)
        assert (EP.row_digests(d)["run_roots"], EP.row_digests(d)["verdict"]) == ([], "PASS"), verdict
    _put(d / "verdict.json", {"run_roots": ("r1", "r2")})
    assert EP.row_digests(d)["run_roots"] == ["r1", "r2"]
    row: dict[str, Any] = {"row": "a", "verdict": None, "manifest_digest": None, "partition_digests": [], "run_roots": []}
    dashes = "| a | - | - | - | 0 | - | 0 | - |"
    left_out = [{}, {"0": None}, {"0": {}}, {"0": {"step": None, "workload": None, "requests": None}}, {"0": {"requests": {}}}]
    for rank0 in left_out:
        assert EP.digests_markdown([{**row, "programs": rank0}]).splitlines()[2] == dashes, rank0
    assert EP.digests_markdown([{**row, "programs": {"0": {"step": "s", "workload": "w", "requests": {"request": "p", "build_request_LP8": "q"}}}}]
                               ).splitlines()[2] == "| a | - | s | w | 2 | - | 0 | - |"

