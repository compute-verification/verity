"""What a row's stages read back from the files beside the row, for the readers no other test runs: the lines of ``stages.txt``
(`verity_vllm.pipeline.row_records`: `build_line`, `match_line`, `commit_line`), the Match and Build records' fields
(`missing_derived`, `fold_steps`, `served_shape`, `snapshot_steps_of_record`, `program_digest_of_record`, `manifest_verify_ok`),
the workload's (`is_stochastic`, `tp_world`), the run store's job (`verity_vllm.pipeline.research`: `job_input`, `preserve`), a
manifest build that fails (`verity_vllm.pipeline.row_build.required_manifest`), the launch contexts a workload's shapes are built
under (`verity_vllm.pipeline.launch_context.main`) and the card's line for a stage that is not ok
(`verity_vllm.pipeline.report.summarize_md`).

Each reads the document a stage wrote and gives a word, a line or a default; a document that is absent or does not parse gives the
default its docstring names, never an error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.launch_context as LC
import verity_vllm.pipeline.report as CARD
import verity_vllm.pipeline.research as RS
import verity_vllm.pipeline.row_build as RB
import verity_vllm.pipeline.row_records as R
from verity_vllm.program import prefix_plan as PP


def _put(path: Path, doc: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(path)


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


@pytest.mark.parametrize(("shape", "said"), [
    ({"LP": 8, "T": 3, "cap_T": 7}, "LP=8 T_r=3 cap_T=7"),
    ({"LP": "8", "T": "3", "cap_T": "7"}, "LP=8 T_r=3 cap_T=7"),
    ({"LP": 8, "T": 7, "cap_T": 7}, ""),
    ({"LP": 8, "T": 0, "cap_T": 7}, ""),
    ({"LP": 8, "T": 3}, ""),
    ("{", ""),
])
def test_the_served_shape_is_said_when_a_request_stopped_before_its_cap(tmp_path, shape, said):
    assert R.served_shape(str(tmp_path)) == ""
    _put(tmp_path / "match_shape.json", shape)
    assert R.served_shape(str(tmp_path)) == said


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


def test_a_jobs_input_is_the_artifact_its_run_recorded(tmp_path):
    assert RS.job_input(None, "build") == "" and RS.job_input(str(tmp_path), "build") == ""
    _put(tmp_path / "job.json", {"store": {"inputs": {"build": "art:b", "match": ""}}})
    assert RS.job_input(str(tmp_path), "build") == "art:b"
    assert RS.job_input(str(tmp_path), "match") == "" and RS.job_input(str(tmp_path), "commit") == ""
    unreadable: tuple[Any, ...] = ({}, {"store": None}, [], "{")
    for doc in unreadable:
        _put(tmp_path / "job.json", doc)
        assert RS.job_input(str(tmp_path), "build") == ""


def test_a_run_is_put_on_the_preserve_list_once(tmp_path, capsys):
    path = tmp_path / "guard" / "preserve.txt"
    RS.preserve(str(path), "run-a")
    RS.preserve(str(path), "run-b")
    RS.preserve(str(path), "run-a")
    assert path.read_text() == "run-a\nrun-b\n"
    RS.preserve(str(tmp_path / "guard" / "preserve.txt" / "under-a-file"), "run-c")
    assert "not updated" in capsys.readouterr().err and path.read_text() == "run-a\nrun-b\n"


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
    wl = _put(tmp_path / "w.json", {"sweep": {"input_lengths": [8, 5], "output_caps": [3, 2]}})
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
