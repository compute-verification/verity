"""What a row's scripts ask the adapter and read from its output, for the answers no other CPU test runs in process: the epoch's
commands (`verity_vllm.pipeline.epoch`: `job`, `resolve`, `row-state`, `word-check`, `store-meta`, `custody-env`, `device`, with
`has_role`, `custody_exports` and the word check's reading of a log that shows an error), the row's result for the run store
(`verity_vllm.pipeline.research_result`: `build_result`, `main`), the run store's reading of a row's argv and of its interpreter's
pins (`verity_vllm.pipeline.research_tools`: `parse`, `_probe_pins`, `_research`), and a row started under a `$VERITY_TREE` that
names another tree (`verity_vllm.pipeline.row.run_row`).  CPU only: torch and vLLM are stood in, and no process starts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.epoch as EP
import verity_vllm.pipeline.research_result as RES
import verity_vllm.pipeline.research_tools as RT
import verity_vllm.pipeline.row as ROW_CMD
import verity_vllm.pipeline.row_records as R
from tests import fake_vllm
from tests.pipeline.test_row import ROW, _opts
from verity_vllm import config

WORD_LINE = "Q_word_v1{X=16,W=32,R=no-recompute}: 10 calls -> 20 units over 30 gates\n"
START = 1790640000                                   # 2026-09-29T00:00:00Z, in epoch seconds


def _put(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _built(d: Path, log: str = WORD_LINE) -> Path:
    """A row directory whose Build wrote a manifest under the query of record, with `log` as the manifest step's log."""
    _put(d / "manifest.json", {"manifest_digest": "m0", "query": {"query_id": R.QUERY_OF_RECORD, "partition": {"object": "p", "digest": "pd"},
                                                                  "required_families": ["attention", "norm"]}})
    (d / "manifest.log").write_text(log)
    _put(d / "build_summary.json", {"request": {"program_digest": "p0"}, "step": {"program_digest": "s0"}, "workload": {"workload_digest": "w0"}})
    return d


@pytest.fixture
def no_row_file(tmp_path, monkeypatch) -> Path:
    """The epoch's commands read a row's file from a place that holds none, until the test writes one there."""
    wl = tmp_path / "workloads" / "row.json"
    monkeypatch.setattr(EP, "row_file", lambda row: wl)
    return wl


# ---------------------------------------------------------------------------------------------------------------- epoch: a row's checkpoint

@pytest.mark.parametrize(("stated", "role", "serves"), [
    ("B2", "B2", True), ("B2", "B3", False), ("B1-B4", "B1", True), ("B1-B4", "B3", True), ("B1-B4", "B4", True), ("B1-B4", "B5", False),
    ("B2-B4", "B1", False), ("B1-4", "B4", True), ("B1-B4", "A2", False), ("A1, B1-B4", "B2", True), ("A1,B1-B4", "A1", True),
    ("A1, B1-B4", "A2", False), ("B1-B4", "LLAMA32_1B", False), ("LLAMA32_1B", "LLAMA32_1B", True), ("B1-B4", "B", False),
    (None, "B1", False), ("", "B1", False)])
def test_a_checkpoint_serves_a_role_it_names_or_one_inside_a_range_it_names(stated, role, serves):
    assert EP.has_role({"role": stated}, role) is serves
    assert EP.has_role({}, role) is False


def test_a_rows_checkpoint_is_the_one_of_its_role_else_the_local_one_whose_range_holds_it(tmp_path, monkeypatch, no_row_file, capsys):
    """The checkpoint that names the row's role is taken first; without one, the first that serves the role and has a local path.
    The world is the sweep's, else the row's own `tp`, else 2 for a row id that says `__tp2__` and 1 for any other."""
    monkeypatch.setattr(EP, "ROOT", tmp_path)
    ranged = [{"role": "A1-A4", "repo": "org/a", "revision": "ra", "local_path": "/a"}, {"role": "B1-B4", "repo": "org/far", "revision": "r0"},
              {"role": "B1-B4", "repo": "org/local", "revision": "r1", "local_path": "/b"},
              {"role": "B2-B3", "repo": "org/later", "revision": "r2", "local_path": "/c"}]
    _put(tmp_path / "manifests" / "checkpoints.json", {"checkpoints": ranged})
    _put(no_row_file, {"case": "B3", "sweep": {"tp": 4}})
    assert EP.resolve("a-row") == ("B3", "org/local", "r1", 4)
    assert EP.main(EP.ResolveOptions(row="a-row")) == 0 and capsys.readouterr().out == "B3 org/local r1 4\n"
    _put(tmp_path / "manifests" / "checkpoints.json", {"checkpoints": [*ranged, {"role": "B3", "repo": "org/exact", "revision": "r3"}]})
    assert EP.resolve("a-row") == ("B3", "org/exact", "r3", 4)
    for workload, row, world in (({"tp": "8"}, "a-row", 8), ({"sweep": {}}, "a__tp2__row", 2), ({"sweep": None}, "a-row", 1),
                                 ({"sweep": {"tp": 4}, "tp": 8}, "a__tp2__row", 4)):
        _put(no_row_file, {"case": "B3", **workload})
        assert EP.resolve(row)[3] == world, workload


# ---------------------------------------------------------------------------------------------------------------- epoch: the word check

@pytest.mark.parametrize("shown", ["Traceback (most recent call last):\n", "QueryRuleViolation: a unit over 32 bits\n", "RecomputeViolation\n"])
def test_a_manifest_log_that_shows_an_error_fails_the_word_check_with_every_line_it_needs(tmp_path, no_row_file, shown):
    rep = EP.word_check("a-row", _built(tmp_path, WORD_LINE + shown), 1, "", "", "")
    assert rep["problems"] == ["manifest.log shows a traceback or a query-rule violation"] and rep["ok"] is False
    assert (rep["word_lines"], rep["components"], rep["calls"], rep["units"], rep["gates"]) == (1, 1, 10, 20, 30)


def test_the_word_check_command_writes_its_report_says_it_in_one_line_and_exits_by_it(tmp_path, no_row_file, capsys):
    d, out = _built(tmp_path / "row"), tmp_path / "word.json"
    assert EP.main(EP.WordCheckOptions(row="a-row", row_dir=str(d), out=str(out))) == 0
    assert capsys.readouterr().out == "PASS fast: 1/1 word lines, 10 calls -> 20 units over 30 gates, manifest m0\n"
    rep = json.loads(out.read_text())
    assert out.read_text() == json.dumps(rep, indent=1) and rep["ok"] is True and rep["problems"] == [] and rep["row"] == "a-row"
    _put(d / "manifest.json", {"manifest_digest": "d" * 40, "query": {"query_id": R.QUERY_OF_RECORD, "partition": {"object": "p", "digest": "pd"}}})
    asked = EP.WordCheckOptions(row="a-row", row_dir=str(d), out=str(out), world=2, build_word_check="8/16", build_max_gates="9", allowed_max_gates="7")
    assert EP.main(asked) == 1
    rep = json.loads(out.read_text())
    assert capsys.readouterr().out == ("FAIL fast: 1/2 word lines, 10 calls -> 20 units over 30 gates, manifest " + "d" * 16 + ": "
                                       + "; ".join(rep["problems"]) + "\n")
    assert rep["ok"] is False and len(rep["problems"]) == 3 and rep["env"] == {"VERITY_WORD_CHECK": "8/16", "VERITY_QWORD_MAX_GATES": "9"}
    assert rep["problems"][2] == "VERITY_QWORD_MAX_GATES=9 in the Build's environment (allowed for this row: 7)"


# ---------------------------------------------------------------------------------------------------------------- epoch: the job's lines

def test_the_job_command_prints_the_runs_end_the_jobs_end_and_the_keys_expiry_with_a_dash_for_each_unknown(tmp_path, capsys):
    run, custody = tmp_path / "run", tmp_path / "custody"
    assert EP.main(EP.JobOptions(run_dir=str(run), custody_dir=str(custody))) == 0 and capsys.readouterr().out == "- - -\n"
    _put(run / "status.json", {"transitions": [{"state": "running", "started_utc": "2026-09-29T00:00:00.000Z", "timeout_s": 3600}]})
    assert EP.main(EP.JobOptions(run_dir=str(run), custody_dir=str(custody))) == 0
    assert capsys.readouterr().out == f"{START + 3600} {START + 3600 - 2400} -\n"                # the store's reserve, by default 2400 s
    _put(custody / "cred.json", {"expires_utc": "2026-09-29T02:00:00Z"})
    asked = EP.JobOptions(run_dir=str(run), custody_dir=str(custody), store_reserve_s=600)
    assert EP.main(asked) == 0 and capsys.readouterr().out == f"{START + 3600} {START + 3000} 2026-09-29T02:00:00Z\n"
    asked = EP.JobOptions(run_dir=str(run), custody_dir=str(custody), store_reserve_s=600, job_deadline="2026-09-29T00:10Z")
    assert EP.main(asked) == 0 and capsys.readouterr().out == f"{START + 3600} {START + 600} 2026-09-29T02:00:00Z\n"


def test_the_row_state_command_prints_the_families_the_outcome_and_the_manifest_check_with_a_dash_for_each_unknown(tmp_path, capsys):
    d, header = _built(tmp_path / "row"), tmp_path / "header.json"
    assert EP.main(EP.RowStateOptions(row_dir=str(tmp_path / "none"))) == 0 and capsys.readouterr().out == "- - -\n"
    assert EP.main(EP.RowStateOptions(row_dir=str(d))) == 0 and capsys.readouterr().out == "attention,norm - -\n" and not header.exists()
    _put(d / "verdict.json", {"outcome": "FAIL"})
    _put(d / "commit" / "manifest_verify.json", {"ok": False})
    assert EP.main(EP.RowStateOptions(row_dir=str(d), header=str(header))) == 0 and capsys.readouterr().out == "attention,norm FAIL False\n"
    assert json.loads(header.read_text())["manifest_digest"] == "m0"
    _put(d / "commit" / "manifest_verify.json", {"ok": True})
    _put(d / "manifest.json", {"query": {"required_families": []}})
    assert EP.main(EP.RowStateOptions(row_dir=str(d))) == 0 and capsys.readouterr().out == "- FAIL True\n"


def test_the_store_meta_command_prints_the_stored_trees_meta_as_one_document(tmp_path, capsys):
    d = _built(tmp_path / "a-row")
    _put(d / "verdict.json", {"outcome": "PASS"})
    asked = EP.StoreMetaOptions(row_dir=str(d), kind="build", role="B3", rownum=7, run_id="r1", epoch_sha="e" * 40)
    assert EP.main(asked) == 0
    printed = capsys.readouterr().out
    assert printed.endswith("}\n") and printed.count("\n") == 1 and json.loads(printed) == {
        "name": "epoch-build/a-row", "role": "B3", "row": 7, "row_id": "a-row", "epoch": "vllm-rebaseline-epoch",
        "program_digests": {"rank0/step": "s0", "rank0/request": "p0"}, "workload_digests": {"rank0": "w0"}, "manifest_digest": "m0",
        "query_id": R.QUERY_OF_RECORD, "verdict": "PASS", "build": {"gpu": None, "machine": None, "run": "r1", "source": "e" * 40, "stage": "build"}}
    named = EP.StoreMetaOptions(row_dir=str(d), kind="commit", role="B3", rownum=7, run_id="r1", epoch_sha="e" * 40, gpu="L40S", pod="p1", epoch="other")
    assert EP.main(named) == 0
    meta = json.loads(capsys.readouterr().out)
    assert (meta["name"], meta["epoch"], meta["build"]) == ("epoch-commit/a-row", "other", {"gpu": "L40S", "machine": "p1", "run": "r1",
                                                                                           "source": "e" * 40, "stage": "commit"})


def test_the_custody_lines_export_each_part_of_the_runs_key_that_is_stated_quoted_for_the_shell(tmp_path, capsys):
    """`custody-env` prints one `export` line a variable of the key, in the key's order, each value quoted as the shell reads it; a
    part the key leaves out or leaves empty has no line, and a directory with no key gives none."""
    first, second, third = EP.KEY_VARS
    assert EP.custody_exports(tmp_path) == []
    assert EP.main(EP.CustodyEnvOptions(custody_dir=str(tmp_path))) == 0 and capsys.readouterr().out == "\n"
    _put(tmp_path / "cred.json", {third: "it's a dummy", "expires_utc": "2026-09-29T02:00:00Z", second: "", first: "dummy"})
    lines = EP.custody_exports(tmp_path)
    assert lines == [f"export {first}=dummy", f"export {third}=" + "'it'\"'\"'s a dummy'"]
    assert EP.main(EP.CustodyEnvOptions(custody_dir=str(tmp_path))) == 0 and capsys.readouterr().out == "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------- epoch: the device

def _torch(monkeypatch, *, available: bool = True, sms: int = 142, cc: tuple[int, int] = (8, 9), count: int = 2) -> list[str]:
    """A stand-in torch whose CUDA has one kind of device; what it was asked, in order."""
    asked: list[str] = []
    props = SimpleNamespace(name="a device", multi_processor_count=sms, major=cc[0], minor=cc[1])

    def said(what: str, answer: Any) -> Any:
        def call(*a: Any) -> Any:
            asked.append(what if not a else f"{what} {a[0]}")
            return answer
        return call
    cuda = SimpleNamespace(init=said("init", None), is_available=said("is_available", available),
                           get_device_properties=said("properties", props), device_count=said("count", count))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return asked


def test_the_device_is_read_once_cuda_initialises_and_vllm_imports_and_is_the_expected_one_by_its_sms_and_capability(monkeypatch):
    fake_vllm.install(monkeypatch)
    asked = _torch(monkeypatch)
    dev = {"name": "a device", "sms": 142, "cc": "8.9", "count": 2}
    assert EP.device(None, "9.0") == (dev, True) and asked == ["init", "is_available", "properties 0", "count"]
    assert EP.device(142, "8.9") == (dev, True) and EP.device(141, "8.9") == (dev, False) and EP.device(142, "9.0") == (dev, False)


def test_the_device_raises_until_cuda_is_available_and_until_vllm_imports(monkeypatch):
    fake_vllm.install(monkeypatch)
    asked = _torch(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\) is False after init"):
        EP.device(None, "8.9")
    assert asked == ["init", "is_available"]
    asked = _torch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ImportError):
        EP.device(None, "8.9")
    assert asked == ["init", "is_available"]


def test_the_device_command_prints_the_device_and_exits_3_for_one_that_is_not_expected(monkeypatch, capsys):
    fake_vllm.install(monkeypatch)
    _torch(monkeypatch, sms=80, cc=(12, 0), count=1)
    dev = {"name": "a device", "sms": 80, "cc": "12.0", "count": 1}
    assert EP.main(EP.DeviceOptions()) == 0 and json.loads(capsys.readouterr().out) == dev
    assert EP.main(EP.DeviceOptions(expect_sms=80, expect_cc="12.0")) == 0 and json.loads(capsys.readouterr().out) == dev
    assert EP.main(EP.DeviceOptions(expect_sms=80)) == 3 and json.loads(capsys.readouterr().out) == dev      # 8.9 is expected by default
    assert EP.main(EP.DeviceOptions(expect_sms=142, expect_cc="12.0")) == 3 and json.loads(capsys.readouterr().out) == dev
    _torch(monkeypatch, sms=80, cc=(8, 9), count=1)
    assert EP.main(EP.DeviceOptions(expect_sms=80)) == 0 and json.loads(capsys.readouterr().out) == {**dev, "cc": "8.9"}


# ---------------------------------------------------------------------------------------------------------------- the row's result

def _result(d: Path, lines: list[str], stages: str = "build,match,commit") -> dict[str, Any]:
    (d / "stages.txt").write_text("".join(ln + "\n" for ln in lines))
    return RES.build_result(d, "a-row", "B3", stages.split(","), "r1", None)


def test_a_rows_result_is_failed_by_a_stage_or_a_precheck_that_did_not_pass_and_not_run_while_a_stage_it_asked_for_has_no_line(tmp_path):
    """Every stage asked for has a line: passed when each is a PASS, else failed.  Before that the result is failed once a stage that
    has a line, or the precheck, did not pass, and not run otherwise.  A stage's last line is the one read, and only Build, Match and
    Commit are stages of the result."""
    def status(lines: list[str], stages: str = "build,match,commit") -> str:
        return _result(tmp_path, lines, stages)["validation"]["status"]
    assert status([]) == status(["build PASS t ok"]) == status(["precheck PASS t ok", "build PASS t ok", "match PASS t ok"]) == "not_run"
    assert status(["build PASS t ok", "match FAIL t why"]) == status(["match INCOMPLETE t why"]) == "failed"
    assert status(["precheck FAIL t why"]) == status(["precheck FAIL t why", "build PASS t ok"]) == "failed"
    assert status(["build PASS t ok", "match PASS t ok", "commit PASS t ok"]) == "passed"
    assert status(["build PASS t ok", "match PASS t ok", "commit FAIL t why"]) == "failed"
    assert status(["build FAIL t why", "build PASS t ok"], "build") == "passed" and status(["build PASS t ok", "build FAIL t why"], "build") == "failed"
    assert status(["precheck FAIL t why", "build PASS t ok"], "build") == "passed"            # the stages asked for all passed
    doc = _result(tmp_path, ["precheck PASS t ok", "build PASS t ok", "replay FAIL t why"], "precheck,build,replay")
    assert doc["workload_fingerprint"]["stages"] == ["build"] and doc["validation"]["status"] == "passed"
    assert doc["validation"]["detail"] == "precheck PASS; build PASS; replay FAIL" and _result(tmp_path, [])["validation"]["detail"] == "no stages.txt"


def test_a_rows_result_reads_the_builds_program_digest_and_each_stages_wall_from_their_lines(tmp_path):
    digest = "0123456789abcdef" * 2
    doc = _result(tmp_path, [f"build PASS t digest={digest} wall=12s", "match PASS t wall=7s tokens equal", "commit PASS t no wall stated"])
    assert doc["workload_fingerprint"]["program_digest"] == digest
    assert doc["measurements"] == [{"name": "proofs/one-stage.build-wall", "value": 12, "unit": "s"},
                                 {"name": "proofs/one-stage.match-wall", "value": 7, "unit": "s"}]
    doc = _result(tmp_path, [f"build PASS t step digest {digest[:16]} of 3 wall=3s", f"match PASS t digest={digest} wall=7s"], "match")
    assert doc["workload_fingerprint"]["program_digest"] == digest[:16] and doc["measurements"] == [
        {"name": "proofs/one-stage.match-wall", "value": 7, "unit": "s"}]
    for line in ("build PASS t digest=0123456789abcde", "build PASS t wall=12", f"match PASS t digest={digest}"):
        doc = _result(tmp_path, [line])
        assert "program_digest" not in doc["workload_fingerprint"] and doc["measurements"] == [], line


def _asked(ev: Path, run_dir: Path | None) -> RES.Options:
    return RES.Options(evidence_dir=str(ev), row="a-row", role="B3", stages="build,match", run_dir=str(run_dir) if run_dir else None, run_id="r1")


def test_the_result_command_writes_the_result_under_the_run_and_never_fails_the_row(tmp_path, capsys):
    """Without a run directory nothing is written, and stderr says so.  With one, `result.json` is written whole (through a file
    beside it that does not stay) with the row's small records copied under `evidence/`, and stdout names it with its status.  A
    result that cannot be written is said on stderr; the command returns 0 each time, since the row's exit is its own."""
    ev, run = tmp_path / "ev", tmp_path / "run"
    ev.mkdir()
    (ev / "stages.txt").write_text("build PASS t ok\nmatch PASS t ok\n")
    assert RES.main(_asked(ev, None)) == 0
    assert capsys.readouterr() == ("", "research_result: RESEARCH_RUN_DIR not set; nothing written\n") and not run.exists()
    assert RES.main(_asked(ev, run)) == 0
    assert capsys.readouterr() == (f"research_result: {run / 'result.json'} validation=passed\n", "")
    doc = json.loads((run / "result.json").read_text())
    assert (run / "result.json").read_text() == json.dumps(doc, indent=1, sort_keys=True) and sorted(p.name for p in run.iterdir()) == ["evidence", "result.json"]
    assert (doc["run_id"], doc["validation"]["status"], doc["artifacts"]) == ("r1", "passed", ["evidence/stages.txt"])
    assert doc["workload_fingerprint"] == {"row": "a-row", "role": "B3", "stages": ["build", "match"], "evidence_dir": str(ev)}
    assert (run / "evidence" / "stages.txt").read_text() == "build PASS t ok\nmatch PASS t ok\n"
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the run directory would be")
    assert RES.main(_asked(ev, blocked)) == 0
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("research_result: FAILED to write result.json: NotADirectoryError: ") and err.count("\n") == 1


# ---------------------------------------------------------------------------------------------------------------- the run store's reading

def test_an_argv_that_names_no_runner_is_read_as_the_rows_own_arguments(tmp_path):
    """Without `verity-vllm row`, the module's `row` or an earlier runner in it, the argv is the row's arguments from its first
    word, and the stage is the Tool's."""
    got = RT.parse([ROW, "B3", "org/m", "rev", "--lp", "8"], stage="match")
    assert (got["stage"], got["row"], got["role"], got["model"], got["revision"], got["lp"]) == ("match", ROW, "B3", "org/m", "rev", "8")
    assert (got["model_tag"], got["tp"], got["batch"], got["query"]) == ("llama32-1b", 1, 1, RT.DEFAULT_QUERY)
    assert RT.parse(["python", "-m", "some.module", "row", "stage", "commit", ROW], stage="build")["row"] == "python"
    bare = RT.parse([], stage="build")
    assert (bare["stage"], bare["row"], bare["role"], bare["model_tag"], bare["query"]) == ("build", None, None, None, RT.DEFAULT_QUERY)


def test_a_rows_flags_are_read_with_their_values_their_equals_signs_and_their_bare_switches():
    """`--k v`, `--k=v` and a bare `--k` (true) are each read, dashes as underscores; a value that is itself a list of flags is a
    value, and a word between flags that belongs to none is kept as a positional.  A flag the stage keys only when set is in the
    reading when it is on, and any other flag the row took is recorded as it was given."""
    argv = ["verity-vllm", "row", "stage", "commit", ROW, "B3", "org/m", "rev", "--pairs=5", "--commit-args", "--bounded-staging --retain-exclude m1",
            "--replay-deferred", "--gpu-util", "0.5", "stray", "--some-new-flag=a=b", "--last"]
    got = RT.parse(argv, stage="commit")
    assert (got["stage"], got["pairs"], got["commit_args"], got["replay_deferred"]) == ("commit", "5", "--bounded-staging --retain-exclude m1", True)
    assert (got["gpu_util"], got["_positional"], got["some_new_flag"], got["last"]) == ("0.5", ["stray"], "a=b", True)
    assert (got["required_classes"], got["replay_args"], got["extra_engine_args"]) == (None, None, None)
    for off in ([], ["--replay-deferred", "0"], ["--replay-deferred=0"]):
        assert "replay_deferred" not in RT.parse([*argv[:8], *off], stage="commit"), off
    assert RT.parse([*argv[:8], "--replay-deferred", "1"], stage="commit")["replay_deferred"] == "1"
    built = RT.parse(["verity-vllm", "row", "stage", "build", ROW, "B3", "org/m", "rev", "--build-programs-only", "--replay-deferred"], stage="build")
    assert built["build_programs_only"] is True and built["replay_deferred"] is True and "pairs" not in built


def test_the_interpreters_pins_are_what_it_prints_of_torch_and_vllm_and_none_where_it_is_absent_or_does_not_answer(tmp_path, monkeypatch):
    """The pins are read from the workload's interpreter (`$PY`, else the pod's), asked once for both versions with a bound on its
    time.  An interpreter that is not there, cannot be started or does not answer in time pins neither, and a line that names
    another distribution or no version pins nothing."""
    py = tmp_path / "python"
    ran: list[tuple[list[str], dict[str, Any]]] = []
    answer: list[Any] = [""]

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
        ran.append((cmd, kw))
        if isinstance(answer[0], Exception):
            raise answer[0]
        return subprocess.CompletedProcess(cmd, 0, stdout=answer[0], stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("PY", str(py))
    assert RT._probe_pins() == {"torch": None, "vllm": None, "py": str(py)} and not ran
    py.write_text("")
    answer[0] = "torch 2.6.0+cu124\nvllm   0.9.1.dev3+gabc1234  \n"
    assert RT._probe_pins() == {"torch": "2.6.0+cu124", "vllm": "0.9.1.dev3+gabc1234", "py": str(py)}
    (cmd, kw), = ran
    assert cmd[:2] == [str(py), "-c"] and "importlib.metadata" in cmd[2] and kw == {"capture_output": True, "text": True, "timeout": 30}
    answer[0] = "torch \nnumpy 2.1\nvllm\n\ntorch 2.6 extra words\n"
    assert RT._probe_pins() == {"torch": "2.6 extra words", "vllm": None, "py": str(py)}
    for failed in (OSError("cannot start"), subprocess.TimeoutExpired([str(py)], 30)):
        answer[0] = failed
        assert RT._probe_pins() == {"torch": None, "vllm": None, "py": str(py)}, failed
    ran.clear()
    monkeypatch.setattr(RT, "DEFAULT_PY", str(tmp_path / "the-pods-python"))
    for unset in ("", None):
        monkeypatch.setenv("PY", unset) if unset is not None else monkeypatch.delenv("PY")
        assert RT._probe_pins() == {"torch": None, "vllm": None, "py": str(tmp_path / "the-pods-python")} and not ran, unset


def test_a_run_store_module_that_is_missing_is_never_read_as_none(monkeypatch):
    """`_research` hands the missing module to `config.run_store`, which refuses by the reader's name; were that refusal ever to
    return, the import's own error is raised, so no reader gets a module that is not there."""
    monkeypatch.setitem(sys.modules, "research_stand_in_that_is_absent", None)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(config, "run_store", lambda module, entry: asked.append((module, entry)))
    with pytest.raises(ModuleNotFoundError):
        RT._research("research_stand_in_that_is_absent", "CLOSURE")
    assert asked == [("research_stand_in_that_is_absent", "research_tools.CLOSURE")]
    assert RT._research("json", "CLOSURE") is json and len(asked) == 1


# ---------------------------------------------------------------------------------------------------------------- a row refused at its start

def test_a_row_started_under_a_verity_tree_that_names_another_tree_is_refused_before_it_writes_anything(tmp_path, capsys):
    sweep = tmp_path / "sweep"
    o = _opts(sweep, environ={"VERITY_TREE": str(tmp_path)})
    assert ROW_CMD.run_row(o, "build") == 3
    out, err = capsys.readouterr()
    assert out == "" and err == (f"PRECHECK_FAIL_SOURCE_IDENTITY: row at {os.path.realpath(config.ROOT)} but $VERITY_TREE={tmp_path} "
                                 "names another tree\n")
    assert not sweep.exists()
