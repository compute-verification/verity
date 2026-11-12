"""A row's setup and precheck past the tree's own checks (`Row.setup`, `prepare`, `run`): its stages' environment, the cuBLAS
names, the release line, the target-family preflight, the Commit's session checks. A row's stages run with the run store's tool
(`tools/research/src`) on their path, so these run only where it is beside the tree; `test_row` holds a tree without it to the
refusal at setup, by name.

Then what the run store brings a row's stages (`pipeline/research.py`'s readers): a job's recorded inputs, an artifact's meta, an
Attempt's outputs, tool, status and source, each for a document that leaves the block out or states it as null or empty.  The
research CLI is a stand-in that answers ``data show`` with the document a test gives it."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline.test_cublas_names import _node
from tests.pipeline.test_row import (
    GLM53,
    ROW,
    TP_DENSE,
    TP_MOE,
    _row,
    _timeline,
    _tp_row,
)
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import research as RS
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.records.schema_ids import TIMELINE_SCHEMA

TREE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------------------------- the stages' environment

@pytest.mark.parametrize("row, execution, bi", [(ROW, "bi-eager", "1"), ("glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", "eager", "0")])
def test_a_rows_stages_run_batch_invariant_exactly_when_its_execution_is(tmp_path, monkeypatch, row, execution, bi):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    r = _row(tmp_path, row=row)
    r.setup({})
    assert (r.execution(), r.child_env["VLLM_BATCH_INVARIANT"], r.bi_eager()) == (execution, bi, execution == "bi-eager")


@pytest.mark.parametrize("doc, label", [
    ({"execution": "eager", "sweep": {"execution": "compiled"}}, "eager"),      # the file's own label first
    ({"execution": "", "sweep": {"execution": "compiled"}}, "compiled"),        # then its sweep block's
    ({"sweep": {"execution": "compiled"}}, "compiled"),
    ({"sweep": {}}, "bi-eager"),                                                # then the id's last token
    ({"sweep": ["execution"]}, "bi-eager"),                                     # a sweep that is no block names none
    ({"sweep": None}, "bi-eager"),
    ({}, "bi-eager"),
    (["execution"], None),                                                      # a file that is no document has no label
])
def test_a_rows_execution_is_its_files_label_then_its_sweep_blocks_then_its_ids_last_token(tmp_path, doc, label):
    r = _row(tmp_path)
    r.wl = str(tmp_path / "workload.json")
    Path(r.wl).write_text(json.dumps(doc))
    assert r.execution() == label
    os.remove(r.wl)
    assert r.execution() is None
    Path(r.wl).write_text("{")                                                  # nor has a file that does not parse
    assert r.execution() is None


def test_a_row_whose_shape_is_not_read_yet_names_no_target_to_its_derives(tmp_path):
    """Before `shape` reads the row's declared target, a derive is built for the device: no ``--target`` and the plain limit."""
    r = _row(tmp_path)
    r.MML = 288
    cmd = r.request_cmd(8, 1, "out")
    assert cmd[-2:] == ["--out", "out"] and "--target" not in cmd and r.derive_timeout(256, 31) == 900


def test_remote_verifier_reaches_the_single_gpu_commit_which_checks_its_session_before_anything_runs(tmp_path):
    """REMOTE_VERIFIER reaches the single-GPU Commit as ``--remote-verifier``, and a TP row refuses it; the Commit
    refuses a bad address, a session of more than one mode or a Commit with no instrumented run to audit before anything
    runs (`store.remote.CommitSession`)."""
    from verity_vllm.pipeline import commit

    single = _row(tmp_path, environ={"REMOTE_VERIFIER": "10.0.0.7:7781"})
    single.hidden_so, single.B = "/x/verity_fa2_matReqG.so", 1
    cmd = single.commit_cmd([])
    assert cmd[cmd.index("--remote-verifier") + 1] == "10.0.0.7:7781"
    assert "--remote-verifier" not in _tp_row(tmp_path, TP_DENSE).commit_cmd([])
    with pytest.raises(SystemExit, match="REMOTE_VERIFIER"):
        _tp_row(tmp_path, TP_DENSE, environ={"REMOTE_VERIFIER": "10.0.0.7:7781"}).commit_cmd([])
    args = cli.parse("commit", cmd[cmd.index("commit") + 1:], {})
    with pytest.raises(ValueError, match="one mode"):
        commit.main(dataclasses.replace(args, remote_plan="serve,servedrop:1"))
    with pytest.raises(ValueError, match="HOST:PORT"):
        commit.main(dataclasses.replace(args, remote_verifier="7781"))
    with pytest.raises(SystemExit, match="first instrumented run"):
        commit.main(dataclasses.replace(args, only_arm="control"))


@pytest.mark.parametrize("row, p2p", [
    ("glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", None),   # an H box: its engine.env names none
    (TP_DENSE, "1"),                                                          # an L40S row's engine.env names it
    (TP_MOE, "1"),
])
def test_a_tp_rows_stages_have_nccl_p2p_disable_only_when_its_workload_requires_it(tmp_path, monkeypatch, row, p2p):
    """The caller's NCCL_P2P_DISABLE=1 (an L40S operator's shell) reaches the stages only when the row's workload requires it."""
    t = _tp_row(tmp_path, row, environ={"NCCL_P2P_DISABLE": "1"})

    def precheck_reached() -> str:
        raise Stop(99)
    monkeypatch.setattr(t, "hidden_build", precheck_reached)
    with pytest.raises(Stop) as e:
        t.prepare()
    assert e.value.rc == 99 and t.nccl == p2p
    os.makedirs(t.d, exist_ok=True)
    stage = [sys.executable, "-c", "import os; print('NCCL_P2P_DISABLE=' + os.environ.get('NCCL_P2P_DISABLE', 'unset'))"]
    assert t.run_to(0, f"{t.d}/stage.log", stage) == 0
    assert Path(f"{t.d}/stage.log").read_text().splitlines()[-1] == f"NCCL_P2P_DISABLE={p2p or 'unset'}"


def test_a_single_and_a_tp_rows_folds_store_the_instances_form_of_record(tmp_path, monkeypatch):
    """Every child of either row (a TP row's per-rank folds under tp-fold-match among them) stores instances.jsonl in `--instances-form`,
    progressions by default, not the fold library's run-by-run default (a transposed weight's operand is ~1e6 runs per instance)."""
    def precheck_reached(*a, **kw):
        raise Stop(99)
    for argv, form in (((), "progressions"), (("--instances-form", "runs"), "runs")):
        single, tp = _row(tmp_path, *argv), _tp_row(tmp_path, GLM53, *argv)
        monkeypatch.setattr(single, "release_line", precheck_reached)
        monkeypatch.setattr(tp, "hidden_build", precheck_reached)
        for r in (single, tp):
            with pytest.raises(Stop):
                r.prepare()
            assert r.child_env["VERITY_INSTANCES_FORM"] == form


def test_a_tensor_parallel_row_declaring_prefix_caching_is_refused_before_any_stage(tmp_path, monkeypatch):
    """A TP row's stages serve with prefix caching off and its rank Programs take no cached prefix, so a TP row whose file declares it is
    refused at its precheck; a single-GPU prefix row passes it (its warm requests' Programs read their cached rows as an input)."""
    tp = _tp_row(tmp_path, "glm53__fp8__h200__tp8__b8__i256__o32__prefix-arrivals__greedy__eager")
    monkeypatch.setattr(tp, "hidden_build", lambda: "/x/verity_fa2_matReqG.so")
    monkeypatch.setattr(tp, "release_line", lambda *a: pytest.fail("a stage after the precheck"))
    with pytest.raises(Stop) as e:
        tp.prepare()
    last = Path(f"{tp.d}/stages.txt").read_text().splitlines()[-1].split(" ", 3)
    assert e.value.rc == 3 and last[:2] == ["precheck", "FAIL"] and "enable_prefix_caching" in last[3]
    single = _row(tmp_path, row="llama32-1b__bf16__h200__tp1__b2__i168__o10__prefix__greedy__bi-eager")

    def past_precheck(*a):
        raise Stop(99)
    monkeypatch.setattr(single, "release_line", past_precheck)
    with pytest.raises(Stop) as e:
        single.prepare()
    assert e.value.rc == 99


@pytest.mark.parametrize("argv, reads", [
    (("--stages", "build,match", "--match-phase", "cpu"), False),   # a declared-target Build and the CPU half: a CPU node runs them
    (("--stages", "match", "--match-phase", "gpu"), True),
    (("--stages", "match"), True),                                  # the GPU stages and the CPU half in one run
    (("--stages", "commit"), True),
])
def test_a_tp_rows_target_precheck_reads_the_device_only_when_a_stage_runs_on_it(tmp_path, monkeypatch, argv, reads):
    t = _tp_row(tmp_path, GLM53, *argv)
    seen: list[tuple[str, ...]] = []

    def target_family(*extra, out):
        seen.append(extra)
        raise Stop(99)
    for name, fn in (("hidden_build", lambda: "/x/verity_fa2_matReqG.so"), ("release_line", lambda arg: ("PASS|x", None)),
                     ("export_commit", lambda: ""), ("source_identity", lambda pys, out: False), ("target_family", target_family)):
        monkeypatch.setattr(t, name, fn)
    with pytest.raises(Stop) as e:
        t.prepare()
    assert e.value.rc == 99 and seen == [() if reads else ("--no-device",)]


# ---------------------------------------------------------------------------------------------------------------- cuBLAS names

def test_a_rows_stages_get_the_names(tmp_path, monkeypatch, capsys):
    py, _, lib = _node(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    r = _row(tmp_path, "--py", py)
    r.setup({})
    d = r.child_env["LD_LIBRARY_PATH"].split(":")[0]
    assert os.path.dirname(d) == str(tmp_path) and os.path.realpath(os.path.join(d, "libcublas.so")) == os.path.realpath(lib / "libcublas.so.12")
    shutil.copy(lib / "libcublasLt.so.12", lib / "libcublasLt.so.13")
    with pytest.raises(Stop) as e:
        _row(tmp_path, "--py", py).setup({})
    assert e.value.rc == 3 and "PRECHECK_FAIL_CUBLAS_NAMES" in capsys.readouterr().err
    assert isinstance(e.value.__cause__, ValueError) and "not one cuBLAS of each name" in str(e.value.__cause__)


# ---------------------------------------------------------------------------------------------------------------- a forwarded signal

def test_a_rows_timeline_event_carries_the_timelines_id_and_its_numbers_unquoted(tmp_path):
    """`Row.tl` writes one line an event under the id the timeline's readers know (`TIMELINE_SCHEMA`): a value that reads as a number,
    true, false or null goes unquoted, and any other as a string with its quotes and backslashes kept."""
    r = _row(tmp_path)
    r.tl("match.fold", "end", ("rc", 0), ("wall", "2.5"), ("ok", "true"), ("left", "null"), ("why", 'said "no" in a\\b'), ("by", None))
    (event,) = [e for e in _timeline(r) if e["stage"] == "match.fold"]
    assert (event["schema"], event["proc"], event["pid"], event["ev"], event["hot"]) == (TIMELINE_SCHEMA, r.proc, os.getpid(), "end", False)
    assert (event["rc"], event["wall"], event["ok"], event["left"], event["why"], event["by"]) == (0, 2.5, True, None, 'said "no" in a\\b', "None")


def test_a_signal_goes_to_every_running_stage_and_one_that_has_ended_is_passed_over(tmp_path, monkeypatch):
    r = _row(tmp_path)
    sent: list[tuple[int, int]] = []

    def kill(pid: int, signum: int) -> None:
        sent.append((pid, signum))
        if pid == 7:
            raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", kill)
    r._running = {7, 9}
    r._on_signal(signal.SIGTERM, None)
    assert r.signalled == "TERM" and sorted(sent) == [(7, 15), (9, 15)]
    assert (Path(r.d) / "row.log").read_text().count("SIGTERM received: forwarding to the running stage (timeout pid ") == 2


# ---------------------------------------------------------------------------------------------------------------- the launch context

def _launcher(tmp_path: Path, script: str) -> str:
    """An interpreter that is this shell script: what a row's stage command runs in place of ``verity-vllm``."""
    py = tmp_path / "launcher"
    py.write_text("#!/bin/sh\n" + script)
    py.chmod(0o755)
    return str(py)


def test_a_batchs_request_program_is_built_under_its_launch_context_and_a_context_that_does_not_read_fails_the_shape(tmp_path):
    """B=1 asks for none.  B>=2 runs ``launch-context WORKLOAD LP T [C]``: its lines are the Build's ``--launch-context`` arguments
    (none for a context of no lines), and a failure is None with the end of its stderr in row.log, the whole of it kept beside the row."""
    args = 'shift 2; echo "$@" >&2; '                                        # -m verity_vllm.pipeline.cli, then the command line
    r = _row(tmp_path, "--py", _launcher(tmp_path, args + "printf 'k=1\\nk=2\\n'"))
    r.B = 1
    assert r.launch_args(8, 3) == []
    r.B = 2
    assert r.launch_args(8, 3) == ["--launch-context", "k=1", "k=2"] and r.launch_args(8, 3, 4) == ["--launch-context", "k=1", "k=2"]
    assert not [f for f in os.listdir(r.d) if f.endswith(".err")]

    r.py = _launcher(tmp_path, args + "exit 0")
    assert r.launch_args(8, 3) == []

    r.py = _launcher(tmp_path, args + "echo 'ValueError: the workload names no arrivals' >&2; exit 2")
    assert r.launch_args(8, 3, 4) is None
    (err,) = [f for f in os.listdir(r.d) if f.endswith(".err")]
    assert err == "launch_context_LP8_T3_C4.err"
    assert (Path(r.d) / err).read_text() == f"launch-context {r.wl} 8 3 4\nValueError: the workload names no arrivals\n"
    line = (Path(r.d) / "row.log").read_text().splitlines()[-1]
    assert line.endswith(f"Build FAIL shape LP8_T3_C4: launch_context rc=2 -- launch-context {r.wl} 8 3 4 ValueError: the workload names no arrivals ")


# ---------------------------------------------------------------------------------------------------------------- the source identity

def test_a_rows_shipped_commit_is_the_one_its_trees_export_file_states(tmp_path):
    """A shipped tree's ``EXPORT.json`` states the commit it was cut from.  A tree with no such file, or with one that cannot be
    read, does not parse, is no document or states no commit, has none."""
    r = _row(tmp_path)
    r.tree = str(tmp_path)
    export = tmp_path / "EXPORT.json"
    assert r.export_commit() == ""
    for text in ("{", "[]", '"c0ffee"', "{}", '{"commit": null}', '{"commit": ""}', '{"sha": "c0ffee"}'):
        export.write_text(text)
        assert r.export_commit() == "", text
    export.write_text(json.dumps({"commit": "c0ffee" * 6 + "abcd", "dirty": False}))
    assert r.export_commit() == "c0ffee" * 6 + "abcd"
    export.unlink()
    export.mkdir()
    assert r.export_commit() == ""


def test_a_passed_source_identity_goes_into_the_runs_job_record_which_is_left_alone_when_it_does_not_read(tmp_path, monkeypatch):
    """The precheck copies the stage interpreter's identity to ``source_identity.json`` and, inside a run, into ``job.json``
    (``computation.source_identity``, its reason cleared); a job record that is no JSON object is left as it is and named in row.log,
    and the precheck still passes."""
    r = _row(tmp_path)
    d = Path(r.d)
    monkeypatch.setattr(r, "export_commit", lambda: "abc123")

    def source_identity(pys: list[str], out_of) -> int:
        for py in pys:
            Path(out_of(py)).write_text(json.dumps({"ok": True, "py": py}))
        return 0

    monkeypatch.setattr(r, "source_identity", source_identity)
    identity = {"ok": True, "py": r.py}
    r.precheck_identity()                                                            # outside a run: no job record, none written
    assert json.loads((d / "source_identity.json").read_text()) == identity and not (d / "job.json").exists()

    (d / "job.json").write_text(json.dumps({"run": "r1", "computation": {"kind": "row", "source_identity_reason": "not read yet"}}))
    r.precheck_identity()
    assert (d / "job.json").read_text() == json.dumps(
        {"run": "r1", "computation": {"kind": "row", "source_identity_reason": None, "source_identity": identity}}, indent=1) + "\n"
    (d / "job.json").write_text(json.dumps({"run": "r2"}))
    r.precheck_identity()
    assert json.loads((d / "job.json").read_text()) == {"run": "r2", "computation": {"source_identity": identity, "source_identity_reason": None}}
    assert not (d / "job.json.tmp").exists() and not (d / "row.log").exists()

    for text, error in (("{", "Expecting property name"), ('["a list"]', "'list' object has no attribute 'setdefault'")):
        (d / "job.json").write_text(text)
        r.precheck_identity()
        assert (d / "job.json").read_text() == text and not (d / "job.json.tmp").exists()
        assert f"job.json not updated with the source identity: {error}" in (d / "row.log").read_text().splitlines()[-1]
    assert [ln.split()[:2] for ln in (d / "stages.txt").read_text().splitlines()] == [["precheck", "PASS"]] * 5


# ---------------------------------------------------------------------------------------------------------------- the release line

def test_the_row_reads_the_release_through_the_module(tmp_path, monkeypatch):
    """The row's release line: from `release-json line` (tree-local first), the record USED is the one copied into the evidence, and a
    released tree takes the release's collector build dir unless the caller set NATIVE_COLLECT_BUILD."""
    rel = tmp_path / "RELEASE.json"
    rel.write_text(json.dumps({"tag": "r16-int-3", "build_dirs": {"native_collect": str(tmp_path / "nc_rel")}}))

    def prepared(*flags, environ=None):
        r = _row(tmp_path / "sweep", "--native-collect-default", str(tmp_path / "nc_default"), *flags, environ=environ)
        monkeypatch.setattr(r, "release_line", lambda fallback: (f"r16-int-3|commit=a2dabe493 tree={r.tree}", str(rel)))
        for step in ("precheck_identity", "precheck_target", "install_signal_forwarding"):
            monkeypatch.setattr(r, step, lambda: None)

        def shape():
            r.B = r.I = r.O = r.LP = r.T = 1
            r.ck = r.tgt = r.hidden_so = r.code9 = ""
        monkeypatch.setattr(r, "shape", shape)
        r.prepare()
        return r
    r = prepared()
    assert json.loads((Path(r.d) / "release.json").read_text()) == json.loads(rel.read_text())
    assert r.ncb == r.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_rel")
    assert json.loads((Path(r.d) / "tree.json").read_text())["release_line"].startswith("r16-int-3|commit=a2dabe493 ")
    for pinned in (prepared("--native-collect-build", str(tmp_path / "nc_own")),
                   prepared(environ={"NATIVE_COLLECT_BUILD": str(tmp_path / "nc_own")})):
        assert pinned.ncb == pinned.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_own")
    no_dir: tuple[dict, ...] = ({}, {"build_dirs": None}, {"build_dirs": {}}, {"build_dirs": {"native_collect": ""}}, {"build_dirs": {"fa2": "/x"}})
    for names_none in no_dir:
        rel.write_text(json.dumps({"tag": "r16-int-3", **names_none}))                 # a release that names no collector build dir
        r = prepared()
        assert r.ncb == r.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_default")


# ---------------------------------------------------------------------------------------------------------------- the target-family preflight

def test_the_row_runs_the_preflight_before_build_and_canary_waives_by_name(tmp_path, monkeypatch):
    r = _row(tmp_path, "--stages", "build,match")
    seen = []
    for step in ("precheck_identity", "shape", "build", "match"):
        monkeypatch.setattr(r, step, lambda step=step: seen.append(step))
    monkeypatch.setattr(r, "release_line", lambda fallback: (f"UNRELEASED|commit=unknown tree={r.tree}{fallback}", "-"))

    def target_family(*a, out):
        seen.append(("target_family", r.stages, a))
        return 3, "PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9"
    monkeypatch.setattr(r, "target_family", target_family)
    assert r.run() == 3
    assert seen == ["precheck_identity", "shape", ("target_family", "build,match", ())]
    assert "precheck-target FAIL" in (tmp_path / r.row / "stages.txt").read_text()
    can = (TREE / "verity_vllm" / "ops" / "canary.sh").read_text()
    assert 'TARGET_FAMILY_WAIVER=${TARGET_FAMILY_WAIVER:-"canary $TAG"}' in can


# ---------------------------------------------------------------------------------------------------------------- what the run store brings

def _shows(monkeypatch, *docs: Any, publish: str = "") -> tuple[tuple[list[str], dict[str, str]], list[list[str]]]:
    """The research CLI as a stand-in: ``data show`` answers with the newest of ``docs`` (a str as it is, else its JSON), and any
    other command with ``publish`` as its stdout.  Returns the CLI as `research_argv` gives it, and the commands asked."""
    asked: list[list[str]] = []

    def run(argv, **kw):
        asked.append(list(argv))
        if asked[-1][1:3] != ["data", "show"]:
            return subprocess.CompletedProcess(argv, 0, stdout=publish)
        return subprocess.CompletedProcess(argv, 0, stdout=docs[-1] if isinstance(docs[-1], str) else json.dumps(docs[-1]))

    monkeypatch.setattr(RS.subprocess, "run", run)
    return (["research"], {}), asked


def test_a_jobs_input_is_nothing_where_its_record_lists_none(tmp_path):
    """A run's ``job.json`` lists its inputs under ``store.inputs``; a record without the block, with it null or empty, with no
    such input, or with a block that is no table, has none."""
    job = tmp_path / "job.json"
    lists_none: list[Any] = [{"store": {}}, {"store": None}, {"store": {"inputs": None}}, {"store": {"inputs": {}}},
                             {"store": {"inputs": {"build": None}}}, {"store": {"inputs": {"match": "art:m"}}}, {"store": "inputs"},
                             {"store": ["inputs"]}, {"store": {"inputs": ["build"]}}, {"store": {"inputs": "build"}}]
    for doc in lists_none:
        job.write_text(json.dumps(doc))
        assert RS.job_input(str(tmp_path), "build") == "", doc
    job.write_text(json.dumps({"store": {"inputs": {"build": "art:b", "match": "art:m"}}}))
    assert (RS.job_input(str(tmp_path), "build"), RS.job_input(str(tmp_path), "match")) == ("art:b", "art:m")


@pytest.mark.parametrize("doc, digest", [
    ({"manifest": {"meta": {"program_digest": "ab" * 32}}}, "ab" * 32),         # the local store shows a manifest, and its meta is read
    ({"meta": {"program_digest": "ab" * 32}}, "ab" * 32),                       # the remote shows the meta at the top
    ({"manifest": {}, "meta": {"program_digest": "cd" * 32}}, "cd" * 32),       # an empty manifest is none
    ({"manifest": None, "meta": {"program_digest": 7}}, "7"),
    ({"manifest": {"meta": None}, "meta": {"program_digest": "cd" * 32}}, ""),  # a manifest that is there is the one read
    ({"manifest": {"meta": {}}}, ""),
    ({"meta": None}, ""),
    ({"meta": {}}, ""),
    ({"meta": {"program_digest": None}}, ""),
    ({"meta": {"workload_digest": "ef" * 32}}, ""),
    ({}, ""),
    ([{"meta": {"program_digest": "ab" * 32}}], ""),                            # a list is no artifact
    ("null", ""),
    ("not a document", ""),
])
def test_an_artifacts_meta_is_its_manifests_and_nothing_where_the_store_shows_none(monkeypatch, doc, digest):
    research, _ = _shows(monkeypatch, doc)
    assert RS.artifact_meta(research, "art:" + "a" * 64, "program_digest") == digest


@pytest.mark.parametrize("doc, art", [
    ({"outputs": {"build": "art:b", "plan": "art:p"}}, "art:b"),
    ([{"outputs": {"build": "art:b"}}, {"outputs": {"build": "art:c"}}], "art:b"),   # the store may show a list: its first is the Attempt
    ({"outputs": {"build": None}}, ""),
    ({"outputs": {"plan": "art:p"}}, ""),
    ({"outputs": {}}, ""),
    ({"outputs": None}, ""),
    ({}, ""),
    ([], ""),
    (["r1"], ""),
    ("7", ""),
    ("not a document", ""),
])
def test_an_attempts_output_is_the_artifact_it_published_under_that_name(monkeypatch, doc, art):
    research, asked = _shows(monkeypatch, doc)
    assert RS.attempt_output(research, "r1", "build") == art and asked == [["research", "data", "show", "r1", "--json"]]


def test_a_cited_attempt_is_its_stages_passing_one_with_its_account_artifact(monkeypatch):
    """`--match-run` (and `--build-run`) cite an Attempt of the stage's tool that exited 0 and published the stage's account
    artifact; an Attempt that leaves one of the three out, or states it as null or empty, is refused with what it has."""
    passing = {"tool": {"name": "vllm.match"}, "status": {"rc": 0}, "outputs": {"match": "art:m", "build": "art:b"}}
    research, _ = _shows(monkeypatch, passing)
    assert RS.cited(research, "match", "r1") == {"run_id": "r1", "artifact": "art:m", "rc": 0}
    has = {"tool": "tool vllm.match", "status": "rc 0", "outputs": "match artifact art:m"}
    lacks = {"tool": "tool None", "status": "rc None", "outputs": "match artifact None"}
    for block in has:
        without = {k: v for k, v in passing.items() if k != block}
        for attempt in (without, {**passing, block: None}, {**passing, block: {}}):
            research, _ = _shows(monkeypatch, attempt)
            said = ", ".join({**has, block: lacks[block]}.values())
            with pytest.raises(UsageError, match=re.escape(f"--match-run r1: not a passing Match Attempt ({said})")):
                RS.cited(research, "match", "r1")
    for attempt, said in (({**passing, "tool": {"name": "vllm.build"}}, "tool vllm.build, rc 0, match artifact art:m"),
                          ({**passing, "status": {"rc": 1}}, "tool vllm.match, rc 1, match artifact art:m"),
                          ({**passing, "outputs": {"build": "art:b"}}, "tool vllm.match, rc 0, match artifact None")):
        research, _ = _shows(monkeypatch, attempt)
        with pytest.raises(UsageError, match=re.escape(f"({said})")):
            RS.cited(research, "match", "r1")
    research, _ = _shows(monkeypatch, {"tool": {"name": "vllm.build"}, "status": {"rc": 0}, "outputs": {"build": "art:b"}})
    assert RS.cited(research, "build", "r0") == {"run_id": "r0", "artifact": "art:b", "rc": 0}


def test_a_record_is_published_only_from_a_commit_attempt_that_ran_a_clean_commit(tmp_path, monkeypatch, capsys):
    """`publish` asks the store which source the Commit Attempt ran: one that records no source, no commit, or a tree that was
    dirty or not said to be clean, publishes nothing (exit 21) and says so in row.log; a clean commit is the record's tree."""
    record = "art:" + "e" * 64
    o = SimpleNamespace(row=ROW)
    runs = {st: {"artifact": f"art:{st}", "run_id": f"r-{st}"} for st in RS.OF_RECORD}
    how = {"ref": "main", "by": "vllm", "tier": None}
    no_clean_commit: list[dict[str, Any]] = [{}, {"source": None}, {"source": {}}, {"source": {"commit": "c0ffee"}},
                                             {"source": {"commit": "c0ffee", "dirty": True}}, {"source": {"commit": "c0ffee", "dirty": None}},
                                             {"source": {"commit": "", "dirty": False}}, {"source": {"dirty": False}}]
    for attempt in no_clean_commit:
        research, asked = _shows(monkeypatch, attempt, publish=record)
        assert RS.publish(o, str(tmp_path), runs, research, how) == RS.PUBLISH_REFUSED
        ran = attempt.get("source") or "unknown"
        said = (tmp_path / "row.log").read_text().splitlines()[-1]
        assert said.endswith(f"publishes nothing: the Commit r-commit ran source {ran}, not a clean commit"), attempt
        assert asked == [["research", "data", "show", "r-commit", "--json"]] and not (tmp_path / "row.json").exists()
    research, asked = _shows(monkeypatch, {"source": {"commit": "c0ffee", "dirty": False}}, publish=f"published {record}\n")
    assert RS.publish(o, str(tmp_path), runs, research, how) == 0
    words = asked[-1]
    assert words[1:5] == [*RS.CLI, "records", "publish"] and words[words.index("--tree") + 1] == "main@c0ffee" and "--tier" not in words
    assert json.loads((tmp_path / "row.json").read_text())["record"] == {"artifact": record, "rc": 0}
    assert f"publish - {record} rc=0" in capsys.readouterr().out
