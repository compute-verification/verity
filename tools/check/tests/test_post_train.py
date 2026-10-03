"""`post_train.py`: the label a landing's check attempt gets from its manifest rebuilds, which landing it labels, and the job it runs."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("post_train_under_test", ROOT / "tools" / "check" / "post_train.py")
PT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PT)


def _junit(tmp_path: Path, cases: dict[str, str]) -> Path:
    body = {"pass": "", "fail": "<failure message='x'/>", "error": "<error message='x'/>", "skip": "<skipped message='retired'/>"}
    xml = "".join(f"<testcase classname='tests.regression.test_regression' name='{n}'>{body[k]}</testcase>" for n, k in cases.items())
    p = tmp_path / "junit.xml"
    p.write_text(f"<testsuites><testsuite name='pytest'>{xml}</testsuite></testsuites>")
    return p


def test_a_pass_is_every_rebuild_passing_or_retired(tmp_path):
    j = _junit(tmp_path, {"test_reproduces[T0-manifest_digest-r23]": "pass", "test_reproduces[T0-manifest_digest-r74]": "skip",
                          "test_negative_57_refusal_reproduces": "pass"})
    assert PT.verdict(j) == "regression pass"


def test_a_failure_names_its_rows_in_row_order_and_the_refusal_by_name(tmp_path):
    j = _junit(tmp_path, {"test_reproduces[T0-manifest_digest-r57]": "fail", "test_reproduces[T0-manifest_digest-r11]": "error",
                          "test_reproduces[T0-manifest_digest-r23]": "pass", "test_negative_57_refusal_reproduces": "fail"})
    assert PT.verdict(j) == "regression fail:r11,r57,negative_57"


def test_no_result_is_a_failure_that_says_why(tmp_path):
    assert PT.verdict(tmp_path / "absent.xml", 3) == "regression fail:no junit (run rc=3)"
    assert PT.verdict(_junit(tmp_path, {"test_reproduces[T0-manifest_digest-r74]": "skip"}), 0) == "regression fail:nothing ran (run rc=0)"


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def test_only_a_landing_has_a_check_attempt_to_label(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "a lane's commit")
    lane = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "Merge #1 #2 (abc)\n\nCheck: r20261003-145201-e806\n\nCo-authored-by: x <x@x>")
    assert PT.check_attempt(_git(tmp_path, "rev-parse", "HEAD"), cwd=tmp_path) == "r20261003-145201-e806"
    assert PT.check_attempt(lane, cwd=tmp_path) is None


def test_check_starts_it_after_each_landing():
    spec = importlib.util.spec_from_file_location("check_tool_under_test", ROOT / "tools" / "check" / "tool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CHECK.after_land[-1] == "tools/check/post_train.py" and (ROOT / mod.CHECK.after_land[-1]).is_file()


SHA = "f" * 40
PREV = "r20261003-120000-0001"


def _fake_landing(tmp_path, monkeypatch, attempt, prev_label=None, untouched=False):
    """A landing ``SHA`` admitted on ``attempt``, after a landing admitted on PREV that carries ``prev_label``; every call to
    `research` is recorded, and a run fails r39 (with the lock held: jobs run one at a time)."""
    runs = tmp_path / "runs"
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(PT, "LOCK", tmp_path / "post-train.lock")
    calls: list[tuple[str, ...]] = []

    def research(*args, cwd=None):
        calls.append(args)
        if args[0] == "run":
            with open(PT.LOCK) as other, pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (runs / "r20261003-000000-abcd" / "out").mkdir(parents=True)
            _junit(runs / "r20261003-000000-abcd" / "out", {"test_reproduces[T0-manifest_digest-r39]": "fail"})
            return subprocess.CompletedProcess(args, 1, "research: run r20261003-000000-abcd launched\n", "")
        if args[:2] == ("data", "labels"):
            labs = [prev_label] if args[2] == PREV and prev_label else []
            return subprocess.CompletedProcess(args, 0, json.dumps(labs), "")
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(PT, "research", research)
    monkeypatch.setattr(PT, "git", lambda *a, **k: SHA if a[0] == "rev-parse" else "")
    monkeypatch.setattr(PT, "check_attempt", lambda sha, cwd=PT.ROOT: {SHA: attempt, f"{SHA}^1": PREV}.get(sha))
    monkeypatch.setattr(PT, "untouched", lambda tree, rev: untouched and rev == f"{SHA}^1")
    import research.runs
    monkeypatch.setattr(research.runs, "runs_root", lambda: runs)
    return calls


def test_a_landing_gets_its_label_and_a_dry_run_writes_none(tmp_path, monkeypatch):
    calls = _fake_landing(tmp_path, monkeypatch, "r20261003-145201-e806")
    assert PT.main(["--cores", "96-127"]) == 1
    assert ("data", "label", "r20261003-145201-e806", "note", "regression fail:r39", "--by", "regression",
            "--ref", "r20261003-000000-abcd") in calls
    calls = _fake_landing(tmp_path / "dry", monkeypatch, None)
    assert PT.main(["HEAD", "--dry-run"]) == 1
    assert [c[0] for c in calls] == ["run", "fetch"], "a dry run reads no label and writes none"
    assert PT.main(["HEAD"]) == 2, "without --dry-run, a commit that is no landing has nothing to label"


def test_a_landing_that_touches_none_of_the_inputs_carries_the_verdict_before_it(tmp_path, monkeypatch):
    for value, rc in (("regression pass", 0), ("regression fail:r57", 1)):
        prev = {"key": "note", "value": value, "by": "regression", "ref": "r20261003-120500-beef"}
        calls = _fake_landing(tmp_path / value[11:15], monkeypatch, "r20261003-145201-e806", prev, untouched=True)
        assert PT.main([]) == rc
        assert ("data", "label", "r20261003-145201-e806", "note", value, "--by", "regression", "--ref", "r20261003-120500-beef") in calls
        assert not any(c[0] == "run" for c in calls)


def test_a_verdict_is_not_carried_over_a_change_to_the_inputs_nor_from_a_run_that_never_tested(tmp_path, monkeypatch):
    ok = {"key": "note", "value": "regression pass", "by": "regression", "ref": "r20261003-120500-beef"}
    broken = {**ok, "value": "regression fail:no junit (run rc=3)"}
    for name, prev, untouched in (("touched", ok, False), ("broken", broken, True), ("unlabelled", None, True)):
        calls = _fake_landing(tmp_path / name, monkeypatch, "r20261003-145201-e806", prev, untouched=untouched)
        assert PT.main([]) == 1, name
        assert [c[:2] for c in calls if c[0] == "run" or c[:2] == ("data", "label")] == [("run", "--on"), ("data", "label")], name
        assert calls[-1][-1] == "r20261003-000000-abcd", name


def test_untouched_is_suites_saying_the_change_touches_none(monkeypatch):
    def fake(out: str, rc: int = 0):
        monkeypatch.setattr(PT.subprocess, "run", lambda argv, **_k: subprocess.CompletedProcess(argv, rc, out, ""))
    fake("suites: the change from 1a2b3c4d5e6f touches 0: none\n")
    assert PT.untouched(ROOT, "HEAD^1")
    fake("suites: the change from 1a2b3c4d5e6f touches 1: verity-vllm\nverity-vllm  integrations/vllm  not cached\n")
    assert not PT.untouched(ROOT, "HEAD^1")
    fake("suites: the change from 1a2b3c4d5e6f touches 0: none\n", rc=2)
    assert not PT.untouched(ROOT, "HEAD^1")


def test_the_job_pins_the_rebuilds_to_the_cores_it_is_given():
    pinned, free = PT.job("96-127", 12), PT.job("", 4)
    assert "taskset -c 96-127 ../../.venv/bin/python -m pytest" in pinned and "taskset" not in free
    assert "-m 'regression and rebuild' -n 12" in pinned and "-n 4" in free
    assert "VERITY_REGRESSION_REBUILD=1 VERITY_STORE_REQUIRED=1" in pinned and "fetch-fixtures" in pinned
