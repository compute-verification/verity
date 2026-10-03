"""`post_train.py`: the label a landing's check attempt gets from its manifest rebuilds, which landing it labels, and the job it runs."""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

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


def test_the_job_pins_the_rebuilds_to_the_cores_it_is_given():
    pinned, free = PT.job("96-127", 12), PT.job("", 4)
    assert "taskset -c 96-127 ../../.venv/bin/python -m pytest" in pinned and "taskset" not in free
    assert "-k 'manifest_digest or negative_57' -n 12" in pinned and "-n 4" in free
    assert "VERITY_REGRESSION_REBUILD=1 VERITY_STORE_REQUIRED=1" in pinned and "fetch-fixtures" in pinned
