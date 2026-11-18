"""`research run --on` makes a run whose shipped tree isn't main's a trial (`cli.branch_trial`): a measurement taken from a branch or a
dirty checkout wrote a point of record before, and the dashboard read a branch's number."""
import os
import subprocess
from pathlib import Path

from research import cli

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
T = cli.TRIAL_ENV


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True, env=ENV)


def _repo(tmp_path: Path, origin_main: bool = True) -> Path:
    r = tmp_path / "src"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "a.py").write_text("a = 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "a")
    if origin_main:
        _git(r, "update-ref", "refs/remotes/origin/main", "HEAD")
    return r


def test_a_run_of_mains_commit_is_untouched(tmp_path):
    r = _repo(tmp_path)
    env = {"X": "1"}
    assert cli.branch_trial(r, env, False) is None and env == {"X": "1"}
    _git(r, "checkout", "-q", "-b", "work")   # a branch at main's commit is main's
    assert cli.branch_trial(r, env, False) is None and env == {"X": "1"}


def test_a_run_ahead_of_main_is_a_trial_unless_it_says_otherwise(tmp_path):
    r = _repo(tmp_path)
    (r / "a.py").write_text("a = 2\n")
    _git(r, "commit", "-q", "-am", "b")
    env: dict[str, str] = {}
    said = cli.branch_trial(r, env, False)
    assert env == {T: "1"} and "isn't on origin/main" in said and "--of-record" in said and f"--env {T}=0" in said
    for env, of_record in (({T: "0"}, False), ({}, True)):
        before = dict(env)
        assert cli.branch_trial(r, env, of_record) is None and env == before


def test_a_dirty_checkout_of_mains_commit_is_a_trial(tmp_path):
    r = _repo(tmp_path)
    (r / "b.py").write_text("b = 1\n")
    env: dict[str, str] = {}
    said = cli.branch_trial(r, env, False)
    assert env == {T: "1"} and "uncommitted changes" in said and "isn't on origin/main" not in said


def test_a_checkout_without_origin_main_is_left_as_it_was_and_says_so(tmp_path):
    r = _repo(tmp_path, origin_main=False)
    (r / "a.py").write_text("a = 2\n")
    env: dict[str, str] = {}
    said = cli.branch_trial(r, env, False)
    assert env == {} and "can't tell" in said
    assert cli.branch_trial(None, env, False) is None and env == {}
