"""`run_minutes.py`: the minutes from the first run's start to the last run's end, written only for an accepting verdict on
a commit of main, and refused with its reason otherwise."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import run_minutes as M

METRIC = "test/minutes"
CHECKER = "tests/benchmarks/demo_minutes/test_run_minutes.py"


def git(repo: Path, *args: str, stdin: str = "") -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository whose ``main`` holds one commit, and a commit off it: (path, on main, off main)."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    tree = git(r, "mktree")
    on = git(r, "commit-tree", tree, "-m", "on main")
    git(r, "update-ref", "refs/heads/main", on)
    return r, on, git(r, "commit-tree", tree, "-p", on, "-m", "off main")


def run(root: Path, rid: str, *, started: str | None = None, ended: str | None = None, sha: str | None = None,
        verdict: str | None = None) -> str:
    d = root / rid
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": "done", "rc": 0, "started_utc": started, "ended_utc": ended}))
    (d / "job.json").write_text(json.dumps({"computation": {"tree_sha": sha}}))
    if verdict is not None:
        (d / "verdict").write_text(verdict)
    return rid


def read(d: Path) -> str:
    f = d / "verdict"
    if not f.is_file():
        raise M.Refused(f"{d.name}: no verdict")
    return f.read_text()


def runs3(root: Path, sha: str, verdict: str | None = "accept") -> list[str]:
    """Three runs, 10:18:08Z to 13:39:26Z on the same day, the last with ``verdict``."""
    return [run(root, "r-serve", started="2026-11-12T10:18:08.000Z", ended="2026-11-12T11:39:23.000Z", sha=sha),
            run(root, "r-audit", started="2026-11-12T12:00:44.000Z", ended="2026-11-12T13:01:00.000Z", sha=sha),
            run(root, "r-compose", started="2026-11-12T13:05:00.000Z", ended="2026-11-12T13:39:26.000Z", sha=sha,
                verdict=verdict)]


def main(argv: list[str]) -> int:
    return M.main(argv, read=read, metric=METRIC, checker=CHECKER)


def test_an_accepting_verdict_on_a_commit_of_main_is_its_minutes(tmp_path, repo, monkeypatch, capsys):
    r, on, _ = repo
    runs = runs3(tmp_path / "runs", on)
    m = M.minutes(runs, tmp_path / "runs", "main", r, read=read, metric=METRIC)
    assert m == {"name": METRIC, "value": 201.3, "unit": "minutes", "better": "lower", "target": 60,
                 "verdict": "accept", "runs": ["r-serve", "r-audit", "r-compose"], "commit": on}
    out = tmp_path / "me"
    out.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-minutes")
    monkeypatch.setattr(M, "ROOT", r)
    assert main([*runs, "--runs-dir", str(tmp_path / "runs"), "--main", "main"]) == 0
    doc = json.loads((out / "result.json").read_text())
    assert (doc["schema"], doc["run_id"], doc["validation"]["status"]) == ("research/result/v0.1", "r-minutes", "passed")
    assert doc["measurements"] == [m]
    assert capsys.readouterr().out.startswith("MINUTES {")


def test_a_rejecting_verdict_gives_its_minutes_to_the_log_and_no_value(tmp_path, repo, monkeypatch, capsys):
    r, on, _ = repo
    runs = runs3(tmp_path / "runs", on, "reject")
    with pytest.raises(M.Rejected, match="r-compose: the verdict is reject, after 201.3 minutes"):
        M.minutes(runs, tmp_path / "runs", "main", r, read=read, metric=METRIC)
    out = tmp_path / "me"
    out.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out))
    monkeypatch.setattr(M, "ROOT", r)
    assert main([*runs, "--runs-dir", str(tmp_path / "runs"), "--main", "main"]) == 1
    doc = json.loads((out / "result.json").read_text())
    assert (doc["measurements"], doc["validation"]["status"]) == ([], "failed")
    assert capsys.readouterr().out == ("MINUTES refused: r-compose: the verdict is reject, after 201.3 minutes; only an "
                                       "accepting demo has minutes\n")


def test_what_read_refuses_is_refused(tmp_path, repo):
    r, on, _ = repo
    with pytest.raises(M.Refused, match="r-compose: no verdict"):
        M.minutes(runs3(tmp_path / "runs", on, None), tmp_path / "runs", "main", r, read=read, metric=METRIC)


def test_a_served_commit_off_main_is_refused(tmp_path, repo):
    r, _, off = repo
    with pytest.raises(M.Refused, match=f"the served commit {off} is not a commit of main"):
        M.minutes(runs3(tmp_path / "runs", off), tmp_path / "runs", "main", r, read=read, metric=METRIC)
    with pytest.raises(M.Refused, match="can't tell whether"):
        M.minutes(runs3(tmp_path / "other", "0" * 40), tmp_path / "other", "main", r, read=read, metric=METRIC)


def test_a_record_without_its_times_or_runs_out_of_order_is_refused(tmp_path, repo):
    r, on, _ = repo
    root = tmp_path / "runs"
    first = run(root, "r-first", ended="2026-11-12T11:00:00Z", sha=on)
    last = run(root, "r-last", started="2026-11-12T11:00:00Z", verdict="accept")
    with pytest.raises(M.Refused, match="r-first: its status.json has no started_utc"):
        M.minutes([first, last], root, "main", r, read=read, metric=METRIC)
    with pytest.raises(M.Refused, match="no run record"):
        M.minutes(["r-absent", last], root, "main", r, read=read, metric=METRIC)
    runs = runs3(tmp_path / "again", on)
    with pytest.raises(M.Refused, match="not in the demo's order"):
        M.minutes(runs[::-1], tmp_path / "again", "main", r, read=read, metric=METRIC)


def test_a_refusal_writes_no_measurement_and_its_reason(tmp_path, repo, monkeypatch, capsys):
    r, _, off = repo
    runs = runs3(tmp_path / "runs", off)
    out = tmp_path / "me"
    out.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out))
    monkeypatch.setattr(M, "ROOT", r)
    assert main([*runs, "--runs-dir", str(tmp_path / "runs"), "--main", "main"]) == 1
    doc = json.loads((out / "result.json").read_text())
    assert doc["measurements"] == []
    assert doc["validation"]["status"] == "not_run"
    assert "is not a commit of main" in doc["validation"]["detail"]
    assert capsys.readouterr().out.startswith("MINUTES refused: ")
