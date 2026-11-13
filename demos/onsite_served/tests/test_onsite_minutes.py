"""`minutes.py`: the onsite demo's minutes from the prover node's run's start to the auditor host's run's end, written
only for an accepting ``VERDICT`` line of the goal's seven checks, from a research run, on a commit of main."""
from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import minutes as M
import pytest


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", "-C", str(repo), *args], input="", capture_output=True, text=True, check=True,
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


def line(verdict: str = "accept", run: str = "r-prover", **components: str) -> str:
    c = {k: "accept" for k in M.CHECKS} | components
    return "VERDICT " + json.dumps({"block": 0, "coins": "development", "components": c, "product": "verity/onsite/v0",
                                    "reason": "", "registration": "ab" * 32, "run": run, "simulated": [],
                                    "verdict": verdict}, sort_keys=True)


def demo(root: Path, sha: str, *log: str) -> list[str]:
    """The prover node's run, 06:42:19Z to 08:07:28Z, and the auditor host's, to 08:50:31Z, whose log is ``log``."""
    out = []
    for rid, started, ended, lines in (("r-prover", "06:42:19", "08:07:28", ()),
                                       ("r-auditor", "08:10:00", "08:50:31", log)):
        d = root / rid
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"started_utc": f"2026-11-09T{started}Z",
                                                   "ended_utc": f"2026-11-09T{ended}Z"}))
        (d / "job.json").write_text(json.dumps({"computation": {"tree_sha": sha}}))
        (d / "stdout.log").write_text("".join(f"{x}\n" for x in ("PARTY {}", *lines)))
        out.append(rid)
    return out


def test_the_checks_are_the_served_tables_seven():
    """`CHECKS` repeats `served_table.ONSITE_COMPONENTS`, which this reader can't import (it is stdlib only)."""
    tree = ast.parse((Path(M.__file__).parent / "served_table.py").read_text())
    (seven,) = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "ONSITE_COMPONENTS" for t in n.targets)]
    assert M.CHECKS == seven


def test_an_accepting_verdict_of_the_seven_checks_from_research_runs_on_main_is_its_minutes(tmp_path, repo,
                                                                                         monkeypatch, capsys):
    r, on, _ = repo
    runs = demo(tmp_path / "runs", on, line())
    out = tmp_path / "me"
    out.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-minutes")
    monkeypatch.setattr(M.R, "ROOT", r)
    assert M.main([*runs, "--runs-dir", str(tmp_path / "runs"), "--main", "main"]) == 0
    doc = json.loads((out / "result.json").read_text())
    assert doc["measurements"] == [{"name": "vllm/onsite-demo-minutes", "value": 128.2, "unit": "minutes",
                                    "better": "lower", "target": 60, "verdict": "accept",
                                    "runs": ["r-prover", "r-auditor"], "commit": on}]
    assert (doc["run_id"], doc["validation"]["status"], doc["validation"]["checker"]) == (
        "r-minutes", "passed", "demos/onsite_served/minutes.py")
    assert capsys.readouterr().out.startswith("MINUTES {")


@pytest.mark.parametrize("log, why", [
    ((), "r-auditor: no verdict line in its stdout.log"),
    ((line(run="local"),), "its verdict line's run is local"),
    ((line(wipe="absent", pous="absent"),), "covers 5 of the goal's 7 checks, not pous, wipe"),
    ((line(), line(verdict="reject", pouw="reject")), "the verdict is reject, after 128.2 minutes"),
])
def test_no_accepting_verdict_of_the_seven_checks_from_a_research_run_is_refused(tmp_path, repo, log, why):
    r, on, _ = repo
    with pytest.raises(M.R.Refused, match=why):
        M.R.minutes(demo(tmp_path / "runs", on, *log), tmp_path / "runs", "main", r, read=M.verdict, metric=M.METRIC)


def test_a_rejecting_line_before_an_accepting_one_rejects_the_period_with_no_value(tmp_path, repo, monkeypatch, capsys):
    """Accept only where every line accepts: a period whose last line accepts after one that rejects has no minutes."""
    r, on, _ = repo
    runs = demo(tmp_path / "runs", on, line(verdict="reject", pouw="reject"), line())
    out = tmp_path / "me"
    out.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-minutes")
    monkeypatch.setattr(M.R, "ROOT", r)
    assert M.main([*runs, "--runs-dir", str(tmp_path / "runs"), "--main", "main"]) == 1
    doc = json.loads((out / "result.json").read_text())
    assert not doc["measurements"] and doc["validation"]["status"] == "failed"
    assert capsys.readouterr().out == ("MINUTES refused: r-auditor: the verdict is reject, after 128.2 minutes; only an "
                                       "accepting demo has minutes\n")


def test_a_served_commit_off_main_or_a_third_run_is_refused(tmp_path, repo, capsys):
    r, _, off = repo
    with pytest.raises(M.R.Refused, match=f"the served commit {off} is not a commit of main"):
        M.R.minutes(demo(tmp_path / "runs", off, line()), tmp_path / "runs", "main", r, read=M.verdict)
    with pytest.raises(SystemExit):
        M.main(["r-prover", "r-other", "r-auditor"])
    assert "unrecognized arguments" in capsys.readouterr().err
