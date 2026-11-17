"""`sampled_accepted.py`: the share of a served run's sampled proof blocks whose proofs the Lean verifier accepted on the
host that composed its record, 100 only for SPOT's accept recomputed there on a commit of main, else 0 with the
first condition it fails."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import sampled_accepted as M


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


LEAN = M.LEAN_HERE + " (vstar-recursive sampled check) from the proofs"


def spot(verdict: str = "accept", answer_from: str = LEAN, sampled: int | None = 6) -> dict:
    profile = {"levels": [{"name": "blocks"}, {"name": "units"}], "drawn": {"blocks": 2, "units": sampled}}
    return {"verdict": verdict, "answer_from": answer_from,
            "answer": {"verdict": verdict, "profile": profile if verdict == "accept" else None}}


def demo(root: Path, sha: str, row: dict | None, served_node: bool | None = False, report: dict | None = None) -> list[str]:
    """The served run and the run whose record holds SPOT's row ``row`` (no record where None), or is ``report``."""
    for rid in ("r-served", "r-verdicts"):
        d = root / rid
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"started_utc": "2026-11-15T06:00:00Z", "ended_utc": "2026-11-15T07:00:00Z"}))
        (d / "job.json").write_text(json.dumps({"computation": {"tree_sha": sha}}))
    if row is not None or report is not None:
        f = root / "r-verdicts" / M.RECORD
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps(report if report is not None else
                                {"protocols": {"integrity": row}, "composed_on": {"host": "auditor", "served_node": served_node}}))
    return ["r-served", "r-verdicts"]


def test_lean_here_is_the_head_of_served_verdicts_own():
    """`LEAN_HERE` repeats the head of `served_verdicts.LEAN_HERE`, which this reader can't import (it is stdlib only)."""
    src = Path(M.__file__).resolve().parents[1] / "one_stage" / "served_verdicts.py"
    (whole,) = [ast.literal_eval(n.value) for n in ast.parse(src.read_text()).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "LEAN_HERE" for t in n.targets)]
    assert whole.startswith(M.LEAN_HERE)


@pytest.mark.parametrize("served_node", [False, True])
def test_an_accept_recomputed_on_the_auditors_machine_from_main_is_every_sampled_proof_block(tmp_path, repo, monkeypatch, capsys,
                                                                                            served_node):
    r, on, _ = repo
    runs = demo(tmp_path / "runs", on, spot(), served_node)
    out = tmp_path / "me"
    out.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-accepted")
    monkeypatch.setattr(M.R, "ROOT", r)
    assert M.main([*runs, "--runs-dir", str(tmp_path / "runs"), "--main", "main"]) == 0
    doc = json.loads((out / "result.json").read_text())
    (m,) = doc["measurements"]
    assert (m["name"], m["value"], m["target"], m["sampled"], m["accepted"], m["reason"]) == (M.METRIC, 100, 100, 6, 6, None)
    assert m["inputs"][0].startswith("measured: ") and m["mark"] == "measured"
    where = ("its record was composed on the workload's node (auditor), which the prover controls, so no guarantee "
             "applies to it" if served_node else "its record was composed off the workload's node, on auditor")
    assert m["composed_on"] == {"host": "auditor", "served_node": served_node}
    assert (doc["validation"]["status"], doc["validation"]["detail"]) == ("passed",
                                                                         f"6 of 6 sampled proof blocks accepted; {where}")
    assert capsys.readouterr().out.startswith("ACCEPTED {")


@pytest.mark.parametrize("row, served_node, off, why", [
    (spot(), False, True, "served from .*, not a commit of main"),
    (spot(answer_from="as the served node wrote it"), False, False, "not recomputed there by the Lean verifier"),
    (spot(verdict="reject"), False, False, "the Lean verifier's verdict is reject"),
])
def test_anything_short_of_that_accepts_no_unit_and_says_why(tmp_path, repo, row, served_node, off, why):
    r, on, off_sha = repo
    m = M.share(demo(tmp_path / "runs", off_sha if off else on, row, served_node), tmp_path / "runs", "main", r)
    assert (m["value"], m["accepted"]) == (0, 0) and m["composed_on"]["served_node"] is served_node
    assert re.search(why, m["reason"]), m["reason"]


@pytest.mark.parametrize("report, why", [
    ({"protocols": {}, "composed_on": {"host": "auditor", "served_node": False}}, "holds no SPOT row"),
    ({"protocols": {"integrity": {}}, "composed_on": {"host": "auditor", "served_node": False}}, "holds no SPOT row"),
    ({"protocols": {"integrity": spot()}}, "names no composed_on.served_node"),
    ({"protocols": {"integrity": spot()}, "composed_on": {"host": "auditor", "served_node": None}}, "names no composed_on"),
])
def test_a_partial_record_is_refused_not_measured_as_0(tmp_path, repo, report, why):
    """A record whose SPOT row is missing or empty, or without where it was composed is no measurement, not a run that failed."""
    r, on, _ = repo
    with pytest.raises(M.R.Refused, match=why):
        M.share(demo(tmp_path / "runs", on, None, report=report), tmp_path / "runs", "main", r)


def test_no_record_or_an_accept_sampling_nothing_is_refused(tmp_path, repo):
    r, on, _ = repo
    with pytest.raises(M.R.Refused, match="no out/record/verdicts.json"):
        M.share(demo(tmp_path / "a", on, None), tmp_path / "a", "main", r)
    with pytest.raises(M.R.Refused, match="names no sampled proof block"):
        M.share(demo(tmp_path / "b", on, spot(sampled=None)), tmp_path / "b", "main", r)
