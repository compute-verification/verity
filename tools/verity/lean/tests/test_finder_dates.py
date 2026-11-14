"""`tools/verity/lean/finder_dates.py` on a repository it makes: an entry due before the given day fails, naming the
definition, its lock, owner, issue and date and the two ways out; one due that day or later passes; with `--base` only an
entry the change made overdue fails; the controls' locks are not read; a lock it can't read can't be judged. Every test
passes `--today`: nothing reads a clock."""
from __future__ import annotations

import datetime
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("finder_dates_under_test", HERE.parent / "finder_dates.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

DAY = datetime.date(2030, 3, 10)


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def commit(repo: Path, files: dict[str, object]) -> str:
    for f, doc in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(doc if isinstance(doc, str) else json.dumps(doc))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c")
    return git(repo, "rev-parse", "HEAD")


def lock(**due: str) -> dict:
    return {"guarantees": {}, "choosing": {d: {"owner": "@proofs", "issue": "ISSUE-1", "due": day} for d, day in due.items()}}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    monkeypatch.chdir(root)
    return root


def test_an_entry_due_before_the_day_fails_and_names_its_way_out(repo, capsys):
    commit(repo, {"a/lean-audit.json": lock(**{"A.late": "2030-03-09", "A.today": "2030-03-10"}),
                  "b/lean-audit.json": lock(**{"B.later": "2030-04-01"})})
    doc = D.dates("HEAD", DAY)
    assert doc["listed"] == 3 and [f["definition"] for f in doc["failures"]] == ["A.late"]
    f = doc["failures"][0]
    assert (f["lock"], f["owner"], f["issue"], f["due"]) == ("a/lean-audit.json", "@proofs", "ISSUE-1", "2030-03-09")
    assert "restate A.late as a search" in f["fix"][0] and "moves its `due`" in f["fix"][1]
    assert D.main(["--today", "2030-03-10", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["kind"] == D.KIND
    assert D.main(["--today", "2030-03-09"]) == 0


def test_with_a_base_only_the_change_s_own_entry_fails(repo):
    base = commit(repo, {"a/lean-audit.json": lock(**{"A.old": "2030-03-01", "A.moved": "2030-05-01"})})
    commit(repo, {"a/lean-audit.json": lock(**{"A.old": "2030-03-01", "A.moved": "2030-03-02", "A.new": "2030-03-03"})})
    assert [f["definition"] for f in D.dates("HEAD", DAY)["failures"]] == ["A.moved", "A.new", "A.old"]
    assert [f["definition"] for f in D.dates("HEAD", DAY, base)["failures"]] == ["A.moved", "A.new"]


def test_the_controls_locks_are_not_read(repo):
    commit(repo, {f"{D.FIXTURES}control/lean-audit.json": lock(**{"C.late": "2000-01-01"}), "a/lean-audit.json": lock()})
    assert D.dates("HEAD", DAY)["failures"] == []


def test_a_lock_it_cannot_read_cannot_be_judged(repo, capsys):
    commit(repo, {"a/lean-audit.json": {"choosing": ["A.x"]}})
    assert D.main(["--today", "2030-03-10"]) == 2
    assert "not a map" in capsys.readouterr().err


def test_the_day_is_required():
    with pytest.raises(SystemExit):
        D.main([])
