"""`tools/verity/lean/finder_gate.py` on a repository it makes: a change that refuses more or lists less passes; one that
adds a `pass` row, drops a `refuse` row, a head or a position, adds or narrows a `fields` row, adds a `choosing` entry or
moves a `due` later fails, naming each; the change judged is the head's against its merge base with `main`, or the tip a
queue merge merged onto."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("finder_gate_under_test", HERE.parent / "finder_gate.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

TABLE = {"heads": {"H.cr": [1], "H.strict": [3, 4, 5]}, "fields": {"F.mk": ["out", "S"]}, "refuse": ["Classical.choice"],
         "pass": ["P.loop"]}
ENTRY = {"owner": "@proofs", "issue": "ISSUE-1", "due": "2030-03-10"}


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def commit(repo: Path, files: dict[str, object], msg: str = "c") -> str:
    for f, doc in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(json.dumps(doc))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def lock(**entries: dict) -> dict:
    return {"guarantees": {}, "choosing": entries}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """`main` with the table and one lock listing `D.a` and `D.b`, and a branch `pr` from it."""
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    commit(root, {G.TABLE: TABLE, "s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY})})
    git(root, "checkout", "-q", "-b", "pr")
    monkeypatch.chdir(root)
    return root


def whys(doc: dict) -> list[str]:
    return [f["why"] for f in doc["failures"]]


def test_refusing_more_and_listing_less_passes(repo):
    commit(repo, {G.TABLE: TABLE | {"heads": TABLE["heads"] | {"H.new": [0]}, "refuse": ["Classical.choice", "Nat.find"],
                                    "pass": [], "fields": {"F.mk": ["out", "S", "cost"]}},
                  "s/lean-audit.json": lock(**{"D.a": ENTRY | {"due": "2030-03-01"}})})
    assert G.gate("main", "HEAD")["failures"] == []
    assert G.main(["main", "HEAD"]) == 0


@pytest.mark.parametrize("change, said", [
    ({"pass": ["P.loop", "P.new"]}, "adds the `pass` row P.new"),
    ({"refuse": []}, "drops the `refuse` row Classical.choice"),
    ({"heads": {"H.strict": [3, 4, 5]}}, "drops the head H.cr"),
    ({"heads": {"H.cr": [1], "H.strict": [3, 5]}}, "drops the position 4 of H.strict"),
    ({"fields": {"F.mk": ["out", "S"], "G.mk": ["out"]}}, "adds the `fields` row G.mk"),
    ({"fields": {"F.mk": ["out"]}}, "narrows the `fields` row F.mk"),
])
def test_a_table_change_that_refuses_less_fails(repo, change, said):
    commit(repo, {G.TABLE: TABLE | change})
    assert [w for w in whys(G.gate("main", "HEAD")) if said in w], whys(G.gate("main", "HEAD"))


def test_an_added_entry_and_a_later_due_fail(repo, capsys):
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY | {"due": "2030-04-01"}, "D.b": ENTRY, "D.c": ENTRY}),
                  "t/lean-audit.json": lock(**{"D.a": ENTRY})})
    got = whys(G.gate("main", "HEAD"))
    assert len(got) == 3 and "moves D.a's `due` from 2030-03-10 to 2030-04-01" in got[0]
    assert "s/lean-audit.json lists D.c" in got[1] and "t/lean-audit.json lists D.a" in got[2]
    assert G.main(["main", "HEAD", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["kind"] == G.KIND


def test_the_change_is_judged_against_its_merge_base(repo):
    """A `main` that moved on since the branch adds an entry of its own: not this change's."""
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY})})
    git(repo, "checkout", "-q", "main")
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY, "D.z": ENTRY})})
    git(repo, "checkout", "-q", "pr")
    assert G.gate("main", "pr")["failures"] == []


def test_a_queue_merge_is_judged_against_the_tip_it_merged_onto(repo):
    git(repo, "checkout", "-q", "main")
    tip = commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY})})
    git(repo, "checkout", "-q", "pr")
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY, "D.y": ENTRY})})
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-X", "theirs", "-m", "Queue: #1 the change", "pr")
    doc = G.gate("main", "HEAD")
    assert doc["below"] == tip and [w for w in whys(doc) if "lists D.b" in w] and [w for w in whys(doc) if "lists D.y" in w]


def test_a_table_it_cannot_read_cannot_be_judged(repo, capsys):
    commit(repo, {G.TABLE: {"heads": []}})
    assert G.main(["main", "HEAD"]) == 2
    assert "not the walk's table" in capsys.readouterr().err
