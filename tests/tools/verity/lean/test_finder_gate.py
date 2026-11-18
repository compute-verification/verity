"""`tools/verity/lean/finder_gate.py` on a repository it makes: a change that refuses more or lists less passes; one that
adds a `pass` row, drops a `refuse` row, a head or a position, adds or narrows a `fields` row, adds a `choosing` entry or
moves a `due` later fails, naming each; the change judged is the head's against its merge base with `main`, whatever its
merges' subjects say; an entry a move re-keys is no addition; a record newly reaching an overdue entry fails; and drive's
copy of the script alone judges, in drive's form, through the queue's own `gates_at` and `run_gate`."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "finder_gate_under_test", HERE.parents[3] / "tools" / "verity" / "lean" / "finder_gate.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

TABLE = {"heads": {"H.cr": [1], "H.strict": [3, 4, 5]}, "fields": {"F.mk": ["out", "S"]}, "refuse": ["Classical.choice"],
         "pass": ["P.loop"], "types": ["Option"]}
ENTRY = {"owner": "@proofs", "issue": "ISSUE-1", "due": "2030-03-10"}


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def commit(repo: Path, files: dict[str, object], msg: str = "c") -> str:
    for f, doc in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(doc if isinstance(doc, str) else json.dumps(doc))
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
    ({"types": ["Option", "Real"]}, "adds the `types` row Real"),
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


def test_a_queue_subject_on_the_pr_s_own_merge_is_judged_against_the_merge_base(repo):
    """A PR that moves a due a year later, then merges `main` under a `Queue: #` subject, still fails."""
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY | {"due": "2031-03-10"}, "D.b": ENTRY})})
    git(repo, "checkout", "-q", "main")
    tip = commit(repo, {"t/lean-audit.json": lock()})
    git(repo, "checkout", "-q", "pr")
    git(repo, "merge", "-q", "--no-ff", "-m", "Queue: #99999 merge main", "main")
    doc = G.gate("main", "pr")
    assert doc["below"] == tip and [w for w in whys(doc) if "moves D.a's `due` from 2030-03-10 to 2031-03-10" in w]


def test_drive_s_copy_alone_judges_in_its_form(repo, tmp_path):
    """Drive runs the gate as `python3 PATH MAIN HEAD --json` from a copy of the file alone: exit 1 with
    `{"failures": [{"why", "fix"}]}`."""
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY, "D.c": ENTRY})})
    alone = tmp_path / "alone" / "finder_gate.py"
    alone.parent.mkdir()
    alone.write_text((HERE.parents[3] / "tools" / "verity" / "lean" / "finder_gate.py").read_text())
    r = subprocess.run([sys.executable, str(alone), "main", "HEAD", "--json"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    assert [(f["why"].split(" (")[0], len(f["fix"])) for f in json.loads(r.stdout)["failures"]] == [
        ("s/lean-audit.json lists D.c under `choosing`", 3)]


def test_a_table_it_cannot_read_cannot_be_judged(repo, capsys):
    commit(repo, {G.TABLE: {"heads": []}})
    assert G.main(["main", "HEAD"]) == 2
    assert "not the walk's table" in capsys.readouterr().err


def reads(**defs: str) -> dict:
    return {"M": {"definitions": defs}}


@pytest.mark.parametrize("old, new, moved, read_old, read_new, fails", [
    ("D.a", "D.n", None, {"D.a": "h1"}, {"D.n": "h1"}, False),
    ("D.a._f", "D.n._f", None, {"D.a": "h1"}, {"D.n": "h1"}, False),
    ("D.a", "E.a", {"as": "D.a", "sha256": "h1"}, {"D.a": "h1"}, {"E.a": "h9"}, False),
    ("D.a._f", "E.a._f", {"as": "D.a._f", "sha256": "h1"}, {"D.a": "h1"}, {"E.a": "h9"}, False),
    ("D.a", "D.n", None, {"D.a": "h1"}, {"D.n": "h2"}, True),
    ("D.a", "E.a", {"as": "D.a", "sha256": "h2"}, {"D.a": "h1"}, {"E.a": "h9"}, True),
    ("D.a", "E.a", None, {"D.a": "h1"}, {"E.a": "h9"}, True),
])
def test_an_entry_a_move_re_keys_is_no_addition(repo, old, new, moved, read_old, read_new, fails):
    """Re-keyed onto a dropped key whose `reads` hash it has, as built or under the old names (`moved`): no addition;
    otherwise one, whatever the change's moves file says."""
    git(repo, "checkout", "-q", "main")
    commit(repo, {"s/lean-audit.json": lock(**{old: ENTRY}) | {"reads": reads(**read_old)}})
    git(repo, "checkout", "-q", "-B", "pr")
    commit(repo, {"s/lean-audit.json": lock(**{new: ENTRY | ({"moved": moved} if moved else {})})
                  | {"reads": reads(**read_new)}, "tools/move/x_moves.json": {old: new}})
    assert [w.split(" (")[0] for w in whys(G.gate("main", "HEAD"))] == (
        [f"s/lean-audit.json lists {new} under `choosing`"] if fails else [])


def test_one_dropped_entry_re_keys_one_added_entry(repo):
    git(repo, "checkout", "-q", "main")
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY}) | {"reads": reads(**{"D.a": "h1"})}})
    git(repo, "checkout", "-q", "-B", "pr")
    commit(repo, {"s/lean-audit.json": lock(**{"D.b": ENTRY, "D.m": ENTRY, "D.n": ENTRY})
                  | {"reads": reads(**{"D.m": "h1", "D.n": "h1"})}})
    assert [w.split(" (")[0] for w in whys(G.gate("main", "HEAD"))] == ["s/lean-audit.json lists D.n under `choosing`"]


def test_a_moved_entry_s_due_owner_and_issue_are_held_as_its_old_one_s(repo):
    git(repo, "checkout", "-q", "main")
    commit(repo, {"s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY}) | {"reads": reads(**{"D.a": "h1"})}})
    git(repo, "checkout", "-q", "-B", "pr")
    moved = {"moved": {"as": "D.a", "sha256": "h1"}}
    commit(repo, {"s/lean-audit.json": lock(**{"E.a": ENTRY | moved | {"due": "2031-01-01", "owner": "@nobody"},
                                               "D.b": ENTRY | {"issue": "X-1"}}) | {"reads": reads(**{"E.a": "h9"})}})
    assert whys(G.gate("main", "HEAD")) == ["s/lean-audit.json changes D.b's `issue` from ISSUE-1 to X-1",
                                            "s/lean-audit.json moves E.a's `due` from 2030-03-10 to 2031-01-01",
                                            "s/lean-audit.json changes E.a's `owner` from @proofs to @nobody"]


@pytest.mark.parametrize("day, fails", [("2030-03-11", True), ("2030-03-10", False)])
def test_a_record_newly_reaching_an_overdue_entry_fails(repo, monkeypatch, day, fails):
    """A guarantee newly reaching `D.a` after its due day fails; one the moves file renames is not new."""
    held = lock(**{"D.a": ENTRY, "D.b": ENTRY}) | {"guarantees": {"G.t": {"premises": {"choosing": ["D.a"]}}}}
    git(repo, "checkout", "-q", "main")
    commit(repo, {"s/lean-audit.json": held})
    git(repo, "checkout", "-q", "-B", "pr")
    monkeypatch.setenv("GIT_COMMITTER_DATE", f"{day}T12:00:00Z")
    commit(repo, {"s/lean-audit.json": held | {"guarantees": {"H.t": {"premises": {"choosing": ["D.a"]}},
                                                              "G.u": {"premises": {"choosing": ["D.a"]}}}},
                  "tools/move/x_moves.json": {"G.t": "H.t"}})
    assert whys(G.gate("main", "HEAD")) == (
        ["s/lean-audit.json: G.u newly reaches D.a, whose `choosing` entry was due 2030-03-10"] if fails else [])


def test_a_move_whose_old_name_the_lock_still_holds_hides_no_reach(repo, monkeypatch):
    """A moves file `{D.b: D.a}` would rename `G.t`'s reach of `D.b` onto overdue `D.a`; `D.b` is still listed, so the
    move renames nothing and `G.t`'s new reach of `D.a` fails."""
    held = lock(**{"D.a": ENTRY, "D.b": ENTRY}) | {"guarantees": {"G.t": {"premises": {"choosing": ["D.b"]}}}}
    git(repo, "checkout", "-q", "main")
    commit(repo, {"s/lean-audit.json": held})
    git(repo, "checkout", "-q", "-B", "pr")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2030-03-12T12:00:00Z")
    commit(repo, {"s/lean-audit.json": held | {"guarantees": {"G.t": {"premises": {"choosing": ["D.a", "D.b"]}}}},
                  "tools/move/x_moves.json": {"D.b": "D.a"}})
    assert whys(G.gate("main", "HEAD")) == ["s/lean-audit.json: G.t newly reaches D.a, whose `choosing` entry was due "
                                            "2030-03-10"]


def test_drive_s_path_runs_main_s_copy(repo):
    """The preflight's own `gates_at` and `run_gate`, which runs `main`'s copy of the gate alone."""
    from research.queue import gates_at, run_gate

    gate_py = "tools/verity/lean/finder_gate.py"
    git(repo, "checkout", "-q", "main")
    main = commit(repo, {"tools/check/drive.toml": f'landing_gates = ["{gate_py}"]\n',
                         gate_py: (HERE.parents[3] / gate_py).read_text(),
                         "s/lean-audit.json": lock(**{"D.a": ENTRY, "D.b": ENTRY}) | {"reads": reads(**{"D.a": "h1"})}})
    git(repo, "checkout", "-q", "-B", "pr")
    e_a = ENTRY | {"moved": {"as": "D.a", "sha256": "h1"}}
    moved = commit(repo, {"s/lean-audit.json": lock(**{"E.a": e_a, "D.b": ENTRY}) | {"reads": reads(**{"E.a": "h9"})}})
    assert gates_at(repo, main) == (gate_py,)
    assert run_gate(repo, main, moved, gate_py)[0] == 0
    added = commit(repo, {"s/lean-audit.json": lock(**{"E.a": e_a, "D.b": ENTRY, "D.c": ENTRY})
                          | {"reads": reads(**{"E.a": "h9"})}})
    rc, out = run_gate(repo, main, added, gate_py)
    assert rc == 1 and [f["why"].split(" (")[0] for f in json.loads(out)["failures"]] == [
        "s/lean-audit.json lists D.c under `choosing`"]


def reaching(*gs: str) -> dict:
    return lock(**{"D.a": ENTRY, "D.b": ENTRY}) | {"guarantees": {g: {"premises": {"choosing": ["D.a"]}} for g in gs}}


@pytest.mark.parametrize("m_day, p_reaches", [("2030-03-09", False), ("2030-03-09", True), ("2030-03-12", False)])
def test_the_sync_judges_a_merge_s_overdue_reach_against_its_first_parent(repo, monkeypatch, m_day, p_reaches):
    """Drive's sync merges each member onto `next`, dated as it is made, and runs `main`'s gate on the merge: P merged
    above M after `D.a`'s due is refused for its own new reach only, never for M's, which M's own merge was judged on;
    P's head at the preflight passes."""
    from research.queue import gate_refusal, run_gate

    gate_py = "tools/verity/lean/finder_gate.py"
    git(repo, "checkout", "-q", "main")
    main = commit(repo, {"tools/check/drive.toml": f'landing_gates = ["{gate_py}"]\n',
                         gate_py: (HERE.parents[3] / gate_py).read_text(), "s/lean-audit.json": reaching("G.t"),
                         "t/lean-audit.json": reaching()})
    git(repo, "checkout", "-q", "-B", "m")
    commit(repo, {"s/lean-audit.json": reaching("G.t", "G.m")})
    git(repo, "checkout", "-q", "-B", "p", "main")
    p = commit(repo, {"README.md": "docs\n"} | ({"t/lean-audit.json": reaching("G.p")} if p_reaches else {}))
    git(repo, "checkout", "-q", "-B", "next", "main")
    refused = {}
    for branch, day in (("m", m_day), ("p", "2030-03-12")):
        monkeypatch.setenv("GIT_COMMITTER_DATE", f"{day}T12:00:00Z")
        git(repo, "merge", "-q", "--no-ff", "-m", f"Queue: {branch}", branch)
        got, unjudged = gate_refusal(repo, main, git(repo, "rev-parse", "HEAD"), (gate_py,), onto="next")
        assert unjudged == []
        refused[branch] = [g.split(": ", 2)[2].split(",")[0] for g in got]
    assert refused == {"m": [] if m_day < "2030-03-11" else ["G.m newly reaches D.a"],
                       "p": ["G.p newly reaches D.a"] if p_reaches else []}
    assert run_gate(repo, main, p, gate_py)[0] == 0


def test_generated_is_the_check_s():
    check = importlib.util.spec_from_file_location("lean_audit_gen", HERE.parents[3] / "tools/verity/lean/check.py")
    A = importlib.util.module_from_spec(check)
    check.loader.exec_module(A)
    assert G.GENERATED.pattern == A.GENERATED.pattern
