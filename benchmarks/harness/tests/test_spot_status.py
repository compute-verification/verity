"""`spot_status`: each resampling hole is closed only where its fix's commit is an ancestor of the commit a run's record
names, decided in a tiny git repository of made-up commits; a run whose code can't be read is not known; the label is
`SPOT soundness` only when both holes are closed; and the theorem's open premises are the fixture lock's."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_bench import cli, prove_records
from verity_bench import exfiltration as E
from verity_bench import spot_status as S

#: a lock's shape under a name no Lean tool takes for a package's lock
LOCK = Path(__file__).parent / "fixtures" / "spot_status" / "lock.json"
REDRAWN, RECOMMITTED = S.HOLES
BASE, ONE, BOTH = "r20261112-000001-aaaa", "r20261112-000002-bbbb", "r20261112-000003-cccc"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository of three commits in a line (the base, the first fix, the second) and a side commit off the base,
    each adding a file of its own, with the fixes' data file naming the two fixes."""

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    commits = {}

    def commit(name: str) -> str:
        (root / f"{name}.txt").write_text(f"the {name} commit's line\n")
        _git(root, "add", f"{name}.txt")
        _git(root, "commit", "-q", "-m", name)
        return _git(root, "rev-parse", "HEAD")

    for name in ("base", "fix1", "fix2"):
        commits[name] = commit(name)
    _git(root, "checkout", "-q", "-b", "side", commits["base"])
    commits["side"] = commit("side")
    data = tmp_path / "spot_holes.json"
    data.write_text(json.dumps({"fixes": {REDRAWN: commits["fix1"], RECOMMITTED: commits["fix2"]}}))
    monkeypatch.setattr(S, "FIXES", data)
    monkeypatch.setattr(S, "LOCK", LOCK)
    return SimpleNamespace(root=root, data=data, **commits)


class Store:
    def __init__(self, sources: dict[str, dict | None]):
        self.sources = sources

    def get_attempt(self, run: str) -> dict | None:
        src = self.sources[run]
        return None if src is None else {"id": run, "source": src}


def _status(repo, **runs: dict | None) -> dict:
    return S.status(list(runs), store=Store(runs), root=repo.root)


def _by_hole(doc: dict) -> dict[str, str]:
    return {h["hole"]: h["status"] for h in doc["holes"]}


def test_both_holes_open_at_the_base(repo):
    doc = _status(repo, **{BASE: {"commit": repo.base, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert "- label: **SPOT**\n- redrawn sample: open\n" in text and "- re-committed execution phase: open\n" in text
    assert f"  - at every commit read: the fix's commit {repo.fix1[:12]} is not an ancestor" in text
    assert "proved" not in text


def test_one_hole_closed_where_its_fix_is_an_ancestor(repo):
    doc = _status(repo, **{ONE: {"commit": repo.fix1, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    assert "- redrawn sample: closed\n- re-committed execution phase: open\n" in S.render(doc)


def test_both_closed_flips_the_label(repo):
    doc = _status(repo, **{BOTH: {"commit": repo.fix2, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "closed"} and doc["label"] == "SPOT soundness"
    text = S.render(doc)
    assert "- label: **SPOT soundness**\n- redrawn sample: closed\n- re-committed execution phase: closed\n" in text
    assert f"- runs read: 1, at 1 commit: `{repo.fix2[:12]}` (1 run)" in text


def test_a_side_commit_holds_neither_fix(repo):
    """An ancestor, not a later date: a commit off the base after both fixes still has neither."""

    doc = _status(repo, **{BOTH: {"commit": repo.side, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"}


def test_one_open_run_holds_the_hole_open_over_the_rest(repo):
    doc = _status(repo, **{BASE: {"commit": repo.base, "dirty": False}, BOTH: {"commit": repo.fix2, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    assert f"  - at `{repo.fix2[:12]}`: closed\n" in S.render(doc)


def test_a_commit_this_repository_lacks_is_not_known(repo):
    lost = "d" * 40
    doc = _status(repo, **{BOTH: {"commit": lost, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"} and doc["label"] == "SPOT"
    assert f"doesn't hold the run's commit {lost} (`git fetch origin {lost}`)" in S.render(doc)


def test_a_run_whose_code_cant_be_read_is_not_known_and_never_closes(repo):
    doc = _status(repo, **{BOTH: {"commit": repo.fix2, "dirty": False}, ONE: {"commit": repo.fix2, "dirty": True},
                           BASE: None})
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert f"  - {ONE}: not known: it ran a dirty tree at {repo.fix2[:12]}" in text
    assert f"  - {BASE}: not known: the store holds no record of it" in text
    assert _by_hole(S.status([], store=Store({}), root=repo.root)) == {REDRAWN: "not known", RECOMMITTED: "not known"}


def test_a_hole_with_no_fix_declared_is_open(repo):
    repo.data.write_text(json.dumps({"fixes": {REDRAWN: repo.fix1}}))
    doc = _status(repo, **{BOTH: {"commit": repo.fix2, "dirty": False}})
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    assert "  - at every commit read: no fix's commit is declared" in S.render(doc)
    lost = _status(repo, **{BOTH: {"commit": "d" * 40, "dirty": False}})
    assert _by_hole(lost) == {REDRAWN: "not known", RECOMMITTED: "open"}
    assert _by_hole(S.status([], store=Store({}), root=repo.root)) == {REDRAWN: "not known", RECOMMITTED: "open"}


#: a line a fix adds that counts toward its presence (8 characters or more, not a comment alone)
FIX = "refuse_redraw(sample)"
#: lines a fix removes (`OLD`) and moves within its file (`MOVED`)
OLD, MOVED = "accept_redraw(sample)", "register(sample, salt)"


@pytest.fixture
def changes(tmp_path):
    """A repository whose fix changes a file, for the checks that read what a run's tree holds: `commit(text, msg,
    path)` writes the file (`f.txt` unless named) and commits it, returning the commit."""

    root = tmp_path / "changes"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")

    def commit(text: str, msg: str, path: str = "f.txt") -> str:
        (root / path).write_text(text)
        _git(root, "add", path)
        _git(root, "commit", "-q", "-m", msg)
        return _git(root, "rev-parse", "HEAD")

    commit("a\n", "base")
    return SimpleNamespace(root=root, commit=commit, head=lambda: _git(root, "rev-parse", "HEAD"))


def test_a_reverted_fix_is_open_and_names_its_revert(changes):
    """An ancestor is not enough: a fix reverted after it is no longer in the run's code."""

    fix = changes.commit(f"a\n{FIX}\n", "fix")
    assert S.closes(changes.root, fix, fix) == {"status": "closed", "why": None}
    _git(changes.root, "revert", "--no-edit", fix)
    revert = changes.head()
    got = S.closes(changes.root, fix, revert)
    assert got["status"] == "open" and got["why"] == f"reverted by {revert[:12]}, which reverts the fix"
    assert S.holes(revert, changes.root, {REDRAWN: fix, RECOMMITTED: fix})[REDRAWN]["status"] == "open"


def test_a_reverted_merge_that_brought_the_fix_is_open(changes):
    """`git revert -m 1` of the landing's merge, as a red step's culprit is reverted, takes the fix out with it."""

    _git(changes.root, "checkout", "-q", "-b", "side")
    fix = changes.commit(f"a\n{FIX}\n", "fix")
    _git(changes.root, "checkout", "-q", "main")
    changes.commit("other\n", "other", "g.txt")
    _git(changes.root, "merge", "-q", "--no-ff", "-m", "land the fix", "side")
    merge = changes.head()
    assert S.closes(changes.root, fix, merge)["status"] == "closed"
    _git(changes.root, "revert", "--no-edit", "-m", "1", merge)
    revert = changes.head()
    got = S.closes(changes.root, fix, revert)
    assert got["status"] == "open"
    assert got["why"] == f"reverted by {revert[:12]}, which reverts the merge {merge[:12]} that brought it in"


def test_a_revert_of_the_revert_closes_the_hole_again(changes):
    """A revert a later revert undoes doesn't count; undoing that one puts the first revert back in force."""

    fix = changes.commit(f"a\n{FIX}\n", "fix")
    _git(changes.root, "revert", "--no-edit", fix)
    first = changes.head()
    _git(changes.root, "revert", "--no-edit", "HEAD")
    assert S.closes(changes.root, fix, changes.head()) == {"status": "closed", "why": None}
    _git(changes.root, "revert", "--no-edit", "HEAD")
    again = S.closes(changes.root, fix, changes.head())
    assert again == {"status": "open", "why": f"reverted by {first[:12]}, which reverts the fix"}


def test_a_fix_whose_added_lines_were_rewritten_reads_not_known_and_declaring_the_rewrite_reads_closed(changes):
    """The check under the revert line: a tree that lost a line the fix added, with no revert's message, can't be
    told to hold it, until the commit that rewrote it is declared as keeping the fix."""

    fix = changes.commit(f"a\n{FIX}\n", "fix")
    rewrite = changes.commit("a\nrefuse_redraw(sample, strict=True)\n", "rewrite the fix's line")
    run = changes.commit("other\n", "later", "g.txt")
    short = _git(changes.root, "rev-parse", "--short", rewrite)
    assert S.closes(changes.root, fix, run) == {"status": "not known", "why": (
        f"1 of the 1 line the fix {fix[:12]} added to f.txt is not in f.txt at the run's commit, among them `{FIX}`. "
        f"{short} changed those lines since: if {short} keeps the fix, add it to this hole's commits in "
        "spot_holes.json")}
    assert S.closes(changes.root, [fix, rewrite], run) == {"status": "closed", "why": None}
    assert S.holes(run, changes.root, {REDRAWN: (fix, rewrite), RECOMMITTED: (fix, rewrite)})[REDRAWN]["status"] == "closed"
    _git(changes.root, "revert", "--no-edit", rewrite)
    got = S.closes(changes.root, [fix, rewrite], changes.head())
    assert got == {"status": "open",
                   "why": f"reverted by {changes.head()[:12]}, which reverts the fix's later commit {rewrite[:12]}"}
    with pytest.raises(ValueError, match=f"the commits of the fix {rewrite} are out of order: {rewrite[:12]} is not an "
                                         f"ancestor of {fix[:12]}"):
        S.closes(changes.root, [rewrite, fix], run)


def test_a_merge_declared_as_the_fix_says_its_lines_are_all_it_brings(changes):
    """A merge's lines are everything it brings to its first parent, so where one is gone the reason says to declare
    the landing merge."""

    _git(changes.root, "checkout", "-q", "-b", "side")
    changes.commit(f"a\n{FIX}\n", "fix")
    _git(changes.root, "checkout", "-q", "main")
    changes.commit("other\n", "other", "g.txt")
    _git(changes.root, "merge", "-q", "--no-ff", "-m", "land the fix", "side")
    merge = changes.head()
    assert S.closes(changes.root, merge, merge)["status"] == "closed"
    changes.commit("a\n", "drop the fix's line")
    got = S.closes(changes.root, merge, changes.head())
    assert got["status"] == "not known"
    assert got["why"].startswith(f"The fix's commit {merge[:12]} is a merge, so the lines it added are all it brings "
                                 "to its first parent, in 1 file: declare the commit whose first parent is the base its change "
                                 "landed on (its landing merge), or the fix's own commits as a range `A..B`. ")


def test_a_deletion_only_fix_is_closed_while_its_lines_stay_absent(changes):
    """A fix that only deletes is decided by its removed lines alone: closed while they stay out, not known once one
    is back."""

    changes.commit(f"a\n{OLD}\n", "the hole")
    fix = changes.commit("a\n", "fix: delete the redraw")
    run = changes.commit("other\n", "later", "g.txt")
    assert S.closes(changes.root, fix, run) == {"status": "closed", "why": None}
    back = changes.commit(f"a\n{OLD}\n", "put the line back")
    short = _git(changes.root, "rev-parse", "--short", back)
    assert S.closes(changes.root, fix, back) == {"status": "not known", "why": (
        f"1 of the 1 line the fix {fix[:12]} removed from f.txt is in f.txt again at the run's commit, among them "
        f"`{OLD}`. {short} changed those lines since: if {short} keeps the fix, add it to this hole's commits in "
        "spot_holes.json")}


def test_a_fix_that_deletes_a_file_whole_is_not_known_once_the_file_is_back(changes):
    """A file the fix deletes is read under its path before the fix: its lines stay expected absent there, so restoring
    it is caught though the fix's other file still holds its change."""

    changes.commit(f"{OLD}\n", "the hole", "bad.py")
    changes.commit("g = 0\n", "its caller", "g.py")
    _git(changes.root, "rm", "-q", "bad.py")
    fix = changes.commit(f"{FIX}\n", "fix: delete bad.py and refuse in g.py", "g.py")
    run = changes.commit("other\n", "later", "h.txt")
    assert S.closes(changes.root, fix, run) == {"status": "closed", "why": None}
    back = changes.commit(f"{OLD}\n", "restore bad.py", "bad.py")
    got = S.closes(changes.root, fix, back)
    assert got["status"] == "not known"
    assert got["why"].startswith(f"1 of the 1 line the fix {fix[:12]} removed from bad.py is in bad.py again at the "
                                 f"run's commit, among them `{OLD}`.")


def test_a_fix_with_no_line_that_counts_is_never_closed(changes):
    """Short lines, blank lines and comments alone leave nothing to look for, so the run's tree can't be told to hold
    the fix."""

    fix = changes.commit("a\nx = 1\n\n# a comment alone, long enough\n", "fix in short lines")
    assert S.closes(changes.root, fix, fix) == {"status": "not known", "why": (
        f"the fix {fix[:12]} adds no line this check can look for, and removes none (a line counts stripped, from 8 "
        "characters, and not when it is a comment alone), so whether the run's tree holds it can't be told")}


def test_a_hand_revert_restoring_a_removed_line_is_not_known(changes):
    """Putting back a line the fix removed, with no revert's message and the fix's own line kept, is caught; a line the
    fix only moved within the file isn't one it removed."""

    changes.commit(f"{MOVED}\na\n{OLD}\n", "the hole")
    fix = changes.commit(f"a\n{MOVED}\n{FIX}\n", "fix: replace the redraw")
    assert S.closes(changes.root, fix, fix) == {"status": "closed", "why": None}
    back = changes.commit(f"a\n{MOVED}\n{FIX}\n{OLD}\n", "restore the redraw by hand")
    got = S.closes(changes.root, fix, back)
    assert got["status"] == "not known"
    assert got["why"].startswith(f"1 of the 1 line the fix {fix[:12]} removed from f.txt is in f.txt again at the "
                                 f"run's commit, among them `{OLD}`.")


def test_a_fix_whose_files_changed_elsewhere_reads_closed(changes):
    """Context churn around the fix (its neighbours rewritten, its line reindented, its file renamed) and the loss of
    lines that don't count (a comment alone, a short line) leave the fix's added lines in place, so it holds."""

    kept = "".join(f"line {i} that stays as it was\n" for i in range(8))
    fix = changes.commit(f"a\n{kept}# refuse a redraw\nx = 1\n{FIX}\nb\n", "fix")
    changes.commit(f"header line rewritten\n{kept}if strict:\n    {FIX}\ntrailer rewritten\n", "churn")
    _git(changes.root, "mv", "f.txt", "moved.txt")
    _git(changes.root, "commit", "-q", "-m", "rename")
    run = changes.head()
    assert S.closes(changes.root, fix, run) == {"status": "closed", "why": None}
    _git(changes.root, "rm", "-q", "moved.txt")
    _git(changes.root, "commit", "-q", "-m", "drop it")
    gone = S.closes(changes.root, fix, changes.head())
    assert gone["status"] == "not known"
    assert gone["why"].startswith(f"f.txt is gone at the run's commit, with the 9 lines the fix {fix[:12]} added to it")


def test_the_fixes_file_names_only_holes_by_full_commit(repo):
    repo.data.write_text(json.dumps({"fixes": {"another hole": repo.fix1}}))
    with pytest.raises(ValueError, match="names another hole, which is no hole of"):
        S.fixes(repo.data)
    repo.data.write_text(json.dumps({"fixes": {REDRAWN: repo.fix1[:12]}}))
    with pytest.raises(ValueError, match="not a full commit id"):
        S.fixes(repo.data)
    span = f"{repo.base}..{repo.fix1}"
    repo.data.write_text(json.dumps({"fixes": {REDRAWN: [span, repo.fix2]}}))
    assert S.fixes(repo.data) == {REDRAWN: (span, repo.fix2)}
    for bad in ([f"..{repo.fix1}"], [f"{repo.base}.."], [repo.fix1, span]):
        repo.data.write_text(json.dumps({"fixes": {REDRAWN: bad}}))
        with pytest.raises(ValueError, match="not a full commit id"):
            S.fixes(repo.data)


def test_a_range_fix_is_closed_where_its_tip_is_held_and_its_change_kept(changes):
    """A fix of several commits is the range ``A..B``: its change is ``git diff A B``, every commit of it counts."""

    base = changes.head()
    changes.commit(f"a\n{FIX}\n", "the fix's first commit")
    tip = changes.commit(f"{MOVED}\n", "the fix's second commit", "h.txt")
    run = changes.commit("other\n", "later", "g.txt")
    assert S.closes(changes.root, [f"{base}..{tip}"], run) == {"status": "closed", "why": None}
    assert S.closes(changes.root, f"{base}..{tip}", base) == {"status": "open",
                                                             "why": f"the fix's commit {tip[:12]} is not an ancestor"}
    with pytest.raises(ValueError, match="out of order"):
        S.closes(changes.root, f"{tip}..{base}", run)


def test_a_range_fix_rewritten_later_is_closed_with_the_rewrite_declared_and_not_known_without(changes):
    """A later commit's hunk that rewrites the fix's lines moves the lines it removes to expected absent and the ones
    it adds to expected present; its other hunks expect nothing; a run that doesn't hold it holds the fix as it was."""

    base = changes.head()
    changes.commit(f"a\n{FIX}\nb\n", "the fix's first commit")
    tip = changes.commit(f"{MOVED}\n", "the fix's second commit", "h.txt")
    rewrite = changes.commit("a\nrefuse_redraw(sample, strict=True)\nb\nunrelated_line = 1\n", "rewrite the fix's line")
    run = changes.commit("a\nrefuse_redraw(sample, strict=True)\nb\nunrelated_line = 2\n", "change the unrelated line")
    span = f"{base}..{tip}"
    undeclared = S.closes(changes.root, [span], run)
    short = _git(changes.root, "rev-parse", "--short", rewrite)
    assert undeclared == {"status": "not known", "why": (
        f"1 of the 1 line the fix {base[:12]}..{tip[:12]} added to f.txt is not in f.txt at the run's commit, among "
        f"them `{FIX}`. {short} changed those lines since: if {short} keeps the fix, add it to this hole's commits in "
        "spot_holes.json")}
    assert S.closes(changes.root, [span, rewrite], run) == {"status": "closed", "why": None}
    assert S.closes(changes.root, [span, rewrite], tip) == {"status": "closed", "why": None}
    changes.commit(f"a\nrefuse_redraw(sample, strict=True)\n{FIX}\nb\n", "the old line back beside the new")
    back = S.closes(changes.root, [span, rewrite], changes.head())
    assert back["status"] == "not known" and back["why"].startswith(
        f"1 of the 1 line the fix {base[:12]}..{tip[:12]} as {rewrite[:12]} keeps it removed from f.txt is in f.txt "
        "again at the run's commit")


def test_a_range_whose_inner_commit_is_reverted_is_open(changes):
    base = changes.head()
    first = changes.commit(f"a\n{FIX}\n", "the fix's first commit")
    tip = changes.commit(f"{MOVED}\n", "the fix's second commit", "h.txt")
    _git(changes.root, "revert", "--no-edit", first)
    revert = changes.head()
    assert S.closes(changes.root, [f"{base}..{tip}"], revert) == {"status": "open", "why": (
        f"reverted by {revert[:12]}, which reverts the fix's commit {first[:12]} of {base[:12]}..{tip[:12]}")}


def test_the_tree_declares_only_holes():
    assert set(S.fixes(S.FIXES)) <= set(S.HOLES)


def test_the_theorem_and_its_open_premises_are_the_locks(tmp_path):
    t = S.theorem(tmp_path, LOCK)
    assert t["name"] == "Verity.Sampled.remote_soundness_deployedReady"
    assert t["open_premises"] == ["Verity.Sampled.Instance.SessionsReady", "Verity.Vocab.PouwSound"]
    lock = json.loads(LOCK.read_text())
    lock["guarantees"]["Verity.Sampled.onsite_soundness_deployedReady"] = {"premises": {"open": ["Verity.Vocab.WipeSound"]}}
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    assert S.theorem(tmp_path, tmp_path / "lock.json")["name"] == "Verity.Sampled.onsite_soundness_deployedReady"
    del lock["guarantees"]["Verity.Sampled.onsite_soundness_deployedReady"]
    del lock["guarantees"]["Verity.Sampled.remote_soundness_deployedReady"]
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    assert S.theorem(tmp_path, tmp_path / "lock.json")["name"] is None


def test_the_block_cites_the_theorem_with_its_open_premises(repo):
    text = S.render(_status(repo, **{BASE: {"commit": repo.base, "dirty": False}}))
    assert ("- theorem: `Verity.Sampled.remote_soundness_deployedReady`, on the 2 open premises its record in "
            f"`{LOCK}` takes: `Verity.Sampled.Instance.SessionsReady`, `Verity.Vocab.PouwSound`") in text


def _entry(run: str, commit: str | None, **more) -> dict:
    """A run's entry in a served record's `runs`, as its reader writes it."""

    return {"run": run, "commit": commit, "dirty": False, "why": None, "machine": "a-made-up-machine"} | more


def _served(commit: str | None, **runs: dict | None) -> dict:
    """A served record as its reader gives it, its second stage at ``commit``, with only the runs ``runs`` given."""

    return {"run": BOTH, "commit": commit, "runs": dict.fromkeys(S.SERVED_RUNS) | runs}


def test_a_served_record_is_read_at_its_own_commits(repo):
    """The block over a served run reads the commit its record names and each of its four runs' commits, listed; with
    all four read, a hole whose fix is an ancestor of every one is closed, and both closed is `SPOT soundness`."""

    record = _served(repo.fix1, **{name: _entry(ONE, repo.fix1) for name in S.SERVED_RUNS})
    doc = S.served(record, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "closed", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert "is an ancestor of the commit the served run's record names and of each of its runs' commits." in text
    assert "- redrawn sample: closed\n- re-committed execution phase: open\n" in text
    assert (f"- the code {BOTH} ran: its record's `commit`, the second stage's, `{repo.fix1[:12]}`; its runs:\n"
            + "".join(f"  - {name}: {ONE} at `{repo.fix1[:12]}` on a-made-up-machine\n" for name in S.SERVED_RUNS)) in text
    assert "- runs read:" not in text and "not given" not in text
    both = _served(repo.fix2, **{name: _entry(BOTH, repo.fix2) for name in S.SERVED_RUNS})
    assert S.served(both, root=repo.root)["label"] == "SPOT soundness"


def test_a_hole_never_reads_closed_where_some_of_the_served_runs_were_not_given(repo):
    """The rehearsal's shape: the reader was given only the first and second stages, so the serve and the replay are not
    given and no commit of theirs is read. A hole whose fix is an ancestor of every commit read is closed at the runs
    read, naming those not given, never bare closed, and the label stays `SPOT` with both holes so."""

    record = _served(repo.fix2, first=_entry(ONE, repo.fix2), second=_entry(BOTH, repo.fix2))
    doc = S.served(record, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: S.CLOSED_READ, RECOMMITTED: S.CLOSED_READ} and doc["label"] == "SPOT"
    assert (doc["served"]["read"], doc["served"]["not_given"]) == (["first", "second"], ["serve", "replay"])
    text = S.render(doc)
    assert ("- label: **SPOT**\n"
            "- redrawn sample: closed at the runs read (first, second); serve, replay not given\n"
            "- re-committed execution phase: closed at the runs read (first, second); serve, replay not given\n") in text
    assert (f"  - serve: {S.NOT_GIVEN}\n  - first: {ONE} at `{repo.fix2[:12]}` on a-made-up-machine\n"
            f"  - replay: {S.NOT_GIVEN}\n  - second: {BOTH} at `{repo.fix2[:12]}` on a-made-up-machine\n") in text
    assert ": closed\n" not in text and "SPOT soundness**" not in text
    # an open hole stays open, and one the read commits can't settle stays not known: only closed is qualified
    one = S.served(_served(repo.fix1, first=_entry(ONE, repo.fix1), second=_entry(BOTH, repo.fix1)), root=repo.root)
    assert _by_hole(one) == {REDRAWN: S.CLOSED_READ, RECOMMITTED: "open"}
    lost = S.served(_served("d" * 40, second=_entry(BOTH, "d" * 40)), root=repo.root)
    assert _by_hole(lost) == {REDRAWN: "not known", RECOMMITTED: "not known"}


def test_a_hole_is_closed_on_a_served_run_only_at_every_one_of_its_commits(repo):
    """One run at a commit that lacks a fix holds the hole open over the record's own commit, which holds it."""

    record = _served(repo.fix2, first=_entry(ONE, repo.base), second=_entry(BOTH, repo.fix2))
    doc = S.served(record, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    text = S.render(doc)
    assert (f"  - at `{repo.fix2[:12]}` (second): closed\n" in text
            and f"  - at `{repo.base[:12]}` (first): open: the fix's commit" in text)


def test_a_served_run_passes_each_runs_open_reverted_and_not_known_why_through(changes, tmp_path, monkeypatch):
    """Each served run's commit is read by `closes`, and its why reaches the record's runs and the print, by run: an
    ancestor is closed, a reverted fix is open naming its revert, and a tree that rewrote the fix's lines is not known,
    naming the commit to declare."""

    fix = changes.commit(f"a\n{FIX}\n", "fix")
    _git(changes.root, "checkout", "-q", "-b", "reverted")
    _git(changes.root, "revert", "--no-edit", fix)
    revert = changes.head()
    _git(changes.root, "checkout", "-q", "main")
    rewrite = changes.commit("a\nrewritten\n", "rewrite the fix's line")
    (data := tmp_path / "spot_holes.json").write_text(json.dumps({"fixes": {REDRAWN: fix, RECOMMITTED: fix}}))
    monkeypatch.setattr(S, "FIXES", data)
    monkeypatch.setattr(S, "LOCK", LOCK)
    record = _served(rewrite, first=_entry(ONE, fix), replay=_entry(BASE, revert), second=_entry(BOTH, rewrite))
    doc = S.served(record, root=changes.root)
    assert _by_hole(doc) == {REDRAWN: "open", RECOMMITTED: "open"} and doc["label"] == "SPOT"
    runs = {r["name"]: r["holes"].get(REDRAWN) for r in doc["served"]["runs"]}
    reverted = f"reverted by {revert[:12]}, which reverts the fix"
    assert runs["serve"] is None and runs["first"] == {"status": "closed", "why": None}
    assert runs["replay"] == {"status": "open", "why": reverted}
    assert runs["second"]["status"] == "not known" and runs["second"]["why"].startswith(
        f"1 of the 1 line the fix {fix[:12]} added to f.txt is not in f.txt at the run's commit, among them `{FIX}`. ")
    text = S.render(doc)
    assert (f"- redrawn sample: open\n  - at `{rewrite[:12]}` (second): not known: 1 of the 1 line the fix" in text
            and f"  - at `{fix[:12]}` (first): closed\n  - at `{revert[:12]}` (replay): open: {reverted}\n" in text)
    assert "add it to this hole's commits in spot_holes.json" in text


def test_a_served_commit_this_repository_lacks_or_a_run_with_no_commit_is_not_known(repo):
    lost = "d" * 40
    doc = S.served(_served(lost, second=_entry(BOTH, lost)), root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"}
    assert f"doesn't hold the run's commit {lost} (`git fetch origin {lost}`)" in S.render(doc)
    nameless = _served(repo.fix2, first=_entry(ONE, None, why="its directory holds no `job.json`"),
                       replay=_entry(BASE, repo.fix2, dirty=True), second=_entry(BOTH, repo.fix2))
    doc = S.served(nameless, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"} and doc["label"] == "SPOT"
    # a dirty tree's commit names no code it ran: its entry holds no hole, though the second is read at that commit
    runs = {r["name"]: r["holes"] for r in doc["served"]["runs"]}
    assert runs["replay"] == {} and runs["first"] == {} and runs["serve"] == {}
    assert {h: s["status"] for h, s in runs["second"].items()} == {REDRAWN: "closed", RECOMMITTED: "closed"}
    text = S.render(doc)
    assert f"  - first: {ONE} on a-made-up-machine: not known: its directory holds no `job.json`\n" in text
    assert f"  - replay: {BASE} at `{repo.fix2[:12]}` on a-made-up-machine: not known: it ran a dirty tree at" in text
    doc = S.served({"run": BOTH, "runs": {"second": _entry(BOTH, repo.fix2)}}, root=repo.root)
    assert _by_hole(doc) == {REDRAWN: "not known", RECOMMITTED: "not known"}
    assert ("its record's `commit`, the second stage's, not known: the record names no `commit`, the commit its second "
            "stage ran; its runs:") in S.render(doc)
    for bad, why in (({"runs": {"third": None}}, "an object of serve, first, replay, second, not third"),
                     ({"runs": {"first": "r1"}}, "`runs.first` is a run's entry, one JSON object, not str")):
        with pytest.raises(ValueError, match=re.escape(why)):
            S.served({"run": BOTH, "commit": repo.fix2} | bad, root=repo.root)


def test_exfiltration_reads_the_holes_at_a_served_records_commit(repo, tmp_path, monkeypatch, capsys):
    """`verity bench exfiltration --served-run` reads the holes at the served run's own commits, not at the planner's
    runs of record (here at the base, where both are open); without it, at those runs. Proofs' section on what the
    audit showed (`verity_bench.served_row`) is printed between the figures and the holes where the tree holds it, and
    is skipped where it doesn't."""

    store = Store({BASE: {"commit": repo.base, "dirty": False}})
    monkeypatch.setattr(S.jobs, "StoreCompletion", lambda **k: SimpleNamespace(store=store))
    monkeypatch.setattr(S.jobs, "repository_root", lambda *a: repo.root)
    monkeypatch.setattr(E, "build", lambda **k: {"runs": [BASE]})
    monkeypatch.setattr(E, "render", lambda doc: "the figures")
    (served := tmp_path / "served.json").write_text(json.dumps(
        _served(repo.fix2, **{name: _entry(BOTH, repo.fix2) for name in S.SERVED_RUNS})))

    find_spec, name = importlib.util.find_spec, "verity_bench.served_row"
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == name else find_spec(n, *a))
    assert E.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("the figures\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- label: **SPOT**\n- redrawn sample: open\n" in out and "- runs read: 1, at 1 commit" in out
    assert E.main(["--served-run", str(served)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("the figures\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- label: **SPOT soundness**\n- redrawn sample: closed\n- re-committed execution phase: closed\n" in out
    assert f"its record's `commit`, the second stage's, `{repo.fix2[:12]}`; its runs:\n" in out

    section = types.ModuleType(name)
    section.account = lambda record: {"run": record["run"]}
    section.render = lambda shown: f"## what {shown['run']}'s audit showed"
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: importlib.machinery.ModuleSpec(n, None)
                        if n == name else find_spec(n, *a))
    monkeypatch.setitem(sys.modules, name, section)
    monkeypatch.setattr(sys.modules["verity_bench"], "served_row", section, raising=False)
    assert E.main(["--served-run", str(served)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"the figures\n\n## what {BOTH}'s audit showed\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- label: **SPOT soundness**\n" in out
    assert E.main(["--json", "--served-run", str(served)]) == 0
    held = json.loads(capsys.readouterr().out)
    assert held["served_run"] == {"run": BOTH} and held["spot_holes"]["served"]["commit"] == repo.fix2
    assert E.main(["--served-run", str(tmp_path / "none.json")]) == 2


def test_prove_spot_prints_the_block_over_its_input_runs(repo, monkeypatch, capsys):
    """`verity bench prove --system spot` reads the runs among its inputs (not its arts) and prints the block after its
    table, and holds it in `--json`; `--system flock` prints none."""

    doc = {"inputs": [{"targets": ["art:" + "a" * 64, BOTH]}, {"targets": [BOTH, ONE]}]}
    monkeypatch.setattr(prove_records, "build", lambda *a, **k: json.loads(json.dumps(doc)))
    monkeypatch.setattr(prove_records, "render", lambda d, rows: "the table\n")
    store = Store({BOTH: {"commit": repo.fix2, "dirty": False}, ONE: {"commit": repo.fix1, "dirty": False}})
    monkeypatch.setattr(S.jobs, "StoreCompletion", lambda **k: SimpleNamespace(store=store))
    monkeypatch.setattr(S.jobs, "repository_root", lambda *a: repo.root)
    args = argparse.Namespace(system="spot", workload="glm53", sampling=None, provers=1, cores=10, local_only=True,
                              steady_rows=None, json=False, rows=None)
    assert cli._prove(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("the table\n\n## SPOT's resampling holes in the code that ran\n")
    assert "- redrawn sample: closed\n- re-committed execution phase: open\n" in out and "- runs read: 2, at 2 commits" in out
    assert cli._prove(argparse.Namespace(**{**vars(args), "json": True})) == 0
    held = json.loads(capsys.readouterr().out)["spot_holes"]
    assert held["label"] == "SPOT" and held["runs"] == 2
    assert cli._prove(argparse.Namespace(**{**vars(args), "system": "flock"})) == 0
    assert capsys.readouterr().out == "the table\n"
