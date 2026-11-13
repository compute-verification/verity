"""`spot_status`: each resampling hole is closed only where its fix's commit is an ancestor of the commit a run's record
names, decided in a tiny git repository of made-up commits; a run whose code can't be read is not known; the label is
`SPOT soundness` only when both holes are closed; and the theorem's open premises are the fixture lock's."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_bench import cli, prove_records
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
    with the fixes' data file naming the two fixes."""

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    commits = {}
    for name in ("base", "fix1", "fix2"):
        _git(root, "commit", "-q", "--allow-empty", "-m", name)
        commits[name] = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "-b", "side", commits["base"])
    _git(root, "commit", "-q", "--allow-empty", "-m", "side")
    commits["side"] = _git(root, "rev-parse", "HEAD")
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


@pytest.fixture
def changes(tmp_path):
    """A repository whose fix changes a file, for the checks that read what a run's tree holds: `commit(text, msg)`
    writes `f.txt` and commits it, returning the commit."""

    root = tmp_path / "changes"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")

    def commit(text: str, msg: str) -> str:
        (root / "f.txt").write_text(text)
        _git(root, "add", "f.txt")
        _git(root, "commit", "-q", "-m", msg)
        return _git(root, "rev-parse", "HEAD")

    commit("a\n", "base")
    return SimpleNamespace(root=root, commit=commit, head=lambda: _git(root, "rev-parse", "HEAD"))


def test_a_reverted_fix_is_open_and_names_its_revert(changes):
    """An ancestor is not enough: a fix reverted after it is no longer in the run's code."""

    fix = changes.commit("a\nfix\n", "fix")
    assert S.closes(changes.root, fix, fix) == {"status": "closed", "why": None}
    _git(changes.root, "revert", "--no-edit", fix)
    revert = changes.head()
    got = S.closes(changes.root, fix, revert)
    assert got["status"] == "open" and got["why"] == f"reverted by {revert[:12]}, which reverts the fix"
    assert S.holes(revert, changes.root, {REDRAWN: fix, RECOMMITTED: fix})[REDRAWN]["status"] == "open"


def test_a_reverted_merge_that_brought_the_fix_is_open(changes):
    """`git revert -m 1` of the landing's merge, as a red step's culprit is reverted, takes the fix out with it."""

    _git(changes.root, "checkout", "-q", "-b", "side")
    fix = changes.commit("a\nfix\n", "fix")
    _git(changes.root, "checkout", "-q", "main")
    (changes.root / "g.txt").write_text("other\n")
    _git(changes.root, "add", "g.txt")
    _git(changes.root, "commit", "-q", "-m", "other")
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

    fix = changes.commit("a\nfix\n", "fix")
    _git(changes.root, "revert", "--no-edit", fix)
    first = changes.head()
    _git(changes.root, "revert", "--no-edit", "HEAD")
    assert S.closes(changes.root, fix, changes.head()) == {"status": "closed", "why": None}
    _git(changes.root, "revert", "--no-edit", "HEAD")
    again = S.closes(changes.root, fix, changes.head())
    assert again == {"status": "open", "why": f"reverted by {first[:12]}, which reverts the fix"}


def test_a_fix_whose_patch_no_longer_reverses_at_the_run_is_not_known(changes):
    """The belt under the revert line: a tree that took the fix out with no revert's message, or rewrote its lines,
    can't be told to hold it."""

    fix = changes.commit("a\nfix\n", "fix")
    rewrite = changes.commit("a\nrewritten\n", "rewrite the fix's line")
    got = S.closes(changes.root, fix, changes.head())
    short = _git(changes.root, "rev-parse", "--short", rewrite)
    assert got == {"status": "not known", "why": (
        f"the fix's own patch (`git diff {fix[:12]}^ {fix[:12]}`) doesn't apply in reverse at the run's tree: lines it "
        "changed were changed again since, so whether that tree still holds it can't be told. "
        f"{short} touches its files since: if {short} keeps the fix, add it to this hole's commits in spot_holes.json")}
    changes.commit("a\nfix\n", "put it back")
    assert S.closes(changes.root, fix, changes.head())["status"] == "closed"


def test_a_rewrite_declared_after_the_fix_is_the_one_read(changes):
    """A later edit to the fix's lines that keeps it holds the label at `SPOT` only until it is declared."""

    fix = changes.commit("a\nfix\n", "fix")
    rewrite = changes.commit("a\nfix, reworded\n", "reword the fix's line")
    (changes.root / "g.txt").write_text("other\n")
    _git(changes.root, "add", "g.txt")
    _git(changes.root, "commit", "-q", "-m", "later")
    run = changes.head()
    assert S.closes(changes.root, fix, run)["status"] == "not known"
    assert S.closes(changes.root, [fix, rewrite], run) == {"status": "closed", "why": None}
    assert S.holes(run, changes.root, {REDRAWN: (fix, rewrite), RECOMMITTED: (fix, rewrite)})[REDRAWN]["status"] == "closed"
    _git(changes.root, "revert", "--no-edit", rewrite)
    got = S.closes(changes.root, [fix, rewrite], changes.head())
    assert got == {"status": "open",
                   "why": f"reverted by {changes.head()[:12]}, which reverts the fix's later commit {rewrite[:12]}"}
    with pytest.raises(ValueError, match=f"the commits of the fix {rewrite} are out of order: {rewrite[:12]} is not an "
                                         f"ancestor of {fix[:12]}"):
        S.closes(changes.root, [rewrite, fix], run)


def test_the_fixes_file_names_only_holes_by_full_commit(repo):
    repo.data.write_text(json.dumps({"fixes": {"another hole": repo.fix1}}))
    with pytest.raises(ValueError, match="names another hole, which is no hole of"):
        S.fixes(repo.data)
    repo.data.write_text(json.dumps({"fixes": {REDRAWN: repo.fix1[:12]}}))
    with pytest.raises(ValueError, match="not a full commit id"):
        S.fixes(repo.data)


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
