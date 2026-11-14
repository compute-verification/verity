"""`research.carry`: a new head carries its older head's land verdict only when everything between them is merges of trusted
commits that the queue's own merge reproduces, and generated content (pins, and a lock's records but not its policy)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research import carry

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_AUTHOR_DATE": "2026-11-05T00:00:00Z", "GIT_COMMITTER_DATE": "2026-11-05T00:00:00Z", "HOME": "/nonexistent",
       "GIT_CONFIG_NOSYSTEM": "1", "PATH": "/usr/bin:/bin"}
LOCK = "Pkg/lean-audit.json"
PINS = 'PINS = (Pin("pins/TRUSTED.sha256", None, "src"),)\n'
ROOT = Path(__file__).resolve().parents[3]
MOVE_SCRIPTS = ("tools/move/rename.py", "tools/move/layout.py", "tools/move/restack.py")
MAP = "tools/move/demo_map.toml"
PROTO = "import old_proto.a\n\nX = old_proto.a.f()\n"


def lock(type_hash: str = "aa", digest: str = "1", **policy) -> str:
    """A lock laid out one entry a line, as `check.py` writes it, so changes to different entries merge."""
    doc = {"axioms": ["propext"], **policy, "guarantees": {"G": {"owner": "@proofs", "type_hash": type_hash}},
           "reads": {"M": {"digest": digest}}}
    return json.dumps(doc, indent=1) + "\n"


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *a], cwd=repo, env=ENV, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for f, text in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


def merge(repo: Path, other: str, resolve: dict[str, str] | None = None) -> str:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "merge", "--no-ff", "-qm", f"Merge {other[:8]}", other], cwd=repo,
                       env=ENV, capture_output=True, text=True)
    if r.returncode:
        assert resolve, r.stdout + r.stderr
        return commit(repo, resolve, f"Merge {other[:8]}")
    if resolve:
        for f, text in resolve.items():
            (repo / f).write_text(text)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "--amend", "--no-edit")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, {"a.py": "a\n", "b.py": "b\n", LOCK: lock(), "docs/README.md": "r\n", "ci/pins.py": PINS,
                      "src/x": "x\n", "pins/TRUSTED.sha256": "x0  src/x\n"}, "base")
    return tmp_path


def branch(repo: Path, name: str, at: str = "main") -> None:
    git(repo, "checkout", "-q", "-b", name, at)


def landed(repo: Path, *heads: str):
    return carry.trust(repo, [git(repo, "rev-parse", "main")], heads)


def script(repo: Path, *argv: str) -> str:
    """A move script of ``repo``'s run at its root, writing no bytecode into the tree it rewrites (`rename.py` refuses a dirty
    one)."""
    return subprocess.run([sys.executable, *argv], cwd=repo, env={**ENV, "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def moved(tmp_path: Path) -> Path:
    """A `main` whose move (this checkout's `rename.py` with a map from `old_proto` to `new_proto`, as a move lands) came after
    the PR `pr` branched, then one more landed commit; the PR writes a line beside one the move rewrites."""
    repo = tmp_path / "moved"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit(repo, {**{p: (ROOT / p).read_text() for p in MOVE_SCRIPTS}, "tools/verity/check/pins.py": "PINS = ()\n",
                  "tools/research/src/research/store/moves.json": '{"moves": []}\n',
                  MAP: '[rewrite]\nexclude = ["tools/move/**", "tools/research/src/research/store/moves.json"]\n\n'
                       '[words]\n"old_proto" = "new_proto"\n',
                  "old_proto/__init__.py": "", "old_proto/a.py": "def f():\n    return 1\n", "b.py": PROTO, "c.py": "c\n"},
           "base")
    branch(repo, "pr")
    commit(repo, {"b.py": PROTO + "Y = old_proto.a.f()\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    script(repo, "tools/move/rename.py", "--map", MAP)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", f"Move old_proto to new_proto\n\nMove: {MAP}\n")
    commit(repo, {"c.py": "c2\n"}, "landed after the move")
    return repo


def cross(repo: Path) -> str:
    """`pr` crossed to `main` across its move by this checkout's `restack.py`: the crossing's head."""
    [r] = json.loads(script(repo, "tools/move/restack.py", "pr", "--move-commit", git(repo, "rev-parse", "main~1"), "--onto",
                            "main", "--main", "main", "--json", "--no-pins"))
    assert r["status"] == "clean", r
    return r["result"]


def test_a_merge_of_landed_main_carries_and_names_what_it_merged(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    m = commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main")
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries, d.parents) == ("merge-only", True, (m,))
    assert d.why == f"its 1 commit(s) since only merge {m[:12]} as the queue would"


def test_a_lock_that_differs_from_the_queues_merge_only_in_records_carries_since_the_lean_check_regenerates_them(repo: Path):
    """A lock's records (each guarantee's statement record, `reads`, each assumption's digest) are the build's, which the Lean
    check rewrites as drive lands, and a pin is held to the tree, so neither needs a reviewer's eye."""
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n", LOCK: lock(type_hash="bb")}, "the PR")
    git(repo, "checkout", "-q", "main")
    m = commit(repo, {"b.py": "b2\n", LOCK: lock(digest="2"), "src/x": "x2\n", "pins/TRUSTED.sha256": "x2  src/x\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {LOCK: lock(type_hash="cc", digest="3"), "pins/TRUSTED.sha256": "rebuilt  src/x\n"})
    regen = commit(repo, {LOCK: lock(type_hash="dd", digest="4")}, "records")
    d = carry.delta(repo, old, regen, landed(repo))
    assert (d.kind, d.parents, d.generated) == ("merge-only", (m,), (LOCK, "pins/TRUSTED.sha256")), d
    assert d.why.endswith(f"leaving {LOCK}, pins/TRUSTED.sha256 to the checks") and new != regen


def test_a_lock_policy_change_or_a_lock_conflict_is_the_prs_own_change(repo: Path):
    """The rest of a lock is policy the Lean check keeps as written (its axioms, which guarantees and assumptions it names, their
    owners), and the lock's merge driver refuses only over policy: a commit, a merge or a resolution that writes policy needs a
    verdict."""
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n", LOCK: lock(type_hash="bb")}, "the PR")
    axioms = commit(repo, {LOCK: lock(type_hash="bb", axioms=["propext", "sorryAx"])}, "an axiom")
    d = carry.delta(repo, old, axioms, landed(repo))
    assert (d.kind, d.carries) == ("authored", False) and LOCK in d.why
    git(repo, "reset", "-q", "--hard", old)
    broken = commit(repo, {LOCK: "{not json\n"}, "a lock that isn't JSON")
    d = carry.delta(repo, old, broken, landed(repo))
    assert (d.kind, d.carries) == ("authored", False) and LOCK in d.why
    git(repo, "reset", "-q", "--hard", old)
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n", LOCK: lock(digest="2")}, "landed")
    git(repo, "checkout", "-q", "pr")
    dropped = json.loads(lock(type_hash="bb", digest="2"))
    dropped["guarantees"] = {}
    gone = merge(repo, "main", {LOCK: json.dumps(dropped, indent=1) + "\n"})
    assert carry.delta(repo, old, gone, landed(repo)).kind == "edited-merge"
    git(repo, "reset", "-q", "--hard", old)
    git(repo, "checkout", "-q", "main")
    commit(repo, {LOCK: lock(type_hash="zz", digest="2")}, "landed over the same record")
    git(repo, "checkout", "-q", "pr")
    both = merge(repo, "main", {LOCK: lock(type_hash="bb", digest="2"), "a.py": "a2\n", "b.py": "b2\n"})
    d = carry.delta(repo, old, both, landed(repo))
    assert (d.kind, d.carries) == ("hand-resolved", False) and LOCK in d.why


def test_the_owners_own_commit_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = commit(repo, {"b.py": "sneaked\n"}, "more")
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries) == ("authored", False) and "b.py" in d.why


def test_a_merge_that_edits_beyond_the_queues_merge_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {"b.py": "b2 and an edit\n"})
    assert carry.delta(repo, old, new, landed(repo)).kind == "edited-merge"


def test_the_old_heads_tree_by_another_history_needs_a_verdict(repo: Path):
    """A head with the reviewed tree can still undo what it merged: landing it reverts that commit's change."""
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {"b.py": "b\n"})
    assert git(repo, "rev-parse", f"{new}^{{tree}}") == git(repo, "rev-parse", f"{old}^{{tree}}")
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries) == ("edited-merge", False) and "b.py" in d.why


def test_a_conflict_resolved_by_hand_outside_generated_files_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "pr\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"a.py": "main\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {"a.py": "both\n"})
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries) == ("hand-resolved", False) and "a.py" in d.why


def test_merging_another_prs_head_carries_only_once_that_head_is_trusted(repo: Path):
    branch(repo, "other")
    theirs = commit(repo, {"b.py": "other\n"}, "another PR")
    branch(repo, "pr", "main")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = merge(repo, "other")
    assert carry.delta(repo, old, new, landed(repo)).kind == "untrusted-parent"
    assert carry.delta(repo, old, new, landed(repo, theirs)).kind == "merge-only"
    branch(repo, "above", "pr")
    above = commit(repo, {"docs/README.md": "above\n"}, "a reviewed PR stacked on the new head")
    assert carry.delta(repo, old, new, landed(repo, above)).kind == "untrusted-parent"
    git(repo, "checkout", "-q", "other")
    newer = commit(repo, {"b.py": "other, re-headed\n"}, "the other PR moves on")
    assert carry.delta(repo, old, new, landed(repo, newer)).kind == "untrusted-parent"


def test_a_reviewed_pr_built_on_the_old_head_and_merged_back_carries(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    branch(repo, "dependant", "pr")
    theirs = commit(repo, {"b.py": "dependant\n"}, "a PR stacked on it")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "dependant")
    assert carry.delta(repo, old, new, landed(repo)).kind == "untrusted-parent"
    assert carry.delta(repo, old, new, landed(repo, theirs)).kind in carry.CARRIES


def test_a_reviewed_pr_stacked_on_the_new_head_vouches_for_none_of_its_own_commits(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = commit(repo, {"b.py": "sneaked\n"}, "more")
    branch(repo, "above", "pr")
    above = commit(repo, {"docs/README.md": "above\n"}, "a reviewed PR stacked on the new head")
    assert carry.delta(repo, old, new, landed(repo, above)).kind == "authored"


def test_a_rebased_head_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    git(repo, "rebase", "-q", "main")
    new = git(repo, "rev-parse", "HEAD")
    assert carry.delta(repo, old, new, landed(repo)).kind == "rebased"


def test_a_docs_only_commit_needs_a_verdict_too(repo: Path):
    """Only merges carry: a change the low-risk lane would land on its own still waits for a reviewer here, where it rides
    with a change that needed one."""
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = commit(repo, {"docs/README.md": "trimmed\n"}, "trim")
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries) == ("authored", False) and "docs/README.md" in d.why


def test_the_command_prints_the_delta_and_exits_0_only_when_it_carries(repo: Path, capsys):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    merge(repo, "main")
    assert carry.main([old, "pr", "--repo", str(repo), "--landed", "main"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "merge-only"
    commit(repo, {"b.py": "sneaked\n"}, "more")
    assert carry.main([old, "pr", "--repo", str(repo), "--landed", "main"]) == 1
    assert json.loads(capsys.readouterr().out)["kind"] == "authored"


def test_a_crossing_carries_when_its_restack_commit_is_the_moves_script_rerun_on_its_parent(moved: Path, capsys):
    """A head crossed across a move (`tools/move/restack.py`) is a commit that reruns the move's script on the PR's tree, named
    by its `Restack-move:` trailer, and a merge of the moved `main` taking the move for their merge base: from their natural
    base, the PR's line beside one the move rewrote conflicts.  Without the move's script rerun, that commit is the PR's own,
    so a reviewer compared every crossed head by hand."""
    old, move, main = (git(moved, "rev-parse", r) for r in ("pr", "main~1", "main"))
    new = cross(moved)
    assert carry.main([old, new, "--repo", str(moved), "--landed", "main"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert (d["kind"], d["parents"], d["moves"]) == ("merge-only", [main], [move]), d
    assert d["why"] == (f"its 2 commit(s) since only merge {main[:12]} as the queue would and rerun the move {move[:12]} as a "
                        "restack does")
    assert carry.delta(moved, old, new, landed(moved)).kind == "authored"


def test_a_restack_commit_that_changes_more_than_the_moves_script_is_the_prs_own(moved: Path):
    old, move, main = (git(moved, "rev-parse", r) for r in ("pr", "main~1", "main"))
    git(moved, "checkout", "-q", f"{cross(moved)}^1")
    (moved / "b.py").write_text((moved / "b.py").read_text() + "Z = 'sneaked'\n")
    git(moved, "commit", "-q", "-a", "--amend", "--no-edit")
    sneaked = git(moved, "rev-parse", "HEAD")
    tree = git(moved, "merge-tree", "--write-tree", f"--merge-base={move}", sneaked, main)
    new = git(moved, "commit-tree", tree, "-p", sneaked, "-p", main, "-m", "Merge main (the layout move), restacked")
    d = carry.delta(moved, old, new, landed(moved), restack=main)
    assert (d.kind, d.moves) == ("authored", ()) and d.why == f"{sneaked[:12]} changes b.py beyond the move {move[:12]}'s script"


def test_a_move_never_reruns_on_a_tree_that_another_commit_of_the_line_leaves_unvouched(moved: Path, tmp_path: Path):
    """A move's script runs a tree's own code (`rename.py` loads the tree's pins list), so carry reruns it only once every other
    commit of the PR's line passed: here an unreviewed commit below the crossing plants code in that list."""
    old, main = git(moved, "rev-parse", "pr"), git(moved, "rev-parse", "main")
    planted = tmp_path / "planted"
    git(moved, "checkout", "-q", "pr")
    commit(moved, {"tools/verity/check/pins.py": f"import pathlib\npathlib.Path({str(planted)!r}).touch()\nPINS = ()\n"}, "more")
    new = cross(moved)
    assert planted.exists()
    planted.unlink()
    d = carry.delta(moved, old, new, landed(moved), restack=main)
    assert d.kind == "authored" and "tools/verity/check/pins.py" in d.why and not planted.exists()
