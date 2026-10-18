"""`tools/verity/check/precheck.py`: a lane's check of its commit against a `next` fetched now, on a throwaway origin whose `next`
holds the queue's merge commits, with the suite and Lean runners stood in for."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_precheck", ROOT / "tools" / "verity" / "check" / "precheck.py")
PC = sys.modules[spec.name] = importlib.util.module_from_spec(spec)     # its dataclasses look their module up there
spec.loader.exec_module(PC)


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


class Origin:
    """A bare origin with `main`, and `dev`, a clone that writes commits and the queue's merges onto `next`."""

    def __init__(self, tmp: Path):
        self.tmp, self.bare, self.dev = tmp, tmp / "origin.git", tmp / "dev"
        _git(tmp, "init", "-q", "--bare", "-b", "main", str(self.bare))
        files = {"a.txt": "a\n", "b.txt": "b\n", "c.txt": "c\n", "Security/lean-audit.json": "{}\n", "Security/X.lean": "x\n",
                 "Security/Proofs/lean-audit.json": "{}\n", "Security/Proofs/P.lean": "p\n"}
        self.dev.mkdir()
        self.write(self.dev, files)
        _git(self.dev, "init", "-q", "-b", "main")
        _git(self.dev, "add", "-A")
        _git(self.dev, "commit", "-q", "-m", "init")
        _git(self.dev, "remote", "add", "origin", str(self.bare))
        _git(self.dev, "push", "-q", "origin", "main")

    @staticmethod
    def write(repo: Path, files: dict[str, str]) -> None:
        for rel, text in files.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text)

    def change(self, files: dict[str, str]) -> str:
        """A commit on main writing ``files``."""
        _git(self.dev, "checkout", "-q", "--detach", "main")
        self.write(self.dev, files)
        _git(self.dev, "commit", "-qam", f"change {sorted(files)}")
        return _git(self.dev, "rev-parse", "HEAD")

    def queue(self, prs: dict[int, str]) -> str:
        """`next`: main and each of ``prs`` (number: head) merged on top as the queue merges, with its trailers."""
        _git(self.dev, "checkout", "-q", "--detach", "main")
        for n, head in prs.items():
            _git(self.dev, "merge", "-q", "--no-ff", "-m", f"Queue: #{n} change {n}\n\nPR: #{n}\nHead: {head}\nOwner: lane-{n}", head)
        _git(self.dev, "push", "-q", "-f", "origin", "HEAD:refs/heads/next")
        return _git(self.dev, "rev-parse", "HEAD")

    def lane(self, name: str = "lane") -> Path:
        """A lane's checkout: a clone of origin."""
        _git(self.tmp, "clone", "-q", str(self.bare), name)
        return self.tmp / name

    def commit(self, lane: Path, files: dict[str, str]) -> str:
        _git(lane, "checkout", "-q", "-B", "work", "origin/main")
        self.write(lane, files)
        _git(lane, "add", "-A")
        _git(lane, "commit", "-q", "-m", "work")
        return _git(lane, "rev-parse", "HEAD")


@pytest.fixture
def origin(tmp_path: Path) -> Origin:
    return Origin(tmp_path)


class Runners:
    """The suite and Lean runners, recording what precheck asked of them."""

    def __init__(self, suites: list[dict] | str = (), lean_rc: int = 0):
        self.out, self.lean_rc, self.calls = list(suites) if not isinstance(suites, str) else suites, lean_rc, []

    def suites(self, root, base, quick, jobs):
        self.calls.append(("suites", base, quick))
        return self.out

    def lean(self, root, pkgs):
        self.calls.append(("lean", pkgs))
        return [(p, self.lean_rc) for p in pkgs]


def test_a_commit_that_merges_onto_main_but_conflicts_with_next_names_the_member_there_against_a_next_fetched_now(origin):
    """17 Oct: nine PRs read 'admitted' and a worker's report called a PR 'clean against next' while each conflicted with a
    member of `next` (#2392 with #2384 there): `next`, fetched before the run, had moved."""
    lane = origin.lane()
    mine = origin.commit(lane, {"a.txt": "mine\n"})
    nxt = origin.queue({7: origin.change({"a.txt": "seven\n"}), 8: origin.change({"c.txt": "eight\n"})})
    main = _git(origin.bare, "rev-parse", "main")
    run = Runners([{"name": "research", "status": "passed", "counts": {"passed": 3, "skipped": 1}}])
    rep = PC.precheck(lane, suites=run.suites, lean=run.lean)
    assert (rep.commit, rep.next, rep.main) == (mine, nxt, main)
    assert [m.pr for m in rep.members] == [7, 8] and rep.by == [7]
    assert rep.conflicts == {"main": [], "next": ["a.txt"]}
    assert run.calls == [("suites", main, False)]                  # no Lake package changed
    text = PC.summary(rep)
    assert f"precheck of {mine[:12]} on work" in text
    assert f"next:   {nxt[:12]}, fetched now (#7, #8 above main {main[:12]})" in text
    assert "merge:  onto next conflicts in a.txt (#7 there): merge #7's branch and re-mark" in text
    assert "merge:  onto main clean" in text
    assert f"suites: full, since the merge base {main[:12]}: research passed (3 passed, 1 skipped)" in text
    assert "lean:   no Lake package changed" in text
    assert text.endswith("result: not clean (conflicts with `next`)")


def test_a_clean_commit_runs_the_suites_from_its_merge_base_and_the_records_of_each_lake_package_it_changed(origin):
    nxt = origin.queue({7: origin.change({"a.txt": "seven\n"})})
    lane = origin.lane()
    origin.commit(lane, {"b.txt": "mine\n", "Security/X.lean": "x2\n", "Security/Proofs/P.lean": "p2\n"})
    run = Runners([{"name": "research", "status": "passed", "counts": {"passed": 3}},
                   {"name": "verity-check", "status": "cached", "counts": {"passed": 9}}])
    rep = PC.precheck(lane, quick=True, suites=run.suites, lean=run.lean)
    assert rep.changed_packages == ["Security", "Security/Proofs"] and ("lean", ["Security", "Security/Proofs"]) in run.calls
    text = PC.summary(rep)
    assert "suites: quick" in text and "research passed (3 passed); verity-check cached" in text
    assert "lean:   records of Security passed; Security/Proofs passed" in text
    assert text.endswith(f"result: clean against `next` {nxt[:12]}: mark it ready") and rep.problems() == []


def test_a_failing_suite_or_record_or_a_skipped_part_leaves_the_result_unclean(origin):
    origin.queue({7: origin.change({"a.txt": "seven\n"})})
    lane = origin.lane()
    origin.commit(lane, {"Security/X.lean": "x2\n"})
    run = Runners([{"name": "research", "status": "failed", "counts": {"passed": 2, "failed": 1}}], lean_rc=1)
    rep = PC.precheck(lane, suites=run.suites, lean=run.lean)
    assert rep.problems() == ["suite research failed", "Lean records of Security failed"]
    assert "research failed (2 passed, 1 failed)" in PC.summary(rep)
    skipped = PC.precheck(lane, suites=None, lean=None)
    assert skipped.problems() == ["suites not run", "Lean records not run"]
    assert "lean:   not run: skipped (--no-lean) (Security changed)" in PC.summary(skipped)


def test_without_a_next_on_origin_it_checks_against_main_and_says_so(origin):
    lane = origin.lane()
    origin.commit(lane, {"b.txt": "mine\n"})
    origin.change({"b.txt": "main moved\n"})
    _git(origin.dev, "push", "-q", "origin", "HEAD:main")
    rep = PC.precheck(lane, suites=Runners().suites, lean=None)
    assert rep.next is None and rep.conflicts == {"main": ["b.txt"]}
    text = PC.summary(rep)
    assert "next:   origin has none" in text
    assert "onto main conflicts in b.txt: merge `main` into the branch and re-mark" in text
    assert "the branch touches none" in text


def test_a_tree_with_changes_git_sees_is_refused(origin):
    lane = origin.lane()
    origin.commit(lane, {"b.txt": "mine\n"})
    (lane / "new.txt").write_text("untracked\n")
    with pytest.raises(PC.Refused, match="1 uncommitted or untracked file.*commit or stash them first"):
        PC.precheck(lane, suites=Runners().suites)
