"""`research queue`: `next` is `main` plus the admitted PRs, one merge commit each; `main` fast-forwards to the newest commit on it
with a passing `check` of that exact commit.  Every test runs on a throwaway bare origin, with a fake GitHub that serves the open
PRs and writes their `refs/pull/{n}/head`."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from research.cli import _custody_ttl_problem, main as research_main
from research.pods.nebius import slot_only
from research.queue import (DRIVE_FLAGS, PR, SHADOW_BRANCH, Queue, QueueError, Rules, conflicting_files, credential_args,
                            label_target)
from research.store import ids, labels_sync, vocab
from research.store.local import LocalStore
from research.store.model import Label
from test_merge_gate import CHECK_FILES

RULES = 'hold_label = "hold"\n'
AGENT = "bc-605d7c89-ca73-5a32-a582-ee77c49e762a"


def PASS(pr: PR) -> None:
    """A quick tier that passes."""
    return None


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


class FakeGitHub:
    def __init__(self, origin: Path):
        self.origin, self.prs, self.ready = origin, {}, set()

    def put(self, n: int, head: str, *, body: str | None = None, draft: bool = False, labels: tuple[str, ...] = (), ready: bool = True) -> None:
        """A PR at `head`; `ready`: its owner has said so at this head (`research queue ready`), as every PR here has unless told."""
        _git(self.origin, "update-ref", f"refs/pull/{n}/head", head)
        self.prs[n] = PR(n, head, f"change {n}", f"Lane `lane-{n}`." if body is None else body, draft, labels)
        if ready:
            self.ready.add((n, head))

    def open_prs(self) -> list[PR]:
        return list(self.prs.values())


class World:
    def __init__(self, tmp: Path):
        self.tmp, self.origin, self.dev = tmp, tmp / "origin.git", tmp / "dev"
        _git(tmp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        for rel, text in {"a.txt": "a\n", "b.txt": "b\n", "c.txt": "c\n", "ci/queue.toml": RULES, ".gitignore": "__pycache__/\n",
                          "pkg/lean-audit.json": json.dumps({"guarantees": {"T": "1"}, "reads": {}, "dependencies": {"M": "1"}}),
                          "mytools/decl_check.py": "from research.store.tool import Tool\nT = Tool(name='check', version='1')\n",
                          **CHECK_FILES}.items():
            (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dev / rel).write_text(text)
        _git(self.dev, "init", "-q", "-b", "main")
        _git(self.dev, "add", "-A")
        _git(self.dev, "commit", "-q", "-m", "init")
        _git(self.dev, "remote", "add", "origin", str(self.origin))
        _git(self.dev, "push", "-q", "origin", "main")
        self.gh, self.logs = FakeGitHub(self.origin), []
        self.store = LocalStore(tmp / "store")

    def change(self, files: dict[str, str | None], base: str = "origin/main") -> str:
        """A commit on `base` writing `files` (deleting those whose text is None); returns its sha."""
        _git(self.dev, "fetch", "-q", "origin")
        _git(self.dev, "checkout", "-q", "--detach", base)
        for rel, text in files.items():
            if text is None:
                (self.dev / rel).unlink()
                continue
            (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dev / rel).write_text(text)
        _git(self.dev, "add", "-A")
        _git(self.dev, "commit", "-q", "-m", f"change {sorted(files)}")
        sha = _git(self.dev, "rev-parse", "HEAD")
        _git(self.dev, "push", "-q", "origin", f"{sha}:refs/dev/{sha}")         # the object on the origin, as a PR's head is
        return sha

    def with_store_remote(self) -> None:
        """Give `store` a remote (a directory): `StoreLabels` answers only from labels it pulled from one."""
        (self.tmp / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{self.tmp / "vault"}"\n')
        self.store = LocalStore(self.tmp / "store", config_path=self.tmp / "store.toml")

    def queue(self, **kw) -> Queue:
        kw.setdefault("ready", lambda n, h: (n, h) in self.gh.ready)
        return Queue(self.tmp / "q" / "verity", remote=str(self.origin), github=self.gh,
                     store=self.store, log=self.logs.append, **kw)

    def on_next(self, q: Queue) -> list[int]:
        q.fetch()
        return [m.pr for m in q.chain()]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_ready_prs_merge_onto_next_in_number_order_drafts_too_and_held_ownerless_or_unmarked_prs_wait(world):
    world.gh.put(3, world.change({"c.txt": "c3\n"}))
    world.gh.put(1, world.change({"a.txt": "a1\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}), draft=True)
    world.gh.put(4, world.change({"d.txt": "d\n"}), labels=("hold",))
    world.gh.put(5, world.change({"e.txt": "e\n"}), body="no owner here")
    world.gh.put(6, world.change({"f.txt": "f\n"}), body=f"[Open in Web](https://cursor.com/agents/{AGENT})")
    world.gh.put(7, world.change({"g.txt": "g\n"}), draft=True, labels=("hold",))
    world.gh.put(8, h8 := world.change({"h.txt": "h\n"}), draft=True, ready=False)
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2, 3, 6] == world.on_next(q)
    assert members[3].owner == AGENT and members[0].owner == "lane-1"
    assert entries[4].why == entries[7].why == "held (`hold` label)" and entries[5].why.startswith("no owner")
    assert entries[8].why == f"not marked ready at {h8[:8]}: `research queue ready 8 --on MACHINE` (or --local; --mark without the tier)"
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    assert _git(world.origin, "rev-parse", f"{tip}^1^1^1^1") == _git(world.origin, "rev-parse", "main")    # one merge commit each
    msg = _git(world.origin, "log", "-1", "--format=%B", tip)
    assert msg.startswith("Queue: #6 change 6") and f"Head: {world.gh.prs[6].head}" in msg
    status = q.status()
    assert [n for n, e in sorted(status.items()) if e.state == "queued"] == [1, 2, 3, 6] and status[3].position == 3


def test_a_new_head_on_top_of_the_admitted_one_keeps_its_place_and_any_other_goes_to_the_back(world):
    """A PR re-merged each time the stacked PRs below it take fixes isn't sent behind the rest of the queue by each re-mark: a
    head that descends from the one `next` holds keeps its place; a rewritten one doesn't."""
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}))
    world.gh.put(3, h3 := world.change({"c.txt": "c3\n"}))
    q = world.queue()
    q.sync()
    assert world.on_next(q) == [1, 2, 3]
    world.gh.put(1, world.change({"a.txt": "a1 fixed\n"}, base=h1))           # a fix on top of #1's admitted head
    members, _ = q.sync()
    assert [m.pr for m in members] == [1, 2, 3] and members[0].head == world.gh.prs[1].head
    world.gh.put(3, world.change({"c.txt": "c3 again\n"}, base=h3))          # #3 fixed too: it keeps its place
    world.gh.put(2, world.change({"b.txt": "b2 rewritten\n"}))               # #2 rewritten from main: to the back
    members, _ = q.sync()
    assert [m.pr for m in members] == [1, 3, 2]


def test_an_unmarked_new_head_on_top_keeps_the_admitted_head_in_place_until_its_mark(world):
    """A PR pushed one commit on doesn't drop off `next` while its new head's preflight runs and rejoin at the back on its mark:
    a new head that descends from the admitted one keeps the admitted head on `next` until it is marked; a rewritten one still
    drops the PR until it is."""
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}))
    world.gh.put(2, h2 := world.change({"b.txt": "b2\n"}))
    world.gh.put(3, world.change({"c.txt": "c3\n"}))
    q = world.queue()
    q.sync()
    assert world.on_next(q) == [1, 2, 3]
    world.gh.put(1, h1b := world.change({"a.txt": "a1 fixed\n"}, base=h1), ready=False)    # pushed, its preflight still running
    members, entries = q.sync()
    assert [(m.pr, m.head) for m in members][0] == (1, h1) and [m.pr for m in members] == [1, 2, 3]
    world.gh.ready.add((1, h1b))                                                             # marked
    members, _ = q.sync()
    assert [(m.pr, m.head) for m in members][0] == (1, h1b) and [m.pr for m in members] == [1, 2, 3]
    world.gh.put(2, world.change({"b.txt": "b2 rewritten\n"}), ready=False)                  # not on top of h2: dropped
    members, _ = q.sync()
    assert [m.pr for m in members] == [1, 3] and h2


def test_the_queues_merges_carry_the_time_they_are_made_whatever_date_the_caller_pins(world, monkeypatch):
    """A caller that pins the git dates (for reproducible merge ids) doesn't date the queue's merges: a merge that landed with
    an old pinned date would stop `git log --since` over `main` early there."""
    world.gh.put(1, world.change({"a.txt": "a1\n"}))
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-09-28T23:00:00+0000")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-09-28T23:00:00+0000")
    q = world.queue()
    [m] = q.sync()[0]
    assert "2026-09-28" not in _git(q.workdir, "log", "-1", "--format=%aI %cI", m.commit)


def test_a_new_push_moves_its_pr_to_the_tip_and_keeps_the_commits_before_it(world):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    first, _ = q.sync()
    world.gh.put(2, world.change({"b.txt": "2, again\n"}))
    again, _ = q.sync()
    assert [m.pr for m in again] == [1, 3, 2]
    assert again[0].commit == first[0].commit                  # the unchanged prefix is kept, not re-merged
    assert again[2].head == world.gh.prs[2].head


def test_an_api_head_ahead_of_its_pull_ref_waits_without_blocking_the_queue(world):
    """GitHub's REST PR head can update before refs/pull/N/head; the queue must not walk the absent REST commit or stop the tick."""
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, old := world.change({"b.txt": "old\n"}))
    q = world.queue()
    assert [m.pr for m in q.sync()[0]] == [1, 2]

    new = world.change({"b.txt": "new\n"})
    world.gh.prs[2] = PR(2, new, "change 2", "Lane `lane-2`.")
    world.gh.ready.add((2, new))
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "waiting"
    assert entries[2].why == f"GitHub's API head {new[:12]} and pull ref {old[:12]} disagree; retrying next sync"
    assert world.logs[-2] == f"research queue: #2's API head {new[:12]} and fetched pull ref {old[:12]} disagree; retrying next sync"

    world.gh.put(2, new)
    assert [m.pr for m in q.sync()[0]] == [1, 2]


def test_a_pull_ref_github_left_behind_takes_the_head_from_its_branch_when_the_branch_is_at_the_api_head(world):
    """GitHub can leave a PR's refs/pull/N/head behind its branch and API head for hours, and the PR waited every sync. A
    branch whose tip is the API head stands in for the pull ref; one at any other commit doesn't."""
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, old := world.change({"b.txt": "old\n"}))
    q = world.queue()
    assert [m.pr for m in q.sync()[0]] == [1, 2]

    new = world.change({"b.txt": "new\n"})
    _git(world.origin, "update-ref", "refs/heads/feature-2", old)
    world.gh.prs[2] = PR(2, new, "change 2", "Lane `lane-2`.", branch="feature-2")
    world.gh.ready.add((2, new))
    members, entries = q.sync()
    assert [m.pr for m in members] == [1] and entries[2].state == "waiting" and "disagree" in entries[2].why

    _git(world.origin, "update-ref", "refs/heads/feature-2", new)
    members, _ = q.sync()
    assert [(m.pr, m.head) for m in members] == [(1, world.gh.prs[1].head), (2, new)]
    assert any("#2's pull ref isn't at its API head" in line for line in world.logs)
    assert _git(world.origin, "rev-parse", "refs/pull/2/head") == old


def test_a_pr_github_has_no_pull_ref_for_waits_and_the_others_sync(world):
    """GitHub can list an open PR before it makes refs/pull/N/head, and one missing ref would fail the whole fetch, and every
    drive tick with it: that PR waits, and the rest of the queue syncs."""
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, two := world.change({"b.txt": "two\n"}))
    _git(world.origin, "update-ref", "-d", "refs/pull/2/head")
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "waiting" and "no pull ref" in entries[2].why
    assert any("#2's pull ref can't be fetched" in line for line in world.logs)

    _git(world.origin, "update-ref", "refs/pull/2/head", two)
    assert [m.pr for m in q.sync()[0]] == [1, 2]


def test_first_puts_a_ready_fix_at_the_bottom_of_next_ahead_of_members_that_keep_their_place(world):
    """A fix for a test that fails on `main` needn't sit behind the next train: `--first` restacks `next` so the next train
    starts with the fix.  A PR in `--first` that isn't ready is passed over, and another queue's sync without the
    flag keeps the new order."""
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    world.queue().sync()
    world.gh.put(9, world.change({"fix.txt": "fix\n"}))
    world.gh.put(8, world.change({"other.txt": "x\n"}), ready=False)
    members, _ = world.queue(first=(8, 9)).sync()
    assert [m.pr for m in members] == [9, 1, 2, 3]
    again, _ = world.queue().sync()
    assert [m.commit for m in again] == [m.commit for m in members]


def test_a_sync_merges_what_changes_no_lean_below_every_lean_change_in_first_then_admission_order(world):
    """A member above a Lean change waits for that change's Lean check, ~1 h, though
    it changes no Lean itself.  What a sync merges goes non-Lean first, `--first`'s order then admission's within each part; a PR
    that contains a Lean change stays above it, and one stacked on a member already on `next` is judged by its own change."""
    world.gh.put(1, h1 := world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, h2 := world.change({"a.txt": "2\n"}))
    world.gh.put(3, h3 := world.change({"pkg/B.lean": "-- 3\n"}))
    world.gh.put(4, world.change({"b.txt": "4\n"}))
    world.gh.put(5, world.change({"c.txt": "5\n"}, base=h1))                 # changes no Lean, but contains #1
    world.gh.put(6, world.change({"d.txt": "6\n"}, base=h2))
    members, _ = world.queue(first=(3, 4)).sync()
    assert [m.pr for m in members] == [4, 2, 6, 3, 1, 5]
    world.gh.put(9, world.change({"pkg/C.lean": "-- 9\n"}))
    world.gh.put(10, world.change({"e.txt": "10\n"}, base=h3))               # its own change, against #3's head, is no Lean
    world.gh.put(11, world.change({"f.txt": "11\n"}))
    again, _ = world.queue().sync()
    assert [m.pr for m in again] == [4, 2, 6, 3, 1, 5, 10, 11, 9]
    assert [m.commit for m in again[:6]] == [m.commit for m in members]      # what the sync keeps never moves for it
    world.gh.put(4, world.change({"b.txt": "4 rewritten\n"}))                # off the bottom: the rest merged again, in order
    rebuilt, _ = world.queue().sync()
    assert [m.pr for m in rebuilt] == [2, 6, 3, 1, 5, 10, 11, 9, 4]


def test_a_sync_with_one_new_admission_keeps_every_member_and_merges_only_it(world):
    """Each PR a sync merges runs the landing gates (~15 s each), so a new admission placed low re-merged most of `next`, a
    160-member pass taking 40-50 min.  A new PR, Lean or not, goes on top: the sync keeps every member."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/A.lean": "-- 2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    members, _ = world.queue().sync()
    assert [m.pr for m in members] == [1, 3, 2]
    for n, files in ((4, {"c.txt": "4\n"}), (5, {"pkg/B.lean": "-- 5\n"})):
        world.gh.put(n, world.change(files))
        again, _ = world.queue().sync()
        assert [m.commit for m in again[:len(members)]] == [m.commit for m in members] and again[-1].pr == n
        assert world.logs[-1].endswith(f"kept {len(members)}, merged 1"), world.logs[-1]
        members = again


def test_members_merged_again_above_a_moved_one_keep_their_order_and_only_new_prs_go_nonlean_first(world):
    """Members above a moved or dropped member are merged again in `next`'s order: sorting them non-Lean first again would undo
    drive's lift (reviewed members first), and the next lift would merge them all again, each tick.  What comes new is sorted."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/A.lean": "-- 2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    world.gh.put(4, world.change({"pkg/B.lean": "-- 4\n"}))
    q = world.queue()
    q.sync()
    q.lift = [2, 3]                                                          # drive's lift: #2 and #3 have a land verdict
    assert [m.pr for m in q.sync()[0]] == [2, 3, 1, 4]
    q.lift = []
    world.gh.put(3, world.change({"b.txt": "3 rewritten\n"}))                # moved: to the top
    world.gh.put(6, world.change({"pkg/C.lean": "-- 6\n"}))
    world.gh.put(5, world.change({"c.txt": "5\n"}))
    members, _ = q.sync()
    assert [m.pr for m in members] == [2, 1, 4, 3, 5, 6] and world.logs[-1].endswith("kept 1, merged 5"), world.logs[-1]
    assert world.queue().sync()[0] == members and world.logs[-1].endswith("kept 6, merged 0")


def test_a_first_pr_still_goes_to_the_bottom_and_sorts_what_is_above_it(world):
    """`--first` restacks `next` from the bottom as it always has: the PRs it names go below every member, and what is merged
    again above them goes non-Lean first; the next sync with the same `--first` keeps it all."""
    world.gh.put(1, world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"pkg/B.lean": "-- 3\n"}))
    _git(world.dev, "push", "-q", "origin", f"{world.change({DRIVE_FLAGS: 'nonlean_first = false'})}:refs/heads/main")
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2, 3]
    world.gh.put(9, world.change({"fix.txt": "9\n"}))
    members, _ = world.queue(first=(9,), nonlean_first=True).sync()
    assert [m.pr for m in members] == [9, 2, 1, 3] and world.logs[-1].endswith("kept 0, merged 4")
    again, _ = world.queue(first=(9,), nonlean_first=True).sync()
    assert again == members and world.logs[-1].endswith("kept 4, merged 0")


def test_nonlean_first_off_at_main_keeps_first_then_admission_order_and_turning_it_on_rebuilds_nothing(world):
    """drive.toml's `nonlean_first` at `main` decides a sync that isn't given it.  Turned on, it reorders only what a sync merges
    anyway: a rebuild just to reorder would take a running Lean check's commit off `next`."""
    _git(world.dev, "push", "-q", "origin", f"{world.change({DRIVE_FLAGS: 'nonlean_first = false'})}:refs/heads/main")
    world.gh.put(1, world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    members, _ = world.queue(first=(3,)).sync()
    assert [m.pr for m in members] == [3, 1, 2]
    assert [m.commit for m in world.queue(nonlean_first=True).sync()[0]] == [m.commit for m in members]
    _git(world.dev, "push", "-q", "origin", f"{world.change({'x.txt': 'x\n'})}:refs/heads/main")      # `next` is rebuilt
    assert [m.pr for m in world.queue(nonlean_first=True).sync()[0]] == [2, 3, 1]


def test_first_naming_a_lean_change_before_a_nonlean_one_builds_a_next_the_same_sync_keeps(world):
    """A sync compares `next` with the order it would build, non-Lean first, so `--first`'s order, which non-Lean first
    overrides, rebuilds nothing on the next tick."""
    world.gh.put(1, world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"pkg/B.lean": "-- 3\n"}))
    world.gh.put(4, world.change({"b.txt": "4\n"}))
    members, _ = world.queue(first=(3, 4), nonlean_first=True).sync()
    assert [m.pr for m in members] == [4, 2, 3, 1]
    again, _ = world.queue(first=(3, 4), nonlean_first=True).sync()
    assert [m.commit for m in again] == [m.commit for m in members] and world.logs[-1].endswith("kept 4, merged 0")


def test_a_pr_the_last_sync_left_out_is_tried_above_the_rest_so_a_sync_with_nothing_new_keeps_next(world):
    """A PR that didn't merge, left where `--first`, or non-Lean first, put it in the order the sync compares `next` with, would
    have every member above it merged again (new commits) on each tick with `main` still, throwing away every Lean check of
    `next` but drive's own.  Left out at a head, it (and a PR that contains it) is tried above the rest until that head
    moves."""
    world.gh.put(1, world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    world.gh.put(4, world.change({"d.txt": "4\n"}, base=world.gh.prs[3].head))
    _git(world.dev, "push", "-q", "origin", f"{world.change({'c.txt': 'main\n'})}:refs/heads/main")
    first = (3, 1, 4)
    members, entries = world.queue(first=first).sync()
    assert [m.pr for m in members] == [2, 1] and entries[3].state == "refused" and entries[4].state == "waiting"
    for _ in range(2):
        again, entries = world.queue(first=first).sync()
        assert [m.commit for m in again] == [m.commit for m in members] and world.logs[-1].endswith("kept 2, merged 0")
        assert entries[3].state == "refused" and entries[4].state == "waiting"
    world.gh.put(3, world.change({"c.txt": "3 on main\n"}))                                       # fixed: first again
    members, entries = world.queue(first=first).sync()
    assert [m.pr for m in members] == [3, 2, 1] and entries[4].state == "refused"     # #4 holds the old #3, which conflicts
    again, _ = world.queue(first=first).sync()
    assert [m.commit for m in again] == [m.commit for m in members] and world.logs[-1].endswith("kept 3, merged 0")


def test_the_queues_clone_merges_lean_audit_records_with_the_record_driver(world):
    """`.gitattributes` names `lean-audit` for every lean-audit.json; the queue's own clone must define it, or two PRs that each
    add a record conflict as text."""
    from research.clone import RECORD_DRIVER

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    tools = {f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}
    rec = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    base = world.change({**tools, ".gitattributes": "lean-audit.json merge=lean-audit\n"})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    for n, g in ((1, "A"), (2, "B")):
        world.gh.put(n, world.change({"pkg/lean-audit.json": json.dumps({**rec, "guarantees": {**rec["guarantees"], g: "1"}})}))
    q = world.queue()
    members, entries = q.sync()
    for k, v in RECORD_DRIVER.items():
        assert _git(q.workdir, "config", "--get", k) == v
    assert [m.pr for m in members] == [1, 2], entries[2].why
    merged = json.loads(_git(world.origin, "show", "refs/heads/next:pkg/lean-audit.json"))
    assert set(merged["guarantees"]) == {"T", "A", "B"}


def test_a_sync_by_merge_tree_builds_the_next_and_refusals_the_worktree_merges_built(world, monkeypatch):
    """A sync that merges each candidate in the clone's worktree takes tens of minutes when many conflict.  `_merge`
    merges with `git merge-tree` and `git commit-tree` and touches no worktree; over 30 PRs (conflicts in one and two files,
    lock records only the merge driver an earlier member adds to `.gitattributes` merges, a stacked PR, a second sync that
    appends) it builds the same `next` (tree, first-parent messages) and the same refusals as the worktree merges."""
    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    base = world.change({f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    rec = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    heads = {}
    for n in range(1, 31):
        if n == 2:
            files = {".gitattributes": "lean-audit.json merge=lean-audit\n"}
        elif n % 7 == 0:
            files = {"pkg/lean-audit.json": json.dumps({**rec, "guarantees": {**rec["guarantees"], f"G{n}": "1"}})}
        elif n % 5 == 0:
            files = {"shared.txt": f"{n}\n", **({"a.txt": f"{n}\n"} if n % 10 == 0 else {})}
        else:
            files = {"a.txt" if n == 1 else f"f{n}.txt": f"{n}\n"}
        heads[n] = world.change(files, base=heads[3] if n == 29 else "origin/main")
        world.gh.put(n, heads[n])
    new = world.queue()
    monkeypatch.setattr(new, "_merge_in_worktree", lambda *a: pytest.fail("merged in the worktree"))
    old = Queue(world.tmp / "q-old" / "verity", remote=str(world.origin), github=world.gh, store=world.store,
                log=world.logs.append, ready=lambda n, h: (n, h) in world.gh.ready, shadow=True)
    old._attr_source = False

    def same() -> list:
        (got, ge), (want, we) = new.sync(), old.sync()
        assert [m.pr for m in got] == [m.pr for m in want]
        assert {n: (e.state, e.why) for n, e in ge.items()} == {n: (e.state, e.why.replace(SHADOW_BRANCH, "next"))
                                                                  for n, e in we.items()}
        tips = [_git(world.origin, "rev-parse", f"refs/heads/{b}") for b in ("next", SHADOW_BRANCH)]
        assert len({_git(world.origin, "rev-parse", f"{t}^{{tree}}") for t in tips}) == 1
        assert len({_git(world.origin, "log", "--first-parent", "--format=%B", f"main..{t}") for t in tips}) == 1
        assert _git(new.workdir, "rev-parse", "HEAD") == tips[0]
        return got, ge

    first, entries = same()
    two, one = "conflicts with `next` in a.txt, shared.txt (touched by #1, #5)", "conflicts with `next` in shared.txt (touched by #5)"
    assert {n: e.why for n, e in entries.items() if e.state == "refused"} == {10: two, 15: one, 20: two, 25: one, 30: two}
    assert {7, 14, 21, 28, 29} <= {m.pr for m in first}
    world.gh.put(31, world.change({"f31.txt": "31\n"}))
    again, _ = same()
    assert [m.commit for m in again[:-1]] == [m.commit for m in first] and again[-1].pr == 31


def test_a_conflicting_pr_is_left_out_with_the_queued_pr_that_touched_the_file_and_no_note_is_written(world, monkeypatch):
    """A sync writes no conflict handoff into the notes clone of the VM it runs on, where it would wait for that VM's next notes
    push; drive tells each conflict in the merge thread (`Driver.refuse`)."""
    monkeypatch.setenv("RESEARCH_NOTES", str(world.tmp / "notes"))
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, world.change({"a.txt": "two\n", "b.txt": "b2\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "refused" and entries[2].why == "conflicts with `next` in a.txt (touched by #1)"
    assert q.load_state()["conflicts"]["2"]["why"] == entries[2].why
    assert not (world.tmp / "notes").exists()
    assert q.status()[2].why == "conflicts with `next` in a.txt (touched by #1) (at the last sync)"


POUS = "verity/core/protocols/pous/lean"
TRUSTED, EXFIL = f"{POUS}/TRUSTED.sha256", "verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json"
UP = "../../../../../Security/Proofs/lakefile.toml"


def _trusted(paths: list[str], tree: dict[str, str]) -> str:
    """PoUS's pin, a sha256sum listing of ``paths`` (relative to its directory) as ``tree`` holds them."""
    return "".join(f"{hashlib.sha256(tree[posixpath.normpath(f'{POUS}/{p}')].encode()).hexdigest()}  {p}\n" for p in paths)


def _exfil(paths: list[str], tree: dict[str, str]) -> str:
    """A JSON pin under `inputs`, followed by vectors one per line as its generate.sh writes them."""
    doc = {"source": "s", "inputs": {p: hashlib.sha256(tree[p].encode()).hexdigest() for p in paths}}
    return json.dumps(doc, indent=1)[:-2] + ',\n "vectors": [\n{"a": 1},\n{"b": 2}\n ]\n}\n'


def _pinned(world: World, tree: dict[str, str], trusted: list[str], exfil: list[str], base: str = "origin/main") -> str:
    return world.change({**tree, TRUSTED: _trusted(trusted, tree), EXFIL: _exfil(exfil, tree)}, base=base)


def _pin_main(world: World) -> dict[str, str]:
    """`main` with the real pins tool and both pins over a few inputs; returns those inputs."""
    tree = {f"{POUS}/lakefile.toml": "l0\n", f"{POUS}/lean-toolchain": "t0\n", f"{POUS}/Old.lean": "o0\n",
            "Security/Proofs/lakefile.toml": "s0\n", "x.lean": "x0\n"}
    pins = (Path(__file__).resolve().parents[2] / "verity" / "check" / "pins.py").read_text()
    base = _pinned(world, {**tree, "tools/verity/check/pins.py": pins}, ["lean-toolchain", "lakefile.toml", "Old.lean", UP],
                   ["Security/Proofs/lakefile.toml", "x.lean"])
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    return tree


def test_prs_whose_only_conflicts_are_in_pins_merge_with_each_pin_rebuilt_from_the_merged_tree(world):
    """Two PRs that each regenerated PoUS's TRUSTED.sha256 for their own change to a pinned input both merge onto `next`: each conflicted pin records both sides' paths (OURS's
    order, then THEIRS's new ones) that the merged tree still holds, at their digests there, and a JSON pin keeps the rest of
    its text byte for byte."""
    tree = _pin_main(world)
    one = {**tree, f"{POUS}/lakefile.toml": "l1\n", "Security/Proofs/lakefile.toml": "s1\n", f"{POUS}/New1.lean": "n1\n",
           "y.lean": "y\n"}
    world.gh.put(1, _pinned(world, one, ["lean-toolchain", "lakefile.toml", "Old.lean", UP, "New1.lean"],
                            ["Security/Proofs/lakefile.toml", "x.lean", "y.lean"]))
    two = {**tree, f"{POUS}/lean-toolchain": "t1\n", f"{POUS}/Old.lean": None, f"{POUS}/New2.lean": "n2\n", "x.lean": "x1\n",
           "z.lean": "z\n"}
    world.gh.put(2, _pinned(world, two, ["lean-toolchain", "lakefile.toml", UP, "New2.lean"],
                            ["Security/Proofs/lakefile.toml", "x.lean", "z.lean"]))
    raw = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", world.gh.prs[1].head,
                          world.gh.prs[2].head], cwd=world.dev, capture_output=True, text=True)
    assert sorted(raw.stdout.splitlines()[1:]) == sorted([TRUSTED, EXFIL])
    # the preflight and `status` judge it as the sync merges it: no conflict
    assert conflicting_files(world.dev, world.gh.prs[1].head, world.gh.prs[2].head) == []
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2], entries[2].why
    assert f"research queue: #2 conflicts only in pins: {', '.join(sorted([TRUSTED, EXFIL]))} rebuilt from the merged tree" in world.logs
    merged = {k: v for k, v in {**one, **two, f"{POUS}/lakefile.toml": "l1\n", "Security/Proofs/lakefile.toml": "s1\n"}.items()
              if v is not None}
    assert all(_git(world.origin, "show", f"refs/heads/next:{p}") + "\n" == t for p, t in merged.items())
    show = lambda p: subprocess.run(["git", "show", f"refs/heads/next:{p}"], cwd=world.origin, check=True, capture_output=True,
                                    text=True).stdout
    assert show(TRUSTED) == _trusted(["lean-toolchain", "lakefile.toml", UP, "New1.lean", "New2.lean"], merged)
    assert show(EXFIL) == _exfil(["Security/Proofs/lakefile.toml", "x.lean", "y.lean", "z.lean"], merged)
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    assert _git(world.origin, "rev-parse", f"{tip}^2") == world.gh.prs[2].head
    assert _git(world.origin, "log", "-1", "--format=%B", tip) == f"Queue: #2 change 2\n\nPR: #2\nHead: {world.gh.prs[2].head}\nOwner: lane-2"


def test_a_conflict_in_any_file_besides_the_pins_still_refuses_the_merge(world):
    tree = _pin_main(world)
    paths = ["lean-toolchain", "lakefile.toml", "Old.lean", UP]
    for n, f in ((1, "lakefile.toml"), (2, "lean-toolchain")):
        world.gh.put(n, _pinned(world, {**tree, "a.txt": f"{n}\n", f"{POUS}/{f}": f"{n}\n"}, paths, ["x.lean"]))
    assert conflicting_files(world.dev, world.gh.prs[1].head, world.gh.prs[2].head) == sorted(["a.txt", TRUSTED])
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "refused" and entries[2].why == f"conflicts with `next` in a.txt, {TRUSTED} (touched by #1)"
    assert _git(world.origin, "show", f"refs/heads/next:{TRUSTED}") + "\n" == _trusted(paths, {**tree, f"{POUS}/lakefile.toml": "1\n"})


#: a landing gate: refuses a head whose change touches a file `gates/banned.txt` at `main` names, and can't judge one touching
#: crash.txt
GATE = """\
import json, subprocess, sys
main, head = sys.argv[1:3]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
banned = git("show", f"{main}:gates/banned.txt").split()
files = git("diff", "--name-only", git("merge-base", main, head).strip(), head).split()
if "crash.txt" in files:
    print("gate: can't read crash.txt", file=sys.stderr)
    sys.exit(2)
bad = sorted(set(files) & set(banned))
failures = [{"commit": head, "why": f"touches {', '.join(bad)}", "fix": [f"delete {f}" for f in bad]}] if bad else []
print(json.dumps({"failures": failures}))
sys.exit(1 if failures else 0)
"""
GATED = {DRIVE_FLAGS: 'landing_gates = ["gates/g.py"]\n', "gates/g.py": GATE, "gates/banned.txt": "bad.txt\n"}


def _gate_says(main: str, what: str, merged: bool = False) -> str:
    """A refusal of `GATE`'s: of a head (the preflight's), or of its merge onto `next` (the sync's)."""
    f = what.split(" ", 1)[1]
    on, where = (" on its merge onto `next`", "on the branch merged with `origin/next`") if merged else ("", "in the branch")
    return (f"the landing gate gates/g.py at `main` {main[:12]}{on}: {what}; delete {f} (`python3 gates/g.py origin/main HEAD` "
            f"{where} runs it)")


def test_a_pr_a_landing_gate_at_main_refuses_is_left_out_with_its_fix_and_only_the_members_a_sync_merges_are_judged(world, monkeypatch):
    """A landing gate judges a PR as the sync merges it onto `next`, never at landing, where ejecting a
    member low on `next` rebuilds it and cancels the Lean check running there.  A member the sync keeps isn't judged again; one
    it merges again after `main` moved is, and is left out once the gate at the new `main` refuses it."""
    from research import queue as Q

    main = _move_main(world, GATED)
    real, judged = Q.gate_refusal, {}

    def gate_refusal(repo, m, merge, gates, onto):
        judged[_git(repo, "rev-parse", f"{merge}^2")] = merge            # each judged commit is a PR's merge onto `next`
        return real(repo, m, merge, gates, onto=onto)

    monkeypatch.setattr(Q, "gate_refusal", gate_refusal)
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, h2 := world.change({"bad.txt": "2\n", "gates/g.py": "import sys\nsys.exit(0)\n"}))   # its own copy is moot
    world.gh.put(3, world.change({"c.txt": "3\n"}, base=h2))
    world.gh.put(4, h4 := world.change({"d.txt": "4\n"}))
    world.gh.put(5, h5 := world.change({"crash.txt": "5\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 4] == world.on_next(q)
    assert entries[2].state == "refused" and entries[2].why == _gate_says(main, "touches bad.txt", merged=True)
    assert entries[3].state == "waiting" and entries[3].why == "contains #2, which didn't merge"
    assert entries[5].state == "refused" and entries[5].why == (
        f"the landing gate gates/g.py can't judge its merge onto `next` ({judged[h5][:12]}) on `main` {main[:12]} "
        f"(exit 2: gate: can't read crash.txt)")
    assert {n: c["why"] for n, c in q.load_state()["conflicts"].items()} == {"2": entries[2].why, "5": entries[5].why}
    assert q.status()[2].why == entries[2].why + " (at the last sync)"

    judged.clear()
    world.gh.put(4, h4b := world.change({"d.txt": "4 fixed\n"}, base=h4))     # #1 is kept, #4 merged again at its new head
    members, _ = q.sync()
    assert [(m.pr, m.head) for m in members] == [(1, h1), (4, h4b)] and h1 not in judged and set(judged) == {h2, h4b, h5}

    main = _move_main(world, {"gates/banned.txt": "bad.txt\na.txt\n"})       # `main` moves: `next` is rebuilt on it
    members, entries = q.sync()
    assert [m.pr for m in members] == [4]
    assert entries[1].state == "refused" and entries[1].why == _gate_says(main, "touches a.txt", merged=True)


def test_a_member_merged_again_onto_the_tree_it_was_judged_on_takes_its_verdict_and_main_moving_judges_it_again(
        world, monkeypatch):
    """A lift merges every member above the lifted one again, each running the landing gates (~15 s), though those above where
    it was have the same members below and so the same trees.  Their verdicts, a refusal too, are kept under `main` and the gates;
    a new `main` judges every merge again."""
    from research import queue as Q

    main = _move_main(world, GATED)
    real, judged = Q.gate_refusal, []

    def gate_refusal(repo, m, merge, gates, onto):
        judged.append(int(_git(repo, "log", "-1", "--format=%s", merge).split("#")[1].split()[0]))
        return real(repo, m, merge, gates, onto=onto)

    monkeypatch.setattr(Q, "gate_refusal", gate_refusal)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt"), (5, "bad.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2, 3, 4] and judged == [1, 2, 3, 4, 5] and entries[5].state == "refused"
    judged.clear()
    q.lift = [2]
    lifted, entries = q.sync()
    assert [m.pr for m in lifted] == [2, 1, 3, 4] and judged == [2, 1], judged
    assert [_git(world.origin, "rev-parse", f"{m.commit}^{{tree}}") for m in lifted[2:]] == [
        _git(world.origin, "rev-parse", f"{m.commit}^{{tree}}") for m in members[2:]]
    assert entries[5].state == "refused" and entries[5].why == _gate_says(main, "touches bad.txt", merged=True)
    assert world.logs[-1].endswith("kept 0, merged 4, gates' verdicts reused for 3"), world.logs[-1]
    judged.clear()
    _move_main(world, {"x.txt": "x\n"})
    q.lift = []
    assert [m.pr for m in q.sync()[0]] == [1, 2, 3, 4] and judged == [1, 2, 3, 4, 5]


#: the Markdown caps' landing gate and the caps it reads at `main`, committed into the origin as they are in this checkout
CAP_GATE = "tools/verity/check/gates/markdown_caps.py"
CAP_FILES = {DRIVE_FLAGS: f'landing_gates = ["{CAP_GATE}"]\n', **{f: (Path(__file__).resolve().parents[3] / f).read_text()
                                                                   for f in (CAP_GATE, "tests/test_repository.py")}}
#: `doc.md` as `main` holds it (70 bytes), and with line ``at`` replaced by ``text``
LINES = [f"line {i}\n" for i in range(10)]


def _doc(at: int | None = None, text: str = "") -> str:
    return "".join(text if i == at else line for i, line in enumerate(LINES))


def test_a_pr_whose_merge_onto_next_puts_a_markdown_file_over_its_cap_is_left_out_and_the_members_below_stay(world):
    """A PR whose head alone is under a Markdown file's cap can take the file over it once merged onto `next` above the members
    that also grow it, and then every Lean check above it fails `preflight-lints` until someone trims the file.  The sync gives the landing gates each PR's merge onto `next`, so the bytes the members below add count."""
    main = _move_main(world, {**CAP_FILES, "doc.md": _doc()})
    world.gh.put(1, h1 := world.change({"doc.md": _doc(1, "x" * 30000 + "\n")}))            # 30,064 bytes: under 49,152
    world.gh.put(2, h2 := world.change({"doc.md": _doc(8, "y" * 30000 + "\n")}))            # as much alone, 60,058 onto #1
    world.gh.put(3, h3 := world.change({"doc.md": _doc(5, "short\n")}))                     # 30,063 merged onto #1: under
    for h in (h1, h2, h3):                                                                  # each head alone passes
        assert subprocess.run([sys.executable, CAP_GATE, main, h], cwd=world.dev, capture_output=True).returncode == 0
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 3] == world.on_next(q)
    assert entries[2].state == "refused"
    said = entries[2].why
    assert said.startswith(f"the landing gate {CAP_GATE} at `main` {main[:12]} on its merge onto `next`: doc.md is 60058 bytes "
                           "at "), said
    assert (f"over its 49152-byte cap (tests/test_repository.py), and was 30064 bytes at {members[0].commit[:12]} below; trim "
            "doc.md to 49152 bytes or fewer (10906 over): move detail to the notes store (cite it by note:<id>) (`python3 "
            f"{CAP_GATE} origin/main HEAD` on the branch merged with `origin/next` runs it)") in said, said
    assert q.load_state()["conflicts"]["2"]["why"] == said
    assert int(_git(world.origin, "cat-file", "-s", "refs/heads/next:doc.md")) == 30063

    again, entries = q.sync()                                    # the members below stay; #2 is judged again, and refused
    assert [m.commit for m in again] == [m.commit for m in members] and entries[2].state == "refused"


def test_a_file_already_over_its_cap_below_refuses_only_a_pr_whose_merge_grows_it(world, monkeypatch):
    """The gate judges only what a merge adds, against the tip it merged onto.  A member kept on `next` that is
    already over a cap (merged before the gate was listed) mustn't make the gate refuse every PR above it: one that doesn't touch
    the file merges, one that shrinks it merges (the gate notes it), and one that grows it is refused."""
    from research import queue as Q

    _move_main(world, {**CAP_FILES, "doc.md": _doc(), "other.md": "other\n"})
    world.gh.put(1, world.change({"doc.md": _doc(1, "x" * 60000 + "\n")}))                  # 60,064 bytes: over 49,152
    q = world.queue()
    with monkeypatch.context() as m:
        m.setattr(Q, "gate_refusal", lambda *a, **k: ([], []))                              # no gate listed yet
        [kept], _ = q.sync()
    world.gh.put(2, world.change({"a.txt": "2\n", "other.md": "other, longer\n"}))
    world.gh.put(3, world.change({"doc.md": _doc(8, "y" * 10 + "\n")}))                     # grows it
    world.gh.put(4, world.change({"doc.md": _doc(5, "l\n")}))                                # shrinks it, still over
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2, 4] == world.on_next(q) and members[0].commit == kept.commit
    assert entries[3].state == "refused" and "doc.md is 60068 bytes at " in entries[3].why
    assert f"and was 60064 bytes at {members[1].commit[:12]} below" in entries[3].why
    merge = members[2].commit
    doc = json.loads(subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / CAP_GATE), "origin/main", merge, "--json"], cwd=q.workdir,
                                    capture_output=True, text=True).stdout)
    assert doc["below"] == members[1].commit and doc["failures"] == [] and doc["notes"] == [
        f"doc.md is 60059 bytes, over its 49152-byte cap, but was 60064 bytes at {members[1].commit[:12]} below: not this change's"]


def test_status_names_a_ready_pr_that_conflicts_with_main_without_a_sync_here(world):
    """`status` on a VM other than drive's doesn't call a PR that conflicts with `main` 'admitted' just because only the drive
    VM's last sync recorded the conflict."""
    world.gh.put(1, world.change({"a.txt": "pr\n"}))
    world.gh.put(2, world.change({"b.txt": "fine\n"}))
    _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'main moved\n'})}:refs/heads/main")
    got = world.queue().status()
    assert got[1].state == "refused" and got[1].why.startswith("conflicts with `main` in a.txt")
    assert got[2].state == "ready" and got[2].why.startswith("admitted")


def test_a_pr_containing_another_goes_in_after_it_and_waits_while_it_waits(world):
    parent = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, parent, ready=False)
    world.gh.put(4, world.change({"c.txt": "stacked\n"}, base=parent))
    q = world.queue()
    members, entries = q.sync()
    assert members == [] and entries[5].why.startswith(f"not marked ready at {parent[:8]}")
    assert entries[4].state == "waiting" and entries[4].after == (5,)
    world.gh.ready.add((5, parent))
    members, _ = q.sync()
    assert [m.pr for m in members] == [5, 4]


def test_a_pr_holding_an_older_head_of_an_unmarked_pr_waits_for_it_and_shared_history_alone_holds_nothing(world):
    """A PR that branched from another at an older head, which then moved on unmarked, isn't admitted with the other's unmarked
    commits in it, as it would be if drive looked only for the other's current head."""
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old, ready=False)
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))             # #7 branched from #5 at `old`
    q = world.queue()
    q.sync()                                                                    # the queue sees #5 at `old`
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old), ready=False)  # #5 moved on from `old`, unmarked
    base = world.change({"m.txt": "landed\n"})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    world.gh.put(8, world.change({"y.txt": "y\n"}, base=base))                 # shares only `main`'s history with #5
    world.gh.put(9, world.change({"z.txt": "z\n"}, base=world.gh.prs[5].head))  # on #5's current head: as before
    members, entries = q.sync()
    assert [m.pr for m in members] == [8]
    assert entries[7].state == "waiting" and entries[7].why == "contains #5's commits (at an older head), which isn't admitted"
    assert entries[7].after == entries[7].older == (5,) and entries[5].after == ()
    assert entries[8].after == () and entries[9].why == "contains #5, which isn't admitted" and entries[9].older == ()
    assert q.status()[7].why == entries[7].why
    world.gh.ready.add((5, world.gh.prs[5].head))
    assert [m.pr for m in q.sync()[0]] == [8, 5, 7, 9]


@pytest.mark.parametrize("marked_at", ["nowhere", "older head"])
def test_a_pr_holding_an_older_head_of_a_pr_not_admitted_at_its_current_head_waits_for_it(world, marked_at):
    """As above, with #5 unmarked or marked only at the older head #7 holds."""
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old, ready=marked_at == "older head")
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))             # #7 branched from #5 at `old`, marked
    q = world.queue()
    q.sync()
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old), ready=False)  # #5 moved on from `old`
    members, entries = q.sync()
    assert 7 not in [m.pr for m in members] and entries[7].state == "waiting", entries[7].why
    world.gh.ready.add((5, world.gh.prs[5].head))
    assert [m.pr for m in q.sync()[0]] == [5, 7]


def _merge(world: World, into: str, other: str) -> str:
    """A merge commit of ``other`` into ``into`` (first parent ``into``)."""
    _git(world.dev, "checkout", "-q", "--detach", into)
    _git(world.dev, "merge", "-q", "--no-edit", other)
    sha = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/dev/{sha}")
    return sha


@pytest.mark.parametrize("marked", [True, False])
def test_the_pr_that_merged_another_prs_branch_waits_for_it_and_never_the_other_way_by_number(world, marked):
    """#5, opened first, merges #6's branch, and #6 moves on; a rule that took the lower number to carry the higher one's
    commits would make #6, queued, wait for #5.  The shared commits are on #6's first-parent line and reach #5 only through a
    merge, so #5 carries #6: #5 waits for #6, #6 for nothing.  The queue never saw #6 at the merged head: ancestry alone
    decides it."""
    period = world.change({"p.txt": "period\n"})
    world.gh.put(6, world.change({"m.txt": "merged main\n"}, base=period))        # #6's branch
    soundness = world.change({"s.txt": "soundness\n"})
    world.gh.put(5, _merge(world, soundness, world.gh.prs[6].head))              # #5 merges #6's branch
    world.gh.put(6, world.change({"p.txt": "period 2\n"}, base=world.gh.prs[6].head), ready=marked)  # #6 moves on
    members, entries = world.queue().sync()
    assert entries[6].after == () and entries[5].after == entries[5].older == (6,)
    if marked:
        assert [m.pr for m in members] == [6, 5]
    else:
        assert members == [] and entries[6].why.startswith("not marked ready")
        assert entries[5].why == "contains #6's commits (at an older head), which isn't admitted"


def test_prs_that_merely_share_commits_wait_for_nothing(world):
    """Two PRs that each merged a third branch (on neither first-parent line), and two forked from a commit the queue never saw
    either at (on both): neither knows which carries the other, so neither waits, whatever their numbers."""
    third = world.change({"t.txt": "third\n"})
    world.gh.put(9, _merge(world, world.change({"a9.txt": "9\n"}), third), ready=False)
    world.gh.put(10, _merge(world, world.change({"a10.txt": "10\n"}), third))
    fork = world.change({"f.txt": "fork\n"})
    world.gh.put(11, world.change({"a11.txt": "11\n"}, base=fork), ready=False)
    world.gh.put(12, world.change({"a12.txt": "12\n"}, base=fork))
    members, entries = world.queue().sync()
    assert {n: entries[n].after for n in (9, 10, 11, 12)} == {9: (), 10: (), 11: (), 12: ()}
    assert [m.pr for m in members] == [10, 12]


def _resolved_main(world: World, head: str) -> str:
    """``head`` with `main` merged into it, each conflict taken from `main`: it contains ``head`` but no longer conflicts."""
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", head)
    _git(world.dev, "merge", "-q", "--no-edit", "-X", "theirs", "origin/main")
    sha = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/dev/{sha}")
    return sha


def test_a_ready_pr_containing_one_that_conflicts_with_main_shows_waiting_naming_it_and_the_sync_records_why(world):
    """A ready PR the sync leaves out because a PR it contains conflicts with `main` isn't called 'admitted' by `status`: the
    sync records why, and `status` shows it waiting, naming the contained PR."""
    a = world.change({"a.txt": "pr\n"})
    _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'main moved\n'})}:refs/heads/main")
    b = _resolved_main(world, world.change({"b.txt": "b2\n"}, base=a))
    world.gh.put(1, a)
    world.gh.put(2, b)
    q = world.queue()
    got = q.status()
    assert got[1].state == "refused" and got[1].why.startswith("conflicts with `main` in a.txt")
    assert got[2].state == "waiting" and got[2].why == f"contains #1, which is refused: {got[1].why}"
    members, entries = q.sync()
    assert members == [] and entries[2].state == "waiting" and entries[2].why == "contains #1, which didn't merge"
    assert {n: (w["head"], w["why"]) for n, w in q.load_state()["waiting"].items()} == {"2": (b, entries[2].why)}
    assert sum("#2 at" in line and "waits off `next`" in line for line in world.logs) == 1
    q.sync()
    assert sum("#2 at" in line and "waits off `next`" in line for line in world.logs) == 1      # once per head
    assert q.status()[2].why.startswith("contains #1, which is refused: ") and q.status()[2].why.endswith("(at the last sync)")


def test_status_judges_a_chain_of_contained_prs_each_after_what_it_contains(world):
    """C contains B contains A, and A conflicts with `main`: B waits naming A, and C waits too, numbered below both so a judgment
    in number order would have called it admitted."""
    a = world.change({"a.txt": "pr\n"})
    _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'main moved\n'})}:refs/heads/main")
    b = _resolved_main(world, world.change({"b.txt": "b2\n"}, base=a))
    c = world.change({"c.txt": "c3\n"}, base=b)
    world.gh.put(9, a)
    world.gh.put(5, b)
    world.gh.put(1, c)
    got = world.queue().status()
    assert got[9].state == "refused"
    assert got[5].state == "waiting" and got[5].why.startswith("contains #9, which is refused: conflicts with `main` in a.txt")
    assert got[1].state == "waiting" and got[1].why.startswith("contains #5, which is waiting: contains #9, which is refused")
    members, entries = world.queue().sync()
    assert members == [] and entries[1].why == "contains #5, which didn't merge"
    assert set(world.queue().load_state()["waiting"]) == {"1", "5"}


def _based_on(world: World, n: int, base: str, branch: str = "") -> None:
    import dataclasses

    world.gh.prs[n] = dataclasses.replace(world.gh.prs[n], base=base, branch=branch or f"lane-{n}/work")


def test_status_lists_a_marked_pr_based_on_another_branch_as_waiting_and_drive_still_admits_only_main_based_ones(world):
    """A PR marked ready while based on another PR's branch isn't invisible to `status` until its parent lands and someone
    retargets it."""
    parent = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, parent)
    _based_on(world, 5, "main", "lane-5/parent")
    world.gh.put(6, world.change({"c.txt": "stacked\n"}, base=parent))
    _based_on(world, 6, "lane-5/parent")
    world.gh.put(7, world.change({"b.txt": "elsewhere\n"}))
    _based_on(world, 7, "some/branch")
    world.gh.put(8, world.change({"b.txt": "unmarked\n"}), ready=False)
    _based_on(world, 8, "some/branch")
    q = world.queue()
    got = q.status()
    assert got[6].state == "waiting" and got[6].why == "based on lane-5/parent (#5's head), not main: retarget it to main"
    assert got[7].state == "waiting" and got[7].why == "based on some/branch, not main: retarget it to main"
    assert 8 not in got and got[5].state == "ready"
    members, entries = q.sync()
    assert [m.pr for m in members] == [5] and set(entries) == {5}


def test_ready_marks_a_pr_based_on_another_prs_branch_and_says_drive_wont_see_it(world, capsys):
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    parent = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, parent, ready=False)
    _based_on(world, 5, "main", "lane-5/parent")
    world.gh.put(6, child := world.change({"c.txt": "stacked\n"}, base=parent), ready=False)
    _based_on(world, 6, "lane-5/parent")
    assert ready(world.gh, world.store, 6, "lane-6", MarkOnly()) == 0
    note = ("#6 is based on lane-5/parent (#5's head), not main: drive won't see it until #5 lands, one stack level per landing. "
            "Retarget it to main (drive orders stacked PRs itself: a PR containing another waits for it)")
    assert note in capsys.readouterr().err
    assert StoreLabels(world.store, log=world.logs.append).ready(6, child) is True

    _based_on(world, 6, "some/branch")
    assert ready(world.gh, world.store, 6, "lane-6", MarkOnly(), as_json=True) == 0
    out = capsys.readouterr()
    got = json.loads(out.out)
    assert got["pr"] == 6 and got["head"] == child and got["base"] == "some/branch" and got["exit"] == 0
    assert got["warning"].startswith("#6 is based on some/branch, not main: drive won't see it until it is. Retarget it to main")
    assert "marked on lane-6's word" in out.err

    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly()) == 0
    assert "not main" not in capsys.readouterr().err


def test_a_pr_carries_a_contained_unmarked_pr_of_its_own_owner_or_one_whose_owner_withdrew_it(world):
    """Drive doesn't hold a PR containing another open PR's head until that one closes, since no one's tooling may close a
    PR, and a lane would have to rebuild it by hand.  A rebuild by the same owner, or one whose owner withdrew the PR it
    contains, carries that change and lands without it."""
    parent = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, parent, ready=False, body="Lane `Lane-4`.")
    world.gh.put(4, world.change({"c.txt": "stacked\n"}, base=parent))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [4] and entries[4].after == () and entries[5].state == "waiting"
    world.gh.put(5, parent, ready=False)                                    # another lane's PR: #4 waits behind it
    members, entries = q.sync()
    assert members == [] and entries[4].why == "contains #5, which isn't admitted"
    q.marked_by = lambda n, h: "lane-5" if n == 5 else None                 # #5's owner wrote its `ready = false`
    assert [m.pr for m in q.sync()[0]] == [4]
    q.before_cutoff = lambda n, h: "2026-10-01T00:00Z" if n == 5 else None   # its owner's `ready = true`, but before cutoff
    assert q.sync()[0] == []
    q.marked_by = q.before_cutoff = None
    assert q.sync()[0] == []
    world.gh.put(5, parent)
    q.eject(5, "superseded by #4", "lane-5")
    members, entries = q.sync()
    assert [m.pr for m in members] == [4] and entries[5].state == "refused"
    world.gh.put(5, parent, ready=False, labels=("hold",))                  # a hold isn't carried
    assert q.sync()[0] == []


def test_an_ejected_head_stays_out_until_its_pr_is_pushed_again(world):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    q.sync()
    q.eject(2, "breaks suite `verity` at r20260929-000000-abcd", "test")
    assert world.on_next(q) == [1, 2, 3]                        # an ejection takes effect at the next sync
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 3] and entries[2].why.startswith("ejected at")
    assert "breaks suite `verity`" in entries[2].why                # the notice is `tell_ejected`'s post, tested below
    world.gh.put(2, world.change({"b.txt": "fixed\n"}))
    assert [m.pr for m in q.sync()[0]] == [1, 3, 2]


def test_a_main_moved_outside_the_queue_rebuilds_next_on_it(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    q = world.queue()
    q.sync()
    moved = world.change({"d.txt": "landed by hand\n"})
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    members, _ = q.sync()
    assert [m.pr for m in members] == [1, 2]
    _git(world.origin, "merge-base", "--is-ancestor", moved, "refs/heads/next")      # raises unless next is on the new main


def test_the_clone_follows_the_callers_origin_so_a_url_that_stopped_working_does_not_strand_it(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.queue().sync()
    moved = world.tmp / "origin-now.git"
    world.origin.rename(moved)                  # the URL the clone was made from fails now, as one with a rotated token does
    world.origin = world.gh.origin = moved
    members, _ = world.queue().sync()
    assert [m.pr for m in members] == [1]
    assert _git(world.tmp / "q" / "verity", "remote", "get-url", "origin") == str(moved)


HTTP_BACKEND = Path(subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout.strip()) / "git-http-backend"


class _GitHTTP(BaseHTTPRequestHandler):
    """`git http-backend` over `root` behind Basic auth as `u:p`, as GitHub serves a private repository: without credentials, a 401."""
    root: Path

    def do_GET(self):
        if self.headers.get("Authorization") != "Basic " + base64.b64encode(b"u:p").decode():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="origin"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        path, _, query = self.path.partition("?")
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        env = {"PATH": os.environ["PATH"], "GIT_PROJECT_ROOT": str(self.root), "GIT_HTTP_EXPORT_ALL": "1", "REMOTE_USER": "u",
               "PATH_INFO": path, "QUERY_STRING": query, "REQUEST_METHOD": self.command, "CONTENT_LENGTH": str(len(body)),
               "CONTENT_TYPE": self.headers.get("Content-Type", ""), "HTTP_CONTENT_ENCODING": self.headers.get("Content-Encoding", ""),
               "GIT_PROTOCOL": self.headers.get("Git-Protocol", "")}
        cgi = subprocess.run([str(HTTP_BACKEND)], input=body, env=env, capture_output=True, check=True).stdout
        head, _, out = cgi.partition(b"\r\n\r\n")
        headers = [line.split(": ", 1) for line in head.decode().split("\r\n") if line]
        self.send_response(int(next((v for k, v in headers if k == "Status"), "200").split()[0]))
        for k, v in headers:
            if k != "Status":
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_POST = do_GET

    def log_message(self, *args):
        pass


def _no_machine_git_config(tmp: Path, monkeypatch) -> None:
    """A cloud VM's view: no credential helper (nor prompt) from git's global or system config, only from a checkout's own."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp / "global.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_ASKPASS", "SSH_ASKPASS"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def served(world, monkeypatch):
    """`world`'s origin over http behind a password; its URL."""
    _no_machine_git_config(world.tmp, monkeypatch)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (_GitHTTP,), {"root": world.tmp}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/{world.origin.name}"
    srv.shutdown()
    srv.server_close()


@pytest.mark.skipif(not HTTP_BACKEND.exists(), reason="no git-http-backend in this git")
def test_the_clone_authenticates_as_the_callers_checkout_does_and_fails_rather_than_prompt_or_answer_stale(world, served):
    """A cloud VM's credential helper is in its checkout's own config (one that needs the path, as the VMs' broker does), which the
    queue's own clone never read, so `status` failed there at the clone.  It takes the checkout's settings for the remote now; and
    with no credentials git fails at once instead of prompting, also once the clone has fetched before."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _git(world.dev, "config", f"credential.{served}.helper", "")
    _git(world.dev, "config", "--add", f"credential.{served}.helper",
         "!f() { test \"$1\" = get && grep -qx path=origin.git && printf 'username=u\\npassword=p\\n'; }; f")
    _git(world.dev, "config", f"credential.{served}.useHttpPath", "true")

    def queue(auth: tuple[str, ...]) -> Queue:
        return Queue(world.tmp / "q" / "verity", remote=served, github=world.gh, ready=lambda n, h: True,
                     log=world.logs.append, auth=auth)

    with pytest.raises(QueueError, match=r"git clone .*terminal prompts disabled; no credential helper applies to the remote"):
        queue(()).status()
    auth = credential_args(world.dev, served)
    members, _ = queue(auth).sync()
    assert [m.pr for m in members] == [1] and _git(world.origin, "rev-parse", "refs/heads/next") == members[0].commit
    assert [(n, e.state) for n, e in queue(auth).status().items()] == [(1, "queued")]
    with pytest.raises(QueueError, match=r"git fetch .*could not read Username"):
        queue(()).status()


def test_credential_args_take_the_checkouts_own_settings_else_gh_where_no_helper_applies(tmp_path, monkeypatch):
    _no_machine_git_config(tmp_path, monkeypatch)
    url, repo, bin_dir = "https://verity-agent@github.com/o/r", tmp_path / "checkout", tmp_path / "bin"
    _git(tmp_path, "init", "-q", str(repo))
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/bin/sh\n")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    gh = ("-c", "credential.helper=", "-c", f"credential.helper=!{shlex.quote(str(bin_dir / 'gh'))} auth git-credential")
    assert credential_args(repo, url) == credential_args(tmp_path, url) == gh, "no helper anywhere, in a checkout or not: gh's"
    assert credential_args(repo, "/srv/origin.git") == credential_args(repo, "git@github.com:o/r") == (), "not http: no credentials"
    _git(repo, "config", "credential.https://verity-agent@github.com.helper", "")
    _git(repo, "config", "--add", "credential.https://verity-agent@github.com.helper", "!broker credential")
    _git(repo, "config", "credential.https://verity-agent@github.com.useHttpPath", "true")
    _git(repo, "config", "credential.https://elsewhere.example.helper", "!not-this-one")
    assert credential_args(repo, url) == ("-c", "credential.helper=", "-c", "credential.helper=!broker credential",
                                          "-c", "credential.usehttppath=true")
    (tmp_path / "global.gitconfig").write_text("[credential]\n\thelper = store\n")
    assert credential_args(tmp_path, url) == (), "a helper from git's global config serves the clone as it is"


def test_a_rewritten_pr_head_is_fetched_by_force_and_merged(world):
    old = world.change({"a.txt": "1\n"})
    world.gh.put(1, old)
    q = world.queue()
    q.sync()
    new = world.change({"a.txt": "rewritten\n"})                # not a descendant of the old head: a force-push
    world.gh.put(1, new)
    members, _ = q.sync()
    assert members[0].head == new and _git(world.origin, "show", "refs/heads/next:a.txt") == "rewritten"


def test_pr_label_targets(tmp_path, monkeypatch):
    head = "0123456789abcdef0123456789abcdef01234567"
    assert ids.is_pr(label_target(12, head)) and ids.kind_of_id(f"pr:12@{head}") == "pr"
    assert not any(ids.is_pr(x) for x in (f"pr:0@{head}", f"pr:12@{head[:12]}", f"pr:x@{head}", f"12@{head}"))
    assert Label(target=f"pr:12@{head[:8]}", key="ready", value=True, by="t", ts="2026-09-29T05:30:00Z").errors()
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane, queue_host = LocalStore(tmp_path / "lane"), LocalStore(tmp_path / "queue")
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-09-29T05:30:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert len(labels_sync.pull_targets(queue_host, [label_target(12, head)])["pulled"]) == 1
    assert [lab.value for lab in queue_host.labels(label_target(12, head), key=vocab.READY)] == [True]


class _Unreachable:
    """A configured remote whose endpoint is down: every call raises."""

    def describe(self):
        return "fs:/unreachable"

    def __getattr__(self, name):
        def fail(*a, **k):
            raise OSError("injected: endpoint unreachable")
        return fail


def test_every_git_command_that_talks_to_the_remote_aborts_a_stalled_transfer(tmp_path, monkeypatch):
    import research.queue as Q

    q = World(tmp_path).queue()
    seen = []
    real = Q._git
    monkeypatch.setattr(Q, "_git", lambda cwd, *a, **k: seen.append(a) or real(cwd, *a, **k))
    q._clone()
    q.fetch()
    remote = [a for a in seen if {"clone", "fetch", "push", "ls-remote"} & set(a)]
    assert remote and all(a[: len(Q.STALL)] == Q.STALL for a in remote), seen


def test_ready_marks_read_through_to_the_remote_and_without_that_pull_the_queue_refuses_to_answer(tmp_path, monkeypatch):
    """A failed pull, an unusable remote, or none at all (a cloud VM's shell that lost its store secrets) used to leave the queue
    answering from this machine's labels, with a warning at most, so its ready marks could be stale.  Each is `STALE` now."""
    from research.queue import StoreLabels

    head = "0123456789abcdef0123456789abcdef01234567"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane, queue_host = LocalStore(tmp_path / "lane"), LocalStore(tmp_path / "queue")
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-09-29T05:31:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    logs: list[str] = []
    labels = StoreLabels(queue_host, log=logs.append)
    real = LocalStore.remote.fget
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    with pytest.raises(QueueError, match=rf"^STALE: the labels of pr:12@{head} weren't pulled from the store's remote "
                                         r"\(pulling from fs:/unreachable failed \(OSError: injected"):
        labels.ready(12, head)
    monkeypatch.setattr(LocalStore, "remote", property(real))
    assert labels.ready(12, head) is True and logs == []
    # the labels are on this machine now, and still don't answer without a pull
    absent, unusable = tmp_path / "absent.toml", tmp_path / "s3.toml"
    unusable.write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                        'access_key_id_env = "Q_AK_ABSENT"\nsecret_access_key_env = "Q_SK_ABSENT"\n')
    with pytest.raises(QueueError, match=r"^STALE: .*\(no remote configured"):
        StoreLabels(LocalStore(tmp_path / "queue", config_path=absent)).ready(12, head)
    with pytest.raises(QueueError, match=r"^STALE: .*Q_AK_ABSENT"):
        StoreLabels(LocalStore(tmp_path / "queue", config_path=unusable)).ready(12, head)
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", "R2_ENDPOINT,R2_BUCKET")
    with pytest.raises(QueueError, match=r"^STALE: .*R2_ENDPOINT, R2_BUCKET not in this shell .*fresh tmux login shell"):
        StoreLabels(LocalStore(tmp_path / "queue", config_path=absent)).ready(12, head)


def test_a_cutoff_admits_only_ready_marks_written_at_or_after_it(tmp_path, monkeypatch):
    """`--marked-after`: a mark then means "land this now", and older marks need nothing from their owners."""
    from research.queue import StoreLabels

    head = "0123456789abcdef0123456789abcdef01234567"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane = LocalStore(tmp_path / "lane")
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-12T05:59:59Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q1")).ready(12, head) is True
    assert StoreLabels(LocalStore(tmp_path / "q2"), since="2026-10-12T06:00:00Z").ready(12, head) is False
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-12T06:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q3"), since="2026-10-12T06:00:00Z").ready(12, head) is True
    lane.add_label(label_target(12, head), vocab.READY, False, "memory-accounting", ts="2026-10-12T10:20:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q4")).ready(12, head) is False         # an ejection, seen by every queue
    lane.add_label(label_target(12, head), vocab.READY, True, "lane-12", ts="2026-10-12T11:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(12, head)])["failed"] == []
    assert StoreLabels(LocalStore(tmp_path / "q5")).ready(12, head) is True          # until its owner marks it again


def test_ready_marks_are_pulled_for_every_pr_at_once_and_each_pr_alone_when_that_fails(tmp_path, monkeypatch):
    """One listing of the store's remote per open PR, in turn, would take seconds of every drive tick."""
    from research.queue import StoreLabels

    heads = {n: f"{n:040x}" for n in (1, 2, 3)}
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane = LocalStore(tmp_path / "lane")
    for n, value in ((1, True), (2, False)):
        lane.add_label(label_target(n, heads[n]), vocab.READY, value, f"lane-{n}", ts="2026-10-12T07:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(n, heads[n]) for n in (1, 2)])["failed"] == []
    pulls: list[tuple[int, int]] = []
    pull = labels_sync.pull_targets
    monkeypatch.setattr(labels_sync, "pull_targets", lambda store, targets, jobs=1: pulls.append((len(targets), jobs))
                        or pull(store, targets, jobs=jobs))
    labels = StoreLabels(LocalStore(tmp_path / "q1"))
    labels.prefetch(list(heads.items()))
    assert [labels.ready(n, heads[n]) for n in (1, 2, 3)] == [True, False, False]
    assert pulls == [(3, labels_sync.SYNC_JOBS)]

    pulls.clear()
    monkeypatch.setattr(labels_sync, "remote_labels_for", lambda r, t, f=labels_sync.remote_labels_for:
                        (_ for _ in ()).throw(OSError("listing refused")) if not pulls[1:] else f(r, t))
    labels = StoreLabels(LocalStore(tmp_path / "q2"))
    labels.prefetch(list(heads.items()))
    assert [labels.ready(n, heads[n]) for n in (1, 2, 3)] == [True, False, False]
    assert pulls == [(3, labels_sync.SYNC_JOBS), (1, 1), (1, 1), (1, 1)]


def test_admission_prefetches_the_marks_of_every_pr_it_will_ask_of(world):
    asked: list[list[tuple[int, str]]] = []
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}), labels=("hold",))
    world.gh.put(3, h3 := world.change({"c.txt": "3\n"}))
    world.queue(prefetch=asked.append).sync()
    assert asked == [[(1, h1), (3, h3)]]


def test_status_on_any_vm_applies_the_rules_cutoff_as_drive_does(world, monkeypatch, capsys):
    """`research queue status` on any VM, run without `--marked-after`, doesn't list as admitted old marks drive, with its
    cutoff, never takes.  The cutoff is `marked_after` in the rules on `main`, and a PR whose mark it ignores says so, so its
    owner re-marks it or closes it rather than it sitting unseen."""
    import research.queue as Qm

    cfg = world.tmp / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{world.tmp / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    (world.dev / "ci" / "queue.toml").write_text(RULES + 'marked_after = "2026-10-12T06:00:00Z"\n')
    _git(world.dev, "commit", "-qam", "the cutoff")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    lane = LocalStore(world.tmp / "lane")
    for n, ts in ((1, "2026-10-05T09:00:00Z"), (2, "2026-10-12T07:00:00Z")):
        world.gh.put(n, head := world.change({f"{n}.txt": "x\n"}), ready=False)
        lane.add_label(label_target(n, head), vocab.READY, True, f"lane-{n}", ts=ts)
        assert labels_sync.push_targets(lane, [label_target(n, head)])["failed"] == []
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    argv = ["queue", "status", "--json", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin),
            "--repo", "o/verity", "--store", str(world.tmp / "queue-store")]
    assert research_main(argv) == 0
    got = {e["pr"]: (e["state"], e["why"]) for e in json.loads(capsys.readouterr().out)}
    assert got[1][0] == "waiting" and got[1][1].startswith("ignored: marked before cutoff") and "2026-10-05T09:00:00Z" in got[1][1]
    assert got[2][0] == "ready" and "the next sync merges it onto `next`" in got[2][1]
    assert research_main([*argv, "--marked-after", "2026-10-01T00:00:00Z"]) == 0
    assert {e["pr"]: e["state"] for e in json.loads(capsys.readouterr().out)} == {1: "ready", 2: "ready"}


def test_an_ejection_withdraws_the_heads_ready_mark_in_the_store(world):
    """An ejection doesn't live only in the ejecting VM's queue state, where drive, on another VM, would rebuild `next` with
    the ejected PR."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q = world.queue()
    q.eject(1, "held for Daniel's decision 10", "memory-accounting")
    head = next(p.head for p in world.gh.open_prs() if p.number == 1)
    (lab,) = world.store.labels(label_target(1, head), key=vocab.READY)
    assert (lab.value, lab.by, lab.ref) == (False, "memory-accounting", "held for Daniel's decision 10")


def test_an_ejection_naming_a_head_refuses_once_its_owner_pushed_another(world):
    """An eject waiting for drive's tick mustn't withdraw the mark of a head pushed meanwhile.  With `--sha` it acts only while that head is the PR's, and a newer head keeps its mark."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q = world.queue()
    old = next(p.head for p in world.gh.open_prs() if p.number == 1)
    world.gh.put(1, world.change({"a.txt": "fixed\n"}))
    new = next(p.head for p in world.gh.open_prs() if p.number == 1)
    with pytest.raises(QueueError, match=f"#1 is at {new[:12]} now, not {old[:12]}"):
        q.eject(1, "breaks a suite", "ci", sha=old)
    assert "1" not in q.load_state()["ejected"] and not world.store.labels(label_target(1, new), key=vocab.READY)
    assert q.eject(1, "breaks a suite", "ci", sha=new[:12]).head == new and q.load_state()["ejected"]["1"]["head"] == new


def test_a_hand_ejection_is_posted_as_its_ejector_to_ci_and_the_owner_and_says_so_when_no_one_was_told(world, monkeypatch, capsys):
    """A hand eject's notice isn't only a handoff note on the ejecting VM, which syncs no notes, and the eject doesn't log that
    it told the owner when it didn't."""
    import research.drive as D
    import research.queue as Qm

    cfg = world.tmp / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{world.tmp / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setenv("RESEARCH_NOTES", str(world.tmp / "notes"))
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    reg = world.dev / Qm.REGISTRY
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"handles": [{"handle": "ci"}, {"handle": "lane-1"}]}))
    _git(world.dev, "add", "-A")
    _git(world.dev, "commit", "-qm", "the registry")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    posts, ok = [], [True]
    monkeypatch.setattr(D, "merge_thread_post", lambda text, to, sender="ci": posts.append((text, to, sender)) or ok[0])
    argv = ["queue", "eject", "1", "--reason", "breaks suite `verity`", "--by", "ci", "--workdir", str(world.tmp / "q" / "verity"),
            "--remote", str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "queue-store")]
    assert research_main(argv) == 0
    ((text, to, sender),) = posts
    assert (to, sender) == (("@ci", "@lane-1"), "ci")
    assert text.startswith(f"@ci @lane-1 queue: #1 at {head[:8]} is ejected from `next` by ci: breaks suite `verity`")
    ok[0] = False
    assert research_main(argv) == 1
    err = capsys.readouterr().err
    assert "#1 is ejected, but posting its notice as @ci failed, so no one was told" in err
    assert "research msg send --as HANDLE --thread" in err and "--to @ci @lane-1" in err


def test_off_drive_the_queue_verbs_never_push_next_or_main(world, monkeypatch, capsys):
    """GitHub's rulesets let only drive's App move `main` and `next`, yet don't stop verity-agents[bot] force-pushing `next` from a
    VM's `research queue eject` that syncs. Off drive an eject only withdraws the mark; sync refuses."""
    import research.drive as D
    import research.queue as Qm

    monkeypatch.delenv("RESEARCH_DRIVE")
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    monkeypatch.setattr(D, "merge_thread_post", lambda *a, **k: True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    base = ["--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo", "o/verity",
            "--store", str(world.tmp / "queue-store")]
    for verb in ("sync",):
        assert research_main(["queue", verb, *base]) == 1
        assert f"/data/infra/drive.sh queue {verb}" in capsys.readouterr().err
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    research_main(["queue", "eject", "1", "--reason", "x", "--by", "ci", *base])
    assert refs() == before and "drive's next tick rebuilds `next` without it" in capsys.readouterr().out


def test_a_ready_mark_written_after_an_ejection_re_admits_the_same_head(world):
    """A PR re-marked by its owner at its ejected head doesn't stay off `next` on the ejecting VM's queue alone: the queue
    state's record of the ejection doesn't hold it against the store's newest mark."""
    from research.queue import StoreLabels

    world.with_store_remote()
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    target = label_target(1, head)
    world.store.add_label(target, vocab.READY, True, "lane-1", ts="2026-10-12T21:00:00Z")
    labels = StoreLabels(world.store)
    q = world.queue(ready=labels.ready, marked_at=labels.marked_at)
    assert [m.pr for m in q.sync()[0]] == [1]
    q.eject(1, "breaks the repository suite", "ci")
    members, entries = q.sync()
    assert members == [] and entries[1].why.startswith("ejected at")
    world.store.add_label(target, vocab.READY, True, "lane-1", ts="2999-01-01T00:00:00Z")
    assert [m.pr for m in q.sync()[0]] == [1]

    state = q.load_state()                      # an ejection recorded to the minute, with no withdrawn mark in the store
    state["ejected"]["1"] = {"head": head, "why": "w", "by": "ci", "at": "2999-01-01T00:00Z"}
    q.save_state(state)
    assert q.sync()[0] == [], "a mark from the ejection's own minute may be older than it"
    state["ejected"]["1"]["at"] = "2998-12-31T23:59Z"
    q.save_state(state)
    assert [m.pr for m in q.sync()[0]] == [1]


def test_rules_name_the_hold_label_and_refuse_anything_else():
    assert Rules.parse(RULES).hold_label == "hold" and Rules.parse("").hold_label == "hold"
    for bad in ('[[grant]]\nrole = "red-team"\npaths = ["x/"]\n', 'colour = "red"\n'):
        with pytest.raises(QueueError):
            Rules.parse(bad)


def test_until_the_move_the_rules_are_read_from_tools_check(world):
    world.change({"tools/verity/check/queue.toml": 'hold_label = "parked"\n'})
    _git(world.dev, "rm", "-q", "ci/queue.toml")
    _git(world.dev, "commit", "-q", "-m", "rules where they are before the move")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    world.gh.put(1, world.change({"x.txt": "x\n"}), labels=("parked",))
    world.gh.put(2, world.change({"y.txt": "y\n"}), labels=("hold",))
    _, entries = world.queue().sync()
    assert entries[1].why == "held (`parked` label)" and entries[2].state == "queued"


def test_a_ready_label_holds_for_one_head_and_research_queue_ready_writes_it(world, capsys):
    from research.queue import StoreLabels, ready

    world.with_store_remote()
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    labels = StoreLabels(world.store, log=world.logs.append)
    assert labels.ready(1, h1) is False
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    assert f"#1 is ready at {h1[:12]} (pr:1@{h1})" in capsys.readouterr().out
    assert labels.ready(1, h1) is True
    q = Queue(world.tmp / "q" / "verity", remote=str(world.origin), github=world.gh, ready=labels.ready,
              store=world.store, log=world.logs.append)
    members, _ = q.sync()
    assert [m.pr for m in members] == [1]
    h2 = world.change({"a.txt": "a2\n"})
    world.gh.put(1, h2, ready=False)                           # a new push: the label was for the old head
    members, entries = q.sync()
    assert members == [] and entries[1].why == f"not marked ready at {h2[:8]}: `research queue ready 1 --on MACHINE` (or --local; --mark without the tier)"
    world.gh.put(1, h2, labels=("hold",), ready=False)
    assert ready(world.gh, world.store, 1, "lane-1", PASS) == 0
    members, entries = q.sync()
    assert members == [] and entries[1].why == "held (`hold` label)"   # hold wins over ready
    assert ready(world.gh, world.store, 99, "lane-1", PASS) == 2


def test_ready_mark_writes_the_mark_with_no_quick_tier(world, capsys):
    """A ready post with no mark behind it admits nothing; `--mark` needs neither a machine nor a tier."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    h = world.change({"m.txt": "m\n"})
    world.gh.put(3, h, ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly()) == 0
    assert "marked on lane-3's word, with no quick tier" in capsys.readouterr().out
    assert StoreLabels(world.store, log=world.logs.append).ready(3, h) is True


def test_ready_without_sha_marks_the_branchs_tip_not_the_head_github_still_shows(world, monkeypatch):
    """`research queue ready N --mark` seconds after a push mustn't preflight GitHub's old head, which may conflict with `next`
    where the pushed one doesn't. Without --sha, the branch's tip (git's refs, which a push moves at once)
    is the head to mark, and GitHub's PR head is waited for as with --sha."""
    import dataclasses

    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    old, new = world.change({"s.txt": "old\n"}), world.change({"s.txt": "new\n"})
    world.gh.put(5, old, ready=False)
    world.gh.prs[5] = dataclasses.replace(world.gh.prs[5], branch="lane-5/work")
    monkeypatch.setattr(world.gh, "branch_tip", lambda branch: new if branch == "lane-5/work" else None, raising=False)
    reads = []

    def sleep(s):
        reads.append(s)
        world.gh.prs[5] = dataclasses.replace(world.gh.prs[5], head=new)

    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly(), sleep=sleep) == 0
    labels = StoreLabels(world.store, log=world.logs.append)
    assert reads and labels.ready(5, new) is True and labels.ready(5, old) is False


def test_ready_sha_waits_for_github_to_show_the_pushed_head_and_never_marks_another(world, capsys):
    """`git push && research queue ready N --mark` mustn't mark the old head while GitHub's headRefOid lags the push."""
    from research.queue import SHA_READS, MarkOnly, StoreLabels, ready

    world.with_store_remote()
    old, new = world.change({"s.txt": "old\n"}), world.change({"s.txt": "new\n"})
    world.gh.put(5, old, ready=False)
    reads = []

    def sleep(s):
        reads.append(s)
        if len(reads) == 2:
            world.gh.put(5, new, ready=False)

    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly(), sha=new[:12], sleep=sleep) == 0
    labels = StoreLabels(world.store, log=world.logs.append)
    assert len(reads) == 2 and labels.ready(5, new) is True and labels.ready(5, old) is False

    other = world.change({"s.txt": "other\n"})
    capsys.readouterr()
    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly(), sha=other, sleep=reads.append) == 1
    assert len(reads) == 2 + SHA_READS - 1
    err = capsys.readouterr().err
    assert f"GitHub has its head at {new[:12]}, not {other[:12]}" in err and "run this again" in err
    assert StoreLabels(world.store, log=world.logs.append).ready(5, other) is False


def test_ready_refuses_a_head_that_was_ejected_and_says_why_unless_again(world, capsys):
    """A head ejected for a lean-audit failure and marked ready again would fail a second train on the same line, holding up
    the PRs without Lean in it."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    h = world.change({"e.txt": "e\n"})
    world.gh.put(4, h, ready=False)
    world.store.add_label(label_target(4, h), vocab.READY, True, "lane-4", ts="2026-10-16T20:00:00Z")
    world.queue().eject(4, "lean-audit in r1: X.lean:189", "ci")
    capsys.readouterr()
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly()) == 1
    err = capsys.readouterr().err
    assert "that head was ejected: lean-audit in r1: X.lean:189 (ejected by ci at" in err and "Push the fix" in err
    assert StoreLabels(world.store, log=world.logs.append).ready(4, h) is False
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), again=True) == 0
    assert StoreLabels(world.store, log=world.logs.append).ready(4, h) is True


def test_a_head_with_the_tree_of_a_head_marked_ready_is_marked_without_running_the_tier_again(world, monkeypatch, capsys):
    """Merging a `main` that changes none of a PR's files (the repository move's merges did this) gave lanes a new head and a
    35-minute tier over the same files.  Only an identical tree, read with labels pulled from the remote, skips the tier."""
    from research import queue as Q

    class Counting:
        def __init__(self):
            self.repo, self.remote, self.ran = world.dev, str(world.origin), []

        def __call__(self, pr: PR) -> None:
            self.ran.append(pr.head)

    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_WRITE_THROUGH", "1")
    tier = Counting()
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and tier.ran == [h1]
    _move_main(world, {"a.txt": "a1\n", "b.txt": "b\n"})              # `main` makes the PR's own change
    world.gh.put(1, h2 := _merge_main(world, h1), ready=False)
    assert h2 != h1 and _git(world.dev, "rev-parse", f"{h2}^{{tree}}") == _git(world.dev, "rev-parse", f"{h1}^{{tree}}")
    capsys.readouterr()
    elsewhere = LocalStore(world.tmp / "elsewhere", config_path=world.tmp / "store.toml")   # h1's mark only on the remote
    assert Q.ready(world.gh, elsewhere, 1, "lane-1", tier) == 0 and tier.ran == [h1]
    assert f"#1's head {h2[:12]} has the tree of {h1[:12]}" in capsys.readouterr().out
    assert Q.StoreLabels(elsewhere).ready(1, h2) is True

    world.gh.put(1, h3 := world.change({"c.txt": "c3\n"}, base=h2), ready=False)
    assert Q.ready(world.gh, elsewhere, 1, "lane-1", tier) == 0 and tier.ran == [h1, h3]

    _move_main(world, {"a.txt": "a1\n", "b.txt": "b4\n"})
    world.gh.put(1, h4 := _merge_main(world, h3), ready=False)        # a merge of `main` that changes a file runs the tier
    assert Q.ready(world.gh, elsewhere, 1, "lane-1", tier) == 0 and tier.ran == [h1, h3, h4]

    h5 = _git(world.dev, "commit-tree", f"{h4}^{{tree}}", "-p", h4, "-m", "h4's tree again")
    _git(world.dev, "push", "-q", "origin", f"{h5}:refs/dev/{h5}")
    world.gh.put(1, h5, ready=False)
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    capsys.readouterr()
    Q.ready(world.gh, LocalStore(world.tmp / "cut-off", config_path=world.tmp / "store.toml"), 1, "lane-1", tier)
    assert tier.ran == [h1, h3, h4, h5]                               # labels it can't pull say nothing: the tier runs
    assert "STALE" in capsys.readouterr().err


def _invariant(marker: str, up: int) -> str:
    """A stand-in invariant test that fails while the tree, ``up`` directories above it, holds ``marker``."""
    return (f"from pathlib import Path\n\n\ndef test_{marker.lower()}_absent():\n"
            f"    assert not (Path(__file__).resolve().parents[{up}] / '{marker}').exists()\n")


#: what the preflight runs on the merged tree, as stand-ins (each invariant test fails while its marker is in the tree,
#: `pins.py --stale` while STALE.txt is), but `suites.py`, the real one: the suites collected are those its own `touched` picks
PREFLIGHT_FILES = {
    "pyproject.toml": '[tool.uv.workspace]\nmembers = ["pkg"]\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
    "pkg/pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
    "pkg/rows.json": '{"r7": 1}\n',
    "pkg/tests/test_pkg.py": "def test_pkg():\n    pass\n",
    "tests/test_module_strings.py": _invariant("BROKEN_MODULE_STRING", 1),
    "tests/test_pins.py": _invariant("BROKEN_PIN", 1),
    "tests/test_doc_paths.py": _invariant("BROKEN_DOC_PATH", 1),
    "tests/test_repository.py": _invariant("REPORT_GENRE_NAME", 1),
    "tools/research/tests/test_pythonpath.py": _invariant("BROKEN_REAL_ROOTS", 3),
    "tests/test_no_wall_clock.py": _invariant("WALL_CLOCK", 1),
    "integrations/vllm/tests/lint/test_p03_one_evaluator.py": _invariant("VLLM_LINT", 4),
    "tools/verity/check/suites.py": (Path(__file__).resolve().parents[2] / "verity" / "check" / "suites.py").read_text(),
    "tools/verity/check/pins.py": ("import sys\nfrom pathlib import Path\nif Path('STALE.txt').exists():\n"
                                   "    print('pkg/pins.json is stale (pkg/Main.lean): run `python3 regen.py`', file=sys.stderr)\n"
                                   "    sys.exit(1)\n"),
}
#: `uv sync --locked` as the preflight runs it, refusing while LOCK_STALE.txt is in the tree
FAKE_SYNC = (sys.executable, "-c", "import sys; from pathlib import Path; s = Path('LOCK_STALE.txt').exists(); s and print("
             "'error: The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.', file=sys.stderr); sys.exit(2 * s)")


def _preflight(world: World, free=lambda: 1.0, fetch=None, store=None, **kw):
    from research.queue import Preflight

    return Preflight(world.dev, str(world.origin), store or world.store, sync=FAKE_SYNC, python=sys.executable,
                     root=world.tmp / "pre", log=world.logs.append, free=free, fetch=fetch, **kw)


def _preflight_main(world: World, extra: dict[str, str] | None = None) -> str:
    """`main` with what the preflight runs, and with #1 on `next`, refitting away the row `pkg/rows.json` had."""
    world.with_store_remote()
    _move_main(world, {**PREFLIGHT_FILES, **(extra or {})})
    world.gh.put(1, h1 := world.change({"pkg/rows.json": '{"r8": 1}\n'}))
    assert [m.pr for m in world.queue().sync()[0]] == [1]
    return h1


def _no_worktree_left(world: World) -> bool:
    pre = world.tmp / "pre"
    return not (pre.exists() and any(pre.iterdir())) and len(_git(world.dev, "worktree", "list").splitlines()) == 1


def test_the_preflight_marks_a_head_that_merges_onto_next_cleanly_and_a_failure_next_has_too_refuses_nothing(world, capsys):
    """The merged tree is checked in a temporary worktree that is removed, and a check `next` already fails (a doc path broken
    there, a fixture this machine never fetched) isn't the PR's."""
    from research.queue import MarkOnly, StoreLabels, ready

    h1 = _preflight_main(world, {"BROKEN_DOC_PATH": ""})
    world.gh.put(2, h2 := world.change({"pkg/tests/test_more.py": "def test_more():\n    pass\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight" in s)
    assert f"#2 at {h2[:12]} merged onto `next` at" in said and "nothing refused" in said and "collected 2 suite(s)" in said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True
    assert _no_worktree_left(world)
    world.logs.clear()                                         # #1, already on `next`, merges onto what's below its merge there
    assert ready(world.gh, world.store, 1, "lane-1", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert f"merged onto `next` below its merge of #1 at {h1[:8]}" in next(s for s in world.logs if "preflight" in s)


def test_the_preflight_collects_no_suite_when_memory_is_short_and_still_marks(world, capsys):
    """Four torch-importing collections reach about 8 GB of a marker's 16 GB VM. Under a quarter of memory free,
    the preflight collects nothing, says so, and still marks; the train's check collects."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world)
    world.gh.put(2, h2 := world.change({"pkg/tests/test_more.py": "def test_more():\n    pass\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world, free=lambda: 0.1)) == 0
    said = next(s for s in world.logs if "preflight" in s)
    assert "no suite collected: 10% of memory available, under 25%" in said and "collected 0 suite(s)" in said, said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


#: a fetched fixture's bytes, and the tree that reads it: pkg's test reads it at import, as benchmarks/numerical's
#: tests/explore/test_authentication.py reads fixtures/typed-obligation-v0's K=1536 vector (`auth.read_openings()`), so the
#: suite's collection fails without it; `other` registers one too, in a suite no change below touches
BIG = b'{"openings": [1, 2, 3]}\n'
BIG_PATH = "pkg/data/big.json"


def _fixture_tree(world: World) -> tuple[dict[str, str], str]:
    from research.store import fixtures as F

    (src := world.tmp / "src").mkdir()
    (src / "big.json").write_bytes(BIG)
    (src / "other.json").write_bytes(b"{}\n")
    art = F.fixture_id(src / "big.json")
    files = {".gitignore": f"__pycache__/\n{BIG_PATH}\nother/data/other.json\n",
             "fixtures/artifacts.json": json.dumps({art: BIG_PATH, F.fixture_id(src / "other.json"): "other/data/other.json"}),
             "pyproject.toml": PREFLIGHT_FILES["pyproject.toml"].replace('["pkg"]', '["pkg", "other"]'),
             "other/pyproject.toml": PREFLIGHT_FILES["pkg/pyproject.toml"].replace('"pkg"', '"other"'),
             "other/tests/test_other.py": "def test_other():\n    pass\n",
             "pkg/tests/test_big.py": ("import json\nfrom pathlib import Path\n\n"
                                       "BIG = json.loads((Path(__file__).parents[1] / 'data' / 'big.json').read_text())\n\n\n"
                                       "def test_big():\n    assert BIG['openings']\n")}
    return files, art


def _stub_fetch(calls: list, why: str = ""):
    """The preflight's fetch, recording what it was asked; it places the bytes, or (``why``) fetches nothing and says why."""
    def fetch(wt: Path, paths: list[str], deadline: float) -> tuple[set[str], str]:
        calls.append(list(paths))
        if why:
            return set(paths), why
        for p in paths:
            (wt / p).parent.mkdir(parents=True, exist_ok=True)
            (wt / p).write_bytes(BIG)
        return set(), ""
    return fetch


#: a pkg test that fails to collect on the merged tree alone: #1, on `next`, refits away the row it reads
ROWS_TEST = {"pkg/tests/test_rows.py": "import json\nfrom pathlib import Path\n"
                                       "ROW = json.loads((Path(__file__).parents[1] / 'rows.json').read_text())['r7']\n"}


def test_the_preflight_fetches_the_fixtures_of_the_suites_it_collects_and_only_those(world, capsys):
    """A suite whose test reads a fetched fixture at import (as `benchmarks/numerical` reads a vector of
    fixtures/typed-obligation-v0) fails to collect in a worktree that never fetched it.  The preflight
    fetches each collected suite's fetched fixtures into the worktree first, and none of a suite it doesn't collect; a suite
    collected again at `next` has them fetched again there, since the worktree is cleaned in between."""
    from research.queue import MarkOnly, StoreLabels, ready

    files, _ = _fixture_tree(world)
    _preflight_main(world, files)
    calls: list = []
    world.gh.put(2, h2 := world.change({"pkg/tests/test_more.py": "def test_more():\n    pass\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world, fetch=_stub_fetch(calls))) == 0
    said = next(s for s in world.logs if "preflight" in s)
    assert "nothing refused" in said and "collected 2 suite(s)" in said and "not judged" not in said, said
    assert calls == [[BIG_PATH]]
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)

    calls.clear()
    world.gh.put(3, world.change(ROWS_TEST), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=_preflight(world, fetch=_stub_fetch(calls))) == 1
    err = capsys.readouterr().err
    found = [ln for ln in err.splitlines() if ln.startswith("  - ")]
    assert found == ["  - `pytest --collect-only` in pkg fails on the merged tree: ERROR tests/test_rows.py - KeyError: 'r7': a "
                     "test reads what another queued PR changes or what only an unlanded PR adds; make it read what `main` and "
                     "`next` hold, or wait for that PR"], err
    assert calls == [[BIG_PATH], [BIG_PATH]] and _no_worktree_left(world)


def test_a_fixture_the_preflight_cannot_fetch_is_said_once_and_leaves_its_suite_unjudged(world, capsys):
    """With no credentials (or no source holding it), the suite that reads it isn't collected, so it can't refuse on bytes it
    never had; one note names the suite and the fixtures, and the head is marked: the after-landing run fetches them."""
    from research.queue import MarkOnly, StoreLabels, ready

    files, _ = _fixture_tree(world)
    _preflight_main(world, files)
    calls: list = []
    world.gh.put(2, h2 := world.change(ROWS_TEST), ready=False)
    pre = _preflight(world, fetch=_stub_fetch(calls, why="no credentials for the store's remote"))
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=pre) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight" in s)
    note = (f"pkg not judged: 1 of the 1 fixture(s) the suites read weren't fetched (no credentials for the store's remote: "
            f"{BIG_PATH}); the after-landing run fetches them")
    assert said.count(note) == 1 and "collected 1 suite(s) (.)" in said and "not judged" not in said.replace(note, ""), said
    assert calls == [[BIG_PATH]]
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflights_fetch_keeps_what_it_fetched_in_this_machines_store_for_the_next_preflight(world, monkeypatch, capsys):
    """The default fetch is `research data fetch-fixtures` through the caller's store: the first preflight takes the fixture
    from the store's remote, and the next one on this machine from the store alone, with the remote gone; a store that never
    fetched it then can't, and its preflight says so and still marks."""
    from research.queue import MarkOnly, ready
    from research.store import fixtures as F
    from research.store.local import manifest_key, object_key
    from research.store.remote import FsRemote

    files, art = _fixture_tree(world)
    _preflight_main(world, files)
    seed = LocalStore(world.tmp / "seed")
    assert seed.put_artifact(F.FIXTURE_KIND, {}, None, file=world.tmp / "src" / "big.json") == art
    vault = FsRemote(world.tmp / "vault")
    [(blob, _)] = seed.get_manifest(art).blobs()
    vault.put(object_key(blob), seed.object_path(blob))
    vault.put(manifest_key(art.split(":", 1)[1]), seed.manifest_path(art.split(":", 1)[1]))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(world.tmp / "store.toml"))
    monkeypatch.setenv("RESEARCH_PUBLIC_URL", "")

    def recorded(store=None):
        pre = _preflight(world, store=store)
        real = pre.fetch
        pre.fetch = lambda wt, paths, deadline: got.append((paths, real(wt, paths, deadline))) or got[-1][1]
        return pre

    got: list = []
    world.gh.put(2, world.change({"pkg/tests/test_more.py": "def test_more():\n    pass\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=recorded()) == 0, capsys.readouterr().err
    assert got == [([BIG_PATH], (set(), ""))] and world.store.object_path(blob).is_file()
    said = next(s for s in world.logs if "preflight: #2" in s)
    assert "collected 2 suite(s)" in said and "not judged" not in said, said

    shutil.rmtree(world.tmp / "vault")
    world.gh.put(3, world.change({"pkg/tests/test_most.py": "def test_most():\n    pass\n"}), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=recorded()) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight: #3" in s)
    assert got[-1] == ([BIG_PATH], (set(), "")) and "collected 2 suite(s)" in said and "not judged" not in said, said

    elsewhere = LocalStore(world.tmp / "elsewhere", config_path=world.tmp / "store.toml")
    world.gh.put(4, world.change({"pkg/tests/test_else.py": "def test_else():\n    pass\n"}), ready=False)
    assert ready(world.gh, elsewhere, 4, "lane-4", MarkOnly(), preflight=recorded(elsewhere)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight: #4" in s)
    assert got[-1][1][0] == {BIG_PATH} and "pkg not judged: 1 of the 1 fixture(s)" in said, said


def test_the_preflight_refuses_a_head_that_conflicts_with_next_naming_the_paths_and_the_pr_that_touched_them(world, capsys):
    """A conflict with a PR already on `next` is found at mark time, not only by drive's sync, which would take the PR off
    `next`."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world)
    world.gh.put(2, h2 := world.change({"pkg/rows.json": '{"r9": 1}\n'}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 1
    err = capsys.readouterr().err
    assert "it conflicts with `next` at" in err and "in pkg/rows.json (touched by #1)" in err and "never merge `next`" in err
    assert "`--no-preflight` skips the preflight" in err
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=None) == 0         # --no-preflight: the lander's


#: what the head writes -> what the refusal says: each a way a train breaks that the merged tree shows at mark time
PREFLIGHT_REFUSALS = {
    "stale pin": ({"STALE.txt": "x\n"}, "`pins.py --stale` on the merged tree: pkg/pins.json is stale (pkg/Main.lean)"),
    "module string": ({"BROKEN_MODULE_STRING": ""}, "the invariant tests in . fail on the merged tree: FAILED "
                      "tests/test_module_strings.py::test_broken_module_string_absent"),
    "pythonpath": ({"BROKEN_REAL_ROOTS": ""}, "the invariant tests in tools/research fail on the merged tree: FAILED "
                   "tests/test_pythonpath.py::test_broken_real_roots_absent"),
    # a skill over the Markdown cap or a report-genre name fails a train's preflight-lints
    "repository lint": ({"REPORT_GENRE_NAME": ""}, "the invariant tests in . fail on the merged tree: FAILED "
                        "tests/test_repository.py::test_report_genre_name_absent"),
    # a test reads a vector row that a PR queued before it refits away: a KeyError at import on the merged tree only
    "collection error": ({"pkg/tests/test_rows.py": "import json\nfrom pathlib import Path\n"
                          "ROW = json.loads((Path(__file__).parents[1] / 'rows.json').read_text())['r7']\n"},
                         "`pytest --collect-only` in pkg fails on the merged tree: ERROR tests/test_rows.py - KeyError: 'r7'"),
    "stale lock": ({"LOCK_STALE.txt": ""}, "`uv sync --locked` on the merged tree: error: The lockfile at `uv.lock` needs to be "
                   "updated"),
    # time.perf_counter in a test fixture, or numpy's rng in vllm (P3), costs a train its check
    "wall-clock lint": ({"pkg/tests/test_slow.py": "def test_slow():\n    pass\n", "WALL_CLOCK": ""},
                        "the invariant tests in tests/test_no_wall_clock.py fail on the merged tree: FAILED "
                        "tests/test_no_wall_clock.py::test_wall_clock_absent"),
    "vllm lint": ({"integrations/vllm/verity_vllm/kernels.py": "", "VLLM_LINT": ""},
                  "the invariant tests in integrations/vllm/tests/lint fail on the merged tree: FAILED "
                  "tests/lint/test_p03_one_evaluator.py::test_vllm_lint_absent"),
}


@pytest.mark.parametrize("case", list(PREFLIGHT_REFUSALS))
def test_the_preflight_refuses_a_head_whose_merge_onto_next_breaks_an_invariant_and_names_the_fix(world, capsys, case):
    from research.queue import MarkOnly, StoreLabels, ready

    files, says = PREFLIGHT_REFUSALS[case]
    _preflight_main(world)
    world.gh.put(2, h2 := world.change(files), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 1
    err = capsys.readouterr().err
    assert f"#2 is not marked ready at {h2[:12]}: the preflight of its head merged onto `next` at" in err
    assert says in err, err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [next(ln for ln in err.splitlines() if says in ln)]
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)


def test_the_preflight_refuses_a_head_a_landing_gate_at_main_fails_with_its_fix_and_notes_one_that_cant_judge(world, capsys):
    """The sync leaves out what a landing gate refuses; the preflight says so at mark time, and a gate that can't judge there is
    a note, since the sync judges again."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world, GATED)
    main = _git(world.origin, "rev-parse", "main")
    world.gh.put(2, h2 := world.change({"bad.txt": "2\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [f"  - {_gate_says(main, 'touches bad.txt')}"], err
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)
    world.gh.put(3, h3 := world.change({"crash.txt": "3\n"}), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if f"preflight: #3 at {h3[:12]}" in s)
    assert "nothing refused" in said and f"the landing gate gates/g.py can't judge {h3[:12]}" in said and "exit 2" in said


def test_the_preflight_runs_a_lint_only_on_a_change_that_reaches_what_it_judges(world, capsys):
    """`PREFLIGHT_LINTS` add up to a minute, so a change outside their prefixes doesn't run them: a head breaking the vllm lint's
    stand-in without touching integrations/vllm is marked, and the train's check runs the lint."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world)
    world.gh.put(2, h2 := world.change({"pkg/notes.txt": "x\n", "VLLM_LINT": ""}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflight_refuses_what_the_sync_holds_a_marked_pr_waiting_on_one_it_contains(world, capsys):
    """A re-mark the sync would hold is refused, rather than taken and left for the next sync to say "contains #N, which isn't
    admitted".  Here #5 is marked but contains lane-4's unmarked #6, which #5 doesn't carry, so the sync
    holds #5, and #4 behind it, though #4 carries #6 itself: the mark of #4 is refused as the sync would hold it."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    world.gh.put(6, six := world.change({"x.txt": "x\n"}), ready=False, body="Lane `lane-4`.")
    world.gh.put(5, five := world.change({"y.txt": "y\n"}, base=six), ready=False)
    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly(), preflight=None) == 0
    world.gh.put(4, four := world.change({"z.txt": "z\n"}, base=five), ready=False)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=_preflight(world)) == 1
    err = capsys.readouterr().err
    assert (f"it contains #5's head {five[:12]}, and #5 isn't admitted (contains #6, which isn't admitted), so drive holds "
            f"this PR behind it") in err and "#6's head" not in err
    assert StoreLabels(world.store, log=world.logs.append).ready(4, four) is False
    world.gh.ready |= {(5, five), (4, four)}                                                  # as the sync judges it
    _, entries = world.queue().sync()
    assert entries[4].why == "contains #5, which isn't admitted" and entries[5].why == "contains #6, which isn't admitted"


def test_the_preflight_names_a_contained_prs_ejection(world, capsys):
    from research.queue import MarkOnly, ready
    from research.store.cli import _label_write_through

    world.with_store_remote()
    world.gh.put(5, five := world.change({"x.txt": "x\n"}), ready=False)
    world.store.add_label(label_target(5, five), vocab.READY, True, "lane-5", ts="2026-10-26T10:00:00Z")
    world.store.add_label(label_target(5, five), vocab.READY, False, "drive", ref="the train's check failed x.txt",
                          ts="2026-10-26T11:00:00Z")
    _label_write_through(world.store, label_target(5, five))
    world.gh.put(4, world.change({"y.txt": "y\n"}, base=five), ready=False)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=_preflight(world)) == 1
    assert (f"and #5 isn't admitted (ejected at {five[:8]} by drive: the train's check failed x.txt)"
            in capsys.readouterr().err)


def test_the_preflight_runs_on_a_head_its_pull_ref_doesnt_have_yet_from_the_branch(world, capsys):
    """A mark seconds after a push reads the branch's tip while GitHub's pull ref may still have the old head; the preflight
    still runs, fetching the head from the branch."""
    import dataclasses

    from research.queue import MarkOnly, Preflight, ready

    world.with_store_remote()
    world.gh.put(5, five := world.change({"x.txt": "x\n"}), ready=False)
    world.gh.put(4, old := world.change({"y.txt": "y\n"}, base=five), ready=False)
    new = world.change({"y.txt": "y2\n"}, base=old)
    _git(world.dev, "push", "-q", "origin", f"{new}:refs/heads/lane-4/change")
    world.gh.prs[4] = dataclasses.replace(world.gh.prs[4], head=new, branch="lane-4/change")    # refs/pull/4/head is still `old`
    _git(world.tmp, "clone", "-q", str(world.origin), str(marker := world.tmp / "marker"))     # a checkout without `new`
    pre = Preflight(marker, str(world.origin), world.store, sync=FAKE_SYNC, python=sys.executable, root=world.tmp / "pre",
                    log=world.logs.append)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    assert f"it contains #5's head {five[:12]}, and #5 isn't admitted" in capsys.readouterr().err


def test_the_preflight_refuses_a_head_containing_an_open_pr_that_isnt_admitted_and_names_it(world, capsys):
    """Drive holds a PR that contains one never marked, or one whose mark was withdrawn; the preflight refuses it first."""
    from research.queue import MarkOnly, ready

    world.with_store_remote()
    world.gh.put(5, parent := world.change({"x.txt": "x\n"}), ready=False)
    world.gh.put(4, world.change({"y.txt": "y\n"}, base=parent), ready=False)
    pre = _preflight(world)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert (f"it contains #5's head {parent[:12]}, and #5 isn't admitted (not marked ready at {parent[:8]}), so drive holds "
            f"this PR behind it: mark #5 ready first, so it lands first, or drop #5's commits from this branch (if this PR "
            f"carries #5's change, #5's owner withdraws its mark: `research data label pr:5@{parent[:12]} ready false "
            f"--by OWNER`)") in err
    assert "merged onto `main` at" in err                                                    # no `next` yet
    world.gh.put(5, parent, labels=("hold",), ready=False)
    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly(), preflight=pre) == 0
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    assert "and #5 isn't admitted (held (`hold` label))" in capsys.readouterr().err
    world.gh.put(5, parent, ready=False)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 0         # #5 admitted: drive lands it first
    world.gh.put(6, six := world.change({"z.txt": "z\n"}), ready=False, body="Lane `lane-4`.")
    world.gh.put(4, world.change({"w.txt": "w\n"}, base=six), ready=False)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 0         # #6 is lane-4's own: #4 carries it


#: Lake packages as the Lean step reads them: `lean/a` (library `A`), `lean/a/Proofs` nested in it with `srcDir = ".."` and its
#: library's roots listed (as Security/Proofs'), `lean/b` requiring `lean/a` by path, and `lean/c`, which doesn't
LEAN_TREE = {
    "lean/a/lakefile.toml": 'name = "a"\n\n[[lean_lib]]\nname = "A"\n',
    "lean/a/lean-audit.json": "{}\n",
    "lean/a/A/Base.lean": "def foo := 1\n",
    "lean/a/A/Mid.lean": "/- Copyright\n  /- nested -/ still a comment -/\n-- a line\nimport A.Base\n\ndef mid := 1\nimport A.Alone\n",
    "lean/a/A/Top.lean": "module\n\npublic import A.Mid\nimport Mathlib.Logic.Basic\n",
    "lean/a/A/Alone.lean": "import Mathlib\n",
    "lean/a/Scratch.lean": "import A.Top\n",
    "lean/a/Proofs/lakefile.toml": 'name = "p"\nsrcDir = ".."\n\n[[require]]\nname = "a"\npath = ".."\n\n'
                                   '[[lean_lib]]\nname = "Proofs"\nroots = ["Proofs.Core"]\n',
    "lean/a/Proofs/lean-audit.json": "{}\n",
    "lean/a/Proofs/Core/X.lean": "import A.Mid\n",
    "lean/a/Proofs/Misc.lean": "import A.Base\n",
    "lean/b/lakefile.toml": 'name = "b"\n\n[[require]]\nname = "a"\npath = "../a"\n\n[[lean_lib]]\nname = "B"\n',
    "lean/b/lake-manifest.json": json.dumps({"packages": [{"type": "path", "name": "a", "dir": "../a"}]}),
    "lean/b/lean-audit.json": "{}\n",
    "lean/b/B/Use.lean": "import A.Top\n",
    "lean/b/B/Other.lean": "import B.Use\n",
    "lean/b/B/Old.lean": "import A.Gone\n",
    "lean/c/lakefile.toml": 'name = "c"\n\n[[lean_lib]]\nname = "C"\n',
    "lean/c/lean-audit.json": "{}\n",
    "lean/c/C/Base.lean": "import A.Base\n",
}


def test_the_lean_step_builds_the_changed_modules_and_their_importers_transitively_in_the_package_and_its_path_requirers(tmp_path):
    """A signature change can break a module two imports away: the importers are followed to the end, into the
    packages requiring the changed one by path (a nested one too), and only modules a library's root holds are built."""
    from research.queue import lean_graph, lean_imports

    for rel, text in LEAN_TREE.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    files = sorted(LEAN_TREE)
    assert lean_imports(LEAN_TREE["lean/a/A/Mid.lean"]) == ["A.Base"], "comments skipped, the header ends at the first command"
    assert lean_imports(LEAN_TREE["lean/a/A/Top.lean"]) == ["A.Mid", "Mathlib.Logic.Basic"]
    graph = lean_graph(tmp_path, files, ["lean/a/A/Base.lean", "README.md"])
    assert graph.targets == {"lean/a/A/Base.lean": "A.Base", "lean/a/A/Mid.lean": "A.Mid", "lean/a/A/Top.lean": "A.Top",
                             "lean/a/Proofs/Core/X.lean": "Proofs.Core.X", "lean/b/B/Use.lean": "B.Use",
                             "lean/b/B/Other.lean": "B.Other"}
    assert graph.upstream("lean/b/B/Other.lean") == ["lean/a/A/Base.lean", "lean/a/A/Mid.lean", "lean/a/A/Top.lean",
                                                     "lean/b/B/Use.lean"]
    assert lean_graph(tmp_path, files, ["lean/b/B/Use.lean"]).targets == {"lean/b/B/Use.lean": "B.Use",
                                                                          "lean/b/B/Other.lean": "B.Other"}
    assert lean_graph(tmp_path, files, ["lean/a/A/Gone.lean"]).targets == {"lean/b/B/Old.lean": "B.Old"}, "a deleted module's"
    assert lean_graph(tmp_path, files, ["lean/a/README.md", "a.txt"]).targets == {}


#: `lean_changed.py --build`'s stand-in for Lake: a file whose `-- needs NAME` line names no `def NAME` of the tree's Lean files
#: fails, as Lean's `unknown identifier` would; each call's files are appended to $FAKE_LEAN_SEEN, and $FAKE_LEAN_SKIP skips
FAKE_LEAN_BUILDER = """\
import json, os, re, subprocess, sys
from pathlib import Path
files = sys.argv[sys.argv.index("--build") + 1:]
with open(os.environ["FAKE_LEAN_SEEN"], "a") as f:
    f.write(json.dumps(files) + "\\n")
if os.environ.get("FAKE_LEAN_SKIP"):
    print("lean-build: " + json.dumps({"built": {}, "failed": {}, "skipped": {"lean/a": os.environ["FAKE_LEAN_SKIP"]}}))
    sys.exit(2)
lean = subprocess.run(["git", "ls-files", "*.lean"], capture_output=True, text=True).stdout.split()
defined = {n for g in lean for n in re.findall(r"^def (\\w+)", Path(g).read_text(), re.M)}
failed = {g: f"error: {g}:2:0: unknown identifier '{n}'" for g in files
          for n in re.findall(r"^-- needs (\\w+)", Path(g).read_text(), re.M) if n not in defined}
built = {"/".join(g.split("/")[:2]): int(any(h.startswith("/".join(g.split("/")[:2])) for h in failed)) for g in files}
print("lean-build: " + json.dumps({"built": built, "failed": failed, "skipped": {}}))
sys.exit(1 if failed else 0)
"""


def _lean_world(world: World, monkeypatch) -> tuple[Path, Path]:
    """`main` with what the preflight runs, `LEAN_TREE` and the stand-in builder, and #1 on `next`, whose B.Use reads `foo`; the
    seen file, and a Lean slot file listing a `build` pool (a node's)."""
    world.with_store_remote()
    _move_main(world, {**PREFLIGHT_FILES, **LEAN_TREE, "tools/verity/lean/lean_changed.py": FAKE_LEAN_BUILDER})
    world.gh.put(1, world.change({"lean/b/B/Use.lean": "import A.Top\n-- needs foo\n"}))
    assert [m.pr for m in world.queue().sync()[0]] == [1]
    monkeypatch.setenv("FAKE_LEAN_SEEN", str(seen := world.tmp / "lean-seen.jsonl"))
    monkeypatch.delenv("FAKE_LEAN_SKIP", raising=False)
    (slots := world.tmp / "lean-slots").write_text("build 2\n")
    return seen, slots


def test_the_preflight_refuses_a_head_that_breaks_a_lean_module_a_pr_on_next_owns_naming_it_its_error_and_that_pr(
        world, monkeypatch, capsys):
    """A PR that passes its own Lean check can break a module of another PR on `next`, which drive's Lean check of the stack
    finds only an hour later. A module failing on `next` too isn't the head's; one of the head's own that
    fails only on the merge names the PR on `next` that changed what it imports."""
    from research.queue import LakeBuild, MarkOnly, StoreLabels, ready

    seen, slots = _lean_world(world, monkeypatch)
    world.gh.put(2, h2 := world.change({"lean/a/A/Base.lean": "def bar := 1\n"}), ready=False)
    pre = _preflight(world, lean=LakeBuild(slots=slots))
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [
        "  - Lean: `B.Use` (lean/b/B/Use.lean), of #1 on `next`, fails to build with this head merged: error: lean/b/B/Use.lean:2:0: "
        "unknown identifier 'foo': merge or stack on that PR, never merge `next` (it carries unlanded PRs)"], err
    calls = [json.loads(ln) for ln in seen.read_text().splitlines()]
    assert calls == [["lean/a/A/Base.lean", "lean/a/A/Mid.lean", "lean/a/A/Top.lean", "lean/a/Proofs/Core/X.lean",
                      "lean/b/B/Other.lean", "lean/b/B/Use.lean"], ["lean/b/B/Use.lean"]], "the merge, then what failed, on `next`"
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)

    world.gh.put(2, h2, ready=True)                                    # #2 on `next` anyway (marked without the preflight)
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2]
    seen.unlink()
    world.gh.put(3, h3 := world.change({"lean/a/A/Top.lean": LEAN_TREE["lean/a/A/Top.lean"] + "-- tidied\n",
                                        "lean/b/B/Use3.lean": "import A.Top\n-- needs foo\n"}), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [
        "  - Lean: `B.Use3` (lean/b/B/Use3.lean), this head's own, fails to build with this head merged: error: "
        "lean/b/B/Use3.lean:2:0: unknown identifier 'foo': it imports what #2 on `next` changed: merge or stack on that PR, never "
        "merge `next` (it carries unlanded PRs)"], err
    calls = [json.loads(ln) for ln in seen.read_text().splitlines()]
    assert calls[1] == ["lean/b/B/Use.lean"] and "lean/b/B/Use.lean" in calls[0], "B.Use fails on `next` too: not #3's"
    assert StoreLabels(world.store, log=world.logs.append).ready(3, h3) is False and _no_worktree_left(world)


def test_the_lean_step_notes_why_it_built_nothing_and_refuses_nothing(world, monkeypatch, capsys):
    """An agent VM keeps no warm Lean tree and builds nothing; a change more modules import than the preflight builds, a node
    without Mathlib for the package, and a spent budget each leave the break to drive's Lean check: noted, never refused."""
    from research.queue import LakeBuild, MarkOnly, ready

    seen, slots = _lean_world(world, monkeypatch)
    cases = [
        (LakeBuild(slots=world.tmp / "absent"), {}, "this machine keeps no warm Lean tree", False),
        (LakeBuild(slots=slots), {"lean_modules": 5}, "6 modules are or import what it changes, over the 5", False),
        (LakeBuild(slots=slots), {"lean_s": 0}, "budget was spent before it started", False),
        (LakeBuild(slots=slots), {}, "lean/a not built: no Mathlib here", True),
    ]
    for n, (lean, kw, says, built) in enumerate(cases, start=2):
        if built:
            monkeypatch.setenv("FAKE_LEAN_SKIP", "no Mathlib here")
        seen.unlink(missing_ok=True)
        world.gh.put(n, world.change({"lean/a/A/Base.lean": "def bar := 1\n", f"n{n}.txt": "x\n"}), ready=False)
        world.logs.clear()
        assert ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), preflight=_preflight(world, lean=lean, **kw)) == 0, \
            capsys.readouterr().err
        said = next(s for s in world.logs if f"preflight: #{n} at" in s)
        assert "nothing refused" in said and says in said, said
        assert seen.exists() is built


def test_the_lean_step_ships_the_merged_tree_to_a_node_for_ready_on_and_reads_its_result(tmp_path):
    """Off a node, `ready --on` builds as `verity lean try --on vy-nebius-cpu-3` does: a `research run` of the tree on
    `PREFLIGHT_LEAN_ON`, whose `build` Lean slot is the gate (a pool only when given), bounded by the step's budget, whose stdout
    carries the result line."""
    from research.queue import LEAN_BUILDER, PREFLIGHT_LEAN_ON, LakeBuild

    assert (PREFLIGHT_LEAN_ON, LakeBuild().pool) == ("vy-nebius-cpu-3", None), "infra's placement, 26 Oct 14:44Z"
    (tmp_path / "wt" / LEAN_BUILDER).parent.mkdir(parents=True)
    (tmp_path / "wt" / LEAN_BUILDER).write_text("")
    rid = "r20261026-120000-abcd"
    line = "lean-build: " + json.dumps({"built": {"lean/b": 1}, "failed": {"lean/b/B/Use.lean": "error: B/Use.lean:2:0: x"},
                                        "skipped": {}})
    tail = ["--as", "ci", "--project", "verity", "--source", str(tmp_path / "wt"), "--cwd", "clone", "--timeout", "90", "--wait",
            "--wait-every", "5", "--wait-max", "90", "--", "python3", LEAN_BUILDER, "--build", "lean/b/B/Use.lean"]
    for pool, flags in ((None, []), ("research", ["--pool", "research", "--pool-wait", "90"])):
        argvs = []

        def run(argv, cwd, left, argvs=argvs):
            argvs.append((argv[3:], left))
            if argv[3] == "fetch":
                (tmp_path / "runs" / rid).mkdir(parents=True, exist_ok=True)
                (tmp_path / "runs" / rid / "stdout.log").write_text(f"lean-build: lean/b ...\n{line}\n")
            return 1, f"research: run {rid} launched on {PREFLIGHT_LEAN_ON} (pod) pid=1 -> /r\n"

        got = LakeBuild(on=PREFLIGHT_LEAN_ON, pool=pool, by="ci", slots=tmp_path / "absent", run=run,
                        runs=lambda: tmp_path / "runs")(tmp_path / "wt", ["lean/b/B/Use.lean"], 100)
        assert got.failed == {"lean/b/B/Use.lean": "error: B/Use.lean:2:0: x"} and got.notes == []
        (launch, left), (fetch, _) = argvs
        assert left == 100 and fetch == ["fetch", rid, "--on", PREFLIGHT_LEAN_ON, "--all"]
        assert launch == ["run", "--on", PREFLIGHT_LEAN_ON, *flags, *tail]


def test_research_queue_ready_runs_the_preflight_unless_no_preflight(world, monkeypatch, capsys):
    from research import queue as Q

    seen, gates = [], []
    monkeypatch.setattr(Q, "ready", lambda *a, preflight=None, records=None, **kw: seen.append(preflight) or gates.append(records) or 0)
    monkeypatch.chdir(world.dev)
    base = ["queue", "ready", "3", "--mark", "--by", "lane-3", "--remote", str(world.origin), "--repo", "o/r"]
    assert research_main(base) == 0 and research_main([*base, "--no-preflight"]) == 0
    assert isinstance(seen[0], Q.Preflight) and seen[1] is None
    assert isinstance(seen[0].lean, Q.LakeBuild) and seen[0].lean.on is None, "only `ready --on` ships the Lean step to a node"
    assert all(isinstance(g, Q.RecordsGate) for g in gates) and len(gates) == 2        # the records gate isn't the preflight's
    with pytest.raises(SystemExit) as e:
        research_main(["queue", "status", "--no-preflight"])
    assert e.value.code == 2 and "--no-preflight is for ready only" in capsys.readouterr().err


LOCK = "pkg/lean-audit.json"


def _lock(**entries: str) -> str:
    """A lock with one entry a line, so git merges two sides' changes to entries apart."""
    return json.dumps({k: entries.get(k, "1") for k in "ABCDE"}, indent=1) + "\n"


def _published(world: World, commit: str, *, pkg: str = "pkg", passed: bool = True, path: str = "slot",
               files: dict[str, str] | None = None, steps: list | None = None, at: float | None = None) -> str:
    """A run of ``commit`` that another machine published to the store's remote, as `research run` does: a records run
    (`lean_changed.py --records PKG`, result `lean-fast/v1`) with ``files`` as its run files, or with ``steps`` a check; ``at``
    its start (its id's time), else now."""
    from research.runs import new_run_id

    node = LocalStore(world.tmp / "node", config_path=world.tmp / "store.toml")
    rid, outputs = new_run_id(at), {}
    if files is not None:
        tree = world.tmp / "run-files" / rid
        for rel, text in files.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(text)
        outputs["run_files"] = node.put_artifact("run-files/v1", {}, tree=tree)
    kind = "check/v1" if steps is not None else "lean-fast/v1"
    meta = {"schema": "research/result/v0.1", "kind": kind, "run_id": rid, "validation": {"status": "passed" if passed else "failed"},
            **({"steps": steps} if steps is not None else {"workload_fingerprint": {"package": pkg, "update": True, "path": path}})}
    outputs["result"] = node.put_artifact(kind, meta)
    node.put_attempt({
        "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "python3", "version": None}, "derivation": "drv:" + "0" * 64,
        "reuse_key": None, "argv": ["python3", "tools/verity/lean/lean_changed.py", "--records", pkg, "--update"], "params": {},
        "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [], "source": {"commit": commit, "dirty": False},
        "closure": None, "execution": {}, "cost": None, "status": {"state": "done" if passed else "failed", "rc": 0 if passed else 1},
        "outputs": outputs, "telemetry": None})
    for art in outputs.values():
        node.push(art)
    node.push_attempt(rid)
    return rid


def _audit(pkg: str = "pkg", failure: str | None = None) -> str:
    """check.py's `audit.json` of a records run on its slot's tree, ``pkg`` failing with ``failure`` when given."""
    at = f"/w/lean-records/build-0/{pkg}"
    return json.dumps({"passed": failure is None, "failed": [at] if failure else [],
                       "packages": [{"package": at, "failures": [failure] if failure else []}]})


def _gate(world: World, store=None):
    from research.queue import RecordsGate

    return RecordsGate(world.dev, str(world.origin), store or world.store, log=world.logs.append,
                       lock_merge=world.tmp / "lock_merge.py")


def _mark(world: World, n: int, head: str, store=None) -> int:
    from research.queue import MarkOnly, ready

    world.gh.put(n, head, ready=False)
    return ready(world.gh, store or world.store, n, f"lane-{n}", MarkOnly(), records=_gate(world, store))


def test_a_head_that_changes_a_lock_is_marked_only_on_a_records_run_of_it_that_passed_and_a_refusal_names_the_run(world, capsys):
    """A records run that printed `AUDIT: FAIL` still writes a record, which reads as a pass, and a head marked on it fails a
    train's Lean check about 50 minutes later.  With no run the refusal names the command to start; with a failing one, that run and the first line of the package's failure."""
    from research.queue import RECORDS_COMMAND, StoreLabels

    world.with_store_remote()
    h = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    capsys.readouterr()
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert f"#1 is not marked ready at {h[:12]}: it changes the Lean lock of pkg against `main`" in err
    assert f"start `{RECORDS_COMMAND.format(pkg='pkg')}`" in err
    assert ("research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd clone --declared-output "
            "'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records pkg --update") in err
    failed = _published(world, h, passed=False, files={"lean-audit/audit.json": _audit(
        failure="guarantees.readers: T's readers in the record are stale\n  more detail")})
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert f"records run {failed} failed pkg: guarantees.readers: T's readers in the record are stale" in err
    assert "more detail" not in err and RECORDS_COMMAND.format(pkg="pkg") in err
    assert StoreLabels(world.store, log=world.logs.append).ready(1, h) is False

    passed = _published(world, h, files={"lean-audit/audit.json": _audit()})
    assert _mark(world, 1, h) == 0
    assert f"records gate: pkg: records run {passed} of {h[:12]} passed it" in world.logs[-1]
    assert StoreLabels(world.store, log=world.logs.append).ready(1, h) is True

    plain = world.change({"a.txt": "no lock\n"})                     # no lock changed: nothing is read, not even the store
    assert _mark(world, 2, plain, store=LocalStore(world.tmp / "bare", config_path=world.tmp / "none.toml")) == 0


def test_a_package_passes_as_a_requirer_on_a_reused_path_or_in_a_lean_check_and_only_its_own_verdict_counts(world, capsys):
    """A records run of PKG audits each package whose guarantees read PKG's build (`requirer-<path>/`): its pass there counts for
    that package, and a run that failed only PKG still passed it.  A run that took `path: reused` passed on the same inputs; a check
    whose `lean-audit` step passed without regenerating the lock passed it too."""
    world.with_store_remote()
    two = world.change({LOCK: _lock(A="2"), "lib/lean-audit.json": _lock(B="2")})
    _published(world, two, pkg="lib", passed=False, files={"lean-audit/audit.json": _audit("lib", failure="lean: X.lean:3: error"),
                                                             "lean-audit/requirer-pkg/audit.json": _audit("pkg")})
    capsys.readouterr()
    assert _mark(world, 1, two) == 1
    err = capsys.readouterr().err
    assert "it changes the Lean lock of lib against" in err and "failed lib: lean: X.lean:3: error" in err
    _published(world, two, pkg="lib", path="reused")
    assert _mark(world, 1, two) == 0

    h = world.change({LOCK: _lock(C="2")})
    _published(world, h, steps=[{"step": "lean-audit", "status": "passed"}],
               files={"lean-audit/audit.json": json.dumps({"passed": True, "regenerated": {"pkg": "lean-audit/x/lean-audit.json"}}),
                      "lean-audit/x/lean-audit.json": "{}"})
    assert _mark(world, 2, h) == 1                                     # it passed by regenerating this lock: not the head's
    rid = _published(world, h, steps=[{"step": "lean-audit", "status": "passed"}],
                     files={"lean-audit/audit.json": json.dumps({"passed": True, "regenerated": {}})})
    assert _mark(world, 2, h) == 0 and f"the Lean check {rid} of {h[:12]} passed it" in world.logs[-1]


def test_a_package_audited_beside_pkg_is_read_from_its_own_report(world, capsys):
    """A records run of `Security` audits `Security/Proofs` too (its `proved_in`), with that package's verdict in
    `audit-Proofs/report.json`: a head changing both locks is vouched for by one run when the report passed, and refused with its
    first failure line when it didn't."""
    world.with_store_remote()
    both = {LOCK: _lock(A="2"), "pkg/Proofs/lean-audit.json": _lock(B="2")}
    def report(failure=None):
        return json.dumps({"package": "/w/lean-records/build-0/pkg/Proofs", "failures": [failure] if failure else []})

    h = world.change(both)
    _published(world, h, passed=False, files={"lean-audit/audit.json": _audit(),
                                              "lean-audit/audit-Proofs/report.json": report("replay: P.t: stale\n  more")})
    capsys.readouterr()
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert "the Lean lock of pkg/Proofs against" in err and "failed pkg/Proofs: replay: P.t: stale" in err
    assert "--records pkg/Proofs --update" in err and "more" not in err

    h2 = world.change({"pkg/Proofs/P.lean": "theorem p : True := trivial\n"}, base=h)
    rid = _published(world, h2, files={"lean-audit/audit.json": _audit(), "lean-audit/audit-Proofs/report.json": report()})
    assert _mark(world, 2, h2) == 0
    assert any(f"records gate: pkg/Proofs: records run {rid} of {h2[:12]} passed it" in line for line in world.logs)


def test_a_records_run_whose_update_wrote_a_record_vouches_only_for_the_commit_of_that_record(world, capsys):
    """`--update` passes by writing the record, kept as `record/lean-audit.json` when it differs from the commit's lock: the head
    it ran on still has the old lock, which a train's Lean check fails.  The commit of exactly that record over it is vouched for
    by the run below it, so a lane needs no second run; any other lock there is refused."""
    world.with_store_remote()
    h1 = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    rec = _lock(A="2", B="2")
    rid = _published(world, h1, files={"lean-audit/audit.json": _audit(), "lean-audit/record/lean-audit.json": rec})
    capsys.readouterr()
    assert _mark(world, 1, h1) == 1
    err = capsys.readouterr().err
    assert f"records run {rid} of {h1[:12]} passed pkg only with the record its `--update` wrote" in err
    assert "commit its `record/lean-audit.json`" in err
    assert _mark(world, 1, world.change({LOCK: _lock(A="2", B="3")}, base=h1)) == 1
    h2 = world.change({LOCK: rec}, base=h1)
    assert _mark(world, 1, h2) == 0
    assert f"{h2[:12]} commits the record records run {rid} of {h1[:12]} regenerated, where pkg passed" in world.logs[-1]


def test_a_pass_at_a_commit_whose_inputs_for_the_package_are_the_heads_vouches_for_it_and_one_that_differs_or_cant_be_read_doesnt(
        world, capsys):
    """A PR that carries another PR's Lean change needn't rerun records at its own head, hours of queue on a full node, when the
    package's inputs are byte-identical to a head where it passed.  A pass at any commit whose files of the package and the
    packages it requires by path, their tests and docs aside, and whose toolchain and checker (`tools/verity/lean/`, `tools/verity/check/common.py`) are the head's vouches for it; a commit this clone lacks is fetched, and
    one that can't be read vouches for nothing."""
    world.with_store_remote()
    _move_main(world, {"pkg/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": "../lib", "name": "lib"}]}),
                       "lib/lakefile.toml": 'name = "lib"\n', "lib/L.lean": "def l := 1\n", "lean-toolchain": "v1\n",
                       "lib/tests/T.lean": "def t := 1\n", "lib/README.md": "lib\n", "pkg/test/x.txt": "x\n",
                       "tools/verity/lean/check.py": "# the checker\n", "tools/verity/check/common.py": "# its helpers\n"})
    other = world.tmp / "other"
    _git(world.tmp, "clone", "-q", str(world.origin), str(other))
    (other / LOCK).write_text(_lock(A="2"))
    (other / "c.txt").write_text("the other PR's\n")
    _git(other, "commit", "-q", "-am", "the other PR's Lean change")
    c = _git(other, "rev-parse", "HEAD")
    rid = _published(world, c, files={"lean-audit/audit.json": _audit()})

    h = world.change({LOCK: _lock(A="2"), "a.txt": "carries it\n"})
    capsys.readouterr()
    assert _mark(world, 1, h) == 1                                     # c is on no remote: nothing vouches
    err = capsys.readouterr().err
    assert "1 of 1 commit(s) with runs can't be read here or from" in err and "--records pkg --update" in err
    _git(other, "push", "-q", "origin", f"{c}:refs/dev/other")
    assert _mark(world, 1, h) == 0
    assert f"research queue: records gate: pkg: a records run of {c[:12]} ({rid}) with identical inputs" in world.logs[-1]

    for n, files in ((6, {"lib/tests/T.lean": "def t := 2\n"}), (7, {"lib/README.md": "lib, said again\n", "lib/READMEs.txt": "x\n"}),
                     (8, {"pkg/test/x.txt": "y\n", "pkg/NOTES.md": "notes\n"})):        # tests and docs: no records run reads them
        assert _mark(world, n, world.change({LOCK: _lock(A="2"), **files})) == 0
        assert f"a records run of {c[:12]} ({rid}) with identical inputs" in world.logs[-1]

    for n, files in ((2, {"lib/L.lean": "def l := 2\n"}), (3, {"lean-toolchain": "v2\n"}),
                     (9, {"lib/lakefile.toml": 'name = "lib"\ndefaultTargets = ["L"]\n'}), (10, {"lib/tests.lean": "def t := 1\n"}),
                     (4, {"tools/verity/lean/check.py": "# the checker, changed\n"}),
                     (5, {"tools/verity/check/common.py": "# its helpers, changed\n"})):
        capsys.readouterr()
        assert _mark(world, n, world.change({LOCK: _lock(A="2"), **files})) == 1
        assert "nor at a commit whose inputs for it are the head's" in capsys.readouterr().err


def test_the_same_inputs_search_reads_every_candidate_to_its_bound_newest_first_splitting_a_fetch_the_remote_refuses(world, capsys):
    """A search bounded to a few dozen commits would refuse carried locks with 'N commit(s) with runs can't be read here'.
    It fetches every candidate up to `RECORDS_FETCH`, in chunks, and a chunk the
    remote refuses for one commit it doesn't serve is split until that commit stands alone; the rest are read."""
    from research.queue import RECORDS_FETCH

    world.with_store_remote()
    other = world.tmp / "other"
    _git(world.tmp, "clone", "-q", str(world.origin), str(other))
    (other / LOCK).write_text(_lock(A="2"))
    _git(other, "commit", "-q", "-am", "the carried change")
    c = _git(other, "rev-parse", "HEAD")
    rid = _published(world, c, files={"lean-audit/audit.json": _audit()}, at=int(_git(other, "log", "-1", "--format=%ct", c)) - 60)   # older than the rest
    for i in range(70):                                                  # newer runs, of commits whose inputs differ
        (other / LOCK).write_text(_lock(A=str(10 + i)))
        _git(other, "commit", "-q", "-am", f"another change {i}")
        _published(world, _git(other, "rev-parse", "HEAD"))
    _git(other, "push", "-q", "origin", "HEAD:refs/dev/other")
    _published(world, "ab" * 20)                                         # the newest: on no remote

    h = world.change({LOCK: _lock(A="2")})
    capsys.readouterr()
    assert RECORDS_FETCH > 72 and _mark(world, 1, h) == 0
    assert f"a records run of {c[:12]} ({rid}) with identical inputs" in world.logs[-1]


#: `tools/verity/lean/merge.py` as the gate runs it on a lock both sides changed: ours, with each entry theirs changed
LOCK_MERGE = """\
import json, sys
from pathlib import Path
o, a, b = (json.loads(Path(p).read_text() or "{}") for p in sys.argv[1:4])
Path(sys.argv[2]).write_text(json.dumps({**a, **{k: v for k, v in b.items() if o.get(k) != v}}))
"""


def test_a_head_that_only_merged_main_into_a_vouched_head_is_marked_with_no_run_of_its_own(world, capsys):
    """A lane merges `main` after its records run passed.  The fast path builds nothing for such a head and reports `path: reused`
    (every Lean input it changed is main's whole, and its lock is merge.py's merge of the two sides): that lock is a candidate,
    which the train's Lean check rebuilds and regenerates where another PR moved what it reads, so the gate marks it too, from
    the run below, with no new run.  A merge whose lock isn't merge.py's (a hand resolution), or that changes a file of its own,
    is no such head and needs its own run."""
    world.with_store_remote()
    (world.tmp / "lock_merge.py").write_text(LOCK_MERGE)
    _move_main(world, {LOCK: _lock()})
    h1 = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    rid = _published(world, h1, files={"lean-audit/audit.json": _audit()})
    _move_main(world, {LOCK: _lock(E="2"), "b.txt": "b on main\n"})
    h2 = _merge_main(world, h1)
    assert _git(world.dev, "show", f"{h2}:{LOCK}") == _lock(A="2", E="2").strip()
    assert _mark(world, 1, h2) == 0
    assert (f"{h2[:12]} only merges `main`'s" in world.logs[-1] and
            f"where records run {rid} of {h1[:12]} passed it" in world.logs[-1])

    for files in ({LOCK: _lock(A="2", C="9", E="2")}, {"pkg/A.lean": "theorem a : True := by trivial\n"}):
        _git(world.dev, "checkout", "-q", "--detach", h1)
        _git(world.dev, "merge", "-q", "--no-ff", "--no-commit", "origin/main")
        for rel, text in files.items():
            (world.dev / rel).write_text(text)
        _git(world.dev, "commit", "-q", "-am", "merge main, by hand")
        h3 = _git(world.dev, "rev-parse", "HEAD")
        _git(world.dev, "push", "-q", "origin", f"{h3}:refs/dev/{h3}")
        capsys.readouterr()
        assert _mark(world, 1, h3) == 1
        assert "--records pkg --update" in capsys.readouterr().err


def test_the_records_gate_refuses_when_the_store_cant_be_read_and_never_fails_open(world, capsys, monkeypatch):
    """A VM whose shell lost its store secrets has no remote, and one whose endpoint is down can't list it: the runs other machines
    published aren't here, so a mark would rest on nothing."""
    h = world.change({LOCK: _lock(A="2")})
    capsys.readouterr()
    assert _mark(world, 1, h, store=LocalStore(world.tmp / "bare", config_path=world.tmp / "none.toml")) == 1
    assert "the store's records runs can't be read" in capsys.readouterr().err
    world.with_store_remote()
    _published(world, h, files={"lean-audit/audit.json": _audit()})
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    assert _mark(world, 1, h, store=LocalStore(world.tmp / "cut-off", config_path=world.tmp / "store.toml")) == 1
    err = capsys.readouterr().err
    assert "can't be read (OSError: injected: endpoint unreachable)" in err and "never fails open" in err


#: a `suites.py` that says what it was asked and what it found, and fails one test when FAKE_FAIL names it
FAKE_SUITES = """\
# it knows "--changed"
import json, os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
umask = os.umask(0)
Path(os.environ["FAKE_SEEN"]).write_text(json.dumps({"argv": a, "files": sorted(p.name for p in Path.cwd().iterdir()),
                                                     "umask": umask, "cores": sorted(os.sched_getaffinity(0)),
                                                     "pythonpath": os.environ.get("PYTHONPATH"),
                                                     "built": Path("verity/core/service/flock/.lake/built").is_file(),
                                                     "lake": shutil.which("lake"),
                                                     "env": {k: os.environ.get(k) for k in ("VERITY_STORE_REQUIRED",
                                                             "CHECK_STORE_CUSTODY", "RESEARCH_RUN_ID", "GIT_AUTHOR_EMAIL",
                                                             "PYTHONPYCACHEPREFIX")}}))
fail = os.environ.get("FAKE_FAIL")
Path(a[a.index("--json") + 1]).write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed" if fail else "passed",
                                                       "failed_tests": [fail] if fail else None}]))
sys.exit(1 if fail else 0)
"""


def test_the_quick_tier_runs_the_suites_a_pr_changes_from_a_clean_checkout_at_its_head_and_a_failure_refuses(world, monkeypatch, capsys):
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    seen = world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("PYTHONPATH", str(world.tmp / "the-callers-tree"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert tier(world.gh.prs[1]) is None
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", main] and s["pythonpath"] is None   # from the PR's merge base, in its own tree
    cores = os.sched_getaffinity(0)
    assert s["umask"] == 0o002 and s["cores"] == sorted(cores - {min(cores)} if len(cores) > 1 else cores)   # a node's umask, off core 0
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    assert _git(wt, "rev-parse", "HEAD") == h1
    (wt / "stray.txt").write_text("left by the last PR\n")
    (wt / ".venv").mkdir()

    monkeypatch.setenv("FAKE_FAIL", "tests/test_a.py::test_a")
    assert tier(world.gh.prs[1]) == "pkg: tests/test_a.py::test_a"
    assert {"stray.txt", ".venv"} & set(json.loads(seen.read_text())["files"]) == {".venv"}     # cleaned, its environment kept
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: pkg: tests/test_a.py::test_a" in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False             # no label without a pass

    monkeypatch.delenv("FAKE_FAIL")
    h2 = world.change({"b.txt": "b2\n"})
    world.gh.put(1, h2, ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and _git(wt, "rev-parse", "HEAD") == h2
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is True
    h3 = world.change({"tools/verity/check/suites.py": FAKE_SUITES.replace('"--changed"', "nothing")})
    world.gh.put(1, h3, ready=False)
    assert tier(world.gh.prs[1]) is None
    assert json.loads(seen.read_text())["argv"][:2] == ["--quick", "--json"]     # a runner from before --changed: every suite


def test_the_quick_tier_builds_the_lean_verifier_before_its_tests_and_keeps_that_build_warm(world, monkeypatch):
    """The flock suite's workers each run `lake build` in the verifier; from the cold checkout `git clean` leaves, they collided."""
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "verity/core/service/flock/lakefile.toml": "name = \"flock\"\n"})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    bin_, seen, ran = world.tmp / "bin", world.tmp / "seen.json", world.tmp / "lake.log"
    bin_.mkdir()
    (bin_ / "lake").write_text('#!/bin/sh\necho "$PWD $*" >> "$FAKE_LAKE_LOG"\n[ -z "$FAKE_LAKE_FAIL" ] || exit 1\n'
                               'mkdir -p .lake && echo built > .lake/built\n')
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("FAKE_LAKE_LOG", str(ran))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_LAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert tier(world.gh.prs[1]) is None and json.loads(seen.read_text())["built"]
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    lean = wt / "verity" / "core" / "service" / "flock"
    assert ran.read_text().split("\n") == [f"{lean} build", ""]
    (lean / ".lake" / "built").write_text("from the last PR\n")
    (lean / "stray.olean").write_text("left by the last PR\n")
    monkeypatch.setenv("FAKE_LAKE_FAIL", "1")
    assert tier(world.gh.prs[1]) == "`lake build` in verity/core/service/flock exited 1 (its output is above)"
    assert (lean / ".lake" / "built").read_text() == "from the last PR\n" and not (lean / "stray.olean").exists()


def _move_main(world: World, files: dict[str, str]) -> str:
    moved = world.change(files)
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    return moved


def _merge_main(world: World, head: str) -> str:
    """``head`` with `main` merged into it, on the origin as a PR's head is."""
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", head)
    _git(world.dev, "merge", "-q", "--no-ff", "--no-edit", "origin/main")
    sha = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/dev/{sha}")
    return sha


#: a check slot's `slot.py` as the machine runs it: says how it was asked, then runs CMD
FAKE_SLOT = """\
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
Path(os.environ["FAKE_SLOT_SEEN"]).write_text(json.dumps(a))
cmd = a[a.index("--") + 1:]
os.execvp(cmd[0], cmd)
"""


class Machine:
    """`research run --on` and `research fetch` against a directory, as a machine serves them: the launch ships the `--source`
    checkout's HEAD (a shared clone of its repository, detached there; a dirty checkout is refused), puts each `--send` file in the
    run's inputs and runs the command in the shipped tree with RESEARCH_RUN_DIR and RESEARCH_RUN_ID, then writes status.json; a
    fetch copies the records into the local run dir.  ``outcome(run_dir, run_id, head)``, when given, stands in for the command.
    With ``custody_r2``, the launch stages the run's custody key under `<root>/requests/<run id>/custody/` first, as `--custody-r2`
    (the default with --on) does."""

    def __init__(self, tmp: Path, outcome=None, fetched: dict | None = None, reachable: bool = True, custody_r2: bool = False):
        self.root, self.outcome, self.fetched, self.reachable = tmp / "machine", outcome, fetched or {}, reachable
        self.custody_r2 = custody_r2
        self.launched: list[list[str]] = []

    def launch(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        self.launched.append(argv)
        flags = argv[:argv.index("--")]
        rid, src = flags[flags.index("--id") + 1], Path(flags[flags.index("--source") + 1])
        assert _git(src, "status", "--porcelain") == "", "`research run --on` ships only a clean checkout"
        head = _git(src, "rev-parse", "HEAD")
        tree, run = self.root / "src" / head, self.root / "runs" / rid
        if not tree.exists():
            tree.parent.mkdir(parents=True, exist_ok=True)
            repo = _git(src, "rev-parse", "--path-format=absolute", "--git-common-dir")
            _git(tree.parent, "clone", "-q", "--shared", "--no-checkout", repo, str(tree))
            _git(tree, "checkout", "-q", "--detach", head)
        (run / "inputs").mkdir(parents=True)
        for i, x in enumerate(flags):
            if x == "--send":
                shutil.copy(flags[i + 1], run / "inputs")
        if self.custody_r2:
            (key := self.root / "requests" / rid / "custody").mkdir(parents=True)
            (key / "store.toml").write_text('[remote]\nkind = "s3"\n')
            (key / "cred.json").write_text("{}\n")
        (Path(os.environ["RESEARCH_RUNS"]) / rid).mkdir(parents=True)
        if self.outcome is not None:
            self.outcome(run, rid, head)
        else:
            env = {**os.environ, "RESEARCH_RUN_DIR": str(run), "RESEARCH_RUN_ID": rid}
            rc = subprocess.run(argv[argv.index("--") + 1:], cwd=tree, env=env).returncode
            (run / "status.json").write_text(json.dumps({"state": "done" if rc == 0 else "failed", "rc": rc}))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fetch(self, local: Path) -> dict:
        if not self.reachable:
            raise OSError("injected: the machine is unreachable")
        for f in ("status.json", "result.json"):
            if (self.root / "runs" / local.name / f).exists():
                shutil.copy(self.root / "runs" / local.name / f, local / f)
        return dict(self.fetched)


def _remote_tier(world, monkeypatch, machine: Machine, **kw):
    """A `RemoteQuickTier` on machine `vy-test` for `world`, its clock the time its sleeps took; and
    those sleeps."""
    from research import queue as Q

    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_RUNS", str(world.tmp / "runs"))
    monkeypatch.setenv("FAKE_SLOT_SEEN", str(world.tmp / "slot-seen.json"))
    sleeps: list[float] = []
    return Q.RemoteQuickTier(world.dev, str(world.origin), "vy-test", by="lane-1", root=world.tmp / "ready", sleep=sleeps.append,
                             **{"clock": lambda: sum(sleeps), "launch": machine.launch, "fetch": machine.fetch,
                                **kw}), sleeps


def test_a_remote_quick_tier_runs_in_a_check_slot_on_the_machine_and_is_judged_as_the_local_one_is(world, monkeypatch, capsys):
    """The quick tier ran on the caller's VM, so a lane waited for its own tier and several PRs' tiers ran one at a time; `--on`
    runs it as a run on the machine, in a check slot, from the PR head's shipped tree, and the worktree's lock is free while it runs."""
    import fcntl

    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    seen = world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.setenv("PYTHONPATH", str(world.tmp / "the-callers-tree"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    machine = Machine(world.tmp)
    free_while_waiting = []

    def fetch(d: Path) -> dict:
        with open(next((world.tmp / "ready").glob("*.lock")), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)        # raises while the tier holds the worktree
            free_while_waiting.append(d.name)
        return machine.fetch(d)

    tier, sleeps = _remote_tier(world, monkeypatch, machine)
    tier.fetch = fetch
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    [argv] = machine.launched
    flags = argv[:argv.index("--")]
    rid = flags[flags.index("--id") + 1]
    assert flags[:4] == [sys.executable, "-m", "research", "run"] and "--require-result" in flags
    assert [flags[flags.index(k) + 1] for k in ("--on", "--project", "--cwd", "--declared-output")] == ["vy-test", "verity", "source", "out/*"]
    assert float(flags[flags.index("--timeout") + 1]) == Q.REMOTE_WAIT_S + Q.REMOTE_TIMEOUT_S
    ttl = flags[flags.index("--custody-ttl") + 1]
    assert int(ttl) >= Q.REMOTE_WAIT_S + Q.REMOTE_TIMEOUT_S + Q.REMOTE_GRACE_S and _custody_ttl_problem(ttl, flags) is None, \
        "the custody key outlasts the run and the wait for its upload, so `research run` takes the launch"
    assert [Path(flags[i + 1]).name for i, x in enumerate(flags) if x == "--send"] == ["quick_tier.py"]
    assert slot_only.through_slot(argv[argv.index("--") + 1:]), "cpu-1's launcher refuses a run whose command isn't slot.py's"
    assert json.loads((world.tmp / "slot-seen.json").read_text())[:8] == [
        "--by", "lane-1", "--wait-max", f"{Q.REMOTE_WAIT_S:g}", "--run-max", f"{Q.REMOTE_TIMEOUT_S:g}", "--", "sh"]
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", main] and s["pythonpath"] is None
    assert Path(s["argv"][s["argv"].index("--json") + 1]) == machine.root / "runs" / rid / "out" / "suites.json"
    assert free_while_waiting == [rid] and sleeps == [Q.REMOTE_POLL_S]
    assert f"the quick tier runs as {rid} on vy-test, in a check slot" in capsys.readouterr().out
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True

    monkeypatch.setenv("FAKE_FAIL", "tests/test_a.py::test_a")
    h2 = world.change({"b.txt": "b2\n"})
    world.gh.put(1, h2, ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: pkg: tests/test_a.py::test_a." in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is False


def test_a_remote_quick_tier_does_not_wait_for_a_local_tier_of_the_same_repository(world, monkeypatch):
    """A local tier holds its worktree's lock for the whole tier (about an hour on a lane's VM), and a remote tier from any worktree
    of the same repository that took that lock to check out and ship would make every `--on` tier on the VM wait for it;
    a remote tier ships from a worktree of its own."""
    import fcntl
    import hashlib

    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h1 = world.change({"a.txt": "a1\n"})
    world.gh.put(1, h1, ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(world.tmp / "seen.json"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)

    class NonBlocking:
        LOCK_EX, LOCK_NB = fcntl.LOCK_EX, fcntl.LOCK_NB

        @staticmethod
        def flock(f, op):
            fcntl.flock(f, op | fcntl.LOCK_NB)                  # a held lock raises here instead of hanging the test

    monkeypatch.setattr(Q, "fcntl", NonBlocking)
    common = _git(world.dev, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    (ready := world.tmp / "ready").mkdir()
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp))
    with open(ready / f"{hashlib.sha256(common.encode()).hexdigest()[:12]}.lock", "w") as local:
        fcntl.flock(local, fcntl.LOCK_EX)
        assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True


def _reports(status: dict | None = None, result=lambda doc: doc, raw: str | None = None):
    """A run's outcome on the machine: status.json `status` (done, exit 0, unless given) and the result.json of a pass at the head as
    quick_tier.py writes it, through `result` (None: no result.json), or `raw` as the file."""
    def outcome(run: Path, rid: str, head: str) -> None:
        (run / "status.json").write_text(json.dumps(status or {"state": "done", "rc": 0}))
        facts = {"kind": "quick-tier/v1", "head": head, "base": "0" * 40, "dirty": [], "lean": [], "changed": True, "rc": 0, "rows": [],
                 "logs": "out/logs", "error": None}
        doc = result({"schema": "research/result/v0.1", "kind": "quick-tier/v1", "run_id": rid, "validation": {"status": "passed"},
                      "tier": facts})
        if raw is not None:
            (run / "result.json").write_text(raw)
        elif doc is not None:
            (run / "result.json").write_text(json.dumps(doc))
    return outcome


REMOTE_OUTCOMES = {
    "a pass": ({}, None),
    "no result": ({"result": lambda d: None}, r"ended done \(exit 0\) without a result\.json"),
    "an unreadable result": ({"raw": "{not json"}, r"result\.json is unreadable: not valid JSON"),
    "another run's result": ({"result": lambda d: {**d, "run_id": "r20260101-000000-0000"}},
                             r"result\.json is unreadable: run_id 'r20260101-000000-0000' does not match this run"),
    "another kind": ({"result": lambda d: {**d, "kind": "check/v1"}}, r"result is a check/v1, not a quick-tier/v1 report"),
    "no kind": ({"result": lambda d: {k: v for k, v in d.items() if k != "kind"}}, r"result is a result with no kind, not a quick-tier/v1"),
    "no facts": ({"result": lambda d: {k: v for k, v in d.items() if k != "tier"}}, r"it reported no quick-tier/v1 facts"),
    "another head": ({"result": lambda d: {**d, "tier": {**d["tier"], "head": "f" * 40}}}, r"it ran at ffffffffffff, not at "),
    "a dirty tree": ({"result": lambda d: {**d, "tier": {**d["tier"], "dirty": ["?? stray.txt"]}}},
                     r"the checkout differed from \w{12} before anything ran: \?\? stray\.txt"),
    "a run that failed": ({"status": {"state": "failed", "rc": 1}}, r"facts show a pass, but it ended failed \(exit 1\)"),
    "a failed validation": ({"result": lambda d: {**d, "validation": {"status": "failed"}}}, r"ended done \(exit 0\) with validation failed"),
}


@pytest.mark.parametrize("case", list(REMOTE_OUTCOMES))
def test_a_remote_quick_tier_passes_only_on_a_readable_report_of_a_pass_at_the_head_from_a_run_that_ended_done(world, monkeypatch, capsys, case):
    from research import queue as Q

    kw, refusal = REMOTE_OUTCOMES[case]
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports(**kw)))
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == (0 if refusal is None else 1)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is (refusal is None)
    if refusal is not None:
        assert re.search(r"the quick tier failed: .*" + refusal, capsys.readouterr().err)


def _orphaned(world, monkeypatch, **reports):
    """A `research queue ready 1 --on vy-test` whose caller died while its run waited (a killed shell, a suspended VM): the run on the
    machine ends as ``reports`` says, and nothing marks the PR; the machine, the run's id and the head it was for."""
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports(**reports))
    tier, _ = _remote_tier(world, monkeypatch, machine)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)

    def died(d: Path) -> dict:
        raise KeyboardInterrupt("injected: the caller is gone")

    tier.fetch = died
    with pytest.raises(KeyboardInterrupt):
        Q.ready(world.gh, world.store, 1, "lane-1", tier)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    flags = machine.launched[-1]
    return machine, flags[flags.index("--id") + 1], h1


def test_judge_marks_a_pr_ready_from_its_quick_tiers_run_once_it_ended_without_its_caller(world, monkeypatch, capsys):
    """`ready --on` writes the mark only once its own wait sees the run end, so a tier that passed after its caller died left the PR
    unmarked, and the only way to mark it was to run the whole tier again."""
    from research import queue as Q

    machine, rid, h1 = _orphaned(world, monkeypatch)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 0
    assert f"#1 is ready at {h1[:12]} (pr:1@{h1})" in capsys.readouterr().out
    [label] = world.store.labels(label_target(1, h1), key=vocab.READY)
    assert label.value is True and label.by == "lane-1"
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True


def test_judge_marks_a_pr_whose_head_moved_to_the_same_tree_and_no_other(world, monkeypatch, capsys):
    """A lane that merged `main` (changing no file) while its tier ran had a passing run whose head no open PR has now."""
    from research import queue as Q

    machine, rid, h1 = _orphaned(world, monkeypatch)
    _move_main(world, {"a.txt": "a1\n", "b.txt": "b\n"})              # `main` makes the PR's own change
    world.gh.put(1, h2 := _merge_main(world, h1), ready=False)
    assert h2 != h1
    world.gh.put(2, world.change({"b.txt": "b2\n"}), ready=False)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 1           # no clone to compare trees in
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch, repo=world.dev, url=str(world.origin)) == 0
    assert f"#1's head {h2[:12]} has the tree of {h1[:12]}, where run {rid} ran the quick tier" in capsys.readouterr().out
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is True
    h3 = _git(world.dev, "commit-tree", f"{h2}^{{tree}}", "-p", h2, "-m", "the same tree")
    _git(world.dev, "push", "-q", "origin", f"{h3}:refs/dev/{h3}")
    world.gh.put(3, h3, ready=False)
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch, repo=world.dev, url=str(world.origin)) == 1
    assert "the head of each of #1, #3" in capsys.readouterr().err                          # two PRs with that tree: neither


JUDGE_REFUSALS = {
    "a run that hasn't ended": ({"status": {"state": "running"}}, r"run (\S+) hasn't ended \(last seen running\): `research inspect \1`"),
    "not a quick tier": ({"result": lambda d: {**d, "kind": "check/v1"}}, r"run \S+'s result is a check/v1, not a quick-tier/v1 report"),
    "no result": ({"result": lambda d: None}, r"run \S+ on its machine ended done \(exit 0\) without a result\.json"),
    "a failed tier": ({"status": {"state": "failed", "rc": 1}},
                      r"#1 is not marked ready at \w{12}: the quick tier failed: run \S+'s facts show a pass, but it ended failed"),
    "a head that moved": ({}, r"run \S+ ran the quick tier at \w{12}, which is no open PR's head now, nor its tree \(pushed again "
                              r"since, or closed\?\)"),
    "a run that isn't there": ({}, r"no run matching 'r-nowhere' under "),
}


@pytest.mark.parametrize("case", list(JUDGE_REFUSALS))
def test_judge_refuses_a_run_that_has_not_ended_is_not_a_quick_tier_failed_or_whose_pr_moved(world, monkeypatch, capsys, case):
    from research import queue as Q

    kw, refusal = JUDGE_REFUSALS[case]
    machine, rid, h1 = _orphaned(world, monkeypatch, **kw)
    if case == "a head that moved":
        world.gh.put(1, world.change({"a.txt": "a2\n"}), ready=False)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, "r-nowhere" if case == "a run that isn't there" else rid, "lane-1", fetch=machine.fetch,
                   repo=world.dev, url=str(world.origin)) == 1
    err = capsys.readouterr().err
    assert err.startswith("research queue judge: ") and err.count("\n") == 1 and re.search(refusal, err)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False


def test_a_remote_quick_tier_that_cannot_launch_or_is_never_seen_to_end_fails_after_a_bounded_wait(world, monkeypatch):
    from research import queue as Q

    monkeypatch.setattr(Q, "REMOTE_GRACE_S", 30)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    pr = world.gh.prs[1]
    refused = lambda argv, cwd: subprocess.CompletedProcess(argv, 3, "", "research run: refused: vy-test isn't in machines.toml\n")
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp), launch=refused)
    assert tier(pr) == "`research run --on vy-test` failed (exit 3): research run: refused: vy-test isn't in machines.toml" and sleeps == []
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports({"state": "running"})), timeout_s=60, wait_s=30)
    assert re.fullmatch(r"run r\S+ on vy-test didn't end within 120 s \(last seen running\): `research inspect r\S+`", tier(pr))
    assert sleeps == [Q.REMOTE_POLL_S] * 4, "the run's wait for a slot, its timeout and the grace, at one fetch per poll"
    tier, sleeps = _remote_tier(world, monkeypatch, Machine(world.tmp, outcome=_reports(), reachable=False), timeout_s=60, wait_s=30)
    assert re.fullmatch(r"run r\S+ on vy-test didn't end within 120 s \(last seen without a status\): .*", tier(pr)) and len(sleeps) == 4
    exited = Machine(world.tmp, outcome=lambda run, rid, head: None, fetched={"launcher_exited": {"last_line": "Traceback: boom"}})
    tier, sleeps = _remote_tier(world, monkeypatch, exited)
    assert re.fullmatch(r"run r\S+'s runner on vy-test exited before the run ended \(Traceback: boom\)", tier(pr)) and len(sleeps) == 1
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False


def test_a_remote_quick_tier_that_never_got_a_check_slot_never_started_and_did_not_fail(world, monkeypatch, capsys):
    """A tier that spends its whole wait in a node's line for a check slot isn't reported failed.  slot.py gives up at --wait-max with exit 75 before running anything; that is no verdict on the head, and says so."""
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports({"state": "failed", "rc": Q.SLOT_NOT_STARTED}, result=lambda d: None))
    tier, _ = _remote_tier(world, monkeypatch, machine)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == Q.READY_NOT_STARTED
    err = capsys.readouterr().err
    assert re.search(r"#1 is not marked ready at \w{12}: the quick tier never started: run \S+ on vy-test waited for a check slot "
                     r"until its --wait-max and never ran the tier", err) and "the quick tier failed" not in err
    assert "A PR lands on `checked` on a review, and reaches `main` on the full check after, so the quick tier isn't required" in err and "research run --on vy-nebius-cpu-K" in err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    rid = machine.launched[-1][machine.launched[-1].index("--id") + 1]
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 1
    assert "never ran the tier" in capsys.readouterr().err


def test_a_remote_quick_tier_on_auto_runs_on_the_machine_placed_and_without_one_never_starts(world, monkeypatch, capsys):
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports())
    asked = []
    placed = lambda spec, repo: asked.append(spec) or ("vy-test", "a check slot and a Lean check slot are free (check slots: d free)")
    tier, _ = _remote_tier(world, monkeypatch, machine, place=placed)
    tier.on = "auto"
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and asked == ["auto"]
    flags = machine.launched[-1][:machine.launched[-1].index("--")]
    assert flags[flags.index("--on") + 1] == "vy-test"
    assert "research queue: --on auto: vy-test: a check slot and a Lean check slot are free" in capsys.readouterr().out
    tier.place, tier.on = (lambda spec, repo: (None, "no machine's slots could be read")), "vy-a,vy-b"
    world.gh.put(1, h2 := world.change({"a.txt": "a2\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == Q.READY_NOT_STARTED and len(machine.launched) == 1
    assert "the quick tier never started: --on vy-a,vy-b: no machine's slots could be read" in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is False


def test_a_runner_seen_gone_just_after_it_recorded_its_end_is_not_lost(tmp_path):
    """A tier that passed with rc 0 mustn't be reported lost. A fetch copies
    status.json, then probes the runner's pid; a runner that records its end and exits in between reads as gone with a status
    that isn't terminal. The wait fetches once more before calling it lost, and only a runner still gone then is."""
    from research import queue as Q

    fetched = []

    def fetch(d: Path, ended: bool = True) -> dict:
        fetched.append(d)
        if len(fetched) == 1 or not ended:
            (d / "status.json").write_text(json.dumps({"state": "running", "pid": 7}))
            return {"runner_gone": {"why": "its runner (pid 7) is gone without recording an end"}}
        (d / "status.json").write_text(json.dumps({"state": "done", "rc": 0}))
        return {}

    (d := tmp_path / "runs" / "r1").mkdir(parents=True)
    tier = Q.RemoteQuickTier(tmp_path, "origin", "vy-test", poll_s=30, fetch=fetch, sleep=lambda s: None)
    assert tier._wait(d, "vy-test") == ({"state": "done", "rc": 0}, None) and len(fetched) == 2
    fetched.clear()
    tier.fetch = lambda d: fetch(d, ended=False)
    _, why = tier._wait(d, "vy-test")
    assert why == "run r1 on vy-test ended without recording it: its runner (pid 7) is gone without recording an end"
    assert len(fetched) == 2


def test_a_remote_quick_tiers_wait_is_bounded_in_time_however_long_each_fetch_takes(tmp_path):
    """The wait doesn't count polls as if a fetch took no time: `remote.fetch` may take its ssh's 900 s each poll, and against a
    machine that takes the ssh and then hangs, 180 polls of 930 s are about two days, not the run's timeout and the grace.
    Simulated time only."""
    from research import queue as Q

    clock = [0.0]

    def advance(s: float) -> None:
        clock[0] += s

    def hung_fetch(d: Path) -> dict:
        clock[0] += 900
        return {}

    bound = 1800 + 3600 + Q.REMOTE_GRACE_S
    tier = Q.RemoteQuickTier(tmp_path, "origin", "vy-test", timeout_s=3600, wait_s=1800, max_wait_s=bound + 1, poll_s=30,
                             fetch=hung_fetch, sleep=advance, clock=lambda: clock[0])
    (d := tmp_path / "runs" / "r1").mkdir(parents=True)
    _, why = tier._wait(d, "vy-test")
    assert why == f"run r1 on vy-test didn't end within {bound:g} s (last seen without a status): `research inspect r1`"
    assert bound <= clock[0] <= bound + 30 + 900, "it ends with the poll under way at the deadline"
    clock[0] = 0.0
    tier.max_wait_s = Q.waitcap.WAIT_CAP_S
    _, why = tier._wait(d, "vy-test")
    assert isinstance(why, Q.StillGoing) and Q.waitcap.WAIT_CAP_S <= clock[0] <= Q.waitcap.WAIT_CAP_S + 900


def test_ready_on_stops_waiting_at_twenty_minutes_and_judge_marks_the_pr_once_the_run_ends(world, monkeypatch, capsys):
    """No tool call may block more than 20 minutes, and a quick tier may wait hours for a check slot: `ready --on`
    leaves its run going, marks nothing, names `judge RUN`, and exits STILL_GOING; `judge` marks the PR once the run passed."""
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports())
    tier, sleeps = _remote_tier(world, monkeypatch, machine, max_wait_s=60)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    fetched = []
    tier.fetch = lambda d: fetched.append(d) or {}
    capsys.readouterr()
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == Q.waitcap.STILL_GOING
    assert sum(sleeps) == 60 and len(fetched) == 2
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    rid = (flags := machine.launched[-1])[flags.index("--id") + 1]
    err = capsys.readouterr().err
    assert f"#1 is not marked ready at {h1[:12]} yet" in err and f"the quick tier's run {rid} on vy-test" in err
    assert f"Check back with `research queue judge {rid} --by lane-1`" in err and "Nothing was cancelled" in err
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 0
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True


def test_a_remote_quick_tier_gives_its_suites_the_store_as_a_recorded_check_does(world, monkeypatch):
    """On the machine the workload's environment holds no store credential, so `check.py` hands its steps the run's custody key
    (CHECK_STORE_CUSTODY) and VERITY_STORE_REQUIRED=1.  A tier that handed neither would skip C-Flock's store-backed tests as "the
    evidence store is not reachable", and the skip would pass.  It hands both as check.py does, never the caller's,
    and nothing of the recording run's own; with no key staged, the store is still required.  The machine's PYTHONPYCACHEPREFIX
    doesn't reach the suites, or test_remote_cwd.py's runs would write their bytecode there instead of under their own root and
    fail every tier that runs tools/research on that machine."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen := world.tmp / "seen.json"))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(world.tmp / "the-machines-pycache"))
    monkeypatch.setenv("CHECK_STORE_CUSTODY", str(world.tmp / "the-callers-custody"))
    monkeypatch.delenv("VERITY_STORE_REQUIRED", raising=False)
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    machine = Machine(world.tmp, custody_r2=True)
    tier, _ = _remote_tier(world, monkeypatch, machine)
    assert tier(world.gh.prs[1]) is None
    flags = machine.launched[-1]
    key = machine.root / "requests" / flags[flags.index("--id") + 1] / "custody"
    env = json.loads(seen.read_text())["env"]
    assert env["VERITY_STORE_REQUIRED"] == "1" and env["CHECK_STORE_CUSTODY"] == str(key.resolve())
    assert env["RESEARCH_RUN_ID"] is None and env["PYTHONPYCACHEPREFIX"] is None and env["GIT_AUTHOR_EMAIL"]
    machine.custody_r2 = False
    assert tier(world.gh.prs[1]) is None
    env = json.loads(seen.read_text())["env"]
    assert env["VERITY_STORE_REQUIRED"] == "1" and env["CHECK_STORE_CUSTODY"] is None


def test_a_quick_tier_with_no_lake_on_the_path_builds_with_elans_and_with_none_refuses(world, monkeypatch):
    """On a machine the run's PATH is a non-login shell's, without elan's bin, which check.py puts first itself.  A tier that
    built the verifier only with `lake` on the PATH would skip its Lean-dependent tests and label the PR ready.  It takes `lake` from ~/.elan/bin, first on the suites' PATH too, and with neither refuses."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT, "verity/core/service/flock/lakefile.toml": 'name = "flock"\n'})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen := world.tmp / "seen.json"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier, _ = _remote_tier(world, monkeypatch, Machine(world.tmp))
    (bare := world.tmp / "no-lake-bin").mkdir()
    for t in ("git", "python3", "sh", "mkdir"):
        (bare / t).symlink_to(shutil.which(t))
    monkeypatch.setenv("PATH", str(bare))
    monkeypatch.setenv("HOME", str(home := world.tmp / "home"))
    lean = "verity/core/service/flock"
    assert tier(world.gh.prs[1]) == f"no `lake` on the PATH or in {home}/.elan/bin to build {lean} with, and without that build its tests skip"
    (elan := home / ".elan" / "bin").mkdir(parents=True)
    (elan / "lake").write_text("#!/bin/sh\nmkdir -p .lake && echo built > .lake/built\n")
    (elan / "lake").chmod(0o755)
    assert tier(world.gh.prs[1]) is None
    s = json.loads(seen.read_text())
    assert s["built"] and s["lake"] == str(elan / "lake")


def test_research_queue_ready_takes_one_of_on_and_local(world, monkeypatch, capsys):
    monkeypatch.chdir(world.dev)
    for argv in (["queue", "ready", "1"], ["queue", "ready", "1", "--on", "vy-test", "--local"]):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and "ready takes one of --on MACHINE" in capsys.readouterr().err
    for argv, why in ((["queue", "judge"], "judge needs the quick tier's run id"),
                      (["queue", "eject", "r20261004-160000-abcd", "--reason", "x"], "eject: 'r20261004-160000-abcd' is not a PR number"),
                      (["queue", "eject", "1", "--reason", "x", "--dry-run"], "--dry-run is for sync and drive only"),
                      (["queue", "ready", "1", "--local", "--dry-run"], "--dry-run is for sync and drive only"),
                      (["queue", "drive"], "drive needs --on MACHINE")):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err


def test_research_queue_ready_and_judge_name_a_lane_never_merge_queue(world, monkeypatch, capsys):
    """Ready is the owner's word, and the merge gate refuses one by `merge-queue`: with no --by and no $RESEARCH_LANE, `ready`
    and `judge` refuse before writing anything; `eject` still defaults to merge-queue."""
    monkeypatch.chdir(world.dev)
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    for argv in (["queue", "ready", "1", "--local"], ["queue", "judge", "r20261004-160000-abcd"]):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and "writes the owner's ready mark: pass --by <lane>" in capsys.readouterr().err


def test_quick_tier_py_refuses_a_call_it_cannot_place(tmp_path, monkeypatch):
    from research import quick_tier

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    for argv in ([], ["--head", "h", "--base", "b"], ["--head", "h", "--"], ["--head", "h", "--base", "b", "--what", "x", "--", "c"]):
        assert quick_tier.main(argv) == 2
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    assert quick_tier.main(["--head", "h", "--base", "b", "--", "c"]) == 2 and not (tmp_path / "result.json").exists()


def test_a_quick_tier_in_a_check_slot_starts_lake_from_the_slots_warm_copy(tmp_path):
    """Each tier's checkout is fresh, so the verifier's `lake build` would run cold in every one and hold a node's slot for most
    of a small PR's tier. In a slot (CHECK_SLOT) it starts from the slot's last passing `.lake`; a warm build that fails
    is retried cold, and outside a slot nothing is kept."""
    from research import quick_tier

    lake = tmp_path / "lake"
    lake.write_text('#!/bin/sh\n[ -f .lake/warm ] && w=warm || w=cold\necho "$w" >> "$LOG"\n'
                    '[ "$w" = warm ] && [ -n "$FAIL_WARM" ] && exit 3\nmkdir -p .lake && echo x > .lake/warm\n')
    lake.chmod(0o755)
    log = tmp_path / "log"
    env = {"HOME": str(tmp_path / "home"), "PATH": os.environ["PATH"], "LOG": str(log), "CHECK_SLOT": "a"}
    d = "verity/core/service/flock"

    def run(n, **extra):
        (pkg := tmp_path / f"t{n}" / d).mkdir(parents=True)
        return quick_tier.build(str(lake), pkg, d, {**env, **extra})

    assert run(1) == {"dir": d, "rc": 0, "start": "cold"}
    warm = tmp_path / "home" / ".cache" / "verity" / "lake-slot-a" / "verity--core--service--flock"
    assert (warm / "warm").is_file()
    assert run(2) == {"dir": d, "rc": 0, "start": "warm"}
    assert run(3, FAIL_WARM="1") == {"dir": d, "rc": 0, "start": "cold"}
    assert log.read_text().split() == ["cold", "warm", "warm", "cold"]
    assert quick_tier.warm_dir({"HOME": env["HOME"]}, d) is None and quick_tier.warm_dir({**env, "CHECK_SLOT": "../x"}, d) is None
    env.pop("CHECK_SLOT")
    assert run(4) == {"dir": d, "rc": 0, "start": "cold"} and log.read_text().split()[-1] == "cold"


def test_a_gh_call_that_never_answers_fails_the_tick_instead_of_holding_it(monkeypatch):
    """Drive's listing of the open PRs can hang for minutes on a dead connection, and gh has no timeout of its own."""
    import research.merge as M
    import research.queue as Q

    asked = []

    def hangs(argv, **kw):
        asked.append(kw.get("timeout"))
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(subprocess, "run", hangs)
    with pytest.raises(QueueError, match=f"no answer in {Q.GH_TIMEOUT_S} s"):
        Q.GitHub("o/r").open_prs()
    with pytest.raises(M.Refused, match=f"no answer in {M.GH_TIMEOUT_S} s"):
        M._gh_cli(Path("."), "pr", "view", "1")
    assert asked == [Q.GH_TIMEOUT_S, M.GH_TIMEOUT_S]


def test_handle_reads_mains_registry_once_per_fetch(world, monkeypatch):
    """A review tick asks `handle` of every member of `next` (`Driver._owners`), and each read of main's registry is a
    `git show` of ~70 ms in drive's clone on the network volume, times over a hundred members a tick.  A fetch reads it afresh."""
    from research import queue as Q

    def registry(*handles: str) -> None:
        main = world.change({".agents/skills/using-slack/registry.json": json.dumps({"handles": [{"handle": h} for h in handles]})})
        _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")

    registry("circuits")
    q = world.queue()
    q._clone()
    q.fetch()
    shows = []
    real = Q._git
    monkeypatch.setattr(Q, "_git", lambda repo, *a, **kw: (shows.append(a) if a[:1] == ("show",) and a[1].endswith(str(Q.REGISTRY))
                                                            else None) or real(repo, *a, **kw))
    prs = [PR(n, "h", "t", body) for n, body in ((1, "Lane circuits"), (2, "Lane infra"), (3, "Lane circuits"))]
    assert [q.handle(p) for p in prs] == ["circuits", None, "circuits"]
    assert len(shows) == 1
    registry("circuits", "infra")
    q.fetch()
    assert [q.handle(p) for p in prs] == ["circuits", "infra", "circuits"]
    assert len(shows) == 2


def test_a_fetch_moves_a_clone_main_has_passed_to_main_and_keeps_one_that_contains_main(world):
    """The merge drivers run from the clone's checkout: an agent VM's clone still checked out at an old `main` would judge a
    lean-audit.json with that day's merge.py and refuse a merge main's resolves.  Drive's checkout, `next`'s tip, stays."""
    q = world.queue()
    q._clone()
    q.fetch()
    old = _git(q.workdir, "rev-parse", "HEAD")
    main = world.change({"a.txt": "new\n"})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    q.fetch()
    assert _git(q.workdir, "rev-parse", "HEAD") == main != old
    ahead = world.change({"b.txt": "on next\n"})
    _git(world.dev, "push", "-q", "origin", f"{ahead}:refs/heads/next")
    q.fetch()
    _git(q.workdir, "checkout", "-q", "--detach", ahead)
    q.fetch()
    assert _git(q.workdir, "rev-parse", "HEAD") == ahead


def test_a_lane_line_names_its_handle_without_a_trailing_period():
    """A body that says `Lane proofs.` is owned by `proofs`, not `proofs.`, a handle no one holds."""
    for body, lane in (("Lane proofs.", "proofs"), ("Lane `lane-2`.", "lane-2"), ("Lane a.", "a"), ("Lane ci-x_", "ci-x"),
                       ("Lane: vllm\n\nmore", "vllm")):
        assert PR(1, "h", "t", body).owner == lane, body
