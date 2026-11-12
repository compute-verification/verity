"""`research queue`: `next` is `main` plus the admitted PRs, one merge commit each; `main` fast-forwards to the newest commit on it
with a passing `check` of that exact commit.  Every test runs on a throwaway bare origin, with a fake GitHub that serves the open
PRs and writes their `refs/pull/{n}/head`."""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import itertools
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import pytest

from goals_page import DOWN, PAGE, serve
import research.queue as Q
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
    repo = Q.REPO

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

    def queue(self, first: tuple[int, ...] | None = None, **kw) -> Queue:
        """A queue on this world's origin; ``first``: the PRs drive's priority file lists, first to land first (`prioritize`)."""
        kw.setdefault("ready", lambda n, h: (n, h) in self.gh.ready)
        q = Queue(self.tmp / "q" / "verity", remote=str(self.origin), github=self.gh,
                  store=self.store, log=self.logs.append, **kw)
        if first is not None:
            self.prioritize(q, *first)
        return q

    def prioritize(self, q: Queue, *prs: int) -> None:
        """Write drive's priority file, ``prs`` in their order, first to land first, and give it to ``q`` as `--priority-file`
        does: each of ``q``'s syncs reads it again."""
        (path := self.tmp / "priority").write_text(" ".join(map(str, prs)) + "\n")
        q.priority_file = path

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
    assert entries[2].why == f"GitHub's API head {new[:12]} and pull ref {old[:12]} disagree; retried at the next reading"
    assert world.logs[-2] == (f"research queue: #2's API head {new[:12]} and fetched pull ref {old[:12]} disagree; retried at the "
                              "next reading")

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


def test_the_priority_list_puts_a_ready_fix_at_the_bottom_of_next_ahead_of_members_that_keep_their_place(world):
    """A fix for a test that fails on `main` needn't sit behind the next train: drive's priority list restacks `next` so the
    next train starts with the fix.  A PR in the list that isn't ready is passed over, and another queue's sync without the
    list keeps the new order."""
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    world.queue().sync()
    world.gh.put(9, world.change({"fix.txt": "fix\n"}))
    world.gh.put(8, world.change({"other.txt": "x\n"}), ready=False)
    members, _ = world.queue(first=(8, 9)).sync()
    assert [m.pr for m in members] == [9, 1, 2, 3]
    again, _ = world.queue().sync()
    assert [m.commit for m in again] == [m.commit for m in members]


def test_an_edit_of_the_priority_file_reorders_next_at_the_next_sync_from_the_first_pr_whose_place_changed(world):
    """Each sync reads drive's priority file again: a number repeated keeps its first place, so it moves nothing; swapping two
    PRs rebuilds `next` from the lower one, and listing one more after them merges again only what is above them."""
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    q.sync()
    for n in (7, 9):
        world.gh.put(n, world.change({f"{n}.txt": f"{n}\n"}))
    world.prioritize(q, 9, 7)
    assert [m.pr for m in q.sync()[0]] == [9, 7, 1, 2, 3] and "kept 0, merged 5" in world.logs[-1], world.logs[-1]
    world.prioritize(q, 9, 7, 9)
    assert [m.pr for m in q.sync()[0]] == [9, 7, 1, 2, 3] and "kept 5, merged 0" in world.logs[-1], world.logs[-1]
    world.prioritize(q, 7, 9)
    assert [m.pr for m in q.sync()[0]] == [7, 9, 1, 2, 3] and "kept 0, merged 5" in world.logs[-1], world.logs[-1]
    world.prioritize(q, 7, 9, 2)
    members, _ = q.sync()
    assert [m.pr for m in members] == [7, 9, 2, 1, 3] and "kept 2, merged 3" in world.logs[-1], world.logs[-1]
    assert q.sync()[0] == members and "kept 5, merged 0" in world.logs[-1], world.logs[-1]


def test_a_sync_merges_a_lean_change_like_any_other_in_the_priority_lists_then_admission_order(world):
    """A Lean change waits for no check before it lands, so a sync merges it like any other member: drive's priority
    list's order, then admission's, each PR above the PRs it contains; what the sync keeps never moves for what it adds."""
    world.gh.put(1, h1 := world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, h2 := world.change({"a.txt": "2\n"}))
    world.gh.put(3, h3 := world.change({"pkg/B.lean": "-- 3\n"}))
    world.gh.put(4, world.change({"b.txt": "4\n"}))
    world.gh.put(5, world.change({"c.txt": "5\n"}, base=h1))                 # changes no Lean, but contains #1
    world.gh.put(6, world.change({"d.txt": "6\n"}, base=h2))
    members, _ = world.queue(first=(3, 4)).sync()
    assert [m.pr for m in members] == [3, 4, 1, 2, 5, 6]
    world.gh.put(9, world.change({"pkg/C.lean": "-- 9\n"}))
    world.gh.put(10, world.change({"e.txt": "10\n"}, base=h3))
    world.gh.put(11, world.change({"f.txt": "11\n"}))
    again, _ = world.queue().sync()
    assert [m.pr for m in again] == [3, 4, 1, 2, 5, 6, 9, 10, 11]
    assert [m.commit for m in again[:6]] == [m.commit for m in members]      # what the sync keeps never moves for it
    world.gh.put(4, world.change({"b.txt": "4 rewritten\n"}))                # moved: the rest merged again, in order
    rebuilt, _ = world.queue().sync()
    assert [m.pr for m in rebuilt] == [3, 1, 2, 5, 6, 9, 10, 11, 4]


def test_a_sync_with_one_new_admission_keeps_every_member_and_merges_only_it(world):
    """Each PR a sync merges runs the landing gates (~15 s each), so a new admission placed low re-merged most of `next`, a
    160-member pass taking 40-50 min.  A new PR, Lean or not, goes on top: the sync keeps every member."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/A.lean": "-- 2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    members, _ = world.queue().sync()
    assert [m.pr for m in members] == [1, 2, 3]
    for n, files in ((4, {"c.txt": "4\n"}), (5, {"pkg/B.lean": "-- 5\n"})):
        world.gh.put(n, world.change(files))
        again, _ = world.queue().sync()
        assert [m.commit for m in again[:len(members)]] == [m.commit for m in members] and again[-1].pr == n
        assert world.logs[-1].endswith(f"kept {len(members)}, merged 1"), world.logs[-1]
        members = again


def test_members_merged_again_above_a_moved_one_keep_their_order_and_only_new_prs_are_sorted(world):
    """Members above a moved or dropped member are merged again in `next`'s order: sorting them again would undo drive's lift
    (reviewed members first), and the next lift would merge them all again, each tick.  What comes new is sorted."""
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


def test_a_pr_of_the_priority_list_goes_to_the_bottom_and_the_members_above_it_keep_their_order(world):
    """Drive's priority list restacks `next` from the bottom: the PRs it names go below every member, which are merged
    again above them in their order; the next sync with the same list keeps it all."""
    world.gh.put(1, world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"pkg/B.lean": "-- 3\n"}))
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2, 3]
    world.gh.put(9, world.change({"fix.txt": "9\n"}))
    members, _ = world.queue(first=(9,)).sync()
    assert [m.pr for m in members] == [9, 1, 2, 3] and world.logs[-1].endswith("kept 0, merged 4")
    again, _ = world.queue(first=(9,)).sync()
    assert again == members and world.logs[-1].endswith("kept 4, merged 0")


def test_a_pr_the_last_sync_left_out_is_tried_above_the_rest_so_a_sync_with_nothing_new_keeps_next(world):
    """A PR that didn't merge, left where drive's priority list put it in the order the sync compares `next` with, would have
    every member above it merged again (new commits) on each tick with `main` still.  Left out at a head, it (and a PR that
    contains it) is tried above the rest until that head moves."""
    world.gh.put(1, world.change({"pkg/A.lean": "-- 1\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    world.gh.put(4, world.change({"d.txt": "4\n"}, base=world.gh.prs[3].head))
    _git(world.dev, "push", "-q", "origin", f"{world.change({'c.txt': 'main\n'})}:refs/heads/main")
    first = (3, 2, 4, 1)
    members, entries = world.queue(first=first).sync()
    assert [m.pr for m in members] == [2, 1] and entries[3].state == "refused" and entries[4].state == "waiting"
    for reused in ("", ", merges reused for 1"):        # #3 tried above the rest, then that conflict taken from its record
        again, entries = world.queue(first=first).sync()
        assert [m.commit for m in again] == [m.commit for m in members] and world.logs[-1].endswith(f"kept 2, merged 0{reused}")
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


def test_a_seeded_clone_gets_the_record_drivers_it_lacks_once_and_plain_git_then_merges_two_records_with_merge_py(
        world, monkeypatch):
    """A queue clone seeded from a copy without the drivers (or with another value) gets every `RECORD_DRIVER` key on its
    first sync, logged once, and a later sync only reads them; then a git without `git_env`'s settings merges two PRs'
    lean-audit.json records in that clone through merge.py rather than line by line."""
    from research.clone import RECORD_DRIVER

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    tools = {f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}
    base = world.change({**tools, ".gitattributes": "lean-audit.json merge=lean-audit\n"})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    wd = world.tmp / "q" / "verity"
    _git(world.tmp, "clone", "-q", str(world.origin), str(wd))
    _git(wd, "config", "--local", "merge.lean-caps.driver", "true")

    def said() -> list[str]:
        return [x for x in world.logs if "set the record merge drivers" in x]

    world.queue().sync()
    (line,) = said()
    assert line.endswith(", ".join(RECORD_DRIVER)), line
    assert {k: _git(wd, "config", "--local", "--get", k) for k in RECORD_DRIVER} == RECORD_DRIVER
    world.queue().sync()
    assert said() == [line]

    rec = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    a, b = (world.change({"pkg/lean-audit.json": json.dumps({**rec, "guarantees": {**rec["guarantees"], g: "1"}})}, base=base)
            for g in "AB")
    for k in [k for k in os.environ if k.startswith("GIT_CONFIG")]:
        monkeypatch.delenv(k)
    _git(wd, "fetch", "-q", "origin", f"refs/dev/{a}", f"refs/dev/{b}")
    tree = _git(wd, f"--attr-source={base}", "merge-tree", "--write-tree", "--no-messages", a, b)
    assert set(json.loads(_git(wd, "show", f"{tree}:pkg/lean-audit.json"))["guarantees"]) == {"T", "A", "B"}


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
    old = Queue(world.tmp / "q-old" / "verity", remote=str(world.origin), github=world.gh, store=world.store,
                log=world.logs.append, ready=lambda n, h: (n, h) in world.gh.ready, shadow=True)

    def in_worktree(tip, pr, members):
        _git(old.workdir, "checkout", "-q", "--detach", tip)
        r = Q._git(old.workdir, "merge", "--no-ff", "--no-edit", "-m", Q._message(pr), pr.head, check=False, env=Q.IDENTITY)
        if r.returncode == 0:
            return _git(old.workdir, "rev-parse", "HEAD"), ""
        files = _git(old.workdir, "diff", "--name-only", "--diff-filter=U").split()
        Q._git(old.workdir, "merge", "--abort", check=False)
        _git(old.workdir, "reset", "-q", "--hard", tip)
        return None, old._unmerged((r.stderr or r.stdout).strip()[-200:], files, members)

    monkeypatch.setattr(old, "_merge", in_worktree)

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


def test_a_sync_on_a_git_without_attr_source_raises_rather_than_merging_without_the_pin_merge(world, monkeypatch):
    """A git before 2.40 refuses `--attr-source` (exit 129); merging without it left out every PR whose only conflict is a pin,
    silently.  The sync raises, naming the fix, and pushes no `next`."""
    real = Q._git

    def old_git(repo, *args, **kw):
        if args[0].startswith("--attr-source"):
            return subprocess.CompletedProcess(args, 129, "", f"unknown option: {args[0]}\nusage: git [--version] [--help]")
        return real(repo, *args, **kw)

    monkeypatch.setattr(Q, "_git", old_git)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    with pytest.raises(QueueError, match=r"has no `--attr-source`, which merging `next` needs: install git >= 2\.40"):
        world.queue().sync()
    assert not _git(world.origin, "for-each-ref", "refs/heads/next")


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


POUS = "verity/protocols/pous/lean"
TRUSTED, EXFIL = f"{POUS}/TRUSTED.sha256", "verity/protocols/spot/verifier/tests/fixtures/exfiltration_lean.json"
UP = posixpath.relpath("security/Proofs/lakefile.toml", POUS)
#: what `_pin_main`'s pins record: PoUS's, relative to its directory, and the JSON pin's
MAIN_TRUSTED, MAIN_EXFIL = ["lean-toolchain", "lakefile.toml", "Old.lean", UP], ["security/Proofs/lakefile.toml", "x.lean"]
VECTORS = ('{"a": 1}', '{"b": 2}')


def _trusted(paths: list[str], tree: dict[str, str]) -> str:
    """PoUS's pin, a sha256sum listing of ``paths`` (relative to its directory) as ``tree`` holds them."""
    return "".join(f"{hashlib.sha256(tree[posixpath.normpath(f'{POUS}/{p}')].encode()).hexdigest()}  {p}\n" for p in paths)


def _exfil(paths: list[str], tree: dict[str, str], vectors: tuple[str, ...] = VECTORS) -> str:
    """A JSON pin under `inputs`, followed by ``vectors`` one per line as its generate.sh writes them."""
    doc = {"source": "s", "inputs": {p: hashlib.sha256(tree[p].encode()).hexdigest() for p in paths}}
    return json.dumps(doc, indent=1)[:-2] + ',\n "vectors": [\n' + ",\n".join(vectors) + "\n ]\n}\n"


def _pinned(world: World, tree: dict[str, str], trusted: list[str], exfil: list[str], base: str = "origin/main",
            vectors: tuple[str, ...] = VECTORS) -> str:
    return world.change({**tree, TRUSTED: _trusted(trusted, tree), EXFIL: _exfil(exfil, tree, vectors)}, base=base)


def _pin_main(world: World) -> dict[str, str]:
    """`main` with the real pins tool and both pins over a few inputs; returns those inputs."""
    tree = {f"{POUS}/lakefile.toml": "l0\n", f"{POUS}/lean-toolchain": "t0\n", f"{POUS}/Old.lean": "o0\n",
            "security/Proofs/lakefile.toml": "s0\n", "x.lean": "x0\n"}
    pins = (Path(__file__).resolve().parents[2] / "verity" / "check" / "pins.py").read_text()
    base = _pinned(world, {**tree, "tools/verity/check/pins.py": pins}, MAIN_TRUSTED, MAIN_EXFIL)
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    return tree


def test_prs_whose_only_conflicts_are_in_pins_merge_with_each_pin_rebuilt_from_the_merged_tree(world):
    """Two PRs that each regenerated PoUS's TRUSTED.sha256 for their own change to a pinned input both merge onto `next`: each conflicted pin records both sides' paths (OURS's
    order, then THEIRS's new ones) that the merged tree still holds, at their digests there, and a JSON pin keeps the rest of
    its text byte for byte."""
    tree = _pin_main(world)
    one = {**tree, f"{POUS}/lakefile.toml": "l1\n", "security/Proofs/lakefile.toml": "s1\n", f"{POUS}/New1.lean": "n1\n",
           "y.lean": "y\n"}
    world.gh.put(1, _pinned(world, one, ["lean-toolchain", "lakefile.toml", "Old.lean", UP, "New1.lean"],
                            ["security/Proofs/lakefile.toml", "x.lean", "y.lean"]))
    two = {**tree, f"{POUS}/lean-toolchain": "t1\n", f"{POUS}/Old.lean": None, f"{POUS}/New2.lean": "n2\n", "x.lean": "x1\n",
           "z.lean": "z\n"}
    world.gh.put(2, _pinned(world, two, ["lean-toolchain", "lakefile.toml", UP, "New2.lean"],
                            ["security/Proofs/lakefile.toml", "x.lean", "z.lean"]))
    raw = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", world.gh.prs[1].head,
                          world.gh.prs[2].head], cwd=world.dev, capture_output=True, text=True)
    assert sorted(raw.stdout.splitlines()[1:]) == sorted([TRUSTED, EXFIL])
    # the preflight and `status` judge it as the sync merges it: no conflict
    assert conflicting_files(world.dev, world.gh.prs[1].head, world.gh.prs[2].head) == []
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2], entries[2].why
    assert f"research queue: #2 conflicts only in pins: {', '.join(sorted([TRUSTED, EXFIL]))} rebuilt from the merged tree" in world.logs
    merged = {k: v for k, v in {**one, **two, f"{POUS}/lakefile.toml": "l1\n", "security/Proofs/lakefile.toml": "s1\n"}.items()
              if v is not None}
    assert all(_git(world.origin, "show", f"refs/heads/next:{p}") + "\n" == t for p, t in merged.items())
    def show(p: str) -> str:
        return subprocess.run(["git", "show", f"refs/heads/next:{p}"], cwd=world.origin, check=True, capture_output=True,
                              text=True).stdout

    assert show(TRUSTED) == _trusted(["lean-toolchain", "lakefile.toml", UP, "New1.lean", "New2.lean"], merged)
    assert show(EXFIL) == _exfil(["security/Proofs/lakefile.toml", "x.lean", "y.lean", "z.lean"], merged)
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    assert _git(world.origin, "rev-parse", f"{tip}^2") == world.gh.prs[2].head
    assert _git(world.origin, "log", "-1", "--format=%B", tip) == f"Queue: #2 change 2\n\nPR: #2\nHead: {world.gh.prs[2].head}\nOwner: lane-2"


def test_prs_that_each_changed_a_json_pins_vectors_conflict_in_the_pin_rather_than_one_losing_its_vectors(world):
    """Two PRs that each added vectors to a JSON pin, beside its digests for their own inputs, conflict only in the pin, but
    neither side's text is the merge's: rebuilding it from `next`'s text dropped the PR's vectors, and the test reading one
    failed after landing.  The merge refuses, naming the pin and the PR that touched it, and so do `status` and the preflight
    (`conflicting_files`)."""
    tree = _pin_main(world)
    world.gh.put(1, _pinned(world, {**tree, "x.lean": "x1\n"}, MAIN_TRUSTED, MAIN_EXFIL, vectors=(*VECTORS, '{"c": 3}')))
    world.gh.put(2, _pinned(world, {**tree, "y.lean": "y\n"}, MAIN_TRUSTED, [*MAIN_EXFIL, "y.lean"],
                            vectors=(*VECTORS, '{"d": 4}')))
    assert conflicting_files(world.dev, world.gh.prs[1].head, world.gh.prs[2].head) == [EXFIL]
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1]
    assert entries[2].state == "refused" and entries[2].why == f"conflicts with `next` in {EXFIL} (touched by #1)"
    assert not [line for line in world.logs if "conflicts only in pins" in line], world.logs


@pytest.mark.parametrize("by", ["next", "the PR"])
def test_a_json_pins_vectors_one_side_alone_changed_are_kept_with_its_digests_rebuilt(world, by):
    """A conflict in a JSON pin whose vectors one side alone changed, `next`'s member or the PR, while the other changed only
    its digests is a pin's: the merge keeps that side's vectors, with the digests rebuilt from the merged tree, where
    rebuilding it from `next`'s text dropped a PR's."""
    tree = _pin_main(world)
    more = (*VECTORS, '{"c": 3}')
    one, two = {**tree, "x.lean": "x1\n"}, {**tree, "z.lean": "z\n"}
    world.gh.put(1, _pinned(world, one, MAIN_TRUSTED, MAIN_EXFIL, vectors=more if by == "next" else VECTORS))
    world.gh.put(2, _pinned(world, two, MAIN_TRUSTED, [*MAIN_EXFIL, "z.lean"], vectors=more if by == "the PR" else VECTORS))
    raw = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", world.gh.prs[1].head,
                          world.gh.prs[2].head], cwd=world.dev, capture_output=True, text=True)
    assert raw.returncode == 1 and raw.stdout.splitlines()[1:] == [EXFIL], raw.stdout
    assert conflicting_files(world.dev, world.gh.prs[1].head, world.gh.prs[2].head) == []
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1, 2], entries[2].why
    assert f"research queue: #2 conflicts only in pins: {EXFIL} rebuilt from the merged tree" in world.logs
    pin = subprocess.run(["git", "show", f"refs/heads/next:{EXFIL}"], cwd=world.origin, check=True, capture_output=True,
                         text=True).stdout
    assert pin == _exfil([*MAIN_EXFIL, "z.lean"], {**one, **two, "x.lean": "x1\n"}, more)


def test_a_sync_admits_a_ready_pr_whose_merge_is_clean_and_runs_no_preflight(world, monkeypatch):
    """A ready mark admits a PR: its merge onto `next` is the only check before it joins (a conflict, the landing gates, a lock's
    drops), so a sync makes no `Preflight`, runs none, waits on none and keeps no record of one."""
    def ran(*args, **kwargs):
        raise AssertionError("the sync made or ran a preflight")

    monkeypatch.setattr(Q.Preflight, "__init__", ran)
    monkeypatch.setattr(Q.Preflight, "__call__", ran)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2] == world.on_next(q), world.logs
    assert [entries[n].state for n in (1, 2)] == ["queued", "queued"]
    assert not {"preflights", "preflight_waited"} & q.load_state().keys()
    assert not [line for line in world.logs if "preflight" in line], world.logs


def test_a_pr_of_drives_priority_list_and_those_it_contains_go_first_of_what_a_sync_merges_anew(world):
    """Drive's priority list (`Queue.priority`), the landing order: a listed PR and the PRs it contains go first of what a sync
    merges anew, a Lean change among them too, and with no preflight.  A listed number that is no open PR changes nothing."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, two := world.change({"pkg/A.lean": "-- 2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}, base=two))
    q = world.queue(first=(3, 99))
    assert [m.pr for m in q.sync()[0]] == [2, 3, 1] == world.on_next(q), world.logs


def test_drives_priority_list_is_read_again_at_each_sync_in_its_order_and_one_it_cant_read_is_empty_its_error_logged_once(world):
    """`--priority-file` (`Queue.priority_file`): each sync reads the file again, in its order, first to land first, a number
    repeated keeping its first place, and logs the list when it changes, its order too; no file is an empty list, and so is one
    it can't read, its error logged once while it stays the same, whichever tick's queue meets it."""
    path = world.tmp / "priority"

    def sync() -> tuple[int, ...]:
        q = world.queue()
        q.priority_file = path
        q.sync()
        return q.priority

    def said() -> list[str]:
        return [line for line in world.logs if "priority" in line]

    def listed(prs: str) -> str:
        return f"research queue: drive's priority list ({path}), first to land first: {prs}"

    assert sync() == () and said() == []
    path.write_text("12, 7\n3,")
    assert sync() == sync() == (12, 7, 3) and said() == [listed("#12 #7 #3")]
    path.write_text("7 12 7, 3 12")
    assert sync() == sync() == (7, 12, 3) and said()[1:] == [listed("#7 #12 #3")]
    path.write_text("12 #7")
    assert sync() == sync() == ()
    assert said()[2:] == [f"research queue: --priority-file {path} holds '#7', which is not a PR number: write PR numbers "
                          "separated by spaces or commas; drive's priority list is empty until the file is fixed"]
    path.unlink()
    path.mkdir()
    assert sync() == sync() == () and len(said()) == 4
    assert said()[3].startswith(f"research queue: --priority-file {path} can't be read (Is a directory): "), said()
    path.rmdir()
    assert sync() == () and len(said()) == 4
    path.write_text("5")
    assert sync() == (5,) and said()[4:] == [listed("#5")]
    path.unlink()
    assert sync() == () and said()[5:] == [listed("empty")]


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


def test_a_landing_gate_left_in_its_directory_still_judges(tmp_path, monkeypatch):
    """On drive's NFS TMPDIR the gate script just run is often left as a .nfsXXXX file, so removing its directory fails with
    ENOTEMPTY, and every tick crashed on it.  The gate's verdict stands and the directory is left for the sweep."""
    import errno
    import tempfile

    from research.queue import run_gate

    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    _git(tmp_path, "init", "-q", "-b", "main")
    for p, text in GATED.items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(text)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "gated")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "bad.txt").write_text("x\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "bad")
    rmdir = os.rmdir

    def nfs_rmdir(path, *a, **k):
        if "queue-gate-" in str(path):
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
        return rmdir(path, *a, **k)

    monkeypatch.setattr(os, "rmdir", nfs_rmdir)
    rc, out = run_gate(tmp_path, base, _git(tmp_path, "rev-parse", "HEAD"), "gates/g.py")
    assert rc == 1 and json.loads(out)["failures"][0]["why"] == "touches bad.txt", out


def test_a_merged_tree_whose_caller_names_no_root_goes_under_merged_root_not_tmpdir(tmp_path, monkeypatch):
    """`ported`'s codemod names no root: on drive's control pod its tree went to TMPDIR, a network volume where each worktree add
    and remove took minutes; --preflight-root (`MERGED_ROOT`) now takes it to the container disk like the preflight's."""
    import tempfile

    from research import queue

    repo, tmp, fast = tmp_path / "repo", tmp_path / "tmp", tmp_path / "fast"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setattr(queue, "MERGED_ROOT", fast)
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "f")
    with queue.merged_worktree(repo, _git(repo, "rev-parse", "HEAD")) as wt:
        assert wt.is_relative_to(fast) and (wt / "f.txt").read_text() == "x\n"
    assert not list(fast.glob(f"{queue.MERGED_PREFIX}*")) and not list(tmp.iterdir())


def test_a_pr_a_landing_gate_at_main_refuses_is_left_out_with_its_fix_and_only_the_members_a_sync_merges_are_judged(world, monkeypatch):
    """A landing gate judges a PR as the sync merges it onto `next`, never at landing, where ejecting a
    member low on `next` rebuilds it and cancels the Lean check running there.  A member the sync keeps isn't judged again; one
    it merges again after `main` moved is, and is left out once the gate at the new `main` refuses it."""
    from research import queue as Q

    main = _move_main(world, GATED)
    real, judged = Q.gate_refusal, {}

    def gate_refusal(repo, m, merge, gates, onto, kind):
        judged[_git(repo, "rev-parse", f"{merge}^2")] = merge            # each judged commit is a PR's merge onto `next`
        return real(repo, m, merge, gates, onto=onto, kind=kind)

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
    assert entries[5].state == "waiting" and entries[5].why == (
        f"a landing gate can't judge it yet: the landing gate gates/g.py can't judge its merge onto `next` "
        f"({judged[h5][:12]}) on `main` {main[:12]} (exit 2: gate: can't read crash.txt); retried next sync")
    assert {n: c["why"] for n, c in q.load_state()["conflicts"].items()} == {"2": entries[2].why}
    assert q.status()[2].why == entries[2].why + " (at the last sync)"

    judged.clear()
    world.gh.put(4, h4b := world.change({"d.txt": "4 fixed\n"}, base=h4))     # #1 is kept, #4 merged again at its new head
    members, _ = q.sync()
    assert [(m.pr, m.head) for m in members] == [(1, h1), (4, h4b)] and h1 not in judged and set(judged) == {h2, h4b, h5}

    main = _move_main(world, {"gates/banned.txt": "bad.txt\na.txt\n"})       # `main` moves: `next` is rebuilt on it
    members, entries = q.sync()
    assert [m.pr for m in members] == [4]
    assert entries[1].state == "refused" and entries[1].why == _gate_says(main, "touches a.txt", merged=True)


def test_a_pr_a_landing_gate_can_t_judge_waits_with_its_mark_and_merges_once_a_sync_s_gates_judge_it(world):
    """A gate that exits 2 says nothing against a PR (the unread gate does until a Lean landing's facts are published, about an
    hour after it), so the PR isn't left out as a refused one is, whose mark its owner gives again: it waits with its mark,
    logged once a head, and so does a PR that contains it, and a later sync whose gates judge it merges both."""
    _move_main(world, GATED)
    world.gh.put(1, h1 := world.change({"crash.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}, base=h1))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [3] == world.on_next(q)
    assert entries[1].state == "waiting" and entries[1].why.startswith(
        "a landing gate can't judge it yet: the landing gate gates/g.py can't judge its merge onto `next`")
    assert entries[1].why.endswith("(exit 2: gate: can't read crash.txt); retried next sync"), entries[1].why
    assert entries[2].state == "waiting" and entries[2].why == "contains #1, which didn't merge"
    state = q.load_state()
    assert state["conflicts"] == {} and set(state["waiting"]) == {"1", "2"} and "1" not in state.get("ejected", {})
    assert (q.status()[1].state, q.status()[1].why) == ("waiting", entries[1].why + " (at the last sync)")
    q.sync()
    assert sum(f"#1 at {h1[:8]} waits off `next`" in s for s in world.logs) == 1, "logged once a head"
    _move_main(world, {"gates/g.py": GATE.replace('if "crash.txt" in files:', "if False:")})
    members, _ = q.sync()
    assert [m.pr for m in members] == [1, 2, 3] == world.on_next(q) and q.load_state()["waiting"] == {}


def test_a_member_merged_again_onto_the_tree_it_was_judged_on_takes_its_verdict_and_main_moving_judges_it_again(
        world, monkeypatch):
    """A lift merges every member above the lifted one again, each running the landing gates (~15 s), though those above where
    it was have the same members below and so the same trees.  Their verdicts, a refusal too, are kept under `main` and the gates;
    a new `main` judges every merge again."""
    from research import queue as Q

    main = _move_main(world, GATED)
    real, judged = Q.gate_refusal, []

    def gate_refusal(repo, m, merge, gates, onto, kind):
        judged.append(int(_git(repo, "log", "-1", "--format=%s", merge).split("#")[1].split()[0]))
        return real(repo, m, merge, gates, onto=onto, kind=kind)

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
    assert world.logs[-1].endswith("kept 0, merged 4, merges reused for 3, gates' verdicts reused for 3"), world.logs[-1]
    judged.clear()
    _move_main(world, {"x.txt": "x\n"})
    q.lift = []
    assert [m.pr for m in q.sync()[0]] == [1, 2, 3, 4] and judged == [1, 2, 3, 4, 5]


def _merge_trees(monkeypatch) -> list[str]:
    """The head of each `git merge-tree` the queue runs, from now on."""
    real, heads = Q._git, []

    def git(repo, *args, **kw):
        if "merge-tree" in args:
            heads.append(args[-1])
        return real(repo, *args, **kw)

    monkeypatch.setattr(Q, "_git", git)
    return heads


def _trees(world: World, members) -> list[str]:
    return [_git(world.origin, "rev-parse", f"{m.commit}^{{tree}}") for m in members]


def test_a_member_merged_again_onto_the_same_tree_takes_its_recorded_merge_and_so_does_a_rebuild_on_a_moved_base(
        world, monkeypatch):
    """Each merge costs `git merge-tree` with the merge drivers, the port's codemod and the lock check: tens of seconds on
    drive's control pod, and a rebuild of `next` merges every member again.  A merge is recorded under the parent's tree (not
    its commit), the head, their merge bases and what every merge runs with, so a later sync (a new process) merging the same
    head onto the same tree runs no `git merge-tree` and no lock check, and neither does a rebuild on a `main` moved to a
    commit with the same tree; the merge commits hold the same trees, on the new base."""
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, h2 := world.change({"b.txt": "2\n"}))
    members, _ = world.queue().sync()
    calls, drops = _merge_trees(monkeypatch), []
    real = Q.lock_drops
    monkeypatch.setattr(Q, "lock_drops", lambda *a: drops.append(a[2]) or real(*a))
    _git(world.origin, "update-ref", "-d", "refs/heads/next")
    again, _ = world.queue().sync()
    assert calls == [] and drops == [] and _trees(world, again) == _trees(world, members)
    assert world.logs[-1].endswith("kept 0, merged 2, merges reused for 2"), world.logs[-1]
    main = _git(world.origin, "rev-parse", "refs/heads/main")
    moved = _git(world.origin, "commit-tree", f"{main}^{{tree}}", "-p", main, "-m", "main moved, its tree as it was")
    _git(world.origin, "update-ref", "refs/heads/main", moved)
    rebuilt, _ = world.queue().sync()
    assert [(m.pr, m.head) for m in rebuilt] == [(1, h1), (2, h2)] and calls == [] and drops == []
    assert _trees(world, rebuilt) == _trees(world, members) and _git(world.origin, "rev-parse", f"{rebuilt[0].commit}^1") == moved


def test_a_merge_is_made_again_when_its_head_its_parents_tree_or_the_drivers_change_and_judged_again_when_the_gates_do(
        world, monkeypatch):
    """A merge record stands only for what it read: a new head, a parent tree that changed (`main` moved under `next`) or a
    merge driver's command each merge again; a merge taken from its record with the gates changed is judged again, as the
    gates' verdicts are kept only under `main` and its gates."""
    from research import clone

    _move_main(world, {**GATED, "gates/h.py": "import sys\nsys.exit(0)\n"})
    real, judged = Q.gate_refusal, []
    monkeypatch.setattr(Q, "gate_refusal", lambda repo, m, merge, gates, onto, kind: judged.append(merge) or real(
        repo, m, merge, gates, onto=onto, kind=kind))
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.queue().sync()
    calls = _merge_trees(monkeypatch)
    world.gh.put(2, h2 := world.change({"b.txt": "2 again\n"}))
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2] and calls == [h2]
    calls.clear()
    _move_main(world, {"x.txt": "x\n"})
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2] and calls == [h1, h2]
    calls.clear()
    monkeypatch.setitem(clone.RECORD_DRIVER, "merge.lean-audit.driver", clone.RECORD_DRIVER["merge.lean-audit.driver"] + " ")
    _git(world.origin, "update-ref", "-d", "refs/heads/next")
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2] and calls == [h1, h2]
    calls.clear()
    judged.clear()
    _git(world.origin, "update-ref", "-d", "refs/heads/next")
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2] and calls == [] and judged == []
    monkeypatch.setattr(Q, "gates_at", lambda repo, ref, key="landing_gates": ("gates/g.py", "gates/h.py")
                        if key == "landing_gates" else ())
    _git(world.origin, "update-ref", "-d", "refs/heads/next")
    members, _ = world.queue().sync()
    assert [m.pr for m in members] == [1, 2] and calls == [] and judged == [m.commit for m in members]


def test_a_recorded_conflict_stays_a_conflict_onto_its_tree_and_the_member_merges_afresh_onto_another(world, monkeypatch):
    """A conflict is recorded as one, never as a clean merge: the PR left out is tried again above the rest each sync, and
    onto the same tree it is refused as before without `git merge-tree`; once the member it conflicts with leaves, its merge
    onto the new tree is made afresh, and it merges."""
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, h2 := world.change({"a.txt": "2\n"}))
    members, entries = world.queue().sync()
    why = "conflicts with `next` in a.txt (touched by #1)"
    assert [m.pr for m in members] == [1] and entries[2].why == why
    calls = _merge_trees(monkeypatch)
    again, entries = world.queue().sync()
    assert again == members and entries[2].state == "refused" and entries[2].why == why and calls == []
    assert world.logs[-1].endswith("kept 1, merged 0, merges reused for 1"), world.logs[-1]
    del world.gh.prs[1]
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [2] and calls == [h2]
    assert _git(world.origin, "show", "refs/heads/next:a.txt") == "2"


def test_a_merge_record_whose_tree_the_clone_lacks_is_merged_again_and_the_state_keeps_the_newest_records(world, monkeypatch):
    """A record names a tree in drive's clone, which a fresh clone or a prune drops: a record whose tree is gone merges again.
    The state keeps at most `MERGES` records, those the last syncs used."""
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, h2 := world.change({"b.txt": "2\n"}))
    q = world.queue()
    members, _ = q.sync()
    state = q.load_state()
    for rec in state["merges"]["of"].values():
        rec["tree"] = "0" * 40
    q.save_state(state)
    calls = _merge_trees(monkeypatch)
    _git(world.origin, "update-ref", "-d", "refs/heads/next")
    again, _ = world.queue().sync()
    assert calls == [h1, h2] and _trees(world, again) == _trees(world, members)
    monkeypatch.setattr(Q, "MERGES", 1)
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    world.queue().sync()
    assert len(q.load_state()["merges"]["of"]) == 1


#: the Markdown caps' warn gate and the caps it reads at `main`, committed into the origin as they are in this checkout
CAP_GATE = "tools/verity/check/gates/markdown_caps.py"
CAP_FILES = {DRIVE_FLAGS: f'warn_gates = ["{CAP_GATE}"]\n', **{f: (Path(__file__).resolve().parents[3] / f).read_text()
                                                                for f in (CAP_GATE, "tests/test_repository.py")}}
#: `doc.md` as `main` holds it (70 bytes), and with line ``at`` replaced by ``text``
LINES = [f"line {i}\n" for i in range(10)]


def _doc(at: int | None = None, text: str = "") -> str:
    return "".join(text if i == at else line for i, line in enumerate(LINES))


def test_a_pr_whose_merge_onto_next_puts_a_markdown_file_over_its_cap_joins_next_and_its_warning_is_kept_for_drive(world):
    """A PR whose head alone is under a Markdown file's cap can take the file over it once merged onto `next` above the members
    that also grow it.  The caps are a warn gate: the sync gives it each PR's merge onto `next`, so the bytes the members below
    add count, keeps what it says by PR and head (the state's `warned`, which drive tells the owner) and leaves no PR out."""
    main = _move_main(world, {**CAP_FILES, "doc.md": _doc()})
    world.gh.put(1, h1 := world.change({"doc.md": _doc(1, "x" * 30000 + "\n")}))            # 30,064 bytes: under 49,152
    world.gh.put(2, h2 := world.change({"doc.md": _doc(8, "y" * 30000 + "\n")}))            # as much alone, 60,058 onto #1
    world.gh.put(3, h3 := world.change({"doc.md": _doc(5, "short\n")}))                     # 60,057 onto #2: over, but shrinks it
    for h in (h1, h2, h3):                                                                  # each head alone passes
        assert subprocess.run([sys.executable, CAP_GATE, main, h], cwd=world.dev, capture_output=True).returncode == 0
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2, 3] == world.on_next(q) and q.load_state()["conflicts"] == {}
    (said,) = [w["why"] for n, w in q.load_state()["warned"].items() if (n, w["head"]) == ("2", h2)]
    assert said == (f"the warn gate {CAP_GATE} at `main` {main[:12]} on its merge onto `next`: doc.md is 60058 bytes at "
                    f"{members[1].commit[:12]}, over its 49152-byte cap (tests/test_repository.py), and was 30064 bytes at "
                    f"{members[0].commit[:12]} below; trim doc.md to 49152 bytes or fewer (10906 over): move detail to the notes "
                    f"store (cite it by note:<id>) (`python3 {CAP_GATE} origin/main HEAD` on the branch merged with `origin/next` "
                    "runs it)"), said
    assert int(_git(world.origin, "cat-file", "-s", "refs/heads/next:doc.md")) == 60057

    again, _ = q.sync()                                          # the members stay, and so does the warning at #2's head
    assert [m.commit for m in again] == [m.commit for m in members] and list(q.load_state()["warned"]) == ["2"]
    world.gh.put(2, world.change({"doc.md": _doc(8, "y\n")}, base=h2))                     # a new head drops it
    q.sync()
    assert q.load_state()["warned"] == {}


def test_a_file_already_over_its_cap_below_warns_only_of_a_pr_whose_merge_grows_it(world, monkeypatch):
    """The gate judges only what a merge adds, against the tip it merged onto.  A member kept on `next` that is
    already over a cap (merged before the gate was listed) mustn't lay a warning on every PR above it: one that doesn't touch
    the file merges, one that shrinks it merges (the gate notes it), and one that grows it merges with a warning."""
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
    assert [m.pr for m in members] == [1, 2, 3, 4] == world.on_next(q) and members[0].commit == kept.commit
    warned = q.load_state()["warned"]
    assert list(warned) == ["3"] and "doc.md is 60068 bytes at " in warned["3"]["why"]
    assert f"and was 60064 bytes at {members[1].commit[:12]} below" in warned["3"]["why"]
    merge = members[3].commit
    doc = json.loads(subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / CAP_GATE), "origin/main", merge, "--json"], cwd=q.workdir,
                                    capture_output=True, text=True).stdout)
    assert doc["below"] == members[2].commit and doc["failures"] == [] and doc["notes"] == [
        f"doc.md is 60063 bytes, over its 49152-byte cap, but was 60068 bytes at {members[2].commit[:12]} below: not this change's"]


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


def test_an_older_head_two_prs_were_seen_at_is_carried_by_the_marked_one(world):
    """#6 was opened on #5's branch at its head `old`; both move on, #5 marked and #6 not.  #7, branched at `old`, holds
    commits #5's marked head covers, so it goes after #5 and waits for nothing of #6's."""
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old, ready=False)
    world.gh.put(6, old, ready=False)
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))
    q = world.queue()
    q.sync()                                                                    # the queue sees #5 and #6 at `old`
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old))               # #5 moves on from `old`, marked
    world.gh.put(6, world.change({"s.txt": "soundness\n"}, base=old), ready=False)
    members, entries = q.sync()
    assert entries[7].after == entries[7].older == (5,), entries[7].why
    assert [m.pr for m in members] == [5, 7]


def _carried_on_next(world: World) -> tuple[Queue, str]:
    """#5 at `old`, unmarked; #6, #5's own lane's rebuild on `old` (so it carries #5's commits), marked and merged onto `next`;
    then #5 moves on, still unmarked.  `old` is an older head of #5's that only #6 brings to `next`."""
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old, ready=False)
    world.gh.put(6, world.change({"r.txt": "rebuilt\n"}, base=old), body="Lane `lane-5`.")
    q = world.queue()
    assert [m.pr for m in q.sync()[0]] == [6]                                  # the queue sees #5 at `old`
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old), ready=False)
    return q, old


def test_a_pr_holding_an_older_head_that_admitted_members_already_bring_waits_for_nothing(world):
    """#7 holds `old`, once #5's head, but not #6's head, which brings `old` to `next`: #7 lands nothing of #5's that `next`
    doesn't already land, so it doesn't wait for unmarked #5.  Once #6 isn't admitted, #6 carries nothing and #7 waits again."""
    q, old = _carried_on_next(world)
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))             # #7 branched from #5 at `old`, beside #6
    members, entries = q.sync()
    assert [m.pr for m in members] == [6, 7] and entries[7].after == entries[7].older == (), entries[7].why
    world.gh.ready.discard((6, world.gh.prs[6].head))                           # #6's mark withdrawn: it leaves `next`
    members, entries = q.sync()
    assert [m.pr for m in members] == [] and entries[7].older == (5,)
    assert entries[7].why == "contains #5's commits (at an older head), which isn't admitted"


def test_a_pr_holding_an_older_head_no_member_brings_still_waits(world):
    """As above, but #7 branched from #5 at `mid`, a later head the queue saw #5 at that #6 doesn't bring: #7 waits for #5 at
    `mid`, though `next` brings the older `old`."""
    q, old = _carried_on_next(world)
    world.gh.put(5, mid := world.change({"x.txt": "x = 2\n"}, base=old), ready=False)
    q.sync()                                                                    # the queue sees #5 at `mid`
    world.gh.put(5, world.change({"x.txt": "x = 3\n"}, base=mid), ready=False)
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=mid))
    members, entries = q.sync()
    assert [m.pr for m in members] == [6] and entries[7].older == (5,) and entries[7].held == {5: mid}
    assert entries[7].why == "contains #5's commits (at an older head), which isn't admitted"


def test_a_member_of_next_doesnt_excuse_its_own_commits(world):
    """#7 went onto `next` while #5's head `old`, which #7 holds, was marked; #5 moves on unmarked and its mark at `old` goes
    (so it doesn't stay at `old`, `Queue._pending`).  `next` holds `old` only through #7's own merge (and #5's merge at the head
    it left), so #7 waits for #5."""
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old)
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))
    q = world.queue()
    assert [m.pr for m in q.sync()[0]] == [5, 7]
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old), ready=False)
    world.gh.ready.discard((5, old))
    members, entries = q.sync()
    assert [m.pr for m in members] == [] and entries[7].after == entries[7].older == (5,)
    assert entries[7].why == "contains #5's commits (at an older head), which isn't admitted"


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


def test_ready_marks_a_pr_based_on_another_prs_branch_and_says_the_queue_wont_see_it(world, capsys):
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    parent = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, parent, ready=False)
    _based_on(world, 5, "main", "lane-5/parent")
    world.gh.put(6, child := world.change({"c.txt": "stacked\n"}, base=parent), ready=False)
    _based_on(world, 6, "lane-5/parent")
    assert ready(world.gh, world.store, 6, "lane-6", MarkOnly()) == 0
    note = ("#6 is based on lane-5/parent (#5's head), not main: the verified queue won't see it until #5 lands, one stack level "
            "per landing. Retarget it to main (the queue orders stacked PRs itself: a PR containing another waits for it)")
    assert note in capsys.readouterr().err
    assert StoreLabels(world.store, log=world.logs.append).ready(6, child) is True

    _based_on(world, 6, "some/branch")
    assert ready(world.gh, world.store, 6, "lane-6", MarkOnly(), as_json=True) == 0
    out = capsys.readouterr()
    got = json.loads(out.out)
    assert got["pr"] == 6 and got["head"] == child and got["base"] == "some/branch" and got["exit"] == 0
    assert got["warning"].startswith("#6 is based on some/branch, not main: the verified queue won't see it until it is. Retarget "
                                     "it to main")
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


def test_the_sync_leaves_off_next_a_head_that_merged_next_carrying_a_pr_head_its_pr_no_longer_lands(world):
    """A mark written as a label skips the preflight's (g), so the sync judges such a head itself: one that merged `next`
    lands after the PRs it carries from there while they land as they stand, and is refused, with the fix, once one of them
    is rebuilt without the carried head, or withdrawn by its owner, which carries it only for a rebuild."""
    world.gh.put(5, five := world.change({"x.txt": "x\n"}))
    world.gh.put(6, six := world.change({"y.txt": "y\n"}))
    q = world.queue()
    assert [m.pr for m in q.sync()[0]] == [5, 6]
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    by = {n: _git(world.origin, "log", "-1", "--format=%H", f"--grep=^Queue: #{n} ", tip) for n in (5, 6)}
    world.gh.put(4, world.change({"z.txt": "z\n"}, base=tip))
    assert [m.pr for m in q.sync()[0]] == [5, 6, 4]
    world.gh.put(5, new := world.change({"x.txt": "x2\n"}))
    members, entries = q.sync()
    assert sorted(m.pr for m in members) == [5, 6] and entries[4].state == "refused"
    assert entries[4].why == (f"it carries #5's head {five[:12]} through `next`'s merge {by[5][:12]}, which neither `main` nor "
                              f"#5's current head {new[:12]} holds: merge origin/main and #5's current head instead of `next`, "
                              "or drop #5's commits")
    world.gh.put(5, five)
    world.gh.ready.discard((6, six))
    q.marked_by = lambda n, h: "lane-6" if n == 6 else None                 # #6's owner wrote its `ready = false`
    members, entries = q.sync()
    assert [m.pr for m in members] == [5] and entries[4].why == (
        f"it carries #6's head {six[:12]} through `next`'s merge {by[6][:12]}, and #6 isn't admitted at {six[:12]} (not marked "
        f"ready at {six[:8]}): merge origin/main and #6's current head instead of `next`, or drop #6's commits")


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


def test_the_clone_keeps_the_merge_drivers_side_files_of_a_reader_that_let_the_lock_go(world):
    """`status` and a listing `review` let the queue's lock go once they've fetched, and go on merging in the clone, where git's
    merge drivers read their sides from `.merge_file_*` at its root.  A command that takes the lock meanwhile cleans the clone:
    it keeps those, and removes everything else untracked."""
    q = world.queue()
    q._clone()
    side, stray = q.workdir / ".merge_file_Ab12Cd", q.workdir / "stray.txt"
    side.write_text("a side\n")
    stray.write_text("left over\n")
    world.queue()._clone()
    assert side.read_text() == "a side\n" and not stray.exists()


def _queue_at(world: World, workdir: Path, remote: str) -> Queue:
    return Queue(workdir, remote=remote, github=world.gh, ready=lambda n, h: True, store=world.store, log=world.logs.append)


def _bare(world: World, name: str) -> Path:
    """Another repository than `world`'s origin, holding its `main`."""
    path = world.tmp / name
    _git(world.tmp, "init", "-q", "--bare", "-b", "main", str(path))
    _git(world.dev, "push", "-q", str(path), "main")
    return path


def test_repo_alone_takes_its_github_remote_and_a_clone_of_its_own_and_a_remote_of_another_repository_is_refused(
        world, monkeypatch, capsys):
    """`--remote` defaulted to the caller's origin whatever `--repo` said, so `status --repo` of another repository than the
    caller's read that one's PRs against the caller's pull refs, each waiting forever on a head the two disagreed on.
    `--repo OWNER/NAME` alone takes https://github.com/OWNER/NAME.git and ~/.research/queue/NAME; with neither, the caller's
    origin and ~/.research/queue/verity stay; a `--remote` naming another repository on GitHub is refused before any clone."""
    from research.queue import remote_and_repo

    assert remote_and_repo(None, "o/infra", world.dev) == ("https://github.com/o/infra.git", "o/infra")
    assert remote_and_repo(str(world.origin), "o/infra", world.dev) == (str(world.origin), "o/infra")     # a path: no OWNER/NAME
    with pytest.raises(QueueError, match=re.escape(f"can't tell OWNER/NAME from '{world.origin}': pass --repo")):
        remote_and_repo(None, None, world.dev)
    for same in ("https://github.com/o/infra", "https://u@github.com/O/Infra.git", "git@github.com:o/infra.git",
                 "ssh://git@github.com/o/infra.git/"):
        assert remote_and_repo(same, "o/infra", world.dev) == (same, "o/infra")
    for other, shown in (("https://x-access-token:" "dummy@github.com/o/verity.git", "https://github.com/o/verity.git"),
                         ("git@github.com:o/verity", "git@github.com:o/verity")):
        with pytest.raises(QueueError) as e:
            remote_and_repo(other, "o/infra", world.dev)
        assert str(e.value) == (f"--remote {shown} is o/verity on GitHub, not --repo o/infra: pass --repo alone (its remote is "
                                "https://github.com/o/infra.git), or a --remote of o/infra")
    with pytest.raises(QueueError, match="--repo 'https://github.com/o/infra': OWNER/NAME on GitHub"):
        remote_and_repo(None, "https://github.com/o/infra", world.dev)

    _git(world.dev, "remote", "set-url", "origin", "https://verity-agent@github.com/o/verity")
    monkeypatch.chdir(world.dev)
    monkeypatch.setattr(Q, "WORKDIR", world.tmp / "home" / ".research" / "queue" / "verity")
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    cloned = []

    def clone(q: Queue) -> None:
        cloned.append((q.workdir, q.remote))
        raise QueueError("cloned nothing")

    monkeypatch.setattr(Queue, "_clone", clone)
    store = ["--store", str(world.tmp / "qs")]
    assert research_main(["queue", "status", "--repo", "o/infra", *store]) == 1
    assert research_main(["queue", "status", *store]) == 1
    assert cloned == [(world.tmp / "home" / ".research" / "queue" / "infra", "https://github.com/o/infra.git"),
                      (Q.WORKDIR, "https://verity-agent@github.com/o/verity")]
    capsys.readouterr()
    assert research_main(["queue", "status", "--repo", "o/infra", "--remote", "git@github.com:o/verity.git", *store]) == 1
    assert "research queue: --remote git@github.com:o/verity.git is o/verity on GitHub, not --repo o/infra" in capsys.readouterr().err
    assert len(cloned) == 2


def test_the_queue_never_repoints_the_origin_of_a_checkout_it_did_not_clone(world, monkeypatch, capsys):
    """`_clone` set the workdir's origin to the queue's remote whatever the workdir held: pointed with `--workdir` at a clone of
    another repository, it made that clone's origin the caller's, and a later `git push` there went to the wrong repository.
    A checkout the queue didn't clone keeps its origin: one of another repository is refused before anything in it changes,
    and one of the remote's repository, in another URL's form, is used as it is."""
    infra = _bare(world, "infra.git")
    theirs = world.tmp / "infra-clone"
    _git(world.tmp, "clone", "-q", str(infra), str(theirs))
    (theirs / "notes.txt").write_text("not committed\n")
    with pytest.raises(QueueError) as e:
        _queue_at(world, theirs, str(world.origin)).status()
    assert str(e.value) == (f"{theirs} is a checkout the queue didn't clone (its git config has no research.queueclone), whose "
                            f"origin, {infra}, is another repository than the queue's remote {world.origin}, and the queue never "
                            f"repoints such a checkout: pass --workdir a directory of its own, where the queue clones "
                            f"{world.origin}, or --remote {infra}, the repository that checkout is a clone of")
    monkeypatch.chdir(world.dev)
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    argv = ["queue", "status", "--workdir", str(theirs), "--remote", str(world.origin), "--repo", "o/verity",
            "--store", str(world.tmp / "qs")]
    assert research_main(argv) == 1
    assert f"research queue: {e.value}\n" in capsys.readouterr().err
    assert _git(theirs, "remote", "get-url", "origin") == str(infra) and (theirs / "notes.txt").read_text() == "not committed\n"
    drivers = subprocess.run(["git", "config", "--local", "--get-regexp", r"^merge\."], cwd=theirs, capture_output=True, text=True)
    assert drivers.stdout == "", "nothing in it changed: not even the record merge drivers"

    mine = world.tmp / "verity-clone"
    _git(world.tmp, "clone", "-q", str(world.origin), str(mine))
    _git(mine, "remote", "set-url", "origin", "https://verity-agent@github.com/o/verity")
    for remote in ("git@github.com:o/verity.git", "https://github.com/O/verity"):
        _queue_at(world, mine, remote)._clone()
        assert _git(mine, "remote", "get-url", "origin") == "https://verity-agent@github.com/o/verity"
    with pytest.raises(QueueError, match=f"^{re.escape(str(mine))} is a checkout the queue didn't clone .* is another repository "
                                         r"than the queue's remote https://github\.com/o/infra\.git, and"):
        _queue_at(world, mine, "https://x-access-token:" "dummy@github.com/o/infra.git")._clone()


def test_a_clone_the_queue_made_follows_the_remote_marked_copied_or_at_the_default_workdir(world, monkeypatch):
    """What drive relies on: its clone's origin follows the remote it is given, as the URL changes.  The queue marks each clone
    it makes in that clone's git config, a copy keeps the mark (drive seeds its clone with `cp -a`), and a clone at the
    default workdir is the queue's whether marked or not, since it made those before it marked any."""
    q = world.queue()
    q._clone()
    assert _git(q.workdir, "config", "--local", "--get", "research.queueclone") == "true"
    other = _bare(world, "other.git")
    copy = world.tmp / "copy"
    shutil.copytree(q.workdir, copy, symlinks=True)
    _queue_at(world, copy, str(other))._clone()
    assert _git(copy, "remote", "get-url", "origin") == str(other)

    monkeypatch.setattr(Q, "WORKDIR", world.tmp / "home" / ".research" / "queue" / "verity")
    _git(world.tmp, "clone", "-q", str(other), str(Q.WORKDIR))
    _queue_at(world, Q.WORKDIR, str(world.origin))._clone()
    assert _git(Q.WORKDIR, "remote", "get-url", "origin") == str(world.origin)


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
    """One listing of the store's remote per open PR, in turn, would take seconds of every drive tick; and one GET at a time
    took a fresh VM's listing four minutes for `next`'s ~1,000 labels, so the listings and the GETs each run
    `PREFETCH_JOBS` at once."""
    from research.queue import PREFETCH_JOBS, StoreLabels

    pools: list[int] = []
    pool = labels_sync.ThreadPoolExecutor
    monkeypatch.setattr(labels_sync, "ThreadPoolExecutor", lambda max_workers: pools.append(max_workers) or pool(max_workers))
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
    for n in (1, 2):
        lane.add_label(label_target(n, heads[n]), vocab.REVIEWED, "land", "ci", ts="2026-10-12T08:00:00Z")
    assert labels_sync.push_targets(lane, [label_target(n, heads[n]) for n in (1, 2)])["failed"] == []
    pools.clear()
    labels = StoreLabels(LocalStore(tmp_path / "q1"))
    labels.prefetch(list(heads.items()))
    assert [labels.ready(n, heads[n]) for n in (1, 2, 3)] == [True, False, False]
    assert [len(labels.reviews(n, heads[n])) for n in (1, 2, 3)] == [1, 1, 0]
    assert pulls == [(3, PREFETCH_JOBS)] and pools == [3, 4]          # the listings, then the four labels' GETs

    pulls.clear()
    monkeypatch.setattr(labels_sync, "remote_labels_for", lambda r, t, f=labels_sync.remote_labels_for:
                        (_ for _ in ()).throw(OSError("listing refused")) if not pulls[1:] else f(r, t))
    labels = StoreLabels(LocalStore(tmp_path / "q2"))
    labels.prefetch(list(heads.items()))
    assert [labels.ready(n, heads[n]) for n in (1, 2, 3)] == [True, False, False]
    assert pulls == [(3, PREFETCH_JOBS), (1, 1), (1, 1), (1, 1)]


def test_admission_prefetches_the_marks_of_every_pr_it_will_ask_of(world):
    asked: list[list[tuple[int, str]]] = []
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}), labels=("hold",))
    world.gh.put(3, h3 := world.change({"c.txt": "3\n"}))
    world.queue(prefetch=asked.append).sync()
    assert asked == [[(1, h1), (3, h3)]]


def test_above_is_each_heads_rev_list_and_first_parent_line(world):
    """`entries` reads what each head has above the base from one walk, not two `rev-list`s a PR."""
    from research.queue import above

    a = world.change({"a.txt": "1\n"})
    b = world.change({"b.txt": "2\n"}, a)
    c = world.change({"c.txt": "3\n"})
    c2 = world.change({"e.txt": "5\n"}, c)
    _git(world.dev, "checkout", "-q", "--detach", b)
    _git(world.dev, "merge", "-q", "--no-edit", c2)
    m = _git(world.dev, "rev-parse", "HEAD")
    d = world.change({"d.txt": "4\n"}, m)
    main = _git(world.dev, "rev-parse", "origin/main")
    heads = [a, b, c, c2, m, d, main]
    got = above(world.dev, "origin/main", heads)
    for h in heads:
        assert got[h] == (_git(world.dev, "rev-list", "--topo-order", f"origin/main..{h}").split(),
                          set(_git(world.dev, "rev-list", "--first-parent", f"origin/main..{h}").split())), h
    assert got[d][1] == {d, m, b, a} and got[main] == ([], set())


def test_status_merges_each_head_onto_main_once_until_main_or_the_clones_head_moves(world, monkeypatch):
    """A merge of an old head onto `main` takes about a second, and `status` merged every admitted PR not on `next` each call."""
    import research.queue as Qm

    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    world.gh.put(2, h2 := world.change({"b.txt": "2\n"}))
    merged: list[str] = []
    real = Qm.conflicting_files
    monkeypatch.setattr(Qm, "conflicting_files", lambda repo, base, head: merged.append(head) or real(repo, base, head))
    q = world.queue()
    assert {n: e.state for n, e in q.status().items()} == {1: "ready", 2: "ready"} and sorted(merged) == sorted([h1, h2])
    merged.clear()
    assert {n: e.state for n, e in q.status().items()} == {1: "ready", 2: "ready"} and merged == []
    _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'main\n'})}:refs/heads/main")
    for again in (False, True):
        merged.clear()
        st = q.status()
        assert st[1].state == "refused" and st[1].why.startswith("conflicts with `main` in a.txt") and st[2].state == "ready"
        assert sorted(merged) == ([] if again else sorted([h1, h2]))


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
    assert got[2][0] == "ready" and "the verified queue merges it into a candidate on `main`" in got[2][1]
    assert research_main([*argv, "--marked-after", "2026-10-01T00:00:00Z"]) == 0
    assert {e["pr"]: e["state"] for e in json.loads(capsys.readouterr().out)} == {1: "ready", 2: "ready"}


def test_status_lists_every_open_pr_whose_body_names_no_linear_issue(world, monkeypatch, capsys):
    """A key after a Linear magic word names an issue whatever its team, a bare key only for a team main's registry gives a
    handle (so `SHA-512` names none), and an `Issue: none:` line names none on purpose; an unmarked PR on another branch,
    which has no entry, is listed too."""
    import research.queue as Qm

    cfg = world.tmp / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{world.tmp / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    reg = world.dev / Qm.REGISTRY
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"handles": [{"handle": "ci", "linear": "CI"}, {"handle": "infra"}]}))
    _git(world.dev, "add", "-A")
    _git(world.dev, "commit", "-qm", "the registry")
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    bodies = {1: "Lane a\n\nRefs ENG-5\n", 2: "Lane b\n\nCI-12 (the thing)\n", 3: "Lane c\n\nSHA-512 everywhere, as INF-3\n",
              4: "Lane d\n\nIssue: none: docs only\n", 5: "Lane e\n\npart of nothing\n"}
    for n, body in bodies.items():
        world.gh.put(n, world.change({f"{n}.txt": "x\n"}), body=body, ready=False)
    world.gh.prs[5] = dataclasses.replace(world.gh.prs[5], base="feature")
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    argv = ["queue", "status", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo", "o/verity",
            "--store", str(world.tmp / "queue-store"), "--marked-after", "2026-10-01T00:00:00Z"]
    assert research_main(argv) == 0
    section = capsys.readouterr().out.split("\n\nno Linear issue named ")[1].splitlines()
    assert section[0].startswith("(2 of 5 open PRs; add a `Refs KEY` or `Issue: none: REASON` line")
    assert [r.split()[0] for r in section[1:]] == ["#3", "#5"] and section[1].split()[1:] == ["c", "change", "3"]
    assert research_main([*argv, "--json"]) == 0
    assert {e["pr"]: e["issue"] for e in json.loads(capsys.readouterr().out)} == {1: "ENG-5", 2: "CI-12", 3: None,
                                                                                  4: "none: docs only"}


def test_the_status_command_says_when_a_move_lands_on_its_mark_and_refuses_one_that_conflicts_with_main(world):
    """`status --loop` refuses a move PR that doesn't merge onto `main`, and says of one that does that it lands with a land
    verdict, or on its mark alone once the loop certifies its move on `main` and drive.toml allows it."""
    base = _git(world.origin, "rev-parse", "main")
    _move_main(world, {"a.txt": "main\n"})
    world.gh.put(1, world.change({"b.txt": "1\n"}))
    for n, files, at in ((2, {"c.txt": None, "moved/c.txt": "c\n"}, "origin/main"), (3, {"a.txt": "3\n"}, base)):
        world.gh.put(n, world.change(files, base=at), body=f"Lane `lane-{n}`.\nMove: tools/move/x_map.toml\n")
    loop = world.queue(land_into="checked").status(loop=True)
    assert [loop[n].state for n in (1, 2, 3)] == ["ready", "ready", "refused"], loop
    assert loop[1].why.endswith(", or on its mark alone when it is low-risk"), loop[1]
    assert loop[2].why.endswith(", or on its mark alone once its move is certified on `main`, as drive.toml allows"), loop[2]
    assert loop[3].why == "conflicts with `main` in a.txt: merge `main` into the branch and re-mark it", loop[3]


def test_status_and_the_review_listing_free_the_queue_lock_before_they_read_labels(world, monkeypatch, capsys):
    """The lock refuses a second queue command at once, and a fresh VM's listing read labels for minutes under it, so every
    verdict meanwhile was refused.  A command that only reads holds it while it clones and fetches, not after."""
    import research.queue as Qm

    cfg = world.tmp / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{world.tmp / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    world.gh.put(1, world.change({"1.txt": "x\n"}), ready=False)
    workdir = world.tmp / "q" / "verity"
    free: list[bool] = []

    def prefetch(self, heads) -> None:
        try:
            Qm._lock(workdir).close()
            free.append(True)
        except QueueError:
            free.append(False)

    monkeypatch.setattr(Qm.StoreLabels, "prefetch", prefetch)
    common = ["--workdir", str(workdir), "--remote", str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "qs")]
    assert research_main(["queue", "status", "--json", *common]) == 0 and free == [True]
    assert research_main(["queue", "review", *common]) == 0 and free == [True, True]
    capsys.readouterr()


def test_the_review_listing_names_the_marked_prs_onto_main_not_on_it_lacking_a_land_verdict_oldest_mark_first(world, capsys):
    """The verified queue builds from the marked PRs, not from `next`'s members, so the listing reviewers work from is those:
    each open PR onto `main` marked at its head, not on `main` yet, with no land verdict there by a non-owner, oldest mark first."""
    from research.queue import StoreLabels

    world.with_store_remote()
    world.gh.put(1, world.change({"1.txt": "x\n"}))
    world.gh.put(2, world.change({"2.txt": "x\n"}))
    world.gh.put(3, world.change({"3.txt": "x\n"}), ready=False)
    world.gh.put(4, on := world.change({"4.txt": "x\n"}))
    world.gh.put(5, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(6, world.change({"6.txt": "x\n"}))
    world.gh.prs[6] = dataclasses.replace(world.gh.prs[6], base="checked")
    _git(world.origin, "update-ref", "refs/heads/main", on)
    world.store.add_label(label_target(2, world.gh.prs[2].head), vocab.REVIEWED, "land", "reviewer", ts="2026-11-09T12:00:00Z")
    world.store.add_label(label_target(1, world.gh.prs[1].head), vocab.REVIEWED, "land", "lane-1", ts="2026-11-09T12:00:00Z")
    marks = {n: f"2026-11-09T{h:02d}:00:00Z" for n, h in {1: 10, 2: 8, 4: 7, 5: 9, 6: 6}.items()}
    q = world.queue(marked_at=lambda n, h: marks.get(n))
    labels = StoreLabels(world.store, log=lambda s: None)
    assert Q.review(q, world.store, labels, None, None, None, None, as_json=True) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["pr"], r["marked"], r["why"]) for r in rows] == [
        (5, marks[5], "no land verdict"), (1, marks[1], "no land verdict but its owner's own (lane-1), which doesn't count")]
    assert (rows[0]["lean"], rows[0]["files"], rows[1]["lean"], rows[1]["files"]) == (True, ["pkg/lean-audit.json"], False, ["1.txt"])
    assert Q.review(q, world.store, labels, None, None, None, None) == 0
    out = capsys.readouterr().out
    assert out.index("#5 ") < out.index("#1 ") and "2 marked PR(s) not on `main` lack a land verdict" in out, out


def test_the_review_listing_says_low_risk_of_a_head_the_loops_rule_lands_on_its_mark_and_never_of_a_move_or_held_pr(world, capsys):
    """The verified queue's loop takes a marked head on its mark alone when drive.toml at `main` excludes none of its files
    (`lands_on_mark`), so its row says so instead of waiting on a reviewer: one whose owner's own land verdict doesn't count
    too, but not a head touching an excluded path, a move PR or a held PR, nor any once `main` turns the lane off.  A head a
    reviewer's land verdict lands is no row."""
    from research.queue import StoreLabels

    world.with_store_remote()
    _move_main(world, {DRIVE_FLAGS: 'lowrisk_excludes = ["*.lean", "core/"]\n'})
    world.gh.put(1, world.change({"docs/a.md": "x\n"}))
    world.gh.put(2, world.change({"docs/b.md": "x\n", "core/v.py": "x\n"}))
    world.gh.put(3, world.change({"docs/c.md": "x\n", "pkg/T.lean": "x\n"}))
    world.gh.put(4, world.change({"c.txt": None, "moved/c.txt": "c\n"}), body="Lane `lane-4`.\nMove: tools/move/x_map.toml\n")
    for n in (5, 6, 7):
        world.gh.put(n, world.change({f"docs/{n}.md": "x\n"}))
    for n, verdict, by in ((5, "hold", "reviewer"), (6, "land", "lane-6"), (7, "land", "reviewer")):
        world.store.add_label(label_target(n, world.gh.prs[n].head), vocab.REVIEWED, verdict, by,
                              ref="wait" if verdict == "hold" else None, ts="2026-11-09T12:00:00Z")
    q, labels = world.queue(), StoreLabels(world.store, log=lambda s: None)
    assert Q.review(q, world.store, labels, None, None, None, None, as_json=True) == 0
    rows = {r["pr"]: r for r in json.loads(capsys.readouterr().out)}
    assert {n: r["why"] for n, r in rows.items()} == {
        1: "low-risk: lands on its mark", 2: "no land verdict", 3: "no land verdict", 4: "no land verdict",
        5: "held by reviewer (wait)", 6: "low-risk: lands on its mark"}, rows
    assert rows[1]["files"] == ["docs/a.md"] and rows[4]["files"] == ["c.txt", "moved/c.txt"], rows
    assert Q.review(q, world.store, labels, None, None, None, None) == 0
    out = capsys.readouterr().out
    assert f"#1 {world.gh.prs[1].head[:12]} lane-1 Lean change: no, marked ?: low-risk: lands on its mark\n" in out, out
    _move_main(world, {DRIVE_FLAGS: 'lowrisk_lane = false\nlowrisk_excludes = ["*.lean", "core/"]\n'})
    assert Q.review(q, world.store, labels, None, None, None, None, as_json=True) == 0
    assert {r["pr"]: r["why"] for r in json.loads(capsys.readouterr().out)}[1] == "no land verdict"


def test_review_records_a_verdict_at_an_open_prs_head_and_refuses_its_owner_and_its_ready_marks_author_whatever_the_case(
        world):
    """`research queue review N`: a PR the queue's branch doesn't hold takes the verdict at its head on GitHub, or at the head
    ``sha`` names, as a `reviewed` label with its ref; its owning lane and the author of its ready mark write none, and a PR
    neither held nor open takes none."""
    from research.queue import StoreLabels

    world.with_store_remote()
    world.gh.put(1, old := world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.store.add_label(label_target(2, world.gh.prs[2].head), vocab.READY, "true", "marker")
    q, labels = world.queue(), StoreLabels(world.store, log=lambda s: None)
    assert Q.review(q, world.store, labels, 1, "land", "note:ci/x", "lane-9") == 0
    (lab,) = world.store.labels(label_target(1, old), key=vocab.REVIEWED)
    assert (lab.value, lab.by, lab.ref) == ("land", "lane-9", "note:ci/x")
    world.gh.put(1, world.change({"a.txt": "2\n"}))
    assert Q.review(q, world.store, labels, 1, "hold", "read the old head", "lane-9", sha=old) == 0
    assert [x.value for x in world.store.labels(label_target(1, old), key=vocab.REVIEWED)] == ["land", "hold"]
    for pr, by in ((1, "Lane-1"), (2, "MARKER")):
        with pytest.raises(QueueError, match=f"#{pr} is {by}'s own PR"):
            Q.review(q, world.store, labels, pr, "land", None, by)
    assert world.store.labels(label_target(2, world.gh.prs[2].head), key=vocab.REVIEWED) == []
    with pytest.raises(QueueError, match="#9 is neither on `next` nor open"):
        Q.review(q, world.store, labels, 9, "land", None, "lane-1")


def test_review_without_sha_refuses_a_head_pushed_since_the_sync_and_takes_a_short_or_full_sha(world):
    from research.queue import StoreLabels

    world.with_store_remote()
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q = world.queue()
    q.sync()
    old = world.gh.prs[1].head
    world.gh.put(1, new := world.change({"a.txt": "2\n"}))
    labels = StoreLabels(world.store, log=lambda s: None)
    with pytest.raises(QueueError, match=f"GitHub has its head at {new[:12]}.*pass --sha"):
        Q.review(q, world.store, labels, 1, "land", None, "lane-9")
    assert Q.review(q, world.store, labels, 1, "land", None, "lane-9", sha=new[:10]) == 0
    assert Q.review(q, world.store, labels, 1, "hold", "old", "lane-9", sha=old[:10]) == 0
    assert [lab.value for lab in world.store.labels(label_target(1, new), key=vocab.REVIEWED)] == ["land"]
    assert [lab.value for lab in world.store.labels(label_target(1, old), key=vocab.REVIEWED)] == ["hold"]
    with pytest.raises(QueueError, match="full 40-character SHA"):
        Q.review(q, world.store, labels, 1, "land", None, "lane-9", sha="0badc0de")
    with pytest.raises(QueueError, match="neither #1's head"):                     # a mistyped full SHA
        Q.review(q, world.store, labels, 1, "land", None, "lane-9", sha=new[:12] + "0" * 28)
    assert Q.review(q, world.store, labels, 1, "land", None, "lane-9", sha=new) == 0


def test_under_land_into_main_review_reads_no_next_and_records_on_the_open_head(world):
    """`next` can hold a PR at an older head than GitHub's.  Under `land_into = "main"` written in drive.toml the verified queue
    lands on `main` and nothing reads `next`, so a verdict without --sha goes on the open head."""
    from research.queue import StoreLabels

    world.with_store_remote()
    q = world.queue()
    _move_main(world, {DRIVE_FLAGS: 'land_into = "main"\n'})
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q.sync()
    old = world.gh.prs[1].head
    assert [m.head for m in q.chain() if m.pr == 1] == [old]
    world.gh.put(1, new := world.change({"a.txt": "2\n"}))
    labels = StoreLabels(world.store, log=lambda s: None)
    assert Q.review(q, world.store, labels, 1, "land", None, "lane-9") == 0
    assert [lab.value for lab in world.store.labels(label_target(1, new), key=vocab.REVIEWED)] == ["land"]
    assert not world.store.labels(label_target(1, old), key=vocab.REVIEWED)


def test_review_with_signed_verdicts_refuses_a_verdict_it_cant_sign_and_writes_one_its_holder_signs(world):
    """drive.toml's `signed_verdicts`: `review` writes through a signer, which refuses a verdict it can't sign while the flag
    is on, and a verdict its handle's holder signed counts (`StoreLabels` with a verifier)."""
    from research import labelauth
    from research.queue import StoreLabels
    from test_labelauth import OTHER, TS, _mint, _verifier

    world.with_store_remote()
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q, labels = world.queue(), StoreLabels(world.store, log=lambda s: None, verify=_verifier())

    def down(nonce: str) -> str:
        raise labelauth.AuthError("the Cursor identity socket is unavailable")

    with pytest.raises(QueueError, match="isn't written: drive.toml's `signed_verdicts` is on at `main`, and it can't be signed"):
        Q.review(q, labelauth.Signer(world.store, down, required=True, now=lambda: TS), labels, 1, "land", None, "proofs")
    assert labels.reviews(1, world.gh.prs[1].head) == []
    assert Q.review(q, labelauth.Signer(world.store, _mint(OTHER), now=lambda: TS), labels, 1, "land", None, "proofs") == 0
    assert [(lab.by, lab.value) for lab in labels.reviews(1, world.gh.prs[1].head)] == [("proofs", "land")]


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
    import research.landing as L
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
    monkeypatch.setattr(L, "merge_thread_post", lambda text, to, sender="ci": posts.append((text, to, sender)) or ok[0])
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
    import research.landing as L
    import research.queue as Qm

    monkeypatch.delenv("RESEARCH_DRIVE")
    monkeypatch.setattr(Qm, "GitHub", lambda repo: world.gh)
    monkeypatch.setattr(L, "merge_thread_post", lambda *a, **k: True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    base = ["--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo", "o/verity",
            "--store", str(world.tmp / "queue-store")]
    for verb in ("sync",):
        assert research_main(["queue", verb, *base]) == 1
        assert f"/data/infra/drive.sh queue {verb}" in capsys.readouterr().err
    def refs() -> str:
        return _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")

    before = refs()
    research_main(["queue", "eject", "1", "--reason", "x", "--by", "ci", *base])
    assert refs() == before and "the verified queue leaves it out until a ready mark re-admits it" in capsys.readouterr().out


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


def test_a_merge_of_main_keeps_the_heads_mark_and_verdict_and_a_real_change_loses_them(world):
    """A merge of `main` into a marked, verdicted head leaves its own change (`Queue.own_change`: the whitespace-kept patch-id of
    its diff from its merge base with the landing branch) as it was, so the new head takes the old one's mark and verdict, and
    the log names the head they came from: no new mark and no new review.  A real change on top takes none; a head with a label
    of its own reads only its own; a carried label is judged on its own head's companions; and a withdrawn mark carries as the
    mark did."""
    from research.queue import StoreLabels

    world.with_store_remote()
    world.gh.put(1, a := world.change({"a.txt": "1\n"}), ready=False)
    world.store.add_label(label_target(1, a), vocab.READY, True, "lane-1", ts="2026-11-06T10:00:00Z")
    world.store.add_label(label_target(1, a), vocab.REVIEWED, "land", "reviewer", ts="2026-11-06T11:00:00Z")
    _git(world.dev, "push", "-q", "origin", f"{(main := world.change({'b.txt': 'main moved\n'}))}:refs/heads/main")
    world.gh.put(1, b := _merge(world, a, main), ready=False)
    q = world.queue()
    q._clone()
    q.fetch(world.gh.open_prs())
    labels = StoreLabels(world.store, log=world.logs.append, change=q.own_change)
    assert q.own_change(b) == q.own_change(a) is not None
    assert labels.ready(1, b) and labels.ready_by(1, b) == "lane-1" and labels.marked_at(1, b) == "2026-11-06T10:00:00Z"
    assert [(lab.value, lab.by) for lab in labels.reviews(1, b)] == [("land", "reviewer")]
    assert [s for s in world.logs if "takes the" in s] == [
        f"research queue: #1 at {b[:8]} takes the `{key}` labels at {a[:8]}: none of its own counts, and its own change is "
        f"theirs (patch-id {q.own_change(b)[:12]})" for key in (vocab.READY, vocab.REVIEWED)]

    class Refuse:
        def why(self, lab, beside):
            return "no companion"

    strict = StoreLabels(world.store, verify=Refuse(), change=q.own_change)
    assert not strict.ready(1, b) and strict.refusals(1, b, vocab.READY) == [
        "ready = True by lane-1 at 2026-11-06T10:00:00Z: no companion"]

    world.gh.put(1, c := world.change({"a.txt": "1 fixed\n"}, base=b), ready=False)
    q.fetch(world.gh.open_prs())
    assert q.own_change(c) not in (None, q.own_change(a))
    assert not labels.ready(1, c) and labels.reviews(1, c) == [] and labels.ready_by(1, c) is None

    world.store.add_label(label_target(1, b), vocab.REVIEWED, "hold", "reviewer", ref="wait", ts="2026-11-06T12:00:00Z")
    assert [lab.value for lab in StoreLabels(world.store, change=q.own_change).reviews(1, b)] == ["hold"]
    world.store.add_label(label_target(1, a), vocab.READY, False, "ci", ref="breaks x", ts="2026-11-06T13:00:00Z")
    assert not StoreLabels(world.store, change=q.own_change).ready(1, b)


def test_a_head_whose_own_diff_differs_only_in_indentation_takes_neither_mark_nor_verdict(world):
    """A new head whose own diff differs from the marked, verdicted head's only in indentation takes neither its mark nor its
    verdict: a `return True` moved under `if not ok:` makes `gate(True)` None rather than True, though the two diffs are one
    with their whitespace stripped (`Queue.own_change` keeps it)."""
    from research.queue import StoreLabels

    world.with_store_remote()
    gate = "def gate(ok):\n    if not ok:\n        print('no')\n"
    _git(world.dev, "push", "-q", "origin", f"{world.change({'gate.py': gate})}:refs/heads/main")
    world.gh.put(1, a := world.change({"gate.py": gate + "    return True\n"}), ready=False)
    world.store.add_label(label_target(1, a), vocab.READY, True, "lane-1", ts="2026-11-06T10:00:00Z")
    world.store.add_label(label_target(1, a), vocab.REVIEWED, "land", "reviewer", ts="2026-11-06T11:00:00Z")
    world.gh.put(1, b := world.change({"gate.py": gate + "        return True\n"}, base=a), ready=False)
    q = world.queue()
    q._clone()
    q.fetch(world.gh.open_prs())

    def stripped(head: str) -> list[str]:
        diff = _git(q.workdir, "diff-tree", "-r", "-p", q.base, head).splitlines()
        return [re.sub(r"\s", "", s) for s in diff if not s.startswith("index ")]

    assert stripped(b) == stripped(a)
    assert q.own_change(b) not in (None, q.own_change(a))
    labels = StoreLabels(world.store, log=world.logs.append, change=q.own_change)
    assert labels.ready(1, a) and [(lab.value, lab.by) for lab in labels.reviews(1, a)] == [("land", "reviewer")]
    assert not labels.ready(1, b) and labels.reviews(1, b) == [] and labels.ready_by(1, b) is None
    assert not [s for s in world.logs if "takes the" in s]


def test_ready_mark_writes_the_mark_with_no_quick_tier(world, capsys):
    """A ready post with no mark behind it admits nothing; `--mark` needs neither a machine nor a tier."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    h = world.change({"m.txt": "m\n"})
    world.gh.put(3, h, ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly()) == 0
    assert "marked on lane-3's word, with no quick tier" in capsys.readouterr().out
    assert StoreLabels(world.store, log=world.logs.append).ready(3, h) is True


GOAL_FIX = ("add a `Goal <id>` line naming the live goal it serves (its id on the Goals page, `research goals show`), or an "
            "`Issue: none: REASON` line, to its body (`research queue edit {n} --body-file F`)")


def test_ready_warns_of_a_body_naming_no_live_goal_and_no_issue_and_marks_it_all_the_same(world, monkeypatch, capsys):
    """The goal gate (here under `--mark`, as under every tier) warns on stderr when the body names no live goal of the Goals page
    on a `Goal <id>` line and no issue on purpose on an `Issue: none: REASON` line, and never refuses: the gates after it run,
    and the head is marked."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    serve(monkeypatch, world.tmp, dict(PAGE))
    ran: list[str] = []
    gate = lambda name: lambda *a: ran.append(name)            # noqa: E731  (each gate records that it ran, and passes)
    bodies = {"Lane `lane-3`.": "its body names no goal",
              "Lane `lane-3`.\n\nGoal ci/old-landing\n": "its body names no live goal (ci/old-landing is done)",
              "Lane `lane-3`.\n\nGoal ci/unlisted\nGoal ci/old-landing\n":
                  "its body names no live goal (the Goals page has no goal ci/unlisted; ci/old-landing is done)",
              "Lane `lane-3`.\n\nRefs ENG-123\nGoal ci/faster-landing\n": None,
              "Lane `lane-3`.\n\n> Goal: `ci/merge-speed`.\nGoal ci/old-landing\n": None,
              "Lane `lane-3`.\n\nIssue: none: docs only\n": None}
    for i, (body, why) in enumerate(bodies.items()):
        world.gh.put(3, h := world.change({"m.txt": f"{i}\n"}), body=body, ready=False)
        assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), records=gate("records"), preflight=gate("preflight"),
                     goals=Q.goal_gate) == 0
        assert StoreLabels(world.store, log=world.logs.append).ready(3, h) is True
        warned = [line for line in capsys.readouterr().err.splitlines() if line.startswith("research queue ready: warning:")]
        assert warned == ([f"research queue ready: warning: #3 at {h[:12]}: {why}: {GOAL_FIX.format(n=3)}"] if why else []), warned
    assert ran == ["records", "preflight"] * len(bodies)


def test_ready_reads_the_goals_this_machine_last_read_whatever_their_age_while_the_page_is_down_and_warns_with_neither(
        world, monkeypatch, capsys):
    """Each read of the Goals page is kept on this machine; while the page can't be read the gate judges by that copy, saying so in
    one line, and with no copy it warns, naming the outage.  Each head is marked."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    page, cache = serve(monkeypatch, world.tmp, None), world.tmp / "cache" / "goals.json"
    marked = lambda n, h: StoreLabels(world.store, log=world.logs.append).ready(n, h)    # noqa: E731
    mark = lambda n: ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), goals=Q.goal_gate)  # noqa: E731
    body = "Lane `lane-3`.\n\nGoal ci/merge-speed\n"
    world.gh.put(3, h := world.change({"m.txt": "1\n"}), body=body, ready=False)
    assert mark(3) == 0 and marked(3, h) and not cache.exists()
    assert (f"research queue ready: warning: #3 at {h[:12]}: the Goals page can't be read ({DOWN}) and this machine has no copy "
            f"of it ({cache}): its `Goal` lines go unchecked") in capsys.readouterr().err
    world.gh.put(4, h4 := world.change({"n.txt": "4\n"}), body="Lane `lane-4`.\n\nIssue: none: docs only\n", ready=False)
    assert mark(4) == 0 and marked(4, h4)
    page.goals = dict(PAGE)
    world.gh.put(3, h := world.change({"m.txt": "2\n"}), body=body, ready=False)
    capsys.readouterr()
    assert mark(3) == 0 and marked(3, h) and "warning" not in capsys.readouterr().err
    kept = json.loads(cache.read_text())
    assert kept["goals"] == [{"id": g, "status": PAGE[g]} for g in sorted(PAGE)]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", kept["read"])
    page.goals = None
    cache.write_text(json.dumps({**kept, "read": "2020-01-01T00:00:00Z"}))
    for i, (b, warns) in enumerate(((body, False), ("Lane `lane-3`.\n\nGoal ci/old-landing\n", True))):
        world.gh.put(3, h := world.change({"m.txt": f"{i + 3}\n"}), body=b, ready=False)
        capsys.readouterr()
        assert mark(3) == 0 and marked(3, h)
        said = [line for line in capsys.readouterr().err.splitlines() if "Goals page" in line]
        assert said[0] == (f"research queue: the Goals page can't be read ({DOWN}): using its 4 goals as this machine read them "
                           f"at 2020-01-01T00:00:00Z ({cache})")
        assert said[1:] == ([f"research queue ready: warning: #3 at {h[:12]}: its body names no live goal (ci/old-landing is "
                             f"done): {GOAL_FIX.format(n=3)}"] if warns else [])


def test_pr_from_api_reads_the_head_repository_and_the_author():
    d = {"number": 7, "head": {"sha": "a" * 40, "ref": "lane/x", "repo": {"full_name": "someone/verity"}}, "base": {"ref": "main"},
         "user": {"login": "someone"}, "author_association": "CONTRIBUTOR"}
    pr = PR.from_api(d)
    assert (pr.head_repo, pr.author, pr.association) == ("someone/verity", "someone", "CONTRIBUTOR")
    assert PR.from_api({**d, "head": {**d["head"], "repo": None}}).head_repo is None
    assert PR.from_api({**d, "user": None, "author_association": None}).association == ""


def _outside_prs(world: World) -> dict[int, str]:
    """PRs 1-4 refused (a fork's head, a deleted head repository, an outside author, a collaborator), 5 and 6 not (a member's, the
    App's), each marked ready at its head; their heads."""
    import dataclasses

    who = {1: dict(head_repo="someone/verity"), 2: dict(head_repo=None), 3: dict(author="stranger", association="CONTRIBUTOR"),
           4: dict(author="helper", association="COLLABORATOR"), 5: dict(author="alice", association="MEMBER"),
           6: dict(author="verity-agents[bot]", association="NONE")}
    heads = {}
    for n, fields in who.items():
        world.gh.put(n, heads.setdefault(n, world.change({f"o{n}.txt": f"{n}\n"})))
        world.gh.prs[n] = dataclasses.replace(world.gh.prs[n], **fields)
    return heads


def test_a_fork_or_an_outside_author_is_refused_at_mark_before_any_gate_runs_and_a_members_or_the_apps_pr_is_marked(world, capsys):
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    heads = _outside_prs(world)
    ran: list[str] = []
    gate = lambda name: lambda *a: ran.append(name)            # noqa: E731  (each gate records that it ran, and passes)
    tier = lambda pr: ran.append("tier")                        # noqa: E731
    for n in (1, 2, 3, 4):
        assert ready(world.gh, world.store, n, f"lane-{n}", tier, records=gate("records"), preflight=gate("preflight")) == 1
        err = capsys.readouterr().err
        assert f"#{n} is not marked ready: " in err and "a maintainer opens it from a branch of this repository" in err, err
        assert StoreLabels(world.store, log=world.logs.append).ready(n, heads[n]) is False
    assert ran == []
    ready(world.gh, world.store, 1, "lane-1", MarkOnly())
    assert f"its head is on someone/verity, not {Q.REPO}" in capsys.readouterr().err
    ready(world.gh, world.store, 2, "lane-2", MarkOnly())
    assert "its head is on a deleted repository" in capsys.readouterr().err
    ready(world.gh, world.store, 4, "lane-4", MarkOnly())
    assert "its author helper is no owner or member of the organization (COLLABORATOR)" in capsys.readouterr().err
    for n in (5, 6):
        assert ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), records=gate("records"), preflight=gate("preflight")) == 0
        assert StoreLabels(world.store, log=world.logs.append).ready(n, heads[n]) is True
    assert ran == ["records", "preflight"] * 2


def test_a_fork_or_an_outside_author_is_refused_at_admission_though_marked_and_a_members_or_the_apps_pr_is_admitted(world):
    _outside_prs(world)
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [5, 6] == world.on_next(q)
    assert entries[1].state == "refused" and entries[1].why == (f"its head is on someone/verity, not {Q.REPO}: a maintainer opens "
                                                                "it from a branch of this repository")
    assert entries[2].why.startswith("its head is on a deleted repository")
    assert entries[3].why.startswith("its author stranger is no owner or member of the organization (CONTRIBUTOR)")
    assert all(entries[n].state == "refused" for n in (1, 2, 3, 4))
    status = q.status()
    assert [n for n, e in sorted(status.items()) if e.state == "queued"] == [5, 6] and status[4].state == "refused"


def _old_next(world: World) -> tuple[str, str, str]:
    """#1 merged onto `next`, a head built on that `next` (as a demo branch that merged it is), then #1 closed unlanded and
    `next` rebuilt without it: (#1's head, its merge onto the old `next`, the head built on it)."""
    q = world.queue()
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    q.sync()
    merge = _git(world.origin, "rev-parse", "refs/heads/next")
    head = world.change({"o.txt": "on the old next\n"}, base=merge)
    del world.gh.prs[1]
    q.sync()
    assert world.on_next(q) == []
    return h1, merge, head


OLD_NEXT_FIX = ("which would land with it: retarget the PR to the demo branch it builds on, or restack its own commits onto `main` "
                "(a branch from `origin/main` with each cherry-picked, and the branch of each open PR it needs merged, never `next`) "
                "and mark that head")


def test_the_sync_refuses_a_head_carrying_merges_of_an_old_next_of_prs_closed_unlanded_naming_the_fix(world):
    """A branch that merged `next` (a demo's) carries the merge of every PR on it then; once `next` is rebuilt without one that
    closed unlanded, a PR from that branch would land that PR's change with its own."""
    h1, merge, head = _old_next(world)
    world.gh.put(2, head)
    q = world.queue()
    members, entries = q.sync()
    assert entries[2].state == "refused" and entries[2].why == (
        f"its head carries 1 merge of an old `next` that `main` doesn't hold, of PRs closed unlanded ({merge[:12]} (#1)), "
        + OLD_NEXT_FIX)
    assert members == [] and q.status()[2].state == "refused"
    _git(world.dev, "push", "-q", "origin", f"{h1}:refs/heads/main")      # #1's change landed after all, at its head
    members, entries = q.sync()
    assert [m.pr for m in members] == [2], "a merge of a PR that landed carries nothing `main` lacks"


def test_a_head_stacked_on_an_open_pr_or_built_on_todays_next_is_not_refused(world):
    """`next` holds merges of open PRs only: a head built on it, or on another open PR's head, carries no change that is neither
    open nor landed, and goes after those PRs."""
    q = world.queue()
    world.gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    q.sync()
    world.gh.put(2, world.change({"b.txt": "2\n"}, base=h1), ready=False)
    world.gh.put(3, world.change({"c.txt": "3\n"}, base=_git(world.origin, "rev-parse", "refs/heads/next")))
    members, entries = q.sync()
    assert not [n for n, e in entries.items() if e.state == "refused"], {n: e.why for n, e in entries.items()}
    assert [m.pr for m in members] == [1, 3] and entries[3].after == (1,) and entries[2].state == "waiting"


def test_the_preflight_refuses_a_head_on_main_carrying_an_old_next_before_any_check_and_one_based_elsewhere_by_its_carry(world, capsys):
    """At the mark as at admission, in the sync's words, and first, since the fix makes another head.  A PR based on another
    branch (a demo's) isn't judged for it, as drive never admits it; the carried-head check (`stale_carried`) refuses its mark."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    _, merge, head = _old_next(world)
    world.gh.put(2, head, ready=False)
    world.gh.put(3, demo := world.change({"p.txt": "3\n"}, base=merge), ready=False)
    world.gh.prs[3] = dataclasses.replace(world.gh.prs[3], base="demo", branch="cursor/w-3")
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 1
    assert capsys.readouterr().err == (
        f"research queue ready: #2 is not marked ready at {head[:12]}: its head carries 1 merge of an old `next` that `main` "
        f"doesn't hold, of PRs closed unlanded ({merge[:12]} (#1)), {OLD_NEXT_FIX}\n")
    assert not [s for s in world.logs if "merged onto" in s], "no other check ran"
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=_preflight(world)) == 1
    err = capsys.readouterr().err
    assert "old `next`" not in err and f"through `next`'s merge {merge[:12]}, and #1 is closed" in err, err
    labels = StoreLabels(world.store, log=world.logs.append)
    assert [labels.ready(2, head), labels.ready(3, demo)] == [False, False]


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
    "adapters/vllm/tests/lint/test_p03_one_evaluator.py": _invariant("VLLM_LINT", 4),
    "tools/verity/check/suites.py": (Path(__file__).resolve().parents[2] / "verity" / "check" / "suites.py").read_text(),
    "tools/verity/check/settings.py": (Path(__file__).resolve().parents[2] / "verity" / "check" / "settings.py").read_text(),
    "tools/verity/check/pins.py": ("import sys\nfrom pathlib import Path\nif Path('STALE.txt').exists():\n"
                                   "    print('pkg/pins.json is stale (pkg/Main.lean): run `python3 regen.py`', file=sys.stderr)\n"
                                   "    sys.exit(1)\n"),
}
#: `uv sync --locked` as the preflight runs it, refusing while LOCK_STALE.txt is in the tree
FAKE_SYNC = (sys.executable, "-c", "import sys; from pathlib import Path; s = Path('LOCK_STALE.txt').exists(); s and print("
             "'error: The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.', file=sys.stderr); sys.exit(2 * s)")
#: the ratchet gate (`PREFLIGHT_RATCHET`) as a stand-in, run as `ratchet_gate.py BASE TOP --json`: it writes where it ran and
#: with what to $RATCHET_SEEN, fails TOP on an import cycle while TOP's tree holds RATCHET_NEW, and exits 2 (it can't judge)
#: while it holds RATCHET_CANT; like the gate, it reads TOP from git, never the checkout it runs in
RATCHET_GATE = (
    "import json, os, subprocess, sys\nfrom pathlib import Path\n"
    "holds = lambda name: subprocess.run(['git', 'cat-file', '-e', f'{sys.argv[2]}:{name}']).returncode == 0\n"
    "head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()\n"
    "Path(os.environ['RATCHET_SEEN']).write_text(json.dumps({'script': sys.argv[0], 'argv': sys.argv[1:], 'cwd': os.getcwd(), "
    "'head': head, 'cache': os.environ.get('RATCHET_GATE_CACHE'), 'tmp': os.environ.get('TMPDIR')}))\n"
    "if holds('RATCHET_CANT'):\n"
    "    print('ratchet gate: the base has no lints to run', file=sys.stderr)\n"
    "    sys.exit(2)\n"
    "cycle = {'why': 'the base finds 1 violation its lists miss', 'fix': ['p09_layering: [package-cycle] commit <-> engine']}\n"
    "fails = [cycle] if holds('RATCHET_NEW') else []\n"
    "print(json.dumps({'failures': fails}))\n"
    "sys.exit(1 if fails else 0)\n")


def _preflight(world: World, free=lambda: 1.0, fetch=None, store=None, repo: Path | None = None, sync=FAKE_SYNC,
               budget_s: float = math.inf, **kw):
    """The preflight on ``world``.  Its checks' budget never runs out, so each waits on its check's end: the preflight is
    fail-open, and a check a slow disk kept past the real `PREFLIGHT_CHECKS_S` would be skipped, the refusal a test expects
    missed.  Nor does the budget of its read of the marked heads (`collide_s`), unless a test gives one."""
    from research.queue import Preflight

    kw.setdefault("collide_s", math.inf)
    return Preflight(repo or world.dev, str(world.origin), store or world.store, sync=sync, python=sys.executable,
                     root=world.tmp / "pre", log=world.logs.append, free=free, fetch=fetch, budget_s=budget_s, **kw)


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
    never had; one note names the suite and the fixtures, and the head is marked: its candidate's check fetches them."""
    from research.queue import MarkOnly, StoreLabels, ready

    files, _ = _fixture_tree(world)
    _preflight_main(world, files)
    calls: list = []
    world.gh.put(2, h2 := world.change(ROWS_TEST), ready=False)
    pre = _preflight(world, fetch=_stub_fetch(calls, why="no credentials for the store's remote"))
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=pre) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight" in s)
    note = (f"pkg not judged: 1 of the 1 fixture(s) the suites read weren't fetched (no credentials for the store's remote: "
            f"{BIG_PATH}); its candidate's check fetches them")
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


def test_under_land_into_main_the_preflight_judges_a_head_on_main_and_reads_no_next(world, capsys):
    """Under `land_into = "main"` written in drive.toml the verified queue merges each candidate onto `main` and drive's `next`
    is frozen: a head that conflicts only with what `next` holds is judged on `main`, as the loop merges it, and marked."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world)
    main = _move_main(world, {DRIVE_FLAGS: 'land_into = "main"\n'})
    world.gh.put(2, h2 := world.change({"pkg/rows.json": '{"r9": 1}\n'}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight" in s)
    assert f"merged onto `main` at {main[:12]}:" in said, said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflight_merges_onto_the_landing_tip_when_next_no_longer_holds_it(world, capsys):
    """A `next` the landing branch has moved past is no queue above it, so the head merges onto the landing branch's tip, and a
    conflict with a PR only that `next` holds refuses nothing."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world)
    main = _move_main(world, {"pkg/other.py": "X = 1\n"})
    world.gh.put(2, h2 := world.change({"pkg/rows.json": '{"r9": 1}\n'}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight" in s)
    assert f"merged onto `main` at {main[:12]} (`next` at" in said and "doesn't hold it)" in said, said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflight_finds_a_prs_merge_on_next_past_a_lock_commit_on_checked_when_it_walks_from_main(world, capsys):
    """When `checked` can't be fetched the preflight walks `next` from `main`, across the commit drive puts on `checked`
    without the queue's trailers (`Lock records regenerated for #N`).  A PR already on `next` still merges onto the commit
    below its merge there, so its rewritten head isn't refused for conflicting with its old one, and one landed on `checked`
    (open until `main` reaches it) and on `next` again below its newest merge; a conflict with a PR landed on `checked` still
    names that PR."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    _move_main(world, {DRIVE_FLAGS: 'land_into = "checked"\n'})
    _git(world.origin, "update-ref", "refs/heads/checked", "refs/heads/main")
    world.gh.put(3, h3 := world.change({"c.txt": "c3\n"}))
    assert [m.pr for m in world.queue().sync()[0]] == [3]
    landed = _git(world.origin, "rev-parse", "refs/heads/next")
    lock = _git(world.origin, "commit-tree", f"{landed}^{{tree}}", "-p", landed, "-m", "Lock records regenerated for #3")
    _git(world.origin, "update-ref", "refs/heads/checked", lock)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}))
    world.gh.put(3, h3b := world.change({"d.txt": "d3\n"}, base=h3))           # a fix on top of #3's landed head
    one, three = world.queue().sync()[0]
    assert (one.pr, one.head, three.pr, three.head) == (1, h1, 3, h3b)
    assert _git(world.origin, "rev-parse", f"{one.commit}^1") == lock
    _git(world.origin, "update-ref", "-d", "refs/heads/checked")               # its fetch fails: the preflight's base is `main`
    for m, files, base in ((one, {"a.txt": "a1, again\n"}, "origin/main"), (three, {"d.txt": "d3, again\n"}, h3)):
        world.gh.put(m.pr, new := world.change(files, base=base), ready=False)   # rewritten, not on top of the head `next` has
        world.logs.clear()
        pre = _preflight(world)
        assert ready(world.gh, world.store, m.pr, f"lane-{m.pr}", MarkOnly(), preflight=pre) == 0, capsys.readouterr().err
        said, below = next(s for s in world.logs if "preflight" in s), _git(world.origin, "rev-parse", f"{m.commit}^1")
        assert f"merged onto `next` below its merge of #{m.pr} at {m.head[:8]} ({below[:12]})" in said, said
        assert "conflict" not in capsys.readouterr().err and StoreLabels(world.store, log=world.logs.append).ready(m.pr, new) is True
    world.gh.put(4, world.change({"c.txt": "c4\n"}), ready=False)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=_preflight(world)) == 1
    assert "it conflicts with `next` at" in (err := capsys.readouterr().err) and "in c.txt (touched by #3)" in err, err
    assert _no_worktree_left(world)


def test_the_preflight_finds_a_prs_merge_on_next_while_a_landing_has_moved_checked_off_it(world, capsys):
    """A landing moves `checked` off `next` (here drive's lock commit on the prefix it landed) until the sync rebuilds `next`
    on it: `checked` isn't below `next` then, and a PR on `next` above that prefix still merges onto the commit below its merge
    there, not onto `next`'s tip, where its rewritten head conflicted with its old one."""
    from research.queue import MarkOnly, ready

    world.with_store_remote()
    _move_main(world, {DRIVE_FLAGS: 'land_into = "checked"\n'})
    _git(world.origin, "update-ref", "refs/heads/checked", "refs/heads/main")
    world.gh.put(1, world.change({"a.txt": "a1\n"}))
    world.gh.put(2, h2 := world.change({"b.txt": "b2\n"}))
    one, two = world.queue().sync()[0]
    assert (one.pr, two.pr, two.head) == (1, 2, h2)
    lock = _git(world.origin, "commit-tree", f"{one.commit}^{{tree}}", "-p", one.commit, "-m", "Lock records regenerated for #1")
    _git(world.origin, "update-ref", "refs/heads/checked", lock)
    world.gh.put(2, world.change({"b.txt": "b2, again\n"}), ready=False)          # rewritten, not on top of h2
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight" in s)
    assert f"merged onto `next` below its merge of #2 at {h2[:8]} ({one.commit[:12]})" in said, said
    assert "conflict" not in capsys.readouterr().err


def test_the_preflight_runs_its_checks_on_a_merge_whose_only_conflict_is_in_a_pin_rebuilt_as_the_sync_rebuilds_it(world, capsys):
    """A PR that regenerated a pin for its own input conflicts with one on `next` that did the same, only in the pin, which drive's
    sync rebuilds from the merged tree (as PoUS's TRUSTED.sha256): the preflight merges it the same way and checks that tree,
    so a check failing there refuses the mark, rather than the checks being skipped while the mark goes through."""
    from research.queue import MarkOnly, StoreLabels, ready

    pin = "pkg/PIN.sha256"
    listing = lambda a, b: "".join(f"{hashlib.sha256(t.encode()).hexdigest()}  {n}\n"            # noqa: E731
                                   for n, t in (("a.txt", a), ("b.txt", b)))
    world.with_store_remote()
    _move_main(world, {**PREFLIGHT_FILES, "ci/pins.py": f'PINS = (Pin("{pin}", None, "pkg"),)\n', "pkg/a.txt": "a0\n",
                       "pkg/b.txt": "b0\n", pin: listing("a0\n", "b0\n")})
    world.gh.put(1, world.change({"pkg/a.txt": "a1\n", pin: listing("a1\n", "b0\n")}))
    assert [m.pr for m in world.queue().sync()[0]] == [1]
    next_ = _git(world.origin, "rev-parse", "refs/heads/next")
    for n, broken in ((2, {"BROKEN_MODULE_STRING": ""}), (3, {})):
        world.gh.put(n, h := world.change({"pkg/b.txt": f"b{n}\n", pin: listing("a0\n", f"b{n}\n"), **broken}), ready=False)
        raw = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", next_, h], cwd=world.dev,
                             capture_output=True, text=True)
        assert raw.returncode == 1 and raw.stdout.splitlines()[1:] == [pin], raw.stdout
        world.logs.clear()
        assert ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), preflight=_preflight(world)) == (1 if broken else 0)
        err = capsys.readouterr().err
        if broken:
            [found] = [ln for ln in err.splitlines() if ln.startswith("  - ")]
            assert found.startswith("  - the invariant tests in . fail on the merged tree: FAILED "
                                    "tests/test_module_strings.py::test_broken_module_string_absent"), err
            assert f"conflicts only in pins: {pin}, rebuilt from the merged tree" in err and "checks skipped" not in err, err
        else:
            said = next(s for s in world.logs if f"preflight: #{n} at" in s)
            assert "nothing refused" in said and f"conflicts only in pins: {pin}, rebuilt from the merged tree" in said, said
            assert "collected 2 suite(s) (pkg, .)" in said and "checks skipped" not in said, said
        assert StoreLabels(world.store, log=world.logs.append).ready(n, h) is (not broken) and _no_worktree_left(world)


def test_the_preflight_merges_with_the_drivers_of_next_not_those_of_the_older_checkout_it_runs_in(world, capsys):
    """A branch from before `merge.py --caps`, marked from its own checkout, merged onto `next` with that checkout's `merge.py`
    as the `lean-caps` driver: it errored on `--caps`, and git reported a conflict in `tests/test_lean_packages.py` that the
    sync, with `next`'s driver, merges as counters.  The preflight runs `next`'s drivers, so the head merges cleanly."""
    from research.queue import MarkOnly, StoreLabels, base_drivers, merged_tree, ready

    caps = "tests/test_lean_packages.py"

    def counts(sec: int, props: int) -> str:
        return f'UNWITNESSED = {{"security": {sec}, "security/Properties": {props}}}\n'

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    world.with_store_remote()
    old = _move_main(world, {**PREFLIGHT_FILES, caps: counts(354, 184), "tools/verity/lean/merge.py":
                             "import sys\nsys.exit('merge.py: error: unrecognized arguments: --caps')\n"})
    _move_main(world, {**{f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")},
                       ".gitattributes": f"{caps} merge=lean-caps\n"})
    world.gh.put(1, world.change({caps: counts(355, 184)}))
    assert [m.pr for m in world.queue().sync()[0]] == [1]
    world.gh.put(2, h2 := world.change({caps: counts(354, 185)}, base=old), ready=False)    # the checkout stays at #2's head
    _git(world.dev, "config", "merge.lean-caps.driver", "python3 tools/verity/lean/merge.py --caps %O %A %B %P")
    next_ = _git(world.origin, "rev-parse", "refs/heads/next")
    raw = subprocess.run(["git", f"--attr-source={next_}", "merge-tree", "--write-tree", "--name-only", "--no-messages", next_, h2],
                         cwd=world.dev, capture_output=True, text=True)
    assert raw.returncode == 1 and raw.stdout.splitlines()[1:] == [caps], raw.stdout + raw.stderr
    with base_drivers(world.dev, next_) as config:
        tree, files, _ = merged_tree(world.dev, next_, h2, config=config)
    assert tree and files == [] and _git(world.dev, "show", f"{tree}:{caps}") == counts(355, 185).strip()
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if "preflight: #2 at" in s)
    assert "nothing refused" in said and "checks skipped" not in said, said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_sync_merges_onto_each_tip_with_that_tips_drivers_not_those_of_its_checkout(world):
    """Drive's clone sits at `checked`'s tip, so its `merge.py` can predate one a member of `next` brings, and the sync ran that
    old driver for every merge above that member: a PR's lock merged onto `next` by a `merge.py` from before readerless reads
    collapsed the Security lock's reads groups.  Each merge now runs the tip's own drivers."""
    caps = "tests/test_lean_packages.py"

    def counts(sec: int, props: int) -> str:
        return f'UNWITNESSED = {{"security": {sec}, "security/Properties": {props}}}\n'

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    _move_main(world, {caps: counts(354, 184), ".gitattributes": f"{caps} merge=lean-caps\n",
                       "tools/verity/lean/merge.py": "import sys\nsys.exit('merge.py: error: unrecognized arguments: --caps')\n"})
    world.gh.put(1, world.change({f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}))
    world.gh.put(2, world.change({caps: counts(355, 184)}))
    world.gh.put(3, world.change({caps: counts(354, 185)}))
    assert [m.pr for m in world.queue().sync()[0]] == [1, 2, 3], world.logs     # #3 conflicted under the checkout's driver
    assert _git(world.origin, "show", f"refs/heads/next:{caps}") == counts(355, 185).strip()


#: what the head writes -> what the refusal says: each a way a train breaks that the merged tree shows at mark time
PREFLIGHT_REFUSALS = {
    "stale pin": ({"STALE.txt": "x\n"}, "`pins.py --stale` on the merged tree: pkg/pins.json is stale (pkg/Main.lean)"),
    "module string": ({"BROKEN_MODULE_STRING": ""}, "the invariant tests in . fail on the merged tree: FAILED "
                      "tests/test_module_strings.py::test_broken_module_string_absent"),
    "pythonpath": ({"BROKEN_REAL_ROOTS": ""}, "the invariant tests in tools/research fail on the merged tree: FAILED "
                   "tests/test_pythonpath.py::test_broken_real_roots_absent"),
    # a report-genre name fails a train's preflight-lints
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
    "vllm lint": ({"adapters/vllm/verity_vllm/kernels.py": "", "VLLM_LINT": ""},
                  "the invariant tests in adapters/vllm/tests/lint fail on the merged tree: FAILED "
                  "tests/lint/test_p03_one_evaluator.py::test_vllm_lint_absent"),
}


@pytest.mark.parametrize("case", list(PREFLIGHT_REFUSALS))
def test_the_preflight_refuses_a_head_whose_merge_onto_next_breaks_an_invariant_and_names_the_fix(world, capsys, case):
    """Each check runs to its end however long it takes: under a clock that leaps an hour a reading, the budget `_preflight`
    gives can't run out, so a slow machine skips no check and misses no refusal."""
    from research.queue import MarkOnly, StoreLabels, ready

    files, says = PREFLIGHT_REFUSALS[case]
    _preflight_main(world)
    world.gh.put(2, h2 := world.change(files), ready=False)
    hours = itertools.count(0.0, 3600.0).__next__
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world, clock=hours)) == 1
    err = capsys.readouterr().err
    assert f"#2 is not marked ready at {h2[:12]}: the preflight of its head merged onto `next` at" in err
    assert says in err, err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [next(ln for ln in err.splitlines() if says in ln)]
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)


@pytest.mark.parametrize("deadline, timeout", [(125.0, 120.0), (math.inf, None)])
def test_a_preflight_check_waits_what_is_left_of_its_budget_and_with_none_to_run_out_waits_on_its_end(world, monkeypatch,
                                                                                                   deadline, timeout):
    """A check's timeout is what is left of the budget when it starts; under an unbounded budget it has none, since
    `subprocess.run` can't wait an infinite one (OverflowError)."""
    asked = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: asked.append(kw["timeout"]) or subprocess.CompletedProcess(argv, 0))
    got, why = _preflight(world, budget_s=deadline - 5.0, clock=lambda: 5.0)._run(["true"], world.tmp, deadline)
    assert (got.returncode, why, asked) == (0, "", [timeout])


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
    assert f"checks skipped: the landing gate gates/g.py can't judge {h3[:12]}" in said and "exit 2" in said, said
    assert "nothing refused" not in said, said


def test_the_preflight_prints_what_a_warn_gate_at_main_says_as_a_warning_and_takes_the_mark(world, capsys):
    """A warn gate (drive.toml's `warn_gates`) keeps no PR out, so the ready mark prints what it says and refuses nothing."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world, {**GATED, DRIVE_FLAGS: 'warn_gates = ["gates/g.py"]\n'})
    main = _git(world.origin, "rev-parse", "main")
    world.gh.put(2, h2 := world.change({"bad.txt": "2\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert [s for s in world.logs if "warning" in s] == [
        f"research queue: preflight: warning: #2 at {h2[:12]}: the warn gate gates/g.py at `main` {main[:12]}: touches bad.txt; "
        "delete bad.txt (`python3 gates/g.py origin/main HEAD` in the branch runs it). A warn gate refuses nothing: the PR "
        "lands, and its owner fixes it in their next PR"], world.logs
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflight_runs_a_lint_only_on_a_change_that_reaches_what_it_judges(world, capsys):
    """`PREFLIGHT_LINTS` add up to a minute, so a change outside their prefixes doesn't run them: a head breaking the vllm lint's
    stand-in without touching adapters/vllm is marked, and the train's check runs the lint."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world)
    world.gh.put(2, h2 := world.change({"pkg/notes.txt": "x\n", "VLLM_LINT": ""}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflight_prints_what_the_ratchet_gate_fails_on_the_merge_as_a_warning_and_notes_one_that_cant_judge(
        world, capsys, monkeypatch):
    """The ratchet gate tells a new violation by its text, so it can read an honest change's (a moved line, a cycle through a
    module the PR adds) as one: on a change that reaches what it judges, the preflight runs the base's copy on the merge as
    check's step does, keeping no verdict and its scratch in the merged worktree's directory, and the mark prints what it fails
    and goes through.  A gate that can't judge is named under "checks skipped"."""
    from research.queue import PREFLIGHT_RATCHET, RATCHET_SAYS, MarkOnly, StoreLabels, ready

    monkeypatch.setenv("RATCHET_SEEN", str(seen := world.tmp / "seen.json"))
    _preflight_main(world, {PREFLIGHT_RATCHET[0]: RATCHET_GATE})
    nxt = _git(world.origin, "rev-parse", "next")
    own = "print('{\"failures\": []}')\n"                              # the head's copy, which passes: the base's judges
    world.gh.put(2, h2 := world.change({"adapters/vllm/verity_vllm/commit.py": "", "RATCHET_NEW": "", PREFLIGHT_RATCHET[0]: own}),
                 ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert [s for s in world.logs if "warning" in s] == [
        f"research queue: preflight: warning: #2 at {h2[:12]}: the ratchet gate ({PREFLIGHT_RATCHET[0]} as `next` at "
        f"{nxt[:12]} has it) fails the merge: the base finds 1 violation its lists miss\n"
        f"    p09_layering: [package-cycle] commit <-> engine\n  {RATCHET_SAYS}"], world.logs
    ran = json.loads(seen.read_text())
    base, top, *rest = ran["argv"]
    assert base == nxt and rest == ["--json"] and _git(world.dev, "log", "-1", "--format=%P", top) == f"{nxt} {h2}", ran
    assert ran["cache"] == "" and ran["head"] == top, ran
    assert Path(ran["script"]).resolve().parent == Path(ran["cwd"]).resolve().parent == Path(ran["tmp"]).resolve(), ran
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)

    world.gh.put(3, h3 := world.change({"adapters/vllm/verity_vllm/engine.py": "", "RATCHET_CANT": ""}), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    said = next(s for s in world.logs if f"preflight: #3 at {h3[:12]}" in s)
    assert "checks skipped: ratchet-gate not checked: exited 2: ratchet gate: the base has no lints to run" in said, said
    assert [s for s in world.logs if "warning" in s] == [s for s in world.logs if "warning" in s and "#2" in s]
    assert StoreLabels(world.store, log=world.logs.append).ready(3, h3) is True and _no_worktree_left(world)


def test_the_preflight_runs_the_ratchet_gate_after_the_run_at_next_that_a_refusal_needs(world, capsys, monkeypatch):
    """The gate comes last, after the run at `next` that tells a failure the PR causes from one `next` has too, so the time it
    takes never turns a refusal into a check that couldn't run on `next` to compare; it judges the merge by its commit, though
    the worktree is at `next` by then."""
    from research.queue import PREFLIGHT_RATCHET, MarkOnly, StoreLabels, ready

    monkeypatch.setenv("RATCHET_SEEN", str(seen := world.tmp / "seen.json"))
    _preflight_main(world, {PREFLIGHT_RATCHET[0]: RATCHET_GATE})
    nxt = _git(world.origin, "rev-parse", "next")
    world.gh.put(2, h2 := world.change({"adapters/vllm/verity_vllm/commit.py": "", "RATCHET_NEW": "", "BROKEN_DOC_PATH": ""}),
                 ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 1
    assert "FAILED tests/test_doc_paths.py::test_broken_doc_path_absent" in (err := capsys.readouterr().err), err
    assert [s for s in world.logs if f"warning: #2 at {h2[:12]}: the ratchet gate" in s], world.logs
    ran = json.loads(seen.read_text())
    assert ran["head"] == nxt and _git(world.dev, "log", "-1", "--format=%P", ran["argv"][1]) == f"{nxt} {h2}", ran
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)


def test_the_preflight_runs_the_ratchet_gate_only_on_a_change_that_reaches_what_it_judges(world, capsys, monkeypatch):
    """The gate judges the vLLM integration, the repository's lints and lists and the root's ruff configs, so a change
    elsewhere, a ruff.toml below the root included, doesn't run it, whatever its tree holds."""
    from research.queue import PREFLIGHT_RATCHET, MarkOnly, StoreLabels, ready

    monkeypatch.setenv("RATCHET_SEEN", str(seen := world.tmp / "seen.json"))
    _preflight_main(world, {PREFLIGHT_RATCHET[0]: RATCHET_GATE})
    world.gh.put(2, h2 := world.change({"pkg/notes.txt": "x\n", "pkg/ruff.toml": "", "RATCHET_NEW": ""}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert not seen.exists() and not [s for s in world.logs if "warning" in s], world.logs
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def _marked(world: World, n: int, files: dict[str, str], base: str = "origin/main") -> str:
    """Open PR #``n`` changing ``files`` on ``base``, marked ready at its head on its owner's word: the head."""
    from research.queue import MarkOnly, ready

    world.gh.put(n, head := world.change(files, base=base), ready=False)
    assert ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), preflight=None) == 0
    return head


def _merges(monkeypatch) -> list[str]:
    """The heads `merged_tree` merges from here on, in order: the preflight's own merge of the head it marks, and each merge
    its read of the marked heads makes."""
    heads, real = [], Q.merged_tree
    monkeypatch.setattr(Q, "merged_tree", lambda repo, base, head, **kw: heads.append(head) or real(repo, base, head, **kw))
    return heads


def _on_main(world: World) -> str:
    """`main` as the verified queue lands onto it, with what the preflight runs there: its tip."""
    world.with_store_remote()
    return _move_main(world, {**PREFLIGHT_FILES, DRIVE_FLAGS: 'land_into = "main"\n'})


def test_the_preflight_names_each_marked_pr_a_head_doesnt_merge_with_as_a_warning_and_takes_the_mark(world, capsys):
    """Two marked heads that change the same lines each merge onto `main`, so neither mark is refused, and hours later the
    queue refuses whichever it merges second.  The mark names the other PR, its lane, the files (the first few of many) and
    the fix, and goes through; an open PR that isn't marked is no collision yet."""
    from research.queue import COLLIDE_SAYS, PREFLIGHT_COLLIDE_NAMED, MarkOnly, StoreLabels, ready

    main = _on_main(world)
    five = _marked(world, 5, {"a.txt": "a5\n", "b.txt": "b5\n"})
    many = [f"d/{i}.txt" for i in range(PREFLIGHT_COLLIDE_NAMED + 2)]
    seven = _marked(world, 7, dict.fromkeys(many, "7\n"))
    world.gh.put(6, world.change({"a.txt": "a6\n"}), ready=False)
    world.gh.put(2, h2 := world.change({"a.txt": "a2\n", "b.txt": "b2\n", **dict.fromkeys(many, "2\n")}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert [s for s in world.logs if "warning" in s] == [
        f"research queue: preflight: warning: #2 at {h2[:12]}: it doesn't merge with 2 marked PR(s), each merged onto `main` at "
        f"{main[:12]} and this head onto that, as a candidate holding both merges them:\n"
        f"    #5 (lane-5) at {five[:12]}, in a.txt, b.txt\n"
        f"    #7 (lane-7) at {seven[:12]}, in {', '.join(many[:PREFLIGHT_COLLIDE_NAMED])} and 2 more\n  {COLLIDE_SAYS}"], world.logs
    assert "wait for the PR named to land, merge `main` and mark the new head" in COLLIDE_SAYS
    assert "nothing refused" in next(s for s in world.logs if f"preflight: #2 at {h2[:12]}" in s)
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_the_preflight_merges_no_marked_head_that_shares_no_changed_path_with_the_head(world, monkeypatch, capsys):
    """A conflict needs a file both sides change, so only the marked heads that share a changed path with the head are merged:
    the read stays a few merges however many heads are marked."""
    from research.queue import MarkOnly, ready

    _on_main(world)
    five, six = _marked(world, 5, {"b.txt": "b5\n"}), _marked(world, 6, {"a.txt": "a\nboth\n", "c.txt": "c6\n"})
    world.gh.put(2, world.change({"a.txt": "a\nboth\n", "d.txt": "d2\n"}), ready=False)
    merged = _merges(monkeypatch)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert five not in merged and six in merged, merged                           # #6 shares a.txt, and merges
    assert not [s for s in world.logs if "warning" in s], world.logs


def test_the_preflight_merges_no_marked_head_the_head_contains_or_that_contains_it(world, monkeypatch, capsys):
    """A stack is no collision: the head merges with a marked head below it, and with one above it, by ancestry, and the
    queue lands them in that order.  Neither is merged, though each shares every path the lower one changes."""
    from research.queue import MarkOnly, ready

    _on_main(world)
    five = _marked(world, 5, {"a.txt": "a5\n"})
    world.gh.put(2, h2 := world.change({"a.txt": "a5\nand 2\n"}, base=five), ready=False)
    merged = _merges(monkeypatch)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert h2 in merged and five not in merged, merged                            # #2 contains #5's head
    del merged[:]
    assert ready(world.gh, world.store, 5, "lane-5", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    assert five in merged and h2 not in merged, merged                            # #2, marked by now, contains #5's
    assert not [s for s in world.logs if "warning" in s], world.logs


def test_the_read_of_the_marked_heads_names_what_its_own_budget_left_unread_and_takes_none_of_the_checks_time(
        world, monkeypatch, capsys):
    """The read has a budget of its own, read from the clock before each group of merges: past it the marked heads still to
    merge are named, every one, and not merged, the collision already found is still told, and the mark goes through.  The
    merged tree's checks get their whole budget from when the read ends, so a slow read never costs a lint its time."""
    from research.queue import COLLIDE_SAYS, PREFLIGHT_COLLIDE_NAMED, MarkOnly, Preflight, StoreLabels, ready

    main = _on_main(world)
    five, *rest = (_marked(world, n, {"a.txt": f"a{n}\n"}) for n in range(5, 7 + PREFLIGHT_COLLIDE_NAMED))
    world.gh.put(2, h2 := world.change({"a.txt": "a2\n"}), ready=False)
    now, real, deadlines = [0.0], Q.conflicting_files, []

    def slow(repo: Path, base: str, head: str) -> list[str]:
        now[0] += 1000.0                                                          # one merge outlasts the read's 50 s
        return real(repo, base, head)

    monkeypatch.setattr(Q, "conflicting_files", slow)
    monkeypatch.setattr(Preflight, "_checks", lambda self, *a: deadlines.append(a[5]) or {})
    merged = _merges(monkeypatch)
    pre = _preflight(world, budget_s=120.0, collide_s=50.0, collide_jobs=1, clock=lambda: now[0])
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=pre) == 0, capsys.readouterr().err
    assert five in merged and not set(rest) & set(merged), merged
    assert [s for s in world.logs if "warning" in s] == [
        f"research queue: preflight: warning: #2 at {h2[:12]}: it doesn't merge with 1 marked PR(s), each merged onto `main` at "
        f"{main[:12]} and this head onto that, as a candidate holding both merges them:\n"
        f"    #5 (lane-5) at {five[:12]}, in a.txt\n  {COLLIDE_SAYS}",
        f"research queue: preflight: warning: #2 at {h2[:12]}: whether it merges with these PRs wasn't read: "
        f"{', '.join(f'#{n}' for n in range(6, 7 + PREFLIGHT_COLLIDE_NAMED))} (not merged in time). The read has 50 s of its own "
        "and refuses nothing; the queue's merge judges each"], world.logs
    assert deadlines == [1000.0 + 120.0], deadlines
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is True and _no_worktree_left(world)


def test_a_read_of_the_marked_heads_that_cant_read_the_marks_says_so_and_takes_the_mark(world, capsys):
    """Without the store's remote the ready marks here may be old, so the read answers from none of them: it says why in a
    warning, and the mark goes through as it did before the read was there."""
    from research.queue import MarkOnly, ready

    _move_main(world, {**PREFLIGHT_FILES, DRIVE_FLAGS: 'land_into = "main"\n'})
    world.gh.put(5, world.change({"a.txt": "a5\n"}), ready=False)
    world.gh.put(2, h2 := world.change({"a.txt": "a2\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    [said] = [s for s in world.logs if "warning" in s]
    assert said.startswith(f"research queue: preflight: warning: #2 at {h2[:12]}: the marked PRs it doesn't merge with weren't "
                           "read (QueueError: STALE: the labels of pr:5@") and said.endswith("the queue's merge judges each"), said


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
    assert (f"it contains #5's head {five[:12]}, and #5 isn't admitted (contains #6, which isn't admitted), so the verified "
            f"queue holds this PR behind it") in err and "#6's head" not in err
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


def test_the_preflight_never_holds_a_head_behind_a_contained_head_on_the_landing_branch(world, capsys):
    """Under `land_into = "checked"` the preflight judges on `main`, which a PR landed on `checked` hasn't reached: that PR, still
    open and here held, holds no head containing it.  A contained PR that hasn't landed and isn't admitted still does."""
    from research.queue import MarkOnly, ready

    world.with_store_remote()
    _move_main(world, {"tools/verity/check/drive.toml": 'land_into = "checked"\n'})
    world.gh.put(5, five := world.change({"x.txt": "x\n"}), labels=("hold",), ready=False)
    _git(world.dev, "push", "-q", "origin", f"{five}:refs/heads/checked")
    world.gh.put(4, world.change({"y.txt": "y\n"}, base=five), ready=False)
    pre = _preflight(world)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 0, capsys.readouterr().err
    world.gh.put(6, six := world.change({"z.txt": "z\n"}, base=five), labels=("hold",), ready=False)
    world.gh.put(3, world.change({"w.txt": "w\n"}, base=six), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert f"it contains #6's head {six[:12]}, and #6 isn't admitted (held (`hold` label))" in err and "#5's" not in err, err


def test_the_preflight_refuses_a_head_containing_an_open_pr_that_isnt_admitted_and_names_it(world, capsys):
    """Drive holds a PR that contains one never marked, or one whose mark was withdrawn; the preflight refuses it first."""
    from research.queue import MarkOnly, ready

    world.with_store_remote()
    world.gh.put(5, parent := world.change({"x.txt": "x\n"}), ready=False)
    world.gh.put(4, world.change({"y.txt": "y\n"}, base=parent), ready=False)
    pre = _preflight(world)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert (f"it contains #5's head {parent[:12]}, and #5 isn't admitted (not marked ready at {parent[:8]}), so the verified "
            f"queue holds this PR behind it: mark #5 ready first, so it lands first, or drop #5's commits from this branch (if this PR "
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


def test_the_preflight_refuses_a_head_that_merged_next_carrying_a_pr_head_its_pr_no_longer_lands(world, capsys):
    """Branches that merged an older `next` carried the heads of PRs rebuilt or withdrawn since, which would land under their
    marks.  A head carried through `next`'s merge (`Queue: #N`) refuses the mark, with the fix, when #N's current head doesn't
    hold it, #N is closed (unlanded, the merge of an old `next`: `old_next`'s refusal, alone), or #N isn't admitted, even when
    #N's owner withdrew it, which carries it for a rebuild; one #N still lands as it stands refuses nothing, nor does one the
    landing branch holds (an older merge of a PR landed since)."""
    from research.queue import MarkOnly, carried_heads, ready
    from research.store.cli import _label_write_through

    world.with_store_remote()
    world.gh.put(5, five := world.change({"x.txt": "x\n"}))
    world.gh.put(6, six := world.change({"y.txt": "y\n"}))
    assert [m.pr for m in world.queue().sync()[0]] == [5, 6]
    tip = _git(world.origin, "rev-parse", "refs/heads/next")
    by = {n: _git(world.origin, "log", "-1", "--format=%H", f"--grep=^Queue: #{n} ", tip) for n in (5, 6)}
    world.gh.put(4, four := world.change({"z.txt": "z\n"}, base=tip), ready=False)
    world.gh.put(5, new := world.change({"x.txt": "x2\n"}), ready=False)                   # #5 rebuilt on `main`
    for n in (5, 6):
        assert ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), preflight=None) == 0
    pre = _preflight(world)
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert (f"it carries #5's head {five[:12]} through `next`'s merge {by[5][:12]}, which neither `main` nor #5's current head "
            f"{new[:12]} holds: merge origin/main and #5's current head instead of `next`, or drop #5's commits") in err, err
    assert "#6's" not in err                                                     # #6 lands at the head #4 carries
    del world.gh.prs[5]
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    assert capsys.readouterr().err == (
        f"research queue ready: #4 is not marked ready at {four[:12]}: its head carries 1 merge of an old `next` that `main` "
        f"doesn't hold, of PRs closed unlanded ({by[5][:12]} (#5)), {OLD_NEXT_FIX}\n")
    world.gh.put(5, new, ready=False)
    world.store.add_label(label_target(6, six), vocab.READY, False, "lane-6")
    _label_write_through(world.store, label_target(6, six))
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=pre) == 1
    err = capsys.readouterr().err
    assert (f"it carries #6's head {six[:12]} through `next`'s merge {by[6][:12]}, and #6 isn't admitted at {six[:12]} (not "
            f"marked ready at {six[:8]}): merge origin/main and #6's current head instead of `next`, or drop #6's commits") in err
    assert "it contains #6's head" not in err                                   # a withdrawn PR a rebuild would carry
    assert [n for n, _, _ in carried_heads(world.dev, five, four)] == [6]       # #5's head on the landing branch


#: Lake packages as the Lean step reads them: `lean/a` (library `A`), `lean/a/Proofs` nested in it with `srcDir = ".."` and its
#: library's roots listed (as security/Proofs'), `lean/b` requiring `lean/a` by path, and `lean/c`, which doesn't
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


def test_the_lean_step_skips_past_its_bounds_and_refuses_what_it_cannot_build(world, monkeypatch, capsys):
    """A change more modules import than the preflight builds, and a spent budget, leave the break to drive's Lean check: said
    as a skipped check, never as nothing refused, and never refused.  A head whose few modules nothing builds (no warm tree and
    no node to build on, a node without Mathlib for the package) is refused with the fix: an unbuilt module is how a head that
    breaks `next`'s Lean got marked."""
    from research.queue import LakeBuild, MarkOnly, StoreLabels, ready

    seen, slots = _lean_world(world, monkeypatch)
    cases = [
        (LakeBuild(slots=slots), {"lean_modules": 5}, "6 modules are or import what it changes, over the 5", False, False),
        (LakeBuild(slots=slots), {"lean_s": 0}, "lean build timed out (6 module(s) not built: the Lean step's budget was spent",
         False, False),
        (LakeBuild(slots=world.tmp / "absent"), {}, "this machine keeps no warm Lean tree (no `build` pool in "
         f"{world.tmp / 'absent'}; an agent VM builds nothing) and was given no node to build on: `ready --on MACHINE` builds "
         "them on vy-nebius-cpu-9", False, True),
        (LakeBuild(slots=slots), {}, "lean/a not built: no Mathlib here: mark again once this node builds them", True, True),
    ]
    for n, (lean, kw, says, built, refused) in enumerate(cases, start=2):
        if built:
            monkeypatch.setenv("FAKE_LEAN_SKIP", "no Mathlib here")
        seen.unlink(missing_ok=True)
        world.gh.put(n, h := world.change({"lean/a/A/Base.lean": "def bar := 1\n", f"n{n}.txt": "x\n"}), ready=False)
        world.logs.clear()
        rc = ready(world.gh, world.store, n, f"lane-{n}", MarkOnly(), preflight=_preflight(world, lean=lean, **kw))
        err = capsys.readouterr().err
        assert seen.exists() is built and StoreLabels(world.store, log=world.logs.append).ready(n, h) is (not refused)
        if refused:
            found = [ln for ln in err.splitlines() if ln.startswith("  - ")]
            assert rc == 1 and len(found) == 1, err
            assert found[0].startswith(f"  - Lean: 6 module(s) that are or import what it changes couldn't be built: {says}"), err
            continue
        assert rc == 0, err
        said = next(s for s in world.logs if f"preflight: #{n} at" in s)
        assert "checks skipped: " in said and "nothing refused" not in said and says in said, said
        assert said.index("checks skipped: ") < said.index(says), said


def test_ready_mark_from_an_agent_vm_builds_the_lean_it_reaches_on_a_node_refuses_what_none_can_and_skips_a_timeout(
        world, monkeypatch, capsys):
    """`ready --mark` from a VM without a warm Lean tree used to skip the Lean step, so a head whose merge onto `next` broke 5
    modules was marked.  The step ships the merged tree to `PREFLIGHT_LEAN_ON` whatever the tier (a stand-in for `research
    run` runs the stand-in builder in the shipped tree) and refuses the break; when no node starts the run, it refuses with
    each node's reason and the fix; one that hasn't ended by the deadline marks, said as 'checks skipped: lean build timed
    out'."""
    from research.queue import PREFLIGHT_LEAN_ON, LakeBuild, MarkOnly, StoreLabels, ready

    seen, _ = _lean_world(world, monkeypatch)
    rid, slow, runs, launched, mode = "r20261030-120000-abcd", "r20261030-120000-beef", world.tmp / "runs", [], {}
    first, second = PREFLIGHT_LEAN_ON

    def run(argv, cwd, left):
        if argv[3] == "fetch":
            return 0, ""
        launched.append(argv[3:])
        node = argv[argv.index("--on") + 1]
        if mode["now"] == "unreachable":
            return 1, f"research run: {node}: ssh: connect to host: Connection refused\n"
        if mode["now"] == "slow":
            return None, f"research: run {slow} launched on {node} (pod) pid=1 -> /r\n"
        source, tail = Path(argv[argv.index("--source") + 1]), argv[argv.index("--") + 2:]
        p = subprocess.run([sys.executable, *tail], cwd=source, capture_output=True, text=True)
        (runs / rid).mkdir(parents=True, exist_ok=True)
        (runs / rid / "stdout.log").write_text(p.stdout)
        return p.returncode, f"research: run {rid} launched on {node} (pod) pid=1 -> /r\n"

    lean = LakeBuild(on=PREFLIGHT_LEAN_ON, by="lane-2", slots=world.tmp / "absent", run=run, runs=lambda: runs,
                     clock=lambda: 0.0)
    mode["now"] = "builds"
    world.gh.put(2, h2 := world.change({"lean/a/A/Base.lean": "def bar := 1\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world, lean=lean)) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [
        "  - Lean: `B.Use` (lean/b/B/Use.lean), of #1 on `next`, fails to build with this head merged: error: lean/b/B/Use.lean:2:0: "
        "unknown identifier 'foo': merge or stack on that PR, never merge `next` (it carries unlanded PRs)"], err
    assert [a[:3] for a in launched] == [["run", "--on", first]] * 2, "the merge, then what failed, on `next`"
    assert [json.loads(ln) for ln in seen.read_text().splitlines()][1] == ["lean/b/B/Use.lean"]
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False

    mode["now"], launched[:] = "unreachable", []
    world.gh.put(3, h3 := world.change({"lean/a/A/Base.lean": "def baz := 1\n"}), ready=False)
    assert ready(world.gh, world.store, 3, "lane-3", MarkOnly(), preflight=_preflight(world, lean=lean)) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [
        "  - Lean: 6 module(s) that are or import what it changes couldn't be built: "
        + "; ".join(f"its build on {n} exited 1 without a result line: research run: {n}: ssh: connect to host: Connection "
                    "refused" for n in PREFLIGHT_LEAN_ON)
        + f": mark again once {first} or {second} builds them in its `build` Lean slots (`research status` there says "
          "why not)"], err
    assert [a[2] for a in launched] == [first, second]
    assert StoreLabels(world.store, log=world.logs.append).ready(3, h3) is False and _no_worktree_left(world)

    mode["now"], launched[:] = "slow", []
    world.gh.put(4, h4 := world.change({"lean/a/A/Base.lean": "def qux := 1\n"}), ready=False)
    world.logs.clear()
    assert ready(world.gh, world.store, 4, "lane-4", MarkOnly(), preflight=_preflight(world, lean=lean)) == 0, \
        capsys.readouterr().err
    said = next(s for s in world.logs if "preflight: #4 at" in s)
    assert "checks skipped: lean build timed out (6 module(s) not built: its build on " \
           f"{first} (run {slow}) hadn't ended after" in said and "nothing refused" not in said, said
    assert [a[2] for a in launched] == [first], "time spent is spent: no second node"
    assert StoreLabels(world.store, log=world.logs.append).ready(4, h4) is True and _no_worktree_left(world)


def test_the_lean_step_falls_back_to_the_next_node_when_one_cannot_start_the_build_and_refuses_when_none_can(tmp_path):
    """One node (cpu-3) that can't start a run (its `build` slots full, drained, unreachable) would refuse every Lean mark: each
    node but the last takes its slot with `--no-wait`, and a node that can't start passes the build to the next, within the
    one deadline; only when none builds is the mark refused, naming each node's reason and the fix. Each launch's budget carries
    its unit, as `research run` refuses a bare number under five minutes for a machine."""
    from research.cli import _short_bare_timeout
    from research.queue import LEAN_BUILDER, PREFLIGHT_LEAN_ON, LakeBuild

    assert (PREFLIGHT_LEAN_ON, LakeBuild().pool) == (("vy-nebius-cpu-9", "vy-nebius-cpu-5"), None)
    first, second = PREFLIGHT_LEAN_ON
    (tmp_path / "wt" / LEAN_BUILDER).parent.mkdir(parents=True)
    (tmp_path / "wt" / LEAN_BUILDER).write_text("")
    full = "lean-build: " + json.dumps({"built": {}, "failed": {}, "skipped": {
        "*": "no `build` Lean slot here (all 8 Lean audit slot(s) of pool build are taken, and this request doesn't wait in line)"}})
    built = "lean-build: " + json.dumps({"built": {"lean/b": 0}, "failed": {}, "skipped": {}})
    drained = f"research: run {{rid}} refused on {second}: the node is drained\n"
    for k, says in enumerate(({first: (2, full), second: (0, built)}, {first: (255, None), second: (1, drained)})):
        argvs, now, runs, timeouts = [], [0.0], tmp_path / f"runs-{k}", []

        def run(argv, cwd, left, argvs=argvs, says=says, now=now, runs=runs, timeouts=timeouts):
            if argv[3] == "fetch":
                return 0, ""
            node = argv[argv.index("--on") + 1]
            argvs.append((node, argv[argv.index("--") + 3:], left))
            timeouts.append(argv[argv.index("--timeout") + 1])
            now[0] += 30                                               # each launch spends 30 s of the budget
            rc, out = says[node]
            if out is None:
                return rc, f"ssh: connect to host {node} port 22: Connection timed out\n"
            rid = f"r20261030-1900{len(argvs):02d}-abcd"
            if out.startswith("lean-build: "):
                (runs / rid).mkdir(parents=True, exist_ok=True)
                (runs / rid / "stdout.log").write_text(out + "\n")
                return rc, f"research: run {rid} launched on {node} (pod) pid=1 -> /r\n"
            return rc, out.format(rid=rid)

        got = LakeBuild(on=PREFLIGHT_LEAN_ON, by="ci", slots=tmp_path / "absent", run=run, runs=lambda runs=runs: runs,
                        clock=lambda now=now: now[0])(tmp_path / "wt", ["lean/b/B/Use.lean"], 100)
        assert [(n, tail[:2], left) for n, tail, left in argvs] == [(first, ["--no-wait", "--build"], 100),
                                                                     (second, ["--build", "lean/b/B/Use.lean"], 70)]
        assert [_short_bare_timeout(["--timeout", t]) for t in timeouts] == [None, None], "`research run` refuses it"
        if says[second][0] == 0:
            assert got.failed == {} and got.notes == [] and got.where == f"on {second} (run r20261030-190002-abcd)"
            assert got.passed == [f"* not built on {first} (run r20261030-190001-abcd): no `build` Lean slot here (all 8 Lean "
                                  "audit slot(s) of pool build are taken, and this request doesn't wait in line)"]
            continue
        assert isinstance(got, str) and got == (
            f"its build on {first} exited 255 without a result line: ssh: connect to host {first} port 22: Connection timed "
            f"out; its build on {second} (run r20261030-190002-abcd) exited 1 without a result line: research: run "
            f"r20261030-190002-abcd refused on {second}: the node is drained: mark again once {first} or {second} builds them in its `build` Lean slots (`research status` there says why not)"), got


def test_the_lean_step_ships_the_merged_tree_to_a_node_for_ready_on_and_reads_its_result(tmp_path):
    """Off a node, the Lean step builds as `verity lean try --on vy-nebius-cpu-3` does: a `research run` of the tree on a node of
    `PREFLIGHT_LEAN_ON`, whose `build` Lean slot is the gate (a pool only when given), bounded by the step's budget, whose stdout
    carries the result line; the last node waits in its slot's line (no `--no-wait`)."""
    from research.queue import LEAN_BUILDER, PREFLIGHT_LEAN_ON, LakeBuild

    node = PREFLIGHT_LEAN_ON[0]
    (tmp_path / "wt" / LEAN_BUILDER).parent.mkdir(parents=True)
    (tmp_path / "wt" / LEAN_BUILDER).write_text("")
    rid = "r20261026-120000-abcd"
    line = "lean-build: " + json.dumps({"built": {"lean/b": 1}, "failed": {"lean/b/B/Use.lean": "error: B/Use.lean:2:0: x"},
                                        "skipped": {}})
    tail = ["--as", "ci", "--project", "verity", "--source", str(tmp_path / "wt"), "--cwd", "clone", "--timeout", "90s", "--wait",
            "--wait-every", "5", "--wait-max", "90", "--", "python3", LEAN_BUILDER, "--build", "lean/b/B/Use.lean"]
    for pool, flags in ((None, []), ("research", ["--pool", "research", "--pool-wait", "90"])):
        argvs = []

        def run(argv, cwd, left, argvs=argvs):
            argvs.append((argv[3:], left))
            if argv[3] == "fetch":
                (tmp_path / "runs" / rid).mkdir(parents=True, exist_ok=True)
                (tmp_path / "runs" / rid / "stdout.log").write_text(f"lean-build: lean/b ...\n{line}\n")
            return 1, f"research: run {rid} launched on {node} (pod) pid=1 -> /r\n"

        got = LakeBuild(on=(node,), pool=pool, by="ci", slots=tmp_path / "absent", run=run, runs=lambda: tmp_path / "runs",
                        clock=lambda: 0.0)(tmp_path / "wt", ["lean/b/B/Use.lean"], 100)
        assert got.failed == {"lean/b/B/Use.lean": "error: B/Use.lean:2:0: x"} and got.notes == []
        (launch, left), (fetch, _) = argvs
        assert left == 100 and fetch == ["fetch", rid, "--on", node, "--all"]
        assert launch == ["run", "--on", node, *flags, *tail]


def test_research_queue_ready_runs_the_preflight_unless_no_preflight(world, monkeypatch, capsys):
    from research import queue as Q

    seen, gates, goal_gates = [], [], []
    monkeypatch.setattr(Q, "ready", lambda *a, preflight=None, records=None, goals=None, **kw:
                        seen.append(preflight) or gates.append(records) or goal_gates.append(goals) or 0)
    monkeypatch.chdir(world.dev)
    base = ["queue", "ready", "3", "--mark", "--by", "lane-3", "--remote", str(world.origin), "--repo", "o/r"]
    assert research_main(base) == 0 and research_main([*base, "--no-preflight"]) == 0
    assert research_main([*base[:3], "--local", *base[4:]]) == 0
    assert isinstance(seen[0], Q.Preflight) and seen[1] is None
    assert [(p.lean.on, p.lean.by) for p in (seen[0], seen[2])] == [(Q.PREFLIGHT_LEAN_ON, "lane-3")] * 2, \
        "--mark and --local ship the Lean step to a node too"
    assert all(isinstance(g, Q.RecordsGate) for g in gates) and len(gates) == 3        # the records gate isn't the preflight's
    assert goal_gates == [Q.goal_gate] * 3                                               # nor is the goal gate
    with pytest.raises(SystemExit) as e:
        research_main(["queue", "status", "--no-preflight"])
    assert e.value.code == 2 and "--no-preflight is for ready only" in capsys.readouterr().err


def test_research_queue_judge_marks_through_the_records_and_goal_gates(world, monkeypatch):
    from research import queue as Q

    got = []
    monkeypatch.setattr(Q, "judge", lambda *a, records=None, goals=None, **kw: got.append((records, goals)) or 0)
    monkeypatch.chdir(world.dev)
    assert research_main(["queue", "judge", "r20261004-160000-abcd", "--by", "lane-1", "--remote", str(world.origin), "--repo",
                          "o/r"]) == 0
    [(records, goals)] = got
    assert isinstance(records, Q.RecordsGate) and goals is Q.goal_gate


LOCK = "pkg/lean-audit.json"


def _lock(**entries: str) -> str:
    """A lock with one entry a line, so git merges two sides' changes to entries apart."""
    return json.dumps({k: entries.get(k, "1") for k in "ABCDE"}, indent=1) + "\n"


ROOT = Path(__file__).resolve().parents[3]
#: what `lean_changed.py --inputs-key` runs, which `_keyed` commits into the origin so the records gate keys a commit's Lean
#: inputs with that commit's own code (`lean_keys`)
KEY_FILES = ("tools/verity/lean/lean_changed.py", "tools/verity/lean/lean_audit.py", "tools/verity/lean/module_keys.py",
             "tools/verity/lean/headers.py", "tools/verity/lean/check.py", "tools/verity/lean/siblings.py",
             "tools/verity/lean/axiom_probe.py", "tools/verity/lean/settings.py", "tools/verity/check/common.py",
             "tools/verity/check/slot.py", "tools/verity/check/lean_slot.py", "tools/verity/graph/verity_graph/contract.py")
#: two Lake packages, `pkg` requiring `lib` by path, each with its toolchain
LAKE = {"pkg/lakefile.toml": 'name = "pkg"\n', "pkg/lean-toolchain": "v1\n",
        "pkg/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": "../lib", "name": "lib"}]}),
        "lib/lakefile.toml": 'name = "lib"\n', "lib/lean-toolchain": "v1\n", "lib/L.lean": "def l := 1\n"}


def _keyed(world: World, files: dict[str, str] | None = None) -> str:
    """`main` moved to a commit holding the key code (`KEY_FILES`, as this checkout has it), `LAKE` and ``files``."""
    return _move_main(world, {**{f: (ROOT / f).read_text() for f in KEY_FILES}, **LAKE, **(files or {})})


def _key(repo: Path, commit: str, pkg: str = "pkg") -> str:
    """``pkg``'s Lean inputs key at ``commit`` of ``repo``, as `lean_changed.py --inputs-key` prints it."""
    r = subprocess.run([sys.executable, str(ROOT / KEY_FILES[0]), "--inputs-key", pkg, "--head", commit, "--repo", str(repo)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)[pkg]


def _published(world: World, commit: str, *, pkg: str = "pkg", passed: bool = True, passes: dict[str, str] | None = None,
               records: dict | None = None, steps: list | None = None, kind: str | None = None, at: float | None = None,
               argv: list[str] | None = None, state: str | None = None, rc: int | None = None, rid: str | None = None) -> str:
    """A run of ``commit`` that another machine published to the store's remote, as `research run` does: a records run
    (`lean_changed.py --records PKG`, result `lean-fast/v1` whose `records` are ``records``, else ``pkg``'s outcome) or, with
    ``steps``, a check (`check.py`, result ``kind``, else `check/v1`), its result listing ``passes`` ({Lean inputs key:
    package}); ``rid`` its id, else a new one started ``at``, else now; ``argv`` its command; ``state`` and ``rc`` its
    attempt's end, else done and 0 when it ``passed``, failed and 1 when not."""
    from research.runs import new_run_id

    node = LocalStore(world.tmp / "node", config_path=world.tmp / "store.toml")
    if rid is None:
        rid = new_run_id(at)
        while node.attempt_path(rid).exists():  # an id is a second and 16 random bits: runs published in one second can share one
            rid = new_run_id(at)
    check = steps is not None
    kind = kind or ("check/v1" if check else "lean-fast/v1")
    meta = {"schema": "research/result/v0.1", "kind": kind, "run_id": rid, "validation": {"status": "passed" if passed else "failed"},
            "passes": passes or {},
            **({"steps": steps} if check else {"records": records or {pkg: {"outcome": "pass" if passed else "failed"}}})}
    result = node.put_artifact(kind, meta)
    node.put_attempt({
        "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "python3", "version": None}, "derivation": "drv:" + "0" * 64,
        "reuse_key": None, "argv": argv or (["python3", "tools/verity/check/check.py", "--record"] if check else
                                            ["python3", "tools/verity/lean/lean_changed.py", "--records", pkg, "--update"]),
        "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
        "source": {"commit": commit, "dirty": False}, "closure": None, "execution": {}, "cost": None,
        "status": {"state": state or ("done" if passed else "failed"), "rc": rc if rc is not None else 0 if passed else 1},
        "outputs": {"result": result}, "telemetry": None})
    node.push(result)
    node.push_attempt(rid)
    return rid


def _gate(world: World, store=None, now: Callable[[], float] | None = None):
    from research.queue import RecordsGate

    return RecordsGate(world.dev, str(world.origin), store or world.store, log=world.logs.append,
                       lock_merge=world.tmp / "lock_merge.py", **({"now": now} if now else {}))


def _mark(world: World, n: int, head: str, store=None, now: Callable[[], float] | None = None) -> int:
    """`ready N --mark` at ``head`` through a records gate: 1 when the gate warned, 0 when it vouched for each lock the head
    changes.  It never refuses, so the head is marked either way (in ``store``, whether or not its remote takes the label)."""
    from research.queue import MarkOnly, ready

    world.gh.put(n, head, ready=False)
    store, gate, said = store or world.store, _gate(world, store, now), []
    marks = lambda: len(store.labels(label_target(n, head), key=vocab.READY))     # noqa: E731
    before = marks()
    ready(world.gh, store, n, f"lane-{n}", MarkOnly(), records=lambda pr: said.append(gate(pr)) or said[-1])
    assert marks() == before + 1, said
    return int(said[0] is not None)


def test_a_head_that_changes_a_lock_is_vouched_for_only_by_a_pass_on_its_lean_inputs_and_a_warning_names_a_failed_run(
        world, capsys):
    """A records run that failed a package still writes a record, which reads as a pass, and vouches for nothing.  Only a
    result's `passes`, the Lean inputs key of each package it passed, vouches: with no run the warning names the command to
    start; with a failing one, that run too."""
    from research.queue import RECORDS_COMMAND, StoreLabels

    world.with_store_remote()
    _keyed(world, {LOCK: _lock()})
    h = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    capsys.readouterr()
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert f"research queue ready: warning: #1 at {h[:12]}: it changes the Lean lock of pkg against `main`" in err
    assert f"the store holds no pass of it on {h[:12]}'s Lean inputs" in err
    assert f"start `{RECORDS_COMMAND.format(pkg='pkg')}`" in err
    assert ("research run --on auto --mem-gb 32 --max-min 120 --project verity --source . --cwd clone --declared-output "
            "'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records pkg --update") in err
    failed = _published(world, h, passed=False)
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert f"records run {failed} of {h[:12]} failed pkg (`research status {failed}`" in err
    assert RECORDS_COMMAND.format(pkg="pkg") in err
    assert StoreLabels(world.store, log=world.logs.append).ready(1, h) is True

    key = _key(world.dev, h)
    passed = _published(world, h, passes={key: "pkg"})
    assert _mark(world, 1, h) == 0
    assert (f"records gate: pkg: records run {passed} of {h[:12]} passed it on {h[:12]}'s Lean inputs (key {key[:12]})"
            in world.logs[-1])
    assert StoreLabels(world.store, log=world.logs.append).ready(1, h) is True

    plain = world.change({"a.txt": "no lock\n"})                     # no lock changed: nothing is read, not even the store
    assert _mark(world, 2, plain, store=LocalStore(world.tmp / "bare", config_path=world.tmp / "none.toml")) == 0


def test_a_head_that_changes_a_lock_no_records_run_passed_is_marked_and_the_warning_names_each_package(world, capsys):
    """The lock run after a landing regenerates the records of the locks it changed, so a head no records run vouches for is
    marked all the same: `ready` exits 0, the mark is on the store's remote, and the warning names each package, the run that
    would record it, and why it doesn't refuse."""
    from research.queue import RECORDS_AFTER_LANDING, RECORDS_COMMAND, MarkOnly, StoreLabels, ready

    world.with_store_remote()
    h = world.change({LOCK: _lock(A="2"), "lib/lean-audit.json": _lock(B="2")})
    world.gh.put(1, h, ready=False)
    capsys.readouterr()
    assert ready(world.gh, world.store, 1, "lane-1", MarkOnly(), records=_gate(world)) == 0
    out, err = capsys.readouterr()
    assert err.startswith(f"research queue ready: warning: #1 at {h[:12]}: it changes the Lean lock of lib, pkg against `main`, "
                          f"and the store holds no pass of each on {h[:12]}'s Lean inputs"), err
    assert f": {RECORDS_AFTER_LANDING}:\n  - " in err
    for pkg in ("lib", "pkg"):
        assert f"to record {pkg} before it lands, start `{RECORDS_COMMAND.format(pkg=pkg)}`" in err
    assert "is not marked" not in err and f"#1 at {h[:12]}: marked on lane-1's word, with no quick tier" in out
    assert StoreLabels(world.store, log=world.logs.append).ready(1, h) is True


def test_a_lock_of_the_auditors_fixtures_needs_no_records_run(world):
    """No records run reads a lock under `LEAN_FIXTURES` (`lean_changed.py --records` matches no package there, and the
    controls write its records), so a head that changes only such locks is marked with nothing read, not even the store."""
    fixture = world.change({"tools/verity/lean/tests/fixtures/f/lean-audit.json": _lock(A="2")})
    assert _mark(world, 1, fixture, store=LocalStore(world.tmp / "bare", config_path=world.tmp / "none.toml")) == 0


def test_under_land_into_checked_a_head_is_judged_against_checked_not_main(world, capsys):
    """Under `land_into = "checked"` a lock change lands on `checked` well before `main` follows it, and a PR that merges
    `checked` (to resolve a conflict with one landed there) carries it: against `main` the gate blamed that PR for every lock
    the landings changed.  Against `checked` it judges the PR's own change, so a lock change of its own still samples the
    warning, naming `checked`."""
    world.with_store_remote()
    _move_main(world, {"tools/verity/check/drive.toml": 'land_into = "checked"\n'})
    landed = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    _git(world.dev, "push", "-q", "origin", f"{landed}:refs/heads/checked")
    pr = world.change({"x.txt": "x\n"})
    _git(world.dev, "checkout", "-q", "--detach", pr)
    _git(world.dev, "merge", "-q", "--no-ff", "--no-edit", landed)
    merged = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{merged}:refs/heads/pr-1")
    capsys.readouterr()
    assert _mark(world, 1, merged) == 0, capsys.readouterr().err
    own = world.change({LOCK: _lock(A="2", B="3")}, base=landed)
    assert _mark(world, 2, own) == 1
    assert f"warning: #2 at {own[:12]}: it changes the Lean lock of pkg against `checked`" in capsys.readouterr().err


def test_a_store_put_and_label_made_while_a_marks_preflight_runs_take_the_catalog_without_waiting_for_it(world):
    """A mark refreshes the store's attempts from the remote (its records gate) and then runs its preflight (a merge, the Lean
    gate, the suites' collection: minutes), while the machine's other lanes write the same catalog: `research data put`, its
    write-through and `research data label`.  The mark holds the catalog only around its own short reads and writes, so theirs
    take it at once while the preflight runs; their connection waits for no lock (busy timeout 0), so one held fails them here."""
    from research.queue import MarkOnly, StoreLabels, ready

    world.with_store_remote()
    _keyed(world, {LOCK: _lock()})
    h = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    passed = _published(world, h, passes={_key(world.dev, h): "pkg"})    # what the records gate refreshes, and passes on
    world.gh.put(1, h, ready=False)
    inside, release, preflighted, rc = threading.Event(), threading.Event(), [], []

    def preflight(pr: PR, prs: list[PR]) -> str | None:
        preflighted.append(pr.head)
        inside.set()
        release.wait()
        return None

    def mark() -> None:
        try:
            rc.append(ready(world.gh, world.store, 1, "lane-1", MarkOnly(), preflight=preflight, records=_gate(world)))
        finally:
            inside.set()

    t = threading.Thread(target=mark)
    try:
        t.start()
        inside.wait()
        assert preflighted == [h], f"the mark ended before its preflight ran: {world.logs}"
        other = LocalStore(world.tmp / "store", config_path=world.tmp / "store.toml")      # another lane's process on the machine
        other.index.db.execute("PRAGMA busy_timeout = 0")
        (world.tmp / "checkpoint.json").write_text('{"step": 1}\n')
        art = other.put_artifact("evidence/v1", {"step": 1}, file=world.tmp / "checkpoint.json")
        assert other.push(art)["preserved"] is True
        other.add_label(art, vocab.NOTE, "kept", "lane-9")
        assert not rc, "the preflight is still running"
    finally:
        release.set()
        t.join()
    assert rc == [0] and f"records gate: pkg: records run {passed} of {h[:12]} passed it" in world.logs[-1]
    assert StoreLabels(world.store, log=world.logs.append).ready(1, h) is True
    assert other.index.artifact(art)["remote"] is True
    assert [x["value"] for x in other.index.labels(art, vocab.NOTE)] == ["kept"]


def test_a_result_whose_attempt_cant_be_read_vouches_for_nothing_and_the_warning_says_so(world, capsys):
    """The records gate reads the results of the attempts that may list passes, picked by the command in the index's manifest of
    each; a row that doesn't parse is skipped, which vouches for nothing, and the warning says how many were and which, rather
    than warning with no trace of why."""
    world.with_store_remote()
    _keyed(world, {LOCK: _lock()})
    h = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    rid = _published(world, h, passes={_key(world.dev, h): "pkg"})
    world.store.refresh(attempts_only=True)
    with world.store.index.db as db:
        db.execute("UPDATE attempts SET manifest_json = '{\"argv\": [\"lean_changed.py' WHERE id = ?", (rid,))
    capsys.readouterr()
    assert _mark(world, 1, h) == 1
    assert f"1 result(s) that may list passes couldn't be read, e.g. {rid}: its manifest can't be read (JSONDecodeError)" in \
        capsys.readouterr().err


def test_a_pass_on_the_heads_lean_inputs_at_any_commit_vouches_for_it_and_a_change_to_a_file_they_hold_warns(world, capsys):
    """A PR that carries another PR's Lean change needn't rerun records at its own head, hours of queue on a full node: a pass
    whose Lean inputs key is the head's vouches for it wherever it ran, here a commit on no branch the gate could walk to.  The
    key holds every file of the package and of each package it requires by path, their toolchains and Lean tests included, and
    the checker's, so a change to any of them samples the warning; not the files no audit reads (`lean_audit.UNREAD_NAMES`: a
    README, a pytest file), so a change to one of those keeps the pass."""
    world.with_store_remote()
    _keyed(world, {LOCK: _lock(), "lib/tests/T.lean": "def t := 1\n", "lib/README.md": "lib\n"})
    other = world.tmp / "other"
    _git(world.tmp, "clone", "-q", str(world.origin), str(other))
    (other / LOCK).write_text(_lock(A="2"))
    (other / "c.txt").write_text("the other PR's\n")
    _git(other, "commit", "-q", "-am", "the other PR's Lean change")
    c = _git(other, "rev-parse", "HEAD")
    rid = _published(world, c, passes={_key(other, c): "pkg"})

    h = world.change({LOCK: _lock(A="2"), "a.txt": "carries it\n"})
    capsys.readouterr()
    assert _mark(world, 1, h) == 0, capsys.readouterr().err
    assert f"records gate: pkg: records run {rid} of {c[:12]} passed it on {h[:12]}'s Lean inputs" in world.logs[-1]

    changed = lambda f: (ROOT / f).read_text() + "# changed\n"                   # noqa: E731
    for n, files in ((2, {"lib/L.lean": "def l := 2\n"}), (3, {"pkg/lean-toolchain": "v2\n"}), (4, {"lib/lean-toolchain": "v2\n"}),
                     (5, {"lib/lakefile.toml": 'name = "lib"\ndefaultTargets = ["L"]\n'}), (6, {"lib/tests/T.lean": "def t := 2\n"}),
                     (9, {"tools/verity/lean/check.py": changed("tools/verity/lean/check.py")}),
                     (10, {"tools/verity/check/common.py": changed("tools/verity/check/common.py")})):
        head = world.change({LOCK: _lock(A="2"), **files})
        capsys.readouterr()
        assert _mark(world, n, head) == 1, files
        assert f"the store holds no pass of it on {head[:12]}'s Lean inputs" in capsys.readouterr().err
    for n, files in ((7, {"lib/README.md": "lib, said again\n"}), (8, {"pkg/tests/test_t.py": "def test_t(): pass\n"})):
        head = world.change({LOCK: _lock(A="2"), **files})
        capsys.readouterr()
        assert _mark(world, n, head) == 0, (files, capsys.readouterr().err)
        assert f"records gate: pkg: records run {rid} of {c[:12]} passed it on {head[:12]}'s Lean inputs" in world.logs[-1]


BOTH = {LOCK: _lock(A="2"), "pkg/Proofs/lean-audit.json": _lock(B="2")}


def test_a_run_vouches_for_exactly_the_packages_its_result_lists_as_passed_on_the_heads_lean_inputs(world, capsys):
    """A records run of `lib` that failed `lib` passed its requirer `pkg` all the same, a shard of a check audits only its own
    parts, and a check whose `lean-audit` step passed may have passed a package only by regenerating its lock: each vouches for a
    package only through its result's `passes`, the Lean inputs key of each package it passed on the commit's own lock."""
    world.with_store_remote()
    _keyed(world, {LOCK: _lock(), "pkg/Proofs/lean-audit.json": _lock(), "lib/lean-audit.json": _lock()})
    h = world.change({**BOTH, "lib/lean-audit.json": _lock(C="2")})
    keys = {p: _key(world.dev, h, p) for p in ("pkg", "pkg/Proofs", "lib")}
    t = int(_git(world.dev, "log", "-1", "--format=%ct", h))                # the warning lists the runs newest first
    lib = _published(world, h, pkg="lib", passed=False, passes={keys["pkg"]: "pkg"}, at=t - 120)
    steps = [{"step": "lean-audit", "status": "passed"}]
    shard = _published(world, h, steps=steps, kind="check-shard/v1", passes={keys["pkg/Proofs"]: "pkg/Proofs"}, at=t - 60)
    check = _published(world, h, steps=steps)
    capsys.readouterr()
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert "it changes the Lean lock of lib against `main`" in err and "--records lib --update" in err
    assert "--records pkg --update" not in err and "--records pkg/Proofs --update" not in err
    assert f"records run {lib} of {h[:12]} failed lib" in err
    assert f"the Lean check {check} of {h[:12]} lists no pass of lib on the head's lock among its result's `passes`" in err
    assert any(f"records gate: pkg: records run {lib} of {h[:12]} passed it" in line for line in world.logs)
    assert any(f"records gate: pkg/Proofs: the Lean check {shard} of {h[:12]} passed it" in line for line in world.logs)
    _published(world, h, pkg="lib", passes={keys["lib"]: "lib"})
    assert _mark(world, 1, h) == 0


def test_a_cancelled_or_unfinished_lean_check_vouches_for_nothing(world, capsys):
    """A check cancelled or killed after its `lean-audit` step passed (rc 137) didn't finish the run its result describes: it
    vouches for no package, whatever its `passes` say, and the warning says it didn't finish."""
    world.with_store_remote()
    _keyed(world, {LOCK: _lock(), "pkg/Proofs/lean-audit.json": _lock()})
    h = world.change(BOTH)
    steps = [{"step": "lean-audit", "status": "passed"}]
    passes = {_key(world.dev, h, p): p for p in ("pkg", "pkg/Proofs")}
    ids = [_published(world, h, steps=steps, passes=passes, state="cancelled", rc=137),
           _published(world, h, steps=steps, passes=passes, passed=False, rc=137),
           _published(world, h, steps=steps, passes=passes, state="running")]
    capsys.readouterr()
    assert _mark(world, 1, h) == 1
    err = capsys.readouterr().err
    assert "it changes the Lean lock of pkg/Proofs, pkg against `main`" in err
    for rid in ids:
        assert f"the Lean check {rid} of {h[:12]} didn't finish" in err
    assert not any("records gate:" in line for line in world.logs)


def test_a_records_run_whose_update_wrote_a_record_vouches_only_for_the_commit_of_that_record(world, capsys):
    """`--update` passes by writing the record, kept as `record/<path>/lean-audit.json` when it differs from the commit's lock:
    the commit it ran on still has the old lock, so the run doesn't vouch for it.  Its result lists the package under the key
    of the commit of exactly that record over it, which is then vouched for with no run of its own, and so is a merge of another
    PR that resolves to that commit's tree; the commit it ran on, any other lock, and a change above it to a file the key holds
    are warned of."""
    world.with_store_remote()
    _keyed(world, {LOCK: _lock()})
    h1 = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    h2 = world.change({LOCK: _lock(A="2", B="2")}, base=h1)
    rid = _published(world, h1, records={"pkg": {"outcome": "regenerated", "record": "lean-audit/record/pkg/lean-audit.json"}},
                     passes={_key(world.dev, h2): "pkg"})
    capsys.readouterr()
    assert _mark(world, 1, h1) == 1
    err = capsys.readouterr().err
    assert f"records run {rid} of {h1[:12]} passed pkg only on the record its `--update` wrote" in err
    assert "commit its `lean-audit/record/pkg/lean-audit.json` over `pkg/lean-audit.json`" in err
    assert _mark(world, 1, world.change({LOCK: _lock(A="2", B="3")}, base=h1)) == 1
    assert _mark(world, 1, h2) == 0
    assert f"records gate: pkg: records run {rid} of {h1[:12]} passed it on {h2[:12]}'s Lean inputs" in world.logs[-1]

    other = world.change({"c.txt": "another PR's\n"})
    _git(world.dev, "checkout", "-q", "--detach", h2)
    _git(world.dev, "merge", "-q", "--no-ff", "-s", "ours", "-m", "merge another PR, keeping our lock", other)
    h3 = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{h3}:refs/dev/{h3}")
    assert _mark(world, 1, h3) == 0
    assert f"records gate: pkg: records run {rid} of {h1[:12]} passed it on {h3[:12]}'s Lean inputs" in world.logs[-1]

    for files in ({"pkg/A.lean": "theorem a : 1 = 1 := rfl\n"}, {"pkg/tests/T.lean": "def t := 1\n"}):
        capsys.readouterr()
        assert _mark(world, 1, world.change(files, base=h2)) == 1
        assert "--records pkg --update" in capsys.readouterr().err


def test_a_requirer_whose_record_the_run_regenerated_is_vouched_for_only_at_the_commit_of_that_record(world, capsys):
    """`--records PKG --update` also updates a requirer that failed only on record differences, and lists it under the key of
    the commit of every record it kept, not of the commit it ran on: that commit samples the warning for each, naming each
    record, and the commit of both records is vouched for by that one run."""
    world.with_store_remote()
    req = "req/lean-audit.json"
    _keyed(world, {LOCK: _lock(), req: _lock()})
    h1 = world.change({LOCK: _lock(A="2"), req: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    h2 = world.change({LOCK: _lock(A="2", B="2"), req: _lock(A="2", C="2")}, base=h1)
    records = {p: {"outcome": "regenerated", "record": f"lean-audit/record/{p}/lean-audit.json"} for p in ("pkg", "req")}
    rid = _published(world, h1, records=records, passes={_key(world.dev, h2, p): p for p in ("pkg", "req")})
    capsys.readouterr()
    assert _mark(world, 1, h1) == 1
    err = capsys.readouterr().err
    assert f"records run {rid} of {h1[:12]} passed req only on the record its `--update` wrote" in err
    assert "commit its `lean-audit/record/req/lean-audit.json`" in err and "commit its `lean-audit/record/pkg/lean-audit.json`" in err
    assert _mark(world, 1, world.change({LOCK: _lock(A="2", B="2"), req: _lock(A="2", C="3")}, base=h1)) == 1
    assert _mark(world, 1, h2) == 0
    for pkg in ("pkg", "req"):
        assert any(f"records gate: {pkg}: records run {rid} of {h1[:12]} passed it on {h2[:12]}'s Lean inputs" in line
                   for line in world.logs)


def test_a_head_that_changes_only_other_packages_locks_over_a_vouched_commit_is_vouched_for_through_it(world, capsys):
    """A lane commits another package's record over a commit where `pkg` passed: `pkg`'s key moves with it (its closure holds
    `lib`, lock and all), but no file `pkg`'s records were checked against, so the gate follows such a commit down to the pass.
    A commit that changes anything besides other packages' locks isn't followed."""
    world.with_store_remote()
    _keyed(world, {LOCK: _lock(), "lib/lean-audit.json": _lock()})
    a = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    rid = _published(world, a, passes={_key(world.dev, a): "pkg"})
    h = world.change({"lib/lean-audit.json": _lock(B="2")}, base=a)
    assert _key(world.dev, h) != _key(world.dev, a)
    _published(world, h, pkg="lib", passes={_key(world.dev, h, "lib"): "lib"})
    capsys.readouterr()
    assert _mark(world, 1, h) == 0, capsys.readouterr().err
    assert any(f"records gate: pkg: {h[:12]} changes only other packages' locks over {a[:12]}, where records run {rid} of "
               f"{a[:12]} passed it" in line for line in world.logs)

    h2 = world.change({"lib/lean-audit.json": _lock(B="2"), "lib/L.lean": "def l := 2\n"}, base=a)
    _published(world, h2, pkg="lib", passes={_key(world.dev, h2, "lib"): "lib"})
    capsys.readouterr()
    assert _mark(world, 2, h2) == 1
    err = capsys.readouterr().err
    assert "it changes the Lean lock of pkg against `main`" in err and "--records pkg --update" in err


def test_a_change_to_checker_files_the_key_leaves_out_leaves_a_pass_standing(world, capsys):
    """A test or a doc of `tools/verity/lean/` made every pass at another commit vouch for nothing, as the whole directory was
    one input.  The key holds only the checker files a records run reads (`lean_changed.tool_inputs`): not its tests, its
    Markdown nor `lean_audit.NOT_READ`, `merge.py` among them (a run that merges a lock across a merge of `main` reuses a pass
    and lists none); a new file of it counts, and so does the task graph's contract, which the audit loads."""
    contract = "tools/verity/graph/verity_graph/contract.py"
    world.with_store_remote()
    _keyed(world, {LOCK: _lock(), "tools/verity/lean/merge.py": "# the lock driver\n", "tools/verity/lean/tool.py": "# the runs\n",
                   "tools/verity/lean/README.md": "the checker\n", "tools/verity/lean/tests/test_check.py": "x = 1\n"})
    c = world.change({LOCK: _lock(A="2")})
    rid = _published(world, c, passes={_key(world.dev, c): "pkg"})
    for n, files in ((1, {"tools/verity/lean/tests/test_check.py": "x = 2\n", "tools/verity/lean/tests/fixtures/T.lean": "def t := 1\n"}),
                     (2, {"tools/verity/lean/README.md": "the checker, said again\n", "tools/verity/lean/NOTES.md": "notes\n"}),
                     (3, {"tools/verity/lean/tool.py": "# the runs, changed\n"}),
                     (4, {"tools/verity/lean/merge.py": "# the lock driver, changed\n"})):
        head = world.change({LOCK: _lock(A="2"), **files})
        capsys.readouterr()
        assert _mark(world, n, head) == 0, capsys.readouterr().err
        assert f"records gate: pkg: records run {rid} of {c[:12]} passed it on {head[:12]}'s Lean inputs" in world.logs[-1]

    for n, files in ((5, {"tools/verity/lean/new.py": "x = 1\n"}), (6, {contract: (ROOT / contract).read_text() + "# changed\n"})):
        head = world.change({LOCK: _lock(A="2"), **files})
        capsys.readouterr()
        assert _mark(world, n, head) == 1, files
        assert f"the store holds no pass of it on {head[:12]}'s Lean inputs" in capsys.readouterr().err


def test_the_gate_reads_the_results_of_the_lookback_that_may_list_passes_and_keeps_those_it_read(world, capsys):
    """The store holds thousands of runs a day, most of which are neither a records run nor a Lean check: the gate lists the
    attempts of the last `PASSES_LOOKBACK_S` and fetches only the results of those that may list passes.  A pass it indexed
    keeps vouching once its run is past the lookback."""
    from research.queue import PASSES_LOOKBACK_S

    world.with_store_remote()
    _keyed(world, {LOCK: _lock()})
    h = world.change({LOCK: _lock(A="2")})
    key, t = _key(world.dev, h), int(_git(world.dev, "log", "-1", "--format=%ct", h))
    _published(world, h, passes={key: "pkg"}, at=t - PASSES_LOOKBACK_S - 86400)
    suite = _published(world, h, argv=["python3", "tools/verity/check/suites.py", "--quick"])
    capsys.readouterr()
    assert _mark(world, 1, h, now=lambda: t + 60) == 1
    assert "--records pkg --update" in capsys.readouterr().err
    assert not world.store.index.sql("SELECT a.id FROM attempt_outputs o JOIN artifacts a ON a.id = o.artifact "
                                     "WHERE o.attempt = ?", (suite,))

    rid = _published(world, h, passes={key: "pkg"}, at=t - 86400)
    assert _mark(world, 1, h, now=lambda: t + 60) == 0
    assert f"records run {rid} of {h[:12]} passed it" in world.logs[-1]
    assert _mark(world, 1, h, now=lambda: t + 3 * PASSES_LOOKBACK_S) == 0
    assert f"records run {rid} of {h[:12]} passed it" in world.logs[-1]


#: `tools/verity/lean/merge.py` as the gate runs it on a lock both sides changed: ours, with each entry theirs changed
LOCK_MERGE = """\
import json, sys
from pathlib import Path
o, a, b = (json.loads(Path(p).read_text() or "{}") for p in sys.argv[1:4])
Path(sys.argv[2]).write_text(json.dumps({**a, **{k: v for k, v in b.items() if o.get(k) != v}}))
"""


def test_a_head_that_only_merged_main_into_a_vouched_head_is_vouched_for_with_no_run_of_its_own(world, capsys):
    """A lane merges `main` after its records run passed.  The fast path builds nothing for such a head and reports `path: reused`
    (every Lean input it changed is main's whole, and its lock is merge.py's merge of the two sides): that lock is a candidate,
    which the lock run after its landing rebuilds and regenerates where another PR moved what it reads, so the gate vouches for
    it too, from the run below, with no new run.  A merge whose lock isn't merge.py's (a hand resolution), or that changes a file
    of its own, is no such head and samples the warning."""
    world.with_store_remote()
    (world.tmp / "lock_merge.py").write_text(LOCK_MERGE)
    _keyed(world, {LOCK: _lock()})
    h1 = world.change({LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    rid = _published(world, h1, passes={_key(world.dev, h1): "pkg"})
    _move_main(world, {LOCK: _lock(E="2"), "b.txt": "b on main\n"})
    h2 = _merge_main(world, h1)
    assert _git(world.dev, "show", f"{h2}:{LOCK}") == _lock(A="2", E="2").strip()
    assert _mark(world, 1, h2) == 0
    assert (f"{h2[:12]} only merges `main`'s" in world.logs[-1] and
            f"where records run {rid} of {h1[:12]} passed it on {h1[:12]}'s Lean inputs" in world.logs[-1])

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


def test_the_records_gate_vouches_for_none_and_says_why_when_it_cant_read_the_store_or_tell_the_locks(world, capsys, monkeypatch):
    """A VM whose shell lost its store secrets has no remote, and one whose endpoint is down can't list it: the runs other machines
    published aren't here, so the gate vouches for none of the head's locks rather than answer from what's here, and warns
    saying why, as it does when it can't fetch the head to tell which locks it changes.  The head is marked each time."""
    from research.queue import RECORDS_AFTER_LANDING, MarkOnly, RecordsGate, ready

    h = world.change({LOCK: _lock(A="2")})
    capsys.readouterr()
    assert _mark(world, 1, h, store=LocalStore(world.tmp / "bare", config_path=world.tmp / "none.toml")) == 1
    assert "the store's records runs can't be read" in capsys.readouterr().err
    world.with_store_remote()
    world.gh.put(2, h, ready=False)
    gone = RecordsGate(world.dev, str(world.tmp / "gone.git"), world.store, log=world.logs.append)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), records=gone) == 0
    err = capsys.readouterr().err
    assert f"research queue ready: warning: #2 at {h[:12]}: the records gate can't tell which Lean locks it changes (git " in err
    assert f"), so it vouches for none: {RECORDS_AFTER_LANDING}\n" in err

    _published(world, h)
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: _Unreachable()))
    assert _mark(world, 1, h, store=LocalStore(world.tmp / "cut-off", config_path=world.tmp / "store.toml")) == 1
    err = capsys.readouterr().err
    assert (f"research queue ready: warning: #1 at {h[:12]}: it changes the Lean lock of pkg, and the store's records runs "
            f"can't be read (OSError: injected: endpoint unreachable; ") in err
    assert f"so the records gate vouches for none: {RECORDS_AFTER_LANDING}\n" in err


def test_a_head_stacked_on_a_landed_pr_that_merged_main_is_judged_by_its_own_change_not_one_merge_base(world, capsys, monkeypatch):
    """A head stacked on a PR since landed on `checked`, which then merged `main`, has two merge bases with `checked`: the
    stacked-on PR's head and that `main`.  Measured from the first (git's pick when it is the newer), every lock `main` changed
    since counts as the head's, and the gate warned for records runs of locks it never touched.  A file is the head's change only
    where it differs from both."""
    world.with_store_remote()
    base = _move_main(world, {"tools/verity/check/drive.toml": 'land_into = "checked"\n'})
    monkeypatch.setenv("GIT_COMMITTER_DATE", "1000000000 +0000")
    moved = _move_main(world, {LOCK: _lock(A="2"), "pkg/A.lean": "theorem a : True := trivial\n"})
    monkeypatch.delenv("GIT_COMMITTER_DATE")
    below = world.change({"a.txt": "stacked on\n"}, base=base)
    _git(world.dev, "checkout", "-q", "--detach", moved)
    _git(world.dev, "merge", "-q", "--no-ff", "--no-edit", below)
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/checked")
    head = _merge_main(world, world.change({"x.txt": "x\n"}, base=below))
    bare = LocalStore(world.tmp / "bare", config_path=world.tmp / "none.toml")
    capsys.readouterr()
    assert _mark(world, 1, head, store=bare) == 0, capsys.readouterr().err
    own = world.change({LOCK: _lock(A="2", B="3")}, base=head)
    assert _mark(world, 2, own) == 1
    assert f"warning: #2 at {own[:12]}: it changes the Lean lock of pkg against `checked`" in capsys.readouterr().err


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
                                                     "built": Path("verity/protocols/flock/verifier/.lake/built").is_file(),
                                                     "lake": shutil.which("lake"),
                                                     "env": {k: os.environ.get(k) for k in ("VERITY_STORE_REQUIRED",
                                                             "CHECK_STORE_CUSTODY", "RESEARCH_RUN_ID", "GIT_AUTHOR_EMAIL",
                                                             "PYTHONPYCACHEPREFIX", "VERITY_REGRESSION")}}))
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
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "verity/protocols/flock/verifier/lakefile.toml": "name = \"flock\"\n"})
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
    lean = wt / "verity" / "protocols" / "flock" / "verifier"
    assert ran.read_text().split("\n") == [f"{lean} build", ""]
    (lean / ".lake" / "built").write_text("from the last PR\n")
    (lean / "stray.olean").write_text("left by the last PR\n")
    monkeypatch.setenv("FAKE_LAKE_FAIL", "1")
    assert tier(world.gh.prs[1]) == "`lake build` in verity/protocols/flock/verifier exited 1 (its output is above)"
    assert (lean / ".lake" / "built").read_text() == "from the last PR\n" and not (lean / "stray.olean").exists()


def test_the_quick_tier_runs_on_the_heads_merge_onto_mains_tip_and_a_head_that_does_not_merge_there_refuses(world, monkeypatch,
                                                                                                           capsys):
    """What lands is a head's merge with `main`, and a head can pass the tier on its own and fail on that merge: a head behind
    `main` is judged on its merge onto `main`'s tip, its change measured from that tip, and marked at its head."""
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    moved = _move_main(world, {"m.txt": "main moved\n"})
    seen = world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(seen))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    s = json.loads(seen.read_text())
    assert s["argv"][:3] == ["--quick", "--changed", moved] and {"a.txt", "m.txt"} <= set(s["files"])
    [wt] = [p for p in (world.tmp / "ready").iterdir() if p.is_dir() and p.name != "logs"]
    at = _git(wt, "rev-parse", "HEAD")
    assert _git(wt, "rev-parse", f"{at}^1", f"{at}^2").split() == [moved, h1]
    assert f"#1's head {h1[:12]} merged onto `main` at {moved[:12]}: the tier runs on {at[:12]}" in capsys.readouterr().out
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True

    theirs = _move_main(world, {"a.txt": "main's a\n"})
    world.gh.put(1, h2 := world.change({"a.txt": "a2\n"}, base=h1), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert (f"the quick tier failed: it conflicts with `main` at {theirs[:12]} in a.txt: merge `main` into the branch"
            in capsys.readouterr().err)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is False


#: the Lean audit (`LEAN_AUDIT`) as the tier runs it: records its arguments, the commit it ran at and whether the suites ran yet,
#: and fails with a last line when FAKE_AUDIT_FAIL is set
FAKE_AUDIT = """\
import json, os, subprocess, sys
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
with open(os.environ["FAKE_AUDIT_SEEN"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "head": head, "suites_ran": os.path.exists(os.environ["FAKE_SEEN"])}) + "\\n")
print("audited what the change reaches")
if os.environ.get("FAKE_AUDIT_FAIL"):
    print("Pkg.A: unknown identifier 'x'")
    sys.exit(1)
"""


def test_a_change_that_touches_lean_is_marked_only_after_the_lean_audit_passes_on_its_merge_onto_main(world, monkeypatch,
                                                                                                         capsys):
    """The Lean audit on the merge publishes what the audit of the candidate landing it stages: it runs in the tier of a change
    that touches Lean, on its merge onto `main`'s tip and before the suites, and a failure refuses the mark.  A change that
    touches no Lean runs none, and neither does any change while `LEAN_AUDIT` is None."""
    from research import queue as Q

    world.with_store_remote()
    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "audit.py": FAKE_AUDIT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    world.gh.put(1, h1 := world.change({"Pkg/A.lean": "theorem a : True := trivial\n"}), ready=False)
    world.gh.put(2, world.change({"a.txt": "a1\n"}), ready=False)
    moved = _move_main(world, {"m.txt": "main moved\n"})
    seen, suites = world.tmp / "audits.jsonl", world.tmp / "seen.json"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(suites))
    monkeypatch.setenv("FAKE_AUDIT_SEEN", str(seen))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_AUDIT_FAIL", raising=False)
    tier = Q.QuickTier(world.dev, str(world.origin), root=world.tmp / "ready")

    assert Q.LEAN_AUDIT is None and tier(world.gh.prs[1]) is None and not seen.exists()
    monkeypatch.setattr(Q, "LEAN_AUDIT", [sys.executable, "audit.py"])
    assert tier(world.gh.prs[2]) is None and not seen.exists()
    suites.unlink()
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    [ran] = [json.loads(ln) for ln in seen.read_text().splitlines()]
    assert ran["argv"][:3] == ["--base", moved, "--out"] and not ran["suites_ran"] and suites.exists()
    assert _git(world.dev, "rev-parse", f"{ran['head']}^1", f"{ran['head']}^2").split() == [moved, h1]
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True

    monkeypatch.setenv("FAKE_AUDIT_FAIL", "1")
    world.gh.put(1, h3 := world.change({"Pkg/A.lean": "theorem a : True := x\n"}, base=h1), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 1
    assert "the quick tier failed: the Lean audit on the merge exited 1: Pkg.A: unknown identifier 'x'" in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h3) is False


def _move_main(world: World, files: dict[str, str]) -> str:
    moved = world.change(files)
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    return moved


def _landed_move(world: World, rows: str, files: dict[str, str | None]) -> tuple[str, str]:
    """`main` moved by a layout move: its map `tools/move/x_map.toml` (``rows``) lands, then the move writing ``files`` on it,
    with its `Layout-move:` line, as `tools/move/stage.py` commits one.  Returns (the move's base, the move)."""
    base = _move_main(world, {"tools/move/x_map.toml": rows})
    tree = _git(world.dev, "rev-parse", f"{world.change(files)}^{{tree}}")
    move = _git(world.dev, "commit-tree", tree, "-p", base, "-m", "Move\n\nLayout-move: tools/move/x_map.toml")
    _git(world.dev, "push", "-q", "origin", f"{move}:refs/heads/main")
    return base, move


def test_a_move_map_retires_its_rows_old_paths_and_dotted_modules_and_a_word_map_its_words_in_any_case(world, capsys):
    """`map_retires` reads a map as `layout.py` and `rename.py` take it: a row that keeps its path and a one-word module retire
    nothing, and a map missing at the base or that doesn't parse can't be read, the one that doesn't parse said on stderr."""
    main = _move_main(world, {"tools/move/l_map.toml": '[paths]\n"old/" = "new/"\n"kept.py" = "kept.py"\n'
                                                       '[retired]\npaths = ["gone/"]\n[drop]\npaths = ["x.toml"]\n'
                                                       '[modules]\n"old" = "new"\n"pkg.old" = "pkg.new"\n',
                              "tools/move/w_map.toml": '[words]\n"Warden" = "NetworkCertifier"\n'
                                                       '[[words_in]]\nfiles = ["a/*"]\nwords = { tap = "certifier" }\n',
                              "tools/move/bad_map.toml": "[paths\n"})
    layout, words = (Q.map_retires(world.origin, main, f"tools/move/{m}_map.toml") for m in ("l", "w"))
    assert layout == Q.Retired(frozenset({"old", "gone", "x.toml", "pkg.old"})), layout
    assert words == Q.Retired(words=frozenset({"warden", "tap"})), words
    assert words.named("a NETWORK WARDEN, pkg.old") == ["warden"] and layout.named("from pkg.old import y; Old") == ["old", "pkg.old"]
    capsys.readouterr()
    assert Q.map_retires(world.origin, main, "tools/move/bad_map.toml") is None
    assert f"tools/move/bad_map.toml at {main[:12]} doesn't parse" in capsys.readouterr().err
    assert Q.map_retires(world.origin, main, "tools/move/none_map.toml") is None
    assert not capsys.readouterr().err


def test_the_status_command_refuses_a_head_lacking_a_move_whose_change_names_what_it_retires_and_admits_the_rest(world):
    """`status --loop` says what `World.ready` decides of a head below a move `main` landed: one whose own change names none of
    the move's retired names is admitted, and one that names one, in a path or an added line, is refused with the move and
    the fix."""
    base, move = _landed_move(world, '[paths]\n"c.txt" = "moved/c.txt"\n', {"c.txt": None, "moved/c.txt": "c\n"})
    world.gh.put(1, world.change({"b.txt": "1\n"}, base=base))
    world.gh.put(2, world.change({"e.txt": "reads c.txt\n"}, base=base))
    loop = world.queue(land_into="checked").status(loop=True)
    assert [loop[n].state for n in (1, 2)] == ["ready", "refused"], loop
    assert loop[1].why.startswith("admitted: "), loop[1]
    assert loop[2].why == (f"it lacks the move {move[:12]} `main` landed (tools/move/x_map.toml), whose retired names its own "
                           "change names (`c.txt`): merge `main` into it and rename them, or cross it with "
                           "`tools/move/restack.py`, and mark the new head ready"), loop[2]


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
        LOCK_EX, LOCK_SH, LOCK_NB = fcntl.LOCK_EX, fcntl.LOCK_SH, fcntl.LOCK_NB

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


def test_judge_warns_of_a_pr_whose_body_names_no_live_goal_and_marks_it_all_the_same(world, monkeypatch, capsys):
    """`judge` writes the mark `ready` would have, so `ready`'s goal gate warns there too, with the same fix, and never
    refuses."""
    from research import queue as Q

    machine, rid, h1 = _orphaned(world, monkeypatch)
    serve(monkeypatch, world.tmp, dict(PAGE))
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch, goals=Q.goal_gate) == 0
    assert capsys.readouterr().err == (f"research queue judge: warning: #1 at {h1[:12]}: its body names no goal: "
                                       f"{GOAL_FIX.format(n=1)}\n")
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True


def test_judge_warns_of_a_lock_the_records_gate_doesnt_vouch_for_and_marks_it_all_the_same(world, monkeypatch, capsys):
    """`judge` writes the mark `ready` would have, so `ready`'s records gate warns there too, and never refuses."""
    from research import queue as Q

    machine, rid, h1 = _orphaned(world, monkeypatch)
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch,
                   records=lambda pr: "it changes the Lean lock of pkg against `main`") == 0
    assert capsys.readouterr().err == f"research queue judge: warning: #1 at {h1[:12]}: it changes the Lean lock of pkg against `main`\n"
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


def test_a_remote_tier_on_a_merge_onto_main_reports_the_head_it_merged_and_judge_marks_that_head(world, monkeypatch, capsys):
    """A tier on a head's merge onto `main`'s tip runs at a commit that is no PR's head, so `judge`, marking a run whose caller
    died, marks the head its facts say it merged (`merged`); a checkout that isn't that merge refuses."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    machine = Machine(world.tmp)
    tier, _ = _remote_tier(world, monkeypatch, machine)
    world.gh.put(1, h1 := world.change({"a.txt": "a1\n"}), ready=False)
    moved = _move_main(world, {"m.txt": "main moved\n"})
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setenv("FAKE_SEEN", str(world.tmp / "seen.json"))
    monkeypatch.delenv("FAKE_FAIL", raising=False)

    def died(d: Path) -> dict:
        raise KeyboardInterrupt("injected: the caller is gone")

    tier.fetch = died
    with pytest.raises(KeyboardInterrupt):
        Q.ready(world.gh, world.store, 1, "lane-1", tier)
    flags = machine.launched[-1]
    rid = flags[flags.index("--id") + 1]
    facts = json.loads((machine.root / "runs" / rid / "result.json").read_text())["tier"]
    assert (facts["merged"], facts["base"]) == (h1, moved) and facts["head"] not in (h1, moved)
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    capsys.readouterr()
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 0
    assert f"#1 is ready at {h1[:12]} (pr:1@{h1})" in capsys.readouterr().out
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is True

    from research import quick_tier
    _git(world.dev, "worktree", "add", "-q", "--detach", str(wt := world.tmp / "not-a-merge"), h1)
    facts = quick_tier.tier(wt, world.tmp / "q", h1, moved, ["true"], merged=main)
    assert facts["error"] == f"{h1[:12]} isn't {main[:12]} merged onto {moved[:12]}" and facts["merged"] is None


def test_a_remote_tier_runs_the_lean_audit_on_the_merge_it_ships(world, monkeypatch):
    """`LEAN_AUDIT` reaches the machine's quick_tier.py as `--lean-audit`, shell-quoted, and runs in the shipped merge."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT, "audit.py": FAKE_AUDIT})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    machine = Machine(world.tmp)
    tier, _ = _remote_tier(world, monkeypatch, machine)
    world.gh.put(1, h1 := world.change({"Pkg/A.lean": "theorem a : True := trivial\n"}), ready=False)
    moved = _move_main(world, {"m.txt": "main moved\n"})
    seen = world.tmp / "audits.jsonl"
    monkeypatch.setattr(Q, "QUICK_TIER", [sys.executable, "tools/verity/check/suites.py", "--quick"])
    monkeypatch.setattr(Q, "LEAN_AUDIT", [sys.executable, "audit.py"])
    monkeypatch.setenv("FAKE_SEEN", str(world.tmp / "seen.json"))
    monkeypatch.setenv("FAKE_AUDIT_SEEN", str(seen))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_AUDIT_FAIL", raising=False)

    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    flags = machine.launched[-1]
    rid = flags[flags.index("--id") + 1]
    [ran] = [json.loads(ln) for ln in seen.read_text().splitlines()]
    assert ran["argv"] == ["--base", moved, "--out", str(machine.root / "runs" / rid / "out" / "lean-audit")]
    facts = json.loads((machine.root / "runs" / rid / "result.json").read_text())["tier"]
    assert (facts["merged"], facts["audit"]["rc"], facts["audit"]["command"]) == (h1, 0, [sys.executable, "audit.py"])
    assert ran["head"] == facts["head"]


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
    def refused(argv, cwd):
        return subprocess.CompletedProcess(argv, 3, "", "research run: refused: vy-test isn't in machines.toml\n")

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
    assert "A PR lands once its candidate's full check passes, so the quick tier isn't required" in err and "research run --on auto" in err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h1) is False
    rid = machine.launched[-1][machine.launched[-1].index("--id") + 1]
    assert Q.judge(world.gh, world.store, rid, "lane-1", fetch=machine.fetch) == 1
    assert "never ran the tier" in capsys.readouterr().err


def test_a_remote_quick_tier_on_auto_runs_on_the_machine_placed_and_without_one_never_starts(world, monkeypatch, capsys):
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports())
    asked = []
    def placed(spec, repo):
        asked.append(spec)
        return "vy-test", "a check slot and a Lean check slot are free (check slots: d free)"

    tier, _ = _remote_tier(world, monkeypatch, machine, place=placed)
    tier.on = "auto"
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0 and asked == ["auto"]
    flags = machine.launched[-1][:machine.launched[-1].index("--")]
    assert flags[flags.index("--on") + 1] == "vy-test"
    assert "research queue: --on auto: vy-test: a check slot and a Lean check slot are free" in capsys.readouterr().out
    tier.place, tier.on = (lambda spec, repo: (None, "no machine's slots could be read")), "vy-a,vy-b"
    world.gh.put(1, h2 := world.change({"a.txt": "a2\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == Q.READY_NOT_STARTED and len(machine.launched) == 1
    assert "the quick tier never started: --on vy-a,vy-b: no machine's slots could be read" in capsys.readouterr().err
    assert Q.StoreLabels(world.store, log=world.logs.append).ready(1, h2) is False


def test_a_remote_quick_tier_placed_in_a_research_pool_runs_through_that_pools_slots(world, monkeypatch, capsys):
    """With node 2 stopped, `--on auto` found no machine and the tier never started, while `research run --on auto` spilled to a
    CPU node's research pool. `place` gives such a pool as `MACHINE:POOL`, and the tier's slot.py takes that pool's slots."""
    from research import queue as Q

    machine = Machine(world.tmp, outcome=_reports())
    tier, _ = _remote_tier(world, monkeypatch, machine, place=lambda spec, repo: ("vy-test:research", "a free slot"))
    tier.on = "auto"
    world.gh.put(1, world.change({"a.txt": "a1\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 1, "lane-1", tier) == 0
    argv = machine.launched[-1]
    assert argv[argv.index("--on") + 1] == "vy-test"
    cmd = argv[argv.index("--", argv.index("--declared-output")) + 1:]
    assert cmd[:4] == ["env", "CHECK_SLOTS=/workspace/research/locks/research/slots", "python3", "tools/verity/check/slot.py"]
    assert "in a slot of its research pool" in capsys.readouterr().out


def test_place_on_auto_spills_to_a_research_pool_only_when_no_auto_machine_can_be_read(tmp_path):
    from research import queue as Q

    repo = tmp_path / "repo"
    (repo / "tools/verity/check").mkdir(parents=True)
    (repo / "tools/verity/check/check.py").write_text(
        'AUTO = ("vy-nebius-2",)\n'
        "def read_slots(names):\n    return {}\n"
        "def choose(names, docs, priority=False):\n"
        '    return None, "no machine\'s slots could be read", [f"{n}: {d}" for n, d in docs.items()]\n')
    spilled = []

    def spill(r):
        spilled.append(r)
        return ("vy-nebius-cpu-1", "research"), "vy-nebius-cpu-1's research pool has a free slot"
    got, why = Q.place("auto", repo, read=lambda names: {n: "ssh: connect timed out" for n in names}, spill=spill)
    assert got == "vy-nebius-cpu-1:research" and spilled == [repo]
    assert why.startswith("vy-nebius-cpu-1's research pool has a free slot, as none of vy-nebius-2 could be read")
    got, why = Q.place("auto", repo, read=lambda names: {n: "ssh: connect timed out" for n in names},
                       spill=lambda r: (None, "no pool fits"))
    assert got is None and why.endswith("; no research pool either: no pool fits")
    got, _ = Q.place("vy-a,vy-b", repo, read=lambda names: {n: "down" for n in names}, spill=spill)
    assert got is None and len(spilled) == 1, "a named list never spills"


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
    (CHECK_STORE_CUSTODY) and VERITY_STORE_REQUIRED=1.  A tier that handed neither would skip Flock's store-backed tests as "the
    evidence store is not reachable", and the skip would pass.  It hands both as check.py does, never the caller's,
    and nothing of the recording run's own; with no key staged, the store is still required.  It runs the vLLM reference
    regression as check's steps do (VERITY_REGRESSION=1), so a PR that breaks a row fails before its mark, not first in the
    full check after landing.  The machine's PYTHONPYCACHEPREFIX
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
    monkeypatch.setenv("VERITY_REGRESSION", "0")
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    machine = Machine(world.tmp, custody_r2=True)
    tier, _ = _remote_tier(world, monkeypatch, machine)
    assert tier(world.gh.prs[1]) is None
    flags = machine.launched[-1]
    key = machine.root / "requests" / flags[flags.index("--id") + 1] / "custody"
    env = json.loads(seen.read_text())["env"]
    assert env["VERITY_STORE_REQUIRED"] == "1" and env["CHECK_STORE_CUSTODY"] == str(key.resolve()) and env["VERITY_REGRESSION"] == "1"
    assert env["RESEARCH_RUN_ID"] is None and env["PYTHONPYCACHEPREFIX"] is None and env["GIT_AUTHOR_EMAIL"]
    machine.custody_r2 = False
    assert tier(world.gh.prs[1]) is None
    env = json.loads(seen.read_text())["env"]
    assert env["VERITY_STORE_REQUIRED"] == "1" and env["CHECK_STORE_CUSTODY"] is None and env["VERITY_REGRESSION"] == "1"


def test_a_quick_tier_with_no_lake_on_the_path_builds_with_elans_and_with_none_refuses(world, monkeypatch):
    """On a machine the run's PATH is a non-login shell's, without elan's bin, which check.py puts first itself.  A tier that
    built the verifier only with `lake` on the PATH would skip its Lean-dependent tests and label the PR ready.  It takes `lake` from ~/.elan/bin, first on the suites' PATH too, and with neither refuses."""
    from research import queue as Q

    main = world.change({"tools/verity/check/suites.py": FAKE_SUITES, "tools/verity/check/slot.py": FAKE_SLOT, "verity/protocols/flock/verifier/lakefile.toml": 'name = "flock"\n'})
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
    lean = "verity/protocols/flock/verifier"
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
                      (["queue", "eject", "1", "--reason", "x", "--dry-run"], "--dry-run is for sync only: `eject` would act"),
                      (["queue", "ready", "1", "--local", "--dry-run"], "--dry-run is for sync only: `ready` would act"),
                      (["queue", "status", "--priority-file", "p"], "--priority-file is for sync only")):
        with pytest.raises(SystemExit) as e:
            research_main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err


def test_delegate_takes_an_agent_id_where_a_pr_number_goes(world, monkeypatch, capsys):
    """`delegate AGENT` reaches `delegate`, which judges the agent id, rather than refusing it as a PR number."""
    monkeypatch.chdir(world.dev)
    argv = ["queue", "delegate", "bc-x", "--by", "ci", "--repo", "o/verity", "--store", str(world.tmp / "qs")]
    assert research_main(argv) == 1
    assert "'bc-x' isn't a Cursor agent id" in capsys.readouterr().err


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
    d = "verity/protocols/flock/verifier"

    def run(n, **extra):
        (pkg := tmp_path / f"t{n}" / d).mkdir(parents=True)
        return quick_tier.build(str(lake), pkg, d, {**env, **extra})

    assert run(1) == {"dir": d, "rc": 0, "start": "cold"}
    warm = tmp_path / "home" / ".cache" / "verity" / "lake-slot-a" / d.replace("/", "--")
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
    assert set(asked[:-1]) == {Q.GH_TIMEOUT_S} and asked[-1] == M.GH_TIMEOUT_S       # the pages not yet asked for are cancelled


def test_the_open_prs_are_read_a_wave_of_pages_at_a_time_until_one_isnt_full(monkeypatch):
    """`gh api --paginate` read the ~10 pages one after another, seconds of every `research queue status`."""
    import research.queue as Q

    numbers = [*range(250, 150, -1), *range(151, 51, -1), *range(51, 0, -1)]       # 151 again: the listing shifted a page
    asked: list[int] = []

    def gh(argv, **kw):
        page = int(re.search(r"&page=(\d+)$", argv[-1]).group(1))
        asked.append(page)
        rows = [{"number": n, "head": {"sha": f"{n:040x}", "ref": f"b{n}"}, "body": "Lane `ci`."}
                for n in numbers[(page - 1) * Q.GH_PER_PAGE:page * Q.GH_PER_PAGE]]
        return subprocess.CompletedProcess(argv, 0, json.dumps(rows), "")

    monkeypatch.setattr(subprocess, "run", gh)
    monkeypatch.setattr(Q, "GH_PAGES", 2)
    assert [p.number for p in Q.GitHub("o/r").open_prs()] == list(range(250, 0, -1))
    assert sorted(asked) == [1, 2, 3, 4]


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


#: a lock as `check.py --update` writes one: F, G and H listed, F frozen, G's two inline premises and H's one, G and H reading M
PREMISE_LOCK = {"guarantees": {"F": {"type_hash": "f"}, "G": {"owner": "@lean", "signature": ["G : Guarantees.G"], "type_hash": "g"},
                               "H": {"signature": ["H : Guarantees.H"], "type_hash": "h"}},
        "shape": {"frozen": {"F": "frozen: a game's bound"},
                  "inline": {"G": {"p1": "a membership", "p2": "a bound"}, "H": {"q": "a bound"}}},
        "reads": {"M": {"definitions": {"Guarantees.G": "1", "Guarantees.H": "2"}}}}


def _lock_text(edit: Callable[[dict], object] = lambda d: None) -> str:
    """`PREMISE_LOCK` as a lock's text, after ``edit`` changed it in place."""
    doc = json.loads(json.dumps(PREMISE_LOCK))
    edit(doc)
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def _lock_files() -> dict[str, str]:
    """The record merge driver, named for every lock, and `PREMISE_LOCK`."""
    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    return {**{f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")},
            ".gitattributes": "lean-audit.json merge=lean-audit\n", "pkg/lean-audit.json": _lock_text()}


def _renamed(d: dict) -> None:
    d["guarantees"]["G2"] = d["guarantees"].pop("G")
    d["shape"]["inline"]["G2"] = d["shape"]["inline"].pop("G")


def _removed(d: dict) -> None:
    del d["guarantees"]["G"], d["shape"]["inline"]["G"]


def _rekeyed(d: dict) -> None:
    d["shape"]["inline"]["G"] = {"p1 at its new statement": "a membership", "p2": "a bound"}


def _restated(d: dict) -> None:
    d["reads"]["M"]["definitions"]["Guarantees.G"] = "3"
    del d["shape"]["inline"]["G"]["p1"]


#: the head's lock -> what its merge onto `next` drops that `next` needs; None: the head leaves the lock alone
LOCK_DROPS = {
    # the record driver drops a premise the head dropped where `next` didn't restate its guarantee: the head's own drop
    "a dropped premise": (lambda d: d["shape"]["inline"]["G"].pop("p1"), []),
    "a dropped frozen key": (lambda d: d["shape"]["frozen"].pop("F"), ["pkg/lean-audit.json: shape.frozen[F]"]),
    "a renamed guarantee": (_renamed, []),
    "a removed guarantee": (_removed, []),
    "a premise re-keyed at its restated statement": (_rekeyed, []),
    "a restated guarantee": (_restated, []),
    "a new guarantee": (lambda d: d["guarantees"].update(N={"type_hash": "n"}), []),
    "an unchanged lock": (None, []),
}


def _lock_world(world: World) -> None:
    """`main` with `_lock_files`, and #1 on `next`, which adds guarantee K and its premise to the lock: the driver merges every
    head that changes it."""
    _move_main(world, _lock_files())

    def k(d: dict) -> None:
        d["guarantees"]["K"] = {"type_hash": "k"}
        d["shape"]["inline"]["K"] = {"r": "a bound"}

    world.gh.put(1, world.change({"pkg/lean-audit.json": _lock_text(k)}))
    assert [m.pr for m in world.queue().sync()[0]] == [1]


@pytest.mark.parametrize("case", list(LOCK_DROPS))
def test_a_lock_merge_onto_next_names_each_premise_and_frozen_key_it_drops_that_next_still_needs(world, case):
    """A head's regenerated lock, merged onto `next` by the record driver as the sync merges it, dropped `shape.inline` premises
    of guarantees `next` still states unchanged, and the train's Lean check failed on each.  What it may drop: a removed or
    restated guarantee's, one renamed, or re-keyed at a restated statement."""
    from research.queue import lock_drops

    _lock_world(world)
    edit, drops = LOCK_DROPS[case]
    head = world.change({"a.txt": "a2\n"} if edit is None else {"pkg/lean-audit.json": _lock_text(edit)})
    _git(world.dev, "fetch", "-q", "origin")
    assert lock_drops(world.dev, _git(world.dev, "rev-parse", "origin/next"), head) == drops


def test_dropped_entries_reads_a_frozen_reason_reworded_at_its_key_as_kept_and_a_removed_key_as_dropped():
    """`check.py` reads a frozen entry's key and its reason's category, not its wording: a head that rewords the reason under
    the same key drops nothing, and one that removes the key still drops it."""
    from research.queue import dropped_entries

    def lock(edit: Callable[[dict], object]) -> dict:
        return json.loads(_lock_text(edit))

    base = lock(lambda d: None)
    (f, why), = base["shape"]["frozen"].items()
    assert dropped_entries(base, lock(lambda d: d["shape"]["frozen"].update({f: why + " (reworded)"}))) == []
    assert dropped_entries(base, lock(lambda d: d["shape"]["frozen"].pop(f))) == [f"shape.frozen[{f}]"]


@pytest.mark.parametrize("names", [
    {"witnesses": {"W": {"nontrivial": "F"}}},
    {"witnesses": {"W": {"nontrivial": ["E", "F"], "honest": ["H"]}}},
    {"witnesses": {"W": {"nontrivial": "E", "honest": ["F"]}}},
    {"claims": {"witnesses": {"C": {"fails": "F"}}}},
    {"claims": {"witnesses": {"C": {"fails": "E", "meets": ["F"]}}}},
    {"claims": {"generic": {"C": "F"}}},
])
def test_dropped_entries_lets_a_head_drop_the_frozen_entry_of_a_theorem_held_to_its_form(names):
    """`check.py` holds a theorem `witnesses`, `claims.witnesses` or `claims.generic` names to its form instead of `shape`,
    and reads no frozen entry of its: a head may drop it while the theorem stays listed, and the same drop of a guarantee
    none of them names still drops it."""
    from research.queue import dropped_entries

    def lock(edit: Callable[[dict], object]) -> dict:
        return json.loads(_lock_text(edit))

    base = lock(lambda d: d.update(json.loads(json.dumps(names))))
    assert dropped_entries(base, lock(lambda d: (d.update(json.loads(json.dumps(names))),
                                                 d["shape"]["frozen"].pop("F")))) == []
    assert dropped_entries(lock(lambda d: None), lock(lambda d: d["shape"]["frozen"].pop("F"))) == ["shape.frozen[F]"]


def test_dropped_entries_keeps_a_premise_moved_under_a_new_key_and_one_of_a_guarantee_recorded_otherwise():
    """A premise `next` lists under G that the merged lock lists under a key it adds is renamed, and a guarantee whose record
    (its owner aside) differs, or a definition its signature names has another hash, was restated; a new owner alone, or
    another guarantee's definition rehashed, restates nothing."""
    from research.queue import dropped_entries

    def lock(edit: Callable[[dict], object]) -> dict:
        return json.loads(_lock_text(edit))

    base = lock(lambda d: None)
    assert dropped_entries(base, lock(lambda d: d["shape"]["inline"].update(G2=d["shape"]["inline"].pop("G")))) == []
    assert dropped_entries(base, lock(lambda d: (d["guarantees"]["G"].update(type_hash="g2"),
                                                 d["shape"]["inline"]["G"].pop("p1")))) == []
    assert dropped_entries(base, lock(_restated)) == [], "G is lifted into Guarantees.G, rehashed"
    assert dropped_entries(base, lock(lambda d: (d["reads"]["M"]["definitions"].update({"Guarantees.H": "4"}),
                                                 d["shape"]["inline"]["G"].pop("p1")))) == ["shape.inline[G]: p1"]
    assert dropped_entries(base, lock(lambda d: (d["guarantees"]["G"].update(owner="@proofs"),
                                                 d["shape"]["inline"]["G"].pop("p1")))) == ["shape.inline[G]: p1"]
    long = "∀ (i : Fin n), " + "x" * 80
    assert dropped_entries(lock(lambda d: d["shape"]["inline"]["G"].update({long: "a bound"})), base) == [
        f"shape.inline[G]: {long[:60]}…"]


def test_the_preflight_refuses_a_head_whose_lock_merge_drops_a_frozen_key_next_needs_naming_it_and_the_fix(world, capsys):
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world, _lock_files())
    world.gh.put(2, h2 := world.change({"pkg/lean-audit.json": _lock_text(lambda d: d["shape"]["frozen"].pop("F"))}),
                 ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.startswith("  - ")] == [
        "  - its lock merge onto `next` drops 1 entry `next`'s statements still need: pkg/lean-audit.json: shape.frozen[F]: "
        "restore it verbatim from `next` in a lock-only commit (`git show origin/next:PATH`) and mark that head"], err
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h2) is False and _no_worktree_left(world)


def test_a_premise_the_heads_own_regeneration_dropped_is_no_refusal_and_one_the_merge_alone_drops_is(world):
    """`check.py --update` keeps a premise only while a statement prints it, so a premise the head's own lock dropped against
    its merge base is printed nowhere at the head: a stale entry, which the head may drop though `next`'s lock still lists it.
    One the head's lock keeps and the merge drops is still refused."""
    from research.queue import lock_drops

    _move_main(world, _lock_files())
    main = _git(world.origin, "rev-parse", "main")
    stale = world.change({"pkg/lean-audit.json": _lock_text(lambda d: d["shape"]["inline"]["G"].pop("p1"))})
    _git(world.dev, "fetch", "-q", "origin")
    assert lock_drops(world.dev, main, stale) == []

    def new(d: dict) -> None:
        d["guarantees"]["N"] = {"type_hash": "n"}

    kept = world.change({"pkg/lean-audit.json": _lock_text(new)})
    lost = world.change({"pkg/lean-audit.json": _lock_text(lambda d: (new(d), d["shape"]["inline"]["G"].pop("p1")))})
    _git(world.dev, "fetch", "-q", "origin")
    assert lock_drops(world.dev, main, kept, lost) == ["pkg/lean-audit.json: shape.inline[G]: p1"]


def test_the_sync_leaves_off_next_a_head_whose_lock_merge_drops_what_next_needs_and_keeps_the_refusal_for_its_owner(world):
    """Drive posts each refusal the sync records to the PR's owner, once per head; a rename beside it merges."""
    _lock_world(world)
    world.gh.put(2, world.change({"pkg/lean-audit.json": _lock_text(lambda d: d["shape"]["frozen"].pop("F"))}))
    world.gh.put(3, world.change({"pkg/lean-audit.json": _lock_text(_renamed)}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 3]
    why = ("its lock merge onto `next` drops 1 entry `next`'s statements still need: pkg/lean-audit.json: shape.frozen[F]: "
           "restore it verbatim from `next` in a lock-only commit (`git show origin/next:PATH`) and mark that head")
    assert entries[2].state == "refused" and entries[2].why == why
    assert q.load_state()["conflicts"]["2"]["why"] == why
    assert set(json.loads(_git(world.origin, "show", "refs/heads/next:pkg/lean-audit.json"))["guarantees"]) == {"F", "G2", "H", "K"}



def test_drive_and_ready_take_the_preflight_root_they_are_given(tmp_path):
    """Drive's pod keeps TMPDIR on a network volume, where a merged tree's `uv sync` can't hardlink from uv's cache and outruns
    the budget; `--preflight-root` puts the trees on a local disk instead, and without it they go to the temporary directory."""
    from research.queue import Preflight

    assert Preflight(tmp_path, "origin", None, root=tmp_path / "fast").root == tmp_path / "fast"
    assert Preflight(tmp_path, "origin", None).root == Path(tempfile.gettempdir())


def test_a_fetch_holds_the_clones_worktree_lock_shared_and_other_git_takes_none(tmp_path, monkeypatch):
    """A fetch reads every worktree's HEAD, and `git worktree add` writes a placeholder one first: a sync's fetch beside a
    restack job's add failed on it (`bad object worktrees/restack-…/HEAD`). So each fetch holds the clone's worktree lock
    shared, which `tools/move/restack.py` holds alone while it adds or forgets a scratch worktree, and fetches side by side,
    like every other git command, wait on nothing."""
    import fcntl

    _git(tmp_path, "init", "-q", "-b", "main", "origin")
    _git(tmp_path / "origin", "commit", "-q", "--allow-empty", "-m", "c")
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin"), "clone")
    clone, seen, real = tmp_path / "clone", [], subprocess.run

    def probed(argv, *args, **kw):
        if (sub := Q._subcommand(tuple(argv[1:]))) in ("fetch", "status"):
            with open(clone / ".git" / Q.WORKTREES_LOCK, "a") as f:
                for op in (fcntl.LOCK_SH, fcntl.LOCK_EX):
                    try:
                        fcntl.flock(f, op | fcntl.LOCK_NB)
                        fcntl.flock(f, fcntl.LOCK_UN)
                        seen.append((sub, op, "free"))
                    except BlockingIOError:
                        seen.append((sub, op, "held"))
        return real(argv, *args, **kw)

    monkeypatch.setattr(Q.subprocess, "run", probed)
    Q._git(clone, *Q.STALL, "fetch", "-q", "origin")
    Q._git(clone, "status", "--short")
    assert seen == [("fetch", fcntl.LOCK_SH, "free"), ("fetch", fcntl.LOCK_EX, "held"),
                    ("status", fcntl.LOCK_SH, "free"), ("status", fcntl.LOCK_EX, "free")]


@pytest.mark.parametrize("rc, log, results, said, how, interrupted", [
    (0, "", [{"status": "clean", "result": "c0ffee"}], {"result": "c0ffee"}, "exit 0", False),
    (0, "", [{"status": "up-to-date", "result": "c0ffee"}], {"result": "c0ffee"}, "exit 0", False),
    (1, "a\nb", [{"status": "conflicted", "step": "moved", "files": ["a.py", "b.py"]}],
     {"why": f"{Q.RESTACK} conflicts at its moved merge in a.py, b.py"}, "exit 1: a | b", False),
    (1, "", [{"status": "based-after", "error": "based after the move\nmerge it as usual"}],
     {"why": f"{Q.RESTACK}: based-after: based after the move"}, "exit 1", False),
    (0, "", None, None, "exit 0; its results don't read (JSONDecodeError: Expecting value: line 1 column 1 (char 0))", False),
    (137, "Killed", None, None, "exit 137: Killed", False),
    (124, "", None, None, f"exit 124, stopped {Q.RESTACK_WAIT_S // 60} min after its start", False),
    (143, "Terminated", None, None, "exit 143: Terminated", True),
    (130, "", None, None, "exit 130", True),
])
def test_a_restack_jobs_end_reads_as_the_sync_caches_it(tmp_path, rc, log, results, said, how, interrupted):
    """What `restack.py --json` said of its head, as a sync caches it ({result} or {why}), how its job ended, and whether
    something else interrupted it (SIGTERM's or SIGINT's exit, not SIGKILL's or `timeout`'s own); its directories removed."""
    d, out = tmp_path / "job", tmp_path / "out"
    d.mkdir()
    out.mkdir()
    (d / "log").write_text(log)
    (d / "rc").write_text(f"{rc}\n")
    (out / Q.RESTACK_OUT).write_text("" if results is None else json.dumps(results))
    assert Q.restack_end({"dir": str(d), "out": str(out), "pid": os.getpid(), "at": 0.0}, 0.0) == (said, how, interrupted)
    assert not d.exists() and not out.exists()


@pytest.mark.parametrize("verb", ["drive", "baseline", "promote", "accept-checked", "drive-shard-add"])
def test_no_verb_moves_main_or_checked_beside_the_verified_queues_loop(verb, capsys):
    """`main` moves only as the verified queue's loop moves it: drive's tick and the verbs that moved `main` or `checked`
    beside it are gone, so a script still calling one fails before it reads or pushes anything."""
    with pytest.raises(SystemExit) as e:
        Q.main([verb, "0123456789ab", "--by", "ci"])
    assert e.value.code == 2 and f"invalid choice: '{verb}'" in capsys.readouterr().err


def test_first_is_refused_naming_the_priority_file_which_holds_the_landing_order(world, monkeypatch, capsys):
    """The landing order has one source, the priority file: a `--first` refuses the command before it reads anything, with
    the fix."""
    monkeypatch.chdir(world.dev)
    with pytest.raises(SystemExit) as e:
        Q.main(["sync", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo", "o/r",
                "--store", str(world.tmp / "store"), "--first", "7,8"])
    assert e.value.code == 2 and not (world.tmp / "q").exists()
    assert Q.FIRST_RETIRED in capsys.readouterr().err


def test_sync_takes_the_priority_file_it_is_given(world, monkeypatch):
    seen = []
    monkeypatch.setattr(Q.Queue, "sync", lambda self, dry_run=False: seen.append((self.priority_file, dry_run)))
    monkeypatch.chdir(world.dev)
    path = world.tmp / "priority"
    assert Q.main(["sync", "--dry-run", "--priority-file", str(path), "--workdir", str(world.tmp / "q" / "verity"), "--remote",
                   str(world.origin), "--repo", "o/r", "--store", str(world.tmp / "store")]) == 0
    assert [(str(p), d) for p, d in seen] == [(str(path), True)]

