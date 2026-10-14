"""`research queue report`: one tick checks each of GitHub's merge groups and posts its `verity/check` status, once per state.
On `test_queue`'s throwaway origin, with the launch, the runs' states, the statuses and the merge-thread post injected."""
from __future__ import annotations

import pytest

from research.mergequeue import CONTEXT, GROUPS, Reporter
from test_queue import World, _git
from test_queue_drive import _check


class Statuses:
    def __init__(self):
        self.posted: list[tuple[str, str, str]] = []
        self.refuse = ""

    def post(self, sha, state, description):
        if self.refuse:
            return self.refuse
        self.posted.append((sha, state, description))
        return None


class Harness:
    def __init__(self, world: World, max_running: int = 1, **kw):
        self.world, self.launched, self.posts, self.cancelled, self.run_state = world, [], [], [], {}
        self.statuses = Statuses()
        self.q = world.queue(**kw)
        self.r = Reporter(self.q, on="m1", statuses=self.statuses, max_running=max_running, launch=self.launch,
                          states=lambda repo, ids: {i: self.run_state[i] for i in ids},
                          post=lambda text, to=(): self.posts.append(text) or True, now=lambda: 1_000_000.0,
                          cancel=lambda repo, ids, reason: self.cancelled.extend(ids), evidence=lambda repo, ids: [])

    def launch(self, repo, commit, machine, agreement):
        rid = f"r20261013-0000{len(self.launched):02d}-abcd"
        self.launched.append((commit, machine, agreement))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def group(self, pr: int, files: dict[str, str], base: str = "origin/main") -> str:
        """GitHub's merge group for ``pr``: a branch `gh-readonly-queue/main/pr-{pr}-{main}` on ``base`` writing ``files``."""
        sha = self.world.change(files, base)
        main = _git(self.world.origin, "rev-parse", "main")
        _git(self.world.origin, "update-ref", f"refs/heads/{GROUPS}pr-{pr}-{main}", sha)
        return sha

    def drop(self, pr: int) -> None:
        main = _git(self.world.origin, "rev-parse", "main")
        _git(self.world.origin, "update-ref", "-d", f"refs/heads/{GROUPS}pr-{pr}-{main}")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_an_unchecked_group_is_checked_once_and_reported_pending_once(world):
    h = Harness(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    g = h.group(1, {"a.txt": "1\n"})
    assert h.r.tick() == f"#1 {g[:12]} pending"
    assert h.launched == [(g, "m1", False)]
    (sha, state, why), = h.statuses.posted
    assert (sha, state) == (g, "pending") and why == "check r20261013-000000-abcd running"
    h.r.tick()
    assert len(h.launched) == 1 and len(h.statuses.posted) == 1                  # still running: no second check, no post


def test_groups_wait_while_max_running_checks_run_oldest_first_and_a_guarantee_change_sends_agreement(world):
    h = Harness(world)
    g1 = h.group(1, {"a.txt": "1\n"})
    g2 = h.group(2, {"pkg/lean-audit.json": "{}\n"}, base=g1)
    h.r.tick()
    assert h.launched == [(g1, "m1", False)]
    assert dict((s, w) for s, _, w in h.statuses.posted)[g2] == "waits: 1 merge-group checks running"
    h.r.max_running = 2
    h.r.tick()
    assert h.launched[-1] == (g2, "m1", True)


def test_a_passing_check_posts_success(world, monkeypatch):
    h = Harness(world)
    g = h.group(1, {"a.txt": "1\n"})
    h.r.tick()
    _check(world, monkeypatch, g)
    h.r.tick()
    sha, state, why = h.statuses.posted[-1]
    assert (sha, state) == (g, "success") and why.startswith("check r") and why.endswith(" passed")


def test_a_failed_check_posts_failure_and_tells_ci_and_the_owner_once(world, monkeypatch):
    h = Harness(world)
    h.r.d._handle = {1: "lane-1"}.get                                           # the registry's handle for #1's owner
    g = h.group(1, {"a.txt": "1\n"})
    h.r.tick()
    _check(world, monkeypatch, g, ("false",))
    h.r.tick()
    h.r.tick()
    assert [s for sha, s, _ in h.statuses.posted if sha == g][-1] == "failure"
    assert len(h.posts) == 1 and h.posts[0].startswith("@ci @lane-1 queue: merge group #1") and "failed its check" in h.posts[0]


def test_a_dropped_groups_running_check_is_cancelled_and_forgotten(world):
    h = Harness(world, max_running=2)
    g1 = h.group(1, {"a.txt": "1\n"})
    g2 = h.group(2, {"b.txt": "2\n"}, base=g1)
    h.r.tick()
    h.drop(2)
    h.r.tick()
    assert h.cancelled == ["r20261013-000001-abcd"]
    assert set(h.q.load_state()["report"]["launched"]) == {g1} and g2 not in h.q.load_state()["report"]["posted"]


def test_a_refused_status_is_posted_again_next_tick(world):
    h = Harness(world)
    g = h.group(1, {"a.txt": "1\n"})
    h.statuses.refuse = "HTTP 403: Resource not accessible by integration"
    h.r.tick()
    assert h.statuses.posted == [] and any("was refused (HTTP 403" in line for line in world.logs)
    h.statuses.refuse = ""
    h.r.tick()
    assert [(sha, s) for sha, s, _ in h.statuses.posted] == [(g, "pending")]


def test_dry_run_launches_nothing_posts_nothing_and_tells_no_one(world):
    h = Harness(world)
    h.group(1, {"a.txt": "1\n"})
    assert h.r.tick(dry_run=True).endswith("pending")
    assert h.launched == h.statuses.posted == h.posts == []
    assert any(f"would post {CONTEXT} pending" in line for line in world.logs)


def test_no_merge_groups_is_idle_and_other_branches_are_not_groups(world):
    h = Harness(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _git(world.origin, "update-ref", f"refs/heads/{GROUPS}not-a-group", world.change({"b.txt": "2\n"}))
    assert h.r.tick() == "idle"
    assert h.launched == h.statuses.posted == []
