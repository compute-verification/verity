"""The stale rule (`research.stale`) and the stacked-PR notice: `research queue status` marks a PR that
isn't marked ready at its head and has been idle `stale_days` stale; drive rings its owner once a day and closes it at
`stale_close_days` with a comment, keeping its branch, but never one held, a draft that says "keep open", or one on `next`;
after a landing, the owner of each unmarked PR stacked on a landed one hears once to merge `main`.  On `test_queue`'s
throwaway origin with its fake GitHub; commit dates and drive's clock are injected."""
from __future__ import annotations

import dataclasses
import json
import time

import pytest

import research.queue as Q
from research.queue import DRIVE_FLAGS, PR, DriveFlags, Member, QueueError, label_target
from research.stale import DAY_S, close_comment, stale_prs
from research.store import vocab
from test_queue import World, _git
from test_queue_drive import Harness

T0 = 1_800_000_000.0                                    # drive's clock, injected
DAY = int(DAY_S)


def _at(world: World, monkeypatch, files: dict[str, str], when: float, base: str = "origin/main") -> str:
    """`World.change` committed at ``when``."""
    monkeypatch.setenv("GIT_COMMITTER_DATE", f"@{int(when)} +0000")
    try:
        return world.change(files, base)
    finally:
        monkeypatch.delenv("GIT_COMMITTER_DATE")


def _iso(when: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_status_marks_an_unmarked_pr_idle_three_days_stale_by_its_head_and_its_labels(world, monkeypatch):
    gh = world.gh
    gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, T0 - 4 * DAY), ready=False)
    gh.put(2, _at(world, monkeypatch, {"b.txt": "2\n"}, T0 - DAY), ready=False)                     # pushed yesterday
    gh.put(3, _at(world, monkeypatch, {"c.txt": "3\n"}, T0 - 10 * DAY))                             # marked ready
    gh.put(4, _at(world, monkeypatch, {"d.txt": "4\n"}, T0 - 10 * DAY), labels=("hold",), ready=False)
    gh.put(5, h5 := _at(world, monkeypatch, {"e.txt": "5\n"}, T0 - 10 * DAY), ready=False)        # ejected yesterday
    gh.put(6, h6 := _at(world, monkeypatch, {"f.txt": "6\n"}, T0 - 10 * DAY), ready=False)        # reviewed 2 days ago
    gh.prs[7] = PR(7, _at(world, monkeypatch, {"g.txt": "7\n"}, T0 - 5 * DAY), "change 7", "Lane `lane-7`.", base="feature",
                   branch="cursor/w-7")
    _git(world.origin, "update-ref", "refs/pull/7/head", gh.prs[7].head)
    marks = {(5, h5): _iso(T0 - DAY)}
    q = world.queue(marked_at=lambda n, h: marks.get((n, h)))
    reviews = {(6, h6): [_iso(T0 - 2 * DAY)]}
    entries = q.status(lambda prs: stale_prs(q, prs, T0, 3, lambda n, h: reviews.get((n, h), [])), 7)
    stale = {n for n, e in entries.items() if e.idle_since is not None}
    assert stale == {1, 7}
    assert entries[1].idle_since == T0 - 4 * DAY
    assert entries[1].why.startswith(f"stale, idle since {_iso(T0 - 4 * DAY)[:10]} (drive closes it 7 days idle; push, mark it "
                                     "ready or close it): not marked ready at ")
    assert entries[7].state == "waiting" and entries[7].why.endswith("based on feature, and not marked ready at "
                                                                      f"{gh.prs[7].head[:8]}")
    assert stale_prs(q, gh.open_prs(), T0, 0) == {}, "0 is off"


def test_status_reads_stale_days_from_drive_toml_at_main(world, monkeypatch, capsys):
    """The command's own path: `stale_days` from drive.toml at `main`, and `stale`/`idle_since` in its JSON."""
    world.with_store_remote()
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(world.tmp / "store.toml"))
    sha = world.change({DRIVE_FLAGS: 'on = ["m1"]\nstale_days = 3\nstale_close_days = 7\n'})
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/heads/main")
    world.gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, 1_000_000_000), ready=False)         # 2001: long idle
    world.gh.put(2, world.change({"b.txt": "2\n"}), ready=False)                                     # just now
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    assert Q.main(["status", "--json", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo",
                   "o/r", "--store", str(world.tmp / "store")]) == 0
    rows = {r["pr"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows[1]["stale"] and rows[1]["idle_since"] == "2001-09-09T01:46:40Z" and "drive closes it 7 days idle" in rows[1]["why"]
    assert not rows[2]["stale"] and rows[2]["idle_since"] is None


def _driven(world, monkeypatch, **kw) -> Harness:
    h = Harness(world, monkeypatch)
    h.clock = T0
    h.d.stale_days, h.d.stale_close_days = kw.get("stale_days", 3), kw.get("stale_close_days", 7)
    h.d.reviews = lambda n, head: []
    h.q.handle = lambda pr: pr.owner
    h.closed = []

    def close(n: int, comment: str) -> None:
        h.closed.append((n, comment))
        del world.gh.prs[n]

    world.gh.close = close
    return h


def test_drive_rings_a_stale_prs_owner_once_a_day_and_closes_it_at_seven_days_keeping_its_branch(world, monkeypatch):
    gh = world.gh
    gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, T0 - 4 * DAY), ready=False)
    gh.put(2, _at(world, monkeypatch, {"b.txt": "2\n"}, T0 - 8 * DAY), ready=False)
    gh.prs[2] = dataclasses.replace(gh.prs[2], branch="cursor/w-2")
    gh.put(3, _at(world, monkeypatch, {"c.txt": "3\n"}, T0 - 30 * DAY), draft=True, ready=False,
           body="Lane `lane-3`.\n\nA reference branch: keep open.")
    gh.put(4, _at(world, monkeypatch, {"d.txt": "4\n"}, T0 - 30 * DAY), labels=("hold",), ready=False)
    gh.put(5, _at(world, monkeypatch, {"e.txt": "5\n"}, T0 - 30 * DAY))                              # marked: never stale
    gh.put(6, h6 := _at(world, monkeypatch, {"f.txt": "6\n"}, T0 - 30 * DAY))
    h = _driven(world, monkeypatch, stale_days=0, stale_close_days=0)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [5, 6] and h.posts == []
    gh.put(6, _at(world, monkeypatch, {"f.txt": "6 more\n"}, T0 - 20 * DAY, base=h6), ready=False)  # on `next` at its marked head
    h.d.stale_days, h.d.stale_close_days = 3, 7

    h.tick(dry_run=True)
    assert h.closed == [] and h.posts == [] and any("would close #2" in x for x in world.logs)

    h.tick()
    assert h.closed == [(2, close_comment(7))]
    assert close_comment(7) == ("Closed by drive after 7 days unmarked and idle (TOP-90 row 36). The branch stays: reopen the PR, "
                                "push or mark it ready, and it queues again.")
    assert sorted(h.posts) == [
        "@lane-1 queue: stale, not marked ready at their heads and idle 3+ days: #1 (4 d). Push, mark ready or close each; drive "
        "closes one 7 days idle, its branch kept",
        "@lane-2 queue: closed after 7 days unmarked and idle, branches kept (reopen one to go on): #2"]
    state = h.q.load_state()["drive"]
    assert state["stale_closed"] == {"cursor/w-2": 2}, "the branch sweep never archives it (`Driver._sweep`)"

    h.posts.clear()
    gh.put(7, _at(world, monkeypatch, {"g.txt": "7\n"}, T0 - 5 * DAY), ready=False)
    h.clock = T0 + 60
    h.tick()
    assert h.posts == [], "the rule runs at most hourly (`STALE_EVERY_S`)"
    h.clock = T0 + 3600
    h.tick()
    assert h.posts == ["@lane-7 queue: stale, not marked ready at their heads and idle 3+ days: #7 (5 d). Push, mark ready or "
                       "close each; drive closes one 7 days idle, its branch kept"], "#1 once a day"
    assert h.closed == [(2, close_comment(7))]
    h.posts.clear()
    h.clock = T0 + DAY
    h.tick()
    assert sorted(h.posts) == [f"@lane-{n} queue: stale, not marked ready at their heads and idle 3+ days: #{n} ({d} d). Push, "
                               "mark ready or close each; drive closes one 7 days idle, its branch kept" for n, d in ((1, 5), (7, 6))]
    told = h.q.load_state()["drive"]["told"]
    assert sorted(k for k in told if k.startswith("idle:")) == [f"idle:{n}@{_iso(T0 + DAY)[:10]}" for n in (1, 7)], \
        "a past day's rings leave `told`"
    h.clock = T0 + 3 * DAY
    h.tick()
    assert [n for n, _ in h.closed] == [2, 1, 7] and sorted(gh.prs) == [3, 4, 5, 6]


def test_drive_closes_nothing_with_stale_close_days_off(world, monkeypatch):
    world.gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, T0 - 30 * DAY), ready=False)
    h = _driven(world, monkeypatch, stale_close_days=0)
    h.tick()
    assert h.closed == [] and h.posts == ["@lane-1 queue: stale, not marked ready at their heads and idle 3+ days: #1 (30 d). "
                                          "Push, mark ready or close each"]


def test_a_pr_drive_toml_keeps_is_never_stale_rung_or_closed(world, monkeypatch):
    world.gh.put(1, _at(world, monkeypatch, {"a.txt": "1\n"}, T0 - 30 * DAY), ready=False)
    world.gh.put(2, _at(world, monkeypatch, {"b.txt": "2\n"}, T0 - 30 * DAY), ready=False)
    h = _driven(world, monkeypatch)
    h.d.stale_keep = (2,)
    h.tick()
    assert [n for n, _ in h.closed] == [1] and 2 in world.gh.prs and not any("#2" in x for x in h.posts)
    assert DriveFlags.parse("stale_keep = [2692, 2693]\n").stale_keep == (2692, 2693)
    with pytest.raises(QueueError, match="stale_keep"):
        DriveFlags.parse('stale_keep = ["2692"]\n')


def test_after_a_landing_the_owner_of_each_unmarked_pr_stacked_on_it_hears_once_to_merge_main(world, monkeypatch):
    """A PR whose branch contains a landed PR's head, based on `main` or on that PR's branch, needs `main`
    merged in; one marked ready lands from `next` as it is, and a merge would cost it its mark."""
    gh = world.gh
    gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    gh.put(2, world.change({"b.txt": "2\n"}, base=h1), ready=False)                                  # stacked, unmarked
    gh.put(3, world.change({"c.txt": "3\n"}, base=h1))                                                # stacked, marked
    gh.prs[4] = PR(4, world.change({"d.txt": "4\n"}, base=h1), "change 4", "Lane `lane-4`.", base="cursor/w-1",
                   branch="cursor/w-4")                                                               # on #1's branch
    _git(world.origin, "update-ref", "refs/pull/4/head", gh.prs[4].head)
    gh.put(5, world.change({"e.txt": "5\n"}), ready=False)                                           # not stacked
    h = _driven(world, monkeypatch, stale_days=0)
    h.d.reviews = lambda n, head: world.store.labels(label_target(n, head), key=vocab.REVIEWED)
    h.d.ready_by = lambda n, head: None
    assert Q.baseline(h.q, _git(world.origin, "rev-parse", "main"), "ci") == 0
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 3]
    world.store.add_label(label_target(1, h1), vocab.REVIEWED, "land", "reviewer")
    assert h.tick().startswith("landed #1"), world.logs
    tip = _git(world.origin, "rev-parse", "main")
    assert sorted(h.posts) == [f"@lane-2 queue: #2's base #1 landed in main at {tip[:12]}; merge main into #2",
                               f"@lane-4 queue: #4's base #1 landed in main at {tip[:12]}; merge main into #4"]
    told = h.q.load_state()["drive"]["told"]
    assert {"stacked:2<1", "stacked:4<1"} <= set(told)
    h.posts.clear()
    h.d._stacked({"told": told}, [Member(1, h1, "lane-1", tip)], tip)
    assert h.posts == [], "once per (PR, base)"
