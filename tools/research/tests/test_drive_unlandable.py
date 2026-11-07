"""The unlandable rule (drive.toml's `close_unlandable`, `drive.Driver._unlandable`): at most hourly, drive closes each unmarked
open PR based on `main` that nothing will land as it is, with a comment naming why and how to keep it open, its branch kept:
one whose head a landed PR of the same owner contains, or one whose change the landing branch has already.  With "log" it
only logs what it would close.  Never a PR marked ready at its head, held, a draft that says "keep open", one `stale_keep`
lists, one based on another branch or from another repository, nor one reopened at the head drive closed it at.  On
`test_queue`'s throwaway origin with its fake GitHub, `main` the landing branch; drive's clock is injected."""
from __future__ import annotations

import dataclasses

import pytest

import research.queue as Q
from research.drive import STALE_EVERY_S, UNLANDABLE_COMMENT
from research.queue import PR
from test_drive_stale import _driven
from test_queue import World, _git


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _land(world: World, n: int, head: str, owner: str) -> str:
    """``head`` merged onto `main` as the queue lands PR ``n`` of ``owner`` (`queue._message`): the landing's commit."""
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    _git(world.dev, "merge", "-q", "--no-ff", "-m", Q._message(PR(n, head, f"change {n}", f"Lane `{owner}`.")), head)
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    return _git(world.dev, "rev-parse", "HEAD")


def test_drive_closes_a_pr_a_landed_pr_of_its_owner_carries_or_whose_change_main_has_and_with_log_only_logs_it(world, monkeypatch):
    gh = world.gh
    h3 = world.change({"c.txt": "3\n"})
    h9 = world.change({"i.txt": "9\n"}, base=h3)
    h2 = world.change({"b.txt": "2\n"}, base=h9)
    h1 = world.change({"a.txt": "1\n"}, base=h2)                   # #1 contains #2's head, #9's and #3's
    h4 = world.change({"d.txt": "4\n"})
    h5 = world.change({"d.txt": "4\n", "e.txt": "5\n"})            # #5 brings #4's change, and one of its own
    h6 = world.change({"d.txt": "4\n", "f.txt": "6\n"})            # one of its files isn't on `main`
    kept = {10: {"e.txt": "5\n"}, 11: {"d.txt": "4\n", "e.txt": "5\n"}, 12: {"a.txt": "1\n"}, 13: {"b.txt": "2\n"},
            14: {"c.txt": "3\n"}}                                  # each change on `main` too, once #1 and #5 land
    heads = {n: world.change(files) for n, files in kept.items()}
    landing, landing5 = _land(world, 1, h1, "lane-1"), _land(world, 5, h5, "lane-5")
    gh.put(2, h2, body="Lane `lane-1`.", ready=False)
    gh.prs[2] = dataclasses.replace(gh.prs[2], branch="cursor/w-2")
    gh.put(3, h3, body="Lane `lane-1`.")                            # marked at its head
    gh.put(9, h9, ready=False)                                      # another owner's
    gh.put(4, h4, ready=False)
    gh.put(6, h6, ready=False)
    gh.put(10, heads[10], labels=("hold",), ready=False)
    gh.put(11, heads[11], draft=True, body="Lane `lane-11`.\n\nA reference branch: keep open.", ready=False)
    gh.put(12, heads[12], ready=False)
    gh.put(13, heads[13], ready=False)
    gh.prs[13] = dataclasses.replace(gh.prs[13], base="feature", branch="cursor/w-13")
    gh.put(14, heads[14], ready=False)
    gh.prs[14] = dataclasses.replace(gh.prs[14], head_repo="someone/verity")
    h = _driven(world, monkeypatch, stale_days=0, stale_close_days=0)
    h.d.stale_keep = (12,)

    h.tick()
    assert not any("close #" in x for x in world.logs) and "unlandable_judged" not in h.q.load_state()["drive"], "off by default"
    h.d.close_unlandable = "log"
    h.tick()
    why2 = f"#1, of the same owner, contains its head {h2[:12]} and landed on `main` at {landing[:12]}"
    why4 = (f"its change is on `main` already: the one file it changes is there as its head {h4[:12]} has it, since #5's landing "
            f"{landing5[:12]}")
    would = [f"research queue drive: would close #2: {why2}", f"research queue drive: would close #4: {why4}"]
    assert [x for x in world.logs if "close #" in x] == would and h.closed == [] and h.posts == []

    world.logs.clear()
    h.d.close_unlandable = "on"
    h.tick()
    assert not any("close #" in x for x in world.logs), "at most once an hour (`STALE_EVERY_S`)"
    h.clock += STALE_EVERY_S
    h.tick(dry_run=True)
    assert [x for x in world.logs if "close #" in x] == would and h.closed == []
    world.logs.clear()
    h.tick()
    assert h.closed == [(2, UNLANDABLE_COMMENT.format(why=why2, n=2)), (4, UNLANDABLE_COMMENT.format(why=why4, n=4))]
    assert h.closed[0][1] == (f"Closed by drive: {why2}, so nothing would land it as it is. Its branch stays: reopen the PR (one "
                              "click) and drive leaves it open at this head, or mark it ready (`research queue ready 2 --mark`) and "
                              "it queues again.")
    assert [x for x in world.logs if x.startswith("research queue drive: closed #")] == [
        f"research queue drive: closed #2: {why2}", f"research queue drive: closed #4: {why4}"]
    assert sorted(h.posts) == [f"@lane-{lane} queue: closed #{n}: nothing would land it as it stands (each PR's comment says "
                               "why), branches kept; reopen one and drive leaves it open at that head" for lane, n in ((1, 2), (4, 4))]
    state = h.q.load_state()["drive"]
    assert state["unlandable_closed"] == {"2": h2, "4": h4}
    assert state["stale_closed"] == {"cursor/w-2": 2}, "the branch sweeps keep its branch (`Driver._sweep`, `merge.sweep`)"
    assert sorted(gh.prs) == [3, 6, 9, 10, 11, 12, 13, 14]

    gh.put(2, h2, body="Lane `lane-1`.", ready=False)              # its owner reopens it
    h.clock += STALE_EVERY_S
    h.tick()
    assert [n for n, _ in h.closed] == [2, 4] and 2 in gh.prs, "a PR reopened at the head drive closed it at stays open"
    gh.put(2, h7 := world.change({"g.txt": "7\n"}, base=h2), body="Lane `lane-1`.", ready=False)
    _land(world, 7, h7, "lane-1")
    h.clock += STALE_EVERY_S
    h.tick()
    assert [n for n, _ in h.closed] == [2, 4, 2], "at a new head it is judged again"
