"""The review landing (`land = "review"`: PRs land on an eyeball review, and the full check runs after landing).  A tick
lands `next`'s longest prefix whose every member has a land verdict at its head by a reviewer who isn't its owner, a Lean
change only once a Lean check passed; the after-landing run checks `main`, and its failure reverts what it implicates once
`main` passed that step before, bisecting when several PRs are implicated."""
from __future__ import annotations

import contextlib
import calendar
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import research.queue as Q
from research import merge
from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL, LEAN_STEP
from research.queue import PR, QueueError, StoreLabels, label_target, lean_change
from research.store import vocab
from test_queue import World, _git
from test_queue_drive import Harness, _failed_run, _killed_run, _live_run, _steps_run, _tip

EVERY = {s: "passed" for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _registry(world: World, *handles: str) -> None:
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane"} for h in handles]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")


def _tree(world: World, ref: str) -> str:
    return _git(world.origin, "rev-parse", f"{ref}^{{tree}}")


class Review(Harness):
    """A drive landing on reviews, whose verdicts are `reviewed` labels in the world's store.  Unless ``baseline`` is off, ci has
    recorded `main` green (`research queue baseline`), so a tick lands without first running the after-landing run on `main`."""

    def __init__(self, world, monkeypatch, baseline: bool = True, **kw):
        super().__init__(world, monkeypatch, **kw)
        self.d.reviews = lambda pr, head: world.store.labels(label_target(pr, head), key=vocab.REVIEWED)
        self.d.ready_by = lambda pr, head: next((lab.by for lab in sorted(world.store.labels(label_target(pr, head), key=vocab.READY),
                                                                          key=lambda lab: lab.ts, reverse=True)), None)
        self.hand: dict[str, list[str]] = {}                    # Lean checks of drive's run registry it didn't launch, by commit
        self.d.lean_runs = lambda commits, since: {c: ids for c, ids in self.hand.items() if c in commits}
        if baseline:
            assert Q.baseline(self.q, _git(world.origin, "rev-parse", "main"), "ci") == 0

    def review(self, pr: int, verdict: str = "land", by: str = "reviewer", ref: str | None = None) -> None:
        self.world.store.add_label(label_target(pr, self.world.gh.prs[pr].head), vocab.REVIEWED, verdict, by, ref=ref)

    def lean_ids(self) -> list[str]:
        return [r for r in self.run_state if not r.startswith("r-post-")]

    def postland(self) -> dict:
        return self.d.q.load_state()["drive"]["postland"]


def _subject(world: World, ref: str) -> str:
    return _git(world.origin, "log", "-1", "--format=%s", ref)


def _main(world: World) -> str:
    return _git(world.origin, "rev-parse", "main")


def _move_main(world: World, files: dict[str, str]) -> None:
    """A commit onto `main` that isn't a landing of drive's (the lander's `research merge`), so the next sync rebuilds `next`."""
    _git(world.dev, "push", "-q", "origin", f"{world.change(files)}:refs/heads/main")


def test_a_reviewed_prefix_lands_and_awaits_the_after_landing_run_of_every_step(world, monkeypatch):
    h = Review(world, monkeypatch)
    h.d.after_landing_no_cache = True
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    main = _main(world)
    assert _subject(world, "main").startswith("Queue: #2") and _subject(world, "main^1").startswith("Queue: #1")
    assert _subject(world, "next").startswith("Queue: #3") and _git(world.origin, "rev-parse", "next^1") == main
    assert h.launched == []                                     # no Lean check: no Lean changes
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": False, "no_cache": True, "circuit_cache": True,
                                                            "no_regenerate": True})]
    assert h.after and h.after[0][1] == {main: "", _git(world.origin, "rev-parse", "main^1"): ""}
    pl = h.postland()
    assert pl["pending"] == [main] and pl["run"]["skip"] == ["lean-agreement", "lean-changed"]
    assert pl["run"]["as"] == {"steps": [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL], "no_cache": True}
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", EVERY)]
    h.tick()
    pl = h.postland()
    assert pl["verified"] == main and pl["verified_as"]["no_cache"] is True
    assert "lean-agreement" in pl["verified_as"]["steps"]         # ci's baseline passed it, and nothing it checks changed since


def test_an_unreviewed_member_new_to_next_or_a_held_one_stops_the_prefix_below_it(world, monkeypatch):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(3)
    assert h.tick().startswith("landed #1"), world.logs        # the lift moves only what `next` held before the sync
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2)
    h.review(2, "hold", by="other", ref="reads a stale fixture")
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #2") and "held by other (reads a stale fixture)" in out, out
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2, by="other")                                     # each reviewer's newest verdict counts
    assert h.tick().startswith("landed #3")
    assert _main(world) == _tip(world)
    assert h.launched == []


def test_no_verdict_counts_by_the_owner_its_handle_or_its_ready_marks_author_whatever_the_case(world, monkeypatch):
    world.with_store_remote()
    _registry(world, "lane-1")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.store.add_label(label_target(1, world.gh.prs[1].head), vocab.READY, "true", "marker")
    for by in ("Lane-1", "@LANE-1", "Marker"):                  # #1's `Lane lane-1`, its handle, its ready mark's author
        h.review(1, by=by)
        out = h.tick()
        assert "no land verdict but its owner's own" in out and _subject(world, "main") == "change ['.agents/skills/using-slack/registry.json']", out
    with pytest.raises(QueueError, match="#1 is MARKER's own PR"):
        Q.review(h.q, world.store, StoreLabels(world.store, log=lambda s: None), 1, "land", None, "MARKER")
    h.review(1, by="@lane-7")
    assert h.tick().startswith("landed #1")


def test_a_lean_change_without_a_lock_change_waits_for_a_lean_check_and_the_after_landing_lean_audit_takes_no_regenerated_record(
        world, monkeypatch):
    """A PR that changes a guarantee's statement, or a definition it reads, without its lock would land unchecked if only a
    changed lean-audit.json held it, and the after-landing lean-audit, passing on regenerated records nothing commits, would
    leave `main`'s lock stale and the spec-change DM unsent."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"other/X.lean": "-- no lock in other/\n"}))
    world.gh.put(2, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    for n in (1, 2):
        h.review(n)
    h.q.sync()
    chain = h.q.chain()
    assert [lean_change(h.d.repo, m.commit) for m in chain] == [False, True]
    assert h.tick().startswith("landed #1"), world.logs
    assert h.launched == [(_tip(world), "m1", False)]           # `train.sh TOP MACHINES --deferred lean-audit`
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {"lean-build": "passed", LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #2"), world.logs
    assert all(kw["no_regenerate"] for _, _, kw in h.posted)


def test_a_lock_change_waits_for_a_passed_lean_check_of_a_next_commit_holding_it(world, monkeypatch):
    h = Review(world, monkeypatch, nonlean_first=False)       # a member that changes no Lean above a Lean change
    lock = json.dumps({"guarantees": {"T": "2"}, "reads": {}, "dependencies": {"M": "1"}})
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": lock}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #1"), world.logs
    top = _tip(world)
    assert h.launched == [(top, "m1", False)]                   # of the prefix's top
    (rid,) = h.lean_ids()
    assert _subject(world, "main").startswith("Queue: #1")
    assert not h.tick().startswith("landed") and len(h.launched) == 1   # it waits for the check it launched
    assert _subject(world, "main").startswith("Queue: #1")
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {"lean-build": "passed", LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #3"), world.logs
    assert _main(world) == top and len(h.launched) == 1


def _rebuild_next(world: World, monkeypatch) -> None:
    """`next` made again from the same merges onto the same `main`, as a sync rebuilding it from below makes it: each commit new,
    each tree the same."""
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-10-24T18:00:00Z")
    base = _main(world)
    commits = _git(world.origin, "rev-list", "--first-parent", "--reverse", f"{base}..next").split()
    top = base
    for c in commits:
        heads = _git(world.origin, "rev-list", "--parents", "-n", "1", c).split()[2:]
        top = _git(world.origin, "commit-tree", f"{c}^{{tree}}", "-p", top, *(x for h in heads for x in ("-p", h)), "-m",
                   _git(world.origin, "log", "-1", "--format=%B", c))
    _git(world.origin, "update-ref", "refs/heads/next", top)


def test_a_running_lean_check_holds_a_rebuilt_next_with_the_same_prs_heads_and_tree(world, monkeypatch):
    """A re-mark low on `next` rebuilds it and the check's commit leaves it; cancelling the check mid-audit would land nothing
    for hours.  A rebuilt `next` with the same PRs at the same heads onto the same
    `main` has the same tree there, which the check still covers."""
    h = Review(world, monkeypatch, nonlean_first=False)       # a member that changes no Lean above a Lean change
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    old = _tip(world)
    assert h.launched == [(old, "m1", False)]
    _rebuild_next(world, monkeypatch)
    new = _tip(world)
    assert new != old and _git(world.origin, "rev-parse", f"{new}^{{tree}}") == _git(world.origin, "rev-parse", f"{old}^{{tree}}")
    assert not h.tick().startswith("landed")
    assert h.cancelled == [] and len(h.launched) == 1 and list(h.d.q.load_state()["drive"]["lean"]) == [new]
    h.run_state[h.lean_ids()[0]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[0], {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #2"), world.logs
    assert _main(world) == new and len(h.launched) == 1


def test_a_moved_lean_check_that_regenerated_locks_runs_again_on_the_rebuilt_commit(world, monkeypatch):
    """`_land` builds the regenerated locks on the commit the check covers, and its lock gate takes only a run of that very
    commit: a moved check that regenerated locks would cover the rebuilt `next` and be refused at every tick."""
    h = Review(world, monkeypatch)
    h.d.regenerate_locks = True
    h.d._regenerated = lambda drive, run: True
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    _rebuild_next(world, monkeypatch)
    new = _tip(world)
    h.tick()
    h.run_state[h.lean_ids()[0]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[0], {LEAN_STEP: "passed"})]
    assert not h.tick().startswith("landed")
    assert len(h.launched) == 2 and h.launched[1][0] == new, h.launched
    assert h.d.q.load_state()["drive"]["lean"][new]["state"] is None


def _lock_landing(world, monkeypatch, tmp_path, above: dict[str, str], lock: str | None = None,
                  passed: bool = False) -> tuple[Review, str, str]:
    """#1 (a Lean change with its lock) checked and landed with the records its check regenerated, `{"records": 2}`, while a
    check stacked at `next`'s top for #2 (a Lean change in the same package: without its lock, or stacked on #1 with ``lock``
    as its lock) runs, or with ``passed`` has passed already; its run dir holds ``above``'s records.  (the harness, that check's
    run id, the commit it ran on)"""
    h = Review(world, monkeypatch, nonlean_first=False)
    h.d.regenerate_locks = True
    world.gh.put(1, world.change({"pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": '{"records": 1}\n'}))
    world.gh.put(2, world.change({"pkg/B.lean": "theorem b : True := trivial\n", **({"pkg/lean-audit.json": lock} if lock else {})},
                                 base=world.gh.prs[1].head if lock else "origin/main"))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    h.review(1)
    h.tick()
    h.tick()
    one, top = h.q.chain()[0].commit, _tip(world)
    assert [c for c, *_ in h.launched] == [one, top], world.logs        # #1's, and one stacked at the top for #2
    first, second = h.lean_ids()
    runs = {first: _live_run(tmp_path, first, one, {"pkg": '{"records": 2}\n'})[1],
            second: _live_run(tmp_path, second, top, above)[1]}
    h.d.fetch = lambda r, names: runs[r](r, names) if r in runs else None
    h.run_state[first] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, first, {LEAN_STEP: "passed"})]
    if passed:
        h.run_state[second] = ("done", 0, None)
        h.evidence.append(_steps_run(world.tmp, second, {LEAN_STEP: "passed"}))
    assert h.tick().startswith("landed #1"), world.logs
    assert _subject(world, "main").startswith("Lock records regenerated for #1") and _git(world.origin, "rev-parse", "main^") == one
    world.gh.prs.pop(1)                                                 # GitHub closes it as merged
    return h, second, top


@pytest.mark.parametrize("records", ['{"records": 3}\n', '{"records": 2}\n'])
def test_a_landing_with_regenerated_locks_rebuilds_next_on_its_lock_commit_and_the_check_above_moves_with_it(
        world, monkeypatch, tmp_path, records):
    """A landing with a `Lock records regenerated` commit on top leaves `main` no commit of `next`, and a sync would merge every
    member again: the passed and running Lean checks above would go with their commits.  `next` is rebuilt on the lock commit at once, each member with its merge's tree and those records, the
    next sync keeps them, and the check above, running on, moves to its commit's image and lands it with its own records: the
    tree it checked with them, which the lock gate checks against the commit it ran on.  When its records are the ones below,
    the image holds them already, and it lands as it is."""
    h, rid, top = _lock_landing(world, monkeypatch, tmp_path, {"pkg": records})
    lock, new = _main(world), _tip(world)
    assert _git(world.origin, "rev-parse", f"{new}~2") == lock                  # #2's image, then #3's, on the lock commit
    assert _git(world.origin, "diff", "--name-only", top, new) == "pkg/lean-audit.json"
    assert _git(world.origin, "show", f"{new}:pkg/lean-audit.json") == '{"records": 2}'
    lean = h.d.q.load_state()["drive"]["lean"]
    assert top not in lean and lean[new]["carried"] == {"from": top, "files": ["pkg/lean-audit.json"]}, lean
    assert lean[new]["state"] is None and h.cancelled == []
    logged = len(world.logs)
    h.tick()
    assert any("2 PR(s) above `main`, kept 2, merged 0" in x for x in world.logs[logged:]), world.logs[logged:]
    assert _tip(world) == new and h.cancelled == [] and list(h.d.q.load_state()["drive"]["lean"]) == [new]
    h.review(2)
    h.review(3)
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #3"), world.logs
    main = _main(world)
    assert len(h.launched) == 2 and _git(world.origin, "show", f"{main}:pkg/lean-audit.json") == records.strip()
    assert _git(world.origin, "diff", "--name-only", top, main) == "pkg/lean-audit.json"
    if records == '{"records": 2}\n':
        assert main == new
        return
    assert _git(world.origin, "rev-parse", "main^") == new
    kw = {"tool": "check", "kinds": merge.produced(h.q.workdir, "check"), "command": merge.declared_command(h.q.workdir, "check"),
          "deferred": rid, "fetch": h.d.fetch}
    assert merge.regenerated_parent(world.origin, main, h.q.store, ran=top, **kw) == rid
    assert merge.regenerated_parent(world.origin, main, h.q.store, **kw) is None     # it didn't run on the parent


def test_a_member_above_whose_merge_changes_the_regenerated_lock_keeps_its_own_record_and_its_place(world, monkeypatch, tmp_path):
    """In the S1 tail most Lean changes above a landing change its lock too: each keeps its merge's record, so the rebuild
    doesn't stop there, and a check above it, of the same tree, covers its image as it is."""
    h, rid, top = _lock_landing(world, monkeypatch, tmp_path, {}, lock='{"records": 5}\n')
    lock, new = _main(world), _tip(world)
    assert _git(world.origin, "rev-parse", f"{new}~2") == lock and _tree(world, new) == _tree(world, top)
    assert h.d.q.load_state()["drive"]["lean"][new]["carried"] == {"from": top, "files": []}
    logged = len(world.logs)
    h.review(2)
    h.review(3)
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #3"), world.logs
    assert any("kept 2, merged 0" in x for x in world.logs[logged:]) and _main(world) == new and len(h.launched) == 2


def test_a_moved_check_that_didnt_regenerate_the_records_below_it_covers_nothing(world, monkeypatch, tmp_path):
    """The image holds the lock commit's records, and a check that passed without regenerating them checked a tree with other
    ones: it covers nothing, and the commit is checked again."""
    h, rid, top = _lock_landing(world, monkeypatch, tmp_path, {})
    lock, new = _main(world), _tip(world)
    h.review(2)
    h.review(3)
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert not h.tick().startswith("landed"), world.logs
    assert _main(world) == lock and len(h.launched) == 3 and h.launched[2][0] == new, h.launched
    assert any(f"{rid} ran on {top[:12]} and didn't regenerate pkg/lean-audit.json" in x for x in world.logs), world.logs


def test_pre_passed_carried_check_that_regenerated_nothing_covers_nothing(world, monkeypatch, tmp_path):
    """The check above passed before the landing below it moved it, so `_follow_lean` never judged it on
    its image.  Having regenerated nothing, it checked a tree without the lock commit's records, and there is no lock commit
    for the lock gate to refuse: it covers nothing, and the image is checked again."""
    h, _, top = _lock_landing(world, monkeypatch, tmp_path, {}, passed=True)
    lock, new = _main(world), _tip(world)
    rec = h.d.q.load_state()["drive"]["lean"][new]
    assert rec["state"] == "passed" and rec["carried"] == {"from": top, "files": ["pkg/lean-audit.json"]}, rec
    h.review(2)
    h.review(3)
    assert not h.tick().startswith("landed"), world.logs
    assert _main(world) == lock and len(h.launched) == 3 and h.launched[2][0] == new, h.launched


def test_a_pre_passed_carried_check_that_regenerated_the_lock_lands_with_its_records(world, monkeypatch, tmp_path):
    h, _, top = _lock_landing(world, monkeypatch, tmp_path, {"pkg": '{"records": 3}\n'}, passed=True)
    h.review(2)
    h.review(3)
    assert h.tick().startswith("landed #3"), world.logs
    main = _main(world)
    assert _git(world.origin, "diff", "--name-only", top, main) == "pkg/lean-audit.json" and len(h.launched) == 2
    assert _git(world.origin, "show", f"{main}:pkg/lean-audit.json") == '{"records": 3}'


def test_a_failed_lean_check_below_a_moved_check_that_didnt_regenerate_its_records_holds_the_landing(world, monkeypatch, tmp_path):
    """`_past_failed` lands past a failed Lean check only on a passed one above it that checked the tree that lands.  ci's hand
    check of `next`'s top passed without regenerating records, #1 landed below it with its own, and a hand check of #2's image
    failed: with no Lean change between them, nothing judges the moved check, which checked the tree without #1's records."""
    h = Review(world, monkeypatch, nonlean_first=False)
    h.d.regenerate_locks = True
    world.gh.put(1, world.change({"pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": '{"records": 1}\n'}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    h.review(1)
    h.tick()
    one, _, top = (m.commit for m in h.q.chain())
    assert [c for c, *_ in h.launched] == [one], world.logs
    h.d.hand_lean(top, "ci")
    first, hand = h.lean_ids()
    runs = {first: _live_run(tmp_path, first, one, {"pkg": '{"records": 2}\n'})[1], hand: _live_run(tmp_path, hand, top, {})[1]}
    h.d.fetch = lambda r, names: runs[r](r, names) if r in runs else None
    h.run_state[first] = h.run_state[hand] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, first, {LEAN_STEP: "passed"}), _steps_run(world.tmp, hand, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    world.gh.prs.pop(1)
    lock, (two, new) = _main(world), (m.commit for m in h.q.chain())
    rec = h.d.q.load_state()["drive"]["lean"][new]
    assert rec["state"] == "passed" and rec["carried"] == {"from": top, "files": ["pkg/lean-audit.json"]}, rec
    h.d.hand_lean(two, "ci")
    h.run_state[h.lean_ids()[2]] = ("done", 1, None)
    h.review(2)
    h.review(3)
    out = h.tick()
    assert h.d.q.load_state()["drive"]["lean"][two]["state"] == "failed"
    assert out.startswith("nothing lands: #2") and _main(world) == lock, (out, world.logs)


def test_a_moved_running_check_keeps_only_the_members_above_the_lock_commit_and_survives_a_rebuild(world, monkeypatch, tmp_path):
    """`_rekey_lean` matches a check's members against `next`'s bottom: the image of a moved check holds the members above the
    lock commit, so a rebuilt `next` with the same PRs at the same heads on it keeps the check running."""
    h, _, _ = _lock_landing(world, monkeypatch, tmp_path, {})
    new = _tip(world)
    assert h.d.q.load_state()["drive"]["lean"][new]["members"] == [[2, world.gh.prs[2].head], [3, world.gh.prs[3].head]]
    _rebuild_next(world, monkeypatch)
    rebuilt = _tip(world)
    assert rebuilt != new
    assert not h.tick().startswith("landed")
    lean = h.d.q.load_state()["drive"]["lean"]
    assert h.cancelled == [] and list(lean) == [rebuilt] and lean[rebuilt]["rekeyed_from"] == new, (lean, world.logs)


def test_a_store_run_of_a_moved_checks_image_lands_its_records_checked_against_the_image(world, monkeypatch, tmp_path):
    """The lock commit names, and the lock gate checks it against, the commit `_carry` moved a check from only when that check's
    run is the one whose records land: a recorded run of the image itself covers it while the moved check runs on, and ran on
    the image, which the gate checks it against."""
    h, _, _ = _lock_landing(world, monkeypatch, tmp_path, {})
    new, rid = _tip(world), "r20261027-010000-abcd"
    fetch, live = h.d.fetch, _live_run(tmp_path, rid, new, {"pkg": '{"records": 4}\n'})[1]
    h.d.fetch = lambda r, names: live(r, names) if r == rid else fetch(r, names)
    index, sql, status = type(h.q.store.index), type(h.q.store.index).sql, merge.step_status
    monkeypatch.setattr(index, "sql", lambda self, query, params=(): [(rid,)] if "source_dirty = 0" in query and params == (new,)
                        else sql(self, query, params))
    monkeypatch.setattr(merge, "step_status", lambda store, r, step: "passed" if r == rid else status(store, r, step))
    h.review(2)
    h.review(3)
    assert h.tick().startswith("landed #3"), world.logs
    main = _main(world)
    assert _git(world.origin, "rev-parse", "main^") == new and _git(world.origin, "show", f"{main}:pkg/lean-audit.json") == '{"records": 4}'
    assert f"Checked: {rid} (the Lean check of {new[:12]}," in _git(world.origin, "log", "-1", "--format=%B", main)


def test_the_lean_check_stops_below_a_member_that_came_onto_next_within_the_settle_ticks(world, monkeypatch):
    """A PR marked moments ago is the one most likely to be re-marked: a check of the prefix's top through it would leave `next`
    at its next push, so the check stops below it."""
    h = Review(world, monkeypatch, nonlean_first=False)       # a member that changes no Lean above a Lean change
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (2, 3):
        h.review(n)
    h.tick()
    two, three = (m.commit for m in h.q.chain())
    assert h.launched == [(two, "m1", False)]                             # not #3's, which just came
    h.run_state[h.lean_ids()[0]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[0], {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #3") and _main(world) == three     # #3 changes no Lean


def test_a_member_that_came_onto_next_a_tick_ago_no_longer_stops_the_lean_check(world, monkeypatch):
    """One new head at position 2 doesn't cap the check at position 1 for more than one settle tick."""
    h = Review(world, monkeypatch, nonlean_first=False)       # a member that changes no Lean above a Lean change
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    h.review(3)
    assert not h.tick().startswith("landed") and h.launched == []         # #2 unreviewed: no reviewed Lean change
    h.review(2)
    h.tick()
    assert h.launched == [(_tip(world), "m1", False)], world.logs         # #3 came a tick ago: the check reaches it


def _stacked(world, monkeypatch) -> tuple[Review, str, str]:
    """A Lean check running at #2, the reviewed top, and one stacked at #3, an unreviewed Lean change that came above it a tick
    later: (the harness, #2's commit, #3's)."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.review(2)
    h.tick()
    two, three = (m.commit for m in h.q.chain())
    assert h.launched == [(two, "m1", False)], world.logs                 # #3 just came: nothing stacked through it yet
    assert not h.tick().startswith("landed")
    assert h.launched == [(two, "m1", False), (three, "m1", False)], world.logs
    assert h.d.q.load_state()["drive"]["lean"][three]["stacked"] is True
    return h, two, three


def test_a_proof_only_lean_change_lands_on_its_verdict_and_a_lock_or_code_lean_change_waits(world, monkeypatch):
    """Lean in `Security/` that touches no lock is proofs and lands on its land verdict (the
    after-landing lean-audit reverts it on a failure); a lock, or Lean outside `Security/` (it runs as code), still waits."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"Security/lean-audit.json": "{}\n", "Security/Proofs/A.lean": "theorem A : True := trivial\n"}))
    world.gh.put(2, world.change({"Security/Proofs/B.lean": "theorem B : True := trivial\n"}))
    world.gh.put(3, world.change({"pkg/C.lean": "theorem C : True := trivial\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.q.sync()
    assert {m.pr: lean_change(h.d.repo, m.commit) for m in h.q.chain()} == {1: True, 2: False, 3: True}
    assert h.q.chain()[0].pr == 2                                         # proofs only: non-Lean first puts it below


def test_a_two_demos_lean_change_lands_on_its_verdict_and_another_waits(world, monkeypatch):
    """A PR whose title says "two demos" lands on its land verdict with no Lean check first (its records passed at its
    ready mark, and the after-landing lean-audit checks it); any other lock change still waits for one."""
    h = Review(world, monkeypatch)
    world.gh.put(1, head := world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.prs[1] = PR(1, head, "Two demos: the lock", "Lane `lane-1`.")
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{\n}"}, base=head))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #1"), world.logs
    assert h.launched and h.launched[0][0] != head, world.logs              # #2's own check, never #1's


def test_a_lean_check_is_stacked_at_nexts_top_for_a_lean_change_above_a_running_one(world, monkeypatch):
    """Lean changes that come onto `next` above a running Lean check's commit don't wait for it to pass, then for a whole
    check of their own (~1 h).  While it runs, one is stacked at `next`'s top, reviewed or not, with no free check slot needed,
    at most `max_lean_checks` running; the lower one is never cancelled for it, and the landing takes the highest that passed."""
    h, two, three = _stacked(world, monkeypatch)
    world.gh.put(4, world.change({"pkg/U.lean": "theorem U : True := trivial\n"}))
    h.tick()
    h.tick()                                                              # #4 settled: two running, so no third
    assert len(h.launched) == 2 and h.cancelled == [], world.logs
    h.d.max_lean_checks = 3
    h.tick()
    assert len(h.launched) == 3 and h.launched[2][0] == _tip(world) and h.cancelled == [], world.logs
    rid = h.lean_ids()[1]
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    h.review(3)
    assert h.tick().startswith("landed #3"), world.logs                   # #3's check passed, #2's still runs
    assert _main(world) == three


def test_nothing_is_stacked_for_a_member_above_the_running_check_that_changes_no_lean(world, monkeypatch):
    """No check is stacked at the reviewed top just because a check slot is free: a member above the running check that
    changes no Lean lands on that check alone."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    h.tick()
    one = _tip(world)
    assert h.launched == [(one, "m1", False)]
    world.gh.put(2, world.change({"c.txt": "2\n"}))
    h.review(2)
    h.tick()
    h.tick()
    assert h.launched == [(one, "m1", False)], world.logs
    h.run_state[h.lean_ids()[0]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[0], {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #2"), world.logs


def test_an_ejection_from_a_stacked_checks_prefix_rebuilds_next_at_once_and_cancels_only_that_check(world, monkeypatch):
    """An ejection from a running Lean check's prefix waits for it to end before `next` is rebuilt; a stacked check reaches
    `next`'s top, so that would hold every sync for its whole run.  It is rebuilt at once, the stacked check cancelled, and the
    lower one runs on."""
    h, two, three = _stacked(world, monkeypatch)
    world.gh.ready.discard((3, world.gh.prs[3].head))
    assert Q.lean_running(h.d.q.load_state(), 3) is None
    h.tick()
    assert [m.pr for m in h.q.chain()] == [2] and h.cancelled == [], world.logs
    h.tick()
    assert h.cancelled == [h.lean_ids()[1]] and list(h.d.q.load_state()["drive"]["lean"]) == [two]


@pytest.mark.parametrize("case, lean, changes, churned, want", [
    ("a Lean change above the running check: the top", {"c1": None}, "13", None, 4),
    ("capped below what came just now", {"c1": None}, "13", 4, 3),
    ("only what came just now changes Lean", {"c1": None}, "14", 4, None),
    ("no Lean change above the running check", {"c3": None}, "13", None, None),
    ("nothing runs", {"c1": "passed"}, "13", None, None),
    ("one failed", {"c1": None, "c0": "failed"}, "13", None, None),
    ("at the bound", {"c1": None, "c3": None}, "134", None, None),
    ("the bound counts a check off `next`", {"c1": None, "gone": None}, "13", None, None),
    ("covered by a passed check above", {"c1": None, "c3": "passed"}, "13", None, None),
    ("a Lean change above a passed check above the running one", {"c1": None, "c2": "passed"}, "13", None, 4),
])
def test_lean_stack_decides_from_the_checks_verdicts_where_a_lean_check_is_stacked(case, lean, changes, churned, want):
    """`next` is c0..c4; ``changes`` names the members that change Lean, and ``churned`` the one that came onto it just now."""
    from research.drive import lean_stack
    from research.queue import Member

    chain = [Member(n, f"h{n}", "o", f"c{n}") for n in range(5)]
    got = lean_stack(chain, {c: {"ids": ["r"], "state": st} for c, st in lean.items()}, lambda m: str(m.pr) in changes,
                     lambda m: m.pr == churned, 2)
    assert got == want, case
    assert lean_stack(chain, {"c0": {"ids": ["r"], "state": None}}, lambda m: m.pr > 0, lambda m: False, 3) == 2   # two free


@pytest.mark.parametrize("case, changes, lo, top, slots, want", [
    ("four Lean changes, two checks: below the third", "0123", 0, 3, 2, 1),
    ("what changes no Lean below the next Lean change goes with the cut", "0245", 0, 5, 2, 3),
    ("three checks: every two", "012345", 0, 5, 3, 1),
    ("one check: the top", "0123", 0, 3, 1, 3),
    ("none free counts as one", "0123", 0, 3, 0, 3),
    ("one Lean change: the top", "2", 0, 5, 2, 5),
    ("an odd count rounds the lower part up", "123", 1, 4, 2, 2),
    ("only from lo", "0145", 2, 5, 2, 4),
])
def test_lean_cut_spaces_the_lean_checks_through_the_lean_changes(case, changes, lo, top, slots, want):
    """`next` is c0..c5; ``changes`` names the members that change Lean."""
    from research.drive import lean_cut
    from research.queue import Member

    chain = [Member(n, f"h{n}", "o", f"c{n}") for n in range(6)]
    assert lean_cut(chain, lo, top, lambda m: str(m.pr) in changes, slots) == want, case


def test_lean_checks_start_at_cut_points_lowest_first_and_a_failure_holds_only_what_is_above_it(world, monkeypatch):
    """With room for two Lean checks, four reviewed Lean changes get one check through the lower two and one stacked at the top
    in the same tick; the upper one failing holds only #3 and #4, which then check their own commit, and #1 and #2 land on the
    lower check."""
    h = Review(world, monkeypatch)
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/L{n}.lean": f"-- {n}\n"}))
        h.review(n)
    assert not h.tick().startswith("landed"), world.logs
    chain = h.q.chain()
    assert [m.pr for m in chain] == [1, 2, 3, 4]
    assert h.launched == [(chain[1].commit, "m1", False), (chain[3].commit, "m1", False)], world.logs
    low, high = h.lean_ids()
    assert h.d.q.load_state()["drive"]["lean"][chain[3].commit]["stacked"] is True
    h.run_state[high] = ("done", 1, None)
    h.evidence = [_steps_run(world.tmp, high, {LEAN_STEP: "failed"})]
    assert not h.tick().startswith("landed") and h.cancelled == [] and len(h.launched) == 2, world.logs
    h.run_state[low] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, low, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #2"), world.logs
    assert _main(world) == chain[1].commit
    assert h.launched[2:] == [(chain[2].commit, "m1", False)], world.logs       # #3's own commit, before blaming it


def test_members_that_change_no_lean_go_below_a_lean_change_and_land_while_its_lean_check_runs(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    assert [m.pr for m in h.q.chain()] == [1] and h.launched == [(_tip(world), "m1", False)]


def test_reviewed_members_that_change_no_lean_above_a_lean_change_at_the_bottom_go_below_it_before_its_check_runs(
        world, monkeypatch):
    """A Lean change already at the bottom of `next` would hold every reviewed member above it for its Lean check: non-Lean
    first orders only what a sync adds, and the check launches on the reviewed prefix's top, holding them all.  While no Lean check of `next` runs, they go below it and land that tick, and its check starts on
    what is left.  Once it runs, nothing moves below it: one reviewed after waits above it, and lands on its pass."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3] and h.launched == []        # #1, kept at the bottom, holds #2 and #3
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    assert _subject(world, "main").startswith("Queue: #3") and _subject(world, "main^1").startswith("Queue: #2")
    assert [m.pr for m in h.q.chain()] == [1] and h.launched == [(_tip(world), "m1", False)] and h.cancelled == []
    assert any("#2 #3 #1 (reviewed) go to the bottom" in x for x in world.logs), world.logs
    world.gh.put(4, world.change({"c.txt": "4\n"}))
    h.review(4)
    assert not h.tick().startswith("landed")
    assert [m.pr for m in h.q.chain()] == [1, 4] and h.cancelled == [] and len(h.launched) == 1
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #4"), world.logs
    assert len(h.launched) == 1


def test_a_lift_leaves_an_unreviewed_member_a_pr_stacked_on_the_lean_change_and_a_lean_change_whose_own_check_failed(
        world, monkeypatch):
    """Only what lands at once goes below the Lean change: an unreviewed member there would keep it from its check, a PR
    stacked on it contains it, and a Lean change whose own check failed waits for its owner's fix or hold, not a check
    again on every lift."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.tick()
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}, base=world.gh.prs[1].head))
    world.gh.put(4, world.change({"c.txt": "4\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4]
    for n in (3, 4):
        h.review(n)
    assert h.tick().startswith("landed #4"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 2, 3] and h.launched == []
    h.review(1)
    h.tick()
    (rid,) = h.lean_ids()
    assert h.launched == [(h.q.chain()[0].commit, "m1", False)]                  # #1's own: #2 is unreviewed
    h.run_state[rid] = ("done", 1, None)
    h.tick()
    assert h.d.q.load_state()["drive"]["lean"][h.q.chain()[0].commit]["state"] == "failed"
    h.review(2)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3] and len(h.launched) == 1, world.logs


def test_an_unreviewed_member_below_reviewed_ones_no_longer_stops_them_landing(world, monkeypatch):
    """A member with no land verdict held every reviewed member above it, deep in `next`.  The members with a land verdict
    go below those waiting for one, and land that tick; the one waiting holds only itself."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3]
    h.review(2)
    h.review(3)
    assert h.tick().startswith("landed #3"), world.logs
    assert _subject(world, "main").startswith("Queue: #3") and _subject(world, "main^1").startswith("Queue: #2")
    assert [m.pr for m in h.q.chain()] == [1]
    assert any("#2 #3 (reviewed) go to the bottom" in x for x in world.logs), world.logs
    h.review(1)
    assert h.tick().startswith("landed #1") and _main(world) == _tip(world)


def test_a_held_member_still_stops_the_members_above_it_a_hold_is_a_verdict(world, monkeypatch):
    """Nothing moves past a held member: what is below it lands, past an unreviewed one, and what is above it waits."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    h.review(4)
    assert h.tick().startswith("landed #2"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 3, 4]
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #1"), out
    assert [m.pr for m in h.q.chain()] == [1, 3, 4] and _subject(world, "main").startswith("Queue: #2")


def test_reviewed_first_keeps_a_stacked_pr_after_the_pr_it_contains(world, monkeypatch):
    """A reviewed PR stacked on an unreviewed one stays above it; one stacked on a reviewed one goes down after it."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}, base=world.gh.prs[1].head))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    world.gh.put(4, world.change({"d.txt": "4\n"}, base=world.gh.prs[3].head))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4]
    for n in (2, 3, 4):
        h.review(n)
    assert h.tick().startswith("landed #4"), world.logs
    assert _subject(world, "main").startswith("Queue: #4") and _subject(world, "main^1").startswith("Queue: #3")
    assert [m.pr for m in h.q.chain()] == [1, 2]
    h.review(1)
    assert h.tick().startswith("landed #2") and _main(world) == _tip(world)


def test_a_member_whose_labels_cant_be_read_stops_the_lift_not_the_tick(world, monkeypatch):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    h.review(3)
    reviews = h.d.reviews

    def stale(pr, head):
        if pr == 3:
            raise QueueError(f"STALE: the labels of pr:3@{head} weren't pulled from the store's remote")
        return reviews(pr, head)

    h.d.reviews = stale
    assert h.tick().startswith("landed #2"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 3]
    assert any("the lift stops below #3: STALE" in x for x in world.logs), world.logs


def test_reviewed_first_off_keeps_an_unreviewed_member_where_it_is(world, monkeypatch):
    h = Review(world, monkeypatch, reviewed_first=False)
    for n, f in ((1, "a.txt"), (2, "b.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 2]


@pytest.mark.parametrize("fault", ["enospc", "after-audit"])
def test_a_lean_check_failed_by_its_machine_runs_again_once_on_another_machine(world, monkeypatch, tmp_path, fault):
    """A Lean check that hits ENOSPC after every audit passed is no failure of its change.  A check whose step ran out of disk, or that failed after its lean-audit passed, runs again once, on the machines it didn't
    run on; failed by its machine again, it is the change's."""
    h = Review(world, monkeypatch)
    h.d.on = "vy-a,vy-b,vy-c"
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    h.tick()
    top = _tip(world)
    (rid,) = h.lean_ids()
    d = tmp_path / rid
    d.mkdir()
    if fault == "enospc":
        (d / "steps.jsonl").write_text(json.dumps({"step": LEAN_STEP, "status": "failed", "rc": 1, "log": "lean-audit.log"}) + "\n")
        (d / "lean-audit.log").write_text("AUDIT /x/lean-audit-scratch-1/pkg: PASS\nOSError: [Errno 28] No space left on device\n")
    else:
        (d / "steps.jsonl").write_text(json.dumps({"step": LEAN_STEP, "status": "passed"}) + "\n")
    h.evidence = [d]
    h.run_state[rid] = ("done", 1, None, None, "vy-b")
    assert not h.tick().startswith("landed")
    assert h.launched == [(top, "vy-a,vy-b,vy-c", False), (top, "vy-a,vy-c", False)], world.logs
    assert any("a machine's fault: again, once, on vy-a,vy-c" in x for x in world.logs), world.logs
    again = h.lean_ids()[1]
    h.run_state[again] = ("done", 1, None, None, "vy-a")
    h.tick()
    assert len(h.launched) == 2 and h.d.q.load_state()["drive"]["lean"][top]["state"] == "failed"
    (post,) = [p for p in h.posts if "of its own commit" in p]
    assert "rc=1" in post, post


def test_an_ejection_during_a_lean_check_of_its_prefix_waits_to_rebuild_next_and_lands_nothing_from_it_up(
        world, monkeypatch, capsys):
    """An eject synced `next` at once, so the running Lean check's commit left it and the check was cancelled."""
    import research.drive as D
    from research.cli import main as research_main

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    top = _tip(world)
    assert h.launched == [(top, "m1", False)]
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    monkeypatch.setattr(D, "merge_thread_post", lambda *a, **k: True)
    monkeypatch.setenv("RESEARCH_DRIVE", "1")
    assert research_main(["queue", "eject", "3", "--reason", "x", "--by", "ci", "--workdir", str(h.q.workdir), "--remote",
                          str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "store")]) == 0
    assert "the tick after that check ends" in capsys.readouterr().out and _tip(world) == top
    world.gh.ready.discard((3, world.gh.prs[3].head))                    # the store's `ready = false`, as drive's `ready` reads it
    h.run_state[h.lean_ids()[0]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[0], {LEAN_STEP: "passed"})]
    out = h.tick()
    assert out.startswith("landed #2"), out
    assert h.cancelled == [] and _subject(world, "main").startswith("Queue: #2")


def test_a_pr_re_marked_at_its_ejected_head_lands(world, monkeypatch):
    """The queue's record of an ejection outlives a re-mark at the same head (`Queue._marked_again` re-admits it): read as
    withdrawn, it held the PR, and every PR above it, off `main` for good."""
    h = Review(world, monkeypatch)
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    h.review(1)
    h.q.sync()
    h.q.eject(1, "x", "ci")
    world.gh.ready.discard((1, head))
    assert not h.tick().startswith("landed")
    world.gh.ready.add((1, head))                                           # its owner marks it again at that head
    h.q.marked_at = lambda n, hd: "2099-01-01T00:00:00Z"
    assert h.tick().startswith("landed #1"), world.logs


def test_a_lean_check_covers_only_while_its_commit_is_on_next(world, monkeypatch):
    """A passed Lean check of a commit `next` was rebuilt away from checked a tree that no longer lands: `main` moved under it, so
    the check is dropped and the Lean change waits for one of the new `next`."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    assert not h.tick().startswith("landed")
    old = _tip(world)
    assert h.launched == [(old, "m1", False)]
    (rid,) = h.lean_ids()
    _move_main(world, {"b.txt": "moved\n"})
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    out = h.tick()
    assert not out.startswith("landed"), out
    new = _tip(world)
    assert new != old and h.launched == [(old, "m1", False), (new, "m1", False)], world.logs
    assert list(h.d.q.load_state()["drive"]["lean"]) == [new]
    h.run_state[h.lean_ids()[-1]] = ("done", 0, None)
    assert h.tick().startswith("landed #1")


def test_a_failed_lean_check_above_a_lean_change_checks_its_own_commit_before_blaming_it(world, monkeypatch):
    """The check of the prefix's top failing says only that something at or below it broke Lean: the Lean change is checked at
    its own commit, lands when that passes, and its owner hears nothing; the failed top, no Lean change of its own, is `CI`'s."""
    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch, nonlean_first=False)       # a member that changes no Lean above a Lean change
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    own, top = (m.commit for m in h.q.chain())
    assert h.launched == [(top, "m1", False)]
    h.run_state[h.lean_ids()[0]] = ("done", 1, None)
    assert not h.tick().startswith("landed")
    assert h.launched[1] == (own, "m1", False) and not [p for p in h.posts if "@lane-1" in p]
    h.run_state[h.lean_ids()[1]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[1], {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #2"), world.logs                 # the failed top dropped: it wedged all above it
    assert _main(world) == top and len(h.launched) == 2
    (post,) = [p for p in h.posts if "changes no Lean" in p]
    assert post.startswith("@ci queue:") and "@lane-2" not in post and "drive drops it" in post, post


def test_a_killed_lean_check_runs_again_once_and_one_lost_past_the_bound_too(world, monkeypatch):
    """A Lean check the machine killed (or whose state can't be read for hours) said nothing of the Lean change, yet counted as
    failed for good, and an `unknown` one held `next` forever."""
    from research.drive import LOST_AFTER_S

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    h.tick()
    top = _tip(world)
    h.run_state[h.lean_ids()[0]] = ("done", 137, None)
    assert not h.tick().startswith("landed")
    assert h.launched == [(top, "m1", False)] * 2
    h.run_state[h.lean_ids()[1]] = ("unknown", None, None)
    h.tick()
    assert len(h.launched) == 2                                          # within the bound it still waits
    h.clock += LOST_AFTER_S + 1
    out = h.tick()
    assert len(h.launched) == 2 and not out.startswith("landed"), out   # resent once: lost again, it failed
    (post,) = [p for p in h.posts if "of its own commit" in p]
    assert "lost on its resend" in post, post


def test_a_failed_lean_check_of_its_own_commit_tells_its_owner_and_runs_again_once_the_commit_leaves_next(world, monkeypatch):
    _registry(world, "lane-2")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    h.run_state[h.lean_ids()[0]] = ("done", 1, None)
    h.tick()
    h.tick()
    (post,) = [p for p in h.posts if "changes Lean" in p]
    assert post.startswith("@ci @lane-2 queue: #2 at") and "of its own commit" in post and "failed" in post, post
    assert len(h.launched) == 1 and _subject(world, "main").startswith("Queue: #1")
    _move_main(world, {"c.txt": "moved\n"})
    h.tick()
    assert len(h.launched) == 2 and h.launched[1][0] == _tip(world)


def test_nothing_lands_before_the_after_landing_run_of_main_itself_and_a_red_main_lands_reverting_nothing_on_its_red_step(
        world, monkeypatch, tmp_path):
    """A first after-landing run fails on anything `main` already had: reverting the landings it follows would revert innocent
    PRs.  So the review landing's first run is of `main` itself, and nothing lands until it ends.  Failed, `main` is verified
    with the steps it passed alone: landing goes on (a red `main` cleared only by a false baseline wedged it), and a later
    failure of a step red then reverts nothing."""
    _registry(world, "lane-2")
    h = Review(world, monkeypatch, baseline=False)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    h.review(2)
    main = _main(world)
    out = h.tick()
    assert out.startswith("nothing lands before an after-landing run of `main`"), out
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": True, "no_cache": False, "circuit_cache": False,
                                                            "no_regenerate": True})]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {**{s: "passed" for s in AFTER_LANDING}, "lean-suites": "failed"})]
    assert h.tick().startswith("landed #2"), world.logs
    (post,) = [p for p in h.posts if "baseline after-landing run" in p]
    assert "was red before any review landing" in post and "until a run of `main` passes lean-suites" in post, post
    assert h.postland()["verified"] == main and "lean-suites" not in h.postland()["verified_as"]["steps"]
    tip = _main(world)
    h.run_state["r-post-1"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert _main(world) == tip
    (post,) = [p for p in h.posts if "Reverted nothing" in p]
    assert "suite pkg is red" in post and "It never passed in an after-landing run since" in post, post
    assert "@lane-2" not in post, post


def test_ci_records_a_baseline_only_on_the_control_pod_and_it_clears_an_open_bisection(world, monkeypatch):
    """Off the control pod `research queue baseline` wrote a state file there that drive never reads, and said nothing."""
    from research.merge import DRIVE_ENV

    h = Review(world, monkeypatch, baseline=False)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    main = _main(world)
    h.q._clone()
    state = h.q.load_state()
    state["drive"] = {"launched": {}, "told": [], "postland": {"verified": main, "pending": [], "run": None, "bisect": {
        "tip": main, "base": main, "steps": ["pytest"], "why": "x", "key": "step pytest", "no_cache": False, "cands": [], "good": -1, "bad": 0,
        "run": {"ids": ["r-gone"], "i": 0, "at": 0}}}}
    h.q.save_state(state)
    argv = ["baseline", main, "--by", "ci", "--workdir", str(h.q.workdir), "--remote", str(world.origin), "--repo", "o/r",
            "--store", str(world.tmp / "store"), "--rules", str(world.dev / "ci" / "queue.toml")]
    monkeypatch.delenv(DRIVE_ENV, raising=False)
    assert Q.main(argv) == 1 and "bisect" in h.postland()
    with pytest.raises(QueueError, match="only ci"):
        Q.baseline(h.q, main, "lane-3")
    monkeypatch.setenv(DRIVE_ENV, "1")
    with pytest.raises(QueueError, match="not after-landing steps"):
        Q.baseline(h.q, main, "ci", "nonsense")
    assert Q.main([*argv, "--steps", "lean-audit"]) == 0 and h.postland()["verified_as"]["steps"] == ["lean-audit"]
    assert Q.main(argv) == 0 and "bisect" not in h.postland()
    assert h.tick().startswith("landed #1")


def _owning(world: World, owns: dict[str, list[str]]) -> None:
    """`main`'s registry, each handle owning its paths (`research msg owners`)."""
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane", "owns": paths} for h, paths in owns.items()]}
    _git(world.dev, "push", "-q", "origin", f"{world.change({'.agents/skills/using-slack/registry.json': json.dumps(reg)})}"
                                          ":refs/heads/main")


def _red_landing(world, monkeypatch, tmp_path) -> tuple[Review, str]:
    """#1, #2 and #3 land; #2 breaks suite pkg (owned by @pkg-owner), whose after-landing run `r-post-0` fails."""
    _owning(world, {"lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    assert [c for c, *_ in h.posted] == [tip]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    return h, tip


def test_an_after_landing_failure_rings_its_culprit_and_the_suites_owner_and_reverts_it_an_hour_later(world, monkeypatch,
                                                                                                       tmp_path):
    """Without a culprit, PRs keep landing on a `main` whose after-landing runs all fail, each failure going to @ci alone.
    Drive names the one landing whose change touches the failing test, rings its owner and the suite's with the run and the first
    error line, and reverts it once `FIX_WINDOW_S` passed with no fix marked."""
    from research.drive import FIX_WINDOW_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    assert _main(world) == tip
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert ring.startswith("@ci @pkg-owner @lane-2 queue: suite pkg is red on `main`") and "(run r-post-0)" in ring, ring
    assert "failed tests/test_x.py::test_y" in ring and "NameError: name '_once' is not defined" in ring, ring
    assert "Culprit: #2 (its change touches pkg/tests/test_x.py)" in ring and "unless a fix is marked" in ring, ring
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 2
    h.clock += FIX_WINDOW_S - 1
    h.tick()
    assert _main(world) == tip and not [p for p in h.posts if "reverted" in p]
    h.clock += 2
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == tip                            # one revert on top of the landing
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`") and "no fix was marked within 60 min" in post, post
    assert h.posted[-1][0] == main and h.launched == []                                  # the next run checks the new tip
    h.run_state[f"r-post-{len(h.posted) - 1}"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, f"r-post-{len(h.posted) - 1}", {s: "passed" for s in h.posted[-1][1].split(",")})]
    (h.evidence[0] / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "passed"}]))
    h.tick()
    assert h.postland()["verified"] == main and not h.postland()["red"] and h.postland()["green"]["suite pkg"] == main


@pytest.mark.parametrize("fix", ["names-the-culprit", "fixes-suite", "touches-the-failure", "new-head"])
def test_a_marked_fix_holds_the_culprits_revert_while_it_stays_marked(world, monkeypatch, tmp_path, fix):
    """A fix is a ready PR whose body names the culprit or says `fixes suite NAME`, or whose change touches what the failure
    named, or a new head of the culprit; each body case changes an unrelated file, so only its body can count."""
    from research.drive import FIX_WINDOW_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    if fix == "new-head":
        world.gh.put(n := 2, head := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    elif fix == "touches-the-failure":
        world.gh.put(n := 9, head := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body="Lane `lane-9`.")
    else:
        world.gh.put(n := 9, head := world.change({"elsewhere.txt": "9\n"}),
                     body="Lane `lane-9`. " + ("FIXES suite pkg: test_y." if fix == "fixes-suite" else "Fixes #2's test."))
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == tip and any(f"a fix is marked, #{n} at {head[:8]}" in x for x in world.logs), world.logs
    world.gh.ready.discard((n, head))                                   # its owner withdrew the fix: the clock ran out
    h.tick()
    assert _git(world.origin, "rev-parse", "main^") == tip and any("#2 reverted on `main`" in p for p in h.posts), h.posts


def test_a_landed_fix_holds_the_culprits_revert_until_a_run_of_a_main_with_it_is_red_again(world, monkeypatch, tmp_path):
    """A fix counted only while it is an open PR would have drive ask for the culprit's revert after its fix landed.  A fix that landed after the red tip
    holds the revert, by its merge's title; once a run of a `main` with it is red again, it no longer does."""
    from research.drive import FIX_WINDOW_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    world.gh.put(9, world.change({"elsewhere.txt": "9\n"}))
    world.gh.prs[9] = Q.PR(9, world.gh.prs[9].head, "fixes suite pkg", "Lane `lane-9`.")
    h.review(9)
    assert h.tick().startswith("landed #9"), world.logs
    fixed = _main(world)
    del world.gh.prs[9]                                                  # landed: no open PR holds the revert any more
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == fixed and any("a fix is marked, #9, landed at" in x for x in world.logs), world.logs
    h.run_state[f"r-post-{len(h.posted) - 1}"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path / "again", "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    h.tick()
    assert any("#2 reverted on `main`" in p for p in h.posts), (h.posts, world.logs)


def test_an_unrelated_pr_that_names_the_red_suite_holds_no_revert(world, monkeypatch, tmp_path):
    """Counting any ready PR whose body names the suite would let a suite called `research` be held
    for good by whatever PR mentions research.  Neither the name nor `suite research` without `fixes` holds the revert."""
    from research.drive import FIX_WINDOW_S

    _owning(world, {"lane-2": [], "ci": ["tools/research/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"tools/research/tests/test_x.py": "def test_y():\n    assert False\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2")
    tip = _main(world)
    run = tmp_path / "run"
    run.mkdir()
    (run / "steps.jsonl").write_text(json.dumps({"step": "pytest", "status": "failed"}) + "\n")
    (run / "suites.json").write_text(json.dumps([{"name": "research", "suite": "tools/research", "status": "failed",
                                                  "failed_tests": ["tests/test_x.py::test_y"]}]))
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "suite research is red" in ring and "Culprit: #2" in ring and "says `fixes suite research`" in ring, ring
    world.gh.put(9, world.change({"docs/notes.txt": "9\n"}),
                 body="Lane `lane-9`. research: the research store's docs, as suite research reads them.")
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _git(world.origin, "rev-parse", "main^") == tip and any("#2 reverted on `main`" in p for p in h.posts), h.posts
    assert not any("a fix is marked, #" in x for x in world.logs), world.logs


def _one_red(world, monkeypatch, tmp_path) -> tuple[Review, str]:
    """#1 breaks suite pkg and lands; its after-landing run `r-post-0` fails, ringing #1 as its culprit."""
    _registry(world, "lane-1")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 1
    return h, tip


def _reverts(world: World, since: str) -> int:
    return len(_git(world.origin, "log", "--format=%H", "--grep=^This reverts commit", f"{since}..main").split())


def test_a_culprits_revert_that_conflicts_leaves_main_and_tells_its_owner_to_revert_by_hand(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)
    _move_main(world, {"pkg/tests/test_x.py": "def test_y():\n    assert False  # edited on main\n"})
    moved = _main(world)
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == moved
    (post,) = [p for p in h.posts if "must be reverted by hand" in p]
    assert post.startswith("@ci @lane-1 queue: #1 must be reverted by hand") and "the revert didn't apply" in post, post


def test_a_culprit_reverted_by_hand_is_not_reverted_again_nor_told_as_a_hand_revert(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)
    merge_commit = _git(world.origin, "log", "--format=%H", "--grep=^PR: #1$", "main").split()[0]
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    _git(world.dev, "revert", "--no-edit", "-m", "1", merge_commit)
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    fixed = _main(world)
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == fixed and _reverts(world, tip) == 1
    assert h.postland()["red"]["suite pkg"]["reverted"] == "undone"
    assert not any("by hand" in p or "reverted on `main`" in p for p in h.posts), h.posts


def test_a_revert_whose_push_loses_a_race_is_retried_on_the_next_tick(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)
    real, raced = h.q._remote, []

    def remote(repo, *args, **kw):
        if "main:refs/heads/main" in args and not raced:
            _move_main(world, {"c.txt": "landed in between\n"})
            raced.append(_main(world))
        return real(repo, *args, **kw)

    monkeypatch.setattr(h.q, "_remote", remote)
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert raced and _main(world) == raced[0] and not any("reverted" in p or "by hand" in p for p in h.posts), h.posts
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == raced[0] and _reverts(world, tip) == 1
    assert _git(world.origin, "show", f"{main}:c.txt") == "landed in between"
    assert any(p.startswith("@ci @lane-1 queue: #1 reverted on `main`") for p in h.posts), h.posts


def test_a_drive_killed_after_a_revert_pushed_does_not_revert_again_on_restart(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)

    def killed(text, to=()):
        raise KeyboardInterrupt("the drive is killed")

    h.clock += FIX_WINDOW_S + 1
    h.d.post = killed
    with pytest.raises(KeyboardInterrupt):
        h.tick()
    assert _reverts(world, tip) == 1
    again = Review(world, monkeypatch, baseline=False)
    again.clock, again.run_state, again.evidence, again.posted = h.clock, h.run_state, h.evidence, h.posted
    again.tick()
    assert _reverts(world, tip) == 1 and not any("by hand" in p for p in again.posts), again.posts


def test_a_drive_restarted_while_its_after_landing_run_runs_reads_it_back_and_verifies_its_tip(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    again = Review(world, monkeypatch, baseline=False)                  # a new process: only state.json carries over
    again.run_state, again.posted = h.run_state, h.posted
    again.run_state["r-post-0"] = ("done", 0, None)
    again.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING})]
    again.tick()
    assert again.postland()["verified"] == tip and again.postland()["pending"] == [] and len(h.posted) == 1


def test_an_after_landing_run_the_machine_killed_runs_again_once_and_killed_again_is_told(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 137, None)
    h.evidence = [_killed_run(tmp_path, "r-post-0")]
    h.tick()
    assert [c for c, *_ in h.posted] == [tip, tip] and h.postland()["run"]["resend"]
    h.run_state["r-post-1"] = ("done", 137, None)
    h.evidence = [_killed_run(tmp_path / "again", "r-post-1")]
    h.tick()
    assert [p for p in h.posts if "killed again on its resend" in p], h.posts
    assert _main(world) == tip and h.postland()["failed"] == tip


def test_a_red_after_landing_run_runs_on_and_is_cancelled_only_with_its_circuit_check_tail_left(world, monkeypatch, tmp_path):
    """Cancelling an after-landing run at its first failed suite leaves the rest of a ~4 h run unknown, and the cancel kills
    lean-agreement mid-step, which reads as failed.  A red run rings at once and runs on until every step but
    circuit-check and flock-rust-tests has ended, so every red suite is named; a step a cancel stopped reads `not run`."""
    from research.drive import TAIL

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    run = _failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2", steps="")
    (run / "suites-lean.failed.json").write_text((run / "suites-lean.json").read_text())
    (run / "suites-lean.json").unlink()
    steps = {"lean-build": "passed", "lean-agreement": "stopped"}
    (run / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    h.run_state["r-post-0"] = ("running", None, None, "suite pkg (tests/test_x.py::test_y)")
    h.evidence = [run]
    h.tick()
    assert h.cancelled == [] and h.postland()["run"]["ids"] == ["r-post-0"]
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "suite pkg is red" in ring and "Culprit: #1" in ring and "Not run (a cancel stopped them): lean-agreement" in ring
    h.tick()
    assert h.cancelled == [] and len([p for p in h.posts if "is red on" in p]) == 1             # rung once
    steps |= {s: "passed" for s in AFTER_LANDING if s not in TAIL and s not in steps}
    steps["lean-suites"] = "failed"
    (run / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    h.tick()
    assert h.cancelled == ["r-post-0"] and h.postland()["run"] is None and h.postland()["green"]["step lean-audit"] == _main(world)
    assert "step lean-agreement" not in h.postland()["red"] and "step lean-suites" not in h.postland()["red"]


def test_a_read_guard_failure_rings_the_pr_that_added_the_reading_test(world, monkeypatch, tmp_path):
    """A suite the read guard failed names no failing test: its record's `outside` (path -> the test module that read it) is what
    implicates the PR that added the reading test."""
    _owning(world, {"lane-1": [], "lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_reads.py": "def test_r():\n    open('data/x.json')\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2")
    run = tmp_path / "run"
    run.mkdir()
    (run / "steps.jsonl").write_text(json.dumps({"step": "pytest", "status": "failed"}) + "\n")
    (run / "suites.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed",
                                                  "outside": {"data/x.json": "pkg/tests/test_reads.py"}}]))
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert ring.startswith("@ci @pkg-owner @lane-1 queue: suite pkg is red"), ring
    assert "read outside its declared inputs: data/x.json" in ring and "Culprit: #1" in ring, ring


def test_main_far_behind_its_last_verified_commit_rings_ci_and_top_every_hour(world, monkeypatch, tmp_path):
    """A `main` many hours without a verified commit must reach someone.  Once its oldest unverified landing is
    `BEHIND_S` old, drive rings @ci and @top, again every `BEHIND_EVERY_S` while it stays behind."""
    from research.drive import BEHIND_EVERY_S, BEHIND_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    world.gh.put(9, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body="Lane `lane-9`. Fixes pkg.")
    h.clock += BEHIND_S - 10
    h.tick()
    assert not [p for p in h.posts if "behind its tip" in p] and _main(world) == tip
    h.clock += 20
    h.tick()
    (alarm,) = [p for p in h.posts if "behind its tip" in p]
    assert alarm.startswith("@ci @top queue: `main`'s last verified commit") and "red: suite pkg (culprit #2)" in alarm, alarm
    h.clock += BEHIND_EVERY_S - 30
    h.tick()
    assert len([p for p in h.posts if "behind its tip" in p]) == 1
    h.clock += 60
    h.tick()
    assert len([p for p in h.posts if "behind its tip" in p]) == 2


def test_a_healthy_after_landing_run_under_way_is_not_an_alarm(world, monkeypatch):
    """A cold after-landing run takes ~4 h, so `main` is routinely more than `BEHIND_S` behind while its run is green so far:
    drive rings only once that run has gone `POSTLAND_RUN_S`."""
    from research.drive import BEHIND_S, POSTLAND_RUN_S

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    h.clock += BEHIND_S + 60
    h.tick()
    assert not [p for p in h.posts if "behind its tip" in p] and h.postland()["run"]["ids"] == ["r-post-0"]
    h.clock += POSTLAND_RUN_S
    h.tick()
    (alarm,) = [p for p in h.posts if "behind its tip" in p]
    assert "its run r-post-0" in alarm and "red: nothing named red" in alarm, alarm


def test_an_after_landing_failure_several_prs_touch_is_bisected_to_the_one_that_broke_it(world, monkeypatch, tmp_path):
    """Reverting every landing a failure's files reach took innocent PRs with the culprit: each step of the bisection checks one
    landing's commit and reads the failed suite alone, and only the first landing that fails it is rung, then reverted.
    Landing goes on meanwhile: the bisection checks fixed commits."""
    from research.drive import FIX_WINDOW_S

    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch)
    world.gh.put(1, h1 := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}, base=h1))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    one, two = _git(world.origin, "rev-parse", "main^1^1"), _git(world.origin, "rev-parse", "main^1")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    world.gh.put(4, world.change({"c.txt": "4\n"}))
    h.review(4)
    out = h.tick()
    assert out.startswith("landed #4"), out
    assert h.posted[1] == (one, "lean-suites", {"agreement": False, "no_cache": False, "circuit_cache": False,
                                                   "no_regenerate": True})
    assert h.posted[2][0] == _main(world)                                        # #4's after-landing run
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "#1 #2 each touch it" in ring and "drive bisects them" in ring, ring
    h.run_state["r-post-1"] = ("done", 1, None)                                  # another suite failed there: pkg passed
    h.evidence = [_steps_run(world.tmp, "r-post-1", {"lean-suites": "failed"})]
    (h.evidence[0] / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "passed"},
                                                                 {"name": "other", "suite": "other", "status": "failed"}]))
    h.tick()
    assert h.posted[3][0] == two and _main(world) != tip
    h.run_state["r-post-3"] = ("done", 1, None)
    h.evidence = [_failed_run(world.tmp / "r3", "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert "bisect" not in h.postland() and not [p for p in h.posts if "reverted" in p]
    (ring,) = [p for p in h.posts if "Culprit: #2, by bisection" in p]
    assert ring.startswith("@ci @lane-2 queue: suite pkg is red") and "passes on #1's landing" in ring, ring
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    main = _main(world)
    revert = _git(world.origin, "rev-list", "--first-parent", "--reverse", f"{tip}..main").split()[-1]
    assert _subject(world, revert).startswith('Revert "Queue: #2')
    assert _git(world.origin, "show", f"{main}:pkg/tests/test_x.py") == "def test_y():\n    pass"
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`"), post


def _bisecting(world, monkeypatch, tmp_path, revert: bool = False) -> tuple[Review, str]:
    """#1 and #2 both change the failing test's file and land with #3; the after-landing run fails there, and drive bisects them
    (its first run, `r-post-1`, of #1's landing).  With ``revert``, a hand revert lands on `main` first."""
    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch)
    if revert:
        _git(world.dev, "fetch", "-q", "origin")
        _git(world.dev, "checkout", "-q", "--detach", "origin/main")
        _git(world.dev, "commit", "-q", "--allow-empty", "-m", f"Revert \"x\"\n\nThis reverts commit {_main(world)}.")
        _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    world.gh.put(1, h1 := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}, base=h1))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    return h, tip


def test_a_bisection_run_lost_past_the_bound_runs_once_more_then_ends_reverting_nothing(world, monkeypatch, tmp_path):
    """A run whose state can't be read counted as running for ever: while a bisection is open nothing lands, and ci's baseline
    was refused, so `next` stopped until someone edited drive's state by hand."""
    from research.drive import LOST_AFTER_S

    h, tip = _bisecting(world, monkeypatch, tmp_path)
    assert len(h.posted) == 2
    h.run_state["r-post-1"] = ("unknown", None, None)
    h.tick()
    assert len(h.posted) == 2
    h.clock += LOST_AFTER_S + 1
    h.tick()
    assert len(h.posted) == 3 and h.posted[2][0] == h.posted[1][0]       # the same landing once more
    h.run_state["r-post-2"] = ("unknown", None, None)
    h.clock += LOST_AFTER_S + 1
    out = h.tick()
    assert "bisect" not in h.postland() and _main(world) == tip, out
    (post,) = [p for p in h.posts if "ended twice" in p]
    assert post.startswith("@ci queue:") and "(lost)" in post and "Reverted nothing" in post, post


def test_a_bisection_run_fails_a_landing_only_on_a_failed_step_it_bisects(world, monkeypatch, tmp_path):
    """Any run that wasn't clean marked its landing bad, so an exit 0 whose records couldn't be read, or a failure in another
    step (setup, an older tree's ci.toml), reverted an innocent PR."""
    h, tip = _bisecting(world, monkeypatch, tmp_path)
    h.run_state["r-post-1"] = ("done", 0, None)
    h.evidence = []                                                         # exit 0, its steps.jsonl unread
    h.tick()
    assert len(h.posted) == 3 and h.posted[2][0] == h.posted[1][0] and h.postland()["bisect"]["bad"] == 2
    h.run_state["r-post-2"] = ("done", 2, None)
    h.evidence = [_steps_run(world.tmp, "r-post-2", {"setup": "failed"})]
    h.tick()
    assert "bisect" not in h.postland() and _main(world) == tip and not [p for p in h.posts if "reverted" in p]
    (post,) = [p for p in h.posts if "ended twice" in p]
    assert "neither passing nor failing pkg" in post, post


def test_a_bisection_with_a_revert_among_its_landings_reverts_nothing(world, monkeypatch, tmp_path):
    """A revert inside the range means a landing's commit isn't what `main` ran: bisecting it could blame the wrong PR."""
    h, tip = _bisecting(world, monkeypatch, tmp_path, revert=True)
    assert "bisect" not in h.postland() and len(h.posted) == 1 and _main(world) == tip
    (post,) = [p for p in h.posts if "a revert lies in" in p]
    assert "Reverted nothing" in post, post


def test_a_lean_check_above_the_landing_top_covers_it_only_with_no_lean_change_between(world, monkeypatch):
    """A passed check of a commit above what lands checked a tree with a Lean change the landing leaves out, which may fix what
    the landed one breaks: the check runs again on the prefix's top."""
    h = Review(world, monkeypatch)
    h.d.max_lean_checks = 1                                   # one check through both, no cut between them
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    one, top = (m.commit for m in h.q.chain())
    assert h.launched == [(top, "m1", False)]
    h.review(2, "hold", by="other", ref="not yet")
    h.run_state[h.lean_ids()[0]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[0], {LEAN_STEP: "passed"})]
    assert not h.tick().startswith("landed")
    assert h.launched[1] == (one, "m1", False), h.launched
    h.run_state[h.lean_ids()[1]] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, h.lean_ids()[1], {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1") and _main(world) == one


def test_a_hand_lean_check_holds_the_members_up_to_its_commit_until_it_ends(world, monkeypatch):
    """A `--first` rebuild of `next` before hand Lean checks (`train.sh COMMIT MACHINES --deferred lean-audit`) landed anything
    would throw them away: drive holds `next` under them as under its own."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    assert not h.tick().startswith("landed")
    one, two = (m.commit for m in h.q.chain()[:2])
    h.hand[two] = ["r20261026-063429-f8e3"]
    h.run_state["r20261026-063429-f8e3"] = ("running", None, None)
    h.q.first = (4,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 4, 3], world.logs        # `--first` goes above the running check
    assert [m.commit for m in h.q.chain()[:2]] == [one, two] and h.launched == []
    assert any("didn't launch hold `next` up to them: " + two[:12] in s for s in world.logs), world.logs
    h.run_state["r20261026-063429-f8e3"] = ("done", 0, None)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [4, 1, 2, 3], world.logs


def test_lean_runs_reads_the_registry_s_live_lean_checks_of_the_commits_asked_and_a_registry_that_cant_be_read_holds_none(
        world, monkeypatch, tmp_path):
    from research.drive import lean_runs

    check = ["python3", "tools/verity/check/slot.py", "--priority", "--", "uv", "run", "python", "tools/verity/check/check.py",
             "--keep-going"]

    def run(rid: str, commit: str, extra: list[str], state: str | None = "running") -> None:
        d = tmp_path / rid
        d.mkdir()
        (d / "remote.json").write_text(json.dumps({"run_id": rid, "phase": "launched", "command": check + extra,
                                                   "source_sha": commit, "remote_run_dir": f"/r/{rid}"}))
        if state:
            (d / "status.json").write_text(json.dumps({"state": state}))

    run("r20261026-063429-f8e3", "c1", ["--deferred", "lean-audit"])
    run("r20261026-063430-8e1b", "c1", ["--deferred", "pytest,lean-audit"], state=None)     # launched, no records yet
    run("r20261026-063554-88c4", "c2", ["--deferred", "lean-audit"], state="cancelled")
    run("r20261026-063555-a3e0", "c2", [])                                               # a whole check runs lean-audit
    run("r20261026-063556-0001", "c2", ["--deferred", "pytest"])
    run("r20261026-063557-0002", "c3", ["--deferred", "lean-audit"])                     # not a commit asked of
    run("r20261025-063557-0003", "c1", ["--deferred", "lean-audit"])                     # before the window
    (tmp_path / "src").mkdir()
    since = calendar.timegm((2026, 10, 26, 0, 0, 0))
    assert lean_runs({"c1", "c2"}, since, tmp_path) == {"c1": ["r20261026-063429-f8e3", "r20261026-063430-8e1b"],
                                                        "c2": ["r20261026-063555-a3e0"]}

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    assert not h.tick().startswith("landed")

    def unreadable(commits, since):
        raise OSError("runs root gone")

    h.d.lean_runs = unreadable
    h.q.first = (2,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [2, 1]
    assert sum("run registry's Lean checks can't be read (OSError: runs root gone)" in s for s in world.logs) == 1, world.logs


def test_a_landing_that_does_not_compile_lands_what_is_below_the_pr_that_broke_it(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/x.py": "def f(:\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #1"), world.logs
    assert _subject(world, "main").startswith("Queue: #1")
    (post,) = [p for p in h.posts if "doesn't compile" in p]
    assert "#2: pkg/x.py: SyntaxError" in post, post


def test_a_store_that_cant_be_read_covers_nothing_and_a_launched_lean_check_survives_a_failed_tick(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    for n in (1, 2):
        h.review(n)

    index, sql = type(h.q.store.index), type(h.q.store.index).sql

    def broken(self, query, *a, **k):
        if "source_dirty = 0" in query:
            raise OSError("catalog locked")
        return sql(self, query, *a, **k)

    def dies(*a):
        raise RuntimeError("git died")

    monkeypatch.setattr(index, "sql", broken)
    h.d._compiles = dies
    with pytest.raises(RuntimeError):
        h.tick()
    assert len(h.launched) == 1 and any("can't be read" in s for s in world.logs)
    del h.d._compiles
    assert h.tick().startswith("landed #1") and len(h.launched) == 1


def test_review_lists_the_members_lacking_a_land_verdict_and_records_one_refusing_the_owner(world, monkeypatch, capsys):
    world.with_store_remote()
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    h.d.review = False
    h.q.sync()
    labels = StoreLabels(world.store, log=lambda s: None)
    assert Q.review(h.q, world.store, labels, 1, "land", "note:ci/x", "lane-9") == 0
    with pytest.raises(QueueError, match="#2 is lane-2's own PR"):
        Q.review(h.q, world.store, labels, 2, "land", None, "lane-2")
    capsys.readouterr()
    assert Q.review(h.q, world.store, StoreLabels(world.store, log=lambda s: None), None, None, None, None, as_json=True) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert (row["pr"], row["owner"], row["lean"], row["files"], row["why"]) == (2, "lane-2", True, ["pkg/lean-audit.json"],
                                                                               "no land verdict")
    (lab,) = world.store.labels(label_target(1, world.gh.prs[1].head), key=vocab.REVIEWED)
    assert (lab.value, lab.by, lab.ref) == ("land", "lane-9", "note:ci/x")


def test_review_without_sha_refuses_a_head_pushed_since_the_sync_and_takes_a_short_sha(world, monkeypatch):
    world.with_store_remote()
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.d.review = False
    h.q.sync()
    old = world.gh.prs[1].head
    world.gh.put(1, new := world.change({"a.txt": "2\n"}))
    labels = StoreLabels(world.store, log=lambda s: None)
    with pytest.raises(QueueError, match=f"GitHub has its head at {new[:12]}.*pass --sha"):
        Q.review(h.q, world.store, labels, 1, "land", None, "lane-9")
    assert Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha=new[:10]) == 0
    assert Q.review(h.q, world.store, labels, 1, "hold", "old", "lane-9", sha=old[:10]) == 0
    assert [lab.value for lab in world.store.labels(label_target(1, new), key=vocab.REVIEWED)] == ["land"]
    assert [lab.value for lab in world.store.labels(label_target(1, old), key=vocab.REVIEWED)] == ["hold"]
    with pytest.raises(QueueError, match="full 40-character SHA"):
        Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha="0badc0de")
    with pytest.raises(QueueError, match="neither #1's head"):                     # a mistyped full SHA
        Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha=new[:12] + "0" * 28)
    assert Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha=new) == 0


def test_the_owner_of_an_unmarked_pr_a_marked_one_waits_for_is_told_once_per_pair_of_heads(world, monkeypatch):
    """A marked PR can wait on another's unmarked commits, and that one's owner is told.  Its owner hears once per
    (head of the waiting PR, head of the unmarked one), and nothing once it's admitted or when the waiting PR isn't marked."""
    _registry(world, "lane-5", "lane-7", "lane-8", "lane-9")
    h = Review(world, monkeypatch)
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old, ready=False)
    h.tick()                                                                    # drive sees #5 at `old`
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old), ready=False)  # moved on from `old`, unmarked
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))             # branched from #5 at `old`
    world.gh.put(8, world.change({"y.txt": "y\n"}, base=old), ready=False)      # waits too, but isn't marked: no one hears
    world.gh.put(9, world.change({"z.txt": "z\n"}, base=world.gh.prs[5].head))  # on #5's current head
    world.store.add_label(label_target(7, world.gh.prs[7].head), vocab.READY, "true", "marker7")

    def told() -> list[str]:
        return [p for p in h.posts if "waits for your mark" in p]

    h.tick()
    five = world.gh.prs[5].head
    assert told() == [
        f"@ci @lane-5 queue: #7 (marked by marker7) waits for your mark of #5, which it contains (at {old[:12]}): mark #5 ready at "
        f"{five[:8]}, or withdraw it (`research queue eject 5 --reason ...`) and #7 carries it",
        f"@ci @lane-5 queue: #9 (marked by lane-9) waits for your mark of #5, which it contains (at {five[:12]}): mark #5 ready at "
        f"{five[:8]}, or withdraw it (`research queue eject 5 --reason ...`) and #9 carries it"], told()
    h.tick()
    assert len(told()) == 2                                     # once per pair of heads
    world.gh.put(7, world.change({"k.txt": "kernels 2\n"}, base=world.gh.prs[7].head))
    h.tick()
    assert len(told()) == 3 and told()[-1].startswith("@ci @lane-5 queue: #7 (marked by lane-7) waits for your mark of #5")
    world.gh.put(5, world.change({"x.txt": "x = 3\n"}, base=five), ready=False)
    h.tick()
    # #9 holds #5's older head now; it only shares `old` with #8, which waits for nothing of #9's or #7's (`Queue.entries`)
    assert [p.split(": mark")[0] for p in told()[3:]] == [
        f"@ci @lane-5 queue: #7 (marked by lane-7) waits for your mark of #5, which it contains (at {old[:12]})",
        f"@ci @lane-5 queue: #9 (marked by lane-9) waits for your mark of #5, which it contains (at {five[:12]})"], told()
    world.gh.ready.add((5, world.gh.prs[5].head))
    h.tick()
    assert len(told()) == 5 and h.q.status()[7].state != "waiting"


def _detached(h: Review, tmp_path, lean: bool = True, post: bool = False) -> list[str]:
    """``h``'s Lean checks (``lean``) and post-landing runs (``post``) launched as train.sh is: detached, the launch returning at
    once with no run ids (`drive.detach`'s record), each launch's commit in the list returned.  `h.printed[launch["dir"]]` is
    what its train.sh has printed (`drive.launched`): unset while it runs, its run ids, or [] once it exited without them."""
    starts, h.printed = [], {}

    def start(repo, commit, machine, *a, **kw):
        starts.append(commit)
        return {"dir": str(tmp_path / f"launch-{len(starts)}"), "pid": 0}

    if lean:
        h.d.launch_lean = start
    if post:
        h.d.launch_postland = start
    h.d.launched = lambda launch: h.printed.get(launch["dir"], (None, "train: abc on m1; log x"))
    return starts


def test_a_lean_check_launches_detached_and_a_later_tick_reads_its_run_ids(world, monkeypatch, tmp_path):
    """A tick that waited on each Lean check's train.sh for its run ids would take minutes per launch.  The launch returns at once; a later tick reads the ids."""
    h = Review(world, monkeypatch)
    starts = _detached(h, tmp_path)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    assert not h.tick().startswith("landed")
    top = _tip(world)
    rec = h.d.q.load_state()["drive"]["lean"][top]
    assert starts == [top] and rec["ids"] == [] and rec["launch"]["dir"] == str(tmp_path / "launch-1") and rec["state"] is None
    out = h.tick()
    assert not out.startswith("landed") and starts == [top], out          # still launching: it waits, and nothing launches
    assert any("waits for the Lean check (launching)" in s for s in world.logs)
    h.printed[rec["launch"]["dir"]] = (["r-lean"], "RUN r-lean")
    h.run_state["r-lean"] = ("running", None, None)
    h.clock += 600
    assert not h.tick().startswith("landed")
    rec = h.d.q.load_state()["drive"]["lean"][top]
    assert rec["ids"] == ["r-lean"] and "launch" not in rec and rec["at"] == h.clock and rec["state"] is None
    h.run_state["r-lean"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-lean", {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    assert starts == [top]


def test_a_detached_lean_launch_that_ends_without_run_ids_registered_no_run_and_launches_again(world, monkeypatch, tmp_path):
    from research.drive import LAUNCH_LOST_S

    h = Review(world, monkeypatch)
    starts = _detached(h, tmp_path)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    h.tick()
    top = _tip(world)
    h.printed[str(tmp_path / "launch-1")] = ([], "train: no run launched in the wait; stopped its launcher")
    assert not h.tick().startswith("landed")
    (post,) = [p for p in h.posts if "registered no run" in p]
    assert f"train.sh {top[:12]} m1 --deferred {LEAN_STEP} registered no run: train: no run launched" in post, post
    assert f"{tmp_path / 'launch-1'}/log" in post
    assert starts == [top, top]                                           # dropped, and launched again in the same tick
    rec = h.d.q.load_state()["drive"]["lean"][top]
    assert rec["launch"]["dir"] == str(tmp_path / "launch-2") and rec["state"] is None
    h.clock += LAUNCH_LOST_S                                              # within the bound it still waits
    h.tick()
    assert starts == [top, top]
    h.clock += 1                                                          # printing nothing past it, it registered no run
    h.tick()
    assert starts == [top, top, top] and len([p for p in h.posts if "registered no run" in p]) == 1   # told once per commit
    rec = {"ids": [], "launch": {"dir": "d", "pid": 0}, "at": h.clock - LAUNCH_LOST_S - 1}
    assert h.d._resolve(rec, "x").startswith("no run ids 60 min after its launch") and "launch" not in rec


def test_a_launching_lean_check_holds_next_on_its_base_and_an_ejection_waits_for_it(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch)
    starts = _detached(h, tmp_path)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    h.tick()
    top = _tip(world)
    _move_main(world, {"b.txt": "moved\n"})
    out = h.tick()
    assert "stays on" in out and _tip(world) == top and starts == [top], out
    assert Q.lean_running(h.d.q.load_state(), 1) == "(launching)"


def test_launching_lean_checks_count_against_max_lean_checks(world, monkeypatch, tmp_path):
    """As `test_a_lean_check_is_stacked_at_nexts_top_for_a_lean_change_above_a_running_one`, with every launch still waiting
    for its run ids: one still launching is a running check for the stack and for the cap."""
    h = Review(world, monkeypatch)
    starts = _detached(h, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.review(2)
    h.tick()
    two, three = (m.commit for m in h.q.chain())
    assert starts == [two], world.logs
    h.tick()
    assert starts == [two, three], world.logs                             # stacked above the one still launching
    assert h.d.q.load_state()["drive"]["lean"][three]["stacked"] is True
    world.gh.put(4, world.change({"pkg/U.lean": "theorem U : True := trivial\n"}))
    h.tick()
    h.tick()
    assert starts == [two, three] and h.cancelled == [], world.logs       # two launching: max_lean_checks
    h.d.max_lean_checks = 3
    h.tick()
    assert len(starts) == 3 and starts[2] == _tip(world), world.logs


def test_the_after_landing_run_launches_detached_and_one_that_registered_no_run_is_told_and_launched_again(
        world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, baseline=False)
    starts = _detached(h, tmp_path, lean=False, post=True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    main = _main(world)
    assert h.tick().startswith("nothing lands before an after-landing run")
    run = h.postland()["run"]
    assert starts == [main] and run["ids"] == [] and run["baseline"] is True
    out = h.tick()
    assert out.startswith("nothing lands before the post-landing run (launching) of `main`") and starts == [main], out
    h.printed[run["launch"]["dir"]] = ([], "train: research run exited 1 before launching a run")
    h.tick()
    (post,) = [p for p in h.posts if "registered no run" in p]
    assert f"train.sh {main[:12]} m1 --deferred" in post and "exited 1 before launching a run" in post, post
    assert starts == [main, main]
    run = h.postland()["run"]
    h.printed[run["launch"]["dir"]] = (["r-post"], "RUN r-post")
    h.run_state["r-post"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post", {s: "passed" for s in AFTER_LANDING})]
    assert h.tick().startswith("landed #1"), world.logs
    assert h.postland()["verified"] == main


def test_detach_returns_before_train_sh_prints_its_run_ids_and_launched_reads_them_or_an_exit_without_them(tmp_path, monkeypatch):
    """`detach`'s job, with a stand-in for train.sh that prints its `RUN` line only once the test lets it."""
    from research.drive import LAUNCHES, detach, launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    gate, rid = tmp_path / "gate", "r20261026-093000-abcd"
    os.mkfifo(gate)
    job = detach(tmp_path, ["sh", "-c", f'echo "train: abc on m1"; cat "$1" > /dev/null; echo "RUN {rid}"', "sh", str(gate)],
                 dict(os.environ))
    assert job["dir"].startswith(str(tmp_path / LAUNCHES)) and launched(job)[0] is None
    with open(gate, "w"):
        pass
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    assert launched(job) == ([rid], f"train: abc on m1\nRUN {rid}")
    job = detach(tmp_path, ["sh", "-c", "echo 'train: no run launched' >&2; exit 1"], dict(os.environ))
    with contextlib.suppress(ChildProcessError):
        os.waitpid(job["pid"], 0)
    assert launched(job) == ([], "train: no run launched") and (Path(job["dir"]) / "rc").read_text() == "1\n"
    gone = subprocess.Popen(["true"])
    gone.wait()
    (Path(job["dir"]) / "rc").unlink()
    assert launched({**job, "pid": gone.pid})[0] == []


def test_a_detached_train_launch_runs_its_own_copy_of_train_sh(tmp_path, monkeypatch):
    """Bash reads a script as it runs, and a later tick resets and checks out the queue's clone, so
    a train.sh that `main` changed meanwhile would change under a running launch.  It runs a copy in its launch directory, from
    the clone, and a rewrite of the clone's train.sh doesn't reach it."""
    from research.drive import LAUNCHES, launched, train_launch

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    repo, gate, rid = tmp_path / "clone", tmp_path / "gate", "r20261026-093000-abcd"
    script = repo / "tools" / "verity" / "check" / "train.sh"
    script.parent.mkdir(parents=True)
    script.write_text(f'echo "train: $0 in $PWD"\ncat {gate} > /dev/null\necho "RUN {rid}"\n')
    os.mkfifo(gate)
    job = train_launch(repo, "c0ffee", "m1", False, deferred=LEAN_STEP, detached=True)
    copy = Path(job["dir"]) / "train.sh"
    assert job["dir"].startswith(str(tmp_path / LAUNCHES)) and copy.read_text() == script.read_text()
    with open(script, "w") as f:                                           # in place, as a checkout of a changed train.sh
        f.write('echo "RUN r20261026-093000-ffff"\n' * 8)
    with open(gate, "w"):
        pass
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    ids, tail = launched(job)
    assert ids == [rid] and f"train: {copy} in {repo}" in tail, tail


def test_a_hand_lean_check_through_drive_holds_next_from_its_launch_before_train_sh_registers_a_run(world, monkeypatch, tmp_path):
    """Train.sh registers a hand Lean check's run minutes after its launch, and a tick in that window could reorder `next` and
    throw the check away.  `research queue lean-check` records it in drive's state as it launches."""
    h = Review(world, monkeypatch)
    starts = _detached(h, tmp_path)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    assert not h.tick().startswith("landed")
    one, two = (m.commit for m in h.q.chain()[:2])
    rec = h.d.hand_lean(two[:12], "ci")
    assert starts == [two] and h.hand == {}                               # launching: the run registry knows nothing yet
    saved = h.d.q.load_state()["drive"]["lean"][two]
    assert saved == rec and rec["state"] is None and rec["ids"] == [] and rec["launch"]["dir"] == str(tmp_path / "launch-1")
    assert (rec["hand"], rec["by"], rec["members"]) == (True, "ci", [[m.pr, m.head] for m in h.q.chain()[:2]])
    assert Q.lean_running(h.d.q.load_state(), 2) == "(launching)"
    h.q.first = (4,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 4, 3], world.logs        # `--first` goes above the hand check
    assert [m.commit for m in h.q.chain()[:2]] == [one, two] and starts == [two]


def test_lean_check_refuses_a_commit_that_is_not_a_member_of_next(world, monkeypatch):
    from research.merge import DRIVE_ENV

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    assert not h.tick().startswith("landed")
    argv = ["--workdir", str(h.q.workdir), "--remote", str(world.origin), "--repo", "o/r", "--store", str(world.tmp / "store"),
            "--rules", str(world.dev / "ci" / "queue.toml"), "--on", "m1", "--by", "ci"]
    monkeypatch.delenv(DRIVE_ENV, raising=False)
    assert Q.main(["lean-check", _tip(world), *argv]) == 1                # drive's state is the control pod's
    monkeypatch.setenv(DRIVE_ENV, "1")
    for c in (_main(world), world.gh.prs[1].head, "nonsense"):            # `main`, a PR's head, no commit
        assert Q.main(["lean-check", c, *argv]) == 1
        with pytest.raises(QueueError, match="is not a commit of `next`"):
            h.d.hand_lean(c, "ci")
    assert h.launched == [] and not h.d.q.load_state()["drive"].get("lean")


def test_lean_check_refuses_a_commit_a_running_lean_check_at_or_above_it_holds(world, monkeypatch):
    h = Review(world, monkeypatch, nonlean_first=False)       # a member that changes no Lean above a Lean change
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #1"), world.logs
    two, top = (m.commit for m in h.q.chain())
    assert h.launched == [(top, "m1", False)]                             # drive's own, of the prefix's top
    for c in (two, top):
        with pytest.raises(QueueError, match=f"running Lean check {h.lean_ids()[0]} of {top[:12]} \\(#3\\) already holds"):
            h.d.hand_lean(c, "ci")
    assert len(h.launched) == 1 and "hand" not in h.d.q.load_state()["drive"]["lean"][top]


def test_a_hand_lean_check_is_followed_and_covers_as_drives_own(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    assert not h.tick().startswith("landed") and h.launched == []
    top = _tip(world)
    h.d.hand_lean(top, "ci")
    assert h.launched == [(top, "m1", False)]
    (rid,) = h.lean_ids()
    h.review(1)
    assert not h.tick().startswith("landed") and len(h.launched) == 1     # drive waits for it, and launches none of its own
    assert any(f"waits for the Lean check {rid} of {top[:12]}" in s for s in world.logs), world.logs
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {"lean-build": "passed", LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    assert _main(world) == top and len(h.launched) == 1
    assert any(f"the Lean check {rid} of {top[:12]} passed" in s for s in world.logs), world.logs


def _passed_hand_check(world, monkeypatch) -> tuple[Review, str, str]:
    """A Review whose hand Lean check of `next`'s one member (a lock change, unreviewed, so it stays on `next`) passed: (it, the
    commit, the run that passed)."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.tick()
    top = _tip(world)
    h.d.hand_lean(top, "ci")
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    h.tick()
    assert h.d.q.load_state()["drive"]["lean"][top]["state"] == "passed" and _tip(world) == top, world.logs
    return h, top, rid


def test_a_passed_lean_check_of_a_reviewed_prefix_lands_it_though_a_new_head_below_conflicts_and_its_own_commit_failed(
        world, monkeypatch):
    """A hand check passes over the reviewed PRs, and in the same tick a new head below it rebuilds `next` and conflicts: the
    pass would leave `next` with its commit, and nothing land.  Below it, a member's own commit failed its Lean check, which
    must not hold everything above it when the pass checked the tree that lands."""
    _registry(world, "lane-1")
    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}", "x.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    own, _, top = (m.commit for m in h.q.chain())
    assert h.launched == [(top, "m1", False)]
    h.run_state[h.lean_ids()[0]] = ("done", 1, None)
    h.tick()
    assert h.launched[1] == (own, "m1", False), h.launched
    h.run_state[h.lean_ids()[1]] = ("done", 1, None)
    h.tick()
    assert [p for p in h.posts if "of its own commit" in p and "@lane-1" in p], h.posts
    h.d.hand_lean(top, "ci")
    (rid,) = h.lean_ids()[2:]
    world.gh.put(2, world.change({"b.txt": "2\n", "x.txt": "2\n"}))       # a new head that conflicts with #1, marked ready
    old = h.q.chain()
    assert not h.tick().startswith("landed") and h.q.chain() == old, world.logs
    h.clock += 600
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    out = h.tick()
    assert out.startswith("landed #3"), (out, world.logs)
    assert _main(world) == top and len(h.launched) == 3
    assert any(f"the Lean check {rid} of {top[:12]} passed" in s for s in world.logs), world.logs


def test_a_passed_lean_check_holds_next_until_it_lands_and_an_ejection_below_it_rebuilds_next(world, monkeypatch):
    h = Review(world, monkeypatch, baseline=False, nonlean_first=False)    # nothing lands before `main`'s own run
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    one, two, top = (m.commit for m in h.q.chain())
    admitted = world.gh.prs[2].head
    h.d.hand_lean(top, "ci")
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("nothing lands before")
    h.q.first = (3,)
    world.gh.put(2, world.change({"b.txt": "two\n"}))                      # its owner pushes, and marks the new head
    h.tick()
    assert [m.commit for m in h.q.chain()] == [one, two, top], world.logs  # the pass keeps them, `--first` too
    world.gh.ready.discard((2, world.gh.prs[2].head))
    world.gh.ready.discard((2, admitted))                                  # #2 ejected, both heads
    h.q.first = ()
    h.tick()
    chain = h.q.chain()
    assert [m.pr for m in chain] == [1, 3] and chain[0].commit == one and top not in {m.commit for m in chain}, world.logs


def test_an_eject_off_drive_leaves_next_under_a_running_lean_check_and_records_the_ejection(world, monkeypatch, capsys):
    """`research queue eject N` of a PR not on `next`, run off drive, would rebuild every member of `next` ("kept 0, merged
    N") and force-push it, dropping running hand Lean checks."""
    import research.drive as D
    from research.cli import main as research_main

    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(9, head := world.change({"c.txt": "9\n"}), ready=False)    # open, never on `next`
    h.review(1)
    h.tick()
    one = h.q.chain()[0].commit
    assert h.launched == [(one, "m1", False)], world.logs
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    monkeypatch.setattr(D, "merge_thread_post", lambda *a, **k: True)
    monkeypatch.delenv("RESEARCH_DRIVE")
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    assert research_main(["queue", "eject", "9", "--reason", "x", "--by", "compute-accounting", "--workdir", str(h.q.workdir),
                          "--remote", str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "store")]) == 0
    assert refs() == before and "drive's next tick rebuilds `next` without it" in capsys.readouterr().out
    state = h.q.load_state()
    assert state["ejected"]["9"]["head"] == head and state["ejected"]["9"]["by"] == "compute-accounting"
    (mark,) = world.store.labels(label_target(9, head), key=vocab.READY)
    assert mark.value is False and mark.by == "compute-accounting"
    assert state["drive"]["lean"][one]["state"] is None
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.q.sync()                                                          # nor any other caller
    assert refs() == before
    monkeypatch.setenv("RESEARCH_DRIVE", "1")
    h.tick()
    assert h.q.chain()[0].commit == one and h.cancelled == [], world.logs


def test_a_hand_sync_on_drive_keeps_next_up_to_drives_running_and_passed_lean_checks(world, monkeypatch):
    h = Review(world, monkeypatch, nonlean_first=False)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    chain = h.q.chain()
    assert [m.pr for m in chain] == [1, 2, 3]
    h.d.hand_lean(chain[1].commit, "ci")
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    h.tick()
    assert h.d.q.load_state()["drive"]["lean"][chain[1].commit]["state"] == "passed"
    world.gh.put(1, world.change({"a.txt": "one\n"}))                       # a new head at the bottom, marked
    hand = world.queue(nonlean_first=False)
    assert hand.hand_sync() is None and hand.hold == 2
    assert [m.commit for m in hand.chain()[:2]] == [m.commit for m in chain[:2]], world.logs
    world.queue(nonlean_first=False).sync()                                 # without drive's hold it rebuilds from the bottom
    assert [m.pr for m in h.q.chain()] == [2, 3, 1], world.logs


def test_a_hand_sync_waits_for_a_running_lean_check_whose_prefix_holds_an_ejected_member(world, monkeypatch):
    """Drive's tick waits for such a check to end before the rebuild takes its commit off `next`; a hand sync waits
    too, or it would rebuild from the ejected member and throw the check away."""
    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    top = _tip(world)
    assert h.launched == [(top, "m1", False)], world.logs
    world.gh.ready.discard((1, world.gh.prs[1].head))                      # #1 ejected below the running check
    why = world.queue(nonlean_first=False).hand_sync()
    assert why and f"waits for the Lean check of {top[:12]}" in why and _tip(world) == top


def test_research_queue_drive_refuses_off_drive_and_pushes_nothing(world, monkeypatch, capsys):
    """A tick lands onto `main` and rebuilds `next`, which only drive's App may move."""
    import research.merge as M
    from research.cli import main as research_main

    calls = []

    def gh(repo, *args):
        calls.append(args)
        assert args[:2] == ("pr", "list"), args
        return "[]"

    monkeypatch.setattr(M, "_gh_cli", gh)                                   # the branch sweep's, which runs before the sync
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    monkeypatch.delenv("RESEARCH_DRIVE")
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    assert research_main(["queue", "drive", "--on", "m1", "--workdir", str(world.tmp / "q" / "verity"), "--remote",
                          str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "store"), "--rules",
                          str(world.dev / "ci" / "queue.toml")]) == 1
    assert "/data/infra/drive.sh once" in capsys.readouterr().err and refs() == before and calls == []


def test_a_member_that_doesnt_compile_under_a_passed_lean_check_lets_its_pushed_fix_in(world, monkeypatch):
    """A passed Lean check must not hold #2's old head on `next` (`_landable`) after `_compiles` cut below it and `_eject` left
    it for its owner's fix, or drive lands #1 and then logs "nothing lands: #2 ... waits" every tick."""
    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(2, broken := world.change({"pkg/bad.py": "def (:\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    top = _tip(world)
    assert h.launched == [(top, "m1", False)], world.logs
    world.gh.put(2, world.change({"pkg/bad.py": "x = 1\n"}, base=broken))   # its owner's fix, on top, marked
    h.review(2)
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    assert any("not ejecting #2" in s for s in world.logs), world.logs
    out = h.tick()
    assert out.startswith("landed #3"), (out, world.logs)
    assert _git(world.origin, "show", "main:pkg/bad.py") == "x = 1"


def test_a_cut_below_a_lean_change_under_a_passed_lean_check_waits_for_a_lean_check_of_what_lands(world, monkeypatch):
    """#1 and #2 both change pkg's lock, and #2 doesn't compile; a Lean check of #1 + #2 + #3 passed and `_compiles` cuts
    below #2: #1 alone, a tree no Lean check saw, doesn't land on that check."""
    h = Review(world, monkeypatch, nonlean_first=False)
    h.d.max_lean_checks = 1                                                # one check holds both Lean changes
    world.gh.put(1, one := world.change({"pkg/lean-audit.json": '{"a": 1}'}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": '{"a": 1, "b": 2}', "pkg/bad.py": "def (:\n"}, base=one))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    top, below = _tip(world), _main(world)
    assert h.launched == [(top, "m1", False)], world.logs
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    out = h.tick()
    assert not out.startswith("landed") and _main(world) == below, (out, world.logs)
    own = h.q.chain()[0].commit
    assert h.launched[1] == (own, "m1", False) and top not in h.d.q.load_state()["drive"]["lean"], world.logs
    (again,) = h.lean_ids()[1:]
    h.run_state[again] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, again, {LEAN_STEP: "passed"})]
    out = h.tick()
    assert out.startswith("landed #"), (out, world.logs)
    assert _git(world.origin, "show", "main:pkg/lean-audit.json") == '{"a": 1}'
    assert f"research queue drive: landed #1 #3 on `main` at {_main(world)[:12]} (check {again})" in world.logs, world.logs


def test_a_failed_lean_check_below_a_passed_one_lands_nothing_on_a_check_of_another_lean(world, monkeypatch):
    """#1 and #3 change pkg's lock, #4 is unreviewed; the check of #1 + #2 (#1's Lean) failed and the one of #1..#4
    passed: #1 alone doesn't land on the latter."""
    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, one := world.change({"pkg/lean-audit.json": '{"a": 1}'}))
    world.gh.put(2, world.change({"a.txt": "2\n"}, base=one))
    world.gh.put(3, world.change({"pkg/lean-audit.json": '{"a": 1, "b": 2}'}, base=one))
    world.gh.put(4, world.change({"b.txt": "4\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    chain = h.q.chain()
    assert [m.pr for m in chain] == [1, 2, 3, 4]
    assert [c for c, *_ in h.launched] == [chain[1].commit, chain[3].commit], world.logs
    low, high = h.lean_ids()
    h.run_state[low] = ("done", 1, None)
    h.run_state[high] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, low, {LEAN_STEP: "failed"}), _steps_run(world.tmp, high, {LEAN_STEP: "passed"})]
    below = _main(world)
    out = h.tick()
    assert not out.startswith("landed") and _main(world) == below, (out, world.logs)
    assert h.launched[2] == (chain[0].commit, "m1", False), world.logs                     # #1's own commit, blamed on its own
    assert not any("each Lean change below it has a passed Lean check" in x for x in world.logs), world.logs
    (own,) = h.lean_ids()[2:]
    h.run_state[own] = ("done", 0, None)
    h.evidence = [*h.evidence, _steps_run(world.tmp, own, {LEAN_STEP: "passed"})]
    out = h.tick()
    assert out.startswith("landed #"), (out, world.logs)
    assert _git(world.origin, "show", "main:pkg/lean-audit.json") == '{"a": 1, "b": 2}'
    assert f"research queue drive: landed #1 #2 #3 on `main` at {_main(world)[:12]} (check {high})" in world.logs, world.logs


def test_a_lean_change_below_a_failed_lean_check_waits_for_no_running_check_of_another_lean(world, monkeypatch):
    """The check of #1 + #2 failed while the one of #1..#4 runs; #3 changes the lock between them, so that one's pass
    can't cover #1, and #1 doesn't wait for it before its own check launches."""
    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, one := world.change({"pkg/lean-audit.json": '{"a": 1}'}))
    world.gh.put(2, world.change({"a.txt": "2\n"}, base=one))
    world.gh.put(3, world.change({"pkg/lean-audit.json": '{"a": 1, "b": 2}'}, base=one))
    world.gh.put(4, world.change({"b.txt": "4\n"}))
    for n in (1, 2, 3):
        h.review(n)
    h.tick()
    chain = h.q.chain()
    low, high = h.lean_ids()
    h.run_state[low] = ("done", 1, None)
    h.evidence = [_steps_run(world.tmp, low, {LEAN_STEP: "failed"})]
    out = h.tick()
    assert not out.startswith("landed"), (out, world.logs)
    assert h.launched[2] == (chain[0].commit, "m1", False), world.logs
    assert h.q.load_state()["drive"]["lean"][chain[3].commit]["state"] is None              # still running: nothing cancels it


def test_drives_own_pushes_of_main_refuse_off_drive_and_in_shadow_too(world, monkeypatch):
    """`_land` and `_revert` push `main` whoever calls them; off drive they refuse before anything, `--shadow` too,
    which never reaches them."""
    h = Review(world, monkeypatch, baseline=False)                         # nothing lands before `main`'s own run
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert not h.tick().startswith("landed")
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    monkeypatch.delenv("RESEARCH_DRIVE")
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.d.land_reviewed(h.q.load_state()["drive"], dry_run=False)
    _move_main(world, {"z.txt": "z\n"})
    h.q.fetch()
    before, moved = refs(), _main(world)
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.d._revert([moved])
    h.q.shadow = True
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.d._revert([moved])
    assert refs() == before
    h.q.shadow = False
    monkeypatch.setenv("RESEARCH_DRIVE", "1")
    assert h.d._revert([moved]) == "pushed" and _git(world.origin, "rev-parse", "main^1") == moved


def test_lean_check_refuses_a_commit_whose_recorded_lean_check_passed_naming_its_run(world, monkeypatch):
    """A hand check at a commit drive records as passed would replace the pass with a pending check, so what it covered
    would wait again."""
    from research.merge import DRIVE_ENV

    h, top, rid = _passed_hand_check(world, monkeypatch)
    with pytest.raises(QueueError, match=f"of {top[:12]} \\(#1\\) already passed: run {rid} \\(run ids {rid}\\); a deliberate "
                                         "rerun takes --again"):
        h.d.hand_lean(top, "ci")
    monkeypatch.setenv(DRIVE_ENV, "1")
    argv = ["lean-check", top, "--workdir", str(h.q.workdir), "--remote", str(world.origin), "--repo", "o/r", "--store",
            str(world.tmp / "store"), "--rules", str(world.dev / "ci" / "queue.toml"), "--on", "m1", "--by", "ci"]
    assert Q.main(argv) == 1
    rec = h.d.q.load_state()["drive"]["lean"][top]
    assert len(h.launched) == 1 and (rec["state"], rec["run"]) == ("passed", rid)


def test_lean_check_again_replaces_a_passed_record_with_a_pending_hand_check(world, monkeypatch):
    h, top, rid = _passed_hand_check(world, monkeypatch)
    rec = h.d.hand_lean(top, "ci", again=True)
    assert h.launched == [(top, "m1", False), (top, "m1", False)]
    (again,) = [r for r in h.lean_ids() if r != rid]
    assert h.d.q.load_state()["drive"]["lean"][top] == rec and "run" not in rec
    assert (rec["ids"], rec["state"], rec["hand"], rec["by"]) == ([again], None, True, "ci")
