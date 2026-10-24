"""The review landing (`land = "review"`; Daniel, 24 Oct 06:05Z: PRs land on an eyeball review, and the full check runs after
landing).  A tick lands `next`'s longest prefix whose every member has a land verdict at its head by a reviewer who isn't its owner,
a lock change only once a Lean check passed, and launches no train check; the after-landing run checks `main`, and its failure
reverts what it implicates."""
from __future__ import annotations

import json

import pytest

import research.queue as Q
from research.drive import AFTER_LANDING, LEAN_STEP
from research.queue import QueueError, StoreLabels, label_target
from research.store import vocab
from test_drive_postland import _registry
from test_queue import World, _git
from test_queue_drive import Harness, _failed_run, _steps_run, _tip


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Review(Harness):
    """A drive landing on reviews, whose verdicts are `reviewed` labels in the world's store, and whose after-landing launches are
    recorded (`posted`) as `r-post-N`; the Harness's `launched` holds its Lean checks."""

    def __init__(self, world, monkeypatch, **kw):
        super().__init__(world, monkeypatch, **kw)
        self.d.review, self.posted = True, []
        self.d.reviews = lambda pr, head: world.store.labels(label_target(pr, head), key=vocab.REVIEWED)
        self.d.launch_postland = self.launch_post

    def launch_post(self, repo, commit, machine, steps, **kw):
        rid = f"r-post-{len(self.posted)}"
        self.posted.append((commit, steps, kw))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def review(self, pr: int, verdict: str = "land", by: str = "reviewer", ref: str | None = None) -> None:
        self.world.store.add_label(label_target(pr, self.world.gh.prs[pr].head), vocab.REVIEWED, verdict, by, ref=ref)


def _subject(world: World, ref: str) -> str:
    return _git(world.origin, "log", "-1", "--format=%s", ref)


def test_a_reviewed_prefix_lands_with_no_train_check_and_awaits_the_after_landing_run_of_every_step(world, monkeypatch):
    h = Review(world, monkeypatch)
    h.d.after_landing_no_cache = True
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    main = _git(world.origin, "rev-parse", "main")
    assert _subject(world, "main").startswith("Queue: #2") and _subject(world, "main^1").startswith("Queue: #1")
    assert _subject(world, "next").startswith("Queue: #3") and _git(world.origin, "rev-parse", "next^1") == main
    assert h.launched == []                                     # no train check, and no Lean check: no lock changes
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": False, "no_cache": True})]
    assert h.after and h.after[0][1] == {main: "", _git(world.origin, "rev-parse", "main^1"): ""}
    pl = h.d.q.load_state()["drive"]["postland"]
    assert pl["pending"] == [main] and pl["run"]["skip"] == ["lean-agreement", "lean-changed"]
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING if s not in ("lean-agreement", "lean-changed")})]
    h.tick()
    assert h.d.q.load_state()["drive"]["postland"]["verified"] == main


def test_an_unreviewed_or_held_member_stops_the_prefix_below_it(world, monkeypatch):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(3)
    assert h.tick().startswith("landed #1"), world.logs
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2)
    h.review(2, "hold", by="other", ref="reads a stale fixture")
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #2") and "held by other (reads a stale fixture)" in out, out
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2, by="other")                                     # each reviewer's newest verdict counts
    assert h.tick().startswith("landed #3")
    assert _git(world.origin, "rev-parse", "main") == _tip(world)
    assert h.launched == []


def test_the_owners_own_verdict_does_not_count(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1, by="lane-1")                                    # #1's `Lane lane-1`
    out = h.tick()
    assert "no land verdict but its owner's own (lane-1), which doesn't count" in out, out
    assert _subject(world, "main") == "init"
    h.review(1, by="@lane-7")
    assert h.tick().startswith("landed #1")


def test_a_lock_change_waits_for_a_passed_lean_check_of_a_next_commit_holding_it(world, monkeypatch):
    h = Review(world, monkeypatch)
    lock = json.dumps({"guarantees": {"T": "2"}, "reads": {}, "dependencies": {"M": "1"}})
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": lock}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #1"), world.logs
    top = _tip(world)
    assert h.launched == [(top, "m1", False)]                   # `train.sh TOP MACHINES --deferred lean-audit`, of the prefix's top
    (rid,) = [r for r in h.run_state if not r.startswith("r-post-")]
    assert _subject(world, "main").startswith("Queue: #1")
    assert not h.tick().startswith("landed") and len(h.launched) == 1   # it waits for the check it launched
    assert _subject(world, "main").startswith("Queue: #1")
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {"lean-build": "passed", LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #3"), world.logs
    assert _git(world.origin, "rev-parse", "main") == top and len(h.launched) == 1


def test_a_failed_lean_check_holds_its_lock_change_and_tells_its_owner(world, monkeypatch):
    _registry(world, "lane-2")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    for n in (1, 2):
        h.review(n)
    h.tick()
    (rid,) = [r for r in h.run_state if not r.startswith("r-post-")]
    h.run_state[rid] = ("done", 1, None)
    h.tick()
    (post,) = [p for p in h.posts if "changes a lock" in p]
    assert post.startswith("@ci @lane-2 queue: #2 at") and "failed" in post and "--deferred lean-audit" in post, post
    assert len(h.launched) == 1 and _subject(world, "main").startswith("Queue: #1")


def test_an_after_landing_failure_reverts_the_pr_it_implicates(world, monkeypatch, tmp_path):
    _registry(world, "lane-2")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _git(world.origin, "rev-parse", "main")
    assert [c for c, *_ in h.posted] == [tip]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{main}^") == tip                            # one revert on top of the landing
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`") and "#1 #3 stay on `main`" in post, post
    assert h.posted[-1][0] == main and h.launched == []                                  # the next run checks the new tip


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
    assert (row["pr"], row["owner"], row["lock"], row["files"], row["why"]) == (2, "lane-2", True, ["pkg/lean-audit.json"],
                                                                               "no land verdict")
    (lab,) = world.store.labels(label_target(1, world.gh.prs[1].head), key=vocab.REVIEWED)
    assert (lab.value, lab.by, lab.ref) == ("land", "lane-9", "note:ci/x")
