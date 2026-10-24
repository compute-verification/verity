"""The review landing (`land = "review"`; Daniel, 24 Oct 06:05Z: PRs land on an eyeball review, and the full check runs after
landing).  A tick lands `next`'s longest prefix whose every member has a land verdict at its head by a reviewer who isn't its owner,
a Lean change only once a Lean check passed, and launches no train check; the after-landing run checks `main`, and its failure
reverts what it implicates once `main` passed that step before, bisecting when several PRs are implicated."""
from __future__ import annotations

import json

import pytest

import research.queue as Q
from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL, LEAN_STEP
from research.queue import QueueError, StoreLabels, label_target, lean_change
from research.store import vocab
from test_drive_postland import _registry
from test_queue import World, _git
from test_queue_drive import Harness, _failed_run, _steps_run, _tip

EVERY = {s: "passed" for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Review(Harness):
    """A drive landing on reviews, whose verdicts are `reviewed` labels in the world's store, and whose after-landing launches are
    recorded (`posted`) as `r-post-N`; the Harness's `launched` holds its Lean checks.  Unless ``baseline`` is off, ci has
    recorded `main` green (`research queue baseline`), so a tick lands without first running the after-landing run on `main`."""

    def __init__(self, world, monkeypatch, baseline: bool = True, **kw):
        super().__init__(world, monkeypatch, **kw)
        self.d.review, self.posted = True, []
        self.d.reviews = lambda pr, head: world.store.labels(label_target(pr, head), key=vocab.REVIEWED)
        self.d.ready_by = lambda pr, head: next((lab.by for lab in sorted(world.store.labels(label_target(pr, head), key=vocab.READY),
                                                                          key=lambda lab: lab.ts, reverse=True)), None)
        self.d.launch_postland = self.launch_post
        if baseline:
            assert Q.baseline(self.q, _git(world.origin, "rev-parse", "main"), "ci") == 0

    def launch_post(self, repo, commit, machine, steps, **kw):
        rid = f"r-post-{len(self.posted)}"
        self.posted.append((commit, steps, kw))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

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


def test_a_reviewed_prefix_lands_with_no_train_check_and_awaits_the_after_landing_run_of_every_step(world, monkeypatch):
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
    assert h.launched == []                                     # no train check, and no Lean check: no Lean changes
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": False, "no_cache": True, "no_regenerate": True})]
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
    leave `main`'s lock stale and Daniel's DM unsent."""
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
    h = Review(world, monkeypatch)
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
    h = Review(world, monkeypatch)
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
    assert h.tick().startswith("landed #1"), world.logs
    assert _main(world) == own and _tip(world) == top
    (post,) = [p for p in h.posts if "changes no Lean" in p]
    assert post.startswith("@ci queue:") and "@lane-2" not in post, post


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


def test_nothing_lands_before_the_after_landing_run_of_main_itself_passes_or_ci_records_a_baseline(world, monkeypatch):
    """A first after-landing run fails on anything `main` already had: reverting the landings it follows would revert innocent
    PRs.  So the review landing's first run is of `main` itself, and nothing lands until it passes; its failure reverts nothing."""
    h = Review(world, monkeypatch, baseline=False)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    main = _main(world)
    out = h.tick()
    assert out.startswith("nothing lands before an after-landing run of `main`"), out
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": True, "no_cache": False, "no_regenerate": True})]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {"pytest": "failed"})]
    out = h.tick()
    assert "failed its baseline" in out and _main(world) == main, out
    (post,) = [p for p in h.posts if "Reverted nothing" in p]
    assert "never passed on `main`" in post and "research queue baseline SHA --by ci" in post, post
    with pytest.raises(QueueError, match="only ci"):
        Q.baseline(h.q, main, "lane-3")
    assert Q.main(["baseline", main, "--by", "ci", "--workdir", str(h.q.workdir), "--remote", str(world.origin), "--repo", "o/r",
                   "--store", str(world.tmp / "store"), "--rules", str(world.dev / "ci" / "queue.toml")]) == 0
    assert h.tick().startswith("landed #1")


def test_a_baseline_waits_for_check_modes_last_post_landing_run_and_then_checks_main_itself(world, monkeypatch):
    h = Review(world, monkeypatch, baseline=False)
    main = _main(world)
    h.q._clone()
    state = h.q.load_state()
    state["drive"] = {"launched": {}, "told": [], "postland": {"verified": main, "pending": [main], "run": {
        "ids": ["r-old"], "tip": main, "at": 0}, "steps": ["circuit-check"]}}
    h.q.save_state(state)
    h.run_state["r-old"] = ("running", None, None)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("nothing lands before the post-landing run r-old") and h.posted == []
    assert h.postland()["steps"] == ["circuit-check"]
    h.run_state["r-old"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-old", {"circuit-check": "passed"})]
    assert h.tick().startswith("nothing lands before an after-landing run")
    assert [c for c, *_ in h.posted] == [main]
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING})]
    assert h.tick().startswith("landed #1"), world.logs


def test_an_after_landing_failure_reverts_the_pr_it_implicates(world, monkeypatch, tmp_path):
    _registry(world, "lane-2")
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
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == tip                            # one revert on top of the landing
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`") and "#1 #3 stay on `main`" in post, post
    assert h.posted[-1][0] == main and h.launched == []                                  # the next run checks the new tip


def test_an_after_landing_failure_several_prs_touch_is_bisected_to_the_one_that_broke_it(world, monkeypatch, tmp_path):
    """Reverting every landing a failure's files reach took innocent PRs with the culprit: each step of the bisection checks one
    landing's commit with the failed step alone, and only the first landing that fails is reverted.  Nothing lands meanwhile."""
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
    assert out.startswith("nothing lands while drive bisects"), out
    assert h.posted[1] == (one, "lean-suites", {"agreement": False, "no_cache": False, "no_regenerate": True})
    h.run_state["r-post-1"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-1", {"lean-suites": "passed"})]
    h.tick()
    assert h.posted[2][0] == two and _main(world) == tip
    h.run_state["r-post-2"] = ("done", 1, None)
    h.tick()
    main = _main(world)
    revert = _git(world.origin, "rev-list", "--first-parent", "--reverse", f"{tip}..main").split()[0]
    assert _git(world.origin, "rev-parse", f"{revert}^") == tip and _subject(world, revert).startswith('Revert "Queue: #2')
    assert _git(world.origin, "show", f"{main}:pkg/tests/test_x.py") == "def test_y():\n    pass"
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`") and "passes on #1's landing" in post, post
    assert "#1 #3 stay on `main`" in post
    assert "bisect" not in h.postland() and h.posted[-1][0] == main


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
