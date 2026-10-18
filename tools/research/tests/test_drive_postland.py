"""CI-30, the post-landing run: with `--defer`, a train lands once every step but the deferred ones passed, and drive cancels the
rest of its check (18 Oct: several such tails held a check slot each for 1-4 h).  One run at a time checks the deferred steps on
`main`'s tip; a failure reverts only the PR it implicates (18 Oct: a failed tail reverted every merge from its base, 12-26
innocent PRs, which the coordinator prevented by hand seven times)."""
from __future__ import annotations

import json

import pytest

from research.drive import CHECK_STEPS
from test_queue import World, _git
from test_queue_drive import Harness, _check, _failed_run, _killed_run, _steps_run, _tip

DEFER = frozenset({"circuit-check", "flock-rust-tests"})


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Postland(Harness):
    """A drive with circuit-check deferred, whose post-landing launches are recorded (`posted`) as `r-post-N`."""

    def __init__(self, world, monkeypatch, tmp_path, **kw):
        super().__init__(world, monkeypatch, **kw)
        self.tmp, self.posted = tmp_path, []
        self.d.defer = DEFER
        self.d.launch_postland = self.launch_post

    def launch_post(self, repo, commit, machine, steps):
        rid = f"r-post-{len(self.posted)}"
        self.posted.append((commit, steps))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def land_deferred(self) -> tuple[str, str]:
        """Check `next`'s tip and land it on every step but the deferred ones: (its check's run id, the tip)."""
        before = set(self.run_state)
        self.tick()
        (rid,) = [r for r in self.run_state if r not in before and not r.startswith("r-post-")]
        tip = _tip(self.world)
        self.evidence = [_steps_run(self.tmp, rid, {x: "passed" for x in CHECK_STEPS if x not in DEFER})]
        assert self.tick().startswith("landed"), self.world.logs
        assert _git(self.world.origin, "rev-parse", "main") == tip
        return rid, tip

    def postland(self) -> dict:
        return self.d.q.load_state()["drive"].get("postland")

    def passed(self, rid: str) -> None:
        """``rid``, a post-landing run, ended with every deferred step passed in its steps.jsonl."""
        self.run_state[rid] = ("done", 0, None)
        self.evidence = [_steps_run(self.tmp, rid, {s: "passed" for s in DEFER})]


def _registry(world: World, *handles: str) -> None:
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane"} for h in handles]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")


def test_a_deferred_landing_cancels_its_checks_tail_and_awaits_one_post_landing_run_of_main(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    rid, tip = h.land_deferred()
    assert h.cancelled == [rid]                                          # once: the tick's rebuild doesn't cancel it again
    pl = h.postland()
    assert pl["verified"] == base and pl["pending"] == [tip] and pl["steps"] == sorted(DEFER)
    assert pl["run"]["tip"] == tip and h.posted == [(tip, "circuit-check,flock-rust-tests")]
    assert not h.d.q.load_state()["drive"].get("deferred")              # nothing for the old rule to follow
    h.tick()
    assert len(h.posted) == 1 and h.cancelled == [rid]                  # its run is running: nothing more


def test_one_post_landing_run_at_a_time_and_a_pass_verifies_its_tip_then_the_next_checks_main(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, one = h.land_deferred()
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    rid2, two = h.land_deferred()
    assert h.posted == [(one, "circuit-check,flock-rust-tests")] and rid2 in h.cancelled   # the first run still runs
    assert h.postland()["pending"] == [one, two]
    h.passed("r-post-0")
    h.tick()
    pl = h.postland()
    assert pl["verified"] == one and pl["pending"] == [two] and h.posted[-1][0] == two == pl["run"]["tip"]
    h.passed("r-post-1")
    h.tick()
    h.tick()
    pl = h.postland()
    assert pl["verified"] == two and pl["pending"] == [] and pl["run"] is None and len(h.posted) == 2
    assert not h.posts


def test_a_failed_post_landing_run_reverts_only_the_pr_it_implicates_and_tells_its_owner(world, monkeypatch, tmp_path):
    _registry(world, "lane-2")
    h = Postland(world, monkeypatch, tmp_path)
    h.d.train_size = 3
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    _, tip = h.land_deferred()
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{main}^") == tip                            # one revert on top of the landing
    assert _git(world.origin, "show", f"{main}:a.txt") == "1" and _git(world.origin, "show", f"{main}:b.txt") == "3"
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    (post,) = h.posts
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`") and "#1 #2 #3" in post, post
    assert "#1 #3 stay on `main`" in post, post
    assert h.posted[-1][0] == main and len(h.posted) == 2                                # the next run checks the new tip
    h.passed("r-post-1")
    h.tick()
    assert h.postland()["verified"] == main and h.postland()["pending"] == []


def _tree(world: World, ref: str) -> str:
    return _git(world.origin, "rev-parse", f"{ref}^{{tree}}")


def test_a_post_landing_failure_no_pr_is_implicated_in_reverts_every_landing_since_verified(world, monkeypatch, tmp_path):
    """Daniel, 11 Oct: a failure there reverts the landing at once, so `main` doesn't stay red when no single PR is to blame.
    One that no PR touches reverts nothing (`test_drive_postland_redteam`)."""
    _registry(world, "lane-1", "lane-2")
    h = Postland(world, monkeypatch, tmp_path)
    h.d.train_size = 2
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    _, tip = h.land_deferred()
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = []                                                         # no failed suite's report reached drive
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _tree(world, main) == _tree(world, base) and _git(world.origin, "rev-list", "--count", f"{tip}..{main}") == "2"
    (post,) = h.posts
    assert post.startswith(f"@ci @lane-1 @lane-2 queue: Every landing since {base[:12]} reverted on `main` (#1 #2)"), post
    assert "No single PR is implicated" in post, post
    assert h.posted[-1][0] == main and len(h.posted) == 2                                # the next run checks the new tip


def test_reverting_every_landing_reverts_only_the_queues_merges_and_a_lander_commit_in_the_range_stays(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.land_deferred()
    moved = world.change({"c.txt": "a lander's pure move\n"})
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    h.run_state["r-post-0"] = ("done", 75, None)                                       # never ran: it runs on `main`'s tip now
    h.tick()
    assert h.posted[-1][0] == moved
    h.run_state["r-post-1"] = ("done", 1, None)
    h.evidence = []                                                                    # no failed suite's report reached drive
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{main}^") == moved                          # one revert, of #1's merge alone
    assert _git(world.origin, "show", f"{main}:a.txt") == _git(world.origin, "show", f"{base}:a.txt")
    assert _git(world.origin, "show", f"{main}:c.txt") == "a lander's pure move"
    assert any("Every landing since" in p and "(#1)" in p for p in h.posts), h.posts


def test_reverting_every_landing_reaches_a_train_landed_onto_a_moved_main(world, monkeypatch, tmp_path):
    """There the first-parent commit is drive's `Merge` onto the moved `main`, and the PR's `Queue: #` merge is its second parent."""
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    (rid,) = h.run_state
    other = world.change({"c.txt": "outside the queue\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    h.evidence = [_steps_run(tmp_path, rid, {x: "passed" for x in CHECK_STEPS if x not in DEFER})]
    assert h.tick().startswith("landed #1")
    landed = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{landed}^1") == other and h.posted[-1][0] == landed
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = []                                                                    # no failed suite's report reached drive
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _tree(world, main) == _tree(world, other) and _git(world.origin, "rev-parse", f"{main}^") == landed


def test_a_post_landing_failure_several_prs_are_implicated_in_reverts_every_landing_since_verified(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    h.d.train_size = 3
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"pkg/helpers.py": "X = 2\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    h.land_deferred()
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", 'File "/workspace/research/src/abc/pkg/helpers.py", line 1')]
    h.tick()
    assert _tree(world, "main") == _tree(world, base)
    (post,) = h.posts
    assert "Every landing since" in post and "which #1 #2 change" in post, post


def test_a_post_landing_run_the_machine_killed_runs_again_once_and_killed_again_is_told(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, tip = h.land_deferred()
    h.run_state["r-post-0"] = ("done", 137, None)                                      # an OOM kill, and no report
    h.tick()
    assert h.posted == [(tip, "circuit-check,flock-rust-tests")] * 2 and not h.posts
    assert h.postland()["run"]["resend"] is True and _git(world.origin, "rev-parse", "main") == tip
    h.run_state["r-post-1"] = ("done", 1, None)
    h.evidence = [_killed_run(tmp_path, "again")]                                      # circuit-check -9, naming no file
    h.tick()
    (post,) = h.posts
    assert post.startswith("@ci queue: the post-landing run r-post-1") and "killed again on its resend" in post, post
    h.tick()
    assert len(h.posted) == 2 and _git(world.origin, "rev-parse", "main") == tip


def test_a_post_landing_run_that_has_already_failed_is_acted_on_and_cancelled(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    h.d.train_size = 2
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.land_deferred()
    h.run_state["r-post-0"] = ("running", None, None, "suite pkg (tests/test_x.py::test_y)")
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert "r-post-0" in h.cancelled and any(p.startswith("@ci queue: #2 reverted on `main`") for p in h.posts), h.posts


def test_a_deferred_record_from_before_the_post_landing_run_still_finishes_under_the_old_rule(world, monkeypatch, tmp_path):
    """A live drive switches over mid-flight: a landing it recorded in `drive["deferred"]` is still followed to its end, and a
    deferred step that fails reverts that landing."""
    h = Harness(world, monkeypatch)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    tip = _tip(world)
    _check(world, monkeypatch, tip)
    assert h.tick().startswith("landed #1")
    state = h.q.load_state()
    state["drive"]["deferred"] = {tip: {"ids": ["r-old"], "base": base, "prs": "#1"}}
    h.q.save_state(state)
    h.run_state["r-old"] = ("running", None, None)
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == tip and "r-old" not in h.cancelled
    h.run_state["r-old"] = ("done", 1, None)
    h.evidence = [_steps_run(tmp_path, "r-old", {"circuit-check": "failed"})]
    h.tick()
    assert _git(world.origin, "diff", "--stat", base, "main") == "" and any("reverted on `main`" in x for x in h.posts), h.posts
    assert not h.q.load_state()["drive"]["deferred"]
