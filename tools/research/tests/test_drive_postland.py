"""CI-30, the post-landing run: with `--defer`, a train lands once every step but the deferred ones passed, and drive cancels the
rest of its check (18 Oct: several such tails held a check slot each for 1-4 h).  One run at a time checks the deferred steps on
`main`'s tip; a failure reverts only the PR it implicates (18 Oct: a failed tail reverted every merge from its base, 12-26
innocent PRs, which the coordinator prevented by hand seven times)."""
from __future__ import annotations

import json

import pytest

from research.drive import CHECK_STEPS
from test_queue import World, _git
from test_queue_drive import Harness, _check, _failed_run, _steps_run, _tip

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
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    pl = h.postland()
    assert pl["verified"] == one and pl["pending"] == [two] and h.posted[-1][0] == two == pl["run"]["tip"]
    h.run_state["r-post-1"] = ("done", 0, None)
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
    assert h.posted[-1][0] == main and len(h.posted) == 2                                # the next run checks the new tip
    h.run_state["r-post-1"] = ("done", 0, None)
    h.tick()
    assert h.postland()["verified"] == main and h.postland()["pending"] == []


def test_a_post_landing_failure_no_pr_is_implicated_in_reverts_nothing_and_tells_ci(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, tip = h.land_deferred()
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]     # in pkg, which #1 doesn't touch
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == tip
    (post,) = h.posts
    assert post.startswith("@ci queue: the post-landing run r-post-0 of `main`") and "Reverted nothing" in post, post
    h.tick()
    assert len(h.posted) == 1 and len(h.posts) == 1                                      # that tip isn't checked again
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    _, two = h.land_deferred()
    assert h.posted[-1][0] == two                                                        # a new landing is


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
