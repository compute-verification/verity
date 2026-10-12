"""`research queue drive`: one tick lands what passed, syncs `next`, and checks its tip, with no agent in between; what it refuses
goes to the lander once.  On `test_queue`'s throwaway origin, with the launch, the runs' states and the merge-thread post injected."""
from __future__ import annotations

import os
import pytest

from research import merge
from research.drive import PUBLISH_GRACE_S, Driver
from research.cli import main as research_main
from test_merge_gate import CHECKED
from test_queue import World, _git


class Harness:
    def __init__(self, world: World, monkeypatch, **kw):
        self.world, self.launched, self.posts, self.after, self.cancelled = world, [], [], [], []
        self.run_state: dict[str, tuple] = {}
        self.clock = 1_000_000.0
        monkeypatch.setattr(merge, "caught_up", lambda repo, remote, tool, since="", log=print, runs=None:
                            self.after.append((since, runs)))
        monkeypatch.setattr(merge, "_sweep_after", lambda *a: None)
        self.q = world.queue(**kw)
        self.d = Driver(self.q, on="m1", launch=self.launch, states=lambda repo, ids: {i: self.run_state[i] for i in ids},
                        post=lambda text: self.posts.append(text) or True, now=lambda: self.clock,
                        cancel=lambda repo, ids, reason: self.cancelled.extend(ids))

    def launch(self, repo, commit, machine, agreement):
        rid = f"r20261012-0000{len(self.launched):02d}-abcd"
        self.launched.append((commit, machine, agreement))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def tick(self, **kw) -> str:
        return self.d.tick(**kw)


def _check(world: World, monkeypatch, commit: str, command=CHECKED) -> None:
    """A recorded `check` attempt of `commit` that runs `command` (CHECKED passes)."""
    monkeypatch.setenv("RESEARCH_STORE", str(world.tmp / "store"))
    tree = world.tmp / f"check-{commit[:12]}"
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "worktree", "add", "-q", "--detach", str(tree), commit)
    monkeypatch.chdir(tree)
    research_main(["run", "--runs-dir", str(world.tmp / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                   "--cwd", "source", "--", *command])


def _tip(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/next")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_an_unchecked_tip_is_checked_once_with_agreement_only_for_a_guarantee_change(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    assert h.tick().startswith("launched")
    assert h.launched == [(_tip(world), "m1", False)]
    assert "running" in h.tick() and len(h.launched) == 1          # its launch is still running: no second check
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}\n"}))
    h.tick()
    assert h.launched[-1] == (_tip(world), "m1", True)             # a new tip, and it changes a lean-audit.json


def test_launches_wait_while_max_running_checks_of_next_run(world, monkeypatch):
    h = Harness(world, monkeypatch)
    h.d.max_running = 1
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    assert "waits" in h.tick() and len(h.launched) == 1


def test_a_passing_tip_fast_forwards_main_and_the_landing_post_names_its_check(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    tip, before = _tip(world), _git(world.origin, "rev-parse", "main")
    _check(world, monkeypatch, tip)
    assert h.tick().startswith("landed #2")
    assert _git(world.origin, "rev-parse", "main") == tip           # exactly the checked commit
    (since, runs), = h.after
    assert since == before and set(runs.values()) == {next(iter(runs.values()))} and tip in runs
    lines = merge.landing_lines(h.q.workdir, before, tip, runs=runs)
    assert [ln.split(" ", 4)[-1] for ln in lines] == [f"#1; checked by {runs[tip]}", f"#2; checked by {runs[tip]}"]


def test_the_steps_after_a_landing_run_with_the_queues_credentials_and_git_never_prompts(world, monkeypatch):
    """12 Oct 08:49Z: `caught_up`'s `ls-remote` in the queue's clone, which has no credential helper, waited on a password."""
    h = Harness(world, monkeypatch)
    seen = []
    monkeypatch.setattr(merge, "caught_up", lambda *a, **kw: seen.append({k: os.environ.get(k) for k in
                                                                           ("GIT_TERMINAL_PROMPT", "GIT_CONFIG_COUNT")}))
    monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    h.q.auth = ("-c", "credential.helper=!h get")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world))
    assert h.tick().startswith("landed #1")
    assert seen == [{"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1"}]
    assert "GIT_TERMINAL_PROMPT" not in os.environ and "GIT_CONFIG_COUNT" not in os.environ


def test_a_tip_checked_before_main_moved_lands_as_a_merge_commit_when_they_share_no_file(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    tip = _tip(world)
    _check(world, monkeypatch, tip)
    other = world.change({"c.txt": "outside the queue\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert h.tick().startswith("landed #1")
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-list", "--parents", "-n", "1", main).split()[1:] == [other, tip]
    body = _git(world.origin, "log", "-1", "--format=%B", main)
    assert body.startswith(f"Merge {tip[:12]} (") and "\nChecked: r" in body and f"Skipped-check: main " in body


def test_a_tip_checked_before_main_moved_over_one_of_its_files_is_rebuilt_and_checked_again(world, monkeypatch):
    h = Harness(world, monkeypatch)
    lines = [f"{i}\n" for i in range(8)]
    _git(world.dev, "push", "-q", "origin", f"{world.change({'m.txt': ''.join(lines)})}:refs/heads/main")
    world.gh.put(1, world.change({"m.txt": "".join(["first\n", *lines[1:]])}))
    h.tick()
    old = _tip(world)
    _check(world, monkeypatch, old)
    other = world.change({"m.txt": "".join([*lines[:-1], "last\n"])})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert h.tick().startswith("launched")
    assert _git(world.origin, "rev-parse", "main") == other and h.launched[-1][0] == _tip(world) != old


def test_a_main_that_changed_check_tooling_is_never_landed_on_unchecked(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    old = _tip(world)
    _check(world, monkeypatch, old)
    other = world.change({"tools/verity/check/check.py": "# a new step\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert h.tick().startswith("launched")
    assert _git(world.origin, "rev-parse", "main") == other and h.launched[-1][0] == _tip(world) != old


def test_a_failed_check_and_a_launch_that_registers_nothing_are_told_to_the_lander_once(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.tick()
    h.tick()
    assert len(h.posts) == 1 and "failed its check" in h.posts[0] and h.posts[0].startswith("@old-circuits-and-proofs")
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.d.launch = lambda *a: ([], "train: no parent store key")
    h.tick()
    h.tick()
    assert len(h.posts) == 2 and "registered no run: train: no parent store key" in h.posts[1]


def test_a_launch_that_ended_without_an_attempt_is_lost_after_the_grace_and_one_whose_twins_never_started_is_relaunched(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    (rid,) = h.run_state
    h.run_state[rid] = ("failed", 75, None)
    h.tick()
    assert len(h.launched) == 2                                     # no twin took a slot: launched again
    rid = sorted(h.run_state)[-1]
    h.run_state[rid] = ("done", 0, None)
    assert "running" in h.tick() and not h.posts                    # its attempt may still be on its way to the store
    h.clock += PUBLISH_GRACE_S + 1
    h.tick()
    assert len(h.posts) == 1 and "no attempt reached the store" in h.posts[0]


def test_next_is_checked_in_stacked_trains_and_a_failed_train_holds_up_only_the_ones_above_it(world, monkeypatch):
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 3
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt"), (5, "e.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    members = h.q.chain()
    assert [c for c, _, _ in h.launched] == [members[1].commit, members[3].commit, members[4].commit]
    _check(world, monkeypatch, members[1].commit)
    _check(world, monkeypatch, members[3].commit, ("false",))
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == members[1].commit       # the train below the failure lands
    assert len(h.posts) == 1 and f"{members[3].commit[:12]} on `next` (#3 #4) failed" in h.posts[0]


def test_shadow_launches_lands_and_tells_nothing(world, monkeypatch):
    h = Harness(world, monkeypatch, shadow=True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    before = _git(world.origin, "rev-parse", "main")
    assert h.tick().startswith("would check")
    _check(world, monkeypatch, _git(world.origin, "rev-parse", "refs/heads/queue-shadow"), ("false",))
    h.tick()
    assert not h.launched and not h.posts and _git(world.origin, "rev-parse", "main") == before
    assert any("would tell the lander" in s for s in world.logs)


def test_train_sh_runs_with_the_queues_credentials_and_git_never_prompts(tmp_path, monkeypatch):
    """12 Oct, drive's first live tick: train.sh's `git fetch` in the queue's clone, which has no credential helper of its own,
    waited on a password at the tty, and drive with it."""
    from research.drive import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text(
        'echo "prompt=$GIT_TERMINAL_PROMPT n=$GIT_CONFIG_COUNT $GIT_CONFIG_KEY_1=$GIT_CONFIG_VALUE_1 $GIT_CONFIG_KEY_2=$GIT_CONFIG_VALUE_2"\n'
        'read -r line && echo "stdin: $line"\necho "RUN r20261012-060000-aaaa"\n')
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "*")
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, auth=("-c", "credential.helper=", "-c", "credential.helper=!h get"))
    assert ids == ["r20261012-060000-aaaa"]
    assert "prompt=0 n=3 credential.helper= credential.helper=!h get" in tail and "stdin:" not in tail


def test_a_disjoint_move_of_main_keeps_next_while_its_check_runs_and_a_shared_one_rebuilds_it(world, monkeypatch):
    """12 Oct: the lander held a passed PR rather than land it beside drive's running check, since any move of `main` rebuilt
    `next` and wasted the check."""
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    old = _tip(world)
    other = world.change({"z.txt": "lander's\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert "stays on" in h.tick() and _tip(world) == old and len(h.launched) == 1
    _check(world, monkeypatch, old)
    assert h.tick().startswith("landed #1")
    assert _git(world.origin, "rev-parse", "main^1", "main^2").split() == [other, old]


def test_a_move_of_main_sharing_a_file_with_the_running_train_rebuilds_next(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    old = _tip(world)
    other = world.change({"a.txt": "lander's\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    h.tick()
    assert _tip(world) != old
    assert h.cancelled == ["r20261012-000000-abcd"]               # the old tip's check, still waiting, is cancelled
