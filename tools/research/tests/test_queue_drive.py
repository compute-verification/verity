"""`research queue drive`: what a tick tells and to whom, and the branch sweep, on `test_queue`'s throwaway origin with the
launch, the runs' states and the merge-thread post injected.  `Harness` is the drive `test_drive_review` and `test_drive_stale`
build on."""
from __future__ import annotations

import pytest

from research import merge
from research.drive import AFTER_LANDING, Driver
from test_landing import Harness as LandingHarness
from test_queue import World, _git


class Harness(LandingHarness):
    """`test_landing.Harness` on a `Driver`, which ticks."""

    make = Driver

    def tick(self, **kw) -> str:
        return self.d.tick(**kw)

    def baseline(self) -> None:
        """Drive's state with `main` verified on every after-landing step, as a passed after-landing run of it leaves it, so a
        tick lands without first running one."""
        self.q._clone()
        self.q.fetch()
        state = self.q.load_state()
        state.setdefault("drive", {})["postland"] = {
            "verified": _git(self.world.origin, "rev-parse", "main"), "pending": [], "run": None,
            "verified_as": {"steps": list(AFTER_LANDING), "no_cache": None}}
        self.q.save_state(state)


def _tip(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/next")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_a_driver_is_a_landing_on_drive_tomls_machines_and_warn_steps(world):
    d = Driver(world.queue(), on="m1,m2", warn_steps=("ratchet-gate",))
    assert (d.on, d.warn_steps) == ("m1,m2", ("ratchet-gate",))


def test_a_conflict_is_told_to_ci_and_the_lane_that_owns_the_pr(world, monkeypatch):
    """An ejection post names the PR's owner too, not only drive's own handle, so the owner doesn't hear of a conflict through a relay.  The
    owner's handle comes from main's registry: the lane the body names, the agent id it carries, the ready mark's author, or
    the agent whose id ends in the branch's suffix."""
    import dataclasses
    import json

    agent = "bc-b8aaadaa-aefd-503e-9863-5eabb3a28c79"
    reg = {"handles": [{"handle": "circuits", "holder": f"the circuits coordinator ({agent})"},
                       {"handle": "infra", "holder": "infra (bc-17cc41f1-6227-5fab-bc64-3fa2f224558b)"}]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append((text, to)) or True
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, world.change({"a.txt": "two\n"}), body="Lane `catalog`.")
    world.gh.prs[2] = dataclasses.replace(world.gh.prs[2], branch="cursor/catalog-8c79")
    h.tick()
    assert [to for _, to in sent] == [("@ci", "@circuits")]
    assert sent[0][0].startswith("@ci @circuits queue: #2 at ")
    q = h.q
    for pr, want in ((dataclasses.replace(world.gh.prs[2], body=f"[agent]({agent})", branch=""), "circuits"),
                     (dataclasses.replace(world.gh.prs[2], body="Lane `infra`.", branch=""), "infra"),
                     (dataclasses.replace(world.gh.prs[2], branch="cursor/x-0000"), None)):
        assert q.handle(pr) == want


def _lanes(world: World, *ns: int) -> None:
    """Main's registry with the handle `lane-N` for each of ``ns``, so each PR's owner is tagged."""
    import json

    reg = {"handles": [{"handle": f"lane-{n}", "holder": f"the lane-{n} lane"} for n in ns]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")


def _conflict_with_dependants(world: World, h: Harness, deps: tuple[int, ...]) -> str:
    """#1 on `next`, #2 conflicting with it, and ``deps`` each built on #2; #2's head."""
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, two := world.change({"a.txt": "two\n"}))
    for n in deps:
        world.gh.put(n, world.change({f"f{n}.txt": f"{n}\n"}, base=two))
    return two


def test_a_pr_that_didnt_merge_is_told_once_naming_every_pr_it_takes_out(world, monkeypatch):
    """#2 at one head stopped merging and drive posted a refusal for each PR containing it, each owner apart: one root cause is one
    post, naming why, every PR it takes out and each owner once; an identical sync posts nothing."""
    _lanes(world, 1, 2, 3, 4, 5)
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append((text, to)) or True
    two = _conflict_with_dependants(world, h, (3, 4, 5))
    h.tick()
    assert sent == [(f"@ci @lane-2 @lane-3 @lane-4 @lane-5 queue: #2 at {two[:8]} is out of `next`: conflicts with `next` in a.txt "
                     "(touched by #1). It takes out #3, #4, #5, which contain it", ("@ci", "@lane-2", "@lane-3", "@lane-4", "@lane-5"))]
    assert h.q.load_state()["drive"]["out"] == {"2": [two, [3, 4, 5], "conflicts with `next` in a.txt (touched by #1)"]}
    h.tick()
    assert len(sent) == 1
    assert h.q.status()[4].why == "contains #2, which is refused: conflicts with `next` in a.txt (touched by #1) (at the last sync)"


def test_a_root_cause_is_told_again_once_when_the_prs_it_takes_out_change(world, monkeypatch):
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append(text) or True
    two = _conflict_with_dependants(world, h, (3,))
    h.tick()
    assert sent == [f"@ci queue: #2 at {two[:8]} is out of `next`: conflicts with `next` in a.txt (touched by #1). It takes out #3, "
                    "which contains it"]
    world.gh.put(6, world.change({"f6.txt": "6\n"}, base=two))
    h.tick()
    h.tick()
    assert sent[1:] == [f"@ci queue: #2 at {two[:8]} is out of `next`: conflicts with `next` in a.txt (touched by #1). It takes out "
                        "#3, #6, which contain it"]


def test_two_independent_root_causes_are_two_posts(world, monkeypatch):
    """Each PR left out for a reason of its own is its own root, told with the PRs it takes out and only those."""
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append(text) or True
    two = _conflict_with_dependants(world, h, (3,))
    world.gh.put(7, world.change({"b.txt": "seven\n"}))
    world.gh.put(8, eight := world.change({"b.txt": "eight\n"}))
    world.gh.put(9, world.change({"f9.txt": "9\n"}, base=eight))
    h.tick()
    h.tick()
    assert sent == [f"@ci queue: #2 at {two[:8]} is out of `next`: conflicts with `next` in a.txt (touched by #1). It takes out #3, "
                    "which contains it",
                    f"@ci queue: #8 at {eight[:8]} is out of `next`: conflicts with `next` in b.txt (touched by #7). It takes out #9, "
                    "which contains it"]


def test_a_refusal_whose_post_was_refused_is_told_again(world, monkeypatch):
    h = Harness(world, monkeypatch)
    logged = []
    h.q.log = logged.append
    ok = [False]
    h.d.post = lambda text, to: ok[0]
    state = {"told": []}
    h.d.refuse(state, "k", "#1 conflicts", "circuits")
    assert state["told"] == [] and "could not post" in logged[-1]
    ok[0] = True
    h.d.refuse(state, "k", "#1 conflicts", "circuits")
    assert state["told"] == ["k"] and logged[-1].startswith("research queue drive: told @ci @circuits")


def test_an_absolute_path_touches_the_file_it_ends_in_under_a_directory():
    """A traceback's frame names an installed copy of a file: it touches the landing's file it ends in, but a bare
    `conftest.py` would match every conftest an installed package has."""
    from research.drive import touches

    assert touches({"conftest.py", "pkg/conftest.py"}, {"/x/site/pkg/conftest.py"}) == {"pkg/conftest.py"}
    assert touches({"tools/verity/lean/tests/test_lean_changed.py", "a.txt"},
                   {"tools/verity/lean/tests/test_lean_changed.py", "elsewhere/data.txt"}) == {
        "tools/verity/lean/tests/test_lean_changed.py"}


def test_drive_sweeps_the_repositorys_branches_once_a_day(world, monkeypatch):
    """Something deletes landed PRs' branches and idle ones, or the repository's branches pile up by the thousand."""
    from research.drive import BRANCH_SWEEP_S

    h = Harness(world, monkeypatch)
    calls = []
    monkeypatch.setattr(merge, "sweep_branches", lambda *a, **k: calls.append(a) or (2, 1))
    h.tick()
    h.tick()
    assert len(calls) == 1 and any("branch sweep: 2 deleted (on main), 1 archived" in x for x in world.logs)
    h.clock += BRANCH_SWEEP_S + 1
    h.tick()
    assert len(calls) == 2


def test_drive_deletes_the_head_branch_of_a_pr_it_closed_as_landed_unless_something_still_needs_it(world, monkeypatch):
    """A PR drive closes once `main` reaches it (`_close_carried`) leaves no branch behind, but drive keeps one an open PR
    uses as its head or base, a fork's, one of `KEEP_BRANCHES`, and one pushed to since (the lease)."""
    import dataclasses

    h = Harness(world, monkeypatch)
    h.q._clone()
    heads = {}
    for n, branch, base, fork in ((1, "lane/one", "main", False), (2, "lane/two", "main", False), (3, "lane/three", "lane/two", False),
                                  (4, "lane/four", "main", True), (5, "lane/five", "main", False), (6, "next", "main", False)):
        heads[n] = world.change({f"{n}.txt": f"{n}\n"})
        world.gh.put(n, heads[n])
        world.gh.prs[n] = dataclasses.replace(world.gh.prs[n], branch=branch, base=base, fork=fork)
        _git(world.origin, "update-ref", f"refs/heads/{branch}", heads[n])
    _git(world.origin, "update-ref", "refs/heads/lane/five", world.change({"5.txt": "pushed since\n"}))
    for n in (1, 2, 4, 5, 6):
        h.d._delete_head(n, heads[n])
    left = {r.split("refs/heads/")[1] for r in _git(world.origin, "for-each-ref", "--format=%(refname)", "refs/heads/").split()}
    assert {"lane/two", "lane/three", "lane/four", "lane/five", "next"} <= left and "lane/one" not in left
    logs = [x for x in world.logs if "#1's head" in x or "kept" in x]
    assert f"research queue drive: deleted origin's lane/one, #1's head at {heads[1][:12]}" in logs
    assert "research queue drive: kept lane/two, #2's head: another open PR's head or base" in logs
    assert "research queue drive: kept lane/four, #4's head: a fork's" in logs
    assert "research queue drive: kept next, #6's head: kept by drive" in logs
    assert any(x.startswith("research queue drive: kept lane/five, #5's head: ") and "stale info" in x for x in logs), logs


def test_a_branch_sweep_at_its_limit_sweeps_again_the_next_tick(world, monkeypatch):
    """A sweep pushes at most `SWEEP_LIMIT` deletions; the rest go on the ticks after, not a day later."""
    from research.merge import SWEEP_LIMIT

    h = Harness(world, monkeypatch)
    calls = []
    monkeypatch.setattr(merge, "sweep_branches", lambda *a, **k: calls.append(a) or ((SWEEP_LIMIT, 0) if len(calls) == 1 else (3, 0)))
    h.tick()
    assert any(f"at its limit of {SWEEP_LIMIT}, so the next tick sweeps again" in x for x in world.logs)
    h.tick()
    h.tick()
    assert len(calls) == 2


def test_the_after_landing_launch_gives_train_sh_no_cache_with_circuit_checks_cache(tmp_path):
    """Drive's after-landing run reuses no suite or Lean pass (`after_landing_no_cache`) but circuit-check's per-target ones: a
    cold `circuit-check --all` (~4 h) would outlast every after-landing run, so `main` would never be verified."""
    from research.drive import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text('echo "argv: $*"\necho "RUN r20261027-120000-aaaa"\n')
    extra = {"agreement": False, "no_cache": True, "circuit_cache": True, "no_regenerate": True}    # `start_postland`'s
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", extra.pop("agreement"), deferred="pytest,circuit-check", **extra)
    assert ids == ["r20261027-120000-aaaa"]
    assert f"argv: {'c' * 40} m1 --deferred pytest,circuit-check --no-cache --circuit-cache --no-regenerate" in tail, tail
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, deferred="ratchet-gate", base="b" * 40, ratchet_base="a" * 40)
    assert f"argv: {'c' * 40} m1 --deferred ratchet-gate --base {'b' * 40} --ratchet-base {'a' * 40}" in tail, tail


def test_the_after_landing_run_and_the_lock_run_take_a_check_slot_ahead_of_unranked_priority_checks(tmp_path, monkeypatch):
    """Each part of the after-landing run (a shard, circuit-check's records step) and the lock run ask for their check slot at
    `POSTLAND_SLOT_RANK` (CHECK_SLOT_RANK, which check.py passes on as `slot.py --rank`), whatever drive's own environment
    says: a ranked priority waiter takes the next free slot ahead of the unranked, so `main` never waits on a slot for a
    `train.sh` launched by hand."""
    import contextlib
    import os
    import tempfile
    import types

    from research.landing import POSTLAND_SLOT_RANK, launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("CHECK_SLOT_RANK", "7")
    repo = tmp_path / "clone"
    (repo / "tools" / "verity" / "check").mkdir(parents=True)
    (repo / "tools" / "verity" / "check" / "train.sh").write_text(
        'echo "rank=[$CHECK_SLOT_RANK]"\necho "RUN r20261103-140000-aaaa"\n')
    d = Driver(types.SimpleNamespace(auth=(), workdir=repo), on="m1")

    def tail(job: dict) -> str:
        with contextlib.suppress(ChildProcessError):                     # its end, not a clock
            os.waitpid(job["pid"], 0)
        return launched(job)[1]

    ranked = f"rank=[{POSTLAND_SLOT_RANK}]"
    assert ranked in tail(d.launch_postland(repo, "c" * 40, "m1", "pytest", shard="1/3"))
    assert ranked in tail(d.launch_postland(repo, "c" * 40, "m1", "circuit-check", cc_records=True))
    assert ranked in tail(d.launch_locks(repo, "c" * 40, "m1"))


def test_a_tick_sets_the_record_drivers_its_clone_lacks_and_plain_git_then_merges_two_records_with_merge_py(world, monkeypatch):
    """A queue clone seeded from a copy without the drivers (or with another value) gets every `RECORD_DRIVER` key on a tick,
    logged once, and a later tick only reads them; then a git without `git_env`'s settings merges two PRs' lean-audit.json
    records in that clone through merge.py rather than line by line."""
    import json
    import os
    from pathlib import Path

    from research.clone import RECORD_DRIVER

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    tools = {f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}
    base = world.change({**tools, ".gitattributes": "lean-audit.json merge=lean-audit\n"})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    wd = world.tmp / "q" / "verity"
    _git(world.tmp, "clone", "-q", str(world.origin), str(wd))
    _git(wd, "config", "--local", "merge.lean-caps.driver", "true")
    h = Harness(world, monkeypatch)
    assert h.q.workdir == wd

    def said() -> list[str]:
        return [x for x in world.logs if "set the record merge drivers" in x]

    h.tick()
    (line,) = said()
    assert line.endswith(", ".join(RECORD_DRIVER)), line
    assert {k: _git(wd, "config", "--local", "--get", k) for k in RECORD_DRIVER} == RECORD_DRIVER
    h.tick()
    assert said() == [line]

    rec = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    a, b = (world.change({"pkg/lean-audit.json": json.dumps({**rec, "guarantees": {**rec["guarantees"], g: "1"}})}, base=base)
            for g in "AB")
    for k in [k for k in os.environ if k.startswith("GIT_CONFIG")]:
        monkeypatch.delenv(k)
    _git(wd, "fetch", "-q", "origin", f"refs/dev/{a}", f"refs/dev/{b}")
    tree = _git(wd, f"--attr-source={base}", "merge-tree", "--write-tree", "--no-messages", a, b)
    assert set(json.loads(_git(wd, "show", f"{tree}:pkg/lean-audit.json"))["guarantees"]) == {"T", "A", "B"}
