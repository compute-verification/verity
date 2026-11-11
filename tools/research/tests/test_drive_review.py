"""The review landing (`land = "review"`, here with `land_into = "main"`: PRs land on review, and the full check runs on
`main` after; `test_drive_checked.py` has `land_into = "checked"`, where `main` moves only to a checked commit, and
`test_drive_locks.py` the lock run after a Lean landing).  A tick lands `next`'s longest prefix whose every member has a land
verdict at its head by a reviewer who isn't its owner, a Lean change as any other; the after-landing run checks `main`, and
its failure names the one landing it implicates once `main` passed that step before, to be fixed forward or reverted
`FIX_FORWARD_S` after; several implicated are each rung, and none is reverted."""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

import research.drive as D
import research.queue as Q
from research import merge
from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL, LEAN_STEP, SIGTERM_RC
from research.queue import PR, QueueError, StoreLabels, label_target, lean_change
from research.store import vocab
from test_queue import World, _git
from test_queue_drive import Harness, _failed_run, _killed_run, _steps_run, _tip

EVERY = {s: "passed" for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL}
#: a ci.toml with a job for each after-landing step, and drive's policy taking circuit-check off the merge gate as in the tree
CI_TOML = 'schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{s}"\nowner = "ci"\ngroup = 1\n' for s in AFTER_LANDING)
POLICY_TOML = 'schema = "policy/v0"\noff_gate = ["circuit-check"]\n'
#: the files `_move_main` writes for a tree declaring `CI_TOML` under `POLICY_TOML`
GATED_TREE = {"tools/verity/check/ci.toml": CI_TOML, "tools/verity/check/policy.toml": POLICY_TOML}


def test_the_after_landing_run_runs_circuit_check_which_is_off_the_merge_gate():
    """circuit-check is off the merge gate (policy.toml's `off_gate`): no recorded check runs it, so the after-landing run must,
    and must pass it to verify `main` (it is no step the run may skip by name)."""
    from research.drive import CHECK_STEPS

    assert AFTER_LANDING == tuple(s for s in CHECK_STEPS if not s.startswith("preflight-"))
    assert "circuit-check" in AFTER_LANDING and "circuit-check" not in AFTER_LANDING_OPTIONAL and "circuit-check" in EVERY


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_main_waits_for_the_steps_on_the_gate_of_the_commits_own_ci_toml(world, monkeypatch):
    """`_promote` waits at a commit for every after-landing step but those a run skips by name and the jobs that commit's own
    drive policy takes off the merge gate (circuit-check); for every such step when its ci.toml or policy can't be read."""
    h = Review(world, monkeypatch)
    every = [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL]
    assert h.d._gating(_main(world)) == every
    _move_main(world, GATED_TREE)
    h.q.fetch()
    assert h.d._gating(_main(world)) == [s for s in every if s != "circuit-check"]
    _move_main(world, {"tools/verity/check/policy.toml": "not toml ["})
    h.q.fetch()
    assert h.d._gating(_main(world)) == every
    _move_main(world, {"tools/verity/check/policy.toml": POLICY_TOML, "tools/verity/check/ci.toml": "not toml ["})
    h.q.fetch()
    assert h.d._gating(_main(world)) == every


def test_a_git_without_merge_tree_attr_source_refuses_the_tick_and_rings_ci_and_infra_hourly(world, monkeypatch):
    """A git before 2.40 has no `merge-tree --attr-source`, and a sync without it left out every PR whose only conflict is a pin,
    ringing no one.  Drive refuses the tick on it (no sync, nothing pushed) and rings the lander's and the machines' handles
    with the version and the fix, again an hour later while it stays; on a git that has it the same tick syncs and lands."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    refs = _git(world.origin, "for-each-ref")
    monkeypatch.setattr(D, "git_version", lambda: "git version 2.34.1")
    for _ in range(2):
        assert h.tick().startswith("drive refuses its ticks: `git version 2.34.1` has no `merge-tree --attr-source`"), world.logs
    assert _git(world.origin, "for-each-ref") == refs
    (ring,) = h.posts
    assert ring.startswith("@ci @infra queue: drive refuses its ticks") and "Fix: install git >= 2.40" in ring
    h.clock += D.OLD_GIT_EVERY_S
    h.tick()
    assert h.posts == [ring, ring]
    monkeypatch.setattr(D, "git_version", lambda: "git version 2.40.0")
    assert h.tick().startswith("landed #1"), world.logs


def _registry(world: World, *handles: str) -> None:
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane"} for h in handles]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")


def _tree(world: World, ref: str) -> str:
    return _git(world.origin, "rev-parse", f"{ref}^{{tree}}")


class Review(Harness):
    """A drive landing on reviews, whose verdicts are `reviewed` labels in the world's store.  Unless ``baseline`` is off, `main`
    is verified (`Harness.baseline`), so a tick lands without first running the after-landing run on `main`."""

    def __init__(self, world, monkeypatch, baseline: bool = True, **kw):
        super().__init__(world, monkeypatch, **kw)
        self.d.reviews = lambda pr, head: world.store.labels(label_target(pr, head), key=vocab.REVIEWED)
        self.d.ready_by = lambda pr, head: next((lab.by for lab in sorted(world.store.labels(label_target(pr, head), key=vocab.READY),
                                                                          key=lambda lab: lab.ts, reverse=True)), None)
        if baseline:
            self.baseline()

    def review(self, pr: int, verdict: str = "land", by: str = "reviewer", ref: str | None = None) -> None:
        self.world.store.add_label(label_target(pr, self.world.gh.prs[pr].head), vocab.REVIEWED, verdict, by, ref=ref)

    def lock_ids(self) -> list[str]:
        """The lock runs' ids, oldest first."""
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
    assert h.launched == []                                     # no lock run: no Lean changes
    verified = _git(world.origin, "rev-parse", "main^1^1")
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": False, "no_cache": True, "circuit_cache": True,
                                                            "no_regenerate": True, "base": verified})]
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


@pytest.mark.parametrize("ended", ["passed", "red"])
def test_the_next_after_landing_run_launches_before_the_sync_of_the_tick_its_predecessor_ended_in(world, monkeypatch, tmp_path,
                                                                                                  ended):
    """The sync takes most of a tick, and an after-landing run launched only after it left `main`'s tip unchecked that long
    each time a run ended.  The tick that reads the end launches the next before the sync, once the reverts due are pushed
    (a red's culprit named in that tick has `FIX_FORWARD_S` to be fixed forward first), and launches it once: what lands in
    that tick waits for the run after."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"} if ended == "red" else
                                 {"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs                 # while r-post-0 runs
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    h.review(3)
    calls: list[str] = []
    sync, launch = h.q.sync, h.d.launch_postland
    monkeypatch.setattr(h.q, "sync", lambda dry_run: calls.append("sync") or sync(dry_run))
    h.d.launch_postland = lambda repo, commit, *a, **kw: calls.append(commit) or launch(repo, commit, *a, **kw)
    if ended == "red":
        h.run_state["r-post-0"] = ("done", 1, None)
        h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    else:
        h.run_state["r-post-0"] = ("done", 0, None)
        h.evidence = [_steps_run(tmp_path, "r-post-0", {s: "passed" for s in h.posted[0][1].split(",")})]
    assert h.tick().startswith("landed #3"), world.logs
    first = _git(world.origin, "rev-parse", "main^1")                  # below #3's landing: #2's, #1 not reverted yet
    assert _subject(world, first).startswith("Queue: #2"), _subject(world, first)
    assert calls == [first, "sync"] and [c for c, *_ in h.posted[1:]] == [first], (calls, world.logs)
    assert h.postland()["run"]["tip"] == first and h.postland()["pending"][-1] == _main(world) != first


def test_a_sharded_after_landing_run_verifies_main_only_once_every_shard_passed(world, monkeypatch):
    """`after_landing_shards = N`: the after-landing run is N runs at once, shard K of ci.toml's plan with N shards each, so it
    takes about its longest shard rather than every step on one slot (2.5 h). `main` is verified once each shard passed; a shard
    that never started launches the run again, whole."""
    h = Review(world, monkeypatch)
    h.d.after_landing_shards = 3
    art = f"art:{'a' * 64}@{'b' * 64}"
    h.d.records_artifact = lambda rid: art if rid == "r-post-0" else None
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    main = _main(world)
    assert h.posted == [(main, "circuit-check", {"cc_records": True})], "circuit-check's records step first"
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    assert [(c, kw.get("shard"), kw.get("cc_records_from")) for c, _, kw in h.posted[1:]] == [(main, f"{k}/3", art) for k in (1, 2, 3)]
    h.run_state.update({"r-post-1": ("done", 0, None), "r-post-2": ("running", None, None), "r-post-3": ("done", 0, None)})
    h.evidence = [_steps_run(world.tmp, "r-post-1", {s: "passed" for s in AFTER_LANDING})]
    h.tick()
    pl = h.postland()
    assert pl["run"]["groups"] == [["r-post-1"], ["r-post-2"], ["r-post-3"]] and pl["verified"] != main, "one shard runs on"
    h.run_state["r-post-2"] = ("done", 75, None)
    h.tick()
    assert h.postland()["verified"] != main and len(h.posted) == 7, "a shard that never started: the run again, whole"
    assert [kw.get("cc_records_from") for _, _, kw in h.posted[4:]] == [art] * 3, "on the records it had: no second records step"
    h.run_state.update({f"r-post-{i}": ("done", 0, None) for i in (4, 5, 6)})
    h.tick()
    pl = h.postland()
    assert pl["verified"] == main and pl["run"] is None, world.logs


@pytest.mark.parametrize("ends", ["failed", "no artifact", "lost"])
def test_a_sharded_run_s_shards_build_the_records_themselves_when_the_records_step_gives_none(world, monkeypatch, ends):
    """The records step holds the shards only while it may still give its artifact: failed, passed without one by
    `RECORDS_WAIT_S`, or still running then (cancelled), the shards launch without it and each builds the records itself."""
    from research.drive import RECORDS_WAIT_S

    h = Review(world, monkeypatch)
    h.d.after_landing_shards = 3
    h.d.records_artifact = lambda rid: None
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    h.run_state["r-post-0"] = {"failed": ("failed", 1, None), "no artifact": ("done", 0, None), "lost": ("running", None, None)}[ends]
    if ends != "failed":
        h.tick()
        assert len(h.posted) == 1, "it waits for the artifact"
        h.clock += RECORDS_WAIT_S + 1
    h.tick()
    assert [(kw.get("shard"), "cc_records_from" in kw) for _, _, kw in h.posted[1:]] == [(f"{k}/3", False) for k in (1, 2, 3)]
    assert h.cancelled == (["r-post-0"] if ends == "lost" else [])


def test_an_unsharded_after_landing_run_has_no_records_step(world, monkeypatch):
    """One run of every step runs circuit-check's `--all` whole, whose prologue builds the records once anyway."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    [(_, steps, kw)] = h.posted
    assert steps == ",".join(AFTER_LANDING) and "cc_records" not in kw and "shard" not in kw


class Planner:
    """Drive's auto plans (`plan_shards`, `planned`): each one asked for, and the plan (or why there is none) each answers once
    ``answer`` is set."""

    def __init__(self, h: Review, tmp: Path):
        self.asked: list[tuple[str, str, int | None]] = []
        self.answer: tuple[dict | None, str | None] = (None, None)
        self.tmp = tmp
        h.d.plan_shards = self.plan
        h.d.planned = lambda rec: self.answer

    def plan(self, repo, tip, steps, max_shards=None) -> dict:
        from research.drive import AUTO_PLAN

        self.asked.append((tip, steps, max_shards))
        out = self.tmp / f"plan-{len(self.asked)}"
        out.mkdir()
        (out / AUTO_PLAN).write_text("{}\n")
        return {"dir": str(out / "job"), "pid": None, "out": str(out)}


@pytest.mark.parametrize("rc", [0, 2])
def test_the_plan_job_plans_in_a_worktree_of_the_tip_and_planned_reads_its_plan(tmp_path, rc):
    """`PLAN_JOB`, as `detach` runs it (here to its end): `check.py --shard-plan` in a worktree of the tip under TRAIN_TREES,
    its venv kept beside, the plan written into the plan's directory, the worktree removed after; `planned` gives the plan,
    or why there is none, and keeps the plan's directory (its shards send the file) but not the job's."""
    from research.drive import AUTO_PLAN, DETACH_JOB, PLAN_JOB, planned

    repo, out, job, trees, bin_ = (tmp_path / x for x in ("repo", "plan", "job", "trees", "bin"))
    for d in (repo, out, job, bin_):
        d.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "tip")
    tip = _git(repo, "rev-parse", "HEAD")
    doc = {"kind": "check-shard-plan/v1", "n": 3, "digest": "d" * 64}
    (bin_ / "uv").write_text("#!/bin/sh\n"
                             f'printf "%s\\n" "$PWD" "$UV_PROJECT_ENVIRONMENT" "$(git rev-parse HEAD)" "$@" > {tmp_path}/argv\n'
                             'while [ "$1" != --out ]; do shift; done\n'
                             f"echo '{json.dumps(doc)}' > \"$2\"\necho 'shard plan: N = 3'\nexit {rc}\n")
    (bin_ / "uv").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:/usr/bin:/bin", "TRAIN_TREES": str(trees)}
    subprocess.run(["sh", "-c", DETACH_JOB, "sh", str(job), "sh", "-c", PLAN_JOB, "sh", str(out), tip, "--deferred", "pytest",
                    "--max-shards", "3"], cwd=repo, env=env, check=True)
    cwd, venv, head, *argv = (tmp_path / "argv").read_text().splitlines()
    assert (cwd, venv, head) == (str(trees / "plan-plan"), str(trees / "plan-venv"), tip)
    assert argv == ["run", "-q", "python", "tools/verity/check/check.py", "--shard-plan", "--out", str(out / AUTO_PLAN),
                    "--deferred", "pytest", "--max-shards", "3"]
    assert not (trees / "plan-plan").exists() and _git(repo, "worktree", "list").count("\n") == 0, "the worktree is removed"
    got = planned({"dir": str(job), "pid": None, "out": str(out)})
    assert got == ((doc, None) if rc == 0 else (None, "`check.py --shard-plan` exited 2: shard plan: N = 3"))
    assert not job.exists() and (out / AUTO_PLAN).is_file()


def _plan_doc(n: int) -> dict:
    return {"kind": "check-shard-plan/v1", "n": n, "digest": "d" * 64, "predicted_s": 3600.0}


@pytest.mark.parametrize("gone", [False, True])
def test_an_auto_after_landing_run_launches_the_planners_n_shards_of_one_plan_after_the_records_step(world, monkeypatch, gone):
    """`after_landing_shards = "auto:MAX"`: the plan (`check.py --shard-plan --deferred STEPS --max-shards MAX`) is made once a
    tip beside circuit-check's records step, and the run launches once both are done: the plan's N shards, each `--shard
    auto:K/N` with the plan's file and the records' artifact.  A shard that never started launches the run again on the same
    plan, or on a new one when its file is gone (a restart of the pod); `main` is verified once every shard passed."""
    from research.drive import AUTO_PLAN

    h = Review(world, monkeypatch)
    h.d.after_landing_shards = "auto:4"
    planner = Planner(h, world.tmp)
    art = f"art:{'a' * 64}@{'b' * 64}"
    h.d.records_artifact = lambda rid: art if rid == "r-post-0" else None
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    main = _main(world)
    assert h.posted == [(main, "circuit-check", {"cc_records": True})] and planner.asked == [(main, ",".join(AFTER_LANDING), 4)]
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    assert len(h.posted) == 1, "the records step passed, but the plan is still being made"
    planner.answer = (_plan_doc(2), None)
    h.tick()
    plan = str(world.tmp / "plan-1" / AUTO_PLAN)
    assert [(c, kw.get("shard"), kw.get("shard_plan"), kw.get("cc_records_from")) for c, _, kw in h.posted[1:]] == [
        (main, f"auto:{k}/2", plan, art) for k in (1, 2)]
    assert h.postland()["run"]["plan"] == "d" * 64 and planner.asked == [(main, ",".join(AFTER_LANDING), 4)]
    h.run_state.update({"r-post-1": ("done", 0, None), "r-post-2": ("done", 75, None)})
    if gone:
        os.unlink(plan)
        planner.answer = (None, None)
        h.tick()
        assert len(h.posted) == 3 and len(planner.asked) == 2 and not (world.tmp / "plan-1").exists(), "a new plan first"
        plan = str(world.tmp / "plan-2" / AUTO_PLAN)
        planner.answer = (_plan_doc(2), None)
    h.tick()
    assert [(kw.get("shard"), kw.get("shard_plan")) for _, _, kw in h.posted[3:]] == [("auto:1/2", plan), ("auto:2/2", plan)]
    assert len(planner.asked) == 1 + gone, "a shard that never started: the run again, whole, on the plan it had"
    h.evidence = [_steps_run(world.tmp, "r-post-3", {s: "passed" for s in AFTER_LANDING})]
    h.run_state.update({"r-post-3": ("done", 0, None), "r-post-4": ("done", 0, None)})
    h.tick()
    pl = h.postland()
    assert pl["verified"] == main and pl["run"] is None, world.logs


def test_a_run_ci_adds_to_an_auto_shard_counts_only_as_a_shard_of_its_plan(world, monkeypatch):
    """`drive-shard-add` to an auto run: the run's record must be `--shard auto:K/N` and its inputs must hold the run's plan
    (`AUTO_PLAN`), so a hand run of another plan's shard K, which ran other parts, verifies nothing."""
    from research.drive import AUTO_PLAN

    h = Review(world, monkeypatch)
    h.d.after_landing_shards = "auto"
    planner = Planner(h, world.tmp)
    planner.answer = (_plan_doc(2), None)
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}"
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    h.run_state.update({"r-post-1": ("done", 0, None), "r-post-2": ("waiting", None, None)})
    h.tick()
    main = _main(world)
    check = ["uv", "run", "python", "tools/verity/check/check.py", "--keep-going", "--deferred", ",".join(AFTER_LANDING)]
    records = {}
    for rid, spec, digest in (("r-hand", "auto:2/2", "d" * 64), ("r-other-plan", "auto:2/2", "e" * 64),
                              ("r-plain", "2/2", "d" * 64), ("r-no-plan", "auto:2/2", None)):
        (d := world.tmp / rid / "inputs").mkdir(parents=True)
        if digest:
            (d / AUTO_PLAN).write_text(json.dumps({"digest": digest}))
        records[rid] = (world.tmp / rid, {"tree_sha": main, "command": [*check, "--shard", spec]})
    monkeypatch.setattr(merge, "fetch_run", lambda rid, names, cwd=None, refresh=False: records.get(rid))
    monkeypatch.setenv(merge.DRIVE_ENV, "1")
    for rid, why in (("r-other-plan", f"auto plan ({AUTO_PLAN} among its inputs) is eeeeeeeeeeee, not the after-landing "
                                      "run's dddddddddddd"),
                     ("r-plain", "r-plain is shard 2/2, not shard auto:2/2"),
                     ("r-no-plan", "auto plan (check-shard-plan.json among its inputs) is unreadable")):
        with pytest.raises(QueueError, match=re.escape(why)):
            Q.drive_shard_add(h.q, main, 2, rid)
    assert Q.drive_shard_add(h.q, main, 2, "r-hand") == 0
    assert h.postland()["run"]["groups"] == [["r-post-1"], ["r-post-2", "r-hand"]]


@pytest.mark.parametrize("ends", ["failed", "lost"])
def test_an_auto_after_landing_run_with_no_plan_runs_unsharded_and_says_so(world, monkeypatch, ends):
    """A plan that failed, or that is still being made `PLAN_WAIT_S` after it started (stopped then), never holds the run: ci is
    told once and the run launches unsharded, one run of every step, its plan's directory removed."""
    from research.drive import PLAN_WAIT_S

    h = Review(world, monkeypatch)
    h.d.after_landing_shards = "auto"
    planner = Planner(h, world.tmp)
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}"
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    assert planner.asked == [(_main(world), ",".join(AFTER_LANDING), None)]
    h.run_state["r-post-0"] = ("done", 0, None)
    if ends == "failed":
        planner.answer = (None, "`check.py --shard-plan` exited 2: no store")
    else:
        h.tick()
        assert len(h.posted) == 1, "it waits for the plan"
        h.clock += PLAN_WAIT_S + 1
    h.tick()
    [(_, steps, kw)] = h.posted[1:]
    assert steps == ",".join(AFTER_LANDING) and "shard" not in kw and "cc_records_from" not in kw
    told = [m for m in h.posts if "auto shard plan" in m]
    assert len(told) == 1 and "runs unsharded" in told[0] and ("exited 2: no store" in told[0] if ends == "failed"
                                                              else f"not made in {PLAN_WAIT_S // 60} min" in told[0]), h.posts
    assert not (world.tmp / "plan-1").exists()


@pytest.mark.parametrize("on, n, want", [
    ("m1,m2,m3,m4,m5", 3, ["m1,m4", "m2,m5", "m3"]),
    ("m1,m2,m3", 3, ["m1", "m2", "m3"]),
    ("m1,m2", 3, ["m1", "m2", "m1"]),
    ("auto", 2, ["auto", "auto"]),
])
def test_each_shard_of_the_after_landing_run_gets_machines_no_other_shard_has(world, monkeypatch, on, n, want):
    """train.sh twins only when no slot is free, so shards launched with one machine list all took its one free slot and
    waited one behind another.  Shard K gets every Nth machine of `on` from the Kth, in `on`'s order; with fewer machines than
    shards they share, and drive says so once."""
    h = Review(world, monkeypatch)
    h.d.on, h.d.after_landing_shards, machines = on, n, []
    h.d.launch_postland = lambda repo, commit, machine, steps, **kw: machines.append(machine) or h.launch_post(repo, commit, machine,
                                                                                                             steps, **kw)
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}"
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    h.run_state["r-post-0"] = ("done", 0, None)                  # circuit-check's records step, on any of `on`
    h.tick()
    assert machines == [on, *want] and [kw["shard"] for _, _, kw in h.posted[1:]] == [f"{k}/{n}" for k in range(1, n + 1)]
    machines = machines[1:]
    shared = [line for line in world.logs if "fewer machines than shards" in line]
    if len(on.split(",")) >= n:
        assert not shared and len({m for ms in machines for m in ms.split(",")}) == sum(len(ms.split(",")) for ms in machines)
    else:
        assert len(shared) == 1, world.logs


def test_a_run_ci_adds_to_a_shard_counts_as_its_twin(world, monkeypatch):
    """A shard whose twin waits behind another shard: ci launches its run elsewhere and `drive-shard-add TIP K RUN_ID` makes it
    one of that shard's twins, so its pass passes the shard.  Refused for another tip, a shard out of range, a run already
    counted, a run still launching, off the control pod, and for a run whose own record isn't that shard of that commit (another
    commit, shard or steps, no check.py, or no readable record), so a passing run of a mistyped id verifies nothing."""
    from research.merge import DRIVE_ENV

    h = Review(world, monkeypatch)
    h.d.after_landing_shards = 2
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}"
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    h.run_state["r-post-0"] = ("done", 0, None)                  # circuit-check's records step
    h.tick()
    main = _main(world)
    check = ["uv", "run", "python", "tools/verity/check/check.py", "--keep-going"]
    steps = ["--deferred", ",".join(AFTER_LANDING)]
    records = {"r-hand": (main, [*check, *steps, "--shard", "2/2"]),
               "r-other-commit": (_git(world.origin, "rev-parse", "main^1"), [*check, *steps, "--shard", "2/2"]),
               "r-other-shard": (main, [*check, *steps, "--shard", "1/2"]),
               "r-unsharded": (main, [*check, *steps]),
               "r-other-steps": (main, [*check, "--deferred", "lean-audit", "--shard", "2/2"]),
               "r-no-check": (main, ["bash", "-lc", "true"])}
    monkeypatch.setattr(merge, "fetch_run", lambda rid, names, cwd=None, refresh=False: (
        (world.tmp, {"tree_sha": records[rid][0], "command": records[rid][1]}) if rid in records else None))
    argv = ["drive-shard-add", main[:12], "2", "r-hand", "--workdir", str(h.q.workdir), "--remote", str(world.origin), "--repo",
            "o/r", "--store", str(world.tmp / "store"), "--rules", str(world.dev / "ci" / "queue.toml")]
    with pytest.raises(QueueError, match="still launching"):
        Q.drive_shard_add(h.q, main, 2, "r-hand")
    h.run_state.update({"r-post-1": ("done", 0, None), "r-post-2": ("waiting", None, None), "r-hand": ("done", 0, None)})
    h.evidence = [_steps_run(world.tmp, "r-hand", {s: "passed" for s in AFTER_LANDING})]
    h.tick()
    assert h.postland()["verified"] != main, "shard 2's only twin waits"
    monkeypatch.delenv(DRIVE_ENV, raising=False)
    assert Q.main(argv) == 1 and h.postland()["run"]["groups"] == [["r-post-1"], ["r-post-2"]]
    monkeypatch.setenv(DRIVE_ENV, "1")
    for tip, k, rid, why in ((_git(world.origin, "rev-parse", "main^1"), 2, "r-hand", "is not the tip"),
                             (main[:6], 2, "r-hand", "is not the tip"), (main, 3, "r-hand", "shard 3 is not one of the run's 2"),
                             (main, 0, "r-hand", "shard 0 is not one"), (main, 2, "r-post-1", "already a run"),
                             (main, 2, "r-other-commit", f"r-other-commit checked {records['r-other-commit'][0][:12]}, not"),
                             (main, 2, "r-other-shard", "r-other-shard is shard 1/2, not shard 2/2"),
                             (main, 2, "r-unsharded", "r-unsharded is no shard, not shard 2/2"),
                             (main, 2, "r-other-steps", "ran --deferred lean-audit, not"),
                             (main, 2, "r-no-check", "is no check.py run"), (main, 2, "r-typo", "can't be read")):
        with pytest.raises(QueueError, match=why):
            Q.drive_shard_add(h.q, tip, k, rid)
    with pytest.raises(SystemExit):
        Q.main([*argv[:3], *argv[4:]])
    assert Q.main(argv) == 0
    run = h.postland()["run"]
    assert run["groups"] == [["r-post-1"], ["r-post-2", "r-hand"]] and run["ids"] == ["r-post-1", "r-post-2", "r-hand"]
    h.tick()
    assert h.postland()["verified"] != main, "drive waits while a twin it replaced still waits: ci cancels it"
    h.run_state["r-post-2"] = ("done", SIGTERM_RC, None)
    h.tick()
    pl = h.postland()
    assert pl["verified"] == main and pl["run"] is None, world.logs
    assert h.cancelled == [] and len(h.posted) == 3


def test_an_after_landing_run_sends_a_base_below_its_tip_so_ratchet_gate_judges_the_landings_above_it(world, monkeypatch):
    """Against origin/main, its own tip, the after-landing run's ratchet-gate judged no change and could never fail.  Drive
    sends the last verified `main` as its base (`train.sh --base`), below the tip with every landing since above it; the
    baseline run, of the verified `main` itself, sends the tip's first parent."""
    _move_main(world, {"x.txt": "x\n"})
    h = Review(world, monkeypatch, baseline=False)
    verified = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(1)
    h.review(2)
    assert h.tick().startswith("nothing lands before an after-landing run of `main`"), world.logs
    ((tip, _, kw),) = h.posted
    assert tip == verified and kw["base"] == _git(world.origin, "rev-parse", f"{verified}^1")
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING})]
    assert h.tick().startswith("landed #2"), world.logs
    tip, _, kw = h.posted[-1]
    assert tip == _main(world) != verified and kw["base"] == verified == h.postland()["verified"]
    assert _git(world.origin, "merge-base", kw["base"], tip) == verified                   # below the tip
    assert [m.pr for m in h.d.landed_since(kw["base"], tip)] == [1, 2]                    # the landings it judges


def test_an_unreviewed_member_new_to_next_stops_the_prefix_below_it_and_a_held_one_goes_to_the_top(world, monkeypatch):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(3)
    assert h.tick().startswith("landed #1"), world.logs        # the lift moves only what `next` held before the sync
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2)
    h.review(2, "hold", by="other", ref="reads a stale fixture")
    assert h.tick().startswith("landed #3"), world.logs
    assert [m.pr for m in h.q.chain()] == [2]
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #2") and "held by other (reads a stale fixture)" in out, out
    assert _subject(world, "main").startswith("Queue: #3")
    h.review(2, by="other")                                     # each reviewer's newest verdict counts
    assert h.tick().startswith("landed #2")
    assert _main(world) == _tip(world)
    assert h.launched == []


def test_no_verdict_counts_by_the_owner_its_handle_or_its_ready_marks_author_whatever_the_case(world, monkeypatch):
    world.with_store_remote()
    _registry(world, "lane-1")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.store.add_label(label_target(1, world.gh.prs[1].head), vocab.READY, "true", "marker")
    for by in ("Lane-1", "@LANE-1", "Marker"):                  # #1's owning lane, its handle, its ready mark's author
        h.review(1, by=by)
        out = h.tick()
        assert "no land verdict but its owner's own" in out and _subject(world, "main") == "change ['.agents/skills/using-slack/registry.json']", out
    with pytest.raises(QueueError, match="#1 is MARKER's own PR"):
        Q.review(h.q, world.store, StoreLabels(world.store, log=lambda s: None), 1, "land", None, "MARKER")
    h.review(1, by="@lane-7")
    assert h.tick().startswith("landed #1")


def test_with_signed_verdicts_drive_lands_only_on_an_authenticated_land_verdict(world, monkeypatch):
    """drive.toml's `signed_verdicts`: drive reads the store's labels through a verifier (`StoreLabels`, `research.labelauth`),
    so a land verdict written as another lane counts only once its holder's token authenticates it."""
    from research import labelauth
    from test_labelauth import OTHER, TS, _mint, _verifier

    world.with_store_remote()
    h = Review(world, monkeypatch)
    labels = StoreLabels(world.store, log=lambda s: None, verify=_verifier())
    h.d.reviews, h.d.ready_by = labels.reviews, labels.ready_by
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1, by="proofs")
    out = h.tick()
    assert "no land verdict" in out and not out.startswith("landed #1"), out

    def down(nonce: str) -> str:
        raise labelauth.AuthError("the Cursor identity socket is unavailable")

    with pytest.raises(QueueError, match="isn't written: drive.toml's `signed_verdicts` is on at `main`, and it can't be signed"):
        Q.review(h.q, labelauth.Signer(world.store, down, required=True, now=lambda: TS), labels, 1, "land", None, "proofs")
    assert Q.review(h.q, labelauth.Signer(world.store, _mint(OTHER), now=lambda: TS), labels, 1, "land", None, "proofs") == 0
    assert h.tick().startswith("landed #1"), world.logs


def test_the_lowest_member_waiting_for_a_land_verdict_rings_the_reviewers_and_top_every_30_min_at_its_head(world, monkeypatch):
    """`next`'s lowest member with no land verdict holds every landing above it, and its tick's log rang nobody: drive rings `CI`
    (who staffs the reviewers) and `TOP` `BOTTOM_WAIT_S` after it became the lowest or its ready mark, whichever is later, once
    per `BOTTOM_WAIT_S` at its head; a hold rings nothing, and a new head starts the wait again."""
    from research.drive import BOTTOM_WAIT_S

    h = Review(world, monkeypatch)
    marks: dict[tuple[int, str], float] = {}
    h.q.marked_at = lambda n, hd: (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(marks[(n, hd)]))
                                   if (n, hd) in marks else None)

    def rings() -> list[str]:
        return [p for p in h.posts if "lowest member" in p]

    def wait(s: float) -> str:
        h.clock += s
        return h.tick()

    world.gh.put(1, world.change({"a.txt": "1\n"}))
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    assert wait(BOTTOM_WAIT_S - 60).startswith("nothing reviewed lands: #1") and rings() == []
    wait(60)
    (ring,) = rings()
    assert ring.startswith("@ci @top queue: #1 (owner lane-1)") and "has waited 30 min with no land verdict" in ring, ring
    wait(BOTTOM_WAIT_S - 60)
    assert len(rings()) == 1, "at most once per head per BOTTOM_WAIT_S"
    wait(60)
    assert len(rings()) == 2 and "has waited 60 min" in rings()[-1]

    world.gh.put(1, world.change({"a.txt": "2\n"}))            # a new head waits afresh, from its ready mark
    marks[(1, world.gh.prs[1].head)] = h.clock + 600
    assert wait(0).startswith("nothing reviewed lands: #1") and len(rings()) == 2
    assert wait(BOTTOM_WAIT_S).startswith("nothing reviewed lands") and len(rings()) == 2, "30 min since it became the lowest, 20 since its mark"
    wait(600)
    assert len(rings()) == 3 and world.gh.prs[1].head[:8] in rings()[-1]

    h.review(1, "hold", by="other", ref="reads a stale fixture")
    wait(BOTTOM_WAIT_S)
    assert len(rings()) == 3, "a held member waits on its owner, not on a reviewer"


def test_the_bottom_ring_goes_only_to_drive_tomls_bottom_ring_after_its_bottom_wait_min(world, monkeypatch):
    """drive.toml's `bottom_ring` names every handle `_bottom` rings, and `bottom_wait_min` the minutes before each ring: with
    ci's reviewer pool reviewing `next`'s bottom, drive.toml rings ci alone, and top hears nothing."""
    h = Review(world, monkeypatch)
    h.d.bottom_ring, h.d.bottom_wait_min = ("@ci",), 45
    rung: list[tuple[str, tuple[str, ...]]] = []
    h.d.post = lambda text, to=(): rung.append((text, to)) or True

    def rings() -> list[tuple[str, tuple[str, ...]]]:
        return [(text, to) for text, to in rung if "lowest member" in text]

    def wait(s: float) -> str:
        h.clock += s
        return h.tick()

    world.gh.put(1, world.change({"a.txt": "1\n"}))
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    assert wait(45 * 60 - 60).startswith("nothing reviewed lands: #1") and rings() == [], "it waits 45 min, not the default's 30"
    wait(60)
    ((text, to),) = rings()
    assert to == ("@ci",) and text.startswith("@ci queue: #1 (owner lane-1)") and "@top" not in text, text
    assert "has waited 45 min with no land verdict" in text, text
    wait(45 * 60 - 60)
    assert len(rings()) == 1, "at most once per head per 45 min"
    wait(60)
    assert [to for _, to in rings()] == [("@ci",), ("@ci",)] and "has waited 90 min" in rings()[-1][0]


def test_a_lean_change_lands_on_its_verdict_and_the_after_landing_lean_audit_takes_no_regenerated_record(world, monkeypatch):
    """A PR that changes a guarantee's statement, or a definition it reads, without its lock lands on its verdict as any PR
    does, with no Lean check before it; the after-landing lean-audit fails on the record it leaves stale (`--no-regenerate`)
    rather than pass on regenerated records nothing commits, which would leave `main`'s lock stale and the spec-change DM
    unsent.  Committing them is the lock run's (`regenerate_locks`, `test_drive_locks.py`)."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"other/X.lean": "-- no lock in other/\n"}))
    world.gh.put(2, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    for n in (1, 2):
        h.review(n)
    h.q.sync()
    chain = h.q.chain()
    assert [lean_change(h.d.repo, m.commit) for m in chain] == [False, True]
    assert h.tick().startswith("landed #2"), world.logs
    assert h.launched == [] and [c for c, *_ in h.posted] == [_main(world)], world.logs
    assert all(kw["no_regenerate"] for _, _, kw in h.posted)


#: a Lake package with a lock outside `Security/`, whose Lean runs as code
FLOCK_LOCK = {"verity/protocols/flock/verifier/lean-audit.json": "{}\n"}


def test_every_lean_change_but_proofs_is_one_a_lock_run_follows_whatever_its_title(world, monkeypatch):
    """A "two demos" title exempts no Lean change.  The spec (Lean under `Security/Definitions/`, `Security/Definitions.lean` and
    `Security/`'s Lake files: what the guarantees' statements read), every lock and Lean that runs as code (outside `Security/`,
    in a package with a lock: Flock's verifier) are Lean changes (`queue.lean_change`) in a two-demos PR too, which a lock
    run follows on `checked`: with no regenerated lock commit after them, they leave a record stale for the after-landing
    lean-audit (`--no-regenerate`), the spec's statements or a code package's `verifier.allow`.  `Security/` proofs alone are
    none."""
    _move_main(world, FLOCK_LOCK)
    h = Review(world, monkeypatch)
    two_demos = {1: {"pkg/C.lean": "theorem C : True := trivial\n"},
                 2: {"Security/Definitions/X.lean": "def x := 1\n"},
                 3: {"pkg/lean-audit.json": "{}"},
                 4: {"Security/Proofs/Y.lean": "theorem Y : True := trivial\n"},
                 5: {"Security/Definitions.lean": "import Definitions.X\n"},
                 6: {"Security/lakefile.toml": "name = \"security\"\n"},
                 7: {"Security/lean-toolchain": "leanprover/lean4:v4.0.0\n"},
                 8: {"verity/protocols/flock/verifier/Flock/X.lean": "def x := Nat.gcd 4 6\n"}}
    for n, files in {**two_demos, 9: {"Security/Definitions/Z.lean": "def z := 1\n"}}.items():
        world.gh.put(n, head := world.change(files))
        if n in two_demos:
            world.gh.prs[n] = PR(n, head, f"two demos: change {n}", f"Lane `lane-{n}`.")
        h.review(n)
    h.q.sync()
    chain = h.q.chain()
    assert {m.pr: lean_change(h.d.repo, m.commit) for m in chain} == {n: n != 4 for n in range(1, 10)}


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


def test_a_held_member_goes_to_the_top_and_the_lift_passes_over_it(world, monkeypatch):
    """A held member held every member above it, deep in `next`.  It goes to the top, and the reviewed members on both sides
    of it go below the unreviewed one and land; the unreviewed one and the held one hold only themselves."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    h.review(4)
    assert h.tick().startswith("landed #4"), world.logs
    assert _subject(world, "main^1").startswith("Queue: #2")
    assert [m.pr for m in h.q.chain()] == [1, 3]
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #1"), out
    assert [m.pr for m in h.q.chain()] == [1, 3] and _subject(world, "main").startswith("Queue: #4")


def test_the_lift_keeps_a_stacked_pr_after_the_pr_it_contains(world, monkeypatch):
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


def test_verdict_first_off_keeps_an_unreviewed_member_where_it_is(world, monkeypatch):
    h = Review(world, monkeypatch, verdict_first=False)
    for n, f in ((1, "a.txt"), (2, "b.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 2]


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
    below = _git(world.origin, "rev-parse", f"{main}^1")
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": True, "no_cache": False, "circuit_cache": False,
                                                            "no_regenerate": True, "base": below})]
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


def _rings(h: Harness) -> list[str]:
    """The rings of reds among ``h``'s posts; a revert's post quotes its red too."""
    return [p for p in h.posts if re.search(r" queue: (?:suite|step) \S+ is red on ", p)]


def _owning(world: World, owns: dict[str, list[str]]) -> None:
    """`main`'s registry, each handle owning its paths (`research msg owners`)."""
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane", "owns": paths} for h, paths in owns.items()]}
    _git(world.dev, "push", "-q", "origin", f"{world.change({'.agents/skills/using-slack/registry.json': json.dumps(reg)})}"
                                          ":refs/heads/main")


def _red_landing(world, monkeypatch, tmp_path, before=lambda h: None) -> tuple[Review, str]:
    """#1, #2 and #3 land; #2 breaks suite pkg (owned by @pkg-owner), whose after-landing run `r-post-0` fails once ``before``
    ran, and the tick that reads it names #2."""
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
    before(h)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    return h, tip


def test_an_after_landing_failure_rings_its_culprits_lane_and_the_suites_owner_and_reverts_it_once_its_window_passed_unfixed(
        world, monkeypatch, tmp_path):
    """Without a culprit, PRs keep landing on a `main` whose after-landing runs all fail, each failure going to ci alone.
    Drive names the one landing whose change touches the failing test and rings its lane with the suite's owner in one post
    (neither ci nor top), with the run, the first error line and the time by which to fix it forward.  No tick before
    `FIX_FORWARD_S` has passed since the naming reverts it, each logging why; the first after, with no fix marked, reverts it
    and tells the same two, and the next run checks the revert."""
    h, tip = _red_landing(world, monkeypatch, tmp_path)
    (ring,) = _rings(h)
    assert ring.startswith("@lane-2 @pkg-owner queue: suite pkg is red on `main`") and "(run r-post-0)" in ring, ring
    assert "@ci" not in ring and "@top" not in ring, ring
    assert "failed tests/test_x.py::test_y" in ring and "NameError: name '_once' is not defined" in ring, ring
    assert "Culprit: #2 (its change touches pkg/tests/test_x.py), of the 3 landing(s)" in ring, ring
    due = time.strftime("%H:%M", time.gmtime(h.clock + D.FIX_FORWARD_S))
    assert ring.endswith(f"Fix it forward by {due} UTC: mark a fix (a new head of #2, a PR whose body names #2 or says "
                         "`fixes suite pkg`, or one whose change touches what failed), or drive reverts #2 then."), ring
    r = h.postland()["red"]["suite pkg"]
    assert r["culprit"][0] == 2 and r["named_at"] == h.clock and not r.get("reverted"), r
    assert any(f"suite pkg's culprit #2 isn't reverted: {D.FIX_FORWARD_S // 60} min are left to fix it forward, until {due} UTC"
               in x for x in world.logs), world.logs
    h.clock += D.FIX_FORWARD_S - 1
    h.tick()
    assert _main(world) == tip and not [p for p in h.posts if "reverted" in p] and len(h.posted) == 1, world.logs
    h.clock += 1
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == tip                            # one revert on top of the landing
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@lane-2 @pkg-owner queue: #2 reverted on `main`: suite pkg is red on `main`"), post
    assert post.endswith(f"#2 is its culprit (its change touches pkg/tests/test_x.py), and {D.FIX_FORWARD_S // 60} min after it "
                         "was named no fix of it is marked. Fix and re-mark"), post
    assert "@ci" not in post and "@top" not in post, post
    assert h.posted[-1][0] == main and h.launched == []                                  # the next run checks the new tip
    h.tick()
    assert len([p for p in h.posts if "reverted" in p]) == 1 and len(h.posted) == 2       # reverted and launched once
    h.run_state[f"r-post-{len(h.posted) - 1}"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, f"r-post-{len(h.posted) - 1}", {s: "passed" for s in h.posted[-1][1].split(",")})]
    (h.evidence[0] / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "passed"}]))
    h.tick()
    assert h.postland()["verified"] == main and not h.postland()["red"] and h.postland()["green"]["suite pkg"] == main


def test_a_red_named_by_a_drive_before_named_at_has_its_two_hours_from_the_first_tick_that_reads_it(world, monkeypatch,
                                                                                                     tmp_path):
    """A red record a drive without `named_at` wrote, its culprit unreverted, would be reverted by the first tick of a drive
    that has it, however long its lane had been fixing it: it is taken as named at that tick."""
    h, tip = _red_landing(world, monkeypatch, tmp_path)
    state = h.q.load_state()
    del state["drive"]["postland"]["red"]["suite pkg"]["named_at"]
    h.q.save_state(state)
    h.clock += 10 * D.FIX_FORWARD_S
    h.tick()
    r = h.postland()["red"]["suite pkg"]
    assert r["named_at"] == h.clock and not r.get("reverted") and _main(world) == tip, world.logs
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert h.postland()["red"]["suite pkg"]["reverted"] == h.clock and _git(world.origin, "rev-parse", "main^") == tip


def _two_shards(world, monkeypatch, tmp_path) -> tuple[Review, str, list[str], Path]:
    """#1, #2 (touching suite pkg, owned by @pkg-owner) and #3 land, and once circuit-check's records step passed the
    after-landing run's two shards launch on m1,m3 and m2,m4 of `on`: (drive, its tip, the machines each launch was given,
    shard 1's records, every step passed but lean-suites, shard 2's)."""
    _owning(world, {"lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    h.d.on, h.d.after_landing_shards, machines, launch = "m1,m2,m3,m4", 2, [], h.d.launch_postland
    h.d.launch_postland = lambda repo, commit, machine, steps, **kw: machines.append(machine) or launch(repo, commit, machine,
                                                                                                         steps, **kw)
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}" if rid == "r-post-0" else None
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    assert machines[1:] == ["m1,m3", "m2,m4"] and [kw.get("shard") for _, _, kw in h.posted[1:]] == ["1/2", "2/2"], h.posted
    h.run_state["r-post-1"] = ("done", 0, None, None, "m1")
    return h, _main(world), machines, _steps_run(tmp_path, "r-post-1", {s: "passed" for s in AFTER_LANDING if s != "lean-suites"})


def _suite_failed(tmp_path: Path, rid: str, test: str | None = None, running: bool = False) -> Path:
    """Run ``rid``'s records, suite pkg failed in its lean-suites: naming ``test``, with an error line in its log, or none, its
    log saying only that it was killed; while ``running``, as what failed so far."""
    d = tmp_path / rid
    (d / "suites").mkdir(parents=True)
    if not running:
        (d / "steps.jsonl").write_text('{"step": "lean-suites", "status": "failed", "rc": 1}\n')
    row = {"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": [test] if test else [],
           "log": f"/workspace/research/runs/{rid}/suites/pkg.log"}
    (d / ("suites-lean.failed.json" if running else "suites-lean.json")).write_text(json.dumps([row]))
    (d / "suites" / "pkg.log").write_text("E       assert False\n" if test else "Killed\n")
    return d


@pytest.mark.parametrize("again", ["passed", "nameless"])
@pytest.mark.parametrize("first", ["ended", "running"])
def test_a_shard_whose_failure_names_nothing_runs_again_once_elsewhere_ringing_infra_then_is_red(world, monkeypatch, tmp_path,
                                                                                                 first, again):
    """A failure naming no failing test and no error line (a kill, a crash with no output) may be its machine's, and its
    culprit, named by the suite's declared inputs alone, would be rung to fix what it didn't break.  Its shard runs again once
    instead, stopped if it still runs, on the machines of `on` it didn't run on, as it was launched, and `INFRA` is rung with
    the run, its machine and what failed: nothing is red.  The rerun's pass passes the shard; failing so again, it is red and
    its culprit named, and the shard runs no third time."""
    h, tip, machines, shard1 = _two_shards(world, monkeypatch, tmp_path)
    h.run_state["r-post-2"] = ("running", None, None, "suite pkg", "m2") if first == "running" else ("done", 1, None, None, "m2")
    h.evidence = [shard1, _suite_failed(tmp_path, "r-post-2", running=first == "running")]
    h.tick()
    assert h.posted[3] == h.posted[2] and machines[3] == "m1,m3,m4", (h.posted, machines)
    assert h.cancelled == (["r-post-2"] if first == "running" else [])
    (infra,) = [p for p in h.posts if p.startswith("@infra queue: ")]
    assert infra.startswith(f"@infra queue: suite pkg failed in shard 2/2 of the after-landing run of `main` {tip[:12]} (run "
                            "r-post-2 on m2) naming nothing: no failing test, no error line"), infra
    assert infra.endswith("It runs again, once, on m1,m3,m4: r-post-3; failing so again, it is red"), infra
    assert not _rings(h) and _main(world) == tip and h.postland()["run"]["reran"] == {"2": ["r-post-2"]}
    h.tick()
    assert h.postland()["run"]["groups"] == [["r-post-1"], ["r-post-3"]] and len(h.posted) == 4, "the rerun runs"
    h.run_state["r-post-3"] = ("done", 0 if again == "passed" else 1, None, None, "m3")
    h.evidence = [shard1, _steps_run(tmp_path, "r-post-3", {"lean-suites": "passed"}) if again == "passed"
                  else _suite_failed(tmp_path, "r-post-3")]
    h.tick()
    assert [kw.get("shard") for c, _, kw in h.posted if c == tip] == [None, "1/2", "2/2", "2/2"], h.posted
    assert len([p for p in h.posts if p.startswith("@infra queue: ")]) == 1
    if again == "passed":
        assert h.postland()["verified"] == tip and not _rings(h) and _main(world) == tip, world.logs
        return
    (ring,) = _rings(h)
    assert f"suite pkg is red on `main` {tip[:12]} (run r-post-1 r-post-3): its log has no error line" in ring, ring
    assert "Culprit: #2 (its change touches suite pkg's declared inputs)" in ring, ring
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 2 and _main(world) == tip, world.logs


def test_a_shard_failure_naming_a_failing_test_is_red_at_once_and_runs_its_shard_no_more(world, monkeypatch, tmp_path):
    """Only a failure that names nothing runs its shard again: one naming its failing test is red in the tick that reads it,
    its culprit named, and `INFRA` hears nothing."""
    h, tip, _, shard1 = _two_shards(world, monkeypatch, tmp_path)
    h.run_state["r-post-2"] = ("done", 1, None, None, "m2")
    h.evidence = [shard1, _suite_failed(tmp_path, "r-post-2", test="tests/test_x.py::test_y")]
    h.tick()
    (ring,) = _rings(h)
    assert "suite pkg is red on" in ring and "failed tests/test_x.py::test_y" in ring and "Culprit: #2" in ring, ring
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 2 and not [p for p in h.posts if p.startswith("@infra")]
    assert [kw.get("shard") for c, _, kw in h.posted if c == tip] == [None, "1/2", "2/2"], h.posted


def test_a_red_ratchet_gate_names_the_merge_it_fails_as_the_culprit(world, monkeypatch, tmp_path):
    """The after-landing run's ratchet-gate step runs the vLLM integration's ratchet gate on each landing's merge, and its log
    names no file of the change, only the PR whose merge grows an allowlist: drive takes that PR as the culprit.  The log
    here is the gate's own, run on the merges drive landed."""
    import sys

    p01 = "adapters/vllm/tests/lint/allowlists/p01_core.json"

    def allowlist(*files: str) -> str:
        return json.dumps({"rule": "p01", "entries": [{"file": f, "kind": "k", "symbol": "s", "detail": "d"} for f in files]})

    _git(world.dev, "push", "-q", "origin", f"{world.change({p01: allowlist('a.py', 'b.py')})}:refs/heads/main")
    h = Review(world, monkeypatch)
    base = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"c.py": "C = 1\n", p01: allowlist("a.py", "b.py", "c.py")}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    merge2 = _git(world.origin, "rev-parse", "main^1")
    assert _subject(world, merge2).startswith("Queue: #2")
    _git(world.dev, "fetch", "-q", "origin")
    gate = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / "adapters/vllm/tools/ratchet_gate.py"),
                           base, tip], cwd=world.dev, capture_output=True, text=True, env={**os.environ, "RATCHET_GATE_CACHE": ""})
    assert gate.returncode == 1 and f"  #2 ({merge2[:12]}): failed: grows {p01} 2 -> 3 entries" in gate.stdout, gate
    run = _steps_run(tmp_path, "r-post-0", {s: "passed" for s in EVERY if s != "ratchet-gate"})
    with open(run / "steps.jsonl", "a") as f:
        f.write(json.dumps({"step": "ratchet-gate", "status": "failed", "rc": 1, "log": "ratchet-gate.log"}) + "\n")
    (run / "ratchet-gate.log").write_text(gate.stdout)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = _rings(h)
    assert f"step ratchet-gate is red on `main` {tip[:12]}" in ring and f"`#2 ({merge2[:12]}): failed: grows {p01}" in ring, ring
    assert "Culprit: #2 (the ratchet gate fails its merge), of the 3 landing(s)" in ring, ring
    assert h.postland()["red"]["step ratchet-gate"]["culprit"][:1] == [2]


def test_ratchet_gate_judges_above_the_last_tip_its_gate_passed_at_while_verified_stays_below(world, monkeypatch, tmp_path):
    """While another step stays red, `verified` doesn't move, and a ratchet-gate based there re-judged every landing since in
    each run, the run's long pole.  Its base is the tip of the last run whose ratchet-gate passed (`train.sh --ratchet-base`),
    and the run's `--base` stays the last verified commit: the run on the red's revert, launched in the tick that reverts
    it, judges none of the landings its red run's gate passed."""
    _owning(world, {"lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2"), world.logs
    t1, verified = _main(world), h.postland()["verified"]
    ((_, _, kw),) = h.posted
    assert kw["base"] == verified != t1 and "ratchet_base" not in kw                     # none judged: the verified commit
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2",
                              steps='{"step": "lean-suites", "status": "failed"}\n{"step": "ratchet-gate", "status": "passed"}\n')]
    h.tick()
    pl = h.postland()
    assert pl["verified"] == verified and pl["red"]["suite pkg"]["culprit"][0] == 2, world.logs
    assert pl["ratchet_judged"] == {"tip": t1, "base": verified, "failed": []}
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert _subject(world, "main").startswith('Revert "Queue: #2'), world.logs
    t2, _, kw = h.posted[-1]
    assert t2 == _main(world) != t1 and kw["base"] == verified == h.postland()["verified"] and kw["ratchet_base"] == t1
    assert h.d.landed_since(kw["ratchet_base"], t2) == []                                 # #1 and #2 aren't judged again
    assert any(f"; ratchet-gate judges those above {t1[:12]}" in line for line in world.logs), world.logs


def test_after_a_red_ratchet_gate_whose_culprit_was_reverted_the_gate_judges_above_that_runs_tip(world, monkeypatch, tmp_path):
    """A ratchet-gate that failed naming the merge it fails is a verdict too: once drive reverted that culprit, the next run's
    gate judges above the red run's tip, where the revert lies, so the revert is judged; `verified` stays below."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"c.py": "C = 1\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip, verified = _main(world), h.postland()["verified"]
    merge2 = _git(world.origin, "rev-parse", "main^1")
    assert _subject(world, merge2).startswith("Queue: #2")
    run = _steps_run(tmp_path, "r-post-0", {s: "passed" for s in EVERY if s != "ratchet-gate"})
    with open(run / "steps.jsonl", "a") as f:
        f.write(json.dumps({"step": "ratchet-gate", "status": "failed", "rc": 1, "log": "ratchet-gate.log"}) + "\n")
    (run / "ratchet-gate.log").write_text(f"  #2 ({merge2[:12]}): failed: grows p01.json 2 -> 3 entries\n")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    pl = h.postland()
    assert pl["red"]["step ratchet-gate"]["culprit"][:1] == [2] and pl["verified"] == verified, world.logs
    assert pl["ratchet_judged"] == {"tip": tip, "base": verified, "failed": [merge2]}
    h.clock += D.FIX_FORWARD_S
    h.tick()
    revert = _main(world)
    assert _git(world.origin, "rev-parse", f"{revert}^") == tip
    assert f"This reverts commit {merge2}" in _git(world.origin, "log", "-1", "--format=%B", revert)
    got, _, kw = h.posted[-1]
    assert got == revert and kw["base"] == verified == h.postland()["verified"] and kw["ratchet_base"] == tip, world.logs
    assert _git(world.origin, "rev-list", f"{kw['ratchet_base']}..{revert}") == revert   # the revert lies above the base


def test_ratchet_gates_base_falls_back_to_verified_unless_the_judged_tip_lies_on_the_tips_history(world, monkeypatch):
    """`_ratchet_base`: the judged tip only strictly below the tip and at or above the verified commit; while a change its gate
    failed stands (no revert at or below the tip), the base that gate judged above, unless ratchet-gate is a warn step, whose
    failed changes no revert undoes; else the verified commit: none judged, or a revert or reset that left the judged tip off
    the tip's history."""
    h = Review(world, monkeypatch)
    m0 = _main(world)
    a = world.change({"rb-a.txt": "a\n"}, base=m0)
    b = world.change({"rb-b.txt": "b\n"}, base=a)
    c = world.change({"rb-c.txt": "c\n"}, base=b)
    off = world.change({"rb-o.txt": "o\n"}, base=m0)
    _git(world.dev, "checkout", "-q", "--detach", c)
    _git(world.dev, "revert", "--no-edit", b)
    rev = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{rev}:refs/dev/{rev}")
    _git(h.d.repo, "fetch", "-q", "origin", *(f"refs/dev/{x}" for x in (a, b, c, off, rev)))

    def base(judged: dict | None, tip: str, below: str = m0) -> str | None:
        return h.d._ratchet_base({"ratchet_judged": judged} if judged else {}, tip, below)

    assert base(None, c) == m0                                                            # none judged
    assert base({"tip": b, "base": m0, "failed": []}, c) == b                              # passed at b
    assert base({"tip": off, "base": m0, "failed": []}, c) == m0                           # off the tip's history
    assert base({"tip": c, "base": m0, "failed": []}, c) == m0                             # the tip itself: nothing to judge
    assert base({"tip": a, "base": m0, "failed": []}, c, below=b) == b                     # below the verified commit
    assert base({"tip": c, "base": a, "failed": [b]}, rev) == c                            # its failure reverted at or below
    assert base({"tip": b, "base": a, "failed": [b]}, c) == a                              # b stands: where that gate began
    assert base({"tip": b, "base": off, "failed": [b]}, c) == m0                           # ... unless that is off the history
    h.d.warn_steps = ("ratchet-gate",)
    assert base({"tip": b, "base": a, "failed": [b]}, c) == b                              # a warn step: b stands, judged once
    assert base({"tip": off, "base": m0, "failed": [off]}, c) == m0                        # ... still only on the tip's history
    assert base({"tip": a, "base": m0, "failed": [a]}, c, below=b) == b                    # ... and at or above verified


def _ratchet_red(tmp_path: Path, rid: str, *failed: tuple[int, str]) -> Path:
    """Run ``rid``'s dir: every step passed but ratchet-gate, whose log fails each of ``failed`` (a PR and its merge)."""
    run = _steps_run(tmp_path, rid, {s: "passed" for s in EVERY if s != "ratchet-gate"})
    with open(run / "steps.jsonl", "a") as f:
        f.write(json.dumps({"step": "ratchet-gate", "status": "failed", "rc": 1, "log": "ratchet-gate.log"}) + "\n")
    (run / "ratchet-gate.log").write_text("".join(f"  #{n} ({m[:12]}): failed: grows p01.json 2 -> 3 entries\n" for n, m in failed))
    return run


def _warned(world, monkeypatch, *lanes: str) -> tuple[Review, str, dict[int, str]]:
    """#1, #2 and #3 land under a drive whose warn step is ratchet-gate (drive.toml's `warn_steps`), ``lanes`` in the registry:
    the harness, the tip, and each PR's merge."""
    _registry(world, *lanes)
    h = Review(world, monkeypatch)
    h.d.warn_steps = ("ratchet-gate",)
    for n in (1, 2, 3):
        world.gh.put(n, world.change({f"w{n}.txt": f"{n}\n"}))
        h.review(n)
    assert h.tick().startswith("landed #3")
    merge = {n: _git(world.origin, "rev-parse", "main" + "^1" * (3 - n)) for n in (1, 2, 3)}
    assert all(_subject(world, m).startswith(f"Queue: #{n}") for n, m in merge.items())
    return h, merge[3], merge


def test_a_red_warn_step_rings_its_culprits_lane_once_saying_no_revert_is_coming_and_never_reverts_it(world, monkeypatch, tmp_path):
    """A red ratchet-gate step never reverts (drive.toml's `warn_steps`): drive names the merge its gate fails as the culprit
    and rings its lane once, saying no revert is coming, so the lane fixes it forward in its next PR.  No tick reverts it, at
    `FIX_FORWARD_S` or long after, and the ring that verification is behind rings `CI` and `TOP`, not the lane again."""
    h, tip, merge = _warned(world, monkeypatch, "lane-2")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_ratchet_red(tmp_path, "r-post-0", (2, merge[2]))]
    h.tick()
    (ring,) = _rings(h)
    assert ring.startswith("@lane-2 ") and "Culprit: #2 (the ratchet gate fails its merge), of the 3 landing(s)" in ring, ring
    assert ("No revert is coming: drive never reverts step ratchet-gate's red (drive.toml's `warn_steps`), so #2's lane "
            "fixes it forward in its next PR.") in ring and "drive reverts" not in ring, ring
    for later in (D.FIX_FORWARD_S, 10 * D.FIX_FORWARD_S):
        h.clock += later
        h.tick()
        assert _main(world) == tip and _reverts(world, tip) == 0, world.logs
    assert h.postland()["red"]["step ratchet-gate"]["culprit"][:1] == [2]
    assert not [line for line in world.logs if "step ratchet-gate's culprit" in line], world.logs
    (behind,) = [p for p in h.posts if "behind its tip" in p]
    assert behind.startswith("@ci @top queue: ") and behind.endswith("red: step ratchet-gate (culprit #2)"), behind
    assert [p for p in h.posts if "@lane-2" in p] == [ring], h.posts


def test_a_red_warn_step_names_each_merge_its_gate_fails_a_culprit_and_rings_each_lane_in_one_post(world, monkeypatch, tmp_path):
    """Each merge a warn step's gate fails is a culprit of its own failure, not one of several candidates: one post rings each
    culprit's lane with each failure line, once per culprit, and none is reverted."""
    h, tip, merge = _warned(world, monkeypatch, "lane-1", "lane-3")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_ratchet_red(tmp_path, "r-post-0", (1, merge[1]), (3, merge[3]))]
    h.tick()
    (ring,) = _rings(h)
    assert ring.startswith("@lane-1 @lane-3 ") and "Culprits: #1, #3 (the ratchet gate fails each one's merge)" in ring, ring
    assert all(f"`#{n} ({merge[n][:12]}): failed: grows p01.json 2 -> 3 entries`" in ring for n in (1, 3)), ring
    assert "so each culprit's lane fixes it forward in its next PR." in ring and "@ci" not in ring, ring
    r = h.postland()["red"]["step ratchet-gate"]
    assert [c[0] for c in r["culprits"]] == [1, 3] and r["culprit"][0] == 1 and not r.get("candidates")
    assert {f"red:step ratchet-gate:{merge[n]}" for n in (1, 3)} <= set(h.q.load_state()["drive"]["told"])
    h.clock += 2 * D.FIX_FORWARD_S
    h.tick()
    assert _main(world) == tip and _reverts(world, tip) == 0 and len(_rings(h)) == 1, world.logs


def test_a_warn_steps_next_run_judges_only_the_merges_above_the_tip_it_judged_with_its_failed_merges_standing(world, monkeypatch, tmp_path):
    """While ratchet-gate is a warn step, the merge its gate failed stays and the next run's gate judges only the merges above
    the red run's tip (`--ratchet-base`), so it doesn't judge or ring that merge again; once that gate passes, the red ends."""
    h, t1, merge = _warned(world, monkeypatch, "lane-2")
    verified = h.postland()["verified"]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_ratchet_red(tmp_path, "r-post-0", (2, merge[2]))]
    h.tick()
    assert h.postland()["ratchet_judged"] == {"tip": t1, "base": verified, "failed": [merge[2]]}
    world.gh.put(4, world.change({"w4.txt": "4\n"}))
    h.review(4)
    h.clock += D.FIX_FORWARD_S
    assert h.tick().startswith("landed #4"), world.logs
    t2, _, kw = h.posted[-1]
    assert t2 == _main(world) != t1 and _reverts(world, verified) == 0, world.logs
    assert kw["base"] == verified == h.postland()["verified"] and kw["ratchet_base"] == t1
    assert [m.pr for m in h.d.landed_since(kw["ratchet_base"], t2)] == [4]                # #2 isn't judged again
    h.run_state["r-post-1"] = ("done", 0, None)
    h.evidence = [_steps_run(tmp_path, "r-post-1", {s: "passed" for s in AFTER_LANDING})]
    h.tick()
    pl = h.postland()
    assert "step ratchet-gate" not in pl.get("red", {}) and pl["verified"] == t2, world.logs
    assert pl["ratchet_judged"] == {"tip": t2, "base": t1, "failed": []} and len(_rings(h)) == 1


def test_a_red_off_the_warn_steps_still_reverts_its_culprit_once_two_hours_pass_unfixed(world, monkeypatch, tmp_path):
    """Only a warn step's red never reverts: in a run whose ratchet-gate and a suite both fail, the suite's culprit is rung to
    fix it forward by `FIX_FORWARD_S`, or drive reverts it then, and is reverted then; ratchet-gate's stays."""
    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch)
    h.d.warn_steps = ("ratchet-gate",)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    world.gh.put(2, world.change({"c.py": "C = 1\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2")
    tip = merge2 = _main(world)
    assert _subject(world, merge2).startswith("Queue: #2")
    run = _failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2",
                      steps='{"step": "lean-suites", "status": "failed"}\n'
                            '{"step": "ratchet-gate", "status": "failed", "rc": 1, "log": "ratchet-gate.log"}\n')
    (run / "ratchet-gate.log").write_text(f"  #2 ({merge2[:12]}): failed: grows p01.json 2 -> 3 entries\n")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    suite, step = sorted(_rings(h), key=lambda p: "step ratchet-gate" in p)
    assert "suite pkg is red" in suite and "Culprit: #1" in suite and "or drive reverts #1 then." in suite, suite
    assert "Culprit: #2" in step and "No revert is coming" in step, step
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert _subject(world, "main").startswith('Revert "Queue: #1') and _reverts(world, tip) == 1, world.logs
    red = h.postland()["red"]
    assert red["suite pkg"]["reverted"] and not red["step ratchet-gate"].get("reverted")


@pytest.mark.parametrize("fix", ["names-the-culprit", "fixes-suite", "touches-the-failure", "new-head"])
def test_a_marked_fix_holds_its_reds_revert_while_it_stays_marked_and_once_ejected_past_two_hours_the_culprit_is_reverted(
        world, monkeypatch, tmp_path, fix):
    """A fix (a PR whose body names the culprit or says `fixes suite NAME`, or whose change touches what the failure named, or
    a new head of the culprit) marked ready at its head holds the culprit's revert past `FIX_FORWARD_S`, each tick logging
    which fix holds it; once it is ejected (its mark at that head no longer `true`), the next tick reverts the culprit.  Each
    body case changes an unrelated file, so only its body could count."""
    n = 2 if fix == "new-head" else 9

    def mark(h):
        if fix == "new-head":
            world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
        elif fix == "touches-the-failure":
            world.gh.put(9, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body="Lane `lane-9`.")
        else:
            world.gh.put(9, world.change({"elsewhere.txt": "9\n"}),
                         body="Lane `lane-9`. " + ("FIXES suite pkg: test_y." if fix == "fixes-suite" else "Fixes #2's test."))

    h, tip = _red_landing(world, monkeypatch, tmp_path, before=mark)
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 2
    for _ in range(2):
        h.clock += D.FIX_FORWARD_S
        h.tick()
        assert _main(world) == tip and not h.postland()["red"]["suite pkg"].get("reverted"), world.logs
    assert any(f"suite pkg's culprit #2 isn't reverted: #{n} at {world.gh.prs[n].head[:8]}" in x
               and x.endswith("which fixes it, is marked ready at its head") for x in world.logs), world.logs
    world.gh.ready.discard((n, world.gh.prs[n].head))
    h.tick()
    assert h.postland()["red"]["suite pkg"]["reverted"] == h.clock, world.logs
    assert _git(world.origin, "rev-parse", "main^") == tip and any("#2 reverted on `main`" in p for p in h.posts), h.posts


def test_a_fix_that_landed_holds_the_revert_until_a_run_holding_it_is_red_again(world, monkeypatch, tmp_path):
    """#9, `fixes suite pkg` in its body, lands while the run that finds #2 red runs: the next run checks it, and while that
    runs #9 holds #2's revert past `FIX_FORWARD_S`.  That run red in suite pkg again, #9 fixed nothing: #2 is reverted in the
    tick that reads it, on top of #9."""
    def land_fix(h):
        world.gh.put(9, world.change({"elsewhere.txt": "9\n"}), body="Lane `lane-9`. Fixes suite pkg: test_y.")
        h.review(9)
        assert h.tick().startswith("landed #9"), world.logs
        fixed.append(_main(world))

    fixed: list[str] = []
    h, tip = _red_landing(world, monkeypatch, tmp_path, before=land_fix)
    assert [c for c, *_ in h.posted] == [tip, fixed[0]], world.logs                 # the next run checks #9
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert _main(world) == fixed[0] and not h.postland()["red"]["suite pkg"].get("reverted"), world.logs
    h.run_state["r-post-1"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path / "again", "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == fixed[0] and any("#2 reverted on `main`" in p for p in h.posts), h.posts
    assert h.postland()["red"]["suite pkg"]["seen"] == fixed[0] and h.posted[-1][0] == main, world.logs


def test_an_unrelated_pr_that_names_the_red_suite_is_no_fix_of_it(world, monkeypatch, tmp_path):
    """A fix of a red (`Driver._fixes`) holds the after-landing run for it (`_fix_wait`) and its culprit's revert
    (`_revert_reds`).  Counting any ready PR whose body names the suite would let a suite called `research` be held for good
    by whatever PR mentions research: neither the name nor `suite research` without `fixes` is a fix, so the culprit is
    reverted once `FIX_FORWARD_S` passed."""
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
    world.gh.put(9, head := world.change({"docs/notes.txt": "9\n"}),
                 body="Lane `lane-9`. research: the research store's docs, as suite research reads them.")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = _rings(h)
    assert "suite research is red" in ring and "Culprit: #2" in ring, ring
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert _git(world.origin, "rev-parse", "main^") == tip and any("#2 reverted on `main`" in p for p in h.posts), h.posts
    r, own = h.postland()["red"]["suite research"], lambda: {"docs/notes.txt"}
    assert h.d._fixes("suite research", r, world.gh.prs[9], own) is None
    assert h.d._fixes("suite research", r, Q.PR(9, head, body="Lane `lane-9`. Moves suite research's docs."), own) is None
    for body in ("Lane `lane-9`. Fixes suite `research`.", "Lane `lane-9`. Fixes #2's test."):
        assert h.d._fixes("suite research", r, Q.PR(9, head, body=body), own) == f"#9 at {head[:8]}", body


def _one_red(world, monkeypatch, tmp_path, before=lambda h: None) -> tuple[Review, str]:
    """#1 breaks suite pkg and lands; its after-landing run `r-post-0` fails once ``before`` ran, the tick that reads it
    names #1 its culprit, and the first tick `FIX_FORWARD_S` later reverts it."""
    _registry(world, "lane-1")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    before(h)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 1
    h.clock += D.FIX_FORWARD_S
    h.tick()
    return h, tip


def _reverts(world: World, since: str) -> int:
    return len(_git(world.origin, "log", "--format=%H", "--grep=^This reverts commit", f"{since}..main").split())


def test_a_culprits_revert_that_conflicts_leaves_main_and_tells_its_owner_to_revert_by_hand(world, monkeypatch, tmp_path):
    moved: list[str] = []

    def edit(h):
        _move_main(world, {"pkg/tests/test_x.py": "def test_y():\n    assert False  # edited on main\n"})
        moved.append(_main(world))

    h, tip = _one_red(world, monkeypatch, tmp_path, before=edit)
    assert _main(world) == moved[0]
    (post,) = [p for p in h.posts if "must be reverted by hand" in p]
    assert post.startswith("@ci @lane-1 queue: #1 must be reverted by hand") and "the revert didn't apply" in post, post


def test_a_culprit_reverted_by_hand_is_not_reverted_again_nor_told_as_a_hand_revert(world, monkeypatch, tmp_path):
    fixed: list[str] = []

    def revert(h):
        merge_commit = _git(world.origin, "log", "--format=%H", "--grep=^PR: #1$", "main").split()[0]
        _git(world.dev, "fetch", "-q", "origin")
        _git(world.dev, "checkout", "-q", "--detach", "origin/main")
        _git(world.dev, "revert", "--no-edit", "-m", "1", merge_commit)
        _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
        fixed.append(_main(world))

    h, tip = _one_red(world, monkeypatch, tmp_path, before=revert)
    assert _main(world) == fixed[0] and _reverts(world, tip) == 1
    assert h.postland()["red"]["suite pkg"]["reverted"] == "undone"
    assert not any("by hand" in p or "reverted on `main`" in p for p in h.posts), h.posts


def test_a_revert_whose_push_loses_a_race_is_retried_on_the_next_tick(world, monkeypatch, tmp_path):
    raced: list[str] = []

    def race(h):
        real = h.q._remote

        def remote(repo, *args, **kw):
            if "main:refs/heads/main" in args and not raced:
                _move_main(world, {"c.txt": "landed in between\n"})
                raced.append(_main(world))
            return real(repo, *args, **kw)

        monkeypatch.setattr(h.q, "_remote", remote)

    h, tip = _one_red(world, monkeypatch, tmp_path, before=race)
    assert raced and _main(world) == raced[0] and not any("reverted" in p or "by hand" in p for p in h.posts), h.posts
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == raced[0] and _reverts(world, tip) == 1
    assert _git(world.origin, "show", f"{main}:c.txt") == "landed in between"
    assert any(p.startswith("@lane-1 queue: #1 reverted on `main`") for p in h.posts), h.posts


def test_a_drive_killed_after_a_revert_pushed_does_not_revert_again_on_restart(world, monkeypatch, tmp_path):
    got: list[Review] = []

    def kill(h):
        post = h.d.post

        def killed(text, to=()):
            if "reverted on" in text:
                raise KeyboardInterrupt("the drive is killed")
            return post(text, to)

        h.d.post = killed
        got.append(h)

    with pytest.raises(KeyboardInterrupt):
        _one_red(world, monkeypatch, tmp_path, before=kill)
    (h,) = got
    tip = h.posted[0][0]
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
    tip = _main(world)
    run = _failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2", steps="")
    (run / "suites-lean.failed.json").write_text((run / "suites-lean.json").read_text())
    (run / "suites-lean.json").unlink()
    steps = {"lean-build": "passed", "lean-agreement": "stopped"}
    (run / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    h.run_state["r-post-0"] = ("running", None, None, "suite pkg (tests/test_x.py::test_y)")
    h.evidence = [run]
    h.tick()
    assert h.cancelled == [] and h.postland()["run"]["ids"] == ["r-post-0"]
    (ring,) = _rings(h)
    assert "suite pkg is red" in ring and "Culprit: #1" in ring and "Not run (a cancel stopped them): lean-agreement" in ring
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert h.cancelled == [] and len(_rings(h)) == 1             # rung once
    assert _git(world.origin, "rev-parse", "main^") == tip                                      # reverted as the run runs on
    steps |= {s: "passed" for s in AFTER_LANDING if s not in TAIL and s not in steps}
    steps["lean-suites"] = "failed"
    (run / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    h.tick()
    assert h.cancelled == ["r-post-0"] and h.postland()["green"]["step lean-audit"] == tip
    assert h.postland()["run"]["ids"] == ["r-post-1"] and h.posted[-1][0] == _main(world) != tip  # the next run checks the revert
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
    (ring,) = _rings(h)
    assert ring.startswith("@lane-1 @pkg-owner queue: suite pkg is red"), ring
    assert "read outside its declared inputs: data/x.json" in ring and "Culprit: #1" in ring, ring


def test_main_far_behind_its_last_verified_commit_for_a_red_rings_who_acts_on_it_every_hour(world, monkeypatch, tmp_path):
    """A `main` many hours without a verified commit must reach someone.  Once its oldest unverified landing is
    `BEHIND_S` old, drive rings who acts on what stands red there unreverted, here the owner of a suite no landing touches
    (not ci and top, which hear only while nothing unreverted explains it), again every `BEHIND_EVERY_S` while it stays
    behind."""
    from research.drive import BEHIND_EVERY_S, BEHIND_S

    _owning(world, {"pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert h.postland()["red"]["suite pkg"]["culprit"] is None
    h.clock += BEHIND_S - 10
    h.tick()
    assert not [p for p in h.posts if "behind its tip" in p] and _main(world) == tip
    h.clock += 20
    h.tick()
    (alarm,) = [p for p in h.posts if "behind its tip" in p]
    assert alarm.startswith("@pkg-owner queue: `main`'s last verified commit") and alarm.endswith("red: suite pkg"), alarm
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
    assert alarm.startswith("@ci @top queue: ") and "its run r-post-0" in alarm and "red: nothing named red" in alarm, alarm


def test_an_after_landing_failure_several_prs_touch_rings_each_ones_lane_with_ci_and_top_and_reverts_none(world, monkeypatch,
                                                                                                           tmp_path):
    """Reverting every landing a failure's files reach takes innocent PRs with the culprit, and bisecting them held a run per
    landing.  A failure several landings touch names no culprit: each one's lane is rung with `CI` and `TOP` in one post,
    naming them all, and none is reverted, however long the red stands."""
    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch)
    world.gh.put(1, h1 := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}, base=h1))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    (ring,) = _rings(h)
    to = ring.split(" queue: ")[0].split()
    assert to[:2] == ["@lane-1", "@lane-2"] and {"@ci", "@top"} <= set(to), ring
    assert "#1 #2 each touch it (3 landing(s) since `main`" in ring, ring
    assert "so drive names no culprit and reverts none of them: find which broke it, and fix it forward." in ring, ring
    r = h.postland()["red"]["suite pkg"]
    assert r["culprit"] is None and [c[0] for c in r["candidates"]] == [1, 2] and "bisect" not in h.postland(), r
    assert [c for c, *_ in h.posted] == [tip], "no run of a landing's own commit"
    h.clock += 2 * D.FIX_FORWARD_S
    h.tick()
    assert _main(world) == tip and not [p for p in h.posts if "reverted" in p] and len(_rings(h)) == 1, world.logs


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


def test_the_owner_of_an_unmarked_pr_marked_ones_wait_for_is_told_once_with_theirs_until_its_head_changes(world, monkeypatch):
    """A marked PR can wait on another's unmarked commits, and that one's owner is told, in one post with every marked PR it takes
    out and their owners.  It is told again when its head moves, not when a PR it takes out moves, and not at all for a waiting PR
    that isn't marked; nothing once it's admitted."""
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
        return [p for p in h.posts if "isn't marked ready" in p]

    h.tick()
    five = world.gh.prs[5].head
    assert told() == [
        f"@ci @lane-5 @lane-7 @lane-9 queue: #5 at {five[:8]} isn't marked ready. It takes out #7 (marked by marker7; at #5's "
        f"older head {old[:12]}), #9 (marked by lane-9), which contain it. Fix: mark #5 ready at {five[:8]}, or withdraw it (`research queue "
        f"eject 5 --reason ...`) and they carry it"], told()
    h.tick()
    assert len(told()) == 1
    world.gh.put(7, world.change({"k.txt": "kernels 2\n"}, base=world.gh.prs[7].head))
    h.tick()
    assert len(told()) == 1                                     # the same root, PRs and reason
    world.gh.put(5, world.change({"x.txt": "x = 3\n"}, base=five), ready=False)
    h.tick()
    # #9 holds #5's older head now; it only shares `old` with #8, which waits for nothing of #9's or #7's (`Queue.entries`)
    new = world.gh.prs[5].head
    assert told()[1:] == [
        f"@ci @lane-5 @lane-7 @lane-9 queue: #5 at {new[:8]} isn't marked ready. It takes out #7 (marked by lane-7; at #5's "
        f"older head {old[:12]}), #9 (marked by lane-9; at #5's older head {five[:12]}), which contain it. Fix: mark #5 ready at {new[:8]}, or "
        f"withdraw it (`research queue eject 5 --reason ...`) and they carry it"], told()
    world.gh.ready.add((5, new))
    h.tick()
    assert len(told()) == 2 and h.q.status()[7].state != "waiting"


def test_a_marked_pr_left_out_for_an_ejected_pr_it_holds_an_older_head_of_is_told_once_with_the_fix(world, monkeypatch):
    """#7 branched from #6, which moved on and was ejected at its new head: #7 holds #6's commits at an older head, so the sync
    leaves it off `next`.  Both owners hear once, in #6's post with the fix, and nothing while that stays."""
    _registry(world, "lane-6", "lane-7")
    h = Review(world, monkeypatch)
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(6, old, ready=False)
    h.tick()                                                                    # drive sees #6 at `old`
    world.gh.put(6, new := world.change({"x.txt": "x = 2\n"}, base=old))
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))
    h.q.eject(6, "breaks circuits", "ci")
    world.gh.ready.discard((6, new))

    def told() -> list[str]:
        return [p for p in h.posts if "isn't admitted" in p]

    h.tick()
    assert told() == [
        f"@ci @lane-6 @lane-7 queue: #6 at {new[:8]} isn't admitted (ejected at {new[:8]}: breaks circuits (a ready mark "
        f"re-admits it)). It takes out #7 (at #6's older head {old[:12]}), which contains it. Fix: mark #6 ready at {new[:8]}, "
        f"drop #6's commits from #7, or ask @top to close #6"], told()
    assert h.q.status()[7].state == "waiting"
    h.tick()
    assert len(told()) == 1


def test_the_notice_names_the_older_head_a_pr_carries_never_a_merge_base_on_main(world, monkeypatch):
    """#7 merged #5's branch at `q1`, then both merged `main` once it moved: their merge bases are `q1` and `main`'s commit, and
    `git merge-base` may name either.  #5 hasn't landed, so #7 still waits for it, and the notice names `q1`, the head of #5's
    whose commits #7 carries, never the commit of `main`."""
    from test_queue import _merge

    _registry(world, "lane-5", "lane-7")
    q1 = world.change({"x.txt": "x = 1\n"})
    seven = _merge(world, world.change({"k.txt": "kernels\n"}), q1)
    _move_main(world, {"m.txt": "landed\n"})
    main = _main(world)
    five, seven = _merge(world, q1, main), _merge(world, seven, main)
    assert main in _git(world.dev, "merge-base", "--all", seven, five).split()
    h = Review(world, monkeypatch)
    world.gh.put(5, five, ready=False)
    world.gh.put(7, seven)
    h.tick()
    (told,) = [p for p in h.posts if "isn't marked ready" in p]
    assert f"#5 at {five[:8]} isn't marked ready. It takes out #7 (marked by lane-7; at #5's older head {q1[:12]})" in told, told
    assert main[:12] not in told and h.q.status()[7].state == "waiting"


def _detached(h: Review, tmp_path, locks: bool = False, post: bool = False) -> list[str]:
    """``h``'s lock runs (``locks``) and after-landing runs (``post``) launched as train.sh is: detached, the launch returning at
    once with no run ids (`drive.detach`'s record), each launch's commit in the list returned.  `h.printed[launch["dir"]]` is
    what its train.sh has printed (`drive.launched`): unset while it runs, its run ids, or [] once it exited without them."""
    starts, h.printed = [], {}

    def start(repo, commit, machine, *a, **kw):
        starts.append(commit)
        return {"dir": str(tmp_path / f"launch-{len(starts)}"), "pid": 0}

    if locks:
        h.d.launch_locks = start
    if post:
        h.d.launch_postland = start
    h.d.launched = lambda launch: h.printed.get(launch["dir"], (None, "train: abc on m1; log x"))
    return starts


def test_the_after_landing_run_launches_detached_and_one_that_registered_no_run_is_told_and_launched_again(
        world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, baseline=False)
    starts = _detached(h, tmp_path, post=True)
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
    assert "postland" not in h.d.q.load_state()["drive"]["launch_failed"]     # its launch registered a run: the count ends


def test_an_after_landing_launch_that_exits_1_rings_ci_and_top_once_an_hour_and_the_behind_ring_names_it(
        world, monkeypatch, tmp_path):
    """A train.sh that refuses before any step exits 1 at every launch, and drive launches the run again each tick.  Told once to
    ci alone, nobody acted for hours while the behind-ring said the run was `(launching)` for 0.0 h.  Each launch that
    registered no run rings ci and top at once with its reason's last line, the same reason at the same tip again only an hour
    later, and the behind-ring counts the failures."""
    from datetime import UTC, datetime

    from research.drive import BEHIND_S, LAUNCH_RING_EVERY_S

    h = Review(world, monkeypatch)
    starts = _detached(h, tmp_path, post=True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    main, landed = _main(world), h.clock
    assert starts == [main]
    refused = "train.sh: refused: its settings check failed (exit 1)"
    h.d.launched = lambda launch: ([], f"train: checking settings\n{refused}")

    def rings() -> list[str]:
        return [p for p in h.posts if "registered no run" in p]

    h.clock += 60
    h.tick()
    (ring,) = rings()
    since = f"{datetime.fromtimestamp(h.clock, UTC):%H:%M}Z"
    assert ring.startswith(f"@ci @top queue: train.sh {main[:12]} m1 --deferred "), ring
    assert ring.endswith(f"registered no run 1 time since {since} (log {tmp_path / 'launch-1'}/log): {refused}"), ring
    assert "checking settings" not in ring and starts == [main, main]
    h.clock += 600
    h.tick()                                                              # the same failure within the hour: no ring
    assert len(rings()) == 1 and starts == [main] * 3
    assert h.d.q.load_state()["drive"]["launch_failed"]["postland"]["n"] == 2
    h.clock += LAUNCH_RING_EVERY_S - 600
    h.tick()
    assert len(rings()) == 2 and f"registered no run 3 times since {since}" in rings()[1], rings()
    h.clock = landed + BEHIND_S + 60
    h.tick()
    (alarm,) = [p for p in h.posts if "behind its tip" in p]
    assert alarm.startswith("@ci @top queue: `main`'s last verified commit"), alarm
    assert f"its launch registered no run 4 times since {since}: {refused}; red: nothing named red" in alarm, alarm
    assert "(launching)" not in alarm and "has run" not in alarm, alarm


def test_a_shard_of_the_after_landing_run_that_prints_no_run_ids_within_the_bound_registered_no_run(world, monkeypatch, tmp_path):
    """A shard's launch record holds no launch time of its own: without the run's, a train.sh that hung before printing its run
    ids would leave the run launching for good, and nothing would land."""
    from research.drive import LAUNCH_LOST_S

    h = Review(world, monkeypatch, baseline=False)
    h.d.after_landing_shards = 2
    starts = _detached(h, tmp_path, post=True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    main = _main(world)
    h.tick()
    h.printed[str(tmp_path / "launch-1")] = ([], "train: no run launched")   # circuit-check's records step: the shards build them
    h.tick()
    shards = h.postland()["run"]["shards"]
    assert starts == [main] * 3 and all("at" not in s for s in shards), world.logs
    h.printed[shards[0]["launch"]["dir"]] = (["r-one"], "RUN r-one")
    h.clock += LAUNCH_LOST_S                                              # within the bound the other still launches
    h.tick()
    assert starts == [main] * 3 and not [p for p in h.posts if "--shard 2/2 registered no run" in p], world.logs
    h.clock += 1
    h.tick()
    (post,) = [p for p in h.posts if "--shard 2/2 registered no run" in p]
    assert "no run ids 60 min after its launch" in post, post
    assert h.cancelled == ["r-one"] and starts == [main] * 5, world.logs


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


def test_an_eject_off_drive_leaves_next_as_it_is_and_records_the_ejection(world, monkeypatch, capsys):
    """`research queue eject N` of a PR not on `next`, run off drive, would rebuild every member of `next` ("kept 0, merged
    N") and force-push it, which only drive's tick does: it records the ejection, and drive's next tick syncs."""
    import research.drive as D
    from research.cli import main as research_main

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(9, head := world.change({"c.txt": "9\n"}), ready=False)    # open, never on `next`
    h.tick()
    one = h.q.chain()[0].commit
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
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.q.sync()                                                          # nor any other caller
    assert refs() == before
    monkeypatch.setenv("RESEARCH_DRIVE", "1")
    h.tick()
    assert h.q.chain()[0].commit == one and h.cancelled == [], world.logs


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


def test_the_low_risk_lanes_rebuild_merges_with_each_tips_drivers_not_its_checkouts(world):
    """The low-risk lane rebuilds `next` on the landing branch with `Driver._clean_merge`, which ran `git merge-tree` with the
    drivers of the clone's checkout: once that checkout held another `merge.py` than the tip it merged onto, every lock it
    merged came out in that driver's format.  It is the sync's merge (`Queue._recorded`), which runs the tip's own drivers;
    any record the sync made of it is dropped, so the merge is made again under the checkout's broken `merge.py`."""
    import types

    caps = "tests/test_lean_packages.py"

    def counts(sec: int, props: int) -> str:
        return f'UNWITNESSED = {{"Security": {sec}, "Security/Properties": {props}}}\n'

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    _move_main(world, {caps: counts(354, 184), ".gitattributes": f"{caps} merge=lean-caps\n",
                       **{f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}})
    two, three = world.change({caps: counts(355, 184)}), world.change({caps: counts(354, 185)})
    q = world.queue()
    q.sync()
    (q.workdir / "tools/verity/lean/merge.py").write_text("import sys\nsys.exit('merge.py: error: unrecognized arguments: --caps')\n")
    q._merge_records = {}
    tree, why = D.Driver._clean_merge(types.SimpleNamespace(repo=q.workdir, q=q), two, three)
    assert tree is not None and _git(q.workdir, "show", f"{tree}:{caps}") == counts(355, 185).strip(), why


def test_a_merge_of_main_into_a_verdicted_head_lands_on_its_verdict_and_a_real_change_waits_for_its_own(world, monkeypatch):
    """Merging `main` into a PR costs no new mark and no new review: the new head's own change (`Queue.own_change`) is the
    verdicted head's, so drive lands it on that head's mark and verdict, and its log names that head.  A PR whose new head
    changes its own diff, marked again by its owner, waits for a land verdict at that head."""
    from test_queue import _merge

    world.with_store_remote()
    old = {1: world.change({"a.txt": "1\n"}), 2: world.change({"b.txt": "2\n"})}
    _move_main(world, {"c.txt": "moved\n"})
    labels = StoreLabels(world.store, log=world.logs.append)
    h = Review(world, monkeypatch, ready=labels.ready, marked_at=labels.marked_at, prefetch=labels.prefetch)
    labels.change, h.d.reviews, h.d.ready_by = h.q.own_change, labels.reviews, labels.ready_by
    for n, head in old.items():
        world.gh.put(n, head, ready=False)
        world.store.add_label(label_target(n, head), vocab.READY, True, f"lane-{n}")
        h.review(n)
    world.gh.put(1, one := _merge(world, old[1], _main(world)), ready=False)
    world.gh.put(2, two := world.change({"b.txt": "2 fixed\n"}, base=old[2]), ready=False)
    world.store.add_label(label_target(2, two), vocab.READY, True, "lane-2")
    assert h.tick().startswith("landed #1"), world.logs
    assert f"Head: {one}" in _git(world.origin, "log", "-1", "--format=%B", "main")
    assert any(s.startswith(f"research queue: #1 at {one[:8]} takes the `reviewed` labels at {old[1][:8]}") for s in world.logs)
    out = h.tick()
    assert out.startswith(f"nothing reviewed lands: #2 at {two[:8]}") and "no land verdict" in out, out


def test_a_refusing_warn_gate_lands_the_pr_and_tells_its_owner_once_a_head_while_a_landing_gate_still_refuses(world,
                                                                                                             monkeypatch):
    """A warn gate (drive.toml's `warn_gates`) judges each PR's merge onto `next` as a landing gate does, and keeps no PR out:
    drive lands the PR and tells its owner and ci what the gate said, once a head, for the owner to fix in their next PR.  A
    landing gate's refusal still keeps its PR off `next`."""
    from test_queue import GATE

    _registry(world, "lane-1", "lane-2")
    _move_main(world, {Q.DRIVE_FLAGS: 'landing_gates = ["gates/g.py"]\nwarn_gates = ["gates/w.py"]\n', "gates/g.py": GATE,
                       "gates/banned.txt": "bad.txt\n", "gates/w.py": GATE.replace("gates/banned.txt", "gates/warned.txt"),
                       "gates/warned.txt": "warn.txt\n"})
    main = _main(world)
    h = Review(world, monkeypatch)
    world.gh.put(1, h1 := world.change({"warn.txt": "1\n"}))
    world.gh.put(2, world.change({"bad.txt": "2\n"}))
    h.review(2)
    h.tick()
    assert world.on_next(h.q) == [1], world.logs
    assert h.q.load_state()["conflicts"]["2"]["why"].startswith(f"the landing gate gates/g.py at `main` {main[:12]} "), world.logs

    def warned() -> list[str]:
        return [p for p in h.posts if "lands all the same" in p]

    assert warned() == [f"@lane-1 @ci queue: #1 at {h1[:8]} lands all the same, as a warn gate keeps no PR out: the warn gate "
                        f"gates/w.py at `main` {main[:12]} on its merge onto `next`: touches warn.txt; delete warn.txt (`python3 "
                        "gates/w.py origin/main HEAD` on the branch merged with `origin/next` runs it). Fix it in your next PR"]
    h.tick()
    assert len(warned()) == 1, "once a head"
    world.gh.put(1, h1 := world.change({"warn.txt": "1, again\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    assert len(warned()) == 2 and warned()[1].startswith(f"@lane-1 @ci queue: #1 at {h1[:8]} lands all the same"), h.posts
    assert world.on_next(h.q) == [] and _subject(world, "main").startswith("Queue: #1")
