"""`research.vqueue_slim`: the run beside each candidate of the verified queue of what graph2 doesn't check, launched and read
through `research.landing`, on its fakes (`test_landing`): `Harness`'s launches, run states, evidence and clock, `_slim`'s
ended run, and a lint log whose failure names its paths.  Pending while it launches or runs, infra when it registered no run,
never ran, was lost or was killed, fail on a step on the gate it failed, pass on every one it passed; each failure keyed by
the content the step reads at the candidate and at its base, so `vqueue.decide` takes one its base fails too as inherited;
`combine`'s table; and the check that gates: the after-landing run as shards, judged on the steps `main` waits for at the
base, each failure by the tests it names."""
from __future__ import annotations

import json
import shutil
import signal
from dataclasses import replace
from pathlib import Path

import pytest

from research import merge
from research.landing import (ACTIVE, AFTER_LANDING, AUTO_PLAN, LAUNCH_LOST_S, LOST_AFTER_S, NOT_STARTED, PLAN_WAIT_S,
                              SIGTERM_RC, SLIM_STEPS, shard_machines, step_paths, step_records)
from research.vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, STATES, Advance, Candidate, Drop, Eject, Failed, Ready,
                             Red, Snapshot, Verdict, _inherited, decide, meets)
from research.vqueue_slim import (BUILD, CHECK, LEAN_INPUTS, OWN_TRAIN, WHOLE, Read, _json, _text, abandon, audit_built,
                                  audit_failures, audited, audits, based, combine, failures, judge, key, lean_inputs, lean_reads,
                                  read_step, read_suite, reads, shard_count, start, step_test, suite_kind, traced_reads, unnamed)
from test_policy import LIVE, TEST
from test_landing import CI_TOML, GATED_TREE, LINT_LOG, PREFLIGHTS, Harness, _main, _move_main, _plan_doc, _policy, _slim
from test_queue import World, _git

#: policy.toml's `off_gate` in the tree: ratchet-gate is off the merge gate
OFF = ("circuit-check", "ratchet-gate")
#: what `step_paths` finds of `LINT_LOG`, a failed preflight-lints: its failing test and the path its failure names
LINT_PATHS = frozenset({"tests/test_repository.py", "tools/research/src/research/store/README.md"})
#: a graph build family declaring lean-agreement's reads in the shape the tree's does
BUILD_SRC = ('VERIFIER = "verity/protocols/flock/agreement"\nLEAN = "verity/protocols/flock/verifier"\n'
             'CHECK_READS = (*(f"tools/verity/check/{m}" for m in ("check.py", "settings.py")),\n'
             '               ".agents/skills/using-slack/registry.json")\n')
ART = "art:" + "ab" * 32
#: a `main` whose files each bounded step reads, beside one none of them does (docs/notes.md)
KEYED = {BUILD: BUILD_SRC, "tools/verity/check/check.py": "# check\n", "tools/verity/check/settings.py": "# settings\n",
         ".agents/skills/using-slack/registry.json": "{}\n", "uv.lock": "version = 1\n", "pyproject.toml": "[project]\n",
         "tools/x/pyproject.toml": "[project]\n", "verity/protocols/flock/agreement/ci.py": "# ci\n",
         "verity/protocols/flock/verifier/Main.lean": "-- main\n", "fixtures/artifacts.json": json.dumps({ART: "fixtures/golden/a.json"}),
         "fixtures/golden/a.json": "{}\n", "docs/notes.md": "notes\n"}
#: each step the run judges whose reads are bounded, with files of `KEYED` it reads and files it doesn't
BOUNDED = {"preflight-lock": (("uv.lock", "tools/x/pyproject.toml", "tools/verity/check/check.py", ".python-version"),
                              ("verity/protocols/flock/agreement/ci.py", "fixtures/golden/a.json")),
           "lean-agreement": (("verity/protocols/flock/agreement/ci.py", "verity/protocols/flock/verifier/Main.lean",
                               "tools/verity/check/settings.py", ".agents/skills/using-slack/registry.json", "uv.lock"),
                              ("tools/x/pyproject.toml", "fixtures/golden/a.json", "tools/verity/check/other.py")),
           "preflight-fixtures": (("fixtures/artifacts.json", "fixtures/golden/a.json", "tools/research/src/research/x.py",
                                   "tools/verity/check/check.py"), ("fixtures/golden/b.json", "tools/x/pyproject.toml"))}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Beside(Harness):
    """`Harness`, whose launch of a run beside a check gives ``got`` when it is set: a detached launch's record, or the ids and
    tail of a train.sh that printed none."""
    got: dict | tuple[list[str], str] | None = None

    def launch_post(self, repo, commit, machine, steps, **kw):
        if self.got is None:
            return super().launch_post(repo, commit, machine, steps, **kw)
        self.posted.append((commit, steps, kw))
        return self.got


def _harness(world: World, monkeypatch, agreement: bool = False, off_gate: tuple[str, ...] = OFF, tree: dict | None = None,
             change: dict | None = None, **kw) -> tuple[Beside, str, str, list]:
    """`Beside` on `main` (with ``tree``'s files) under policy.toml's ``off_gate``, candidate ``sha`` one commit above it
    (``change``, else f.txt) in the queue's clone, and the bases and commits lean-agreement's check asked of
    (`merge.required_steps`), which finds it changed with ``agreement``."""
    _move_main(world, {**GATED_TREE, "tools/verity/check/policy.toml": _policy(off_gate=off_gate), **(tree or {})})
    asked: list[tuple[str, str, str]] = []
    monkeypatch.setattr(merge, "required_steps", lambda repo, base, commit, tool: asked.append((base, commit, tool)) or (
        {"lean-agreement": ["verity/protocols/flock/prover/"]} if agreement else {}))
    h = Beside(world, monkeypatch, **kw)
    h.q._clone()
    base, sha = _main(world), world.change(change or {"f.txt": "1\n"})
    _git(h.q.workdir, "fetch", "-q", "origin", f"refs/dev/{sha}")
    return h, base, sha, asked


def _failed(h: Beside, base: str, sha: str, step: str, inputs: frozenset[str] = frozenset(), kind: str = "own") -> Failed:
    """The `Failed` of ``step`` failed at candidate ``sha`` on ``base``: keyed by the content it reads at each, its one test
    the step's, the paths it names ``inputs``, and what it can have read the step's reads at ``sha`` (`reads`)."""
    got = reads(h.q.workdir, sha, step)
    return Failed(f"step {step}", key(h.q.workdir, sha, step), inputs, key(h.q.workdir, base, step), frozenset({step_test(step)}),
                  reads=got if isinstance(got, frozenset) else None, named=inputs, kind=kind)


def _rewrite(d, **steps: dict | None) -> None:
    """``d``'s steps.jsonl with each step ``steps`` names (``preflight_fixtures=...``) given those fields, or left out when None."""
    rows = []
    for st in step_records(d):
        key = st["step"].replace("-", "_")
        if key not in steps:
            rows.append(st)
        elif steps[key] is not None:
            rows.append({**st, **steps[key]})
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _down(rid: str, sha: str, why: str) -> Verdict:
    return Verdict(INFRA, (), (Failed(f"the run beside its check ({rid}) {why}", f"{sha}:{rid}", frozenset()),))


class Plans:
    """Drive's auto plans (`Driver.plan_shards`, `Driver.planned`) on ``d``: each one asked for gets a job's directory and a
    plan's (``tmp``'s job-K and plan-K, the plan's file in it), and is being made until ``answers`` holds what its plan's
    directory answers.  An answer is read once, as `drive._job_end` reads a job's end: its job's directory goes, and a job
    with none is gone with no exit."""

    def __init__(self, d, tmp: Path):
        self.asked: list[tuple[str, str, int | None, int | None]] = []
        self.answers: dict[str, tuple[dict | None, str | None]] = {}
        self.tmp = tmp
        d.plan_shards, d.planned = self.plan, self.planned

    def plan(self, repo, tip, steps, max_shards=None, n=None) -> dict:
        self.asked.append((tip, steps, max_shards, n))
        job, out = (self.tmp / f"{x}-{len(self.asked)}" for x in ("job", "plan"))
        job.mkdir(parents=True)
        out.mkdir()
        (out / AUTO_PLAN).write_text("{}\n")
        return {"dir": str(job), "pid": 7000 + len(self.asked), "out": str(out)}

    def planned(self, rec: dict) -> tuple[dict | None, str | None]:
        if not Path(rec["dir"]).is_dir():
            return None, "its process is gone with no exit written (a restart of the pod)"
        if (got := self.answers.get(rec["out"], (None, None))) != (None, None):
            shutil.rmtree(rec["dir"])
        return got


# -- start


@pytest.mark.parametrize("agreement", [False, True])
def test_start_launches_the_steps_beside_the_executors_check_with_main_below_as_its_base(world, monkeypatch, agreement):
    """The candidate's run: the steps it sends with `Landing.beside_args`' keywords, ``base`` the `main` below it, lean-agreement
    only when what it checks changed between them, and its record, with ``base``."""
    h, base, sha, asked = _harness(world, monkeypatch, agreement)
    rec = start(h.d, sha, base)
    steps = ["lean-agreement", "ratchet-gate"] if agreement else ["ratchet-gate"]
    assert h.posted == [(sha, ",".join(steps), {"base": base, "agreement": agreement, "no_regenerate": True,
                                                "train_from": OWN_TRAIN})]
    assert rec == {"ids": ["r-post-0"], "tip": sha, "base": base, "at": h.clock, "steps": steps,
                   **({} if agreement else {"skip": ["lean-agreement"]})}
    assert asked == [(base, sha, "check")]
    assert any(x == f"research queue vqueue: train.sh {sha[:12]} m1 --deferred {','.join(steps)} --base {base[:12]}"
               + (" --agreement" if agreement else "") + f" --no-regenerate beside candidate {sha[:12]}'s check: r-post-0"
               for x in world.logs), world.logs


def test_a_detached_launch_is_pending_until_it_prints_its_run_ids_which_judge_puts_in_the_record(world, monkeypatch, tmp_path):
    h, base, sha, _ = _harness(world, monkeypatch)
    h.got = launch = {"dir": str(tmp_path / "launch"), "pid": 1}
    printed: list[list[str] | None] = [None]
    h.d.launched = lambda got: (printed[0], "")
    rec = start(h.d, sha, base)
    assert rec == {"ids": [], "launch": launch, "tip": sha, "base": base, "at": h.clock, "steps": ["ratchet-gate"],
                   "skip": ["lean-agreement"]}
    assert judge(h.d, rec) == Verdict(PENDING) and rec["launch"] == launch
    h.clock += LAUNCH_LOST_S
    printed[0] = ["r-1"]
    h.run_state["r-1"] = ("running", None, None)
    assert judge(h.d, rec) == Verdict(PENDING)
    assert rec == {"ids": ["r-1"], "tip": sha, "base": base, "at": h.clock, "steps": ["ratchet-gate"], "skip": ["lean-agreement"]}


def test_a_launch_that_registers_no_run_is_infra(world, monkeypatch, tmp_path):
    """A train.sh that printed no run ids: at once, `start` gives no record, which `judge` takes as infra; detached, `judge`
    finds it so, once it exited or `LAUNCH_LOST_S` after its launch, and keeps that in the record."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.got = ([], "train.sh: no machine answered")
    assert start(h.d, sha, base) is None
    assert any(x.endswith("registered no run: train.sh: no machine answered") for x in world.logs), world.logs
    assert judge(h.d, None) == Verdict(INFRA, (), (Failed("the run beside its check registered no run", ":no run", frozenset()),))
    h.got = launch = {"dir": str(tmp_path / "launch"), "pid": 1}
    h.d.launched = lambda got: ([], "train.sh: refused")
    rec = start(h.d, sha, base)
    want = Verdict(INFRA, (), (Failed(f"the run beside its check registered no run: train.sh: refused (log {launch['dir']}/log)",
                                      f"{sha}:no run", frozenset()),))
    assert judge(h.d, rec) == want and "launch" not in rec and judge(h.d, rec) == want
    h.d.launched = lambda got: (None, "")
    rec = start(h.d, sha, base)
    h.clock += LAUNCH_LOST_S
    assert judge(h.d, rec) == Verdict(PENDING)
    h.clock += 1
    lost = judge(h.d, rec).infra[0]
    assert lost.task.startswith(f"the run beside its check registered no run: no run ids "
                            f"{LAUNCH_LOST_S // 60} min after its launch") and lost.launch
    assert judge(h.d, rec).infra == (lost,), "the record keeps it, launch and all"


@pytest.mark.parametrize("tail, launch", [
    ("train: no run launched in the wait; stopped its launcher (pid 7), so none launches later; see /l/log", True),
    ("train: 0e4f on m1 (its process is gone with no exit written: a restart of the pod)", True),
    ('Traceback (most recent call last):\n  File "tools/verity/check/check.py", line 9, in <module>\nImportError: boom\n'
     "train: no run launched; see /l/log", False),
    ("train.sh: refused", False)])
def test_only_a_launch_that_outran_its_wait_or_that_the_pod_lost_is_launch_only_infra(world, monkeypatch, tmp_path, tail,
                                                                                         launch):
    """train.sh runs the candidate's own check.py, so a launcher that exited without a run may be the tree's doing: only a
    launcher stopped in its wait, or one the pod lost, is `Failed.launch`, which the loop retries before blaming anyone."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.got = {"dir": str(tmp_path / "launch"), "pid": 1}
    h.d.launched = lambda got: ([], tail)
    got = judge(h.d, start(h.d, sha, base))
    assert got.state == INFRA and got.infra[0].launch is launch, got


@pytest.mark.parametrize("how", ["dry_run", "shadow"])
def test_a_dry_run_or_a_shadow_queue_launches_nothing(world, monkeypatch, how):
    h, base, sha, _ = _harness(world, monkeypatch, **({"shadow": True} if how == "shadow" else {}))
    h.d.dry_run = how == "dry_run"
    assert start(h.d, sha, base) is None and h.posted == []
    assert any(x.startswith(f"research queue vqueue: would launch train.sh {sha[:12]} m1 --deferred ratchet-gate --base")
               for x in world.logs), world.logs


def test_start_needs_the_queues_machines(world, monkeypatch):
    """The loop's Landing is built with no machines unless it is given drive.toml's `on`, and a train.sh with none registers
    nothing: `start` says so rather than launch it."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on = ""
    with pytest.raises(ValueError, match=r"build the Landing with drive.toml's `on` \(`research\.queue\.flags_at\(repo, main\)\.on`\)"):
        start(h.d, sha, base)
    assert h.posted == []


# -- judge


@pytest.mark.parametrize("state", ACTIVE)
def test_judge_is_pending_while_the_run_runs(world, monkeypatch, state):
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    h.run_state["r-post-0"] = (state, None, None)
    assert judge(h.d, rec) == Verdict(PENDING) and "verdict" not in rec


def test_judge_reads_the_runs_state_afresh_each_call(world, monkeypatch, tmp_path):
    """The loop's Landing outlives its ticks: `judge` reads each run's state every call."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    assert judge(h.d, rec) == Verdict(PENDING)
    _slim(h, tmp_path)
    assert judge(h.d, rec) == Verdict(PASS)


def test_a_run_that_passed_every_step_it_judges_passes(world, monkeypatch, tmp_path):
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path)
    assert [st["step"] for st in step_records(d)] == [*PREFLIGHTS, "ratchet-gate"]
    assert judge(h.d, rec) == Verdict(PASS) and rec["verdict"] == {"state": PASS, "passed": list(PREFLIGHTS)}


def test_a_failed_preflight_fails_the_candidate_with_the_paths_its_failure_names(world, monkeypatch, tmp_path):
    """No graph/v2 task runs the preflights: a failed one fails the candidate, its `Failed` holding the paths its failure names
    (`step_paths`), the paths the suspects are the members whose changes meet."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, preflight_lints=LINT_LOG)
    (st,) = [st for st in step_records(d) if st["status"] == "failed"]
    assert step_paths(d, st) == LINT_PATHS
    assert judge(h.d, rec) == Verdict(FAIL, (_failed(h, base, sha, "preflight-lints", LINT_PATHS),), ended=True)


def test_a_failures_frames_are_named_from_the_repositorys_root_and_frames_outside_it_dropped(world, monkeypatch, tmp_path):
    """A traceback's frame is an absolute path, in the check's tree (job.json's `run.cwd`) or outside it: the `Failed` names
    one in the tree as its repository path, which meets that file and no other of its name, and drops one outside it (the
    interpreter's, the venv's), which no change of a member's touches."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, preflight_lints='FAILED tests/test_x.py::test_y - KeyError\n'
                                           '  File "/scratch/r-1/tree/pkg/conftest.py", line 3\n'
                                           '  File "/scratch/r-1/tree/.venv/lib/python3.12/site-packages/x.py", line 9\n'
                                           '  File "/usr/lib/python3.12/runpy.py", line 1\n')
    (d / "job.json").write_text(json.dumps({"run": {"cwd": "/scratch/r-1/tree/"}}))
    (st,) = [st for st in step_records(d) if st["status"] == "failed"]
    (failed,) = judge(h.d, rec).failed
    assert failed.inputs == failed.named == step_paths(d, st) == {"tests/test_x.py", "pkg/conftest.py"}
    assert not meets({"conftest.py"}, failed.inputs) and not meets({"other/pkg/conftest.py"}, failed.inputs)


@pytest.mark.parametrize("off_gate", [OFF, ("circuit-check",)])
def test_a_red_ratchet_gate_fails_the_candidate_only_on_the_gate(world, monkeypatch, tmp_path, off_gate):
    """ratchet-gate is off the gate in the tree, so its red holds no candidate; on the gate, the candidate's own policy.toml
    says so, it fails it."""
    h, base, sha, _ = _harness(world, monkeypatch, off_gate=off_gate)
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, ratchet_gate="  merge 1 (0123456789ab): failed: grows p01.json 2 -> 3 entries\n")
    assert judge(h.d, rec) == (Verdict(PASS) if "ratchet-gate" in off_gate else
                               Verdict(FAIL, (_failed(h, base, sha, "ratchet-gate"),), ended=True))


@pytest.mark.parametrize("agreement", [True, False])
def test_lean_agreement_is_judged_only_when_the_run_ran_it(world, monkeypatch, tmp_path, agreement):
    """Run, a red lean-agreement fails the candidate, with no path when its log names none; skipped, it isn't judged."""
    h, base, sha, _ = _harness(world, monkeypatch, agreement)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, **({"lean_agreement": "lean-agreement: the verifiers disagree on 3 of 40 vectors\n"} if agreement else {}))
    assert ("lean-agreement" in {st["step"] for st in step_records(d)}) is agreement
    assert judge(h.d, rec) == (Verdict(FAIL, (_failed(h, base, sha, "lean-agreement"),), ended=True) if agreement else Verdict(PASS))


@pytest.mark.parametrize("rc, why", [(NOT_STARTED, f"never ran or was stopped (done rc={NOT_STARTED})"),
                                     (SIGTERM_RC, f"never ran or was stopped (done rc={SIGTERM_RC})"),
                                     (137, "was killed (done rc=137)"), (-9, "was killed (done rc=-9)")])
def test_a_run_that_never_ran_was_stopped_or_was_killed_is_infra(world, monkeypatch, rc, why):
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    h.run_state["r-post-0"] = ("done", rc, None)
    assert judge(h.d, rec) == _down("r-post-0", sha, why)


def test_a_step_its_machine_killed_is_infra_and_no_failure(world, monkeypatch, tmp_path):
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, preflight_fixtures="Killed\n")
    _rewrite(d, preflight_fixtures={"rc": 137})
    assert judge(h.d, rec) == _down("r-post-0", sha, "was killed (done rc=1)")


def test_a_run_killed_after_passing_every_step_it_judges_passes(world, monkeypatch, tmp_path):
    """Killed in ratchet-gate, off the gate, once every preflight passed: what it judges passed."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path)
    _rewrite(d, ratchet_gate={"status": "failed", "rc": 137})
    h.run_state["r-post-0"] = ("done", 137, None)
    assert judge(h.d, rec) == Verdict(PASS)


def test_a_run_with_no_state_for_lost_after_s_is_infra(world, monkeypatch):
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    h.run_state["r-post-0"] = ("unknown", None, None)
    h.clock += LOST_AFTER_S
    assert judge(h.d, rec) == Verdict(PENDING)
    h.clock += 1
    assert judge(h.d, rec) == _down("r-post-0", sha, "was lost")


@pytest.mark.parametrize("fixtures", [None, {"status": "stopped"}])
def test_a_run_that_ended_failing_nothing_without_passing_a_step_it_judges_is_infra(world, monkeypatch, tmp_path, fixtures):
    """Its exit 0 says only that nothing failed: a step absent from its steps.jsonl, or one a cancel stopped (`not_run`), gave
    no verdict."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    _rewrite(_slim(h, tmp_path), preflight_fixtures=fixtures)
    assert judge(h.d, rec) == _down("r-post-0", sha, "ended (done rc=0) without passing preflight-fixtures: each skipped, or "
                                                     "absent from its steps.jsonl")


def test_judge_keeps_its_verdict_in_the_record_and_reads_the_run_no_more(world, monkeypatch, tmp_path):
    """The record, saved as JSON, gives the same verdict without reading the run's state or records again: preflight-lints
    reads the whole tree, so its failure keeps no reads, and None comes back."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lints=LINT_LOG)
    got = judge(h.d, rec)
    f = _failed(h, base, sha, "preflight-lints", LINT_PATHS)
    assert rec["verdict"] == {"state": FAIL, "failed": [{"task": "step preflight-lints", "key": f.key, "inputs": sorted(LINT_PATHS),
                                                         "base_key": f.base_key, "tests": ["step:preflight-lints"],
                                                         "named": sorted(LINT_PATHS), "kind": "own"}],
                              "passed": [s for s in PREFLIGHTS if s != "preflight-lints"]}
    assert got.failed[0].reads is None
    saved = json.loads(json.dumps(rec))
    h.d.states = h.d.evidence = None
    assert judge(h.d, saved) == got == judge(h.d, rec)


def test_start_and_judge_write_nothing_to_the_queues_state_and_tell_and_stop_nothing(world, monkeypatch, tmp_path):
    h, base, sha, _ = _harness(world, monkeypatch)
    before = h.q.load_state()
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lints=LINT_LOG)
    assert judge(h.d, rec).state == FAIL
    assert h.q.load_state() == before and h.posts == [] and h.cancelled == []


# -- the check that gates

#: lean-audit's log of a package that failed its build, in the scratch tree ``n``, and one more failure it may print
AUDIT = ("AUDIT /w/lean-audit-scratch-{n}-x/security/Proofs: FAIL; built WITHOUT a sandbox\n  build: exit 1\n"
         "    error: ../Proofs/A.lean:100:2: Tactic `rfl` failed\n  closure: {n} modules checked, 2 reused\n"
         "  trust gap: 2 premises outside their trust\n    G: 2 (a, b)\n")
RECORD = "  record: the statement of G changed\n"


@pytest.mark.parametrize("shards", [1, 3])
def test_start_launches_each_shard_with_the_loops_own_train_sh(world, monkeypatch, shards):
    """A candidate's tree doesn't decide how long its launch is waited for: every launch runs the loop's own train.sh, this
    tree's, which the loop's pin deploys."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, given, launch = "m1,m2,m3", [], h.d.launch_postland
    h.d.launch_postland = lambda repo, commit, machine, steps, train_from=None, **kw: given.append(train_from) or launch(
        repo, commit, machine, steps, **kw)
    start(h.d, sha, base, shards=shards)
    assert given == [OWN_TRAIN] * shards and OWN_TRAIN.is_file()
    assert OWN_TRAIN == Path(__file__).resolve().parents[3] / "tools" / "verity" / "check" / "train.sh"


@pytest.mark.parametrize("agreement", [False, True])
def test_start_with_shards_launches_the_after_landing_run_of_every_step_a_shard_a_launch(world, monkeypatch, agreement):
    """The check that gates is the after-landing run of the candidate above ``base``: every step sent, with
    `Landing.beside_args`' keywords, `--shard K/N` on shard K's machines, the record holding each shard's run ids and its
    ``gate``."""
    h, base, sha, _ = _harness(world, monkeypatch, agreement)
    h.d.on, machines, launch = "m1,m2,m3,m4", [], h.d.launch_postland
    h.d.launch_postland = lambda repo, commit, machine, steps, **kw: machines.append(machine) or launch(repo, commit, machine,
                                                                                                         steps, **kw)
    rec = start(h.d, sha, base, shards=3)
    kw = h.d.beside_args(sha, base)[0]
    assert kw == {"base": base, "agreement": agreement, "no_regenerate": True}
    assert h.posted == [(sha, ",".join(AFTER_LANDING), {**kw, "shard": f"{k}/3", "train_from": OWN_TRAIN}) for k in (1, 2, 3)]
    assert machines == shard_machines("m1,m2,m3,m4", 3) == ["m1,m4", "m2", "m3"]
    assert rec == {"ids": ["r-post-0", "r-post-1", "r-post-2"], "groups": [["r-post-0"], ["r-post-1"], ["r-post-2"]],
                   "tip": sha, "base": base, "at": h.clock, "gate": CHECK,
                   "steps": [s for s in AFTER_LANDING if agreement or s != "lean-agreement"],
                   **({} if agreement else {"skip": ["lean-agreement"]})}
    whole = start(h.d, sha, base, shards=1)
    assert h.posted[3] == (sha, ",".join(AFTER_LANDING), {**kw, "train_from": OWN_TRAIN}) and machines[3] == "m1,m2,m3,m4"
    assert whole == {**{k: v for k, v in rec.items() if k != "groups"}, "ids": ["r-post-3"]}


@pytest.mark.parametrize("shards", [0, 1, 3, "auto:2"])
@pytest.mark.parametrize("drops", [False, True])
def test_a_candidate_is_sent_only_the_steps_its_own_ci_toml_declares_as_jobs(world, monkeypatch, tmp_path, shards, drops):
    """check.py refuses a `--deferred` naming a job the candidate's own ci.toml doesn't declare (`declared_steps`): a candidate
    whose ci.toml drops a step's job (lean-suites, or ratchet-gate beside its check) is sent every other step, beside its
    check, whole, as each shard or to its auto plan, with one line naming the step; one declaring every step is sent them all."""
    dropped = "lean-suites" if shards else "ratchet-gate"
    jobs = CI_TOML.replace(f'[[job]]\nname = "{dropped}"\nowner = "ci"\ngroup = 1\n', "")
    assert jobs != CI_TOML
    h, base, sha, _ = _harness(world, monkeypatch, agreement=True,
                               change={"tools/verity/check/ci.toml": jobs} if drops else None)
    h.d.on, plans = "m1,m2,m3", Plans(h.d, tmp_path)
    start(h.d, sha, base, shards=shards)
    want = ",".join(s for s in (AFTER_LANDING if shards else SLIM_STEPS) if not (drops and s == dropped))
    sent = [steps for _, steps, _ in h.posted] + [steps for _, steps, *_ in plans.asked]
    assert sent == [want] * {0: 1, 1: 1, 3: 3, "auto:2": 1}[shards]
    said = (f"research queue vqueue: candidate {sha[:12]}'s tools/verity/check/ci.toml declares no job {dropped}, which "
            "check.py refuses: left out of its --deferred")
    assert (said in world.logs) == drops and not any("ci.toml can't be read" in x for x in world.logs)


@pytest.mark.parametrize("ci", [None, "not toml ["])
def test_a_candidate_whose_ci_toml_can_t_be_read_is_sent_every_step_and_one_line_says_so(world, monkeypatch, ci):
    """Where git can't read the candidate's ci.toml (gone, or not TOML), `declared_steps` can't say which jobs it declares: its
    check is sent every step, as before the tree's own list was read, and one line says why."""
    h, base, sha, _ = _harness(world, monkeypatch, change={"tools/verity/check/ci.toml": ci})
    start(h.d, sha, base, shards=1)
    assert [steps for _, steps, _ in h.posted] == [",".join(AFTER_LANDING)]
    assert world.logs.count(f"research queue vqueue: candidate {sha[:12]}'s tools/verity/check/ci.toml can't be read: its "
                            f"--deferred keeps every step ({','.join(AFTER_LANDING)})") == 1


def test_every_shard_of_a_candidates_check_leaves_out_the_tests_the_loops_list_names(world, monkeypatch, tmp_path):
    """The loop's deselect list, read at each launch: every shard is sent the node ids it names (`train.sh --deselect`), written
    beside it; a line that is no node id is logged and left out, so a typo drops no test and reds no candidate; no list, or one
    naming none, sends none."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, listed = "m1,m2,m3", tmp_path / "deselect"
    ids = ["tools/verity/circuit_check/tests/test_circuit_check.py::test_every_call_the_suite_checks_is_partitioned",
           "verity/protocols/remote/verifier/composition/tests/test_composition_difftest.py::test_remote_agrees"]
    listed.write_text(f"# over 5 minutes\n\n{ids[0]}\nnot a node id\n/abs/test_x.py::t\n{ids[1]}\n{ids[0]}\n")
    start(h.d, sha, base, shards=3, deselect=listed)
    sent = tmp_path / "deselect.sent"
    assert [kw.get("deselect") for _commit, _steps, kw in h.posted] == [str(sent)] * 3
    assert sent.read_text().splitlines() == ids
    assert any("2 line(s) not a test's node id" in x and "not a node id; /abs/test_x.py::t" in x for x in world.logs), world.logs
    assert any(f"--deselect {sent} (2 tests) as 3 shards" in x for x in world.logs), world.logs
    for text in (None, "# none yet\n"):
        listed.unlink(missing_ok=True)
        if text is not None:
            listed.write_text(text)
        h.posted.clear()
        start(h.d, sha, base, shards=3, deselect=listed)
        assert len(h.posted) == 3 and not any("deselect" in kw for _commit, _steps, kw in h.posted)


@pytest.mark.parametrize("second", [["r-2"], []])
def test_a_sharded_launch_is_read_once_every_shard_printed_and_a_shard_that_registered_no_run_stops_the_others(
        world, monkeypatch, tmp_path, second):
    """Each shard's train.sh runs detached: pending until every one printed its run ids, none read before (a launch read
    once is gone, and the record is saved between), then the run's ids are every shard's.  One that printed none is infra,
    the shards that registered a run cancelled."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, printed = "m1,m2", {}
    launches = [{"dir": str(tmp_path / f"launch-{k}"), "pid": 1} for k in (1, 2)]
    h.d.launch_postland = lambda *a, **kw: launches[int(kw["shard"][0]) - 1]
    h.d.launched = lambda got: (printed.get(got["dir"]), "train.sh: no slot")
    rec = start(h.d, sha, base, shards=2)
    assert rec["ids"] == [] and rec["shards"] == [{"ids": [], "launch": x} for x in launches]
    assert judge(h.d, rec) == Verdict(PENDING)
    printed[launches[0]["dir"]] = ["r-1"]
    assert judge(h.d, rec) == Verdict(PENDING) and [r["launch"] for r in rec["shards"]] == launches
    printed[launches[1]["dir"]], h.run_state["r-1"], h.run_state["r-2"] = second, ("running", None, None), ("running", None, None)
    got = judge(h.d, json.loads(json.dumps(rec)) if second else rec)
    if second:
        assert got == Verdict(PENDING) and h.cancelled == []
        assert judge(h.d, rec) == Verdict(PENDING) and (rec["ids"], rec["groups"]) == (["r-1", "r-2"], [["r-1"], ["r-2"]])
        assert "shards" not in rec and "verdict" not in rec
    else:
        assert got.state == INFRA and "registered no run: shard 2/2: train.sh: no slot" in got.infra[0].task
        assert not got.infra[0].launch and h.cancelled == ["r-1"] and judge(h.d, rec) == got


def test_a_shard_whose_train_sh_printed_no_run_at_once_stops_the_others_and_start_gives_no_record(world, monkeypatch):
    """A shard launched at once whose train.sh printed no run ids: the shards that registered a run are cancelled, and `start`
    logs that shard's tail and gives no record, which `judge` takes as infra."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, launch = "m1,m2,m3", h.d.launch_postland
    h.d.launch_postland = lambda repo, commit, machine, steps, **kw: (
        ([], "train.sh: no slot") if kw["shard"] == "2/3" else launch(repo, commit, machine, steps, **kw))
    assert start(h.d, sha, base, shards=3) is None and h.cancelled == ["r-post-0", "r-post-1"]
    assert world.logs[-1].endswith(f"as 3 shards as candidate {sha[:12]}'s check registered no run: train.sh: no slot")


def test_shards_are_n_of_ci_tomls_plan_or_auto_n_of_an_auto_plan():
    assert [shard_count(s) for s in (0, 3, "auto:6", "auto:12")] == [(0, False), (3, False), (6, True), (12, True)]
    for bad in ("auto:0", "auto:", "auto:6x", "auto:-1", " auto:6", "6", "x:6"):
        with pytest.raises(ValueError, match="or auto:N"):
            shard_count(bad)


@pytest.mark.parametrize("detached", [False, True])
def test_start_with_auto_shards_has_the_candidates_plan_made_and_the_judge_that_finds_it_launches_its_n_shards(
        world, monkeypatch, tmp_path, detached):
    """`auto:N`: `start` launches no shard but has the auto plan of the candidate's tree made, of every after-landing step and
    exactly N shards (`check.py --shard-plan --deferred STEPS --n N`); pending while it is made, and the first `judge` that
    finds it made launches each shard `--shard auto:K/N` with its file on shard K's machines, the plan read once.  The file
    goes once every shard has its run ids, each having sent it with its run."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, machines, printed, launches, post = "m1,m2,m3,m4,m5,m6", [], {}, [], h.d.launch_postland
    plans = Plans(h.d, tmp_path)

    def launch(repo, commit, machine, steps, **kw):
        machines.append(machine)
        if not detached:
            return post(repo, commit, machine, steps, **kw)
        h.posted.append((commit, steps, kw))
        launches.append(got := {"dir": str(tmp_path / f"launch-{len(launches) + 1}"), "pid": 1})
        return got

    h.d.launch_postland, h.d.launched = launch, lambda got: (printed.get(got["dir"]), "")
    rec = start(h.d, sha, base, shards="auto:6")
    kw, out = h.d.beside_args(sha, base)[0], tmp_path / "plan-1"
    assert plans.asked == [(sha, ",".join(AFTER_LANDING), None, 6)] and h.posted == []
    assert rec == {"ids": [], "plan": {"dir": str(tmp_path / "job-1"), "pid": 7001, "out": str(out), "n": 6,
                                       "deferred": ",".join(AFTER_LANDING), "kw": kw},
                   "tip": sha, "base": base, "at": h.clock, "gate": CHECK, "skip": ["lean-agreement"],
                   "steps": [s for s in AFTER_LANDING if s != "lean-agreement"]}
    assert any(x.endswith(f"as an auto plan's 6 shards as candidate {sha[:12]}'s check: planning") for x in world.logs)
    h.clock += PLAN_WAIT_S
    assert judge(h.d, rec) == Verdict(PENDING) and h.posted == [] and "made" not in rec["plan"]
    rec = json.loads(json.dumps(rec))
    plans.answers[str(out)] = (_plan_doc(6), None)
    h.clock += 60
    assert judge(h.d, rec) == Verdict(PENDING)
    assert h.posted == [(sha, ",".join(AFTER_LANDING), {**kw, "shard": f"auto:{k}/6", "shard_plan": str(out / AUTO_PLAN),
                                                        "train_from": OWN_TRAIN}) for k in range(1, 7)]
    assert machines == shard_machines(h.d.on, 6) == [f"m{k}" for k in range(1, 7)]
    assert rec["plan"]["made"] == {"digest": "d" * 64, "predicted_s": 3600.0} and rec["at"] == h.clock
    ids = [f"r-{k}" for k in range(1, 7)] if detached else [f"r-post-{k}" for k in range(6)]
    said = "launching" if detached else " ".join(ids)
    assert any(x.endswith(f": auto plan {'d' * 12}, predicted wall 60 min: {said}") for x in world.logs), world.logs
    if detached:
        assert rec["ids"] == [] and [r["launch"] for r in rec["shards"]] == launches and (out / AUTO_PLAN).is_file()
        rec = json.loads(json.dumps(rec))
        printed.update({x["dir"]: [f"r-{k}"] for k, x in enumerate(launches, 1)})
        h.run_state.update({f"r-{k}": ("running", None, None) for k in range(1, 7)})
        assert judge(h.d, rec) == Verdict(PENDING)
    assert (rec["ids"], rec["groups"], "shards" in rec) == (ids, [[i] for i in ids], False)
    assert not out.exists() and not (tmp_path / "job-1").exists()
    assert judge(h.d, json.loads(json.dumps(rec))) == Verdict(PENDING) and len(h.posted) == 6, "its shards launch once"


def test_a_shard_of_an_auto_plan_that_registered_no_run_stops_the_others_and_the_plans_file_goes(world, monkeypatch, tmp_path):
    """As a shard of ci.toml's plan does, one of an auto plan's that printed no run ids stops the others, and no shard sends the
    plan's file any more."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, printed = "m1,m2", {}
    plans = Plans(h.d, tmp_path)
    launches = [{"dir": str(tmp_path / f"launch-{k}"), "pid": 1} for k in (1, 2)]
    h.d.launch_postland = lambda *a, **kw: launches[int(kw["shard"].removeprefix("auto:").split("/")[0]) - 1]
    h.d.launched = lambda got: (printed.get(got["dir"]), "train.sh: no slot")
    rec = start(h.d, sha, base, shards="auto:2")
    plans.answers[rec["plan"]["out"]] = (_plan_doc(2), None)
    assert judge(h.d, rec) == Verdict(PENDING) and (Path(rec["plan"]["out"]) / AUTO_PLAN).is_file()
    printed.update({launches[0]["dir"]: ["r-1"], launches[1]["dir"]: []})
    h.run_state["r-1"] = ("running", None, None)
    got = judge(h.d, rec)
    assert got.state == INFRA and "registered no run: shard 2/2: train.sh: no slot" in got.infra[0].task
    assert h.cancelled == ["r-1"] and not Path(rec["plan"]["out"]).exists()


@pytest.mark.parametrize("how", ["failed", "lost", "late", "another n"])
def test_an_auto_plan_that_failed_or_was_not_made_in_time_is_infra_its_directories_gone(world, monkeypatch, tmp_path, how):
    """A plan that failed (`check.py --shard-plan` exited nonzero, or the pod lost its job), has another N than asked, or wasn't
    made `PLAN_WAIT_S` after the start (its job stopped): infra, kept, no shard launched, the plan's directories removed.  Only
    a lost job points at the machines (`Failed.launch`): the planner runs the candidate's own check.py."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, stopped = "m1,m2", []
    plans = Plans(h.d, tmp_path)
    monkeypatch.setattr("research.vqueue_slim.os.killpg", lambda pid, sig: stopped.append((pid, sig)))
    rec = start(h.d, sha, base, shards="auto:2")
    gone = "its process is gone with no exit written (a restart of the pod)"
    plans.answers[rec["plan"]["out"]] = {"failed": (None, "`check.py --shard-plan` exited 2: no costs"), "lost": (None, gone),
                                         "late": (None, None), "another n": (_plan_doc(3), None)}[how]
    h.clock += PLAN_WAIT_S
    if how == "late":
        assert judge(h.d, rec) == Verdict(PENDING) and stopped == []
        h.clock += 1
    got = judge(h.d, rec)
    why = {"failed": "failed: `check.py --shard-plan` exited 2: no costs", "lost": f"failed: {gone}",
           "late": f"was not made in {PLAN_WAIT_S // 60} min (stopped)", "another n": "has 3 shards, not 2"}[how]
    assert got == Verdict(INFRA, (), (Failed(f"the run beside its check registered no run: its auto plan {why}", f"{sha}:no run",
                                             frozenset(), launch=how == "lost"),))
    assert stopped == ([(7001, signal.SIGTERM)] if how == "late" else []) and h.posted == []
    assert not Path(rec["plan"]["out"]).exists() and not Path(rec["plan"]["dir"]).exists()
    assert judge(h.d, json.loads(json.dumps(rec))) == got and rec["verdict"]["state"] == INFRA


@pytest.mark.parametrize("made", [False, True])
def test_abandon_gives_up_a_plan_still_being_made_and_keeps_one_its_launching_shards_send(world, monkeypatch, tmp_path, made):
    """A candidate dropped while its plan is made launches no shard: the plan's directories go, so its job writes no plan.  Once
    made, its shards' launches send its file, which stays until each has its run ids."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, h.got = "m1,m2", {"dir": str(tmp_path / "launch"), "pid": 1}
    plans = Plans(h.d, tmp_path)
    rec = start(h.d, sha, base, shards="auto:2")
    if made:
        plans.answers[rec["plan"]["out"]] = (_plan_doc(2), None)
        assert judge(h.d, rec) == Verdict(PENDING) and len(h.posted) == 2 and "made" in rec["plan"]
    abandon(rec)
    assert (Path(rec["plan"]["out"]) / AUTO_PLAN).is_file() == made and not Path(rec["plan"]["dir"]).exists()
    abandon({"ids": ["r-1"], "tip": sha})


@pytest.mark.parametrize("red, state", [("circuit-check", PASS), ("ratchet-gate", PASS),
                                        ("flock-rust-tests", FAIL), ("preflight-lock", FAIL)])
def test_the_check_that_gates_is_judged_on_the_steps_main_waits_for_at_its_base(world, monkeypatch, tmp_path, red, state):
    """Every step `main` waits for at the base gates, the preflights among them: not one policy.toml takes off the gate there
    (circuit-check) or drive.toml's warn step (ratchet-gate).  The candidate's
    own policy.toml takes flock-rust-tests off the gate, and it fails on it all the same."""
    h, base, sha, _ = _harness(world, monkeypatch, off_gate=("circuit-check",), change={
        "tools/verity/check/policy.toml": _policy(off_gate=("circuit-check", "flock-rust-tests"))})
    h.d.warn_steps = ("ratchet-gate",)
    rec = start(h.d, sha, base, shards=1)
    _slim(h, tmp_path, **{red.replace("-", "_"): "error: it failed\n"})
    got = judge(h.d, rec)
    assert got.state == state and [f.task for f in got.failed] == [f"step {red}"] * (state == FAIL)
    gates = [s for s in (*PREFLIGHTS, *AFTER_LANDING) if s not in ("lean-agreement", "circuit-check", "ratchet-gate")]
    if red not in PREFLIGHTS:
        assert sorted(rec["verdict"]["passed"]) == sorted(s for s in gates if s != red or state == PASS)


def test_the_check_that_gates_names_each_test_of_a_logs_short_summary_by_its_whole_id(world, monkeypatch, tmp_path):
    """pytest's short summary prints a test's whole id, a parametrized one's spaces among it, then ` - ` and its error: the
    failure names each test by all of it, so two cases of one test that differ past a space are two tests."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base, shards=1)
    _slim(h, tmp_path, preflight_lints="FAILED tests/test_x.py::test_y[a b] - KeyError: 'k'\nFAILED tests/test_x.py::test_y[a c]\n"
                                       "ERROR tests/test_z.py - ImportError: no module named z\n")
    (f,) = judge(h.d, rec).failed
    assert (f.task, f.tests) == ("step preflight-lints", {"tests/test_x.py::test_y[a b]", "tests/test_x.py::test_y[a c]",
                                                          "tests/test_z.py"})


#: suite pkg's report row: one test failed, and nothing more
PKG = {"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": [TEST], "beyond_tests": False, "log": "/x/suites/pkg.log"}


@pytest.mark.parametrize("at", ["base", "candidate", "unbased"])
def test_the_check_that_gates_passes_a_failure_only_an_allowance_at_its_base_covers(world, monkeypatch, tmp_path, at):
    """An unexpired allowance of policy.toml at the base covers a suite's failure as drive's covers it on `checked`, but the
    candidate's own (it adds the allowance) never covers its own failure: that fails it, on the suite's test.  A record
    with no base reads the candidate's."""
    allowing = {"tools/verity/check/policy.toml": _policy((f"suite pkg::{TEST}", "ci", LIVE), off_gate=OFF)}
    h, base, sha, _ = _harness(world, monkeypatch, tree=allowing if at == "base" else None,
                               change=None if at == "base" else allowing)
    rec = start(h.d, sha, base, shards=1)
    if at == "unbased":
        rec["base"] = None
    (_slim(h, tmp_path, pytest="") / "suites.json").write_text(json.dumps([PKG]))
    got = judge(h.d, rec)
    if at == "candidate":
        (f,) = got.failed
        assert (got.state, f.task, f.tests) == (FAIL, "step pytest", {f"pkg::{TEST}"})
        assert not h.d._only_allowed(rec, base) and h.d._only_allowed(rec)
    else:
        assert got == Verdict(PASS)


@pytest.mark.parametrize("own", [False, True])
def test_a_check_killed_on_what_only_the_candidates_own_allowance_covers_is_infra(world, monkeypatch, tmp_path, own):
    """A kill (a step's exit 137, on a suite that names no test) is the machine's fault unless an allowance at the base covers
    every failure: then the check passes; covered only by the candidate's own allowance, it is infra, to run again."""
    allowing = {"tools/verity/check/policy.toml": _policy(("suite pkg", "ci", LIVE), off_gate=OFF)}
    h, base, sha, _ = _harness(world, monkeypatch, tree=None if own else allowing, change=allowing if own else None)
    rec = start(h.d, sha, base, shards=1)
    d = _slim(h, tmp_path, pytest="Killed\n")
    _rewrite(d, pytest={"rc": 137})
    (d / "suites.json").write_text(json.dumps([{**PKG, "failed_tests": []}]))
    got = judge(h.d, rec)
    if own:
        assert got.state == INFRA and got.infra[0].task.endswith("was killed (done rc=1)"), got
    else:
        assert got == Verdict(PASS)


def test_lean_audit_s_failures_are_what_it_prints_under_each_failed_package_but_its_run_s_counts():
    log = ("lean-audit: verity/hardware/world: PASS reused\n" + AUDIT.format(n=1)
           + "  tamper: /w/lean-audit-scratch-1-x/security/Proofs/A.olean changed\n"
           "AUDIT /w/lean-audit-scratch-1-x/security: PASS  12 declarations\n  note: fine\n"
           "AUDIT /w/lean-audit-scratch-1-x/Flock: FAIL\nAUDIT: FAIL (security/Proofs, Flock); reports in /out\n"
           "lean-audit: FAIL: unread: Core grew\n")
    named = {"security/Proofs::build: exit 1", "security/Proofs::error: ../Proofs/A.lean:100:2: Tactic `rfl` failed",
             "security/Proofs::tamper: security/Proofs/A.olean changed", "Flock::rest", "lean-audit: FAIL: unread: Core grew"}
    assert audit_failures(log, "rest") == named == audit_failures(log.replace("scratch-1-x", "scratch-22-y"), "rest")
    assert audit_failures("AUDIT: PASS; reports in /out\n", "rest") == set()


# -- a failure's records, read: copies of real checks' records, cut to what the reader reads

TREE = "/workspace/research/scratch/r-x/tree"
BOUNDARY = "tests/test_boundaries.py::test_circuit_check_imports_no_workspace_distribution_but_verity"
ROOT_SUITE_LOG = f"""=================================== FAILURES ===================================
_______ test_circuit_check_imports_no_workspace_distribution_but_verity ________

    def test_circuit_check_imports_no_workspace_distribution_but_verity() -> None:
        packages = _workspace_packages() - {{"circuit_check"}}
>       assert not offenders, ("circuit-check imports `verity` and itself, and no other distribution of the workspace. "
                               "CIRCUIT_CHECK_IMPORTS only shrinks.\\n" + "\\n".join(offenders))
E       AssertionError: circuit-check imports `verity` and itself, and no other distribution of the workspace. CIRCUIT_CHECK_IMPORTS only shrinks.
E         tools/verity/circuit_check/src/circuit_check/structural_slots.py: verity_flock
E       assert not ['tools/verity/circuit_check/src/circuit_check/structural_slots.py: verity_flock']

tests/test_boundaries.py:796: AssertionError
=============================== warnings summary ===============================
tests/lint/test_lean_bar.py::test_no_new_findings
  {TREE}/tests/lint/test_lean_bar.py:261: Stale: 1 of tests/lint/allowlists/lean_bar.json's entries allow more than the tree holds:
    verity/protocols/spot/verifier/lean/Spot/OneStageZk.lean [module-doc]: 1 allowed, 0 found: delete the entry
    held(scan(), allowed(), SOFT)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED {BOUNDARY}
============ 1 failed, 306 passed, 20 warnings in 353.31s (0:05:53) ============
"""
REGRESSION = "tests/regression/test_regression.py::test_reproduces[T0-stoch_value-r101]"
VLLM_LOG = f"""=================================== FAILURES ===================================
_____________________ test_reproduces[T0-stoch_value-r101] _____________________
[gw11] linux -- Python 3.12.14 {TREE}/.venv/bin/python
>       assert not problems, (f"#{{row.number}} {{row.key}}: {{check_name}} does not reproduce {{what}}")
E       AssertionError: #101 llama32-1b__bf16__l40s: stoch_value [T0, strong] does not reproduce the expected contract (expected/<row>.json)
E           $.verdict: expected "PASS" != actual "FAIL"

tests/regression/test_regression.py:206: AssertionError
=========================== short test summary info ============================
FAILED {REGRESSION}
"""
#: a suite whose tests' own runs its machine refused, one test's code quoting a full disk it fakes
REFUSED_LOG = """=================================== FAILURES ===================================
___________ test_a_stage_timeout_ends_the_runs_workers_outside_its_group ___________

        def dump(procs, **kw):
            raise OSError(28, "No space left on device")
>       assert trun.main(argv) == 0
E       AssertionError: assert 75 == 0
----------------------------- Captured stderr call -----------------------------
[tele run] refused to start the workload: this machine caps each login session at MemoryMax=80% (/etc/systemd/system/session-.scope.d), but this run's cgroup /user.slice/check.scope has memory.max max, so its workload would run with no cap of its own.
=========================== short test summary info ============================
FAILED tests/test_telemetry.py::test_a_stage_timeout_ends_the_runs_workers_outside_its_group - AssertionError: assert 75 == 0
"""
SCRATCH = "/workspace/research/cache/verity-check/lean-audit-scratch-{n}"
BUILD_LOG = f"""AUDIT {SCRATCH}/Security/Proofs: FAIL; built in the sandbox
  staged: 868 of its 4967 keyed modules from the artifact cache
  build: exit 1
    warning: ../Proofs/Flock/SessionLoop/Concrete/Inputs.lean:86:6: `dif_pos` has been deprecated: Use `dite_eq_left` instead
    ✔ [14248/14263] Built Proofs.Flock.SessionLoop.Concrete.Sizes ({{s}}s)
    ⚠ [14250/14263] Built Proofs.Flock.SessionLoop.Concrete.Inputs ({{s}}s)
    Some required targets logged failures:
    - Proofs.Flock.SpotFlock.ScopeOf
    error: build failed

AUDIT {SCRATCH}/Security/Properties: FAIL; built in the sandbox
  staged: 560 of its 658 keyed modules from the artifact cache
  sampled: 12 of the 560 modules its build didn't compile, recompiled in {{s}}.1 s (0 equal as object graphs): Spot.Registration
  lean: proved_in: ../Proofs's build did not pass (its report says why), so its facts are not read
AUDIT: FAIL ({SCRATCH}/Security/Proofs, {SCRATCH}/Security/Properties); reports in /workspace/research/runs/r-x/lean-audit
"""
TAMPER_LOG = f"""AUDIT {SCRATCH}/verity/core/devices/certifier/lean: FAIL; built in the sandbox
  tamper: {SCRATCH}/verity/core/service/flock/.lake/build/lib/lean/Flock/Canon.olean changed
  build: exit 1
    error: object file verity/core/service/flock/.lake/build/lib/lean/Flock/Canon.olean of module Flock.Canon does not exist
AUDIT: FAIL ({SCRATCH}/verity/core/devices/certifier/lean); reports in /workspace/research/runs/r-x/lean-audit
"""


def _row(name: str, suite: str, *tests: str, **more) -> dict:
    return {"name": name, "suite": suite, "status": "failed", "step": "pytest", "failed_tests": list(tests),
            "beyond_tests": False, **more}


def test_the_repository_s_suite_failing_a_boundary_test_is_read_as_that_test_naming_its_file_and_the_offender():
    """The repository's own suite failed a boundary test, its report naming the file at fault: the failure is that test,
    named by the test's file and the offender, from the report's failures and not its warnings; its reads are unbounded,
    since the suite's directory is the tree, and the failure is its own."""
    got = read_suite("repository", _row("repository", ".", BOUNDARY), ROOT_SUITE_LOG, "rest", None, frozenset(), TREE)
    assert got == Read(frozenset({f"repository::{BOUNDARY}"}), frozenset(
        {"tests/test_boundaries.py", "tools/verity/circuit_check/src/circuit_check/structural_slots.py"}), None, "own")
    assert read_suite("repository", _row("repository", "."), ROOT_SUITE_LOG, "rest", None).tests == {"repository::rest"}


def test_a_suite_s_failing_test_is_read_by_its_id_and_its_file_from_the_repository_s_root():
    """lean-suites failed one parametrized test of the vLLM suite: read as that test, named by its file under the suite's
    directory, the interpreter's path in the venv left out; its reads are what the suite declares."""
    declared = frozenset({"integrations/vllm", "pyproject.toml"})
    got = read_suite("verity-vllm", _row("verity-vllm", "integrations/vllm", REGRESSION, step="lean-suites"), VLLM_LOG, "rest",
                     declared, frozenset({"integrations/vllm"}), TREE)
    assert got == Read(frozenset({f"verity-vllm::{REGRESSION}"}),
                       frozenset({"integrations/vllm/tests/regression/test_regression.py"}), declared, "own")


def test_a_suite_failed_by_its_machine_is_infra_by_what_its_run_printed_not_by_what_a_test_s_code_quotes():
    row = _row("research", "tools/research", "tests/test_telemetry.py::test_a_stage_timeout_ends_the_runs_workers_outside_its_group")
    assert suite_kind(row, REFUSED_LOG) == "infra"
    assert suite_kind(row, REFUSED_LOG.replace("[tele run] refused", "[tele run] started")) == "own"
    assert suite_kind({**row, "timed_out": 900.0}, REFUSED_LOG) == "timeout"
    assert suite_kind({**row, "nondeterministic": row["failed_tests"]}, "") == "flaky"


def test_a_lean_build_s_failure_is_read_as_its_failed_modules_alike_in_every_run_and_its_reads_their_imports():
    """lean-audit failed Security/Proofs's build on one module and Security/Properties for reading its facts: read as lake's
    failed module and the audit's lines, the same in every run whatever its scratch tree and timings, named by the module's
    file, and said only of a build's failure (`audit_built`); one whose build outputs another build changed is the store's."""
    one, two = BUILD_LOG.format(n="1-x", s=23), BUILD_LOG.format(n="22-y", s=31)
    got = read_step("lean-audit", {"step": "lean-audit", "rc": 1}, one, "rest")
    assert got.tests == read_step("lean-audit", {"step": "lean-audit", "rc": 1}, two, "rest").tests == {
        "Security/Proofs::build: exit 1", "Security/Proofs::- Proofs.Flock.SpotFlock.ScopeOf",
        "Security/Proofs::error: build failed",
        "Security/Properties::lean: proved_in: ../Proofs's build did not pass (its report says why), so its facts are not read"}
    assert got.named == {"Security/Proofs/Flock/SpotFlock/ScopeOf.lean"} and got.kind == "own" and audit_built(one)
    tamper = read_step("lean-audit", {"step": "lean-audit", "rc": 1}, TAMPER_LOG.format(n="1-x"), "rest")
    assert tamper.kind == "store-state" and tamper.reads is None and not audit_built(TAMPER_LOG.format(n="1-x"))


def test_a_lean_build_reads_its_failed_modules_their_imports_in_the_tree_what_they_include_and_their_packages_lake_files(tmp_path):
    files = {"pkg/lakefile.toml": "", "pkg/lean-toolchain": "", "pkg/data/a.json": "{}",
             "pkg/Pkg/A.lean": 'import Pkg.B\nimport Mathlib.Foo\ndef a := include_str "../data/a.json"\n',
             "pkg/Pkg/B.lean": "public import Pkg.C -- and nothing else\n", "pkg/Pkg/C.lean": "def c := 1\n",
             "pkg/Pkg/D.lean": "import Pkg.A\n", "other/E.lean": "def e := 1\n"}
    for p, text in files.items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(text)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "x")
    got = lean_reads(tmp_path, "HEAD", ["pkg/Pkg/A.lean"])
    assert {"pkg/Pkg/A.lean", "pkg/Pkg/B.lean", "pkg/Pkg/C.lean", "pkg/data/a.json", "pkg/lakefile.toml",
            "pkg/lean-toolchain", "tools/verity/lean"} <= got
    assert "pkg/Pkg/D.lean" not in got and "other/E.lean" not in got
    assert lean_reads(tmp_path, "HEAD", ["pkg/Pkg/Z.lean"]) is None and lean_reads(tmp_path, "HEAD", []) is None


def test_a_suite_s_reads_are_narrowed_by_what_its_run_traced_unless_a_program_no_tracer_sees_into_ran():
    """`traced_reads`: the declared paths but their directories, with what the run traced and what its tests read of each
    other; the declared paths whole where the failing module, or a session fixture, ran an untraced program; and either with
    each path read outside the declared ones."""
    declared, dirs = frozenset({"pkg", "shared", "pyproject.toml"}), frozenset({"pkg", "shared"})
    report = {"traced": ["F\tpkg/a.py", "L\tshared/data", "F\tpkg/tests/test_a.py"], "untraced": {},
              "reads": {"pkg/tests/test_a.py": ["pkg/tests/helpers.py"]}, "outside": {}}
    trace = "F\tpkg/tests/test_a.py\tpkg/b.py\nO\tpkg/tests/test_a.py\tdocs/x.md\n"
    got = traced_reads(declared, dirs, "pkg", {"pkg/tests/test_a.py"}, report, trace)
    assert {"pkg/a.py", "pkg/b.py", "shared/data", "pkg/tests/helpers.py", "docs/x.md", "pyproject.toml"} <= got
    assert not meets(("pkg/c.py",), got) and not meets(("shared/y.py",), got)
    for blind in ({"untraced": {"pkg/tests/test_a.py": ["git"]}}, {"untraced": {"*": ["git"]}}, {"traced": None}):
        assert traced_reads(declared, dirs, "pkg", {"pkg/tests/test_a.py"}, {**report, **blind}, trace) == declared | {"docs/x.md"}
    assert traced_reads(declared, dirs, "pkg", {"pkg/tests/test_a.py"}, report, "X\tpkg/tests/test_a.py\t/usr/bin/git\n") == declared
    assert traced_reads(None, dirs, "pkg", {"pkg/tests/test_a.py"}, report, trace) is None


def test_a_suites_guard_report_that_was_never_fetched_or_was_cut_short_reads_as_none(tmp_path):
    """A failed suite's guard report and children's reads beside its log: absent (a suite with none, or a fetch that brought
    none) reads as no report and no reads, which leave its declared inputs whole; so does a report that is empty, cut short
    or not a JSON object, the one a pytest that died in its session finish leaves, as `suites.read_report` reads it."""
    report, trace = tmp_path / "pkg.json", tmp_path / "pkg.reads"
    assert _json(report) is None and _text(trace) == ""
    report.write_text('{"traced": []}')
    trace.write_text("F\tpkg/tests/test_a.py\tpkg/a.py\n")
    assert _json(report) == {"traced": []} and _text(trace) == "F\tpkg/tests/test_a.py\tpkg/a.py\n"
    for bad in ("", '{"traced": [', "[]"):
        report.write_text(bad)
        assert _json(report) is None


@pytest.mark.parametrize("more", ["", RECORD])
def test_a_lean_audit_failure_is_inherited_only_while_it_prints_no_more_than_the_base_s(world, monkeypatch, tmp_path, more):
    """The base's check and the candidate's fail lean-audit on the same error of the same package, each in its own scratch
    tree: inherited by what the audit printed.  One more failure under the package is the candidate's own."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec_base = start(h.d, base, base, shards=1)
    _slim(h, tmp_path, lean_audit=AUDIT.format(n=1))
    assert judge(h.d, rec_base).state == FAIL
    rec = start(h.d, sha, base, shards=1)
    _slim(h, tmp_path, lean_audit=AUDIT.format(n=2) + more)
    (f,) = based(judge(h.d, rec), rec_base).failed
    assert f.key != f.base_key and f.inputs == {"security/Proofs/A.lean"} and _inherited(f) is (not more)
    assert f.tests - f.base_tests == ({"security/Proofs::record: the statement of G changed"} if more else set())
    assert unnamed(f.key) not in f.tests


# -- lean-audit taken from the base's check

#: the steps a check taking lean-audit from its base's is sent
FREE = ",".join(s for s in AFTER_LANDING if s != "lean-audit")


@pytest.mark.parametrize("shards", [1, 3, "auto:2"])
def test_a_check_taking_lean_audit_from_its_base_s_runs_every_step_but_it_and_its_record_says_whose(world, monkeypatch,
                                                                                                    tmp_path, shards):
    """``audit_from``, the candidate the check is built on: every step of `AFTER_LANDING` but lean-audit is sent, to each
    shard alike and to an auto plan's making, and the record says whose verdict on lean-audit it takes (``lean_audit_from``).
    Judged on no lean-audit, it lends none to a check above it (`audits`), where the check of its base, which ran it, does.
    The run beside a graph2 check runs no lean-audit, so it takes none."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on, plans = "m1,m2,m3", Plans(h.d, tmp_path)
    rec_base = start(h.d, base, base, shards=1)
    rec = start(h.d, sha, base, shards=shards, audit_from=base)
    if shards == "auto:2":
        assert plans.asked == [(sha, FREE, None, 2)] and rec["plan"]["deferred"] == FREE and len(h.posted) == 1
    else:
        assert [steps for _, steps, _ in h.posted[1:]] == [FREE] * shards
    assert (rec["lean_audit_from"], rec["gate"], "lean-audit" in rec["steps"]) == (base, CHECK, False)
    assert any(f"as candidate {sha[:12]}'s check, lean-audit {base[:12]}'s: " in x for x in world.logs), world.logs
    assert audits(h.d, rec_base) and not audits(h.d, rec) and not audits(h.d, None)
    with pytest.raises(ValueError, match="only the check that gates"):
        start(h.d, sha, base, audit_from=base)


@pytest.mark.parametrize("says", ["passed", "failed", "running", "no record"])
def test_a_check_that_took_lean_audit_from_its_base_s_passes_it_exactly_when_that_check_passed_it(world, monkeypatch,
                                                                                                   tmp_path, says):
    """The candidate's own check passed every step it ran, and its base, whose check ran lean-audit, has landed: its verdict
    on lean-audit is the one the run beside `main` keeps, here its base's run beside itself.  Where that run passed it, a
    pass; where it failed it, `main` red on lean-audit, a fail on that very failure, ended, which `based` reads as the
    base's; either kept in the candidate's record as if it had run the step, so it answers for it once its candidate is
    `main`, and read from there.  While that run runs, or with none given, as while its base is in flight, a pass is pending
    and a fail not ended, the record unchanged: the candidate never passes on its other steps alone."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec_base = start(h.d, base, base, shards=1)
    if says in ("passed", "failed"):
        _slim(h, tmp_path, **({"lean_audit": AUDIT.format(n=1)} if says == "failed" else {}))
        assert judge(h.d, rec_base).state == (FAIL if says == "failed" else PASS)
    rec = start(h.d, sha, base, shards=1, audit_from=base)
    _slim(h, tmp_path)
    own = judge(h.d, rec)
    assert own == Verdict(PASS) and "lean-audit" not in rec["verdict"]["passed"]
    was, below = json.loads(json.dumps(rec)), None if says == "no record" else rec_base
    got = audited(h.d, own, rec, below)
    if says == "passed":
        assert got == own and rec["verdict"]["passed"] == [*was["verdict"]["passed"], "lean-audit"]
    elif says == "failed":
        (f,) = based(got, rec_base).failed
        assert (got.state, got.ended, f.task, _inherited(f)) == (FAIL, True, "step lean-audit", True)
        assert rec["verdict"]["state"] == FAIL and rec["verdict"]["failed"] == rec_base["verdict"]["failed"]
    else:
        assert got == Verdict(PENDING) and rec == was
        red = Verdict(FAIL, (_failed(h, base, sha, "pytest"),), ended=True)
        assert audited(h.d, red, rec, below) == replace(red, ended=False)
    if says in ("passed", "failed"):
        assert judge(h.d, rec) == got == audited(h.d, judge(h.d, rec), rec, None), "kept, it is read from the record"


def test_where_lean_audit_gates_nothing_no_check_lends_it_and_none_waits_on_its_base_s(world, monkeypatch, tmp_path):
    """drive.toml's warn step gates nothing: a check that ran it lends no verdict on it, and one that took it from its base's
    check doesn't wait on that check's."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec_base = start(h.d, base, base, shards=1)
    rec = start(h.d, sha, base, shards=1, audit_from=base)
    _slim(h, tmp_path)
    own = judge(h.d, rec)
    assert audits(h.d, rec_base) and audited(h.d, own, rec, rec_base) == Verdict(PENDING)
    h.d.warn_steps = ("lean-audit",)
    assert not audits(h.d, rec_base) and audited(h.d, own, rec, rec_base) == own == Verdict(PASS)


def test_a_lean_input_is_a_lean_or_lake_file_or_lock_anywhere_the_check_s_or_lean_check_s_code_or_the_lean_tree_whole():
    """The conservative set a member's change is judged against: a README under security/ counts as much as a Lean file;
    `Security/`, the tree's name before, too; a path that only starts like one of them doesn't, nor the environment's
    files, which every step reads."""
    lean = ["security/README.md", "security/Definitions/Core.lean", "Security/Proofs/README.md", "Main.lean",
            "verity/core/protocols/pous/lean/Grader.lean", "verity/core/service/flock/lakefile.toml",
            "verity/core/service/flock/lakefile.lean", "lean-toolchain", "verity/x/lake-manifest.json",
            "verity/core/protocols/nci/lean/lean-audit.json", "tools/verity/lean/check.py", "tools/verity/check/policy.toml"]
    other = ["docs/security.md", "verity/security/x.py", "tools/verity/checkers/x.py", "tools/verity/lean-tools/x.py",
             "verity/ml/circuits/ops.py", "uv.lock", ".python-version", "lean.md"]
    assert [p for p in [*lean, *other] if meets((p,), LEAN_INPUTS)] == lean


def test_a_tree_s_lean_inputs_hold_each_directory_of_a_lakefile_or_a_lock_whole_where_the_tree_puts_it():
    """An audit reads every file of its package (`lean_audit.inputs`), one a module reads with `include_str` among them, as
    pouw's `check/models.json`: every directory of the tree holding a lakefile or a lock is a Lean input whole, a file the
    tree doesn't hold yet too, and the root's is all of it; a directory beside or above one isn't, nor one named like it."""
    pouw, rand, pous = ("verity/protocols/pouw/verifier/lean/check", "verity/commitments/randomness",
                        "verity/protocols/pous/lean")
    files = [f"{pouw}/lakefile.toml", f"{pouw}/models.json", f"{rand}/lakefile.lean", f"{rand}/__init__.py",
             f"{pous}/lean-audit.json", f"{pous}/check.sh", "verity/protocols/README.md", "verity/protocols/pouw/models.json",
             "a.txt"]
    inputs = lean_inputs(files)
    assert inputs[:len(LEAN_INPUTS)] == LEAN_INPUTS
    lean = [f"{pouw}/models.json", f"{pouw}/tests/pouw_check_cases.py", f"{rand}/__init__.py", f"{rand}/tests/vectors.json",
            f"{pous}/check.sh", f"{pous}/TRUSTED.sha256"]
    other = ["verity/protocols/README.md", "verity/protocols/pouw/models.json", "verity/commitments/randomness2/x.py",
             "verity/protocols/pous/leaner/x.json", "a.txt"]
    assert [p for p in [*lean, *other] if meets((p,), inputs)] == lean
    assert meets(("a.txt",), lean_inputs(["lakefile.toml", "a.txt"])) and not meets(("a.txt",), lean_inputs(["a.txt"]))


def test_failures_are_what_an_ended_run_failed_of_a_step_by_what_its_log_names_once_its_verdict_is_kept(
        world, monkeypatch, tmp_path):
    """`failures` reads a step `judge` leaves out (circuit-check, ratchet-gate), once the run's verdict is kept: each failure
    its log names, the step itself where it names none, none where it passed; None while the run runs, of a record with no
    run, and where the step didn't run to its end (skipped, stopped, killed, or absent from every shard)."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base, shards=1)
    assert failures(h.d, rec, "circuit-check") is None and failures(h.d, None, "circuit-check") is None
    d = _slim(h, tmp_path, circuit_check="FAIL lowering/dead Callee_v1\n", ratchet_gate="  #4 (000000000004): failed: grows\n",
              lean_build="error: it failed\n")
    assert failures(h.d, rec, "circuit-check") is None                    # its verdict isn't kept yet
    assert judge(h.d, rec).state == FAIL
    assert failures(h.d, rec, "circuit-check") == {"lowering/dead Callee_v1": "lowering/dead Callee_v1"}
    assert failures(h.d, rec, "ratchet-gate") == {"#4": "#4 (000000000004): grows"}
    assert failures(h.d, rec, "lean-build") == {"lean-build": "lean-build failed, its log naming no failure"}
    assert failures(h.d, rec, "pytest") == {} and failures(h.d, {**rec, "ids": []}, "pytest") is None
    for how in ({"rc": 137}, {"rc": -9}, {"rc": SIGTERM_RC}, {"status": "skipped"}, {"status": "stopped"}, None):
        _rewrite(d, circuit_check=how)
        assert failures(h.d, rec, "circuit-check") is None, how
    assert failures(h.d, rec, "ratchet-gate") == {"#4": "#4 (000000000004): grows"}


def test_failures_of_a_sharded_run_are_those_of_each_shard_that_failed_the_step_which_a_twin_s_pass_answers_for(
        world, monkeypatch, tmp_path):
    """Of the check that gates, a step is failed in a shard where no twin passed it, and what it failed is what each twin
    that failed it to its end names (not one that skipped it, or that a SIGTERM stopped); with no live record a shard's steps
    are those its steps.jsonl names, every shard having ended."""
    h, base, sha, _ = _harness(world, monkeypatch)
    twins = {"r-1": ({"status": "failed", "rc": 1}, "A_v1"), "r-1b": ({"status": "passed"}, None),
             "r-2": ({"status": "failed", "rc": 1}, "B_v1"), "r-2b": ({"status": "skipped"}, None),
             "r-2c": ({"status": "failed", "rc": SIGTERM_RC}, "C_v1")}
    h.evidence = []
    for rid, (st, target) in twins.items():
        (d := tmp_path / rid).mkdir()
        (d / "steps.jsonl").write_text(json.dumps({"step": "circuit-check", **st, "log": "circuit-check.log"}) + "\n")
        (d / "circuit-check.log").write_text(f"FAIL lowering/dead {target}\n" if target else "")
        h.evidence.append(d)
    rec = {"ids": list(twins), "groups": [["r-1", "r-1b"], ["r-2", "r-2b", "r-2c"]], "tip": sha, "base": base, "gate": CHECK,
           "verdict": {"state": PASS}}
    assert failures(h.d, rec, "circuit-check") == {"lowering/dead B_v1": "lowering/dead B_v1"}
    rec = {**rec, "groups": [["r-1"], ["r-2", "r-2b", "r-2c"]]}
    assert failures(h.d, rec, "circuit-check") == {"lowering/dead A_v1": "lowering/dead A_v1",
                                                   "lowering/dead B_v1": "lowering/dead B_v1"}


# -- content keys


@pytest.mark.parametrize("step", sorted(BOUNDED))
def test_a_steps_key_changes_with_each_file_it_reads_and_with_no_other(world, step):
    """Each bounded step's reads, as `BUILD`, the replica map and `uv lock --check` declare them at the commit keyed: a change
    to any of them changes the key, a change elsewhere keeps it."""
    _move_main(world, KEYED)
    base = _main(world)
    assert isinstance(reads(world.origin, base, step), frozenset)
    touched, untouched = BOUNDED[step]
    for f in (*touched, *untouched):
        assert (key(world.origin, world.change({f: "changed\n"}), step) != key(world.origin, base, step)) is (f in touched), f


def _decided(sha: str, base: str, files: set[str], v: Verdict) -> list:
    """`vqueue.decide`'s actions on candidate ``sha`` of one member, #1 touching ``files``, on ``base``, `main`, its verdict
    ``v`` as the loop reads it once both checks ended (`Verdict.ended`, which the loop carries past `combine`), its failures
    rerun afresh once already (`vqueue.Rerun`), so a failure of its own is blamed on it."""
    return decide(Snapshot(base, (Ready(1, "h1", frozenset(files)),),
                           (Candidate(sha, base, (1,), ("h1",), replace(v, ended=True), reruns=1),)))


def test_a_failed_step_its_base_reads_unchanged_is_inherited_so_decide_blames_no_member(world, monkeypatch, tmp_path):
    """The member touches nothing preflight-lock reads: its key at the candidate is its key at the base, which therefore fails it
    too, and `decide` takes the failure as inherited: no member is suspected or ejected, and the candidate advances under top's
    rule (a candidate whose every failure is inherited advances), its base ringing red before it."""
    h, base, sha, _ = _harness(world, monkeypatch, tree=KEYED, change={"docs/notes.md": "more\n"})
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lock="error: the lock needs to be updated, but `--check` was provided\n")
    v = judge(h.d, rec)
    (f,) = v.failed
    assert f == _failed(h, base, sha, "preflight-lock", f.inputs) and f.key == f.base_key
    acts = _decided(sha, base, {"docs/notes.md"}, v)
    (drop,) = [a for a in acts if isinstance(a, Drop) and a.sha == sha]
    assert drop.outcome.inherited == (f,) and drop.outcome.suspects == ()
    assert not [a for a in acts if isinstance(a, Eject)]
    assert Advance(sha) in acts or [a.failed for a in acts if isinstance(a, Red)] == [(f,)]


@pytest.mark.parametrize("touched", BOUNDED["preflight-lock"][0])
def test_a_member_touching_what_the_failed_step_reads_owns_the_failure(world, monkeypatch, tmp_path, touched):
    h, base, sha, _ = _harness(world, monkeypatch, tree=KEYED, change={touched: "changed\n"})
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lock="error: the lock needs to be updated, but `--check` was provided\n")
    (f,) = (v := judge(h.d, rec)).failed
    assert f.key != f.base_key is not None
    (drop,) = [a for a in _decided(sha, base, {touched}, v) if isinstance(a, Drop) and a.sha == sha]
    assert drop.outcome.inherited == () and drop.outcome.suspects == (1,)


@pytest.mark.parametrize("step", ["preflight-lints", "ratchet-gate"])
def test_a_step_read_whole_is_keyed_on_the_tree_so_its_failure_is_never_inherited_by_its_key(world, monkeypatch, tmp_path, step):
    """preflight-lints' lints read every file, and ratchet-gate judges the merges above the base: a change to a file no
    bounded step reads still changes their key, and the member is suspected."""
    h, base, sha, _ = _harness(world, monkeypatch, off_gate=("circuit-check",), tree=KEYED, change={"docs/notes.md": "more\n"})
    assert reads(h.q.workdir, sha, step) == WHOLE[step]
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, **{step.replace("-", "_"): "failed\n"})
    (f,) = (v := judge(h.d, rec)).failed
    assert f == _failed(h, base, sha, step, f.inputs) and f.key != f.base_key is not None
    (drop,) = [a for a in _decided(sha, base, {"docs/notes.md"}, v) if isinstance(a, Drop) and a.sha == sha]
    assert drop.outcome.inherited == () and drop.outcome.suspects == (1,)


@pytest.mark.parametrize("below", ["failed", "passed", "running", "killed"])
def test_a_step_read_whole_is_inherited_only_when_the_run_beside_the_base_s_check_failed_it_too(world, monkeypatch, tmp_path,
                                                                                                below):
    """Never at its base's key, a failure of a step read whole is compared by its test (`step_test`) with the run beside the
    base's own check (`based`): failed there too, it is inherited, and `decide` blames no member; passed there, it is the
    member's; while that run runs, or once it ended in infra, it is unknown, so the member is suspected as before."""
    h, base, sha, _ = _harness(world, monkeypatch, tree=KEYED, change={"docs/notes.md": "more\n"})
    rec_base = start(h.d, base, base)
    if below == "killed":
        _rewrite(_slim(h, tmp_path, preflight_fixtures="Killed\n"), preflight_fixtures={"rc": 137})
    elif below != "running":
        _slim(h, tmp_path, **({"preflight_lints": LINT_LOG} if below == "failed" else {}))
    assert judge(h.d, rec_base).state == {"failed": FAIL, "passed": PASS, "running": PENDING, "killed": INFRA}[below]
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lints=LINT_LOG)
    assert based(judge(h.d, rec), None) == judge(h.d, rec)
    (f,) = (v := based(judge(h.d, rec), rec_base)).failed
    assert f.tests == {"step:preflight-lints"} and f.key != f.base_key is not None
    assert f.base_tests == {"failed": {"step:preflight-lints"}, "passed": frozenset(), "running": None, "killed": None}[below]
    (drop,) = [a for a in _decided(sha, base, {"docs/notes.md"}, v) if isinstance(a, Drop) and a.sha == sha]
    assert (drop.outcome.inherited, drop.outcome.suspects) == (((f,), ()) if below == "failed" else ((), (1,)))

@pytest.mark.parametrize("build", [None, BUILD_SRC.replace("CHECK_READS = (", "CHECK_READS = tuple(sorted(") + "))\n",
                                   BUILD_SRC.replace('"settings.py"', "SETTINGS"), "VERIFIER = (\n"])
def test_a_tree_not_declaring_lean_agreements_reads_as_plain_data_keys_every_step_on_the_whole_tree(world, build):
    """Without `BUILD`'s declarations as plain data (none, a call, a name, a syntax error), what lean-agreement and check read
    can't be bounded: each step is keyed on the whole tree, and nothing in the declaration is run."""
    _move_main(world, {k: v for k, v in KEYED.items() if k != BUILD} | ({BUILD: build} if build else {}))
    base = _main(world)
    sha = world.change({"docs/notes.md": "more\n"})
    for step in BOUNDED:
        assert BUILD in reads(world.origin, base, step)
        assert key(world.origin, sha, step) != key(world.origin, base, step)


def test_a_key_is_none_for_a_commit_git_cant_read(world):
    assert key(world.origin, "0" * 40, "preflight-lock") is None is key(world.origin, "0" * 40, "preflight-lints")


# -- combine

#: combine's table: graph2's verdict -> the candidate's, for each of the run's in `COLUMNS`
COLUMNS = (PENDING, PASS, FAIL, INFRA)
TABLE = {PENDING: (PENDING, PENDING, FAIL, INFRA),
         PASS: (PENDING, PASS, FAIL, INFRA),
         FAIL: (FAIL, FAIL, FAIL, FAIL),
         INFRA: (INFRA, INFRA, FAIL, INFRA),
         LIST_FAILED: (PENDING, LIST_FAILED, FAIL, INFRA)}


def _verdict(state: str, who: str) -> Verdict:
    """A verdict of ``state``, with a failure and an infra error named for ``who`` where it has them (a fail can hold both)."""
    failed = (Failed(f"{who} task", f"{who} key", frozenset({who})),) if state == FAIL else ()
    infra = (Failed(f"{who} infra", f"{who} infra key", frozenset()),) if state in (FAIL, INFRA) else ()
    return Verdict(state, failed, infra)


@pytest.mark.parametrize("graph", STATES)
@pytest.mark.parametrize("slim", COLUMNS)
def test_combine_takes_the_candidates_state_from_its_table_with_both_verdicts_failures(graph, slim):
    g, s = _verdict(graph, "graph"), _verdict(slim, "slim")
    assert combine(g, s) == Verdict(TABLE[graph][COLUMNS.index(slim)], g.failed + s.failed, g.infra + s.infra)


def test_combine_refuses_a_state_it_has_no_rule_for():
    with pytest.raises(ValueError, match="no verdict has state 'passed'"):
        combine(Verdict(PASS), Verdict("passed"))
