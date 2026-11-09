"""`research.vqueue_slim`: the run beside each candidate of the verified queue of what graph2 doesn't check, launched as drive
launches its run beside the executor's check (`Driver._start_slim`) and judged as drive judges it (`Driver._follow_slim`), on
drive's own fakes: `Harness`'s launches, run states, evidence and clock, `_slim`'s ended run, and the lint log drive names a
culprit by.  Pending while it launches or runs, infra when it registered no run, never ran, was lost or was killed, fail on a
step on the gate it failed, pass on every one it passed; and `combine`'s table."""
from __future__ import annotations

import json

import pytest

from research import merge
from research.drive import ACTIVE, LAUNCH_LOST_S, LOST_AFTER_S, NOT_STARTED, SIGTERM_RC, step_paths, step_records, touches
from research.vqueue import FAIL, INFRA, LIST_FAILED, PASS, PENDING, STATES, Failed, Verdict, meets
from research.vqueue_slim import combine, judge, start
from test_drive_executor import PREFLIGHTS, _slim
from test_drive_policy import _policy
from test_drive_review import GATED_TREE, _main, _move_main
from test_queue import World, _git
from test_queue_drive import LINT_LOG, Harness

#: policy.toml's `off_gate` in the tree: ratchet-gate is off the merge gate
OFF = ("circuit-check", "ratchet-gate")
#: what drive's naming finds of `LINT_LOG`, a failed preflight-lints: its failing test and the path its failure names
LINT_PATHS = frozenset({"tests/test_repository.py", "tools/research/src/research/store/README.md"})


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Drive(Harness):
    """`Harness`, whose launch of a run beside a check gives ``got`` when it is set: a detached launch's record, or the ids and
    tail of a train.sh that printed none."""
    got: dict | tuple[list[str], str] | None = None

    def launch_post(self, repo, commit, machine, steps, **kw):
        if self.got is None:
            return super().launch_post(repo, commit, machine, steps, **kw)
        self.posted.append((commit, steps, kw))
        return self.got


def _harness(world: World, monkeypatch, agreement: bool = False, off_gate: tuple[str, ...] = OFF,
             **kw) -> tuple[Drive, str, str, list]:
    """Drive's harness on `main` under policy.toml's ``off_gate``, candidate ``sha`` one commit above it in the queue's clone, and
    the bases and commits lean-agreement's check asked of (`merge.required_steps`), which finds it changed with ``agreement``."""
    _move_main(world, {**GATED_TREE, "tools/verity/check/policy.toml": _policy(off_gate=off_gate)})
    asked: list[tuple[str, str, str]] = []
    monkeypatch.setattr(merge, "required_steps", lambda repo, base, commit, tool: asked.append((base, commit, tool)) or (
        {"lean-agreement": ["verity/ml/flock/"]} if agreement else {}))
    h = Drive(world, monkeypatch, **kw)
    h.q._clone()
    base, sha = _main(world), world.change({"f.txt": "1\n"})
    _git(h.q.workdir, "fetch", "-q", "origin", f"refs/dev/{sha}")
    return h, base, sha, asked


def _rewrite(d, **steps: dict) -> None:
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


# -- start


@pytest.mark.parametrize("agreement", [False, True])
def test_start_launches_what_drive_launches_beside_the_executors_check_with_main_below_as_its_verified(world, monkeypatch,
                                                                                                      agreement):
    """The candidate's run is drive's: the steps and keywords `Driver._start_slim` launches for it with ``base`` as the verified
    `main`, lean-agreement only when what it checks changed between them, and drive's record of it, with ``base``."""
    h, base, sha, asked = _harness(world, monkeypatch, agreement)
    rec = start(h.d, sha, base)
    steps = ["lean-agreement", "ratchet-gate"] if agreement else ["ratchet-gate"]
    assert h.posted == [(sha, ",".join(steps), {"base": base, "agreement": agreement, "no_cache": False, "circuit_cache": False,
                                                "no_regenerate": True})]
    assert rec == {"ids": ["r-post-0"], "tip": sha, "base": base, "at": h.clock, "steps": steps,
                   **({} if agreement else {"skip": ["lean-agreement"]})}
    assert asked == [(base, sha, "check")]
    assert any(x == f"research queue vqueue: train.sh {sha[:12]} m1 --deferred {','.join(steps)} --base {base[:12]}"
               + (" --agreement" if agreement else "") + f" --no-regenerate beside candidate {sha[:12]}'s check: r-post-0"
               for x in world.logs), world.logs
    pl = {"verified": base}
    h.d._start_slim({}, pl, sha, False)
    assert h.posted[1] == h.posted[0], "drive's own launch of the same run"
    assert pl["slim"]["ratchet_base"] == base
    assert {k: v for k, v in pl["slim"].items() if k not in ("ids", "ratchet_base")} == \
        {k: v for k, v in rec.items() if k not in ("ids", "base")}


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
    assert judge(h.d, rec).infra[0].task.startswith("the run beside its check registered no run: no run ids 60 min after its launch")


@pytest.mark.parametrize("how", ["dry_run", "shadow"])
def test_a_dry_run_or_a_shadow_queue_launches_nothing(world, monkeypatch, how):
    h, base, sha, _ = _harness(world, monkeypatch, **({"shadow": True} if how == "shadow" else {}))
    h.d.dry_run = how == "dry_run"
    assert start(h.d, sha, base) is None and h.posted == []
    assert any(x.startswith(f"research queue vqueue: would launch train.sh {sha[:12]} m1 --deferred ratchet-gate --base")
               for x in world.logs), world.logs


def test_start_needs_drives_machines(world, monkeypatch):
    """The loop's Driver is built with no machines unless it is given drive.toml's `on`, and a train.sh with none registers
    nothing: `start` says so rather than launch it."""
    h, base, sha, _ = _harness(world, monkeypatch)
    h.d.on = ""
    with pytest.raises(ValueError, match="build the Driver with drive.toml's `on`"):
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
    """The loop's Driver outlives its ticks, and `Driver._states` reads each run once a tick: `judge` reads it every call."""
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
    assert judge(h.d, rec) == Verdict(PASS) and rec["verdict"] == {"state": PASS}


def test_a_failed_preflight_fails_the_candidate_with_the_paths_drive_names_its_culprit_by(world, monkeypatch, tmp_path):
    """No graph/v2 task runs the preflights: a failed one fails the candidate, its `Failed` holding what drive's naming finds
    (`step_paths`), the paths the suspects are the members whose changes meet."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, preflight_lints=LINT_LOG)
    (st,) = [st for st in step_records(d) if st["status"] == "failed"]
    assert step_paths(d, st) == LINT_PATHS
    assert judge(h.d, rec) == Verdict(FAIL, (Failed("step preflight-lints", f"{sha}:preflight-lints", LINT_PATHS),))


def test_a_failures_absolute_paths_meet_the_files_drive_matches_them_to(world, monkeypatch, tmp_path):
    """A traceback's frame is an absolute path, which drive matches to a landing's file it ends in (`touches`): the `Failed`'s
    inputs meet the same files."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, preflight_lints='FAILED tests/test_x.py::test_y - KeyError\n  File "/x/site/pkg/conftest.py", line 3\n')
    (st,) = [st for st in step_records(d) if st["status"] == "failed"]
    (failed,) = judge(h.d, rec).failed
    for files in ({"pkg/conftest.py"}, {"site/pkg/conftest.py"}, {"x/site/pkg/conftest.py"}, {"conftest.py"},
                  {"other/pkg/conftest.py"}, {"tests/test_x.py"}, {"tests/test_z.py"}):
        assert meets(files, failed.inputs) is bool(touches(files, step_paths(d, st))), files
    assert meets({"pkg/conftest.py"}, failed.inputs) and not meets({"conftest.py"}, failed.inputs)


@pytest.mark.parametrize("off_gate", [OFF, ("circuit-check",)])
def test_a_red_ratchet_gate_fails_the_candidate_only_on_the_gate(world, monkeypatch, tmp_path, off_gate):
    """ratchet-gate is off the gate in the tree, so its red holds no candidate, as it holds no `main` in drive; on the gate,
    the candidate's own policy.toml says so, it fails it."""
    h, base, sha, _ = _harness(world, monkeypatch, off_gate=off_gate)
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, ratchet_gate="  merge 1 (0123456789ab): failed: grows p01.json 2 -> 3 entries\n")
    assert judge(h.d, rec) == (Verdict(PASS) if "ratchet-gate" in off_gate else
                               Verdict(FAIL, (Failed("step ratchet-gate", f"{sha}:ratchet-gate", frozenset()),)))


@pytest.mark.parametrize("agreement", [True, False])
def test_lean_agreement_is_judged_only_when_the_run_ran_it(world, monkeypatch, tmp_path, agreement):
    """Run, a red lean-agreement fails the candidate, with no path when its log names none; skipped, it isn't judged."""
    h, base, sha, _ = _harness(world, monkeypatch, agreement)
    rec = start(h.d, sha, base)
    d = _slim(h, tmp_path, **({"lean_agreement": "lean-agreement: the verifiers disagree on 3 of 40 vectors\n"} if agreement else {}))
    assert ("lean-agreement" in {st["step"] for st in step_records(d)}) is agreement
    assert judge(h.d, rec) == (Verdict(FAIL, (Failed("step lean-agreement", f"{sha}:lean-agreement", frozenset()),))
                               if agreement else Verdict(PASS))


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
    """The record, saved as JSON, gives the same verdict without reading the run's state or records again."""
    h, base, sha, _ = _harness(world, monkeypatch)
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lints=LINT_LOG)
    got = judge(h.d, rec)
    assert rec["verdict"] == {"state": FAIL, "failed": [{"task": "step preflight-lints", "key": f"{sha}:preflight-lints",
                                                         "inputs": sorted(LINT_PATHS)}]}
    saved = json.loads(json.dumps(rec))
    h.d.states = h.d.evidence = None
    assert judge(h.d, saved) == got == judge(h.d, rec)


def test_start_and_judge_write_nothing_of_drives(world, monkeypatch, tmp_path):
    h, base, sha, _ = _harness(world, monkeypatch)
    before = h.q.load_state()
    rec = start(h.d, sha, base)
    _slim(h, tmp_path, preflight_lints=LINT_LOG)
    assert judge(h.d, rec).state == FAIL
    assert h.q.load_state() == before and h.posts == [] and h.cancelled == []


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
