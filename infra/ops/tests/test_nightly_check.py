"""The control pod's nightly check of `main` (infra/control/nightly_check.py): the command its settings build, when a night
starts, how a run's records read as an outcome, how its verdicts compare part by part with the recorded check's of the same
commit, each outcome's post, rings and label, and what a launch that names no run, a cleanup that fails, a restart or a
refused post leave. The launcher, the fetched records, the comparison, Slack and the clock are stand-ins, and the recorded
checks are made-up records in a store whose remote is a directory: only the clone's tests run a program (git, and a Python
that prints its environment)."""
import calendar
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from research import monitors as M
from research import slack
from research.jobs import dispatch
from research.store import vocab
from research.store.local import LocalStore, NotFound

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"


def module(name):
    spec = importlib.util.spec_from_file_location(name, CONTROL / f"{name}.py")
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)     # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


N = module("nightly_check")
W = module("watchers")

COMMIT = "c0ffee" + "0" * 34
RUN = "r20261019-090212-ab12"
LAUNCHED = f"check: recording at {COMMIT[:12]}\nresearch: run {RUN} launched\n"
LEAD = "<!subteam^S0INFRA|@infra>"
CI = "<!subteam^S0CI|@ci>"
POSTED = "1760000000.000100"
SHARDS = ["r20261019-041500-0a01", "r20261019-041700-0a02"]     # the recorded check of COMMIT: a check in two shards
OLD = "r20261018-221000-0b03"                                   # an older commit's run, whose pass a shard reused
NO_EARLIER = {"passed": [], "failed": []}                       # no part an earlier recorded check judged otherwise than a later one
AGREED = {"runs": SHARDS, "agree": 11, "disagree": [], "unstored": [], "unchecked": [], "earlier": NO_EARLIER}
SAID = (f" Against the recorded check of this commit (`{SHARDS[0]}`, `{SHARDS[1]}`), part by part: %s; circuit-check is one "
        "part, its targets not compared one by one.")
SAID_AGREED = SAID % "11 agree, 0 disagree, 0 with no stored verdict"


def utc(hour, minute=0, day=19):
    return float(calendar.timegm((2026, 10, day, hour, minute, 0)))


def settings(tmp_path, **kw):
    return N.Settings(**{"on": "node-a", "pool": "pool-a", "timeout": 7200, "hour": 9, "work": tmp_path / "work",
                         "state": tmp_path / "nightly-check.json", **kw})


class Fx:
    """A pass's effects, recorded: none starts a program, reaches a machine or posts."""
    tags = {"infra": LEAD, "ci": CI}

    def __init__(self, tmp_path, out=LAUNCHED, rc=0, kept=""):
        self.tmp, self.out, self.rc, self.kept = tmp_path, out, rc, kept    # kept: the launch log as the last launch left it
        self.reads = []                 # what each read of the run gives, in turn: a Read, or an exception it raises
        self.compared = []              # what each comparison gives, in turn: its tally, or an exception it raises; then AGREED
        self.fail = {}                  # an effect's name -> the exceptions it raises, in turn, before it works
        self.during_launch = None
        self.logs, self.launches, self.cleaned, self.posts, self.rings, self.labels = [], [], [], [], [], []
        self.compares, self.rang = [], []

    def _may_fail(self, what):
        if self.fail.get(what):
            raise self.fail[what].pop(0)

    def log(self, msg):
        self.logs.append(msg)

    def prepare(self, day):
        self._may_fail("prepare")
        return COMMIT, self.tmp / "trees" / f"{day}-{COMMIT[:12]}"

    def launch(self, argv, tree):
        self._may_fail("launch")
        self.launches.append((argv, tree))
        if self.during_launch:
            self.during_launch()
        self.kept = self.out
        return self.rc, self.out

    def launch_output(self):
        return self.kept

    def forget_launch(self):
        self.kept = ""

    def cleanup(self, tree):
        self.cleaned.append(tree)
        self._may_fail("cleanup")

    def read(self, run):
        got = self.reads.pop(0) if self.reads else N.Read(None, "running")
        if isinstance(got, Exception):
            raise got
        return got

    def compare(self, run, commit):
        self.compares.append((run, commit))
        got = self.compared.pop(0) if self.compared else AGREED
        if isinstance(got, Exception):
            raise got
        return json.loads(json.dumps(got))

    def post(self, words):
        self._may_fail("post")
        self.posts.append(words)
        return POSTED

    def ring(self, ts, words, name):
        self._may_fail("ring")
        self.rings.append((ts, words))
        self.rang.append(name)
        return f"rang @{name}"

    def label(self, commit, outcome, run):
        self._may_fail("label")
        self.labels.append((commit, outcome, run))
        return f"commit:{commit} nightly_check={outcome}"


class Loop:
    """The loop as `main` runs it: each pass loads the state file, steps at the time given and saves, so every pass here is
    also a restart."""

    def __init__(self, tmp_path, fx=None, **kw):
        self.cfg, self.fx, self.saves = settings(tmp_path, **kw), fx or Fx(tmp_path), []

    def save(self, state):
        N.save_state(self.cfg.state, state)
        self.saves.append(json.loads(json.dumps(state)))

    def __call__(self, now):
        N.step(N.load_state(self.cfg.state), self.cfg, self.fx, now, self.save)
        return N.load_state(self.cfg.state)

    def running(self):
        """A loop whose night of the 19th has launched its run at 09:00Z."""
        self(utc(20, day=18))
        assert self(utc(9))["night"]["phase"] == "running"
        return self


# ---- the settings and the command ---------------------------------------------------------------------------------------------

def test_the_loop_in_watchers_toml_builds_the_nightly_command_from_its_settings():
    """The node, the pool and the run's timeout are the loop's arguments in watchers.toml and nowhere in the code; the command
    they build is the dispatcher's `audit-main` in a pool of its own."""
    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "nightly-check"]
    assert lp.argv[:3] == ["python3.12", "-s", "/data/watchers/src/infra/control/nightly_check.py"]
    cfg, ns = N.parse(lp.argv[3:])
    assert N.command(cfg, "python") == [
        "python", "tools/verity/check/check.py", "--record", "--on", "vy-nebius-cpu-8", "--pool", "nightly",
        "--base", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "--no-cache", "--timeout", "36000"]
    assert N.command(cfg)[0] == sys.executable
    assert cfg.hour == 9 and not ns.once and not ns.dry_run
    assert f"/{M.NIGHTLY_STATE}" == str(cfg.state), "the monitors' nightly-check reads the file the loop writes"
    assert str(cfg.work).startswith("/root/"), "its clone and worktrees are on the pod's own disk, not /data's NFS"
    assert str(cfg.tmp).startswith("/data/"), "check's lean-deps temp dir is several GiB: on the volume"
    assert lp.env["RESEARCH_STORE"].startswith("/root/") and lp.env["RESEARCH_LANE"] == "infra"
    assert lp.env_files == ["/root/.config/verity/relay.env", "/root/.config/verity/r2.env"]

    other = N.Settings("node-b", "pool-b", 60, 3, Path("/w"), Path("/s.json"))
    assert N.command(other, "py")[3:] == ["--on", "node-b", "--pool", "pool-b", "--base", dispatch.EMPTY_TREE, "--no-cache",
                                          "--timeout", "60"]
    theirs = dispatch.command({"kind": "audit-main", "payload": {}}, "node-b", "py", None)
    assert [a for a in N.command(other, "py") if a not in ("--pool", "pool-b", "--timeout", "60")] == theirs, \
        "the dispatcher's audit-main, with a pool and a timeout"
    code = (CONTROL / "nightly_check.py").read_text()
    assert cfg.on not in code and str(cfg.timeout) not in code, "a setting written into the code"


def test_a_dry_run_says_the_command_and_starts_nothing(tmp_path, capsys):
    assert N.main(["--on", "node-a", "--pool", "pool-a", "--timeout", "7200", "--hour", "9", "--work", str(tmp_path / "work"),
                   "--state", str(tmp_path / "state.json"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "tools/verity/check/check.py --record --on node-a --pool pool-a" in out and "--no-cache --timeout 7200" in out
    assert list(tmp_path.iterdir()) == []


# ---- when a night starts --------------------------------------------------------------------------------------------------------

def test_a_night_starts_once_a_day_from_its_hour_and_not_on_a_first_pass_after_it(tmp_path):
    loop = Loop(tmp_path)
    s = loop(utc(20, day=18))
    assert s["since"] == "2026-10-18T20:00:00Z" and s["night"]["phase"] == "done" and "skipped" in s["night"]
    assert loop.fx.launches == [] and "last" not in s, "a loop first started after the hour waits for the next day's"
    loop(utc(8, 59))
    assert loop.fx.launches == []
    s = loop(utc(9))
    assert s["night"] | {"tree": ""} == {"day": "2026-10-19", "phase": "running", "began": "2026-10-19T09:00:00Z",
                                         "commit": COMMIT, "tree": "", "launching": "2026-10-19T09:00:00Z", "run": RUN,
                                         "launched": "2026-10-19T09:00:00Z"}
    (argv, tree), = loop.fx.launches
    assert argv == N.command(loop.cfg) and tree == tmp_path / "trees" / f"2026-10-19-{COMMIT[:12]}"
    loop.fx.reads = [N.Read("passed", "", 61)]
    assert loop(utc(11))["night"]["phase"] == "done"
    for now in (utc(12), utc(18), utc(23, 59), utc(8, 59, day=20)):
        loop(now)
    assert len(loop.fx.launches) == 1 and len(loop.fx.posts) == 1, "one night a day"
    assert loop(utc(9, 4, day=20))["night"]["day"] == "2026-10-20" and len(loop.fx.launches) == 2


def test_a_loop_first_started_before_the_hour_starts_that_day_and_one_down_at_the_hour_starts_when_it_is_back(tmp_path):
    loop = Loop(tmp_path)
    s = loop(utc(7))
    assert s["since"] == "2026-10-19T07:00:00Z" and "night" not in s
    assert loop(utc(9, 2))["night"]["day"] == "2026-10-19" and len(loop.fx.launches) == 1
    loop.fx.reads = [N.Read("passed", "", 61)]
    loop(utc(10))
    s = loop(utc(14, 30, day=20))                    # no pass since the 19th's: the supervisor was down over the 20th's hour
    assert s["night"]["day"] == "2026-10-20" and s["night"]["began"] == "2026-10-20T14:30:00Z" and len(loop.fx.launches) == 2


def test_the_state_is_saved_as_launching_before_the_launch(tmp_path):
    """A loop killed during the launch must find, when it is back, that a launch was under way: else it would launch again."""
    loop, seen = Loop(tmp_path), []
    loop.fx.during_launch = lambda: seen.append(N.load_state(loop.cfg.state)["night"])
    loop.running()
    (night,) = seen
    assert night["phase"] == "launching" and night["commit"] == COMMIT and night["launching"] == "2026-10-19T09:00:00Z"
    assert "run" not in night


# ---- a run's records as an outcome ----------------------------------------------------------------------------------------------

def result(status, steps, **more):
    """check's result.json, as `write_result` writes it."""
    return {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": RUN,
            "validation": {"status": status, "checker": "tools/verity/check/check.py"},
            "steps": [{"step": s, "status": v, "seconds": 1.0} for s, v in steps], **more}


def suite(path, status, **more):
    return {"suite": path, "name": path.rsplit("/", 1)[-1], "status": status, **more}


STEPS_OK = [("pytest", "passed"), ("lean-build", "passed"), ("lean-audit", "passed")]
STEPS_RED = [("pytest", "failed"), ("lean-build", "passed"), ("lean-audit", "failed"), ("flock", "skipped")]
CONTRADICTS = [{"suite": "verity/circuits", "contradicts": ["tests/test_ops.py::test_add", "tests/test_ops.py::test_mul"]}]
RECORDS = {
    "passed": dict(status={"state": "done", "rc": 0, "wall_s": 14520.4}, result=result("passed", STEPS_OK),
                   suites=[suite("tools/research", "passed"), suite("infra/ops", "cached")]),
    "failed": dict(status={"state": "failed", "rc": 1, "wall_s": 5700}, result=result("failed", STEPS_RED),
                   suites=[suite("tools/research", "failed"), suite("verity/circuits", "passed")],
                   lean=[suite("verity/protocols/flock/verifier", "failed")]),
    "contradicted": dict(status={"state": "failed", "rc": 1, "wall_s": 5700},
                         result=result("failed", STEPS_RED[:2], cache_contradictions=CONTRADICTS),
                         suites=[suite("verity/circuits", "failed", contradicts=CONTRADICTS[0]["contradicts"])]),
    "never-started": dict(status={"state": "failed", "rc": 75, "wall_s": 3.0}),
    "timed-out": dict(status={"state": "cancelled", "rc": None, "signal": "SIGTERM", "wall_s": 7200.0},
                      failure={"class": "STAGE_TIMEOUT", "why": "the run's --timeout"}),
    "no-result": dict(status={"state": "failed", "rc": 137, "signal": "SIGKILL", "wall_s": 900.0}),
    "wrong-result": dict(status={"state": "done", "rc": 0, "wall_s": 60.0}, result=result("passed", STEPS_OK) | {"run_id": "r-other"}),
    "runner-gone": dict(status={"state": "running"}, remote={"runner_gone": {"why": "no runner process for it on the machine"}}),
    "launcher-exited": dict(remote={"launcher_exited": {"last_line": "runner: cannot make the run dir"}}),
    "refused": dict(refused={"run_id": RUN, "reason": "DiskFloor: 3 GB free on /workspace", "fault": "Refused"}),
    "running": dict(status={"state": "running"}),
    "submitted": dict(),
}


def records(root, status=None, result=None, suites=None, lean=None, failure=None, refused=None, remote=None):
    """A launched run's local dir as a fetch leaves it."""
    d = root / RUN
    d.mkdir(parents=True)
    docs = {"remote.json": {"machine": "node-a", "host": "node-a.test", "port": 22, "phase": "launched",
                            "remote_run_dir": f"/workspace/research/runs/{RUN}", **(remote or {})},
            "status.json": status, "result.json": result, "suites.json": suites, "suites-lean.json": lean,
            "failure.json": failure, "refused.json": refused}
    for name, doc in docs.items():
        if doc is not None:
            (d / name).write_text(json.dumps(doc))
    return d


def read(tmp_path, name, rec=None):
    cfg, asked = settings(tmp_path), []
    d = records(tmp_path / "runs", **RECORDS[name])
    got = N.read(RUN, cfg, fetch=lambda run_dir, also=(): asked.append((run_dir, also)) or (rec or {}), root=tmp_path / "runs")
    assert asked == [(d, ("suites.json", "suites-lean.json"))], "the run's records are fetched, with check's suite reports"
    return got


def test_a_passed_check_with_nothing_contradicted_is_passed(tmp_path):
    assert read(tmp_path, "passed") == N.Read("passed", "", 242)


def test_a_failed_check_names_each_failed_step_and_suite(tmp_path):
    got = read(tmp_path, "failed")
    assert got == N.Read("failed", "failed steps: `pytest`, `lean-audit`; failed suites: `tools/research`, "
                         "`verity/protocols/flock/verifier`. Its result names no cached pass it contradicts.", 95)


def test_a_cached_pass_the_run_contradicts_outranks_the_verdict(tmp_path):
    got = read(tmp_path, "contradicted")
    assert got.outcome == "contradicted" and got.minutes == 95
    assert got.detail == ("the uncached check failed, and a cached pass answers for what it failed, so a cache key misses an "
                          "input or a test isn't a function of its inputs: `verity/circuits`: tests/test_ops.py::test_add, "
                          "tests/test_ops.py::test_mul. Its failed steps: `pytest`; failed suites: `verity/circuits`.")
    cfg = settings(tmp_path)
    passed = N.judge({"state": "done", "rc": 0, "validation": "passed", "result_valid": True, "wall_s": 60},
                     result("passed", STEPS_OK, cache_contradictions=CONTRADICTS), [], cfg)
    assert passed.outcome == "contradicted" and passed.detail.startswith("the uncached check passed, and a cached pass")
    many = [{"suite": f"s{i}", "contradicts": [f"t{j}" for j in range(5)]} for i in range(11)]
    long = N.judge({"state": "failed", "rc": 1, "validation": "failed", "result_valid": True},
                   result("failed", STEPS_RED, cache_contradictions=many), [], cfg)
    assert "`s7`: t0, t1, t2 and 2 more" in long.detail and "`s8`" not in long.detail and "; and 3 more suites" in long.detail


@pytest.mark.parametrize("name, words", [
    ("never-started", "it never started: no slot of `node-a`'s `pool-a` pool took it (exit 75), so nothing was checked"),
    ("timed-out", "it timed out at its 7200 s"),
    ("no-result", "it ended without a verdict (no verdict: failed rc=137 signal=SIGKILL"),
    ("wrong-result", "it exited 0, but its result.json isn't this run's valid result"),
    ("runner-gone", "it was lost on `node-a`: no runner process for it on the machine (runner-gone)."),
    ("launcher-exited", "it was lost on `node-a`: runner: cannot make the run dir (launcher-exited)."),
    ("refused", "`node-a` refused it, so nothing ran: DiskFloor: 3 GB free on /workspace (no verdict: refused rc=None"),
])
def test_a_run_that_ends_with_no_verdict_is_an_error_and_never_a_failure(tmp_path, name, words):
    got = read(tmp_path, name)
    assert got.outcome == "error" and words in got.detail, got


@pytest.mark.parametrize("name", ["running", "submitted"])
def test_a_run_that_has_not_ended_has_no_outcome(tmp_path, name):
    assert read(tmp_path, name).outcome is None


def test_a_machine_that_cannot_be_reached_is_read_again(tmp_path):
    got = read(tmp_path, "passed", rec={"last_fetch_error": {"rc": 255}})
    assert got == N.Read(None, "`node-a` unreachable (rc 255)"), "what an earlier fetch left is not this pass's reading"


# ---- a run's verdicts against the recorded check's ------------------------------------------------------------------------------

def key(c):
    return c * 64


def report(pkg, **more):
    """One Lean package's report in lean-audit/audit.json: what its check found, and how it ran (RUN_FIELDS)."""
    return {"package": pkg, "failures": [], "axioms": {"propext": 3}, "guarantees": ["G"], "declarations": 12,
            "heartbeats": [{"declaration": "G", "heartbeats": 900, "module_seconds": 4.5}],
            "seconds": 31.5, "sampled": False, "replay": {"seconds": 12.0}, **more}


def audit(*reports, cached=None, failed=()):
    return {"passed": not failed, "failed": list(failed), "cached": cached or {}, "packages": list(reports)}


def step(name, status, **more):
    return {"step": name, "status": status, "seconds": 1, **more}


def uncached():
    """The night's run: every suite but one it skips, both Lean packages and every step but one it skips ran, under no cache."""
    return dict(
        result={"steps": [step("preflight-lock", "passed"), step("preflight-lints", "passed"), step("pytest", "passed"),
                          step("flock-circuit-build", "passed"), step("flock-rust-tests", "passed"), step("circuit-check", "passed"),
                          step("lean-agreement", "passed"), step("lean-audit", "passed"), step("lean-suites", "passed"),
                          step("lean-changed", "skipped")],
                "passes": {key("1"): "lean/x", key("2"): "lean/y"}},
        suites=[suite("pkg/a", "passed", key=key("a")), suite("pkg/b", "passed", key=key("b")), suite("pkg/split", "passed", key=key("c")),
                suite("pkg/traced", "passed", key=key("d")), suite("pkg/gpu", "skipped"), suite("lean/x", "passed", key=key("e"))],
        audit=audit(report("lean/x"), report("lean/y")))


def recorded():
    """The recorded check of the same commit, in two shards, as a check that reuses passes leaves it. It says what the night's
    run says, in other records: `pkg/a` and `lean/x` are passes reused under the night's keys, `pkg/split` ran as two parts
    with a key each, `pkg/traced` is a traced pass under its traced key, the steps cached whole name their keys, and what it
    reused of `lean/x` carries no build time, no replay and a `staged`. Each shard ran the preflight's steps, and none ran
    `lean-agreement`."""
    reused = {"commit": "0" * 40, "run": OLD, "key": key("a")}
    first = dict(
        result={"steps": [step("preflight-lock", "passed"), step("preflight-lints", "passed", cache={"stored": key("7")}),
                          step("pytest", "passed"), step("flock-rust-tests", "passed", cache={"hit": key("8")}),
                          step("lean-agreement", "skipped")],
                "passes": {key("1"): "lean/x"}},
        suites=[suite("pkg/a", "cached", key=key("a"), since=reused), suite("pkg/b", "passed", key=key("b")),
                suite("pkg/split", "passed", key=key("f"), part={"ks": [1], "m": 2}),
                suite("pkg/traced", "cached", key=key("4"), since={**reused, "key": key("4"), "traced": key("5")}),
                suite("pkg/gpu", "skipped")],
        audit=audit(report("lean/x", seconds=0.2, staged=True, replay=None,
                           heartbeats=[{"declaration": "G", "heartbeats": 900, "module_seconds": None}]), cached={"lean/x": key("9")}))
    second = dict(
        result={"steps": [step("preflight-lock", "passed"), step("preflight-lints", "passed", cache={"hit": key("7")}),
                          step("flock-circuit-build", "passed"), step("circuit-check", "passed"), step("lean-audit", "passed"),
                          step("lean-suites", "passed")],
                "passes": {key("2"): "lean/y"}},
        suites=[suite("pkg/split", "passed", key=key("6"), part={"ks": [2], "m": 2}), suite("lean/x", "passed", key=key("e"))],
        audit=audit(report("lean/y", seconds=40)))
    return {SHARDS[0]: first, SHARDS[1]: second}


def tally(mine, theirs):
    return N.tally(N.verdicts(RUN, **mine), {run: N.verdicts(run, **rec) for run, rec in theirs.items()})


def row(rec, path):
    return next(r for r in rec["suites"] if r["suite"] == path)


def package(rec, pkg):
    return next(r for r in rec["audit"]["packages"] if r["package"] == pkg)


def ran(rec, name):
    return next(s for s in rec["result"]["steps"] if s["step"] == name)


def test_a_check_the_recorded_one_agrees_with_agrees_part_by_part_and_a_part_no_shard_ran_has_no_stored_verdict():
    """Five suites, two Lean packages and six steps: the same verdict on each wherever a shard has one, the same key wherever
    both hold the same kind, and the same Lean reports but for how each check ran."""
    got = tally(uncached(), recorded())
    assert got == {"runs": SHARDS, "agree": 12, "disagree": [], "unstored": ["step `lean-agreement`"], "unchecked": [],
                   "earlier": NO_EARLIER}
    assert N.said(got) == SAID % "12 agree, 0 disagree, 1 with no stored verdict (step `lean-agreement`)"
    parts = N.verdicts(RUN, **uncached())
    assert len(parts) == 13 and ("suite", "lean/x") in parts and ("Lean package", "lean/x") in parts, \
        "a Lean suite and the Lean package at its path are two parts"
    assert {name for kind, name in parts if kind == "step"} == {
        "preflight-lock", "preflight-lints", "flock-circuit-build", "flock-rust-tests", "circuit-check", "lean-agreement"}, \
        "every step that ran is a part, but the three whose parts are the suites and the Lean packages"
    assert ("suite", "pkg/gpu") not in parts, "a suite it skipped has no verdict"


def test_a_step_is_a_part_by_rule_so_one_the_check_gains_is_compared_and_one_compared_by_its_parts_is_not():
    """No list here names the steps to compare: a step no test or constant has heard of is a part from the night both checks
    first hold it, and the three steps of N.BY_PART never are, whose verdicts are their suites' and their packages'."""
    mine, theirs = uncached(), recorded()
    mine["result"]["steps"].append(step("a-step-check-gains", "passed"))
    theirs[SHARDS[1]]["result"]["steps"].append(step("a-step-check-gains", "passed"))
    for shard, name in ((0, "pytest"), (1, "lean-suites"), (1, "lean-audit")):
        ran(theirs[SHARDS[shard]], name).update(status="failed")
    got = tally(mine, theirs)
    assert (got["agree"], got["disagree"], got["unstored"]) == (13, [], ["step `lean-agreement`"])
    ran(theirs[SHARDS[1]], "a-step-check-gains").update(status="failed")
    got = tally(mine, theirs)
    assert (got["agree"], got["disagree"]) == (12, [f"step `a-step-check-gains`: passed here; failed in `{SHARDS[1]}`"])
    assert N.BY_PART == ("pytest", "lean-suites", "lean-audit"), "a name added here is a step the comparison stops reading"


A, B, C, D = "a" * 12, "b" * 12, "c" * 12, "d" * 12
PLANTED = {
    "a stored pass against an uncached failure": (
        lambda u, r: row(u, "pkg/a").update(status="failed"),
        f"suite `pkg/a`: failed here, key `{A}`; passed in `{SHARDS[0]}` (a pass `{OLD}` stored)"),
    "a stored failure against an uncached pass": (
        lambda u, r: row(r[SHARDS[0]], "pkg/b").update(status="failed"),
        f"suite `pkg/b`: passed here, key `{B}`; failed in `{SHARDS[0]}`"),
    "a suite's changed key": (
        lambda u, r: row(r[SHARDS[0]], "pkg/b").update(key=key("0")),
        f"suite `pkg/b`: passed here, key `{B}`; passed in `{SHARDS[0]}`, key `{'0' * 12}`"),
    "a reused pass of a suite under another key": (
        lambda u, r: row(r[SHARDS[0]], "pkg/a").update(key=key("0")),
        f"suite `pkg/a`: passed here, key `{A}`; passed in `{SHARDS[0]}` (a pass `{OLD}` stored), key `{'0' * 12}`"),
    "one part of a suite a shard ran in parts": (
        lambda u, r: row(r[SHARDS[1]], "pkg/split").update(status="failed"),
        f"suite `pkg/split`: passed here, key `{C}`; passed in `{SHARDS[0]}`; failed in `{SHARDS[1]}`"),
    "a traced pass against an uncached failure": (
        lambda u, r: row(u, "pkg/traced").update(status="failed"),
        f"suite `pkg/traced`: failed here, key `{D}`; passed in `{SHARDS[0]}` (a traced pass `{OLD}` stored)"),
    "a Lean package's changed inputs key": (
        lambda u, r: r[SHARDS[1]]["result"].update(passes={key("3"): "lean/y"}),
        f"Lean package `lean/y`: passed here, key `{'2' * 12}`; passed in `{SHARDS[1]}`, key `{'3' * 12}`"),
    "a changed Lean report": (
        lambda u, r: package(r[SHARDS[1]], "lean/y").update(axioms={"propext": 3, "Classical.choice": 1}, guarantees=[]),
        f"Lean package `lean/y`: passed here, key `{'2' * 12}`; passed in `{SHARDS[1]}`, its report differing in `axioms`, "
        "`guarantees`"),
    "a Lean report's changed heartbeats": (
        lambda u, r: package(r[SHARDS[0]], "lean/x")["heartbeats"][0].update(heartbeats=901),
        f"Lean package `lean/x`: passed here, key `{'1' * 12}`; passed in `{SHARDS[0]}` (a pass reused from lean-audit's "
        f"cache, where its key is `{'9' * 12}`), its report differing in `heartbeats`"),
    "a reused pass of a Lean package against an uncached failure": (
        lambda u, r: (package(u, "lean/x").update(failures=["an axiom"]), u["result"]["passes"].pop(key("1"))),
        f"Lean package `lean/x`: failed here; passed in `{SHARDS[0]}` (a pass reused from lean-audit's cache, where its key "
        f"is `{'9' * 12}`), key `{'1' * 12}`, its report differing in `failures`"),
    "an uncached failure of a Lean package that left no report": (
        lambda u, r: (u["audit"]["packages"].remove(package(u, "lean/y")), u["audit"]["failed"].append("lean/y"),
                      u["result"]["passes"].pop(key("2"))),
        f"Lean package `lean/y`: failed here; passed in `{SHARDS[1]}`, key `{'2' * 12}`"),
    "an uncached failure of one part of a suite the night's run ran in parts": (
        lambda u, r: u["suites"].extend([suite("pkg/b", "passed", key=key("0"), part={"ks": [2], "m": 3}),
                                         suite("pkg/b", "failed", key=key("3"), part={"ks": [3], "m": 3})]),
        f"suite `pkg/b`: failed here; passed in `{SHARDS[0]}`, key `{B}`"),
    "a Lean package its report lists as failed": (
        lambda u, r: r[SHARDS[1]]["audit"]["failed"].append("lean/y"),
        f"Lean package `lean/y`: passed here, key `{'2' * 12}`; failed in `{SHARDS[1]}`"),
    "a step's stored failure against an uncached pass": (
        lambda u, r: ran(r[SHARDS[1]], "circuit-check").update(status="failed"),
        f"step `circuit-check`: passed here; failed in `{SHARDS[1]}`"),
    "a step's reused pass against an uncached failure": (
        lambda u, r: ran(u, "flock-rust-tests").update(status="failed"),
        f"step `flock-rust-tests`: failed here; passed in `{SHARDS[0]}` (a pass reused under key `{'8' * 12}`)"),
}


@pytest.mark.parametrize("what", PLANTED)
def test_a_planted_difference_is_the_one_part_that_disagrees_named_with_both_verdicts_the_key_and_the_run(what):
    plant, line = PLANTED[what]
    mine, theirs = uncached(), recorded()
    plant(mine, theirs)
    got = tally(mine, theirs)
    assert got == {"runs": SHARDS, "agree": 11, "disagree": [line], "unstored": ["step `lean-agreement`"], "unchecked": [],
                   "earlier": NO_EARLIER}
    assert N.said(got) == SAID % "11 agree, 1 disagree, 1 with no stored verdict (step `lean-agreement`)" + f"\n\u2022 {line}"


def test_how_a_check_ran_is_no_difference_and_several_differences_are_each_named():
    mine, theirs = uncached(), recorded()
    package(theirs[SHARDS[1]], "lean/y").update(seconds=1, sampled=True, staged=True, replay={"seconds": 1}, pipeline="other",
                                                sandbox={"seconds": 2})
    ran(theirs[SHARDS[1]], "circuit-check").update(seconds=9000, cache=None, parts=[{"part": 1}])
    assert tally(mine, theirs)["disagree"] == []
    several = ["a stored pass against an uncached failure", "a suite's changed key", "a stored failure against an uncached pass",
               "one part of a suite a shard ran in parts", "a changed Lean report", "a step's stored failure against an uncached pass"]
    for what in several:
        PLANTED[what][0](mine, theirs)
    got = tally(mine, theirs)
    assert got["disagree"] == [
        PLANTED["a changed Lean report"][1], PLANTED["a step's stored failure against an uncached pass"][1],
        PLANTED["a stored pass against an uncached failure"][1],
        f"suite `pkg/b`: passed here, key `{B}`; failed in `{SHARDS[0]}`, key `{'0' * 12}`",
        PLANTED["one part of a suite a shard ran in parts"][1]], "each part that differs is one line, however many ways it differs"
    assert got["agree"] == 7 and got["unstored"] == ["step `lean-agreement`"]


def test_a_reused_lean_pass_carrying_how_the_run_that_stored_it_ran_agrees_with_the_uncached_one():
    """A recorded check reuses a pass a records run stored, whose report names its build's facts document (`facts_key`, a
    hash of `.olean` bytes, written only under `--facts-cache`), or one a node without the sandbox stored, which says it
    used no replay cache; the uncached run's own node sets its threads and measures its memory. Each says how and where an
    audit ran, not what it found, so the package agrees: under the same inputs key, with the same findings."""
    mine, theirs = uncached(), recorded()
    package(theirs[SHARDS[0]], "lean/x").update(facts_key="c3" * 32, judged_from=key("9"),
                                                 facts_cache="not stored: this node builds without the sandbox")
    package(theirs[SHARDS[1]], "lean/y").update(replay_cache="not used: this node builds without the sandbox, so a build could "
                                                              "have written the cache")
    package(mine, "lean/y").update(build_threads=8, peak_gib={"build": 12.5, "replay": 3.1})
    got = tally(mine, theirs)
    assert (got["agree"], got["disagree"]) == (12, [])
    package(theirs[SHARDS[0]], "lean/x").update(axioms={"propext": 3, "sorryAx": 1})
    assert tally(mine, theirs)["disagree"] == [
        f"Lean package `lean/x`: passed here, key `{'1' * 12}`; passed in `{SHARDS[0]}` (a pass reused from lean-audit's cache, "
        f"where its key is `{'9' * 12}`), its report differing in `axioms`"], "what it found still differs by name"


def test_a_part_on_one_side_only_is_listed_and_neither_agrees_nor_disagrees():
    mine, theirs = uncached(), recorded()
    mine["suites"].append(suite("pkg/new", "passed", key=key("0")))             # no shard has a row for it
    mine["audit"]["packages"].append(report("lean/new"))
    row(theirs[SHARDS[0]], "pkg/gpu").update(status="passed", key=key("3"))     # the night's run skipped it
    theirs[SHARDS[1]]["result"]["steps"].extend([step("lean-agreement", "skipped"), step("lean-changed", "passed")])
    got = tally(mine, theirs)
    assert got == {"runs": SHARDS, "agree": 12, "disagree": [], "unchecked": ["step `lean-changed`", "suite `pkg/gpu`"],
                   "unstored": ["Lean package `lean/new`", "step `lean-agreement`", "suite `pkg/new`"], "earlier": NO_EARLIER}
    assert N.said(got) == SAID % ("12 agree, 0 disagree, 3 with no stored verdict (Lean package `lean/new`, step `lean-agreement`, "
                                  "suite `pkg/new`)") + (" Only the recorded check has a verdict on step `lean-changed`, suite "
                                                         "`pkg/gpu`.")
    assert N.rung({"outcome": "passed", "compared": got}) == []


AGAIN = "r20261019-043000-0a04"                                 # the first shard of the recorded check, run again later
ONE_PART = "; circuit-check is one part, its targets not compared one by one."


def tried_again():
    """The recorded check with its first shard run again: the two shards as `recorded` has them, and the later run."""
    return {**recorded(), AGAIN: recorded()[SHARDS[0]]}


def test_the_newest_recorded_verdict_on_a_part_counts_and_an_earlier_try_that_failed_is_named_and_rings_nobody():
    """A shard that failed a suite and a step and passed both when it was run again, against a night that passed them: the
    later verdicts are the recorded ones, so every part agrees, and the first try is one sentence of the post."""
    mine, theirs = uncached(), tried_again()
    row(theirs[SHARDS[0]], "pkg/b").update(status="failed")
    ran(theirs[SHARDS[0]], "flock-rust-tests").update(status="failed", cache=None)
    got = tally(mine, theirs)
    assert got == {"runs": [*SHARDS, AGAIN], "agree": 12, "disagree": [], "unstored": ["step `lean-agreement`"], "unchecked": [],
                   "earlier": {"failed": ["step `flock-rust-tests`", "suite `pkg/b`"], "passed": []}}
    assert N.said(got).endswith(f"{ONE_PART} 2 parts failed in an earlier recorded check of this commit and passed in a later "
                                "one: step `flock-rust-tests`, suite `pkg/b`.")
    assert N.rung({"outcome": "passed", "compared": got}) == []


def test_the_newest_recorded_verdict_that_failed_against_a_night_that_passed_disagrees_and_rings_infra_and_ci():
    mine, theirs = uncached(), tried_again()
    row(theirs[AGAIN], "pkg/b").update(status="failed")                         # its first try passed the suite
    got = tally(mine, theirs)
    line = f"suite `pkg/b`: passed here, key `{B}`; failed in `{AGAIN}`"
    assert (got["agree"], got["disagree"], got["earlier"]) == (11, [line], {"passed": ["suite `pkg/b`"], "failed": []})
    assert N.said(got).endswith(f"{ONE_PART} 1 part passed in an earlier recorded check of this commit and failed in a later "
                                f"one: suite `pkg/b`.\n\u2022 {line}")
    assert N.rung({"outcome": "passed", "compared": got}) == ["infra", "ci"]


def test_a_part_only_an_older_recorded_check_holds_has_that_checks_verdict():
    mine, theirs = uncached(), tried_again()
    theirs[AGAIN]["suites"].remove(row(theirs[AGAIN], "pkg/b"))                 # the later run holds no row for the suite
    got = tally(mine, theirs)
    assert (got["agree"], got["unstored"], got["earlier"]) == (12, ["step `lean-agreement`"], NO_EARLIER)
    row(theirs[SHARDS[0]], "pkg/b").update(status="failed")
    got = tally(mine, theirs)
    assert got["disagree"] == [f"suite `pkg/b`: passed here, key `{B}`; failed in `{SHARDS[0]}`"] and got["earlier"] == NO_EARLIER


def test_a_suite_recorded_in_parts_has_a_recorded_verdict_for_each_part_whichever_shard_ran_it():
    """`pkg/split`'s two parts are in the check's two shards. The part the later shard ran is no later try of the part the
    earlier shard ran: a part that failed in the shard launched first disagrees, and only a later run of that same part
    makes its failure an earlier try."""
    mine, theirs = uncached(), recorded()
    row(theirs[SHARDS[0]], "pkg/split").update(status="failed")
    got = tally(mine, theirs)
    assert got["disagree"] == [f"suite `pkg/split`: passed here, key `{C}`; failed in `{SHARDS[0]}`; passed in `{SHARDS[1]}`"]
    assert got["earlier"] == NO_EARLIER
    theirs[AGAIN] = recorded()[SHARDS[0]]
    got = tally(mine, theirs)
    assert (got["agree"], got["disagree"], got["earlier"]) == (12, [], {"failed": ["suite `pkg/split`"], "passed": []})


def test_a_commit_with_no_recorded_check_is_said_to_have_none_and_never_to_agree():
    got = tally(uncached(), {})
    assert got["runs"] == [] and got["agree"] == 0 and got["disagree"] == [] and len(got["unstored"]) == 13
    assert N.said(got) == " The store holds no recorded check of this commit, so its verdicts were compared with none."
    assert N.rung({"outcome": "passed", "compared": got}) == []


# ---- each outcome's post, ring and label ----------------------------------------------------------------------------------------

def night_of(tmp_path, got):
    loop = Loop(tmp_path).running()
    loop.fx.reads = [got]
    return loop, loop(utc(13, 2))


def test_a_pass_that_agrees_is_one_post_that_rings_nobody_and_labels_the_commit(tmp_path):
    loop, s = night_of(tmp_path, N.Read("passed", "", 242))
    assert loop.fx.posts == [f"Nightly check of `main` passed: `{COMMIT[:12]}`, run `{RUN}`, 242 min.{SAID_AGREED}"]
    assert loop.fx.rings == [] and loop.fx.labels == [(COMMIT, "passed", RUN)] and loop.fx.compares == [(RUN, COMMIT)]
    assert s["night"]["phase"] == "done" and "ring" not in s["night"] and "ring_ci" not in s["night"]
    assert s["night"]["compared"] == AGREED
    assert s["last"] == {"day": "2026-10-19", "commit": COMMIT, "run": RUN, "outcome": "passed", "at": "2026-10-19T13:02:00Z"}


@pytest.mark.parametrize("outcome, title", [("failed", "failed"), ("contradicted", "contradicts the cache"),
                                            ("error", "gave no verdict")])
def test_anything_but_a_pass_says_what_happened_rings_infra_and_labels_the_commit(tmp_path, outcome, title):
    """A check that gave a verdict is compared, and one that gave none has nothing to compare; neither rings ci while every
    part agrees."""
    loop, s = night_of(tmp_path, N.Read(outcome, "failed steps: `pytest`.", 95))
    compared = outcome != "error"
    post = (f"{LEAD} *Nightly check of `main` {title}* at `{COMMIT[:12]}` (run `{RUN}`, 95 min): failed steps: `pytest`."
            + (SAID_AGREED if compared else ""))
    assert loop.fx.posts == [post] and loop.fx.rings == [(POSTED, post)] and loop.fx.labels == [(COMMIT, outcome, RUN)]
    assert loop.fx.rang == ["infra"] and loop.fx.compares == ([(RUN, COMMIT)] if compared else [])
    assert s["night"]["phase"] == "done" and s["last"]["outcome"] == outcome and ("compared" in s["night"]) == compared
    assert s["night"]["post"] == POSTED and s["night"]["ring"] == "rang @infra" and "ring_ci" not in s["night"]


def test_every_outcome_is_a_value_of_the_label(tmp_path):
    assert set(N.OUTCOMES) == set(vocab.NIGHTLY_CHECK_OUTCOMES)
    for name in RECORDS:
        got = N.read(RUN, settings(tmp_path), fetch=lambda d, also=(): {}, root=records(tmp_path / name, **RECORDS[name]).parent)
        assert got.outcome is None or vocab.check(vocab.NIGHTLY_CHECK, got.outcome) is None, name


DISAGREES = {**AGREED, "agree": 9, "disagree": [PLANTED["a stored pass against an uncached failure"][1],
                                                PLANTED["a step's stored failure against an uncached pass"][1]]}


@pytest.mark.parametrize("outcome, title, detail", [("passed", "passed", ""), ("failed", "failed", "failed steps: `pytest`."),
                                                    ("contradicted", "contradicts the cache", "a cached pass answers.")])
def test_a_part_that_disagrees_is_named_and_rings_infra_and_ci_whatever_the_outcome(tmp_path, outcome, title, detail):
    """The comparison changes neither the outcome nor the label: a check that passed and disagrees is still `passed`."""
    loop = Loop(tmp_path).running()
    loop.fx.reads, loop.fx.compared = [N.Read(outcome, detail, 95)], [DISAGREES]
    s = loop(utc(13, 2))
    post = (f"{LEAD} {CI} *Nightly check of `main` {title}, and disagrees with the recorded check* at `{COMMIT[:12]}` "
            f"(run `{RUN}`, 95 min):" + (f" {detail}" if detail else "") + SAID % "9 agree, 2 disagree, 0 with no stored verdict"
            + f"\n\u2022 suite `pkg/a`: failed here, key `{A}`; passed in `{SHARDS[0]}` (a pass `{OLD}` stored)"
            + f"\n\u2022 step `circuit-check`: passed here; failed in `{SHARDS[1]}`")
    assert loop.fx.posts == [post] and loop.fx.rings == [(POSTED, post)] * 2 and loop.fx.rang == ["infra", "ci"]
    assert s["night"]["ring"] == "rang @infra" and s["night"]["ring_ci"] == "rang @ci" and s["night"]["phase"] == "done"
    assert loop.fx.labels == [(COMMIT, outcome, RUN)] and s["last"]["outcome"] == outcome
    loop(utc(13, 7))
    assert len(loop.fx.posts) == 1 and loop.fx.rang == ["infra", "ci"] and len(loop.fx.compares) == 1, "compared and said once"


def test_a_part_with_no_stored_verdict_is_listed_and_rings_nobody(tmp_path):
    loop = Loop(tmp_path).running()
    loop.fx.reads = [N.Read("passed", "", 242)]
    loop.fx.compared = [{**AGREED, "unstored": ["step `lean-agreement`"], "unchecked": ["suite `pkg/gpu`"]}]
    s = loop(utc(13, 2))
    assert loop.fx.posts == [f"Nightly check of `main` passed: `{COMMIT[:12]}`, run `{RUN}`, 242 min."
                             + SAID % "11 agree, 0 disagree, 1 with no stored verdict (step `lean-agreement`)"
                             + " Only the recorded check has a verdict on suite `pkg/gpu`."]
    assert loop.fx.rings == [] and s["night"]["phase"] == "done" and loop.fx.labels == [(COMMIT, "passed", RUN)]


def test_a_part_an_earlier_recorded_check_failed_and_a_later_one_passed_is_said_in_the_post_and_rings_nobody(tmp_path):
    loop = Loop(tmp_path).running()
    loop.fx.reads = [N.Read("passed", "", 242)]
    loop.fx.compared = [{**AGREED, "earlier": {"failed": ["suite `pkg/b`"], "passed": []}}]
    s = loop(utc(13, 2))
    assert loop.fx.posts == [f"Nightly check of `main` passed: `{COMMIT[:12]}`, run `{RUN}`, 242 min.{SAID_AGREED} 1 part failed "
                             "in an earlier recorded check of this commit and passed in a later one: suite `pkg/b`."]
    assert loop.fx.rings == [] and s["night"]["phase"] == "done" and loop.fx.labels == [(COMMIT, "passed", RUN)]
    assert "ring" not in s["night"] and "ring_ci" not in s["night"]


def test_a_night_whose_commit_has_no_recorded_check_says_so_and_is_not_said_to_agree(tmp_path):
    loop = Loop(tmp_path).running()
    loop.fx.reads = [N.Read("passed", "", 242)]
    loop.fx.compared = [{"runs": [], "agree": 0, "disagree": [], "unstored": ["suite `pkg/a`", "suite `pkg/b`"], "unchecked": [],
                         "earlier": NO_EARLIER}]
    s = loop(utc(13, 2))
    assert loop.fx.posts == [f"Nightly check of `main` passed: `{COMMIT[:12]}`, run `{RUN}`, 242 min. The store holds no recorded "
                             "check of this commit, so its verdicts were compared with none."]
    assert "agree" not in loop.fx.posts[0] and loop.fx.rings == [] and s["night"]["phase"] == "done"


def test_a_comparison_that_cannot_be_read_is_read_again_each_pass_and_the_night_is_compared_once(tmp_path):
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads = [N.Read("passed", "", 61)]
    fx.compared = [ConnectionError("the run's machine is unreachable (rc 255)"), OSError("the store's remote: timed out")]
    s = loop(utc(11))
    assert s["night"]["phase"] == "comparing" and "compared" not in s["night"] and (fx.posts, fx.labels) == ([], [])
    assert s["last"]["outcome"] == "passed", "the outcome stands from when the run was read, whatever the comparison does"
    assert loop(utc(11, 5))["night"]["phase"] == "comparing" and fx.posts == []
    s = loop(utc(11, 10))
    assert s["night"]["phase"] == "done" and s["night"]["compared"] == AGREED and fx.compares == [(RUN, COMMIT)] * 3
    assert fx.posts == [f"Nightly check of `main` passed: `{COMMIT[:12]}`, run `{RUN}`, 61 min.{SAID_AGREED}"] and fx.rings == []
    assert sum("trying again next pass" in ln and "aren't compared" in ln for ln in fx.logs) == 2
    loop(utc(11, 15))
    assert len(fx.compares) == 3 and len(fx.posts) == 1


def test_a_comparison_unread_for_an_hour_is_said_and_rings_infra_and_the_outcome_is_reported_without_it(tmp_path):
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads = [N.Read("passed", "", 61)]
    fx.compared = [OSError("the store's remote: timed out")] * 13
    for minute in range(0, 60, 5):
        assert loop(utc(11, minute))["night"]["phase"] == "comparing"
    assert fx.posts == [] and fx.labels == []
    s = loop(utc(12))
    post = (f"{LEAD} *Nightly check of `main` passed, and was not compared with the recorded check* at `{COMMIT[:12]}` "
            f"(run `{RUN}`, 61 min): Its verdicts could not be compared with the recorded check's: OSError: the store's remote: "
            "timed out.")
    assert fx.posts == [post] and fx.rings == [(POSTED, post)] and fx.rang == ["infra"] and fx.labels == [(COMMIT, "passed", RUN)]
    assert s["night"]["compared"] == {"unread": "OSError: the store's remote: timed out"} and s["night"]["phase"] == "done"
    assert N.COMPARE_S == 3600 and len(fx.compares) == 13


def test_a_night_reported_before_the_loop_compared_reports_as_it_did(tmp_path):
    """A loop restarted on this code with a night still reporting: its state holds no comparison, and it says none."""
    loop = Loop(tmp_path)
    N.save_state(loop.cfg.state, {"schema": N.SCHEMA, "since": "2026-10-18T20:00:00Z", "night": {
        "day": "2026-10-19", "phase": "reporting", "began": "2026-10-19T09:00:00Z", "commit": COMMIT, "run": RUN,
        "outcome": "failed", "detail": "failed steps: `pytest`.", "minutes": 95, "at": "2026-10-19T11:00:00Z", "post": POSTED}})
    s = loop(utc(11, 5))
    assert s["night"]["phase"] == "done" and loop.fx.posts == [] and loop.fx.rang == ["infra"] and loop.fx.compares == []
    assert loop.fx.rings == [(POSTED, f"{LEAD} *Nightly check of `main` failed* at `{COMMIT[:12]}` (run `{RUN}`, 95 min): "
                              "failed steps: `pytest`.")]


def test_a_launch_that_names_no_run_is_an_error_with_its_last_lines(tmp_path):
    """Nothing is known to run, so there is nothing to wait for: the night ends at once, said and rung."""
    fx = Fx(tmp_path, out="check: preflight\ncheck.py: error: --pool: no pool `pool-a` on node-a\n", rc=2)
    loop = Loop(tmp_path, fx)
    loop(utc(20, day=18))
    s = loop(utc(9))
    post = (f"{LEAD} *Nightly check of `main` gave no verdict* at `{COMMIT[:12]}` (no run): the launch named no run (exit 2), so "
            "nothing is known to run: check: preflight | check.py: error: --pool: no pool `pool-a` on node-a")
    assert fx.posts == [post] and fx.rings == [(POSTED, post)] and fx.labels == [(COMMIT, "error", None)]
    assert s["night"]["phase"] == "done" and s["last"]["outcome"] == "error" and s["last"]["run"] is None
    assert fx.cleaned == [tmp_path / "trees" / f"2026-10-19-{COMMIT[:12]}"]
    loop(utc(9, 5))
    assert len(fx.launches) == 1 and len(fx.posts) == 1, "not launched again that day"

    quiet = Loop(tmp_path / "quiet", Fx(tmp_path, out="", rc=0))
    quiet(utc(20, day=18))
    assert "the launch named no run (exit 0), so nothing is known to run: it printed nothing" in quiet(utc(9))["night"]["detail"]


def test_a_launch_that_cannot_start_is_an_error(tmp_path):
    fx = Fx(tmp_path)
    fx.fail["launch"] = [FileNotFoundError(2, "No such file or directory: 'tools/verity/check/check.py'")]
    loop = Loop(tmp_path, fx)
    loop(utc(20, day=18))
    s = loop(utc(9))
    assert s["last"]["outcome"] == "error" and "the launch could not start, so nothing runs: FileNotFoundError" in fx.posts[0]
    assert fx.cleaned and fx.labels == [(COMMIT, "error", None)]


def test_a_cleanup_that_fails_changes_nothing_of_the_night(tmp_path):
    """Removing the worktree can fail (a file still open, a slow disk): the run was launched all the same, and the night
    reads and reports it exactly as after a clean removal."""
    nights = {}
    for name, errors in (("clean", []), ("failing", [OSError(39, "Directory not empty")]), ("odd", [RuntimeError("git gone")])):
        fx = Fx(tmp_path / name)
        fx.fail["cleanup"] = list(errors)
        loop = Loop(tmp_path / name, fx).running()
        fx.reads = [N.Read("passed", "", 242)]
        s = loop(utc(13, 2))
        s["night"].pop("tree")
        nights[name] = (s["night"], s["last"], fx.posts, fx.rings, fx.labels, len(fx.cleaned))
        assert any("was not removed" in ln and "the night goes on" in ln for ln in fx.logs) == bool(errors)
    assert nights["clean"] == nights["failing"] == nights["odd"]
    assert nights["clean"][1]["outcome"] == "passed" and nights["clean"][5] == 1


# ---- restarts, a lost run, an origin that can't be reached ---------------------------------------------------------------------

def launching(tmp_path, out):
    """A loop back after it was killed during the launch: the state says `launching`, and the launch's log is what it has."""
    fx = Fx(tmp_path, kept=out)
    loop = Loop(tmp_path, fx)
    N.save_state(loop.cfg.state, {"schema": N.SCHEMA, "since": "2026-10-18T20:00:00Z", "night": {
        "day": "2026-10-19", "phase": "launching", "began": "2026-10-19T09:00:00Z", "commit": COMMIT,
        "tree": str(tmp_path / "trees" / "t"), "launching": "2026-10-19T09:00:00Z"}})
    return loop, fx


def test_a_loop_restarted_during_the_launch_reads_its_log_and_never_launches_again(tmp_path):
    loop, fx = launching(tmp_path, "check: sending the source\n")
    assert loop(utc(9, 3))["night"]["phase"] == "launching" and fx.cleaned == [], "the launch may still be using its tree"
    fx.kept = LAUNCHED
    s = loop(utc(9, 8))
    assert s["night"]["phase"] == "running" and s["night"]["run"] == RUN and s["night"]["launched"] == "2026-10-19T09:08:00Z"
    assert fx.launches == [] and len(fx.cleaned) == 1


def test_a_loop_killed_as_its_launch_starts_never_takes_the_last_nights_run_for_this_nights(tmp_path):
    """The launch log still holds the last night's `run R launched` until this night's launch writes its own. A loop that
    dies in between is back in `launching` with that log: it must find it empty, never adopt the run that already ended."""
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads = [N.Read("passed", "", 61)]
    assert loop(utc(11))["last"]["run"] == RUN and fx.launch_output() == LAUNCHED
    fx.fail["launch"] = [KeyboardInterrupt()]                # the loop dies as the launch starts, after its state was saved
    with pytest.raises(KeyboardInterrupt):
        loop(utc(9, day=20))
    night = N.load_state(loop.cfg.state)["night"]
    assert night["day"] == "2026-10-20" and night["phase"] == "launching" and fx.launch_output() == ""
    s = loop(utc(9, 5, day=20))
    assert s["night"]["phase"] == "launching" and "run" not in s["night"] and len(fx.posts) == 1
    s = loop(utc(9, 30, day=20))
    assert s["last"]["day"] == "2026-10-20" and s["last"]["outcome"] == "error" and s["last"]["run"] is None
    assert "the launch named no run (the loop was restarted during it)" in fx.posts[1] and len(fx.launches) == 1


def test_a_launch_cut_by_a_restart_that_never_names_a_run_is_an_error_once_its_time_is_up(tmp_path):
    loop, fx = launching(tmp_path, "check: sending the source\n")
    loop(utc(9, 29))
    assert fx.posts == []
    s = loop(utc(9, 30))
    assert dispatch.LAUNCH_TIMEOUT_S == 30 * 60 and s["last"]["outcome"] == "error" and fx.launches == []
    assert "the launch named no run (the loop was restarted during it)" in fx.posts[0] and len(fx.rings) == 1


def test_a_run_that_never_ends_is_lost_an_hour_after_its_timeout_and_a_read_that_raises_is_read_again(tmp_path):
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads = [ConnectionError("ssh: connection reset"), FileNotFoundError(f"no run matching '{RUN}'")]
    loop(utc(9, 5))
    s = loop(utc(9, 10))
    assert s["night"]["phase"] == "running" and sum("could not be read" in ln for ln in fx.logs) == 2 and fx.posts == []
    assert loop(utc(11, 59))["night"]["phase"] == "running", "under its 7200 s timeout and the hour after it"
    s = loop(utc(12))
    assert s["last"]["outcome"] == "error" and s["last"]["run"] == RUN and fx.labels == [(COMMIT, "error", RUN)]
    assert f"it was lost: not ended 60 min after its 7200 s timeout (running). `research status {RUN}`" in fx.posts[0]


def test_an_origin_that_cannot_be_reached_is_tried_each_pass_and_an_error_after_an_hour(tmp_path):
    fx = Fx(tmp_path)
    fx.fail["prepare"] = [subprocess.SubprocessError("git fetch exited 128: could not resolve host")] * 3
    loop = Loop(tmp_path, fx)
    loop(utc(20, day=18))
    for minute in (0, 5, 10):
        assert loop(utc(9, minute))["night"]["phase"] == "preparing" and fx.launches == []
    assert loop(utc(9, 15))["night"]["phase"] == "running" and len(fx.launches) == 1, "origin is back within the hour"

    fx = Fx(tmp_path)
    fx.fail["prepare"] = [OSError(28, "No space left on device")] * 20
    loop = Loop(tmp_path / "down", fx)
    loop(utc(20, day=18))
    for minute in range(0, 60, 5):
        loop(utc(9, minute))
    assert fx.posts == []
    s = loop(utc(10))
    assert s["last"] == {"day": "2026-10-19", "commit": None, "run": None, "outcome": "error", "at": "2026-10-19T10:00:00Z"}
    assert fx.posts == [f"{LEAD} *Nightly check of `main` gave no verdict* at a commit it never read (no run): `main`'s tree "
                        "could not be prepared for 60 min, so nothing was launched: OSError: [Errno 28] No space left on device"]
    assert len(fx.rings) == 1 and fx.labels == [] and s["night"]["phase"] == "done", "no commit is known, so none is labelled"


# ---- the report, once each ------------------------------------------------------------------------------------------------------

def test_a_refused_post_is_tried_again_and_nothing_is_said_twice(tmp_path):
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads = [N.Read("failed", "failed steps: `pytest`.", 95)]
    fx.fail = {"post": [slack.SlackError("ratelimited")], "ring": [slack.SlackError("ratelimited")] * 2,
               "label": [OSError("store: read-only")]}
    s = loop(utc(11))
    assert s["night"]["phase"] == "reporting" and fx.posts == [] and fx.rings == [] and fx.labels == []
    assert s["last"]["outcome"] == "failed", "the outcome stands from when it was read, whatever Slack says"
    s = loop(utc(11, 5))
    assert len(fx.posts) == 1 and fx.rings == [] and fx.labels == [(COMMIT, "failed", RUN)] and s["night"]["phase"] == "reporting"
    s = loop(utc(11, 10))
    assert s["night"]["phase"] == "reporting" and len(fx.posts) == 1
    s = loop(utc(11, 15))
    assert s["night"]["phase"] == "done" and len(fx.posts) == 1 and len(fx.rings) == 1 and len(fx.labels) == 1
    loop(utc(11, 20))
    assert (len(fx.posts), len(fx.rings), len(fx.labels)) == (1, 1, 1)


def test_a_refused_ring_of_one_inbox_is_tried_again_and_neither_inbox_is_rung_twice(tmp_path):
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads, fx.compared = [N.Read("passed", "", 61)], [DISAGREES]
    fx.fail["ring"] = [slack.SlackError("ratelimited")]
    s = loop(utc(11))
    assert s["night"]["phase"] == "reporting" and fx.rang == ["ci"] and "ring" not in s["night"] and N.owed(s["night"]) == ["ring"]
    s = loop(utc(11, 5))
    assert s["night"]["phase"] == "done" and fx.rang == ["ci", "infra"] and len(fx.posts) == 1 and len(fx.labels) == 1
    loop(utc(11, 10))
    assert fx.rang == ["ci", "infra"] and len(fx.compares) == 1


def test_a_report_still_owed_gives_way_to_the_next_night(tmp_path):
    loop = Loop(tmp_path).running()
    fx = loop.fx
    fx.reads = [N.Read("passed", "", 61)]
    fx.fail["post"] = [slack.SlackError("invalid_auth")] * 400
    assert loop(utc(11))["night"]["phase"] == "reporting"
    assert loop(utc(8, 55, day=20))["night"]["day"] == "2026-10-19"
    s = loop(utc(9, day=20))
    assert s["night"]["day"] == "2026-10-20" and s["night"]["phase"] == "running" and len(fx.launches) == 2
    assert any("its report still lacks post: given up" in ln for ln in fx.logs)
    assert s["last"]["day"] == "2026-10-19" and s["last"]["outcome"] == "passed" and fx.labels == [(COMMIT, "passed", RUN)]


# ---- the guard reads what the loop writes --------------------------------------------------------------------------------------

def test_the_monitors_guard_reads_the_loops_state(tmp_path):
    """The 36 hours are counted from the state this loop writes: quiet after a night, whatever its outcome, and found once
    no night has ended for 36 h, with the one in progress named."""
    root = tmp_path / "pod"
    loop = Loop(tmp_path, state=root / M.NIGHTLY_STATE)

    def guard(now):
        return M.nightly_check(M.Ctx(now=now, root=root))

    assert [f.key for f in guard(utc(20, day=18))] == ["absent"]
    loop(utc(20, day=18))
    assert guard(utc(21, day=18)) == [], "the loop runs and its first night isn't due"
    loop(utc(9))
    assert guard(utc(7, 59, day=20)) == []
    (f,) = guard(utc(8, day=20))                     # 36 h after the loop's first pass, and its one night never ended
    assert f.key == "silent" and "none has ended" in f.text and f"is `running` (run `{RUN}`)" in f.text
    loop.fx.reads = [N.Read("error", "it timed out at its 7200 s.", 120)]
    loop(utc(8, 5, day=20))
    assert guard(utc(8, 10, day=20)) == [] and guard(utc(20, 4, day=21)) == []
    (f,) = guard(utc(20, 5, day=21))
    assert f.key == "silent" and "`error` at 2026-10-20T08:05:00Z" in f.text and f"`{COMMIT[:12]}`" in f.text


# ---- the pod's own effects ------------------------------------------------------------------------------------------------------

def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Api:
    def __init__(self):
        self.calls = []

    def call(self, method, **kw):
        self.calls.append((method, kw))
        return {"ts": POSTED} if method == "chat.postMessage" else {}


class Routes:
    route = {"name": "infra", "holder": "bc-44444444-4444-4444-4444-444444444444", "inbox_ts": "1760000000.000300", "closed": None}
    ci = {"name": "ci", "holder": "bc-55555555-5555-5555-5555-555555555555", "inbox_ts": "1760000000.000400", "closed": None}

    def resolve(self, name):
        return name, {"infra": self.route, "ci": self.ci}.get(name)


def pod(tmp_path, **kw):
    reg = slack.load_registry(ROOT / ".agents" / "skills" / "using-slack" / "registry.json", people=False)
    lines = []
    return N.Pod(settings(tmp_path, **kw), log=lines.append, reg=reg, api=Api(), routes=Routes()), lines


def test_the_clone_follows_origins_main_and_each_night_gets_a_worktree_of_its_tip(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    (origin / "tools" / "research" / "src").mkdir(parents=True)
    (origin / "tools" / "research" / "src" / "marker.py").write_text("FIRST = 1\n")
    git("add", "-A", cwd=origin)
    git("commit", "-q", "-m", "first", cwd=origin)
    first = git("rev-parse", "HEAD", cwd=origin)
    git("branch", "side", cwd=origin)
    p, lines = pod(tmp_path, url=str(origin), helper="", tmp=tmp_path / "scratch" / "tmp")

    commit, tree = p.prepare("2026-10-19")
    assert commit == first and tree == tmp_path / "work" / "trees" / f"2026-10-19-{first[:12]}"
    assert (tree / "tools" / "research" / "src" / "marker.py").read_text() == "FIRST = 1\n"
    assert git("status", "--porcelain", cwd=tree) == "" and git("rev-parse", "HEAD", cwd=tree) == first
    assert git("for-each-ref", "--format=%(refname)", cwd=p.repo) == "refs/remotes/origin/main", "main alone is fetched"

    show = [sys.executable, "-c", "import os; print(os.environ['PYTHONPATH']); print(os.environ['TMPDIR']); print(os.getcwd()); "
            f"print('research: run {RUN} launched')"]
    rc, out = p.launch(show, tree)
    assert rc == 0 and out.splitlines() == [str(tree / "tools" / "research" / "src"), str(tmp_path / "scratch" / "tmp"), str(tree),
                                            f"research: run {RUN} launched"]
    assert p.cfg.launch_log == tmp_path / "nightly-check.launch.log" and p.launch_output() == out
    assert dispatch.LAUNCHED.search(out).group(1) == RUN and f"launch: research: run {RUN} launched" in lines
    assert p.launch([sys.executable, "-c", "import sys; sys.exit(3)"], tree) == (3, ""), "each launch's log starts empty"
    p.launch(show, tree)
    p.forget_launch()
    assert p.launch_output() == "" and not p.cfg.launch_log.exists()
    p.forget_launch()                                # nothing to forget is no error

    (origin / "tools" / "research" / "src" / "marker.py").write_text("SECOND = 2\n")
    git("commit", "-q", "-am", "second", cwd=origin)
    second = git("rev-parse", "HEAD", cwd=origin)
    commit, tree2 = p.prepare("2026-10-20")          # the 19th's tree was never removed: the next prepare removes it
    assert commit == second and not tree.exists() and (tree2 / "tools" / "research" / "src" / "marker.py").read_text() == "SECOND = 2\n"
    assert p.prepare("2026-10-20") == (second, tree2), "a night prepared twice (a restart before its launch) gets the tree anew"
    p.cleanup(tree2)
    assert not tree2.exists() and git("worktree", "list", "--porcelain", cwd=p.repo).count("worktree ") == 1
    p.cleanup(tree2)                                 # removing what is already gone is no error

    gone = tmp_path / "moved"
    origin.rename(gone)
    with pytest.raises(subprocess.SubprocessError, match="git fetch"):
        p.prepare("2026-10-21")


def test_the_post_goes_to_agent_alerts_as_infra_and_a_ring_to_the_inbox_of_the_handle_it_names(tmp_path):
    p, _ = pod(tmp_path)
    lead = p.tags["infra"]
    words = f"{lead} *Nightly check of `main` failed* at `{COMMIT[:12]}` (run `{RUN}`, 95 min): failed steps: `pytest`."
    assert p.tags == {n: slack.tag(slack.entry(p.reg, n)) for n in ("infra", "ci")} and lead.startswith("<!subteam^")
    assert p.tags["ci"] != lead
    assert p.post(words) == POSTED
    method, kw = p.api.calls[0]
    assert method == "chat.postMessage" and kw["channel"] == slack.channel_id_for(p.reg, "agent-alerts")
    assert kw["username"] == "infra" and kw["text"].startswith(words) and kw["thread_ts"] is None

    link = slack.permalink(p.reg, p.alerts, POSTED)
    for name, route in (("infra", Routes.route), ("ci", Routes.ci)):
        del p.api.calls[:]
        assert p.ring(POSTED, words, name) == f"rang @{name}"
        (method, kw), = p.api.calls
        assert method == "chat.postMessage" and kw["thread_ts"] == route["inbox_ts"] and kw["username"] == "router"
        assert kw["channel"] == slack.channel_id_for(p.reg, slack.FRONT_DOOR)
        assert kw["text"] == slack.doorbell("infra", POSTED, POSTED, f"{words} {link}"), "the relay's doorbell, from infra"

    del p.api.calls[:]                               # a post longer than a doorbell carries: cut so that its link is still in it
    long = words + SAID_AGREED + "".join(f"\n\u2022 suite `pkg/s{i}`: failed here; passed in `{SHARDS[0]}`" for i in range(9))
    p.ring(POSTED, long, "ci")
    text = p.api.calls[0][1]["text"]
    assert len(slack.plain(f"{long} {link}")) > slack.DOORBELL_MAX and text.endswith(f"\u2026 {link}") and "the rest" not in text
    assert text.startswith(f"from infra, thread {POSTED}: @infra *Nightly check of `main` failed* at `{COMMIT[:12]}`")
    assert len(text.split(": ", 1)[1]) <= slack.DOORBELL_MAX

    del p.api.calls[:]
    p.routes = type("Closed", (), {"resolve": lambda self, name: (name, {**Routes.route, "closed": "2026-10-18T21:00Z"})})()
    assert "ci has no open inbox" in p.ring(POSTED, words, "ci") and p.api.calls == []


def published(tmp_path, rid, rec=None, *, commit=COMMIT, kind="check-shard/v1", argv=("--shard", "1/2"), dirty=False,
              files=True):
    """A run another machine published to the store's remote, as `research run` does: its result (`rec`'s, of `kind`; none
    for a run that left no result), the files its result names (`rec`'s suite reports and Lean report, when it has any) and
    its attempt."""
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    node = LocalStore(tmp_path / "node", config_path=tmp_path / "store.toml")
    outputs = {"result": None, "run_files": None, "run_record": None}
    if rec is not None:
        outputs["result"] = node.put_artifact(kind, {"schema": "research/result/v0.1", "kind": kind, "run_id": rid, **rec["result"]})
        d = tmp_path / "files" / rid
        (d / "lean-audit").mkdir(parents=True)
        (d / "logs.txt").write_text("check: done\n")
        if files:
            (d / "suites.json").write_text(json.dumps([r for r in rec["suites"] if not r["suite"].startswith("lean/")]))
            (d / "suites-lean.json").write_text(json.dumps([r for r in rec["suites"] if r["suite"].startswith("lean/")]))
            (d / "lean-audit" / "audit.json").write_text(json.dumps(rec["audit"]))
        outputs["run_files"] = node.put_artifact("run-files/v1", {"run_id": rid}, tree=d)
    node.put_attempt({
        "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "python3", "version": None}, "derivation": "drv:" + "0" * 64,
        "reuse_key": None, "argv": ["python3", "tools/verity/check/check.py", *argv], "params": {}, "key_params": [], "inputs": {},
        "conditions": {}, "key_conditions": [], "source": {"commit": commit, "dirty": dirty}, "closure": None, "execution": {},
        "cost": None, "status": {"state": "done", "rc": 0}, "outputs": outputs, "telemetry": None})
    for art in (outputs["result"], outputs["run_files"]):
        if art:
            node.push(art)
    node.push_attempt(rid)


def the_pods_store(tmp_path):
    return LocalStore(tmp_path / "pod-store", config_path=tmp_path / "store.toml")


def test_the_recorded_checks_of_a_commit_are_its_clean_attempts_whose_result_is_a_checks_but_the_uncached_ones(tmp_path):
    """The store's remote holds the commit's two shards among what else ran: only they are recorded checks of it. Their
    verdicts are read from each one's result and from the files its result names, fetched from the remote by name."""
    theirs, mine, bare = recorded(), uncached(), {"result": {"steps": [step("circuit-check", "failed")]}, "suites": [], "audit": {}}
    published(tmp_path, SHARDS[0], theirs[SHARDS[0]])
    published(tmp_path, SHARDS[1], theirs[SHARDS[1]], kind="check-deferred/v1", argv=("--shard", "2/2"))
    others = [f"r20261019-05{i}000-0c0{i}" for i in range(5)]
    published(tmp_path, RUN, mine, kind="check/v1", argv=())                                           # the night's own run, by its id
    published(tmp_path, others[0], bare, kind="check/v1", argv=("--no-cache", "--timeout", "60"))      # another uncached one
    published(tmp_path, others[1], bare, kind="lean-fast/v1", argv=())                                 # a result that is no check's
    published(tmp_path, others[2], None, argv=())                                                      # a run that left no result
    published(tmp_path, others[3], bare, commit="d" * 40)                                              # a check of another commit
    published(tmp_path, others[4], bare, dirty=True)                                                   # a check of a changed tree
    store = the_pods_store(tmp_path)
    got = N.stored(COMMIT, RUN, "r20261019-031405", store)
    assert got == {run: N.verdicts(run, **rec) for run, rec in theirs.items()}
    assert sorted(p.stem for p in (store.root / "attempts").iterdir()) == sorted([*SHARDS, RUN, *others]), \
        "the store listed them all: what it left out, it left out by what each is"
    assert N.tally(N.verdicts(RUN, **mine), got) == {"runs": SHARDS, "agree": 12, "disagree": [], "unchecked": [],
                                                    "unstored": ["step `lean-agreement`"], "earlier": NO_EARLIER}

    published(tmp_path, "r20261019-041900-0a03", bare, argv=("--shard", "2/2"), files=False)
    got = N.stored(COMMIT, RUN, "r20261019-031405", the_pods_store(tmp_path))
    assert got["r20261019-041900-0a03"] == {("step", "circuit-check"): [
        {"run": "r20261019-041900-0a03", "status": "failed", "key": None, "how": "", "report": None, "row": None}]}, \
        "a shard whose files hold no suite report and no Lean report still has its steps"
    again = N.tally(N.verdicts(RUN, **mine), got)
    assert again["disagree"] == ["step `circuit-check`: passed here; failed in `r20261019-041900-0a03`"], \
        "the second shard run again, later: its verdict on the step is the recorded one"
    assert again["earlier"] == {"passed": ["step `circuit-check`"], "failed": []} and again["runs"] == [*SHARDS, "r20261019-041900-0a03"]

    late = N.stored(COMMIT, RUN, "r20261019-041600", LocalStore(tmp_path / "late", config_path=tmp_path / "store.toml"))
    assert sorted(late) == [SHARDS[1], "r20261019-041900-0a03"], "only the attempts since the id given are listed from the remote"
    assert N.stored("e" * 40, RUN, "r20261019-031405", the_pods_store(tmp_path)) == {}, "a commit with no recorded check"


def test_a_recorded_checks_report_that_cant_be_fetched_is_an_unread_comparison_never_a_part_with_no_stored_verdict(tmp_path):
    """The remote lost a suite report a recorded check's files list: read without it, every suite of that shard would count
    as having no stored verdict and ring nobody. The read raises, so the loop reads it again and then reports it unread."""
    theirs = recorded()
    published(tmp_path, SHARDS[0], theirs[SHARDS[0]])
    lost = hashlib.sha256((tmp_path / "files" / SHARDS[0] / "suites.json").read_bytes()).hexdigest()
    (tmp_path / "vault" / "objects" / "sha256" / lost).unlink()
    with pytest.raises(NotFound, match=f"blob {lost} is not local and not on"):
        N.stored(COMMIT, RUN, "r20261019-031405", the_pods_store(tmp_path))


def test_the_comparison_fetches_the_runs_lean_report_once_and_reads_the_recorded_checks_from_the_store(tmp_path):
    theirs, mine = recorded(), uncached()
    for i, run in enumerate(SHARDS):
        published(tmp_path, run, theirs[run], argv=("--shard", f"{i + 1}/2"))
    d, asked = records(tmp_path / "runs", status={"state": "done", "rc": 0}, result=mine["result"],
                       suites=mine["suites"][:-1], lean=mine["suites"][-1:]), []

    def fetch(run_dir, also=()):
        asked.append((run_dir, also))
        (run_dir / "lean-audit").mkdir(exist_ok=True)
        (run_dir / "lean-audit" / "audit.json").write_text(json.dumps(mine["audit"]))
        return {}

    got = N.compared(RUN, COMMIT, "r20261019-031405", fetch=fetch, root=tmp_path / "runs", store=the_pods_store(tmp_path))
    assert asked == [(d, ("lean-audit/audit.json",))], "the Lean report, beside the records the read already fetched"
    assert got == {"runs": SHARDS, "agree": 12, "disagree": [], "unstored": ["step `lean-agreement`"], "unchecked": [],
                   "earlier": NO_EARLIER}

    row(mine, "pkg/b").update(status="failed")
    (d / "suites.json").write_text(json.dumps(mine["suites"][:-1]))
    got = N.compared(RUN, COMMIT, "r20261019-031405", fetch=fetch, root=tmp_path / "runs", store=the_pods_store(tmp_path))
    assert got["disagree"] == [f"suite `pkg/b`: failed here, key `{B}`; passed in `{SHARDS[0]}`"] and got["agree"] == 11

    with pytest.raises(ConnectionError, match="unreachable \\(rc 255\\)"):
        N.compared(RUN, COMMIT, "r20261019-031405", fetch=lambda run_dir, also=(): {"last_fetch_error": {"rc": 255}},
                   root=tmp_path / "runs", store=the_pods_store(tmp_path))


def test_the_pod_looks_for_a_commits_recorded_checks_from_an_hour_before_it_was_made(tmp_path, monkeypatch):
    """A recorded check of a commit ran after it was made, and a run's id starts with when it was launched: the store's
    attempts are listed from an hour before the commit's own time, whatever today is."""
    p, _ = pod(tmp_path)
    p.repo.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=p.repo)
    (p.repo / "f").write_text("1\n")
    git("add", "f", cwd=p.repo)
    subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=p.repo, check=True, capture_output=True,
                   env={**os.environ, "GIT_COMMITTER_DATE": "2026-10-19T04:14:05Z", "GIT_AUTHOR_DATE": "2026-10-12T00:00:00Z"})
    commit, asked = git("rev-parse", "HEAD", cwd=p.repo), []
    monkeypatch.setattr(N, "compared", lambda run, commit, since: asked.append((run, commit, since)) or AGREED)
    assert p.compare(RUN, commit) == AGREED and asked == [(RUN, commit, "r20261019-031405")] and N.BEFORE_S == 3600


def test_the_label_is_the_outcome_on_the_commit_with_the_run_as_its_ref(tmp_path):
    p, _ = pod(tmp_path)
    said = p.label(COMMIT, "contradicted", RUN)
    assert said.startswith(f"commit:{COMMIT} nightly_check=contradicted; ")
    (lab,) = LocalStore().labels(f"commit:{COMMIT}")
    assert (lab.key, lab.value, lab.by, lab.ref) == (vocab.NIGHTLY_CHECK, "contradicted", "nightly-check", RUN)
    p.label(COMMIT, "error", None)
    assert [(x.value, x.ref) for x in LocalStore().labels(f"commit:{COMMIT}", key="nightly_check")] == [
        ("contradicted", RUN), ("error", None)]
    with pytest.raises(ValueError, match="nightly_check"):
        p.label(COMMIT, "green", RUN)
