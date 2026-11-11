"""The control pod's nightly check of `main` (infra/control/nightly_check.py): the command its settings build, when a night
starts, how a run's records read as an outcome, each outcome's post, ring and label, and what a launch that names no run, a
cleanup that fails, a restart or a refused post leave. The launcher, the fetched records, Slack and the clock are stand-ins:
only the clone's test runs a program (git, and a Python that prints its environment)."""
import calendar
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from research import monitors as M
from research import slack
from research.jobs import dispatch
from research.store import vocab
from research.store.local import LocalStore

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
POSTED = "1760000000.000100"


def utc(hour, minute=0, day=19):
    return float(calendar.timegm((2026, 10, day, hour, minute, 0)))


def settings(tmp_path, **kw):
    return N.Settings(**{"on": "node-a", "pool": "pool-a", "timeout": 7200, "hour": 9, "work": tmp_path / "work",
                         "state": tmp_path / "nightly-check.json", **kw})


class Fx:
    """A pass's effects, recorded: none starts a program, reaches a machine or posts."""
    lead = LEAD

    def __init__(self, tmp_path, out=LAUNCHED, rc=0, kept=""):
        self.tmp, self.out, self.rc, self.kept = tmp_path, out, rc, kept    # kept: the launch log as the last launch left it
        self.reads = []                 # what each read of the run gives, in turn: a Read, or an exception it raises
        self.fail = {}                  # an effect's name -> the exceptions it raises, in turn, before it works
        self.during_launch = None
        self.logs, self.launches, self.cleaned, self.posts, self.rings, self.labels = [], [], [], [], [], []

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

    def post(self, words):
        self._may_fail("post")
        self.posts.append(words)
        return POSTED

    def ring(self, ts, words):
        self._may_fail("ring")
        self.rings.append((ts, words))
        return "rang @infra"

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
SAID = [{"suite": "verity/circuits", "contradicts": ["tests/test_ops.py::test_add", "tests/test_ops.py::test_mul"]}]
RECORDS = {
    "passed": dict(status={"state": "done", "rc": 0, "wall_s": 14520.4}, result=result("passed", STEPS_OK),
                   suites=[suite("tools/research", "passed"), suite("infra/ops", "cached")]),
    "failed": dict(status={"state": "failed", "rc": 1, "wall_s": 5700}, result=result("failed", STEPS_RED),
                   suites=[suite("tools/research", "failed"), suite("verity/circuits", "passed")],
                   lean=[suite("verity/protocols/flock/verifier", "failed")]),
    "contradicted": dict(status={"state": "failed", "rc": 1, "wall_s": 5700},
                         result=result("failed", STEPS_RED[:2], cache_contradictions=SAID),
                         suites=[suite("verity/circuits", "failed", contradicts=SAID[0]["contradicts"])]),
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
                     result("passed", STEPS_OK, cache_contradictions=SAID), [], cfg)
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


# ---- each outcome's post, ring and label ----------------------------------------------------------------------------------------

def night_of(tmp_path, got):
    loop = Loop(tmp_path).running()
    loop.fx.reads = [got]
    return loop, loop(utc(13, 2))


def test_a_pass_is_one_line_that_rings_nobody_and_labels_the_commit(tmp_path):
    loop, s = night_of(tmp_path, N.Read("passed", "", 242))
    assert loop.fx.posts == [f"Nightly check of `main` passed: `{COMMIT[:12]}`, run `{RUN}`, 242 min."]
    assert loop.fx.rings == [] and loop.fx.labels == [(COMMIT, "passed", RUN)]
    assert s["night"]["phase"] == "done" and "ring" not in s["night"]
    assert s["last"] == {"day": "2026-10-19", "commit": COMMIT, "run": RUN, "outcome": "passed", "at": "2026-10-19T13:02:00Z"}


@pytest.mark.parametrize("outcome, title", [("failed", "failed"), ("contradicted", "contradicts the cache"),
                                            ("error", "gave no verdict")])
def test_anything_but_a_pass_says_what_happened_rings_infra_and_labels_the_commit(tmp_path, outcome, title):
    loop, s = night_of(tmp_path, N.Read(outcome, "failed steps: `pytest`.", 95))
    post = f"{LEAD} *Nightly check of `main` {title}* at `{COMMIT[:12]}` (run `{RUN}`, 95 min): failed steps: `pytest`."
    assert loop.fx.posts == [post] and loop.fx.rings == [(POSTED, post)] and loop.fx.labels == [(COMMIT, outcome, RUN)]
    assert s["night"]["phase"] == "done" and s["last"]["outcome"] == outcome
    assert s["night"]["post"] == POSTED and s["night"]["ring"] == "rang @infra"


def test_every_outcome_is_a_value_of_the_label(tmp_path):
    assert set(N.OUTCOMES) == set(vocab.NIGHTLY_CHECK_OUTCOMES)
    for name in RECORDS:
        got = N.read(RUN, settings(tmp_path), fetch=lambda d, also=(): {}, root=records(tmp_path / name, **RECORDS[name]).parent)
        assert got.outcome is None or vocab.check(vocab.NIGHTLY_CHECK, got.outcome) is None, name


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

    def resolve(self, name):
        return name, self.route if name == "infra" else None


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


def test_the_post_goes_to_agent_alerts_as_infra_and_the_ring_to_infras_inbox(tmp_path):
    p, _ = pod(tmp_path)
    words = f"{p.lead} *Nightly check of `main` failed* at `{COMMIT[:12]}` (run `{RUN}`, 95 min): failed steps: `pytest`."
    assert p.lead == slack.tag(slack.entry(p.reg, "infra")) and p.lead.startswith("<!subteam^")
    assert p.post(words) == POSTED
    method, kw = p.api.calls[0]
    assert method == "chat.postMessage" and kw["channel"] == slack.channel_id_for(p.reg, "agent-alerts")
    assert kw["username"] == "infra" and kw["text"].startswith(words) and kw["thread_ts"] is None

    del p.api.calls[:]
    assert p.ring(POSTED, words) == "rang @infra"
    (method, kw), = p.api.calls
    assert method == "chat.postMessage" and kw["thread_ts"] == Routes.route["inbox_ts"] and kw["username"] == "router"
    assert kw["channel"] == slack.channel_id_for(p.reg, slack.FRONT_DOOR)
    assert kw["text"] == slack.doorbell("infra", POSTED, POSTED, f"{words} {slack.permalink(p.reg, p.alerts, POSTED)}")

    del p.api.calls[:]
    p.routes = type("Closed", (), {"resolve": lambda self, name: (name, {**Routes.route, "closed": "2026-10-18T21:00Z"})})()
    assert "no open inbox" in p.ring(POSTED, words) and p.api.calls == []


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
