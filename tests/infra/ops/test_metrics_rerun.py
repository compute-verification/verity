"""The control pod's reruns of the declared numbers (infra/control/metrics_rerun.py): a number older than its `every` is launched
as a `research run` on a machine, at most `--max` a pass and none while `--max-in-flight` are in flight; never one whose last
launch is in flight or passed less than its `every` ago, whatever the door says; a failed launch rings its owners and is tried
again within the hour, and is held after three that ended alike until its declaration changes, while a run that waited out
its pool's line is launched again uncounted, and rings once at the third such launch running; a placeholder's owner is rung
once; a row that says `runs = "control"` runs on the pod itself, as a local run the pass waits for, with the Linear app's
keys and no other of the pod's; the record and its rename are synced to disk; the stop file stops every launch,
read again before each; a dry run launches nothing; a CPU command (no `--on`, `auto`, a CPU node, a pool's `NODE`) runs in
the loop's pool up to a CPU machine's size, and one asking for GPUs or naming another machine only on a machine the plan
lists for it now; never `auto` or a small-disk machine; a row's options outside the allowed set are refused; the record
says each number's reason; a pass takes a lock. The clone, the door, the runs' records, the launch, the rings and the clock
are stand-ins: no test reads the clock or the network."""
import calendar
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research import runs

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"


def module(name):
    spec = importlib.util.spec_from_file_location(name, CONTROL / f"{name}.py")
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)     # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


R = module("metrics_rerun")
W = module("watchers")
NOW = float(calendar.timegm((2026, 11, 14, 12, 0, 0)))
DAY, DAY_MS = 86400, 86_400_000
CPU = "research run --on auto --project verity --source . --cwd source --store-read -- uv run python benchmarks/compute/run.py"
GPU = "research run --on NODE --source . --cwd source -- bash benchmarks/serve_tax/run.sh --model glm53"
RUNNING = ("running", None, True)


def run_id(n):
    return f"r20261114-1200{n:02d}-ab12"


def row(i, command, every="7d"):
    return {"id": i, "title": "t", "owner": i.split("/")[0], "unit": "s", "better": "lower", "command": command, "every": every}


def toml(rows):
    return "\n".join("[[metric]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in r.items()) for r in rows)


def free(machine, metrics=None, start="2026-11-14T00:00:00Z", until="2026-11-15T00:00:00Z"):
    return (f'[[free]]\nmachine = "{machine}"\nfrom = {start}\nuntil = {until}\n'
            + (f"metrics = {json.dumps(metrics)}\n" if metrics is not None else ""))


class Pod:
    """The pass's world: the door's ages (None: listed without a reading), each run's state, rc and whether it started (a run
    is registered once its launch printed its run line, or when `states` holds it), and each launch's output; it records what
    it was asked."""

    def __init__(self, ages=None, states=None, out=None, listed=None, on_refresh=None, ring_fails=0):
        self.ages, self.states, self.lines, self.launched = ages or {}, states or {}, [], []
        self.listed, self.on_refresh, self.named, self.registered = listed, on_refresh, 0, set()
        self.out = out or (lambda n: f"research: run {run_id(n)} launched\n")
        self.rang, self.ring_fails = [], ring_fails

    def log(self, msg):
        self.lines.append(msg)

    def refresh(self):
        if self.on_refresh:
            self.on_refresh()
        return "c0ffee" + "0" * 34

    def door(self):
        ids = self.listed if self.listed is not None else self.ids
        return {"metrics": [{"id": i, "ageMs": self.ages.get(i)} for i in ids]}

    def new_id(self, now):
        self.named += 1
        return run_id(self.named)

    def status(self, run):
        if run not in self.states and run not in self.registered:
            return None
        s = self.states.get(run, RUNNING)
        if isinstance(s, Exception):
            raise s
        return {"state": s[0], "rc": s[1], "started": s[2], **(s[3] if len(s) > 3 else {})}

    def ring(self, run, rc, owner, text):
        if self.ring_fails:
            self.ring_fails -= 1
            raise RuntimeError("ring-run exited 1: slack is down")
        self.rang.append((run, rc, owner, text))

    def launch(self, argv):
        i = argv.index("--id")
        self.launched.append(argv[:i] + argv[i + 2:])
        said = self.out(len(self.launched))
        if f"run {argv[i + 1]} launched" in said:
            self.registered.add(argv[i + 1])
        return 0, said


def setup(tmp_path, files, **kw):
    tree = tmp_path / "tree"
    for where, rows in files.items():
        (tree / where).mkdir(parents=True, exist_ok=True)
        (tree / where / "metrics.toml").write_text(toml(rows))
    return R.Settings(tree, tmp_path / "state.json", tmp_path / "stop", tmp_path / "plan.toml", "vy-nebius-cpu-12", "research", **kw)


def run(cfg, pod, state=None, now=NOW, dry_run=False):
    state = {} if state is None else state
    jobs, _ = R.declarations(cfg.tree)
    pod.ids = [i for j in jobs for i in j.ids]          # the door lists every declared number unless `listed` says otherwise
    saved = []
    R.step(state, cfg, pod, now, lambda s: saved.append(json.loads(json.dumps(s))), dry_run=dry_run)
    return state, saved


def test_a_stale_cpu_number_is_launched_on_the_named_machine_and_pool_and_recorded(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(ages={"compute-accounting/share": 8 * DAY_MS})
    state, saved = run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-nebius-cpu-12", "--pool", "research", "--project", "verity",
                             "--source", ".", "--cwd", "source", "--store-read", "--", "uv", "run", "python",
                             "benchmarks/compute/run.py"]]
    (e,) = state["launches"]
    assert (e["metrics"], e["run"], e["at"], e["machine"], e["state"]) == (
        ["compute-accounting/share"], run_id(1), "2026-11-14T12:00:00Z", "vy-nebius-cpu-12", "submitted")
    assert saved[0]["launches"][0]["state"] == "launching", "the record holds the launch before it starts"


def test_rows_sharing_a_command_are_one_launch_and_a_fresh_number_or_a_loops_is_never_launched(tmp_path):
    loop = row("infra/loop-number", "sh infra/control/store_index.sh", every="5m")
    fresh = row("top/fresh", "research run --on auto --cwd source -- uv run python benchmarks/tree/x.py", every="1d")
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/a", CPU), row("compute-accounting/b", CPU)],
                           "benchmarks/tree": [fresh], "infra/control": [loop]})
    pod = Pod(ages={"compute-accounting/a": 8 * DAY_MS, "top/fresh": DAY_MS // 2})
    state, _ = run(cfg, pod)
    assert len(pod.launched) == 1 and state["launches"][0]["metrics"] == ["compute-accounting/a", "compute-accounting/b"]
    assert not any("top/fresh" in ln or "infra/loop-number" in ln for ln in pod.lines)


def test_a_number_whose_script_is_under_infra_but_not_a_control_loop_is_rerun(tmp_path):
    """Only the control pod's loops send their own numbers: the dashboard's count, measured by infra/dashboard/run.py, is
    launched like any other stale number."""
    dashboard = row("console/top-goals-running",
                    "research run --on auto --project verity --source . --cwd source -- uv run python infra/dashboard/run.py",
                    every="1d")
    loop = row("infra/loop-number", "sh infra/control/store_index.sh", every="5m")
    cfg = setup(tmp_path, {"infra/dashboard": [dashboard], "infra/control": [loop]})
    pod = Pod(ages={"console/top-goals-running": 2 * DAY_MS, "infra/loop-number": DAY_MS})
    state, _ = run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-nebius-cpu-12", "--pool", "research", "--project", "verity",
                             "--source", ".", "--cwd", "source", "--", "uv", "run", "python", "infra/dashboard/run.py"]]
    assert [e["metrics"] for e in state["launches"]] == [["console/top-goals-running"]]
    assert not any("infra/loop-number" in ln for ln in pod.lines)


RECURSION = "uv run python benchmarks/recursion_e2e/record.py run"
PARKED_LINE = ("skip benchmarks/recursion-run-seconds: benchmarks/recursion-run-seconds has no reading; but its goal is parked "
               "(top/parked): a parked number is rerun only by hand")


def parked_and_not(tmp_path):
    """Two rows alike but for their goal, each with no reading and its every long past."""
    parked = {**row("benchmarks/recursion-run-seconds", RECURSION, every="30d"), "goal": "top/parked"}
    other = {**row("benchmarks/recursion-other-seconds", RECURSION, every="30d"), "goal": "benchmarks/proof-system-performance"}
    cfg = setup(tmp_path, {"benchmarks/recursion_e2e": [parked], "benchmarks/recursion_other": [other]})
    return cfg, Pod(ages={"benchmarks/recursion-run-seconds": None, "benchmarks/recursion-other-seconds": None})


def test_a_parked_number_is_never_launched_and_the_pass_says_why(tmp_path):
    """A row whose goal is top/parked is parked work, which is rerun only by hand; an identical row under another goal is
    launched."""
    cfg, pod = parked_and_not(tmp_path)
    state, _ = run(cfg, pod)
    assert [e["metrics"] for e in state["launches"]] == [["benchmarks/recursion-other-seconds"]]
    assert pod.launched[0][-5:] == ["uv", "run", "python", "benchmarks/recursion_e2e/record.py", "run"]
    assert pod.lines.count(PARKED_LINE) == 1


def test_a_dry_run_judges_a_parked_number_the_same(tmp_path):
    cfg, pod = parked_and_not(tmp_path)
    state, _ = run(cfg, pod, dry_run=True)
    assert pod.launched == [] and state["launches"] == []
    assert PARKED_LINE in pod.lines
    assert any(ln.startswith("would launch benchmarks/recursion-other-seconds on vy-nebius-cpu-12: ") for ln in pod.lines)
    assert not any(ln.startswith("would launch benchmarks/recursion-run-seconds") for ln in pod.lines)


def test_a_command_one_parked_row_shares_with_another_is_not_launched(tmp_path):
    """One launch measures every row of its command, so a parked row keeps the command from the loop, and the line names it."""
    parked = {**row("benchmarks/a", RECURSION, every="30d"), "goal": "top/parked"}
    cfg = setup(tmp_path, {"benchmarks/recursion_e2e": [parked, row("benchmarks/b", RECURSION, every="30d")]})
    pod = Pod(ages={"benchmarks/a": None, "benchmarks/b": None})
    run(cfg, pod)
    assert pod.launched == []
    assert ("skip benchmarks/a, benchmarks/b: benchmarks/a has no reading; benchmarks/b has no reading; but benchmarks/a's goal "
            "is parked (top/parked): a parked number is rerun only by hand") in pod.lines


def past_default(tmp_path, option):
    """A row whose command takes ``option`` and an identical one without it, each with no reading."""
    asks = row("training/asks", f"research run --on auto {option} -- uv run python benchmarks/asks/run.py")
    plain = row("training/plain", "research run --on auto -- uv run python benchmarks/plain/run.py")
    cfg = setup(tmp_path, {"benchmarks/asks": [asks], "benchmarks/plain": [plain]})
    return cfg, Pod(ages={"training/asks": None, "training/plain": None})


def past_default_line(options):
    return (f"skip training/asks: training/asks has no reading; but its command asks for more than a CPU check machine has "
            f"({options}): such a number is rerun by its owner")


MEM, MAX_S, CPUS = R.CEILING["--mem-gb"], R.CEILING["--timeout"], R.CEILING["--cpus"]


def test_the_ceiling_is_a_cpu_check_machines():
    assert (CPUS, MEM) == (192, 750), "a CPU check machine's logical CPUs and GiB"


@pytest.mark.parametrize("option, said", [
    (f"--mem-gb {MEM:g}", None), (f"--mem-gb {MEM + 1:g}", f"--mem-gb {MEM + 1:g} over {MEM:g}"),
    (f"--timeout {MAX_S:g}", None), (f"--timeout {MAX_S + 1:g}", f"--timeout {MAX_S + 1:g} over {MAX_S:g}"),
    (f"--cpus {CPUS:g}", None), (f"--cpus {CPUS + 1:g}", f"--cpus {CPUS + 1:g} over {CPUS:g}")])
def test_a_number_is_launched_only_when_its_command_asks_no_more_than_a_cpu_check_machine_has(tmp_path, option, said):
    """At or under the machine's CPUs, memory and the loop's longest run, a row is launched like one that names none; one unit
    over, it is left to its owner, and the pass names the option, its value and the ceiling it exceeds."""
    cfg, pod = past_default(tmp_path, option)
    state, _ = run(cfg, pod)
    assert sorted(i for e in state["launches"] for i in e["metrics"]) == (
        ["training/plain"] if said else ["training/asks", "training/plain"])
    assert pod.lines.count(past_default_line(said)) == (1 if said else 0)


@pytest.mark.parametrize("option, said", [(f"--mem-gb={MEM:g}", None), (f"--timeout={MAX_S + 1:g}", f"--timeout {MAX_S + 1:g} over {MAX_S:g}")])
def test_a_dry_run_judges_the_ceiling_the_same(tmp_path, option, said):
    cfg, pod = past_default(tmp_path, option)
    state, _ = run(cfg, pod, dry_run=True)
    assert pod.launched == [] and state["launches"] == []
    assert any(ln.startswith("would launch training/plain on vy-nebius-cpu-12: ") for ln in pod.lines)
    assert any(ln.startswith("would launch training/asks on vy-nebius-cpu-12: ") for ln in pod.lines) == (said is None)
    assert (past_default_line(said) in pod.lines) == (said is not None)


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("option, said", [
    ("--timeout 240m", None),
    ("--timeout 241m", f"--timeout 241m over {MAX_S:g}"),
    ("--timeout 5h", f"--timeout 5h over {MAX_S:g}"),
    ("--timeout abc", "--timeout abc could not be read"),
    ("--mem-gb x", "--mem-gb x could not be read"),
    ("--cpus x", "--cpus x could not be read"),
    ("--timeout nan", "--timeout nan could not be read"),
    ("--mem-gb nan", "--mem-gb nan could not be read"),
    ("--timeout inf", f"--timeout inf over {MAX_S:g}"),
    ("--mem-gb inf", f"--mem-gb inf over {MEM:g}")])
def test_a_value_is_read_as_research_run_reads_it_and_one_it_cant_read_is_its_owners(tmp_path, option, said, dry_run):
    """`--timeout` in research run's own units (`60m`, `2h`) is judged by its seconds; a value research run can't read is past
    the ceiling and said, and the pass goes on to the next row."""
    cfg, pod = past_default(tmp_path, option)
    state, _ = run(cfg, pod, dry_run=dry_run)
    if dry_run:
        launched = sorted(m for m in ("training/asks", "training/plain")
                          if any(ln.startswith(f"would launch {m} on vy-nebius-cpu-12: ") for ln in pod.lines))
    else:
        launched = sorted(i for e in state["launches"] for i in e["metrics"])
    assert launched == (["training/plain"] if said else ["training/asks", "training/plain"])
    assert pod.lines.count(past_default_line(said)) == (1 if said else 0)


def test_a_host_measurement_asking_for_cpus_and_memory_runs_in_the_loops_pool_without_its_cpus(tmp_path):
    """The shape of the host rate rows: `--cpus 48 --mem-gb 64`. `research run --pool` refuses `--cpus` (the slot's cores are
    the run's), so the launch leaves it out and keeps `--mem-gb`, which the slot admits by."""
    rate = row("memory-accounting/host-write-rate", "research run --on auto --cpus 48 --mem-gb 64 --project verity --cwd source "
                                                    "-- bash calibration/bandwidth/run_write_host.sh")
    cfg = setup(tmp_path, {"calibration/bandwidth": [rate]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-nebius-cpu-12", "--pool", "research", "--source", ".", "--mem-gb",
                             "64", "--project", "verity", "--cwd", "source", "--", "bash",
                             "calibration/bandwidth/run_write_host.sh"]]


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_pools_run_on_an_owners_machine_runs_in_the_loops_pool(tmp_path, dry_run):
    """The shape of the network scans' rows: `--on NODE`, a pool of research's, 128 GiB and three hours. A pool's run is a
    slot's, on a CPU node, so the loop runs it on its own machine in its own pool, without the plan."""
    scan = row("network-accounting/scan",
               "research run --on NODE --project verity --source . --cwd clone --pool network --as network-accounting "
               "--mem-gb 128 --timeout 10800 -- bash benchmarks/scan/run.sh")
    cfg = setup(tmp_path, {"benchmarks/scan": [scan]})
    pod = Pod(ages={"network-accounting/scan": None})
    state, _ = run(cfg, pod, dry_run=dry_run)
    argv = ["research", "run", "--on", "vy-nebius-cpu-12", "--pool", "research", "--project", "verity", "--source", ".",
            "--cwd", "clone", "--as", "network-accounting", "--mem-gb", "128", "--timeout", "10800", "--", "bash",
            "benchmarks/scan/run.sh"]
    if dry_run:
        assert pod.launched == [] and any(ln.endswith(" ".join(argv)) for ln in pod.lines)
    else:
        assert pod.launched == [argv]


def test_a_pool_research_does_not_have_is_refused(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/b": [row("benchmarks/b", "research run --on auto --pool mine -- bash benchmarks/b/run.sh")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []
    assert "gives --pool 'mine', which isn't network or proofs or research or soundness" in pod.lines[-1]


def test_at_most_max_launches_a_pass_and_the_rest_are_said(tmp_path):
    files = {f"benchmarks/b{n}": [row(f"benchmarks/n{n}", f"research run --on auto -- bash benchmarks/b{n}/run.sh")] for n in range(3)}
    cfg = setup(tmp_path, files, max=2)
    pod = Pod()
    run(cfg, pod)
    assert len(pod.launched) == 2
    assert "skip benchmarks/n2: benchmarks/n2 has no reading; but this pass has made its 2 launches (--max)" in pod.lines


def test_never_a_number_whose_last_launch_is_in_flight_or_ran_less_than_its_every_ago_whatever_the_door_says(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1d")]})
    pod = Pod()                                               # the door holds no reading, every pass
    state, _ = run(cfg, pod)
    run(cfg, pod, state, now=NOW + 3 * DAY)
    assert len(pod.launched) == 1
    assert pod.lines[-1].endswith(f"its last launch, run {run_id(1)} at 2026-11-14T12:00:00Z, is still in flight")
    pod.states[run_id(1)] = ("done", 0, True)
    run(cfg, pod, state, now=NOW + 4 * DAY)
    assert len(pod.launched) == 1, "it ended at this pass: less than its every ago"
    assert pod.lines[-1].endswith(f"its last launch, run {run_id(1)}, passed at 2026-11-18T12:00:00Z, less than "
                                  "compute-accounting/share's every 1d ago")
    run(cfg, pod, state, now=NOW + 5 * DAY + 1)
    assert len(pod.launched) == 2


def test_a_failed_run_rings_its_owner_and_is_tried_again_within_the_hour_not_after_its_every(tmp_path):
    """A weekly number whose run fails is launched again an hour after, not a week: the owner (the id's handle) is rung once
    with the run id and the failure's reason line, and the launch between is said."""
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod()
    state, _ = run(cfg, pod)
    pod.states[run_id(1)] = ("failed", 1, True, {"class": "WORKLOAD_EXIT", "why": "census has no passing run"})
    run(cfg, pod, state, now=NOW + 900)
    assert pod.rang == [(run_id(1), "1", "compute-accounting",
                         "metrics-rerun: compute-accounting/share failed (WORKLOAD_EXIT: census has no passing run); it is "
                         "launched again after 60 min, or its every if shorter")]
    assert state["launches"][0]["rung"] == "2026-11-14T12:15:00Z"
    run(cfg, pod, state, now=NOW + 1800)
    assert len(pod.launched) == 1 and pod.lines[-1].endswith(
        f"its last launch, run {run_id(1)}, failed at 2026-11-14T12:15:00Z, and it is launched again 60 min after")
    assert state["numbers"]["compute-accounting/share"]["reason"] == "retry"
    run(cfg, pod, state, now=NOW + 900 + 3600)
    assert len(pod.launched) == 2 and len(pod.rang) == 1


def test_three_failures_that_ended_alike_hold_the_command_until_its_declaration_changes(tmp_path):
    """The first failure of a cause rings, the second ended as it did and doesn't, the third rings again and holds the number
    for HOLD_S or until its rows change."""
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    for n in range(1, 4):
        pod.states[run_id(n)] = ("failed", 1, True)
        run(cfg, pod, state, now=NOW + n * 7200 - 3600)      # sees it end
        run(cfg, pod, state, now=NOW + n * 7200)             # an hour on: launched again
    assert len(pod.launched) == 3, "the third failure holds it"
    assert [r[0] for r in pod.rang] == [run_id(1), run_id(3)]
    assert pod.rang[1][3].endswith("held: it ended this way 3 times running, so it is launched again once its rows in "
                                   "benchmarks/compute/metrics.toml change, or 7.0 d after")
    assert pod.lines[-1] == ("skip compute-accounting/share: compute-accounting/share has no reading; but held: its last 3 "
                             "launches ended alike, failed (failed, rc 1) "
                             f"({run_id(1)}, {run_id(2)}, {run_id(3)}), so it is launched again once its rows in "
                             "benchmarks/compute/metrics.toml change, or 7.0 d after 2026-11-14T17:00:00Z")
    assert state["numbers"]["compute-accounting/share"]["reason"] == "failing"
    (cfg.tree / "benchmarks/compute/metrics.toml").write_text(toml([row("compute-accounting/share", CPU, every="2h")]))
    run(cfg, pod, state, now=NOW + 8 * 7200)
    assert len(pod.launched) == 4


def test_failures_that_ended_differently_never_hold_the_command(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    for n, rc in enumerate((1, 2, 1, 2), start=1):
        pod.states[run_id(n)] = ("failed", rc, True)
        run(cfg, pod, state, now=NOW + n * 7200 - 3600)
        run(cfg, pod, state, now=NOW + n * 7200)
    assert len(pod.launched) == 5 and len(pod.rang) == 4, "each failure is the first of its cause"


def test_a_ring_that_fails_is_tried_again_and_then_given_up(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(ring_fails=R.RING_TRIES)
    state, _ = run(cfg, pod)
    pod.states[run_id(1)] = ("failed", 1, True)
    for n in range(1, R.RING_TRIES + 2):
        run(cfg, pod, state, now=NOW + n * 60)
    assert pod.rang == [] and state["launches"][0]["rung"] == "not rung: RuntimeError: ring-run exited 1: slack is down"
    assert sum("ring of compute-accounting/share failed" in ln for ln in pod.lines) == R.RING_TRIES


def test_a_failure_recorded_before_rings_is_never_rung(tmp_path):
    """A record from before this loop rang holds failures with no `rung`: none of them is rung when the loop starts."""
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    old = {"metrics": ["compute-accounting/share"], "command": CPU, "state": "failed", "rc": 1, "run": run_id(9),
           "at": "2026-11-13T00:00:00Z", "ended": "2026-11-13T01:00:00Z"}
    pod = Pod()
    run(cfg, pod, {"launches": [old]})
    assert pod.rang == [] and len(pod.launched) == 1


def test_a_row_with_an_option_outside_the_allowed_set_is_refused_naming_it(tmp_path):
    bad = {"--send /data/watchers/metrics-rerun.env": "takes --send, which a declared command may not carry",
           "--source /data/steward/research-notes": "gives --source '/data/steward/research-notes', which isn't .",
           "--runs-dir /data/research": "takes --runs-dir", "--id r20261114-000000-0000": "takes --id",
           "--project other": "gives --project 'other', which isn't verity", "--pool=../x": "gives --pool '../x'",
           "--store-read=1": "gives the flag --store-read a value",
           "--pool research --pool proofs": "gives --pool twice", "--gpu-host": "takes --gpu-host"}
    files = {f"benchmarks/b{n}": [row(f"benchmarks/n{n}", f"research run --on auto {opt} -- bash benchmarks/b{n}/run.sh")]
             for n, opt in enumerate(bad)}
    cfg = setup(tmp_path, files)
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []
    for n, why in enumerate(bad.values()):
        (line,) = [ln for ln in pod.lines if ln.startswith(f"skip benchmarks/n{n}:")]
        assert why in line, line


def test_launches_that_never_ran_are_held_after_three_in_a_row(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1d")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    for n in range(1, 4):
        pod.states[run_id(n)] = ("failed", 75, False)         # its pool refused it at once: it never took a slot
    for n in range(1, 50):                                    # a pass every 15 min for half a day
        run(cfg, pod, state, now=NOW + n * 900)
    assert len(pod.launched) == 3 and [e["state"] for e in state["launches"]] == ["unrun"] * 3
    assert "held: its last 3 launches ended alike, unrun (failed, rc 75)" in pod.lines[-1]
    assert [r[1] for r in pod.rang] == ["75", "75"]
    run(cfg, pod, state, now=NOW + 11 * 900 + R.HOLD_S)
    assert len(pod.launched) == 4


def test_the_trim_keeps_each_declared_numbers_newest_launches(tmp_path):
    other = [{"metrics": ["benchmarks/churn"], "state": "unrun", "run": f"r-{n}", "at": "2026-11-01T00:00:00Z",
              "ended": "2026-11-01T00:00:00Z"} for n in range(R.KEEP + 50)]
    mine = {"metrics": ["compute-accounting/share"], "command": CPU, "state": "passed", "run": run_id(1),
            "at": "2026-11-14T00:00:00Z", "ended": "2026-11-14T06:00:00Z"}
    record = [mine, *other]
    R.trim(record, {"compute-accounting/share", "benchmarks/churn"})
    assert record[0] is mine and len(record) == R.KEEP + 1
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod()
    run(cfg, pod, {"launches": record})
    assert pod.launched == [], "its newest launch, kept, still holds it for its every"


def test_the_stop_file_stops_every_launch(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    cfg.stop.write_text("")
    pod = Pod()
    state, saved = run(cfg, pod)
    assert pod.launched == [] and saved == [] and state == {}
    assert pod.lines == [f"stopped: {cfg.stop} exists, so nothing is launched; remove it to go on"]


def test_the_stop_file_is_read_again_before_each_launch(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(on_refresh=lambda: cfg.stop.write_text(""))     # touched while the pass refreshes the clone
    run(cfg, pod)
    assert pod.launched == []
    assert pod.lines[-1] == f"stopped: {cfg.stop} exists, so compute-accounting/share is not launched; remove it to go on"


def test_a_dry_run_says_what_it_would_launch_and_why_and_launches_nothing(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(ages={"compute-accounting/share": 9 * DAY_MS})
    state, saved = run(cfg, pod, dry_run=True)
    assert pod.launched == [] and saved == [] and state["launches"] == []
    assert pod.lines[-1].startswith("would launch compute-accounting/share on vy-nebius-cpu-12: compute-accounting/share's newest "
                                    "reading is 9.0 d old, past its every 7d: research run --on vy-nebius-cpu-12 --pool research")
    cfg.stop.write_text("")
    pod.lines.clear()
    run(cfg, pod, dry_run=True)
    assert pod.launched == [] and pod.lines[0].startswith("stopped: ") and pod.lines[-1].startswith("would launch "), (
        "a dry run judges while the stop file stands, so the lead sees what removing it would launch")


def test_a_gpu_command_runs_only_on_a_machine_the_plan_lists_for_it_now(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)],
                           "benchmarks/gpu": [row("vllm/other", "research run --gpus 1 --on auto -- bash benchmarks/gpu/run.sh")]})
    pod = Pod()
    state, _ = run(cfg, pod, dry_run=True)
    assert pod.launched == [] and sum("the plan (" in ln for ln in pod.lines) == 2
    cfg.plan.write_text(free("vy-h200-1", ["vllm/serving-cost-glm53"], start="2026-11-14T13:00:00Z"))
    run(cfg, pod)
    assert pod.launched == [], "the plan's window starts later"
    cfg.plan.write_text(free("vy-h200-1", ["vllm/serving-cost-glm53"]))
    state, _ = run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-h200-1", "--project", "verity", "--source", ".", "--cwd", "source",
                             "--", "bash", "benchmarks/serve_tax/run.sh", "--model", "glm53"]]
    assert any(ln.startswith("skip vllm/other: vllm/other has no reading; but it asks for GPUs, and the plan") for ln in pod.lines)
    cfg.plan.write_text(free("vy-h200-1", ["vllm/serving-cost-glm53", "vllm/other"]))
    run(cfg, pod, state)
    assert len(pod.launched) == 1, "the machine still runs this loop's launch"


def test_a_plan_row_applies_only_to_the_numbers_it_names(tmp_path):
    lease = row("kernels/lease", "research run --on NODE --mem-gb 128 -- bash benchmarks/lease/run.sh")
    cfg = setup(tmp_path, {"benchmarks/lease": [lease], "benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)]})
    cfg.plan.write_text(free("vy-h200-1"))
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == [] and "the plan's row for vy-h200-1 names no metrics, so it is free for none" in pod.lines
    cfg.plan.write_text(free("vy-h200-1", ["vllm/serving-cost-glm53"]))
    run(cfg, pod)
    assert [a[3] for a in pod.launched] == ["vy-h200-1"] and "benchmarks/serve_tax/run.sh" in pod.launched[0]


def test_a_row_naming_a_machine_goes_there_only_by_the_plans_row_for_that_machine(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/x": [row("benchmarks/x", "research run --on vy-h200-1 -- bash benchmarks/x/run.sh")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == [] and pod.lines[-1].endswith(
        f"but it names its machine (vy-h200-1), and the plan ({cfg.plan}) lists no machine free for it now")
    cfg.plan.write_text(free("vy-h200-2", ["benchmarks/x"]))
    run(cfg, pod)
    assert pod.launched == [], "the plan lists another machine for it"
    cfg.plan.write_text(free("vy-h200-1", ["benchmarks/x"]))
    run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-h200-1", "--project", "verity", "--source", ".", "--",
                             "bash", "benchmarks/x/run.sh"]]


def test_a_command_with_no_on_runs_on_the_loops_machine_like_one_on_auto(tmp_path):
    """A `research run` without `--on` would run where it is launched, the pod; unless its row says `runs = "control"`, the
    loop runs it as a CPU command instead. One that asks for GPUs still waits for the plan."""
    cfg = setup(tmp_path, {"benchmarks/tree": [row("top/tree", "research run --project verity --cwd source -- bash "
                                                               "benchmarks/tree/run.sh", every="12h")],
                           "benchmarks/gpu": [row("vllm/gpu", "research run --gpus 1 -- bash benchmarks/gpu/run.sh")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-nebius-cpu-12", "--pool", "research", "--source", ".", "--project",
                             "verity", "--cwd", "source", "--", "bash", "benchmarks/tree/run.sh"]]
    assert state["numbers"]["vllm/gpu"]["reason"] == "plan"


LANDING = "research run --project verity --cwd source -- bash benchmarks/landing/run.sh"


def control(i, command=LANDING, every="12h"):
    return {**row(i, command, every), "runs": "control"}


def write_list(tmp_path, entries):
    """The pod's list of CONTROL rows, as infra writes it on the pod: [[row]] id, command."""
    p = tmp_path / "metrics-rerun.control.toml"
    p.write_text("".join(f"[[row]]\nid = {json.dumps(i)}\ncommand = {json.dumps(c)}\n\n" for i, c in entries))
    return p


def listed(tmp_path, files, **kw):
    """`setup`, with the pod's list holding every CONTROL row of `files` with its command."""
    entries = [(r["id"], r["command"]) for rows in files.values() for r in rows if r.get("runs") == "control"]
    return setup(tmp_path, files, control_rows=write_list(tmp_path, entries), **kw)


def here(n):
    """A local run's output: `research run` with no `--on` names its directory, never a machine it launched on."""
    return f"research: run {run_id(n)} -> /root/.research/runs/{run_id(n)}\n"


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_control_row_runs_on_the_pod_itself_and_nowhere_else(tmp_path, dry_run):
    """The landing rows read the pod's own state, GitHub and the store's labels, which no node reaches: `runs = "control"`
    launches them as a local run on the pod, with no `--on` and no pool and its --timeout the pass waits out, and never on
    the loop's node; the pass judges the run by its own record as soon as it ends."""
    cfg = listed(tmp_path, {"benchmarks/landing": [control("ci/plain"), control("ci/lean")]})
    pod = Pod(out=here, states={run_id(1): ("done", 0, True, {"here": True})})
    state, _ = run(cfg, pod, dry_run=dry_run)
    argv = ["research", "run", "--source", ".", "--timeout", f"{R.CONTROL_S}", "--project", "verity", "--cwd", "source", "--",
            "bash", "benchmarks/landing/run.sh"]
    if dry_run:
        assert pod.launched == [] and f"would launch ci/plain, ci/lean on control: ci/plain has no reading; ci/lean has no " \
                                      f"reading: {' '.join(argv)}" in pod.lines
        return
    assert pod.launched == [argv]
    (e,) = state["launches"]
    assert (e["machine"], e["state"], e["argv"][:4]) == ("control", "passed", ["research", "run", "--id", run_id(1)])
    assert f"run {run_id(1)} (ci/plain, ci/lean): passed (done, rc 0)" in pod.lines


def test_a_control_rows_bounds_still_hold(tmp_path):
    """A row on the pod is held like any other: once it ran, it waits its every; when it fails it rings and is tried again
    after the hour; and --max and the stop file apply to it."""
    cfg = listed(tmp_path, {"benchmarks/landing": [control("ci/plain")],
                           "benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]}, max=1)
    pod = Pod(out=lambda n: here(n) if n == 2 else f"research: run {run_id(n)} launched\n",
              states={run_id(2): ("failed", 1, True, {"here": True})})
    state, _ = run(cfg, pod)
    assert [e["metrics"] for e in state["launches"]] == [["compute-accounting/share"]], "--max: the first in file order"
    state, _ = run(cfg, pod, state, now=NOW + 900)
    assert state["launches"][-1]["state"] == "failed" and pod.rang[0][:3] == (run_id(2), "1", "ci")
    run(cfg, pod, state, now=NOW + 1800)
    assert len(pod.launched) == 2 and "and it is launched again 60 min after" in pod.lines[-1]
    cfg.stop.write_text("")
    run(cfg, pod, state, now=NOW + 900 + 3600)
    assert len(pod.launched) == 2


def test_a_control_run_that_had_not_ended_when_its_launch_did_never_will(tmp_path):
    """The pass waits for a run on the pod; one whose record still says running once its launch returned lost its runner, and
    is a fault of the launch, launched again after the hour."""
    cfg = listed(tmp_path, {"benchmarks/landing": [control("ci/plain")]})
    pod = Pod(out=here, states={run_id(1): ("running", None, True, {"here": True})})
    state, _ = run(cfg, pod)
    (e,) = state["launches"]
    assert (e["state"], e["cause"]) == ("unlaunched", "unlaunched")
    assert e["why"].startswith(f"run {run_id(1)} on the control pod had not ended (running) when its launch did")


@pytest.mark.parametrize("option, said", [
    ("--on auto", "refuses --on"), ("--pool research", "refuses --pool"), ("--mem-gb 8", "refuses --mem-gb"),
    ("--store-read", "refuses --store-read"), ("--gpus 1", "refuses --gpus"), ("--cpus 4", "refuses --cpus"),
    (f"--timeout {R.CONTROL_S + 1}", f"may take (--timeout {R.CONTROL_S + 1} over {R.CONTROL_S:g})")])
def test_a_control_row_takes_no_option_that_places_or_reserves_a_run_elsewhere(tmp_path, option, said):
    cfg = setup(tmp_path, {"benchmarks/landing": [control("ci/plain", f"research run {option} -- bash benchmarks/landing/run.sh")]})
    pod = Pod(out=here)
    state, _ = run(cfg, pod)
    assert pod.launched == [] and said in pod.lines[-1], pod.lines[-1]
    assert state["numbers"]["ci/plain"]["reason"] == ("ceiling" if "--timeout" in option else "refused")


def test_a_runs_value_but_control_or_rows_of_one_command_that_disagree_are_refused(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/landing": [{**control("ci/plain"), "runs": "gpu"}],
                           "benchmarks/landing2": [control("ci/a", "research run -- bash benchmarks/landing2/run.sh"),
                                                   row("ci/b", "research run -- bash benchmarks/landing2/run.sh")]})
    pod = Pod(out=here)
    run(cfg, pod)
    assert pod.launched == []
    assert "skip ci/plain: ci/plain has no reading; but its runs is 'gpu', not 'control'" in pod.lines
    assert any(ln.startswith("skip ci/a, ci/b: ") and "give runs differently" in ln for ln in pod.lines)


COMMUNICATION = "research run --project verity --cwd source -- uv run python benchmarks/communication/response.py"
#: the entries infra writes in the pod's list for the tree's CONTROL rows
ENTRIES = [("ci/plain-mark-to-main-p90", LANDING), ("ci/lean-mark-to-main-p90", LANDING),
           ("top/lead-response-p90-minutes", COMMUNICATION), ("top/lead-asks-open-over-30-min", COMMUNICATION)]


def the_trees_control_rows(tmp_path, **kw):
    """`setup` with the tree's own CONTROL rows, as their metrics.toml files declare them."""
    tree = tmp_path / "tree"
    for where in ("benchmarks/landing", "benchmarks/communication"):
        (tree / where).mkdir(parents=True)
        (tree / where / "metrics.toml").write_text((ROOT / where / "metrics.toml").read_text())
    return setup(tmp_path, {}, **kw)


def test_a_listed_control_row_launches_on_the_pod(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/landing": [control("ci/plain")]}, control_rows=write_list(tmp_path, [("ci/plain", LANDING)]))
    pod = Pod(out=here, states={run_id(1): ("done", 0, True, {"here": True})})
    state, _ = run(cfg, pod)
    assert [e["machine"] for e in state["launches"]] == ["control"] and "--on" not in pod.launched[0]
    assert state["numbers"]["ci/plain"]["reason"] == "launched"


def test_a_control_row_the_pods_list_lacks_runs_nowhere_and_rings_its_owner_and_infra_once(tmp_path):
    """The word in a lane's file is not enough: the pod's own list must hold the row, or it is launched nowhere, not even on
    the loop's node, and its owner and infra are rung once with the entry to write."""
    cfg = setup(tmp_path, {"benchmarks/landing": [control("ci/plain")]},
                control_rows=write_list(tmp_path, [("ci/other", LANDING)]))
    pod = Pod(out=here)
    state = None
    for n in range(3):
        state, _ = run(cfg, pod, state, now=NOW + n * 900)
    line = (f"it says runs = 'control', but the control pod's list {cfg.control_rows.name} doesn't hold it (ci/plain's id is "
            f'missing), so it runs nowhere until infra adds there, on the pod, a [[row]] with id = "ci/plain", command = "{LANDING}"')
    assert pod.launched == [] and state["launches"] == []
    assert state["numbers"]["ci/plain"] == {"reason": "control-unlisted", "why": f"ci/plain has no reading; but {line}"}
    assert f"skip ci/plain: ci/plain has no reading; but {line}" in pod.lines
    key = f"control-unlisted:ci/plain:{R.Job('benchmarks/landing/metrics.toml', LANDING, (control('ci/plain'),)).declared}"
    assert pod.rang == [(key, "control-unlisted", o, f"metrics-rerun: ci/plain: {line}") for o in ("ci", "infra")]
    assert all(len(r[3]) <= 600 for r in pod.rang), "a doorbell carries 600 characters (slack.DOORBELL_MAX)"


def test_a_listed_id_whose_command_changed_is_refused_with_the_command_it_lists(tmp_path):
    changed = "research run --project verity --cwd source -- bash benchmarks/landing/run.sh --all"
    cfg = setup(tmp_path, {"benchmarks/landing": [control("ci/plain", changed)]},
                control_rows=write_list(tmp_path, [("ci/plain", LANDING)]))
    pod = Pod(out=here)
    state, _ = run(cfg, pod)
    assert pod.launched == [] and state["numbers"]["ci/plain"]["reason"] == "control-unlisted"
    why = state["numbers"]["ci/plain"]["why"]
    assert "(ci/plain's command differs from its entry)" in why
    assert why.endswith(f'a [[row]] with id = "ci/plain", command = "{changed}"')
    assert [r[2] for r in pod.rang] == ["ci", "infra"]


def test_each_row_of_a_shared_command_must_be_listed(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/landing": [control("ci/plain"), control("ci/lean")]},
                control_rows=write_list(tmp_path, [("ci/plain", LANDING)]))
    pod = Pod(out=here)
    state, _ = run(cfg, pod)
    assert pod.launched == [] and "(ci/lean's id is missing)" in state["numbers"]["ci/plain"]["why"]
    assert "ci/plain's" not in state["numbers"]["ci/lean"]["why"]


@pytest.mark.parametrize("contents, said", [(None, "is missing"), ("[[row]]\nid = 1\n", "is unreadable (ValueError: "),
                                            ("[[row\n", "is unreadable (TOMLDecodeError: ")])
def test_a_missing_or_unreadable_list_runs_none_of_the_trees_control_rows_and_says_so_once(tmp_path, contents, said):
    lst = tmp_path / "metrics-rerun.control.toml"
    if contents is not None:
        lst.write_text(contents)
    cfg = the_trees_control_rows(tmp_path, control_rows=lst)
    pod = Pod(out=here)
    state, _ = run(cfg, pod)
    assert pod.launched == [] and {i: state["numbers"][i]["reason"] for i, _ in ENTRIES} == dict.fromkeys(
        (i for i, _ in ENTRIES), "control-unlisted")
    assert sum(ln.startswith(f"the control pod's list of rows {lst} {said}") for ln in pod.lines) == 1, pod.lines
    assert sorted({r[2] for r in pod.rang}) == ["ci", "infra", "top"]
    assert all(len(r[3]) <= 600 for r in pod.rang), "a doorbell carries 600 characters (slack.DOORBELL_MAX): the fix is whole"
    assert 'a [[row]] for each of id = "ci/plain-mark-to-main-p90", id = "ci/lean-mark-to-main-p90", command = ' in (
        state["numbers"]["ci/plain-mark-to-main-p90"]["why"])


def test_the_entries_for_the_trees_control_rows_run_them_on_the_pod(tmp_path):
    cfg = the_trees_control_rows(tmp_path, control_rows=write_list(tmp_path, ENTRIES))
    pod = Pod(out=here, states={run_id(n): ("done", 0, True, {"here": True}) for n in (1, 2)})
    state, _ = run(cfg, pod)
    assert sorted(i for e in state["launches"] for i in e["metrics"]) == sorted(i for i, _ in ENTRIES)
    assert {e["machine"] for e in state["launches"]} == {"control"} and pod.rang == []


def test_a_node_row_is_untouched_by_the_list(tmp_path):
    """The list only lets a CONTROL row run on the pod: a row a node runs launches on its node, missing list or not, and even
    when the list holds its id with another command."""
    for lst in (tmp_path / "none.toml", write_list(tmp_path, [("compute-accounting/share", LANDING)])):
        cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]}, control_rows=lst)
        pod = Pod()
        state, _ = run(cfg, pod)
        assert [e["machine"] for e in state["launches"]] == ["vy-nebius-cpu-12"] and pod.rang == []
        assert state["numbers"]["compute-accounting/share"]["reason"] == "launched"


def test_classify_reads_the_pods_list_only_when_given_it(tmp_path):
    cfg = the_trees_control_rows(tmp_path)
    lines = []
    R.census(cfg.tree, lines.append)
    assert lines[0].startswith("the control pod's list of rows was not read (no --control-rows)")
    assert sum(ln.startswith("runnable") and ln.endswith("(on control)") for ln in lines) == 4
    lines = []
    counts = R.census(cfg.tree, lines.append, control_rows=tmp_path / "none.toml")
    assert counts == {"control-unlisted": 4} and sum("is missing, so no row runs on the control pod" in ln for ln in lines) == 1
    lines = []
    assert R.census(cfg.tree, lines.append, control_rows=write_list(tmp_path, ENTRIES)) == {"runnable": 4}


def test_a_control_rows_environment_keeps_the_linear_apps_keys_and_no_other_of_the_pods():
    pods = {"JOBS_TOKEN": "j", "SLACK_BOT_TOKEN": "s", "LINEAR_CLIENT_ID": "i", "LINEAR_CLIENT_SECRET": "c",
            "LINEAR_API_KEY": "k", "RESEARCH_STORE": "r", "AWS_ACCESS_KEY_ID": "a", "PATH": "p"}
    assert sorted(R.environment(pods, here=False)) == ["AWS_ACCESS_KEY_ID", "PATH", "RESEARCH_STORE"]
    assert sorted(R.environment(pods, here=True)) == ["AWS_ACCESS_KEY_ID", "LINEAR_CLIENT_ID", "LINEAR_CLIENT_SECRET", "PATH",
                                                      "RESEARCH_STORE"]


def test_the_pod_reads_a_run_here_from_its_own_directory(tmp_path, monkeypatch):
    """A local run has a status.json and no remote.json: it is read where it is, never fetched or taken for one that didn't
    reach its machine."""
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path))
    d = tmp_path / run_id(1)
    d.mkdir()
    pod = R.Pod(setup(tmp_path, {}))
    (d / "status.json").write_text(json.dumps({"state": "running", "transitions": [{"state": "running", "t_utc": "t0"}]}))
    assert {k: v for k, v in pod.status(run_id(1)).items() if k in ("state", "started", "here")} == {
        "state": "running", "started": True, "here": True}
    (d / "status.json").write_text(json.dumps({"state": "done", "rc": 0, "transitions": [
        {"state": "running", "t_utc": "t0"}, {"state": "done", "t_utc": "t1"}]}))
    assert {k: v for k, v in pod.status(run_id(1)).items() if k in ("state", "rc", "here")} == {
        "state": "done", "rc": 0, "here": True}
    (d / "status.json").unlink()
    assert pod.status(run_id(1))["state"] == R.UNSENT, "no status and no remote.json: a launch that never reached a machine"


def test_a_command_naming_a_cpu_node_runs_there_without_the_plan(tmp_path):
    """The dashboard's count names vy-nebius-cpu-9: a CPU node runs it in the loop's pool. A GPU machine, a small-disk node or
    `--on NODE` alone (a GPU lease's usual shape) still waits for the plan, or never runs."""
    cfg = setup(tmp_path, {"infra/dashboard": [row("console/top-goals-running", "research run --on vy-nebius-cpu-9 --project "
                                                   "verity --source . --cwd source -- uv run python infra/dashboard/run.py",
                                                   every="1d")],
                           "benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)],
                           "benchmarks/x": [row("benchmarks/x", "research run --on vy-nebius-cpu-5 -- bash benchmarks/x/run.sh")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-nebius-cpu-9", "--pool", "research", "--project", "verity",
                             "--source", ".", "--cwd", "source", "--", "uv", "run", "python", "infra/dashboard/run.py"]]
    assert {i: n["reason"] for i, n in state["numbers"].items()} == {
        "console/top-goals-running": "launched", "vllm/serving-cost-glm53": "plan", "benchmarks/x": "small-disk"}


def test_a_launch_that_names_no_run_is_a_fault_of_the_launch_and_never_holds_the_command(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod(out=lambda n: "Traceback: ImportError\n" if n <= 2 else f"research: run {run_id(n)} launched\n")
    state, _ = run(cfg, pod)
    run(cfg, pod, state, now=NOW + 3600)
    assert [e["state"] for e in state["launches"]] == ["unlaunched", "unlaunched"]
    assert any("a fault of the launch and not of the command" in ln for ln in pod.lines)
    run(cfg, pod, state, now=NOW + 5400)
    assert len(pod.launched) == 2 and "unlaunched at 2026-11-14T13:00:00Z, and it is launched again" in pod.lines[-1]
    assert [r[1] for r in pod.rang] == ["unlaunched"], "the second ended as the first, whose ring said so"
    run(cfg, pod, state, now=NOW + 7200)
    assert len(pod.launched) == 3 and state["launches"][-1]["state"] == "submitted"


def test_a_launch_its_timeout_killed_after_its_run_registered_is_never_launched_again(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod(out=lambda n: "research: shipping the tree\n", states={run_id(1): ("waiting", None, False)})
    state, _ = run(cfg, pod)
    (e,) = state["launches"]
    assert (e["run"], e["state"]) == (run_id(1), "waiting")
    assert pod.lines[-1].startswith("launch of compute-accounting/share named no run (exit 0: research: shipping the tree), "
                                    f"but run {run_id(1)} is registered (waiting): it is in flight")
    for n in range(1, 20):                                    # past the launch's timeout and its every, a pass each 15 min
        run(cfg, pod, state, now=NOW + n * 900)
    assert len(pod.launched) == 1
    pod.states[run_id(1)] = ("done", 0, True)
    run(cfg, pod, state, now=NOW + 20 * 900)
    assert len(pod.launched) == 1 and state["launches"][0]["state"] == "passed"


def test_a_launch_that_stopped_before_its_run_reached_its_machine_never_ran(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod(out=lambda n: "", states={run_id(1): (R.UNSENT, None, False)})
    state, _ = run(cfg, pod)
    assert state["launches"][0]["state"] == "unlaunched"
    assert (f"launch of compute-accounting/share named no run (exit 0) and run {run_id(1)} never reached its "
            "machine: it printed nothing, a fault of the launch and not of the command") in pod.lines


def test_the_loop_names_each_run_and_records_its_id_before_launching(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod()
    state, saved = run(cfg, pod)
    assert saved[0]["launches"][0]["run"] == run_id(1) and saved[0]["launches"][0]["state"] == "launching"
    assert state["launches"][0]["argv"][:4] == ["research", "run", "--id", run_id(1)]


def test_a_row_that_does_not_split_is_refused_and_the_pass_goes_on(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/a": [row("benchmarks/a", "research run --on auto -- bash -lc 'benchmarks/a/run.sh")],
                           "benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod()
    run(cfg, pod)
    assert len(pod.launched) == 1
    assert any(ln.startswith("skip benchmarks/a: benchmarks/a has no reading; but its command doesn't split as a shell line")
               for ln in pod.lines)


def test_an_on_equals_auto_row_takes_the_loops_machine(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/b": [row("benchmarks/b", "research run --on=auto --cwd=clone -- bash benchmarks/b/run.sh")]})
    pod = Pod()
    run(cfg, pod)
    (argv,) = pod.launched
    assert argv[:4] == ["research", "run", "--on", "vy-nebius-cpu-12"] and not any(a.startswith("--on=") for a in argv)
    assert argv[argv.index("--cwd") + 1] == "clone"


def test_a_run_whose_state_cannot_be_read_stays_in_flight(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    pod.states[run_id(1)] = ConnectionError("its machine is unreachable")
    run(cfg, pod, state, now=NOW + DAY)
    assert len(pod.launched) == 1 and state["launches"][0]["state"] == "submitted"
    assert any("not read (ConnectionError: its machine is unreachable); it stays in flight" in ln for ln in pod.lines)


def test_a_plan_row_past_its_until_lists_nothing(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)]})
    cfg.plan.write_text(free("vy-h200-1", ["vllm/serving-cost-glm53"], start="2026-11-13T00:00:00Z", until="2026-11-14T12:00:00Z"))
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []


def test_a_launch_waiting_for_its_slot_keeps_its_machine_busy(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)],
                           "benchmarks/gpu": [row("vllm/other", "research run --gpus 1 --on NODE -- bash benchmarks/gpu/run.sh")]})
    cfg.plan.write_text(free("vy-h200-1", ["vllm/serving-cost-glm53", "vllm/other"]))
    pod = Pod(states={run_id(1): ("waiting", None, False)})
    state, _ = run(cfg, pod, {"launches": []})
    assert len(pod.launched) == 1
    run(cfg, pod, state, now=NOW + 900)
    assert len(pod.launched) == 1 and state["launches"][0]["state"] == "waiting"


def test_the_loop_reads_research_runs_own_ended_states():
    assert R.ENDED is runs.ENDED


def test_a_pass_finding_the_lock_held_does_nothing(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod()
    with open(cfg.lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, pod, dry_run=True)
        run(cfg, pod)
    assert pod.launched == [] and pod.lines == [
        f"another pass holds {cfg.lock} (the live loop, a dry run or a second copy), so this one does nothing"] * 2
    run(cfg, pod)
    assert len(pod.launched) == 1, "the lock is free again"


def test_a_corrupt_record_is_moved_aside_said_once_and_every_number_waits_its_every(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1d")]})
    cfg.state.write_text("{not json")
    said = []
    state = R.load_state(cfg.state, NOW, said.append, dry_run=True)
    assert cfg.state.read_text() == "{not json" and said[0].startswith(f"the record {cfg.state} is not a JSON object "
                                                                        "(JSONDecodeError: ")
    assert "it would be moved aside" in said[0]
    state = R.load_state(cfg.state, NOW, said.append)
    assert not cfg.state.exists() and (tmp_path / "state.json.corrupt").read_text() == "{not json" and len(said) == 2
    assert state == {"schema": R.SCHEMA, "held_since": "2026-11-14T12:00:00Z", "launches": []}
    pod = Pod()
    run(cfg, pod, state, now=NOW + 3600)
    assert pod.launched == [] and "the record was reset at 2026-11-14T12:00:00Z" in pod.lines[-1]
    run(cfg, pod, state, now=NOW + DAY)
    assert len(pod.launched) == 1


def test_a_number_the_door_does_not_list_is_said_and_not_due(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(listed=[])
    run(cfg, pod)
    assert pod.launched == [] and pod.lines[-1] == (
        "skip compute-accounting/share: the door doesn't list compute-accounting/share, so it isn't due")


def test_never_auto_nor_a_small_disk_machine(tmp_path, capsys):
    args = ["--tree", "t", "--state", "s", "--stop", "x", "--plan", "p", "--pool", "research", "--every", "900", "--on"]
    for bad in ("auto", "vy-nebius-cpu-4"):
        with pytest.raises(SystemExit):
            R.parse([*args, bad])
        assert f"{bad} is refused" in capsys.readouterr().err
    cfg = setup(tmp_path, {"benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)],
                           "benchmarks/x": [row("benchmarks/x", "research run --on vy-nebius-cpu-6 -- bash benchmarks/x/run.sh")]})
    cfg.plan.write_text(free("vy-nebius-cpu-5", ["vllm/serving-cost-glm53"]) + free("vy-nebius-cpu-6", ["benchmarks/x"]))
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []


def test_a_command_that_takes_a_run_id_is_never_launched(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/memory": [row("memory-accounting/rate", "research run --on auto --store-read -- env "
                                                                              "CARD=CARD uv run python benchmarks/memory/run.py "
                                                                              "--from FIRE_RUN")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []
    assert pod.lines[-1].endswith("but its command takes CARD, FIRE_RUN, which only its owner fills in")


def test_the_checked_in_loop_parses_and_launches_nowhere_it_must_not():
    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "metrics-rerun"]
    assert lp.argv[2] == "/data/watchers/src/infra/control/metrics_rerun.py"
    cfg, ns = R.parse(lp.argv[3:])
    assert not ns.dry_run and not ns.once and cfg.max >= 1 and ns.every == 900, "the row declares its interval"
    assert cfg.stop.parent == cfg.state.parent == Path("/data/watchers")
    assert "--max-in-flight" in lp.argv and cfg.max_in_flight == 4, "the row declares its limit on runs in flight"
    assert cfg.control_rows == Path("/data/watchers/metrics-rerun.control.toml"), "the pod's list is the pod's, beside its state"
    args = ["--tree", "t", "--state", "s", "--stop", "x", "--plan", "p", "--pool", "research", "--every", "900", "--on", "m"]
    with pytest.raises(SystemExit):
        R.parse(args)
    assert R.parse([*args, "--control-rows", "c"])[0].max_in_flight == 4


PRICED = "research run --on auto --store-read --source . --cwd source -- uv run python benchmarks/serving_cost/run.py --capture CAPTURE"


def test_a_placeholder_rings_its_owner_once_for_each_declaration_and_the_record_says_so(tmp_path):
    """A CPU command that holds a value only its owner fills in is never launched: its owner is rung once, the record says
    why, and a change to its rows that still holds one rings again."""
    cfg = setup(tmp_path, {"benchmarks/serving_cost": [row("benchmarks/serving-cost-priced", PRICED)]})
    pod = Pod()
    state, _ = run(cfg, pod, dry_run=True)
    assert pod.rang == [] and any(ln.startswith("would ring @benchmarks: ") for ln in pod.lines)
    for n in range(3):
        state, _ = run(cfg, pod, state if n else None, now=NOW + n * 900)
    assert pod.launched == [] and pod.rang == [(
        "placeholder:benchmarks/serving-cost-priced", "placeholder", "benchmarks",
        "metrics-rerun: benchmarks/serving-cost-priced: its command takes CAPTURE, which only its owner fills in, so the loop "
        "launches it only once its command in benchmarks/serving_cost/metrics.toml holds none")]
    assert state["numbers"]["benchmarks/serving-cost-priced"]["reason"] == "placeholder"
    (cfg.tree / "benchmarks/serving_cost/metrics.toml").write_text(toml([row("benchmarks/serving-cost-priced", PRICED, "1d")]))
    state, _ = run(cfg, pod, state, now=NOW + DAY)
    assert len(pod.rang) == 2 and len(state["rings"]) == 1, "the old declaration's ring is dropped"


def test_a_placeholder_in_a_command_the_plan_places_rings_nobody(tmp_path):
    """A card a GPU lease takes (CARD) waits for the plan as well: filling it in alone wouldn't launch the number."""
    lease = row("kernels/lease", "research run --on NODE -- bash benchmarks/lease/run.sh --on CARD")
    cfg = setup(tmp_path, {"benchmarks/lease": [lease]})
    pod = Pod()
    state, _ = run(cfg, pod)
    assert pod.rang == [] and state["numbers"]["kernels/lease"]["reason"] == "placeholder"


def test_the_record_says_each_declared_numbers_reason(tmp_path):
    loop = row("infra/loop-number", "sh infra/control/store_index.sh", every="5m")
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/a", CPU)],
                           "benchmarks/tree": [row("top/fresh", "research run --on auto -- bash benchmarks/tree/x.sh", "1d"),
                                               row("top/gone", "research run --on auto -- bash benchmarks/tree/y.sh", "1d")],
                           "benchmarks/landing": [row("ci/landing", "research run -- bash benchmarks/landing/run.sh")],
                           "infra/control": [loop]}, max=1)
    pod = Pod(ages={"top/fresh": DAY_MS // 2}, listed=["compute-accounting/a", "top/fresh", "ci/landing", "infra/loop-number"])
    state, _ = run(cfg, pod)
    assert {i: n["reason"] for i, n in state["numbers"].items()} == {
        "ci/landing": "max", "compute-accounting/a": "launched", "infra/loop-number": "loop", "top/fresh": "fresh",
        "top/gone": "unlisted"}
    assert set(r for n in state["numbers"].values() for r in [n["reason"]]) <= set(R.REASONS)
    assert state["judged"] == {"at": "2026-11-14T12:00:00Z", "commit": "c0ffee" + "0" * 34}
    assert any("by reason: " in ln and "launched 1" in ln and "max 1" in ln for ln in pod.lines)


def test_a_number_the_stop_file_stops_before_its_launch_says_so(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(on_refresh=lambda: cfg.stop.write_text(""))
    state, _ = run(cfg, pod)
    assert pod.launched == [] and state["numbers"]["compute-accounting/share"]["reason"] == "stopped"


def test_the_census_classifies_each_number_from_the_files_alone(tmp_path):
    """`classify` reads only the declarations: no door, record, plan, clock or network."""
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/a", CPU)],
                           "benchmarks/landing": [row("ci/landing", "research run -- bash benchmarks/landing/run.sh")],
                           "benchmarks/serving_cost": [row("benchmarks/priced", PRICED)],
                           "benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)],
                           "benchmarks/r": [{**row("benchmarks/r", RECURSION, "30d"), "goal": "top/parked"}],
                           "benchmarks/big": [row("benchmarks/big", "research run --on auto --mem-gb 900 -- bash benchmarks/big/x.sh")],
                           "benchmarks/bad": [row("benchmarks/bad", "research run --on auto --gpu-host -- bash benchmarks/bad/x.sh")],
                           "infra/dashboard": [row("console/n", "research run --on vy-nebius-cpu-9 -- uv run python infra/dashboard/run.py")],
                           "infra/control": [row("infra/loop", "sh infra/control/store_index.sh", "5m")]})
    lines = []
    counts = R.census(cfg.tree, lines.append)
    assert counts == {"runnable": 3, "placeholder": 1, "plan": 1, "parked": 1, "ceiling": 1, "refused": 1, "loop": 1}
    assert "runnable     console/n: launched once it is due (classify) (on vy-nebius-cpu-9)" in lines
    assert lines[-1] == "9 numbers: runnable 3, loop 1, parked 1, refused 1, ceiling 1, placeholder 1, plan 1"


def test_the_census_runs_from_the_command_line(tmp_path, capsys):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/a", CPU)]})
    assert R.main(["classify", "--tree", str(cfg.tree)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "1 numbers: runnable 1"


def test_a_launch_holds_none_of_the_pods_keys():
    held = ["JOBS_TOKEN", "SLACK_BOT_TOKEN", "LINEAR_CLIENT_SECRET"]
    kept = ["RESEARCH_STORE", "RESEARCH_LANE", "PATH", "XSLACK_BOT_TOKEN"]
    assert [k for k in held + kept if R.LEFT_OUT.match(k)] == held


def test_the_loop_rings_with_the_relays_slack_key():
    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "metrics-rerun"]
    assert "/root/.config/verity/relay.env" in lp.env_files


def flying(n):
    """`n` launches of the loop's, of numbers no file declares, queued or running on the loop's machine."""
    return [{"metrics": [f"other/n{k}"], "run": run_id(90 + k), "at": "2026-11-14T11:50:00Z", "state": "running",
             "machine": "vy-nebius-cpu-12"} for k in range(n)]


def test_no_launch_past_max_in_flight_counting_this_passes_and_the_pass_says_how_many_it_held(tmp_path):
    """With three of its runs queued or running and a limit of four, a pass launches one of two due numbers and holds the other
    back for the limit, whatever --max allows; once one of the three ends, the next pass launches it."""
    files = {f"benchmarks/b{n}": [row(f"benchmarks/n{n}", f"research run --on auto -- bash benchmarks/b{n}/run.sh")] for n in range(2)}
    cfg = setup(tmp_path, files, max=2)
    pod = Pod(states={run_id(90 + k): RUNNING for k in range(3)})
    state, _ = run(cfg, pod, {"launches": flying(3)})
    assert len(pod.launched) == 1
    assert any("3 in flight (--max-in-flight 4), 1 held back for it; by reason:" in ln for ln in pod.lines)
    assert ("skip benchmarks/n1: benchmarks/n1 has no reading; but 4 of this loop's launches are in flight, its "
            "--max-in-flight 4") in pod.lines
    assert state["numbers"]["benchmarks/n1"]["reason"] == "limit"
    pod.states[run_id(90)] = ("done", 0, True)
    run(cfg, pod, state, now=NOW + 900)
    assert len(pod.launched) == 2 and pod.launched[1][-1] == "benchmarks/b1/run.sh"


def test_a_pass_launches_nothing_while_its_limit_is_in_flight(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]}, max_in_flight=2)
    pod = Pod(states={run_id(90 + k): ("queued", None, False) for k in range(2)})
    run(cfg, pod, {"launches": flying(2)})
    assert pod.launched == [] and any("2 in flight (--max-in-flight 2), 1 held back for it" in ln for ln in pod.lines)


WAITED = ("failed", 75, False, {"waited": True})


def test_a_slot_timeout_is_launched_again_never_held_and_named_as_congestion(tmp_path):
    """A run whose pool had no slot for it within the wait never started: it counts toward no hold however often it happens,
    and is launched again at the pass that sees it end, which names it on one congestion line."""
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1d")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    for n in range(1, 5):
        pod.states[run_id(n)] = WAITED
        run(cfg, pod, state, now=NOW + n * 4 * 3600)
    assert len(pod.launched) == 5
    assert [e["state"] for e in state["launches"]] == ["waited"] * 4 + ["submitted"]
    assert all("cause" not in e for e in state["launches"])
    (line,) = [ln for ln in pod.lines if ln.startswith("congestion:") and run_id(4) in ln]
    assert line == ("congestion: 1 run(s) had no slot within their pool's wait and never started, each launched again on a "
                    f"later pass, with no count toward a hold and a ring only once 3 of a number's launches running have: "
                    f"{run_id(4)} (compute-accounting/share) on vy-nebius-cpu-12")


def test_three_slot_timeouts_running_ring_the_owner_once_and_again_only_after_a_launch_that_started(tmp_path):
    """A pool that stays full shows to the number's owner: the third launch running that waited out its line rings them once,
    naming the three runs and the machine; a fourth and fifth don't; after a launch that started, three more ring again."""
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    ends = [WAITED] * 5 + [("done", 0, True)] + [WAITED] * 3
    for n, end in enumerate(ends, start=1):
        pod.states[run_id(n)] = end
        run(cfg, pod, state, now=NOW + n * 7200 - 3600)      # sees it end: a slot timeout is launched again here
        run(cfg, pod, state, now=NOW + n * 7200)             # a run that passed, its every on
        if n == 2:
            assert pod.rang == [], "two slot timeouts running ring nobody"
    assert len(pod.launched) == len(ends) + 1
    assert [r[:3] for r in pod.rang] == [(run_id(3), "waited", "compute-accounting"), (run_id(9), "waited", "compute-accounting")]
    assert pod.rang[0][3] == (f"metrics-rerun: compute-accounting/share: its last 3 launches ({run_id(1)}, {run_id(2)}, "
                              f"{run_id(3)}) each waited out vy-nebius-cpu-12's pool line and never started; it is launched "
                              "again at a later pass, and rung again only after a launch of it that starts")
    assert [e.get("rung") is not None for e in state["launches"]][-1] is False, "the newest launch is in flight, unrung"
    assert not any("held:" in ln for ln in pod.lines), "a slot wait holds nothing"


def test_slot_timeouts_between_failures_neither_count_nor_break_their_streak(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    ends = [("failed", 1, True), ("failed", 75, False, {"waited": True})] * 2 + [("failed", 1, True)]
    for n, end in enumerate(ends, start=1):
        pod.states[run_id(n)] = end
        run(cfg, pod, state, now=NOW + n * 7200 - 3600)      # sees it end: a slot timeout is launched again here
        run(cfg, pod, state, now=NOW + n * 7200)             # a failure, an hour on
    assert len(pod.launched) == 5, "the third failure holds it"
    assert [r[0] for r in pod.rang] == [run_id(1), run_id(5)] and "held: it ended this way 3 times" in pod.rang[1][3]
    assert state["numbers"]["compute-accounting/share"]["reason"] == "failing"


def test_a_slot_timeout_is_told_from_a_workloads_own_exit_75_by_its_slot_record():
    left = {"state": "left", "since": "2026-11-14T08:00:00Z", "left": "2026-11-14T12:00:00Z",
            "why": "no check slot within --wait-max 14400 s (2 check(s) ahead in line)"}
    assert R.slot_timeout({"rc": 75, "slot": left})
    assert not R.slot_timeout({"rc": 75, "slot": {"state": "held", "slot": "s1"}}), "it took a slot: the workload exited 75"
    assert not R.slot_timeout({"rc": 75, "slot": None}), "no pool: the workload exited 75"
    assert not R.slot_timeout({"rc": 75, "slot": {**left, "why": "its twin took a slot first (claimed)"}})
    assert not R.slot_timeout({"rc": 1, "slot": left})


def test_the_slot_wait_line_is_the_one_slot_py_writes():
    assert f'gone(f"{R.SLOT_WAIT} {{wait_max:g}} s' in (ROOT / "tools/check/slot.py").read_text(encoding="utf-8")


def test_the_record_is_on_disk_before_its_rename_and_the_rename_after(tmp_path, monkeypatch):
    """The record's bytes are synced before the rename, and its directory after it, so a crash brings back neither a torn
    record nor one without the launch the pass recorded before starting it, which would launch that run again."""
    events = []
    monkeypatch.setattr(R.os, "fsync", lambda fd: events.append(("fsync", R.os.path.realpath(f"/proc/self/fd/{fd}"))))
    real = R.Path.replace
    monkeypatch.setattr(R.Path, "replace", lambda self, to: (events.append(("replace", str(to))), real(self, to))[1])
    R.save_state(tmp_path / "state.json", {"launches": []})
    assert events == [("fsync", str(tmp_path / "state.json.tmp")), ("replace", str(tmp_path / "state.json")),
                      ("fsync", str(tmp_path))]
    assert json.loads((tmp_path / "state.json").read_text()) == {"launches": []}
