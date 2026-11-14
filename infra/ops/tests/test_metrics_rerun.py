"""The control pod's reruns of the declared numbers (infra/control/metrics_rerun.py): a number older than its `every` is launched
as a `research run` on a machine, at most `--max` a pass; never one whose last launch is in flight or ran less than its `every`
ago, whatever the door says; a command that failed twice running is held until its declaration changes, and launches that
never ran are held after two in a row until its `every`; the stop file stops every launch, read again before each; a dry run
launches nothing; a command asking for GPUs or naming a machine runs only on a machine the plan lists for it now; never `auto`
or a small-disk machine; a row's options outside the allowed set are refused; a pass takes a lock. The clone, the door, the
runs' records, the launch and the clock are stand-ins: no test reads the clock or the network."""
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

    def __init__(self, ages=None, states=None, out=None, listed=None, on_refresh=None):
        self.ages, self.states, self.lines, self.launched = ages or {}, states or {}, [], []
        self.listed, self.on_refresh, self.named, self.registered = listed, on_refresh, 0, set()
        self.out = out or (lambda n: f"research: run {run_id(n)} launched\n")

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
        return {"state": s[0], "rc": s[1], "started": s[2]}

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


def test_a_command_that_failed_twice_running_is_held_until_its_declaration_changes(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    pod.states[run_id(1)] = ("failed", 1, True)
    run(cfg, pod, state, now=NOW + 7200)                      # seen ended now: less than its every ago
    run(cfg, pod, state, now=NOW + 2 * 7200)
    pod.states[run_id(2)] = ("failed", 1, True)
    run(cfg, pod, state, now=NOW + 3 * 7200)
    run(cfg, pod, state, now=NOW + 4 * 7200)
    assert len(pod.launched) == 2
    assert pod.lines[-1] == ("skip compute-accounting/share: compute-accounting/share has no reading; but held: its command "
                             f"failed twice running ({run_id(1)}, {run_id(2)}) under this declaration, and is launched "
                             "again once its rows in benchmarks/compute/metrics.toml change")
    (cfg.tree / "benchmarks/compute/metrics.toml").write_text(toml([row("compute-accounting/share", CPU, every="2h")]))
    run(cfg, pod, state, now=NOW + 5 * 7200)
    assert len(pod.launched) == 3


def test_a_row_with_an_option_outside_the_allowed_set_is_refused_naming_it(tmp_path):
    bad = {"--send /data/watchers/metrics-rerun.env": "takes --send, which a declared command may not carry",
           "--source /data/steward/research-notes": "gives --source '/data/steward/research-notes', which isn't .",
           "--runs-dir /data/research": "takes --runs-dir", "--id r20261114-000000-0000": "takes --id",
           "--project other": "gives --project 'other', which isn't verity", "--pool=../x": "gives --pool '../x'",
           "--store-read=1": "gives the flag --store-read a value", "--pool a --pool b": "gives --pool twice"}
    files = {f"benchmarks/b{n}": [row(f"benchmarks/n{n}", f"research run --on auto {opt} -- bash benchmarks/b{n}/run.sh")]
             for n, opt in enumerate(bad)}
    cfg = setup(tmp_path, files)
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []
    for n, why in enumerate(bad.values()):
        (line,) = [ln for ln in pod.lines if ln.startswith(f"skip benchmarks/n{n}:")]
        assert why in line, line


def test_launches_that_never_ran_are_held_after_two_in_a_row_and_tried_again_after_its_every(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1d")]})
    pod = Pod()
    state, _ = run(cfg, pod)
    pod.states[run_id(1)] = ("failed", 75, False)             # its pool refused it at once: it never took a slot
    run(cfg, pod, state, now=NOW + 900)
    pod.states[run_id(2)] = ("failed", 75, False)
    for n in range(2, 50):                                    # a pass every 15 min for half a day
        run(cfg, pod, state, now=NOW + n * 900)
    assert len(pod.launched) == 2 and [e["state"] for e in state["launches"]] == ["unrun", "unrun"]
    assert pod.lines[-1].endswith(f"held: its last 2 launches never ran ({run_id(1)} unrun, {run_id(2)} unrun), so it is "
                                  "tried again once compute-accounting/share's every 1d has passed since 2026-11-14T12:30:00Z")
    run(cfg, pod, state, now=NOW + 1800 + DAY)
    assert len(pod.launched) == 3


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
    scan = row("network-accounting/scan", "research run --on NODE --pool network --mem-gb 128 -- bash benchmarks/scan/run.sh")
    cfg = setup(tmp_path, {"benchmarks/scan": [scan], "benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)]})
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


def test_a_local_run_is_never_launched(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/landing": [row("ci/landing", "research run --project verity --cwd source -- bash "
                                                                    "benchmarks/landing/run.sh", every="12h")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == [] and pod.lines[-1].endswith(
        "but its command is a local run (a `research run` without --on), which runs where it is launched, and the loop runs "
        "no workload on the pod")


def test_a_launch_that_names_no_run_is_a_fault_of_the_launch_and_never_holds_the_command(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod(out=lambda n: "Traceback: ImportError\n" if n <= 2 else f"research: run {run_id(n)} launched\n")
    state, _ = run(cfg, pod)
    run(cfg, pod, state, now=NOW + 900)
    assert [e["state"] for e in state["launches"]] == ["unlaunched", "unlaunched"]
    assert "a fault of the launch and not of the command" in pod.lines[-1]
    run(cfg, pod, state, now=NOW + 1800)
    assert len(pod.launched) == 2 and "never ran" in pod.lines[-1] and "failed twice" not in pod.lines[-1]
    run(cfg, pod, state, now=NOW + 900 + 3601)
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
    assert pod.lines[-1] == (f"launch of compute-accounting/share named no run (exit 0) and run {run_id(1)} never reached its "
                             "machine: it printed nothing, a fault of the launch and not of the command")


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
    cfg = setup(tmp_path, {"benchmarks/b": [row("benchmarks/b", "research run --on=auto --timeout=600 -- bash benchmarks/b/run.sh")]})
    pod = Pod()
    run(cfg, pod)
    (argv,) = pod.launched
    assert argv[:4] == ["research", "run", "--on", "vy-nebius-cpu-12"] and not any(a.startswith("--on=") for a in argv)
    assert argv[argv.index("--timeout") + 1] == "600"


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
    cfg = setup(tmp_path, {"benchmarks/memory": [row("memory-accounting/rate", "research run --on auto --store-read -- uv run "
                                                                              "python benchmarks/memory/run.py --from FIRE_RUN")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []
    assert pod.lines[-1].endswith("but its command takes FIRE_RUN, which only its owner fills in")


def test_the_checked_in_loop_parses_and_launches_nowhere_it_must_not():
    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "metrics-rerun"]
    assert lp.argv[2] == "/data/watchers/src/infra/control/metrics_rerun.py"
    cfg, ns = R.parse(lp.argv[3:])
    assert not ns.dry_run and not ns.once and cfg.max >= 1 and ns.every == 900, "the row declares its interval"
    assert cfg.stop.parent == cfg.state.parent == Path("/data/watchers")
