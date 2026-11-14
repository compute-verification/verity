"""The control pod's reruns of the declared numbers (infra/control/metrics_rerun.py): a number older than its `every` is launched
as a `research run` on a machine, at most `--max` a pass; never one whose last launch still runs or ended less than its `every`
ago, whatever the door says; a command that failed twice is held until its declaration changes; the stop file stops every
launch; a dry run launches nothing; a GPU command runs only on a machine the plan lists free now; never `auto` or a small-disk
machine. The clone, the door, the runs' records, the launch and the clock are stand-ins: no test reads the clock or the network."""
import calendar
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
DAY_MS = 86_400_000
CPU = "research run --on auto --project verity --source . --cwd source --store-read -- uv run python benchmarks/compute/run.py"
GPU = "research run --on NODE --source . --cwd source -- bash benchmarks/serve_tax/run.sh --model glm53"


def row(i, command, every="7d"):
    return {"id": i, "title": "t", "owner": i.split("/")[0], "unit": "s", "better": "lower", "command": command, "every": every}


def toml(rows):
    out = []
    for r in rows:
        out.append("[[metric]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in r.items()))
    return "\n".join(out)


class Pod:
    """The pass's world: the door's ages, each run's state, and each launch's output; it records what it was asked."""

    def __init__(self, ages=None, states=None, out=None):
        self.ages, self.states, self.lines, self.launched = ages or {}, states or {}, [], []
        self.out = out or (lambda n: f"research: run r20261114-1200{n:02d}-ab12 launched\n")

    def log(self, msg):
        self.lines.append(msg)

    def refresh(self):
        return "c0ffee" + "0" * 34

    def door(self):
        return {"metrics": [{"id": i, "ageMs": a} for i, a in self.ages.items()]}

    def status(self, run):
        return self.states.get(run, ("running", None))

    def launch(self, argv):
        self.launched.append(argv)
        return 0, self.out(len(self.launched))


def setup(tmp_path, files, **kw):
    tree = tmp_path / "tree"
    for where, rows in files.items():
        (tree / where).mkdir(parents=True, exist_ok=True)
        (tree / where / "metrics.toml").write_text(toml(rows))
    cfg = R.Settings(tree, tmp_path / "state.json", tmp_path / "stop", tmp_path / "plan.toml", "vy-nebius-cpu-12", "research", **kw)
    return cfg


def run(cfg, pod, state=None, now=NOW, dry_run=False):
    state = {} if state is None else state
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
        ["compute-accounting/share"], "r20261114-120001-ab12", "2026-11-14T12:00:00Z", "vy-nebius-cpu-12", "submitted")
    assert saved[0]["launches"][0]["state"] == "launching", "the record holds the launch before it starts"


def test_rows_sharing_a_command_are_one_launch_and_a_fresh_number_or_a_loops_is_never_launched(tmp_path):
    loop = row("infra/loop-number", "sh infra/control/store_index.sh", every="5m")
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/a", CPU), row("compute-accounting/b", CPU)],
                           "benchmarks/tree": [row("top/fresh", "research run --on auto --cwd source -- uv run python benchmarks/tree/x.py",
                                                   every="1d")],
                           "infra/control": [loop]})
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


def test_never_a_number_whose_last_launch_runs_or_ended_less_than_its_every_ago_whatever_the_door_says(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1d")]})
    pod = Pod()                                               # the door holds no reading, every pass
    state, _ = run(cfg, pod)
    run(cfg, pod, state, now=NOW + 3 * 86400)
    assert len(pod.launched) == 1
    assert pod.lines[-1].endswith("its last launch, run r20261114-120001-ab12 at 2026-11-14T12:00:00Z, still runs")
    pod.states["r20261114-120001-ab12"] = ("done", 0)
    run(cfg, pod, state, now=NOW + 4 * 86400)
    assert len(pod.launched) == 1, "it ended at this pass: less than its every ago"
    assert pod.lines[-1].endswith("its last launch, run r20261114-120001-ab12, passed at 2026-11-18T12:00:00Z, less than "
                                  "compute-accounting/share's every 1d ago")
    run(cfg, pod, state, now=NOW + 5 * 86400 + 1)
    assert len(pod.launched) == 2


def test_a_command_that_failed_twice_is_held_until_its_declaration_changes(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU, every="1h")]})
    pod = Pod(out=lambda n: "research: no machine answered\n" if n == 1 else f"research: run r20261114-1200{n:02d}-ab12 launched\n")
    state, _ = run(cfg, pod)
    assert state["launches"][0]["state"] == "failed", "a launch that names no run has failed"
    run(cfg, pod, state, now=NOW + 7200)
    pod.states["r20261114-120002-ab12"] = ("failed", 1)
    run(cfg, pod, state, now=NOW + 3 * 7200)
    run(cfg, pod, state, now=NOW + 5 * 7200)
    assert len(pod.launched) == 2
    assert pod.lines[-1] == ("skip compute-accounting/share: compute-accounting/share has no reading; but held: its command "
                             "failed twice running ((none named), r20261114-120002-ab12) under this declaration, and is "
                             "launched again once its rows in benchmarks/compute/metrics.toml change")
    (cfg.tree / "benchmarks/compute/metrics.toml").write_text(toml([row("compute-accounting/share", CPU, every="2h")]))
    run(cfg, pod, state, now=NOW + 7 * 7200)
    assert len(pod.launched) == 3


def test_a_run_its_pool_never_started_is_no_failure_and_counts_as_no_launch(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod()
    state, _ = run(cfg, pod)
    for n in range(1, 4):
        pod.states[f"r20261114-1200{n:02d}-ab12"] = ("failed", 75)
        run(cfg, pod, state, now=NOW + n * 900)
    assert len(pod.launched) == 4 and [e["state"] for e in state["launches"]] == ["unrun"] * 3 + ["submitted"]


def test_the_stop_file_stops_every_launch(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    cfg.stop.write_text("")
    pod = Pod()
    state, saved = run(cfg, pod)
    assert pod.launched == [] and saved == [] and state == {}
    assert pod.lines == [f"stopped: {cfg.stop} exists, so nothing is launched; remove it to go on"]


def test_a_dry_run_says_what_it_would_launch_and_why_and_launches_nothing(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/compute": [row("compute-accounting/share", CPU)]})
    pod = Pod(ages={"compute-accounting/share": 9 * DAY_MS})
    state, saved = run(cfg, pod, dry_run=True)
    assert pod.launched == [] and saved == [] and state["launches"] == []
    assert pod.lines[-1].startswith("would launch compute-accounting/share on vy-nebius-cpu-12: compute-accounting/share's newest "
                                    "reading is 9.0 d old, past its every 7d: research run --on vy-nebius-cpu-12 --pool research")


def test_a_gpu_command_runs_only_on_a_machine_the_plan_lists_free_for_it_now(tmp_path):
    cfg = setup(tmp_path, {"benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)],
                           "benchmarks/gpu": [row("vllm/other", "research run --gpus 1 --on auto -- bash benchmarks/gpu/run.sh")]})
    pod = Pod()
    state, _ = run(cfg, pod, dry_run=True)
    assert pod.launched == [] and sum("needs a GPU and the plan" in ln for ln in pod.lines) == 2
    cfg.plan.write_text('[[free]]\nmachine = "vy-h200-1"\nfrom = 2026-11-14T13:00:00Z\nuntil = 2026-11-15T00:00:00Z\n')
    run(cfg, pod)
    assert pod.launched == [], "the plan's window starts later"
    cfg.plan.write_text('[[free]]\nmachine = "vy-h200-1"\nfrom = 2026-11-14T00:00:00Z\nuntil = 2026-11-15T00:00:00Z\n'
                        'metrics = ["vllm/serving-cost-glm53"]\n')
    state, _ = run(cfg, pod)
    assert pod.launched == [["research", "run", "--on", "vy-h200-1", "--project", "verity", "--source", ".", "--cwd", "source",
                             "--", "bash", "benchmarks/serve_tax/run.sh", "--model", "glm53"]]
    assert any(ln.startswith("skip vllm/other: vllm/other has no reading; but it needs a GPU and the plan") for ln in pod.lines)
    cfg.plan.write_text('[[free]]\nmachine = "vy-h200-1"\nfrom = 2026-11-14T00:00:00Z\nuntil = 2026-11-15T00:00:00Z\n')
    run(cfg, pod, state)
    assert len(pod.launched) == 1, "the machine still runs this loop's launch"


def test_never_auto_nor_a_small_disk_machine(tmp_path, capsys):
    args = ["--tree", "t", "--state", "s", "--stop", "x", "--plan", "p", "--pool", "research", "--on"]
    for bad in ("auto", "vy-nebius-cpu-4"):
        with pytest.raises(SystemExit):
            R.parse([*args, bad])
        assert f"{bad} is refused" in capsys.readouterr().err
    cfg = setup(tmp_path, {"benchmarks/x": [row("benchmarks/x", "research run --on vy-nebius-cpu-6 -- bash benchmarks/x/run.sh")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == [] and pod.lines[-1].endswith("but its row names vy-nebius-cpu-6, a small-disk machine")
    cfg.plan.write_text('[[free]]\nmachine = "vy-nebius-cpu-5"\nfrom = 2026-11-14T00:00:00Z\nuntil = 2026-11-15T00:00:00Z\n')
    cfg2 = setup(tmp_path / "g", {"benchmarks/serve_tax": [row("vllm/serving-cost-glm53", GPU)]})
    run(R.Settings(cfg2.tree, cfg2.state, cfg2.stop, cfg.plan, cfg2.on, cfg2.pool), pod)
    assert pod.launched == []


def test_a_command_that_takes_a_run_id_is_never_launched(tmp_path):
    cfg = setup(tmp_path, {"demos/vllm": [row("vllm/remote-demo-minutes",
                                              "research run --cwd source -- python3 demos/vllm/minutes.py FIRST_RUN [RUN ...] VERDICT_RUN")]})
    pod = Pod()
    run(cfg, pod)
    assert pod.launched == []
    assert pod.lines[-1].endswith("but its command takes FIRST_RUN, RUN, VERDICT_RUN, which only its owner fills in")


def test_the_checked_in_loop_parses_and_launches_nowhere_it_must_not():
    _, loops = W.load(CONTROL / "watchers.toml")
    (lp,) = [lp for lp in loops if lp.name == "metrics-rerun"]
    assert lp.argv[2] == "/data/watchers/src/infra/control/metrics_rerun.py"
    cfg, ns = R.parse(lp.argv[3:])
    assert not ns.dry_run and not ns.once and cfg.max >= 1
    assert cfg.stop.parent == cfg.state.parent == Path("/data/watchers")
