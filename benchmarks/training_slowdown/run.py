"""`verity bench training-slowdown CONFIG`: every job of an attack config and of its honest config, each found by its
label and run where it is missing.

    verity bench training-slowdown NAME|CONFIG.toml [--dry-run] [--local [--root DIR] | --on MACHINE [--pool NAME]]
        [--weights] [--starts FILE] [--starts-root DIR] [--device DEVICE]

A config (`training_slowdown.config`; a bare NAME is `attacks/NAME.toml`) and its honest config expand into jobs, one
per seed and split, each labelled `training-slowdown=<digest>/<seed>/<split>` (`config.Job.label`). Each time it runs,
the command looks every job up and runs the missing ones:

- by default each goes on the queue (`research run --queue`, kind `training-attack` in infra/cluster/kinds/training.toml:
  one GPU, preemptible), or with `--on MACHINE` to that machine, for the time its config's budget gives it
  (`run_minutes`). On the queue it asks for the disk its start's size needs (`run_disk_gb`: the copies of a kept
  state a run holds at once, at the state's largest); a machine named with `--on` is asked for none. The command
  labels a job's run as it queues it, and the run's record once the run ends with one (`research data label`), so
  the store's labels are its only state and the next run of the command picks up where this one stopped;
- with `--local` each runs here, one after another, with no `research` call, into `ROOT/<digest>/<seed>/<split>/`
  (`run.json`, and `job.json` with the job's exit status); ROOT is `--root`, else `$OUT`.

A job keeps what continuing needs after each checkpoint (`training_slowdown.run --state`), and one told to stop,
evicted or out of its time, writes a record that says so (`training_slowdown.TOLD`). The command reads a stop there
and never in the run's exit status, which is whatever stopped it reports: a lease out of time ends the job through
`timeout`, whose status is 124 whatever the job's own. So a run that ended with any status but 0 is stopped when
its record says it was told to stop, and failed when it holds no record or one that doesn't say so. A job told to
stop before its own process is up, while its environment is made or its state fetched, is stopped too: its shell
writes the record of the stop, the job's own (`stop_record`), where the job wrote none (`ENV`). A crash with no
stop stays a failure. Nothing runs a stopped job again by itself: the command's
next run finds it stopped and starts it again, from the latest state a run of it kept (a run stopped before it
was up kept none, and loses no earlier one), or from θ where none kept one. A job whose last `STALLS` runs were
each stopped and kept no newer state than the one before them is stalled: it is stopped every time before its
next checkpoint, and only `--continue-stalled` continues it, for `STALLS` more such stops. A job that continues
is given the CPU thread count its kept record states. Where the state is between the two runs is
`NODE_STATE`: by default with the stopped run's output, on the store, from where the next run fetches it on
whichever machine it gets. The command tells that run which state it continues from (`--continues`: the sha256 the
store lists for the `state.json` in the stopped run's record, or `nothing`), and the job refuses any other it finds
in its state directory; a state on a node's disk has no copy in a record, and the job is told nothing of it. The
state is fetched beside `$OUT` and moved into `$OUT/state` once whole, so a run stopped while it fetched keeps no
part of one. A
state is a cache: one that is lost costs the job a start from θ, and no result is read from it.

A queued job writes nothing into the tree it runs from, which every run of its commit shares. Its Python
environment is the run's own: it makes a directory under /tmp, has `uv sync` build the environment there
(`UV_PROJECT_ENVIRONMENT`) from the tree's lock as it is, and its shell removes the directory when the job ends,
stopped or not (`ENV`, `SYNC`).

A job's output holds no weights: its record, and its last checkpoint's E (the bytes that rebuild its output model
from θ) when E is under 50 MiB. `--weights` makes each job this call runs keep its last checkpoint's whole state too,
for a promoted config; a job that already has its record isn't run again for them.

`--dry-run` prints what it would run and runs nothing. The command exits 0 when every job has its record, 3 when a job
is still pending (its run id says which) or stopped, 4 when a job stalled, and 1 when a job failed: a failed job is
shown, and never run again.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from training_slowdown import BUDGET_SLACK, NOTHING, config, starts, stop_record, told  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

KIND = "training-attack"
#: the asserter of the command's labels
BY = "training-slowdown"
RUN_RE = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
#: where a queued job makes its directory, for its Python environment
TMP = "/tmp"
#: a queued job's Python environment is the run's own, in a directory it makes under `TMP`, never the tree's `.venv`,
#: which every run of the commit would share. Per run and not per commit: a node's /tmp is every run's and nothing
#: empties it, so an environment a commit's runs shared would stay there after the last of them, and each run would
#: execute what an earlier one left in it. The job's shell removes the directory when it ends (`trap ... EXIT`), and
#: `$fetching`, where a kept state is fetched before it moves into `$OUT/state` whole. It outlives a stop (`trap ...
#: TERM`): what stops a run signals every process of it, so the job's own process is told beside its shell, and bash
#: runs the trap once the command it waits on has ended, the job included, which keeps the run's end the job's. The
#: trap then writes the job's record of the stop (`stop_record`, in `$stopped`) where the job wrote no record (one
#: stopped before its own process was up), and ends the script with the status of the command the stop ended, so
#: nothing after it starts. A script whose shell wrote that record never ends with 0, which a stop between two
#: commands, or during one that ends 0 when told to stop, would give it: it ends with 143 (SIGTERM's) instead, so
#: status 0 with a record is still the job's own record of its end (`store_state`)
ENV = ('tmp=$(mktemp -d {tmp}/training-slowdown.XXXXXX) && fetching="$(dirname "$OUT")/fetching" && '
       'trap \'rm -rf "$tmp" "$fetching"\' EXIT && stopped={record} && '
       'trap \'status=$?; [ -e "$OUT/run.json" ] || {{ mkdir -p "$OUT" && printf "%s\\n" "$stopped" > "$OUT/run.json"; '
       '[ "$status" -ne 0 ] || status=143; }}; exit $status\' TERM && export UV_PROJECT_ENVIRONMENT="$tmp/venv"')
#: the environment's programs
PYTHON, RESEARCH = '"$UV_PROJECT_ENVIRONMENT/bin/python"', '"$UV_PROJECT_ENVIRONMENT/bin/research"'
#: what goes into the environment: the workspace with the members' CUDA torch (the suites' `--extra torch-cpu`
#: elsewhere), from the lock as the tree has it (`--frozen`), so the lock isn't written either
SYNC = "uv sync -q --frozen --all-packages --extra torch"
#: where a job's state is between two of its runs. None: under the run's $OUT, so a stopped run's state is in its
#: record on the store, and the job's next run fetches it into its own $OUT on whichever machine it gets. A directory
#: of the nodes' own disks: the state stays there under the job's label, never leaves the node, and the job's next run
#: goes to the stopped run's machine.
NODE_STATE: str | None = None
#: the minutes of a run that its config's `budget.minutes` doesn't count: the environment, the start's and the data's
#: load, a kept state's fetch, the last checkpoint and the trace
OUTSIDE_MIN = 30
#: a kept state is at most this many times its model: the weights and, for each, two buffers of its optimizer
STATE_OVER_MODEL = 3
#: the copies of a kept state a run's disk holds at once: the one the route's process hands out, and the new state
#: beside the old one until it takes its place (`training_slowdown.run.Kept.write`)
STATE_COPIES = 3
#: the GB a run writes besides: its environment, its tree, a checkpoint on its way to be scored and its output. The
#: queue's own figure for a job that declares none. The queue holds a run for room on the filesystem of its runs, so
#: where `TMP` is on another one the environment is in the figure and not in what the hold measures.
OUTSIDE_GB = 20
DONE, PENDING, STOPPED, STALLED, FAILED, MISSING = "done", "pending", "stopped", "stalled", "failed", "missing"
#: a job whose last this many runs were each stopped and kept no newer state than the one before them is stalled, and
#: `--continue-stalled` continues it for this many more
STALLS = 3
#: where a run's record holds the state its job kept
STATE_JSON = "out/state/state.json"


@dataclass
class State:
    state: str
    where: str = ""
    note: str = ""
    #: of a stopped job: the record of its latest run that holds a state, the store's sha256 of the state.json in it,
    #: the CPU thread count that state's record states, and its latest run's machine
    kept: str = ""
    stated: str = ""
    machine: str = ""
    threads: int | None = None


def find(arg: str) -> Path:
    for p in (Path(arg), HERE / arg, HERE / "attacks" / f"{arg}.toml"):
        if p.is_file():
            return p.resolve()
    raise SystemExit(f"training-slowdown: no config {arg!r} (a file, or a name in {HERE / 'attacks'})")


def job_args(job: config.Job, config_path: str, a: argparse.Namespace) -> list[str]:
    out = ["-m", "training_slowdown.run", "--config", config_path, "--seed", str(job.seed), "--split", job.split]
    if a.starts != starts.REGISTRY.resolve():
        out += ["--starts", str(a.starts)]
    if a.starts_root:
        out += ["--starts-root", str(a.starts_root)]
    if a.device:
        out += ["--device", a.device]
    if a.weights:
        out += ["--keep-weights"]
    return out


def run_minutes(cfg: config.Config) -> int | None:
    """The longest a job of `cfg` runs, in minutes, from its budget: the minutes past which the job stops itself
    (`BUDGET_SLACK` times `budget.minutes`, counted over every run it continues in) and `OUTSIDE_MIN`. None where
    the budget states no minutes: the kind's time, or the machine's. Nothing caps it at the kind's: a run that asks
    for more than its kind's `max_wall_min` is warned about by the queue and never refused."""
    minutes = cfg.budget.get("minutes")
    return math.ceil(BUDGET_SLACK * minutes) + OUTSIDE_MIN if minutes else None


def run_disk_gb(start: starts.Start, weights: bool = False, fetched: bool = False) -> int:
    """The disk a run of a job at `start` asks the queue for, in its GB (10^9 bytes), from the start's size: its
    parameters at 4 bytes each are its model, a kept state is at most `STATE_OVER_MODEL` models, and the run holds
    `STATE_COPIES` of them at once, and one more when it `fetched` the state it continues from, of which the store
    keeps its own copy on the machine. With `weights` the run keeps the last checkpoint's whole output, one more
    model. `OUTSIDE_GB` is the rest."""
    model = 4 * int(start.shape["parameters"])
    return math.ceil(((STATE_COPIES + fetched) * STATE_OVER_MODEL + weights) * model / 1e9) + OUTSIDE_GB


def research_argv(job: config.Job, a: argparse.Namespace, after: State | None = None) -> list[str]:
    """The `research run` of `job`; `after` its stopped run, it continues from the state that run kept."""
    try:
        rel = job.config.path.resolve().relative_to(HERE)
    except ValueError:
        raise SystemExit(f"training-slowdown: a queued job reads its config from the tree it ships, and "
                         f"{job.config.path} is outside {HERE}") from None
    threads = ["--threads", str(after.threads)] if after and after.threads is not None else []
    py = " ".join(shlex.quote(x) for x in job_args(job, str(rel), a) + threads)
    # the stopped run's out/state, into this run's $OUT (the run directory's out/) once whole
    fetch = (f'{RESEARCH} data fetch {after.kept} --path "out/state/*" --to "$fetching" && mkdir -p "$OUT" && '
             f'mv "$fetching/out/state" "$OUT/state" && ' if after and after.kept else "")
    if NODE_STATE:
        # no record holds a copy of a state on a node's disk: the job continues from the one it finds there
        state = shlex.quote(f"{NODE_STATE}/{job.label}")
    else:
        # the job is told which state.json the stopped run's record holds, and continues from no other
        state = f'"$OUT/state" --continues {after.stated if fetch else NOTHING}'
    # no exec: the shell stays, to remove the environment once the job has ended (`ENV`)
    record = shlex.quote(json.dumps(stop_record(job.seed, job.split)))
    cmd = (f'{ENV.format(tmp=TMP, record=record)} && {SYNC} && {fetch}cd {HERE.relative_to(ROOT)} && {PYTHON} {py} '
           f'--state {state} --out "$OUT"')
    minutes = run_minutes(job.config)
    if a.on:
        # a machine named by hand declares nothing to the queue, its disk included
        where = ["--on", a.on] + (["--pool", a.pool] if a.pool else [])
        asks = ["--timeout", f"{minutes}m"] if minutes else []
    else:
        where = ["--queue", "--kind", KIND, "--preemptible"]
        asks = (["--disk-gb", str(run_disk_gb(job.config.start, a.weights, bool(fetch)))]
                + (["--max-min", str(minutes)] if minutes else [])
                + (["--on", after.machine] if after and NODE_STATE else []))
    return ["research", "run", *where, "--project", "verity", "--source", str(ROOT), "--cwd", "source", "--gpus", "1",
            *asks, *(["--store-read"] if fetch else []), "--", "bash", "-lc", cmd]


def reads(path: Path, of: str) -> object:
    """The run.json (or state.json) at `path`, which `of` holds, as its job wrote it. One that doesn't read is no
    record (None), which leaves its job failed, and the command says what it found there."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"training-slowdown: {of} holds no {path.name} that reads ({type(e).__name__}: {e}): read as no record",
              file=sys.stderr)
        return None


class Store:
    """The evidence store, through `research`: the jobs' labels, their runs' attempts, and the runs it starts."""

    def __init__(self, execute=None):
        self.execute = execute or (lambda argv: subprocess.run(argv, cwd=ROOT, capture_output=True, text=True))

    def research(self, *args: str, check: bool = True):
        p = self.execute(["research", *args])
        if check and p.returncode:
            raise SystemExit(f"training-slowdown: `research {' '.join(args[:3])}` failed ({p.returncode}): "
                             f"{(p.stderr or p.stdout or '').strip()[-400:]}")
        return p

    def labels(self) -> dict[str, list[str]]:
        """Every target of a `training-slowdown` label, by its value; the remote's pulled first (unreachable: this
        store's alone)."""
        if p := self.research("data", "labels-sync", "--pull-only", check=False).returncode:
            print(f"training-slowdown: research data labels-sync failed ({p}): reading this machine's labels alone",
                  file=sys.stderr)
        rows = json.loads(self.research("data", "sql", f"SELECT target, value FROM labels WHERE key = "
                                                       f"'{config.LABEL_KEY}'", "--json").stdout or "[]")
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["value"], []).append(r["target"])
        return out

    def attempt(self, run: str) -> dict | None:
        p = self.research("data", "show", run, "--json", check=False)
        return json.loads(p.stdout) if p.returncode == 0 else None

    def says(self, record: str, path: str = "out/run.json") -> object:
        """The run.json (or the file at `path`) a run's record holds, as its job wrote it; None where it holds none,
        or one that doesn't read (`reads`)."""
        with tempfile.TemporaryDirectory(prefix="training-slowdown-") as d:
            if self.research("data", "fetch", record, "--path", path, "--to", d, check=False).returncode:
                return None
            return reads(Path(d) / path, record)

    def keeps(self, record: str) -> str:
        """The sha256 of the state.json a run's record holds, as the store lists it: '' where it holds no state."""
        p = self.research("data", "fetch", record, "--list", "--json", "--path", "out/state/state.json", check=False)
        rows = json.loads(p.stdout or "[]") if p.returncode == 0 else []
        sha = str(rows[0].get("sha256")) if rows else ""
        return sha if re.fullmatch(r"[0-9a-f]{64}", sha) else ""

    def label(self, target: str, value: str) -> None:
        p = self.research("data", "label", target, config.LABEL_KEY, value, "--by", BY, check=False)
        if p.returncode:
            print(f"training-slowdown: the label on {target} isn't on the store's remote yet "
                  f"({(p.stderr or '').strip()[-200:]})", file=sys.stderr)

    def launch(self, argv: list[str]) -> str:
        p = self.execute(argv)
        text = (p.stdout or "") + (p.stderr or "")
        if p.returncode or not (m := RUN_RE.search(text)):
            raise SystemExit(f"training-slowdown: `{' '.join(argv[:4])} ...` started no run ({p.returncode}): "
                             f"{text.strip()[-400:]}")
        return m.group(0)


def store_state(job: config.Job, labels: dict[str, list[str]], store: Store, write: bool) -> State:
    """Where `job` stands by its latest run. One that ended with status 0 and a record is done. Any other is stopped
    when the run.json in its record says the job was told to stop (`told`), whatever its status, and failed when
    its record holds none or one that doesn't say so; a stopped job goes on as `continued` says."""
    targets = labels.get(job.label, [])
    runs = sorted(t for t in targets if RUN_RE.fullmatch(t))
    if not runs:
        return State(MISSING)
    run = runs[-1]
    a = store.attempt(run)
    if a is None:
        return State(PENDING, run)
    rc, record = (a.get("status") or {}).get("rc"), (a.get("outputs") or {}).get("run_record")
    if rc == 0 and record:
        if write and record not in targets:
            store.label(record, job.label)
        return State(DONE, record, run)
    said = store.says(record) if record else None
    if told(said):
        machine = (a.get("execution") or {}).get("machine") or ""
        if NODE_STATE:
            return State(STOPPED, run, f"its state is on {machine}, under {NODE_STATE}", machine=machine)
        return continued(runs, record, store, machine)
    return State(FAILED, run, f"rc {rc}, " + ("no record" if said is None else "a record that doesn't say the job was "
                                              "told to stop"))


def continued(runs: list[str], record: str, store: Store, machine: str) -> State:
    """Where a job whose latest run (`runs[-1]`, its record `record`) was told to stop goes on: from the state of its
    latest run that kept one, since a run stopped before its job was up kept none and loses no earlier one, with the
    CPU thread count that state's record states. Every run of a label but its latest was a stopped one: a label is
    run again only while it is stopped. A job whose runs since its last new state number a multiple of `STALLS` is
    stalled: each was stopped before its next checkpoint, and only `--continue-stalled` continues it, for `STALLS`
    more such stops. A stall is no failure: it chooses nothing, and the job it continues ends with the unbroken
    job's record; the bound keeps the command from continuing by itself a job that never reaches a checkpoint."""
    records = [((store.attempt(r) or {}).get("outputs") or {}).get("run_record") or "" for r in runs[:-1]] + [record]
    shas = [store.keeps(r) if r else "" for r in records]
    newest, stalls = "", 0
    for sha in shas:
        stalls = 0 if sha and sha != newest else stalls + 1
        newest = sha or newest
    i = max((i for i, sha in enumerate(shas) if sha), default=None)
    if i is None:
        how = State(STOPPED, runs[-1], "it kept no state: it starts again from θ", machine=machine)
    else:
        kept = store.says(records[i], STATE_JSON)
        rec = kept.get("record") if isinstance(kept, dict) else None
        threads = rec.get("threads") if isinstance(rec, dict) else None
        how = State(STOPPED, runs[-1], "its record holds its state" if i == len(runs) - 1 else
                    f"it kept no state: it goes on from the state {runs[i]} kept", kept=records[i], stated=shas[i],
                    machine=machine, threads=threads if isinstance(threads, int) else None)
    if stalls and stalls % STALLS == 0:
        how.state, how.note = STALLED, (
            f"its last {STALLS} runs, {', '.join(runs[-STALLS:])}, were each stopped and kept no newer state than the "
            f"one before them: it is stopped every time before its next checkpoint, and `--continue-stalled` "
            f"continues it for {STALLS} more such stops")
    return how


def local_state(job: config.Job, root: Path) -> State:
    """Where `job` stands by its directory under `root`, read as `store_state` reads a run: its exit status says only
    that it is done, and its run.json whether it was told to stop."""
    d = root / job.label
    if not (d / "job.json").is_file():
        return State(MISSING, str(d))
    rc = json.loads((d / "job.json").read_text())["rc"]
    if rc == 0:
        return State(DONE, str(d))
    said = reads(d / "run.json", str(d)) if (d / "run.json").is_file() else None
    if not told(said):
        return State(FAILED, str(d), f"rc {rc}")
    kept = (d / "state" / "state.json").is_file()
    return State(STOPPED, str(d), "its state is in state/" if kept else "it kept no state: it starts again from θ")


def local_argv(job: config.Job, a: argparse.Namespace, root: Path) -> list[str]:
    d = root / job.label
    return [sys.executable, *job_args(job, str(job.config.path), a), "--state", str(d / "state"), "--out", str(d)]


def run_local(job: config.Job, a: argparse.Namespace, root: Path) -> None:
    d = root / job.label
    d.mkdir(parents=True, exist_ok=True)
    # a stopped run's record: this run writes its own, and a run that fails is not read as that stop
    (d / "run.json").unlink(missing_ok=True)
    argv = local_argv(job, a, root)
    rc = subprocess.run(argv, cwd=HERE).returncode
    (d / "job.json").write_text(json.dumps({"rc": rc, "argv": argv}) + "\n")


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="an attack config: a file, or a name in attacks/")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--local", action="store_true", help="run the missing jobs here, one after another")
    where.add_argument("--on", metavar="MACHINE", help="run the missing jobs on MACHINE (research run --on)")
    ap.add_argument("--root", type=Path, help="--local's directory of jobs (default $OUT)")
    ap.add_argument("--pool", help="with --on, the machine's pool")
    ap.add_argument("--weights", action="store_true",
                    help="each job this call runs keeps its last checkpoint's whole state (default: no weights)")
    ap.add_argument("--starts", type=Path, default=starts.REGISTRY, help="the starts registry (default starts.toml)")
    ap.add_argument("--starts-root", type=Path, help="where the starts are installed (default the registry's)")
    ap.add_argument("--device", help="the jobs' device (default: theirs, cuda where there is one)")
    ap.add_argument("--continue-stalled", action="store_true",
                    help=f"continue each stalled job too, for {STALLS} more stops before its next checkpoint (a "
                         "failed job is never run again)")
    a = ap.parse_args(argv)
    a.starts = a.starts.resolve()
    a.starts_root = a.starts_root.resolve() if a.starts_root else None
    return a


def main(argv=None, store: Store | None = None) -> int:
    a = parse(argv)
    reg = starts.registry(a.starts)
    cfg = config.load(find(a.config), reg)
    configs = [cfg] + [h for h in [config.honest(cfg, reg)] if h]
    jobs = [j for c in configs for j in c.jobs()]
    if a.local:
        root = a.root or (Path(SHARED.OUT) if SHARED.OUT else None)
        if root is None:
            raise SystemExit("training-slowdown: --local writes its jobs under --root DIR (or $OUT)")
        states = [local_state(j, root) for j in jobs]
    else:
        store = store or Store()
        labels = store.labels()
        states = [store_state(j, labels, store, not a.dry_run) for j in jobs]
    for i, (j, s) in enumerate(zip(jobs, states)):
        if not (s.state in (MISSING, STOPPED) or s.state == STALLED and a.continue_stalled):
            continue
        after = s if s.state in (STOPPED, STALLED) else None
        if a.dry_run:
            argv = local_argv(j, a, root) if a.local else research_argv(j, a, after)
            print("would run: " + " ".join(shlex.quote(x) for x in argv))
        elif a.local:
            run_local(j, a, root)
            states[i] = local_state(j, root)
        else:
            run = store.launch(research_argv(j, a, after))
            store.label(run, j.label)
            states[i] = State(PENDING, run, f"continues {s.where}" if after else "")
    for c in configs:
        print(f"{'honest ' if c is not cfg else ''}{c.name}: digest {c.digest[:16]}, start {c.start.name}, route "
              f"{c.route}, {c.budget['steps']} steps of {c.budget['batch']} {c.windows and c.windows + ' '}windows, "
              f"gain {c.target['gain']}, "
              f"policy {c.policy}")
        for j, s in zip(jobs, states):
            if j.config is c:
                print(f"  {j.split:6} {j.seed:<5} {s.state:8} {s.where}" + (f" ({s.note})" if s.note else ""))
    if not a.local:
        print(f"queued as {KIND} (infra/cluster/kinds/training.toml): one GPU, preemptible; a stopped job continues "
              f"from its kept state when this command runs again" if not a.on else f"run on {a.on}")
    n = {k: sum(s.state == k for s in states) for k in (DONE, PENDING, STOPPED, STALLED, FAILED, MISSING)}
    if n[FAILED]:
        print(f"{n[FAILED]} of {len(jobs)} jobs failed: a failed job is never run again")
        return 1
    if n[STALLED]:
        print(f"{n[STALLED]} of {len(jobs)} jobs stalled: run the command with --continue-stalled to continue them")
        return 4
    if n[PENDING] or n[MISSING] or n[STOPPED]:
        print(f"{n[PENDING] + n[MISSING] + n[STOPPED]} of {len(jobs)} jobs pending" +
              (f", {n[STOPPED]} of them stopped: run the command again to continue them" if n[STOPPED] else ""))
        return 3
    print(f"every job has its record ({len(jobs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
