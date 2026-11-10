"""`verity bench training-slowdown CONFIG`: every job of an attack config and of its honest config, each found by its
label and run where it is missing.

    verity bench training-slowdown NAME|CONFIG.toml [--dry-run] [--local [--root DIR] | --on MACHINE [--pool NAME]]
        [--starts FILE] [--starts-root DIR] [--device DEVICE]

A config (`training_slowdown.config`; a bare NAME is `attacks/NAME.toml`) and its honest config expand into jobs, one
per seed and split, each labelled `training-slowdown=<digest>/<seed>/<split>` (`config.Job.label`). Each time it runs,
the command looks every job up and runs the missing ones:

- by default each goes on the queue (`research run --queue`, kind `training-attack` in infra/cluster/kinds/training.toml:
  one GPU, preemptible, and a requeued job starts again from θ), or with `--on MACHINE` to that machine. The command
  labels a job's run as it queues it, and the run's record once the run ends with one (`research data label`), so the
  store's labels are its only state and the next run of the command picks up where this one stopped;
- with `--local` each runs here, one after another, with no `research` call, into `ROOT/<digest>/<seed>/<split>/`
  (`run.json`, and `job.json` with the job's exit status); ROOT is `--root`, else `$OUT`.

`--dry-run` prints what it would run and runs nothing. The command exits 0 when every job has its record, 3 when a job
is still pending (its run id says which), and 1 when a job failed: a failed job is shown, and never run again.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from training_slowdown import config, starts  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

KIND = "training-attack"
#: the asserter of the command's labels
BY = "training-slowdown"
RUN_RE = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
#: a queued job's environment: the workspace with the members' CUDA torch (the suites' `--extra torch-cpu` elsewhere)
SYNC = "uv sync -q --all-packages --extra torch"
DONE, PENDING, FAILED, MISSING = "done", "pending", "failed", "missing"


@dataclass
class State:
    state: str
    where: str = ""
    note: str = ""


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
    return out


def research_argv(job: config.Job, a: argparse.Namespace) -> list[str]:
    try:
        rel = job.config.path.resolve().relative_to(HERE)
    except ValueError:
        raise SystemExit(f"training-slowdown: a queued job reads its config from the tree it ships, and "
                         f"{job.config.path} is outside {HERE}") from None
    py = " ".join(shlex.quote(x) for x in job_args(job, str(rel), a))
    cmd = f'{SYNC} && cd {HERE.relative_to(ROOT)} && ../../.venv/bin/python {py} --out "$OUT"'
    where = (["--on", a.on] + (["--pool", a.pool] if a.pool else [])) if a.on else ["--queue", "--kind", KIND,
                                                                                    "--preemptible"]
    return ["research", "run", *where, "--project", "verity", "--source", str(ROOT), "--cwd", "source", "--gpus", "1",
            "--", "bash", "-lc", cmd]


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
    return State(FAILED, run, f"rc {rc}" + ("" if record else ", no record"))


def local_state(job: config.Job, root: Path) -> State:
    d = root / job.label
    if not (d / "job.json").is_file():
        return State(MISSING, str(d))
    rc = json.loads((d / "job.json").read_text())["rc"]
    return State(DONE, str(d)) if rc == 0 else State(FAILED, str(d), f"rc {rc}")


def run_local(job: config.Job, a: argparse.Namespace, root: Path) -> None:
    d = root / job.label
    d.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, *job_args(job, str(job.config.path), a), "--out", str(d)]
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
    ap.add_argument("--starts", type=Path, default=starts.REGISTRY, help="the starts registry (default starts.toml)")
    ap.add_argument("--starts-root", type=Path, help="where the starts are installed (default the registry's)")
    ap.add_argument("--device", help="the jobs' device (default: theirs, cuda where there is one)")
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
        if s.state != MISSING:
            continue
        if a.dry_run:
            argv = ([sys.executable, *job_args(j, str(j.config.path), a), "--out", str(root / j.label)] if a.local
                    else research_argv(j, a))
            print("would run: " + " ".join(shlex.quote(x) for x in argv))
        elif a.local:
            run_local(j, a, root)
            states[i] = local_state(j, root)
        else:
            run = store.launch(research_argv(j, a))
            store.label(run, j.label)
            states[i] = State(PENDING, run)
    for c in configs:
        print(f"{'honest ' if c is not cfg else ''}{c.name}: digest {c.digest[:16]}, start {c.start.name}, route "
              f"{c.route}, {c.budget['steps']} steps of {c.budget['batch']} windows, gain {c.target['gain']}, "
              f"policy {c.policy}")
        for j, s in zip(jobs, states):
            if j.config is c:
                print(f"  {j.split:6} {j.seed:<5} {s.state:8} {s.where}" + (f" ({s.note})" if s.note else ""))
    if not a.local:
        print(f"queued as {KIND} (infra/cluster/kinds/training.toml): one GPU, preemptible, a requeued job starts again "
              f"from θ" if not a.on else f"run on {a.on}")
    n = {k: sum(s.state == k for s in states) for k in (DONE, PENDING, FAILED, MISSING)}
    if n[FAILED]:
        print(f"{n[FAILED]} of {len(jobs)} jobs failed: a failed job is never run again")
        return 1
    if n[PENDING] or n[MISSING]:
        print(f"{n[PENDING] + n[MISSING]} of {len(jobs)} jobs pending")
        return 3
    print(f"every job has its record ({len(jobs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
