"""Each declared number measured again on its schedule, which the control pod does by itself: each pass reads every
`metrics.toml` of a clean checkout of `main` (benchmarks/README.md's "Metrics"), each number's newest reading from the site's
GET /api/metrics (`research.metrics`), and launches the command of each number older than its `every` as a `research run` on a
machine. The pod runs no workload of its own.

A pass:

    refresh   fetch origin's `main` into `--tree` (the loop's own clone, made on first use) and check its tip out, detached and
              clean: the declarations are read there, and every command is launched from there, `--source .` shipping it.
    due       a number is due when the door holds no reading of it or its newest is older than its `every`. The rows of one
              file that share a command are one launch, which measures each of them.
    bounds    a due command is launched only when each of these holds, and the pass says which one stopped it:
              - the stop file (`--stop`) is absent: it stops every launch;
              - it is no loop's: a row whose command's script is under infra/, or that has no command, is a loop's number,
                which the loop sends itself;
              - it names no placeholder but its machine's (a run id such as FIRE_RUN, which only its owner picks);
              - none of its numbers has a launch of this loop's still running, or one that ended less than that number's
                `every` ago, whatever the door says (the record, `--state`);
              - it has not failed twice running under the declaration it has now: a change to its rows launches it again;
              - it has a machine. A GPU command, one that asks for GPUs (GPU_FLAGS) or leaves its machine to its owner
                (`--on NODE`), runs only on a machine the plan (`--plan`) lists as free for it now and that none of this
                loop's launches still runs on. Any other runs on the machine and pool its row names, else `--on` and
                `--pool`. Never `auto`, never a small-disk machine (SMALL_DISK);
              - this pass has launched fewer than `--max`.
    launch    in the tree, `research run --on MACHINE [--pool POOL] --project verity --source .`, then the row's own options
              but its `--on`, then its workload. The record holds the launch before it starts (its numbers, command,
              declaration, machine and time) and its run id once the launch names one; a launch that names none has failed.
              Each pass reads how the record's running launches stand (`research.runs`) before it judges. A run its pool
              never started (exit 75) is no failure and counts as no launch.

The plan, written by the lead for the week (TOML; a missing file lists no machine):

    [[free]]
    machine = "vy-nebius-h200-1"
    from = 2026-11-16T00:00:00Z
    until = 2026-11-18T00:00:00Z
    metrics = ["vllm/serving-cost-glm53"]      # optional: free for these alone; without it, for any GPU command

usage (the watchers' `metrics-rerun` loop on the control pod, infra/control/watchers.toml; research importable, the store's
key, and a site key with events:read in $JOBS_TOKEN):
  python3 infra/control/metrics_rerun.py --tree DIR --state FILE --stop FILE --plan FILE --on MACHINE --pool POOL [--max 2]
      [--every 900] [--once] [--dry-run]
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from research.jobs import dispatch

SCHEMA = "verity/metrics-rerun/v1"
STAMP = "%Y-%m-%dT%H:%M:%SZ"
NAME = "metrics.toml"
TREES = ("benchmarks", "demos", "infra", "tools")      # where a metrics.toml may sit (tests/test_metrics_registry.py)
SKIP = {".git", ".lake", ".venv", "node_modules", "__pycache__", "target", "build", "dist"}
LOOPS = "infra/"                                        # a row whose command's script is here is a loop's, which sends it itself
SCRIPT = re.compile(rf"(?:^|\s)((?:{'|'.join(TREES)})/\S+\.(?:sh|py))(?:\s|$)")
EVERY = re.compile(r"^([1-9][0-9]*)([mhd])$")
UNIT_S = {"m": 60, "h": 3600, "d": 86400}
#: the machines whose disks are too small for a measurement's tree and outputs
SMALL_DISK = frozenset({"vy-nebius-cpu-2", "vy-nebius-cpu-4", "vy-nebius-cpu-5", "vy-nebius-cpu-6"})
GPU_FLAGS = ("--gpus", "--gpu-model", "--gpu-ids", "--gpu-host")
OWNERS = "NODE"                                         # `--on NODE`: a machine its owner picks
PLACEHOLDER = re.compile(r"^\[?([A-Z][A-Z0-9_]+)\]?$")  # FIRE_RUN, [RUN: what the command's owner fills in
ENDED = ("done", "failed", "cancelled", "refused", "launcher-exited", "runner-gone")    # research.runs.ENDED
NOT_STARTED = 75                                        # a pool run that never took a slot: no failure of its command, and no launch
OVER = ("passed", "failed", "unrun")
LAUNCH_S = dispatch.LAUNCH_TIMEOUT_S                    # a launch that names no run by then has failed
KEEP = 400                                              # launches the record keeps
URL = "https://github.com/compute-verification/verity.git"
HELPER = "!/data/infra/gh_app_token.sh git"             # hands git the App's token when git asks: never in a config or an argument
GIT_S = 1800


@dataclass(frozen=True)
class Settings:
    tree: Path
    state: Path
    stop: Path
    plan: Path
    on: str
    pool: str
    max: int = 2
    url: str = URL
    helper: str = HELPER


@dataclass(frozen=True)
class Job:
    """The rows of one metrics.toml that share a command: one launch measures each of them."""
    where: str
    command: str | None
    rows: tuple[dict, ...]

    @property
    def ids(self) -> list[str]:
        return [str(r.get("id")) for r in self.rows]

    @property
    def declared(self) -> str:
        """The declaration a failure count holds under: a digest of the rows."""
        return hashlib.sha256(json.dumps(sorted(self.rows, key=lambda r: str(r.get("id"))), sort_keys=True,
                                         default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Decision:
    job: Job
    launch: bool
    why: str
    machine: str | None = None
    argv: tuple[str, ...] = ()


def iso(t: float) -> str:
    return time.strftime(STAMP, time.gmtime(t))


def at(stamp: str) -> float:
    return float(calendar.timegm(time.strptime(stamp, STAMP)))


def every_s(row: dict) -> int | None:
    m = EVERY.match(str(row.get("every", "")))
    return int(m.group(1)) * UNIT_S[m.group(2)] if m else None


def span(s: float) -> str:
    return f"{s / 86400:.1f} d" if s >= 2 * 86400 else f"{s / 3600:.1f} h" if s >= 7200 else f"{s / 60:.0f} min"


# ---- the declarations and their commands ----------------------------------------------------------------------------------------

def declarations(tree: Path) -> tuple[list[Job], list[str]]:
    """The jobs of every metrics.toml under TREES, and a line for each file that isn't TOML."""
    jobs, broken = [], []
    for top in TREES:
        for d, dirs, files in os.walk(tree / top):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            if NAME not in files:
                continue
            f = (Path(d) / NAME).relative_to(tree).as_posix()
            try:
                rows = tomllib.loads((tree / f).read_text(encoding="utf-8")).get("metric", [])
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
                broken.append(f"{f}: not TOML ({e}), so none of its numbers is launched")
                continue
            by: dict[str | None, list[dict]] = {}
            for r in rows:
                by.setdefault(r.get("command") or None, []).append(r)
            jobs += [Job(f, c, tuple(rs)) for c, rs in by.items()]
    return sorted(jobs, key=lambda j: (j.where, j.ids)), broken


def split(command: str) -> tuple[list[str], list[str], bool]:
    """A command as (research run's options, the workload, whether it already is a `research run`)."""
    tokens = shlex.split(command)
    if tokens[:2] == ["research", "run"] and "--" in tokens:
        i = tokens.index("--")
        return tokens[2:i], tokens[i + 1:], True
    return [], tokens, False


def option(opts: list[str], name: str) -> str | None:
    for i, o in enumerate(opts):
        if o == name and i + 1 < len(opts):
            return opts[i + 1]
        if o.startswith(name + "="):
            return o.split("=", 1)[1]
    return None


def without(opts: list[str], name: str) -> list[str]:
    out, skip = [], False
    for o in opts:
        if skip:
            skip = False
        elif o == name:
            skip = True
        elif not o.startswith(name + "="):
            out.append(o)
    return out


def placeholders(opts: list[str], work: list[str]) -> list[str]:
    """What the command leaves for its owner to fill in, but its machine (`--on NODE`, which the loop chooses)."""
    return [m.group(1) for t in without(opts, "--on") + work if (m := PLACEHOLDER.match(t))]


def needs_gpu(opts: list[str]) -> bool:
    return any(o == f or o.startswith(f + "=") for o in opts for f in GPU_FLAGS) or option(opts, "--on") == OWNERS


def argv_for(job: Job, machine: str, pool: str | None) -> tuple[str, ...]:
    opts, work, wrapped = split(job.command or "")
    rest = without(opts, "--on")
    head = ["research", "run", "--on", machine]
    if pool and option(rest, "--pool") is None:
        head += ["--pool", pool]
    if option(rest, "--project") is None:
        head += ["--project", "verity"]
    if option(rest, "--source") is None:
        head += ["--source", "."]
    if not wrapped:
        head += ["--cwd", "source"]
    return (*head, *rest, "--", *work)


# ---- the record ---------------------------------------------------------------------------------------------------------------

def failed(e: dict) -> bool:
    return e.get("state") == "failed"


def last_of(record: list[dict], metric: str) -> dict | None:
    """The newest launch of `metric` that ran or still may: one its pool never started counts as none."""
    return next((e for e in reversed(record) if metric in e.get("metrics", []) and e.get("state") != "unrun"), None)


def refresh(record: list[dict], status: Callable[[str], tuple[str, int | None]], now: float, say: Callable[[str], None]) -> None:
    """How each launch the record holds as under way stands now: ended (passed or failed), or still going."""
    for e in record:
        if e.get("state") in OVER:
            continue
        if not e.get("run"):
            if now - at(e["at"]) >= LAUNCH_S:
                e.update(state="failed", ended=iso(now), why="the launch named no run")
            continue
        try:
            state, rc = status(e["run"])
        except Exception as x:  # noqa: BLE001  (a run whose machine can't be read counts as running; read again next pass)
            say(f"run {e['run']} ({', '.join(e['metrics'])}): not read ({type(x).__name__}: {x}); it counts as running")
            continue
        if state in ENDED:
            ok = "passed" if state == "done" and rc == 0 else "unrun" if rc == NOT_STARTED else "failed"
            e.update(state=ok, ended=iso(now), rc=rc, how=state)
            say(f"run {e['run']} ({', '.join(e['metrics'])}): {ok} ({state}, rc {rc})")
        else:
            e["state"] = state


# ---- the judgment -------------------------------------------------------------------------------------------------------------

def plan_rows(path: Path, say: Callable[[str], None]) -> list[dict]:
    """The plan's [[free]] rows, each with its `from` and `until` as epoch seconds; an unreadable plan lists no machine."""
    if not path.exists():
        return []
    try:
        rows = tomllib.loads(path.read_text(encoding="utf-8")).get("free", [])
        out = []
        for r in rows:
            when = {k: r.get(k) for k in ("from", "until")}
            if not all(isinstance(t, dt.datetime) for t in when.values()) or not isinstance(r.get("machine"), str):
                raise ValueError(f"row {r!r} lacks a machine or a from and an until as TOML date-times")
            out.append({**r, **{k: t.replace(tzinfo=t.tzinfo or dt.UTC).timestamp() for k, t in when.items()}})
        return out
    except (tomllib.TOMLDecodeError, ValueError, KeyError, TypeError) as e:
        say(f"the plan {path} is unreadable ({type(e).__name__}: {e}), so it lists no machine")
        return []


def decide(jobs: list[Job], door: dict, record: list[dict], plan: list[dict], cfg: Settings, now: float) -> list[Decision]:
    """What a pass does with each job that is due or that a bound stops; a job with no due number is left out."""
    readings = {str(m.get("id")): m for m in door.get("metrics") or [] if isinstance(m, dict)}
    busy = {e.get("machine") for e in record if e.get("state") not in OVER}
    out, launched = [], 0
    for job in jobs:
        def skip(why: str, job: Job = job) -> None:
            out.append(Decision(job, False, why))

        if job.command is None or (SCRIPT.findall(job.command) or [""])[0].startswith(LOOPS):
            continue                                       # a loop's number: the loop sends it, and nothing launches it
        due = []
        for r in job.rows:
            i, e = str(r.get("id")), every_s(r)
            age = (readings.get(i) or {}).get("ageMs")
            if e is None:
                skip(f"{i}'s every {r.get('every')!r} is no count of minutes, hours or days")
                break
            if age is None:
                due.append(f"{i} has no reading")
            elif age / 1000 > e:
                due.append(f"{i}'s newest reading is {span(age / 1000)} old, past its every {r['every']}")
        else:
            if not due:
                continue
            why = "; ".join(due)
            opts, work, _ = split(job.command)
            if held := placeholders(opts, work):
                skip(f"{why}; but its command takes {', '.join(held)}, which only its owner fills in")
                continue
            blocked = None
            for r in job.rows:
                last = last_of(record, str(r.get("id")))
                if last and last.get("state") not in OVER:
                    blocked = f"its last launch, run {last.get('run') or '(not yet named)'} at {last['at']}, still runs"
                elif last and now - at(last["ended"]) < (every_s(r) or 0):
                    blocked = (f"its last launch, run {last.get('run') or '(none named)'}, {last['state']} at {last['ended']}, "
                               f"less than {r.get('id')}'s every {r['every']} ago")
                if blocked:
                    break
            if blocked:
                skip(f"{why}; but {blocked}")
                continue
            mine = [e for e in record if e.get("command") == job.command and e.get("declared") == job.declared
                    and e.get("state") in ("passed", "failed")]
            if len(mine) >= 2 and all(failed(e) for e in mine[-2:]):
                runs = ", ".join(e.get("run") or "(none named)" for e in mine[-2:])
                skip(f"{why}; but held: its command failed twice running ({runs}) under this declaration, and is launched "
                     f"again once its rows in {job.where} change")
                continue
            pool = option(opts, "--pool")
            if needs_gpu(opts):
                free = [p["machine"] for p in plan if p["from"] <= now < p["until"] and p["machine"] not in SMALL_DISK
                        and p["machine"] not in busy and (not p.get("metrics") or set(p["metrics"]) & set(job.ids))]
                if not free:
                    skip(f"{why}; but it needs a GPU and the plan ({cfg.plan}) lists no machine free for it now")
                    continue
                machine = free[0]
            else:
                named = option(opts, "--on")
                machine = named if named and named != "auto" and not PLACEHOLDER.match(named) else cfg.on
                pool = pool or cfg.pool
                if machine in SMALL_DISK:
                    skip(f"{why}; but its row names {machine}, a small-disk machine")
                    continue
            if launched >= cfg.max:
                skip(f"{why}; but this pass has made its {cfg.max} launches (--max)")
                continue
            launched += 1
            busy.add(machine)
            out.append(Decision(job, True, why, machine, argv_for(job, machine, pool)))
    return out


# ---- one pass ------------------------------------------------------------------------------------------------------------------

def step(state: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None], dry_run: bool = False) -> list[Decision]:
    """One pass at `now` (above); `fx` does what reaches outside the state: git, the door, the runs' records and the launch."""
    say = fx.log
    if cfg.stop.exists():
        say(f"stopped: {cfg.stop} exists, so nothing is launched; remove it to go on")
        return []
    state.setdefault("schema", SCHEMA)
    record = state.setdefault("launches", [])
    refresh(record, fx.status, now, say)
    try:
        commit = fx.refresh()
    except Exception as e:  # noqa: BLE001  (origin or the disk: tried again next pass, launching nothing meanwhile)
        say(f"main's tree could not be refreshed ({type(e).__name__}: {e}), so nothing is launched")
        return []
    jobs, broken = declarations(cfg.tree)
    for line in broken:
        say(line)
    try:
        door = fx.door()
    except Exception as e:  # noqa: BLE001  (without readings nothing is known to be due: nothing is launched)
        say(f"the door could not be read ({type(e).__name__}: {e}), so nothing is launched")
        return []
    out = decide(jobs, door, record, plan_rows(cfg.plan, say), cfg, now)
    say(f"main at {commit[:12]}: {sum(len(j.rows) for j in jobs)} numbers declared, {len(out)} due, "
        f"{sum(d.launch for d in out)} to launch" + (" (dry run: none is)" if dry_run else ""))
    for d in out:
        nums = ", ".join(d.job.ids)
        if not d.launch:
            say(f"skip {nums}: {d.why}")
            continue
        say(f"{'would launch' if dry_run else 'launch'} {nums} on {d.machine}: {d.why}: {shlex.join(d.argv)}")
        if dry_run:
            continue
        e = {"metrics": d.job.ids, "command": d.job.command, "declared": d.job.declared, "machine": d.machine,
             "argv": list(d.argv), "commit": commit, "at": iso(now), "run": None, "state": "launching"}
        record.append(e)
        save(state)                                      # before the launch: a restart during it must not launch it again
        try:
            rc, said = fx.launch(list(d.argv))
        except (OSError, subprocess.SubprocessError) as x:
            rc, said = None, f"{type(x).__name__}: {x}"
        if m := dispatch.LAUNCHED.search(said):
            e.update(run=m.group(1), state="submitted")
            say(f"launched {nums}: run {m.group(1)}")
        else:
            tail = " | ".join(ln.strip() for ln in said.splitlines()[-3:] if ln.strip())[-400:]
            e.update(state="failed", ended=iso(now), rc=rc, why=f"the launch named no run: {tail or 'it printed nothing'}")
            say(f"launch of {nums} failed (exit {rc}): it named no run: {tail or 'it printed nothing'}")
        save(state)
    del record[:-KEEP]
    if not dry_run:
        save(state)
    return out


# ---- the control pod ----------------------------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"{iso(time.time())} {msg}", flush=True)


class Pod:
    """What a pass does outside its state on the control pod: the clone, the door, the runs' records and `research run`."""

    def __init__(self, cfg: Settings, log: Callable[[str], None] = log) -> None:
        self.cfg, self.log = cfg, log

    def git(self, *args: str) -> str:
        r = subprocess.run(["git", "-C", str(self.cfg.tree), *args], capture_output=True, text=True, timeout=GIT_S,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        if r.returncode != 0:
            raise subprocess.SubprocessError(f"git {' '.join(args[:2])} exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return r.stdout.strip()

    def refresh(self) -> str:
        """Origin's `main` checked out in the clone, detached and clean: its commit."""
        if not (self.cfg.tree / ".git").exists():
            subprocess.run(["git", "clone", "-q", "--no-checkout", "--filter=blob:none", "-c", f"credential.helper={self.cfg.helper}",
                            self.cfg.url, str(self.cfg.tree)], check=True, capture_output=True, timeout=GIT_S,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        self.git("fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main")
        self.git("checkout", "-q", "--force", "--detach", "refs/remotes/origin/main")
        self.git("clean", "-ffdxq")
        return self.git("rev-parse", "HEAD")

    def door(self) -> dict:
        from research import metrics

        return metrics.Route(metrics.METRICS, metrics._key()).call()

    def status(self, run: str) -> tuple[str, int | None]:
        from research import remote, runs

        d = runs.resolve(runs.runs_root(), run)
        rec = remote.fetch(d)
        if rec.get("last_fetch_error"):
            raise ConnectionError(f"its machine is unreachable (rc {rec['last_fetch_error'].get('rc')})")
        s = runs.summarize(d)
        return s["state"], s["rc"]

    def launch(self, argv: list[str]) -> tuple[int | None, str]:
        """`argv`, a `research run`, in the clone with the clone's own research: (its exit code, its output)."""
        assert argv[:2] == ["research", "run"], argv
        env = {**os.environ, "PYTHONPATH": str(self.cfg.tree / "tools" / "research" / "src")}
        try:
            r = subprocess.run([sys.executable, "-m", "research", *argv[1:]], cwd=self.cfg.tree, env=env, capture_output=True,
                               text=True, stdin=subprocess.DEVNULL, timeout=LAUNCH_S)
        except subprocess.TimeoutExpired as e:
            return None, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        for ln in (r.stdout + r.stderr).splitlines():
            self.log(f"  {ln}")
        return r.returncode, r.stdout + r.stderr


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(path)


def machine(name: str) -> str:
    if name == "auto" or name in SMALL_DISK:
        raise argparse.ArgumentTypeError(f"{name} is refused: name a machine by itself, not `auto` or a small-disk one "
                                         f"({', '.join(sorted(SMALL_DISK))})")
    return name


def parse(argv: list[str] | None = None) -> tuple[Settings, argparse.Namespace]:
    ap = argparse.ArgumentParser(prog="metrics_rerun", description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True, type=Path, help="the loop's clone of main, made on first use")
    ap.add_argument("--state", required=True, type=Path, help="the record of launches")
    ap.add_argument("--stop", required=True, type=Path, help="while this file exists, nothing is launched")
    ap.add_argument("--plan", required=True, type=Path, help="the week's GPU plan: [[free]] machine, from, until")
    ap.add_argument("--on", required=True, type=machine, metavar="MACHINE", help="a CPU command's machine, unless its row names one")
    ap.add_argument("--pool", required=True, help="a CPU command's pool on that machine, unless its row names one")
    ap.add_argument("--max", type=int, default=2, help="launches a pass at most")
    ap.add_argument("--url", default=URL, help="origin")
    ap.add_argument("--credential-helper", default=HELPER, help="git's credential helper for origin ('' for none)")
    ap.add_argument("--every", type=float, default=900.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="one pass that says what it would launch and why, launching nothing "
                                                           "and writing no record")
    ns = ap.parse_args(argv)
    return Settings(ns.tree, ns.state, ns.stop, ns.plan, ns.on, ns.pool, ns.max, ns.url, ns.credential_helper), ns


def main(argv: list[str] | None = None) -> int:
    cfg, ns = parse(argv)
    pod = Pod(cfg)
    while True:
        state = load_state(cfg.state)
        step(state, cfg, pod, time.time(), lambda s: save_state(cfg.state, s), dry_run=ns.dry_run)
        if ns.once or ns.dry_run:
            return 0
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main())
