"""Each declared number measured again on its schedule, which the control pod does by itself: each pass reads every
`metrics.toml` of a clean checkout of `main` (benchmarks/README.md's "Metrics"), each number's newest reading from the site's
GET /api/metrics (`research.metrics`), and launches the command of each number older than its `every` as a `research run` on a
machine. The pod runs no workload of its own.

A pass holds a lock beside the record (`--state`.lock), so a dry run and the live loop never share the clone at once; a pass
that finds it held does nothing and says so. Then:

    refresh   fetch origin's `main` into `--tree` (the loop's own clone, made on first use) and check its tip out, detached and
              clean: the declarations are read there, and every command is launched from there, `--source .` shipping it.
    due       a number is due when the door lists it with no reading, or with its newest older than its `every`. One the door
              doesn't list (a site that reads another `main`) is not due, and is said. The rows of one file that share a
              command are one launch, which measures each of them.
    bounds    a due command is launched only when each of these holds, and the pass says which one stopped it:
              - it is no loop's: a row whose command's script is under infra/ is a loop's number, which the loop sends;
              - its command splits as a shell line, and is a `research run` that names its machine (`--on`): one without
                `--on` is a local run, which runs where it is launched, and the loop launches nothing on the pod; a command
                that is no `research run` is run under one;
              - its `research run` options are all in ALLOWED, each with a value of its form: no option names a path of the
                pod (`--send`, `--source DIR`), so a row can't ship one of the pod's files to a machine;
              - it names no placeholder but its machine's (a run id such as FIRE_RUN, which only its owner picks);
              - none of its numbers has a launch of this loop's still in flight, or one that ran and ended less than that
                number's `every` ago, whatever the door says (the record);
              - none of its numbers has had NEVER_MAX launches in a row that never ran (a run that never registered or never
                reached its machine, or one that ended without taking its slot) unless that number's `every` has passed
                since the last; then it is tried once more;
              - it has not failed twice running under the declaration it has now (two runs that started and failed): a
                change to its rows launches it again;
              - the record was not reset (a corrupt one moved aside) less than that number's `every` ago;
              - it has a machine. A command that asks for GPUs (`--gpus`) or names any machine but `auto` (`--on NODE`,
                `--on vy-...`) runs only on a machine the plan (`--plan`) lists for one of its numbers now, the machine it
                names when it names one, and that none of this loop's launches is in flight on. Any other runs on `--on`,
                in its row's `--pool`, else in `--pool`. Never `auto`, never a small-disk machine (SMALL_DISK);
              - this pass has launched fewer than `--max`;
              - the stop file (`--stop`) is absent, read again just before each launch: it stops every launch (a dry run
                judges all the same).
    launch    in the tree, `research run --id RUN --on MACHINE [--pool POOL] --project verity --source .`, then the row's
              allowed options but its `--on`, then its workload, with the pod's site key left out of its environment. The
              loop names the run: the record holds the launch and its run id before it starts (its numbers, command,
              declaration, machine and time). A launch that prints no run line (it failed, or its timeout killed it) is
              judged by that run's own record: a run that registered and reached its machine is in flight, and never
              launched again while it is; one that didn't never ran. Each pass reads how the record's launches in flight
              stand (`research.runs`) before it judges; one whose run can't be read stays in flight.

The record keeps the newest KEEP launches, every launch in flight, and each declared number's newest NEVER_MAX + 1. A record
that isn't JSON is moved aside (`--state`.corrupt) and said once, and every number is then held for its `every`, so nothing
in flight is launched twice. A missing record is a first pass: a launch in flight when it was lost may be launched once more.

The plan, written by the lead for the week (TOML; a missing file lists no machine), one table a machine:

    [[free]]
    machine = "MACHINE"
    from = <a TOML date-time, UTC>
    until = <a TOML date-time, UTC>
    metrics = ["OWNER/NAME", ...]      # the numbers it is free for; a row that names none is free for none

usage (the watchers' `metrics-rerun` loop on the control pod, infra/control/watchers.toml; research importable, the store's
key, and a site key with events:read in $JOBS_TOKEN):
  python3 infra/control/metrics_rerun.py --tree DIR --state FILE --stop FILE --plan FILE --on MACHINE --pool POOL --every S
      [--max 2] [--once] [--dry-run]
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import datetime as dt
import fcntl
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
from research.runs import ENDED

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
WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
NUMBER = re.compile(r"^[0-9]+(\.[0-9]+)?$")
#: the `research run` options a declared command may carry, each with the form of its value (None: a flag). Anything else is
#: refused, naming the option: the loop launches as the pod, so a row must not reach a file of the pod (`--send`,
#: `--source DIR`, `--runs-dir`) or the run's machine-side settings (`--id`, `--custody-dir`, `--claimed`)
ALLOWED: dict[str, re.Pattern | frozenset | None] = {
    "--on": WORD,                    # replaced by the machine the loop picks; `auto`, NODE or a name says which kind
    "--project": frozenset({"verity"}),
    "--source": frozenset({"."}),    # the loop's own clone of main, and nothing else
    "--cwd": frozenset({"source", "clone"}),
    "--pool": WORD,                  # the slot pool on the machine
    "--as": WORD,                    # the lane its slot names
    "--mem-gb": NUMBER,              # memory reserved on the machine
    "--timeout": NUMBER,             # the run's own bound
    "--gpus": NUMBER,                # GPUs it waits for on the machine; the plan decides the machine
    "--store-read": None,            # a read-only store key for the workload, minted for the run and deleted after it
}
OWNERS = "NODE"                                         # `--on NODE`: a machine its owner picks
PLACEHOLDER = re.compile(r"^\[?([A-Z][A-Z0-9_]+)\]?$")  # FIRE_RUN, [RUN: what the command's owner fills in
NEVER_RAN = ("unlaunched", "unrun")                     # the run never registered or reached its machine; it never took its slot
UNSENT = "unsent"                                       # a registered run whose launch stopped before it reached its machine
OVER = ("passed", "failed", *NEVER_RAN)
NEVER_MAX = 2                                           # launches in a row that never ran, before the number waits its every
LAUNCH_S = dispatch.LAUNCH_TIMEOUT_S                    # a launch that names no run by then never ran
KEEP = 400                                              # launches the record keeps, beside those it always keeps
URL = "https://github.com/compute-verification/verity.git"
HELPER = "!/data/infra/gh_app_token.sh git"             # hands git the App's token when git asks: never in a config or an argument
GIT_S = 1800
SITE_KEY = "JOBS_TOKEN"                                 # the door's key, which no launch needs


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

    @property
    def lock(self) -> Path:
        return self.state.with_name(self.state.name + ".lock")


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


class Refused(ValueError):
    """A command the loop never launches, with why."""


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


def options(command: str) -> tuple[list[tuple[str, str | None]], list[str], bool]:
    """A command as (its `research run` options, each checked against ALLOWED, the workload, whether it is a `research run`).
    Raises Refused for a line that doesn't split, an option outside ALLOWED, a value not of its option's form, or one given
    twice."""
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        raise Refused(f"its command doesn't split as a shell line ({e})") from None
    if tokens[:2] != ["research", "run"]:
        return [], tokens, False
    if "--" not in tokens:
        raise Refused("its `research run` has no `--` before its workload")
    i = tokens.index("--")
    raw, out, seen = tokens[2:i], [], set()
    while raw:
        tok = raw.pop(0)
        name, eq, value = tok.partition("=")
        if name not in ALLOWED:
            raise Refused(f"its `research run` takes {name}, which a declared command may not carry (allowed: "
                          f"{', '.join(ALLOWED)})")
        if name in seen:
            raise Refused(f"its `research run` gives {name} twice")
        seen.add(name)
        form = ALLOWED[name]
        if form is None:
            if eq:
                raise Refused(f"its `research run` gives the flag {name} a value")
            out.append((name, None))
            continue
        if not eq:
            if not raw:
                raise Refused(f"its `research run` gives {name} no value")
            value = raw.pop(0)
        if not (value in form if isinstance(form, frozenset) else form.match(value)):
            raise Refused(f"its `research run` gives {name} {value!r}, which isn't "
                          + (" or ".join(sorted(form)) if isinstance(form, frozenset) else f"of the form {form.pattern}"))
        out.append((name, value))
    return out, tokens[i + 1:], True


def get(opts: list[tuple[str, str | None]], name: str) -> str | None:
    return next((v for n, v in opts if n == name), None)


def placeholders(opts: list[tuple[str, str | None]], work: list[str]) -> list[str]:
    """What the command leaves for its owner to fill in, but its machine (`--on NODE`, which the loop chooses)."""
    return [m.group(1) for t in [v for n, v in opts if n != "--on" and v] + work if (m := PLACEHOLDER.match(t))]


def planned(opts: list[tuple[str, str | None]]) -> bool:
    """Whether the command takes a machine from the plan: it asks for GPUs or names any machine but `auto`."""
    return any(n == "--gpus" for n, _ in opts) or get(opts, "--on") not in (None, "auto")


def argv_for(opts: list[tuple[str, str | None]], work: list[str], wrapped: bool, machine: str, pool: str | None) -> tuple[str, ...]:
    rest = [x for n, v in opts if n != "--on" for x in ((n,) if v is None else (n, v))]
    head = ["research", "run", "--on", machine]
    if pool and get(opts, "--pool") is None:
        head += ["--pool", pool]
    if get(opts, "--project") is None:
        head += ["--project", "verity"]
    if get(opts, "--source") is None:
        head += ["--source", "."]
    if not wrapped:
        head += ["--cwd", "source"]
    return (*head, *rest, "--", *work)


# ---- the record ---------------------------------------------------------------------------------------------------------------

def of(record: list[dict], metric: str) -> list[dict]:
    return [e for e in record if metric in e.get("metrics", [])]


def settle(e: dict, status: Callable[[str], dict | None], now: float, over: bool, say: Callable[[str], None]) -> None:
    """How launch `e` stands, read from its run's own record: passed, failed (it ran), unrun (it never took its slot), unlaunched
    (no record of the run, or one that never reached its machine, once its launch is `over`), or in flight. A run whose record
    can't be read stays in flight."""
    try:
        s = status(e["run"]) if e.get("run") else None
    except Exception as x:  # noqa: BLE001  (a run whose machine can't be read stays in flight; read again next pass)
        say(f"run {e['run']} ({', '.join(e['metrics'])}): not read ({type(x).__name__}: {x}); it stays in flight")
        return
    if s is None or s["state"] == UNSENT:
        if over:
            why = f"run {e.get('run')} never reached its machine" if s else f"no record of run {e.get('run')}"
            e.update(state="unlaunched", ended=iso(now), why=why)
        return
    if s["state"] in ENDED:
        ok = "passed" if s["state"] == "done" and s["rc"] == 0 else "failed" if s["started"] else "unrun"
        e.update(state=ok, ended=iso(now), rc=s["rc"], how=s["state"])
        say(f"run {e['run']} ({', '.join(e['metrics'])}): {ok} ({s['state']}, rc {s['rc']})")
    else:
        e["state"] = s["state"]


def refresh(record: list[dict], status: Callable[[str], dict | None], now: float, say: Callable[[str], None]) -> None:
    """How each launch the record holds as in flight stands now (`settle`). A launch is over once LAUNCH_S has passed since it
    began: until then a pass that died during it may have left its `research run` still registering the run."""
    for e in record:
        if e.get("state") not in OVER:
            settle(e, status, now, now - at(e["at"]) >= LAUNCH_S, say)


def trim(record: list[dict], declared: set[str]) -> None:
    """The newest KEEP launches, every launch in flight, and for each declared number its newest NEVER_MAX + 1 launches and
    its newest two that ran, which the bounds read."""
    keep = set(range(max(0, len(record) - KEEP), len(record)))
    keep |= {i for i, e in enumerate(record) if e.get("state") not in OVER}
    for m in declared:
        mine = [i for i, e in enumerate(record) if m in e.get("metrics", [])]
        keep |= set(mine[-(NEVER_MAX + 1):])
        keep |= set([i for i in mine if record[i].get("state") in ("passed", "failed")][-2:])
    record[:] = [e for i, e in enumerate(record) if i in keep]


# ---- the judgment -------------------------------------------------------------------------------------------------------------

def plan_rows(path: Path, say: Callable[[str], None]) -> list[dict]:
    """The plan's [[free]] rows that name numbers, each with its `from` and `until` as epoch seconds; an unreadable plan lists
    no machine, and a row that names no number is free for none. Each is said."""
    if not path.exists():
        return []
    try:
        out = []
        for r in tomllib.loads(path.read_text(encoding="utf-8")).get("free", []):
            when = {k: r.get(k) for k in ("from", "until")}
            if not all(isinstance(t, dt.datetime) for t in when.values()) or not isinstance(r.get("machine"), str):
                raise ValueError(f"row {r!r} lacks a machine or a from and an until as TOML date-times")
            if not r.get("metrics"):
                say(f"the plan's row for {r['machine']} names no metrics, so it is free for none")
                continue
            out.append({**r, **{k: t.replace(tzinfo=t.tzinfo or dt.UTC).timestamp() for k, t in when.items()}})
        return out
    except (tomllib.TOMLDecodeError, ValueError, KeyError, TypeError) as e:
        say(f"the plan {path} is unreadable ({type(e).__name__}: {e}), so it lists no machine")
        return []


def bound(record: list[dict], r: dict, now: float) -> str | None:
    """What in the record stops row `r`'s number now, or None."""
    i, e = str(r.get("id")), every_s(r) or 0
    mine = of(record, i)
    last = mine[-1] if mine else None
    if last and last.get("state") not in OVER:
        return f"its last launch, run {last.get('run') or '(not yet named)'} at {last['at']}, is still in flight"
    if last and last["state"] in ("passed", "failed") and now - at(last["ended"]) < e:
        return f"its last launch, run {last['run']}, {last['state']} at {last['ended']}, less than {i}'s every {r['every']} ago"
    streak = []
    for x in reversed(mine):
        if x.get("state") not in NEVER_RAN:
            break
        streak.append(x)
    if len(streak) >= NEVER_MAX and now - at(streak[0]["ended"]) < e:
        runs = ", ".join(f"{x.get('run') or '(none named)'} {x['state']}" for x in reversed(streak[:NEVER_MAX]))
        return (f"held: its last {len(streak)} launches never ran ({runs}), so it is tried again once {i}'s every "
                f"{r['every']} has passed since {streak[0]['ended']}")
    return None


def decide(jobs: list[Job], door: dict, record: list[dict], plan: list[dict], cfg: Settings, now: float,
           held_since: str | None = None) -> list[Decision]:
    """What a pass does with each job that is due, or that the door doesn't list; any other job is left out."""
    readings = {str(m.get("id")): m for m in door.get("metrics") or [] if isinstance(m, dict)}
    busy = {e.get("machine") for e in record if e.get("state") not in OVER}
    out, launched = [], 0
    for job in jobs:
        def skip(why: str, job: Job = job) -> None:
            out.append(Decision(job, False, why))

        if job.command is None or (SCRIPT.findall(job.command) or [""])[0].startswith(LOOPS):
            continue                                       # a loop's number: the loop sends it, and nothing launches it
        due, unlisted, bad = [], [], None
        for r in job.rows:
            i, e = str(r.get("id")), every_s(r)
            if e is None:
                bad = f"{i}'s every {r.get('every')!r} is no count of minutes, hours or days"
                break
            if i not in readings:
                unlisted.append(i)
            elif (age := readings[i].get("ageMs")) is None:
                due.append(f"{i} has no reading")
            elif age / 1000 > e:
                due.append(f"{i}'s newest reading is {span(age / 1000)} old, past its every {r['every']}")
        if bad:
            skip(bad)
            continue
        gone = f"the door doesn't list {', '.join(unlisted)}, so it isn't due" if unlisted else ""
        if not due:
            if gone:
                skip(gone)
            continue
        why = "; ".join(due + ([gone] if gone else []))
        try:
            opts, work, wrapped = options(job.command)
        except Refused as x:
            skip(f"{why}; but {x}")
            continue
        if wrapped and get(opts, "--on") is None:
            skip(f"{why}; but its command is a local run (a `research run` without --on), which runs where it is launched, "
                 "and the loop runs no workload on the pod")
            continue
        if held := placeholders(opts, work):
            skip(f"{why}; but its command takes {', '.join(held)}, which only its owner fills in")
            continue
        if blocked := next((b for r in job.rows if (b := bound(record, r, now))), None):
            skip(f"{why}; but {blocked}")
            continue
        if held_since and (r := next((r for r in job.rows if now - at(held_since) < (every_s(r) or 0)), None)):
            skip(f"{why}; but the record was reset at {held_since} (a corrupt one moved aside), so it waits until "
                 f"{r.get('id')}'s every {r['every']} has passed since")
            continue
        mine = [e for e in record if e.get("command") == job.command and e.get("declared") == job.declared
                and e.get("state") in ("passed", "failed")]
        if len(mine) >= 2 and all(e["state"] == "failed" for e in mine[-2:]):
            skip(f"{why}; but held: its command failed twice running ({mine[-2]['run']}, {mine[-1]['run']}) under this "
                 f"declaration, and is launched again once its rows in {job.where} change")
            continue
        pool = get(opts, "--pool")
        if planned(opts):
            named = get(opts, "--on")
            free = [p["machine"] for p in plan if p["from"] <= now < p["until"] and p["machine"] not in SMALL_DISK
                    and p["machine"] not in busy and set(p["metrics"]) & set(job.ids)
                    and named in (None, "auto", OWNERS, p["machine"])]
            if not free:
                what = "asks for GPUs" if get(opts, "--gpus") else f"names its machine ({named})"
                skip(f"{why}; but it {what}, and the plan ({cfg.plan}) lists no machine free for it now")
                continue
            machine = free[0]
        else:
            machine, pool = cfg.on, pool or cfg.pool
        if launched >= cfg.max:
            skip(f"{why}; but this pass has made its {cfg.max} launches (--max)")
            continue
        launched += 1
        busy.add(machine)
        out.append(Decision(job, True, why, machine, argv_for(opts, work, wrapped, machine, pool)))
    return out


# ---- one pass ------------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def locked(path: Path):
    """Whether this process now holds the pass's lock, held until the block ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def step(state: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None], dry_run: bool = False) -> list[Decision]:
    """One pass at `now` (above), under the pass's lock; `fx` does what reaches outside the state: git, the door, the runs'
    records and the launch."""
    with locked(cfg.lock) as mine:
        if not mine:
            fx.log(f"another pass holds {cfg.lock} (the live loop, a dry run or a second copy), so this one does nothing")
            return []
        return _pass(state, cfg, fx, now, save, dry_run)


def stopped(cfg: Settings, say: Callable[[str], None], dry_run: bool, what: str) -> bool:
    if not cfg.stop.exists():
        return False
    say(f"stopped: {cfg.stop} exists, so {what}; remove it to go on")
    return not dry_run


def _pass(state: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None], dry_run: bool) -> list[Decision]:
    say = fx.log
    if stopped(cfg, say, dry_run, "nothing is launched"):
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
    out = decide(jobs, door, record, plan_rows(cfg.plan, say), cfg, now, state.get("held_since"))
    say(f"main at {commit[:12]}: {sum(len(j.rows) for j in jobs)} numbers declared, {len(out)} said, "
        f"{sum(d.launch for d in out)} to launch" + (" (dry run: none is)" if dry_run else ""))
    for d in out:
        nums = ", ".join(d.job.ids)
        if not d.launch:
            say(f"skip {nums}: {d.why}")
            continue
        if stopped(cfg, say, dry_run, f"{nums} is not launched"):
            break
        say(f"{'would launch' if dry_run else 'launch'} {nums} on {d.machine}: {d.why}: {shlex.join(d.argv)}")
        if dry_run:
            continue
        rid = fx.new_id(now)
        argv = [*d.argv[:2], "--id", rid, *d.argv[2:]]
        e = {"metrics": d.job.ids, "command": d.job.command, "declared": d.job.declared, "machine": d.machine,
             "argv": argv, "commit": commit, "at": iso(now), "run": rid, "state": "launching"}
        record.append(e)
        save(state)                                      # before the launch: a restart during it must not launch it again
        try:
            rc, said = fx.launch(argv)
        except (OSError, subprocess.SubprocessError) as x:
            rc, said = None, f"{type(x).__name__}: {x}"
        if dispatch.LAUNCHED.search(said):
            e["state"] = "submitted"
            say(f"launched {nums}: run {rid}")
        else:
            tail = " | ".join(ln.strip() for ln in said.splitlines()[-3:] if ln.strip())[-400:] or "it printed nothing"
            settle(e, fx.status, now, True, say)
            if e["state"] == "unlaunched":
                e.update(rc=rc, why=f"{e['why']}: {tail}")
                say(f"launch of {nums} named no run (exit {rc}) and {e['why']}, a fault of the launch and not of the command")
            elif e["state"] not in OVER:
                say(f"launch of {nums} named no run (exit {rc}: {tail}), but run {rid} is registered ({e['state']}): it is "
                    "in flight, and no later pass launches it again while it is")
        save(state)
    trim(record, {i for j in jobs for i in j.ids})
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

    def new_id(self, now: float) -> str:
        from research import runs

        return runs.new_run_id(now)

    def status(self, run: str) -> dict | None:
        """The run's state and exit code, and whether it started: took its slot, or, outside a pool, ran and didn't exit as
        slot.py does when it never ran its command. None when no run of that id is registered here; UNSENT when its launch
        stopped before it named the run's directory on its machine (remote.json's `remote_run_dir`), so it never got there."""
        from research import asker, monitors, remote, runs

        d = runs.runs_root() / run
        if not d.is_dir():
            return None
        p = d / "remote.json"
        if not (json.loads(p.read_text()) if p.exists() else {}).get("remote_run_dir"):
            return {"state": UNSENT, "rc": None, "started": False}
        rec = remote.fetch(d)
        if rec.get("last_fetch_error"):
            raise ConnectionError(f"its machine is unreachable (rc {rec['last_fetch_error'].get('rc')})")
        s = runs.summarize(d)
        started = asker.started(s) or (s.get("slot") is None and bool(s.get("started_utc")) and s.get("rc") != monitors.NOT_STARTED)
        return {"state": s["state"], "rc": s["rc"], "started": started}

    def launch(self, argv: list[str]) -> tuple[int | None, str]:
        """`argv`, a `research run`, in the clone with the clone's own research and without the site key: (its exit code,
        its output)."""
        assert argv[:2] == ["research", "run"], argv
        env = {k: v for k, v in os.environ.items() if k != SITE_KEY}
        env["PYTHONPATH"] = str(self.cfg.tree / "tools" / "research" / "src")
        try:
            r = subprocess.run([sys.executable, "-m", "research", *argv[1:]], cwd=self.cfg.tree, env=env, capture_output=True,
                               text=True, stdin=subprocess.DEVNULL, timeout=LAUNCH_S)
        except subprocess.TimeoutExpired as e:
            return None, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        for ln in (r.stdout + r.stderr).splitlines():
            self.log(f"  {ln}")
        return r.returncode, r.stdout + r.stderr


def load_state(path: Path, now: float, say: Callable[[str], None], dry_run: bool = False) -> dict:
    """The record; one that isn't a JSON object is moved aside (but in a dry run) and said, and every number is held for its
    `every` from `now`, so nothing in flight is launched twice."""
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        state, why = None, f"{type(e).__name__}: {e}"
    else:
        why = f"a JSON {type(state).__name__}"
    if isinstance(state, dict):
        return state
    aside = path.with_name(path.name + ".corrupt")
    if not dry_run:
        path.replace(aside)
    say(f"the record {path} is not a JSON object ({why}): {'it would be' if dry_run else 'it is'} moved aside to {aside}, and "
        f"every number is held for its every from {iso(now)}")
    return {"schema": SCHEMA, "held_since": iso(now), "launches": []}


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
    ap.add_argument("--state", required=True, type=Path, help="the record of launches; its lock is beside it")
    ap.add_argument("--stop", required=True, type=Path, help="while this file exists, nothing is launched")
    ap.add_argument("--plan", required=True, type=Path, help="the week's GPU plan: [[free]] machine, from, until, metrics")
    ap.add_argument("--on", required=True, type=machine, metavar="MACHINE", help="a CPU command's machine")
    ap.add_argument("--pool", required=True, help="a CPU command's pool on that machine, unless its row names one")
    ap.add_argument("--every", required=True, type=float, metavar="S", help="seconds between passes")
    ap.add_argument("--max", type=int, default=2, help="launches a pass at most")
    ap.add_argument("--url", default=URL, help="origin")
    ap.add_argument("--credential-helper", default=HELPER, help="git's credential helper for origin ('' for none)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="one pass that says what it would launch and why, launching nothing "
                                                           "and writing no record")
    ns = ap.parse_args(argv)
    return Settings(ns.tree, ns.state, ns.stop, ns.plan, ns.on, ns.pool, ns.max, ns.url, ns.credential_helper), ns


def main(argv: list[str] | None = None) -> int:
    cfg, ns = parse(argv)
    pod = Pod(cfg)
    while True:
        now = time.time()
        state = load_state(cfg.state, now, log, ns.dry_run)
        step(state, cfg, pod, now, lambda s: save_state(cfg.state, s), dry_run=ns.dry_run)
        if ns.once or ns.dry_run:
            return 0
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main())
