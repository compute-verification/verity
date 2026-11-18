"""Each declared number measured again on its schedule, which the control pod does by itself: each pass reads every
`metrics.toml` of a clean checkout of `main` (benchmarks/README.md's "Metrics"), each number's newest reading from the site's
GET /api/metrics (`research.metrics`), and launches the command of each number older than its `every` as a `research run` on a
machine. The pod runs a workload of its own only for a row that says `runs = "control"`: one that reads what only the pod
reaches (its own state, GitHub through its read token, the Linear app), which runs on the pod and nowhere else.

A pass holds a lock beside the record (`--state`.lock), so a dry run and the live loop never share the clone at once; a pass
that finds it held does nothing and says so. Then:

    refresh   fetch origin's `main` into `--tree` (the loop's own clone, made on first use) and check its tip out, detached and
              clean: the declarations are read there, and every command is launched from there, `--source .` shipping it.
    due       a number is due when the door lists it with no reading, or with its newest older than its `every`. One the door
              doesn't list (a site that reads another `main`) is not due, and is said. The rows of one file that share a
              command are one launch, which measures each of them.
    bounds    a due command is launched only when each of these holds, and the pass says which one stopped it, as one of
              REASONS (`classify` judges the first six whatever the door, the record and the plan say):
              - it is no loop's: a row whose command's script is under infra/control/ is a loop's number, which the loop
                sends;
              - none of its rows is parked: a row whose `goal` is PARKED (top/parked) is rerun only by hand;
              - its command splits as a shell line, and its `research run` options are all in ALLOWED, each with a value of
                its form: no option names a path of the pod (`--send`, `--source DIR`), so a row can't ship one of the pod's
                files to a machine, and a `--pool` is one of research's pools (POOLS). A command that is no `research run`
                is run under one. Its rows give one `runs`, absent or CONTROL, and a CONTROL row's command takes none of
                NOT_LOCAL, which a `research run` that runs where it is launched refuses;
              - it asks no more than a CPU check machine has (CEILING): `--cpus`, `--mem-gb` and `--timeout` each at or
                under its CPUs, GiB and seconds, read as `research run` reads them (a value it can't read is over); a
                CONTROL row's `--timeout` is at most CONTROL_S, since the pass waits for it;
              - it names no placeholder but its machine's (a run id such as FIRE_RUN, which only its owner picks): such a
                number's owner is rung once for each declaration of it;
              - it has a machine. A CONTROL row runs on the pod itself, and nowhere else. A command that asks for GPUs
                (`--gpus`), names a machine that is no CPU node (`research.auto.CPU_NODE`), or names `NODE` without a pool
                runs only on a machine the plan (`--plan`) lists
                for one of its numbers now, the machine it names when it names one, and that none of this loop's launches is
                in flight on. A command that names a CPU node runs there, in `--pool`; any other (no `--on`, `--on auto`,
                `--on NODE` with a pool) runs on `--on`, in `--pool`. Never `auto`, never a small-disk machine (SMALL_DISK);
              - none of its numbers has a launch of this loop's still in flight, or one that passed less than that number's
                `every` ago, or one that failed (it ran and ended non-zero, it never ran, or its launch named no run) less
                than RETRY_S or its `every`, the shorter, ago, whatever the door says (the record). A run that waited out its
                pool's line (`slot_timeout`) is no failure: it is launched again at a later pass, unrung and uncounted;
              - it has not ended the same way (`cause`) SAME_CAUSE times running under the declaration it has now, unless
                HOLD_S has passed since the last: a change to its rows launches it again;
              - the record was not reset (a corrupt one moved aside) less than that number's `every` ago;
              - fewer than `--max-in-flight` of this loop's launches are in flight (launching, queued or running), counting
                this pass's;
              - this pass has launched fewer than `--max`;
              - the stop file (`--stop`) is absent, read again just before each launch: it stops every launch (a dry run
                judges all the same).
    launch    in the tree, `research run --id RUN --on MACHINE [--pool POOL] --project verity --source .`, then the row's
              allowed options but its `--on`, `--cpus` and, in `--pool`, its `--pool` (the slot's cores are the run's, and
              the slot's memory rule admits it), then its workload, with the pod's site, Slack and Linear keys left out of
              its environment. A CONTROL row's is `research run --id RUN --project verity --source . [--timeout CONTROL_S]`,
              then its allowed options, then its workload: a local run on the pod, which the pass waits for, with the pod's
              site and Slack keys left out of its environment and the Linear app's (CONTROL_KEEPS) kept. The loop names
              the run: the record holds the launch and its run id before it starts (its numbers, command, declaration,
              machine and time). A launch that prints no run line (it failed, or its timeout killed it, or it ran here) is
              judged by that run's own record: a run that registered and reached its machine is in flight, and never
              launched again while it is; one that didn't never ran; one here that had not ended when its launch did, or
              LAUNCH_S after it began, never will. Each pass reads how the record's launches in flight stand
              (`research.runs`) before it judges; one whose run can't be read stays in flight.
    ring      a launch that fails rings its numbers' owners (the handle before each id's slash) once, with its run id and
              its reason line (`research msg ring-run`): at the first failure of a cause, and when SAME_CAUSE of them hold
              it. A number whose last SLOT_WAITS launches running each waited out its pool's line rings them once, at the
              last of them, and again only after a launch of it that didn't. A ring that fails is tried again next pass,
              RING_TRIES times in all.

The record keeps the newest KEEP launches, every launch in flight, and each declared number's newest SAME_CAUSE + 1; beside
them, each declared number's reason at the last pass that judged it (`numbers`: its reason of REASONS and the pass's line),
and the placeholder rings made. A record that isn't JSON is moved aside (`--state`.corrupt) and said once, and every number is
then held for its `every`, so nothing in flight is launched twice. A missing record is a first pass: a launch in flight when
it was lost may be launched once more.

The plan, written by the lead for the week (TOML; a missing file lists no machine), one table a machine:

    [[free]]
    machine = "MACHINE"
    from = <a TOML date-time, UTC>
    until = <a TOML date-time, UTC>
    metrics = ["OWNER/NAME", ...]      # the numbers it is free for; a row that names none is free for none

usage (the watchers' `metrics-rerun` loop on the control pod, infra/control/watchers.toml; research importable, the store's
key, a site key with events:read in $JOBS_TOKEN, and $SLACK_BOT_TOKEN for its rings):
  python3 infra/control/metrics_rerun.py --tree DIR --state FILE --stop FILE --plan FILE --on MACHINE --pool POOL --every S
      [--max 2] [--max-in-flight 4] [--once] [--dry-run]
  python3 infra/control/metrics_rerun.py classify --tree DIR     # each number of a checkout by `classify`, and the counts
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
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

from research.auto import CPU_NODE
from research.jobs import dispatch, runline
from research.metrics import REGISTRY_ROOTS
from research.pods.nebius.slot_only import POOLS
from research.remote import timeout_seconds
from research.runs import ENDED

SCHEMA = "verity/metrics-rerun/v1"
STAMP = "%Y-%m-%dT%H:%M:%SZ"
NAME = "metrics.toml"
SKIP = {".git", ".lake", ".venv", "node_modules", "__pycache__", "target", "build", "dist"}
LOOPS = "infra/control/"                                # a row whose command's script is here is a loop's, which sends it itself
PARKED = "top/parked"                                   # a row whose goal is this is parked work, rerun only by hand
#: what a CPU check machine has (logical CPUs, GiB) and the longest run the loop holds a slot for (seconds): the loop launches a
#: row only when its `research run` asks no more than this, and its owner reruns any other
CEILING: dict[str, float] = {"--cpus": 192, "--mem-gb": 750, "--timeout": 4 * 3600}
SCRIPT = re.compile(rf"(?:^|\s)((?:{'|'.join(REGISTRY_ROOTS)})/\S+\.(?:sh|py))(?:\s|$)")
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
    "--pool": frozenset(POOLS),      # a pool of research's; a CPU command runs in the loop's own
    "--as": WORD,                    # the lane its slot names
    "--cpus": WORD,                  # judged against CEILING (over_ceiling), and never passed on: the slot's cores are the run's
    "--mem-gb": WORD,                # memory reserved on the machine, read as `research run` reads it (over_ceiling)
    "--timeout": WORD,               # the run's own bound, read as `research run` reads it (over_ceiling)
    "--gpus": NUMBER,                # GPUs it waits for on the machine; the plan decides the machine
    "--store-read": None,            # a read-only store key for the workload, minted for the run and deleted after it
}
OWNERS = "NODE"                                         # `--on NODE`: a machine its owner picks
PLACEHOLDER = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=)?\[?([A-Z][A-Z0-9_]+)\]?$")  # FIRE_RUN, [RUN], A5_CARD=CARD: the owner's
NEVER_RAN = ("unlaunched", "unrun")                     # the run never registered or reached its machine; it never took its slot
UNSENT = "unsent"                                       # a registered run whose launch stopped before it reached its machine
WAITED = "waited"                                       # its pool had no slot for it within the wait (`slot_timeout`): not started
NOT_STARTED = 75                                        # slot.py's exit when it never ran its command
#: slot.py's slot.json `why` when its waiter left the line at --wait-max, before its command could run
SLOT_WAIT = "no check slot within --wait-max"
OVER = ("passed", "failed", WAITED, *NEVER_RAN)
FAILED = ("failed", *NEVER_RAN)                         # a launch that rings its owners and is tried again
RETRY_S = 3600                                          # a failed number is launched again this long after, or its every if shorter
SAME_CAUSE = 3                                          # failures in a row that ended alike, before the number is held
HOLD_S = 7 * 86400                                      # how long such a number is held, unless its rows change
SLOT_WAITS = 3                                          # launches in a row that waited out their pool's line, before a ring
RING_TRIES = 3                                          # passes that try a ring that failed
RING = ("research", "msg", "ring-run")
LAUNCH_S = dispatch.LAUNCH_TIMEOUT_S                    # a launch that names no run by then never ran
RUNS = "runs"                                           # a row's key saying where it must run; absent: a machine, as above
CONTROL = "control"                                     # `runs = "control"`: on the control pod itself, and nowhere else
#: what a run on the pod never takes: `research run` with no `--on` runs where it is launched, and refuses these
NOT_LOCAL = ("--on", "--pool", "--cpus", "--mem-gb", "--gpus", "--store-read")
CONTROL_S = LAUNCH_S // 2                               # a CONTROL run's longest, its --timeout: the pass waits for it
#: the pod's keys a CONTROL run keeps, which LEFT_OUT drops from every other: the Linear app's, which the reads of Linear take
CONTROL_KEEPS = frozenset({"LINEAR_CLIENT_ID", "LINEAR_CLIENT_SECRET"})
KEEP = 400                                              # launches the record keeps, beside those it always keeps
URL = "https://github.com/compute-verification/verity.git"
HELPER = "!/data/infra/gh_app_token.sh git"             # hands git the App's token when git asks: never in a config or an argument
GIT_S = 1800
SITE_KEY = "JOBS_TOKEN"                                 # the door's key, which no launch needs
LEFT_OUT = re.compile(rf"^(?:{SITE_KEY}|SLACK_.*|LINEAR_.*)$")   # the pod's keys a launch's environment never holds
#: why a declared number is or isn't launched, as the record's `numbers` and `classify` say
REASONS = {
    "launched": "launched at this pass",
    "runnable": "launched once it is due (classify)",
    "fresh": "its newest reading is within its every",
    "unlisted": "the door doesn't list it",
    "loop": "a loop's number, which the loop sends",
    "every": "its every is no count of minutes, hours or days",
    "parked": "parked work, rerun only by hand",
    "refused": "its command carries an option or a value the loop never launches",
    "ceiling": "its command asks more than a CPU check machine has",
    "placeholder": "its command holds a placeholder only its owner fills in; the owner is rung once",
    "plan": "it asks for GPUs or names a machine that is no CPU node, and the plan lists none free for it",
    "small-disk": "it names a machine whose disk is too small",
    "in-flight": "its last launch is still in flight",
    "ran": "its last launch passed less than its every ago",
    "retry": "its last launch failed, and it is launched again once RETRY_S or its every has passed",
    "failing": "it ended the same way SAME_CAUSE times running, and is held HOLD_S unless its rows change",
    "reset": "the record was reset less than its every ago",
    "max": "the pass made its --max launches",
    "limit": "the loop has its --max-in-flight launches in flight",
    "stopped": "the stop file stands",
}


@dataclass(frozen=True)
class Settings:
    tree: Path
    state: Path
    stop: Path
    plan: Path
    on: str
    pool: str
    max: int = 2
    max_in_flight: int = 4
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
    """A pass's judgment of a job: `reason` of REASONS for each of its numbers but those `per` names; one not `said` is a
    fresh number's or a loop's, which the pass doesn't print."""
    job: Job
    launch: bool
    why: str
    machine: str | None = None
    argv: tuple[str, ...] = ()
    reason: str = "launched"
    said: bool = True
    per: tuple[tuple[str, str], ...] = ()
    because: str = ""                                   # what `classify` found, for a ring
    rings: bool = False                                 # a placeholder the owner is rung for: its command runs on a CPU node


@dataclass(frozen=True)
class Verdict:
    """What the loop can do with a job whatever the door, the record and the plan say (`classify`)."""
    reason: str
    why: str = ""
    opts: tuple[tuple[str, str | None], ...] = ()
    work: tuple[str, ...] = ()
    wrapped: bool = True
    machine: str | None = None                          # a runnable command's machine; None for one the plan places
    held: tuple[str, ...] = ()                          # the placeholders it holds


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
    """The jobs of every metrics.toml under REGISTRY_ROOTS, and a line for each file that isn't TOML."""
    jobs, broken = [], []
    for top in REGISTRY_ROOTS:
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


def over_ceiling(opts: list[tuple[str, str | None]], ceiling: dict[str, float] = CEILING) -> list[str]:
    """What of a row's options asks more than `ceiling`, each as the skip line says it. A value is read as `research run`
    reads it (`--timeout` by remote.timeout_seconds, `--mem-gb` and `--cpus` as floats); one it can't read is over too."""
    past = []
    for name, value in opts:
        if (most := ceiling.get(name)) is None:
            continue
        try:
            asked = timeout_seconds(value) if name == "--timeout" else float(value)
        except ValueError:
            asked = math.nan
        if not asked <= most:
            past.append(f"{name} {value} could not be read" if math.isnan(asked) else f"{name} {value} over {most:g}")
    return past


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
    try:
        out = runline.options(tokens[2:i], ALLOWED, unknown=lambda name: f"takes {name}, which a declared command may not "
                                                                         f"carry (allowed: {', '.join(ALLOWED)})")
    except runline.Refused as e:
        raise Refused(f"its `research run` {e}") from None
    return out, tokens[i + 1:], True


def get(opts: list[tuple[str, str | None]], name: str) -> str | None:
    return next((v for n, v in opts if n == name), None)


def placeholders(opts: list[tuple[str, str | None]], work: list[str]) -> list[str]:
    """What the command leaves for its owner to fill in, but its machine (`--on NODE`, which the loop chooses)."""
    return [m.group(1) for t in [v for n, v in opts if n != "--on" and v] + work if (m := PLACEHOLDER.match(t))]


def planned(opts: list[tuple[str, str | None]]) -> bool:
    """Whether the command takes a machine from the plan: it asks for GPUs, names a machine that is no CPU node, or names
    `NODE` without a pool (a pool's run is a slot's, on a CPU node; `NODE` alone is often a GPU machine's lease)."""
    named = get(opts, "--on")
    return (any(n == "--gpus" for n, _ in opts) or (named == OWNERS and get(opts, "--pool") is None)
            or named not in (None, "auto", OWNERS) and not CPU_NODE.fullmatch(named))


def classify(job: Job, cfg: Settings) -> Verdict:
    """What the loop can do with `job` whatever the door, the record and the plan say: `runnable` (on `machine`, in `--pool`,
    once due; CONTROL, the pod itself, for a CONTROL row), `plan` (only on a machine the plan lists), or why it never launches
    it, each a reason of REASONS."""
    if job.command is None or (SCRIPT.findall(job.command) or [""])[0].startswith(LOOPS):
        return Verdict("loop", REASONS["loop"])
    if bad := next((r for r in job.rows if every_s(r) is None), None):
        return Verdict("every", f"{bad.get('id')}'s every {bad.get('every')!r} is no count of minutes, hours or days")
    if parked := [str(r.get("id")) for r in job.rows if r.get("goal") == PARKED]:
        whose = "its goal is" if len(parked) == len(job.rows) else f"{', '.join(parked)}'s goal is"
        return Verdict("parked", f"{whose} parked ({PARKED}): a parked number is rerun only by hand")
    if len(places := {r.get(RUNS) for r in job.rows}) > 1:
        return Verdict("refused", f"its rows, which share a command, give {RUNS} differently ({', '.join(map(repr, places))})")
    if (place := places.pop()) not in (None, CONTROL):
        return Verdict("refused", f"its {RUNS} is {place!r}, not {CONTROL!r}")
    try:
        opts, work, wrapped = options(job.command)
    except Refused as x:
        return Verdict("refused", str(x))
    if place == CONTROL and (here := [n for n, _ in opts if n in NOT_LOCAL]):
        return Verdict("refused", f"it runs on the control pod ({RUNS} = {CONTROL!r}), where `research run` runs where it is "
                                  f"launched and refuses {', '.join(here)}")
    if past := over_ceiling(opts, {"--timeout": CONTROL_S} if place == CONTROL else CEILING):
        where = "a run on the control pod, which the pass waits for, may take" if place == CONTROL else "a CPU check machine has"
        return Verdict("ceiling", f"its command asks for more than {where} ({', '.join(past)}): such a number is rerun by its "
                                  "owner")
    shape = {"opts": tuple(opts), "work": tuple(work), "wrapped": wrapped}
    if held := placeholders(opts, work):
        return Verdict("placeholder", f"its command takes {', '.join(held)}, which only its owner fills in", **shape,
                       held=tuple(held))
    if place == CONTROL:
        return Verdict("runnable", REASONS["runnable"], **shape, machine=CONTROL)
    named = get(opts, "--on")
    if planned(opts):
        return Verdict("plan", "asks for GPUs" if get(opts, "--gpus") else f"names its machine ({named})", **shape)
    if named in SMALL_DISK:
        return Verdict("small-disk", f"it names {named}, whose disk is too small for a measurement's tree and outputs", **shape)
    return Verdict("runnable", REASONS["runnable"], **shape, machine=named if CPU_NODE.fullmatch(named or "") else cfg.on)


def argv_for(opts: list[tuple[str, str | None]], work: list[str], wrapped: bool, machine: str, pool: str | None) -> tuple[str, ...]:
    """The launch: on `machine`, or, on CONTROL, a local run on the pod with its --timeout at most CONTROL_S."""
    gone = ("--on", "--cpus", "--pool") if pool else ("--on", "--cpus")
    rest = [x for n, v in opts if n not in gone for x in ((n,) if v is None else (n, v))]
    head = ["research", "run"] if machine == CONTROL else ["research", "run", "--on", machine]
    if pool:
        head += ["--pool", pool]
    if get(opts, "--project") is None:
        head += ["--project", "verity"]
    if get(opts, "--source") is None:
        head += ["--source", "."]
    if machine == CONTROL and get(opts, "--timeout") is None:
        head += ["--timeout", f"{CONTROL_S}"]
    if not wrapped:
        head += ["--cwd", "source"]
    return (*head, *rest, "--", *work)


# ---- the record ---------------------------------------------------------------------------------------------------------------

def of(record: list[dict], metric: str) -> list[dict]:
    return [e for e in record if metric in e.get("metrics", [])]


def settle(e: dict, status: Callable[[str], dict | None], now: float, over: bool, say: Callable[[str], None]) -> None:
    """How launch `e` stands, read from its run's own record: passed, failed (it ran), waited (its pool had no slot for it in
    time: no failure, never rung or counted), unrun (it never took its slot otherwise), unlaunched (no record of the run, or one
    that never reached its machine, or a run on the pod that had not ended, once its launch is `over`), or in flight. A run
    whose record can't be read stays in flight."""
    try:
        s = status(e["run"]) if e.get("run") else None
    except Exception as x:  # noqa: BLE001  (a run whose machine can't be read stays in flight; read again next pass)
        say(f"run {e['run']} ({', '.join(e['metrics'])}): not read ({type(x).__name__}: {x}); it stays in flight")
        return
    if s is not None and s.get("here") and s["state"] not in ENDED:
        if over:                                         # the pass waits for a run here: its runner is gone with its launch
            e.update(state="unlaunched", ended=iso(now), cause="unlaunched", rung=False,
                     why=f"run {e['run']} on the control pod had not ended ({s['state']}) when its launch did")
        return
    if s is None or s["state"] == UNSENT:
        if over:
            why = f"run {e.get('run')} never reached its machine" if s else f"no record of run {e.get('run')}"
            e.update(state="unlaunched", ended=iso(now), why=why, cause="unlaunched", rung=False)
        return
    if s["state"] in ENDED and s.get("waited"):
        e.update(state=WAITED, ended=iso(now), rc=s["rc"], how=s["state"])
        return
    if s["state"] in ENDED:
        ok = "passed" if s["state"] == "done" and s["rc"] == 0 else "failed" if s["started"] else "unrun"
        e.update(state=ok, ended=iso(now), rc=s["rc"], how=s["state"])
        if ok != "passed":
            kind = f", {s['class']}" if s.get("class") else ""
            e.update(cause=f"{ok} ({s['state']}, rc {s['rc']}{kind})", rung=False,
                     why=f"{s['class']}: {s.get('why') or ''}".rstrip(": ") if s.get("class") else f"{s['state']}, rc {s['rc']}")
        say(f"run {e['run']} ({', '.join(e['metrics'])}): {ok} ({s['state']}, rc {s['rc']})")
    else:
        e["state"] = s["state"]


def refresh(record: list[dict], status: Callable[[str], dict | None], now: float, say: Callable[[str], None]) -> None:
    """How each launch the record holds as in flight stands now (`settle`), and one line naming those that waited out their
    pool's line at this pass; the one that makes SLOT_WAITS of them running (`waits`) is to be rung. A launch is over once
    LAUNCH_S has passed since it began: until then a pass that died during it may have left its `research run` still
    registering the run."""
    waited = []
    for e in record:
        if e.get("state") not in OVER:
            settle(e, status, now, now - at(e["at"]) >= LAUNCH_S, say)
            if e["state"] == WAITED:
                waited.append(e)
                if len(waits(record, e)) == SLOT_WAITS:
                    e["rung"] = False
    if waited:
        say(f"congestion: {len(waited)} run(s) had no slot within their pool's wait and never started, each launched again on "
            f"a later pass, with no count toward a hold and a ring only once {SLOT_WAITS} of a number's launches running have: "
            + "; ".join(f"{e['run']} ({', '.join(e['metrics'])}) on {e.get('machine')}" for e in waited))


def waits(record: list[dict], e: dict) -> list[dict]:
    """Launch `e` and the launches of its first number before it, newest first, while each waited out its pool's line."""
    mine = of(record, e["metrics"][0])
    out = []
    for x in reversed(mine[:mine.index(e) + 1]):
        if x.get("state") != WAITED:
            break
        out.append(x)
    return out


def slot_timeout(s: dict) -> bool:
    """Whether the run of summary `s` (`runs.summarize`) ended because its pool had no slot for it within the wait: slot.py
    exited NOT_STARTED having left the line at --wait-max (slot.json `left`, SLOT_WAIT). A workload's own exit 75 is never
    this: its run took a slot (slot.json `held`), or ran in none (no slot.json)."""
    slot = s.get("slot") or {}
    return (s.get("rc") == NOT_STARTED and slot.get("state") == "left"
            and str(slot.get("why") or "").startswith(SLOT_WAIT))


def trim(record: list[dict], declared: set[str]) -> None:
    """The newest KEEP launches, every launch in flight, and for each declared number its newest SLOT_WAITS launches, its
    newest SAME_CAUSE + 1 that didn't wait for a slot and its newest two that ran, which the bounds and the rings read."""
    keep = set(range(max(0, len(record) - KEEP), len(record)))
    keep |= {i for i, e in enumerate(record) if e.get("state") not in OVER}
    for m in declared:
        mine = [i for i, e in enumerate(record) if m in e.get("metrics", [])]
        keep |= set(mine[-SLOT_WAITS:])
        keep |= set([i for i in mine if record[i].get("state") != WAITED][-(SAME_CAUSE + 1):])
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


def cause(e: dict) -> str:
    """How a failed launch ended, which a streak of failures holds by: a launch recorded before causes were holds its state."""
    return e.get("cause") or f"{e.get('state')} rc {e.get('rc')}"


def streak(record: list[dict], metric: str, declared: str | None = None) -> list[dict]:
    """`metric`'s newest launches, newest first, while each failed with the newest's cause (under `declared`, when given); one
    that waited for a slot is passed over, neither counting nor breaking it."""
    out = []
    for x in reversed(of(record, metric)):
        if x.get("state") == WAITED:
            continue
        if x.get("state") not in FAILED or (declared and x.get("declared") != declared) or (out and cause(x) != cause(out[0])):
            break
        out.append(x)
    return out


def bound(record: list[dict], r: dict, job: Job, now: float) -> tuple[str, str] | None:
    """What in the record stops row `r`'s number now, as (its reason, the line), or None."""
    i, e = str(r.get("id")), every_s(r) or 0
    mine = of(record, i)
    last = mine[-1] if mine else None
    if last and last.get("state") not in OVER:
        return "in-flight", f"its last launch, run {last.get('run') or '(not yet named)'} at {last['at']}, is still in flight"
    if last and last["state"] == "passed" and now - at(last["ended"]) < e:
        return "ran", f"its last launch, run {last['run']}, passed at {last['ended']}, less than {i}'s every {r['every']} ago"
    held = streak(record, i, job.declared)
    if len(held) >= SAME_CAUSE and now - at(held[0]["ended"]) < HOLD_S:
        runs = ", ".join(x.get("run") or "(none named)" for x in reversed(held[:SAME_CAUSE]))
        return "failing", (f"held: its last {len(held)} launches ended alike, {cause(held[0])} ({runs}), so it is launched again "
                           f"once its rows in {job.where} change, or {span(HOLD_S)} after {held[0]['ended']}")
    if last and last["state"] in FAILED and now - at(last["ended"]) < (gap := min(RETRY_S, e)):
        return "retry", (f"its last launch, run {last.get('run') or '(none named)'}, {last['state']} at {last['ended']}, and it "
                         f"is launched again {span(gap)} after")
    return None


def in_flight(record: list[dict]) -> int:
    """The loop's launches not yet ended: launching, queued or running, as `refresh` last read them."""
    return sum(e.get("state") not in OVER for e in record)


def decide(jobs: list[Job], door: dict, record: list[dict], plan: list[dict], cfg: Settings, now: float,
           held_since: str | None = None) -> list[Decision]:
    """What a pass does with each job, with its reason of REASONS: a job that is due, or that the door doesn't list, is said;
    a fresh one and a loop's are not."""
    readings = {str(m.get("id")): m for m in door.get("metrics") or [] if isinstance(m, dict)}
    busy = {e.get("machine") for e in record if e.get("state") not in OVER}
    flying = in_flight(record)
    out, launched = [], 0
    for job in jobs:
        def skip(reason: str, why: str, job: Job = job, **kw) -> None:
            out.append(Decision(job, False, why, reason=reason, **kw))

        v = classify(job, cfg)
        if v.reason in ("loop", "every"):
            skip(v.reason, v.why, said=v.reason == "every")
            continue
        due, unlisted = [], []
        for r in job.rows:
            i, e = str(r.get("id")), every_s(r)
            if i not in readings:
                unlisted.append(i)
            elif (age := readings[i].get("ageMs")) is None:
                due.append(f"{i} has no reading")
            elif age / 1000 > e:
                due.append(f"{i}'s newest reading is {span(age / 1000)} old, past its every {r['every']}")
        gone = f"the door doesn't list {', '.join(unlisted)}, so it isn't due" if unlisted else ""
        if not due:
            skip("fresh", gone or REASONS["fresh"], said=bool(gone), per=tuple((i, "unlisted") for i in unlisted))
            continue
        why = "; ".join(due + ([gone] if gone else []))
        if v.reason not in ("runnable", "plan"):
            skip(v.reason, f"{why}; but {v.why}", because=v.why, rings=v.reason == "placeholder" and not planned(list(v.opts)))
            continue
        if blocked := next((b for r in job.rows if (b := bound(record, r, job, now))), None):
            skip(blocked[0], f"{why}; but {blocked[1]}")
            continue
        if held_since and (r := next((r for r in job.rows if now - at(held_since) < (every_s(r) or 0)), None)):
            skip("reset", f"{why}; but the record was reset at {held_since} (a corrupt one moved aside), so it waits until "
                          f"{r.get('id')}'s every {r['every']} has passed since")
            continue
        if v.reason == "plan":
            named = get(v.opts, "--on")
            free = [p["machine"] for p in plan if p["from"] <= now < p["until"] and p["machine"] not in SMALL_DISK
                    and p["machine"] not in busy and set(p["metrics"]) & set(job.ids)
                    and named in (None, "auto", OWNERS, p["machine"])]
            if not free:
                skip("plan", f"{why}; but it {v.why}, and the plan ({cfg.plan}) lists no machine free for it now")
                continue
            machine, pool = free[0], None
        else:
            machine, pool = v.machine, None if v.machine == CONTROL else cfg.pool
        if flying + launched >= cfg.max_in_flight:
            skip("limit", f"{why}; but {flying + launched} of this loop's launches are in flight, its --max-in-flight "
                        f"{cfg.max_in_flight}")
            continue
        if launched >= cfg.max:
            skip("max", f"{why}; but this pass has made its {cfg.max} launches (--max)")
            continue
        launched += 1
        busy.add(machine)
        out.append(Decision(job, True, why, machine, argv_for(list(v.opts), list(v.work), v.wrapped, machine, pool)))
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
    reasons = {i: [d.reason, d.why] for d in out for i in d.job.ids} | {i: [r, d.why] for d in out for i, r in d.per}
    counts = sorted(((n, r) for r in REASONS if (n := sum(x[0] == r for x in reasons.values()))), key=lambda x: (-x[0], x[1]))
    said = [d for d in out if d.said]
    say(f"main at {commit[:12]}: {sum(len(j.rows) for j in jobs)} numbers declared, {len(said)} said, "
        f"{sum(d.launch for d in out)} to launch" + (" (dry run: none is)" if dry_run else "")
        + f"; {in_flight(record)} in flight (--max-in-flight {cfg.max_in_flight}), "
          f"{sum(d.reason == 'limit' for d in out)} held back for it; by reason: {', '.join(f'{r} {n}' for n, r in counts)}")
    ring_placeholders(state, jobs, out, fx, now, say, dry_run)
    for k, d in enumerate(said):
        nums = ", ".join(d.job.ids)
        if not d.launch:
            say(f"skip {nums}: {d.why}")
            continue
        if stopped(cfg, say, dry_run, f"{nums} is not launched"):
            for i in (i for x in said[k:] if x.launch for i in x.job.ids):
                reasons[i] = ["stopped", f"the stop file {cfg.stop} stood before its launch"]
            break
        say(f"{'would launch' if dry_run else 'launch'} {nums} on {d.machine}: {d.why}: {shlex.join(d.argv)}")
        if dry_run:
            continue
        rid = fx.new_id(now)
        argv = [*d.argv[:2], "--id", rid, *d.argv[2:]]
        e = {"metrics": d.job.ids, "command": d.job.command, "declared": d.job.declared, "where": d.job.where,
             "machine": d.machine, "argv": argv, "commit": commit, "at": iso(now), "run": rid, "state": "launching"}
        record.append(e)
        save(state)                                      # before the launch: a restart during it must not launch it again
        try:
            rc, said_ = fx.launch(argv)
        except (OSError, subprocess.SubprocessError) as x:
            rc, said_ = None, f"{type(x).__name__}: {x}"
        for i in d.job.ids:
            reasons[i] = ["launched", f"run {rid} on {d.machine}: {d.why}"]
        if dispatch.LAUNCHED.search(said_):
            e["state"] = "submitted"
            say(f"launched {nums}: run {rid}")
        else:
            tail = " | ".join(ln.strip() for ln in said_.splitlines()[-3:] if ln.strip())[-400:] or "it printed nothing"
            settle(e, fx.status, now, True, say)
            if e["state"] == "unlaunched":
                e.update(rc=rc, why=f"{e['why']}: {tail}")
                say(f"launch of {nums} named no run (exit {rc}) and {e['why']}, a fault of the launch and not of the command")
            elif e["state"] not in OVER:
                say(f"launch of {nums} named no run (exit {rc}: {tail}), but run {rid} is registered ({e['state']}): it is "
                    "in flight, and no later pass launches it again while it is")
        save(state)
    ring_failures(record, fx, now, say, dry_run)
    trim(record, {i for j in jobs for i in j.ids})
    if not dry_run:
        state["numbers"] = {i: {"reason": r, "why": w} for i, (r, w) in sorted(reasons.items())}
        state["judged"] = {"at": iso(now), "commit": commit}
        save(state)
    return out


def owners(ids: list[str]) -> list[str]:
    """The handles that own these numbers: each id's part before its slash."""
    return sorted({i.split("/")[0] for i in ids})


def ring(fx, entry: dict, run: str, rc: str, ids: list[str], text: str, now: float, say: Callable[[str], None],
         dry_run: bool) -> None:
    """Ring each owner of `ids` with `text`, once: `entry` holds when it was rung, or how many passes tried a ring that
    failed, RING_TRIES at most."""
    if dry_run:
        say(f"would ring {', '.join('@' + o for o in owners(ids))}: {text}")
        return
    try:
        for o in owners(ids):
            fx.ring(run, rc, o, text)
    except (OSError, subprocess.SubprocessError, RuntimeError) as x:
        entry["ring_tries"] = tries = entry.get("ring_tries", 0) + 1
        gave_up = tries >= RING_TRIES
        if gave_up:
            entry["rung"] = f"not rung: {type(x).__name__}: {x}"
        say(f"ring of {', '.join(ids)} failed ({type(x).__name__}: {x})" + ("; it is not tried again" if gave_up else
                                                                            "; it is tried again next pass"))
        return
    entry["rung"] = iso(now)
    say(f"rang {', '.join('@' + o for o in owners(ids))} for {', '.join(ids)}: {text}")


def ring_failures(record: list[dict], fx, now: float, say: Callable[[str], None], dry_run: bool) -> None:
    """Each failed launch not yet rung, rung to its numbers' owners: the first of a cause, and the one that holds it; a
    failure between them ended as the one before it, whose ring said so, and is not rung. A launch that waited out its pool's
    line is rung only as the SLOT_WAITS-th running (`refresh`)."""
    for e in record:
        if e.get("rung") is not False:
            continue
        if e["state"] == WAITED:
            runs_ = ", ".join(x.get("run") or "-" for x in reversed(waits(record, e)))
            text = (f"metrics-rerun: {', '.join(e['metrics'])}: its last {SLOT_WAITS} launches ({runs_}) each waited out "
                    f"{e.get('machine')}'s pool line and never started; it is launched again at a later pass, and rung "
                    "again only after a launch of it that starts")
            ring(fx, e, e.get("run") or "-", WAITED, e["metrics"], text, now, say, dry_run)
            continue
        run_ = streak(record, e["metrics"][0], e.get("declared"))
        k = len(run_) - run_.index(e) if e in run_ else 1
        if 1 < k < SAME_CAUSE:
            if not dry_run:
                e["rung"] = f"not rung: it ended as run {run_[run_.index(e) + 1].get('run')} did"
            continue
        then = (f"held: it ended this way {k} times running, so it is launched again once its rows in "
                f"{e.get('where') or 'its metrics.toml'} change, or {span(HOLD_S)} after" if k >= SAME_CAUSE else
                f"it is launched again after {span(RETRY_S)}, or its every if shorter")
        text = f"metrics-rerun: {', '.join(e['metrics'])} {e['state']} ({e.get('why') or cause(e)}); {then}"
        rc = e["state"] if e["state"] == "unlaunched" or e.get("rc") is None else str(e["rc"])   # an unlaunched rc is its launcher's
        ring(fx, e, e.get("run") or "-", rc, e["metrics"], text, now, say, dry_run)


def ring_placeholders(state: dict, jobs: list[Job], out: list[Decision], fx, now: float, say: Callable[[str], None],
                      dry_run: bool) -> None:
    """The owners of a due number whose command holds a placeholder and would otherwise run on a CPU node, rung once for each
    declaration of it (`rings`): one the plan places waits for the plan as well."""
    rings = state.get("rings", {})
    for d in out:
        key = f"{d.job.where} {d.job.declared}"
        if not d.rings or (r := rings.setdefault(key, {"ids": d.job.ids})).get("rung"):
            continue
        text = (f"metrics-rerun: {', '.join(d.job.ids)}: {d.because}, so the loop launches it only once its command in "
                f"{d.job.where} holds none")
        ring(fx, r, f"placeholder:{d.job.ids[0]}", "placeholder", d.job.ids, text, now, say, dry_run)
    if not dry_run:
        live = {f"{j.where} {j.declared}" for j in jobs}
        state["rings"] = {k: r for k, r in rings.items() if k in live}


# ---- the control pod ----------------------------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"{iso(time.time())} {msg}", flush=True)


def environment(env: dict[str, str], here: bool) -> dict[str, str]:
    """A launch's environment: `env` without the pod's keys (LEFT_OUT), but for a run on the pod, which keeps CONTROL_KEEPS."""
    return {k: v for k, v in env.items() if not LEFT_OUT.match(k) or (here and k in CONTROL_KEEPS)}


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
        """The run's state and exit code, whether it started (took its slot, or, outside a pool, ran and didn't exit as
        slot.py does when it never ran its command), and its failure's class and line where `research run` classified one.
        None when no run of that id is registered here; UNSENT when its launch stopped before it named the run's directory on
        its machine (remote.json's `remote_run_dir`), so it never got there; `here` for a run on this pod (a status.json and
        no remote.json), read from its own directory."""
        from research import asker, remote, runs

        d = runs.runs_root() / run
        if not d.is_dir():
            return None
        p = d / "remote.json"
        here = not p.exists() and (d / "status.json").exists()
        if not here and not (json.loads(p.read_text()) if p.exists() else {}).get("remote_run_dir"):
            return {"state": UNSENT, "rc": None, "started": False}
        if not here and (rec := remote.fetch(d)).get("last_fetch_error"):
            raise ConnectionError(f"its machine is unreachable (rc {rec['last_fetch_error'].get('rc')})")
        s = runs.summarize(d)
        started = asker.started(s) or (s.get("slot") is None and bool(s.get("started_utc")) and s.get("rc") != NOT_STARTED)
        return {"state": s["state"], "rc": s["rc"], "started": started, "class": s.get("class"), "why": s.get("why"),
                "waited": slot_timeout(s), "here": here}

    def launch(self, argv: list[str]) -> tuple[int | None, str]:
        """`argv`, a `research run`, in the clone with the clone's own research and without the pod's keys (LEFT_OUT) but, for
        a run here (no `--on`), the Linear app's (CONTROL_KEEPS): (its exit code, its output)."""
        assert argv[:2] == ["research", "run"], argv
        env = environment(os.environ, here="--on" not in argv[:argv.index("--")])
        env["PYTHONPATH"] = str(self.cfg.tree / "tools" / "research" / "src")
        try:
            r = subprocess.run([sys.executable, "-m", "research", *argv[1:]], cwd=self.cfg.tree, env=env, capture_output=True,
                               text=True, stdin=subprocess.DEVNULL, timeout=LAUNCH_S)
        except subprocess.TimeoutExpired as e:
            return None, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        for ln in (r.stdout + r.stderr).splitlines():
            self.log(f"  {ln}")
        return r.returncode, r.stdout + r.stderr

    def ring(self, run: str, rc: str, owner: str, text: str) -> None:
        """`owner`'s inbox rung with `text` (`research msg ring-run`, which rings a run once however often called), with the
        pod's own research; RuntimeError when it fails."""
        r = subprocess.run([sys.executable, "-m", *RING, "--run", run, "--rc", rc, "--to", f"@{owner}", "--text", text],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        if r.returncode:
            raise RuntimeError(f"ring-run exited {r.returncode}: {r.stderr.strip()[-300:]}")


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
    with open(tmp, "w") as f:
        f.write(json.dumps(state, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())                             # on disk before the rename, so a crash leaves the old or the new
    tmp.replace(path)
    d = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(d)                                      # the rename on disk too, so a crash can't bring back a record without it
    finally:
        os.close(d)


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
    ap.add_argument("--pool", required=True, help="a CPU command's pool on that machine")
    ap.add_argument("--every", required=True, type=float, metavar="S", help="seconds between passes")
    ap.add_argument("--max", type=int, default=2, help="launches a pass at most")
    ap.add_argument("--max-in-flight", type=int, default=4, metavar="N",
                    help="the loop's launches not yet ended (queued or running) at most: a pass launches nothing new at N")
    ap.add_argument("--url", default=URL, help="origin")
    ap.add_argument("--credential-helper", default=HELPER, help="git's credential helper for origin ('' for none)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="one pass that says what it would launch and why, launching nothing "
                                                           "and writing no record")
    ns = ap.parse_args(argv)
    return Settings(ns.tree, ns.state, ns.stop, ns.plan, ns.on, ns.pool, ns.max, ns.max_in_flight, ns.url,
                    ns.credential_helper), ns


def census(tree: Path, out=print) -> dict[str, int]:
    """Each declared number of `tree` by `classify` (one line each: reason, id, why), then the count of each reason: what the
    loop could ever launch, read from the files alone."""
    cfg = Settings(tree, Path("-"), Path("-"), Path("-"), "MACHINE", "POOL")
    jobs, broken = declarations(tree)
    for line in broken:
        out(line)
    counts: dict[str, int] = {}
    for job in jobs:
        v = classify(job, cfg)
        for i in job.ids:
            counts[v.reason] = counts.get(v.reason, 0) + 1
            out(f"{v.reason:<12} {i}: {v.why}" + (f" (on {v.machine})" if v.machine and v.machine != cfg.on else ""))
    out(f"{sum(counts.values())} numbers: " + ", ".join(f"{r} {counts[r]}" for r in REASONS if r in counts))
    return counts


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["classify"]:
        ap = argparse.ArgumentParser(prog="metrics_rerun classify", description=census.__doc__.split("\n")[0])
        ap.add_argument("--tree", required=True, type=Path, help="a checkout, read as it stands")
        census(ap.parse_args(argv[1:]).tree)
        return 0
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
