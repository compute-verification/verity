"""`research queue drive`: one tick of the landing automation (Daniel, 11 Oct: from an owner's ready mark to `main` moving, no
agent acts unless the automation refuses).  A loop runs it every few minutes.  How `next` lands is drive.toml's `land`: `review`
(drive's of record; "Review landing" below), or `check`, the default, in which each tick does at most one of each step:

1. Land.  Walk the queue's branch down from its tip to the newest commit with a passing `check` and land it as the lander does:
   a merge commit onto `main` naming its PRs, with a `Checked:` trailer for the run.  On a `main` that moved outside the queue,
   the commit still lands when the commits `main` gained share no file with it (`Skipped-check:` names them) and change
   no check tooling (`CHECK_TOOLING`), which would have checked it differently.  Then
   `merge.caught_up` (the spec DM, node 1's mirror, the post-train regression, the Lean cache publish, the landing post) and
   the sweep.
2. Sync `next` (`Queue.sync`): admit and drop PRs, rebuild on a moved `main`; but while a check of `next` runs and `main` moved
   by files its train doesn't change (and no check tooling), `next` stays on its base so the check isn't wasted (`held`, the
   rule `queue.hold` keeps for every sync: `eject`'s and a hand one read drive's launches from the state file).
3. Check `next` in trains of `train_size` PRs, each stacked on the one below (every `train_size`-th member's commit, and the
   tip): `train.sh COMMIT MACHINE`, with `--agreement` when the train changes a `lean-audit.json` (the lander's rule; train.sh
   itself sends the build for C-Flock's paths), the lowest unchecked first, while fewer than `max_running` checks run, and
   none above a member whose failure drive is still settling, which it would hold; a tick's launches run at once
   (`LAUNCH_JOBS`), each with its train's rank in the slot queue.  `max_running` counts the checks that have started (a twin
   holds a check slot) and at most `UNSTARTED_COUNTED` that wait in slot lines; past those, a train launches only onto a free
   slot.  A check still waiting while a machine of `on` without a twin of it could start it now is cancelled and launched
   again there (`move_waiting`).  A former tip a re-cut left inside a train
   runs on as a hedge outside the count, until a train above it passes its check (`drop_hedges`).

What it refuses goes to `@ci` in the merge thread, once per commit or head, with the owner of the PR it concerns: a failed
check, a conflict, a check that ran and passed but lacks a step the gate needs, a launch that registered no run.  A failed
train's cause is found by bisection over the checked prefixes of `next` (`Driver.failed`): it is among the members above the
highest lower prefix whose check passed, or the base.  One member there is ejected (`research queue eject`, by `ci`); of
several, those whose change touches a file the failure names (a traceback's frames, a lint's offender: `implicated`) are
ejected at once, and otherwise drive checks the prefix halfway between and acts again when that check ends.  Overlap with
what failed's declared inputs only informs: no ejection rests on it.  Bisection alone ejects only a failure seen twice (its
train's check resent, or a bisect check failing as its train did), and the lowest member only once a check of `main` itself
passed: a failed one is told to `@ci` as `main` red, and nobody is ejected.  A PR pushed since the head that failed is not
ejected; its new head is checked.  A kill that names nothing, or a failure with no report, goes to `@ci`, who ejects the PR
that broke it.  The next sync rebuilds `next` without it.

With `--defer`, a train lands once every step but the deferred ones passed, and drive cancels the rest of its check.  One
post-landing run at a time (`train.sh MAIN MACHINE --deferred STEPS`, in a slot `max_running` doesn't count) checks the deferred
steps on `main`'s tip while a landing awaits it (`drive["postland"]`).  A pass in which every deferred step passed verifies that
tip; a failure reverts the one PR landed since the last verified `main` that it implicates (`landed_culprit`), else every landing
since, and tells `@ci` and the owners, but reverts nothing when no landing since touches what failed or its inputs (`main` was
broken before them); a run the machine killed runs again once.

`--shadow` drives `queue-shadow`: it lands nothing, launches nothing and posts nothing, and logs what it would do.

`--per-pr-shadow N` (drive.toml's `per_pr_shadow`, 0 is off) runs the per-PR landing (parallel landing's step 3) in shadow beside
the trains, to measure it: it lands, ejects and posts nothing, and never touches `next`, `main` or the trains' records.  Each
ready PR, oldest mark first, while fewer than N of its checks run, gets one check (`train.sh`, as an ordinary quiet job,
`PER_PR_ENV`) of its landing commit L, `main` merged with its head, pushed as `per-pr/N` (train.sh checks what origin holds).
A new head restarts only that PR's check.  L would land once its check passes by drive's rule (every step but the deferred
ones, as a train lands; the whole check too is recorded), when `main` is still the one L was built on; on a moved `main`, the
new L is planned by its parts' keys (`check.py --plan --packs-only --from` its checks, on the node its check ran on: `plan_launch`,
detached, at most N at once) and would land when every part the rule needs has a cited pass (`merge.plan_refusal`), else
it is checked again.  What it would land, and when, is logged and kept under `drive["per_pr"]` (`research queue stats --per-pr`
reads it); a record ends once the trains land the PR's head, or it closes, leaves the queue or gets a new head.

Review landing (`land = "review"`; Daniel, 24 Oct 06:05Z: PRs land on an eyeball review, and the full check runs after landing).
A tick syncs `next` as above, then lands its longest prefix whose every member has a land verdict at its head and no hold
(`research queue review N --land|--hold`, a `reviewed` label on `pr:N@HEAD`; one by its owner, the owner's handle or its ready
mark's author doesn't count: `queue.unreviewed`), a fast-forward of `main` through the same landing steps.  Two gates cut the
prefix first.  A Lean change (a member whose own change touches a lock, `lean-audit.json`, or a Lean source outside `Security/`
of a package that has one, Lean that runs as code: `queue.lean_change`; Lean proofs alone land on the verdict, checked after
landing, Daniel 25 Oct 11:19 PM PDT) lands only once a lean-audit passed on a commit of `next` at or above it: drive's own Lean check
(`train.sh TOP MACHINES --deferred lean-audit`, of the prefix's top, while `next` stays on its base for it as for a train's
check), or any recorded run of the store whose lean-audit passed there; one that regenerated locks lands exactly at its commit,
the locks on top.  While one runs, a Lean change above it, reviewed or not, gets a check stacked at `next`'s top, below what
came this tick (`lean_stack`, at most `max_lean_checks` running), so it doesn't wait for the lower one to pass and then for a
whole check of its own; the lower one runs on.  Each Lean check launched or stacked stops at a cut point, just below the Lean
change after the first ceil(n / f) of the n it would hold, f the Lean checks still free (`lean_cut`), lowest first: a failure
holds only what's above it, and the lower ones land without waiting for the rest.  `next` is built non-Lean first (drive.toml's
`nonlean_first`, `queue.Queue._nonlean_order`: what a sync merges anyway, never a member it keeps), so a member that changes
no Lean seldom waits above a Lean change for its check.  A failed Lean check is run again on the Lean change's own commit, and only a
failure there is told to its owner; nothing at or above a failed commit lands until it leaves `next`.  And every .py file the
landing leaves must compile (`py_compile`): the PR that last changed one that doesn't is ejected.  No train is checked.  While
any Lean check of a commit of `next` runs, drive's or a hand one of its run registry (`_running_lean`), the members up to it keep
their place, as under a train's check (`--first` goes above them).  ci's hand Lean check goes through drive
(`/data/infra/drive.sh queue lean-check COMMIT`, `Driver.hand_lean`): recorded in drive's state at its launch, it holds `next`
from its first second and counts, is followed and covers as drive's own.  Each
landing awaits the after-landing run, the post-landing run above with every ci.toml job (`AFTER_LANDING`), `--no-cache` while
`after_landing_no_cache`, and `--no-regenerate` (a lock record difference fails it).  Nothing lands before one such run of `main`
itself ended (its baseline, verifying the steps it passed; `/data/infra/drive.sh queue baseline SHA --by ci` records a green one
by hand).  A failure reverts what it implicates only when each failed step passed on the last verified `main` under the same
flags; with several PRs implicated and no revert among their landings it bisects them, each step a run of one landing's commit
with the failed steps alone, and nothing lands meanwhile.  A run with no readable state `LOST_AFTER_S` after its launch is lost.

The Lean checks and the post-landing runs (the after-landing run and its bisection's) are launched detached (`detach`), so no tick
waits on train.sh, which took 9-12 min to print a Lean check's run ids (26 Oct: a tick that launched two took 15-26 min).  A later
tick reads the ids from its log (`launched`, `_resolve`); until then the launch counts as running for every rule that counts
running checks (the hold on `next`, `max_lean_checks`, the waits), and a train.sh that exits without them registered no run.

State (launches, refusals told) is the queue's `state.json`, under `drive`."""
from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .queue import (CHECK_TOOLING, IDENTITY, PR, TOUCHED_LOGS, TOUCHED_REPORT, Member, Queue, QueueError, RemoteSuites, StoreLabels,
                    _git, _message, _out, built_on, changed, hold, lean_change, touched_suites, unreviewed)
from .runs import SLOT_FILE

#: who hears what drive can't settle (13 Oct: the other Project's lander, its addressee before, acts on none of it)
CI = "@ci"
#: a pytest location (`tests/test_x.py:503: NameError`) or a traceback frame (`File "/.../x.py", line 12`) in a suite's log
TRACE_RE = re.compile(r'^\s*([\w./+-]+\.py):\d+:|File "([^"]+\.py)", line \d+', re.M)
#: a repository path a pytest failure report names (`E   tools/x/README.md: 48 KB > 48 KB`, 13 Oct: a lint's offender is in
#: its assertion, not its traceback); relative, with a directory
NAMED_RE = re.compile(r"(?<![\w./-])((?:[\w+-]+/)+[\w+.-]*\w\.\w+)")
#: pytest's failing tests in its short summary
FAILED_RE = re.compile(r"^FAILED ([^\s:]+)::", re.M)
#: a test module pytest couldn't collect, in its short summary (`ERROR tests/x.py - KeyError: ...`)
ERROR_RE = re.compile(r"^ERROR ([^\s:]+\.py)", re.M)
#: in lean-audit's log: the package an audit failed (its path under the audit's scratch tree), and a Lean error's file, relative
#: to that package
LEAN_AUDIT_RE = re.compile(r"^AUDIT \S*?/lean-audit-scratch-[^/\s]+/(\S+?): FAIL", re.M)
LEAN_ERROR_RE = re.compile(r"^\s*error: (\S+?\.lean):\d+", re.M)
#: the audit's own failures under its `AUDIT .../PKG: FAIL` line: a file no root imports (relative to PKG), and a module that
#: imports outside its layer (22-23 Oct: #3215's orphan and #3140's Credit imports named no file drive matched, so it resent
#: their trains instead of ejecting the one PR)
LEAN_ORPHAN_RE = re.compile(r"^\s+orphan: (\S+?\.lean) is not imported")
LEAN_LAYERS_RE = re.compile(r"^\s+layers: ([A-Za-z][\w.]*) imports ")
#: a run's or a step's exit that a kill gave, not a failure of what it checked: SIGHUP, SIGKILL, SIGTERM as 128+N, or a signal
KILLED_RC = (129, 137, 143)
#: what the Lean check says of a process of its own that was killed (`tools/verity/lean/check.py`: the replay, `--fresh`)
KILLED_SAYS = "not a kernel verdict"
#: circuit-check's report of a failure: `FAIL <check> <target>`, and `circuit-check/killed` for a target whose process died
CIRCUIT_FAIL_RE = re.compile(r"^FAIL (\S+)", re.M)
CIRCUIT_KILLED = "circuit-check/killed"
#: how often drive sweeps the repository's branches (`merge.sweep_branches`)
BRANCH_SWEEP_S = 86400
#: what `Driver.culprit` says of a failure that names no file and was a kill
KILLED = "it was killed (a signal or 128+N) and names no file: a machine's fault"
#: what it says of one whose files, and the declared inputs of what failed, no PR's change touches: a machine's fault too
UNTOUCHED = "no PR's change touches what failed or its declared inputs: a machine's fault"
#: what it says when no report of the failed check reached drive
NO_REPORT = "no failed suite's report reached drive"
#: Lean's own inputs, for a Lean step that failed with no file named
LEAN_INPUTS = (".lean", "lakefile.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json", "lean-audit.json")
#: states `research status` reports for a run that may still publish an attempt
ACTIVE = ("submitted", "waiting", "running", "unknown")
#: the states of a twin that holds no check slot yet: launched, or waiting in its machine's slot line (slot.py's RUN_SLOT); a
#: launch all of whose live twins are in one of them hasn't started
WAITS = ("submitted", "waiting")
#: launches not yet started that `max_running` counts: beyond them a waiting train takes no room from one that would start (24 Oct
#: 05:15Z: four trains waiting in full lines filled the count while two new nodes' slots stood empty)
UNSTARTED_COUNTED = 2
#: after a launched run ends, how long its attempt may take to reach the store before the launch counts as lost
PUBLISH_GRACE_S = 1800
#: check.py's exit for a twin that never took a slot
NOT_STARTED = 75
#: the steps of a check (`tools/verity/check/check.py`), for `--defer`: a train lands once each one not deferred has passed
CHECK_STEPS = ("preflight-lock", "preflight-lints", "preflight-fixtures", "lean-changed", "flock-circuit-build", "lean-build",
               "lean-unit-cut", "lean-agreement", "lean-audit", "lean-suites", "pytest", "circuit-check", "flock-rust-tests")
#: what a verdict says when it passed with `--defer`'s steps still running
DEFERRED = "every step but the deferred ones passed"
#: what a review landing's verdict says (`land = "review"`)
REVIEWED = "a land verdict at each head"
#: the after-landing run of a review landing: every ci.toml job (`--deferred` takes jobs; the preflights are none)
AFTER_LANDING = tuple(s for s in CHECK_STEPS if not s.startswith("preflight-"))
#: the after-landing steps a run may skip by name: lean-agreement when nothing it checks changed, lean-changed under --no-cache
AFTER_LANDING_OPTIONAL = ("lean-agreement", "lean-changed")
#: ticks a member must keep its head on `next` before the review landing's Lean check reaches it (`Driver._lean_check`): one
#: of a prefix capped below recent churn isn't taken off `next` by the next re-mark (24 Oct: three checks lost to it in 3 h;
#: three ticks capped a check below one new head for an hour)
LEAN_SETTLE_TICKS = 1
#: how long a run may show no readable state (`unknown`) before drive takes it as lost (`Driver._lost`)
LOST_AFTER_S = 3 * 3600
#: the step a Lean change's Lean check before a review landing runs (`train.sh TOP MACHINES --deferred lean-audit`)
LEAN_STEP = "lean-audit"
#: a Lake package's lock, which a train's Lean check may regenerate (`merge.LOCK`)
LOCK = "lean-audit.json"
#: a run stopped by SIGTERM (128 + 15): a cancel, or a twin whose sibling won the slot
SIGTERM_RC = 143
#: pre-warms running at once: each is a whole check in a slot, and a train waits behind every one (18 Oct 00:43Z: four took the
#: room drive counts, and no train started)
PREWARM_MAX = 1
RUN_RE = re.compile(r"^RUN ((?:r20\d{6}-\d{6}-[0-9a-f]{4} ?)+)$", re.M)
#: what `git revert` writes of the commit it undoes
REVERTS_RE = re.compile(r"^This reverts commit ([0-9a-f]{40})", re.M)
#: the per-PR shadow's branch of a PR's landing commit, which train.sh needs on origin (`--per-pr-shadow`)
PER_PR_BRANCH = "per-pr/"
#: its records kept once they end
PER_PR_DONE = 1000
#: its landing commits' identity, so none equals a commit of `next`
PER_PR_IDENTITY = {**IDENTITY, "GIT_COMMITTER_NAME": "research queue per-PR shadow"}
#: its records' phases in which it still acts: a terminal one (would-land, failed, conflict) waits for a new head or the trains
PER_PR_LIVE = ("queued", "checking", "planning")


@dataclass
class Verdict:
    state: str                    # passed | running | failed | none
    attempt: str = ""
    why: str = ""


def needs_agreement(repo: Path, base: str, commit: str) -> bool:
    """The lander's rule: a train that adds or changes a guarantee anywhere is checked with the pinned upstream build."""
    return any(p == "lean-audit.json" or p.endswith("/lean-audit.json") for p in changed(repo, base, commit))


def auth_env(auth: tuple[str, ...]) -> dict[str, str]:
    """The queue's `git -c` options (`credential_args`) as GIT_CONFIG_* variables, after any the caller carries, so every git a
    child runs in the queue's clone authenticates as the queue's own does."""
    pairs = [o.split("=", 1) for flag, o in zip(auth[::2], auth[1::2]) if flag == "-c"]
    n = int(os.environ.get("GIT_CONFIG_COUNT") or 0)
    env = {"GIT_CONFIG_COUNT": str(n + len(pairs))}
    for i, (k, v) in enumerate(pairs, n):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
    return env if pairs else {}


#: what train.sh's check.py passes on as `slot.py --rank`: a train's first-parent depth (`Driver._rank`, the lowest train's the
#: least), so the train that lands next takes a node's next free check slot (CI-110); the post-landing run has none, so it takes
#: a slot only where no train waits (top, 24 Oct 04:53Z: trains come first)
SLOT_RANK = "CHECK_SLOT_RANK"
#: the per-PR shadow's checks' train.sh: an ordinary job, no `--priority` or rank, behind every priority check and in turn with
#: lane checks, so it never takes a lane's or a train's slot; and quiet, posting nothing when a launch registers nothing (ci, 21 Oct)
PER_PR_ENV = {SLOT_RANK: None, "TRAIN_ORDINARY": "1", "TRAIN_QUIET": "1"}


#: this thread's `launch_env` values: a tick launches its trains at once (`LAUNCH_JOBS`), each with its own rank, so they
#: never go through os.environ, which every thread shares
_LAUNCH = threading.local()


@contextlib.contextmanager
def launch_env(values: dict[str, str | None]):
    """``values`` set (a None unset) over os.environ for the train.sh launches this thread makes inside (`launch_vars`)."""
    was = getattr(_LAUNCH, "values", {})
    _LAUNCH.values = {**was, **values}
    try:
        yield
    finally:
        _LAUNCH.values = was


def launch_vars() -> dict[str, str]:
    """The environment a launch's train.sh gets: os.environ with this thread's `launch_env` values."""
    env = dict(os.environ)
    for k, v in getattr(_LAUNCH, "values", {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def slot_rank(rank: int | None):
    """SLOT_RANK set to ``rank`` (unset when None) for this thread's launches (`launch_env`)."""
    return launch_env({SLOT_RANK: None if rank is None else str(rank)})


@contextlib.contextmanager
def authed(auth: tuple[str, ...]):
    """os.environ with the queue's credentials (`auth_env`) and git never prompting, for the steps after a landing
    (`merge.caught_up`, `_sweep_after`), whose git children read os.environ (12 Oct 08:49Z: `caught_up`'s `ls-remote` in the
    queue's clone waited on a password at the tty and held drive for 13 min)."""
    extra = {**auth_env(auth), "GIT_TERMINAL_PROMPT": "0"}
    was = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        yield
    finally:
        for k, v in was.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def train_launch(repo: Path, commit: str, machine: str, agreement: bool, auth: tuple[str, ...] = (),
                 deferred: str | None = None, no_cache: bool = False, no_regenerate: bool = False,
                 detached: bool = False) -> tuple[list[str], str] | dict:
    """`train.sh COMMIT MACHINE [--agreement]` from the queue's clone: (the run ids it printed, its output's tail); ``detached``,
    started in the background (`detach`) from a copy of train.sh, which reads nothing by its own path (``$0`` only for its
    usage) and finds the tree from its working directory, and returned at once as the record `launched` reads.  Git never
    prompts (12 Oct: train.sh's fetch in the queue's clone, which has no credential helper, waited on a password at the tty)."""
    argv = ["bash", "tools/verity/check/train.sh", commit, machine, *(["--agreement"] if agreement else []),
            *(["--deferred", deferred] if deferred else []), *(["--no-cache"] if no_cache else []),
            *(["--no-regenerate"] if no_regenerate else [])]
    env = {**launch_vars(), **auth_env(auth), "GIT_TERMINAL_PROMPT": "0", "TRAIN_AS": "ci"}
    if detached:
        return detach(repo, argv, env, script=argv[1])
    r = subprocess.run(argv, cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    m = RUN_RE.search(r.stdout)
    return (m[1].split() if m else []), (r.stdout + r.stderr).strip()[-600:]


#: where detached launches (`detach`) keep their output and exit, under drive's TMPDIR
LAUNCHES = "drive-launches"
#: how long a detached launch may print no run ids before drive takes it as having registered none: train.sh stops its launcher
#: 30 min into its wait for one (`launch_wait_s`), and a restart of the pod can leave its pid to another process
LAUNCH_LOST_S = 3600
#: a detached launch's job: the command's output to `log` in its directory, then its exit status to `rc`
DETACH_JOB = 'd=$1; shift; "$@" > "$d/log" 2>&1; echo "$?" > "$d/rc.tmp" && mv "$d/rc.tmp" "$d/rc"'


def detach(repo: Path, argv: list[str], env: dict[str, str], script: str | None = None) -> dict:
    """``argv`` in ``repo``, in a session of its own, so no tick waits on it (26 Oct: train.sh took 9-12 min to print a Lean
    check's run ids, and a tick that launched two took 15-26 min): its output and exit go to a directory of their own under
    `LAUNCHES`.  ``script``, a path of ``repo`` in ``argv``, runs from a copy there: bash reads a script as it runs, and a later
    tick resets and checks out ``repo`` (infra, 26 Oct 09:54Z).  Returns the record `launched` reads."""
    root = Path(tempfile.gettempdir()) / LAUNCHES
    root.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(prefix="train-", dir=root)
    if script:
        copy = shutil.copy(repo / script, Path(d) / Path(script).name)
        argv = [str(copy) if a == script else a for a in argv]
    p = subprocess.Popen(["sh", "-c", DETACH_JOB, "sh", d, *argv], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"dir": d, "pid": p.pid}


def _ids(rec: dict) -> str:
    """A launch record's run ids, for a message, while its train.sh runs detached too."""
    return " ".join(rec.get("ids") or ()) or ("(launching)" if rec.get("launch") else "(no run)")


def launched(launch: dict) -> tuple[list[str] | None, str]:
    """(the run ids a detached train.sh (`detach`) printed, [] once it exited without them or its process is gone without an
    exit, None while it runs; its output's tail)."""
    d = Path(launch["dir"])
    try:
        out = (d / "log").read_text(errors="replace")
    except OSError:
        out = ""
    tail = out.strip()[-600:]
    if m := RUN_RE.search(out[: out.rfind("\n") + 1]):          # a whole line: train.sh may be writing it
        return m[1].split(), tail
    if (d / "rc").is_file():
        return [], tail
    try:
        os.kill(int(launch["pid"]), 0)
    except ProcessLookupError:
        return [], f"{tail} (its process is gone with no exit written: a restart of the pod)".strip()
    except (PermissionError, ValueError, TypeError, KeyError):
        pass
    return None, tail


def touched_launch(repo: Path, commit: str, base: str, auth: tuple[str, ...] = (),
                   how: RemoteSuites | None = None) -> tuple[list[str], str]:
    """A train's touched-suites run (`queue.touched_suites` on a node, `RemoteSuites`: `research run --on auto`, outside the check
    slots) of ``commit`` against ``base``, shipped from a temporary worktree of the queue's clone: (its run ids, its output's
    tail).  It runs as lane ci, with the queue's credentials, and git never prompts."""
    how = how or RemoteSuites(env={**auth_env(auth), "RESEARCH_LANE": "ci"})
    t = touched_suites(repo, commit, base, how)
    return t.ids, (t.tail or "; ".join(t.notes))[-600:]


#: what a running check has written of its steps and suites so far (check.py's `STEPS_LOG`, suites.py's `failing_path` of each
#: `--json`), fetched with its records
PROGRESS = ("steps.jsonl", "suites.failed.json", "suites-lean.failed.json")


def failing(run_dir: Path) -> str | None:
    """What has failed so far in a check still running under `--keep-going`, which fails it whatever else it finds, or None."""
    out = []
    try:
        lines = (run_dir / PROGRESS[0]).read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("status") == "failed":
            out.append(f"step {rec.get('step')}")
    for name in PROGRESS[1:]:
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError):
            continue
        for s in doc if isinstance(doc, list) else ():
            tests = s.get("failed_tests") or []
            out.append(f"suite {s.get('name')}" + (f" ({', '.join(tests[:2])}{' ...' if len(tests) > 2 else ''})" if tests else ""))
    return "; ".join(out) or None


def run_states(repo: Path, ids: list[str]) -> dict[str, tuple]:
    """Each run's (state, rc, ended_utc, what has failed so far while it runs: `failing`, the machine it runs on) from `research
    status RUN --json`, which fetches `PROGRESS` with its records (a check waiting in its machine's slot line reads `waiting`); a run
    it can't read is ("unknown", None, None, None)."""
    also = [x for n in PROGRESS for x in ("--also", n)]

    def one(rid: str) -> tuple:
        r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json", *also], cwd=repo, capture_output=True, text=True)
        try:
            run = json.loads(r.stdout)["runs"][0]
            state = run.get("state") or "unknown"
            return (state, run.get("rc"), run.get("ended_utc"),
                    failing(Path(run["dir"])) if state in ACTIVE and run.get("dir") else None, (run.get("remote") or {}).get("machine"))
        except (ValueError, KeyError, IndexError):
            return ("unknown", None, None, None)

    return dict(zip(ids, _each(one, ids)))


#: how far back `lean_runs` reads drive's run registry: a Lean check waits in a slot line and runs well inside it
LEAN_RUN_WINDOW_S = 12 * 3600
RUN_ID_RE = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")


def lean_runs(commits: set[str], since: float, root: Path | None = None) -> dict[str, list[str]]:
    """{commit: [run id, ...]} of the runs of this machine's run registry launched since ``since`` that check one of ``commits``
    with lean-audit among their steps (`check.py` with lean-audit in its `--deferred`, or a whole check), and that haven't ended
    by their local records: a hand `train.sh COMMIT MACHINES --deferred lean-audit` on the control pod registers here as drive's
    own launches do (26 Oct: ci's two at 7d11a5e2 and 2823fe1e).  The store has no such run before it ends (its README, §10).
    Reads one listing and the records of the runs it names since ``since``."""
    from . import runs as R

    root = R.runs_root(root)
    cut = time.strftime("r%Y%m%d-%H%M%S", time.gmtime(since))
    out: dict[str, list[str]] = {}
    for d in sorted(root.iterdir()) if root.is_dir() else ():
        if not RUN_ID_RE.match(d.name) or d.name < cut:
            continue
        rec = R._read(d / "remote.json") or {}
        job = R._read(d / "job.json") or {}
        cmd = [str(x) for x in rec.get("command") or (job.get("execution") or {}).get("command") or ()]
        commit = rec.get("source_sha") or (job.get("computation") or {}).get("tree_sha")
        if commit not in commits or not any(x.endswith("check/check.py") for x in cmd) or "--shard" in cmd:
            continue
        steps = cmd[cmd.index("--deferred") + 1].split(",") if "--deferred" in cmd[:-1] else [LEAN_STEP]
        if LEAN_STEP in steps and R.summarize(d)["state"] in R.ACTIVE:
            out.setdefault(commit, []).append(d.name)
    return out


#: how many runs drive reads at once: each `research status` is an ssh round trip to the run's node (20 Oct: the control pod's
#: first tick took 33 min reading about 40 runs one at a time)
STATUS_JOBS = 8
#: how many trains' checks a tick launches at once: each train.sh waits minutes for check.py to gather verdicts and launch a
#: run per twin (21 Oct 06:08Z: a tick launched seven trains one after another, in 40 min, and landed nothing meanwhile); the
#: control pod has 2 cores
LAUNCH_JOBS = 4


def _each(fn, items: list, jobs: int = STATUS_JOBS) -> list:
    """``fn`` of each of ``items``, ``jobs`` at a time, in their order."""
    if len(items) < 2:
        return [fn(x) for x in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items))


def cancel_runs(repo: Path, ids: list[str], reason: str) -> dict[str, str]:
    """`research cancel` of each of ``ids``: {run id: why} of each cancel that failed (21 Oct: drive's cancels of a rebuilt
    train's queued touched-suites runs failed unseen, and 12 of them held vy-nebius-2's agent-test line for hours; CI-146)."""
    failed = {}
    for rid in ids:
        r = subprocess.run([sys.executable, "-m", "research", "cancel", rid, "--by", "ci", "--reason", reason], cwd=repo,
                           capture_output=True, text=True)
        if r.returncode:
            failed[rid] = (r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}"
    return failed


def free_slots(repo: Path, on: str) -> dict[str, int]:
    """Each machine of ``on`` (`auto`, or `M1,M2`) with the train checks it could start now (check.py's `spare` of its
    `read_slots`, with the queue's clone's check.py, as train.sh's `--on` places them), in ``on``'s order."""
    import importlib.util

    from .store import tools_registry

    path = tools_registry.resolve("tools/verity/check/check.py", lambda p: (repo / p).is_file()) or "tools/verity/check/check.py"
    s = importlib.util.spec_from_file_location("drive_check", repo / path)
    check = importlib.util.module_from_spec(s)
    sys.modules[s.name] = check
    s.loader.exec_module(check)
    names = list(check.AUTO) if on == "auto" else [n for n in on.split(",") if n]
    docs = check.read_slots(names)
    return {n: check.spare(docs[n], priority=True) if isinstance(docs.get(n), dict) else 0 for n in names}


def merge_thread_post(text: str, to: tuple[str, ...] = (CI,), sender: str = "ci") -> bool:
    """Post ``text`` to the merge thread as ``sender``, addressed to ``to`` but not the sender, and with the sender dropped from
    the names ``text`` starts with: `research msg send` refuses a `--to` naming the sender (13 Oct: from 05:07Z every notice
    drive sent named @ci and none was posted) and a post that starts with its sender's name alone (17:41Z: b22a's failure)."""
    others = [t for t in to if t != f"@{sender}"]
    if lead := re.match(r"(?:@[\w.-]+ )+", text):
        text = " ".join([*(n for n in lead[0].split() if n != f"@{sender}"), text[lead.end():]])
    r = subprocess.run([sys.executable, "-m", "research", "msg", "send", "--as", sender, "--thread", _merge_thread(),
                        *(["--to", *others] if others else []), "--quiet", "--text", text], capture_output=True, text=True)
    return r.returncode == 0


def _merge_thread() -> str:
    from .monitors import MERGE_THREAD

    return MERGE_THREAD


def _utc(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return time.mktime(time.strptime(ts.rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except ValueError:
        return None



def _docs(run_dir: Path, *names: str) -> list[dict]:
    out = []
    for name in names:
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError):
            continue
        out += [x for x in doc if isinstance(x, dict)] if isinstance(doc, list) else []
    return out


def failed_suites(run_dir: Path) -> list[dict]:
    """Each failed suite's record: from `suites.json` and `suites-lean.json` once the check ended, else from what it has written
    of its failures so far (`PROGRESS`)."""
    done = [s for s in _docs(run_dir, "suites.json", "suites-lean.json", TOUCHED_REPORT) if s.get("status") == "failed"]
    return done or _docs(run_dir, *PROGRESS[1:])


def touched_report(run_dir: Path) -> list[dict] | None:
    """A touched-suites run's `TOUCHED_REPORT` (suites.py's `--json`, which it writes whenever it ends), None when it isn't
    there or doesn't parse."""
    try:
        doc = json.loads((run_dir / TOUCHED_REPORT).read_text())
    except (OSError, ValueError):
        return None
    return [x for x in doc if isinstance(x, dict)] if isinstance(doc, list) else None


def _fetch(repo: Path, rid: str, names: list[str]) -> Path | None:
    from .merge import fetch_run

    got = fetch_run(rid, names, repo)
    return got[0] if got else None


def evidence(repo: Path, ids: list[str]) -> list[Path]:
    """The local run dirs of ``ids`` with their steps, suite reports and failed suites' logs fetched (`research status --also`)."""
    def one(rid: str) -> Path | None:
        if (d := _fetch(repo, rid, [*PROGRESS, "suites.json", "suites-lean.json", TOUCHED_REPORT])) is None:
            return None
        logs = [f"{where}/{Path(s['log']).name}" for s in failed_suites(d) if s.get("log") for where in ("suites", TOUCHED_LOGS)]
        logs += [Path(st["log"]).name for st in failed_steps(d) if st.get("log") and st.get("step") not in ("pytest", "lean-suites")]
        if logs:
            _fetch(repo, rid, logs)
        return d

    return [d for d in _each(one, ids) if d is not None]


def failed_steps(run_dir: Path) -> list[dict]:
    out = []
    with contextlib.suppress(OSError):
        for line in (run_dir / PROGRESS[0]).read_text().splitlines():
            with contextlib.suppress(ValueError):
                if (rec := json.loads(line)).get("status") == "failed":
                    out.append(rec)
    return out


def suite_inputs(repo: Path, commit: str, suite: str) -> list[str]:
    """The paths suite ``suite`` (its directory) reads at ``commit``: its directory and its `[tool.verity.tests] inputs`; the
    repository's own suite (``.``) only its inputs."""
    import tomllib

    own = "pyproject.toml" if suite.strip("/") in ("", ".") else f"{suite.strip('/')}/pyproject.toml"
    r = _git(repo, "show", f"{commit}:{own}", check=False)
    try:
        inputs = tomllib.loads(r.stdout).get("tool", {}).get("verity", {}).get("tests", {}).get("inputs", []) if not r.returncode else []
    except tomllib.TOMLDecodeError:
        inputs = []
    return [i for i in ([] if suite in ("", ".") else [suite]) + [str(x) for x in inputs] if i]


def under(f: str, paths: list[str]) -> bool:
    return any(f == p or f.startswith(p.rstrip("/") + "/") for p in paths)


def lean_errors(log: str) -> set[str]:
    """The repository paths of the Lean files lean-audit's ``log`` reports errors in: each `error: FILE.lean:LINE`, each
    `orphan: FILE.lean` and the module of each `layers: MODULE imports ...` after an `AUDIT .../PKG: FAIL` line, FILE relative to
    PKG."""
    out, pkg, failures = set(), None, False
    for line in log.splitlines():
        if m := LEAN_AUDIT_RE.match(line):
            pkg, failures = m[1], False
        elif pkg and ((m := LEAN_ERROR_RE.match(line)) or (m := LEAN_ORPHAN_RE.match(line))):
            out.add(os.path.normpath(os.path.join(pkg, m[1])))
        elif pkg and (m := LEAN_LAYERS_RE.match(line)):
            parts = m[1].split(".")
            root = os.path.dirname(pkg) if parts[0] == os.path.basename(pkg) else pkg
            out.add(os.path.normpath(os.path.join(root, *parts) + ".lean"))
        elif pkg and "Some required targets logged failures:" in line:
            failures = True                                 # lake's own list: the step's log keeps it when the errors scroll by
        elif pkg and failures and (m := re.match(r"^\s*- ([A-Za-z][\w.]*)\s*$", line)):
            parts = m[1].split(".")                         # a module of a package rooted at its own name sits beside it
            root = os.path.dirname(pkg) if parts[0] == os.path.basename(pkg) else pkg
            out.add(os.path.normpath(os.path.join(root, *parts) + ".lean"))
        elif failures and not line.strip().startswith("-"):
            failures = False
    return out


def killed(run_dir: Path, why: str = "") -> bool:
    """Whether a check failed by being killed rather than by what it checked: a failed step that exited by a signal or 128+N,
    or the run's own exit (``why``, the verdict's `rc=N`) one of those."""
    steps = failed_steps(run_dir)
    if any(isinstance(rc, int) and (rc < 0 or rc in KILLED_RC) for rc in (st.get("rc") for st in steps)):
        return True
    for st in steps:                                     # a step that says one of its own processes was killed (17 Oct: the
        if st.get("log"):                                # Lean check's kernel replay, exit 137 on cpu-2, ejected #2224, #2231)
            with contextlib.suppress(OSError):
                if KILLED_SAYS in (run_dir / Path(st["log"]).name).read_text(errors="replace"):
                    return True
    # circuit-check alone failed, and only on targets whose process died (19 Oct: three PoUS SHA-512 targets OOM-killed at
    # 149-164 GiB on cpu-1 in 399ea461's post-landing run, and drive tried to revert all 19 landings since 6c4f254d)
    if len(steps) == 1 and steps[0].get("step") == "circuit-check" and steps[0].get("log"):
        with contextlib.suppress(OSError):
            fails = CIRCUIT_FAIL_RE.findall((run_dir / Path(steps[0]["log"]).name).read_text(errors="replace"))
            if fails and all(f == CIRCUIT_KILLED for f in fails):
                return True
    return any(f"rc={rc}" in why for rc in (*KILLED_RC, -9, -15))


def _named(log: str, base: str) -> set[str]:
    """The files a pytest log's failures name: its tracebacks' frames, and the repository paths in its failure report."""
    rel = lambda p: p if p.startswith("/") else os.path.normpath(os.path.join(base, p))  # noqa: E731
    paths = {rel(m[1] or m[2]) for m in TRACE_RE.finditer(log)}
    report = log.partition("= FAILURES =")[2].partition("short test summary info")[0]
    return paths | {rel(m[1]) for m in NAMED_RE.finditer(report)}


def implicated(run_dir: Path) -> set[str] | None:
    """The paths a failed check's failure points at: each failed suite's failing test files and the files its log's failures
    name (`_named`: relative to the repository, or absolute as the log wrote them), and the same of a failed step that runs
    pytest on the repository's own tests (`preflight-lints`). A suite the read guard failed with no failing test names the
    files its tests (or their children, `child_outside`) read outside its inputs and the test modules that read them (its
    record's `outside`, path -> module), so the PR that added the reading test, or changed the file read, is implicated.
    None when another step failed, or a failed suite or step names no failing test and no outside read (a timeout, a kill),
    so no file says what broke it."""
    paths: set[str] = set()
    for st in failed_steps(run_dir):
        if st.get("step") in ("pytest", "lean-suites"):
            continue
        try:
            log = (run_dir / Path(st.get("log") or "").name).read_text(errors="replace") if st.get("log") else ""
        except OSError:
            log = ""
        if st.get("step") == "lean-audit" and (lean := lean_errors(log)):
            paths |= lean
            continue
        if not (tests := FAILED_RE.findall(log)):
            return None
        paths |= {os.path.normpath(t) for t in tests} | _named(log, "")
    suites = failed_suites(run_dir)
    if not suites and not paths:
        return None
    for s in suites:
        base, tests = s.get("suite") or "", s.get("failed_tests") or []
        log = ""
        if s.get("log"):
            with contextlib.suppress(OSError):
                log = next((f.read_text(errors="replace") for where in ("suites", TOUCHED_LOGS)
                            if (f := run_dir / where / Path(s["log"]).name).is_file()), "")
        if not tests and (errors := ERROR_RE.findall(log)):     # 17 Oct: a collection error (#2299's test of a vector only
            tests = errors                                       # #2286 adds) named no file, and 8 PRs were ejected by inputs
        if not tests:
            out, child = s.get("outside") or {}, s.get("child_outside") or {}
            modules = list(out.values() if isinstance(out, dict) else ()) + list(child)
            files = list(out) + [f for fs in child.values() for f in fs]
            if not files:
                return None
            paths |= {os.path.normpath(p) for p in files} | {os.path.normpath(m) for m in modules if m and m != "*"}
            continue
        paths |= {os.path.normpath(os.path.join(base, t.split("::")[0])) for t in tests}
        if log:
            paths |= _named(log, base)
    return paths


def touches(files: set[str], paths: set[str]) -> set[str]:
    """The ``files`` (repository paths) among ``paths``, an absolute path matching the file it ends in when that file is under a
    directory (a bare `conftest.py` would match every conftest an installed package has)."""
    return {f for f in files for p in paths if p == f or ("/" in f and p.startswith("/") and p.endswith("/" + f))}


#: the plan's job (`plan_launch`): `check.py --plan` on each machine in turn, each plan to its own file, stopping at the first
#: fully passed; its exit in `rc` last, once the worktree it ran in is gone
PLAN_JOB = ('i=0; rc=2; for m in $PLAN_ON; do "$@" --on "$m" > "$PLAN_OUT/plan-$i.json" 2>> "$PLAN_OUT/plan.log"; rc=$?; '
            'i=$((i + 1)); [ "$rc" -eq 0 ] && break; done; cd / && git -C "$PLAN_REPO" worktree remove --force "$PLAN_TREE" '
            '>> "$PLAN_OUT/plan.log" 2>&1; echo "$rc" > "$PLAN_OUT/rc.tmp" && mv "$PLAN_OUT/rc.tmp" "$PLAN_OUT/rc"')


def plan_launch(repo: Path, commit: str, base: str, runs: list[str], machines: list[str], out: Path,
                auth: tuple[str, ...] = ()) -> dict:
    """Plan ``commit`` by its parts' keys, detached, so no tick waits on it: in a worktree of ``base`` (`main`, whose check.py is
    the gate's) under ``out``, `merge.PLAN COMMIT --json --packs-only --base BASE --from RUN...` (``runs``: the PR's own checks)
    `--on` each of ``machines`` in turn (`PLAN_JOB`).  Returns the record `plan_state` reads."""
    from .merge import PLAN

    out.mkdir(parents=True, exist_ok=True)
    tree = out / "tree"
    _git(repo, "worktree", "add", "-q", "--detach", str(tree), base)
    argv = [*PLAN, commit, "--json", "--packs-only", "--base", base, *(x for r in runs for x in ("--from", r))]
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_RUN_RECURSION_DEPTH")}
    env.update(auth_env(auth), GIT_TERMINAL_PROMPT="0", PLAN_ON=" ".join(machines), PLAN_OUT=str(out), PLAN_REPO=str(repo),
               PLAN_TREE=str(tree))
    p = subprocess.Popen(["sh", "-c", PLAN_JOB, "sh", *argv], cwd=tree, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"dir": str(out), "pid": p.pid, "machines": machines}


def plan_state(plan: dict) -> tuple[str, list[dict], str]:
    """(running | ended | lost, the plans it has written, its log's tail) of a `plan_launch`: lost when its process is gone
    without writing its exit (a restart of the pod)."""
    d = Path(plan["dir"])
    docs = []
    for f in sorted(d.glob("plan-*.json")):
        with contextlib.suppress(OSError, ValueError):
            if isinstance(doc := json.loads(f.read_text()), dict):
                docs.append(doc)
    try:
        tail = (d / "plan.log").read_text(errors="replace").strip()[-400:]
    except OSError:
        tail = ""
    if (d / "rc").is_file():
        return "ended", docs, tail
    try:
        os.kill(int(plan["pid"]), 0)
    except ProcessLookupError:
        return "lost", docs, tail
    except (PermissionError, ValueError, TypeError, KeyError):
        pass
    return "running", docs, tail


def undeferred(doc: dict, defer: frozenset[str]) -> dict:
    """A `check.py --plan` document with the parts of the ``defer`` steps taken out, as a train lands without them: a build step
    (`implied`) keeps the parts that read what it builds that are left, and goes when none is."""
    parts = [p for p in doc.get("parts") or [] if isinstance(p, dict) and p.get("step") not in defer]
    names = {p.get("part") for p in parts}
    out = []
    for p in parts:
        if p.get("state") == "implied" and isinstance(p.get("implied_by"), list):
            if not (left := [n for n in p["implied_by"] if n in names]):
                continue
            p = {**p, "implied_by": left}
        out.append(p)
    return {**doc, "parts": out}


def lean_cut(chain: list[Member], lo: int, top: int, is_lean: Callable[[Member], bool], slots: int) -> int:
    """The index in ``chain`` to check the Lean changes (``is_lean``) of ``chain[lo:top + 1]`` up to, when ``slots`` Lean checks
    may share them: just below the Lean change after the first ceil(n / ``slots``) of its n, or ``top``.  Spaced cut points
    each hold only what's above them when one fails, and the lower ones land without waiting for the rest."""
    at = [j for j in range(lo, top + 1) if is_lean(chain[j])]
    per = math.ceil(len(at) / max(slots, 1)) if at else 0
    return at[per] - 1 if 0 < per < len(at) else top


def lean_stack(chain: list[Member], lean: dict, is_lean: Callable[[Member], bool], churned: Callable[[Member], bool],
               most: int) -> int | None:
    """The index in ``chain`` (`next`'s members) of the commit to stack a review-landing Lean check on, or None: `next`'s top,
    capped below the lowest member above the checks of ``lean`` (`drive["lean"]`) that came onto `next` just now
    (``churned``), and cut to share the Lean changes up to it with the checks still free (`lean_cut`), when a Lean change
    (``is_lean``) at or below it is above every check of `next` that runs or passed, one runs (only lower ones, then), none
    failed, and fewer than ``most`` run.  25 Oct: members that came above a running check's commit waited for it to pass, then
    for a whole check of their own."""
    pos = {m.commit: i for i, m in enumerate(chain)}
    on = {pos[c]: rec.get("state") for c, rec in lean.items() if c in pos}
    running = sum(rec.get("state") is None for rec in lean.values())
    if None not in on.values() or "failed" in on.values() or running >= most:
        return None
    hi = max(on)
    top = next((j for j in range(hi + 1, len(chain)) if churned(chain[j])), len(chain)) - 1
    return lean_cut(chain, hi + 1, top, is_lean, most - running) if any(is_lean(m) for m in chain[hi + 1:top + 1]) else None


class Driver:
    def __init__(self, q: Queue, *, on: str, max_running: int = 2, train_size: int = 4,
                 launch: Callable[[Path, str, str, bool], tuple[list[str], str]] = train_launch,
                 states: Callable[[Path, list[str]], dict] = run_states, post: Callable[[str, tuple[str, ...]], bool] = merge_thread_post,
                 now: Callable[[], float] = time.time, cancel: Callable[[Path, list[str], str], dict[str, str] | None] = cancel_runs,
                 evidence: Callable[[Path, list[str]], list[Path]] = evidence, defer: frozenset[str] = frozenset(),
                 regenerate_locks: bool = False, per_pr_shadow: int = 0,
                 slots: Callable[[Path, str], dict[str, int]] = free_slots, land: str = "check",
                 after_landing_no_cache: bool = False, reviews: Callable[[int, str], list] | None = None,
                 ready_by: Callable[[int, str], str | None] | None = None, max_lean_checks: int = 2):
        self.q, self.on, self.max_running, self.train_size = q, on, max_running, train_size
        #: the review landing's Lean checks that may run at once, a lower one and those stacked above it (`lean_stack`)
        self.max_lean_checks = max_lean_checks
        #: `land = "review"` (Daniel, 24 Oct 06:05Z): each member lands on a reviewer's verdict, a Lean change once a Lean check
        #: passed, and the full check runs after landing (`_review`), reusing no pass with ``after_landing_no_cache``
        self.review, self.after_landing_no_cache = land == "review", after_landing_no_cache
        #: the `reviewed` labels on a PR's head (`StoreLabels.reviews`)
        self.reviews: Callable[[int, str], list] = reviews or (lambda pr, head: StoreLabels(q.store, q.log).reviews(pr, head))
        #: who wrote the ready mark on a PR's head (`StoreLabels.ready_by`), one of those whose verdict doesn't count (`_owners`)
        self.ready_by: Callable[[int, str], str | None] = ready_by or (lambda pr, head: StoreLabels(q.store, q.log).ready_by(pr, head))
        #: each machine of `on` with the train checks it could start now (`free_slots`), read at most once a tick (`_spare`)
        self.slots, self._free = slots, None
        #: the per-PR landing in shadow (`per_pr`): at most this many of its checks, and of its plans, at once; 0 is off
        self.per_pr_shadow = per_pr_shadow
        self.plan_launch: Callable[..., dict] = functools.partial(plan_launch, auth=q.auth)
        self.plan_state: Callable[[dict], tuple[str, list[dict], str]] = plan_state
        #: commit the locks a train's Lean check regenerated onto it as it lands (CI-64: Daniel through top, 18 Oct)
        self.regenerate_locks = regenerate_locks
        #: steps a train lands without (Daniel through top, 17 Oct; CI-30): the post-landing run checks them on `main`
        #: (`postland`), and one that fails reverts the PR it implicates, else every landing since the last verified tip (none
        #: when no landing touches what failed).  The Lean check's steps are never deferred; lean-suites, the pytest suites that use
        #: its build, may be (Daniel, 11 Oct: a part whose cold run is over ~10 min; lean-suites takes 28-65 min, top 24 Oct 04:46Z).
        defer = {x for x in defer if not x.startswith("lean") or x == "lean-suites"}
        self.defer = frozenset(defer | ({"flock-rust-tests"} if "circuit-check" in defer else set()))  # it runs after it
        #: the post-landing run of the deferred steps on `main`'s tip (`postland`): `train.sh COMMIT MACHINE --deferred STEPS`,
        #: detached (`_started`); an injected `launch` launches it too
        self.launch_postland: Callable[..., tuple[list[str], str] | dict] = (
            (lambda repo, commit, machine, steps, agreement=False, no_cache=False, no_regenerate=False:
             train_launch(repo, commit, machine, agreement, auth=q.auth, deferred=steps, no_cache=no_cache,
                          no_regenerate=no_regenerate, detached=True))
            if launch is train_launch else (lambda repo, commit, machine, steps, **_: self.launch(repo, commit, machine, False)))
        #: a Lean change's Lean check before a review landing: `train.sh TOP MACHINES --deferred lean-audit`, detached
        self.launch_lean: Callable[[Path, str, str], tuple[list[str], str] | dict] = (
            (lambda repo, commit, machine: train_launch(repo, commit, machine, False, auth=q.auth, deferred=LEAN_STEP,
                                                        detached=True))
            if launch is train_launch else (lambda repo, commit, machine: self.launch(repo, commit, machine, False)))
        #: what a detached launch has printed so far (`launched`, `_resolve`)
        self.launched: Callable[[dict], tuple[list[str] | None, str]] = launched
        #: the running Lean checks of drive's run registry, its own and hand ones (`lean_runs`), which hold `next` as its own do
        self.lean_runs: Callable[[set[str], float], dict[str, list[str]]] = lean_runs
        if launch is train_launch:
            launch = functools.partial(train_launch, auth=q.auth)
        #: with pytest deferred, the suites a train's change reaches run before it lands (CI-57: #2271 and #2382 each reverted
        #: landings a 2-minute run of one suite would have caught): `touched_launch`, on a node outside the check slots, so
        #: ``machine`` (the trains') isn't its
        self.launch_touched: Callable[[Path, str, str, str], tuple[list[str], str]] = (
            lambda repo, commit, machine, base: touched_launch(repo, commit, base, q.auth))
        self.launch, self.states, self.post, self.now, self.cancel, self.evidence = launch, states, post, now, cancel, evidence
        self.repo, self.dry_run, self._seen, self._prs, self._ev, self._relaunched = q.workdir, False, {}, None, {}, set()
        self._drive: dict | None = None
        #: the commits of `next` that `trains` checks, and each member's place on it, once this tick's sync has rebuilt it (None,
        #: {} before)
        self._tips: set[str] | None = None
        self._pos: dict[str, int] = {}
        q.warming = self.warming
        q.holding = lambda state: None          # `tick` asks `held`, with its verdicts, before it syncs

    # -- what the store says of a commit

    def _states(self, ids: list[str]) -> dict:
        """`states` of ``ids``, each run read once a tick: one `research status` is a few seconds over ssh, and a tick asks of the
        same launch for `held`, before the sync, for the running count and for its train (12 Oct: 25 s of a 52 s tick)."""
        if missing := [rid for rid in ids if rid not in self._seen]:
            self._seen.update(self.states(self.repo, missing))
        return {rid: self._seen[rid] for rid in ids}

    @staticmethod
    def _started(got: tuple[list[str], str] | dict) -> tuple[dict | None, str]:
        """A launch record's fields from what `launch_lean` or `launch_postland` returned: {"ids"} of a launch that printed
        them, {"ids": [], "launch"} of one still running detached (`detach`), which `_resolve` reads on later ticks and which
        counts as running until then; None, with train.sh's output's tail, of one that registered no run."""
        if isinstance(got, dict):
            return {"ids": [], "launch": got}, ""
        ids, tail = got
        return ({"ids": ids} if ids else None), tail

    def _resolve(self, rec: dict, what: str) -> str | None:
        """Read a detached launch's run ids into ``rec`` (`launched`) once train.sh has printed them, its ``at`` then.  None
        while it runs (``rec["launch"]`` stays) or once ``rec`` has its ids; else, its ``launch`` dropped, the tail of a train.sh
        that registered no run, or printed none within `LAUNCH_LOST_S`."""
        if not (launch := rec.get("launch")) or self.dry_run or self.q.shadow:
            return None
        ids, tail = self.launched(launch)
        if ids is None and self.now() - rec.get("at", self.now()) <= LAUNCH_LOST_S:
            return None
        del rec["launch"]
        if not ids:
            return (tail if ids is not None else f"no run ids {LAUNCH_LOST_S // 60} min after its launch: {tail}") + \
                f" (log {launch['dir']}/log)"
        rec.update(ids=ids, at=self.now())
        shutil.rmtree(launch["dir"], ignore_errors=True)
        self.q.log(f"research queue drive: {what}: {' '.join(ids)}")
        return None

    def _evidence(self, v: Verdict) -> list[Path]:
        """`evidence` of a failed verdict, fetched once a tick: `failed` reads it before the sync and again for the trains, and a
        bisection can wait hours on a check slot."""
        if v.attempt not in self._ev:
            self._ev[v.attempt] = self.evidence(self.repo, v.attempt.split())
        return self._ev[v.attempt]

    def verdict(self, commit: str, base: str, drive: dict) -> Verdict:
        """`commit`'s check: passed (the gate's rule, against `base`), failed, running (a launch of ours not yet in the store),
        or none (never launched, or every twin of the last launch exited without starting)."""
        from .commitmap import Maps
        from .merge import declared_command, passing, produced, refused, required_steps, step_status

        store = self.q.store
        tree = _out(self.repo, "rev-parse", f"{commit}^{{tree}}")
        kinds, command = produced(self.repo, "check"), declared_command(self.repo, "check")
        ids = passing(store, commit, "check", kinds=kinds, command=command, tree=tree, maps=Maps.load(store), repo=self.repo)
        need = required_steps(self.repo, base, commit, "check")
        for rid in ids:
            if all(step_status(store, rid, s) == "passed" for s in need):
                return Verdict("passed", rid)
        rows = store.index.sql("SELECT id, state, rc FROM attempts WHERE tool = 'check' AND source_commit = ?", (commit,))
        old = set(drive.get("resent", {}).get(commit, ()))
        # a cancel or a SIGTERM is a decision, not a verdict: a twin a sibling won the slot from, or a run someone stopped
        # (17 Oct 11:47Z: bd74745daa5f read as failed from twin 30ce, SIGTERMed unstarted, while twin 4af6 ran on)
        ran = [(rid, st, rc) for rid, st, rc in rows
               if rc not in (NOT_STARTED, SIGTERM_RC) and st != "cancelled" and rid not in old]
        if ids:
            return Verdict("failed", ids[0], f"{ids[0]} passed without {', '.join(need)}, which the gate needs for "
                                            f"{', '.join(sorted({p for ps in need.values() for p in ps}))}")
        if ran:
            rid, st, rc = ran[-1]
            why = refused(store, rid, kinds, "check", commit, command, self.repo) if st == "done" and rc == 0 else f"rc={rc}"
            return Verdict("failed", rid, f"{rid} {st} {why}")
        launch = drive["launched"].get(commit) or drive.get("base", {}).get(commit)
        if not launch:
            return Verdict("none")
        states = self._states(launch["ids"])
        if early := next((s[3] for s in states.values() if len(s) > 3 and s[0] in ACTIVE and s[3]), None):
            launch["failing"] = early
            return Verdict("failed", " ".join(launch["ids"]), f"{' '.join(launch['ids'])} is still running (`--keep-going`, for "
                                                              f"the full picture) and has already failed: {early}")
        if any(st in ACTIVE for st, *_ in states.values()):
            # `main`'s own check lands nothing, and only its deferred steps may clear it
            if self.defer and commit not in drive.get("base", {}) and (rid := self._passed_but_deferred(launch["ids"])):
                if (held := self._touched(commit, base, drive, launch)) is not None:
                    return held
                return Verdict("passed", rid, f"{DEFERRED} ({', '.join(sorted(self.defer))} run on `main` after landing)")
            return Verdict("running", " ".join(launch["ids"]))
        if all(rc in (NOT_STARTED, SIGTERM_RC) for _, rc, *_ in states.values()):
            return Verdict("none", why="every twin exited without starting, or was stopped")
        ended = max((_utc(e) or launch["at"] for _, _, e, *_ in states.values()), default=launch["at"])
        if self.now() - ended < PUBLISH_GRACE_S:
            return Verdict("running", " ".join(launch["ids"]), "ended; its attempt isn't in the store yet")
        return Verdict("failed", " ".join(launch["ids"]),
                       f"{' '.join(launch['ids'])} ended ({', '.join(f'{s} rc={rc}' for s, rc, *_ in states.values())}) "
                       f"and no attempt reached the store in {PUBLISH_GRACE_S // 60} min")

    def _touched(self, commit: str, base: str, drive: dict, launch: dict) -> Verdict | None:
        """What a train's touched-suites run (`launch["touched"]`) says before a deferred landing.  None, the gate passed, only
        for an exit 0 whose `TOUCHED_REPORT` reached drive with no failed row; failed for one with a failed row, or another
        exit (the report names what failed, as a check's suites do); else the landing is held ("running").  A run that never
        started (every twin 75), or a launch that registered none, runs again (`_relaunch_touched`).  A killed one (a signal or
        128+N, the cap's 143 among them; a cancel; no exit at all) runs again once beside the same check, and killed again is
        told to `CI` and stays held, for the check's own pytest to decide when it ends.  Drive's own cancels of a touched-suites
        run never read as kills: it cancels one only once no live record holds it (`resend` replaces the record first; a train
        off the chain drops it)."""
        if "pytest" not in self.defer:
            return None
        # the train's own check already passed every suite at this commit, a superset of the touched ones (19 Oct: a5b2d8dc's
        # check passed pytest in 6,566 s while its touched run crawled on vy-nebius-2's disk for 100 min)
        if launch.get("ids") and not self._unrun(launch["ids"], ["pytest"]):
            return None
        if not (ids := launch.get("touched")):
            return self._relaunch_touched(commit, base, drive, launch, "its touched suites have no run")
        states = self._states(ids)
        said = ", ".join(f"{rid} {st} rc={rc}" for rid, (st, rc, *_) in states.items())
        if any(st in ACTIVE for st, *_ in states.values()):
            return Verdict("running", " ".join(ids), "its touched suites are running")
        if ran := [rid for rid, (_, rc, *_) in states.items() if rc == 0]:
            reports = [r for d in self._evidence(Verdict("passed", " ".join(ran))) if (r := touched_report(d)) is not None]
            if any(s.get("status") == "failed" for r in reports for s in r):
                return Verdict("failed", " ".join(ran), f"its touched suites failed ({said}; {TOUCHED_REPORT} has a failed row)")
            if reports:
                return None
            ended = max((_utc(e) or launch["at"] for _, _, e, *_ in states.values()), default=launch["at"])
            if self.now() - ended >= PUBLISH_GRACE_S:
                self.refuse(drive, f"touched-report:{commit}", f"{commit[:12]} on `{self.q.branch}`: its touched-suites run "
                                                               f"({said}) exited 0, but no {TOUCHED_REPORT} of it reached drive "
                                                               f"in {PUBLISH_GRACE_S // 60} min. Held, not landed: find why")
            return Verdict("running", " ".join(ids), f"its touched suites exited 0 with no {TOUCHED_REPORT} yet: held")
        if all(rc == NOT_STARTED for _, rc, *_ in states.values()):
            return self._relaunch_touched(commit, base, drive, launch, f"its touched suites never started ({said})")
        if all(rc == NOT_STARTED or st == "cancelled" or rc is None or (isinstance(rc, int) and (rc < 0 or rc > 128))
               for st, rc, *_ in states.values()):
            if not launch.get("touched_killed"):
                return self._relaunch_touched(commit, base, drive, launch, f"its touched suites were killed ({said})", killed=True)
            self.refuse(drive, f"touched-killed:{commit}", f"{commit[:12]} on `{self.q.branch}`: its touched-suites run {said} "
                                                           f"was killed again on its relaunch (the first: "
                                                           f"{' '.join(launch['touched_killed'])}), a machine's fault. Held, "
                                                           "not landed: its check's own pytest decides when it ends")
            return Verdict("running", " ".join(ids), f"its touched suites were killed twice ({said}): held")
        return Verdict("failed", " ".join(ids), f"its touched suites failed ({said})")

    def _relaunch_touched(self, commit: str, base: str, drive: dict, launch: dict, why: str, killed: bool = False) -> Verdict:
        """Run ``commit``'s touched suites again beside its check (at most once a tick: a tick reads a verdict several times),
        recording the run a kill ended (``killed``) in `touched_killed`, and hold its landing."""
        old = launch.get("touched") or []
        held = Verdict("running", " ".join(old or launch["ids"]), f"{why}: held")
        if commit in self._relaunched:
            return held
        self._relaunched.add(commit)
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue drive: would run {commit[:12]}'s touched suites again: {why}")
            return held
        ids, out = self.launch_touched(self.repo, commit, self.on, base)
        launch["touched"] = ids
        if killed:
            launch["touched_killed"] = old
        if not ids:
            self.refuse(drive, f"touched-launch:{commit}", f"{commit[:12]} on `{self.q.branch}`: {why}, and its touched-suites "
                                                           f"run registered no run ({out[-200:]}). Held, not landed: drive "
                                                           "launches it again each tick")
            return held
        self.q.log(f"research queue drive: {commit[:12]}: {why}: its touched suites run again as {' '.join(ids)}, and its "
                   "landing waits on them")
        return Verdict("running", " ".join(ids), f"{why}: running again")

    def _passed_but_deferred(self, ids: list[str]) -> str | None:
        """The running check among ``ids`` whose every step but the deferred ones has passed (or was skipped), none failed."""
        need = [x for x in CHECK_STEPS if x not in self.defer]
        for d in self.evidence(self.repo, ids):
            got: dict[str, str] = {}
            with contextlib.suppress(OSError):
                for line in (d / PROGRESS[0]).read_text().splitlines():
                    with contextlib.suppress(ValueError):
                        rec = json.loads(line)
                        got[rec.get("step")] = rec.get("status")
            if any(v == "failed" for v in got.values()):
                continue
            if all(got.get(x) in ("passed", "skipped") for x in need):
                return d.name
        return None

    def follow_deferred(self, drive: dict) -> None:
        """Each landing recorded in `drive["deferred"]`, the rule before the post-landing run (`postland`), which records none:
        once its check ends, drop it if its deferred steps passed; else revert it on `main` (`git revert -m 1` of each merge it
        landed, newest first) and tell `CI` and the owners."""
        q = self.q
        for tip, rec in list(drive.get("deferred", {}).items()):
            states = self._states(rec["ids"])
            if any(st in ACTIVE for st, *_ in states.values()):
                continue
            failed = [st.get("step") for d in self.evidence(self.repo, rec["ids"]) for st in failed_steps(d)]
            if not failed:
                q.log(f"research queue drive: {rec['prs']}'s deferred steps passed ({' '.join(rec['ids'])})")
                del drive["deferred"][tip]
                continue
            if self.dry_run or q.shadow:
                q.log(f"research queue drive: would revert {rec['prs']} ({tip[:12]}): {', '.join(failed)} failed after landing")
                continue
            got = self._revert(_out(self.repo, "rev-list", "--first-parent", f"{rec['base']}..{tip}").split())
            if got == "refused":
                q.log(f"research queue drive: reverting {rec['prs']} lost a race to `main`'s push; the next tick tries again")
                continue
            what = {"pushed": f"{rec['prs']} reverted on `main`", "undone": f"{rec['prs']} was reverted on `main` already"}.get(got)
            what = (f"{what}: {', '.join(failed)} failed after landing ({' '.join(rec['ids'])}). Fix and re-mark") if what else (
                f"{rec['prs']} must be reverted by hand: {', '.join(failed)} failed after landing ({' '.join(rec['ids'])}), and the "
                "revert didn't apply")
            self.refuse(drive, f"deferred:{tip}", what, *(self._handle(int(n.lstrip('#'))) for n in rec["prs"].split()))
            if got != "conflict":
                del drive["deferred"][tip]

    def _undone(self, c: str) -> bool:
        """Whether a revert on `main` above ``c`` undid it (and no revert of that revert redid it): a hand revert, or drive's own
        before a restart lost its record."""
        by = {}
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--format=%H%x00%B%x1e",
                                                          f"{c}..refs/remotes/origin/main").split("\x1e"))):
            h, body = rec.split("\x00", 1)
            by.update(dict.fromkeys(REVERTS_RE.findall(body), h))
        n = 0
        while c in by:
            c, n = by[c], n + 1
        return n % 2 == 1

    def _revert(self, commits: list[str]) -> str:
        """`git revert` each of ``commits`` `main` hasn't undone already (`_undone`) on `main`, in order (`-m 1` for a merge), and
        push: "pushed", "undone" (each was undone already), "conflict" (one doesn't apply) or "refused" (the push lost a race
        with a push to `main`: the next tick tries again).  `main` stays as it was unless "pushed"."""
        _git(self.repo, "checkout", "-q", "-B", "main", "refs/remotes/origin/main")
        if not (todo := [c for c in commits if not self._undone(c)]):
            return "undone"
        for c in todo:
            merge = len(_out(self.repo, "rev-list", "--parents", "-n", "1", c).split()) > 2
            r = _git(self.repo, "revert", "--no-edit", *(["-m", "1"] if merge else []), c, check=False, env=IDENTITY)
            if r.returncode:
                _git(self.repo, "revert", "--abort", check=False)
                _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
                return "conflict"
        refused = self.q._remote(self.repo, "push", "-q", "--no-verify", "origin", "main:refs/heads/main", ok=None).returncode
        _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
        return "refused" if refused else "pushed"

    def _lock_commits(self, base: str, tip: str, reverted: list[Member], whole: bool) -> list[str]:
        """The `Lock records regenerated for #N...` commits in ``base``..``tip`` (`regenerate_locks`) that go with ``reverted``: one
        naming any of them when the whole range is reverted (``whole``), else one that regenerated a lock in a package a reverted
        PR's change touches.  Left in place, it rewrites the lean-audit.json the PR's own revert restores, which then conflicts,
        or keeps records of reverted Lean."""
        prs = {x.pr for x in reverted}
        touched = set().union(*(changed(self.repo, f"{x.commit}^1", x.commit) for x in reverted)) if reverted else set()
        out = []
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--format=%H%x00%B%x1e", f"{base}..{tip}").split("\x1e"))):
            c, body = rec.split("\x00", 1)
            named = {int(n) for n in re.findall(r"#(\d+)", body.split("\n")[0])}
            if not body.startswith("Lock records regenerated for ") or not prs & named:
                continue
            m = re.search(r"^Regenerated: (.+)$", body, re.M)
            pkgs = [os.path.dirname(f.strip()) for f in (m[1].split(",") if m else [])]
            if whole or any(under(f, [p]) if p else True for f in touched for p in pkgs):
                out.append(c)
        return out

    def _newest_first(self, base: str, tip: str, commits: list[str]) -> list[str]:
        order = {c: i for i, c in enumerate(_out(self.repo, "rev-list", "--topo-order", f"{base}..{tip}").split())}
        return sorted(set(commits), key=lambda c: order.get(c, -1))

    # -- the post-landing run (CI-30): a train lands once its steps but the deferred ones passed, and one run at a time checks
    #    those on `main`'s tip.  `drive["postland"]`: {"verified": the newest `main` whose deferred steps passed, "pending": the
    #    tips landed since, "run": the run of them (with "acting": why it failed, when its revert lost a push race), or None,
    #    "steps": the deferred steps, "resend": the next run is the one resend of a run the machine killed, "failed": a tip whose
    #    run failed with nothing drive could revert, or passed without a deferred step, which no run checks again}

    def _hand_over(self, drive: dict, m: Member, prs: str, tip: str, base: str, v: Verdict) -> None:
        """A train landed at ``tip`` on ``base`` with its deferred steps still running: cancel its check, whose slot goes to the
        trains, and record the landing as unverified, for the post-landing run."""
        ids = (drive["launched"].get(m.commit) or {}).get("ids") or v.attempt.split()
        if live := [rid for rid, (st, *_) in self._states(ids).items() if st in ACTIVE]:
            self.cancel(self.repo, live, f"drive: {prs} landed at {tip[:12]}; its deferred steps run in the post-landing run of `main`")
            self._seen.update({rid: ("cancelled", SIGTERM_RC, None, None) for rid in live})
            self.q.log(f"research queue drive: cancelled {' '.join(live)}, {prs}'s deferred steps: the post-landing run checks them")
        if "postland" not in drive:                 # a landing the old rule still follows (`deferred`) isn't verified either
            olds = [r["base"] for r in drive.get("deferred", {}).values() if self._ancestor(r["base"], base)]
            base = max(olds, key=lambda b: int(_out(self.repo, "rev-list", "--count", f"{b}..{base}")), default=base)
        pl = drive.setdefault("postland", {"verified": base, "pending": [], "run": None})
        pl["pending"].append(tip)
        pl["steps"] = sorted(set(pl.get("steps") or ()) | self.defer)
        # now, not at the tick's end: `main` has the landing, and a drive killed before then would take it as verified
        self.q.save_state({**self.q.load_state(), "drive": drive})

    def _ancestor(self, a: str, b: str) -> bool:
        return not _git(self.repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    def landed_since(self, base: str, tip: str) -> list[Member]:
        """The PRs the queue landed in ``base``..``tip``, oldest first: each merge of a PR onto `next` (`Queue: #N`, with its
        `PR`, `Head` and `Owner` trailers) that `main` reached, first parent or inside a train merged onto a moved `main`, but
        those a revert in that range undid.  Nothing else: never a lander's commit."""
        from .queue import TRAILER_RE

        out, undone = [], set()
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--reverse", "--topo-order", "--format=%H%x00%P%x00%B%x1e",
                                                          f"{base}..{tip}").split("\x1e"))):
            commit, parents, body = rec.split("\x00", 2)
            undone |= set(REVERTS_RE.findall(body))
            t = dict(TRAILER_RE.findall(body))
            if len(parents.split()) > 1 and body.startswith("Queue: #") and {"PR", "Head", "Owner"} <= set(t):
                out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
        return [x for x in out if x.commit not in undone]

    def follow_postland(self, drive: dict) -> None:
        """The post-landing run, once it ends: passed, its tip is verified; failed, `_postland_failed`; never started (every twin
        exited 75, or was stopped), the next `start_postland` launches it again.  One already failing under `--keep-going` is
        acted on at once, and cancelled.  A pass verifies its tip only when every deferred step passed in it (`_unrun`); one
        whose revert lost a push race (`acting`) is acted on again next tick.  One still launching (`_resolve`) runs; one whose
        train.sh registered no run is told to `CI`, and launched again."""
        if not (pl := drive.get("postland")) or not (run := pl.get("run")):
            return
        if (tail := self._resolve(run, f"the post-landing run of `main` {run['tip'][:12]}")) is not None:
            pl["run"] = None
            if run.get("resend"):
                pl["resend"] = True
            self.refuse(drive, f"launch:postland:{run['tip']}", f"train.sh {run['tip'][:12]} {self.on} --deferred "
                                                                f"{','.join(pl.get('steps') or sorted(self.defer))} registered no run: {tail}")
            return
        if "launch" in run:
            return
        tip, signalled = run["tip"], False
        if (why := run.get("acting")) is None:
            states = self._states(run["ids"])
            if self._lost(run, states):
                if not run.get("resend"):
                    pl["run"], pl["resend"] = None, True
                    self.q.log(f"research queue drive: the post-landing run {' '.join(run['ids'])} of {tip[:12]} is lost (no "
                               f"state read {LOST_AFTER_S // 3600} h after its launch): it launches again once")
                    return
                states = {r: ("lost", -1) for r in states}
            early = next((s[3] for s in states.values() if len(s) > 3 and s[0] in ACTIVE and s[3]), None)
            if not early and any(st in ACTIVE for st, *_ in states.values()):
                return
            if not early and any(rc == 0 for _, rc, *_ in states.values()):
                pl["run"] = None
                if unrun := self._unrun(run["ids"], [s for s in pl.get("steps") or sorted(self.defer) if s not in run.get("skip", ())]):
                    if run.get("baseline"):
                        self._red_baseline(drive, pl, run, f"it passed without passing {', '.join(unrun)}")
                        return
                    pl["failed"] = tip
                    self.refuse(drive, f"postland:{tip}", f"the post-landing run {' '.join(run['ids'])} of `main` {tip[:12]} "
                                                          f"passed without passing {', '.join(unrun)} (skipped, or absent from its "
                                                          "steps.jsonl): it verified nothing and reverted nothing. Find why; "
                                                          "the next landing's run checks `main` again")
                    return
                pl["verified"] = tip
                if "as" in run:
                    # a step it skipped by name (`start_postland`) still holds from the run that verified `main` before
                    kept = set((pl.get("verified_as") or {}).get("steps") or ()) & set(run.get("skip", ()))
                    pl["verified_as"] = {**run["as"], "steps": [s for s in AFTER_LANDING if s in kept | set(run["as"]["steps"])]}
                else:
                    pl.pop("verified_as", None)
                pl["pending"] = [p for p in pl["pending"] if not self._ancestor(p, tip)]
                self.q.log(f"research queue drive: `main` {tip[:12]} verified: its post-landing run {' '.join(run['ids'])} passed "
                           f"({', '.join(pl.get('steps') or ())})" + (f"; {len(pl['pending'])} landing(s) above it wait" if pl["pending"] else ""))
                return
            if not early and all(rc in (NOT_STARTED, SIGTERM_RC) for _, rc, *_ in states.values()):
                pl["run"] = None
                self.q.log(f"research queue drive: the post-landing run {' '.join(run['ids'])} of {tip[:12]} never ran or was "
                           "stopped: it launches again")
                return
            if early and not (self.dry_run or self.q.shadow):
                self.cancel(self.repo, [r for r, (st, *_) in states.items() if st in ACTIVE],
                            f"drive: the post-landing run of {tip[:12]} has failed ({early}); drive acted on it")
            why = (f"it has already failed, still running: {early}" if early
                   else ", ".join(f"{st} rc={rc}" for st, rc, *_ in states.values()))
            # not SIGTERM: a twin a sibling won the slot from, or a cancel (all of them: launched again above)
            signalled = not early and any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                          for _, rc, *_ in states.values())
        if self._postland_failed(drive, pl, run, why, signalled):
            pl["run"] = None
        else:
            run["acting"] = why

    def _unrun(self, ids: list[str], steps: list[str]) -> list[str]:
        """The deferred ``steps`` no run among ``ids`` records as passed in its steps.jsonl: a run's exit 0 says only that
        nothing failed (a step `skipped:`, or one `--deferred` maps to no part, passes it too)."""
        got = set()
        for d in self.evidence(self.repo, ids):
            with contextlib.suppress(OSError):
                for line in (d / PROGRESS[0]).read_text().splitlines():
                    with contextlib.suppress(ValueError):
                        if (rec := json.loads(line)).get("status") == "passed":
                            got.add(rec.get("step"))
        return [s for s in steps if s not in got]

    def _postland_failed(self, drive: dict, pl: dict, run: dict, why: str, signalled: bool = False) -> bool:
        """A post-landing run of ``run["tip"]`` that failed.  Killed by the machine (``signalled``: by a signal or 128+N, or
        `landed_culprit` says `KILLED`), it runs again once, on `main`'s tip then, and killed again is told to `CI`.  When no PR
        landed since `verified` touches what failed or its declared inputs (`UNTOUCHED`), `main` was broken before them (18
        Oct: a revert left benchmarks/stacked_share red) and reverting them would leave it red, and revert every landing after
        too: it reverts nothing and tells `CI`.  Else the one PR landed since `verified` whose change touches what failed
        (`landed_culprit`) is reverted on `main` (`git revert -m 1` of its merge onto `next`), the rest staying; with no single such
        PR, every landing since `verified` is (Daniel, 11 Oct: a failure there reverts the landing at once): each of their
        merges onto `next`, newest first, and no other commit, so `main` doesn't stay red.  Either takes the lock commits that
        go with them (`_lock_commits`), and passes over a merge `main` reverted already (`_undone`).  `CI` and the owners are
        told, and the next run checks `main`'s new tip.  False when the revert's push lost a race: the next tick acts again.
        A review landing's run reverts only when each step that failed passed on `verified` under the same flags
        (`_unconfirmed`), and with several PRs implicated it bisects them (`_bisect_step`) rather than revert them all."""
        tip, ids, base = run["tip"], " ".join(run["ids"]), pl["verified"]
        members = self.landed_since(base, tip)
        one, hit, what = (self.landed_culprit(members, Verdict("failed", ids, why)) if members
                          else (None, [], f"no PR the queue landed since {base[:12]} is in it"))
        since = " ".join(f"#{x.pr}" for x in members) or "none"
        head = (f"the post-landing run {ids} of `main` {tip[:12]} ({', '.join(pl.get('steps') or ())}) failed ({why}); PRs "
                f"landed since {base[:12]}, its last verified `main`: {since}")
        key = f"postland:{tip}"
        if (signalled or what == KILLED) and not run.get("resend"):
            pl["resend"] = True
            self.q.log(f"research queue drive: {head}. It was killed, a machine's fault: it runs again once, on `main`'s tip")
            return True
        if run.get("baseline"):
            self._red_baseline(drive, pl, run, why)
            return True
        if signalled or what == KILLED:
            pl["failed"] = tip
            self.refuse(drive, key, f"{head}. Reverted nothing: it was killed again on its resend, a machine's fault")
            return True
        if "as" in run and (unsure := self._unconfirmed(pl, run)):
            pl["failed"] = tip
            self.refuse(drive, key, f"{head}. Reverted nothing: {unsure}, so `main` may have been red before them. Fix `main`, or "
                                    "once it is green record it: `/data/infra/drive.sh queue baseline SHA --by ci`")
            return True
        if not members or what == UNTOUCHED:
            pl["failed"] = tip
            self.refuse(drive, key, f"{head}. Reverted nothing: " + (what if not members else f"no PR landed since {base[:12]} "
                                    "touches what failed or its declared inputs") + ", so `main` was broken before them, or the "
                                    "machine failed. Fix `main`; the next landing's run checks it again")
            return True
        if "as" in run and one is None and len(cands := hit or members) > 1:
            if REVERTS_RE.search(_out(self.repo, "log", "--format=%B", f"{base}..{tip}")):
                pl["failed"] = tip
                self.refuse(drive, key, f"{head}. Reverted nothing: a revert lies in {base[:12]}..{tip[:12]}, so a landing's "
                                        "commit there isn't what `main` ran, and bisecting could blame the wrong PR. Find the "
                                        "culprit by hand")
                return True
            pl["bisect"] = {"tip": tip, "base": base, "steps": self._failing(run), "why": why,
                            "no_cache": (run.get("as") or {}).get("no_cache", self.after_landing_no_cache),
                            "cands": [[x.pr, x.head, x.owner, x.commit] for x in cands], "good": -1,
                            "bad": len(cands) - 1 if cands[-1].commit == tip else len(cands), "run": None}
            self.q.log(f"research queue drive: {head}. {what or 'No single PR is implicated'}: bisecting "
                       f"{' '.join(f'#{x.pr}' for x in cands)} with {', '.join(pl['bisect']['steps'])} alone; nothing lands meanwhile")
            return True
        return self._revert_landed(drive, pl, tip, base, members, one, hit, what, head, key)

    def _lost(self, run: dict, states: dict) -> bool:
        """Whether every twin of ``run`` that hasn't ended has had no readable state (`unknown`) for `LOST_AFTER_S` since its
        launch: it is taken as lost, as a killed run is, so a store or node that never answers doesn't hold drive forever."""
        live = [st for st, *_ in states.values() if st in ACTIVE]
        return bool(live) and all(st == "unknown" for st in live) and self.now() - run.get("at", self.now()) > LOST_AFTER_S

    def _red_baseline(self, drive: dict, pl: dict, run: dict, why: str) -> None:
        """The review landing's first run, of `main` itself, failed: `main` was red before any review landing.  It is verified
        anyway with the steps it passed alone (`verified_as`), so landing goes on and a later failure reverts only on a step that
        passed here (`_unconfirmed`); `CI` is told."""
        tip, steps = run["tip"], (run.get("as") or {}).get("steps") or list(AFTER_LANDING)
        unrun = self._unrun(run["ids"], steps)
        pl["verified"], pl["verified_as"] = tip, {"steps": [s for s in steps if s not in unrun], "no_cache": run["as"]["no_cache"]}
        pl["pending"] = [p for p in pl["pending"] if not self._ancestor(p, tip)]
        self.refuse(drive, f"postland:{tip}", f"the baseline after-landing run {' '.join(run['ids'])} of `main` {tip[:12]} failed "
                                              f"({why}): `main` was red before any review landing. Landing goes on, and until a "
                                              f"run of `main` passes {', '.join(unrun) or 'them'}, their failure reverts nothing. "
                                              "Fix `main`")

    def _failing(self, run: dict) -> list[str]:
        """The steps a post-landing run's steps.jsonl records as failed."""
        return sorted({s for d in self.evidence(self.repo, run["ids"]) for st in failed_steps(d) if (s := st.get("step"))})

    def _unconfirmed(self, pl: dict, run: dict) -> str | None:
        """Why a review landing's failed after-landing run can't revert a landing: the step that failed didn't pass on `verified`
        under the same flags (`verified_as`), so it may have failed there too; None when each did."""
        vs, fails = pl.get("verified_as"), self._failing(run)
        if not fails:
            return "its steps.jsonl names no failed step"
        if not vs:
            return f"{', '.join(fails)} never passed on `main` {pl['verified'][:12]} in an after-landing run"
        if vs.get("no_cache") is not None and vs["no_cache"] != (run.get("as") or {}).get("no_cache"):
            return f"`main` {pl['verified'][:12]} was verified {'without' if vs['no_cache'] else 'with'} reuse, and this run differs"
        if missing := [s for s in fails if s not in vs.get("steps", ())]:
            return f"{', '.join(missing)} didn't pass on `main` {pl['verified'][:12]}"
        return None

    def _revert_landed(self, drive: dict, pl: dict, tip: str, base: str, members: list[Member], one: Member | None,
                       hit: list[Member], what: str, head: str, key: str, blame: str | None = None) -> bool:
        """Revert ``one`` on `main`, else every landing since ``base`` (`_postland_failed`), and tell `CI` and the owners."""
        since = " ".join(f"#{x.pr}" for x in members) or "none"
        if self.dry_run or self.q.shadow:
            self.q.log(f"research queue drive: would revert {f'#{one.pr}' if one else f'every landing since {base[:12]}'} on "
                       f"`main`: {head}")
            return True
        reverted = [one] if one is not None else members
        locks = self._lock_commits(base, tip, reverted, one is None)
        commits = self._newest_first(base, tip, [x.commit for x in reverted] + locks)
        got = self._revert(commits)
        if got == "refused":
            self.q.log(f"research queue drive: reverting {' '.join(f'#{x.pr}' for x in reverted)} lost a race to `main`'s push; "
                       "the next tick tries again")
            return False
        if got == "conflict":
            pl["failed"] = tip
        undo = f"`git revert {' '.join(c[:12] for c in commits)}` (`-m 1` for a merge)"
        with_locks = f", with the lock commit {' '.join(c[:12] for c in locks)} regenerated for it" if locks else ""
        if one is not None:
            lead = {"pushed": f"#{one.pr} reverted on `main`{with_locks}", "undone": f"#{one.pr} was reverted on `main` already",
                    "conflict": f"#{one.pr} must be reverted by hand ({undo})"}[got]
            rest = " ".join(f"#{x.pr}" for x in members if x is not one)
            self.refuse(drive, key, f"{lead}: {head}. " + (blame or f"The failure is in {what}, which of those only #{one.pr}'s "
                                                                    "change touches")
                                    + (f"; {rest} stay on `main`" if rest else "")
                                    + (", and the revert didn't apply" if got == "conflict" else ". Fix and re-mark"), self._owner(one))
            return True
        blame = f"the failure is in {what}, which {' '.join(f'#{x.pr}' for x in hit)} change" if hit else what
        lead = {"pushed": f"Every landing since {base[:12]} reverted on `main` ({since}){with_locks}",
                "undone": f"Every landing since {base[:12]} ({since}) was reverted on `main` already",
                "conflict": f"Every landing since {base[:12]} ({since}) must be reverted by hand ({undo})"}[got]
        self.refuse(drive, key, f"{lead}: {head}. No single PR is implicated: {blame}"
                                + (", and the revert didn't apply" if got == "conflict" else ". Fix and re-mark"),
                    *(self._owner(x) for x in members))
        return True

    def start_postland(self, drive: dict) -> None:
        """Launch the one post-landing run, on `main`'s tip, when none runs, a landing awaits it, and that tip is neither verified
        nor one whose run failed with nothing drive could revert."""
        if not (pl := drive.get("postland")) or pl.get("run") or pl.get("bisect"):
            return
        q, main = self.q, _out(self.repo, "rev-parse", "origin/main")
        pl["pending"] = [p for p in pl["pending"] if not self._ancestor(p, pl["verified"])]
        base = self.review and not self._baselined(pl)            # the review landing's first run: `main` itself (`_unbaselined`)
        if not base and (not pl["pending"] or main == pl.get("failed") or main == pl["verified"]):
            return
        if self.review:
            pl["steps"] = list(AFTER_LANDING)
        steps = ",".join(pl.get("steps") or sorted(self.defer))
        extra, skip = {}, []
        if self.review:
            # the run checks `main` against itself, so it sends the upstream build only when told to: when the landings since
            # the last verified `main` change what lean-agreement checks; lean-changed has no cache to compare with under --no-cache;
            # lean-audit fails on a record difference, since nothing commits a regenerated record after landing
            from .merge import required_steps

            extra = {"agreement": base or "lean-agreement" in required_steps(self.repo, pl["verified"], main, "check"),
                     "no_cache": self.after_landing_no_cache, "no_regenerate": True}
            skip = [s for s, off in (("lean-agreement", not extra["agreement"]), ("lean-changed", extra["no_cache"])) if off]
        flags = "".join(f" --{k.replace('_', '-')}" for k, on in extra.items() if on)
        if self.dry_run or q.shadow:
            q.log(f"research queue drive: would launch train.sh {main[:12]} {self.on} --deferred {steps}{flags}")
            return
        with slot_rank(None):
            rec, tail = self._started(self.launch_postland(self.repo, main, self.on, steps, **extra))
        if rec is None:
            self.refuse(drive, f"launch:postland:{main}", f"train.sh {main[:12]} {self.on} --deferred {steps}{flags} registered no "
                                                          f"run: {tail}")
            return
        pl["run"] = {**rec, "tip": main, "at": self.now(), **({"resend": True} if pl.pop("resend", False) else {}),
                     **({"skip": skip} if skip else {}), **({"baseline": True} if base else {}),
                     **({"as": {"steps": [s for s in AFTER_LANDING if s not in skip], "no_cache": extra["no_cache"]}}
                        if self.review else {})}
        q.log(f"research queue drive: post-landing run of `main` {main[:12]} ({steps}): {_ids(rec)}; "
              f"{len(pl['pending'])} landing(s) since {pl['verified'][:12]}")

    def _owner(self, m: Member) -> str | None:
        """The owning lane of a PR the queue landed: its open PR's (`_handle`), else the one its merge's `Owner` names."""
        from .queue import PR

        return self._handle(m.pr) or self.q.handle(PR(m.pr, m.head, body=f"Lane {m.owner}"))

    # -- landing

    def _chain(self) -> tuple[str, list[Member]]:
        """(the base `next` was built on, its members oldest first), whether or not `main` has moved since."""
        return built_on(self.repo, f"refs/remotes/origin/{self.q.branch}")

    def held(self, drive: dict) -> str | None:
        """Why `next` stays on its base this tick (`queue.hold`), a member's check running by its verdict."""
        return hold(self.repo, self.q.branch, lambda c, base: self.verdict(c, base, drive).state == "running")

    def land(self, drive: dict, dry_run: bool) -> Member | None:
        """Land the newest member of `next` with a passing check (see the module's step 1): a fast-forward when `main` is still
        `next`'s base, so the members above it keep their commits and their checks; a merge commit onto a moved `main`."""
        q = self.q
        base, chain = self._chain()
        if not chain:
            return None
        main = _out(self.repo, "rev-parse", "origin/main")
        moved = sorted(changed(self.repo, base, main)) if main != base else []
        for i in range(len(chain) - 1, -1, -1):
            m = chain[i]
            v = self.verdict(m.commit, base, drive)
            if v.state != "passed":
                continue
            self._land_saw = m.commit
            members = chain[: i + 1]
            prs = " ".join(f"#{x.pr}" for x in members)
            if moved and ((shared := sorted(set(moved) & changed(self.repo, base, m.commit)))
                          or (shared := [p for p in moved if p.startswith(CHECK_TOOLING)])):
                q.log(f"research queue drive: {prs} passed ({v.attempt}) on {base[:12]}, but `main` moved to {main[:12]} and "
                      f"{'shares' if set(shared) & changed(self.repo, base, m.commit) else 'changed check tooling in'} "
                      f"{', '.join(shared[:5])}: the next sync rebuilds `{q.branch}` on `main` for a new check")
                return None
            if dry_run or q.shadow:
                q.log(f"research queue drive: would land {prs} ({m.commit[:12]}, check {v.attempt})"
                      + (f" onto the moved `main` {main[:12]}" if moved else "") + (" [shadow]" if q.shadow else ""))
                return m
            scratch = Path(tempfile.mkdtemp(prefix="drive-locks-"))
            try:
                return self._land(drive, m, members, prs, v, base, main, moved, scratch)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        return None

    def _locks(self, v: Verdict, scratch: Path) -> tuple[str, dict[str, Path], str | None]:
        """(the check that regenerated them, {package: its regenerated lock, fetched under ``scratch`` or from its node}, why the
        train is held instead) of a passed train, when drive commits regenerated locks; ("", {}, None) when its check regenerated
        none.  Read from the check's files in the store, else, for a check still running (`--defer`), from its run dir on its
        node (`merge.live_records`).  Held, never landed without them (CI-122), when its Lean report names records neither
        holds, or can't be read at all unless its lean-audit step was skipped (`_lean_audit_ran`)."""
        if not self.regenerate_locks:
            return "", {}, None
        from .merge import AUDIT, live_records, regenerated_records

        for rid in v.attempt.split():
            try:
                recs = regenerated_records(self.q.store, rid, scratch / rid)
                if recs is None:
                    recs = live_records(rid, self.fetch)
            except Exception as e:  # noqa: BLE001  (a guard never fails open: `merge.Unreadable`, or the store failing)
                return rid, {}, f"its regenerated lock records can't be read: {type(e).__name__}: {e}"
            if recs is None and self._lean_audit_ran(rid):
                return rid, {}, (f"its {AUDIT}, which says whether its Lean check regenerated lock records, can't be read "
                                 "from the store or from its run dir on its node")
            if recs:
                return rid, {pkg: path for pkg, (_, path) in recs.items() if path is not None}, None
        return "", {}, None

    def _unvouched(self, v: Verdict, rid: str) -> str | None:
        """Why the merge gate refuses the lock commit at HEAD (`merge.regenerated_parent`, reading the records itself: from the
        store, else, for a landing with `--defer`, from ``rid``'s run dir on its node), or None when it vouches for it."""
        from .merge import declared_command, produced, regenerated_parent

        deferred = rid if v.why.startswith((DEFERRED, REVIEWED)) else None
        if regenerated_parent(self.repo, "HEAD", self.q.store, tool="check", kinds=produced(self.repo, "check"),
                              command=declared_command(self.repo, "check"), deferred=deferred, fetch=self.fetch):
            return None
        return (f"the merge gate refuses its lock commit: its records aren't byte for byte the ones {rid} regenerated "
                f"(`merge.regenerated_parent`{', on the live run' if deferred else ''})")

    def fetch(self, rid: str, names: list[str]) -> tuple[Path, dict] | None:
        """Run ``rid``'s local dir and status row with ``names`` fetched from its node (`merge.fetch_run`)."""
        from .merge import fetch_run

        return fetch_run(rid, names, self.repo)

    def _lean_audit_ran(self, rid: str) -> bool:
        """Whether run ``rid``'s lean-audit step may have regenerated records: true unless its result in the store, else its
        steps.jsonl (`evidence`), says the step was skipped or never ran; a run neither can be read of may have."""
        from .merge import step_status

        try:
            stored = bool(((self.q.store.get_attempt(rid) or {}).get("outputs") or {}).get("result"))
        except Exception:  # noqa: BLE001  (unknown: it may have)
            return True
        if stored:
            return step_status(self.q.store, rid, "lean-audit") not in ("skipped", "absent")
        for d in self.evidence(self.repo, [rid]):
            with contextlib.suppress(OSError):
                for line in (d / PROGRESS[0]).read_text().splitlines():
                    with contextlib.suppress(ValueError):
                        if (rec := json.loads(line)).get("step") == "lean-audit":
                            return rec.get("status") != "skipped"
        return True

    def _land(self, drive: dict, m: Member, members: list[Member], prs: str, v: Verdict, base: str, main: str,
              moved: list[str], scratch: Path) -> Member | None:
        """Land ``m``'s train (`land`), with the locks its check regenerated committed on top (`regenerate_locks`)."""
        q = self.q
        rid, locks, held = self._locks(v, scratch)
        if held:
            self.refuse(drive, f"locks:{m.commit}", f"{prs} passed ({v.attempt}) but is held, not landed: {held}. Find why; "
                                                     "drive lands it once they can be read (the store has them when the check "
                                                     "ends)", *(self._owner(x) for x in members))
            q.log(f"research queue drive: {prs} held: {rid} {held}")
            return None
        if locks and moved:
            q.log(f"research queue drive: {prs} passed ({v.attempt}) by regenerating {', '.join(sorted(locks))}'s locks on "
                  f"{base[:12]}, but `main` moved to {main[:12]}: the next sync rebuilds `{q.branch}` for a new check")
            return None
        _git(self.repo, "checkout", "-q", "-B", "main", "refs/remotes/origin/main")
        if moved:
            gained = _out(self.repo, "log", "--first-parent", "--format=%s", f"{base}..{main}").splitlines()
            body = (f"Merge {m.commit[:12]} ({' + '.join(x.head[:8] for x in members)})\n\n{prs}\n\n"
                    f"Checked: {v.attempt} (full check of {m.commit[:12]} = main {base[:12]} + "
                    f"{' + '.join(f'#{x.pr}@{x.head[:9]}' for x in members)}: passed)\n"
                    f"Skipped-check: main {base[:12]}..{main[:12]} ({'; '.join(gained)}) shares no files with {prs}\n")
            r = _git(self.repo, "merge", "--no-ff", "--no-edit", "-m", body, m.commit, check=False, env=IDENTITY)
        else:
            r = _git(self.repo, "merge", "-q", "--ff-only", m.commit, check=False)
        if r.returncode:
            _git(self.repo, "merge", "--abort", check=False)
            _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
            self.refuse(drive, f"land:{m.commit}", f"{prs} passed ({v.attempt}) but doesn't merge onto `main` {main[:12]}: "
                                                    f"{(r.stderr or r.stdout).strip()[-200:]}")
            return None
        if locks:
            files = []
            for pkg, src in sorted(locks.items()):
                dst = f"{pkg}/{LOCK}" if pkg not in ("", ".") else LOCK
                shutil.copyfile(src, self.repo / dst)
                files.append(dst)
            _git(self.repo, "add", "--", *files)
            _git(self.repo, "commit", "-q", "-m", f"Lock records regenerated for {prs}\n\nChecked: {rid} (the Lean check of "
                                                  f"{m.commit[:12]}, passed by regenerating them)\nRegenerated: {', '.join(files)}\n",
                 env=IDENTITY)
            if why := self._unvouched(v, rid):
                _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
                self.refuse(drive, f"lock-gate:{m.commit}", f"{prs} passed ({v.attempt}) but is held, not landed: {why}")
                return None
        p = q._remote(self.repo, "push", "-q", "--no-verify", "origin", "main:refs/heads/main", ok=None)
        if p.returncode:
            _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
            q.log(f"research queue drive: pushing `main` was refused ({p.stderr.strip()[-200:]}); the next tick tries again")
            return None
        tip = _out(self.repo, "rev-parse", "HEAD")
        q.log(f"research queue drive: landed {prs} on `main` at {tip[:12]} (check {v.attempt})"
              + (f", with the locks it regenerated for {', '.join(sorted(locks))}" if locks else ""))
        if v.why.startswith(DEFERRED):
            self._hand_over(drive, m, prs, tip, main, v)
        elif v.why.startswith(REVIEWED):
            self._after_landing(drive, tip, main)
        for x in members:
            if (now := self._head(x.pr)) and now != x.head:
                self.refuse(drive, f"stale:{x.pr}@{x.head}", f"#{x.pr} landed at {x.head[:12]}, the head its train checked, not "
                                                             f"at {now[:12]}, its head now: a new head doesn't cancel a running "
                                                             f"train, so {now[:12]} rejoins `{q.branch}` above and lands in a later "
                                                             "one", self._owner(x))
        from .merge import _sweep_after, caught_up

        def after() -> None:
            with authed(q.auth):
                caught_up(self.repo, "origin", "check", since=main, log=q.log,
                          runs={x.commit: v.attempt for x in members} | {tip: v.attempt})

        # the landing's after-steps (spec DM, mirror, post_train, posts, labels) wait for the tick's launches (23 Oct: landing ticks
        # held the next trains' launches 20-30 min behind them); the sweep stays here, since it closes what the sync must not read
        if (pending := getattr(self, "_after", None)) is not None:
            pending.append(after)
        else:
            after()
        with authed(q.auth):
            _sweep_after(self.repo, "main", "origin")
        return m

    # -- refusals

    def refuse(self, drive: dict, key: str, text: str, *handles: str | None) -> None:
        """Tell `CI` once per key, and ``handles``, the owning lanes of the PRs it concerns, beside it so each owner hears
        without a relay; with --shadow or --dry-run, only log it."""
        if key in drive["told"]:
            return
        to = tuple(dict.fromkeys((CI, *(f"@{h}" for h in handles if h))))
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue drive: would tell {' '.join(to)}: {text} [{'shadow' if self.q.shadow else 'dry run'}]")
            return
        if self.post(f"{' '.join(to)} queue: {text}", to):
            drive["told"].append(key)
            self.q.log(f"research queue drive: told {' '.join(to)}: {text}")
        else:
            self.q.log(f"research queue drive: could not post to the merge thread (the next tick retries): {' '.join(to)}: {text}")

    def _handle(self, pr: int) -> str | None:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self.q.handle(self._prs[pr]) if pr in self._prs else None

    def _head(self, pr: int) -> str | None:
        """Open PR ``pr``'s head now, None when it isn't open."""
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs[pr].head if pr in self._prs else None

    def culprit(self, members: list[Member], v: Verdict) -> tuple[list[Member], str]:
        """The members of a failed train whose own change (its merge commit against the one below) touches a file the failure
        points at (`implicated`), and what it points at.  With none, what the failure says: `KILLED` for a kill that names no
        file, `NO_REPORT`, `UNTOUCHED` when no change touches the declared inputs of what failed either (`by_inputs`), else
        that overlap, which only informs (18 Oct 01:55Z: ejecting on it took #2348 and nine others with the culprit)."""
        found: set[str] = set()
        dirs = self._evidence(v)
        for d in dirs:
            if (paths := implicated(d)) is None:
                if killed(d, v.why):
                    return [], KILLED
                continue
            found |= paths
        own = {x.commit: changed(self.repo, f"{x.commit}^1", x.commit) for x in members}
        hits = [(x, t) for x in members if (t := touches(own[x.commit], found))] if found else []
        if hits:
            return [x for x, _ in hits], ", ".join(sorted({f for _, t in hits for f in t}))
        if not dirs:
            return [], NO_REPORT
        back, what = self.by_inputs(members, own, dirs)
        if back:
            return [], f"{what}, which {' '.join(f'#{x.pr}' for x in back)} change{'s' if len(back) == 1 else ''}"
        return [], UNTOUCHED

    def landed_culprit(self, members: list[Member], v: Verdict) -> tuple[Member | None, list[Member], str]:
        """`culprit` for the post-landing run, which reverts rather than bisects (no prefix of `main` is checked again): the one
        landed member whose change touches a file the failure points at, or else the declared inputs of what failed
        (`by_inputs`), else None; with the members that touch one, and what the failure points at."""
        found: set[str] = set()
        dirs = self.evidence(self.repo, v.attempt.split())
        for d in dirs:
            if (paths := implicated(d)) is None:
                if killed(d, v.why):
                    return None, [], KILLED
                continue
            found |= paths
        own = {x.commit: changed(self.repo, f"{x.commit}^1", x.commit) for x in members}
        hits = [(x, t) for x in members if (t := touches(own[x.commit], found))] if found else []
        if hits:
            what = ", ".join(sorted({f for _, t in hits for f in t}))
            return (hits[0][0] if len(hits) == 1 else None), [x for x, _ in hits], what
        if not dirs:
            return None, [], NO_REPORT
        back, what = self.by_inputs(members, own, dirs)
        if back:
            return (back[0] if len(back) == 1 else None), back, what
        return None, [], UNTOUCHED

    def by_inputs(self, members: list[Member], own: dict[str, set[str]], dirs: list[Path]) -> tuple[list[Member], str]:
        """The members whose change touches the declared inputs of what failed, when the failure's own files touch none
        (16 Oct: #2164's suite failed in tests/, and the cause was its benchmarks/pouw/pyproject.toml): a failed suite's
        directory and `[tool.verity.tests] inputs`, a Lean step's Lean sources, circuit-check's (`queue.forces_cold`).  It says
        whether a PR may be the cause, so the failure is bisected rather than resent, and names no PR to eject."""
        from .queue import forces_cold

        tip, rules = members[-1].commit, []
        for d in dirs:
            for s in failed_suites(d):
                ins = suite_inputs(self.repo, tip, s.get("suite") or "")
                rules.append((f"suite {s.get('name') or s.get('suite')}'s inputs", lambda f, ins=ins: under(f, ins)))
            for st in failed_steps(d):
                step = st.get("step") or ""
                if step.startswith("lean") and step != "lean-suites":
                    if pkgs := self._failed_packages(d, st, tip):
                        rules.append((f"Lean package {', '.join(sorted(pkgs))}", lambda f, pkgs=pkgs: under(f, sorted(pkgs))))
                    else:
                        rules.append(("Lean sources", lambda f: f.endswith(LEAN_INPUTS)))
                elif step == "circuit-check":
                    rules.append(("circuit-check's inputs", lambda f: forces_cold([f])))
        out, why = [], set()
        for x in members:
            if t := {label for label, ok in rules for f in own[x.commit] if ok(f)}:
                out.append(x)
                why |= t
        return out, ", ".join(sorted(why))

    def _failed_packages(self, run_dir: Path, step: dict, tip: str) -> set[str]:
        """The Lake packages a failed Lean step's log reports `FAIL` for, with each package its manifest requires by path, at
        ``tip``; empty when the log names none (18 Oct: a stale record in verity/core/primitives/requirement, #2386's, failed two
        trains, and the fallback on every Lean source ejected #2341, #2370 and #2371 with it)."""
        try:
            log = (run_dir / Path(step.get("log") or "").name).read_text(errors="replace") if step.get("log") else ""
        except OSError:
            return set()
        out = set()
        for pkg in set(LEAN_AUDIT_RE.findall(log)):
            out.add(pkg)
            r = _git(self.repo, "show", f"{tip}:{pkg}/lake-manifest.json", check=False)
            with contextlib.suppress(ValueError, AttributeError):
                for dep in json.loads(r.stdout).get("packages", []) if not r.returncode else []:
                    if dep.get("type") == "path":
                        out.add(os.path.normpath(os.path.join(pkg, dep.get("dir", ""))))
        return {p for p in out if not p.startswith("..")}

    def failed(self, drive: dict, m: Member, members: list[Member], v: Verdict) -> None:
        """Act on a failed train with no lander (16 Oct, Daniel through top) by bisection over the checked prefixes of `next`:
        its cause is among the members above the highest lower prefix whose check passed, or the base.
        - One member there is the cause: it is ejected, once its failure was seen twice (`_again`; else ``m`` is resent), and,
          the lowest member, once `main` itself passed (`_main_red`).
        - Of several, those whose change touches a file the failure names (`culprit`) are ejected at once; else drive checks
          the prefix halfway between (`bisect`) and acts again when that check ends: log2 of their count checks at most.
        - A lower check that passed with `--defer`'s steps still running clears nobody of a failure in one of them.
        - A kill that points at nothing, or a failure nothing in the train reaches, a machine's fault, is checked again once,
          on ``on``'s machines (`resend`); a kill again, or a failure with no report, is told to `CI`.
        A member a lower running check covers is ejected all the same when the failure names a file only such members change:
        that check carries the same change, and every train above waits on it while it runs (23 Oct 12:05-12:48Z: #3266's
        read-guard failure held the queue behind a lower check whose deferred pytest would have failed it after landing). A
        lower failed check acts first (18 Oct 10:30Z)."""
        key, prs = f"failed:{m.commit}", " ".join(f"#{x.pr}" for x in members)
        if key in drive["told"]:
            return
        main = _out(self.repo, "rev-parse", "origin/main")
        fails = self._signature(v)
        # a lower check that passed with its deferred steps still running clears nobody of a failure in one of them
        lower = {k: "running" if u.state == "passed" and u.why.startswith(DEFERRED) and self._in_deferred(fails) else u.state
                 for k, x in enumerate(members[:-1]) if x.commit in drive["launched"]
                 for u in [self.verdict(x.commit, main, drive)]}
        if "failed" in lower.values():
            return
        passed = max((k for k, s in lower.items() if s == "passed"), default=-1)
        floor = max((k + 1 for k, s in lower.items() if s in ("running", "passed")), default=0)
        if (rec := drive.get("bisect", {}).get(m.commit)) and any(
                s == "running" and members[k].commit == rec["check"] for k, s in lower.items()):
            self.q.log(f"research queue drive: bisecting {m.commit[:12]}: waiting on the check of {rec['check'][:12]}")
            return
        head = f"{m.commit[:12]} on `{self.q.branch}` ({prs}) failed its check: {v.why}"
        hit, what = self.culprit(members, v)
        again = self._again(drive, m, fails)
        if what in (KILLED, UNTOUCHED) and m.commit not in drive.setdefault("resent", {}) and not (what == UNTOUCHED and again):
            self.resend(drive, m, prs, v)
            return
        above = members[passed + 1:]
        known = what not in (KILLED, NO_REPORT)
        if known and len(above) == 1:
            if above[0] not in hit and not again:
                self.resend(drive, m, prs, v, f"bisection alone names #{above[0].pr}, on this one run's word: it runs again "
                                               "before anyone is ejected")
                return
            if passed < 0 and above[0] not in hit and self._main_red(drive, key, head, m, main, fails):
                return
            low = f"{members[passed].commit[:12]}, whose check passed" if passed >= 0 else f"`main` {main[:12]}"
            on = "it failed again on its resend, naming no file a change touches" if what == UNTOUCHED else f"the failure is in {what}"
            self._eject(drive, key, head, m, prs, v, above, f"it is the only PR above {low}; {on}")
            return
        hit = [x for x in hit if members.index(x) > passed]
        if upper := [x for x in hit if members.index(x) >= floor]:
            hit = upper
        if hit:
            self._eject(drive, key, head, m, prs, v, hit, f"{'its' if len(hit) == 1 else 'their'} change touches {what}")
            return
        if known:
            self.bisect(drive, m, members, passed, lower, what, fails)
            return
        why = f"it failed again on its resend ({what})" if what == KILLED else what
        self.refuse(drive, key, f"{head}. Not ejected: {why}. Eject the PR that broke it (`research queue eject N --reason ...`); "
                                "the next sync rebuilds without it")

    def _eject(self, drive: dict, key: str, head: str, m: Member, prs: str, v: Verdict, xs: list[Member], because: str) -> None:
        """Eject ``xs`` from ``m``'s failed train, and tell `CI` and their owners why (``because``).  A PR pushed since the head
        its member merged is not ejected: `eject` marks the PR's current head, often the owner's fix, and a running check kept
        the old one on `next` (`queue.frozen`); the next sync checks the new head in its place."""
        ejected, why, pushed = [], because, []
        for x in xs:
            if (now := self._head(x.pr)) not in (None, x.head):
                pushed.append(f"#{x.pr} was pushed to {now[:8]} since {x.head[:8]}, the head that failed: that head isn't "
                              "ejected, and the next sync checks it in its place")
                self.q.log(f"research queue drive: not ejecting #{x.pr}: {pushed[-1]}")
                continue
            reason = f"train {m.commit[:12]} ({prs}), check {v.attempt}: #{x.pr}: {because}"
            if self.q.shadow or self.dry_run:
                self.q.log(f"research queue drive: would eject #{x.pr}: {reason}")
                continue
            try:
                self.q.eject(x.pr, reason, "ci")
                ejected.append(x)
            except QueueError as e:
                why = f"{because}; ejecting #{x.pr} failed: {e}"
        also = "".join(f". {p}" for p in pushed)
        if ejected:
            names, one = " ".join(f"#{x.pr}" for x in ejected), len(ejected) == 1
            self.refuse(drive, key, f"{head}. Ejected {names}: {because}; the next sync rebuilds without "
                                    f"{'it' if one else 'them'}. Fix and re-mark{also}", *(self._handle(x.pr) for x in xs))
            return
        if pushed and why == because:
            self.refuse(drive, key, f"{head}. The cause: {because}{also}", *(self._handle(x.pr) for x in xs))
            return
        self.refuse(drive, key, f"{head}. Not ejected: {why}. Eject the PR that broke it (`research queue eject N --reason ...`); "
                                f"the next sync rebuilds without it{also}", *(self._handle(x.pr) for x in xs))

    def bisect(self, drive: dict, m: Member, members: list[Member], passed: int, lower: dict[int, str], what: str,
               fails: list[str]) -> None:
        """Halve the members of ``m``'s failed train its cause is among, those above ``passed`` (the highest lower prefix whose
        check passed, -1 for the base): check the prefix halfway up (`drive["bisect"]` records it, with what ``m`` failed on:
        ``fails``), unless a check of a prefix between them already runs, which halves them too.  Once it ends, `failed` acts
        again, on it if it failed, else on ``m`` with it passed (or landed)."""
        q, main = self.q, _out(self.repo, "rev-parse", "origin/main")
        among = " ".join(f"#{x.pr}" for x in members[passed + 1:])
        if run := [members[k] for k, s in sorted(lower.items()) if s == "running" and k > passed]:
            q.log(f"research queue drive: {m.commit[:12]} failed on {what}; its cause is among {among}, and the running check "
                  f"of {run[-1].commit[:12]} (through #{run[-1].pr}) halves them: waiting on it")
            return
        mid = members[passed + (len(members) - 1 - passed) // 2]
        if q.shadow or self.dry_run:
            q.log(f"research queue drive: would bisect {m.commit[:12]} ({among}) at {mid.commit[:12]} (through #{mid.pr})")
            return
        if (running := self._running(drive, main)) >= self.max_running:
            q.log(f"research queue drive: bisecting {m.commit[:12]} ({among}) waits: {running} checks running")
            return
        ids, tail = self._launch(drive, mid, main)
        if not ids:
            self.refuse(drive, f"launch:{mid.commit}", f"train.sh {mid.commit[:12]} {self.on} registered no run (bisecting "
                                                       f"{m.commit[:12]}): {tail}")
            return
        drive.setdefault("bisect", {})[m.commit] = {"check": mid.commit, "among": [x.pr for x in members[passed + 1:]],
                                                    "ids": ids, "at": self.now(), "fails": fails}
        q.log(f"research queue drive: bisecting {m.commit[:12]}: it failed on {what} and its cause is among {among}, whose "
              f"changes name none of it: checking {mid.commit[:12]} (through #{mid.pr}) as {' '.join(ids)}")

    def _main_red(self, drive: dict, key: str, head: str, m: Member, main: str, fails: list[str]) -> bool:
        """Whether ``m``'s ejection on bisection alone, the lowest member's with no lower prefix passed, waits on a check of
        `main` itself or is refused by it: bisection narrows a failure `main` shares to the lowest member, which is innocent
        (18 Oct 17:00Z: a revert left benchmarks/stacked_share red on `main`, and every train failed that suite).  `main` is
        checked once (`drive["base"]`), unless a check of it passed; failed, `CI` hears `main` is red and nobody is ejected.
        A `main` landed with `--defer` whose deferred steps aren't verified (`drive["postland"]`) passed every other step, and
        its post-landing run is that check for a failure in a deferred step."""
        q, pl = self.q, drive.get("postland") or {}
        if main in drive.get("deferred", {}) or (main != pl.get("verified")
                                                  and any(self._ancestor(p, main) for p in pl.get("pending", ()))):
            if not self._in_deferred(fails):
                return False
            if pl.get("failed") == main:
                self.refuse(drive, f"red:{main}", f"{head}. Not ejected: `main` {main[:12]} is red: its post-landing run failed "
                                                  f"too and reverted nothing, so no PR on `{q.branch}` is the cause. Land a fix "
                                                  "with `--first`, or revert what broke `main`; every train fails until it moves")
                if f"red:{main}" in drive["told"]:
                    drive["told"].append(key)
                return True
            q.log(f"research queue drive: {m.commit[:12]} failed in a deferred step with no lower prefix passed: waiting on the "
                  f"post-landing run of `main` {main[:12]} before ejecting #{m.pr}")
            return True
        if main == pl.get("verified") and self._in_deferred(fails):
            return False
        v = self.verdict(main, main, drive)
        if v.state == "passed":
            return False
        if v.state == "running":
            q.log(f"research queue drive: {m.commit[:12]} failed with no lower prefix passed: waiting on the check of `main` "
                  f"{main[:12]} ({v.attempt}) before ejecting #{m.pr}")
            return True
        if v.state == "failed":
            self.refuse(drive, f"red:{main}", f"{head}. Not ejected: `main` {main[:12]} is red: its own check failed too "
                                              f"({v.why}), so no PR on `{q.branch}` is the cause. Land a fix with `--first`, or "
                                              "revert what broke `main`; every train fails until it moves")
            if f"red:{main}" in drive["told"]:
                drive["told"].append(key)
            return True
        if q.shadow or self.dry_run:
            q.log(f"research queue drive: would check `main` {main[:12]} before ejecting #{m.pr} ({m.commit[:12]} failed)")
            return True
        if (running := self._running(drive, main)) >= self.max_running:
            q.log(f"research queue drive: checking `main` {main[:12]} before ejecting #{m.pr} waits: {running} checks running")
            return True
        ids, tail = self.launch(self.repo, main, self.on, False)
        if not ids:
            self.refuse(drive, f"launch:{main}", f"train.sh {main[:12]} {self.on} registered no run (checking `main` before "
                                                 f"ejecting #{m.pr}): {tail}")
            return True
        drive.setdefault("base", {})[main] = {"ids": ids, "at": self.now(), "for": m.commit}
        q.log(f"research queue drive: {m.commit[:12]} failed with no lower prefix passed, and no check of `main` {main[:12]} "
              f"passed: checking `main` as {' '.join(ids)} before ejecting #{m.pr}")
        return True

    def _signature(self, v: Verdict) -> list[str]:
        """What a failed check failed: its failed suites, and its failed steps other than the suites' own (pytest, lean-suites)."""
        out: set[str] = set()
        for d in self._evidence(v):
            out |= {f"suite {s.get('name') or s.get('suite')}" for s in failed_suites(d)}
            out |= {f"step {st.get('step')}" for st in failed_steps(d) if st.get("step") not in ("pytest", "lean-suites")}
        return sorted(out)

    def _in_deferred(self, fails: list[str]) -> bool:
        """Whether a failure (`_signature`) is in a step `--defer` lets a check pass without; one naming nothing may be."""
        return not fails or any(f in {f"step {s}" for s in self.defer} or (f.startswith("suite ") and "pytest" in self.defer)
                                for f in fails)

    def _again(self, drive: dict, m: Member, fails: list[str]) -> bool:
        """Whether ``m``'s failure was seen twice, so a flake can't eject on bisection alone: ``m`` was resent and failed again,
        or ``m`` is a bisect check that failed on something its failed train failed on too (``fails``: `_signature`)."""
        return m.commit in drive.get("resent", {}) or any(
            r.get("check") == m.commit and set(r.get("fails", ())) & set(fails) for r in drive.get("bisect", {}).values())

    def _running(self, drive: dict, main: str) -> int:
        """The checks counted against `max_running`: drive's launches still running (`_counted`), and its pre-warms."""
        warm = sum(1 for r in drive.get("prewarm", {}).values() if not r.get("done") and r.get("ids")
                   and any(st in ACTIVE for st, *_ in self._states(r["ids"]).values()))
        return warm + self._load(drive, main)[0]

    def _load(self, drive: dict, main: str) -> tuple[int, list[str]]:
        """(how many of `_counted`'s running launches `max_running` counts: each that has started, and at most
        `UNSTARTED_COUNTED` that haven't; those that haven't, `_waits`)."""
        live = [c for c in self._counted(drive, main) if self.verdict(c, main, drive).state == "running"]
        idle = [c for c in live if self._waits(drive, c)]
        return len(live) - len(idle) + min(len(idle), UNSTARTED_COUNTED), idle

    def _waits(self, drive: dict, commit: str) -> bool:
        """Whether ``commit``'s launch hasn't started: it has a live twin and none holds a check slot (each in `WAITS`).  A twin
        `research status` can't read (`unknown`) counts as started, so drive never moves its launch."""
        launch = drive["launched"].get(commit) or drive.get("base", {}).get(commit) or {}
        live = [s[0] for s in self._states(launch.get("ids") or []).values() if s[0] in ACTIVE]
        return bool(live) and all(st in WAITS for st in live)

    def _spare(self) -> dict[str, int]:
        """`slots` of `on`, read once a tick; {} when they can't be read, so nothing moves on a guess."""
        if self._free is None:
            try:
                self._free = dict(self.slots(self.repo, self.on))
            except Exception as e:  # noqa: BLE001  (nothing moves; the launches wait where they are)
                self.q.log(f"research queue drive: reading the check slots of {self.on} failed ({type(e).__name__}: {e})")
                self._free = {}
        return self._free

    def move_waiting(self, drive: dict, members: list[Member], trains: list[int], main: str, top: int | None,
                     idle: list[str]) -> list[str]:
        """Move each train's check that hasn't started (``idle``), lowest first, onto a machine of `on` that could start it now
        (`_spare`) and has no twin of it: cancel its twins, each still waiting for a slot (read again just before), and launch it
        again with its rank, its touched suites kept, so `check.py --on` places it on that free slot (24 Oct 05:28Z: ci cancelled 23
        twins by hand so drive would relaunch their trains onto two new nodes, after 1-4 h in lines whose memory gate admitted
        none).  A free slot on a machine where a twin waits is that twin's to take; a twin that holds a slot is never cancelled.
        Returns the commits moved."""
        cands = sorted((c for c in idle if c in (self._tips or ()) and c in drive["launched"]), key=lambda c: self._pos[c])
        free = self._spare() if cands else {}
        moved: list[str] = []
        for c in cands:
            if not any(n > 0 for n in free.values()):
                break
            launch = drive["launched"][c]
            live = {rid: s for rid, s in self._states(launch["ids"]).items() if s[0] in ACTIVE}
            twins = {s[4] if len(s) > 4 else None for s in live.values()}
            if None in twins:
                continue                                    # where it waits is unknown
            if near := next((t for t in twins if free.get(t, 0) > 0), None):
                free[near] -= 1                             # its twin there takes that slot
                continue
            to = next(n for n, k in free.items() if k > 0)  # `choose`: the first machine of `on` with room
            why = f"{c[:12]} waits in the slot lines of {', '.join(sorted(twins))} while {to} has a train slot free"
            if self.dry_run or self.q.shadow:
                self.q.log(f"research queue drive: would move {c[:12]}: {why}")
                free[to] -= 1
                continue
            now = self.states(self.repo, list(live))
            if not now or any(s[0] in ACTIVE and s[0] not in WAITS for s in now.values()):
                continue                                    # a twin took a slot since: it runs where it is
            waiting = [rid for rid, s in now.items() if s[0] in WAITS]
            stuck = self.cancel(self.repo, waiting, f"drive: moved: {why}") or {}
            self._seen.update({rid: ("cancelled", SIGTERM_RC, None, None) for rid in waiting if rid not in stuck})
            if stuck:
                self.q.log(f"research queue drive: not moving {c[:12]}: cancelling {' '.join(stuck)} failed; it waits where it is")
                continue
            floor = self._rank(members[trains[0]].commit) if top is not None else None
            rank = floor + 1 if floor is not None and c == members[top].commit else self._rank(c)
            with slot_rank(rank):
                ids, tail = self.launch(self.repo, c, self.on, bool(launch.get("agreement")))
            if not ids:
                self.q.log(f"research queue drive: moving {c[:12]}: train.sh registered no run ({tail[-200:]}); it launches again "
                           "as a train never launched")
                continue
            drive["launched"][c] = {**launch, "ids": ids, "at": self.now(), "moved": launch.get("moved", 0) + 1}
            self._seen.update({rid: ("submitted", None, None, None) for rid in ids})
            free[to] -= 1
            moved.append(c)
            self.q.log(f"research queue drive: moved {c[:12]} as {' '.join(ids)} (rank {rank}): {why}; cancelled "
                       f"{' '.join(waiting)}")
        return moved

    def _counted(self, drive: dict, main: str) -> list[str]:
        """The launches `max_running` counts: those of the commits `trains` checks now (every launch, before this tick's sync), the
        prefixes bisection checks, `main`'s own checks, and a former tip a re-cut left inside a train until a train above it, which
        contains it, runs its own check.  From then it runs on as a hedge (`land` lands it should that train fail) and takes no
        train's slot (CI-85)."""
        bisecting = {r.get("check") for r in drive.get("bisect", {}).values()}
        if self._tips is None:
            return [*drive["launched"], *drive.get("base", {})]
        pos, tips = self._pos, self._tips

        def above(c: str) -> bool:
            return any(pos[t] > pos[c] and t in drive["launched"] and self.verdict(t, main, drive).state == "running" for t in tips)

        return [*(c for c in drive["launched"] if c in self._tips or c in bisecting or c not in pos or not above(c)),
                *drive.get("base", {})]

    def _cleared(self, commit: str, main: str, drive: dict) -> bool:
        """Whether ``commit``'s train has passed its check: its verdict passed, or its check passed every step but the deferred
        ones and only its touched-suites gate is still pending."""
        v = self.verdict(commit, main, drive)
        if v.state == "passed":
            return True
        launch = drive["launched"].get(commit)
        return (v.state == "running" and "pytest" in self.defer and launch is not None
                and any(st in ACTIVE for st, *_ in self._states(launch["ids"]).values())
                and self._passed_but_deferred(launch["ids"]) is not None)

    def drop_hedges(self, drive: dict, members: list[Member], trains: list[int], main: str) -> None:
        """Cancel the live runs (its check's and its touched suites') of each launched commit `trains` no longer checks, a former tip
        a re-cut of `next` left inside a train, once a train above it, which contains it, has passed its check (`_cleared`), and
        drop its launch record, so its cancel (143) never reads as a kill to resend.  Until then it is a hedge: `land` lands the
        newest member with a passing check, tip or not.  Bisection's checks and the post-landing run's are kept (CI-85: a
        stale former tip held a check slot through the train above it)."""
        pos = {m.commit: k for k, m in enumerate(members)}
        tips = [members[i].commit for i in trains]
        run = (drive.get("postland") or {}).get("run") or {}
        keep = {r.get("check") for r in drive.get("bisect", {}).values()} | {run.get("tip")}
        busy = set(run.get("ids") or ()) | {rid for rec in drive.get("deferred", {}).values() for rid in rec["ids"]}
        for c, rec in list(drive["launched"].items()):
            if c in tips or c not in pos or c in keep or set(rec["ids"]) & busy:
                continue
            if (over := next((t for t in tips if pos[t] > pos[c] and self._cleared(t, main, drive)), None)) is None:
                continue
            live = [rid for rid, (st, *_) in self._states([*rec["ids"], *(rec.get("touched") or [])]).items()
                    if st in ACTIVE and st != "unknown"]
            if not live:
                continue
            why = f"{c[:12]} is no longer a train's tip on `{self.q.branch}`, and {over[:12]}'s train above it has passed its check"
            if self.dry_run or self.q.shadow:
                self.q.log(f"research queue drive: would cancel {' '.join(live)}: {why}")
                continue
            self.cancel(self.repo, live, f"drive: {why}")
            self._seen.update({rid: ("cancelled", SIGTERM_RC, None, None) for rid in live})
            del drive["launched"][c]
            drive.get("resent", {}).pop(c, None)
            self.q.log(f"research queue drive: cancelled {' '.join(live)}: {why}")

    def _cancel_orphans(self, drive: dict) -> None:
        """Cancel again each run whose cancel failed (`drive["orphans"]`, run id: why) while it is still queued or running; one
        that has ended, or that is cancelled now, leaves the list, so no run of a train drive dropped is left holding a line."""
        orphans = drive.get("orphans") or {}
        if not orphans:
            return
        live = [rid for rid, (st, *_) in self._states(list(orphans)).items() if st in ACTIVE and st != "unknown"]
        why = f"drive: a run of a train no longer on `{self.q.branch}`; an earlier cancel failed"
        stuck = (self.cancel(self.repo, live, why) or {}) if live else {}
        drive["orphans"] = {rid: orphans[rid] for rid in stuck}
        if gone := [r for r in live if r not in stuck]:
            self.q.log(f"research queue drive: cancelled {' '.join(gone)} on a retry: their trains left `{self.q.branch}`")

    def _launch(self, drive: dict, m: Member, main: str) -> tuple[list[str], str]:
        """Check ``m``'s commit (`train.sh`), with the suites its change reaches beside it when pytest is deferred, and record
        the launch: (the check's run ids, train.sh's output's tail)."""
        agreement = needs_agreement(self.repo, main, m.commit)
        started = self._start(m, main, agreement, self._rank(m.commit))
        self._record(drive, m, agreement, started)
        return started[0], started[1]

    def _rank(self, commit: str) -> int | None:
        """``commit``'s train's rank for the slot queue (`SLOT_RANK`): its depth on the first-parent line `main` moves along, so a
        rank keeps its order with ranks sent after `main` moved (24 Oct: d836's check, ranked 17 by its members above `main` at
        01:17Z, waited 3 h behind higher trains ranked 15 after the 01:38Z landing)."""
        depth = _out(self.repo, "rev-list", "--count", "--first-parent", commit)
        return int(depth) if depth.isdigit() else None

    def _start(self, m: Member, main: str, agreement: bool, rank: int | None) -> tuple[list[str], str, list[str] | None, str]:
        """Launch ``m``'s check and, with pytest deferred, the suites its change reaches once the check registered: (the check's
        ids, train.sh's tail, the touched run's ids or None, its output).  It records nothing, so a tick starts several at once."""
        with slot_rank(rank):
            ids, tail = self.launch(self.repo, m.commit, self.on, agreement)
        if not ids or "pytest" not in self.defer:
            return ids, tail, None, ""
        touched, out = self.launch_touched(self.repo, m.commit, self.on, main)
        return ids, tail, touched, out

    def _record(self, drive: dict, m: Member, agreement: bool, started: tuple) -> None:
        """Record what `_start` launched for ``m`` under ``drive["launched"]``."""
        ids, _, touched, out = started
        if not ids:
            return
        drive["launched"][m.commit] = {"ids": ids, "at": self.now(), "agreement": agreement}
        if touched is not None:
            drive["launched"][m.commit]["touched"] = touched
            if not touched:
                self.q.log(f"research queue drive: {m.commit[:12]}'s touched-suites run registered no run ({out[-200:]}); its "
                           "landing waits until one passes")

    def resend(self, drive: dict, m: Member, prs: str, v: Verdict,
               because: str = "no PR's change touches what failed, so no PR is ejected") -> None:
        """Check ``m``'s train again on ``on``'s machines, once: its failure was a kill and names no file, so no PR is at fault, or
        it is one run's word (``because``).  Its failed attempts stop counting (`verdict` passes over `resent`), so the new run's
        verdict is the train's, and with pytest deferred its touched suites run again beside it (`_launch`).  What the resend
        replaces is cancelled once it registers: a failed attempt still running under `--keep-going` (18 Oct: one held a check
        slot 4 h after its resend, until the OOM killer took it) and the old touched-suites run, of the ids drive's own launch
        record of the train holds, never another's."""
        main = _out(self.repo, "rev-parse", "origin/main")
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue drive: would resend {m.commit[:12]} ({prs}): {v.why}")
            return
        old = drive["launched"].get(m.commit) or {}
        ours = [rid for rid in v.attempt.split() if rid in old.get("ids", ())] + list(old.get("touched") or ())
        ids, tail = self._launch(drive, m, main)
        if not ids:
            self.refuse(drive, f"launch:{m.commit}", f"train.sh {m.commit[:12]} {self.on} registered no run on a resend: {tail}")
            return
        new = {*ids, *(drive["launched"][m.commit].get("touched") or ())}
        if live := [rid for rid, (st, *_) in self._states([r for r in ours if r not in new]).items()
                    if st in ACTIVE and st != "unknown"]:
            self.cancel(self.repo, live, f"drive: {m.commit[:12]} was resent as {' '.join(ids)}; this run has already failed, "
                                         "or the resend runs its touched suites again")
        drive.setdefault("resent", {})[m.commit] = v.attempt.split()
        self.q.log(f"research queue drive: resent {m.commit[:12]} ({prs}) as {' '.join(ids)}: {v.attempt} failed ({v.why}); "
                   f"{because}")

    # -- pre-warming a cold-forcing PR off `next`

    def warming(self, p) -> str | None:
        """Why ready PR ``p`` stays off `next` for now: its change makes circuit-check run cold (`queue.forces_cold`) or reruns
        every suite (`queue.reruns_suites`), and its head's pre-warm, one recorded check of that head (`prewarm`), hasn't ended.  Its passes then reach the trains it joins,
        which run warm, and no warm PR waits hours on it above it (16 Oct: #2158 and #2199, each at the bottom of `next`, made
        every train above them cold, until a lander held them by hand)."""
        if self._drive is None or "circuit-check" in self.defer:      # deferred, a cold circuit-check holds no landing up
            return None
        pw, key = self._drive.setdefault("prewarm", {}), f"{p.number}@{p.head}"
        if (rec := pw.get(key)) is None:
            if not self.q._reruns_all(p.number):
                return None
            rec = pw[key] = {"pr": p.number, "head": p.head}
        if rec.get("done"):
            return None
        ids = rec.get("ids")
        return (f"pre-warming its circuit-check off `{self.q.branch}`: " + (f"{' '.join(ids)} of {p.head[:12]}" if ids else
                "its check of the head launches when a check slot frees") + "; it joins once that check ends")

    def prewarm(self, drive: dict, running: int) -> int:
        """Launch and follow each pre-warm `warming` asked for (one `train.sh` check of the PR's head, in a check slot, while fewer
        than `max_running` checks run, and `PREWARM_MAX` at a time) and mark it done once it ends, passed or not: the train's check is the gate.  Returns
        how many run now, for the running count."""
        q = self.q
        heads = {f"{p.number}@{p.head}" for p in q.github.open_prs()}
        pw = drive["prewarm"] = {k: v for k, v in drive.get("prewarm", {}).items() if k in heads}
        active = 0
        for key, rec in pw.items():
            if rec.get("done"):
                continue
            if "circuit-check" in self.defer:                          # nothing waits on it: its slot goes to trains
                rec["done"] = True
                if live := [r for r, (st, *_) in self._states(rec.get("ids") or []).items() if st in ACTIVE]:
                    self.cancel(self.repo, live, "drive: circuit-check is deferred, so a pre-warm holds nothing up")
                continue
            if rec.get("ids"):
                states = self._states(rec["ids"])
                if any(st in ACTIVE for st, *_ in states.values()):
                    active += 1
                    continue
                if not all(rc == NOT_STARTED for _, rc, *_ in states.values()):
                    rec["done"] = True
                    q.log(f"research queue drive: #{rec['pr']}'s pre-warm {' '.join(rec['ids'])} ended; it joins `{q.branch}`")
                    continue
                rec.pop("ids")
            if running + active >= self.max_running or active >= PREWARM_MAX or self.dry_run or q.shadow:
                continue
            ids, tail = self.launch(self.repo, rec["head"], self.on, False)
            if not ids:
                rec["done"] = True
                q.log(f"research queue drive: #{rec['pr']}'s pre-warm registered no run ({tail[-200:]}); it joins cold")
                continue
            rec.update(ids=ids, at=self.now())
            active += 1
            q.log(f"research queue drive: pre-warming #{rec['pr']} ({rec['head'][:12]}): {' '.join(ids)}")
        return active

    # -- the per-PR landing, in shadow (`--per-pr-shadow N`; the module's last paragraph)

    def _say(self, text: str) -> None:
        self.q.log(f"research queue drive: per-PR shadow: {text}")

    def per_pr(self, drive: dict) -> str | None:
        """One tick of the per-PR shadow, under `drive["per_pr"]`; a failure of its own is logged, and the trains' tick goes on."""
        if not self.per_pr_shadow:
            return None
        head = _out(self.repo, "rev-parse", "HEAD")
        try:
            return self._per_pr(drive.setdefault("per_pr", {}))
        except Exception as e:  # noqa: BLE001  (a shadow never stops the trains)
            self._say(f"failed this tick ({type(e).__name__}: {e})")
            return None
        finally:
            _git(self.repo, "merge", "--abort", check=False)
            _git(self.repo, "checkout", "-q", "-f", "--detach", head, check=False)

    def _admitted(self) -> tuple[dict[int, PR], set[int]]:
        """(the open PRs against `main` drive admits now, by number; those it admits or ejected at their heads, whose records the
        shadow keeps: the trains' ejection is the trains' verdict, not the shadow's), each head fetched.  The pre-warm's hold
        only orders `next`, and doesn't apply."""
        q = self.q
        if self._prs is None:
            self._prs = {p.number: p for p in q.github.open_prs()}
        prs = [p for p in self._prs.values() if p.base == "main"]
        have = dict(line.split()[::-1] for line in _out(self.repo, "for-each-ref", "--format=%(objectname) %(refname)",
                                                            "refs/queue/pr/").splitlines())
        fresh = [p for p in prs if have.get(f"refs/queue/pr/{p.number}") == p.head]
        if stale := [p for p in prs if p not in fresh]:
            fresh += q.fetch_prs(stale)[0]
        warming, q.warming = q.warming, None
        try:
            entries = q.entries(fresh, q.load_state())
        finally:
            q.warming = warming
        ready = {n: e.pr for n, e in entries.items() if e.state == "ready"}
        return ready, set(ready) | {n for n, e in entries.items() if e.state == "refused" and e.why.startswith("ejected at")}

    def _slot_state(self, rid: str) -> dict:
        """Run ``rid``'s `slot.json` (its `state`, and `took` once a check slot took it), {} when it can't be read."""
        got = self.fetch(rid, [SLOT_FILE])
        with contextlib.suppress(OSError, ValueError, AttributeError, TypeError):
            return json.loads((got[0] / SLOT_FILE).read_text()) if got else {}
        return {}

    def _trains_waiting(self) -> list[str]:
        """The train checks of `next` waiting for a check slot: running, none of their twins in a slot, and one of them waiting.
        The per-PR shadow yields to them (CI-141: 21 Oct its ordinary jobs took vy-nebius-2's two slots while nothing waited, then
        held them an hour while trains queued behind)."""
        launched = (self._drive or {}).get("launched", {})
        live = self._states([r for v in launched.values() for r in v["ids"]])
        act = {c: [r for r in v["ids"] if (st := live.get(r, ("unknown",))[0]) in ACTIVE and st != "unknown"]
               for c, v in launched.items()}
        rids = [r for ids in act.values() for r in ids]
        slots = dict(zip(rids, _each(self._slot_state, rids)))
        return [c for c, ids in act.items() if ids and not any(slots[r].get("took") for r in ids)
                and any(slots[r].get("state") == "waiting" for r in ids)]

    def _per_pr_yield(self, recs: dict, waiting: list[str]) -> None:
        """Cancel each shadow check holding a check slot while ``waiting`` trains wait for one, and queue its PR again: the
        shadow measures, the trains land."""
        for rec in recs.values():
            if rec["phase"] != "checking" or not rec["checks"] or (c := rec["checks"][-1]).get("ended"):
                continue
            self._per_pr_took(rec, c)
            if not c.get("took"):
                continue                                    # an ordinary job waits behind every train's priority anyway
            self.cancel(self.repo, c["ids"], f"drive: per-PR shadow yields #{rec['pr']}'s check slot to the trains waiting "
                                             f"({' '.join(x[:12] for x in waiting)})")
            c.update(ended=self.now(), yielded=True)
            rec["phase"] = "queued"
            self._say(f"#{rec['pr']}'s check {' '.join(c['ids'])} yields its slot to {len(waiting)} waiting train(s); "
                      "it is checked again later")

    def _per_pr(self, sh: dict) -> str:
        recs, done = sh.setdefault("prs", {}), sh.setdefault("done", [])
        self._per_pr_recs = recs
        ready, kept = self._admitted()
        main = _out(self.repo, "rev-parse", "origin/main")
        waiting = self._trains_waiting()
        if waiting:
            self._per_pr_yield(recs, waiting)
        for key, rec in list(recs.items()):
            n = int(key)
            why = ("landed" if self._ancestor(rec["head"], main) else "closed" if n not in self._prs
                   else "new-head" if self._prs[n].head != rec["head"]
                   else "dropped" if n not in kept and rec["phase"] in PER_PR_LIVE else None)
            if why:
                self._per_pr_end(recs, done, key, why, main)
        for rec in list(recs.values()):
            self._per_pr_follow(rec, main)
        new = [p for n, p in ready.items() if str(n) not in recs and not self._ancestor(p.head, main)]
        for p in new:
            p_mark = self.q.marked_at(p.number, p.head) if self.q.marked_at else None
            recs[str(p.number)] = {"pr": p.number, "head": p.head, "owner": p.owner or "", "marked": _utc(p_mark),
                                   "phase": "queued", "new": True, "checks": [], "plans": []}
        order = sorted((r for r in recs.values() if r["phase"] == "queued"),
                       key=lambda r: (r["marked"] is None, r["marked"] or 0, r["pr"]))
        for rec in order:
            if waiting or self._per_pr_running() >= self.per_pr_shadow:
                break                                       # a train waits for a slot: no shadow check takes one
            self._per_pr_check(rec, main)
        for key in [k for k, r in recs.items() if r.pop("new", False) and r["phase"] == "queued" and not r["checks"]]:
            del recs[key]                                   # not sampled this tick: no record until a slot takes it
        phases = [r["phase"] for r in recs.values()]
        return (f"per-PR shadow: {phases.count('checking')} checking, {phases.count('planning')} planning, "
                f"{phases.count('would-land')} would land")

    def _per_pr_running(self) -> int:
        return sum(r["phase"] == "checking" or bool(r["checks"] and not r["checks"][-1].get("ended") and r["checks"][-1].get("ids"))
                   for r in self._per_pr_recs.values())

    def _per_pr_commit(self, rec: dict, main: str) -> tuple[str | None, str]:
        """(L, ""): ``rec``'s PR's head merged onto ``main`` as drive's sync merges it onto `next` (`queue._message`), or (None,
        why) when they conflict.  Its committer is the shadow's (`PER_PR_IDENTITY`): a commit equal to a train's would share
        that train's attempts, and the trains' verdicts with them."""
        pr = self._prs.get(rec["pr"]) or PR(rec["pr"], rec["head"], "", f"Lane {rec['owner']}")
        _git(self.repo, "checkout", "-q", "-f", "--detach", main)
        r = _git(self.repo, "merge", "--no-ff", "--no-edit", "-m", _message(pr), rec["head"], check=False, env=PER_PR_IDENTITY)
        if r.returncode == 0:
            return _out(self.repo, "rev-parse", "HEAD"), ""
        files = _out(self.repo, "diff", "--name-only", "--diff-filter=U").split()
        _git(self.repo, "merge", "--abort", check=False)
        return None, (f"conflicts with `main` {main[:12]} in {', '.join(files[:5])}" if files
                      else f"doesn't merge onto `main` {main[:12]}: {(r.stderr or r.stdout).strip()[-200:]}")

    def _per_pr_check(self, rec: dict, main: str) -> None:
        """Check ``rec``'s landing commit on the current ``main``: build it, push it as its `per-pr/N` branch, and launch
        `train.sh` on it as an ordinary job that never posts (`PER_PR_ENV`)."""
        n, now = rec["pr"], self.now()
        commit, why = self._per_pr_commit(rec, main)
        if commit is None:
            rec.update(phase="conflict", conflict={"at": now, "main": main, "why": why})
            self._say(f"#{n} ({rec['head'][:8]}) {why}: it would be told to its owner, and not checked")
            return
        agreement = needs_agreement(self.repo, main, commit)
        if self.dry_run or self.q.shadow:
            self._say(f"would check #{n} as {commit[:12]} on `main` {main[:12]}")
            return
        branch = f"{PER_PR_BRANCH}{n}"
        p = self.q._remote(self.repo, "push", "-q", "--no-verify", "origin", f"+{commit}:refs/heads/{branch}", ok=None)
        if p.returncode:
            self._say(f"pushing {branch} was refused ({(p.stderr or p.stdout).strip()[-200:]}); #{n} waits for the next tick")
            return
        with launch_env(PER_PR_ENV):
            ids, tail = self.launch(self.repo, commit, self.on, agreement)
        if not ids:
            self._say(f"#{n}'s check of {commit[:12]} registered no run ({tail[-200:]}); it waits for the next tick")
            return
        rec.setdefault("sampled", now)
        rec["checks"].append({"commit": commit, "main": main, "ids": ids, "at": now, "agreement": agreement})
        rec.update(phase="checking", branch=branch)
        self._say(f"checking #{n} ({rec['head'][:8]}) as {commit[:12]} = `main` {main[:12]} + its head: {' '.join(ids)}"
                  + (" with agreement" if agreement else "") + f" [{self._per_pr_running()} of {self.per_pr_shadow}]")

    def _per_pr_verdict(self, c: dict) -> Verdict:
        """The whole check's verdict of ``c`` (`verdict`, with no step deferred: `--defer`'s rule is `_per_pr_follow`'s)."""
        defer, self.defer = self.defer, frozenset()
        try:
            return self.verdict(c["commit"], c["main"], {"launched": {c["commit"]: c}})
        finally:
            self.defer = defer

    def _per_pr_follow(self, rec: dict, main: str) -> None:
        """Follow ``rec``'s last check to its end (past a would-land, which a deferred step may still fail) and its plan."""
        if rec["checks"] and not (c := rec["checks"][-1]).get("ended"):
            self._per_pr_follow_check(rec, c, main)
        if rec["phase"] == "planning":
            self._per_pr_follow_plan(rec, rec["plans"][-1], main)

    def _per_pr_took(self, rec: dict, c: dict) -> None:
        """When a check slot took ``c``'s run (its `slot.json`'s `took`; of twins, the one that ran), kept on ``c`` and, for its
        first check, on ``rec``: the shadow's checks wait as ordinary jobs, a wait a per-PR drive's wouldn't have."""
        for rid in [] if c.get("took") else c["ids"]:
            got, took = self.fetch(rid, [SLOT_FILE]), None
            with contextlib.suppress(OSError, ValueError, AttributeError):
                took = json.loads((got[0] / SLOT_FILE).read_text()).get("took") if got else None
            if took and (t := _utc(took)) is not None:
                c["took"] = t
                rec.setdefault("took", t)
                return

    def _per_pr_follow_check(self, rec: dict, c: dict, main: str) -> None:
        self._per_pr_took(rec, c)
        n, now, v = rec["pr"], self.now(), self._per_pr_verdict(c)
        if v.state == "running":
            if self.defer and not c.get("ready") and (rid := self._passed_but_deferred(c["ids"])):
                c["ready"] = {"at": now, "run": rid}
                self._per_pr_ready(rec, c, rid, main, f"{DEFERRED}, as a train lands")
            return
        c["ended"] = {"at": now, "state": v.state, "attempt": v.attempt, "why": v.why}
        if v.state == "none":
            if rec["phase"] == "checking":
                rec["phase"] = "queued"
            return
        if v.state == "passed":
            wl = rec.get("would_land") or {}
            if wl.get("commit") == c["commit"]:
                wl["full_at"] = now
            if not c.get("ready"):
                c["ready"] = {"at": now, "run": v.attempt}
                self._per_pr_ready(rec, c, v.attempt, main, "every step passed")
            return
        live = [rid for rid, (st, *_) in self._states(c["ids"]).items() if st in ACTIVE and st != "unknown"]
        if live and not (self.dry_run or self.q.shadow):
            self.cancel(self.repo, live, f"drive: per-PR shadow: #{n}'s check of {c['commit'][:12]} has failed")
        if (wl := rec.get("would_land") or {}).get("commit") == c["commit"] or rec["phase"] == "planning":
            (wl if rec["phase"] == "would-land" else rec)["revert"] = v.why
            self._say(f"#{n}'s check of {c['commit'][:12]} failed in a deferred step after it passed drive's rule ({v.why}): "
                      "the post-landing run would revert its landing")
            return
        if rec["phase"] != "checking":
            return
        if not rec.get("resent") and any(killed(d, v.why) for d in self._evidence(v)):
            rec.update(phase="queued", resent=True)
            self._say(f"#{n}'s check {v.attempt} was killed ({v.why}): it is checked again, once")
            return
        rec.update(phase="failed", failed={"at": now, "attempt": v.attempt, "why": v.why})
        self._say(f"#{n}'s check of {c['commit'][:12]} failed ({v.why}): it would be ejected and its owner told [not ejected]")

    def _per_pr_ready(self, rec: dict, c: dict, rid: str, main: str, rule: str) -> None:
        """``c`` passed by drive's rule: L would land on the `main` it was built on; on a moved one, its new L is planned."""
        if rec["phase"] != "checking":
            return
        if main == c["main"]:
            self._would_land(rec, c["commit"], main, f"check {rid}: {rule}")
        else:
            self._per_pr_replan(rec, main)

    def _per_pr_replan(self, rec: dict, main: str) -> None:
        commit, why = self._per_pr_commit(rec, main)
        if commit is None:
            rec.update(phase="conflict", conflict={"at": self.now(), "main": main, "why": why})
            self._say(f"#{rec['pr']} passed, but `main` moved to {main[:12]} and its head {why}: it would be told to its owner")
            return
        rec["plans"].append({"commit": commit, "main": main, "queued": self.now()})
        rec["phase"] = "planning"
        self._per_pr_follow_plan(rec, rec["plans"][-1], main)

    def _machine(self, rid: str) -> str | None:
        """The machine run ``rid`` ran on (its status row's `remote`), whose keys its passes are under."""
        got = self.fetch(rid, [])
        return ((got[1].get("remote") or {}).get("machine") or None) if got else None

    def _per_pr_follow_plan(self, rec: dict, p: dict, main: str) -> None:
        n, now = rec["pr"], self.now()
        if "dir" not in p:
            if not rec["checks"][-1].get("ended"):
                return                                      # its passes are cited from its pack, published when it ends
            live = sum(1 for r in self._per_pr_recs.values() if r["phase"] == "planning" and r["plans"]
                       and "dir" in r["plans"][-1] and not r["plans"][-1].get("ended"))
            if live >= self.per_pr_shadow:
                return
            if self.dry_run or self.q.shadow:
                self._say(f"would plan #{n}'s {p['commit'][:12]} on the moved `main` {main[:12]}")
                return
            runs = [x for c in rec["checks"] for x in [(c.get("ended") or {}).get("attempt") or (c.get("ready") or {}).get("run")]
                    if x and " " not in x]
            machines = [m for m in dict.fromkeys(self._machine(r) for r in runs[-1:]) if m] or self.on.split(",")
            root = self.q.workdir.parent / f"{self.q.workdir.name}.per-pr"
            p.update(self.plan_launch(self.repo, p["commit"], main, runs, machines, root / f"plan-{p['commit'][:12]}-{int(now)}"),
                     at=now)
            self._say(f"planning #{n}'s {p['commit'][:12]} on the moved `main` {main[:12]} by its keys, from "
                      f"{' '.join(runs) or 'no check of its own'}, on {', '.join(machines)}")
            return
        state, docs, tail = self.plan_state(p)
        if state == "running":
            return
        p["ended"] = now
        shutil.rmtree(p["dir"], ignore_errors=True)
        _git(self.repo, "worktree", "prune", check=False)
        whys, judged = [], None
        for doc in docs:
            cited, why, full = self.judge_plan(doc, p["commit"], p["main"])
            if why is None:
                judged = (cited, full)
                break
            whys.append(why)
        if judged is None:
            p.update(result="refused" if docs else "error", why=" | ".join(whys) or f"{state}, no plan: {tail[-300:]}")
            rec["phase"] = "queued"
            self._say(f"#{n}'s {p['commit'][:12]} isn't passed by its keys ({p['why'][:300]}): it is checked again")
            return
        p.update(result="passed", cited=judged[0], full=judged[1])
        if main != p["main"]:
            self._say(f"#{n}'s {p['commit'][:12]} passed by its keys, but `main` moved again, to {main[:12]}: planned again")
            self._per_pr_replan(rec, main)
            return
        self._would_land(rec, p["commit"], main, f"plan cites {', '.join(judged[0][:4])}" + ("" if judged[1] else
                                                                                               f" ({DEFERRED})"))

    def judge_plan(self, doc: dict, commit: str, base: str) -> tuple[list[str], str | None, bool]:
        """(the attempts cited, why not, whether every part passed) of a plan of ``commit`` by drive's rule: every part but the
        deferred steps' (`undeferred`) has a pass a recorded check's pack holds (`merge.plan_refusal`, the `--verdicts` gate's)."""
        from .merge import declared_command, plan_refusal, required_steps

        tree = _out(self.repo, "rev-parse", f"{commit}^{{tree}}")
        command = declared_command(self.repo, "check")
        need = {s: ps for s, ps in required_steps(self.repo, base, commit, "check").items() if s not in self.defer}
        cited, why = plan_refusal(doc, self.q.store, commit, tree, "check", command, need)
        if why is None or not self.defer:
            return cited, why, why is None
        cited, why = plan_refusal(undeferred(doc, self.defer), self.q.store, commit, tree, "check", command, need)
        return cited, why, False

    def _would_land(self, rec: dict, commit: str, main: str, via: str) -> None:
        now = self.now()
        rec.update(phase="would-land", would_land={"at": now, "commit": commit, "main": main, "via": via})
        since = [f"{(now - t) / 3600:.2f} h after its {what}" for what, t in (("ready mark", rec.get("marked")),
                                                                               ("first check launched", rec.get("sampled"))) if t]
        self._say(f"would land #{rec['pr']} ({rec['head'][:8]}) as {commit[:12]} on `main` {main[:12]} ({via}); "
                  f"{', '.join(since) or 'no times'} [not pushed]")

    def _per_pr_end(self, recs: dict, done: list, key: str, why: str, main: str) -> None:
        """End ``key``'s record (``why``: landed by the trains, closed, a new head, out of the queue), cancelling its live runs
        only, and keep it in ``done``."""
        rec, now = recs.pop(key), self.now()
        ids = [rid for c in rec["checks"] for rid in c.get("ids") or []]
        live = [rid for rid, (st, *_) in self._states(ids).items() if st in ACTIVE and st != "unknown"] if ids else []
        if live and not (self.dry_run or self.q.shadow):
            self.cancel(self.repo, live, f"drive: per-PR shadow: #{rec['pr']} {why}")
        if rec.get("branch") and why != "new-head" and not (self.dry_run or self.q.shadow):
            self.q._remote(self.repo, "push", "-q", "--no-verify", "origin", f":refs/heads/{rec['branch']}", ok=None)
        rec["ended"] = {"at": now, "why": why, "main": main}
        done.append(rec)
        del done[:-PER_PR_DONE]
        wl = rec.get("would_land")
        self._say(f"#{rec['pr']} ({rec['head'][:8]}) {why}" + (f" at `main` {main[:12]}" if why == "landed" else "")
                  + (f"; it would have landed {(now - wl['at']) / 3600:.2f} h before" if wl and why == "landed" else "")
                  + (f"; cancelled {' '.join(live)}" if live else ""))

    # -- the tick

    # -- the review landing (`land = "review"`, the module's "Review landing").  `drive["lean"]`: {the commit of `next` a Lean
    #    check ran on: {"ids" ([] with "launch", `detach`'s record, while its train.sh runs detached: `_resolve`), "at",
    #    "members": [[PR, head], ...] up to it, "tree", "stacked" or "hand" and "by" (`hand_lean`) when it is either,
    #    "state": None while it runs, else "passed" (with
    #    "run", the id that passed) or "failed" (with "why")}}, each dropped once its commit leaves `next`.  `drive["postland"]`
    #    also keeps "verified_as" ({"steps", "no_cache"} of the run that verified `verified`, or ci's `research queue baseline`)
    #    and "bisect" (a failed after-landing run's implicated landings, being bisected: `_bisect_step`)

    def _review(self, state: dict, drive: dict, dry_run: bool) -> str:
        """One tick of the review landing: sync `next` (held on its base while a Lean check of it runs, drive's or a hand one:
        `_running_lean`, as a train's check holds it), land its reviewed prefix (`land_reviewed`) once `main` has a baseline (`_unbaselined`), stacking a Lean check
        above a running one (`_stack_lean`), and launch the after-landing run, or the next run of a bisection (`_bisect_step`)."""
        q = self.q
        what: list[str] = []
        self._regen: dict[str, bool | None] = {}
        self._lean_of: dict[str, bool] = {}
        drive["tick"] = drive.get("tick", 0) + 1
        self._stop_trains(drive)
        self._follow_lean(drive)
        self._bisect_step(drive)
        running = self._running_lean(drive)
        # an ejection from a Lean check's prefix (`queue.lean_running`) rebuilds `next` only the tick after that check ends, so
        # the landing uses the check before the rebuild takes its commit off `next`; a stacked one (`_stack_lean`) reaches up
        # to `next`'s top, so one ejection anywhere on `next` would hold every sync for its whole run
        waits = next((c for c, rec in drive["lean"].items()
                      if not rec.get("stacked") and (c in running or rec.get("ended") == drive["tick"])
                      if any(not q.ready(n, h) for n, h in rec.get("members") or ())), None)
        if why := hold(self.repo, q.branch, lambda c, base: c in running):
            what.append(why)
        elif waits:
            what.append(f"`{q.branch}` waits for the Lean check of {waits[:12]} to end before an ejection rebuilds it")
        else:
            chain = q.chain()
            q.hold = max((i + 1 for i, m in enumerate(chain) if m.commit in running), default=0)
            q.frozen = {m.pr for m in chain[:q.hold]}
            q.sync(dry_run)
            state = {**q.load_state(), "drive": drive}
            for n, c in state.get("conflicts", {}).items():
                self.refuse(drive, f"conflict:{n}@{c['head']}", f"#{n} at {c['head'][:8]} is out of `{q.branch}`: {c['why']}",
                            c.get("handle"))
        landed, why = None, self._unbaselined(drive)
        if why is None:
            landed, why = self.land_reviewed(drive, dry_run)
            self._stack_lean(drive, dry_run)
        what.insert(0, f"landed #{landed.pr}" if landed else why)
        self._sweep(drive, dry_run)
        self.start_postland(drive)
        if not dry_run:
            state["ejected"] = {**state.get("ejected", {}), **q.load_state().get("ejected", {})}
            q.save_state(state)
        q.log(f"research queue drive: {'; '.join(what)}")
        return "; ".join(what)

    def _running_lean(self, drive: dict) -> set[str]:
        """The commits of `next` a Lean check runs at, which the sync keeps in place with the members below them (`queue.hold`,
        `frozen`; `--first` goes above them): drive's own (`drive["lean"]`), and any other run of its run registry whose steps
        include lean-audit (`lean_runs`: a hand `train.sh COMMIT MACHINES --deferred lean-audit`) while `research status` reads
        it live (26 Oct 07:16Z: a `--first` rebuild of `next` threw away ci's two hand checks).  A registry that can't be read
        adds none."""
        running = {c for c, rec in drive["lean"].items() if rec.get("state") is None}
        mine = {r for rec in drive["lean"].values() for r in rec.get("ids") or ()}
        try:
            found = self.lean_runs({m.commit for m in self._chain()[1]} - running, self.now() - LEAN_RUN_WINDOW_S)
        except Exception as e:  # noqa: BLE001  (holds nothing: drive's own Lean checks still hold `next`)
            self.q.log(f"research queue drive: the run registry's Lean checks can't be read ({type(e).__name__}: {e}); taken as none")
            return running
        held = {c: live for c, ids in found.items()
                if (live := [r for r, (st, *_) in self._states([r for r in ids if r not in mine]).items()
                             if st in ACTIVE and st != "unknown"])}
        if held:
            self.q.log("research queue drive: Lean checks drive didn't launch hold `next` up to them: "
                       + ", ".join(f"{c[:12]} ({' '.join(ids)})" for c, ids in held.items()))
        return running | set(held)

    def _baselined(self, pl: dict) -> bool:
        """Whether `verified` ran the after-landing run as the review landing runs it, under `after_landing_no_cache`
        (`verified_as`: the steps it passed, every step when green, fewer when `main` was red then, `_red_baseline`), or ci
        recorded it green (`research queue baseline`, either flag)."""
        vs = pl.get("verified_as")
        return bool(vs) and vs.get("no_cache") in (None, self.after_landing_no_cache)

    def _unbaselined(self, drive: dict) -> str | None:
        """Why no review landing happens now, or None: a bisection of a failed after-landing run is going (`_bisect_step`), or
        `main` has no baseline (`_baselined`), so a failure after a landing couldn't be told from one `main` had before it.  The
        review landing's first run is then of `main` itself (`start_postland`), once any post-landing run still going has ended
        (check mode's last); ci records a green `main` by hand with `/data/infra/drive.sh queue baseline SHA --by ci`."""
        main = _out(self.repo, "rev-parse", "origin/main")
        pl = drive.setdefault("postland", {"verified": main, "pending": [], "run": None})
        if b := pl.get("bisect"):
            return (f"nothing lands while drive bisects the failed after-landing run of `main` {b['tip'][:12]} "
                    f"({', '.join(b['steps'])})")
        if self._baselined(pl):
            return None
        if run := pl.get("run"):
            return (f"nothing lands before the post-landing run {_ids(run)} of `main` {run['tip'][:12]} ends and an "
                    "after-landing run of `main` itself has run (its baseline)")
        return f"nothing lands before an after-landing run of `main` {main[:12]} itself has run (its baseline)"

    def _bisect_step(self, drive: dict) -> None:
        """Bisect the landings a failed after-landing run implicates (`_postland_failed`, `postland["bisect"]`): check the commit
        of the middle one not yet judged with the failed steps alone (`--deferred STEPS`, the run's flags), each that passes
        clearing it and those below, each that fails blaming it and those above, until one fails and the one below it passes
        (or `verified` did): that PR is reverted (`_revert_landed`).  When even the newest passes, none of them broke it:
        nothing is reverted, `CI` is told.  A run fails a landing only when it failed one of those steps; one never started or
        stopped runs again, and one that ends otherwise (killed, lost, exit 0 without their records, a failure elsewhere) runs
        once more, then ends the bisection with nothing reverted."""
        if not (pl := drive.get("postland")) or not (b := pl.get("bisect")):
            return
        cands, tip, base = [Member(*x) for x in b["cands"]], b["tip"], b["base"]
        key, steps = f"postland:{tip}", ", ".join(b["steps"])
        head = (f"the after-landing run of `main` {tip[:12]} failed ({b['why']}) in {steps}; bisected "
                f"{' '.join(f'#{x.pr}' for x in cands)}, landed since {base[:12]}")
        if run := b.get("run"):
            c = cands[run["i"]].commit
            if (tail := self._resolve(run, f"bisecting `main` {tip[:12]}'s failed {steps}: #{cands[run['i']].pr}'s landing "
                                           f"{c[:12]}")) is not None:
                b["run"] = None
                self.refuse(drive, f"launch:bisect:{c}", f"train.sh {c[:12]} {self.on} --deferred {','.join(b['steps'])} "
                                                         f"registered no run: {tail}")
                return
            if "launch" in run:
                return
            states = self._states(run["ids"])
            lost = self._lost(run, states)
            if not lost and any(st in ACTIVE for st, *_ in states.values()):
                return
            b["run"] = None
            rcs = [rc for _, rc, *_ in states.values()]
            if not lost and any(rc == 0 for rc in rcs) and not self._unrun(run["ids"], b["steps"]):
                b["good"] = run["i"]
            elif not lost and all(rc in (NOT_STARTED, SIGTERM_RC) for rc in rcs):
                pass
            elif not lost and not any(rc == 0 for rc in rcs) and set(self._failing(run)) & set(b["steps"]):
                b["bad"] = run["i"]
            elif not b.get("unclear"):
                b["unclear"] = True                     # killed, lost, or no failed step of its own read: once more
            else:
                pl["failed"] = tip
                del pl["bisect"]
                self.refuse(drive, key, f"{head}: its run {' '.join(run['ids'])} of {cands[run['i']].commit[:12]} ended twice "
                                        f"neither passing nor failing {steps} ({'lost' if lost else ', '.join(map(str, rcs))}). "
                                        "Reverted nothing; find the culprit by hand")
                return
        if b["bad"] - b["good"] == 1:
            if b["bad"] == len(cands):
                pl["failed"] = tip
                del pl["bisect"]
                self.refuse(drive, key, f"{head}: each of their landings passes {steps}, so none of them broke it. Reverted "
                                        "nothing; fix `main`")
                return
            members = self.landed_since(base, tip)
            one = next((x for x in members if x.commit == cands[b["bad"]].commit), cands[b["bad"]])
            below = f"#{cands[b['good']].pr}'s landing" if b["good"] >= 0 else f"`main` {base[:12]}"
            if self._revert_landed(drive, pl, tip, base, members, one, [one], steps, head, key,
                                   blame=f"{steps} fails from #{one.pr}'s landing {one.commit[:12]} on and passes on {below}"):
                del pl["bisect"]
            return
        mid = (b["good"] + b["bad"]) // 2
        c = cands[mid].commit
        extra = {"agreement": "lean-agreement" in b["steps"], "no_cache": b["no_cache"], "no_regenerate": True}
        if self.dry_run or self.q.shadow:
            self.q.log(f"research queue drive: would launch train.sh {c[:12]} {self.on} --deferred {','.join(b['steps'])} to bisect")
            return
        with slot_rank(None):
            rec, tail = self._started(self.launch_postland(self.repo, c, self.on, ",".join(b["steps"]), **extra))
        if rec is None:
            self.refuse(drive, f"launch:bisect:{c}", f"train.sh {c[:12]} {self.on} --deferred {','.join(b['steps'])} registered no "
                                                     f"run: {tail}")
            return
        b["run"] = {**rec, "i": mid, "at": self.now()}
        self.q.log(f"research queue drive: bisecting `main` {tip[:12]}'s failed {steps}: #{cands[mid].pr}'s landing "
                   f"{c[:12]}: {_ids(rec)}")
        self.q.save_state({**self.q.load_state(), "drive": drive})

    def _stop_trains(self, drive: dict) -> None:
        """Cancel the train checks a drive landing on checks left running: the review landing launches none."""
        ids = [rid for v in (*drive["launched"].values(), *drive.get("base", {}).values()) for rid in (*v["ids"], *(v.get("touched") or []))]
        if not ids or self.dry_run or self.q.shadow:
            return
        if live := [rid for rid, (st, *_) in self._states(ids).items() if st in ACTIVE and st != "unknown"]:
            self.cancel(self.repo, live, "drive: `land = \"review\"` lands on reviews and checks after landing; no train check lands")
            self.q.log(f"research queue drive: cancelled the train checks {' '.join(live)}: the review landing launches none")
        drive["launched"], drive["base"] = {}, {}

    def land_reviewed(self, drive: dict, dry_run: bool) -> tuple[Member | None, str]:
        """Land `next`'s longest prefix whose every member has a land verdict at its head and no hold (`queue.unreviewed`, by none
        of `_owners`), cut below a Lean change no passed Lean check holds (`_lean_gate`) and below a member whose Python doesn't
        compile (`_compiles`): a fast-forward of `main` to its top, through `_land`.  (the member landed or None, what the tick
        did)"""
        q = self.q
        base, chain = self._chain()
        if not chain:
            return None, "nothing queued"
        main = _out(self.repo, "rev-parse", "origin/main")
        if main != base:
            return None, f"`main` {main[:12]} moved off `{q.branch}`'s base {base[:12]}: it lands once a sync rebuilds `{q.branch}`"
        self._note_marks(drive, chain)
        k, why = 0, None
        for m in chain:
            if not q.ready(m.pr, m.head):                # an eject writes `ready = false`; a re-mark at that head is newer
                why = "its ready mark is withdrawn: it leaves `next` at the next rebuild"
                break
            if (why := unreviewed(self.reviews(m.pr, m.head), self._owners(m))) is not None:
                break
            k += 1
        stop = f"#{chain[k].pr} at {chain[k].head[:8]}: {why}" if k < len(chain) else ""
        if not k:
            return None, f"nothing reviewed lands: {stop}"
        t, rid = self._lean_gate(drive, chain, k, dry_run)
        if t >= 0:
            t = self._compiles(drive, chain, base, t)
        if t < 0:
            return None, f"nothing lands: #{chain[0].pr} at {chain[0].head[:8]} waits (see above)"
        m, members = chain[t], chain[: t + 1]
        prs = " ".join(f"#{x.pr}" for x in members)
        v = Verdict("passed", rid, REVIEWED + (f"; its Lean changes' Lean check {rid}" if rid else ""))
        if dry_run or q.shadow:
            q.log(f"research queue drive: would land {prs} ({m.commit[:12]}, {v.why})" + (" [shadow]" if q.shadow else ""))
            return m, f"would land #{m.pr}"
        scratch = Path(tempfile.mkdtemp(prefix="drive-locks-"))
        try:
            got = self._land(drive, m, members, prs, v, base, main, [], scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return got, "" if got else f"{prs} reviewed but not landed (see above)" + (f"; {stop}" if stop else "")

    def _owners(self, m: Member) -> set[str | None]:
        """Whose verdict on ``m`` doesn't count (`queue.review_owners`): its merge's `Owner`, that owner's handle (`_owner`), and
        who wrote its ready mark."""
        return {m.owner, self._owner(m), self.ready_by(m.pr, m.head)}

    def _lean_change(self, m: Member) -> bool:
        if m.commit not in self._lean_of:
            self._lean_of[m.commit] = lean_change(self.repo, m.commit)
        return self._lean_of[m.commit]

    def _lean_gate(self, drive: dict, chain: list[Member], k: int, dry_run: bool) -> tuple[int, str]:
        """(the highest index of ``chain[:k]`` that may land, the Lean check whose regenerated locks land with it or another that
        holds its Lean changes, or ""): every Lean change below it (`queue.lean_change`) has a passed lean-audit of a commit of
        `next` at or above it (`_cover`); one whose check regenerated locks lands exactly at that commit, its locks on top
        (`_land`).  For the lowest Lean change without one, a Lean check is launched, or waited for (`_lean_check`).  Nothing at
        or above a commit whose Lean check failed lands, and when no Lean change's own commit is that one, `CI` is told."""
        lean = drive["lean"]
        self._rekey_lean(drive, chain)
        pos = {m.commit: i for i, m in enumerate(chain)}
        for c in [c for c, rec in lean.items() if c not in pos and rec.get("state") is not None]:
            del lean[c]                                         # of a `next` the sync rebuilt (a running one holds it)
        bad = min((pos[c] for c, rec in lean.items() if rec.get("state") == "failed" and c in pos), default=k)
        t, rid = min(k, bad) - 1, ""
        for i in range(k):
            if i > bad:
                break
            if not self._lean_change(chain[i]):
                continue
            if (got := self._cover(drive, chain, i, k)) is None:
                self._lean_check(drive, chain, k, i, dry_run)
                return min(i - 1, t), rid
            commit, run = got
            if (regen := self._regenerated(drive, run)) is None:
                return min(i - 1, t), rid
            if regen and lean.get(commit, {}).get("rekeyed_from"):
                # `_land` builds the locks on ``commit``, and its lock gate takes only a run of that very commit
                del lean[commit]
                self._lean_check(drive, chain, k, i, dry_run)
                return min(i - 1, t), rid
            if regen:
                if pos[commit] > t:
                    if bad == k:
                        self._lean_check(drive, chain, k, i, dry_run)
                    return min(i - 1, t), rid
                return pos[commit], run
            rid = run
        if bad < k and not self._lean_change(chain[bad]):
            c, rec = chain[bad].commit, lean[chain[bad].commit]
            hi = max((j for j, m in enumerate(chain[:bad]) if self._lean_change(m)), default=-1)
            if hi < 0 or any(hi <= pos[x] < bad for x, r in lean.items() if r.get("state") == "passed" and x in pos):
                del lean[c]                 # every Lean change below it passed below it: what broke is the after-landing run's
                self.refuse(drive, f"lean-dropped:{c}", f"the Lean check {_ids(rec)} of `{self.q.branch}` {c[:12]} "
                                                     f"(#{chain[bad].pr}'s merge, which changes no Lean) failed ({rec.get('why')}), "
                                                     "and each Lean change below it passed a Lean check below it: drive drops it, "
                                                     "and the after-landing run checks the rest")
                return self._lean_gate(drive, chain, k, dry_run)
            self.refuse(drive, f"lean-next:{c}", f"the Lean check {_ids(rec)} of `{self.q.branch}` {c[:12]} (#{chain[bad].pr}"
                                                 f"'s merge, which changes no Lean) failed ({rec.get('why')}), and each Lean change "
                                                 "below it has a passed Lean check: it and every PR above it wait until it leaves "
                                                 f"`{self.q.branch}`. Find the PR that broke it and hold it (`research queue review "
                                                 "N --hold REASON`)")
        return t, rid

    def _cover(self, drive: dict, chain: list[Member], i: int, k: int) -> tuple[str, str] | None:
        """(a commit of `next` at or above ``chain[i]``, its passed lean-audit's run): drive's own Lean checks first (one of
        ``chain[:k]``, the highest), then any recorded run in the store whose lean-audit passed on a commit of ``chain[i:k]`` (a
        full check, or a hand `train.sh COMMIT MACHINES --deferred lean-audit`).  A store that can't be read covers nothing."""
        pos = {m.commit: j for j, m in enumerate(chain)}
        ok = sorted(((pos[c], c, rec["run"]) for c, rec in drive["lean"].items() if rec.get("state") == "passed" and pos.get(c, -1) >= i
                     and not any(self._lean_change(m) for m in chain[k:pos[c] + 1])),      # what lands is the tree it checked
                    key=lambda x: (x[0] >= k, -x[0] if x[0] < k else x[0]))
        if ok:
            return ok[0][1], ok[0][2]
        from .merge import step_status

        try:
            for j in range(k - 1, i - 1, -1):
                rows = self.q.store.index.sql("SELECT id FROM attempts WHERE source_commit = ? AND source_dirty = 0 AND state = 'done' "
                                              "AND rc = 0 ORDER BY start_utc DESC", (chain[j].commit,))
                if rid := next((r[0] for r in rows if step_status(self.q.store, r[0], LEAN_STEP) == "passed"), None):
                    return chain[j].commit, rid
        except Exception as e:  # noqa: BLE001  (not covered: drive's own Lean check settles it)
            self.q.log(f"research queue drive: the store's Lean checks of #{chain[i].pr} can't be read ({type(e).__name__}: {e}); "
                       "taken as none")
        return None

    def _regenerated(self, drive: dict, run: str) -> bool | None:
        """Whether Lean check ``run`` regenerated lock records (`_locks`), None when that can't be read (told to `CI`)."""
        if not self.regenerate_locks:
            return False
        if run in self._regen:
            return self._regen[run]
        scratch = Path(tempfile.mkdtemp(prefix="drive-locks-"))
        try:
            _, locks, held = self._locks(Verdict("passed", run), scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if held:
            self.refuse(drive, f"lean-locks:{run}", f"the Lean check {run} holds lock changes on `{self.q.branch}`, which wait: {held}")
        self._regen[run] = None if held else bool(locks)
        return self._regen[run]

    def _lean_check(self, drive: dict, chain: list[Member], k: int, i: int, dry_run: bool) -> None:
        """A Lean check for ``chain[i]``, the lowest Lean change no passed one holds: wait for a running one of a commit at or
        above it; after a failed one of its own commit, tell `CI` and the owner, and wait for that commit to leave `next` (a new
        head, a hold, an ejection); after a failed one above it, check its own commit, so a PR is blamed only on its own
        check; else launch one of the reviewed prefix's top, ``chain[k - 1]``, capped below the lowest member above ``chain[i]``
        that came onto `next` within `LEAN_SETTLE_TICKS` ticks (`_churned`), so churn above doesn't take it off `next`, and cut
        to share the Lean changes up to it with the Lean checks still free (`lean_cut`), lowest first.  One running below
        stays; `_stack_lean` stacks one above it for the Lean changes it doesn't hold."""
        q, m, lean = self.q, chain[i], drive["lean"]
        pos = {x.commit: j for j, x in enumerate(chain)}
        mine = {c: rec for c, rec in lean.items() if pos.get(c, -1) >= i}
        if c := next((c for c, rec in mine.items() if rec.get("state") is None), None):
            q.log(f"research queue drive: #{m.pr}'s Lean change waits for the Lean check {_ids(lean[c])} of {c[:12]}")
            return
        if (rec := lean.get(m.commit)) and rec.get("state") == "failed":
            self.refuse(drive, f"lean:{m.commit}", f"#{m.pr} at {m.head[:8]} changes Lean, and the Lean check {_ids(rec)} "
                        f"of its own commit {m.commit[:12]} on `{q.branch}` failed ({rec.get('why')}): it and every PR above it wait. "
                        f"Push a fix (a new head checks again), or hold it (`research queue review {m.pr} --hold REASON`)",
                        self._owner(m))
            return
        cap = next((j for j in range(i + 1, k) if self._churned(drive, chain[j])), k)
        free = self.max_lean_checks - sum(rec.get("state") is None for rec in lean.values())
        t = i if any(rec.get("state") == "failed" for rec in mine.values()) else lean_cut(chain, i, cap - 1, self._lean_change, free)
        self._launch_lean(drive, chain, t, f"#{m.pr}'s Lean change", dry_run)

    def _stack_lean(self, drive: dict, dry_run: bool) -> None:
        """While a Lean check runs, stack one above it on `next`, reviewed or not, for the Lean changes it doesn't hold
        (`lean_stack`, at most `max_lean_checks` running), so they don't wait for it to pass and then for a check of their own.
        The lower one runs on.  An ejection from a stacked check's prefix rebuilds `next` at once (`_review`), cancelling it."""
        base, chain = self._chain()
        if not chain or base != _out(self.repo, "rev-parse", "origin/main"):
            return
        self._rekey_lean(drive, chain)
        self._note_marks(drive, chain)
        t = lean_stack(chain, drive["lean"], self._lean_change, lambda m: self._churned(drive, m), self.max_lean_checks)
        if t is not None:
            self._launch_lean(drive, chain, t, f"the Lean changes up to #{chain[t].pr}, above the running one", dry_run,
                              stacked=True)

    def _launch_lean(self, drive: dict, chain: list[Member], t: int, what: str, dry_run: bool, stacked: bool = False,
                     by: str | None = None) -> None:
        """Launch the Lean check of ``chain[t]``, for ``what``, and save it at once, so a tick that fails after it doesn't launch
        it again.  ``by``: a hand check's (`hand_lean`), recorded with `hand` and who asked, and refused to its caller."""
        q, top = self.q, chain[t]
        if dry_run or q.shadow:
            q.log(f"research queue drive: would launch train.sh {top.commit[:12]} {self.on} --deferred {LEAN_STEP} for {what}")
            return
        with slot_rank(None):                  # a running check below stays: its prefix is on `next`, and it may land it
            rec, tail = self._started(self.launch_lean(self.repo, top.commit, self.on))
        if rec is None:
            why = f"train.sh {top.commit[:12]} {self.on} --deferred {LEAN_STEP} registered no run: {tail}"
            if by:
                raise QueueError(f"lean-check: {why}")
            self.refuse(drive, f"launch:lean:{top.commit}", why)
            return
        drive["lean"][top.commit] = {**rec, "at": self.now(), "members": [[x.pr, x.head] for x in chain[: t + 1]],
                                     "tree": self._tree(top.commit), "state": None, **({"stacked": True} if stacked else {}),
                                     **({"hand": True, "by": by} if by else {})}
        q.log(f"research queue drive: the Lean check of {top.commit[:12]} for {what}: {_ids(rec)}")
        q.save_state({**q.load_state(), "drive": drive})

    def hand_lean(self, commit: str, by: str) -> dict:
        """`research queue lean-check COMMIT`: a Lean check of a commit of `next` that ``by`` chose, launched as drive's own
        (`_launch_lean`, detached) and in `drive["lean"]` from its launch, marked `hand`, so the hold on `next`,
        `max_lean_checks`, the cut points, `_follow_lean` and `_cover` take it as drive's own before its train.sh registers a
        run (26 Oct 10:36Z: a tick in those minutes reordered `next`, non-Lean first, and threw away S1's hand check at
        19a382ca).  Refused for a commit that isn't a member's merge on `next`, or one a running Lean check at or above it
        holds.  Run between ticks, under drive's tick lock (`/data/infra/drive.sh queue lean-check COMMIT`); returns the
        record."""
        q = self.q
        if q.shadow:
            raise QueueError("lean-check: the shadow launches nothing")
        q._clone()
        q.fetch()
        full = _git(self.repo, "rev-parse", "--verify", "-q", f"{commit}^{{commit}}", check=False).stdout.strip()
        base, chain = self._chain()
        pos = {m.commit: i for i, m in enumerate(chain)}
        if full not in pos:
            raise QueueError(f"lean-check: {commit} is not a commit of `{q.branch}` (a member's merge above {base[:12]}): "
                             f"`git log --first-parent origin/main..origin/{q.branch}` lists them")
        state = q.load_state()
        drive = state.setdefault("drive", {})
        lean = drive.setdefault("lean", {})
        self._rekey_lean(drive, chain)
        if c := next((c for c, rec in lean.items() if rec.get("state") is None and pos.get(c, -1) >= pos[full]), None):
            raise QueueError(f"lean-check: the running Lean check {_ids(lean[c])} of {c[:12]} (#{chain[pos[c]].pr}) already "
                             f"holds {full[:12]} (#{chain[pos[full]].pr}): its pass covers it")
        self._launch_lean(drive, chain, pos[full], f"{by}'s hand check", False, by=by)
        return lean[full]

    def _tree(self, commit: str) -> str | None:
        return _git(self.repo, "rev-parse", "-q", "--verify", f"{commit}^{{tree}}", check=False).stdout.strip() or None

    def _rekey_lean(self, drive: dict, chain: list[Member]) -> None:
        """Move each Lean check whose commit left `next` to the commit of the rebuilt `next` that holds the same PRs at the same
        heads in the same order up to it, when that commit's tree is the one it checked (the same heads merged onto the same
        `main`): it still covers them, and a running one keeps running (24 Oct: a re-mark low on `next` rebuilt it, and the
        checks r20261024-144858-46af and r20261024-165000-3c72 were cancelled mid-audit; nothing landed for 3 h)."""
        lean, pos = drive.get("lean") or {}, {m.commit for m in chain}
        for c, rec in list(lean.items()):
            n = len(rec.get("members") or ())
            if c in pos or not n or n > len(chain) or [[m.pr, m.head] for m in chain[:n]] != [list(x) for x in rec["members"]]:
                continue
            new = chain[n - 1].commit
            if new not in lean and (tree := rec.get("tree") or self._tree(c)) and tree == self._tree(new):
                lean[new] = {**lean.pop(c), "tree": tree, "rekeyed_from": c}
                self.q.log(f"research queue drive: the Lean check {_ids(rec)} of {c[:12]} holds {new[:12]} of the "
                           f"rebuilt `{self.q.branch}`: the same PRs at the same heads, the same tree")

    def _note_marks(self, drive: dict, chain: list[Member]) -> None:
        """Record the tick each member of `next` was first seen at its head (`drive["marks"]`), for `_churned`; on drive's first
        tick of the review landing, every member counts as settled."""
        old, tick = drive.get("marks"), drive.get("tick", 0)
        drive["marks"] = {str(m.pr): (old[str(m.pr)] if old and (old.get(str(m.pr)) or [None])[0] == m.head
                                      else [m.head, tick if old is not None else tick - LEAN_SETTLE_TICKS]) for m in chain}

    def _churned(self, drive: dict, m: Member) -> bool:
        """Whether ``m`` came onto `next` at its head within the last `LEAN_SETTLE_TICKS` ticks (`_note_marks`)."""
        seen = (drive.get("marks") or {}).get(str(m.pr))
        return bool(seen) and seen[0] == m.head and drive.get("tick", 0) - seen[1] < LEAN_SETTLE_TICKS

    def _follow_lean(self, drive: dict) -> None:
        """Judge each Lean check that ended: passed when a twin exited 0 with lean-audit passed in its steps.jsonl, gone (launched
        again) when none started, else failed.  A check whose commit is no longer on `next`, nor a commit of it with the same
        prefix and tree (`_rekey_lean`), is dropped, and cancelled, once its train.sh has printed its run ids (`_resolve`); one
        whose train.sh registered no run is dropped, told to `CI`, and launched again."""
        lean = drive.setdefault("lean", {})
        chain = self._chain()[1]
        self._rekey_lean(drive, chain)
        on_next = {m.commit for m in chain}
        for c, rec in list(lean.items()):
            if rec.get("state") is None and (tail := self._resolve(rec, f"the Lean check of {c[:12]}")) is not None:
                if rec.get("resend"):
                    rec.update(state="failed", why=f"its resend registered no run: {tail}")
                else:
                    del lean[c]
                    self.refuse(drive, f"launch:lean:{c}", f"train.sh {c[:12]} {self.on} --deferred {LEAN_STEP} registered no run: "
                                                           f"{tail}")
                continue
            if "launch" in rec:
                continue
            if c not in on_next:
                if rec.get("state") is None and not (self.dry_run or self.q.shadow):
                    if live := [rid for rid, (st, *_) in self._states(rec["ids"]).items() if st in ACTIVE]:
                        self.cancel(self.repo, live, f"drive: the Lean check of {c[:12]} is of a commit no longer on `{self.q.branch}`")
                del lean[c]
                continue
            if rec.get("state") is not None:
                continue
            states = self._states(rec["ids"])
            lost = self._lost(rec, states)
            if not lost and any(st in ACTIVE for st, *_ in states.values()):
                continue
            rcs = [rc for _, rc, *_ in states.values()]
            if (lost or (0 not in rcs and any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC)) for rc in rcs))) \
                    and not rec.get("resend") and not (self.dry_run or self.q.shadow):
                with slot_rank(None):
                    again, tail = self._started(self.launch_lean(self.repo, c, self.on))
                if again:                               # killed or lost, a machine's fault: once more, as `follow_postland`
                    rec.update(**again, at=self.now(), resend=True)
                    self.q.save_state({**self.q.load_state(), "drive": drive})
                    self.q.log(f"research queue drive: the Lean check of {c[:12]} was {'lost' if lost else 'killed'}: again, "
                               f"once: {_ids(again)}")
                    continue
                rec.update(state="failed", why=f"its resend registered no run: {tail}")
                continue
            rec["ended"] = drive.get("tick")
            if lost:
                rec.update(state="failed", why=f"{' '.join(rec['ids'])} lost on its resend")
            elif ran := [rid for rid, (_, rc, *_) in states.items() if rc == 0]:
                if self._unrun(ran, [LEAN_STEP]):
                    rec.update(state="failed", why=f"{' '.join(ran)} exited 0 without passing {LEAN_STEP}")
                else:
                    rec.update(state="passed", run=ran[0])
                    self.q.log(f"research queue drive: the Lean check {ran[0]} of {c[:12]} passed")
            elif all(rc in (NOT_STARTED, SIGTERM_RC) for _, rc, *_ in states.values()):
                del lean[c]
            else:
                rec.update(state="failed", why=", ".join(f"{rid} {st} rc={rc}" for rid, (st, rc, *_) in states.items()))

    def _compiles(self, drive: dict, chain: list[Member], base: str, t: int) -> int:
        """The highest index at or below ``t`` whose landing compiles: every .py file ``base``..``chain[t]`` leaves, compiled as
        `python -m py_compile` does.  The member that last changed a file that doesn't is ejected (`_eject`), and what's below
        it lands."""
        top = chain[t].commit
        bad = []
        for f in sorted(changed(self.repo, base, top)):
            if not f.endswith(".py") or (r := _git(self.repo, "show", f"{top}:{f}", check=False)).returncode:
                continue
            try:
                compile(r.stdout, f, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as e:
                bad.append((f, f"{type(e).__name__}: {e}"))
        if not bad:
            return t
        by: dict[int, list[str]] = {}
        for f, err in bad:
            i = max(i for i in range(t + 1) if f in changed(self.repo, f"{chain[i].commit}^1", chain[i].commit))
            by.setdefault(i, []).append(f"{f}: {err}")
        u = min(by)
        prs = " ".join(f"#{x.pr}" for x in chain[: t + 1])
        self._eject(drive, f"compile:{chain[u].commit}", f"`{self.q.branch}` {top[:12]} ({prs}) doesn't compile, the gate before a "
                    "review landing", chain[t], prs, Verdict("failed", "py_compile"), [chain[i] for i in sorted(by)],
                    "; ".join(f"#{chain[i].pr}: {', '.join(errs)}" for i, errs in sorted(by.items())))
        return u - 1

    def _after_landing(self, drive: dict, tip: str, base: str) -> None:
        """A review landing at ``tip`` on ``base`` awaits the after-landing run: every check step on `main` (`start_postland`)."""
        pl = drive.setdefault("postland", {"verified": base, "pending": [], "run": None})
        pl["pending"].append(tip)
        self.q.save_state({**self.q.load_state(), "drive": drive})

    def tick(self, dry_run: bool = False) -> str:
        self._after: list[Callable[[], None]] | None = []    # the landings' after-steps, run once the tick's launches are out
        try:
            return self._tick(dry_run)
        finally:
            self._run_after()

    def _tick(self, dry_run: bool) -> str:
        q = self.q
        self.dry_run, self._seen, self._prs, self._ev, self._relaunched, self._tips, self._pos = dry_run, {}, None, {}, set(), None, {}
        self._land_saw: str | None = None                   # the passed train this tick's land step tried, landed or held
        self._free = None
        q._clone()
        q.fetch()
        if q.store is not None and q.store.remote is not None:
            try:
                q.store.refresh(attempts_only=True)
            except Exception as e:  # noqa: BLE001  (the local catalog still answers)
                q.log(f"research queue drive: refreshing attempts failed ({type(e).__name__}: {e}); using the local catalog")
        state = q.load_state()
        drive = state.setdefault("drive", {})
        drive.setdefault("launched", {})
        drive.setdefault("told", [])
        self._drive = drive
        self.follow_deferred(drive)
        self.follow_postland(drive)
        if self.review:
            return self._review(state, drive, dry_run)
        landed = self.land(drive, dry_run)
        self._sweep(drive, dry_run)
        if landed is None and (why := self.held(drive)):
            self.per_pr(drive)
            if not dry_run:
                q.save_state(state)
            q.log(f"research queue drive: {why}")
            return why
        # before the sync: a rebuild takes a failed commit off `next`, and its failure would go untold (12 Oct 17:23Z: 3d11's)
        # and a train whose check is running, or has ended without its attempt in the store yet, keeps its place (19:47Z: a
        # rebuild in the seconds between 5897's end and its attempt's publication dropped a passing check)
        chain, main = q.chain(), _out(self.repo, "rev-parse", "origin/main")
        q.hold, failed_at, checked = 0, set(), []
        for i, m in enumerate(chain):
            if m.commit not in drive["launched"]:
                continue
            checked.append(i)
            v = self.verdict(m.commit, main, drive)
            if v.state == "failed":
                # a train above a failed one holds its failure, and runs again once `next` is rebuilt (18 Oct 10:30Z: acting
                # on 5c00b3 too ejected #2478 and #2486 beside c93469's #2484)
                if not failed_at:
                    self.failed(drive, m, chain[: i + 1], v)
                failed_at.add(i)
            elif v.state == "running":
                q.hold = i + 1
        # a member a running check covers keeps its place at the head that check runs, though its PR has a new head, unless
        # its own train (the lowest checked one above it) failed: then the push is the fix (top, 17 Oct)
        q.frozen = {m.pr for i, m in enumerate(chain[:q.hold])
                    if next((c for c in checked if c >= i), None) not in failed_at}
        members, entries = q.sync(dry_run)
        state = {**q.load_state(), "drive": drive}
        for n, c in state.get("conflicts", {}).items():
            self.refuse(drive, f"conflict:{n}@{c['head']}", f"#{n} at {c['head'][:8]} is out of `{q.branch}`: {c['why']}",
                        c.get("handle"))
        on_chain = {m.commit for m in members}
        following = {rid for rec in drive.get("deferred", {}).values() for rid in rec["ids"]}
        if not dry_run:
            self._cancel_orphans(drive)
        for c, v in drive["launched"].items():
            if c not in on_chain and not dry_run and not set(v["ids"]) & following:
                live = [rid for rid, (st, *_) in self._states([*v["ids"], *(v.get("touched") or [])]).items()
                        if st in ACTIVE and st != "unknown"]
                if live:                                   # a failed one too: drive acted on its failure, and its slot is a
                                                           # train's (17 Oct 14:41Z: two held slots 40 min with 32 PRs queued)
                    why = f"drive: {c[:12]} is no longer on `{q.branch}` (rebuilt); no check of it lands"
                    stuck = self.cancel(self.repo, live, why) or {}
                    drive.setdefault("orphans", {}).update({rid: why for rid in stuck})
                    q.log(f"research queue drive: cancelled {' '.join(r for r in live if r not in stuck) or 'none'}: "
                          f"{c[:12]} is no longer on `{q.branch}`"
                          + "".join(f"; {rid}'s cancel failed ({e[-160:]}): it is retried each tick" for rid, e in stuck.items()))
        for c, v in drive.get("base", {}).items():
            if (c != main or v["for"] not in on_chain) and not dry_run:
                if live := [rid for rid, (st, *_) in self._states(v["ids"]).items() if st in ACTIVE and st != "unknown"]:
                    self.cancel(self.repo, live, f"drive: `main` moved from {c[:12]}, or the train its check was for left "
                                                 f"`{q.branch}`")
        drive["launched"] = {c: v for c, v in drive["launched"].items() if c in on_chain}
        drive["resent"] = {c: v for c, v in drive.get("resent", {}).items() if c in on_chain}
        drive["bisect"] = {c: v for c, v in drive.get("bisect", {}).items() if c in on_chain}
        drive["base"] = {c: v for c, v in drive.get("base", {}).items() if c == main and v["for"] in on_chain}
        what = [f"landed #{landed.pr}"] if landed else []
        main = _out(self.repo, "rev-parse", "origin/main")
        trains = self.trains(len(members), self._cold(members))
        self._tips, self._pos = {members[i].commit for i in trains}, {m.commit: k for k, m in enumerate(members)}
        self.drop_hedges(drive, members, trains, main)
        running, idle = self._load(drive, main)
        running += self.prewarm(drive, running)
        # a landing below a failed member shifts the trains' indices off it, and a check above it holds the failure drive is
        # still settling (one told to `CI` is a person's to settle)
        settling = {chain[i].commit for i in failed_at if f"failed:{chain[i].commit}" not in drive["told"]} | {
            c for c in drive["resent"] if self.verdict(c, main, drive).state != "passed"}
        stop = next((k for k, x in enumerate(members) if x.commit in settling), len(members))
        passed_now = False                                  # a check whose verdict came in since the land step at the start
        todo: list[tuple[int, Member, str, bool]] = []          # (its place in `what`, the train's tip, its PRs, agreement)
        # the top tip keeps a check of its own, ranked in the slot queues right after the lowest train: one pass of it lands all
        # of `next`, while the trains below it still land what they can (23 Oct 19:20Z: the five checks were nested prefixes of
        # the bottom 15 of 50, cut at their cold-circuit-check boundaries, and nothing checked the 35 above them); the lowest
        # still takes the first free slot (top, 24 Oct 04:44Z)
        top = trains[-1] if len(trains) > 1 and trains[-1] <= stop and self.max_running > 2 else None
        reserve = top is not None and self.verdict(members[top].commit, main, drive).state == "none"
        if idle and (moved := self.move_waiting(drive, members, trains, main, top, idle)):
            left = [c for c in idle if c not in moved]
            running += len(moved) + min(len(left), UNSTARTED_COUNTED) - min(len(idle), UNSTARTED_COUNTED)
            idle = left
        waited = False
        for i in trains:
            m, prs = members[i], " ".join(f"#{x.pr}" for x in members[: i + 1])
            v = self.verdict(m.commit, main, drive)
            if v.state == "failed":
                self.failed(drive, m, members[: i + 1], v)
                what.append(f"{m.commit[:12]} failed")
                break                                       # every train above it holds the failure too
            if v.state == "passed":
                what.append(f"{m.commit[:12]} passed ({v.attempt})")
                passed_now = True
                continue
            if v.state == "running":
                what.append(f"{m.commit[:12]} ({prs}): {v.attempt} running")
                continue
            if i > stop:
                what.append(f"{m.commit[:12]} waits: {members[stop].commit[:12]} (#{members[stop].pr}) below it failed, and "
                            "its failure is settled first")
                break
            full = running >= self.max_running - (reserve and i != top)
            # past `UNSTARTED_COUNTED` waiting, a train launches only onto a free slot (the top keeps its place in the lines)
            queued = i != top and len(idle) >= UNSTARTED_COUNTED and not any(n > 0 for n in self._spare().values())
            if full or queued:
                if not waited:
                    what.append(f"{m.commit[:12]} waits: " + (
                        f"{running} checks of `{q.branch}` running" if full else
                        f"{len(idle)} checks of `{q.branch}` wait for a check slot, and no machine of {self.on} has one free")
                                + (f", and one is kept for the top, {members[top].commit[:12]}" if reserve and i != top else ""))
                    waited = True
                if reserve and i != top:
                    continue
                break
            if to := next((n for n, k in (self._free or {}).items() if k > 0), None):
                self._free[to] -= 1                         # `check.py --on` places it there
            agreement = needs_agreement(self.repo, main, m.commit)
            if dry_run or q.shadow:
                q.log(f"research queue drive: would launch train.sh {m.commit[:12]} {self.on}"
                      + (" --agreement" if agreement else "") + f" ({prs})" + (" [shadow]" if q.shadow else ""))
                what.append(f"would check {m.commit[:12]}")
                running += 1
                continue
            todo.append((len(what), m, prs, agreement))
            what.append("")
            running += 1
        floor = self._rank(members[trains[0]].commit) if top is not None and todo else None
        started = _each(lambda t: self._start(t[1], main, t[3], floor + 1 if floor is not None and t[1] is members[top] else
                                              self._rank(t[1].commit)), todo, LAUNCH_JOBS)
        for (k, m, prs, agreement), (ids, tail, *rest) in zip(todo, started):
            self._record(drive, m, agreement, (ids, tail, *rest))
            if not ids:
                self.refuse(drive, f"launch:{m.commit}", f"train.sh {m.commit[:12]} {self.on} registered no run: {tail}")
                what[k] = "launch failed"
                continue
            q.log(f"research queue drive: checking {m.commit[:12]} ({prs}): {' '.join(ids)}" + (" with agreement" if agreement else "")
                  + (f", touched suites {' '.join(drive['launched'][m.commit].get('touched') or [])}" if "pytest" in self.defer else ""))
            what[k] = f"launched {' '.join(ids)}"
        # after the trains' launches, whose own launch waits it held up to 15 min (24 Oct 00:5x-01:07Z): a slot of its own, since
        # `max_running` counts only the trains' checks
        self.start_postland(drive)
        if shadowed := self.per_pr(drive):
            what.append(shadowed)
        # a train that passed while this tick ran lands now, not a whole tick later (23 Oct: verdict to landed median 0.24 h,
        # p90 0.42 h over 98 PRs, art:cefa7f0035d98a932ecba75c3c75f7df5b6f88c581e861bbc0e4d3c1508f40e4)
        # only when the land step saw no passed train: one it held (its locks, the gate) stays held until the next tick
        if (passed_now and landed is None and self._land_saw is None and not dry_run and not q.shadow
                and (late := self.land(drive, dry_run)) is not None):
            what.append(f"landed #{late.pr} in the same tick")
        if not dry_run:
            state["ejected"] = {**state.get("ejected", {}), **q.load_state().get("ejected", {})}   # what `failed` ejected since
            q.save_state(state)
        q.log(f"research queue drive: {'; '.join(what) or 'nothing queued'}")
        return "; ".join(what) or "idle"

    def _sweep(self, drive: dict, dry_run: bool) -> None:
        q = self.q
        if not dry_run and not q.shadow and self.now() - drive.get("branches_swept", 0) > BRANCH_SWEEP_S:
            from .merge import sweep_branches

            with authed(q.auth):
                deleted, archived = sweep_branches(self.repo, "origin", log=q.log)
            drive["branches_swept"] = self.now()
            q.log(f"research queue drive: branch sweep: {deleted} deleted (on main), {archived} archived (unmerged, idle)")

    def _run_after(self) -> None:
        pending, self._after = self._after, None
        for step in pending or ():
            step()

    def trains(self, n: int, cold: frozenset[int] = frozenset()) -> list[int]:
        """The members whose commits get a check: every `train_size`-th one and the tip, each train stacked on the one below, so a
        failure holds up only the PRs from its train up, and the trains below it still land; and the member just below each one
        in ``cold`` (`queue.forces_cold`) but not itself cold, so the PRs under a run of cold ones get a warm check of their own and
        the run pays one cold circuit-check."""
        if not n:
            return []
        return sorted({*range(self.train_size - 1, n, self.train_size), n - 1,
                       *(i - 1 for i in cold if 0 < i < n and i - 1 not in cold)})

    def _cold(self, members: list) -> frozenset[int]:
        """The indices of ``members`` whose own merge onto `next` makes circuit-check run cold (`queue.forces_cold`)."""
        from .queue import forces_cold

        return frozenset(i for i, m in enumerate(members)
                         if forces_cold(_out(self.repo, "diff", "--name-only", f"{m.commit}^1", m.commit).split()))
