"""The run beside each candidate of the verified queue (`research.vqueue`) of what its graph2 check doesn't check: no graph/v2
task runs lean-agreement or the preflights.  It is `train.sh SHA MACHINES --deferred STEPS` of `SLIM_STEPS`, lean-agreement
left out when a run of every step would skip it (`Landing.beside_args`), whose check runs the preflights before any step.  It
is judged on the merge gate: the preflights and lean-agreement count, policy.toml's `off_gate` (ratchet-gate) doesn't.  A candidate passes only once its
graph2 check's gating tasks and this run both passed (`combine`).

What the loop does.  It starts the run with each candidate's build, beside each candidate's check, so up to its `--depth`
of them at once take the queue's machines, or once for a candidate on `main`'s chain whose start its log doesn't hold; until
then the candidate's run verdict is pending (`Verdict(PENDING)`, not `judge` of None, which is infra), and a candidate dropped
while its run runs (pending, or a check that gates failed before it ended) cancels it (one still launching has no run ids to
cancel yet).  It calls `start` once a candidate, `judge` each tick, and saves the record after
every `judge`: `judge` fills it in (the run ids once a detached launch printed them, the verdict once the run ended or was
given up, which it returns from then on without reading the run again), and a launch read once is gone (`Landing._resolve`
removes its directory).  On
`vqueue.Retry` with this run's verdict infra, it calls `start` again and keeps the new record in place of the old.  It gives
the `Landing` the `on` of drive.toml at its `main` (`MAIN`): `start` refuses a `Landing` without machines.  Under `--dry-run` it
calls neither `start` nor `judge`.  The candidate's verdict for `vqueue.decide` is `combine` of graph2's and this one's.  The
loop's `Landing` outlives its ticks, so `judge` reads the runs' states afresh each call.  Where the check gates, a candidate
sharing a member with one whose check the loop cancelled in the last `CARRY_WAIT_S` (`replaced`) launches only once those
runs' records are in the store or that long has passed since their cancel, and names them to its launch, so its check reuses
every pass they left (`start`'s ``after``).

A failed step's `Failed.key` is its content key at the candidate (`key`): SHA-512 over the step's name and the files it
reads (`reads`) as git holds them in the candidate's tree; its `base_key` is the same at the base, computed from the base's
tree, with no run.  An equal key means the base fails the step too, as graph2's keys of its tasks by their inputs do, so
`vqueue.decide` takes the failure as inherited.  A step whose reads can't be bounded is keyed on the whole tree, so its
failure is never inherited by its key: preflight-lints (its lints read every file, `check.lints_key`) and ratchet-gate (it
judges the merges above the base, not a tree).  Each failed step's one test (`Failed.tests`) is the step's (`step_test`), so
the run beside the base's own check, which the loop keeps beside `main` (started on `main` against itself, so lean-agreement
never runs), says what the base fails (`based`): failed there too, a step's failure is inherited by `vqueue.decide`'s rule
of failing tests.

The check that gates (`research queue vqueue --gate check`).  `start` with ``shards`` launches a whole check of the
candidate in place of the run of `SLIM_STEPS`: every step of `AFTER_LANDING` as that many shards, one launch a shard on its
own machines (`shard_machines`), with the same keywords (`Landing.beside_args`).  ``shards`` `auto:N` (`AUTO_SHARDS`) is N
shards of an auto plan of the candidate's tree, which splits pytest by suite where ci.toml's plans can't split it: `start`
has `check.py --shard-plan --deferred STEPS --n N` make it in the background (`Landing.plan_shards`), and the first `judge`
that finds it made (`Landing.planned`) launches every shard `--shard auto:K/N` with its file, which is removed once each
shard has sent it with its run; pending until then, infra when it failed or took past `PLAN_WAIT_S`.  A candidate dropped
while its plan is made launches nothing (`abandon`).  Its record says so (``gate``), and `judge` then judges every step
`main` waits for at the candidate's base (`Landing._gating` there, the
preflights and a lean-agreement that ran, less policy.toml's `off_gate` and drive.toml's `warn_steps`), a step a shard holds
passed only once every shard holding it passed (`shard_passed`).  A failed step's tests are what its failure names
(`failing`), so `vqueue.decide` takes as inherited only a failure whose every failing test the run beside the base fails
too; what a failure doesn't name is one test no other key has (`unnamed`), so it is never inherited.  Each holds what else
its records say (`failing`): the repository paths and `art:` ids they name (`Failed.named`, a frame from the check's tree,
`landing.tree_root`, and none outside it), which name its culprit; every path it can have read (`Failed.reads`: a suite's
declared inputs narrowed by what its run traced, a failed Lean build's modules and their imports, a step's declared reads;
None when nothing bounds them); and what failed (`Failed.kind`).  It fails as soon as a
shard has failed a step it judges, or a suite of one, while the others run (`_early`: a shard's failures so far, which its
check writes as each one fails), a failure read then as it would be at the end; a candidate on another candidate whose check
runs is compared with what that check has failed so far, the rest unknown until it ends (`based` with ``so_far``).  A
candidate built on another in flight, no member of it changing a Lean input (`lean_inputs`), audits nothing itself where its
base's check is judged on lean-audit (`audits`) or took its verdict on it from the check of a candidate below that is: `start`
with ``audit_from``, the candidate whose check ran it, leaves lean-audit out of its run (``lean_audit_from``), and `audited`
gives it the verdict on the step that the run beside `main` keeps once that candidate has landed (`answers`), pending until
then whatever that check says, so it lands after that candidate, drops with it, and stays pending while it is rerun.  A step
it leaves out still runs: `failures` reads what it failed, which the loop rings to who acts on it, but what the run beside
the base's check failed too (`Loop._off_gate`)."""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from . import quick_tier
from .landing import (ACTIVE, AFTER_LANDING, AUTO_PLAN, KILLED_RC, KILLED_SAYS, LAUNCH_LOST_S, LEAN_AUDIT_RE, LEAN_STEP,
                      PLAN_WAIT_S, PREFLIGHT, SIGTERM_RC, SLIM_JUDGED, SLIM_STEPS, SUITE_STEPS, Landing, _epoch, _step_log,
                      _suite_log, implicated, killed, lean_errors, not_run, shard_machines, shard_passed, shard_steps, step_named,
                      step_paths, step_records, suite_named, suite_pyproject, suite_rows, tree_root)
from .queue import _SUMMARY_RE, CHECK_STEPS, KEY_COMMAND, QueueError, _git, allowed, judge_report, tier_report
from .reds import named
from .store.fixtures import REPLICA_MAP, entries
from .vqueue import FAIL, INFRA, KINDS, LIST_FAILED, PASS, PENDING, STATES, Failed, Verdict, meets
from .vqueue_records import FAULT_RE

#: a candidate's state from its two verdicts' (`combine`): the first of these either has
ORDER = (FAIL, INFRA, PENDING, LIST_FAILED, PASS)
#: the train.sh a candidate's check launches with: the loop's own, deployed with its pin, so a candidate's tree doesn't decide
#: how its launch is waited for
OWN_TRAIN = Path(__file__).resolve().parents[4] / "tools" / "verity" / "check" / "train.sh"
#: a line of the loop's deselect list (`deselection`) that is a test's node id, its file's path from the repository's root
DESELECT_RE = re.compile(r"(?!/|\.\./)[\w./-]+\.py::\S+")
#: what a launch that registered no run says when it points at the machines, not the candidate's tree (`Failed.launch`):
#: train.sh stopped its launcher, still running, at the end of its wait; no run ids within `LAUNCH_LOST_S`; the pod lost its
#: process.  A launcher that exited without a run ran the candidate's own check.py, and says none of these
LAUNCH_ONLY = ("no run launched in the wait", "min after its launch", "its process is gone with no exit written")
#: what declares lean-agreement's reads: the graph build family, whose lean-agreement task reads `VERIFIER`'s files and
#: `CHECK_READS` (check.py, the modules it loads as it starts, and what they read), and the lean-build of `LEAN`, read from its
#: source at the commit keyed (`_declared`), never run
BUILD = "tools/verity/graph/verity_graph/families/build.py"
#: the environment every step runs in: the lock `uv run --locked` installs, and the interpreter it pins
ENV = ("uv.lock", ".python-version")
#: what `uv lock --check` (preflight-lock) reads: the lock and every file the workspace's metadata can come from
LOCK_READS = ("uv.lock", "uv.toml", ".python-version", "pyproject.toml", "**/pyproject.toml", "setup.py", "**/setup.py",
              "setup.cfg", "**/setup.cfg")
#: the replica map preflight-fixtures reads (`research.store.fixtures.load_map`), as a repository path
MAP = REPLICA_MAP.as_posix()
#: the code preflight-fixtures runs beside check.py: `check.fetchable_fixtures` imports `research.store` and what it imports
FIXTURES_CODE = "tools/research/src/research"
#: the tag a content key starts with (`key`)
KEY_TAG = "verity-vqueue-slim-step/v1"
#: a record's ``gate`` when its run is the check that gates (`start` with ``shards``)
CHECK = "check"
#: the record of a candidate's quick tier (`start_quick`), which gates a plain one where the land verdicts gate
#: (`vqueue_loop.VERDICT_GATE`); its suites are pytest's, so its failures are that step's
QUICK, QUICK_STEP = "quick", "pytest"
#: `start`'s ``shards`` naming N shards of an auto plan of the candidate's tree
AUTO_SHARDS = re.compile(r"auto:([1-9]\d*)")
#: lean-audit's own failures of no package, in its log
AUDIT_FAIL_RE = re.compile(r"^lean-audit: FAIL: .*$", re.M)
#: what the audit prints under a package's `AUDIT .../PKG: FAIL` that is no failure: counts and timings of its run
AUDIT_SAYS = ("closure:", "staged:", "sampled:", "trust gap:", "heartbeats:")
#: what the audit prints of a failed package when its build failed, and of a package that reads the facts (`proved_in`) of
#: one whose build failed
AUDIT_BUILT_RE = re.compile(r"build: |lean: proved_in: \S+'s build did not pass ")
#: an audit's line saying the build outputs it read changed under it (another build wrote them), and a test's skip that the
#: evidence store was unreachable (`verity_suite_guard.STORE_UNREACHABLE`), which a recorded check fails: a shared state's
#: failure, not the tree's
STORE_SAYS = re.compile(r"^\s*tamper: |the evidence store is not reachable", re.M)
#: what a log prints when its machine failed it, beyond `vqueue_records.FAULT_RE`: `tele run` refusing a run its machine's cgroups
#: would leave without a memory cap (`telemetry.cgroup.uncapped`)
MACHINE_SAYS = re.compile(r"refused to start the workload: this machine caps each login session")
#: the code that decides every suite's verdict, which every suite's key holds (`suites.RUNNER`)
SUITE_RUNNER = ("tools/verity/check/suites.py", "tools/verity/check/settings.py", "tools/verity/check/guard")
#: a Lean file's imports: each module of each `import` line (`public`, `meta`, `private` and `all` as Lean 4 writes them),
#: and each file it reads as it is compiled (`include_str`, relative to it)
IMPORT_RE = re.compile(r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?([^\n]+)", re.M)
INCLUDE_RE = re.compile(r'include_str\s+"([^"]+)"')
#: the audit's scratch tree, which each run names anew
SCRATCH_RE = re.compile(r"\S*?/lean-audit-scratch-[^/\s]+/")
#: the audit's own declaration of what an audit reads (`lean_audit.own_files`, `tool_files`), in the loop's own tree, deployed
#: with its pin as `OWN_TRAIN` is: `lean_inputs` reads a tree's Lean inputs through it, so it keeps no list of its own
OWN_AUDIT = Path(__file__).resolve().parents[4] / "tools" / "verity" / "lean" / "lean_audit.py"
#: every Lean source is a Lean input, in a Lean package or outside every one
LEAN_FILES = "**/*.lean"
#: the code that runs check's lean-audit step and that no key holds: `check.py`, which builds the step's command and is its
#: `--lean-slot` entry, and the wrapper that entry runs an audit in (`check.LEAN_SLOT_WRAP`).  Only a candidate that audits
#: runs them as the step does, so a change to one that took the verdict of the candidate below would land unrun and fail in
#: the next candidate that audits, on that one's members.  What they load that every candidate's check runs (`settings.py`,
#: `slot.py`) is no Lean input.  `tests/test_gate_paths.py` holds these to the step `check.py` declares
LEAN_AUDIT_STEP = ("tools/verity/check/check.py", "tools/verity/check/lean_slot.py")
#: the files whose directory is a Lean package's: a Lake package's lakefile, an audited one's lock
LEAN_PACKAGE_FILES = ("lakefile.toml", "lakefile.lean", "lean-audit.json")
#: the part of a check that replays every module of the security analysis's proofs, which `start`'s ``heavy_on`` sends to its
#: own machines (`heavy_shards`), and the part that audits it where a plan names it in no shard
HEAVY_PART, HEAVY_REST = "lean-audit:security/Proofs", "lean-audit:*"
#: how long after the loop cancelled a run a check that replaces it waits for the run's record (`start`'s ``after``), in
#: seconds; past it the check launches without that run's passes
CARRY_WAIT_S = 180
#: what names the runs ``after`` holds to the candidate's check.py, whose launch sends their packs first (`check.carried_packs`)
CARRY_ENV = "TRAIN_CARRY_FROM"


@functools.cache
def _declaration():
    """`OWN_AUDIT`, loaded once: the module whose `own_files` and `tool_files` every audit's key (`lean_audit.inputs`) and
    every records key (`lean_changed.closure_inputs`) is built from.  Raises `QueueError` when it can't be loaded."""
    try:
        spec = importlib.util.spec_from_file_location("research_vqueue_lean_audit", OWN_AUDIT)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
    except Exception as e:
        raise QueueError(f"the audit's declaration of what it reads ({OWN_AUDIT}) can't be loaded: {type(e).__name__}: {e}") from e
    return audit


def lean_inputs(files: Iterable[str]) -> tuple[str, ...]:
    """The Lean inputs of the tree whose tracked files are ``files``, as `vqueue.meets` reads them: a candidate one of whose
    members changes any of them runs lean-audit itself rather than take its base's verdict on it (`start`'s ``audit_from``).
    They are what the audit's and the records' keys read, what a lock run writes, and the code that runs the lean-audit
    step, each file by its path:

    - What a key reads (`lean_audit.inputs`, `lean_changed.closure_inputs`), from the audit's own declaration
      (`_declaration`) and none of it listed here, so a file the audit starts or stops reading joins or leaves with it:
      every Lean package's own files (`lean_audit.own_files` of each directory holding one of `LEAN_PACKAGE_FILES`, the
      auditor's fixtures aside: its sources, Lake files and lock, a file a module reads with `include_str`, its scripts and
      vectors, and not its prose, its pytest files or their settings), the auditor's (`lean_audit.tool_files`: of check's
      code `common.py` alone) and the records key's own program (`queue.KEY_COMMAND`).  A key holds files of those alone,
      whichever packages one requires (`tests/test_gate_paths.py` holds that on the tree).
    - What a lock run writes: each package's lock, one of its own files, and the two files beside the locks it repairs
      (`merge.REPAIRED`: the unread ratchet's limits and the repository suite's), which no key holds.
    - The code that runs the step and that no key holds: check's (`LEAN_AUDIT_STEP`) and the unread ratchet
      (`lean_audit.UNREAD`), which lean-audit runs after the audits.

    Beside the three, every Lean source wherever it is (`LEAN_FILES`).  A root that is a Lean package is the tree whole
    (``**``).  The declaration says of a path what it says of a file there, so a change is judged in a list holding the
    paths it deletes too.  The declaration read is the loop's own, not ``files``' tree's: a loop deployed before the audit
    began reading a file doesn't count that file.  It leaves out what the unread ratchet reads outside Lean
    (`lean_audit.unread_ratchet`: code and pyproject.toml naming an analysis declaration or module, the docs a lock names),
    which no audit's key holds: a change touching only those takes its base's verdict on lean-audit.  Raises `QueueError`
    when the declaration can't be loaded."""
    audit, files = _declaration(), sorted(set(files))
    under: dict[str, list[str]] = {f.rpartition("/")[0]: [] for f in files
                                   if f.rpartition("/")[2] in LEAN_PACKAGE_FILES and not f.startswith(audit.LEAN_TOOL)}
    if "" in under:
        return ("**",)
    for f in files:
        d = f
        while "/" in d:
            if (d := d.rpartition("/")[0]) in under:
                under[d].append(f)
    from .merge import REPAIRED
    own = (f for d, inside in under.items() for f in audit.own_files(d, inside))
    return (LEAN_FILES, *sorted({*own, *audit.tool_files(files), KEY_COMMAND, *REPAIRED, *LEAN_AUDIT_STEP, audit.UNREAD}))


def shard_count(shards: int | str) -> tuple[int, bool]:
    """How many shards `start`'s ``shards`` is, and whether of an auto plan: an int N, ci.toml's plan with N shards (1 or less:
    one whole run), or `auto:N` (`AUTO_SHARDS`).  ValueError for anything else."""
    if isinstance(shards, int):
        return shards, False
    if m := AUTO_SHARDS.fullmatch(str(shards)):
        return int(m[1]), True
    raise ValueError(f"shards {shards!r}: N, ci.toml's plan with N shards, or auto:N, N shards of an auto plan")


def deselection(driver: Landing, listed: Path | None) -> Path | None:
    """The file a candidate's check leaves out the tests of (`train.sh --deselect`), from the loop's list ``listed``, read at
    each launch: its node ids (`DESELECT_RE`; blank and `#` lines skipped), written beside it as ``<name>.sent``, each other
    line logged and left out, so a typo there drops no test and reds no candidate.  None while it is absent or names none."""
    if listed is None or not listed.is_file():
        return None
    lines = [s for line in listed.read_text().splitlines() if (s := line.strip()) and not s.startswith("#")]
    ids = list(dict.fromkeys(s for s in lines if DESELECT_RE.fullmatch(s)))
    if bad := [s for s in lines if not DESELECT_RE.fullmatch(s)]:
        driver.q.log(f"research queue vqueue: {listed}: {len(bad)} line(s) not a test's node id (PATH.py::NAME, PATH from the "
                     f"repository's root), left out: {'; '.join(bad[:5])}")
    if not ids:
        return None
    sent = listed.with_name(f"{listed.name}.sent")
    part = sent.with_name(f"{sent.name}.{os.getpid()}")
    part.write_text("".join(f"{i}\n" for i in ids))
    os.replace(part, sent)
    return sent


def declared_steps(driver: Landing, sha: str, steps: list[str]) -> list[str]:
    """``steps`` but those candidate ``sha``'s own `tools/verity/check/ci.toml` declares no job of (`merge.declared`), one line
    logged naming them: check.py refuses a `--deferred` naming a job its tree doesn't declare, so sending one would fail the
    candidate's own check.  Every one of ``steps``, logged, when that ci.toml can't be read."""
    from .merge import declared

    if (jobs := declared(driver.repo, sha)) is None:
        driver.q.log(f"research queue vqueue: candidate {sha[:12]}'s tools/verity/check/ci.toml can't be read: its --deferred "
                     f"keeps every step ({','.join(steps)})")
        return steps
    names = {t["name"] for t in jobs}
    if gone := [s for s in steps if s not in names]:
        driver.q.log(f"research queue vqueue: candidate {sha[:12]}'s tools/verity/check/ci.toml declares no job "
                     f"{', '.join(gone)}, which check.py refuses: left out of its --deferred")
    return [s for s in steps if s in names]


def heavy_shards(driver: Landing, sha: str, n: int, doc: dict | None = None) -> set[int]:
    """The shards, 1 to ``n``, of candidate ``sha``'s check that audit `HEAVY_PART`: of the auto plan ``doc`` (its shards'
    parts), else of the plan with ``n`` shards in ``sha``'s own ci.toml (`merge.shard_plans`), those naming it, else those
    naming `HEAVY_REST`; every shard, logged, where the plan can't be read or names neither."""
    from .merge import shard_plans

    shards = doc.get("shards") if doc is not None else (shard_plans(driver.repo, sha) or {}).get(n)
    parts = [s.get("parts") if isinstance(s, dict) else s for s in shards] if isinstance(shards, list) else []
    if len(parts) == n and all(isinstance(p, list) for p in parts):
        for part in (HEAVY_PART, HEAVY_REST):
            if got := {k for k, p in enumerate(parts, 1) if part in p}:
                return got
    driver.q.log(f"research queue vqueue: candidate {sha[:12]}'s plan of {n} shards can't be read or names neither {HEAVY_PART} "
                 f"nor {HEAVY_REST}: every shard goes on the heavy machines")
    return set(range(1, n + 1))


def replaced(lines: Iterable[dict], members: Iterable[int], now: float) -> dict[str, float]:
    """The runs a check of a candidate holding ``members`` replaces (`start`'s ``after``), each by when the loop cancelled it:
    of each line of the loop's log (``lines``) in the last `CARRY_WAIT_S` that cancelled the check of a candidate sharing a
    member, a drop's or one's that cancelled a check still launching at its drop (`Loop._launched`), its runs
    (``beside_cancelled``, those that had ended already among them) but one its cancel left running (`KEEP_RUNS`), which
    no wait would see end."""
    want, out = set(members), {}
    for r in lines:
        at = _epoch(r.get("at"))
        if at is None or now - at > CARRY_WAIT_S or not want & set(r.get("members") or ()):
            continue
        kept = {i for i, why in (r.get("beside_cancel_failed") or {}).items() if str(why).startswith("left running")}
        for rid in r.get("beside_cancelled") or ():
            if rid not in kept:
                out.setdefault(rid, at)
    return out


def left(driver: Landing, rid: str) -> bool:
    """Whether run ``rid`` has left its passes where a check's launch finds them: it ended and custody published its record,
    so the store's remote holds its attempt, which `check.run_packs` reads its packs from."""
    try:
        return driver.q.store.get_attempt(rid) is not None
    except Exception:  # noqa: BLE001  (no store, or one that can't answer now: the wait runs on to its bound)
        return False


def _awaiting(driver: Landing, runs: dict[str, float]) -> tuple[list[str], list[str]]:
    """Of ``runs`` (`replaced`) whose records `left` doesn't find: those still waited for, cancelled less than `CARRY_WAIT_S`
    ago, and those gone without, past it."""
    now, awaited, gone = driver.now(), [], []
    for rid, at in sorted(runs.items()):
        if not left(driver, rid):
            (awaited if now - at < CARRY_WAIT_S else gone).append(rid)
    return awaited, gone


@contextlib.contextmanager
def _carrying(runs: Iterable[str]):
    """`CARRY_ENV` naming ``runs`` while a launch starts, and in no launch after: `landing.train_launch` hands train.sh this
    process's environment, the one way to the candidate's check.py that every tree's train.sh passes.  Only this sets it in
    the loop's process."""
    if not (runs := ",".join(runs)):
        yield
        return
    os.environ[CARRY_ENV] = runs
    try:
        yield
    finally:
        os.environ.pop(CARRY_ENV, None)


def start(driver: Landing, sha: str, base: str, shards: int | str = 0, deselect: Path | None = None,
          audit_from: str | None = None, after: dict[str, float] | None = None, heavy_on: str = "") -> dict | None:
    """Launch the run beside candidate ``sha``'s check, ``base`` the commit of `main` below it: `SLIM_STEPS` but those a run
    of every step of ``sha`` above ``base`` skips by name (lean-agreement, unless what it checks changed), with that run's
    keywords (`Landing.beside_args`: ratchet-gate judges the merges above ``base``), on the queue's machines (`Landing.on`).
    With ``shards``, the check that gates instead: `--deferred` every step of `AFTER_LANDING` (each check runs the
    preflights first), `--shard K/N` on shard K's machines (`shard_machines`) for each of that many, one whole run for 1;
    `auto:N` has the auto plan its shards launch by made first (`Landing.plan_shards`, ``plan``, which `judge` follows).
    With ``audit_from`` too, the candidate below whose check ran lean-audit (`audits`), its base or the one its base's check
    took it from, whose verdict on it this one takes (`audited`): every step but lean-audit.  Each sends only the steps the
    candidate's own ci.toml declares as jobs (`declared_steps`).  With ``heavy_on`` (machines, a comma list) too, the check
    that gates sends the shard that audits `HEAVY_PART` (`heavy_shards`; the whole run, of one shard) to those machines
    alone, the others to theirs, and its record keeps them (``heavy_on``), so an auto plan's shards launched later do too, as
    does a launch that waited for the records of the runs it replaces (``after``).
    The record to keep (`Landing._started`): ``ids``, none yet while a detached launch runs (its ``launch`` then, or each
    shard's in ``shards``, which `judge` makes ``groups``, each shard's ids); the candidate (``tip``), ``base``, when it
    launched (``at``), the ``steps`` it runs and those it skips (``skip``), ``gate`` (`CHECK`) with ``shards``, and
    ``lean_audit_from`` with ``audit_from``.  None,
    logged, when it registered no run (a shard that registered none stops the others), or when ``driver`` launches nothing (a
    dry run, or its queue's shadow).  Every launch leaves out the tests the loop's list ``deselect`` names (`deselection`), each
    shard's alike; the nightly uncached check, which reads no list, runs them.
    With ``after`` (`replaced`: the runs of dropped candidates' checks it replaces, by when each was cancelled), it names them
    to its launch (`CARRY_ENV`, ``carry``), so its check reuses their passes first, and launches only once `left` finds each
    one's record or `CARRY_WAIT_S` has passed since its cancel: until then the record holds ``wait`` and no ids, and the
    `judge` that finds the wait over launches it, ``at`` then, naming each run it went without (``carry_without``)."""
    if not driver.on:
        raise ValueError("the run beside a candidate's check needs the queue's machines: build the Landing with drive.toml's `on` "
                         "(`research.queue.flags_at(repo, main).on`)")
    n, auto = shard_count(shards)
    if audit_from and not n:
        raise ValueError(f"audit_from {audit_from[:12]}: only the check that gates (``shards``) runs lean-audit, so only it "
                         "can take another check's verdict on it")
    kw, skip, flags = driver.beside_args(sha, base)
    if left := deselection(driver, deselect):
        kw = {**kw, "deselect": str(left)}
        flags += f" --deselect {left} ({len(left.read_text().splitlines())} tests)"
    every = declared_steps(driver, sha, [s for s in (AFTER_LANDING if n else SLIM_STEPS) if not (audit_from and s == LEAN_STEP)])
    skip = [s for s in skip if s in every]
    steps = [s for s in every if s not in skip]
    sent = ",".join(every if n else steps)
    heavy_on = heavy_on if n and LEAN_STEP in every else ""
    what = (f"train.sh {sha[:12]} {driver.on} --deferred {sent}{flags}"
            + (f" as an auto plan's {n} shards" if auto else f" as {n} shards" if n > 1 else "")
            + (f" as candidate {sha[:12]}'s check" if n else f" beside candidate {sha[:12]}'s check")
            + (f", {LEAN_STEP} {audit_from[:12]}'s" if audit_from else "")
            + (f", {HEAVY_PART}'s shard on {heavy_on}" if heavy_on else ""))
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    rec = {"ids": [], "tip": sha, "base": base, "at": driver.now(), "steps": steps, **({"skip": skip} if skip else {}),
           **({"gate": CHECK} if n else {}), **({"lean_audit_from": audit_from} if audit_from else {}),
           **({"carry": sorted(after)} if after else {}), **({"heavy_on": heavy_on} if heavy_on else {})}
    go = {"what": what, "sent": sent, "kw": kw, "n": n, "auto": auto}
    awaited, gone = _awaiting(driver, after or {})
    if awaited:
        driver.q.log(f"research queue vqueue: {what}: waiting for the records of {' '.join(awaited)}, whose passes it reuses, "
                     f"at most {CARRY_WAIT_S // 60} min after their cancel")
        return {**rec, "wait": {"runs": after, **go}}
    return rec if _go(driver, rec, go, gone) is None else None


def start_quick(driver: Landing, sha: str, base: str) -> dict | None:
    """Launch candidate ``sha``'s quick tier since ``base`` (`Landing.launch_quick`: `queue.QUICK_TIER` on its own merge, as a
    `research run`) on one of the queue's machines, the one its sha picks, so a start again takes the same one.  The record to
    keep: ``ids`` (none while its detached launch runs, ``launch`` then), ``tip``, ``base``, ``at``, ``on``, ``steps``
    (`QUICK_STEP`) and ``gate`` (`QUICK`), which `judge` reads by `_quick`.  None, logged, when it registered no run or
    ``driver`` launches nothing (a dry run, or its queue's shadow)."""
    if not driver.on:
        raise ValueError("a candidate's quick tier needs the queue's machines: build the Landing with drive.toml's `on`")
    ms = driver.on.split(",")
    on = ms[int(sha[:8], 16) % len(ms)]
    what = f"the quick tier of candidate {sha[:12]} since {base[:12]} on {on}"
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    got, tail = driver._started(driver.launch_quick(driver.repo, sha, base, on))
    if got is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(got['ids']) or 'launching'}")
    return {**got, "tip": sha, "base": base, "at": driver.now(), "on": on, "steps": [QUICK_STEP], "gate": QUICK}


def _quick(driver: Landing, rec: dict) -> Verdict:
    """The verdict of the quick tier `start_quick` recorded in ``rec``: pending while it launches or runs, or its records can't
    be fetched yet; once it ended, its result.json judged as `research queue ready` judges the remote tier
    (`queue.judge_report`, each failure an allowance at ``base`` covers counting as passed): pass; fail, ended, when only suites
    failed, one `Failed` of step `QUICK_STEP` as the check's (`read_suite` of each, what it declares at the candidate,
    `suite_reads`), so `vqueue.decide` reads it as it reads the check's; infra when it registered no run, was lost, never ran
    or ended without its facts (`queue.tier_report`), or failed on what no suite says (a Lean build, a checkout not at it)."""
    tip, base = rec["tip"], rec["base"]
    if (tail := driver._resolve(rec, f"the quick tier of candidate {tip[:12]}")) is not None:
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    if "launch" in rec:
        return Verdict(PENDING)
    if not (ids := rec["ids"]):
        return _keep(rec, _down(rec, "registered no run"))
    states = driver.states(driver.repo, ids)
    if driver._lost(rec, states):
        return _keep(rec, _down(rec, "was lost"))
    st, rc, *_ = states.get(ids[0]) or ("unknown", None)
    if st in ACTIVE:
        return Verdict(PENDING)
    if (got := driver.fetch(ids[0], ["result.json"])) is None:
        return Verdict(PENDING)
    status, on = {"state": st, "rc": rc}, rec.get("on") or driver.on
    res, why = tier_report(got[0], ids[0], on, status)
    if why is not None:
        return _keep(rec, _down(rec, f"{why}"))
    covers = allowed(driver.repo, base)
    if (why := judge_report(res, ids[0], on, tip, status, covers)) is None:
        return _keep(rec, Verdict(PASS), [QUICK_STEP])
    tier = res.doc.get("tier") or {}
    rows = [r for r in tier.get("rows") or () if isinstance(r, dict) and r.get("status") == "failed"
            and not (covers is not None and covers(r))]
    if not rows or quick_tier.judge({**tier, "rc": 0}, tip) is not None:
        return _keep(rec, _down(rec, f"failed on what no suite says: {why}"))
    k = key(driver.repo, tip, QUICK_STEP) or f"{tip}:{QUICK_STEP}"
    parts = [read_suite(r.get("name") or r.get("suite") or ".", r, "", unnamed(k),
                        *suite_reads(driver.repo, tip, r.get("suite") or ".")) for r in rows]
    return _keep(rec, Verdict(FAIL, (_failure(QUICK_STEP, k, key(driver.repo, base, QUICK_STEP), parts),), ended=True), [])


def _go(driver: Landing, rec: dict, go: dict, gone: list[str]) -> str | None:
    """Launch what `start` made ``go`` of into ``rec`` (``at`` then): an auto plan's job (``plan``), each shard of ``n``
    (`_launch`), or one run, each launch naming ``rec``'s ``carry`` (`_carrying`), the shard that audits `HEAVY_PART` (the
    whole run, of one) on ``rec``'s ``heavy_on``; ``gone``, the runs it waited for in vain, kept as ``carry_without``.  None,
    logged, once launched; the tail of a launch that registered no run, logged."""
    sha, sent, kw, n, carry = rec["tip"], go["sent"], go["kw"], go["n"], rec.get("carry") or ()
    heavy_on = rec.get("heavy_on") or ""
    if go["auto"]:
        got, tail = {"ids": [], "plan": {**driver.plan_shards(driver.repo, sha, sent, n=n), "n": n, "deferred": sent,
                                         "kw": kw}}, ""
    elif n > 1:
        got, tail = _launch(driver, sha, sent, kw, n, carry=carry, heavy_on=heavy_on)
        if got is not None and not any("launch" in r for r in got["shards"]):
            _grouped(got)
    else:
        with _carrying(carry):
            got, tail = driver._started(driver.launch_postland(driver.repo, sha, heavy_on or driver.on, sent, **kw,
                                                               train_from=OWN_TRAIN))
    if got is None:
        driver.q.log(f"research queue vqueue: {go['what']} registered no run: {tail}")
        return tail
    rec.update(got, at=driver.now(), **({"carry_without": gone} if gone else {}))
    driver.q.log(f"research queue vqueue: {go['what']}: {' '.join(rec['ids']) or ('planning' if go['auto'] else 'launching')}"
                 + (f"; without the passes of {' '.join(gone)}, whose records weren't in the store {CARRY_WAIT_S // 60} min "
                    "after their cancel" if gone else ""))
    return None


def _waited(driver: Landing, rec: dict) -> Verdict:
    """The check `start` left waiting in ``rec`` (``wait``) for the records of the runs it replaces: pending while it waits
    (`_awaiting`), then launched (`_go`) and pending; infra, kept, when that launch registered no run."""
    awaited, gone = _awaiting(driver, rec["wait"]["runs"])
    if awaited:
        return Verdict(PENDING)
    go = {k: v for k, v in rec.pop("wait").items() if k != "runs"}
    if (tail := _go(driver, rec, go, gone)) is not None:
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    return Verdict(PENDING)


def _launch(driver: Landing, sha: str, sent: str, kw: dict, n: int, plan: str | None = None,
            carry: Iterable[str] = (), heavy_on: str = "", doc: dict | None = None) -> tuple[dict | None, str]:
    """Each of ``n`` shards of candidate ``sha``'s check launched on its machines (`shard_machines`): `--shard K/N` of ci.toml's
    plan, or of the auto plan ``doc`` whose file is ``plan``, `--shard auto:K/N` with it, each naming the runs ``carry``
    (`_carrying`); with ``heavy_on``, the shards that audit `HEAVY_PART` (`heavy_shards`) on those machines instead.  The
    record of every shard's launch (``shards``); None with the tail of one that registered no run, the others' runs
    cancelled."""
    ons = shard_machines(driver.on, n)
    if heavy_on:
        heavy = heavy_shards(driver, sha, n, doc)
        ons = [heavy_on if k in heavy else on for k, on in enumerate(ons, 1)]
    with _carrying(carry):
        got = [driver._started(driver.launch_postland(driver.repo, sha, on, sent, **kw, shard=f"{'auto:' * bool(plan)}{k}/{n}",
                                                      **({"shard_plan": plan} if plan else {}), train_from=OWN_TRAIN))
               for k, on in enumerate(ons, 1)]
    if any(r is None for r, _ in got):
        if ids := [i for r, _ in got if r for i in r["ids"]]:
            driver.cancel(driver.repo, ids, f"vqueue: a shard of candidate {sha[:12]}'s check registered no run")
        return None, next(t for r, t in got if r is None)
    return {"ids": [], "shards": [r for r, _ in got]}, ""


def _grouped(rec: dict) -> dict:
    """``rec`` once each of its ``shards`` has its run ids: they are its ``groups``, and its ``ids`` every one's; each shard sent
    its auto plan's file with its run (`_unplan`)."""
    rec["groups"] = [list(r["ids"]) for r in rec.pop("shards")]
    rec["ids"] = [i for g in rec["groups"] for i in g]
    _unplan(rec)
    return rec


def _unplan(rec: dict) -> None:
    """Remove the directory of ``rec``'s auto plan (`start` with `auto:N`), which no shard of it sends any more."""
    if plan := rec.get("plan"):
        shutil.rmtree(plan["out"], ignore_errors=True)


def abandon(rec: dict) -> None:
    """Give up ``rec``'s auto plan before its shards launch (its candidate dropped, or the plan failed): its job's directories
    removed, so it writes no plan (`check.py --shard-plan` makes no directory) and no shard launches by it; a job still
    running ends on its own, its worktree with it."""
    if (plan := rec.get("plan")) and "made" not in plan:
        shutil.rmtree(plan["dir"], ignore_errors=True)
        _unplan(rec)


def _planned(driver: Landing, rec: dict, what: str) -> Verdict:
    """The auto plan ``rec``'s shards launch by (`start` with `auto:N`), followed: pending while `check.py --shard-plan` makes
    it, up to `PLAN_WAIT_S` after the start, then stopped; once made, every shard launched `--shard auto:K/N` with its file
    (`_launch`, ``plan``'s ``made``, ``at`` their launch), and pending.  Infra, kept, when it failed, has another N than asked,
    wasn't made in time, or a shard registered no run, its directories removed (`abandon`)."""
    plan = rec["plan"]
    doc, why = driver.planned(plan)
    if why is not None:
        why = f"failed: {why}"
    elif doc is None:
        if driver.now() - rec["at"] <= PLAN_WAIT_S:
            return Verdict(PENDING)
        with contextlib.suppress(ProcessLookupError, PermissionError, ValueError, TypeError, KeyError):
            os.killpg(int(plan["pid"]), signal.SIGTERM)
        why = f"was not made in {PLAN_WAIT_S // 60} min (stopped)"
    elif doc["n"] != plan["n"]:
        why = f"has {doc['n']} shards, not {plan['n']}"
    got, tail = (None, f"its auto plan {why}") if why is not None else _launch(
        driver, rec["tip"], plan["deferred"], plan["kw"], plan["n"], str(Path(plan["out"]) / AUTO_PLAN), rec.get("carry") or (),
        heavy_on=rec.get("heavy_on") or "", doc=doc)
    if got is None:
        abandon(rec)
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    plan["made"] = {"digest": doc["digest"], "predicted_s": doc.get("predicted_s")}
    rec.update(got, at=driver.now())
    if not any("launch" in r for r in rec["shards"]):
        _grouped(rec)
    driver.q.log(f"research queue vqueue: {what}: auto plan {doc['digest'][:12]}, predicted wall "
                 f"{(doc.get('predicted_s') or 0) / 60:.0f} min: {' '.join(rec['ids']) or 'launching'}")
    return Verdict(PENDING)


def judge(driver: Landing, rec: dict | None) -> Verdict:
    """The verdict of the run `start` recorded in ``rec``, read on the steps it judges on the merge gate: `SLIM_JUDGED` but a
    step it skipped and policy.toml's `off_gate` at ``tip``; of the check that gates (``gate``), every step `main` waits for
    at ``base`` (`Landing._gating`), the preflights and a lean-agreement it ran, but policy.toml's `off_gate` there and
    drive.toml's `warn_steps` (`_gate`).  Pending while it waits for the records of the runs it replaces, the call that finds
    the wait over launching it (`_waited`), and while its auto plan is made, the call that finds it made launching the
    shards (`_planned`), and while it launches or runs, each shard's launch read as the run's own is; the check that gates,
    while it runs, fails once a shard has failed one of those steps (`_early`).
    Infra when it registered no run (``rec`` None, its auto plan failed, or a detached launch that printed no run ids,
    a shard's stopping the other shards; `Failed.launch` when it says one of `LAUNCH_ONLY`), never ran or was stopped, or was lost or killed before it passed each of those
    steps; infra's one `Failed` says which.  Once it ended, pass when it passed each of them (the check that gates: in
    every shard that holds it, `shard_passed`), a step that failed only on what an unexpired allowance of policy.toml at
    ``base`` covers counting as passed, so no candidate's allowance covers its own failure; fail with each of them it failed
    (a `Failed`: ``step NAME``, the step's content key at the candidate, the paths that failure names its culprit by,
    `step_paths`, its content key at ``base``, `key`, and the step's one test, `step_test`; the check that gates: the tests
    and paths its failure names; each with what else its records say, `failing`: what it can have read and its class), and
    infra when it failed none but passed fewer (one skipped, or absent from its steps.jsonl).  Each verdict but pending is
    kept in ``rec`` (``verdict``, with the steps it passed) and read from there."""
    if rec is None:
        return _down(None, "registered no run")
    if "verdict" in rec:
        return _kept(rec["verdict"])
    if "wait" in rec:
        return _waited(driver, rec)
    if rec.get("gate") == QUICK:
        return _quick(driver, rec)
    tip, check = rec["tip"], rec.get("gate") == CHECK
    what = f"the run of {', '.join(rec['steps'])} " + (f"as candidate {tip[:12]}'s check" if check else
                                                      f"beside candidate {tip[:12]}'s check")
    if "plan" in rec and "made" not in rec["plan"]:
        return _planned(driver, rec, what)
    shards = rec.get("shards") or ()
    # every shard's launch is read in one call, since a launch read once is gone: the record never holds some read, some not
    printed = [driver.launched(r["launch"])[0] for r in shards if "launch" in r]
    if None in printed and [] not in printed and driver.now() - rec["at"] <= LAUNCH_LOST_S:
        return Verdict(PENDING)
    for k, shard in enumerate(shards, 1):
        shard.setdefault("at", rec["at"])
        if (tail := driver._resolve(shard, f"shard {k}/{len(shards)} of {what}")) is not None:
            if got := [i for r in shards for i in r["ids"]]:
                driver.cancel(driver.repo, got, f"vqueue: shard {k}/{len(shards)} of candidate {tip[:12]}'s check registered "
                                                "no run")
            _unplan(rec)
            return _keep(rec, _down(rec, f"registered no run: shard {k}/{len(shards)}: {tail}"))
    if any("launch" in r for r in shards):
        return Verdict(PENDING)
    if shards:
        _grouped(rec)
    if (tail := driver._resolve(rec, what)) is not None:
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    if "launch" in rec:
        return Verdict(PENDING)
    if not (ids := rec["ids"]):
        return _keep(rec, _down(rec, "registered no run"))
    states = driver.states(driver.repo, ids)
    if lost := driver._lost(rec, states):
        states = {r: ("lost", -1) for r in states}
    if any(st in ACTIVE for st, *_ in states.values()):
        return _early(driver, rec, states) if check else Verdict(PENDING)
    why = "lost" if lost else ", ".join(f"{st} rc={rc}" for st, rc, *_ in states.values())
    if not lost and driver._never_ran(rec, states):
        return _keep(rec, _down(rec, f"never ran or was stopped ({why})"))
    base = rec.get("base")
    allow = base or tip
    if check:
        judged = _gate(driver, rec)
        view = shard_steps(driver._shards(rec), [any(states[r][1] == 0 for r in g) for g in rec.get("groups") or [ids]])
        passes = shard_passed(view, driver._allowed_steps(allow, [d for ds in driver._shards(rec) for d in ds]))
        missed = [s for s in judged if s not in passes]
    else:
        off = driver._policy(tip).off_gate
        judged = [s for s in SLIM_JUDGED if (s in rec["steps"] or s not in SLIM_STEPS) and s not in off]
        missed = driver._unrun(ids, judged, allow)
    if not missed:
        return _keep(rec, Verdict(PASS), judged)
    dirs = [d for ds in driver._shards(rec) for d in ds]
    ok = driver._every_shard(rec, states, lambda rc: rc == 0) or (
        driver._every_shard(rec, states, lambda rc: rc in (0, 1)) and driver._only_allowed(rec, allow))
    if not ok and (lost or any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                   for _, rc, *_ in states.values())
                       or any(implicated(d) is None and killed(d, why) for d in dirs)):
        return _keep(rec, _down(rec, "was lost" if lost else f"was killed ({why})"))
    got: dict[str, list[Read]] = {}
    keys: dict[str, str] = {}
    for d in dirs:
        for st in step_records(d):
            if (s := st.get("step")) in missed and st.get("status") == "failed" and not not_run(st):
                if s not in keys:
                    keys[s] = key(driver.repo, tip, s) or f"{tip}:{s}"
                read = failing(driver, allow, d, st, unnamed(keys[s]), tip)
                if not check:
                    read = read._replace(tests=frozenset({step_test(s)}), named=frozenset(step_paths(d, st) or ()))
                got.setdefault(s, []).append(read)
    passed = [s for s in judged if s not in missed]
    if got:
        return _keep(rec, Verdict(FAIL, tuple(_failure(s, keys[s], key(driver.repo, base, s) if base else None, got[s])
                                              for s in missed if s in got), ended=True), passed)
    return _keep(rec, _down(rec, f"ended ({why}) without passing {', '.join(missed)}: each skipped, or absent from its "
                                 "steps.jsonl"), passed)


def _gated(driver: Landing, rec: dict) -> set[str]:
    """The steps the check that gates (``rec``) answers for, run or not: every step `main` waits for at its base
    (`Landing._gating`), the preflights and lean-agreement, but policy.toml's `off_gate` there and drive.toml's `warn_steps`."""
    at = rec.get("base") or rec["tip"]
    return {*SLIM_JUDGED, *driver._gating(at)} - {*driver._policy(at).off_gate, *driver.warn_steps}


def _gate(driver: Landing, rec: dict) -> list[str]:
    """The steps the check that gates (``rec``) is judged on: those it answers for (`_gated`) that it ran, and the preflights."""
    gate = _gated(driver, rec)
    return [s for s in CHECK_STEPS if s in gate and (s in rec["steps"] or s in PREFLIGHT)]


def _early(driver: Landing, rec: dict, states: dict) -> Verdict:
    """The verdict of the check that gates (``rec``) while a shard of it runs: pending until one of its runs says it failed
    (one running names its failures so far, `landing.failing`; one ended exited 1), then fail with each step it is judged on
    (`_gate`) that such a run failed, not by a signal, or a suite of which it failed a named test, uncovered by an allowance
    at its base, read as once every shard ended (`failing`), from their dirs fetched afresh (`landing.evidence`: the Landing's
    own cache holds what the end's reading reads).  A suite failed beyond its tests alone, as a stop marks each suite it
    cuts short, waits for the end.  Never kept, so not ``ended``: `vqueue.decide` reads a fail whose every failure is
    inherited as pending until the check ends.  ``rec`` holds it (``early``: its failures as `_keep` keeps a verdict's, with
    ``at``, when the first was written: a failed step's end, a failed suite's log's last write), which the check above reads
    (`based`)."""
    said = [r for r, s in states.items() if ((len(s) > 3 and s[3]) if s[0] in ACTIVE else s[1] == 1)]
    if not said:
        return Verdict(PENDING)
    tip, base = rec["tip"], rec.get("base")
    allow, dirs = base or tip, driver.evidence(driver.repo, said)
    allowed = driver._allowed_steps(allow, dirs)
    keys: dict[str, str] = {}
    got: dict[str, list[Read]] = {}
    at: list[float] = []
    for d in dirs:
        if implicated(d) is None and killed(d):
            continue                                     # a kill, as `judge` reads it at the end: infra, never an early red
        ended = {st.get("step"): st for st in step_records(d)}
        for s in _gate(driver, rec):
            if (st := ended.get(s)) is not None:
                if (st.get("status") != "failed" or not_run(st) or s in allowed
                        or isinstance(rc := st.get("rc"), int) and (rc < 0 or rc in KILLED_RC)):
                    continue
                at.append(_epoch(st.get("ended_utc")) or driver.now())
            elif s in SUITE_STEPS and (rows := [row for name, row in suite_rows(d).items() if row["status"] == "failed"
                                                 and row["step"] == s and row.get("failed_tests")
                                                 and not driver._allowed(allow, f"suite {name}", d, row).covered]):
                st = {"step": s}
                at += [_written(d / "suites" / Path(row["log"]).name, driver.now()) for row in rows if row.get("log")]
            else:
                continue
            if s not in keys:
                keys[s] = key(driver.repo, tip, s) or f"{tip}:{s}"
            got.setdefault(s, []).append(failing(driver, allow, d, st, unnamed(keys[s]), tip))
    if not got:
        return Verdict(PENDING)
    v = Verdict(FAIL, tuple(_failure(s, keys[s], key(driver.repo, base, s) if base else None, got[s])
                            for s in _gate(driver, rec) if s in got))
    first = [t for t in (*at, (rec.get("early") or {}).get("at")) if t is not None]
    rec["early"] = {**_doc(v), "at": min(first, default=driver.now())}
    return v


def _written(path: Path, otherwise: float) -> float:
    """When ``path`` was last written (a fetched file keeps its node's time), ``otherwise`` when it wasn't fetched."""
    return path.stat().st_mtime if path.is_file() else otherwise


def unnamed(key: str) -> str:
    """The one test of what a failure keyed ``key`` doesn't name: no failure under another key has it, so
    `vqueue.decide` never takes it as inherited by its tests."""
    return f"unnamed:{key}"


def failures(driver: Landing, rec: dict | None, step: str) -> dict[str, str] | None:
    """What the run ``rec`` records failed of ``step``, a step `judge` leaves out (policy.toml's `off_gate`, drive.toml's
    `warn_steps`), once its verdict is kept: each failure the step's logs name (`reds.named`: ratchet-gate's merges,
    circuit-check's failures), by its key with its line, else the step itself; {} when it passed.  A shard's twin that passed
    it answers for the shard (`shard_steps`).  None while the run runs, and when it didn't run the step to its end: no shard
    holds it, one skipped or stopped it, or a kill ended it (`KILLED_RC`, a signal)."""
    if rec is None or "verdict" not in rec or not rec.get("ids"):
        return None
    groups = driver._shards(rec)
    view = shard_steps(groups, [True] * len(groups)) or []
    if not (held := [k for k, v in enumerate(view) if step in v]) or any(view[k][step] not in ("passed", "failed") for k in held):
        return None
    out: dict[str, str] = {}
    for d in (d for k in held if view[k][step] == "failed" for d in groups[k]):
        for st in step_records(d):
            if st.get("step") != step or st.get("status") != "failed" or not_run(st):
                continue
            if st.get("rc") in KILLED_RC or isinstance(st.get("rc"), int) and st["rc"] < 0:
                return None
            out.update(named(step, _step_log(d, st)) or {step: f"{step} failed, its log naming no failure"})
    return out


class Read(NamedTuple):
    """What a failure's records say (`failing`): its failing tests, the repository paths and `art:` ids they name
    (`Failed.named`), every path it can have read (`Failed.reads`, None when nothing bounds them) and what failed
    (`Failed.kind`)."""
    tests: frozenset[str]
    named: frozenset[str]
    reads: frozenset[str] | None
    kind: str | None


def failing(driver: Landing, at: str, d: Path, st: dict, rest: str, tip: str | None = None) -> Read:
    """What failed step ``st`` of the check that gates says in its run dir ``d``, the check of ``tip``'s tree.  A step that
    runs suites (`SUITE_STEPS`): `read_suite` of each suite of its that failed (`suite_rows`) uncovered by ``at``'s allowances
    (its base's), with its log, its guard's report and the reads its children traced (`suites/SLUG.json` and `.reads` beside
    its log) and what it declares at ``tip`` (`suite_reads`); ``rest`` (`unnamed`) alone, its reads unbounded, when no
    suite's record says it failed.  Any other: `read_step`, a Lean build's failure reading the failing modules' imports at
    ``tip`` (`lean_reads`) and any other step what declares its reads there (`reads`).  Its parts together (`joined`)."""
    step = st.get("step")
    if step not in SUITE_STEPS:
        log = _step_log(d, st)
        got = reads(driver.repo, tip, step) if tip else None
        if step == "lean-audit" and tip and audit_built(log):
            got = lean_reads(driver.repo, tip, lean_errors(log)) or got
        return read_step(step, st, log, rest, tree_root(d), got if isinstance(got, frozenset) else None)
    parts = []
    for name, row in suite_rows(d).items():
        if row["status"] != "failed" or row["step"] != step or driver._allowed(at, f"suite {name}", d, row).covered:
            continue
        declared, dirs = suite_reads(driver.repo, tip, row.get("suite") or ".") if tip else (None, frozenset())
        slug = d / "suites" / Path(row.get("log") or f"{name}.log").stem
        parts.append(read_suite(name, row, _suite_log(d, row), rest, declared, dirs, tree_root(d), _json(slug.with_suffix(".json")),
                                _text(slug.with_suffix(".reads"))))
    return joined(parts, rest)


def read_suite(name: str, row: dict, log: str, rest: str, declared: frozenset[str] | None, dirs: frozenset[str] = frozenset(),
               root: str | None = None, report: dict | None = None, trace: str = "") -> Read:
    """What failed suite ``name``'s record ``row`` (a row of a check's suite report) and its log ``log`` say, the check's tree at
    ``root`` (`landing.tree_root`): `SUITE::TEST` of each test it failed, and ``rest`` under the suite's name when it names none
    or failed for more than its tests (its row's `beyond_tests` isn't false); the paths they name (`landing.suite_named`);
    ``declared``, the paths it declares (`suite_reads`, with ``dirs`` the directories among them), narrowed by what its run
    traced (`traced_reads`: its guard's ``report`` and its children's ``trace``); and `suite_kind`."""
    tests = {f"{name}::{t}" for t in row.get("failed_tests") or ()}
    if not tests or row.get("beyond_tests") is not False:
        tests.add(f"{name}::{rest}")
    base = "" if (row.get("suite") or ".") == "." else row["suite"]
    modules = {os.path.normpath(os.path.join(base, t.split("::")[0])) for t in row.get("failed_tests") or ()}
    return Read(frozenset(tests), frozenset(suite_named(row, log, root) or ()),
                traced_reads(declared, dirs, base, modules, report, trace), suite_kind(row, log))


def read_step(step: str, st: dict, log: str, rest: str, root: str | None = None, reads: frozenset[str] | None = None) -> Read:
    """What failed step ``st`` (a line of a check's steps.jsonl), not a suite step, and its log ``log`` say: the test id of each
    `FAILED` and `ERROR` line of its log's short summary (all of it before ` - `, a parametrized id's spaces included), and of
    lean-audit what each failed package's lines name (`audit_failures`), ``rest`` (`unnamed`) alone when it names none; the
    paths it names (`landing.step_named`); ``reads``, what it read; and `step_kind`."""
    tests = {m[1] for m in _SUMMARY_RE.finditer(log)} | (audit_failures(log, rest) if step == "lean-audit" else set())
    return Read(frozenset(tests or {rest}), frozenset(step_named(step, log, root) or ()), reads, step_kind(st, log))


def joined(parts: Iterable[Read], rest: str) -> Read:
    """One failure of ``parts`` (a step's, of several suites or shards): their tests, names and reads together (unbounded when
    one's are), and the first class of `KINDS` one has, its own work before the rest; of none, ``rest`` (`unnamed`) alone,
    its reads unbounded."""
    if not (parts := list(parts)):
        return Read(frozenset({rest}), frozenset(), None, None)
    kinds = {p.kind for p in parts}
    return Read(frozenset().union(*(p.tests for p in parts)), frozenset().union(*(p.named for p in parts)),
                None if any(p.reads is None for p in parts) else frozenset().union(*(p.reads for p in parts)),
                next((k for k in KINDS if k in kinds), None))


def suite_kind(row: dict, log: str) -> str:
    """What a failed suite's record and log say failed (`KINDS`): a timeout (its row's `timed_out`); a flake, each test it
    failed having passed under the same key (`nondeterministic`); the store or build state it shares (`STORE_SAYS`); its
    machine (`vqueue_records.FAULT_RE`: a full disk, no memory; `MACHINE_SAYS`), in what its run printed, not in a test's code its
    report quotes (`_printed`); else its own work."""
    failed = set(row.get("failed_tests") or ())
    if row.get("timed_out"):
        return "timeout"
    if failed and failed <= set(row.get("nondeterministic") or ()):
        return "flaky"
    if STORE_SAYS.search(log):
        return "store-state"
    printed = _printed(log)
    return "infra" if FAULT_RE.search(printed) or MACHINE_SAYS.search(printed) else "own"


def _printed(log: str) -> str:
    """A pytest log's lines but the source a traceback quotes (indented, its failing line marked `>`): what its run printed,
    its error lines (`E`) among them."""
    return "\n".join(line for line in log.splitlines() if not line.startswith((" ", "\t", ">")))


def step_kind(st: dict, log: str) -> str:
    """What a failed step's record and log say failed (`KINDS`): the store or build state it shares (`STORE_SAYS`: an audit
    whose build outputs another build changed); a kill (a signal or 128+N, or the Lean check's own process killed,
    `landing.KILLED_SAYS`); its machine (`vqueue_records.FAULT_RE`, `MACHINE_SAYS`); else its own work."""
    rc = st.get("rc")
    if STORE_SAYS.search(log):
        return "store-state"
    if isinstance(rc, int) and (rc < 0 or rc in KILLED_RC) or KILLED_SAYS in log:
        return "killed"
    return "infra" if FAULT_RE.search(log) or MACHINE_SAYS.search(log) else "own"


def traced_reads(declared: frozenset[str] | None, dirs: frozenset[str], base: str, modules: set[str], report: dict | None,
                 trace: str) -> frozenset[str] | None:
    """``declared`` narrowed as `suites.py` keys a test module's pass where its run traced what it read ("Traced", `traced_key`):
    the declared paths but the directories ``dirs``, each file the suite's processes and their children opened and each
    directory they listed (the guard's ``report``'s `traced`, and the `F` and `L` lines of the children's ``trace``, the
    suite's `.reads`), what its test modules read of each other (`reads`, `R`), its configuration, its test modules and
    conftests, `SUITE_RUNNER` and `ENV`.  ``declared`` itself when the run traced nothing, ``modules`` (the failing tests'
    files, repository paths as the guard keys them) is empty, or one of them, a session fixture or a conftest started a
    program no tracer sees into (its report's `untraced`, a trace's `X` and `E` lines): `suites.py` keys those on the
    suite's inputs whole.  Either holds each path its processes or their children read outside the declared ones (`outside`,
    `O`), which fail it.  A `.reads` line is its kind, the test module and the path."""
    if declared is None:
        return None
    report = report or {}
    outside = set(report.get("outside") or ())
    blind = set(report.get("untraced") or ())
    read = {line.partition("\t")[2] for line in report.get("traced") or () if line[:2] in ("F\t", "L\t")}
    read |= {p for ps in (report.get("reads") or {}).values() for p in ps}
    for line in trace.splitlines():
        kind, module, path = (line.split("\t") + ["", "", ""])[:3]
        if kind in ("X", "E"):
            blind.add(module)
        elif kind == "O" and path:
            outside.add(path)
        elif kind in ("F", "L", "R") and path:
            read.add(path)
    if not modules or not isinstance(report.get("traced"), list) or "*" in blind or modules & blind:
        return frozenset(declared | outside)
    own = f"{base}/" if base else ""
    return frozenset({*(declared - dirs), *read, *outside, f"{own}pyproject.toml", f"{own}**/test_*.py",
                      f"{own}**/conftest.py", *SUITE_RUNNER, *ENV})


@functools.lru_cache(maxsize=64)
def suite_reads(repo: Path, commit: str, suite: str) -> tuple[frozenset[str] | None, frozenset[str]]:
    """(every path suite ``suite`` (its directory) may read at ``commit``, the directories among them), as `suites.py`'s
    `inputs_of` allows its guard: its directory and its declared inputs; unless it declares `inputs_only`, what each workspace
    package it depends on ships (`_ships`), its dependencies, extras and dependency groups and theirs but their groups
    (`suites.closure`); the workspace's pyproject.toml, `SUITE_RUNNER` and `ENV`.  (None, none) for the repository's own
    suite, whose directory is the tree."""
    suite = suite.strip("/") or "."
    if suite == ".":
        return None, frozenset()
    data = suite_pyproject(repo, commit, suite)
    tests = data.get("tool", {}).get("verity", {}).get("tests", {})
    out = {suite, *map(str, tests.get("inputs") or ()), "pyproject.toml", *SUITE_RUNNER, *ENV}
    if tests.get("inputs_only") is not True:
        pkgs, seen = _members(repo, commit), set()
        todo = [n for n in _requires(data) if n in pkgs]
        while todo:
            if (n := todo.pop()) in seen:
                continue
            seen.add(n)
            todo += [r for r in _requires(suite_pyproject(repo, commit, pkgs[n]), groups=False) if r in pkgs]
        out |= {p for n in seen if pkgs[n] != suite for p in _ships(repo, commit, pkgs[n])}
    r = _git(repo, "rev-parse", "--verify", "-q", f"{commit}^{{tree}}", check=False)
    files = [p for p, _ in _files(repo, r.stdout.strip())] if not r.returncode else []
    return frozenset(out), frozenset(p for p in out if "*" not in p and "?" not in p and _is_dir(files, p))


def _is_dir(files: list[str], path: str) -> bool:
    """Whether ``path`` is a directory of the tree whose files (in git's order) are ``files``."""
    import bisect

    i = bisect.bisect_left(files, path.rstrip("/") + "/")
    return i < len(files) and files[i].startswith(path.rstrip("/") + "/")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requires(data: dict, groups: bool = True) -> list[str]:
    """The distributions a workspace member's pyproject.toml ``data`` names, normalized: its dependencies and extras, and with
    ``groups`` its dependency groups (`suites._requires`)."""
    project = data.get("project", {})
    reqs = [*project.get("dependencies", []), *(r for extra in project.get("optional-dependencies", {}).values() for r in extra)]
    reqs += [r for g in data.get("dependency-groups", {}).values() if groups for r in g if isinstance(r, str)]
    return [_norm(m[0]) for r in reqs if (m := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", r))]


def _members(repo: Path, commit: str) -> dict[str, str]:
    """{distribution: directory} of the workspace packages at ``commit`` (`suites.members`), the root's own under "."."""
    root = suite_pyproject(repo, commit, ".")
    out = {_norm(n): "." for n in [root.get("project", {}).get("name")] if isinstance(n, str)}
    for path in root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", []):
        if isinstance(n := suite_pyproject(repo, commit, path).get("project", {}).get("name"), str):
            out[_norm(n)] = path
    return out


def _ships(repo: Path, commit: str, path: str) -> list[str]:
    """What a workspace package's dependents read of it (`suites.shipped`): its `exports`, else its directory, or for the
    root, which holds the whole tree, the packages its wheel ships."""
    data = suite_pyproject(repo, commit, path)
    if (exports := data.get("tool", {}).get("verity", {}).get("tests", {}).get("exports")) is not None:
        return [str(e) for e in exports]
    if path != ".":
        return [path]
    return [str(p) for p in data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
            .get("packages", ["**"])]


def audit_built(log: str) -> bool:
    """Whether lean-audit's ``log`` says only that builds failed: each failed package's lines (`_package_lines`) are its
    build's, or say a package it reads the facts of failed to build (`AUDIT_BUILT_RE`), and the audit failed nothing of its
    own (`AUDIT_FAIL_RE`)."""
    lines = _package_lines(log)
    return (bool(lines) and all(built or AUDIT_BUILT_RE.match(t) for ts in lines.values() for t, built in ts)
            and not AUDIT_FAIL_RE.search(log))


def lean_reads(repo: Path, commit: str, files: Iterable[str]) -> frozenset[str] | None:
    """What a Lean build of the modules ``files`` (repository paths) reads at ``commit``: each and, transitively, each module
    it imports that the tree holds (the `.lean` files whose path ends in the module's name, every one when several do) and
    each file one reads as it compiles (`INCLUDE_RE`), the Lake files and toolchain beside each package holding one, the Lean
    check's code and `ENV`; an import the tree holds no file for is a dependency's, which those Lake files pin.  None when
    ``files`` is empty or one isn't in the tree."""
    r = _git(repo, "rev-parse", "--verify", "-q", f"{commit}^{{tree}}", check=False) if commit else None
    if r is None or r.returncode or not (files := set(files)):
        return None
    tree = r.stdout.strip()
    listing = [p for p, _ in _files(repo, tree)]
    lean = {p for p in listing if p.endswith(".lean")}
    if not files <= lean:
        return None
    by_name: dict[str, list[str]] = {}
    for p in lean:
        parts = p[:-len(".lean")].split("/")
        for i in range(len(parts)):
            by_name.setdefault(".".join(parts[i:]), []).append(p)
    seen: set[str] = set()
    included: set[str] = set()
    todo = sorted(files)
    while todo:
        seen.update(todo)
        found: set[str] = set()
        for path, text in zip(todo, _blobs(repo, tree, todo)):
            found.update(q for line in IMPORT_RE.findall(text) for name in line.split("--")[0].split()
                         for q in by_name.get(name.replace("«", "").replace("»", ""), ()))
            included.update(os.path.normpath(os.path.join(os.path.dirname(path), f)) for f in INCLUDE_RE.findall(text))
        todo = sorted(found - seen)
    lake = {p.rpartition("/")[0]: p for p in listing if p.rpartition("/")[2] in LEAN_PACKAGE_FILES}
    homes = {d for d in lake if any(f.startswith(d + "/") or d == "" for f in seen)}
    return frozenset({*seen, *included, *(f"{d}/{n}" if d else n for d in homes
                                          for n in (*LEAN_PACKAGE_FILES, "lake-manifest.json", "lean-toolchain")),
                      "tools/verity/lean", *ENV})


def _blobs(repo: Path, tree: str, paths: list[str]) -> list[str]:
    """The text of each of ``paths`` in ``tree`` (one `git cat-file --batch`); a path it doesn't hold is left out."""
    import subprocess

    from .gates import git_env

    p = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, input="".join(f"{tree}:{q}\n" for q in paths).encode(),
                       capture_output=True, env=git_env(GIT_TERMINAL_PROMPT="0"))
    out, data, i = [], p.stdout, 0
    while (nl := data.find(b"\n", i)) >= 0:
        head = data[i:nl].split()
        if len(head) != 3 or not head[2].isdigit():
            i = nl + 1
            continue
        n = int(head[2])
        out.append(data[nl + 1:nl + 1 + n].decode(errors="replace"))
        i = nl + 2 + n
    return out


def _package_lines(log: str) -> dict[str, list[tuple[str, bool]]]:
    """{package: (each line the audit prints under its `AUDIT .../PKG: FAIL`, whether its build printed it)} of lean-audit's
    ``log``, but its run's counts and timings (`AUDIT_SAYS`, with the lines under each), the scratch tree's name left out.
    Of what its build printed (`build: exit N` and lake's output under it) only what names a failure is kept: an `error:`
    line, and a module of lake's list of the targets that failed; lake's progress lines, which hold timings, and its warnings
    never name the same failure twice alike."""
    out: dict[str, list[tuple[str, bool]]] = {}
    pkg, says, build, listing = None, False, False, False
    for line in log.splitlines():
        if line.startswith("AUDIT"):
            pkg = m[1] if (m := LEAN_AUDIT_RE.match(line)) else None
            if pkg:
                out.setdefault(pkg, [])
        elif pkg and line.startswith(" ") and (text := line.strip()):
            if re.match(r"  \S", line):
                says, build, listing = text.startswith(AUDIT_SAYS), text.startswith("build:"), False
                if says:
                    continue
            elif says:
                continue
            elif build:
                if "Some required targets logged failures:" in text:
                    listing = True
                    continue
                listing = listing and text.startswith("-")
                if not (text.startswith("error:") or listing and re.fullmatch(r"- [A-Za-z][\w.]*", text)):
                    continue
            out[pkg].append((SCRATCH_RE.sub("", text), build))
    return out


def audit_failures(log: str, rest: str) -> set[str]:
    """The failures lean-audit's ``log`` names: `PKG::LINE` of each line of a failed package's (`_package_lines`: of its
    build, each failed module and each error), ``rest`` under the name of a package it prints none under; and each
    `lean-audit: FAIL:` line."""
    return ({f"{p}::{t}" for p, ts in _package_lines(log).items() for t in [t for t, _ in ts] or [rest]}
            | set(AUDIT_FAIL_RE.findall(log)))


def _json(path: Path) -> dict | None:
    """The JSON object at ``path`` (a failed suite's guard report), None when there is none to read: missing, empty, cut
    short or not an object, as `suites.read_report` reads the report a pytest that died in its session finish left.  A
    raise here would fail every tick that judges the run, since a refetch brings the same bytes."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        if path.exists():
            print(f"research queue vqueue: {path}: no report ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    return doc if isinstance(doc, dict) else None


def _text(path: Path) -> str:
    """The text at ``path`` (a failed suite's children's reads), empty when the run dir has none."""
    return path.read_text(errors="replace") if path.exists() else ""


def combine(graph: Verdict, slim: Verdict) -> Verdict:
    """A candidate's verdict from its graph2 check's (``graph``: its gating tasks) and the run beside it's (``slim``,
    `judge`): fail when either failed; else infra when either ended in infra errors; else pending while either is; else
    list-failed when graph2's check failed to list; else pass, both passed.  It holds both verdicts' failures and infra
    errors."""
    if bad := sorted({graph.state, slim.state} - set(STATES)):
        raise ValueError(f"no verdict has state {', '.join(map(repr, bad))}: a state is one of {', '.join(STATES)}")
    state = next(s for s in ORDER if s in (graph.state, slim.state))
    return Verdict(state, graph.failed + slim.failed, graph.infra + slim.infra)


#: the steps whose reads can't be bounded, each keyed on the whole tree, and why
WHOLE = {"preflight-lints": "its lints read every file (`check.lints_key`)",
         "ratchet-gate": "it judges the merges above the base, not a tree"}


def reads(repo: Path, commit: str, step: str) -> frozenset[str] | str:
    """What ``step`` reads of ``commit``'s tree, as `vqueue.meets` reads paths, directories and globs; for a step read whole,
    why: one whose reads can't be bounded (`WHOLE`, or a step this doesn't know), or whose declaration ``commit`` doesn't hold
    as plain data.  Each runs under check.py, so each bounded one holds check's reads (`CHECK_READS` at ``commit``) and `ENV`:

    - lean-agreement: the graph build family's lean-agreement task's inputs, `VERIFIER`'s files beside check's, and the
      Lean package `LEAN` whose lean-build it reads, whole (its lock and nested packages too, which `package` cuts);
    - preflight-lock: `LOCK_READS`;
    - preflight-fixtures: the replica map (`MAP`), each path it registers (`research.store.fixtures.entries`), and the code
      its check imports (`FIXTURES_CODE`)."""
    if step in WHOLE:
        return WHOLE[step]
    if step not in ("lean-agreement", "preflight-lock", "preflight-fixtures"):
        return f"nothing declares what {step} reads"
    if isinstance(got := _declared(repo, commit), str):
        return got
    check = (*got["CHECK_READS"], *ENV)
    if step == "lean-agreement":
        return frozenset((got["VERIFIER"], got["LEAN"], *check))
    if step == "preflight-lock":
        return frozenset((*LOCK_READS, *check))
    if isinstance(doc := _map(repo, commit), str):
        return doc
    return frozenset((MAP, *entries(doc).values(), FIXTURES_CODE, *check))


def key(repo: Path, commit: str, step: str) -> str | None:
    """``step``'s content key at ``commit``: SHA-512 over `KEY_TAG`, the step's name and each file of ``commit``'s tree it
    reads (`reads`), as git holds it (mode, object and path), in path order; for a step read whole, over the tag, its name and
    the commit's tree.  Two commits with one key give the step the same files to read.  None when git can't read ``commit``."""
    r = _git(repo, "rev-parse", "--verify", "-q", f"{commit}^{{tree}}", check=False)
    if r.returncode:
        return None
    tree = r.stdout.strip()
    h = hashlib.sha512(f"{KEY_TAG}\0{step}\0".encode())
    if isinstance(want := reads(repo, tree, step), str):
        h.update(f"tree\0{tree}".encode())
        return h.hexdigest()
    for path, entry in _files(repo, tree):
        if meets((path,), want):
            h.update(f"{entry}\t{path}\0".encode(errors="surrogateescape"))
    return h.hexdigest()


def lean_key(repo: Path, commit: str) -> str | None:
    """SHA-512 over `KEY_TAG` and each Lean input of ``commit``'s tree (`lean_inputs`) as git holds it (mode, object and path),
    in path order.  Two commits with one Lean key have the same Lean inputs, which are all a lock run's regenerated records
    depend on, so records regenerated on one are the other's too (`vqueue_records.commit`'s ``onto``).  None when git can't
    read ``commit``; `QueueError`, from `lean_inputs`, when the audit's declaration of what it reads can't be loaded."""
    r = _git(repo, "rev-parse", "--verify", "-q", f"{commit}^{{tree}}", check=False)
    if r.returncode:
        return None
    files = _files(repo, r.stdout.strip())
    inputs = frozenset(lean_inputs(path for path, _ in files))
    h = hashlib.sha512(f"{KEY_TAG}\0lean inputs\0".encode())
    for path, entry in files:
        if meets((path,), inputs):
            h.update(f"{entry}\t{path}\0".encode(errors="surrogateescape"))
    return h.hexdigest()


@functools.lru_cache(maxsize=8)
def _files(repo: Path, tree: str) -> tuple[tuple[str, str], ...]:
    """(path, "MODE TYPE OBJECT") of each entry under ``tree`` (an object id, which never changes, so a listing is kept), in
    git's order, which is path order."""
    out = _git(repo, "ls-tree", "-r", "-z", "--full-tree", tree).stdout
    return tuple((path, entry) for entry, _, path in (line.partition("\t") for line in out.split("\0") if line))


#: the names `_declared` reads of `BUILD`
DECLARED = ("CHECK_READS", "VERIFIER", "LEAN")


def _declared(repo: Path, commit: str) -> dict | str:
    """`BUILD`'s `DECLARED` at ``commit``, read from its source without running it (`_data`): `VERIFIER` and `LEAN` strings,
    `CHECK_READS` a tuple of them; else why not."""
    r = _git(repo, "show", f"{commit}:{BUILD}", check=False)
    if r.returncode:
        return f"{commit[:12]} has no {BUILD}"
    try:
        body = ast.parse(r.stdout).body
    except SyntaxError as e:
        return f"{BUILD} at {commit[:12]} doesn't parse: {e}"
    got = {node.targets[0].id: _data(node.value, {}) for node in body
           if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in DECLARED}
    check = got.get("CHECK_READS")
    if (isinstance(got.get("VERIFIER"), str) and isinstance(got.get("LEAN"), str) and isinstance(check, tuple)
            and all(isinstance(p, str) for p in check)):
        return got
    return f"{BUILD} at {commit[:12]} doesn't declare {', '.join(DECLARED)} as plain data"


def _map(repo: Path, commit: str) -> dict | str:
    """The replica map at ``commit``, else why not."""
    r = _git(repo, "show", f"{commit}:{MAP}", check=False)
    if r.returncode:
        return f"{commit[:12]} has no {MAP}"
    try:
        doc = json.loads(r.stdout)
    except ValueError as e:
        return f"{MAP} at {commit[:12]} isn't JSON: {e}"
    return doc if isinstance(doc, dict) else f"{MAP} at {commit[:12]} isn't a JSON object"


def _data(node: ast.AST, names: dict[str, str]) -> str | tuple | None:
    """The value of ``node``, an expression of string literals, f-strings of the names a comprehension binds (``names``),
    tuples and lists with starred items, and comprehensions over those: what `BUILD` writes its declarations in.  None for
    anything else, which is never run."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = [_data(v.value if isinstance(v, ast.FormattedValue) and v.conversion == -1 and v.format_spec is None else v,
                       names) for v in node.values]
        return "".join(parts) if all(isinstance(p, str) for p in parts) else None
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list = []
        for e in node.elts:
            got = _data(e.value if isinstance(e, ast.Starred) else e, names)
            if not isinstance(e, ast.Starred):
                out.append(got)
            elif isinstance(got, tuple):
                out.extend(got)
            else:
                return None
        return None if None in out else tuple(out)
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)) and len(node.generators) == 1:
        g = node.generators[0]
        if isinstance(g.target, ast.Name) and not g.ifs and not g.is_async and isinstance(over := _data(g.iter, names), tuple):
            items = tuple(_data(node.elt, {**names, g.target.id: x}) for x in over)
            return None if None in items else items
    return None


def _failure(step: str, k: str, base_key: str | None, parts: list[Read]) -> Failed:
    """Failed step ``step``'s `Failed`, keyed ``k`` at the candidate and ``base_key`` at its base, from what its shards'
    records say of it (`joined`): its culprit named by the paths they name (`Failed.inputs`, which `vqueue.decide` reads)."""
    r = joined(parts, unnamed(k))
    return Failed(f"step {step}", k, r.named, base_key, r.tests, reads=r.reads, named=r.named, kind=r.kind)


def _down(rec: dict | None, why: str) -> Verdict:
    """Infra: what became of the run ``rec`` records."""
    ids = " ".join(rec["ids"]) if rec and rec.get("ids") else ""
    return Verdict(INFRA, (), (Failed(f"the run beside its check{f' ({ids})' if ids else ''} {why}",
                                      f"{rec['tip'] if rec else ''}:{ids or 'no run'}", frozenset(),
                                      launch=why.startswith("registered no run") and any(m in why for m in LAUNCH_ONLY)),))


def _keep(rec: dict, v: Verdict, passed: list[str] | None = None) -> Verdict:
    rec["verdict"] = {**_doc(v), **({"passed": passed} if passed is not None else {})}
    return v


def _doc(v: Verdict) -> dict:
    return {"state": v.state, **{k: [{"task": f.task, "key": f.key, "inputs": sorted(f.inputs),
                                      **({"base_key": f.base_key} if f.base_key is not None else {}),
                                      **({"tests": sorted(f.tests)} if f.tests else {}),
                                      **({"launch": True} if f.launch else {}),
                                      **({"reads": sorted(f.reads)} if f.reads is not None else {}),
                                      **({"named": sorted(f.named)} if f.named else {}),
                                      **({"kind": f.kind} if f.kind else {})} for f in fs]
                                 for k, fs in (("failed", v.failed), ("infra", v.infra)) if fs}}


def _kept(got: dict) -> Verdict:
    return Verdict(got["state"], *(tuple(Failed(f["task"], f["key"], frozenset(f["inputs"]), f.get("base_key"),
                                                frozenset(f.get("tests") or ()), launch=bool(f.get("launch")),
                                                reads=frozenset(f["reads"]) if "reads" in f else None,
                                                named=frozenset(f.get("named") or ()), kind=f.get("kind"))
                                         for f in got.get(k) or ()) for k in ("failed", "infra")), ended=got["state"] == FAIL)


def step_test(step: str) -> str:
    """The one test id of ``step`` of the run beside a check: its failures are compared by it (`vqueue.Failed.tests`), as a
    step read whole (`WHOLE`) is never at its base's key."""
    return f"step:{step}"


def based(v: Verdict, base: dict | None, so_far: bool = False) -> Verdict:
    """``v`` with each failed step's `base_tests` read from the run beside the base's check (``base``, that run's record, as
    `judge` keeps it): the tests that run failed of it (the step's, `step_test`, where it kept none), none when it passed it,
    and None while it runs, or when it ended without judging it, so `vqueue.decide` takes a failure the base has too as
    inherited.  With ``so_far`` (a base that is itself a candidate), one whose check hasn't ended answers with what it has
    failed so far (its ``early``, `_early`) where that holds every test failed of the step, and None otherwise, since it
    may yet fail the rest: a candidate above another is failed on its own failures once its base's word on them is known."""
    got = (base or {}).get("verdict")
    if early := so_far and base is not None and not isinstance(got, dict):
        got = base.get("early") or {}
    if not isinstance(got, dict) or not v.failed:
        return v
    failed = {f.get("task"): f.get("tests") for f in got.get("failed") or () if isinstance(f, dict)}
    passed = set(got.get("passed") or ())

    def of(f: Failed) -> Failed:
        s = f.task.removeprefix("step ")
        if s == f.task:
            return f
        tests = frozenset(failed[f.task] or {step_test(s)}) if f.task in failed else frozenset() if s in passed else None
        return replace(f, base_tests=None if early and tests is not None and not f.tests <= tests else tests)

    return replace(v, failed=tuple(of(f) for f in v.failed))


def audits(driver: Landing, rec: dict | None) -> bool:
    """Whether ``rec`` records the check that gates judged on lean-audit (`_gate`), whose verdict on it a check built on its
    candidate, or on one that took it from this one's, may take in place of running it (`start`'s ``audit_from``)."""
    return rec is not None and rec.get("gate") == CHECK and LEAN_STEP in _gate(driver, rec)


def _audit(rec: dict | None) -> tuple[Failed, ...] | None:
    """The verdict on lean-audit that ``rec`` keeps: its failures of the step, none where it passed it, None where it keeps
    none (no verdict kept, or one that holds neither)."""
    got = (rec or {}).get("verdict")
    if not isinstance(got, dict):
        return None
    failed = tuple(f for f in _kept(got).failed if f.task == f"step {LEAN_STEP}")
    return failed if failed or LEAN_STEP in (got.get("passed") or ()) else None


def answers(rec: dict | None) -> bool:
    """Whether ``rec`` keeps a verdict on lean-audit: its check ran the step, or took that verdict once the candidate whose
    check ran it had landed (`audited`)."""
    return _audit(rec) is not None


def audited(driver: Landing, v: Verdict, rec: dict | None, source: dict | None) -> Verdict:
    """``v``, `judge`'s verdict of the check ``rec`` records, with lean-audit's where that check took it from the check of a
    candidate below that ran it (``lean_audit_from``) and lean-audit gates it (`_gated`): the verdict on the step that
    ``source`` keeps, the record of the run beside `main` once that candidate has landed (its own, or one that took it from it
    too), None while it is in flight, whatever its check says then: a failure there may yet be rerun away, and if it isn't,
    that candidate's drop drops this one with it (`vqueue.decide`'s "its base failed").  Passed there (its kept verdict's
    ``passed``), ``v``.  Failed there, ``v`` failed with that failure, ended once ``v``'s own check has, which `based` reads
    as the base's.  Unknown there, or no ``source`` (`answers`), ``v`` never passes: a pass is pending, a fail not ended,
    which `vqueue.decide` holds while every failure is inherited.  Once known, and ``v`` a pass or fail kept, it is kept in
    ``rec`` as if the check had run the step (``verdict``), so the record answers for it where it is the run beside `main`
    (`based`) and is read from there by the checks that took it too: a pass in ``passed``, and a failure only where the run
    beside `main` fails lean-audit, which is `main` red on it, the candidate that ran it having landed on that failure as
    inherited, so a later candidate's failure of the step reads as inherited there."""
    if not rec or not rec.get("lean_audit_from") or LEAN_STEP not in _gated(driver, rec) or answers(rec):
        return v
    if (failed := _audit(source)) is None:
        return Verdict(PENDING) if v.state == PASS else replace(v, ended=False) if v.state == FAIL else v
    out = v if not failed else Verdict(FAIL, (*v.failed, *failed), v.infra,
                                       ended=v.state in (PASS, INFRA) or v.state == FAIL and v.ended)
    if isinstance(kept := rec.get("verdict"), dict) and v.state in (PASS, FAIL):
        _keep(rec, out, [*(kept.get("passed") or ()), *(() if failed else (LEAN_STEP,))])
    return out
