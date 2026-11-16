"""`research queue vqueue`: the verified queue's loop (`research.vqueue`).  Each tick, under its own lock (`--lock`, never
drive's), derives a `vqueue.Snapshot` from what is outside it, asks the policy (``decide``, `vqueue.decide`) what to do and
applies its actions in order, stopping the tick at the first that fails: the next tick derives everything again.  Between ticks
it keeps only what a snapshot reads: the refs, the labels and the decision log (`--log`), from which come the candidates,
`history`, each candidate's `tries`, `reruns` and first enqueue, the record of the run beside each check and of each records
run, a ring's and a red's once-only and the owners already told.  `--state` is a report it rewrites each tick: each
candidate's verdict, its check's and its records run's, and the `queue/*` branches it ignores.

The snapshot.  `main` is origin's `main`, fetched into the clone's own refs (`refs/vqueue/`), so drive's clone, its worktree and
its refs are left as drive keeps them.  `ready` is the open PRs drive would land from `next` now: admitted as drive admits them
(`Queue.entries`, reading drive's state, the labels read afresh each tick as drive reads them: `marked_after`,
`signed_verdicts` (on "shadow" each mark and verdict counts, and each that doesn't authenticate is logged once:
`unauthenticated`), a mark carried to a head with the same own change, or with an older head's own change mapped across the moves
`main` landed since, as `tools/move/restack.py` carries a branch, from an older head with a land verdict of its own only, since
mapping runs its tree's code: `Queue.mapped_change`, which the end of a tick starts working out, `--move-carry-jobs` heads
at once in jobs detached from the tick, which outlive it), with a land verdict by a non-owner (its head's, or with
drive.toml's `verdict_carry` "on" at `main` one the verdict carry takes from another of its heads whose commits since only
merge trusted ones, `Landing._carried_from`, each decision logged once a head: `carry`) or under the low-risk rule
(`queue.lands_on_mark`, which `research queue review`'s rows read too: drive.toml's `lowrisk_lane` and `lowrisk_excludes` at
`main`; never a held PR's; a move PR on its mark alone while that lane is on and its move is certified on `main`,
`World._uncertified` and `_off_mark`), not on
`main` yet, ranked by drive's priority file then mark time, each after the ready PRs it contains, and held by no candidate
on `main`'s chain it isn't a member of (one that is reaches `main` with that candidate: it is logged once a head and
candidate, `riding`, and nobody is told); `files` is its change against its merge base with `main`, `lean` drive's
Lean-change test (`queue.lean_change`) of a change that compiles (a lock's alone doesn't), and `conflict` whether
`Queue._merge` merges it onto `main`, kept by (head, `main`); `crosses` is the heads of the move PRs, ready or in flight at
their heads as built, whose maps retire nothing its own change names (`queue.crossed`, `World.crossing`): a batch on such a
move may take it (`vqueue.decide`).  The owner of one that doesn't merge is told once a head the files it
conflicts in, to rebase it onto `main` and mark the new head (`Loop._conflicts`), unless `main` fetched again holds that
head; its mark stays.  `left` is why each other open PR into `main` isn't ready (`World.ready`: the queue's word on it, a
withdrawn mark's author, a hold and by whom, no land verdict, the freeze, a move `main` landed that it lacks whose retired
names its own change names, its owner told once a head and move: `queue.lacked_moves`, `Loop._lacking`, or a held PR's
head it holds, directly or through others, its owner told once a head and held head which and what releases it:
`queue.held_below`, `Loop._behind_held`), which a candidate's drop for a member no longer
ready names (`vqueue._stale`); `pulled` those of them others than their owners took (a red's eject, any other eject or a
mark withdrawn by another, a hold by a reviewer or a `hold` label, the freeze, the shadow's eject, or a PR holding one of
them), at its head or at the head a candidate in flight holds it at (`World.taken`), the only ones that drop a candidate in
flight; and `flying` each member of the chain not ready at its head as built, with its files and Lean test.
`candidates` are the loop's own builds: each of origin's `queue/<sha12>` branches at the sha a build line of the log names
(the window's, else the whole log's, which its other lines are then read from too), with that line's base and members, lowest
first; any other `queue/*` branch is no candidate, and is logged once (`ignore`).  `now` is the clock, `lean` and `pending_s`
the command's (`--lean`, `--pending-hours`), and `wide` and `pinned` the PRs `--wide-file` and `--pins-file` name, read each
tick as the priority file is (`World.wide`, `World.pinned`), `wide` with the PRs `--heavy-file` names while `--heavy-on` names
machines (`World.heavy`; its `*` makes none wide); a tick that builds nothing while a Lean PR waits for a chain it may go on
logs why, once a why (`wait`).  A tick that applied every action logs why each PR marked ready waits in no candidate, once
for each PR, head and cause (`leave`): the policy's reason for a PR it read (`vqueue.left_out`), or why `World.ready`
doesn't take one the queue admits (`World.held_out`).  A candidate's verdict is its check's (`--gate`): the run beside it
(`research.vqueue_slim`) starts with its candidate's build (so up to `--depth` at once, on drive's machines), or once in the
snapshot for a candidate on `main`'s chain the log holds no start of; pending until it ends, it is judged each tick, its record
logged whenever it changes.  One more runs beside `main`'s own check: the candidate's `main` advanced to, else one the
snapshot starts at startup and each time `main` moves (and once more after infra).  A step a candidate's run failed is
inherited when `main`'s failed it too, and its own when `main`'s passed it (`vqueue_slim.based`); while `main`'s runs, at
most `pending_s` after it started, a fail whose every other failure is inherited is pending.  With ``records``
(`research.vqueue_records`), a candidate built with its records run first (its build line's `records_first`, while no line
enqueued its check) is staged, off main's chain (`vqueue.decide`): its check is pending and doesn't start, and its records
run is judged each tick (`vqueue_records.judge`), its record logged whenever it changes (`regenerate`).  A records commit's
build line, and a staged batch's built again on another base, names the staged batch it `repairs`, and the walk of a
candidate's members steps over a records commit at its tip (`World._walk`).

The gate (`--gate`).  With `check`, the default, drive's after-landing check gates: the run beside each candidate, and beside
`main`, is every after-landing step as `--shards` shards on `--on`'s machines (`vqueue_slim.start` with ``shards``: N,
ci.toml's plan, or `auto:N`, an auto plan of the candidate's tree, which splits pytest by suite), judged on every step `main`
waits for; a run started under another gate is started again as the check.  With `verdict`,
the land verdicts gate: as `check`, but a plain candidate (`Loop._plain`: each member has a land verdict of its own at its
head, is no move, changes no Lean input and meets nothing drive.toml at `main` names in `lowrisk_excludes` but not in
`verdict_plain`, or in `verdict_excludes`) lands on those verdicts and the quick tier on its own merge (`vqueue_slim.start_quick`).  `main`'s whole check follows each such landing
(`Loop._below`), and a red of it (`Loop._main_red`) reverts the one PR landed since `main`'s last passed check whose change
meets what the failure read (`World.revert`), holding the ready PRs that meet it too until `main`'s check passes again; with
no such PR it reverts none, and every candidate takes the whole check until `main`'s passes.  Every other candidate takes the
whole check before it lands, as under `check`.  A Lean
candidate's records run comes first under either gate: its check doesn't start until that run passed, and starts on its
records commit, or on it as built when the run regenerated nothing (`Loop._enqueue`).  A candidate built
on another in flight, no member of it changing a Lean input against its merge base with `main` (`vqueue_slim.lean_inputs`),
runs every step but lean-audit where its base's check is judged on lean-audit, or took its verdict on it from the check of a
candidate below that is.  It reads lean-audit as pending while the candidate whose check ran it (its record's
``lean_audit_from``) is in flight, whatever that check says then, and once that candidate has landed takes the verdict the
run beside `main` keeps (`vqueue_slim.audited`): it lands after it, drops with it as its base failed, and stays pending while
it is rerun, so a failure the rerun clears is never its own.  One started once its base is `main`, or once that candidate
has landed while the run beside `main` keeps no verdict on lean-audit, audits.  Its check also leaves to its base's the
suites keyed in the base's tree as in its own (`Loop._suites_from`, `suites.py --lend-from`), which that check is running:
only while the check of `main` passed both suite steps and the base's is judged on them under the policy of its own base
(`vqueue_slim.lends_suites`).  It passes only once its base's check passed both suite steps (`vqueue_slim.borrowed`), so
its pass of a suite it left out is that check's, and where its base landed on a check that didn't, it is started again on
`main` and runs them (`Loop._unanswered`).  A failed step
is inherited by its failing tests, those the run beside its base's check failed too.  A shard that failed a step it is
judged on, or a suite of one, fails it while the other shards still run (an early red, `vqueue_slim.judge`), decided as a
finished fail is: on the lowest candidate at once, on one above it once its base's check has ended (`vqueue_slim.based`),
nothing built on it meanwhile.  Its check is cancelled with the candidates above it, and its drop's line says when the
first failure was written and when it was dropped (`early_red`).  The steps the
check leaves out (policy.toml's `off_gate`, drive.toml's `warn_steps`) still run, and hold no pass: a check passes once every
step it is judged on passed in every shard holding it (`vqueue_slim._passed_early`), and its landing's drop leaves the runs
still running one, which any other drop cancels.  Once a candidate's verdict is pass or fail and its runs ended, what it
failed of one that the run beside its base's check didn't is rung, once a candidate and step, to who acts on it: the lane of
each member whose merge ratchet-gate fails, the owner of each circuit-check target's pins (`Loop._off_gate`), read on past its
drop while they or the run beside its base's check run (`Past.off_gate_running`); a red its base shares rings nobody.

The actions.  Build merges its members in order onto its base (`Queue._merge`), or, for a base a build of the same tick
rebuilt (`vqueue.Build.replaces`, logged as its line's `replaces`), onto what that build pushed (`Tick.rebuilds`), leaving
out one that doesn't merge (`skip`, its owner told once per head and base, a conflict with members merged above `main`
naming them where it merges onto `main`;
nobody is told of a head what it merges onto holds, `held`, or `main` fetched again does), pushes `queue/<sha12>` (or takes
as its own one no line names that is the same build made before, as a tick cut off after its push leaves: `adopted`), logs
it and starts the run beside its check; with `records` (a member changes
Lean, a lock or a spec) it starts the candidate's records run instead (`vqueue_records.start`: drive's lock run, on drive's
machines, on its tree), logged (`regenerate`), and the candidate is staged, its check waiting; a tick that stages one builds
its batch of PRs with no Lean change too, on the same base.  One after a Commit of its tick (`vqueue.Build.after`) is built
on the candidate that Commit put on main's chain, whose check's record, as started, it reads as its base's; it is built
nowhere, and nothing is logged, when the Commit put none there.  Every check starts at one seam (`Loop._enqueue`: its line
and the run beside it).
A candidate holding a heavy PR (`World.heavy`), or every candidate and `main`'s own check while the heavy file holds `*`,
runs its records run, and, where the check gates, the shard of its check that audits `vqueue_slim.HEAVY_PART`, on
`--heavy-on`'s machines alone, its other shards on theirs (`Loop._heavy_on`); a records run's resend after a machine's fault
stays on them, as a check started again does.
Advance pushes `main` to the candidate with a lease on its base and logs it; then what follows `main` moving
(`Landing._on_main`) is a step of its own (`on_main`), which the next tick runs again until one is logged done.  A push that
took but raised logs no advance done: the drop of a passed candidate as on `main` logs it then (`Loop._reached`).  Outside
--shadow and --dry-run those steps run detached from the tick, one advance's at a time (`--after-landing`, `Loop._moved`),
their output in `AFTER_LOG` beside the log; the first tick after they ended logs how, and a failure is printed and run again.
Regenerate starts the candidate's records run again and logs it.  Commit puts a staged batch whose records run passed onto
main's chain's top (its base) and starts its check (`Loop._commit`).  On its own base: on it as built when the run
regenerated nothing (`vqueue_records.commit`); else on the records commit of what it regenerated, made on top of it in drive's
lock commit's form, which the merge gate vouches for, and pushed: a candidate on the same base with the same members, which
`repairs` the one beneath it, dropped for it the next tick.  On another base, its members are merged onto it again; with every
member merged and the same Lean inputs (`vqueue_records.same_inputs`), that records commit is carried onto the merge and
checked in its place (the merge itself when there is none), else the merge is staged again, its records run first, and with no
member merged it is dropped.  The bookkeeping its records run repaired is told as drive tells it, to `lean` and each owner a
warning names, once a records commit (`bookkeeping`); a records commit that can't be made or vouched for is held, logged
(`commit`) and rung to `CI` once, and its candidate is dropped once it is held `pending_s` (`vqueue.decide`).
Drop cancels the run beside the check and its records run while they run (one
still launching once it printed its run ids, a later tick), deletes the branch and logs its outcome, with its base and its
inherited failures, their inputs whole, which `history` reads back.  Eject withdraws the PR's ready mark at its head
(`Queue.eject`, under the clone's queue lock) and tells its owner the failed tasks, their keys and where their logs are.
Retry logs an enqueue and starts a run beside it that ended in infra again.  Rerun starts the run beside the check again,
cancelling it while it runs.  Ring tells whom it names (`to`), else `landing.INFRA` (or its `pr`'s owner), once a
candidate and why; a ring to an owner is held, only logged, while another candidate's check or the run beside `main`'s reads
infra in the same tick, as in an outage (`Loop._outage`).  Red tells `CI` that the base is red on what it fails, once a
base and task.  A skip's, conflict's, ring's or red's message is logged as about to be posted (`posting`) before it is, and its record says whether it was (`posted`): one
whose tick stopped in between is told, so no one is told twice, and one whose post failed is told again.  Each action and its
result is one JSON line in the log, whose directory also holds the loop's caches.  Each tick prints one line of its steps'
seconds and the peak resident size after each (`Steps`).

`--shadow` logs advance, eject, ring, red and every message with `shadow: true` and does none of them, and builds,
drops, retries, reruns, starts the runs beside and the records runs, and commits records: its `main` is the newest candidate
it advanced (`refs/vqueue/shadow` in the clone, never pushed, else its log's newest advance) while origin's `main` is below
it, a pass its `main` no longer holds is no history, and a PR it ejected stays out until marked again.  `--no-records` starts
every candidate's check at once, with no records run before it: stale records fail it as any failure.  `--dry-run` prints the
snapshot and the actions and writes nothing: no ref on origin, no log, no enqueue, no label, no message, no run.  The loop
never writes drive's state, its priority file, `next`, `checked` or a label, but an eject's without `--shadow`.

On a node it runs as vy-vqueue (tools/research/src/research/pods/nebius/vqueue/loop.sh): one `--once` tick a pass, from ci's
pinned tree in the loop's own clone, with its args file's flags and the node's log, state, lock, priority and freeze files, and
`--dry-run` or `--shadow` as its mode says, neither when live.  `--batch`, `--depth` and `--lean` default to 16, 3 and 3.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

from . import carry, vqueue, vqueue_records
from .landing import CI, Landing, authed, cancel_runs, merge_thread_post
from .landing import INFRA as INFRA_HANDLE
from .queue import (LEAN_SOURCES, MOVE_CHECK, MOVE_SCRIPTS, PR, SIGNED_GUARDS, WORKDIR, DriveFlags, Entry, Member, Queue,
                    QueueError, StoreLabels, _git, _out, _who, base_drivers, conflicting_files, flags_at, forces_cold,
                    label_verifier, lands_on_mark, lean_change, listed, lowrisk, merged_tree, move_of, own_files,
                    priority_list, queue_commit, queue_log, review_owners, rules_at, touched_by, unreviewed)
from .queue import Retired, crossed, held_below, held_waits, lacked_moves, lacks_why, merge_main
from .reds import CIRCUIT_KILLED, CIRCUIT_STEP, RATCHET_STEP, red_owners
from .store import vocab
from .vqueue import (FAIL, INFRA, ON_MAIN, PASS, PENDING, Action, Advance, Build, Candidate, Commit, Drop, Eject,
                     Failed, Leave, Outcome, Ready, RecordsRun, Red, Regenerate, Rerun, Retry, Ring, Snapshot, Verdict, Wait)
from .vqueue_records import LOCK_SUBJECT

LOCK, LOG, STATE = Path("/data/vqueue/vqueue.lock"), Path("/data/vqueue/decisions.jsonl"), Path("/data/vqueue/state.json")
#: drive's priority list (its `--priority-file`)
PRIORITY = Path("/data/drive/priority")
#: a freeze's paths, one git pathspec a line (`#` lines skipped): while the file exists, a PR whose change renames or deletes a
#: file under one waits, as does one that moves a word map there; ordinary edits don't, nor does a head the freeze's owner
#: cleared against the branch it protects (a `cleared-by HANDLE KEY` line: a `KEY = true` label by HANDLE at the head)
FROZEN = Path("/data/vqueue/live/frozen")
#: the tests a candidate's check leaves out, one node id a line from the repository's root (`#` lines skipped), read at each
#: launch (`vqueue_slim.deselection`); the nightly uncached check, which reads no list, runs them
DESELECT = Path("/data/vqueue/live/deselect")
#: the runs a cancel leaves running, one run id a line (`#` starts a comment), read at each cancel: a first shard whose Lean audit
#: runs, which writes its node's module index only when it ends, so that a kill throws its compile away
KEEP_RUNS = Path("/data/vqueue/live/keep-runs")
#: the PRs whose Lean change is wide, its build expensive, one number a line as the priority file holds them: a candidate
#: taking one takes no other Lean member, and one on a candidate holding one takes none until it is on `main` (`vqueue._wide`)
WIDE = Path("/data/vqueue/live/wide")
#: the PRs whose Lean change is heavy, its audit's replay more memory than a check slot holds, one number a line as the wide
#: file holds them: wide too, and a candidate holding one runs its records run and the shard of its check that audits
#: `vqueue_slim.HEAVY_PART` on `--heavy-on`'s machines alone (`Loop._heavy_on`); `*`, every candidate, none of them wide
HEAVY = Path("/data/vqueue/live/heavy")
#: the pinned PRs, one number a line (`#` comments and blank lines allowed): while a pinned Lean PR waits, a Lean batch holds
#: pinned Lean PRs only, under no `LEAN` limit: a pin orders a Lean batch or cuts it, and at `--lean 0` only a pinned Lean PR
#: opens one; it puts no batch ahead of a PR changing a Lean input, which goes first (`vqueue._next`)
PINS = Path("/data/drive/pins")
#: how far back the decision log is read
WINDOW_S = 48 * 3600
#: what gates a candidate (`--gate`): drive's after-landing check, as `SHARDS` shards unless `--shards` says another, which is
#: also the ``gate`` of that check's record (`vqueue_slim.CHECK`); or the land verdicts, the check but for a plain candidate
#: (`Loop._plain`), which lands on its members' land verdicts and the quick tier on its own merge (`vqueue_slim.start_quick`),
#: `main`'s own check after it (`Loop._main_red`)
GATES = CHECK_GATE, VERDICT_GATE = "check", "verdict"
#: how many commits of `main`'s first-parent line the land verdicts' gate reads back for `main`'s own checks (`Loop._main_red`)
MAIN_WALK = 64
#: the cause of a ready PR held while `main`'s own check is red on what its change meets (`Loop.held`), in its `leave` line
MAIN_FAILED = "main's check failed"
SHARDS = 3
INTERVAL_S = 60
#: the most members a candidate holds, candidates stacked at once and members of a Lean batch, each with a Lean change that
#: compiles (a spec's or Lean source's, not a lock's alone: `World._judge`), unless pinned (`PINS`)
BATCH, DEPTH, LEAN = 16, 3, 3
#: the clone's refs the loop fetches into, none of drive's
NS = "refs/vqueue"
MAIN = f"{NS}/main"
#: the old path's landing branch, frozen since the switch: its landings passed drive's rules and reach `main` in aggregates
CHECKED = f"{NS}/checked"
#: the shadow's own `main`, never pushed: the newest candidate it advanced
SHADOW = f"{NS}/shadow"
#: origin's `main` fetched again before a notice is posted (`World.main_now`), leaving the tick's `MAIN` as it read it
MAIN_NOW = f"{NS}/main-now"
#: no automatic gc or maintenance under drive's own commands in its clone
QUIET = ("-c", "gc.auto=0", "-c", "maintenance.auto=false")
TIME, STAMP_RE = "%Y-%m-%dT%H:%M:%SZ", re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
#: who ejects, as drive's ejections are by
EJECTOR = "ci"
#: a lock commit's trailer naming the files it regenerated (`vqueue_records.lock_message`, which post_train reads alike)
REGENERATED_RE = re.compile(r"^Regenerated: \S", re.M)
#: the most first-parent commits a candidate's walk reads above its base
WALK = 64
#: the PRs judged at once (`World._judge`): a merge onto `main` takes seconds in drive's clone, and a moved `main` judges every
#: ready PR again
JOBS = 8
#: what `World.ready` judged of each open head on `main`, in the loop's cache, for the next tick's process (`--once`)
JUDGED_CACHE = "judged.json"
#: the most failures of a step off the gate a ring quotes (`Loop._off_gate`)
OFF_SAID = 10
#: the old heads mapped across `main`'s moves at once, each in a job the end of a tick detaches and no tick waits on
#: (`Queue.work_out_mapped`; `--move-carry-jobs`): each runs every move's script on a whole tree, so few, leaving the control
#: pod's few cores room for the loop
MAPPED_JOBS = 2
#: the directory beside the log of the after-landing jobs (`Loop._moved`), each a directory of its job, and their output
AFTER, AFTER_LOG = "after-landing", "after-landing.log"
#: the after-landing job: the tick's own command with `--after-landing DIR` (``$1``), its output to `log` in its directory
#: and then appended to ``$2``, and its exit to `rc` there, as `landing._job_end` reads them
AFTER_JOB = ('d=$1 all=$2; shift 2; "$@" > "$d/log" 2>&1; rc=$?; cat "$d/log" >> "$all"; '
             'echo "$rc" > "$d/rc.tmp" && mv "$d/rc.tmp" "$d/rc"')
#: how long after its start an after-landing job no exit was written of is taken as lost (a restart of the pod can leave its
#: pid to another process) and started again
AFTER_LOST_S = 3600


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime(TIME)


def unstamp(s: object) -> float | None:
    if not (isinstance(s, str) and STAMP_RE.fullmatch(s)):
        return None
    return datetime.strptime(s, TIME).replace(tzinfo=UTC).timestamp()


def branch(sha: str) -> str:
    return f"refs/heads/queue/{sha[:12]}"


def kept_runs(path: Path | None) -> frozenset[str]:
    """The run ids ``path`` names (`KEEP_RUNS`): none while it is absent.  QueueError, with the fix, when it can't be read, so
    a cancel never stops a run it may name."""
    try:
        if path is None or not path.exists():
            return frozenset()
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise QueueError(f"--keep-runs {path} can't be read ({getattr(e, 'strerror', None) or e}): make it a text file of "
                         "run ids, one a line, or remove it") from None
    return frozenset(w for line in text.splitlines() if (w := line.partition("#")[0].strip()))


def records_commit(commit: str, parents: str, body: str) -> bool:
    """Whether ``commit`` (its ``parents`` as `%P` gives them, its message ``body``) is a lock commit in drive's form, as a
    candidate's records commit is (`vqueue_records.commit`): one parent, and `vqueue_records.lock_message`'s subject and
    `Regenerated:` files."""
    return len(parents.split()) == 1 and body.startswith(LOCK_SUBJECT) and bool(REGENERATED_RE.search(body))


def _peak_gb() -> float:
    """The process's peak resident size so far, in GB (`ru_maxrss`, KiB on Linux)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


class Steps:
    """A tick's steps as its timing line gives them (`Loop.tick`): each step's seconds on ``clock``, summed over its calls, and
    the process's peak resident size once it ended (``peak``, GB), which only grows, so the step it grew in is the one that
    held it.  A launch is timed inside the action that starts it."""

    def __init__(self, clock: Callable[[], float], peak: Callable[[], float]):
        self.clock, self.peak = clock, peak
        self.at = self.last = clock()
        self.s: dict[str, float] = {}
        self.gb: dict[str, float] = {}

    def _add(self, name: str, s: float) -> None:
        self.s[name] = self.s.get(name, 0.0) + s
        self.gb[name] = self.peak()

    def lap(self, name: str) -> None:
        """The time since the last lap (the start, at first) as step ``name``'s."""
        now = self.clock()
        self._add(name, now - self.last)
        self.last = now

    @contextlib.contextmanager
    def __call__(self, name: str):
        """The time inside as step ``name``'s, inside a lap."""
        t0 = self.clock()
        try:
            yield
        finally:
            self._add(name, self.clock() - t0)

    def line(self, n: int) -> str:
        steps = " ".join(f"{k} {s:.1f}/{self.gb[k]:.1f}" for k, s in self.s.items())
        return f"research queue vqueue: tick {n}: {self.clock() - self.at:.1f} s, peak {self.peak():.1f} GB; s/GB: {steps}"


def _failed_doc(f: Failed) -> dict:
    """An inherited failure as an outcome logs it, whole: a red base's batch reads its inputs, and its failing tests beside its
    base's, which make it inherited too (`vqueue.decide`)."""
    return {"task": f.task, "key": f.key, "inputs": sorted(f.inputs), "base_key": f.base_key,
            **({"tests": sorted(f.tests)} if f.tests else {}),
            **({"base_tests": sorted(f.base_tests)} if f.base_tests is not None else {})}


def _as_failed(d: dict) -> Failed:
    return Failed(str(d.get("task")), str(d.get("key")), frozenset(d.get("inputs") or ()), d.get("base_key"),
                  frozenset(d.get("tests") or ()), frozenset(b) if isinstance(b := d.get("base_tests"), list) else None)


@dataclass
class Past:
    """The decision log's window (`WINDOW_S`), oldest first, the last tick it holds, and why each line it skipped isn't one (a
    torn last line, as a crash mid-write leaves)."""
    tick: int = 0
    lines: list[dict] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)

    @classmethod
    def read(cls, path: Path, since: float) -> Past:
        if not path.is_file():
            return cls()
        out, tick, bad = [], 0, []
        for line in reversed(path.read_text().splitlines()):
            try:
                rec = json.loads(line)
            except ValueError as e:
                bad.append(f"no JSON ({e}): {line[:80]!r}")
                continue
            if not isinstance(rec, dict):
                bad.append(f"no object: {line[:80]!r}")
                continue
            tick = tick or int(rec.get("tick") or 0)
            if (at := unstamp(rec.get("at"))) is None or at < since:
                break
            out.append(rec)
        return cls(tick, out[::-1], bad)

    def of(self, action: str, sha: str | None = None) -> list[dict]:
        return [r for r in self.lines if r.get("action") == action and (sha is None or r.get("sha") == sha)]

    def history(self, on: Callable[[str], bool] | None = None) -> tuple[Outcome, ...]:
        """The drops' outcomes; with ``on``, a pass only of a candidate ``on`` holds (a shadow's `main` left behind)."""
        return tuple(Outcome(tuple(o.get("members") or ()), tuple(o.get("heads") or ()), o["state"], tuple(o.get("suspects") or ()),
                             o.get("base"), tuple(_as_failed(f) for f in o.get("inherited") or () if isinstance(f, dict)))
                     for r in self.of("drop") if isinstance(o := r.get("outcome"), dict) and o.get("state")
                     and (on is None or o["state"] != PASS or on(r.get("sha") or "")))

    def tries(self, sha: str) -> int:
        return sum(1 for r in self._enqueues(sha))

    def enqueued(self, sha: str) -> float | None:
        """When `ready:<sha>` was first enqueued."""
        return next((unstamp(r.get("at")) for r in self._enqueues(sha)), None)

    def _enqueues(self, sha: str) -> Iterable[dict]:
        return (r for r in self.lines if r.get("sha") == sha and r.get("action") in ("build", "retry") and r.get("enqueued"))

    def reruns(self, sha: str) -> int:
        return sum(1 for r in self.of("rerun", sha) if r.get("done") or r.get("absent"))

    def builds(self) -> dict[str, dict]:
        """The loop's own candidates, each pushed build's line by its sha (a build whose enqueue failed is one too)."""
        return {r["sha"]: r for r in self.of("build") if r.get("sha")}

    def slim(self, sha: str) -> tuple[bool, dict | None]:
        """Whether the run beside ``sha``'s check was started, and the newest record of it (`vqueue_slim.start`)."""
        recs = self.of("slim", sha)
        return bool(recs), recs[-1].get("record") if recs else None

    def slim_started(self, sha: str, gate: str | None = None) -> tuple[int, float | None]:
        """How many times a run beside ``sha``'s check was started (with ``gate``, but those whose record names another
        ``gate``), and when the newest was."""
        starts = [r for r in self.of("slim", sha) if str(r.get("why") or "").startswith("started")
                  and (gate is None or (r.get("record") or {}).get("gate", gate) == gate)]
        return len(starts), unstamp(starts[-1].get("at")) if starts else None

    def regenerated(self, sha: str) -> tuple[int, dict | None]:
        """How many times ``sha``'s records run was started, and the newest record of it (`vqueue_records.start`)."""
        recs = [r for r in self._since_records_first("regenerate", sha) if not r.get("dropped")]
        return sum(str(r.get("why") or "").startswith("started") for r in recs), recs[-1].get("record") if recs else None

    def held(self, sha: str) -> float | None:
        """When ``sha``'s records commit was first held (a `commit` line not done: `Loop._commit`)."""
        return next((unstamp(r.get("at")) for r in self._since_records_first("commit", sha) if not r.get("done")), None)

    def _since_records_first(self, action: str, sha: str) -> list[dict]:
        """``sha``'s lines of ``action`` after its newest build line with its records run first: a candidate dropped and built
        again on the same tree in the same second (a held records commit's, `vqueue.decide`) has the same sha, and its records
        run and commit start afresh."""
        start = max((i for i, r in enumerate(self.lines) if r.get("action") == "build" and r.get("sha") == sha
                     and r.get("records_first")), default=-1)
        return [r for r in self.lines[start + 1:] if r.get("action") == action and r.get("sha") == sha]

    def launching(self, key: str = "beside_launching", action: str = "slim") -> dict[str, dict]:
        """The drops whose candidate's run beside its check (or, with ``key`` `records_launching` and ``action`` `regenerate`,
        its records run) was still launching (``key``), by sha, but those a line of ``action`` has settled since (`dropped`)."""
        settled = {r.get("sha") for r in self.of(action) if r.get("dropped")}
        return {r["sha"]: r for r in self.of("drop") if isinstance(r.get(key), dict) and r["sha"] not in settled}

    def off_gate_running(self) -> dict[str, dict]:
        """The drops whose candidate kept a verdict while its runs or the run beside its base's check still ran
        (``off_gate_running``, `Loop._drop`), by sha, but those an `off_gate` line has read since (`Loop._off_gate`)."""
        out: dict[str, dict] = {}
        for r in self.lines:
            if r.get("action") == "drop" and isinstance(r.get("off_gate_running"), dict) and r.get("sha"):
                out[r["sha"]] = r
            elif r.get("action") == "off_gate":
                out.pop(r.get("sha"), None)
        return out

    def ignored(self) -> set[tuple[str, str]]:
        return {(r.get("ref"), r.get("sha")) for r in self.of("ignore")}

    def carries(self) -> dict[tuple[int, str], str]:
        """The newest verdict carry decision logged of each PR's head (`World._carry`), by (number, head)."""
        return {(r.get("number"), (r.get("heads") or [""])[0]): r.get("why") for r in self.of("carry")}

    def unauthenticated(self) -> dict[tuple, str]:
        """Why each mark or verdict logged as counted unauthenticated wouldn't count (`Loop._unauthenticated`), by
        `StoreLabels.unauthenticated`'s key."""
        return {(r.get("number"), (r.get("heads") or [""])[0], r.get("label"), r.get("by"), r.get("written")): r.get("why")
                for r in self.of("unauthenticated")}

    def rung(self) -> set[tuple[str, str]]:
        return {(r["sha"], r.get("why")) for r in self.of("ring") if r.get("done") or r.get("shadow")} | self._posting(
            "ring", lambda r: {(r.get("sha"), r.get("why"))})

    def told(self, kind: str = "skip") -> set[str]:
        """Each `head@base` a message of ``kind`` told its owner of: a skip's (`Loop._merged`), a move a head lacks
        (`lacks`, `Loop._lacking`), or a held head a head waits on (`behind_held`, `Loop._behind_held`)."""
        def heads(r: dict) -> set[str]:
            return {f"{h}@{r.get('base')}" for h in r.get("heads") or ()}

        return {k for r in self.of(kind) if r.get("done") or r.get("shadow") for k in heads(r)} | self._posting(kind, heads)

    def conflicted(self) -> set[str]:
        """The heads whose owners were told they don't merge onto `main` (`Loop._conflicts`)."""
        def heads(r: dict) -> set[str]:
            return set(r.get("heads") or ())

        return {h for r in self.of("conflict") if r.get("done") or r.get("shadow") for h in heads(r)} | self._posting(
            "conflict", heads)

    def riding(self) -> set[tuple[str, str]]:
        """Each (head, candidate) logged as riding (`Loop._riding`)."""
        return {(h, r.get("sha")) for r in self.of("riding") for h in r.get("heads") or ()}

    def reds(self, base: str) -> set[str]:
        """The tasks ``base`` was told red on."""
        def tasks(r: dict) -> set[str]:
            return {f.get("task") for f in r.get("failed") or () if isinstance(f, dict)} if r.get("base") == base else set()

        return {x for r in self.of("red") if r.get("done") or r.get("shadow") for x in tasks(r)} | self._posting("red", tasks)

    def _posting(self, kind: str, keys: Callable[[dict], Iterable]) -> set:
        """The keys of each message of ``kind`` whose post was logged as about to start (`posting`) and not as failed since
        (`posted` false): told, since its tick may have stopped between the post and its record, and a message is told at
        most once."""
        out: set = set()
        for r in self.lines:
            if r.get("action") == "posting" and r.get("posting") == kind:
                out.update(keys(r))
            elif r.get("action") == kind and r.get("posted") is False:
                out.difference_update(keys(r))
        return out

    def built(self, sha: str) -> dict | None:
        return next(reversed(self.of("build", sha)), None)

    def moved(self) -> list[tuple[dict, list[str], dict | None]]:
        """Each advance that pushed `main` and logged drive's steps after it as their own (`on_main`) with no record of them
        done, the stacked PRs' owners their tries told, and the job running them detached (`Loop._moved`), while its start is
        their newest record."""
        out = []
        for r in self.of("advance"):
            if not (r.get("on_main") and r.get("done") and r.get("sha")):
                continue
            ons = self.of("on_main", r["sha"])
            if not any(o.get("done") for o in ons):
                job = ons[-1].get("job") if ons and isinstance(ons[-1].get("job"), dict) else None
                out.append((r, list((ons[-1].get("told") if ons else None) or ()), job))
        return out


@dataclass
class Judged:
    """A candidate's verdict with what it rests on: ``slim``, the verdict of its check, the run beside it (`vqueue_slim.judge`
    on its ``record``; pending until it starts), which is ``verdict`` but while a fail waits on the run beside its base's
    check (`Loop._waits`), and ``note``, why it reads so."""
    verdict: Verdict
    note: str = ""
    slim: Verdict | None = None
    record: dict | None = None


@dataclass(frozen=True)
class MainRed:
    """`main`'s own check red where the land verdicts gate (`Loop._main_red`): the newest commit of `main`'s first-parent line
    whose check ended, ``tip``, failed with ``verdict`` (a fail, or infra once `main`'s own was started again); the newest below
    it whose check passed, ``passed`` (None past `MAIN_WALK`); the PRs landed between, each with what its merge changed
    (``landed``, `World.landed`); the one of them it names (``culprit``: `vqueue.culprit`) and why, or why none; what the held
    ready PRs' changes meet (``holds``: what each failure read, with a culprit); the revert of the culprit's merge already on
    `main` above ``tip`` (``reverted``: a tick that stopped between its push and its line); and the line that acted on it
    (``done``, a `revert` or a `fallback` line), None until a tick acts."""
    tip: str
    verdict: Verdict
    passed: str | None
    landed: tuple[tuple[Member, frozenset[str]], ...] = ()
    culprit: Member | None = None
    why: str = ""
    holds: frozenset[str] = frozenset()
    done: dict | None = None
    reverted: str = ""

    def steps(self) -> str:
        """The failed steps, each with the first of the tests it names."""
        def named(f: Failed) -> list[str]:
            return sorted(x for x in f.tests if not x.split("::")[-1].startswith("unnamed:"))

        return "; ".join(f.task + (f" ({', '.join(t[:3])}{', ...' if len(t) > 3 else ''})" if (t := named(f)) else "")
                         for f in self.verdict.failed) or "; ".join(f.task for f in self.verdict.infra) or "nothing named"


def _off_mark(files: Iterable[str], flags: DriveFlags) -> str:
    """Why drive.toml (``flags``, `DriveFlags`) keeps a move PR changing ``files`` off its ready mark alone, however certified
    its move: "" when it doesn't.  With `lowrisk_lane` off no PR lands without a land verdict, and with `signed_verdicts` "on" a
    change to `SIGNED_GUARDS`, a move's renames too, could change who holds a handle or how a label is judged."""
    if not flags.lowrisk_lane:
        return "drive.toml's `lowrisk_lane` is off, so a move lands only with one"
    if flags.signed_verdicts == "on" and (guarded := sorted(f for f in files if vqueue.meets((f,), SIGNED_GUARDS))):
        return (f"it changes {', '.join(guarded)}, and under drive.toml's `signed_verdicts` a move changing `SIGNED_GUARDS` "
                "lands only with one")
    return ""


@dataclass(frozen=True)
class Skip:
    """A PR a build left out (`World.build`) and why: ``held`` names what of the build holds its head already (the member
    whose head does, else the candidate it is built on), and ``by`` the members above `main` it conflicts with where it merges
    onto `main` alone.  A move PR's (`World._move`) ``waits`` for the members below it to land, its owner told nothing, or
    has ``fix``, what its owner is told to do."""
    pr: PR
    why: str
    held: str = ""
    by: tuple[int, ...] = ()
    waits: bool = False
    fix: str = ""


class _Clone(Queue):
    """The queue in a clone it doesn't bring to the remote's state (drive's, or the loop's own): the loop fetches into `NS`, and
    `main` as fetched is its `MAIN`, not drive's `refs/remotes/origin/main`."""

    def _clone(self) -> None:
        return None

    @property
    def base(self) -> str:
        return MAIN


class World:
    """What the loop reads and changes: git in the clone and on origin, the labels, the PRs and the merge thread.  Each
    effect is one method, so a test fakes what it doesn't run for real.  With ``labels_at`` (a new `StoreLabels`), each tick
    reads the labels afresh (`_relabel`).  ``cache`` is where it keeps the heads' own changes, what `ready` judged of them on
    `main` (`JUDGED_CACHE`) and the issuer's keys, not in drive's files.  ``on`` names the machines the run beside each check takes, in place of drive.toml's `on`."""

    def __init__(self, q: Queue, labels, *, priority_file: Path | None = None,
                 post: Callable[[str, tuple[str, ...]], bool] | None = None, now: Callable[[], float] = time.time,
                 labels_at: Callable[[], StoreLabels] | None = None, cache: Path | None = None, on: str = "",
                 frozen_file: Path | None = None, deselect_file: Path | None = None, wide_file: Path | None = None,
                 pins_file: Path | None = None, keep_file: Path | None = None, heavy_file: Path | None = None,
                 heavy_on: str = ""):
        self.q, self.labels, self.on = q, labels, on
        self.priority_file, self.now, self.repo, self.labels_at = priority_file, now, q.workdir, labels_at
        self.frozen_file, self._frozen_told, self.wide_file = frozen_file, set(), wide_file
        self.pins_file, self.keep_file = pins_file, keep_file
        #: the heavy PRs' file (`HEAVY`) and the machines their candidates' heavy runs take (`heavy`); "" turns it off
        self.heavy_file, self.heavy_on = heavy_file, heavy_on
        #: the tests each candidate's check leaves out (`DESELECT`, `vqueue_slim.deselection`)
        self.deselect_file = deselect_file
        self.keys = cache / "issuer-keys.json" if cache is not None else None
        #: the moves regenerated on candidates of the `main` they landed on, as `Queue._made` holds them (`_keep_moves`)
        self._made_at = cache / "moves-made.json" if cache is not None else None
        #: `JUDGED_CACHE` (`_take_judged`, `_keep_judged`): the text last read or written, and the code and git that judge
        self._judged_at, self._judged_text, self._judging = (cache / JUDGED_CACHE if cache is not None else None), None, ""
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            q._kept_at, q._mapped_at = cache / "own-changes.json", cache / "mapped-changes.json"
            q._kept = json.loads(q._kept_at.read_text()) if q._kept_at.is_file() else {}
            q._made = json.loads(self._made_at.read_text()) if self._made_at.is_file() else {}
        self.post: Callable[[str, tuple[str, ...]], bool] = post or merge_thread_post
        #: by (head, `main`): its files (`_changed`), and (lean, conflict) (`_judge`)
        self._files: dict[tuple[str, str], frozenset[str]] = {}
        self._judged: dict[tuple[str, str], tuple[bool, bool]] = {}
        #: by (head, `main`), the moves it lacks that its change names (`_lacks`); by (where, map), what a move's map retires;
        #: by (head, `main`, the move PRs' heads and maps), the moves in flight it crosses (`crossing`)
        self._lacked: dict[tuple[str, str], list[tuple[str, str, list[str] | None]]] = {}
        self._retires: dict[tuple[str, str], Retired | None] = {}
        self._crossings: dict[tuple[str, str, frozenset[tuple[str, str]]], frozenset[str]] = {}
        #: each `carry.delta` judged, by its arguments (`_delta`)
        self._deltas: dict[tuple[str, str, tuple[str, ...], frozenset[str]], carry.Delta] = {}
        #: what the last `ready` decided of the verdict carry (`_carry`): {number: its head, the older head, why, whether it
        #: carries}
        self.carried: dict[int, dict] = {}
        #: under drive.toml's `signed_verdicts = "shadow"`, the labels' `StoreLabels.unauthenticated`, filled as they are read
        self.unauthenticated: dict[tuple, str] = {}
        self._prs: dict[int, PR] | None = None
        #: each PR `handle` read from GitHub as it isn't open (a landed member's), else None when it can't be read
        self._closed: dict[int, PR | None] = {}
        self._driver: Landing | None = None
        self._slim: Landing | None = None
        self._main: str | None = None
        #: what the last `ready` read: {number: the PR at its ready head}, {number: (head, why it isn't ready)} of each other
        #: open PR into `main`, and those of them taken from the queue by others than their owners (`Snapshot.pulled`)
        self.prs: dict[int, PR] = {}
        self.left: dict[int, tuple[str, str]] = {}
        self.pulled: set[int] = set()
        #: of those, the ones the queue admits at their ready marks: {number: (head, cause, why)}, where the cause stays the same
        #: while the reason does (`vqueue.Leave`)
        self.held_out: dict[int, tuple[str, str, str]] = {}
        #: of those left out for a move they lack whose names their change names: {number: (head, the last such move, why)}
        self.lacking: dict[int, tuple[str, str, str]] = {}
        #: of those, the marked ones holding a held PR's head, directly or through others (`queue.held_below`): {number: (head,
        #: the held head, why)}
        self.behind_held: dict[int, tuple[str, str, str]] = {}
        #: the move PRs the last `ready` took on their ready marks alone (`_uncertified`), and {number: what the last `build`
        #: regenerated of each move PR: its map, tip, commit and `move_check.py --json` report} (`_move`)
        self.on_mark: set[int] = set()
        self.certified: dict[int, dict] = {}
        #: the PRs the last `ready` admitted with a land verdict of their own at their heads, not carried nor on a mark alone
        self.reviewed: set[int] = set()

    def log(self, text: str) -> None:
        self.q.log(text)

    # -- reads

    def fetch(self) -> str:
        """origin's `main`, fetched with origin's `queue/*` branches into `NS`."""
        self._prs, self._closed, self.q._registry = None, {}, None
        self.q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--prune", "origin",
                       f"+refs/heads/main:{MAIN}", f"+refs/heads/queue/*:{NS}/queue/*")
        if self.q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "origin",
                          f"+refs/heads/checked:{CHECKED}", ok=None).returncode:
            _git(self.repo, "update-ref", "-d", CHECKED, check=False)
        return self.at_main()

    def at_main(self) -> str:
        """`main` as the clone holds it (`MAIN`), with the queue's rules and the labels read at it, as a tick reads them."""
        main = _out(self.repo, "rev-parse", MAIN)
        if rules := rules_at(self.repo, main):
            self.q._rules = rules
        if main != self._main:                              # the merges' records onto the `main` before, as a sync drops them
            q = self.q
            q._merge_records, q._merge_keys, q._context, q._merged_from, self._main = {}, {}, None, {}, main
            self._driver = None                             # it keeps the registry it read
        if self.labels_at is not None:
            self._relabel(main)
        return main

    def _relabel(self, main: str) -> None:
        """New labels, read as drive's tick reads them (one `StoreLabels` keeps what it pulled, so a mark, a verdict or a
        withdrawal written since would go unread): marks from drive.toml's `marked_after` at ``main`` on (else the queue
        rules'), authenticated under its `signed_verdicts`, and a head's carried to a new head with the same own change, or,
        while the queue maps heads across moves (`Queue.mapped_jobs`), whose own change is the head's mapped across them.
        With `signed_verdicts = "shadow"` each mark and verdict counts, and the labels keep why one wouldn't
        (`StoreLabels.unauthenticated`, which the loop logs); when the registry or the issuer's keys can't be read, none is
        judged and the tick goes on."""
        q, flags = self.q, flags_at(self.repo, main)
        self.labels = labels = self.labels_at()
        labels.since, labels.change = flags.marked_after or q.rules().marked_after, q.own_change
        labels.moved, labels.owners = q.mapped_change if q.mapped_jobs else None, self._owners
        self.unauthenticated = {}
        if flags.signed_verdicts == "on":
            labels.verify = label_verifier(self.repo, MAIN, self.keys, q.log)
        elif flags.signed_verdicts == "shadow":
            try:
                labels.verify, labels.shadow = label_verifier(self.repo, MAIN, self.keys, q.log), True
                self.unauthenticated = labels.unauthenticated
            except Exception as e:  # noqa: BLE001 -- the shadow changes nothing the queue lands
                self.log(f"research queue vqueue: drive.toml's `signed_verdicts` is \"shadow\" and no mark or verdict is judged "
                         f"now: {type(e).__name__}: {e}")
        q.ready, q.before_cutoff, q.marked_by, q.marked_at, q.prefetch = (labels.ready, labels.before_cutoff, labels.marked_by,
                                                                          labels.marked_at, labels.prefetch)

    def open(self) -> dict[int, PR]:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs

    def _owners(self, n: int) -> set[str | None]:
        """Whose land verdict on PR ``n`` counts at none of its heads (`StoreLabels.owners`): its owner and that owner's handle,
        as `review_owners` has them, with each head's mark's author."""
        return {p.owner, self.q.handle(p)} if (p := self.open().get(n)) else set()

    def _missing(self, shas: Iterable[str]) -> set[str]:
        if not (shas := list(dict.fromkeys(shas))):
            return set()
        r = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"], cwd=self.repo, check=True,
                           input="".join(f"{s}\n" for s in shas), capture_output=True, text=True)
        return {line.split()[0] for line in r.stdout.splitlines() if line.endswith(" missing")}

    def _ancestor(self, a: str, b: str) -> bool:
        return not _git(self.repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    def _below(self, shas: Iterable[str], b: str) -> set[str]:
        """Those of ``shas``, full commit ids in the clone, that are ``b`` or below it (`_ancestor` of each), by one walk: what
        they hold that ``b`` doesn't, which lists each of them that isn't."""
        if not (shas := list(dict.fromkeys(shas))):
            return set()
        r = subprocess.run(["git", "rev-list", "--stdin", f"^{b}"], cwd=self.repo, check=True,
                           input="".join(f"{s}\n" for s in shas), capture_output=True, text=True)
        return set(shas) - set(r.stdout.split())

    def ready(self, main: str) -> list[Ready]:
        """The open PRs drive would land from `next` now, as `vqueue.Ready`: a move PR with a land verdict, or, while drive.toml's
        `lowrisk_lane` is on, on its mark alone once its move is certified on `main` (`_uncertified`, `_off_mark`), never under
        the low-risk rule (`queue.lands_on_mark`).  Why each other open PR into `main` isn't one goes to `left`:
        the queue's word on one it doesn't admit (`Entry.why`; a withdrawn mark's author), else a hold and by whom, no land
        verdict, the freeze, a move it lacks whose retired names its own change names (`lacked_moves`, also to `lacking`), or
        an untaken PR whose head it holds; those others than their owners took go to `pulled`."""
        q, labels = self.q, self.labels
        prs = [p for p in self.open().values() if p.base == "main"]
        absent = self._missing(p.head for p in prs)
        if missing := [p for p in prs if p.head in absent]:
            q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "origin",
                      *(f"+refs/pull/{p.number}/head:{NS}/pr/{p.number}" for p in missing))
        gone = self._missing(p.head for p in missing)
        below = self._below((p.head for p in prs if p.head not in gone), main)
        left = {p.number: (p.head, "its head can't be fetched" if p.head in gone else "its head is on `main`")
                for p in prs if p.head in gone or p.head in below}
        if any(p.move and p.number not in left for p in prs):
            landed_heads = queue_log(self.repo, main)[0]
            left.update((p.number, (p.head, "its move, regenerated from its head, is on `main`")) for p in prs
                        if p.move and p.number not in left and p.head in landed_heads)
        prs = [p for p in prs if p.number not in left]
        self._take_judged(main)
        labels.prefetch([(p.number, p.head) for p in prs])
        old = _git(self.repo, "rev-parse", "-q", "--verify", f"{CHECKED}^{{commit}}", check=False).stdout.strip()
        entries = q.entries(prs, state := q.load_state(), base=main, landed=main, queued=main, open_numbers=set(self.open()),
                            old_landed=old or None)
        left.update((n, (e.pr.head, self._unadmitted(e.pr, e.why))) for n, e in entries.items() if e.state != "ready")
        landed = (main, old) if old else (main,)
        pulled = {n for n, e in entries.items() if e.state != "ready" and self._pulled(e, state["ejected"].get(str(n)))}
        flags = flags_at(self.repo, main)
        self._files = {k: v for k, v in self._files.items() if k[1] == main}
        self._lacked = {k: v for k, v in self._lacked.items() if k[1] == main}
        self._crossings = {k: v for k, v in self._crossings.items() if k[1] == main}
        self._judged = {k: v for k, v in self._judged.items() if k[1] == main}
        self._deltas = {k: v for k, v in self._deltas.items() if k[2] == landed}
        admitted = [n for n, e in entries.items() if e.state == "ready"]
        reviews = {n: labels.reviews(n, entries[n].pr.head) for n in admitted}
        why = {n: unreviewed(reviews[n],
                             review_owners(q, labels, n, entries[n].pr.head, entries[n].pr.owner, self.open()))
               for n in admitted}
        self.reviewed = {n for n in admitted if why[n] is None}
        try:
            carried = self._carry(list(landed), entries, why, flags.verdict_carry)
        except Exception as e:  # noqa: BLE001 -- the carry never stops the reading, which takes the verdicts as they are
            self.log(f"research queue vqueue: the verdict carry didn't run: {type(e).__name__}: {e}")
            carried, self.carried = set(), {}
        with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:
            files = dict(zip(admitted, pool.map(lambda n: self._changed(main, entries[n].pr.head), admitted)))
            took, self.on_mark = [], set()
            uncertified = {n: self._uncertified(entries[n].pr, main) for n in admitted if entries[n].pr.move}
            off_mark = {n: _off_mark(files[n], flags) or uncertified[n] for n in uncertified}
            cause: dict[int, str] = {}
            for n in admitted:
                p, held = entries[n].pr, (why[n] or "").startswith("held")
                if why[n] is None or n in carried or lands_on_mark(flags, p, why[n], files[n]):
                    took.append(n)
                elif held:
                    left[n], cause[n] = (p.head, why[n]), "held"
                    if (unreviewed([r for r in reviews[n] if self._another(n, r.by)], ()) or "").startswith("held"):
                        pulled.add(n)
                elif p.move and not off_mark[n]:
                    took.append(n)
                    self.on_mark.add(n)
                else:
                    left[n] = (p.head, f"{why[n]}, and " + (off_mark[n] if p.move else "it isn't low-risk"))
                    cause[n] = "no land verdict"
            specs, clears = self._frozen()
            if specs:
                below = self._frozen_base(main)
                frozen = [n for n in took if not self._cleared(n, entries[n].pr.head, clears)
                          and self._frozen_out(below, n, entries[n].pr.head, specs)]
                left.update((n, (entries[n].pr.head, f"it waits for the freeze to end ({self.frozen_file})")) for n in frozen)
                cause.update((n, "freeze") for n in frozen)
                took, pulled = [n for n in took if n not in frozen], pulled | set(frozen)
            # a plain merge of a head below a move leaves its own lines naming what the move retired: such a head lands
            # crossed, and one whose own change names none of it crosses by that merge (`lacked_moves`)
            lacks = dict(zip(took, pool.map(lambda n: self._lacks(main, entries[n].pr.head, files[n]), took)))
            left.update((n, (entries[n].pr.head, lacks_why(m))) for n, m in lacks.items() if m)
            cause.update((n, "lacks a move") for n, m in lacks.items() if m)
            self.lacking = {n: (entries[n].pr.head, m[-1][0], lacks_why(m)) for n, m in lacks.items() if m}
            took = [n for n in took if not lacks[n]]
            new = any((entries[n].pr.head, main) not in self._judged for n in took)
            with base_drivers(self.repo, main) if new else contextlib.nullcontext(None) as cfg:   # once a tick, not a PR
                chosen = {n: (files[n], *j)
                          for n, j in zip(took, pool.map(lambda n: self._judge(main, entries[n].pr.head, cfg), took))}
        self._keep_judged(main)
        # a land verdict covers its PR's own change: a PR holding the head of an admitted PR that isn't taken (no land verdict
        # there, or held) would carry that change unreviewed, unless the head landed on `checked` under drive's rules
        while out := {n: a for n in chosen for a in entries[n].after if a in entries and entries[a].state == "ready"
                      and a not in chosen and not (old and self._ancestor(entries[n].held[a], old))}:
            left.update((n, (entries[n].pr.head, f"it holds {entries[n].holds(a)}, which isn't taken"
                                                 + (f": {left[a][1]}" if a in left else ""))) for n, a in out.items())
            cause.update((n, "stacked") for n in out)
            chosen = {n: v for n, v in chosen.items() if n not in out}
        self.behind_held = {}
        for n in sorted(left):
            if n in entries and (cause.get(n) == "stacked" or entries[n].why.startswith("contains")) and (
                    below := held_below(entries, n, lambda a: why[a] if cause.get(a) == "held" else None)):
                self.behind_held[n] = (entries[n].pr.head, entries[below[0]].pr.head,
                                       held_waits(below[0], entries[below[0]].pr.head, below[1]))
        while more := {n for n, e in entries.items() if n not in chosen and n not in pulled and pulled & set(e.after)}:
            pulled |= more
        first = {n: i for i, n in enumerate(priority_list(self.priority_file) if self.priority_file else ())}
        # a stack lands its bases first, ranked or not: a PR the priority file names ahead of PRs whose heads its own holds
        # goes after them, so no ranking takes a PR off the ready list and drops the candidate holding it
        order = sorted(chosen, key=lambda n: (first.get(n, len(first)), labels.marked_at(n, entries[n].pr.head) or "", n))
        placed: list[int] = []

        def place(n: int, path: tuple[int, ...] = ()) -> None:
            if n in placed or n in path:
                return
            for a in entries[n].after:
                if a in chosen:
                    place(a, (*path, n))
            placed.append(n)

        for n in order:
            place(n)
        self.prs, self.left, self.pulled = {n: entries[n].pr for n in chosen}, left, pulled
        self.held_out = {n: (left[n][0], c, left[n][1]) for n, c in cause.items()}
        inputs = self._lean_inputs(main, (f for v in chosen.values() for f in v[0])) if chosen else frozenset()
        # a move regenerated on its batch's tip doesn't merge its head, so whether that merges is moot
        return [Ready(n, (p := entries[n].pr).head, *chosen[n][:2], rank=i, conflict=chosen[n][2] and uncertified.get(n) != "",
                      move=bool(p.move), scripts=vqueue.meets(chosen[n][0], MOVE_SCRIPTS),
                      lean_input=vqueue.meets(chosen[n][0], inputs)) for i, n in enumerate(placed)]

    def _uncertified(self, p: PR, main: str) -> str:
        """Why move PR ``p``, with no land verdict, doesn't land on its ready mark alone: its map isn't on ``main``, where it
        was reviewed as it landed (`tools/move/stage.py` refuses one that isn't on its base), or its move regenerated on a
        candidate on ``main`` wasn't certified (`Queue._regenerate`, kept by the move's key); "" when it lands on its mark,
        once its regeneration on its batch's tip is certified (`build`)."""
        if _git(self.repo, "cat-file", "-e", f"{main}:{p.move}", check=False).returncode:
            return f"a move lands on its mark alone only once its map {p.move} is on `main`"
        tail = Queue._move_key("", p, main)
        if why := next((v["why"] for k, v in self.q._made.items() if k.endswith(tail) and v.get("why")), None):
            return f"its move isn't certified on a candidate on `main` {main[:12]} ({why}), and an uncertified move lands only with one"
        return ""

    def _lean_inputs(self, main: str, files: Iterable[str]) -> frozenset[str]:
        """The Lean inputs (`vqueue_slim.lean_inputs`) of ``main``'s tree with ``files``, the ready PRs' changed files, in it, so
        a Lean package a ready PR adds is one: a PR whose own change meets them changes the Lean key (`vqueue_slim.lean_key`)
        of a tree it is merged into (`Ready.lean_input`)."""
        from .vqueue_slim import lean_inputs

        return frozenset(lean_inputs({*self._tree(main), *files}))

    def _carry(self, landed: list[str], entries: dict[int, Entry], why: dict[int, str | None], mode: str) -> set[int]:
        """The verdict carry (`carry`), as drive.toml's `verdict_carry` at `main` says: an admitted PR with no land verdict at
        its head (``why``, `queue.unreviewed`'s) takes the newest at another of its heads with verdicts of their own when
        `Landing._carried_from` says so, trusting ``landed`` (`main`, and the old path's `checked` while origin has it) and the
        heads of the admitted PRs judged before it that land on their verdicts or carry one, a PR judged after those it contains
        (`Entry.after`).  Each decision is kept in `carried`, which the loop logs once a head.  With "on" the PRs carried
        (returned) count as reviewed; with "shadow" none does.  No verdict is written."""
        self.carried = {}
        if mode not in ("shadow", "on"):
            return set()
        heads: dict[int, list[str]] = {}
        for n, w in why.items():
            if w is None:
                continue
            try:
                if hs := sorted(self.labels._other_heads(n, entries[n].pr.head, vocab.REVIEWED)):
                    heads[n] = hs
            except QueueError as e:
                self.log(f"research queue vqueue: #{n}'s other heads can't be read, so no verdict carries to it now: {e}")
        if not heads:
            return set()
        self.labels.prefetch([(n, h) for n, hs in heads.items() for h in hs])
        d, trusted, out, order = self.driver(), set(), set(), []

        def visit(n: int, path: tuple[int, ...] = ()) -> None:
            if n in order or n in path or n not in why:
                return
            for a in entries[n].after:
                visit(a, (*path, n))
            order.append(n)

        for n in sorted(why):
            visit(n)
        for n in order:
            p = entries[n].pr
            if why[n] is None:
                trusted.add(p.head)
                continue
            if n not in heads:
                continue
            try:
                got = d._carried_from(Member(n, p.head, p.owner or "", ""), heads[n], why[n], landed, trusted)
            except Exception as e:  # noqa: BLE001 -- one PR's failure leaves the others judged
                got = "?", f"it can't be judged: {type(e).__name__}: {e}", False
            if got is None:
                continue
            old, because, ok = got
            if ok:
                trusted.add(p.head)
                if mode == "on":
                    out.add(n)
            verb = ("carried" if ok else "did not carry") if mode == "on" else ("would carry" if ok else "would not carry")
            self.carried[n] = {"head": p.head, "old": old, "carries": ok,
                               "why": f"{verb} #{n}'s land from {old[:12]} to {p.head[:12]} ({because})"}
        return out

    def _delta(self, old: str, new: str, landed: list[str], trusted: set[str]) -> carry.Delta:
        """`carry.delta` of ``new`` over ``old`` (`Landing.delta`), kept by its arguments: judging a merge takes a `git
        merge-tree`, and each tick judges every carry again."""
        if (key := (old, new, tuple(landed), frozenset(trusted))) not in self._deltas:
            self._deltas[key] = carry.delta(self.repo, old, new, carry.trust(self.repo, landed, trusted))
        return self._deltas[key]

    def wide(self) -> frozenset[int]:
        """The PRs the wide file names (`WIDE`), parsed and refused as the priority file is (`priority_list`): none while it is
        missing or empty."""
        return frozenset(priority_list(self.wide_file, "--wide-file")) if self.wide_file else frozenset()

    def pinned(self) -> frozenset[int]:
        """The PRs the pins file names (`PINS`), parsed and refused as the wide file is, `#` starting a comment: none while it is
        missing or empty."""
        return frozenset(priority_list(self.pins_file, "--pins-file", comments=True)) if self.pins_file else frozenset()

    def heavy(self) -> tuple[frozenset[int], bool]:
        """(the PRs the heavy file names, `HEAVY`, parsed and refused as the wide file is; whether it names every candidate,
        `queue.EVERY`): none while it is missing or empty, or while no machines take their runs (``heavy_on`` "")."""
        if not (self.heavy_file and self.heavy_on):
            return frozenset(), False
        prs, every = listed(self.heavy_file, "--heavy-file", every=True)
        return frozenset(prs), every

    def _another(self, n: int, by: str | None) -> bool:
        """Whether ``by`` is someone other than PR ``n``'s owner and its handle (`_owners`), compared as `unreviewed` does."""
        return bool(by) and _who(by) not in {_who(o) for o in self._owners(n) if o}

    def _pulled(self, e: Entry, ejected: dict | None) -> bool:
        """Whether another than its owner took ``e``, which the queue doesn't admit, from the queue (`Snapshot.pulled`): a hold
        label, an eject (``ejected``: its record in the queue's state) by another or by `EJECTOR`, a red's, even of its own PR,
        or a mark withdrawn by another; not its owner's withdrawal, a push or a cause no one gave."""
        if e.why.startswith("held ("):
            return True
        if e.why.startswith("ejected at"):
            by = (ejected or {}).get("by") or EJECTOR
            return _who(by) == _who(EJECTOR) or self._another(e.pr.number, by)
        return e.why.startswith("not marked ready") and self.q.marked_by is not None and self._another(
            e.pr.number, self.q.marked_by(e.pr.number, e.pr.head))

    def _unadmitted(self, p: PR, why: str) -> str:
        """Why the queue doesn't admit ``p`` (``why``, its `Entry.why`), with a withdrawn mark's author: one at its head that
        `ready` read and that isn't `true`."""
        if not why.startswith("not marked ready"):
            return why
        by = self.q.marked_by(p.number, p.head) if self.q.marked_by is not None else None
        return f"its ready mark was withdrawn by {by}" if by else "it isn't marked ready"

    def _frozen(self) -> tuple[list[str], list[tuple[str, str]]]:
        """The freeze's pathspecs and its `cleared-by` (handle, label key) pairs (`FROZEN`): none while its file is absent."""
        if self.frozen_file is None or not self.frozen_file.is_file():
            return [], []
        lines = [s for line in self.frozen_file.read_text().splitlines() if (s := line.strip()) and not s.startswith("#")]
        clears = [(w[1], w[2]) for s in lines if s.startswith("cleared-by ") and len(w := s.split()) == 3]
        return [s for s in lines if not s.startswith("cleared-by ")], clears

    def _cleared(self, n: int, head: str, clears: list[tuple[str, str]]) -> bool:
        """Whether the freeze's owner cleared ``n`` at ``head``: a `KEY = true` label there by the handle of a `cleared-by`."""
        return any(str(lab.value).lower() == "true" and lab.by.lstrip("@").lower() == who.lstrip("@").lower()
                   for who, key in clears for lab in self.labels._counted(n, head, key)[0])

    def _frozen_base(self, main: str) -> str:
        """What the freeze reads a PR's own change above: `checked`'s tip while `main` doesn't hold it, else ``main``."""
        old = _git(self.repo, "rev-parse", "-q", "--verify", f"{CHECKED}^{{commit}}", check=False).stdout.strip()
        return old if old and not self._ancestor(old, main) else main

    def _frozen_out(self, below: str, n: int, head: str, specs: list[str]) -> bool:
        """Whether PR ``n``'s own change at ``head``, above ``below``, renames or deletes a file under ``specs``, said once a head
        in the log."""
        hits = _out(self.repo, "diff", "--name-status", "-M", "--diff-filter=RD", f"{below}...{head}", "--",
                    *specs).splitlines()
        if hits and head not in self._frozen_told:
            self._frozen_told.add(head)
            self.log(f"research queue vqueue: #{n} at {head[:12]} waits for the freeze to end ({self.frozen_file}): it "
                     f"renames or deletes {len(hits)} file(s) under its paths: {'; '.join(hits[:5])}")
        return bool(hits)

    def taken(self, flying: Iterable[tuple[int, str]], main: str) -> dict[int, tuple[str, str]]:
        """The PRs another than their owner took at the heads candidates in flight hold them at, ``flying`` (`Snapshot.flying`:
        heads the queue doesn't take now, and the ones that land), each with that head and why, read there as `_pulled` reads a
        PR at its head: a hold by another among its reviews there, an eject there by `EJECTOR` or by another, a mark there
        withdrawn by another, or the freeze, unless cleared there (a head the clone lacks has no files to read for it).
        `ready` reads these at each PR's current head alone.  A closed PR's owner is its mark's author there."""
        flying, (specs, clears), out = sorted(flying), self._frozen(), {}
        below, gone = (self._frozen_base(main), self._missing(h for _, h in flying)) if specs else (main, set())
        ejects = self.q.load_state().get("ejected", {})
        for n, head in flying:
            mine = {_who(o) for o in self._owners(n) or {self.labels.ready_by(n, head)} if o}
            held = unreviewed([r for r in self.labels.reviews(n, head) if r.by and _who(r.by) not in mine], ()) or ""
            e = ejects.get(str(n)) or {}
            ejected, eby = e.get("head") == head and not self.q._marked_again(PR(n, head), e), e.get("by") or EJECTOR
            withdrawn = not ejected and not self.q.ready(n, head) and not (self.q.before_cutoff and self.q.before_cutoff(n, head))
            by = self.q.marked_by(n, head) if withdrawn and self.q.marked_by is not None else None
            if held.startswith("held"):
                out[n] = (head, f"{held}, at {head[:12]}, the head it was built at")
            elif ejected and (_who(eby) == _who(EJECTOR) or _who(eby) not in mine):
                out[n] = (head, f"ejected by {eby} at {head[:12]}, the head it was built at: {e.get('why') or 'no reason given'}")
            elif by and _who(by) not in mine:
                out[n] = (head, f"its ready mark at {head[:12]}, the head it was built at, was withdrawn by {by}")
            elif specs and head not in gone and not self._cleared(n, head, clears) and self._frozen_out(below, n, head, specs):
                out[n] = (head, f"it waits for the freeze to end ({self.frozen_file}) at {head[:12]}, the head it was built at")
        return out

    def _changed(self, main: str, head: str) -> frozenset[str]:
        """The files ``head`` changes against its merge base with ``main`` (`own_files`)."""
        if (key := (head, main)) not in self._files:
            self._files[key] = own_files(self.repo, main, head)
        return self._files[key]

    def _lacks(self, main: str, head: str, files: frozenset[str]) -> list[tuple[str, str, list[str] | None]]:
        """`lacked_moves` of ``head`` on ``main``, kept by (head, main), each map read once."""
        if (key := (head, main)) not in self._lacked:
            self._lacked[key] = lacked_moves(self.repo, main, head, files, self._retires)
        return self._lacked[key]

    def crossing(self, main: str, ready: list[Ready], moves: dict[str, str]) -> list[Ready]:
        """``ready`` with each PR's `Ready.crosses`: those of ``moves`` ({a move PR's head, ready or in flight: its map's path})
        whose map retires nothing its own change against ``main`` names (`queue.crossed`), kept by its head, ``main`` and
        ``moves``, each map read once.  A move PR crosses none, nor, this tick, a PR whose own change can't be read."""
        if not moves:
            return ready
        rows = frozenset(moves.items())

        def of(x: Ready) -> Ready:
            if (key := (x.head, main, rows)) not in self._crossings:
                try:
                    self._crossings[key] = crossed(self.repo, main, x.head, sorted(rows), x.files, self._retires)
                except QueueError as e:
                    self.log(f"research queue vqueue: #{x.number} at {x.head[:12]} waits for the moves in flight to land: "
                             f"its own change can't be read: {e}")
                    return x
            return replace(x, crosses=self._crossings[key])

        with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:
            return list(pool.map(lambda x: x if x.move else of(x), ready))

    def _tree(self, sha: str) -> list[str]:
        """The files of ``sha``'s tree, as `_changed` names them."""
        return _out(self.repo, "ls-tree", "-r", "--name-only", sha).splitlines()

    def _judge(self, main: str, head: str, config: tuple[str, ...] | None = None) -> tuple[bool, bool]:
        """(whether ``head`` is a Lean change that compiles, whether it doesn't merge onto ``main``), kept by (head, main): a
        Lean change (`queue.lean_change`) whose only Lean files are locks (`lean-audit.json` records) compiles nothing.  The
        merge is `git merge-tree`'s alone (`merged_tree`, no worktree, with ``main``'s merge drivers: ``config``, else
        `base_drivers`), so judging every ready PR stays cheap; a build's codemod (`Queue._merge`) checks out a worktree, and a
        member it refuses is left out of that build as one that doesn't merge."""
        if (key := (head, main)) not in self._judged:
            self._judged[key] = (lean_change(self.repo, head, _out(self.repo, "merge-base", main, head))
                                 and any(f.endswith(LEAN_SOURCES) for f in self._changed(main, head)),
                                 merged_tree(self.repo, main, head, config=config)[0] is None)
        return self._judged[key]

    def _judging_key(self) -> str:
        """What decides `_changed`, `_lacks`, `_judge` and `Queue.own_change` of a head on a `main` beyond the two commits: this
        package's code (the merge drivers' command among it) and the clone's git."""
        if not self._judging:
            h = hashlib.sha512(_out(self.repo, "--version").encode())
            for f in sorted(Path(__file__).parent.glob("*.py")):
                h.update(f"{f.name}\0{len(b := f.read_bytes())}\0".encode() + b)
            self._judging = h.hexdigest()
        return self._judging

    def _take_judged(self, main: str) -> None:
        """What the tick before judged of each head on ``main`` (`_keep_judged`), read once a process: nothing from a cache of
        another `main`, code or git (`_judging_key`), or one that doesn't parse."""
        if self._judged_at is None or self._judged_text is not None:
            return
        self._judged_text = ""
        if not self._judged_at.is_file():
            return
        try:
            doc = json.loads(text := self._judged_at.read_text())
            if doc["main"] != main or doc["judging"] != self._judging_key():
                return
            files = {(h, main): frozenset(v) for h, v in doc["files"].items()}
            judged = {(h, main): (bool(lean), bool(conflict)) for h, (lean, conflict) in doc["judged"].items()}
            lacked = {(h, main): [(m, at, names) for m, at, names in v] for h, v in doc["lacked"].items()}
            own = {(self.q.base, h): v for h, v in doc["own"].items()} if self._main == main else {}
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log(f"research queue vqueue: {self._judged_at} can't be read ({type(e).__name__}: {e}): this tick judges "
                     "every head afresh")
            return
        self._judged_text = text
        for mine, got in ((self._files, files), (self._judged, judged), (self._lacked, lacked), (self.q._own, own)):
            mine.update({k: v for k, v in got.items() if k not in mine})

    def _keep_judged(self, main: str) -> None:
        """What this tick judged of each open PR's head on ``main``, kept whole in the loop's cache (`JUDGED_CACHE`) for the next
        tick's process, under the code and git that judged it (`_judging_key`); with the own change on ``main`` of each head the
        labels read, older heads among them (`Queue.own_change`, on the landing branch as fetched)."""
        if self._judged_at is None:
            return
        heads = {p.head for p in self.open().values()}

        def kept(got: dict) -> dict:
            return {h: v for (h, m), v in sorted(got.items()) if m == main and h in heads}

        base = self.q.base if self._main == main else None
        doc = {"main": main, "judging": self._judging_key(), "files": {h: sorted(v) for h, v in kept(self._files).items()},
               "judged": {h: list(v) for h, v in kept(self._judged).items()},
               "lacked": {h: [list(x) for x in v] for h, v in kept(self._lacked).items()},
               "own": {h: v for (b, h), v in sorted(self.q._own.items()) if b == base}}
        if (text := json.dumps(doc, sort_keys=True) + "\n") != self._judged_text:
            tmp = self._judged_at.with_name(f"{self._judged_at.name}.{os.getpid()}.tmp")
            tmp.write_text(text)
            tmp.replace(self._judged_at)
            self._judged_text = text

    def pins(self, sha: str, targets: Iterable[str]) -> dict[str, list[str]]:
        """The circuit-check sidecars' pins files in ``sha``'s tree (`*.circuit-check.json`) that pin each of ``targets``, a
        key of theirs, by target (one at least); a target none pins has none."""
        r = _git(self.repo, "grep", "-F", "-o", *(a for x in targets for a in ("-e", f'"{x}"')), sha, "--",
                 "*.circuit-check.json", check=False)
        out: dict[str, list[str]] = {}
        for line in r.stdout.splitlines():
            path, _, hit = line.removeprefix(f"{sha}:").partition(':"')
            out.setdefault(hit.removesuffix('"'), []).append(path)
        return out

    def refs(self) -> dict[str, str]:
        """origin's `queue/*` branches as fetched: {name under `queue/`: sha}."""
        out = _out(self.repo, "for-each-ref", "--format=%(refname) %(objectname)", f"{NS}/queue/")
        return {ref.removeprefix(f"{NS}/queue/"): sha for ref, sha in (line.split() for line in out.splitlines())}

    def members(self, sha: str, base: str, prs: Iterable[int], heads: Iterable[str]) -> list[Member]:
        """Candidate ``sha``'s members as `Landing._on_main` reads them, oldest first: the `Queue: #N` merges on its first-parent
        line above ``base``, while they are the PRs ``prs`` at ``heads`` as built; else those, without their merges."""
        prs, heads = list(prs), list(heads)
        if (walked := self._walk(sha, base)) is not None and [(m.pr, m.head) for m in walked] == list(zip(prs, heads)):
            return walked
        self.log(f"research queue vqueue: {sha[:12]}'s first-parent line above {base[:12]} isn't its members' merges as built")
        return [Member(p, h, "", "") for p, h in zip(prs, heads)]

    def _walk(self, sha: str, base: str) -> list[Member] | None:
        """The `Queue: #N` merges on ``sha``'s first-parent line above ``base``, oldest first, beneath a records commit at its
        tip (`records_commit`); None when another commit is."""
        recs = [r.strip() for r in _out(self.repo, "log", "--first-parent", f"-n{WALK}", "--format=%H%x00%P%x00%B%x1e",
                                         f"{base}..{sha}").split("\x1e") if r.strip()]
        if recs and records_commit(*recs[0].split("\x00", 2)):
            recs = recs[1:]
        walked = [queue_commit(*r.split("\x00", 2)) for r in reversed(recs)]
        return walked if all(walked) else None

    def landed(self, since: str | None, tip: str) -> list[tuple[Member, frozenset[str]]]:
        """The PRs whose `Queue: #N` merges are on ``tip``'s first-parent line above ``since`` (`MAIN_WALK` commits back, with
        none), oldest first, each with what its merge changed on that line (a rename's both paths), but those a revert on the
        line names (`merge.REVERTS`): a revert's change isn't in ``tip``'s tree."""
        from .merge import REVERTS

        span = [f"{since}..{tip}"] if since else [f"-n{MAIN_WALK}", tip]
        recs = [r.strip() for r in _out(self.repo, "log", "--first-parent", "--format=%H%x00%P%x00%B%x1e", *span).split("\x1e")
                if r.strip()]
        reverted = {m for r in recs for m in REVERTS.findall(r.split("\x00", 2)[-1])}
        return [(m, frozenset(_out(self.repo, "diff", "--name-only", "--no-renames", f"{m.commit}^1", m.commit).splitlines()))
                for r in reversed(recs) if (m := queue_commit(*r.split("\x00", 2))) and m.commit not in reverted]

    def reverts(self, since: str, tip: str) -> dict[str, str]:
        """The merges reverted on ``tip``'s first-parent line above ``since`` (`merge.REVERTS`), each by its revert."""
        from .merge import REVERTS

        out = _out(self.repo, "log", "--first-parent", "--format=%H%x00%B%x1e", f"{since}..{tip}")
        return {m: sha for r in out.split("\x1e") if r.strip() for sha, body in [r.strip().split("\x00", 1)]
                for m in REVERTS.findall(body)}

    def revert(self, merge: str, main: str, why: str) -> str | None:
        """A commit on ``main`` undoing the queue's merge ``merge`` as `git revert -m 1` does, with git's message (`Revert
        "SUBJECT"`, `This reverts commit MERGE, reversing changes made to PARENT.`, which `digest.REVERT_SUBJECT` and
        `merge.REVERTS` read) and ``why`` below it, made without a worktree (`git merge-tree` from ``merge`` onto ``main`` of its
        first parent), as the queue's commits are (`Queue.identity`: the author its repository's rules name); None when it
        doesn't apply cleanly."""
        parent = _out(self.repo, "rev-parse", f"{merge}^1")
        r = _git(self.repo, "merge-tree", "--write-tree", f"--merge-base={merge}", main, parent, check=False)
        if r.returncode:
            return None
        subject = _out(self.repo, "log", "-1", "--format=%s", merge)
        msg = f'Revert "{subject}"\n\nThis reverts commit {merge}, reversing\nchanges made to {parent}.\n\n{why}\n'
        return _git(self.repo, "commit-tree", r.stdout.split()[0], "-p", main, "-m", msg, env=self.q.identity).stdout.strip()

    def recommitted(self, sha: str, tip: str, parent: str, base: str = "", members: list[Member] | None = None) -> bool:
        """Whether ``sha`` is records commit ``tip`` made before on ``parent`` (pushed with no line in the log, as a tick cut off
        leaves): a records commit (`records_commit`) with ``tip``'s tree whose one parent is ``parent``, or, given ``members``
        merged onto ``base`` as ``parent``, that merge built before (`rebuilt`)."""
        if self._missing([sha]):
            return False
        commit, parents, body = _out(self.repo, "log", "-1", "--format=%H%x00%P%x00%B", sha).split("\x00", 2)
        if not records_commit(commit, parents, body) or \
                _out(self.repo, "rev-parse", f"{sha}^{{tree}}") != _out(self.repo, "rev-parse", f"{tip}^{{tree}}"):
            return False
        return parents == parent or (members is not None and " " not in parents and
                                     self.rebuilt(parents, parent, base, members))

    def rebuilt(self, sha: str, tip: str, base: str, members: list[Member]) -> bool:
        """Whether ``sha`` is ``tip``, ``members`` merged onto ``base``, built before: its tree, and their merges at their
        heads on its first-parent line above ``base`` (a build pushed with no line in the log, as a tick cut off leaves)."""
        if self._missing([sha]) or not self._ancestor(base, sha):
            return False
        if _out(self.repo, "rev-parse", f"{sha}^{{tree}}") != _out(self.repo, "rev-parse", f"{tip}^{{tree}}"):
            return False
        walked = self._walk(sha, base)
        return walked is not None and [(m.pr, m.head) for m in walked] == [(m.pr, m.head) for m in members]

    def shadow_main(self) -> str | None:
        """The newest candidate the shadow advanced, as it keeps it (`SHADOW`)."""
        r = _git(self.repo, "rev-parse", "-q", "--verify", f"{SHADOW}^{{commit}}", check=False)
        return r.stdout.strip() or None

    def set_shadow_main(self, sha: str) -> None:
        _git(self.repo, "update-ref", SHADOW, sha)

    # -- effects

    def build(self, prs: list[PR], base: str, main: str | None = None) -> tuple[str, list[Member], list[Skip]]:
        """``prs`` merged in order onto ``base`` (`Queue._merge`), which is ``main`` or stands on it (``base`` when not given):
        (the tip, the members merged, each left out).  A skip names what holds its head already (`Skip.held`), and one that
        conflicts with what is merged above ``main`` but merges onto ``main`` names the members it conflicts with
        (`Skip.by`), not `main`."""
        main = main or base
        tip, merged, left = base, [], []
        self.certified = {}
        for p in sorted(prs, key=lambda p: bool(p.move)):
            if p.move:
                regenerated, skip = self._move(p, tip, main)
                if regenerated is not None:
                    merged.append(Member(p.number, p.head, p.owner or "", regenerated))
                    tip = regenerated
                    continue
                if skip is not None:
                    left.append(skip)
                    continue
            commit, why = self.q._merge(tip, p, merged)
            if commit == tip:
                held = self._holder(p.head, base, main, merged)
                left.append(Skip(p, f"its head {p.head[:12]} is already in what it merges onto: {held} holds it", held=held))
            elif commit is None:
                left.append(self._unmerged(p, why, tip, base, main, merged))
            else:
                merged.append(Member(p.number, p.head, p.owner or "", commit))
                tip = commit
        self._keep_moves(main)
        return tip, merged, left

    def _move(self, p: PR, tip: str, main: str) -> tuple[str | None, Skip | None]:
        """Move PR ``p`` regenerated last in a build on ``tip``, landing tip ``main`` (`Queue._regenerate`, with ``main``'s
        `tools/move/rename.py` and `move_check.py`, kept in the loop's cache): (its certified commit, None), its certificate in
        `certified`; or (None, why it's left out) while members below it change `tools/move/`'s scripts, or when it came on its
        mark alone (`on_mark`) and its move isn't certified or drive.toml keeps what the regenerated commit changes off the
        mark (`_off_mark`, which `ready` judged on its head's files only); or (None, None) when it isn't certified and has a
        land verdict, which covers its head: merged as any PR."""
        if below := self.q._scripts_below(tip, main):
            return None, Skip(p, f"the members below it change {', '.join(below)}, which its move and the restacks across "
                                 "it run: it waits until they land", waits=True)
        commit, why = self.q._regenerate(tip, p, main)
        if commit is not None and p.number in self.on_mark and (off := _off_mark(
                _out(self.repo, "diff", "--name-only", "--no-renames", tip, commit).splitlines(), flags_at(self.repo, main))):
            return None, Skip(p, f"it came on its mark alone, and its move regenerated on {tip[:12]}: {off}",
                              fix="It lands with a land verdict.")
        if commit is not None:
            self.certified[p.number] = {"map": p.move, "tip": tip, "commit": commit, "on_mark": p.number in self.on_mark,
                                        "certificate": self.q._made[Queue._move_key(tip, p, main)].get("certificate")}
            return commit, None
        if p.number in self.on_mark:
            return None, Skip(p, f"it came on its mark alone, and {why}",
                              fix="It lands with a land verdict, or on its mark once a candidate on a later `main` certifies "
                                  "its move.")
        self.log(f"research queue vqueue: #{p.number}'s move isn't regenerated on {tip[:12]} ({why}): its land verdict covers "
                 "its head, merged as any PR's")
        return None, None

    def _keep_moves(self, main: str) -> None:
        """The moves regenerated on candidates on ``main`` (`Queue._made`), kept in the loop's cache for the next tick, which
        `_uncertified` and `Queue._regenerate` read; those on an older `main` are dropped."""
        q = self.q
        q._made = {k: v for k, v in q._made.items() if k.split(":")[1] == main}
        if self._made_at is not None:
            self._made_at.write_text(json.dumps(q._made, sort_keys=True) + "\n")

    def _above(self, base: str, main: str, merged: list[Member]) -> list[Member]:
        """The members of a build on ``base`` above ``main`` so far, oldest first: the candidates' below it (`_walk`), then
        ``merged``."""
        return [*((self._walk(base, main) or []) if base != main else []), *merged]

    def _holder(self, head: str, base: str, main: str, merged: list[Member]) -> str:
        """What of a build holds ``head``: the first member above ``main`` whose head does, else ``base``.  ``main`` holds no
        head a build merges: a ready PR's head isn't on it (`ready`), nor in a candidate on its chain (`Tick.riding`)."""
        return next((f"#{m.pr}'s head" for m in self._above(base, main, merged) if self._ancestor(head, m.head)),
                    f"the candidate {base[:12]}")

    def _unmerged(self, p: PR, why: str, tip: str, base: str, main: str, merged: list[Member]) -> Skip:
        """``p`` left out of a build for not merging onto ``tip`` (``why``, `Queue._merge`'s): where it merges onto ``main``,
        its conflict is with the members above it whose merges touched the files (else all of them), named."""
        if tip == main or not why.startswith("conflicts with"):
            return Skip(p, why)
        files = self.q._recorded(tip, p.head).get("conflict")
        if not files or conflicting_files(self.repo, main, p.head):
            return Skip(p, why)
        above = self._above(base, main, merged)
        by = [int(n[1:]) for n in touched_by(self.repo, above, files, self.q._touched)] or [m.pr for m in above]
        return Skip(p, f"conflicts with {', '.join(f'#{n}' for n in by)}, merged before it, in {', '.join(files)} (it merges "
                       "onto `main`)", by=tuple(by))

    def holding(self, heads: Iterable[str], shas: Iterable[str], main: str) -> dict[str, str]:
        """{head: the first of ``shas`` whose commits above ``main`` hold it} for each of ``heads`` one does: one `rev-list` a
        candidate, not an ancestry check a PR."""
        want, out = set(heads), {}
        for sha in shas:
            if not want:
                break
            got = want.intersection(_out(self.repo, "rev-list", f"{main}..{sha}").split())
            out.update(dict.fromkeys(got, sha))
            want -= got
        return out

    def main_now(self) -> str:
        """origin's `main` fetched again (into `MAIN_NOW`)."""
        self.q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "origin",
                       f"+refs/heads/main:{MAIN_NOW}")
        return _out(self.repo, "rev-parse", MAIN_NOW)

    def push(self, sha: str, ref: str, lease: str | None = None) -> None:
        """origin's ``ref`` to ``sha`` ("": deleted), with a lease on ``lease`` when given."""
        self.q._remote(self.repo, "push", "-q", *([f"--force-with-lease={ref}:{lease}"] if lease is not None else []), "origin",
                       f"{sha}:{ref}" if sha else f":{ref}")

    def cancel_runs(self, ids: Iterable[str], why: str) -> dict[str, str]:
        """`research cancel` of runs ``ids``, as drive cancels a run beside a check: {run id: why} of each that failed, or that
        ``keep_file`` names (`KEEP_RUNS`), which is left running."""
        ids, keep = list(ids), kept_runs(self.keep_file)
        rest = [i for i in ids if i not in keep]
        return {**{i: f"left running: {self.keep_file} names it" for i in ids if i in keep},
                **(cancel_runs(self.repo, rest, why) if rest else {})}

    def advance(self, sha: str, base: str) -> None:
        """origin's `main` fast-forwarded from ``base`` to ``sha``."""
        self.q._drive_only("main", "advance", False)
        self.push(sha, "refs/heads/main", base)

    def on_main(self, sha: str, base: str, members: list[Member], told: list[str]) -> None:
        """What follows ``members`` reaching `main` at ``sha`` from ``base`` (`Landing._on_main`); ``told``, the stacked PRs
        whose owners were told, is the list it extends as it tells."""
        self.driver()._on_main({"told": told, "stale_closed": self.q.load_state().get("drive", {}).get("stale_closed", {})},
                               members, base, sha, "")
        moved = [(m, p) for m in members if m.commit and (p := self.open().get(m.pr)) is not None and p.head == m.head
                 and move_of(self.repo, m.commit)]
        with authed(self.q.auth):
            for m, p in moved:
                try:
                    self.q.github.close(m.pr, f"Landed on `main` at {sha[:12]} as {m.commit[:12]}: its move ({p.move}) "
                                              f"regenerated there by the verified queue and certified by `{MOVE_CHECK}`. Its "
                                              "head was only the request.")
                except QueueError as e:                     # `ready` leaves it out while it stays open
                    self.log(f"research queue vqueue: #{m.pr} landed as its regenerated move, but isn't closed: {e}")

    def start_after(self, job: dict, argv: list[str], root: Path) -> dict:
        """``argv`` with `--after-landing DIR` (`_after_landing`) in a session of its own, as `landing.detach` starts a launch,
        so no tick waits on it: ``job`` in `job.json` of a directory of its own under ``root``, its output in `log` there and
        then appended to `AFTER_LOG` beside ``root``, and its exit written to `rc` there (`AFTER_JOB`).  The record
        `after_ended` reads."""
        import tempfile

        root.mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix=f"{job['sha'][:12]}-", dir=root))
        (d / "job.json").write_text(json.dumps(job, sort_keys=True) + "\n")
        p = subprocess.Popen(["sh", "-c", AFTER_JOB, "sh", str(d), str(root.parent / AFTER_LOG), *argv, "--after-landing", str(d)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        return {"dir": str(d), "pid": p.pid, "out": str(d), "at": self.now()}

    def after_ended(self, job: dict) -> dict | str | None:
        """How the after-landing job ``job`` (`start_after`) ended, its directory removed: what it wrote (`_after_landing`'s
        `result.json`: done, error, told), or {done: false, error} when it exited with none; why once its process is gone
        with no exit written, or `AFTER_LOST_S` after its start; None while it runs."""
        import shutil

        from .landing import _job_end

        if (end := _job_end(job, "result.json")) is None and self.now() - float(job.get("at") or 0) < AFTER_LOST_S:
            return None
        if end is None or isinstance(end, str):
            shutil.rmtree(job["dir"], ignore_errors=True)
            return end or f"no exit written {AFTER_LOST_S // 60} min after its start"
        rc, tail, text = end
        try:
            got = json.loads(text) if text is not None else None
        except ValueError as e:
            return {"done": False, "error": f"exited {rc}, its result unreadable ({e}): {tail}"[-600:]}
        return got if isinstance(got, dict) else {"done": False, "error": f"exited {rc} with no result: {tail}"[-600:]}

    def withdraw(self, number: int, head: str, why: str) -> PR:
        """`Queue.eject` under the clone's queue lock, as `research queue eject` takes it: it writes drive's state, which a
        drive tick holding the lock rewrites whole (held, the eject fails and the next tick ejects again)."""
        from .queue import _lock

        lock = _lock(self.repo)
        try:
            return self.q.eject(number, why, EJECTOR, sha=head)
        finally:
            lock.close()

    def handle(self, number: int) -> str | None:
        """The handle of PR ``number``'s lane (`Queue.handle`): of an open PR, or of one GitHub has closed, as a member is once
        its candidate lands, read once a tick; None when none is known."""
        if (p := self.prs.get(number) or self.open().get(number)) is None:
            if number not in self._closed:
                try:
                    self._closed[number] = self.q.github.pr(number)
                except (QueueError, ValueError, KeyError, TypeError) as e:
                    self._closed[number] = None
                    self.log(f"research queue vqueue: #{number} isn't open and its PR can't be read, so it names no lane: {e}")
            p = self._closed[number]
        return self.q.handle(p) if p is not None else None

    def say(self, text: str, to: tuple[str, ...]) -> bool:
        """``text`` posted to ``to`` in the merge thread, as drive posts."""
        return bool(self.post(f"{' '.join(to)} vqueue: {text}", to))

    def driver(self) -> Landing:
        """The landing's steps as the loop runs them (`Landing`): on the labels and the open PRs this tick reads, each
        `carry.delta` kept (`_delta`)."""
        if self._driver is None:
            self._driver = Landing(self.q, on="", post=self.post, now=self.now, reviews=lambda n, h: self.labels.reviews(n, h),
                                   ready_by=lambda n, h: self.labels.ready_by(n, h))
            self._driver.delta = self._delta
        self._driver._prs = self.open()
        return self._driver

    def slim_driver(self) -> Landing:
        """The `Landing` the run beside each check and each records run are launched and judged with (`vqueue_slim`,
        `vqueue_records`), on ``on``'s machines, else drive's: drive.toml's `on` at `main` when first asked, as its
        `warn_steps`, which no check that gates is judged on.  It outlives the ticks, as `vqueue_slim.judge` and
        `vqueue_records.judge` expect."""
        if self._slim is None:
            flags = flags_at(self.repo, self._main or MAIN)
            if not (on := self.on or flags.on):
                raise QueueError(f"drive.toml at main {(self._main or MAIN)[:12]} names no `on`, drive's machines, so no run "
                                 "beside a candidate's check or records run can launch: name them there or with --on "
                                 "(--no-slim runs only with --shadow or --dry-run, and --no-records checks a Lean candidate "
                                 "with no records run first)")
            self._slim = Landing(self.q, on=on, post=self.post, now=self.now, warn_steps=flags.warn_steps)
        return self._slim


def _running(j: Judged | None) -> dict:
    """The record of the run beside a judged candidate's check while that runs: pending, or a check that gates failed before
    it ended, a fail `vqueue_slim.judge` doesn't keep; {} once it ended or its verdict is kept, as a pass is while its runs
    still run steps off the gate (`vqueue_slim._passed_early`), which only its drop on `main` leaves running (`Loop._drop`)."""
    if j is None or j.slim is None or "verdict" in (j.record or {}) or j.slim.state not in (PENDING, FAIL):
        return {}
    return j.record or {}


#: where a check that takes lean-audit's verdict from candidate ``sha``'s reads it (`_lent`): the record, and whether
#: ``sha`` has landed
Lent = Callable[[str], tuple[dict | None, bool]]


def _lent(judged: dict[str, Judged], beside: dict | None, *landed: str | None) -> Lent:
    """Where a candidate's check's verdict on lean-audit is read: its record as judged now while it is a candidate, though
    landed if ``landed`` names it (`main`, or the candidate this tick advanced it to), else, landed, the run beside `main`'s
    (``beside``), which keeps it from that candidate's or from one that took it from it (`vqueue_slim.audited`).  A launch
    reads it in flight too (`Loop._audit_from`); a judging only once landed (`Loop._beside`)."""
    return lambda sha: (judged[sha].record, sha in landed) if sha in judged else (beside, True)


def _failing_tests(f: Failed, most: int = 5) -> str:
    """The failing tests ``f`` names, the first ``most`` of them, for an owner to read: not what it fails beyond them
    (`vqueue_slim.unnamed`)."""
    tests = sorted(x for x in f.tests if not x.split("::")[-1].startswith("unnamed:"))
    return ", ".join(tests[:most]) + (f" and {len(tests) - most} more" if len(tests) > most else "")


def _target(key: str) -> str:
    """What the failure `vqueue_slim.failures` keys ``key`` failed: circuit-check's target (its key `CHECK/CODE TARGET`), else
    the key itself (a merge's name, the step's)."""
    return key.split(" ", 1)[-1]


@dataclass
class Tick:
    """What one tick saw: its snapshot and what each action needs that the snapshot doesn't hold: ``past``, the log's window,
    or the whole log while a candidate's build is older than the window, so its once-only rules read all of its lines;
    ``foreign``, origin's `queue/*` branches no build of the loop's made ({name: sha}), which are no candidates."""
    n: int
    main: str
    origin: str
    snap: Snapshot
    judged: dict[str, Judged]
    past: Past
    #: each (head, base) a skip told its owner of, as `head@base`
    told: set[str] = field(default_factory=set)
    foreign: dict[str, str] = field(default_factory=dict)
    #: what the snapshot logged: the runs beside the checks it started or judged
    wrote: list[dict] = field(default_factory=list)
    #: the record of the run beside `main`'s own check (`Loop._below`), None while none is kept
    below: dict | None = None
    #: each candidate's records run's record as the snapshot judged it (`Loop._records_run`), by sha
    records: dict[str, dict] = field(default_factory=dict)
    #: the verdict carry's decisions (`World.carried`)
    carried: dict[int, dict] = field(default_factory=dict)
    #: why each mark and verdict the tick's `ready` counted wouldn't count under `signed_verdicts` (`World.unauthenticated`)
    unauthenticated: dict[tuple, str] = field(default_factory=dict)
    #: each ready PR a candidate on `main`'s chain holds without it as a member, left out of `snap.ready` (a build of it
    #: merges nothing): {number: (head, the lowest such candidate)}
    riding: dict[int, tuple[str, str]] = field(default_factory=dict)
    #: the candidate this tick advanced `main` to, what its build reads as `main`
    advanced: str | None = None
    #: what each `Commit` of this tick put on main's chain, by the staged batch it committed: the candidate, and the record of
    #: the run beside its check as started, which a build after it is built on and reads as its base's (`vqueue.Build.after`)
    committed: dict[str, tuple[str, dict | None]] = field(default_factory=dict)
    #: each candidate a build this tick rebuilt (`vqueue.Build.replaces`): what that build pushed, or what it went on when
    #: no member merged, which a later build naming the candidate as its base goes on
    rebuilds: dict[str, str] = field(default_factory=dict)
    #: `main` fetched again, the first time this tick a notice needed it (`Loop._landed`)
    fetched: str | None = None
    #: whether a detached job of the steps after an advance runs (`Loop._moved`)
    after_running: bool = False


class Loop:
    """The verified queue's loop on ``world``: `tick` once, or `run` every ``interval`` seconds.  ``slim`` is
    `research.vqueue_slim` (its `start` and `judge`): what checks each candidate, as ``shards`` shards (N of ci.toml's plan,
    or `auto:N`: `vqueue_slim.shard_count`); None starts no check, so every candidate stays pending.  ``gate`` is what gates
    a candidate (`GATES`): its check, or, for a plain candidate, its members' land verdicts and the quick tier.  ``records``
    is `research.vqueue_records` (its `start`, `judge` and `commit`), or None to start every candidate's check at once, with
    no records run before it (`vqueue.Snapshot.records`).  ``after`` is the tick's own command, which the steps after `main`
    moved run detached as, with `--after-landing` (`_moved`); None runs them in the tick.  Each tick gives ``timing`` a line
    of its steps' seconds on ``clock`` and the peak resident size after each (``peak``, `Steps`)."""

    def __init__(self, world: World, log: Path, *, decide: Callable[[Snapshot], list[Action]] = vqueue.decide,
                 shadow: bool = False, dry_run: bool = False, batch: int = BATCH, depth: int = DEPTH, lean: int = LEAN,
                 pending_s: float = 4 * 3600, slim=None, now: Callable[[], float] = time.time,
                 out: Callable[[str], None] = print, state: Path | None = None, gate: str = CHECK_GATE,
                 shards: int | str = SHARDS, records=None, after: list[str] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 peak: Callable[[], float] = _peak_gb, timing: Callable[[str], None] = print,
                 left_out: Callable[[Snapshot], list[Leave]] | None = vqueue.left_out):
        if gate not in GATES or slim is not None and not _sharded(slim, shards):
            raise ValueError(f"gate {gate!r} with {shards} shard(s): a gate is one of {', '.join(GATES)}, and the check "
                             "(`research.vqueue_slim`) runs as one shard or more")
        self.world, self.log_path, self.decide, self.shadow, self.dry_run = world, Path(log), decide, shadow, dry_run
        self.batch, self.depth, self.lean, self.pending_s, self.slim = batch, depth, lean, pending_s, slim
        self.gate, self.shards = gate, shards
        self.now, self.out, self.state, self.records = now, out, state, records
        self.after, self.clock, self.peak, self.timing, self.steps = after, clock, peak, timing, Steps(clock, peak)
        self.left_out = left_out
        #: where the land verdicts gate, this tick's red of `main`'s own check (`_main_red`), and the plain PRs, by number, at
        #: their heads (`_plain`)
        self.red: MainRed | None = None
        self.plain: dict[int, str] = {}
        #: the ready PRs this tick's snapshot holds for that red, as `World.held_out` names one: {number: (head, cause, why)}
        self.held: dict[int, tuple[str, str, str]] = {}
        #: the last tick this loop ran: a tick that logs nothing still counts
        self.last = 0

    # -- the snapshot

    def snapshot(self, past: Past, n: int) -> Tick:
        w, steps = self.world, self.steps
        origin = w.fetch()
        main = self._shadow_main(past, origin) if self.shadow else origin
        steps.lap("fetch")
        wide, pinned = w.wide() | w.heavy()[0], w.pinned()
        ejected = {(r.get("number"), (r.get("heads") or [None])[0]): r.get("at") for r in past.of("eject") if r.get("shadow")}
        ready, out = [], {}
        for x in w.ready(main):
            at = ejected.get((x.number, x.head))
            if at is not None and not (w.labels.marked_at(x.number, x.head) or "") > at:
                out[x.number] = (x.head, "the shadow ejected it at this head, and it isn't marked ready since")
                continue
            ready.append(x)
        left = {**w.left, **out}
        refs, builds, whole = w.refs(), past.builds(), past
        if older := {sha for name, sha in refs.items() if name == sha[:12] and sha not in builds
                     and (name, sha) not in past.ignored()}:                # past the window: read the whole log for them
            whole = Past.read(self.log_path, 0)
            builds |= {sha: r for sha, r in whole.builds().items() if sha in older}
        mine = {sha for name, sha in refs.items() if name == sha[:12] and sha in builds}
        foreign = {name: sha for name, sha in refs.items() if sha not in mine or name != sha[:12]}
        cands, judged, wrote, regen = [], {}, [], {}
        bases = {sha: builds[sha].get("base") or "" for sha in mine}

        def chained(sha: str, seen: tuple[str, ...] = ()) -> bool:
            return bases[sha] == main or bases[sha] in bases and sha not in seen and chained(bases[sha], (*seen, sha))

        def height(sha: str, seen: tuple[str, ...] = ()) -> int:
            return 1 + height(bases[sha], (*seen, sha)) if bases.get(sha) in bases and sha not in seen else 0

        steps.lap("ready")
        with self._reading(past, whole, main, mine, older):
            steps.lap("reads")
            below = self._below(past, whole, n, main, origin, wrote)
            steps.lap("below")
            if self.gate == VERDICT_GATE:
                self.red, self.held = self._main_red(past, whole, n, main, origin, below[0], wrote), {}
                if self.red is not None and self.red.holds:
                    held = {x.number: (x.head, f"held: `main`'s check at {self.red.tip[:12]} failed {self.red.steps()}, which read "
                                               "what its change meets; it waits until `main`'s check passes again")
                            for x in ready if vqueue.meets(x.files, self.red.holds)}
                    ready = [x for x in ready if x.number not in held]
                    left.update(held)
                    self.held = {k: (head, MAIN_FAILED, why) for k, (head, why) in held.items()}
                self.plain = self._plain(ready, main)
                steps.lap("main_red")
            lent = _lent(judged, below[0], main)
            for sha in sorted(mine, key=lambda s: (height(s), s)):  # each after its base, whose record it reads as judged now
                p, b = whole if sha in older else past, builds[sha]
                base, prs, heads = b.get("base") or "", tuple(b.get("members") or ()), tuple(b.get("heads") or ())
                waits = self.records is not None and bool(b.get("records_first")) and not p.tries(sha)
                if waits:
                    j = Judged(Verdict(PENDING), note="its check waits for its records run")
                elif self.slim is None:
                    j = Judged(Verdict(PENDING), note="no check runs (--no-slim)")
                else:
                    j = self._beside(p, n, main, origin, sha, base, prs, heads, wrote, chained(sha),
                                     below if base == main else ((judged[base].record if base in judged else None)
                                                                 or p.slim(base)[1], p.slim_started(base)[1]), lent,
                                     on_main=below[0])
                judged[sha] = j
                cands.append(Candidate(sha, base, prs, heads, j.verdict, p.tries(sha), reruns=p.reruns(sha),
                                       enqueued=p.enqueued(sha),
                                       records=self._records_run(p, n, main, origin, sha, b, wrote, regen),
                                       repairs=b.get("repairs") if isinstance(b.get("repairs"), str) else None,
                                       waits=waits, held=p.held(sha) if waits else None))
        steps.lap("judge")
        chain = [c.sha for c in cands if chained(c.sha)]
        members = {m for c in cands if c.sha in chain for m in c.members}
        holds = w.holding([x.head for x in ready if x.number not in members], chain, main) if chain else {}
        riding = {x.number: (x.head, holds[x.head]) for x in ready if x.number not in members and x.head in holds}
        history = past.history((lambda sha: bool(sha) and w._ancestor(sha, main)) if self.shadow else None)
        at = {(x.number, x.head) for x in ready}
        flying = {(m, h) for c in cands if c.sha in chain for m, h in zip(c.members, c.heads) if (m, h) not in at}
        gone = w._missing(h for _, h in flying)
        left.update(taken := w.taken({(m, h) for m, h in flying if m not in w.pulled}, main))
        ready = w.crossing(main, ready, {h: p.move for m, h in sorted(at | flying)
                                         if (p := w.open().get(m)) is not None and p.move})
        snap = Snapshot(main, tuple(sorted((x for x in ready if x.number not in riding), key=lambda x: x.rank)), tuple(cands),
                        history, self.batch, self.depth, lean=self.lean, now=self.now(), pending_s=self.pending_s,
                        left=tuple((n, *left[n]) for n in sorted(left)), wide=wide,
                        pinned=pinned, pulled=frozenset(w.pulled | set(out) | set(taken)),
                        flying=tuple(Ready(m, h, (f := frozenset() if h in gone else w._changed(main, h)),
                                           lean=h not in gone and w._judge(main, h)[0],
                                           move=bool(getattr(w.open().get(m), "move", "")),
                                           scripts=vqueue.meets(f, MOVE_SCRIPTS)) for m, h in sorted(flying)),
                        records=self.records is not None)
        steps.lap("chain")
        return Tick(n, main, origin, snap, judged, whole, past.told(), foreign, wrote, below[0], regen, dict(w.carried),
                    dict(w.unauthenticated), riding=riding)

    def _reading(self, past: Past, whole: Past, main: str, mine: set[str], older: set[str]):
        """The tick's reads of the runs its judging reads (`Landing.reading`): every run of a record not yet decided (or a
        pass kept while its runs still run) of the run beside ``main``'s check, a candidate's or its records run's, read at
        once before the first is judged.  Nothing with --dry-run, which judges none."""
        if self.dry_run or (self.slim is None and self.records is None):
            return contextlib.nullcontext()
        recs = [past.slim(main)[1], whole.slim(main)[1]] if self.slim is not None else []
        for sha in mine:
            p = whole if sha in older else past
            recs += [*([p.slim(sha)[1]] if self.slim is not None else ()),
                     *([p.regenerated(sha)[1]] if self.records is not None else ())]
        ids = [i for r in recs if isinstance(r, dict) and ("verdict" not in r or r.get("still_running"))
               for i in r.get("ids") or () if isinstance(i, str)]
        try:
            reading = getattr(self.world.slim_driver(), "reading", None) if ids else None
        except QueueError as e:                             # the judging that needs the driver raises it
            self.world.log(f"research queue vqueue: the tick's runs aren't read at once, each judge reads its own: {e}")
            reading = None
        return reading(ids) if callable(reading) else contextlib.nullcontext()

    def _records_run(self, past: Past, n: int, main: str, origin: str, sha: str, built: dict, wrote: list[dict],
                     regen: dict[str, dict]) -> RecordsRun | None:
        """Candidate ``sha``'s records run once one was started (`Regenerate`; None before): judged each tick
        (`vqueue_records.judge`), its record kept in ``regen`` for the tick's actions and logged (to ``wrote`` too) whenever it
        changed; with --dry-run not judged, so pending."""
        starts, rec = past.regenerated(sha)
        if self.records is None or not starts:
            return None
        if isinstance(rec, dict):
            regen[sha] = rec
        if self.dry_run:
            return RecordsRun(starts=starts)
        was = json.dumps(rec, sort_keys=True)
        r = self.records.judge(self.world.slim_driver(), rec)
        if json.dumps(rec, sort_keys=True) != was:
            self.write(got := self._line(n, main, origin, "regenerate", sha=sha, base=built.get("base") or "",
                                         members=built.get("members") or (), heads=built.get("heads") or (),
                                         why=f"judged {r.state}", record=rec, done=True))
            wrote.append(got)
        return replace(r, starts=starts)

    def _shadow_main(self, past: Past, origin: str) -> str:
        """The shadow's `main`: the newest candidate it advanced (`SHADOW`, else its log's newest advance) while origin's `main`
        is below it; else origin's."""
        w = self.world
        for sha in (w.shadow_main(), next((r["sha"] for r in reversed(past.of("advance")) if r.get("shadow") and r.get("sha")),
                                          None)):
            if sha and not w._missing([sha]) and w._ancestor(origin, sha):
                return sha
        return origin

    def _below(self, past: Past, whole: Past, n: int, main: str, origin: str,
               wrote: list[dict]) -> tuple[dict | None, float | None]:
        """The record of the run beside ``main``'s own check, which says what the steps a candidate on ``main`` fails are
        (`vqueue_slim.based`), and when it started: the run beside the candidate ``main`` advanced to (the same tree), else
        one started here on ``main`` against itself, so lean-agreement never runs (at startup, each time ``main`` moves, and
        once more when it ended in infra).  Judged each tick, its record logged per ``main`` (to ``wrote`` too) whenever it
        changed, and never cancelled; past the window, read from the whole log (``whole``, where the snapshot read it).
        None with --dry-run, which starts nothing."""
        if self.slim is None or self.dry_run:
            return None, None
        if not past.slim(main)[0]:
            past = whole if whole is not past else Past.read(self.log_path, 0)
        (started, rec), (starts, since) = past.slim(main), past.slim_started(
            main, CHECK_GATE if self.gate == VERDICT_GATE else None)
        line = dict(sha=main, base=main, members=(), heads=())
        if self.gate == VERDICT_GATE and isinstance(rec, dict) and rec.get("gate") == self.slim.QUICK and not starts:
            self.write(got := self._started(n, main, origin, line, "started beside `main` after a landing on the quick tier: "
                                            "`main`'s own check, of every landing since it last passed"))
            wrote.append(got)
            return got["record"], self.now()
        if started and not self._other(rec):
            was = json.dumps(rec, sort_keys=True)
            v = self.slim.judge(self.world.slim_driver(), rec)
            if json.dumps(rec, sort_keys=True) != was:
                self.write(got := self._line(n, main, origin, "slim", **line, why=f"judged {v.state} beside `main`", record=rec,
                                             done=True))
                wrote.append(got)
            if v.state != INFRA or starts > 1:
                return rec, since
        self.write(got := self._started(n, main, origin, line, "started again beside `main`" if starts else
                                        "started beside `main`"))
        wrote.append(got)
        return got["record"], self.now()

    def _main_red(self, past: Past, whole: Past, n: int, main: str, origin: str, below: dict | None,
                  wrote: list[dict]) -> MainRed | None:
        """Where the land verdicts gate: `main`'s own check red, read on its first-parent line (`MAIN_WALK` commits) from
        ``main`` (``below``, its record as `_below` judged it) down to the newest commit whose check ended: each commit's
        check is the one `_below` started on it, or its candidate's whole check when that passed (a candidate's that failed
        and landed, its failures inherited, says nothing new); a commit's own check still running below ``main`` is judged
        here, its record logged per commit (to ``wrote`` too) whenever it changed.  Every landing gets its own check and none
        is cancelled; the newest ended one answers for every landing at or below it.  None when that one passed, or none
        ended; else its fail (or `main`'s own infra, once started again), the PRs landed since the newest that passed, the
        one it names (`vqueue.culprit`; none when it landed in more than one merge or the check ended in infra) with its
        revert when one is on `main` already, and, once a tick acted on it, that tick's line (`revert` or `fallback`, read
        from the whole log past the window)."""
        w = self.world
        recs = {r.get("sha"): r.get("record") for r in past.of("slim")}
        tip = passed = None
        for c in _out(w.repo, "rev-list", "--first-parent", f"-n{MAIN_WALK}", main).split():
            rec = below if c == main else recs.get(c)
            if not isinstance(rec, dict) or rec.get("gate") != CHECK_GATE:
                continue
            own = rec.get("tip") == rec.get("base")
            if "verdict" not in rec and own and c != main and tip is None:
                was = json.dumps(rec, sort_keys=True)
                v = self.slim.judge(w.slim_driver(), rec)
                if json.dumps(rec, sort_keys=True) != was:
                    self.write(got := self._line(n, main, origin, "slim", sha=c, base=c, why=f"judged {v.state} beside `main`'s "
                                                 "line, below its tip", record=rec, done=True))
                    wrote.append(got)
            state = (rec.get("verdict") or {}).get("state")
            if state == PASS:
                passed = c
                break
            if tip is None and own and (state == FAIL or state == INFRA and c == main):
                tip, red = c, rec
        if tip is None:
            return None
        v = self.slim.judge(w.slim_driver(), red)
        acted = [r for r in past.of("revert") + past.of("fallback") if r.get("red") == tip and (r.get("done") or r.get("shadow"))]
        if not acted:
            acted = [r for r in Past.read(self.log_path, 0).lines if r.get("action") in ("revert", "fallback")
                     and r.get("red") == tip and (r.get("done") or r.get("shadow"))]
        if acted:
            done = acted[-1]
            held = frozenset(done.get("holds") or ()) if done.get("action") == "revert" else frozenset()
            return MainRed(tip, v, passed, why=done.get("why") or "", holds=held, done=done,
                           culprit=Member(done["number"], (done.get("heads") or [""])[0], "", done.get("merge") or "")
                           if done.get("action") == "revert" else None)
        landed = tuple(w.landed(passed, tip))
        if v.state == INFRA:
            return MainRed(tip, v, passed, landed, why="it ended in infra, after it was started again: it names no PR")
        number, why = vqueue.culprit(v.failed, {m.pr: frozenset().union(*(f for k, f in landed if k.pr == m.pr))
                                                for m, _ in landed})
        merges = [m for m, _ in landed if m.pr == number]
        if number is not None and len(merges) > 1:
            number, why = None, f"#{number} landed in {len(merges)} merges since"
        culprit = merges[0] if number is not None else None
        holds = frozenset().union(*(f.reads or () for f in v.failed)) if culprit is not None else frozenset()
        reverted = w.reverts(tip, main).get(culprit.commit, "") if culprit is not None else ""
        return MainRed(tip, v, passed, landed, culprit, why, holds, reverted=reverted)

    def _beside(self, past: Past, n: int, main: str, origin: str, sha: str, base: str, prs: tuple[int, ...],
                heads: tuple[str, ...], wrote: list[dict], chained: bool,
                below: tuple[dict | None, float | None] = (None, None), lent: Lent | None = None,
                on_main: dict | None = None) -> Judged:
        """Candidate ``sha``'s verdict: its check's, the run beside it (`vqueue_slim`), which its build starts, and here, once,
        one on ``main``'s chain (``chained``) the log holds no start of (its build's tick stopped first); again by a retry of
        its infra or a rerun of its failure.  Judged each tick, its record logged (to ``wrote`` too) whenever it changed; with
        --dry-run neither started nor judged, so pending.  Each step it failed is compared with the run beside its base's
        check (``below``: its record and start, `_below` for ``main``, a base candidate's as judged this tick), which
        `vqueue.decide` takes as inherited when that run failed it too, a base candidate's still running by what it failed so
        far (`vqueue_slim.based`); a fail whose every other failure is inherited waits (pending) while the run beside
        ``main``, started at most `pending_s` ago, hasn't said, and one on a candidate stays a fail, which `vqueue.decide`
        holds until its base's says, or, where only its own members meet the step, until the run beside ``main``'s check says
        (``on_main``, its record, read as a base's into `vqueue.Failed.main_tests`).  A run that isn't the check that gates it
        (`_other`) counts as none started, and a check that took lean-audit's verdict from the check of a candidate below
        reads lean-audit as pending while that candidate is in flight, and, once it has landed, takes the verdict that the
        record ``lent`` gives keeps (`vqueue_slim.audited`, `_lent`); one that left suites to its base's check passes only
        once that check passed them (`vqueue_slim.borrowed`, on ``below``), and is started again, its runs cancelled, once
        its base has landed on a check that didn't (`_unanswered`)."""
        started, rec = past.slim(sha)
        line = dict(sha=sha, base=base, members=prs, heads=heads)
        started = started and not self._other(rec, line)
        if self.dry_run or not started and not chained:
            v = Verdict(PENDING)
        else:
            redo, stopped = "" if not started or base != main else self._unanswered(rec, below[0]), {}
            if redo and "verdict" not in rec and not self.shadow and (ids := list(rec.get("ids") or ())):
                failed = self.world.cancel_runs(ids, f"vqueue: candidate {sha[:12]}'s check is {redo}")
                stopped = {"beside_cancelled": ids, **({"beside_cancel_failed": failed} if failed else {})}
            if not started or redo:
                self.write(got := self._started(n, main, origin, {**line, **stopped}, *([redo] if redo else []),
                                                below=below[0] if base != main else None, lent=lent))
                wrote.append(got)
                rec = got["record"]
            was = json.dumps(rec, sort_keys=True)
            v = self.slim.judge(self.world.slim_driver(), rec)
            ran = (rec or {}).get("lean_audit_from")
            source, landed = lent(ran) if ran and lent else (None, False)
            v = self.slim.audited(self.world.slim_driver(), v, rec, source if landed else None)
            v = self.slim.borrowed(v, rec, below[0])
            if json.dumps(rec, sort_keys=True) != was:
                self.write(got := self._line(n, main, origin, "slim", **line, why=f"judged {v.state}", record=rec, done=True))
                wrote.append(got)
        v = self.slim.based(v, below[0], so_far=base != main)
        if base != main and on_main is not None and v.failed:
            unsaid = replace(v, failed=tuple(replace(f, base_tests=None) for f in v.failed))
            v = replace(v, failed=tuple(replace(f, main_tests=m.base_tests)
                                        for f, m in zip(v.failed, self.slim.based(unsaid, on_main).failed)))
        if v.state == FAIL and base == main and (waits := self._waits(v, *below)):
            return Judged(Verdict(PENDING), slim=v, record=rec,
                          note=f"waiting for the run beside {base[:12]}'s check: {', '.join(waits)}")
        return Judged(v, slim=v, record=rec)

    def _other(self, rec: dict | None, line: dict | None = None) -> bool:
        """Whether ``rec`` records a run that doesn't gate the candidate of ``line`` (`main`, without one): one whose record
        names no check (a run beside another gate's), or, where the land verdicts gate, a quick tier where the candidate isn't
        plain now (`_quick`): the check starts in its place.  A check started where a quick tier would do stands."""
        if rec is None or rec.get("gate") == CHECK_GATE:
            return False
        return self.gate == CHECK_GATE or rec.get("gate") != self.slim.QUICK or not self._quick(line or {})

    def _plain(self, ready: list[Ready], main: str) -> dict[int, str]:
        """The plain PRs of ``ready``, by number at their heads: each with a land verdict of its own at its head
        (`World.reviewed`), no Lean change, no Lean input of the tree changed and no move, and its change meeting nothing
        drive.toml at ``main`` names in `lowrisk_excludes` but not in `verdict_plain`, or in `verdict_excludes`
        (`DriveFlags.plain_excludes`, read as `queue.lowrisk` reads the low-risk lane's list: what that lane keeps off but
        for what only a reader needs to see, and what this gate adds to it; with `lowrisk_excludes` or `verdict_excludes`
        empty, none is plain)."""
        excludes = flags_at(self.world.repo, main).plain_excludes
        return {x.number: x.head for x in ready if x.number in self.world.reviewed and not (x.lean or x.lean_input or x.move)
                and lowrisk(x.files, excludes)}

    def _quick(self, line: dict) -> bool:
        """Whether the candidate of ``line`` is plain, so that the quick tier on its own merge gates it in place of the check
        (`vqueue_slim.start_quick`): where the land verdicts gate and `main`'s own check isn't red with no PR named
        (`MainRed.culprit`), each member is plain (`_plain`) at the head it was built at."""
        members = list(zip(line.get("members") or (), line.get("heads") or ()))
        return (self.gate == VERDICT_GATE and bool(members) and not (self.red is not None and self.red.culprit is None)
                and all(self.plain.get(m) == h for m, h in members))

    def _waits(self, v: Verdict, rec: dict | None, since: float | None) -> list[str]:
        """The failed steps of ``v`` (a candidate's check) whose fail waits: those the run beside its base's check (``rec``,
        started at ``since``) hasn't said of while it runs, up to `pending_s`, when every other failure is inherited; none
        else."""
        unknown = [f for f in v.failed if f.base_tests is None]
        if not unknown or rec is None or "verdict" in rec or since is None or self.now() - since > self.pending_s:
            return []
        if not all(f in unknown or vqueue._inherited(f) for f in v.failed):
            return []
        return [f.task for f in unknown]

    # -- a tick

    def tick(self) -> list[dict]:
        """One tick: the records it logged (with --dry-run, those it would have), and a line of how long each of its steps
        took and the peak resident size after it (`Steps`), printed however it ends."""
        self.steps = Steps(self.clock, self.peak)
        try:
            return self._tick()
        finally:
            self.timing(self.steps.line(self.last))

    def _tick(self) -> list[dict]:
        steps = self.steps
        past = Past.read(self.log_path, self.now() - WINDOW_S)
        self.last = max(past.tick, self.last) + 1
        steps.lap("log")
        t = self.snapshot(past, self.last)
        if self.dry_run:
            self.out(self.summary(t))
        actions = list(self.decide(t.snap))
        steps.lap("decide")
        if self.dry_run:
            self.out("\n".join(f"would {_name(a)}: {a}" for a in actions) or "nothing to do")
            return []
        red = self._red_main(t) if self.gate == VERDICT_GATE and self.red is not None and self.red.done is None else []
        for rec in red:
            self.write(rec)
        if any(r.get("action") == "revert" and r.get("pushed") for r in red):
            self.out(f"research queue vqueue: tick {t.n}: `main` moved to the revert {red[-1]['sha'][:12]} of "
                     f"#{red[-1]['number']}: the tick stops, and the next derives everything again")
            return [*t.wrote, *red]
        new = [self.record(t, "ignore", sha=sha, ref=name, why=f"origin's queue/{name} is no build of the loop's: no candidate")
               for name, sha in t.foreign.items() if (name, sha) not in past.ignored()] + self._carried(t) \
            + self._unauthenticated(t)
        new += ((self._off_gate(t) if self.slim is not None else []) + self._conflicts(t) + self._lacking(t)
                + self._behind_held(t) + self._riding(t) + self._launched(t))
        steps.lap("notices")
        new += self._moved(t)
        steps.lap("on_main")
        for rec in new:
            self.write(rec)
        logged = [*t.wrote, *red, *new]
        stopped = False
        for a in actions:
            try:
                recs = self.apply(a, t)
            except (QueueError, OSError, subprocess.SubprocessError) as e:
                rec = self.record(t, _name(a), sha=getattr(a, "sha", None), base=getattr(a, "base", None),
                                  members=getattr(a, "members", ()), number=getattr(a, "number", None),
                                  why=getattr(a, "why", ""), error=str(e)[-600:])
                self.write(rec)
                logged.append(rec)
                self.out(f"research queue vqueue: tick {t.n}: {_name(a)} failed, the tick stops: {e}")
                stopped = True
                break
            finally:
                steps.lap(_name(a))
            for rec in recs:
                self.write(rec)
            logged += recs
            if failed := next((r["error"] for r in recs if r.get("error")), None):
                self.out(f"research queue vqueue: tick {t.n}: {_name(a)} failed, the tick stops: {failed}")
                stopped = True
                break
            for adv in [r for r in recs if r.get("action") == "advance" and r.get("on_main")]:
                for rec in self._landing(t, adv, []):
                    self.write(rec)
                    logged.append(rec)
                steps.lap("on_main")
        for sha, j in t.judged.items():
            if j.note:
                self.out(f"research queue vqueue: tick {t.n}: {sha[:12]} {j.verdict.state}: {j.note}")
        if past.bad:
            self.out(f"research queue vqueue: tick {t.n}: {len(past.bad)} line(s) of {self.log_path} read as none: {past.bad[0]}")
        self.keep(t)
        steps.lap("keep")
        try:
            self.world.q.work_out_mapped()
        except (QueueError, OSError, subprocess.SubprocessError) as e:
            self.out(f"research queue vqueue: tick {t.n}: mapping old heads across `main`'s moves failed: {e}")
        steps.lap("mapped")
        if not stopped and self.left_out is not None:       # last: a failure here skips none of the tick's bookkeeping
            for rec in self._leaves(t):
                self.write(rec)
                logged.append(rec)
            steps.lap("leave")
        return logged

    def keep(self, t: Tick) -> None:
        """The state file (``state``): this tick, and each candidate's verdict, its check's and its records run's."""
        if self.state is None:
            return
        records = {c.sha: c.records.state for c in t.snap.candidates if c.records is not None}
        doc = {"at": stamp(self.now()), "tick": t.n, "main": t.main, "origin_main": t.origin,
               "candidates": {sha: {"mine": j.verdict.state, "note": j.note,
                                    "beside": j.slim.state if j.slim is not None else None, "records": records.get(sha)}
                              for sha, j in t.judged.items()},
               "ignored": t.foreign}
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.state)

    def run(self, interval: float, sleep: Callable[[float], None] = time.sleep) -> int:
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 -- the loop goes on: the next tick derives everything again
                self.out(f"research queue vqueue: the tick failed ({type(e).__name__}), the next derives everything again: {e}")
            sleep(interval)

    def summary(self, t: Tick) -> str:
        s = t.snap
        lines = [f"tick {t.n}: main {s.main[:12]}" + (f" (origin's {t.origin[:12]})" if t.origin != s.main else ""),
                 f"ready {len(s.ready)} (rank order; ! doesn't merge onto main, L a Lean change, W a wide one, P pinned): "
                 + " ".join(f"#{x.number}" + ("!" if x.conflict else "") + ("L" if x.lean else "")
                            + ("W" if vqueue._wide(x, s.wide) else "") + ("P" if x.number in s.pinned else "")
                            for x in s.ready)]
        for c in s.candidates:
            j = t.judged[c.sha]
            lines.append(f"candidate {c.sha[:12]} on {c.base[:12]}: {' '.join(f'#{n}' for n in c.members)} {c.verdict.state}"
                         f" tries {c.tries} reruns {c.reruns}" + (f" (beside: {j.slim.state})" if j.slim else "")
                         + (f" (records run: {c.records.state})" if c.records else "")
                         + (f" (the records commit of {c.repairs[:12]})" if c.repairs else "")
                         + (f" ({j.note})" if j.note else "")
                         + "".join(f"\n  failed {f.task} {f.key[:19]}" for f in c.verdict.failed)
                         + "".join(f"\n  infra {f.task} {f.key[:19]}" for f in c.verdict.infra))
        lines.append(f"history {len(s.history)}: " + " ".join(o.state for o in s.history[-10:]))
        lines += [c["why"] for _, c in sorted(t.carried.items())]
        lines += [f"ignored queue/{name} at {sha[:12]}: no build of the loop's" for name, sha in t.foreign.items()]
        return "\n".join(lines)

    # -- the actions

    def record(self, t: Tick, action: str, **kw) -> dict:
        return self._line(t.n, t.main, t.origin, action, **kw)

    def _line(self, n: int, main: str, origin: str, action: str, *, sha: str | None = None, base: str | None = None,
              members: Iterable[int] = (), heads: Iterable[str] = (), priority: int | None = None, number: int | None = None,
              why: str = "", outcome: Outcome | None = None, failed: Iterable[Failed] = (), **extra) -> dict:
        return {"at": stamp(self.now()), "tick": n, "shadow": self.shadow, "main": main, "action": action, "sha": sha,
                "base": base, "members": list(members), "heads": list(heads), "priority": priority, "number": number,
                "why": why, "outcome": outcome and {"members": list(outcome.members), "heads": list(outcome.heads),
                                                    "state": outcome.state, "suspects": list(outcome.suspects),
                                                    **({"base": outcome.base} if outcome.base else {}),
                                                    **({"inherited": [_failed_doc(f) for f in outcome.inherited]}
                                                       if outcome.inherited else {})},
                "failed": [{"task": f.task, "key": f.key, **({"tests": sorted(f.tests)} if f.tests else {})} for f in failed],
                **({"origin_main": origin} if origin != main else {}), **extra}

    def write(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab+") as f:
            end = f.seek(0, 2)
            torn = end > 0 and (f.seek(end - 1), f.read(1))[1] != b"\n"     # a crash's torn line stays a line of its own
            f.write((("\n" if torn else "") + json.dumps(rec, sort_keys=True) + "\n").encode())

    def _leaves(self, t: Tick) -> list[dict]:
        """A `leave` line for each PR marked ready that this tick's candidates leave out: of those `World.ready` admits and
        takes no further (`World.held_out`), of those held for a red of `main`'s check (`held`), then of the PRs it took
        (`left_out`), each (PR, head, cause) once."""
        seen = {(r.get("number"), h, r.get("cause")) for r in t.past.of("leave") for h in r.get("heads") or ()}
        out = []
        for n, head, cause, why in [*((n, *x) for n, x in sorted(self.world.held_out.items())),
                                    *((n, *x) for n, x in sorted(self.held.items())),
                                    *((x.number, x.head, x.cause, x.why) for x in self.left_out(t.snap))]:
            if (n, head, cause) not in seen:
                seen.add((n, head, cause))
                out.append(self.record(t, "leave", number=n, heads=[head], why=why, cause=cause))
        return out

    def apply(self, a: Action, t: Tick) -> list[dict]:
        if isinstance(a, Build):
            return self._build(a, t)
        if isinstance(a, Wait):
            return [] if any(r.get("why") == a.why for r in t.past.of("wait")) else [self.record(t, "wait", why=a.why)]
        c = next((c for c in t.snap.candidates if c.sha == getattr(a, "sha", None)), None)
        if isinstance(a, Eject):
            return [self._eject(a, t)]
        if c is None:
            raise QueueError(f"{a}: no such candidate")
        if isinstance(a, Advance):
            return [self._advance(c, t)]
        if isinstance(a, Drop):
            return [self._drop(a, c, t), *self._reached(a, c, t)]
        if isinstance(a, Retry):
            return self._retry(c, t)
        if isinstance(a, Rerun):
            return self._rerun(a, c, t)
        if isinstance(a, Ring):
            return self._ring(a, c, t)
        if isinstance(a, Red):
            return [self._red(a, c, t)]
        if isinstance(a, Regenerate):
            return self._regenerate(t, {"sha": c.sha, "base": c.base, "members": list(c.members), "heads": list(c.heads)},
                                    "started again" if c.records is not None else "started")
        if isinstance(a, Commit):
            return self._commit(a, c, t)
        raise QueueError(f"{a}: no such action")

    def _build(self, a: Build, t: Tick) -> list[dict]:
        w, base = self.world, t.rebuilds.get(a.base, a.base)
        if a.after is not None:
            if a.after not in t.committed:                  # its Commit put nothing on the chain: the next tick decides
                return []
            base = t.committed[a.after][0]
        if self.slim is not None or a.records:
            w.slim_driver()                                 # before anything is pushed
        again = {"replaces": a.replaces} if a.replaces is not None else {}
        tip, merged, recs = self._merged(t, a.members, base)
        if not merged:
            if a.replaces is not None:
                t.rebuilds[a.replaces] = base
            recs.append(self.record(t, "build", base=base, priority=a.priority, why="no member merges onto its base", **again))
            return recs
        tip, adopted = self._pushed(t, tip, base, merged)
        if a.replaces is not None:
            t.rebuilds[a.replaces] = tip
        line = {"sha": tip, "base": base, "members": [m.pr for m in merged], "heads": [m.head for m in merged],
                "priority": a.priority, **({"adopted": True} if adopted else {}), **again, **self._moves(merged)}
        if a.records:
            return [*recs, *self._staged(t, line, "its records run first: its check waits for it")]
        return self._enqueue(t, recs, line)

    def _merged(self, t: Tick, members: Iterable[int], base: str) -> tuple[str, list[Member], list[dict]]:
        """PRs ``members`` merged in order onto ``base`` (`World.build`): (the tip, the members merged, the skip records of
        those left out, each owner told once a head and base)."""
        w = self.world
        if unknown := [n for n in members if n not in w.prs]:
            raise QueueError(f"build: {', '.join(f'#{n}' for n in unknown)} not ready")
        tip, merged, left = w.build([w.prs[n] for n in members], base, t.advanced or t.main)
        recs = []
        for s in left:
            p, key = s.pr, f"{s.pr.head}@{base}"
            line = dict(base=base, members=[p.number], heads=[p.head], number=p.number)
            if s.held or s.waits:                           # it reaches `main` with what holds it, or is built once that has
                recs.append(self.record(t, "skip", **line, why=s.why, **({"held": True} if s.held else {"waits": True}),
                                        done=False))
                continue
            if key not in t.told and (now := self._landed(t, p.head)):
                recs.append(self.record(t, "skip", **line, why=f"{s.why}; `main` {now[:12]}, fetched again, holds its head",
                                        landed=now, done=False))
                continue
            names = ", ".join(f"#{n}" for n in s.by)
            fix = (f"once {names} {'is' if len(s.by) == 1 else 'are'} on it, " if s.by else "") + merge_main("it", p.branch)
            text = None if key in t.told else (
                f"#{p.number} at {p.head[:12]} isn't built on the verified queue's candidate on {base[:12]}: {s.why}. {s.fix}"
                if s.fix else
                f"#{p.number} at {p.head[:12]} doesn't merge onto the verified queue's candidate on {base[:12]}: {s.why}. "
                f"It is left out until it merges: {fix}")
            posts = text is not None and not self.shadow
            done = posts and self._say(t, "skip", text, self.owner(p.number), base=base, heads=[p.head], number=p.number)
            if done or (text and self.shadow):
                t.told.add(key)
            recs.append(self.record(t, "skip", **line, why=s.why, text=text, done=done, **({"by": list(s.by)} if s.by else {}),
                                    **({"posted": done} if posts else {})))
        return tip, merged, recs

    def _pushed(self, t: Tick, tip: str, base: str, merged: list[Member]) -> tuple[str, bool]:
        """Build ``tip`` of ``merged`` onto ``base`` pushed, or one pushed before by a build whose line the log never got
        adopted in its place (`World.rebuilt`): (the candidate, whether it was adopted)."""
        w = self.world
        if adopted := next((sha for name, sha in sorted(t.foreign.items())
                            if name == sha[:12] and w.rebuilt(sha, tip, base, merged)), None):
            return adopted, True
        w.push(tip, branch(tip))
        return tip, False

    def _moves(self, merged: list[Member]) -> dict:
        """The ``moves`` of a line of ``merged``: each regenerated move's certificate (`World.certified`), which it lands on."""
        moves = [{"pr": m.pr, **self.world.certified[m.pr]} for m in merged if m.pr in self.world.certified]
        return {"moves": moves} if moves else {}

    def _staged(self, t: Tick, line: dict, why: str) -> list[dict]:
        """Pushed build ``line`` staged off main's chain: its build line logged with ``records_first``, its records run
        started and its check not, so the planner builds nothing on it (`vqueue.decide`)."""
        return [self.record(t, "build", **line, why=why, enqueued=False, records_first=True, done=True),
                *self._regenerate(t, line)]

    def _enqueue(self, t: Tick, recs: list[dict], line: dict) -> list[dict]:
        """Start pushed candidate ``line["sha"]``'s check: its build line (``line``) logged after ``recs``, enqueued (which
        `Past.tries` counts), and the run beside its check started.  Every check starts here: a build's, and the check of one
        whose records run passed (`_commit`)."""
        return [*recs, self.record(t, "build", **line, enqueued=True, done=True),
                *self._start(t, line["sha"], line["base"], {"members": line["members"], "heads": line["heads"]})]

    def _onto(self, t: Tick, c: Candidate, recs: list[dict], line: dict) -> list[dict]:
        """`_enqueue` of ``line``, the candidate `Commit` puts on main's chain for staged batch ``c``, kept with the record of
        the run beside its check as started for the builds of this tick that go on it (`Tick.committed`)."""
        out = self._enqueue(t, recs, line)
        t.committed[c.sha] = (line["sha"], next((r.get("record") for r in out if r["action"] == "slim"), None))
        return out

    def _regenerate(self, t: Tick, line: dict, why: str = "started") -> list[dict]:
        """Start the records run of candidate ``line["sha"]`` (`vqueue_records.start`), logged with its record; one that fails
        to start is logged with no record, which `vqueue_records.judge` reads as infra."""
        if self.records is None:
            raise QueueError(f"{line['sha'][:12]}: no records run starts without research.vqueue_records")
        d = self.world.slim_driver()                       # drive.toml naming no machines stops the tick
        heavy = self._heavy_on(line)
        try:
            with self.steps("launch"):
                rec, error = self.records.start(d, line["sha"], **({"on": heavy} if heavy else {})), None
        except (QueueError, ValueError, OSError, subprocess.SubprocessError) as e:
            rec, error = None, f"{type(e).__name__}: {e}"[-600:]
        return [self.record(t, "regenerate", sha=line["sha"], base=line["base"], members=line["members"], heads=line["heads"],
                            why=why, record=rec, done=True, **({"start_error": error} if error else {}))]

    def _commit(self, a: Commit, c: Candidate, t: Tick) -> list[dict]:
        """Put staged batch ``c``, whose records run passed, onto main's chain at ``a.base``, its top, and start its check.

        On ``c``'s own base: on ``c`` as built when the run regenerated nothing (`vqueue_records.commit` makes no commit),
        else on the records commit of what it regenerated on top of ``c``, pushed (one pushed before, with no line in the
        log, is adopted: `World.recommitted`) and checked in ``c``'s place, a candidate on ``c``'s base with ``c``'s members,
        which it `repairs`: the next tick drops ``c`` for it (`vqueue.decide`), so a tick cut off after the push still finds
        ``c`` and adopts it.  On another base, ``c``'s members are merged again onto it; when every member merges and the
        merge's Lean inputs are ``c``'s (`vqueue_records.same_inputs`), the records commit made on ``c`` is carried onto the
        merge and checked in ``c``'s place as above (the merge itself when there is none); else the merge is staged again
        in ``c``'s place, its records run first; and when no member merges, ``c`` is dropped and its members built again.
        The bookkeeping its records run repaired is told to `lean` and each owner a warning names (`_bookkeeping`).  A
        records commit that can't be made, or that the merge gate won't vouch for, is held: logged, and rung to `CI` once,
        and ``c`` is dropped once it is held `pending_s` (`vqueue.decide`).  The candidate whose check it starts is kept
        for the builds the tick makes on it (`_onto`)."""
        w = self.world
        rid = ((t.records.get(c.sha) or {}).get("verdict") or {}).get("run")
        line = {"sha": c.sha, "base": c.base, "members": list(c.members), "heads": list(c.heads), "priority": a.priority,
                "run": rid}
        recs, onto, merged = [], None, []
        if a.base != c.base:
            tip, merged, recs = self._merged(t, c.members, a.base)
            if not merged:
                why = f"no member of it merges onto {a.base[:12]}, main's chain's top: its members are built again"
                return [*recs, self._drop(Drop(c.sha, why), c, t)]
            moved = {"base": a.base, "members": [m.pr for m in merged], "heads": [m.head for m in merged], "repairs": c.sha,
                     **self._moves(merged)}
            if moved["members"] != list(c.members) or not self.records.same_inputs(w.repo, c.sha, tip):
                tip, adopted = self._pushed(t, tip, a.base, merged)
                differ = "not every member merges onto it" if moved["members"] != list(c.members) else \
                    "its Lean inputs aren't that tree's"
                return [*recs, *self._staged(t, {**line, "sha": tip, **moved, **({"adopted": True} if adopted else {})},
                                             f"{c.sha[:12]}'s records were regenerated on {c.base[:12]}, and {differ}: "
                                             "staged again, its records run first")]
            onto, line = tip, {**line, **moved}
        got = self.records.commit(w.slim_driver(), c.sha, rid, c.members, **({"onto": onto} if onto else {})) \
            if self.records is not None and rid else "its records run passed, but its record names no run"
        if got is None and onto is None:
            return self._onto(t, c, [], {**line, "why": f"its records run {rid} regenerated nothing: checked as built"})
        if got is None:
            tip, adopted = self._pushed(t, onto, a.base, merged)
            return self._onto(t, c, recs, {**line, "sha": tip, **({"adopted": True} if adopted else {}),
                                           "why": f"its records run {rid} regenerated nothing, and {c.sha[:12]}'s Lean inputs "
                                                  "are its: checked as built"})
        if isinstance(got, str):
            bound = (f"; it is dropped, its members built again, once it is held {self.pending_s / 3600:g} h after it was "
                     "first")
            return [*recs, self.record(t, "commit", **{**line, "sha": c.sha}, why=got, done=False),
                    *self._ring(Ring(c.sha, f"{c.sha[:12]}'s records commit is held: {got}{bound}", to=(CI,)), c, t)]
        tip, files, told = got
        if adopted := next((sha for name, sha in sorted(t.foreign.items())
                            if name == sha[:12] and w.recommitted(sha, tip, onto or c.sha, a.base, merged)), None):
            tip = adopted
        else:
            w.push(tip, branch(tip))
        why = f"{c.sha[:12]}'s records, which {rid} regenerated, committed on it" if onto is None else \
            f"{c.sha[:12]}'s records, which {rid} regenerated, carried onto its members merged again onto {a.base[:12]}"
        return self._onto(t, c, [*recs, *self._bookkeeping(t, c, tip, told)],
                          {**line, "sha": tip, "why": why, "repairs": c.sha, "files": files,
                           **({"adopted": True} if adopted else {})})

    def _bookkeeping(self, t: Tick, c: Candidate, tip: str, told: tuple[str, tuple[str, ...]] | None) -> list[dict]:
        """Tell the bookkeeping ``c``'s records run repaired in its records commit ``tip`` (``told``:
        `vqueue_records.bookkeeping_told`'s text and handles), once a records commit: logged, and only logged with --shadow."""
        if told is None or any(r.get("posting") == "bookkeeping" and r.get("sha") == tip for r in t.past.of("posting")):
            return []
        text, to = told[0], tuple(f"@{h}" for h in told[1])
        line = dict(sha=tip, base=c.base, members=c.members, heads=c.heads, why=f"{tip[:12]}'s records run repaired bookkeeping",
                    text=text, to=list(to))
        if self.shadow:
            return [self.record(t, "bookkeeping", **line, done=False)]
        return [self.record(t, "bookkeeping", **line, posted=self._say(t, "bookkeeping", text, to, sha=tip), done=True)]

    def _start(self, t: Tick, sha: str, base: str, line: dict, why: str = "started", avoid: Iterable[str] = ()) -> list[dict]:
        """The run beside candidate ``sha``'s check started (`vqueue_slim.start`, off the machines ``avoid`` names where it
        has others), its record logged; a start that fails is
        logged with no record, which `vqueue_slim.judge` reads as infra.  A base still in flight lends the record of its
        check, as judged this tick or, put on the chain by a `Commit` of this tick, as started (`Tick.committed`), and the
        candidates below it theirs, or, landed, the run beside `main` (`_audit_from`)."""
        if self.slim is None:
            return []
        began = dict(t.committed.values())
        below = began[base] if base in began else \
            t.judged[base].record if base not in (t.main, t.advanced) and base in t.judged else None
        return [self._started(t.n, t.main, t.origin, {**line, "sha": sha, "base": base}, why, below,
                              _lent(t.judged, t.below, t.main, t.advanced), avoid)]

    def _started(self, n: int, main: str, origin: str, line: dict, why: str = "started", below: dict | None = None,
                 lent: Lent | None = None, avoid: Iterable[str] = ()) -> dict:
        d = self.world.slim_driver()                       # drive.toml naming no machines stops the tick: no run is infra
        if self._quick(line):
            try:
                with self.steps("launch"):
                    rec, error = self.slim.start_quick(d, line["sha"], line["base"], *([tuple(avoid)] if avoid else [])), None
            except (QueueError, ValueError, OSError, subprocess.SubprocessError) as e:
                rec, error = None, f"{type(e).__name__}: {e}"[-600:]
            return self._line(n, main, origin, "slim", **line, why=f"{why}: the quick tier, its members' land verdicts gating it",
                              record=rec, done=True, **({"start_error": error} if error else {}))
        heavy = self._heavy_on(line)
        try:
            audit = self._audit_from(d, main, line, below, lent)
            suites = self._suites_from(d, main, line, below, lent)
            after = self.slim.replaced(Past.read(self.log_path, self.now() - self.slim.CARRY_WAIT_S).lines,
                                       line.get("members") or (), self.now())
            with self.steps("launch"):
                rec, error = self.slim.start(d, line["sha"], line["base"],
                                             shards=self.shards,
                                             **({"audit_from": audit} if audit else {}),
                                             **({"suites_from": suites} if suites else {}),
                                             **({"deselect": f} if (f := getattr(self.world, "deselect_file", None)) else {}),
                                             **({"after": after} if after else {}),
                                             **({"heavy_on": heavy} if heavy else {}),
                                             **({"avoid": tuple(avoid)} if avoid else {})), None
        except (QueueError, ValueError, OSError, subprocess.SubprocessError) as e:
            rec, error = None, f"{type(e).__name__}: {e}"[-600:]
        return self._line(n, main, origin, "slim", **line, why=why, record=rec, done=True,
                          **({"start_error": error} if error else {}))

    def _heavy_on(self, line: dict) -> str:
        """The machines the heavy runs of the candidate of ``line`` take, `World.heavy_on`, when a member of it is a heavy PR or
        the heavy file names every candidate (`World.heavy`, read at each launch), `main`'s own check then too; "" else, and
        when no machines are named."""
        if not (on := getattr(self.world, "heavy_on", "")):
            return ""
        prs, every = self.world.heavy()
        return on if every or set(line.get("members") or ()) & prs else ""

    def _audit_from(self, d: Landing, main: str, line: dict, below: dict | None, lent: Lent | None) -> str | None:
        """The candidate whose check's verdict on lean-audit the check of ``line`` takes in place of running it
        (`vqueue_slim.start`'s ``audit_from``), where its base is a candidate in flight, not `main`
        (``below``, its record), when no member's own change, against its merge base with ``main`` (`World._changed`, as
        `World._judge` reads it), meets a Lean input of the candidate's tree with the paths its members change in it, so that
        one a member deletes is judged as a file there (`vqueue_slim.lean_inputs`): that base, whose check is judged on
        lean-audit (`vqueue_slim.audits`), else the candidate its base's check took it from (``lean_audit_from``), the one
        whose check ran it, while that check is judged on it, or, once that candidate has landed, while the run beside `main`
        keeps a verdict on it (`vqueue_slim.answers`; where each is read: ``lent``, `_lent`).  None else, and when the Lean
        inputs can't be read (git can't list that tree or say what a member changes, or the audit's declaration doesn't
        load): the check audits."""
        if below is None or lent is None or not line.get("heads"):
            return None
        if self.slim.audits(d, below):
            ran = line["base"]
        else:
            rec, landed = lent(ran) if (ran := below.get("lean_audit_from")) else (None, False)
            if not ran or not (self.slim.answers(rec) if landed else self.slim.audits(d, rec)):
                return None
        try:
            changed = frozenset().union(*(self.world._changed(main, h) for h in line["heads"]))
            if vqueue.meets(changed, self.slim.lean_inputs({*self.world._tree(line["sha"]), *changed})):
                return None
        except QueueError as e:
            # safe: a check that audits itself never takes a verdict it shouldn't, it only costs the audit
            d.q.log(f"research queue vqueue: candidate {line['sha'][:12]}'s check runs lean-audit itself, not taking "
                    f"{ran[:12]}'s: its Lean inputs can't be read: {e}")
            return None
        return ran

    def _suites_from(self, d: Landing, main: str, line: dict, below: dict | None, lent: Lent | None) -> str | None:
        """The candidate whose check answers for the suites of the check of ``line`` that are keyed there as here
        (`vqueue_slim.start`'s ``suites_from``): its base, where the check gates and the base is a candidate in flight
        (``below``, its record) whose check lends them (`vqueue_slim.lends_suites`, on the record of the check of ``main``,
        ``lent``).  None else: the check runs every suite it has no pass of."""
        if self.gate != CHECK_GATE or below is None or lent is None:
            return None
        return line["base"] if self.slim.lends_suites(d, below, lent(main)[0]) else None

    def _unanswered(self, rec: dict | None, on_main: dict | None) -> str:
        """Why the check ``rec`` records, of a candidate now on `main`, is started again: it left suites to the check of
        the candidate it was built on (``suites_from``), which has landed, and the record of `main`'s check (``on_main``:
        that check's, or one started on `main` since) keeps a verdict that didn't pass them, or isn't the check that gates
        (`vqueue_slim.answers_suites`), so nothing will have run them; "" else.  Started again on `main`, it leaves none
        out."""
        if not rec or not (took := rec.get("suites_from")) or self.slim.answers_suites(on_main) is not False:
            return ""
        return f"started again: {took[:12]}'s check, which it left suites to, landed without passing them: it runs them itself"

    def _advance(self, c: Candidate, t: Tick) -> dict:
        """Push `main` to ``c`` (with --shadow, the shadow's own `main`).  Logged as soon as it is pushed, before what follows
        its members reaching `main` (`_on_main`), a step of its own the tick runs next."""
        w = self.world
        if self.shadow:
            w.set_shadow_main(c.sha)
        else:
            w.advance(c.sha, c.base)
        t.advanced = c.sha
        return self.record(t, "advance", sha=c.sha, base=c.base, members=c.members, heads=c.heads,
                           why=f"`main` {c.base[:12]} fast-forwarded to it", done=not self.shadow,
                           **({} if self.shadow else {"on_main": True}))

    def _reached(self, a: Drop, c: Candidate, t: Tick) -> list[dict]:
        """The advance of ``c``, dropped as on `main` with a pass (`vqueue.ON_MAIN`), where no line logs one done and this tick
        made none: a push of `main` that took but raised (a timeout) logged only its error, so `Past.moved` never ran the
        steps after its members reached `main`, and no stacked PR's owner was told.  Logged done with `on_main`, as `_advance`
        logs one, so those steps run as an advance's do (`_landing`, then `Past.moved` until done).  Outside --shadow only."""
        if (self.shadow or a.why != vqueue.ON_MAIN or a.outcome is None or a.outcome.state != PASS or t.advanced == c.sha
                or any(r.get("done") for r in t.past.of("advance", c.sha))):
            return []
        return [self.record(t, "advance", sha=c.sha, base=c.base, members=c.members, heads=c.heads,
                            why="`main` holds it and no advance of it is logged done: the push that took it failed after",
                            done=True, on_main=True, reached=True)]

    def _on_main(self, t: Tick, adv: dict, told: list[str]) -> dict:
        """What follows advance ``adv``'s members reaching `main` (`World.on_main`), logged as its own step: one that fails
        is logged with its error, and the next tick runs it again while no record of it is done (`Past.moved`), ``told`` the
        stacked PRs' owners its tries told."""
        w, told = self.world, list(told)
        sha, base, prs, heads = adv["sha"], adv.get("base") or "", adv.get("members") or (), adv.get("heads") or ()
        line = dict(sha=sha, base=base, members=prs, heads=heads)
        try:
            w.on_main(sha, base, w.members(sha, base, prs, heads), told)
        except Exception as e:  # noqa: BLE001 -- `main` has moved: its bookkeeping is retried, and the tick goes on
            return self.record(t, "on_main", **line, why="drive's steps once its members are on `main`", told=told,
                               done=False, error=f"{type(e).__name__}: {e}"[-600:])
        return self.record(t, "on_main", **line, why="drive's steps once its members are on `main`", told=told, done=True)

    def _moved(self, t: Tick) -> list[dict]:
        """What follows `main` moving, for each advance no record says its steps are done of (`Past.moved`): run in the tick
        (`_on_main`), or, with ``after``, detached: each job that ended is logged (`_after_ended`), and while none runs one
        is started for the oldest advance with none (`_landing`), so they run one at a time and no tick waits on them."""
        moved = t.past.moved()
        if self.after is None:
            return [self._on_main(t, adv, told) for adv, told, _ in moved]
        recs: list[dict] = []
        for adv, told, job in moved:
            if job is not None:
                if (rec := self._after_ended(t, adv, told, job)) is None:
                    t.after_running = True
                else:
                    recs.append(rec)
        ended = {r["sha"] for r in recs}
        if todo := next(((adv, told) for adv, told, job in moved if job is None and adv["sha"] not in ended), None):
            recs += self._landing(t, *todo)
        return recs

    def _landing(self, t: Tick, adv: dict, told: list[str]) -> list[dict]:
        """The steps after advance ``adv``'s members reached `main`: run in the tick without ``after`` (`_on_main`); else, while
        no job of them runs, started detached (`World.start_after`, which runs `_after_landing`), logged as started with its
        job (``job``), which `Past.moved` reads back while that is their newest record.  One that fails to start is logged
        with its error, and the next tick starts it again."""
        if self.after is None:
            return [self._on_main(t, adv, told)]
        if t.after_running:
            return []
        w, why = self.world, "drive's steps once its members are on `main`"
        sha, base, prs, heads = adv["sha"], adv.get("base") or "", adv.get("members") or (), adv.get("heads") or ()
        line = dict(sha=sha, base=base, members=prs, heads=heads)
        try:
            merged = [asdict(m) for m in w.members(sha, base, prs, heads)]
            job = w.start_after({**line, "merged": merged, "told": list(told)}, self.after, self.log_path.parent / AFTER)
        except (QueueError, OSError, subprocess.SubprocessError) as e:
            return [self.record(t, "on_main", **line, why=why, told=list(told), done=False,
                                error=f"{type(e).__name__}: {e}"[-600:])]
        t.after_running = True
        return [self.record(t, "on_main", **line, why=f"{why}, detached", told=list(told), job=job, done=False)]

    def _after_ended(self, t: Tick, adv: dict, told: list[str], job: dict) -> dict | None:
        """The record of advance ``adv``'s steps once their detached job ended (`World.after_ended`), as `_on_main` logs them:
        done, or not with the error, which is printed and which the next tick answers by starting them again; None while
        the job runs."""
        if (end := self.world.after_ended(job)) is None:
            return None
        sha = adv["sha"]
        line = dict(sha=sha, base=adv.get("base") or "", members=adv.get("members") or (), heads=adv.get("heads") or ())
        if isinstance(end, str):
            end = {"done": False, "error": f"its job {end}"}
        done, error = bool(end.get("done")), str(end.get("error") or "")[-600:]
        if not done:
            self.out(f"research queue vqueue: tick {t.n}: the steps after `main` moved to {sha[:12]} failed, the next tick "
                     f"runs them again: {error}")
        return self.record(t, "on_main", **line, why="drive's steps once its members are on `main`, detached",
                           told=list(end.get("told") or told), done=done, ended=job.get("dir"),
                           **({} if done else {"error": error}))

    def _drop(self, a: Drop, c: Candidate, t: Tick) -> dict:
        """Cancel ``c``'s check, the run beside it, and its records run while they run (they would hold drive's machines for
        nothing), delete its branch and log its outcome.  A run still launching (its shards, of a sharded one) has no run ids
        yet: its record is logged (`beside_launching`, `records_launching`), and `_launched` cancels it once it has them; an
        auto plan still being made is given up (`vqueue_slim.abandon`), so no shard of it launches.  One decided on a check that gates failed while it ran (an early red) logs when its first failure was written and
        when it was dropped (``early_red``).  One whose check passed while its runs still run steps off the gate
        (`vqueue_slim._passed_early`) leaves them running where it is dropped on `main`, and cancels them otherwise, as
        nothing will land on what they find; its line keeps their record (``off_gate_running``) while it keeps
        a verdict and those runs or the run beside its base's check haven't ended, which `_off_gate` reads until both did."""
        w, j = self.world, t.judged.get(c.sha)
        why = f"vqueue: candidate {c.sha[:12]} was dropped: {a.why}"
        rec, kept = _running(j), (j.record if j is not None else None) or {}
        if stop := a.why != ON_MAIN and bool(kept.get("still_running")):
            rec = kept
        base = t.below if c.base == t.main else t.past.slim(c.base)[1]
        unread = not stop and "verdict" in kept and (kept.get("still_running") or isinstance(base, dict) and (
            "verdict" not in base or base.get("still_running")))
        if "plan" in rec:
            self.slim.abandon(rec)
        ids = list(rec.get("ids") or ())
        failed = w.cancel_runs(ids, why) if ids else {}
        regen = (t.records.get(c.sha) or {}) if c.records is not None and c.records.state == PENDING else {}
        rids = list(regen.get("ids") or ())
        rfailed = w.cancel_runs(rids, why) if rids else {}
        w.push("", branch(c.sha), c.sha)
        early = (rec.get("early") or {}).get("at") if rec and a.outcome is not None and j.slim.state == FAIL else None
        return self.record(t, "drop", sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why, outcome=a.outcome,
                           failed=c.verdict.failed, done=True,
                           **({"beside_cancelled": ids} if ids else {}), **({"beside_cancel_failed": failed} if failed else {}),
                           **({"beside_launching": rec} if "launch" in rec or "shards" in rec else {}),
                           **({"off_gate_running": kept} if unread else {}),
                           **({"records_cancelled": rids} if rids else {}),
                           **({"records_cancel_failed": rfailed} if rfailed else {}),
                           **({"records_launching": regen} if "launch" in regen else {}),
                           **({"early_red": {"failed_at": stamp(early), "dropped_at": stamp(self.now())}}
                              if early is not None else {}))

    def _launched(self, t: Tick) -> list[dict]:
        """The run beside each dropped candidate's check, and its records run, that was still launching at its drop
        (`Past.launching`), judged each tick until its launch ends: cancelled once it printed its run ids while it runs, and
        settled (`dropped`) either way; one that registered no run or already ended has nothing to cancel."""
        out = []
        for run, key, action, prefix in ((self.slim, "beside_launching", "slim", "beside"),
                                         (self.records, "records_launching", "regenerate", "records")):
            if run is None:
                continue
            for sha, drop in t.past.launching(key, action).items():
                rec = dict(drop[key])
                v = run.judge(self.world.slim_driver(), rec)
                ids = list(rec.get("ids") or ())
                if v.state == PENDING and not ids:
                    continue
                running = ids if v.state == PENDING else []
                reason = f"vqueue: candidate {sha[:12]} was dropped: {drop.get('why')}"
                failed = self.world.cancel_runs(running, reason) if running else {}
                out.append(self.record(t, action, sha=sha, base=drop.get("base"), members=drop.get("members") or (),
                                       heads=drop.get("heads") or (), record=rec, dropped=True, done=True,
                                       why=f"its candidate was dropped while it launched: {'cancelled' if running else v.state}",
                                       **({f"{prefix}_cancelled": running} if running else {}),
                                       **({f"{prefix}_cancel_failed": failed} if failed else {})))
        return out

    def _eject(self, a: Eject, t: Tick) -> dict:
        w = self.world
        tasks = "; ".join(f"{f.task} (key {f.key}" + (f", failing {named}" if (named := _failing_tests(f)) else "") + ")"
                          for f in a.failed)
        text = (f"#{a.number} at {a.head[:12]} is ejected from the verified queue: {a.why}. "
                + (f"Failed: {tasks}. Each result, its log's digest with it: `research data labels task:<key> --key result "
                   "--history`. " if tasks else "") + "Fix it and mark it ready again")
        if (p := w.open().get(a.number)) is None or p.head != a.head:
            return self.record(t, "eject", members=[a.number], heads=[a.head], number=a.number, failed=a.failed, done=False,
                               why=f"{a.why}; not ejected: " + ("closed" if p is None else f"pushed to {p.head[:12]} since"))
        told = False
        if not self.shadow:
            w.withdraw(a.number, a.head, a.why)
            told = w.say(text, self.owner(a.number))
        return self.record(t, "eject", members=[a.number], heads=[a.head], number=a.number, why=a.why, failed=a.failed,
                           text=text, told=told, done=not self.shadow)

    def _say(self, t: Tick, kind: str, text: str, to: tuple[str, ...], **line) -> bool:
        """``text`` posted to ``to`` (`World.say`), logged as about to be first (`posting`, with ``line``, the keys `Past`
        reads a message of ``kind`` by): a tick that stops between the post and its record tells no one twice."""
        self.write(self.record(t, "posting", posting=kind, text=text, to=list(to), **line))
        return self.world.say(text, to)

    def owner(self, number: int) -> tuple[str, ...]:
        """Whom a message about PR ``number`` goes to: its owner's handle, else `CI`, as drive tells."""
        handle = self.world.handle(number)
        return (f"@{handle}" if handle else CI,)

    def _retry(self, c: Candidate, t: Tick) -> list[dict]:
        """Try ``c``'s check again, logged as an enqueue (``enqueued``, which `Past.tries` counts): a run beside it that ended
        in infra, or failed only where its machine failed it (`vqueue._erred`), is started again on its machines but those
        its runs failed on (``erred_on``); one still running is left to run."""
        j, line = t.judged[c.sha], dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads)
        built = t.past.built(c.sha) or {}
        priority = built.get("priority") if isinstance(built.get("priority"), int) else 0
        rec = self.record(t, "retry", **line, priority=priority, why=f"{c.verdict.state}: {j.note or 'tried again'}",
                          failed=vqueue._erred(c.verdict), enqueued=True, done=True)
        again = self._start(t, c.sha, c.base, line, "started again", avoid=(j.record or {}).get("erred_on") or ()) if (
            j.slim is not None and (j.slim.state == INFRA or vqueue._erred(j.slim))) else []
        return [rec, *again]

    def _rerun(self, a: Rerun, c: Candidate, t: Tick) -> list[dict]:
        """Run ``a.keys`` afresh: the run beside ``c``'s check, cancelled while it runs, is started again."""
        w, line = self.world, dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads)
        failed = [f for f in c.verdict.failed if f.key in a.keys]
        stopped = list(_running(t.judged[c.sha]).get("ids") or ())
        stop_failed = w.cancel_runs(stopped, f"vqueue: candidate {c.sha[:12]}'s failure is rerun afresh") if stopped else {}
        again = self._start(t, c.sha, c.base, line, "started again: its failure is rerun")
        return [self.record(t, "rerun", **line, why="the lone member's own failures, afresh: the run beside", failed=failed,
                            keys=list(a.keys), done=True, **({"beside_cancelled": stopped} if stopped else {}),
                            **({"beside_cancel_failed": stop_failed} if stop_failed else {})), *again]

    def _carried(self, t: Tick) -> list[dict]:
        """Each verdict carry decision of the tick's `ready` (`World._carry`), logged once a head while it says the same."""
        said = t.past.carries()
        return [self.record(t, "carry", members=[n], heads=[c["head"]], number=n, why=c["why"], old=c["old"],
                            carries=c["carries"]) for n, c in sorted(t.carried.items()) if said.get((n, c["head"])) != c["why"]]

    def _unauthenticated(self, t: Tick) -> list[dict]:
        """Each ready mark and land verdict the tick counted that drive.toml's `signed_verdicts` wouldn't, on "shadow"
        (`StoreLabels.unauthenticated`): the PR, the head the label is on, its key (`label`), author (`by`) and time
        (`written`), and why, logged once a label while the reason is the same."""
        said = t.past.unauthenticated()
        return [self.record(t, "unauthenticated", members=[n], heads=[head], number=n, label=key, by=by, written=ts, why=why)
                for (n, head, key, by, ts), why in sorted(t.unauthenticated.items()) if said.get((n, head, key, by, ts)) != why]

    def _off_gate(self, t: Tick) -> list[dict]:
        """Ring who acts on what a candidate's check failed of each step `vqueue_slim.judge`
        leaves out (policy.toml's `off_gate` at its base, drive.toml's `warn_steps`), once its verdict is pass or fail, and that
        the run beside its base's check (`Tick.below` beside `main`, else the base candidate's) didn't fail too (`_off_new`):
        whom `_off_owners` names, with each failure's line, once a candidate and step (`Past.rung`, by why).  Nothing is held or
        ejected for it.  A post that fails is rung again by the next tick; with --shadow it is only logged.  What a check
        that passed while its runs still ran those steps (`vqueue_slim._passed_early`) failed of them is read once they ended
        (`vqueue_slim.failures`), its candidate dropped by then or not: a dropped one's runs (`Past.off_gate_running`, its
        landing's drop among them) are judged each tick until they and its base's have ended, then rung as a candidate's
        are, and an `off_gate` line says they were read, unless a post failed, which the next tick rings again.  Its base's
        run is judged here too while it has no verdict, as one beside a `main` since moved on, or beside a candidate since
        dropped, is nowhere else; one still launching, waiting or planning isn't, since judging it would launch it."""
        recs, rung, d = [], t.past.rung(), self.world.slim_driver()
        for c in t.snap.candidates:
            if (j := t.judged[c.sha]).slim is None or j.slim.state not in (PASS, FAIL):
                continue
            recs += self._off_rings(t, d, c, j.record, t.below if c.base == t.main else t.past.slim(c.base)[1], rung)
        for sha, drop in t.past.off_gate_running().items():
            c = Candidate(sha, drop.get("base") or "", tuple(drop.get("members") or ()), tuple(drop.get("heads") or ()))
            rec, base = dict(drop["off_gate_running"]), t.below if c.base == t.main else t.past.slim(c.base)[1]
            self.slim.judge(d, rec)
            if base is not None and (base.get("still_running") or "verdict" not in base and base.get("ids")
                                     and not any(k in base for k in ("launch", "wait", "shards"))):
                self.slim.judge(d, base := dict(base))
            if rec.get("still_running") or base is not None and ("verdict" not in base or base.get("still_running")):
                continue
            recs += (rings := self._off_rings(t, d, c, rec, base, rung))
            if not any(r.get("posted") is False for r in rings):
                recs.append(self.record(t, "off_gate", sha=sha, base=c.base, members=c.members, heads=c.heads, record=rec,
                                        done=True, why="its runs ended after its drop: what they failed off the gate is read"))
        return recs

    def _off_rings(self, t: Tick, d: Landing, c: Candidate, rec: dict | None, base: dict | None,
                   rung: set[tuple[str, str]]) -> list[dict]:
        """`_off_gate`'s rings of what ``c``'s check (``rec``) failed of each step off the gate that its base's (``base``)
        didn't (`_off_new`), but one ``rung`` holds."""
        recs = []
        for step in sorted({*d._policy(c.base).off_gate, *d.warn_steps}):
            if (c.sha, why := f"off the gate: {step} failed") in rung or not (new := self._off_new(d, c, rec, base, step)):
                continue
            to, said = self._off_owners(c, step, new), [f"`{x[:200]}`" for x in new.values()]
            text = (f"the verified queue's candidate ready:{c.sha} ({' '.join(f'#{n}' for n in c.members)}) failed {step}, "
                    f"off the merge gate, where its base {c.base[:12]} doesn't: {'; '.join(said[:OFF_SAID])}"
                    + (f"; and {len(said) - OFF_SAID} more" if len(said) > OFF_SAID else "")
                    + ". Nothing is held or ejected for it: fix it forward")
            line = dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=why, step=step, failures=list(new),
                        text=text, to=list(to))
            if self.shadow:
                recs.append(self.record(t, "ring", **line, done=False))
            elif not self._say(t, "ring", text, to, sha=c.sha, why=why):
                recs.append(self.record(t, "ring", **line, done=False, posted=False,
                                        error=f"ringing {', '.join(to)} about {c.sha[:12]}'s {step} failed: the next tick "
                                              "rings again"))
            else:
                recs.append(self.record(t, "ring", **line, done=True, posted=True))
        return recs

    def _off_new(self, d: Landing, c: Candidate, rec: dict, base: dict | None, step: str) -> dict[str, str]:
        """What ``c``'s check (``rec``) failed of ``step``, off the gate, that the run beside its base's check (``base``) didn't,
        by its key with its line (`vqueue_slim.failures`): of a ratchet gate that names what it fails, each merge of ``c``'s
        members it fails, which no run of its base holds; else each failure whose target (`_target`) the base's run failed in
        no way, but a circuit-check target whose process died (`reds.CIRCUIT_KILLED`) and a failure an unexpired allowance at
        ``c`` covers, none of it while the base's run hasn't ended or when either run didn't run the step to its end.  A red
        the base shares, as `main`'s own, rings nobody."""
        if not (mine := self.slim.failures(d, rec, step)):
            return {}
        if step == RATCHET_STEP and step not in mine:
            return {k: x for k, x in mine.items() if k in {f"#{n}" for n in c.members}}
        if (theirs := self.slim.failures(d, base, step)) is None:
            return {}
        pol, was = d._policy(c.sha), {_target(k) for k in theirs}
        return {k: x for k, x in mine.items() if _target(k) not in was and not k.startswith(CIRCUIT_KILLED)
                and not (step == CIRCUIT_STEP and pol.circuit(f"FAIL {x}", self.now()).covered)}

    def _off_owners(self, c: Candidate, step: str, new: dict[str, str]) -> tuple[str, ...]:
        """Who acts on what ``c`` failed of ``step`` off the gate (``new``, `_off_new`), first who acts (`reds.red_owners`): of
        ratchet-gate, the lane of each member whose merge it fails, with the step's owner in ci.toml at ``c`` only when one such
        lane is unknown or it fails no merge, since the gate itself may then be wrong; of circuit-check,
        the owner in `main`'s registry of each sidecar's pins file at ``c`` pinning a failed target (`World.pins`), and for a
        target none owns, the lane of each member whose change meets circuit-check's inputs (`queue.forces_cold`), else the
        step's owner; of another, the step's owner.  `CI` when none is known."""
        w, d = self.world, self.world.driver()
        job, lanes, owners = d._step_owner(step, c.sha), [], []
        if step == RATCHET_STEP:
            lanes = [w.handle(int(k[1:])) for k in new if k != step]
            job = None if lanes and all(lanes) else job
        elif step == CIRCUIT_STEP:
            pins = w.pins(c.sha, (_target(k) for k in new))
            by = {x: [h for p in pins.get(x, ()) if (h := d._path_owner(p))] for x in map(_target, new)}
            owners = [h for hs in by.values() for h in hs]
            if all(by.values()):
                job = None
            elif lanes := [w.handle(n) for n, h in zip(c.members, c.heads) if forces_cold(w._changed(c.base, h))]:
                job = None if all(lanes) else job
        return tuple(f"@{h}" if h else CI for h in red_owners(lanes, [*owners, job]))

    def _conflicts(self, t: Tick) -> list[dict]:
        """Tell the owner of each ready PR whose head doesn't merge onto `main` (`Ready.conflict`, which no build takes, a
        move's or any other's) the files it conflicts in, and to rebase it onto `main` and mark the new head: once a head
        (`Past.conflicted`).  Its mark stays, since the new head needs its own.  A post that fails is told again by the next
        tick; with --shadow it is only logged."""
        recs, told = [], t.past.conflicted()
        for r in t.snap.ready:
            if not r.conflict or r.head in told:
                continue
            if now := self._landed(t, r.head):
                recs.append(self.record(t, "conflict", members=[r.number], heads=[r.head], number=r.number, landed=now,
                                        why=f"`main` {now[:12]}, fetched again, holds its head: nobody is told", done=False))
                continue
            files = conflicting_files(self.world.repo, t.main, r.head)
            text = (f"#{r.number} at {r.head[:12]} doesn't merge onto `main` at {t.main[:12]}"
                    + (f": it conflicts in {', '.join(files)}" if files else "")
                    + ". The verified queue leaves it out of every build until it does: rebase it onto `main` (where a move "
                      "that landed is the cause, carry it across with `tools/move/restack.py`) and mark the new head ready "
                      f"(`research queue ready {r.number}`)")
            to = self.owner(r.number)
            line = dict(members=[r.number], heads=[r.head], number=r.number, files=files, text=text, to=list(to))
            if self.shadow:
                recs.append(self.record(t, "conflict", **line, done=False))
            elif not self._say(t, "conflict", text, to, heads=[r.head], number=r.number):
                recs.append(self.record(t, "conflict", **line, done=False, posted=False,
                                        error=f"telling #{r.number}'s owner it doesn't merge onto `main` failed: the next "
                                              "tick tells again"))
            else:
                recs.append(self.record(t, "conflict", **line, done=True, posted=True))
        return recs

    def _lacking(self, t: Tick) -> list[dict]:
        """Tell the owner of each PR `ready` left out for a move it lacks whose retired names its own change names
        (`World.lacking`) the move and the fix: once a head and base, the base the last such move (`_tell_once`)."""
        return self._tell_once(t, "lacks", "of the move it lacks", {
            n: (head, move, why, f"#{n} at {head[:12]} isn't taken into the verified queue's candidates on `main` "
                                 f"{t.main[:12]}: {why} (`research queue ready {n}`)")
            for n, (head, move, why) in self.world.lacking.items()})

    def _behind_held(self, t: Tick) -> list[dict]:
        """Tell the owner of each marked PR `ready` left out for holding a held PR's head (`World.behind_held`) that head and
        what releases it: once a head and held head, the base (`_tell_once`)."""
        return self._tell_once(t, "behind_held", "that it waits on a held head", {
            n: (head, held, why, f"#{n} at {head[:12]} goes into no candidate on `main` {t.main[:12]}: {why}")
            for n, (head, held, why) in self.world.behind_held.items()})

    def _tell_once(self, t: Tick, kind: str, what: str, notes: dict[int, tuple[str, str, str, str]]) -> list[dict]:
        """Tell each PR's owner its note of ``notes`` ({number: (head, base, why, text)}) once a head and base (`Past.told`, as
        a skip is told), logged as ``kind``.  A post that fails is told again by the next tick; with --shadow it is only
        logged."""
        recs, told = [], t.past.told(kind)
        for n, (head, base, why, text) in sorted(notes.items()):
            if f"{head}@{base}" in told:
                continue
            line = dict(base=base, members=[n], heads=[head], number=n, why=why, text=text, to=list(to := self.owner(n)))
            if self.shadow:
                recs.append(self.record(t, kind, **line, done=False))
            elif not self._say(t, kind, text, to, base=base, heads=[head], number=n):
                recs.append(self.record(t, kind, **line, done=False, posted=False,
                                        error=f"telling #{n}'s owner {what} failed: the next tick tells again"))
            else:
                recs.append(self.record(t, kind, **line, done=True, posted=True))
        return recs

    def _riding(self, t: Tick) -> list[dict]:
        """A line for each ready PR a candidate on `main`'s chain holds without it as a member (`Tick.riding`), once a head and
        candidate: it reaches `main` with that candidate, so nobody is told."""
        logged = t.past.riding()
        return [self.record(t, "riding", sha=sha, members=[n], heads=[head], number=n, done=True,
                            why=f"{sha[:12]} holds its head without it as a member: it reaches `main` with it")
                for n, (head, sha) in sorted(t.riding.items()) if (head, sha) not in logged]

    def _landed(self, t: Tick, head: str) -> str | None:
        """`main` fetched again when it holds ``head``, else None: nobody is told of a head `main` holds.  Fetched once a tick,
        the first time a notice needs it (the candidate the tick advances `main` to is under its build, which holds its heads:
        `held`); a fetch that fails reads the `main` the tick holds."""
        if t.fetched is None:
            try:
                t.fetched = self.world.main_now()
            except (QueueError, OSError, subprocess.SubprocessError) as e:
                t.fetched = t.advanced or t.main
                self.out(f"research queue vqueue: tick {t.n}: fetching `main` again failed, so {t.fetched[:12]} is read: {e}")
        return t.fetched if self.world._ancestor(head, t.fetched) else None

    def _ring(self, a: Ring, c: Candidate, t: Tick) -> list[dict]:
        if (c.sha, a.why) in t.past.rung():
            return []
        erred = vqueue._erred(c.verdict)
        infra = "; ".join(f"{f.task} (key {f.key})" for f in erred)
        members = " ".join(f"#{n}" for n in c.members)
        state = vqueue._reading(c.verdict) if c.verdict.state == FAIL else c.verdict.state
        text = (f"the verified queue's candidate queue/{c.sha[:12]} ({members}), whose check waits for its records run: {a.why}"
                if c.waits else f"the verified queue's candidate ready:{c.sha} ({members}) is {state} after "
                f"{c.tries} enqueues: {a.why}" + (f". Infra: {infra}" if infra else ""))
        line = dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why, failed=erred, text=text)
        if a.pr is not None and (held := self._outage(c, t)) is not None:
            return [self.record(t, "ring", **line, done=True, posted=False, held=held)]
        if self.shadow:
            return [self.record(t, "ring", **line, done=False)]
        to = a.to or ((INFRA_HANDLE,) if a.pr is None else self.owner(a.pr))
        if not self._say(t, "ring", text, to, sha=c.sha, why=a.why):
            return [self.record(t, "ring", **line, done=False, posted=False,
                                error=f"ringing {', '.join(to)} about {c.sha[:12]} failed: the next tick rings again")]
        return [self.record(t, "ring", **line, done=True, posted=True)]

    def _outage(self, c: Candidate, t: Tick) -> str | None:
        """Why a ring to the owner of ``c``'s lone member about its check erring is held, logged and not posted: in the same
        tick another candidate's check, or the run beside `main`'s own, reads infra too, as an outage reads, which can't
        tell from one change's erring.  Infra is rung all the same."""
        if others := [o.sha[:12] for o in t.snap.candidates if o.sha != c.sha and (o.verdict.state == INFRA
                                                                                     or vqueue._erred(o.verdict))]:
            return f"{', '.join(others)} read infra too"
        if ((t.below or {}).get("verdict") or {}).get("state") == INFRA:
            return f"the run beside {t.main[:12]}'s check ended in infra"
        return None

    def _red(self, a: Red, c: Candidate, t: Tick) -> dict:
        """Tell `CI` that ``a.base`` is red on it, once a base and task: a post that fails is told again by the next red on that
        base, and the drop after it goes ahead."""
        tasks, to = "; ".join(f"{f.task} (key {f.key})" for f in a.failed), (CI,)
        new = {f.task for f in a.failed} - t.past.reds(a.base)
        text = None if not new else (
            f"the verified queue's base {a.base[:12]} is red: {a.why}. Failed there as on {c.sha[:12]}: {tasks}. Each result, its "
            f"log's digest with it: `research data labels task:<key> --key result --history`")
        posts = text is not None and not self.shadow
        done = posts and self._say(t, "red", text, to, sha=c.sha, base=a.base, failed=a.failed)
        return self.record(t, "red", sha=c.sha, base=a.base, members=c.members, heads=c.heads, why=a.why, failed=a.failed,
                           text=text, to=list(to), done=done, **({"posted": done} if posts else {}))

    def _red_main(self, t: Tick) -> list[dict]:
        """What `main`'s own red check (`MainRed`, ``self.red``) calls for, once a red commit: with a culprit, its merge reverted
        (`World.revert`) and the revert landed as the loop lands a candidate (`World.advance`: a lease on `main`, never a force),
        the ready PRs whose change meets what the failures read held while that red stands (`snapshot`), and `CI` and the
        culprit's owner rung; with none, or a revert that doesn't apply or push, nothing reverted, plain candidates taking the
        whole check until `main`'s passes (`_quick`), and `CI` rung with the failed steps.  One line each, its reason with it:
        `revert` (``sha`` the revert, ``number`` the culprit, ``merge``, ``holds``) or `fallback`, both with ``red``, the
        red commit, and ``passed``; a line whose post failed isn't done, so the next tick acts again."""
        r, w = self.red, self.world
        landed = " ".join(f"#{m.pr}" for m, _ in r.landed) or "none"
        line = dict(red=r.tip, base=t.main, members=[m.pr for m, _ in r.landed], heads=[m.head for m, _ in r.landed],
                    passed=r.passed, failed=r.verdict.failed or r.verdict.infra)
        said = (f"`main` is red: its own check at {r.tip[:12]} failed {r.steps()}; landed since it last passed"
                f"{f' at {r.passed[:12]}' if r.passed else ''}: {landed}.")
        why = r.why
        if (m := r.culprit) is not None:
            rev = r.reverted or w.revert(m.commit, t.main, f"`main`'s check at {r.tip} failed {r.steps()}: {r.why}.") or ""
            if rev:
                held = sorted(n for n, _, y in t.snap.left if y.startswith("held: `main`'s check at"))
                text = (f"{said} {why}: reverted as {rev[:12]} (`git revert -m 1 {m.commit[:12]}`), and the ready PRs whose "
                        "change meets what the failures read wait until `main`'s check passes again"
                        + (f": {' '.join(f'#{n}' for n in held)}" if held else "") + ".")
                to = tuple(dict.fromkeys((CI, *self.owner(m.pr))))
                rec = dict(**line, sha=rev, number=m.pr, merge=m.commit, holds=sorted(r.holds), held=held, why=why,
                           text=text, to=list(to), landed=line["members"])
                rec["heads"] = [m.head]
                if self.shadow:
                    return [self.record(t, "revert", **rec, done=False)]
                if not r.reverted:
                    try:
                        w.advance(rev, t.main)
                    except (QueueError, OSError, subprocess.SubprocessError) as e:
                        return [self.record(t, "revert", **rec, done=False, pushed=False,
                                            error=f"pushing the revert failed, the next tick tries again: {e}"[-600:])]
                posted = self._say(t, "revert", text, to, red=r.tip)
                return [self.record(t, "revert", **rec, done=posted, posted=posted, pushed=not r.reverted)]
            why = f"{why}, but its revert of {m.commit[:12]} doesn't apply onto {t.main[:12]}"
        text = (f"{said} No PR is reverted: {why}. Plain candidates take the whole check before `main` until `main`'s check "
                "passes again.")
        rec = dict(**line, why=why, text=text, to=[CI])
        if self.shadow:
            return [self.record(t, "fallback", **rec, done=False)]
        posted = self._say(t, "fallback", text, (CI,), red=r.tip)
        return [self.record(t, "fallback", **rec, done=posted, posted=posted)]


def _name(a: Action) -> str:
    return type(a).__name__.lower()


def _slim():
    """`research.vqueue_slim`, None while this tree has none."""
    import importlib
    import importlib.util

    name = f"{__package__}.vqueue_slim"
    return importlib.import_module(name) if importlib.util.find_spec(name) is not None else None


def _shards(text: str) -> int | str:
    """`--shards`: N as an int, anything else as it is given, which `vqueue_slim.shard_count` reads (`auto:N`) or refuses."""
    return int(text) if text.isdigit() else text


def _sharded(slim, shards: int | str) -> bool:
    """Whether ``shards`` is one shard or more to ``slim`` (`vqueue_slim.shard_count`); False for one it refuses."""
    try:
        return slim.shard_count(shards)[0] >= 1
    except ValueError:
        return False


def _locked(path: Path, wait: bool = False):
    """Hold ``path``'s lock, or None when another tick holds it; with ``wait``, once whoever holds it lets it go."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
        f.close()
        return None
    return f


def _after_landing(world: World, d: Path, now: Callable[[], float] = time.monotonic) -> int:
    """`--after-landing DIR`: the steps after `main` moved of the advance `job.json` in ``d`` names (`World.on_main`), which
    a tick started detached (`World.start_after`), at `main` as origin has it now; what they did written to `result.json`
    there (done, the error when not, the stacked PRs' owners told), which the tick after it ended logs (`Loop._moved`)."""
    t0, job = now(), json.loads((d / "job.json").read_text())
    told = list(job.get("told") or ())
    try:
        world.at_main()
        world.on_main(job["sha"], job.get("base") or "", [Member(**m) for m in job.get("merged") or ()], told)
        got: dict = {"done": True, "told": told}
    except Exception as e:  # noqa: BLE001 -- `main` has moved: the tick logs the error and starts them again
        got = {"done": False, "told": told, "error": f"{type(e).__name__}: {e}"[-600:]}
    tmp = d / "result.json.tmp"
    tmp.write_text(json.dumps(got, sort_keys=True) + "\n")
    tmp.replace(d / "result.json")
    print(f"research queue vqueue: after `main` moved to {job['sha'][:12]}: {'done' if got['done'] else got['error']} "
          f"in {now() - t0:.1f} s", flush=True)
    return 0


def main(argv: list[str]) -> int:
    from . import queue
    from .queue import GitHub, credential_args, remote_and_repo
    from .store.local import LocalStore

    ap = argparse.ArgumentParser(prog="research queue vqueue", description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=WORKDIR, help=f"the clone to work in (default the queue's, {WORKDIR}); the "
                                                              f"loop fetches into its own refs ({NS}/)")
    ap.add_argument("--remote", default=None, help="the repository (default the clone's origin)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--lock", type=Path, default=LOCK, help=f"the loop's own lock (default {LOCK})")
    ap.add_argument("--log", type=Path, default=LOG, help=f"the decision log, one JSON line per action (default {LOG}); its "
                                                          f"directory holds the loop's caches")
    ap.add_argument("--state", type=Path, default=STATE, help=f"each tick's candidates and their verdicts, rewritten each tick "
                                                              f"(default {STATE})")
    ap.add_argument("--priority-file", type=Path, default=PRIORITY, help=f"the priority list, the landing order: PR numbers, "
                                                                        f"first to land first (default {PRIORITY})")
    ap.add_argument("--frozen", type=Path, default=FROZEN, help=f"a freeze's pathspecs, one a line: while the file exists, a PR "
                                                                f"renaming or deleting a file under one waits (default {FROZEN})")
    ap.add_argument("--deselect", type=Path, default=DESELECT, help=f"the tests a candidate's check leaves out, one node id a line "
                                                                    f"from the repository's root, read at each launch (default "
                                                                    f"{DESELECT})")
    ap.add_argument("--keep-runs", type=Path, default=KEEP_RUNS, help=f"the runs a cancel leaves running, one run id a line, read "
                                                                      f"at each cancel (default {KEEP_RUNS})")
    ap.add_argument("--graph2", help=argparse.SUPPRESS)                   # ignored: a loop's args file may still name them
    ap.add_argument("--no-graph2-shadow", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=BATCH, help=f"the most members a candidate holds (default {BATCH})")
    ap.add_argument("--depth", type=int, default=DEPTH, help=f"the most candidates stacked at once, each with the run beside "
                                                             f"its check (default {DEPTH})")
    ap.add_argument("--lean", type=int, default=LEAN, help=f"the most members of a Lean candidate, each with a Lean change "
                                                           f"that compiles (a spec's or a Lean source's, not a lock's "
                                                           f"alone): one opens whenever none is staged and no PR "
                                                           f"changing a Lean input with no Lean change waits, of the first "
                                                           f"in rank, whether or not another PR waits; 0: only pinned ones "
                                                           f"open one, which take no limit (--pins-file) (default {LEAN})")
    ap.add_argument("--wide-file", type=Path, default=WIDE,
                    help=f"the PRs whose Lean change is wide, one number a line, read each tick as --priority-file is: a "
                         f"candidate taking one takes no other member --lean counts, and one stacked on a candidate holding "
                         f"one takes none until it is on `main`; none while the file is missing or empty (default {WIDE})")
    ap.add_argument("--pins-file", type=Path, default=PINS,
                    help=f"the pinned PRs, one number a line, `#` comments and blank lines allowed, read each tick as "
                         f"--priority-file is: while a pinned Lean change waits, a Lean candidate holds pinned Lean changes "
                         f"only, under no --lean limit, so a pin orders a Lean candidate or cuts it, and with --lean 0 it is the "
                         f"only way into one; it puts none ahead of a PR changing a Lean input, which goes first; none while "
                         f"the file is missing or empty (default {PINS})")
    ap.add_argument("--heavy-file", type=Path, default=HEAVY,
                    help=f"the PRs whose Lean change is heavy, one number a line, read each tick as --wide-file is: each is "
                         f"wide too, and a candidate holding one runs its records run, and the shard of its check that audits "
                         f"security/Proofs, on --heavy-on's machines alone, a resend after a machine's fault too; none while "
                         f"the file is missing or empty, or --heavy-on is empty; a line '*' makes every candidate's runs so, "
                         f"main's own check's too, and no PR wide (default {HEAVY})")
    ap.add_argument("--heavy-on", default="", help="the machines a heavy PR's candidate's heavy runs take, a comma list, each "
                                                   "run the first free of them (--heavy-file); empty, the default, turns heavy "
                                                   "PRs off")
    ap.add_argument("--pending-hours", type=float, default=4.0,
                    help="hours the lowest candidate's check may stay pending after its first enqueue before it is enqueued "
                         "again; rung at twice that (default 4)")
    ap.add_argument("--no-slim", action="store_true",
                    help="start no check: every candidate stays pending; only with --shadow or --dry-run")
    ap.add_argument("--gate", choices=GATES, default=CHECK_GATE,
                    help=f"what gates a candidate: `{CHECK_GATE}` (the default), its whole check (`check.py --record`), every "
                         "step as --shards shards (but lean-audit where no member changes a Lean input and it is built on a "
                         "candidate in flight whose check runs it or took it from the one below that ran it, whose verdict on "
                         f"it it takes once that one has landed); `{VERDICT_GATE}`, `{CHECK_GATE}` but for a plain candidate, each member with a "
                         "land verdict of its own at its head, no move, and no change to a Lean input or a path of drive.toml's "
                         "`lowrisk_excludes` (but its entries `verdict_plain` lists) or `verdict_excludes` (with either "
                         "of those two empty, none is plain), which lands on those "
                         "verdicts and the quick tier on its own merge (a `research "
                         "run` on --on's machines), `main`'s whole check following each such landing; a red of it reverts "
                         "the one PR landed since `main`'s last passed check whose change meets what the failure read "
                         "(`git revert -m 1`, pushed as the loop lands) and holds the ready PRs that meet it too, and with no "
                         "such PR reverts none and sends plain candidates through the whole check until `main`'s passes, "
                         "ringing ci either way. While it is on, `main` holds a plain candidate's commits before their whole "
                         "check has run; every other candidate (Lean, a lock, a move, the check or the landing code itself) "
                         "takes the whole check before it lands")
    ap.add_argument("--shards", type=_shards, default=SHARDS,
                    help=f"with --gate {CHECK_GATE} or {VERDICT_GATE}, the shards each check runs as, one launch each: an N "
                         "ci.toml has a shard "
                         f"plan for, or auto:N, N shards of an auto plan of the candidate's tree made before they launch, "
                         f"which splits pytest by suite (default {SHARDS})")
    ap.add_argument("--on", default="", help="the machines the run beside each check and each records run take, a comma list, "
                                             "each shard its share of them first, then the rest (`landing.shard_machines`; "
                                             "default drive.toml's `on` at `main`)")
    ap.add_argument("--no-records", action="store_true",
                    help="start the check of a candidate with a member that changes Lean, a lock or a spec at once, with no "
                         "records run before it (research.vqueue_records: lean-audit alone, on --on's machines) "
                         "regenerating its lock records and no records commit of them checked in its place: stale records "
                         "fail it as any failure")
    ap.add_argument("--shadow", action="store_true", help="log advance, eject, ring, red and messages, and do none of "
                                                          "them; its own main is refs/vqueue/shadow in the clone")
    ap.add_argument("--dry-run", action="store_true", help="print the snapshot and the actions; write nothing")
    ap.add_argument("--once", action="store_true", help="one tick")
    ap.add_argument("--interval", type=float, default=INTERVAL_S, help=f"seconds between ticks (default {INTERVAL_S})")
    ap.add_argument("--move-carry-jobs", type=int, default=MAPPED_JOBS, metavar="N",
                    help=f"old heads mapped across `main`'s moves at once, each in a job the end of a tick detaches and none "
                         f"waits on, running every move's script on a scratch worktree under --merged-root and writing the "
                         f"mapping, or its refusal, into mapped-changes.d/ beside the log, so an old head's "
                         f"land verdict and mark carry to a head whose own change is theirs mapped (`Queue.mapped_change`), "
                         f"from a head with a land verdict of its own only (`StoreLabels.lands`); those "
                         f"`research queue map-moves` mapped elsewhere count once copied into mapped-changes.d/ beside the "
                         f"log; 0 carries none across a move (default {MAPPED_JOBS})")
    ap.add_argument("--merged-root", type=Path, default=None, metavar="DIR",
                    help="where the worktrees a build checks out go (`queue.merged_worktree`, one a merge for `ported`'s "
                         "codemod) and those a move's script runs on for --move-carry-jobs (`queue.mapped_tree`), and where "
                         "one a killed tick left is swept (default the temporary directory; the control pod's TMPDIR is a "
                         "network volume, where each worktree add and remove takes minutes)")
    ap.add_argument("--after-landing", type=Path, default=None, metavar="DIR",
                    help=f"run the steps after `main` moved that a tick started detached, as DIR/job.json names them, one "
                         f"such job at a time (`{AFTER}.lock` beside the log), and write DIR/result.json; no tick")
    a = ap.parse_args(argv)
    queue.MERGED_ROOT = a.merged_root or queue.MERGED_ROOT
    if a.no_slim and not (a.shadow or a.dry_run):
        print("research queue vqueue: --no-slim moves `main` past lean-agreement and the preflights unchecked: it runs only "
              "with --shadow or --dry-run", file=sys.stderr)
        return 1
    if (slim := None if a.no_slim else _slim()) is None and not a.no_slim:
        print("research queue vqueue: research.vqueue_slim isn't in this tree, so no candidate's check can start",
              file=sys.stderr)
        return 1
    if slim is not None and not _sharded(slim, a.shards):
        print("research queue vqueue: each candidate's check runs as one shard or more: --shards is N (ci.toml's plan) or "
              "auto:N (an auto plan)", file=sys.stderr)
        return 1
    lock = _locked(a.log.parent / f"{AFTER}.lock", wait=True) if a.after_landing else None if a.dry_run else _locked(a.lock)
    if not a.dry_run and lock is None:
        print(f"research queue vqueue: another tick holds {a.lock}", file=sys.stderr)
        return 1
    try:
        remote, slug = remote_and_repo(a.remote, None, a.repo)
        store = LocalStore(a.store)

        def labels_at() -> StoreLabels:
            return StoreLabels(store, log=lambda s: print(s, file=sys.stderr))

        labels = labels_at()
        q = _Clone(a.repo, remote=remote, github=GitHub(slug), ready=labels.ready, store=store,
                   auth=credential_args(a.repo, remote), before_cutoff=labels.before_cutoff, marked_by=labels.marked_by,
                   marked_at=labels.marked_at, prefetch=labels.prefetch, land_into="main")
        q.branch, q.mapped_jobs = "main", max(0, a.move_carry_jobs)
        world = World(q, labels, priority_file=a.priority_file, labels_at=labels_at, cache=a.log.parent, on=a.on,
                      frozen_file=a.frozen, deselect_file=a.deselect, wide_file=a.wide_file, pins_file=a.pins_file,
                      keep_file=a.keep_runs, heavy_file=a.heavy_file, heavy_on=a.heavy_on)
        if a.after_landing:
            return _after_landing(world, a.after_landing)
        after = None if a.shadow or a.dry_run else [sys.executable, "-m", "research", "queue", "vqueue", *argv]
        loop = Loop(world, a.log, shadow=a.shadow, dry_run=a.dry_run, batch=a.batch, depth=a.depth, lean=a.lean,
                    pending_s=a.pending_hours * 3600, slim=slim, state=a.state, gate=a.gate, shards=a.shards,
                    records=None if a.no_records else vqueue_records, after=after)
        if a.once or a.dry_run:
            loop.tick()
            return 0
        return loop.run(a.interval)
    except (QueueError, OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"research queue vqueue: {e}", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.close()


__all__ = ["Judged", "Loop", "Past", "Tick", "World", "main"]
