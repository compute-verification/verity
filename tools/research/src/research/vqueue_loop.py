"""`research queue vqueue`: the verified queue's loop (`research.vqueue`).  Each tick, under its own lock (`--lock`, never
drive's), derives a `vqueue.Snapshot` from what is outside it, asks the policy (``decide``, `vqueue.decide`) what to do and
applies its actions in order, stopping the tick at the first that fails: the next tick derives everything again.  Between ticks
it keeps only what a snapshot reads: the refs, the labels, graph2's results and the decision log (`--log`), from which come
the candidates, `history`, each candidate's `tries`, `reruns` and first enqueue, the results a retry or a rerun set aside, the
record of the run beside each check and of each records run, a ring's, a red's and a flake's once-only and the owners already
told.  `--state` is a report it rewrites each tick: each candidate's verdict beside graph2's, the run beside it's and its
records run's, the jobs and kinds its graph lacks, and the `queue/*` branches it ignores.

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
`Queue._merge` merges it onto `main`, kept by (head, `main`).  The owner of one that doesn't is told once a head the files it
conflicts in, to rebase it onto `main` and mark the new head (`Loop._conflicts`), unless `main` fetched again holds that
head; its mark stays.  `left` is why each other open PR into `main` isn't ready (`World.ready`: the queue's word on it, a
withdrawn mark's author, a hold and by whom, no land verdict, the freeze, a move `main` landed that it lacks whose retired
names its own change names, its owner told once a head and move: `queue.lacked_moves`, `Loop._lacking`), which a
candidate's drop for a member no longer
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
logs why, once a why (`wait`).  A candidate's verdict is graph2's results for `ready:<sha>` (`Executor.settle`),
the labels graph2 wrote alone (`GRAPH2_BY`), each key's from the first its retry or any candidate's rerun didn't set aside (a
task graph2's `verdict` says failed or infra reads so whatever its labels say, one failed by its own fault, which it labels
no result for, included), on the merge gate's tasks: none `"gate": false`, of ratchet-gate (`reds.RATCHET_STEP`: it judges
each merge against its first parent, which a task keyed by its inputs can't) or off the gate in policy.toml at the candidate,
so graph2's lean-agreement gates as the run beside the check's does: a failed task an unexpired allowance of policy.toml at
its base covers passed (`Landing._task_allowed`), so a candidate's own allowance never covers its own failure; else a failed
or blocked task is fail, each failed task with its inputs (its own, its env's spec files and those of each task it reads),
its failing tests (result/v1's `failed_tests` and those graph2's verdict names of it, with a fault of its key alone where it
failed for more than them: `_tests`; or what a task's own fault names: `_fault_tests`) and its key and failing tests in its
base's graph as graph2 listed it, where there are (`Failed.base_key`, `Failed.base_tests`),
ended once no gating task runs but behind an infra error; else a task whose newest result is graph2's infra error
(`detail.infra`; graph2's `tree` where `Failed` has one) is infra, else every gating task passed is pass once graph2's own
verdict, where it has one, can be read and counts none pending (`World._held`, which an inherited fail's end waits on too),
else pending.  Whatever else it reads, it is pending while graph2's `verdict` lists a task it needs (on the gate, or read by
one that is) as failed or infra that the loop read passed, at the key it made from what the task reads or by a label that
says pass: graph2's word overrides a label only at graph2's key, and `--by executor` is self-asserted (`_disputed`), so it
holds until the two agree.  So too the base's side: a failure is inherited only while graph2's verdict on the base's check
lists its task failed there at the key the loop read, with the failing tests that verdict names (`World._crossed`); one
inherited on the loop's labels alone holds the candidate once graph2 says otherwise of it or ends without naming it, and
waits on the base (`BASE_UNSAID`) until then.  And it is pending while the listing the loop read, the `verity_graph` label
on its commit, which is self-asserted too, isn't the one graph2's verdict says it runs (`art`), so no listing that leaves a
gating task out hides it (`_relisted`), and while graph2 names none (`World._held`).
It is pending while its base's check isn't listed, or, with a failure, while graph2 says the base's listing failed, the
graph it listed of the base can't be read, or a task there hasn't ended, unless it or what it reads there ended in an infra
error (`World._base`): no failure is the candidate's own, so none ejects or bisects, until the base's keys say.  The loop
enqueues such a base's check again each `BASE_RETRY_S` while a candidate waits on it for want of its graph or of graph2's
word on it, and rings `CI` and `landing.INFRA` about it (`Loop._bases`).  No graph while graph2 says its listing failed
(`verdict`, else its `ready_verdict` label), a graph that can't be read, one lacking a job `main` waits for (`landing.gating`)
or a kind of task on the merge gate its base's graph lists, unless that kind's job went to another kind (`World._gone`), and
failing nothing of its own, or no graph of a candidate the log never enqueued is list-failed.  graph2's own verdict, where
it has one, is compared and a disagreement logged.  With ``slim`` (`research.vqueue_slim`), the run beside the check of what
graph2 doesn't check starts with its candidate's build, one beside each candidate's check (so up to `--depth` at once, on
drive's machines), or once in the snapshot for a candidate on `main`'s chain the log holds no start of.  Pending until it
ends, it is judged each tick, its record logged whenever it changes, and combined with the check's verdict: a candidate passes
only when both pass.  One more runs beside `main`'s own check: the candidate's `main` advanced to, else one the snapshot
starts at startup and each time `main` moves (and once more after infra).  A step a candidate's run failed is inherited when
`main`'s failed it too, and its own when `main`'s passed it (`vqueue_slim.based`); while `main`'s runs, at most `pending_s`
after it started, a fail whose every other failure is inherited is pending.  With ``records`` (`research.vqueue_records`), a
candidate built with its records run first (its build line's `records_first`, while no line enqueued its check) is staged,
off main's chain (`vqueue.decide`): its check is pending, no listing of it is read and no run beside it starts, and its
records run is judged each tick (`vqueue_records.judge`), its record logged whenever it changes (`regenerate`).  A records
commit's build line, and a staged batch's built again on another base, names the staged batch it `repairs`, and the walk of a
candidate's members steps over a records commit at its tip (`World._walk`).

The gate (`--gate`).  With `graph2`, the default, a candidate's verdict is the two combined, as above.  With `check`, drive's
after-landing check gates in graph2's place: the run beside each candidate, and beside `main`, is every after-landing step
as `--shards` shards on `--on`'s machines (`vqueue_slim.start` with ``shards``: N, ci.toml's plan, or `auto:N`, an auto
plan of the candidate's tree, which splits pytest by suite), judged on every step `main` waits for,
and its verdict alone is the candidate's; a run started while graph2 gated is started again as the check.  A Lean
candidate's records run comes first under either gate: neither its graph2 check nor its check starts until that run passed,
and both start on its records commit, or on it as built when the run regenerated nothing (`Loop._enqueue`).  A candidate built
on another in flight, no member of it changing a Lean input against its merge base with `main` (`vqueue_slim.lean_inputs`),
runs every step but lean-audit where its base's check is judged on lean-audit, or took its verdict on it from the check of a
candidate below that is.  It reads lean-audit as pending while the candidate whose check ran it (its record's
``lean_audit_from``) is in flight, whatever that check says then, and once that candidate has landed takes the verdict the
run beside `main` keeps (`vqueue_slim.audited`): it lands after it, drops with it as its base failed, and stays pending while
it is rerun, so a failure the rerun clears is never its own.  One started once its base is `main`, or once that candidate
has landed while the run beside `main` keeps no verdict on lean-audit, audits.  A failed step
is inherited by its failing tests, those the run beside its base's check failed too.  A shard that failed a step it is
judged on, or a suite of one, fails it while the other shards still run (an early red, `vqueue_slim.judge`), decided as a
finished fail is: on the lowest candidate at once, on one above it once its base's check has ended (`vqueue_slim.based`),
nothing built on it meanwhile.  Its check is cancelled with the candidates above it, and its drop's line says when the
first failure was written and when it was dropped (`early_red`).  graph2's check is enqueued, read,
retried with the check and cancelled at a drop as before, in shadow: its verdict (`Judged.shadow`) holds, ejects and
advances nothing, no candidate is held or rung for a task the loop and graph2 dispute or a base graph2 hasn't listed, and
once it and the check have both ended they are compared in one logged line, `agree` or `disagree` with the tasks graph2
failed or erred on and the steps the check failed (`Loop._compare`).  The graph2 check of a candidate dropped before it
ended, whose check had ended in a pass or a fail (`main` advanced to it, or it failed; not an early red), isn't cancelled at
its drop: the loop reads it on (`Loop._followed`) until it ends, and compares it with the check's verdict as its drop logged
it, or until `FOLLOW_S` after its enqueue, or until `KEPT_MAX` newer ones run beside it, when it is cancelled and the line
says `unfinished`; one line a candidate either way.  Each compare line holds, in seconds, how long graph2's verdict took from
its enqueue (`graph2_s`; to the give-up, of an unfinished one) and the check's from its launch (`check_s`).  Since nothing
there waits on graph2, its reading of a candidate, and its word that a followed check still runs, stand `G2_READ_S` between
ticks (`Shadow`, kept beside the log), and a tick reads `G2_FRESH` candidates' afresh at most, so `graph2_s` is that coarse
while the check runs: one is read again at once once the candidate's check ended, at its drop, and after an enqueue, a
rerun or a cancel of it.  `--no-graph2-shadow` reads none.  The steps the
check leaves out (policy.toml's `off_gate`, drive.toml's `warn_steps`) still run: once a candidate's verdict is pass or fail,
what it failed of one that the run beside its base's check didn't is rung, once a candidate and step, to who acts on it: the
lane of each member whose merge ratchet-gate fails, the owner of each circuit-check target's pins (`Loop._off_gate`); a red
its base shares rings nobody.

The actions.  Build merges its members in order onto its base (`Queue._merge`), or, for a base a build of the same tick
rebuilt (`vqueue.Build.replaces`, logged as its line's `replaces`), onto what that build pushed (`Tick.rebuilds`), leaving
out one that doesn't merge (`skip`, its owner told once per head and base, a conflict with members merged above `main`
naming them where it merges onto `main`;
nobody is told of a head what it merges onto holds, `held`, or `main` fetched again does), pushes `queue/<sha12>` (or takes
as its own one no line names that is the same build made before, as a tick cut off after its push leaves: `adopted`), logs
it, enqueues `ready:<sha>` as graph/v2 at its priority and starts the run beside its check; with `records` (a member changes
Lean, a lock or a spec) it starts the candidate's records run instead (`vqueue_records.start`: drive's lock run, on drive's
machines, on its tree), logged (`regenerate`), and the candidate is staged, its check waiting; a tick that stages one builds
its batch of PRs with no Lean change too, on the same base.  One after a Commit of its tick (`vqueue.Build.after`) is built
on the candidate that Commit put on main's chain, whose check's record, as started, it reads as its base's; it is built
nowhere, and nothing is logged, when the Commit put none there.  Every check starts at one seam (`Loop._enqueue`: the
enqueue, its line and the run beside it).
A candidate holding a heavy PR (`World.heavy`), or every candidate and `main`'s own check while the heavy file holds `*`,
runs its records run, and, where the check gates, the shard of its check that audits `vqueue_slim.HEAVY_PART`, on
`--heavy-on`'s machines alone, its other shards on theirs (`Loop._heavy_on`); a records run's resend after a machine's fault
stays on them, as a check started again does.
Advance pushes `main` to the candidate with a lease on its base and logs it; then what follows `main` moving
(`Landing._on_main`) is a step of its own (`on_main`), which the next tick runs again until one is logged done.  Outside
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
Drop cancels the check (graph2's `cancel`, where it has one), and the run beside it and its records run while they run (one
still launching once it printed its run ids, a later tick), deletes the branch and logs its outcome, with its base and its
inherited failures, their inputs whole, which `history` reads back.  Eject withdraws the PR's ready mark at its head
(`Queue.eject`, under the clone's queue lock) and tells its owner the failed tasks, their keys and where their logs are.
Retry enqueues again (graph2 lists a check whose listing failed, reruns what ended in infra errors once its own verdict is
infra, and leaves a pending one as it is), reruns those keys itself while graph2's verdict is another (a task still runs, or
one failed), sets the infra results aside, and starts a run beside it that ended in infra again.  Rerun runs the keys afresh
(graph2's `rerun`), their results set aside, a key of the run beside starting that run again; without graph2's `rerun` it is
logged `absent`, and counts as done.  Ring tells whom it names (`to`), else `landing.INFRA` (or its `pr`'s owner), once a
candidate and why; a ring to an owner is held, only logged, while another candidate's check or the run beside `main`'s reads
infra in the same tick, as in an outage (`Loop._outage`); a base candidates wait on for want of its graph or of graph2's word
on it is rung to `CI` and `landing.INFRA`, once a base and why: at once when its listing failed or its graph can't be read, and
when graph2 still hasn't listed it or said `BASE_RING_S` after it was first asked for (`Loop._bases`); a task a candidate is
held on because graph2's verdict and the loop's reading disagree on it, there or at its base, is rung to `CI` with both keys
and both readings, and a listing the loop read that isn't graph2's with both, once each (`Loop._disputes`).  Red tells
`CI` and the owners of what the base fails (as drive names them: a test file's suite's, else its job's) that it is red on
them, once a base and task.  A test graph2's verdict says flaked (`flaky`: a fail a rerun replaced) is told to its file's
owner in drive's registry and `CI`, once a key: a test that flakes is deleted.  A skip's, conflict's, ring's, red's or
flake's message is logged as about to be posted (`posting`) before it is, and its record says whether it was (`posted`): one
whose tick stopped in between is told, so no one is told twice, and one whose post failed is told again.  Each action and its
result is one JSON line in the log, whose directory also holds the loop's caches.  Each tick prints one line of its steps'
seconds and the peak resident size after each, with graph2's verdicts it read, their bytes and seconds (`Steps`).

`--shadow` logs advance, eject, ring, red, flake and every message with `shadow: true` and does none of them, and builds,
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
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from functools import partial, reduce
from operator import or_
from pathlib import Path
from typing import Callable, Iterable

from . import carry, policy, vqueue, vqueue_records
from .landing import CI, JOB_OF_KIND, SLIM_STEPS, SUITES_KIND, Landing, authed, cancel_runs, gating, merge_thread_post
from .landing import INFRA as INFRA_HANDLE
from .executor import BLOCKED, Executor, QueueTransport, fold_results, task_target
from .queue import (LEAN_SOURCES, MOVE_CHECK, MOVE_SCRIPTS, PR, SIGNED_GUARDS, WORKDIR, DriveFlags, Entry, Member, Queue,
                    QueueError, StoreLabels, _git, _out, _who, base_drivers, conflicting_files, flags_at, forces_cold,
                    label_verifier, lands_on_mark, lean_change, listed, merged_tree, move_of, own_files,
                    priority_list, queue_commit, queue_log, review_owners, rules_at, touched_by, unreviewed)
from .queue import Retired, lacked_moves, lacks_why
from .reds import CIRCUIT_KILLED, CIRCUIT_STEP, RATCHET_STEP, red_owners
from .store import vocab
from .vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Action, Advance, Build, Candidate, Commit, Drop, Eject, Failed,
                     Outcome, Ready, RecordsRun, Red, Regenerate, Rerun, Retry, Ring, Snapshot, Verdict, Wait, _inherited)
from .vqueue_records import LOCK_SUBJECT

#: graph2, infra's executor on the control pod: its argv before the subcommand
GRAPH2 = ("python3.12", "/root/executor/infra/graph2.py", "--db", "/root/executor/graph2.db", "--events", "/root/executor/events")
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
#: pinned Lean PRs only, under no `LEAN` cap: a pin orders a Lean batch or cuts it, and at `--lean 0` only a pinned Lean PR
#: opens one; it puts no batch ahead of a PR changing a Lean input, which goes first (`vqueue._next`)
PINS = Path("/data/drive/pins")
#: how far back the decision log is read
WINDOW_S = 48 * 3600
#: what gates a candidate (`--gate`): graph2's check with the run beside it, or drive's after-landing check alone, as `SHARDS`
#: shards unless `--shards` says another; the second is also the ``gate`` of that check's record (`vqueue_slim.CHECK`)
GATES = GRAPH2_GATE, CHECK_GATE = "graph2", "check"
SHARDS = 3
#: how long after its enqueue the loop still reads, in shadow, the graph2 check of a dropped candidate (`Loop._followed`), and
#: a compare line's ``graph2`` when it stopped reading one that hadn't ended
FOLLOW_S, UNFINISHED = 24 * 3600, "unfinished"
#: the most graph2 checks of dropped candidates the loop follows at once (`Loop._followed`): graph2's slots run them beside the
#: live candidates', so past them the oldest is given up as at `FOLLOW_S`
KEPT_MAX = 2
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
#: who graph2 writes its labels as (its `research data label ... --by executor`): a label by anyone else counts for nothing
GRAPH2_BY = "executor"
#: why a candidate's verdict waits on its base's check (`World._base`, `World._gone`, `World._crossed`; `BaseWait`): graph2
#: hasn't listed it, says its listing failed, listed a graph that can't be read, or hasn't said that it fails what the
#: candidate's failures are inherited from
UNLISTED, BASE_LIST_FAILED, BASE_UNREADABLE = "its check isn't listed yet", "its listing failed", "its graph can't be read"
BASE_UNSAID = "graph2 hasn't said it fails them yet"
#: the waits on a base graph2 may still be getting to, rung only `BASE_RING_S` after its check was first asked for
BASE_COMING = (UNLISTED, BASE_UNSAID)
#: how long after the loop last enqueued a base's check it enqueues it again while a candidate waits on it, and how long after
#: its first enqueue one still not listed, or not said, is rung (`Loop._bases`)
BASE_RETRY_S, BASE_RING_S = 15 * 60, 30 * 60
#: who ejects, as drive's ejections are by
EJECTOR = "ci"
#: a lock commit's trailer naming the files it regenerated (`vqueue_records.lock_message`, which post_train reads alike)
REGENERATED_RE = re.compile(r"^Regenerated: \S", re.M)
#: the most first-parent commits a candidate's walk reads above its base
WALK = 64
#: `vqueue.Failed`'s fields: a verdict fills each it has (`_failure`)
FAILED_FIELDS = frozenset(f.name for f in fields(Failed))
#: the PRs judged at once (`World._judge`): a merge onto `main` takes seconds in drive's clone, and a moved `main` judges every
#: ready PR again
JOBS = 8
#: the most failures of a step off the gate a ring quotes (`Loop._off_gate`)
OFF_SAID = 10
#: the old heads mapped across `main`'s moves at once, each in a job the end of a tick detaches and no tick waits on
#: (`Queue.work_out_mapped`; `--move-carry-jobs`): each runs every move's script on a whole tree, so few, leaving the control
#: pod's few cores room for the loop
MAPPED_JOBS = 2
#: what of graph2's `verdict` JSON the loop reads, each entry's of its lists (`_said`, `_named`, `_disputed`, `_relisted`,
#: `_runs`, `World._held`, `Loop._flaky`): the rest, a line for each task of the check, is dropped once parsed (`_trimmed`)
VERDICT_KEYS = ("state", "why", "pending", "art", "tasks", "off_gate", "labelled")
VERDICT_ENTRIES = {"failed": ("task", "key", "why", "tests", "error"), "infra": ("task", "key", "why", "tree"),
                   "flaky": ("task", "key", "inputs")}
#: where the check gates, how long a candidate's graph2 reading, in shadow, stands, and a followed check's word that it still
#: runs (`Shadow`): each graph2 `verdict` is megabytes and seconds, and a reading settles every task of its graph
G2_READ_S = 600
#: where the check gates, how many candidates' graph2 readings a tick reads afresh when nothing forces it (`Loop._read`):
#: each lists the check's whole graph and pulls each pending task's results, so the rest keep the reading kept of them
G2_FRESH = 1
#: the file beside the log the shadow's readings are kept in between ticks (`Shadow`)
G2_READS = "graph2-reads.json"
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


def _digest(doc: object) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _trimmed(doc: dict) -> dict:
    """graph2's `verdict` JSON as the loop reads it (`VERDICT_KEYS`, `VERDICT_ENTRIES`): a count it reads only as an int is
    kept only as one, and an entry only as a dict."""
    out = {k: doc[k] for k in VERDICT_KEYS if k in doc and (k not in ("pending", "tasks", "off_gate") or isinstance(doc[k], int))}
    for k, keep in VERDICT_ENTRIES.items():
        if isinstance(doc.get(k), list):
            out[k] = [{x: e[x] for x in keep if x in e} for e in doc[k] if isinstance(e, dict)]
    return out


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

    def line(self, n: int, g2: dict) -> str:
        steps = " ".join(f"{k} {s:.1f}/{self.gb[k]:.1f}" for k, s in self.s.items())
        return (f"research queue vqueue: tick {n}: {self.clock() - self.at:.1f} s, peak {self.peak():.1f} GB; s/GB: {steps}; "
                f"graph2 {g2['reads']} read(s) {g2['bytes'] / 1e6:.1f} MB {g2['s']:.1f} s, {g2['cached']} cached")


class Shadow:
    """Where the check gates (`CHECK_GATE`), graph2's readings, in shadow, kept between ticks in ``path`` (`G2_READS`), each
    standing `G2_READ_S` (``read_s``) from its read: a candidate's (`World.verdict`, as `_shadow_doc` keeps it: what
    `Loop._compare`, `Loop._drop`, `Loop._flaky`, `Loop._bases` and the state file read of it), under the key it was read
    with (its base and the results its retries and reruns set aside), and a followed check's word that it still runs
    (`_runs`).  graph2 decides nothing there, so a reading as old as that holds, ejects and advances nothing it wouldn't
    have.  The loop reads one again at once where it acts on it (`Loop._reread`, the judging that finds the check ended,
    a followed check graph2 no longer runs or is read once more); an enqueue, a rerun or a cancel of the check forgets it
    (`forget`).  `save` keeps those this tick read or used."""

    def __init__(self, path: Path, now: Callable[[], float], read_s: float = G2_READ_S,
                 log: Callable[[str], None] = print):
        self.path, self.now, self.read_s, self.used, self.hits = path, now, read_s, set(), 0
        try:
            got = json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, ValueError) as e:
            log(f"research queue vqueue: {path} can't be read ({type(e).__name__}: {e}), so every reading is read afresh")
            got = {}
        self.docs: dict[str, dict] = got if isinstance(got, dict) else {}

    def get(self, kind: str, sha: str, key: str = "", any_age: bool = False) -> dict | None:
        """The reading kept of ``sha`` under ``key``, while it stands (whatever its age, with ``any_age``), else None."""
        d = (self.docs.get(sha) or {}).get(kind)
        if not (isinstance(d, dict) and d.get("key") == key and isinstance(at := d.get("at"), (int, float))
                and (any_age or 0 <= self.now() - at < self.read_s)):
            return None
        self.used.add(sha)
        return d

    def put(self, kind: str, sha: str, doc: dict, key: str = "") -> None:
        self.docs.setdefault(sha, {})[kind] = {**doc, "at": self.now(), "key": key}
        self.used.add(sha)

    def forget(self, sha: str) -> None:
        self.docs.pop(sha, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({s: d for s, d in self.docs.items() if s in self.used}, sort_keys=True) + "\n")
        tmp.replace(self.path)


def _shadow_doc(j: Judged) -> dict:
    """``j``, graph2's reading of a candidate, as `Shadow` keeps it: its verdict, each failure with what `_ended` and `_brief`
    read of it, and what a tick reads of the rest where the check gates."""
    v = j.verdict
    return {"state": v.state, "ended": v.ended,
            "failed": [{k: x for k, x in _failed_doc(f).items() if k != "inputs"} for f in v.failed],
            "infra": [{"task": f.task, "key": f.key} for f in v.infra],
            "mark": j.mark, "note": j.note, "graph2": j.graph2, "lacks": list(j.lacks), "listed": j.listed,
            "flaky": list((j.theirs or {}).get("flaky") or ()) if isinstance(j.theirs, dict) else [],
            "waits_on": [j.waits_on[0], j.waits_on[1].why, j.waits_on[1].detail] if j.waits_on else None,
            "tasks": list(j.tasks) if j.tasks else None}


def _unshadowed(d: dict) -> Judged:
    """A `_shadow_doc` as the `Judged` it keeps."""
    w = d.get("waits_on")
    return Judged(Verdict(d["state"], tuple(_as_failed(f) for f in d.get("failed") or ()),
                          tuple(Failed(str(f.get("task")), str(f.get("key")), frozenset()) for f in d.get("infra") or ()),
                          ended=bool(d.get("ended"))),
                  mark=d.get("mark"), note=d.get("note") or "", graph2=d.get("graph2"), lacks=tuple(d.get("lacks") or ()),
                  listed=bool(d.get("listed", True)), theirs={"flaky": d.get("flaky") or []},
                  waits_on=(w[0], BaseWait(w[1], w[2])) if isinstance(w, list) and len(w) == 3 else None,
                  tasks=tuple(d["tasks"]) if isinstance(d.get("tasks"), list) else None)


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

    def base_enqueues(self, sha: str) -> tuple[float | None, float | None]:
        """When `ready:<sha>` was first asked for, as a base (`Loop._bases`, its enqueue failed or not) or by its build or a
        retry where it is a candidate, and when it was last enqueued."""
        asked = [(at, bool(r.get("enqueued"))) for r in self.lines if r.get("sha") == sha and (
            r.get("action") == "base" or r.get("action") in ("build", "retry") and r.get("enqueued"))
            and (at := unstamp(r.get("at"))) is not None]
        return (asked[0][0] if asked else None), next((at for at, ok in reversed(asked) if ok), None)

    def reruns(self, sha: str) -> int:
        return sum(1 for r in self.of("rerun", sha) if r.get("done") or r.get("absent"))

    def since(self, sha: str) -> dict[str, int]:
        """How many results each key had when a retry of ``sha``, or a rerun of any candidate's, set them aside: its verdict
        reads only those after.  graph2's `rerun` drops a key's cached result for every check that has the key."""
        out: dict[str, int] = {}
        for r in self.lines:
            if r.get("done") and (r.get("action") == "rerun" or r.get("action") == "retry" and r.get("sha") == sha):
                for k, n in (r.get("counts") or {}).items():
                    out[k] = max(out.get(k, 0), n)
        return out

    def builds(self) -> dict[str, dict]:
        """The loop's own candidates, each pushed build's line by its sha (a build whose enqueue failed is one too)."""
        return {r["sha"]: r for r in self.of("build") if r.get("sha")}

    def slim(self, sha: str) -> tuple[bool, dict | None]:
        """Whether the run beside ``sha``'s check was started, and the newest record of it (`vqueue_slim.start`)."""
        recs = self.of("slim", sha)
        return bool(recs), recs[-1].get("record") if recs else None

    def slim_started(self, sha: str) -> tuple[int, float | None]:
        """How many times a run beside ``sha``'s check was started, and when the newest was."""
        starts = [r for r in self.of("slim", sha) if str(r.get("why") or "").startswith("started")]
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

    def followed(self) -> dict[str, dict]:
        """The drops whose candidate's graph2 check was left running, to be compared with its check's verdict (`follow`), by
        sha, but those a compare line has closed since."""
        out: dict[str, dict] = {}
        for r in self.lines:
            if r.get("action") == "drop" and isinstance(r.get("follow"), dict) and r.get("sha"):
                out[r["sha"]] = r
            elif r.get("action") == "compare":
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

    def flaked(self) -> set[str]:
        return {r.get("key") for r in self.of("flaky") if r.get("done") or r.get("shadow")} | self._posting(
            "flaky", lambda r: {r.get("key")})

    def rung(self) -> set[tuple[str, str]]:
        return {(r["sha"], r.get("why")) for r in self.of("ring") if r.get("done") or r.get("shadow")} | self._posting(
            "ring", lambda r: {(r.get("sha"), r.get("why"))})

    def told(self, kind: str = "skip") -> set[str]:
        """Each `head@base` a message of ``kind`` told its owner of: a skip's (`Loop._merged`), or a move a head lacks
        (`lacks`, `Loop._lacking`)."""
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


@dataclass(frozen=True)
class BaseWait:
    """Why a candidate's verdict waits on its base's check where the loop acts on it (`Loop._bases`): ``why`` is `UNLISTED`,
    `BASE_LIST_FAILED`, `BASE_UNREADABLE` or `BASE_UNSAID`, ``detail`` what graph2's verdict or the read of the graph said,
    or the tasks graph2 hasn't said fail there."""
    why: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.why} ({self.detail})" if self.detail else self.why


@dataclass(frozen=True)
class Dispute:
    """A task graph2's `verdict` and the loop's own reading disagree on: on a candidate's check, graph2 says it ended ``said``
    (fail or infra) at its key ``theirs`` and the loop read it passed at the key it read, ``ours`` (`_disputed`); on its
    ``base``'s, the loop reads it failed there at ``ours``, so that the candidate's failure of it is inherited, and graph2's
    verdict on the base says ``said`` of it at ``theirs`` (fail at another key, or infra), or names no fail of it (both
    empty: `World._crossed`).  Its candidate is held until the two agree, and `CI` is rung (`Loop._disputes`)."""
    task: str
    said: str
    theirs: str
    ours: str
    base: str | None = None

    def __str__(self) -> str:
        if self.base is None:
            return f"graph2 says {self.task} {self.said} at {self.theirs}, and the loop reads it {PASS} at {self.ours}"
        says = f"says {self.task} {self.said} at {self.theirs}" if self.said else f"names no fail of {self.task}"
        return f"graph2's verdict on its base {self.base[:12]} {says}, and the loop reads it {FAIL} there at {self.ours}"

    def logged(self) -> dict:
        """What a ring about it logs beside its text: the task, both readings at both keys, and the base's check."""
        return dict(task=self.task, graph2={"state": self.said or None, "key": self.theirs or None},
                    mine={"state": PASS if self.base is None else FAIL, "key": self.ours},
                    **({} if self.base is None else {"check": f"ready:{self.base}"}))


@dataclass(frozen=True)
class Relisted:
    """A candidate's listing the loop read, ``ours`` (the `verity_graph` label on its commit, ``listed`` tasks), that isn't the
    one graph2's verdict says it runs, ``theirs`` (``tasks`` tasks, where it counts them; ``lacks`` those of its entries that
    ``ours`` doesn't list): that label is self-asserted too, and a listing that leaves out a gating task hides it
    (`_relisted`).  Its candidate is held until they agree, and `CI` is rung (`Loop._disputes`)."""
    theirs: str
    ours: str
    tasks: int | None
    listed: int
    lacks: tuple[str, ...] = ()

    def __str__(self) -> str:
        n = f" ({self.tasks} task(s))" if self.tasks is not None else ""
        lacks = f", which lacks {', '.join(self.lacks)}" if self.lacks else ""
        return f"graph2 runs the listing {self.theirs}{n}, and the loop read {self.ours} ({self.listed} task(s)){lacks}"

    def logged(self) -> dict:
        """What a ring about it logs beside its text: both listings, with how many tasks each holds, and what ours lacks."""
        return dict(graph2={"art": self.theirs, "tasks": self.tasks}, mine={"art": self.ours, "tasks": self.listed},
                    lacks=list(self.lacks))


@dataclass
class Judged:
    """A candidate's verdict with what it rests on: ``mark`` names the terminal infra or list-failed evidence a retry answers,
    ``graph2`` is graph2's own verdict (``theirs`` its whole JSON), ``lacks`` the jobs `main` waits for (`landing.gating`) that no
    task of the graph runs, which drive's `executor = "on"` path takes as no promotion and this verdict as list-failed, and
    `kind K` of each kind of task its base's graph lists that it lists none of (`World._gone`), ``waits_on`` the base whose
    check its verdict waits on and why, where the loop acts on it (`BaseWait`), ``disputed`` what it is held on: the tasks
    graph2's verdict says failed or erred and the loop read passed, or the loop reads its base failing and graph2's verdict on
    the base doesn't (`Dispute`), or a listing the loop read that isn't the one graph2 runs (`Relisted`), and ``slim`` the
    verdict of the run beside the check (`vqueue_slim.judge` on its ``record``; pending until it starts), combined into
    ``verdict``.  Where the check gates (`CHECK_GATE`), ``verdict`` is that run's alone and ``shadow`` the verdict read of
    graph2's results.  ``tasks`` is how many of its graph's tasks on the merge gate have ended, and how many there are; None
    with no graph read."""
    verdict: Verdict
    mark: str | None = None
    note: str = ""
    graph2: str | None = None
    lacks: tuple[str, ...] = ()
    #: whether graph2 listed it, or said its listing failed
    listed: bool = True
    theirs: dict | None = None
    slim: Verdict | None = None
    shadow: Verdict | None = None
    record: dict | None = None
    waits_on: tuple[str, BaseWait] | None = None
    disputed: tuple[Dispute | Relisted, ...] = ()
    tasks: tuple[int, int] | None = None


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


class _Seen:
    """A transport that keeps each key's results as `Executor.settle` reads them (an infra error folds to pending there), each
    key's from its ``since``-th on: a retry or a rerun set the earlier ones aside.  A key of ``said`` ({key: the fail or the
    infra error graph2's verdict says it ended in}, `_said`) reads so whatever its labels say: a fail with no fail among them
    gets that fail, and an infra error loses its passes and gets that error unless it ends in one."""

    def __init__(self, transport, since: dict[str, int] | None = None, said: dict[str, dict] | None = None):
        self.transport, self.since, self.said, self.got = transport, since or {}, said or {}, {}

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        if isinstance(self.transport, Transport):
            got = self.transport.results(keys, self.since)
        else:
            got = {k: v[self.since.get(k, 0):] for k, v in self.transport.results(keys).items()}
        for k in keys:
            if (doc := self.said.get(k)) is None:
                continue
            have = list(got.get(k) or ())
            if doc["status"] == FAIL and fold_results(have, k)[0] != FAIL:
                got[k] = [*have, doc]
            elif doc["status"] != FAIL:
                have = [d for d in have if not (isinstance(d, dict) and d.get("status") == PASS)]
                got[k] = have if _infra(have, k) else [*have, doc]
        self.got.update(got)
        return got


class Transport(QueueTransport):
    """`QueueTransport`, pulling a key's results from the store's remote again while none of them passed or failed: graph2 labels
    an infra error as a result, and a retry's or a rerun's result comes after it.  With ``since``, each key's from its
    ``since``-th on, so one set aside reads as no result and is pulled again.  Only graph2's labels count (`GRAPH2_BY`): its
    results, and the graph it listed."""

    by = GRAPH2_BY

    def results(self, keys: list[str], since: dict[str, int] | None = None) -> dict[str, list[object]]:
        from .queue import PREFETCH_JOBS
        from .store import labels_sync, vocab

        since = since or {}

        def have(k: str) -> list[object]:
            return [lab.value for lab in self._labels(task_target(k), vocab.TASK_RESULT)][since.get(k, 0):]

        got = {k: have(k) for k in keys}
        if (todo := [k for k, v in got.items() if fold_results(v, k)[0] == PENDING]) \
                and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [task_target(k) for k in todo], jobs=PREFETCH_JOBS)
            got.update({k: have(k) for k in todo})
        return got


class _Clone(Queue):
    """The queue in a clone it doesn't bring to the remote's state (drive's, or the loop's own): the loop fetches into `NS`, and
    `main` as fetched is its `MAIN`, not drive's `refs/remotes/origin/main`."""

    def _clone(self) -> None:
        return None

    @property
    def base(self) -> str:
        return MAIN


class World:
    """What the loop reads and changes: git in the clone and on origin, graph2, the labels, the PRs and the merge thread.  Each
    effect is one method, so a test fakes what it doesn't run for real.  With ``labels_at`` (a new `StoreLabels`), each tick
    reads the labels afresh (`_relabel`).  ``cache`` is where it keeps the heads' own changes and the issuer's keys, not in
    drive's files.  ``on`` names the machines the run beside each check takes, in place of drive.toml's `on`."""

    def __init__(self, q: Queue, labels, tasks: Executor, *, graph2: tuple[str, ...] = (), priority_file: Path | None = None,
                 post: Callable[[str, tuple[str, ...]], bool] | None = None, now: Callable[[], float] = time.time,
                 labels_at: Callable[[], StoreLabels] | None = None, cache: Path | None = None, on: str = "",
                 frozen_file: Path | None = None, deselect_file: Path | None = None, wide_file: Path | None = None,
                 pins_file: Path | None = None, keep_file: Path | None = None, heavy_file: Path | None = None,
                 heavy_on: str = "", clock: Callable[[], float] = time.monotonic):
        self.q, self.labels, self.tasks, self.graph2, self.on = q, labels, tasks, tuple(graph2), on
        self.priority_file, self.now, self.repo, self.labels_at = priority_file, now, q.workdir, labels_at
        #: graph2's readings kept between ticks where the check gates (`Shadow`, which `Loop` sets), and this tick's reads of
        #: graph2's `verdict`: how many, their bytes and seconds on ``clock``
        self.reads: Shadow | None = None
        self.clock, self.g2 = clock, {"reads": 0, "bytes": 0, "s": 0.0}
        self.frozen_file, self._frozen_told, self.wide_file = frozen_file, set(), wide_file
        self.pins_file, self.keep_file = pins_file, keep_file
        #: the heavy PRs' file (`HEAVY`) and the machines their candidates' heavy runs take (`heavy`); "" turns it off
        self.heavy_file, self.heavy_on = heavy_file, heavy_on
        #: the tests each candidate's check leaves out (`DESELECT`, `vqueue_slim.deselection`)
        self.deselect_file = deselect_file
        self.keys = cache / "issuer-keys.json" if cache is not None else None
        #: the moves regenerated on candidates of the `main` they landed on, as `Queue._made` holds them (`_keep_moves`)
        self._made_at = cache / "moves-made.json" if cache is not None else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            q._kept_at, q._mapped_at = cache / "own-changes.json", cache / "mapped-changes.json"
            q._kept = json.loads(q._kept_at.read_text()) if q._kept_at.is_file() else {}
            q._made = json.loads(self._made_at.read_text()) if self._made_at.is_file() else {}
        self.post: Callable[[str, tuple[str, ...]], bool] = post or merge_thread_post
        #: graph2's subcommands it was found to lack (`_graph2`)
        self.absent: set[str] = set()
        #: by (head, `main`): its files (`_changed`), and (lean, conflict) (`_judge`)
        self._files: dict[tuple[str, str], frozenset[str]] = {}
        self._judged: dict[tuple[str, str], tuple[bool, bool]] = {}
        #: by (head, `main`), the moves it lacks that its change names (`_lacks`); by (base, map), what a move's map retires
        self._lacked: dict[tuple[str, str], list[tuple[str, str, list[str] | None]]] = {}
        self._retires: dict[tuple[str, str], Retired | None] = {}
        #: each `carry.delta` judged, by its arguments (`_delta`)
        self._deltas: dict[tuple[str, str, tuple[str, ...], frozenset[str]], carry.Delta] = {}
        #: what the last `ready` decided of the verdict carry (`_carry`): {number: its head, the older head, why, whether it
        #: carries}
        self.carried: dict[int, dict] = {}
        #: under drive.toml's `signed_verdicts = "shadow"`, the labels' `StoreLabels.unauthenticated`, filled as they are read
        self.unauthenticated: dict[tuple, str] = {}
        self._prs: dict[int, PR] | None = None
        self._driver: Landing | None = None
        self._slim: Landing | None = None
        self._main: str | None = None
        #: while a snapshot reads the candidates' verdicts (`Loop.snapshot`), graph2's verdict of each sha read (`_verdict_of`)
        #: and the graph graph2 listed of each check read (`_listing`)
        self._theirs: dict[str, dict | None] | None = None
        self._listed: dict[str, tuple[str, dict] | None] | None = None
        #: what the last `ready` read: {number: the PR at its ready head}, {number: (head, why it isn't ready)} of each other
        #: open PR into `main`, and those of them taken from the queue by others than their owners (`Snapshot.pulled`)
        self.prs: dict[int, PR] = {}
        self.left: dict[int, tuple[str, str]] = {}
        self.pulled: set[int] = set()
        #: of those left out for a move they lack whose names their change names: {number: (head, the last such move, why)}
        self.lacking: dict[int, tuple[str, str, str]] = {}
        #: the move PRs the last `ready` took on their ready marks alone (`_uncertified`), and {number: what the last `build`
        #: regenerated of each move PR: its map, tip, commit and `move_check.py --json` report} (`_move`)
        self.on_mark: set[int] = set()
        self.certified: dict[int, dict] = {}

    def log(self, text: str) -> None:
        self.q.log(text)

    # -- reads

    def fetch(self) -> str:
        """origin's `main`, fetched with origin's `queue/*` branches into `NS`."""
        self._prs, self.q._registry = None, None
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

    def ready(self, main: str) -> list[Ready]:
        """The open PRs drive would land from `next` now, as `vqueue.Ready`: a move PR with a land verdict, or, while drive.toml's
        `lowrisk_lane` is on, on its mark alone once its move is certified on `main` (`_uncertified`, `_off_mark`), never under
        the low-risk rule (`queue.lands_on_mark`).  Why each other open PR into `main` isn't one goes to `left`:
        the queue's word on one it doesn't admit (`Entry.why`; a withdrawn mark's author), else a hold and by whom, no land
        verdict, the freeze, a move it lacks whose retired names its own change names (`lacked_moves`, also to `lacking`), or
        an untaken PR whose head it holds; those others than their owners took go to `pulled`."""
        q, labels = self.q, self.labels
        prs = [p for p in self.open().values() if p.base == "main"]
        if missing := [p for p in prs if p.head in self._missing(p.head for p in prs)]:
            q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "origin",
                      *(f"+refs/pull/{p.number}/head:{NS}/pr/{p.number}" for p in missing))
        gone = self._missing(p.head for p in prs)
        left = {p.number: (p.head, "its head can't be fetched" if p.head in gone else "its head is on `main`")
                for p in prs if p.head in gone or self._ancestor(p.head, main)}
        if any(p.move and p.number not in left for p in prs):
            landed_heads = queue_log(self.repo, main)[0]
            left.update((p.number, (p.head, "its move, regenerated from its head, is on `main`")) for p in prs
                        if p.move and p.number not in left and p.head in landed_heads)
        prs = [p for p in prs if p.number not in left]
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
        self._judged = {k: v for k, v in self._judged.items() if k[1] == main}
        self._deltas = {k: v for k, v in self._deltas.items() if k[2] == landed}
        admitted = [n for n, e in entries.items() if e.state == "ready"]
        reviews = {n: labels.reviews(n, entries[n].pr.head) for n in admitted}
        why = {n: unreviewed(reviews[n],
                             review_owners(q, labels, n, entries[n].pr.head, entries[n].pr.owner, self.open()))
               for n in admitted}
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
            for n in admitted:
                p, held = entries[n].pr, (why[n] or "").startswith("held")
                if why[n] is None or n in carried or lands_on_mark(flags, p, why[n], files[n]):
                    took.append(n)
                elif held:
                    left[n] = (p.head, why[n])
                    if (unreviewed([r for r in reviews[n] if self._another(n, r.by)], ()) or "").startswith("held"):
                        pulled.add(n)
                elif p.move and not off_mark[n]:
                    took.append(n)
                    self.on_mark.add(n)
                else:
                    left[n] = (p.head, f"{why[n]}, and " + (off_mark[n] if p.move else "it isn't low-risk"))
            specs, clears = self._frozen()
            if specs:
                below = self._frozen_base(main)
                frozen = [n for n in took if not self._cleared(n, entries[n].pr.head, clears)
                          and self._frozen_out(below, n, entries[n].pr.head, specs)]
                left.update((n, (entries[n].pr.head, f"it waits for the freeze to end ({self.frozen_file})")) for n in frozen)
                took, pulled = [n for n in took if n not in frozen], pulled | set(frozen)
            # a plain merge of a head below a move leaves its own lines naming what the move retired: such a head lands
            # crossed, and one whose own change names none of it crosses by that merge (`lacked_moves`)
            lacks = dict(zip(took, pool.map(lambda n: self._lacks(main, entries[n].pr.head, files[n]), took)))
            left.update((n, (entries[n].pr.head, lacks_why(m))) for n, m in lacks.items() if m)
            self.lacking = {n: (entries[n].pr.head, m[-1][0], lacks_why(m)) for n, m in lacks.items() if m}
            took = [n for n in took if not lacks[n]]
            new = any((entries[n].pr.head, main) not in self._judged for n in took)
            with base_drivers(self.repo, main) if new else contextlib.nullcontext(None) as cfg:   # once a tick, not a PR
                chosen = {n: (files[n], *j)
                          for n, j in zip(took, pool.map(lambda n: self._judge(main, entries[n].pr.head, cfg), took))}
        # a land verdict covers its PR's own change: a PR holding the head of an admitted PR that isn't taken (no land verdict
        # there, or held) would carry that change unreviewed, unless the head landed on `checked` under drive's rules
        while out := {n: a for n in chosen for a in entries[n].after if a in entries and entries[a].state == "ready"
                      and a not in chosen and not (old and self._ancestor(entries[n].held[a], old))}:
            left.update((n, (entries[n].pr.head, f"it holds {entries[n].holds(a)}, which isn't taken"
                                                 + (f": {left[a][1]}" if a in left else ""))) for n, a in out.items())
            chosen = {n: v for n, v in chosen.items() if n not in out}
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

    def verdict(self, sha: str, base: str = "", since: dict[str, int] | None = None,
                base_since: dict[str, int] | None = None) -> Judged:
        """Candidate ``sha``'s verdict from graph2's results (see the module's docstring), each key's from its ``since``-th on,
        beside graph2's own, each failed task with its key and failing tests in its base ``base``'s graph (`_base`), read with
        ``base_since``, as graph2's verdict on the base bears them out (`_crossed`)."""
        check = f"ready:{sha}"
        theirs = self._verdict_of(sha)
        said = theirs.get("state") if isinstance(theirs, dict) else None
        try:
            got = self._listing(check)
        except ValueError as e:
            return Judged(Verdict(LIST_FAILED), mark=f"unreadable {_digest(str(e))}", note=str(e), graph2=said, theirs=theirs)
        if got is None:
            if said == LIST_FAILED:
                return Judged(Verdict(LIST_FAILED), mark=f"list-failed {_digest(theirs)}", note=str(theirs.get("why") or ""),
                              graph2=said, theirs=theirs)
            return Judged(Verdict(PENDING), note="not listed yet", graph2=said, listed=False, theirs=theirs)
        listing, graph = got
        tasks, envs = {t["id"]: t for t in graph["tasks"]}, graph.get("envs") or {}
        if (relisted := _relisted(theirs, listing, tasks)) is not None:
            says = [f"graph2 says {said}"] if said in vqueue.STATES and said != PENDING else []
            return Judged(Verdict(PENDING), note="; ".join([*says, f"held until graph2 and the loop agree: {relisted}"]),
                          graph2=said, theirs=theirs, disputed=(relisted,))
        if isinstance(pol := policy.at(self.repo, sha), str):
            self.log(f"research queue vqueue: {pol}: every job gates at {sha[:12]}")
            pol = policy.Policy()
        gates = [i for i, t in tasks.items() if t.get("gate", True) is not False
                 and _job(t, tasks) not in (RATCHET_STEP, *pol.off_gate)]
        if not gates:
            return Judged(Verdict(LIST_FAILED), mark=f"empty {_digest(sorted(tasks))}", note="no task on the merge gate",
                          graph2=said, theirs=theirs)
        states, keys, seen = self._settled(graph, since, _said(theirs))
        closure = _Closure(tasks, envs)
        needed = _walk(tasks, gates)
        allowed = self._allowed(sha, base, tasks, states, needed)
        failed = frozenset(i for i in needed if states[i][0] == FAIL and i not in allowed)
        bad: list[str] = []
        for i in gates:
            if i in failed:
                bad.append(i)
            elif states[i][0] == BLOCKED:
                bad += closure.fails(i, failed) or [i]
        infra = [i for i in needed if states[i][0] == PENDING and _infra(seen.got.get(keys[i]) or (), keys[i])]
        erred = frozenset(infra)
        held, disputed = self._held(theirs), _disputed(theirs, needed, states, keys)
        ran = all(states[i][0] != PENDING or closure.meets(i, erred) for i in gates)
        passed = all(states[i][0] == PASS or i in allowed for i in gates)
        ended = ran and held is None
        jobs = {_job(t, tasks) for t in tasks.values()}
        lacks = tuple(s for s in gating(self.repo, sha) if s not in jobs and s not in SLIM_STEPS)
        gone = self._gone(base, tasks, pol)
        based = self._base(base, set(bad), base_since) if bad else {}
        waits = based if not isinstance(based, dict) else gone if isinstance(gone, BaseWait) else None
        based, gone = based if isinstance(based, dict) else {}, () if isinstance(gone, BaseWait) else gone
        missing = (*lacks, *(f"kind {k}" for k in gone))
        tree = {e.get("task"): bool(e.get("tree")) for e in (theirs or {}).get("infra") or () if isinstance(e, dict)}
        named = _named(theirs)
        fail = () if waits is not None else tuple(
            _failure(i, keys[i], closure.inputs(i), base_key=based.get(i, (None, None))[0],
                     tests=_tests(states[i][1], keys[i], tasks[i]["kind"]) | named.get((i, keys[i]), frozenset()),
                     base_tests=based.get(i, (None, None))[1]) for i in dict.fromkeys(bad))
        fail, crossed, unsaid = self._crossed(base, fail)
        if unsaid is not None:
            fail, waits = (), unsaid
        disputed = (*disputed, *crossed)
        down = tuple(_failure(i, keys[i], closure.inputs(i), tree=tree.get(i, False)) for i in infra)
        if disputed:
            mine = Verdict(PENDING)
        elif fail and not (missing and all(_inherited(f) for f in fail)):    # a fail of its own ejects all the same
            mine = Verdict(FAIL, fail, down, ended=ended)
        elif missing:
            mine = Verdict(LIST_FAILED)
        elif waits is not None:
            mine = Verdict(PENDING)
        elif down:
            mine = Verdict(INFRA, (), down)
        elif passed and held is None:
            mine = Verdict(PASS)
        else:
            mine = Verdict(PENDING)
        mark = (f"lacks {' '.join(missing)}" if mine.state == LIST_FAILED else
                f"infra {_digest(sorted((keys[i], len(seen.got.get(keys[i]) or ())) for i in infra))}" if mine.state == INFRA
                else None)
        note = "; ".join([*([f"graph2 says {said}"] if said in vqueue.STATES and said != mine.state else []),
                          *([f"no task of {', '.join(lacks)}, which `main` waits for"] if lacks else []),
                          *([f"no task of kind {', '.join(gone)}, which its base lists"] if gone else []),
                          *([f"waiting for its base {base[:12]}: {waits}"] if waits is not None else []),
                          *([f"held until graph2 and the loop agree: {'; '.join(map(str, disputed))}"] if disputed else []),
                          *([f"held: {held}"] if held is not None and (mine.state == PENDING and passed
                                                                       or mine.state == FAIL and ran) else []),
                          *([f"allowed: {', '.join(sorted(allowed))}"] if allowed else [])])
        return Judged(mine, mark=mark, note=note, graph2=said, lacks=missing, theirs=theirs,
                      waits_on=(base, waits) if isinstance(waits, BaseWait) else None, disputed=disputed,
                      tasks=(sum(states[i][0] != PENDING for i in gates), len(gates)))

    def _gone(self, base: str, tasks: dict[str, dict], pol: policy.Policy) -> tuple[str, ...] | BaseWait:
        """The kinds of task on the merge gate in ``base``'s graph, as graph2 listed it, of which a candidate's ``tasks`` list
        none: a family whose lister is gone lists no task and no error.  A kind may go where its job (`_job`) is now a kind's it
        wasn't there, as lean-suites went to the test files of the suites that run on lean-build's outputs.  Why not yet while
        `ready:<base>` isn't listed (`UNLISTED`); none for no base, or one whose listing failed or whose graph can't be read
        (logged), so a candidate that mends a listing `main` fails can pass."""
        if not base:
            return ()
        try:
            got = self._listing(f"ready:{base}")
        except ValueError as e:
            self.log(f"research queue vqueue: {base[:12]}'s graph can't be read, so no kind it lists is missed: {e}")
            return ()
        if got is None:
            if isinstance(theirs := self._verdict_of(base), dict) and theirs.get("state") == LIST_FAILED:
                self.log(f"research queue vqueue: ready:{base[:12]}'s listing failed, so no kind it lists is missed")
                return ()
            return BaseWait(UNLISTED)

        def kinds(ts: dict[str, dict]) -> dict[str, set[str]]:
            out: dict[str, set[str]] = {}
            for t in ts.values():
                if t.get("gate", True) is not False and (j := _job(t, ts)) not in (RATCHET_STEP, *pol.off_gate):
                    out.setdefault(j, set()).add(t["kind"])
            return out

        was, now = kinds({t["id"]: t for t in got[1]["tasks"]}), kinds(tasks)
        return tuple(sorted(k for j, ks in was.items() if not now.get(j, set()) - ks for k in ks - now.get(j, set())))

    def _held(self, theirs: dict | None) -> str | None:
        """Why graph2's own verdict, from its database, keeps a candidate from passing (or from ending with only inherited
        failures), which the loop's is never more passing than: it can't be read, says pending, cancelled or list-failed,
        counts a task pending, which no label passes, or names no listing it labelled (`art`: graph2 runs a check it listed
        before it labels the listing), so none the loop read is one it runs (`_relisted`).  Its pending count names no task,
        so one off the gate holds a candidate too; each failed or infra task it names reads so on its own (`_said`), and one
        the loop read passed at another key holds it (`_disputed`).  None with no graph2 `verdict` to ask (its
        `ready_verdict` label, the fallback, is written only once nothing is pending)."""
        if not self.graph2 or "verdict" in self.absent:
            return None
        if not isinstance(theirs, dict):
            return "graph2's verdict can't be read"
        if (state := theirs.get("state")) not in (PASS, FAIL, INFRA):
            return f"graph2 says {state}"
        if isinstance(n := theirs.get("pending"), int) and n > 0:
            return f"graph2 counts {n} task(s) pending"
        if "art" in theirs and not theirs["art"]:
            return "graph2 hasn't labelled the listing it runs yet"
        return None

    def _allowed(self, sha: str, base: str, tasks: dict[str, dict], states: dict[str, tuple], ids: Iterable[str]) -> set[str]:
        """The failed tasks of ``ids`` of candidate ``sha`` an allowance of its base ``base``'s policy (``sha``'s, with no
        base) covers, each red record they name at ``sha`` (`Landing._task_units`, `Landing._task_allowed`): each is taken as
        passed."""
        out = set()
        for i in ids:
            if states[i][0] == FAIL:
                d, res = self.driver(), states[i][1] or {}
                if all(d._task_allowed(sha, tasks[i], res, row, at=base).covered
                       for row in d._task_units(sha, tasks[i], res).values()):
                    out.add(i)
        return out

    def _settled(self, graph: dict, since: dict[str, int] | None = None, said: dict[str, dict] | None = None
                 ) -> tuple[dict[str, tuple], dict[str, str], _Seen]:
        """``graph``'s tasks settled on graph2's results (`Executor.settle`), each key's from its ``since``-th on, a key of
        ``said`` read as graph2's verdict says it ended (`_Seen`), each one's key, the graph's or one made from its reads (""
        while they haven't all passed), and the results read."""
        seen = _Seen(self.tasks.transport, since, said)
        keyed: dict[str, str] = {}
        states = Executor(seen).settle(graph, keyed)
        return states, {t["id"]: keyed.get(t["id"], "") for t in graph["tasks"]}, seen

    def _base(self, base: str, ids: set[str], since: dict[str, int] | None = None
              ) -> dict[str, tuple[str | None, frozenset[str] | None]] | str | BaseWait:
        """For each of tasks ``ids`` in ``base``'s graph, as graph2 listed it: its key there, where one is made (graph2 keys a
        task by all it reads, so a candidate's task at its base's key fails as it did there: `vqueue.Failed.base_key`), and its
        failing tests there, None unless it failed naming them, in its result or in graph2's verdict on ``base`` (read once a
        tick: `_named`; `_crossed` keeps the verdict's alone), each a test id of that task's (`vqueue.Failed.base_tests`).  Why
        not yet, while that is to come: a task there hasn't ended, its key waiting on what it reads or its result to come,
        unless it or what it reads there ended in an infra error; or `ready:<base>` has no graph to read the keys from (a
        `BaseWait`, which the loop acts on: `Loop._bases`): it isn't listed (`UNLISTED`), graph2 says its listing failed
        (`BASE_LIST_FAILED`), or the graph it listed can't be read (`BASE_UNREADABLE`).  No failure is the candidate's own until
        then: one `main` has too would be blamed on each candidate built on it."""
        if not base:
            return {}
        try:
            got = self._listing(f"ready:{base}")
        except ValueError as e:
            return BaseWait(BASE_UNREADABLE, str(e)[-200:])
        theirs = self._verdict_of(base)
        if got is None:
            if isinstance(theirs, dict) and theirs.get("state") == LIST_FAILED:
                return BaseWait(BASE_LIST_FAILED, str(theirs.get("why") or "")[-200:])
            return BaseWait(UNLISTED)
        states, keys, seen = self._settled(got[1], since, _said(theirs))
        erred = frozenset(i for i, k in keys.items() if k and _infra(seen.got.get(k) or (), k))
        closure = _Closure({t["id"]: t for t in got[1]["tasks"]}, got[1].get("envs") or {})
        open_ = [i for i in ids if i in keys and states[i][0] == PENDING and not closure.meets(i, erred)]
        if waiting := sorted(i for i in open_ if not keys[i]):
            return f"{', '.join(waiting)} there still wait on what they read"
        if running := sorted(open_):
            return f"{', '.join(running)} there haven't ended"
        named = _named(theirs)
        return {i: (keys[i] or None, (_tests(states[i][1]) | named.get((i, keys[i]), frozenset()) or None)
                    if states[i][0] == FAIL else None) for i in ids if i in keys}

    def _crossed(self, base: str, fail: tuple[Failed, ...]
                 ) -> tuple[tuple[Failed, ...], tuple[Dispute, ...], BaseWait | None]:
        """``fail`` checked against graph2's own verdict on ``base``'s check, as `_disputed` checks a candidate's: a fail there
        counts toward a failure's being inherited (`vqueue._inherited`) only while graph2's verdict lists its task failed (a
        fail, or its own fault) at the key the loop read there, and then with the failing tests graph2 names of it (`_named`),
        not a label's: a result labelled `--by executor` is self-asserted, and a key's first fail is the one read
        (`fold_results`).  One inherited on the loop's reading alone is held on (a `Dispute`) once graph2 says of the task
        otherwise there (fail at another key, or infra) or its verdict on the base can be read and has ended
        (`_held`) without naming it, and waits on the base (`BASE_UNSAID`) until then: the base's result is still to come.
        As they were with no graph2 `verdict` to ask."""
        if not fail or not base or not self.graph2 or "verdict" in self.absent:
            return fail, (), None
        theirs = self._verdict_of(base)
        said, named, ended = _said(theirs), _named(theirs), self._held(theirs) is None
        of = {d["task"]: (FAIL if d["status"] == FAIL else INFRA, k) for k, d in said.items()}
        out, disputed, unsaid = [], [], []
        for f in fail:
            if (d := said.get(f.base_key or "")) is not None and d["status"] == FAIL and d["task"] == f.task:
                out.append(replace(f, base_tests=named.get((f.task, f.base_key)) or None))
                continue
            out.append(f)
            if not _inherited(f):
                continue
            if f.task in of or ended:
                disputed.append(Dispute(f.task, *of.get(f.task, ("", "")), f.base_key or "", base))
            else:
                unsaid.append(f.task)
        return tuple(out), tuple(disputed), BaseWait(BASE_UNSAID, ", ".join(unsaid)) if unsaid else None

    def counts(self, keys: Iterable[str]) -> dict[str, int]:
        """How many results each of graph2's ``keys`` has (`sha512:…`; a key of the run beside the check has none)."""
        keys = [k for k in dict.fromkeys(keys) if k.startswith("sha512:")]
        return {k: len(v) for k, v in self.tasks.transport.results(keys).items()} if keys else {}

    def rerun(self, sha: str, keys: Iterable[str]) -> str | None:
        """graph2's `rerun` of ``keys`` on `ready:<sha>`, as said; None when graph2 has no `rerun`."""
        self._forget(sha)
        if (r := self._graph2("rerun", "--check", f"ready:{sha}", "--keys", *keys)) is None:
            return None
        if r.returncode:
            raise QueueError(f"graph2 rerun ready:{sha[:12]} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout.strip()

    def failing_tests(self, key: str) -> frozenset[str]:
        """The tests ``key``'s newest failed result names, those a rerun set aside included."""
        docs = self.tasks.transport.results([key]).get(key) or ()
        return next((t for d in reversed(list(docs)) if isinstance(d, dict) and d.get("status") == FAIL
                     and (t := _tests(d))), frozenset())

    def _listing(self, check: str) -> tuple[str, dict] | None:
        """`Executor.graph_v2` of ``check``, read once while a snapshot keeps them (`_listed`): a candidate's graph is the base
        of the one above it too."""
        if self._listed is None:
            return self.tasks.graph_v2(check)
        if check not in self._listed:
            self._listed[check] = self.tasks.graph_v2(check)
        return self._listed[check]

    @contextlib.contextmanager
    def one_read(self):
        """A `_listed` of its own inside: the graphs a read lists inside (a candidate's and its base's) are dropped once it
        ends, so the loop holds one read's at a time: a check's graph of thousands of tasks and millions of reads is
        gigabytes in memory."""
        kept, self._listed = self._listed, {}
        try:
            yield
        finally:
            self._listed = kept

    def _verdict_of(self, sha: str) -> dict | None:
        """`graph2_verdict` of ``sha``, read once while a snapshot keeps them (`_theirs`): a subprocess each, and a base's is
        read for each candidate on it that failed or waits on its listing."""
        if self._theirs is None:
            return self.graph2_verdict(sha)
        if sha not in self._theirs:
            self._theirs[sha] = self.graph2_verdict(sha)
        return self._theirs[sha]

    def graph2_verdict(self, sha: str) -> dict | None:
        """graph2's own verdict on `ready:<sha>`: its `verdict` command's JSON, else its `ready_verdict` label's, else None, as
        the loop reads it (`_trimmed`)."""
        t0 = self.clock()
        if (r := self._graph2("verdict", "--check", f"ready:{sha}")) is not None:
            self.g2["reads"] += 1
            self.g2["bytes"] += len(r.stdout or "")
            self.g2["s"] += self.clock() - t0
            if r.returncode:
                self.log(f"research queue vqueue: graph2 verdict ready:{sha[:12]} exited {r.returncode}: {r.stderr.strip()[-300:]}")
                return None
            try:
                got = json.loads(r.stdout)
            except ValueError as e:
                self.log(f"research queue vqueue: graph2 verdict ready:{sha[:12]} printed no JSON ({e}): {r.stdout.strip()[-300:]}")
                return None
            return _trimmed(got) if isinstance(got, dict) else None
        store = getattr(self.tasks.transport, "store", None)
        if store is None:
            return None

        def ours() -> list:
            return [lab for lab in store.labels(f"commit:{sha}", key=vocab.READY_VERDICT) if lab.by == GRAPH2_BY]

        labs = ours()
        if not labs and getattr(store, "remote", None) is not None:
            from .store import labels_sync

            labels_sync.pull_targets(store, [f"commit:{sha}"])
            labs = ours()
        if not labs:
            return None
        v = labs[-1].value
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError as e:
                self.log(f"research queue vqueue: commit:{sha[:12]}'s {vocab.READY_VERDICT} label holds no JSON ({e})")
                return None
        # each verdict its own, for a retry's mark
        return _trimmed({"labelled": labs[-1].ts, **v}) if isinstance(v, dict) else None

    def _graph2(self, *args: str) -> subprocess.CompletedProcess | None:
        """graph2 run with ``args``; None when it has no such subcommand (argparse's `invalid choice`), or there is no graph2."""
        if not self.graph2 or args[0] in self.absent:
            return None
        r = subprocess.run([*self.graph2, *args], cwd=self.repo, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode == 2 and "invalid choice" in r.stderr:
            self.absent.add(args[0])
            self.log(f"research queue vqueue: graph2 has no `{args[0]}`")
            return None
        return r

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

    def _forget(self, sha: str) -> None:
        if self.reads is not None:
            self.reads.forget(sha)

    def enqueue(self, sha: str, priority: int) -> None:
        self._forget(sha)
        try:
            self.tasks.enqueue_graph(f"ready:{sha}", priority)
        except RuntimeError as e:
            raise QueueError(f"enqueue ready:{sha[:12]}: {e}") from None

    def cancel(self, sha: str) -> str:
        """graph2's `cancel` of `ready:<sha>`, as said."""
        self._forget(sha)
        if (r := self._graph2("cancel", "--check", f"ready:{sha}")) is None:
            return "no cancel"
        return "cancelled" if r.returncode == 0 else f"cancel exited {r.returncode}: {(r.stderr or r.stdout).strip()[-200:]}"

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
        p = self.prs.get(number) or self.open().get(number)
        return self.q.handle(p) if p is not None else None

    def _tasks(self, sha: str) -> dict[str, dict]:
        try:
            got = self.tasks.graph_v2(f"ready:{sha}")
        except ValueError as e:
            self.log(f"research queue vqueue: {sha[:12]}'s graph can't be read, so its failed tasks' owners aren't told: {e}")
            got = None
        return {t["id"]: t for t in got[1]["tasks"]} if got else {}

    def task_owners(self, sha: str, ids: Iterable[str]) -> tuple[str, ...]:
        """The handles owning tasks ``ids`` of ``sha``'s graph, the owners of what failed: a test file's suite's in the registry
        (`Landing._path_owner`), else its job's in ci.toml at ``sha`` (`Landing._step_owner`)."""
        by, d, out = self._tasks(sha), self.driver(), []
        for i in ids:
            if (t := by.get(i)) is None:
                continue
            if h := (d._path_owner(s) if isinstance(s := _suite(t), str) else None) or d._step_owner(d._job(t), sha):
                out.append(f"@{h}")
        return tuple(dict.fromkeys(out))

    def test_owners(self, sha: str, task: str, tests: Iterable[str]) -> tuple[str, ...]:
        """The handles owning ``tests`` (result/v1's ids, relative to the suite of test file ``task`` of ``sha``'s graph): each
        one's file's owner in the registry, as drive reads a failed test's path; else the task's owners."""
        d, suite = self.driver(), _suite(self._tasks(sha).get(task) or {})
        out = [f"@{h}" for x in tests if isinstance(suite, str)
               and (h := d._path_owner(os.path.normpath(os.path.join(suite, x.split("::")[0]))))]
        return tuple(dict.fromkeys(out)) or self.task_owners(sha, [task])

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
    it ended, a fail `vqueue_slim.judge` doesn't keep; {} once it ended."""
    if j is None or j.slim is None or not (j.slim.state == PENDING or j.slim.state == FAIL
                                           and "verdict" not in (j.record or {})):
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


def _suite(t: dict) -> str | None:
    """A test file task's suite, the directory its result's test ids are relative to."""
    return s if t.get("kind") == "test-file" and isinstance(s := (t.get("vars") or {}).get("suite"), str) else None


def _tests(doc: object, key: str = "", kind: str = "") -> frozenset[str]:
    """The failing tests a result names (result/v1's `failed_tests`).  A fail of a test file or a lean-suites task (``kind``)
    for more than its tests adds a test of its ``key`` alone, which no base fails, as a cut detail does (`_run_tests`), so a
    new crash beside the base's failing tests is never inherited by them: a test file's that pytest ended with another exit
    than 1, a test's failure (2: collecting failed, or a `pytest.exit`); a lean-suites suite's (each row of `detail.failed`)
    that names no test, failed for more than them (its `beyond_tests` isn't false, as when its list is cut) or timed out."""
    d = (doc.get("detail") or {}) if isinstance(doc, dict) else {}
    got = d.get("failed_tests")
    named = frozenset(x for x in got if isinstance(x, str)) if isinstance(got, list) else frozenset()
    if not isinstance(doc, dict) or doc.get("status") != FAIL:
        return named
    if kind == "test-file" and isinstance(exit_ := d.get("exit"), int) and exit_ != 1:
        return named | {f"fault:its run failed beyond the tests it names (exit {exit_}), at {key}"}
    if kind == SUITES_KIND:
        return named | {f"fault:suite {r.get('name')} failed beyond the tests it names, at {key}" for r in d.get("failed") or ()
                        if isinstance(r, dict) and (not r.get("failed_tests") or r.get("beyond_tests") is not False
                                                    or r.get("timed_out"))}
    return named


def _job(t: dict, tasks: dict[str, dict]) -> str:
    """The job of check's that graph task ``t`` of ``tasks`` stands for (`landing.JOB_OF_KIND`, else its kind): a test file that
    reads what lean-build built is of a suite check runs after lean-build (`landing.SUITES_KIND`), on its outputs."""
    if t["kind"] == "test-file" and any(isinstance(r, dict) and (tasks.get(r.get("task")) or {}).get("kind") == "lean-build"
                                        for r in t.get("reads") or ()):
        return SUITES_KIND
    return JOB_OF_KIND.get(t["kind"], t["kind"])


def _said(theirs: dict | None) -> dict[str, dict]:
    """How graph2's `verdict` says each task it lists as failed or infra ended, from its database, which no label overrides:
    {key: the fail or the infra error it reads as}, at graph2's key only (where the loop keys a task otherwise and its labels
    there pass it, `_disputed` holds the candidate).  A task failed by its own fault (an undeclared read, a race: a failed
    entry with a `why`) has no result labelled at all.  Not from a `ready_verdict` label, which a rerun leaves standing until
    the check ends again."""
    if not isinstance(theirs, dict) or "labelled" in theirs:
        return {}
    out = {k: {"key": k, "task": e.get("task"), "status": "error", "detail": {"infra": True, "why": e.get("why")}}
           for e in theirs.get("infra") or () if isinstance(e, dict) and isinstance(k := e.get("key"), str)}
    for e in theirs.get("failed") or ():
        if isinstance(e, dict) and isinstance(k := e.get("key"), str):
            out[k] = {"key": k, "task": e.get("task"), "status": FAIL,
                      "detail": {"fault": e["why"]} if e.get("why") else {"failed_tests": [
                          t for t in e.get("tests") or () if isinstance(t, str)]}}
    return out


def _disputed(theirs: dict | None, ids: Iterable[str], states: dict[str, tuple], keys: dict[str, str]
              ) -> tuple[Dispute, ...]:
    """Each task of ``ids`` (the tasks a candidate's verdict needs: on the gate, or read by one that is) that graph2's
    `verdict` lists as failed (a fail, or its own fault) or infra and the loop's own reading passed (``states``, at ``keys``).
    graph2's word overrides a label only at graph2's key (`_said`), and a result labelled `--by executor` is self-asserted, so
    a task the loop keys from the outputs it reads, or one graph2 names at another key, can pass on labels graph2's database
    doesn't hold.  Not from a `ready_verdict` label, which a rerun leaves standing until the check ends again."""
    if not isinstance(theirs, dict) or "labelled" in theirs:
        return ()
    ids, out = set(ids), {}
    for said, entries in ((FAIL, theirs.get("failed")), (INFRA, theirs.get("infra"))):
        for e in entries or ():
            if (isinstance(e, dict) and isinstance(i := e.get("task"), str) and i in ids and i not in out
                    and states[i][0] == PASS):
                out[i] = Dispute(i, said, str(e.get("key")), keys[i])
    return tuple(out.values())


def _relisted(theirs: dict | None, listing: str, tasks: dict[str, dict]) -> Relisted | None:
    """How the listing of a check the loop read (``listing``, its art id, holding ``tasks``) isn't the one graph2's `verdict`
    says it runs (its `art`, the id of the graph it listed, published and labelled), else None: the store's ids are its
    contents' digests, so the same id is the same tasks, inputs and reads, and the `verity_graph` label the loop reads it by
    is self-asserted.  None where graph2's verdict names no listing (`World._held` holds a pass until it does), and not from
    a `ready_verdict` label, which names none."""
    art = theirs.get("art") if isinstance(theirs, dict) and "labelled" not in theirs else None
    if not isinstance(art, str) or not art or art == listing:
        return None
    counts = [theirs.get(k) for k in ("tasks", "off_gate")]
    named = dict.fromkeys(e.get("task") for k in ("failed", "infra", "flaky") for e in theirs.get(k) or ()
                          if isinstance(e, dict))
    return Relisted(art, listing, sum(counts) if all(isinstance(n, int) for n in counts) else None, len(tasks),
                    tuple(i for i in named if isinstance(i, str) and i not in tasks))


def _ended(v: Verdict) -> str | None:
    """What a candidate's check, read as ``v`` (graph2's in shadow, or the one that gates), ended in, its inherited failures
    set aside (`vqueue._reading`); None while it is pending, or failed with a task or a shard still running."""
    return None if (got := vqueue._reading(v)) == PENDING or v.state == FAIL and not v.ended else got


def _brief(v: Verdict) -> str:
    """``v`` as a compare line says it: its reading, and the tasks it failed of its own or erred on."""
    tasks = ", ".join(f.task for f in (*(f for f in v.failed if not _inherited(f)), *v.infra))
    return vqueue._reading(v) + (f" ({tasks})" if tasks else "")


def _runs(theirs: dict | None) -> bool:
    """Whether graph2's own verdict (its `verdict` command's JSON) says its check still runs and names no infra error.  The
    loop's reading of its results hasn't ended then (`World._held`), but for a graph lacking a job or a kind, so a check the
    loop only follows (`Loop._followed`) isn't read whole, a graph's worth of work each tick, until graph2 says otherwise."""
    return (isinstance(theirs, dict) and "labelled" not in theirs and not theirs.get("infra")
            and (theirs.get("state") == PENDING or isinstance(n := theirs.get("pending"), int) and n > 0))


def _named(theirs: dict | None) -> dict[tuple[str, str], frozenset[str]]:
    """The failing tests graph2's `verdict` names of each failed task, by (task, key): its result's own `failed_tests` may
    name none (a `lean-suites` result's detail has none), and a task failed by its own fault has its fault's (`_fault_tests`).
    Test ids are relative to the task's suite, so they are compared only within a task.  Not from a `ready_verdict` label,
    which a rerun leaves standing until the check ends again."""
    if not isinstance(theirs, dict) or "labelled" in theirs:
        return {}
    out: dict[tuple[str, str], frozenset[str]] = {}
    for e in theirs.get("failed") or ():
        if isinstance(e, dict) and isinstance(i := e.get("task"), str) and isinstance(k := e.get("key"), str):
            tests = frozenset(x for x in e.get("tests") or () if isinstance(x, str))
            why, error = e.get("why"), e.get("error")
            out[i, k] = tests or (_fault_tests(why, k, error if isinstance(error, dict) else None)
                                  if isinstance(why, str) and why else frozenset())
    return out


#: graph2's fault of a task that read files outside its declared inputs (its `task_fault`): how many, and the first paths
UNDECLARED_RE = re.compile(r"read (\d+) file\(s\) outside its declared inputs: (.*)$")
#: the start of graph2's fault of a task whose runner caught a race (its `task_fault`)
RACE = "its runner caught a race"
#: a run and the node it ran on in graph2's words ("run <id> on <node>"), and a run id alone
RUN_ON_RE, RUN_ID_RE = re.compile(r"\brun \S+ on [^\s:;,]+"), re.compile(r"\br\d{8}-\d{6}-[0-9a-z]+\b")
#: the runner's detail of a faulted task's run in graph2's `why` (JSON graph2 cuts at 200 characters), before its fault
ERROR_RE = re.compile(r"; error: (.*?); read \d+ file\(s\) outside its declared inputs: ")
#: the message of a runner's `{"error": ...}` detail, whole or as far as graph2 keeps it
ERROR_HEAD_RE = re.compile(r'^\{"error": "((?:[^"\\]|\\.)*)')
#: how much of a runner's error message counts: a `why` keeps the first 200 characters of its detail's JSON, so a message is
#: compared by a head that a `why` and a verdict's whole `error` both hold
MESSAGE_KEEP = 120


def _fault_tests(why: str, key: str, error: dict | None = None) -> frozenset[str]:
    """The tests a task failed by its own fault fails, from graph2's `why` of it, so the same fault at its base is the same
    tests (`vqueue.Failed.base_tests`): each file it read outside its declared inputs (`undeclared:<path>`) and what its run
    failed beside them (`_run_tests`), else one test of the fault (`fault:<it>`), from its runner's race on, else the whole
    `why` without the runs and nodes it names.  graph2's verdict keeps a `why`'s last 300 characters, where its fault ends it;
    ``error``, the run's whole result (its `detail` and `undeclared_reads`), where the verdict gives it, else the `why`'s."""
    if m := UNDECLARED_RE.search(why):
        paths = error.get("undeclared_reads") if error is not None else None
        if not (isinstance(paths, list) and paths):
            # a `why` names only the first three files of its sorted reads: one the change adds past them goes unseen, and
            # one it removes from them brings the next into view, as if added
            paths = m.group(2).split(", ")
        reads = frozenset(f"{vqueue.UNDECLARED}{p}" for p in paths if isinstance(p, str) and p)
        if error is not None and isinstance(error.get("detail"), dict):
            return reads | _run_tests(error["detail"], key)
        d = ERROR_RE.search(why)
        return reads | _run_tests(d.group(1) if d else None, key)
    if (i := why.find(RACE)) >= 0:
        return frozenset({f"fault:{why[i:]}"})
    return frozenset({"fault:" + " ".join(RUN_ID_RE.sub("", RUN_ON_RE.sub("run", why)).split())})


def _message(text: str) -> str:
    """A runner's error message as it counts: its run ids out, and its head of `MESSAGE_KEEP` characters."""
    return RUN_ID_RE.sub("", text)[:MESSAGE_KEEP]


def _run_tests(detail: dict | str | None, key: str) -> frozenset[str]:
    """What a run that read files outside its task's declared inputs failed beside that, from its runner's ``detail`` (whole,
    or the JSON a `why` keeps): nothing for a clean run (exit 0, no error, no failure), else its failing tests (pytest node
    ids) and its runner's error (`fault:<its head>`), else its exit and counts.  A detail cut off, or a list of failing tests
    cut, is a test of ``key`` alone, which no base fails: a crash beside the reads (a signal, an OOM kill) is never taken for
    the base's fault."""
    unseen = frozenset({f"fault:the run's detail graph2 cut, at {key}"})
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except ValueError:
            if not (head := ERROR_HEAD_RE.match(detail)):
                return unseen
            try:
                message = json.loads(f'"{head.group(1)}"')
            except ValueError:
                message = head.group(1)
            return frozenset({f"fault:{_message(message)}"})
    if not isinstance(detail, dict):
        return unseen
    failed = frozenset(t for t in detail.get("failed_tests") or () if isinstance(t, str))
    fault = frozenset({f"fault:{_message(detail['error'])}"}) if isinstance(detail.get("error"), str) else frozenset()
    if failed or fault:
        return failed | fault
    if detail.get("exit") == 0 and not detail.get("errors") and not detail.get("failed"):
        return frozenset()
    return frozenset({f"fault:exit {detail.get('exit')}, {detail.get('errors')} error(s), {detail.get('failed')} failed"})


def _failure(task: str, key: str, inputs: frozenset[str], **kw) -> Failed:
    """A `vqueue.Failed`, each of ``kw`` it has a field for."""
    return Failed(task, key, inputs, **{k: v for k, v in kw.items() if k in FAILED_FIELDS})


def _infra(docs: Iterable[object], key: str) -> bool:
    """Whether ``key``'s newest result is graph2's infra error, its attempts run out."""
    mine = [d for d in docs if isinstance(d, dict) and d.get("key") == key]
    return bool(mine) and mine[-1].get("status") == "error" and (mine[-1].get("detail") or {}).get("infra") is True


def _walk(tasks: dict[str, dict], roots: Iterable[str]) -> list[str]:
    """The tasks ``roots`` read, transitively, each once, as a walk from each root in turn lists those it hasn't yet: a task,
    then what each of its reads reads, in order.  One walk for all roots: a Lean graph reads each module's transitive
    imports, millions of reads, so a walk per root is out of reach."""
    seen: set[str] = set()
    out: list[str] = []
    for root in roots:
        stack = [root]
        while stack:
            if (i := stack.pop()) in seen:
                continue
            seen.add(i)
            out.append(i)
            stack.extend(r["task"] for r in reversed(tasks[i]["reads"]))
    return out


def _fold(tasks: dict[str, dict], i: str, own: Callable[[str], object], join: Callable[[object, list], object],
          memo: dict[str, object]) -> object:
    """``join`` of task ``i``'s ``own`` and the values of the tasks it reads, in order, each task's worked out once (in
    ``memo``) after those it reads, so the tasks a graph reads, transitively, cost its reads once, not once per task asked
    about.  A read that closes a cycle counts for nothing."""
    if i in memo:
        return memo[i]
    stack, on = [(i, iter(tasks[i]["reads"]))], {i}
    while stack:
        j, reads = stack[-1]
        for r in reads:
            if (k := r["task"]) not in memo and k not in on:
                on.add(k)
                stack.append((k, iter(tasks[k]["reads"])))
                break
        else:
            stack.pop()
            on.discard(j)
            memo[j] = join(own(j), [memo[r["task"]] for r in tasks[j]["reads"] if r["task"] in memo])
    return memo[i]


class _Closure:
    """What each task of a graph reads, transitively, worked out once per task (`_fold`): the failed tasks among them, in the
    order a walk from it lists them (`_walk`); whether a task of some set is among them; and the inputs of all, as bit masks
    over the graph's paths."""

    def __init__(self, tasks: dict[str, dict], envs: dict):
        self.tasks, self.envs = tasks, envs
        self._memo: dict[object, dict[str, object]] = {}
        self._bit: dict[str, int] = {}

    def fails(self, i: str, failed: frozenset[str]) -> tuple[str, ...]:
        """The tasks of ``failed`` that ``i`` is or reads, transitively, in the order `_walk` from ``i`` lists them."""
        return _fold(self.tasks, i, lambda j: (j,) if j in failed else (),
                     lambda mine, theirs: tuple(dict.fromkeys(x for t in (mine, *theirs) for x in t)),
                     self._memo.setdefault(("fails", failed), {}))

    def meets(self, i: str, these: frozenset[str]) -> bool:
        """Whether ``i`` is or reads, transitively, a task of ``these``."""
        return _fold(self.tasks, i, lambda j: j in these, lambda mine, theirs: mine or any(theirs),
                     self._memo.setdefault(("meets", these), {}))

    def inputs(self, i: str) -> frozenset[str]:
        """``i``'s inputs, its env's spec files and those of each task it reads, transitively."""
        mask = _fold(self.tasks, i, self._mask, lambda mine, theirs: reduce(or_, theirs, mine),
                     self._memo.setdefault("inputs", {}))
        paths = list(self._bit)
        return frozenset(paths[n] for n, c in enumerate(reversed(bin(mask)[2:])) if c == "1")

    def _mask(self, j: str) -> int:
        """Task ``j``'s own inputs and its env's spec files, as a bit mask."""
        t, mask = self.tasks[j], 0
        for p in (*t["inputs"], *((self.envs.get(t["env"]) or {}).get("spec") or ())):
            mask |= 1 << self._bit.setdefault(p, len(self._bit))
        return mask


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
    `research.vqueue_slim` (its `start`, `judge` and `combine`), or None to judge each candidate on its graph2 check alone.
    ``gate`` is what gates a candidate (`GATES`): its graph2 check with the run beside it, or the check ``slim`` runs as
    ``shards`` shards (N of ci.toml's plan, or `auto:N`: `vqueue_slim.shard_count`), alone.  ``records`` is
    `research.vqueue_records` (its `start`, `judge` and `commit`), or None to start every candidate's check at once, with no
    records run before it (`vqueue.Snapshot.records`).  Where the check gates, graph2's readings stand ``g2_read_s`` between
    ticks (`Shadow`), and a tick reads at most ``g2_fresh`` afresh that nothing forces (`_read`).  ``after`` is the tick's own command, which the steps after `main` moved run detached as, with
    `--after-landing` (`_moved`); None runs them in the tick.  Each tick gives ``timing`` a line of its steps' seconds on
    ``clock`` and the peak resident size after each (``peak``, `Steps`)."""

    def __init__(self, world: World, log: Path, *, decide: Callable[[Snapshot], list[Action]] = vqueue.decide,
                 shadow: bool = False, dry_run: bool = False, batch: int = BATCH, depth: int = DEPTH, lean: int = LEAN,
                 pending_s: float = 4 * 3600, slim=None, now: Callable[[], float] = time.time,
                 out: Callable[[str], None] = print, state: Path | None = None, tree_ejects: bool = False,
                 gate: str = GRAPH2_GATE, shards: int | str = SHARDS, records=None, graph2_shadow: bool = True,
                 g2_read_s: float = G2_READ_S, g2_fresh: int = G2_FRESH, after: list[str] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 peak: Callable[[], float] = _peak_gb, timing: Callable[[str], None] = print):
        if gate not in GATES or gate == CHECK_GATE and (slim is None or not _sharded(slim, shards)):
            raise ValueError(f"gate {gate!r} with {shards} shard(s){'' if slim is not None else ' and no run beside a check'}: "
                             f"a gate is one of {', '.join(GATES)}, and `{CHECK_GATE}` is the run beside each check "
                             "(`research.vqueue_slim`) as one shard or more")
        self.world, self.log_path, self.decide, self.shadow, self.dry_run = world, Path(log), decide, shadow, dry_run
        self.batch, self.depth, self.lean, self.pending_s, self.slim = batch, depth, lean, pending_s, slim
        self.gate, self.shards = gate, shards
        #: whether graph2's verdict is read in shadow where the check gates (`--no-graph2-shadow` turns it off): each read
        #: lists every task of a candidate's check and pulls each pending task's results, which the check's gate never needs
        self.graph2_shadow = graph2_shadow or gate != CHECK_GATE
        self.now, self.out, self.state, self.tree_ejects, self.records = now, out, state, tree_ejects, records
        self.after, self.clock, self.peak, self.timing, self.steps = after, clock, peak, timing, Steps(clock, peak)
        self.reads = Shadow(self.log_path.parent / G2_READS, now, g2_read_s, out) \
            if gate == CHECK_GATE and graph2_shadow else None
        if world is not None:
            world.reads = self.reads
        self.g2_fresh = self.fresh_left = g2_fresh
        #: the candidates this tick's snapshot took graph2's reading of from `reads`
        self.stale: set[str] = set()
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
        below = self._below(past, whole, n, main, origin, wrote)
        steps.lap("below")
        lent = _lent(judged, below[0], main)
        w._theirs, w._listed, self.stale = {}, {}, set()
        try:
            for sha in sorted(mine, key=lambda s: (height(s), s)):      # each after its base, whose record it reads as judged now
                p, b = whole if sha in older else past, builds[sha]
                base, prs, heads = b.get("base") or "", tuple(b.get("members") or ()), tuple(b.get("heads") or ())
                waits = self.records is not None and bool(b.get("records_first")) and not p.tries(sha)
                if waits:
                    j = Judged(Verdict(PENDING), note="its check waits for its records run", listed=False)
                else:
                    j, kept = self._read(p, sha, base)
                    # where the check gates, a candidate on `main` is read by its check too: graph2's verdict decides nothing
                    if self.slim is not None and (self.gate == CHECK_GATE or not w._ancestor(sha, main)):
                        again = None if kept is None or kept.get("check_ended") else partial(self._read, p, sha, base,
                                                                                              fresh=True, ended=True)
                        j = self._beside(p, n, main, origin, sha, base, prs, heads, j, wrote, chained(sha),
                                         below if base == main else ((judged[base].record if base in judged else None)
                                                                     or p.slim(base)[1], p.slim_started(base)[1]), lent,
                                         on_main=below[0], again=again)
                judged[sha] = j
                cands.append(Candidate(sha, base, prs, heads, j.verdict, p.tries(sha), reruns=p.reruns(sha),
                                       enqueued=p.enqueued(sha),
                                       records=self._records_run(p, n, main, origin, sha, b, wrote, regen),
                                       repairs=b.get("repairs") if isinstance(b.get("repairs"), str) else None,
                                       waits=waits, held=p.held(sha) if waits else None))
        finally:
            w._theirs = w._listed = None
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
        snap = Snapshot(main, tuple(sorted((x for x in ready if x.number not in riding), key=lambda x: x.rank)), tuple(cands),
                        history, self.batch, self.depth, lean=self.lean, now=self.now(), pending_s=self.pending_s,
                        tree_ejects=self.tree_ejects, left=tuple((n, *left[n]) for n in sorted(left)), wide=wide,
                        pinned=pinned, pulled=frozenset(w.pulled | set(out) | set(taken)),
                        flying=tuple(Ready(m, h, (f := frozenset() if h in gone else w._changed(main, h)),
                                           lean=h not in gone and w._judge(main, h)[0],
                                           move=bool(getattr(w.open().get(m), "move", "")),
                                           scripts=vqueue.meets(f, MOVE_SCRIPTS)) for m, h in sorted(flying)),
                        records=self.records is not None)
        steps.lap("chain")
        return Tick(n, main, origin, snap, judged, whole, past.told(), foreign, wrote, below[0], regen, dict(w.carried),
                    dict(w.unauthenticated), riding=riding)

    def _read(self, p: Past, sha: str, base: str, fresh: bool = False, ended: bool | None = None
              ) -> tuple[Judged, dict | None]:
        """Candidate ``sha``'s verdict on graph2's results (`World.verdict`), list-failed when the log holds no enqueue of it,
        and the reading `Shadow` kept it from, else None: where the check gates, one read under the same key (its base and the
        results set aside there and on it) less than `G2_READ_S` ago, or of any age once this tick read `G2_FRESH` afresh
        (pending, none kept), unless ``fresh`` or its check ended since (the record of its run, as the log holds it, has a
        verdict; ``ended`` says so of a read after this tick's judging found it).  A reading read is kept."""
        rec = p.slim(sha)[1]
        ended = isinstance(rec, dict) and "verdict" in rec if ended is None else ended
        key = _digest([base, p.since(sha), p.since(base)])
        if self.reads is not None and not fresh:
            kept = self.reads.get("judged", sha, key)
            if kept is None and self.fresh_left <= 0:
                kept = self.reads.get("judged", sha, key, any_age=True) or {
                    "state": PENDING, "note": f"graph2's reading waits for a later tick: {self.g2_fresh} a tick"}
            if kept is not None and (kept.get("check_ended") or not ended):
                self.stale.add(sha)
                self.reads.hits += 1
                return _unshadowed(kept), kept
        if self.graph2_shadow:
            with self.world.one_read() if self.reads is not None else contextlib.nullcontext():
                j = self.world.verdict(sha, base, p.since(sha), p.since(base))
            self.fresh_left -= 1
        else:
            j = Judged(Verdict(PENDING), note="graph2's shadow isn't read (--no-graph2-shadow)")
        if not j.listed and not p.tries(sha):           # pushed, and its enqueue failed or never ran
            j = Judged(Verdict(LIST_FAILED), mark="not enqueued", note="the log holds no enqueue of it",
                       graph2=j.graph2, listed=False, theirs=j.theirs)
        if self.reads is not None:
            self.reads.put("judged", sha, {**_shadow_doc(j), "check_ended": ended}, key)
        self.stale.discard(sha)
        return j, None

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
        (started, rec), (starts, since) = past.slim(main), past.slim_started(main)
        line = dict(sha=main, base=main, members=(), heads=())
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

    def _beside(self, past: Past, n: int, main: str, origin: str, sha: str, base: str, prs: tuple[int, ...],
                heads: tuple[str, ...], j: Judged, wrote: list[dict], chained: bool,
                below: tuple[dict | None, float | None] = (None, None), lent: Lent | None = None,
                on_main: dict | None = None, again: Callable[[], tuple[Judged, dict | None]] | None = None) -> Judged:
        """``j`` combined with the verdict of the run beside candidate ``sha``'s check (`vqueue_slim`), which its build starts
        beside each candidate's check, and here, once, one on ``main``'s chain (``chained``) the log holds no start of (its
        build's tick stopped first); again by a retry of its infra or a rerun of its failure.  Judged each tick, its record
        logged (to ``wrote`` too) whenever it changed; with --dry-run neither started nor judged, so pending.  A check that
        failed to list stays list-failed beside a run that hasn't ended, which `combine` reads as pending: one that never
        starts would hold it so.  Each step it failed is compared with the run beside its base's check (``below``: its record
        and start, `_below` for ``main``, a base candidate's as judged this tick), which `vqueue.decide` takes as inherited
        when that run failed it too, a base candidate's still running by what it failed so far (`vqueue_slim.based`); a fail
        whose every other failure is inherited waits (pending) while the run beside ``main``, started at most `pending_s`
        ago, hasn't said, and one on a candidate stays a fail, which `vqueue.decide` holds until its base's says, or, where
        only its own members meet the step, until the run beside ``main``'s check says (``on_main``, its record, read as a
        base's into `vqueue.Failed.main_tests`).  Where the
        check gates (`CHECK_GATE`), the run's verdict alone is the candidate's, graph2's kept beside it (`Judged.shadow`, its
        note marked so), a run started while graph2 gated counts as none started, and a check that took lean-audit's verdict
        from the check of a candidate below reads lean-audit as pending while that candidate is in flight, and, once it has
        landed, takes the verdict that the record ``lent`` gives keeps (`vqueue_slim.audited`, `_lent`).  There, a graph2
        reading `Shadow` kept from before the check ended is read again (``again``) once this judging finds it ended, so the
        compare and the drop to come read graph2 as it is now."""
        started, rec = past.slim(sha)
        started, gates = started and not self._other(rec), self.gate == GRAPH2_GATE
        line = dict(sha=sha, base=base, members=prs, heads=heads)
        if self.dry_run or not started and not chained:
            v = Verdict(PENDING)
        else:
            if not started:
                self.write(got := self._started(n, main, origin, line, below=below[0] if base != main else None, lent=lent))
                wrote.append(got)
                rec = got["record"]
            was = json.dumps(rec, sort_keys=True)
            v = self.slim.judge(self.world.slim_driver(), rec)
            if not gates:
                ran = (rec or {}).get("lean_audit_from")
                source, landed = lent(ran) if ran and lent else (None, False)
                v = self.slim.audited(self.world.slim_driver(), v, rec, source if landed else None)
                if again is not None and isinstance(rec, dict) and "verdict" in rec:
                    j = again()[0]
            if json.dumps(rec, sort_keys=True) != was:
                self.write(got := self._line(n, main, origin, "slim", **line, why=f"judged {v.state}", record=rec, done=True))
                wrote.append(got)
        if gates and j.verdict.state == LIST_FAILED and v.state == PENDING:
            return replace(j, slim=v, record=rec)
        v = self.slim.based(v, below[0], so_far=base != main)
        if base != main and on_main is not None and v.failed:
            unsaid = replace(v, failed=tuple(replace(f, base_tests=None) for f in v.failed))
            v = replace(v, failed=tuple(replace(f, main_tests=m.base_tests)
                                        for f, m in zip(v.failed, self.slim.based(unsaid, on_main).failed)))
        both = self.slim.combine(j.verdict, v) if gates else v
        if not gates:
            j = replace(j, shadow=j.verdict, note=f"graph2, in shadow, reads {j.verdict.state}: {j.note}" if j.note else "")
        if both.state == FAIL:              # `combine` keeps no `ended`: a fail has ended once both checks stopped running,
            g = j.verdict                   # and never while graph2's check failed to list: none of its gating tasks ran
            if gates:
                both = replace(both, ended=(g.state not in (PENDING, LIST_FAILED) and (g.state != FAIL or g.ended))
                               and v.state != PENDING)
            if base == main and (waits := self._waits(both, v, *below)):
                return replace(j, verdict=Verdict(PENDING), slim=v, record=rec, note="; ".join(
                    [*filter(None, [j.note]), f"waiting for the run beside {base[:12]}'s check: {', '.join(waits)}"]))
        return replace(j, verdict=both, slim=v, record=rec)

    def _other(self, rec: dict | None) -> bool:
        """Whether ``rec`` records a run beside a graph2 check where the check now gates: the check starts in its place."""
        return self.gate == CHECK_GATE and rec is not None and rec.get("gate") != CHECK_GATE

    def _waits(self, both: Verdict, v: Verdict, rec: dict | None, since: float | None) -> list[str]:
        """The failed steps of ``v`` (the run beside a check) whose fail ``both`` waits on: those the run beside its base's
        check (``rec``, started at ``since``) hasn't said of while it runs, up to `pending_s`, when every other failure is
        inherited; none else."""
        unknown = [f for f in v.failed if f.base_tests is None]
        if not unknown or rec is None or "verdict" in rec or since is None or self.now() - since > self.pending_s:
            return []
        if not all(f in unknown or vqueue._inherited(f) for f in both.failed):
            return []
        return [f.task for f in unknown]

    # -- a tick

    def tick(self) -> list[dict]:
        """One tick: the records it logged (with --dry-run, those it would have), and a line of how long each of its steps
        took and the peak resident size after it (`Steps`), with how many of graph2's verdicts it read, their bytes and
        seconds, and how many readings `Shadow` kept answered, printed however it ends."""
        self.steps = Steps(self.clock, self.peak)
        if self.world is not None:
            self.world.g2 = {"reads": 0, "bytes": 0, "s": 0.0}
        if self.reads is not None:
            self.reads.used, self.reads.hits = set(), 0
        self.fresh_left = self.g2_fresh
        try:
            return self._tick()
        finally:
            g2 = getattr(self.world, "g2", None) or {"reads": 0, "bytes": 0, "s": 0.0}
            self.timing(self.steps.line(self.last, {**g2, "cached": self.reads.hits if self.reads is not None else 0}))

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
        new = [self.record(t, "ignore", sha=sha, ref=name, why=f"origin's queue/{name} is no build of the loop's: no candidate")
               for name, sha in t.foreign.items() if (name, sha) not in past.ignored()] + self._carried(t) \
            + self._unauthenticated(t)
        new += self._flaky(t) + self._bases(t) \
            + (self._disputes(t) if self.gate == GRAPH2_GATE else self._compare(t) + self._off_gate(t))
        steps.lap("notices")
        new += self._followed(t)
        steps.lap("followed")
        new += self._conflicts(t) + self._lacking(t) + self._riding(t) + self._launched(t)
        steps.lap("notices")
        new += self._moved(t)
        steps.lap("on_main")
        for rec in new:
            self.write(rec)
        logged = [*t.wrote, *new]
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
                break
            finally:
                steps.lap(_name(a))
            for rec in recs:
                self.write(rec)
            logged += recs
            if failed := next((r["error"] for r in recs if r.get("error")), None):
                self.out(f"research queue vqueue: tick {t.n}: {_name(a)} failed, the tick stops: {failed}")
                break
            if isinstance(a, Advance) and not self.shadow:
                for rec in self._landing(t, recs[0], []):
                    self.write(rec)
                    logged.append(rec)
                steps.lap("on_main")
        for sha, j in t.judged.items():
            if j.note:
                self.out(f"research queue vqueue: tick {t.n}: {sha[:12]} {j.verdict.state}: {j.note}")
        if past.bad:
            self.out(f"research queue vqueue: tick {t.n}: {len(past.bad)} line(s) of {self.log_path} read as none: {past.bad[0]}")
        if self.reads is not None:
            self.reads.save()
        self.keep(t)
        steps.lap("keep")
        try:
            self.world.q.work_out_mapped()
        except (QueueError, OSError, subprocess.SubprocessError) as e:
            self.out(f"research queue vqueue: tick {t.n}: mapping old heads across `main`'s moves failed: {e}")
        steps.lap("mapped")
        return logged

    def keep(self, t: Tick) -> None:
        """The state file (``state``): this tick, and each candidate's verdict beside graph2's and the jobs and kinds its graph
        lacks."""
        if self.state is None:
            return
        records = {c.sha: c.records.state for c in t.snap.candidates if c.records is not None}
        doc = {"at": stamp(self.now()), "tick": t.n, "main": t.main, "origin_main": t.origin,
               "candidates": {sha: {"mine": j.verdict.state, "graph2": j.graph2, "lacks": list(j.lacks), "note": j.note,
                                    "beside": j.slim.state if j.slim is not None else None, "records": records.get(sha)}
                              for sha, j in t.judged.items()},
               "ignored": t.foreign,
               "disagree": sorted(sha for sha, j in t.judged.items()
                                  if j.graph2 in vqueue.STATES and j.graph2 != j.verdict.state)}
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
            return [self._drop(a, c, t)]
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
            fix = f"merge `main` into it once {names} {'is' if len(s.by) == 1 else 'are'} on it" if s.by else "merge `main` into it"
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
        """Start pushed candidate ``line["sha"]``'s check: enqueued, its build line (``line``) logged after ``recs``, and the run
        beside its check started.  Every check starts here: a build's, and the check of one whose records run passed
        (`_commit`)."""
        try:
            self.world.enqueue(line["sha"], line["priority"])
        except (QueueError, OSError, subprocess.SubprocessError) as e:  # pushed: the next tick reads it as not enqueued
            return [*recs, self.record(t, "build", **line, enqueued=False, done=True, error=str(e)[-600:])]
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

    def _start(self, t: Tick, sha: str, base: str, line: dict, why: str = "started") -> list[dict]:
        """The run beside candidate ``sha``'s check started (`vqueue_slim.start`), its record logged; a start that fails is
        logged with no record, which `vqueue_slim.judge` reads as infra.  A base still in flight lends the record of its
        check, as judged this tick or, put on the chain by a `Commit` of this tick, as started (`Tick.committed`), and the
        candidates below it theirs, or, landed, the run beside `main` (`_audit_from`)."""
        if self.slim is None:
            return []
        began = dict(t.committed.values())
        below = began[base] if base in began else \
            t.judged[base].record if base not in (t.main, t.advanced) and base in t.judged else None
        return [self._started(t.n, t.main, t.origin, {**line, "sha": sha, "base": base}, why, below,
                              _lent(t.judged, t.below, t.main, t.advanced))]

    def _started(self, n: int, main: str, origin: str, line: dict, why: str = "started", below: dict | None = None,
                 lent: Lent | None = None) -> dict:
        d = self.world.slim_driver()                       # drive.toml naming no machines stops the tick: no run is infra
        heavy = self._heavy_on(line) if self.gate == CHECK_GATE else ""
        try:
            audit = self._audit_from(d, main, line, below, lent)
            after = self.slim.replaced(Past.read(self.log_path, self.now() - self.slim.CARRY_WAIT_S).lines,
                                       line.get("members") or (), self.now()) if self.gate == CHECK_GATE else {}
            with self.steps("launch"):
                rec, error = self.slim.start(d, line["sha"], line["base"],
                                             **({"shards": self.shards} if self.gate == CHECK_GATE else {}),
                                             **({"audit_from": audit} if audit else {}),
                                             **({"deselect": f} if (f := getattr(self.world, "deselect_file", None)) else {}),
                                             **({"after": after} if after else {}),
                                             **({"heavy_on": heavy} if heavy else {})), None
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
        (`vqueue_slim.start`'s ``audit_from``), where the check gates and its base is a candidate in flight, not `main`
        (``below``, its record), when no member's own change, against its merge base with ``main`` (`World._changed`, as
        `World._judge` reads it), meets a Lean input of the candidate's tree with the paths its members change in it, so that
        one a member deletes is judged as a file there (`vqueue_slim.lean_inputs`): that base, whose check is judged on
        lean-audit (`vqueue_slim.audits`), else the candidate its base's check took it from (``lean_audit_from``), the one
        whose check ran it, while that check is judged on it, or, once that candidate has landed, while the run beside `main`
        keeps a verdict on it (`vqueue_slim.answers`; where each is read: ``lent``, `_lent`).  None else, and when the Lean
        inputs can't be read (git can't list that tree or say what a member changes, or the audit's declaration doesn't
        load): the check audits."""
        if self.gate != CHECK_GATE or below is None or lent is None or not line.get("heads"):
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
        """Cancel ``c``'s check, and the run beside it and its records run while they run (they would hold drive's machines for
        nothing), delete its branch and log its outcome.  A run still launching (its shards, of a sharded one) has no run ids
        yet: its record is logged (`beside_launching`, `records_launching`), and `_launched` cancels it once it has them; an
        auto plan still being made is given up (`vqueue_slim.abandon`), so no shard of it launches.  Where the check gates, the
        line says how graph2's check, in shadow, read (``graph2``), and one `_follow` keeps is left running (``follow``).
        One decided on a check that gates failed while it ran (an early red) logs when its first failure was written and
        when it was dropped (``early_red``)."""
        w, j = self.world, self._reread(t, c)
        shadow = j.shadow if j is not None else None
        follow = self._follow(c, t)
        cancel = w.cancel(c.sha) if follow is None else "left running, in shadow"
        why = f"vqueue: candidate {c.sha[:12]} was dropped: {a.why}"
        rec = _running(j)
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
                           failed=c.verdict.failed, cancel=cancel, done=True,
                           **({"graph2": _ended(shadow) or PENDING} if shadow is not None else {}),
                           **({"follow": follow} if follow is not None else {}),
                           **({"beside_cancelled": ids} if ids else {}), **({"beside_cancel_failed": failed} if failed else {}),
                           **({"beside_launching": rec} if "launch" in rec or "shards" in rec else {}),
                           **({"records_cancelled": rids} if rids else {}),
                           **({"records_cancel_failed": rfailed} if rfailed else {}),
                           **({"records_launching": regen} if "launch" in regen else {}),
                           **({"early_red": {"failed_at": stamp(early), "dropped_at": stamp(self.now())}}
                              if early is not None else {}))

    def _reread(self, t: Tick, c: Candidate) -> Judged | None:
        """``c`` as this tick judged it, with graph2's reading, in shadow, read again where the snapshot took it from the one
        `Shadow` kept (`stale`): a drop logs it and follows by it (`_follow`).  A read that fails keeps the kept one."""
        j, w = t.judged.get(c.sha), self.world
        if j is None or j.shadow is None or c.sha not in self.stale:
            return j
        w._theirs, w._listed = {}, {}
        try:
            g = self._read(t.past, c.sha, c.base, fresh=True)[0]
        except Exception as e:  # noqa: BLE001 -- graph2 only shadows here: its reading stops no drop
            w.log(f"research queue vqueue: ready:{c.sha[:12]}, read again at its drop: {type(e).__name__}: {e}"[:300])
            return j
        finally:
            w._theirs = w._listed = None
        t.judged[c.sha] = j = replace(j, shadow=g.verdict, theirs=g.theirs, graph2=g.graph2)
        return j

    def _still_runs(self, sha: str) -> bool:
        """Whether graph2 says followed check ``sha`` still runs (`_runs`): a word `Shadow` kept stands `G2_READ_S`, and one
        that it doesn't is read each time."""
        if self.reads is not None and self.reads.get("runs", sha) is not None:
            self.reads.hits += 1
            return True
        if runs := _runs(self.world._verdict_of(sha)):
            if self.reads is not None:
                self.reads.put("runs", sha, {})
        return runs

    def _follow(self, c: Candidate, t: Tick) -> dict | None:
        """What a later tick compares graph2's verdict of ``c`` with (`_followed`), where the check gates and ``c`` is dropped
        with its check ended in a pass or a fail (`_ended`: not an early red; on `main`, or failed) and graph2's, in shadow,
        not: the check's reading and what it failed as this drop reads them, how long it took (`_check_s`), and when graph2's
        check was enqueued.  None for any other drop, and for every drop while the shadow is off (`--no-graph2-shadow`, under
        which `_followed` follows nothing): its graph2 check is cancelled, and it gets no compare line it hasn't had."""
        j, mine = t.judged.get(c.sha), _ended(c.verdict)
        if (self.gate != CHECK_GATE or not self.graph2_shadow or j is None or j.shadow is None or _ended(j.shadow) is not None
                or mine not in (PASS, FAIL) or c.enqueued is None):
            return None
        return {"check": mine, "said": _brief(c.verdict), "check_s": self._check_s(t, c.sha, j.record),
                "enqueued": stamp(c.enqueued)}

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
        """Enqueue ``c``'s check again: graph2 lists it again, reruns what ended in infra errors once its own verdict is
        infra, or leaves it running; while graph2's verdict is another (a task still runs, or one failed), the infra keys are
        rerun here (graph2's `rerun`).  The infra results so far are set aside (`Past.since`), so it is pending until graph2
        says something newer, and a run beside it that ended in infra is started again."""
        j, line = t.judged[c.sha], dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads)
        built = t.past.built(c.sha) or {}
        priority = built.get("priority") if isinstance(built.get("priority"), int) else 0
        counts = self.world.counts(f.key for f in c.verdict.infra)
        self.world.enqueue(c.sha, priority)
        said = self.world.rerun(c.sha, list(counts)) if counts and j.graph2 != INFRA else None
        rec = self.record(t, "retry", **line, priority=priority, why=f"{c.verdict.state}: {j.note or 'enqueued again'}",
                          failed=c.verdict.infra, mark=j.mark, counts=counts, enqueued=True, done=True,
                          **({"rerun": said} if said is not None else {}))
        again = self._start(t, c.sha, c.base, line, "started again") if j.slim is not None and j.slim.state == INFRA else []
        return [rec, *again]

    def _rerun(self, a: Rerun, c: Candidate, t: Tick) -> list[dict]:
        """Run ``a.keys`` afresh (graph2's `rerun`), their results so far set aside (`Past.since`) so each reads pending until
        it has a newer one; a key of the run beside the check starts that run again.  Without graph2's `rerun`, logged as
        absent: the policy takes the rerun as done, and its failure stands."""
        w, line = self.world, dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads)
        j = t.judged[c.sha]
        beside = {f.key for f in j.slim.failed} if j.slim is not None else set()
        keys = [k for k in a.keys if k not in beside]
        failed = [f for f in c.verdict.failed if f.key in a.keys]
        counts = w.counts(keys)
        try:
            said = w.rerun(c.sha, keys) if keys else ""
        except QueueError as e:
            return [self._relist(t, c, line, failed, a.keys, str(e))]
        if said is None:
            return [self.record(t, "rerun", **line, why="graph2 has no `rerun`: the failure stands", failed=failed,
                                keys=list(a.keys), absent=True, done=False)]
        stopped = list(_running(j).get("ids") or ()) if set(a.keys) & beside else []
        stop_failed = w.cancel_runs(stopped, f"vqueue: candidate {c.sha[:12]}'s failure is rerun afresh") if stopped else {}
        again = self._start(t, c.sha, c.base, line, "started again: its failure is rerun") if set(a.keys) & beside else []
        return [self.record(t, "rerun", **line, why=f"the lone member's own failures, afresh: {said or 'the run beside'}",
                            failed=failed, keys=list(a.keys), counts=counts, done=True,
                            **({"beside_cancelled": stopped} if stopped else {}),
                            **({"beside_cancel_failed": stop_failed} if stop_failed else {})), *again]

    def _relist(self, t: Tick, c: Candidate, line: dict, failed: list[Failed], keys: tuple[str, ...], why: str) -> dict:
        """A rerun graph2 refused: it prunes a check an hour after its verdict once nothing of it runs, and reruns none of it
        until an enqueue lists it again, its finished tasks answered from the cache.  ``c``'s check is enqueued again here and
        the rerun logged not done, so the policy asks for it again and the next tick's finds the check listed."""
        built = t.past.built(c.sha) or {}
        try:
            self.world.enqueue(c.sha, built.get("priority") if isinstance(built.get("priority"), int) else 0)
            what = "enqueued it again, so graph2 lists it afresh, and the next tick reruns"
        except (QueueError, OSError, subprocess.SubprocessError) as e:
            what = f"enqueuing it again failed too ({str(e)[-200:]}): the next tick tries again"
        return self.record(t, "rerun", **line, why=f"graph2 refused the rerun ({why[-200:]}): {what}", failed=failed,
                           keys=list(keys), done=False)

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

    def _flaky(self, t: Tick) -> list[dict]:
        """Ring the owners of each test graph2 says flaked (its verdict's `flaky`: a fail a rerun replaced), once a key: a test
        that flakes is deleted, not rerun.  With --shadow, logged only."""
        w, recs, rung = self.world, [], t.past.flaked()
        for c in t.snap.candidates:
            for e in (t.judged[c.sha].theirs or {}).get("flaky") or ():
                if not isinstance(e, dict) or not isinstance(key := e.get("key"), str) or key in rung:
                    continue
                rung.add(key)
                task, tests = str(e.get("task")), sorted(w.failing_tests(key))
                to = tuple(dict.fromkeys((*w.test_owners(c.sha, task, tests), CI)))
                text = (f"{task} (key {key}) flaked on the verified queue's candidate ready:{c.sha[:12]}: it failed"
                        + (f" ({', '.join(tests)})" if tests else "") + ", then passed when rerun afresh. A test that flakes "
                        "is deleted, not rerun: delete it, or make it deterministic")
                done = not self.shadow and self._say(t, "flaky", text, to, sha=c.sha, key=key)
                recs.append(self.record(t, "flaky", sha=c.sha, base=c.base, members=c.members, heads=c.heads, key=key,
                                        task=task, tests=tests, inputs=e.get("inputs"), text=text, to=list(to), done=done,
                                        **({} if self.shadow else {"posted": done})))
        return recs

    def _bases(self, t: Tick) -> list[dict]:
        """Act on each base whose check a candidate's verdict waits on for want of its graph or graph2's word on it
        (`Judged.waits_on`: not listed yet, its listing failed, its graph can't be read, or graph2's verdict on it hasn't said
        it fails what the candidate's failures are inherited from): its keys and graph2's verdict tell which failures the
        candidate inherits, and its kinds which its graph must list.  `ready:<base>` is enqueued at the first such candidate's
        priority unless it was enqueued less than `BASE_RETRY_S` ago, here or by its build or a retry where it is a candidate
        (`Past.base_enqueues`), so again each `BASE_RETRY_S` while one waits on it: graph2 lists a check whose listing failed
        again, resumes a cancelled one, and leaves one it is listing or has listed as it is.  One that fails to enqueue is
        enqueued by the next tick.  `CI` and `landing.INFRA` are rung, naming it and the candidates on it, once a base and why
        (`Past.rung`): at once when its listing failed or its graph can't be read, and when graph2 still hasn't listed it or
        said `BASE_RING_S` after it was first asked for (`BASE_COMING`); none where the check gates (`CHECK_GATE`), since no
        candidate waits on it there.  Nothing is ejected or bisected on its account: its candidates wait, for as long as
        `vqueue.decide`'s `pending_s` retries and rings them."""
        w, recs, rung, now = self.world, [], t.past.rung(), self.now()
        waiting: dict[str, list[Candidate]] = {}
        for c in t.snap.candidates:
            if (on := t.judged[c.sha].waits_on) is not None:
                waiting.setdefault(on[0], []).append(c)
        for b, cands in waiting.items():
            wait = t.judged[cands[0].sha].waits_on[1]
            first, last = t.past.base_enqueues(b)
            if last is None or now - last >= BASE_RETRY_S:
                built = t.past.built(cands[0].sha) or {}
                line = dict(sha=b, candidate=cands[0].sha, why=f"{', '.join(c.sha[:12] for c in cands)} wait on it: {wait}",
                            priority=built.get("priority") if isinstance(built.get("priority"), int) else 0)
                first = now if first is None else first
                try:
                    w.enqueue(b, line["priority"])
                except (QueueError, OSError, subprocess.SubprocessError) as e:
                    recs.append(self.record(t, "base", **line, enqueued=False, done=False, error=str(e)[-600:]))
                else:
                    recs.append(self.record(t, "base", **line, enqueued=True, done=True))
            why = f"base {b[:12]}: {wait.why}"
            if self.gate == GRAPH2_GATE and (b, why) not in rung and (
                    wait.why not in BASE_COMING or first is not None and now - first >= BASE_RING_S):
                recs.append(self._ring_base(t, b, wait, why, cands, first))
        return recs

    def _ring_base(self, t: Tick, base: str, wait: BaseWait, why: str, cands: list[Candidate], first: float | None) -> dict:
        """Ring `CI` and `landing.INFRA` that ``cands`` wait on ``base``'s check (`_bases`), keyed by ``why``: a post that fails
        is rung again by the next tick."""
        since = (f", {int(self.now() - first) // 60} min after it was first asked for" if wait.why in BASE_COMING and first
                 else "")
        text = (f"the verified queue's base ready:{base} holds up {', '.join(f'ready:{c.sha[:12]}' for c in cands)}: {wait}"
                f"{since}. They wait, and nothing is ejected or bisected for it; it is enqueued again every "
                f"{BASE_RETRY_S // 60} min while they do")
        to, line = (CI, INFRA_HANDLE), dict(sha=base, why=why, candidates=[c.sha for c in cands], text=text)
        if self.shadow:
            return self.record(t, "ring", **line, to=list(to), done=False)
        if not self._say(t, "ring", text, to, sha=base, why=why):
            return self.record(t, "ring", **line, to=list(to), done=False, posted=False,
                               error=f"ringing {', '.join(to)} about base {base[:12]} failed: the next tick rings again")
        return self.record(t, "ring", **line, to=list(to), done=True, posted=True)

    def _disputes(self, t: Tick) -> list[dict]:
        """Ring `CI` about what a candidate is held on (`Judged.disputed`): a task graph2's `verdict` says failed or erred and
        the loop read passed, or one the loop reads its base failing and graph2's verdict on the base doesn't, with the task,
        both keys and both readings (and the base); a listing the loop read that isn't the one graph2 runs, with both.  Once
        each, across the candidates that share it (`Past.rung`, by why).  A post that fails is rung again by the next tick;
        with --shadow it is only logged."""
        recs, rung = [], {why for _, why in t.past.rung()}
        for c in t.snap.candidates:
            for d in t.judged[c.sha].disputed:
                if (why := str(d)) in rung:
                    continue
                rung.add(why)
                what, look = (("its listing", f"Its commit's listings: `research data labels commit:{c.sha} --key "
                                               "verity_graph --history`") if isinstance(d, Relisted) else
                              (d.task, "Each key's results: `research data labels task:<key> --key result --history`"))
                text = (f"the verified queue holds ready:{c.sha} ({' '.join(f'#{n}' for n in c.members)}) while graph2's "
                        f"verdict and the labels it reads disagree: {why}. It waits until they agree. {look}")
                line = dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=why, text=text, to=[CI],
                            **d.logged())
                if self.shadow:
                    recs.append(self.record(t, "ring", **line, done=False))
                elif not self._say(t, "ring", text, (CI,), sha=c.sha, why=why):
                    recs.append(self.record(t, "ring", **line, done=False, posted=False,
                                            error=f"ringing {CI} about {what} on {c.sha[:12]} failed: the next tick rings "
                                                  "again"))
                else:
                    recs.append(self.record(t, "ring", **line, done=True, posted=True))
        return recs

    def _compare(self, t: Tick) -> list[dict]:
        """Where the check gates (`CHECK_GATE`): graph2's verdict of each candidate (`Judged.shadow`) beside its check's, once
        both ended (neither reads pending, neither's fail has a task or a shard still running, the check's not infra, which a
        retry answers), as one line (`_versus`), logged and printed, once a candidate and line.  While the check hasn't ended,
        graph2's verdict that did is logged once (`graph2`), so the compare line to come knows how long it took."""
        recs = []
        for c in t.snap.candidates:
            j, mine = t.judged[c.sha], _ended(c.verdict)
            if j.shadow is None:
                continue
            rec = self._versus(t, c.sha, dict(base=c.base, members=c.members, heads=c.heads), j.shadow, c.enqueued,
                               mine if mine in (PASS, FAIL) else None, _brief(c.verdict), self._check_s(t, c.sha, j.record))
            if rec is not None and not any(r.get("why") == rec["why"] for r in t.past.of("compare", c.sha)):
                recs.append(self._shown(t, rec))
        return recs

    def _off_gate(self, t: Tick) -> list[dict]:
        """Where the check gates (`CHECK_GATE`): ring who acts on what a candidate's check failed of each step `vqueue_slim.judge`
        leaves out (policy.toml's `off_gate` at its base, drive.toml's `warn_steps`), once its verdict is pass or fail, and that
        the run beside its base's check (`Tick.below` beside `main`, else the base candidate's) didn't fail too (`_off_new`):
        whom `_off_owners` names, with each failure's line, once a candidate and step (`Past.rung`, by why).  Nothing is held or
        ejected for it.  A post that fails is rung again by the next tick; with --shadow it is only logged."""
        recs, rung, d = [], t.past.rung(), self.world.slim_driver()
        for c in t.snap.candidates:
            if (j := t.judged[c.sha]).slim is None or j.slim.state not in (PASS, FAIL):
                continue
            base = t.below if c.base == t.main else t.past.slim(c.base)[1]
            for step in sorted({*d._policy(c.base).off_gate, *d.warn_steps}):
                if (c.sha, why := f"off the gate: {step} failed") in rung or not (
                        new := self._off_new(d, c, j.record, base, step)):
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
        ratchet-gate, the lane of each member whose merge it fails, then the step's owner in ci.toml at ``c``; of circuit-check,
        the owner in `main`'s registry of each sidecar's pins file at ``c`` pinning a failed target (`World.pins`), and for a
        target none owns, the lane of each member whose change meets circuit-check's inputs (`queue.forces_cold`), else the
        step's owner; of another, the step's owner.  `CI` when none is known."""
        w, d = self.world, self.world.driver()
        job, lanes, owners = d._step_owner(step, c.sha), [], []
        if step == RATCHET_STEP:
            lanes = [w.handle(int(k[1:])) for k in new if k != step]
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
        (`World.lacking`) the move and the fix: once a head and base, the base the last such move (`Past.told`, as a skip
        is told).  A post that fails is told again by the next tick; with --shadow it is only logged."""
        recs, told = [], t.past.told("lacks")
        for n, (head, move, why) in sorted(self.world.lacking.items()):
            if f"{head}@{move}" in told:
                continue
            text = (f"#{n} at {head[:12]} isn't taken into the verified queue's candidates on `main` {t.main[:12]}: {why} "
                    f"(`research queue ready {n}`)")
            line = dict(base=move, members=[n], heads=[head], number=n, why=why, text=text, to=list(to := self.owner(n)))
            if self.shadow:
                recs.append(self.record(t, "lacks", **line, done=False))
            elif not self._say(t, "lacks", text, to, base=move, heads=[head], number=n):
                recs.append(self.record(t, "lacks", **line, done=False, posted=False,
                                        error=f"telling #{n}'s owner of the move it lacks failed: the next tick tells again"))
            else:
                recs.append(self.record(t, "lacks", **line, done=True, posted=True))
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

    def _versus(self, t: Tick, sha: str, line: dict, shadow: Verdict, enqueued: float | None, mine: str | None, said: str,
                check_s: int | None) -> dict | None:
        """The line of graph2's verdict ``shadow`` of candidate ``sha`` once it ended (`_ended`), None until then.  Beside
        its check's reading ``mine`` (``said`` with what it failed), a compare line: `agree` when both read alike with their
        inherited failures set aside (`vqueue._reading`), else `disagree`, with the tasks graph2 failed or erred on and the
        steps the check failed; ``graph2_s`` is the seconds from ``enqueued`` to the tick that first read graph2's verdict
        so, and ``check_s`` the check's from its launch.  While the check hasn't ended (``mine`` None), a `graph2` line with
        that ``graph2_s``, once a reading, which the compare line to come reads it from."""
        if (theirs := _ended(shadow)) is None:
            return None
        seen = next(reversed(t.past.of("graph2", sha)), None)
        known = seen is not None and seen.get("graph2") == theirs
        took = seen.get("graph2_s") if known else self._since(enqueued)
        if mine is None:
            return None if known else self.record(t, "graph2", sha=sha, **line, graph2=theirs, graph2_s=took, done=True,
                                                  why=f"graph2, in shadow, ended before the check: {_brief(shadow)}")
        why = f"agree: graph2 and the check {mine}" if theirs == mine else f"disagree: graph2 {_brief(shadow)}, the check {said}"
        return self.record(t, "compare", sha=sha, **line, why=why, graph2=theirs, check=mine, graph2_s=took, check_s=check_s,
                           done=True)

    def _shown(self, t: Tick, rec: dict) -> dict:
        """``rec``, a compare line printed as it is logged."""
        if rec["action"] == "compare":
            self.out(f"research queue vqueue: tick {t.n}: {rec['sha'][:12]} graph2 in shadow: {rec['why']}")
        return rec

    def _followed(self, t: Tick) -> list[dict]:
        """The graph2 check of each candidate dropped before it ended (`Past.followed`), read as a candidate's is
        (`World.verdict`) once graph2's own verdict no longer says it runs (`_runs`): ended, it is compared with the check's
        verdict as the drop logged it (`_versus`).  `FOLLOW_S` after its enqueue, or while it runs beside `KEPT_MAX` newer
        ones graph2 says still run (read newest first, so the oldest give way; its line names them, ``followed``), it is
        read once more, and one that still hasn't ended is cancelled and gets its compare line all the same: `unfinished`,
        with how long it ran and how many of its tasks on the merge gate ended.  Either line closes it: a candidate has one.  A read that fails is
        logged and reads as pending: graph2 only shadows here, so it stops no tick.  None with the shadow off."""
        if not self.graph2_shadow:
            return []
        w, recs, kept = self.world, [], []
        w._theirs, w._listed = {}, {}
        try:
            for sha, drop in reversed(t.past.followed().items()):
                f, base = drop["follow"], drop.get("base") or ""
                line = dict(base=base, members=drop.get("members") or (), heads=drop.get("heads") or ())
                enqueued = unstamp(f.get("enqueued"))
                late, over = enqueued is None or self.now() - enqueued >= FOLLOW_S, False
                try:
                    if not late and self._still_runs(sha):
                        if len(kept) < KEPT_MAX:
                            kept.append(sha)
                            continue
                        over = True
                    with w.one_read():
                        j = w.verdict(sha, base, t.past.since(sha), t.past.since(base))
                except Exception as e:  # noqa: BLE001 -- the gate's tick goes on whatever reading the shadow raises
                    j = Judged(Verdict(PENDING), note=f"its read failed ({type(e).__name__}: {e})"[:300])
                    w.log(f"research queue vqueue: ready:{sha[:12]}, followed in shadow: {j.note}")
                rec = self._versus(t, sha, line, j.verdict, enqueued, f.get("check"), f.get("said") or "", f.get("check_s"))
                if rec is None and (late or over):
                    did = f"{j.tasks[0]} of its {j.tasks[1]} tasks on the merge gate ended" if j.tasks else j.note or "no graph"
                    try:
                        cancel = w.cancel(sha)
                    except (OSError, subprocess.SubprocessError) as e:
                        cancel = f"cancel failed: {e}"[:200]
                    rec = self.record(t, "compare", sha=sha, **line, graph2=UNFINISHED, check=f.get("check"),
                                      graph2_s=self._since(enqueued), check_s=f.get("check_s"), cancel=cancel, done=True,
                                      why=f"{UNFINISHED}: graph2 {j.verdict.state} {self._since(enqueued)} s after its enqueue "
                                          f"({did}), the check {f.get('said')}"
                                          + (f"; at most {KEPT_MAX} are followed, and newer ones are" if over else ""),
                                      **({"tasks": list(j.tasks)} if j.tasks else {}),
                                      **({"followed": list(kept)} if over else {}))
                if rec is not None:
                    recs.append(rec)
        finally:
            w._theirs = w._listed = None
        return [self._shown(t, rec) for rec in reversed(recs)]

    def _since(self, at: float | None) -> int | None:
        return None if at is None else round(self.now() - at)

    def _check_s(self, t: Tick, sha: str, rec: dict | None) -> int | None:
        """How long the run ``rec`` records of candidate ``sha`` took, in seconds: from its launch to the line that first
        logged its verdict, this tick's (``t.wrote``) before the log's, whose record a judging this tick wrote over; None
        while none has."""
        if not isinstance(at := (rec or {}).get("at"), (int, float)):
            return None
        judged = next((unstamp(r.get("at")) for r in (*t.wrote, *t.past.of("slim", sha))
                       if r.get("action") == "slim" and r.get("sha") == sha and isinstance(x := r.get("record"), dict)
                       and x.get("at") == at and "verdict" in x), None)
        return None if judged is None else round(judged - at)

    def _ring(self, a: Ring, c: Candidate, t: Tick) -> list[dict]:
        if (c.sha, a.why) in t.past.rung():
            return []
        infra = "; ".join(f"{f.task} (key {f.key})" for f in c.verdict.infra)
        members = " ".join(f"#{n}" for n in c.members)
        text = (f"the verified queue's candidate queue/{c.sha[:12]} ({members}), whose check waits for its records run: {a.why}"
                if c.waits else f"the verified queue's candidate ready:{c.sha} ({members}) is {c.verdict.state} after "
                f"{c.tries} enqueues: {a.why}" + (f". Infra: {infra}" if infra else ""))
        line = dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why, failed=c.verdict.infra, text=text)
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
        tick another candidate's check, or the run beside `main`'s own, reads infra too, as an outage reads, which graph2 can't
        tell from one change's erring.  Infra is rung all the same."""
        if others := [o.sha[:12] for o in t.snap.candidates if o.sha != c.sha and (o.verdict.state == INFRA or o.verdict.infra)]:
            return f"{', '.join(others)} read infra too"
        if ((t.below or {}).get("verdict") or {}).get("state") == INFRA:
            return f"the run beside {t.main[:12]}'s check ended in infra"
        return None

    def _red(self, a: Red, c: Candidate, t: Tick) -> dict:
        """Tell `CI` and the owners of what failed that ``a.base`` is red on it, once a base and task: a post that fails is told
        again by the next red on that base, and the drop after it goes ahead."""
        w, tasks = self.world, "; ".join(f"{f.task} (key {f.key})" for f in a.failed)
        to = tuple(dict.fromkeys((CI, *w.task_owners(c.sha, (f.task for f in a.failed)))))
        new = {f.task for f in a.failed} - t.past.reds(a.base)
        text = None if not new else (
            f"the verified queue's base {a.base[:12]} is red: {a.why}. Failed there as on {c.sha[:12]}: {tasks}. Each result, its "
            f"log's digest with it: `research data labels task:<key> --key result --history`")
        posts = text is not None and not self.shadow
        done = posts and self._say(t, "red", text, to, sha=c.sha, base=a.base, failed=a.failed)
        return self.record(t, "red", sha=c.sha, base=a.base, members=c.members, heads=c.heads, why=a.why, failed=a.failed,
                           text=text, to=list(to), done=done, **({"posted": done} if posts else {}))


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
    ap.add_argument("--state", type=Path, default=STATE, help=f"each tick's candidates, their verdicts beside graph2's and what "
                                                              f"their graphs lack, rewritten each tick (default {STATE})")
    ap.add_argument("--priority-file", type=Path, default=PRIORITY, help=f"the priority list, the landing order: PR numbers, "
                                                                        f"first to land first (default {PRIORITY})")
    ap.add_argument("--frozen", type=Path, default=FROZEN, help=f"a freeze's pathspecs, one a line: while the file exists, a PR "
                                                                f"renaming or deleting a file under one waits (default {FROZEN})")
    ap.add_argument("--deselect", type=Path, default=DESELECT, help=f"the tests a candidate's check leaves out, one node id a line "
                                                                    f"from the repository's root, read at each launch (default "
                                                                    f"{DESELECT})")
    ap.add_argument("--keep-runs", type=Path, default=KEEP_RUNS, help=f"the runs a cancel leaves running, one run id a line, read "
                                                                      f"at each cancel (default {KEEP_RUNS})")
    ap.add_argument("--graph2", default=shlex.join(GRAPH2), help="graph2's argv before its subcommand (default the control pod's)")
    ap.add_argument("--batch", type=int, default=BATCH, help=f"the most members a candidate holds (default {BATCH})")
    ap.add_argument("--depth", type=int, default=DEPTH, help=f"the most candidates stacked at once, each with the run beside "
                                                             f"its check (default {DEPTH})")
    ap.add_argument("--lean", type=int, default=LEAN, help=f"the most members of a Lean candidate, each with a Lean change "
                                                           f"that compiles (a spec's or a Lean source's, not a lock's "
                                                           f"alone): one opens whenever none is staged and no PR "
                                                           f"changing a Lean input with no Lean change waits, of the first "
                                                           f"in rank, whether or not another PR waits; 0: only pinned ones "
                                                           f"open one, which take no cap (--pins-file) (default {LEAN})")
    ap.add_argument("--wide-file", type=Path, default=WIDE,
                    help=f"the PRs whose Lean change is wide, one number a line, read each tick as --priority-file is: a "
                         f"candidate taking one takes no other member --lean counts, and one stacked on a candidate holding "
                         f"one takes none until it is on `main`; none while the file is missing or empty (default {WIDE})")
    ap.add_argument("--pins-file", type=Path, default=PINS,
                    help=f"the pinned PRs, one number a line, `#` comments and blank lines allowed, read each tick as "
                         f"--priority-file is: while a pinned Lean change waits, a Lean candidate holds pinned Lean changes "
                         f"only, under no --lean cap, so a pin orders a Lean candidate or cuts it, and with --lean 0 it is the "
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
    ap.add_argument("--tree-ejects", action="store_true",
                    help="bisect on infra errors graph2 marks `tree`, and eject a lone member for them, as for a fail; off "
                         "(the default), they are retried and rung like any infra error, while graph2 marks `tree` on any "
                         "error that ran out of attempts on two or more nodes, as an outage's do")
    ap.add_argument("--no-slim", action="store_true",
                    help="judge each candidate on its graph2 check alone, with no run beside it of lean-agreement and the "
                         "preflights (research.vqueue_slim), which it otherwise launches on --on's machines beside each "
                         "candidate's check when it builds it; only with --shadow or --dry-run, since `main` would move past "
                         "what that run checks")
    ap.add_argument("--gate", choices=GATES, default=GRAPH2_GATE,
                    help=f"what gates a candidate: `{GRAPH2_GATE}` (the default), its graph2 check with the run beside it; "
                         f"`{CHECK_GATE}`, its whole check (`check.py --record`), every step as --shards shards (but "
                         "lean-audit where no member changes a Lean input and it is built on a candidate in flight whose check "
                         "runs it or took it from the one below that ran it, whose verdict on it it takes once that one has "
                         "landed), with graph2's check in shadow: read, compared with the check's verdict in one logged line "
                         f"once both end, a drop's after it (at most {KEPT_MAX} so at once, the oldest cancelled past them), "
                         "and deciding nothing")
    ap.add_argument("--shards", type=_shards, default=SHARDS,
                    help=f"with --gate {CHECK_GATE}, the shards each check runs as, one launch each: an N ci.toml has a shard "
                         f"plan for, or auto:N, N shards of an auto plan of the candidate's tree made before they launch, "
                         f"which splits pytest by suite (default {SHARDS})")
    ap.add_argument("--on", default="", help="the machines the run beside each check and each records run take, a comma list, "
                                             "each shard its share of them (default drive.toml's `on` "
                                             "at `main`)")
    ap.add_argument("--no-records", action="store_true",
                    help="start the check of a candidate with a member that changes Lean, a lock or a spec at once, with no "
                         "records run before it (research.vqueue_records: lean-audit alone, on --on's machines) "
                         "regenerating its lock records and no records commit of them checked in its place: stale records "
                         "fail it as any failure")
    ap.add_argument("--shadow", action="store_true", help="log advance, eject, ring, red, flakes and messages, and do none of "
                                                          "them; its own main is refs/vqueue/shadow in the clone")
    ap.add_argument("--dry-run", action="store_true", help="print the snapshot and the actions; write nothing")
    ap.add_argument("--no-graph2-shadow", action="store_true",
                    help=f"with --gate {CHECK_GATE}, read no graph2 verdict in shadow (no compare lines, no followed drops); "
                         "graph2's checks are still enqueued")
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
        print("research queue vqueue: research.vqueue_slim isn't in this tree, so no run beside a candidate's check can start: "
              "--no-slim judges each candidate on its graph2 check alone", file=sys.stderr)
        return 1
    if a.gate == CHECK_GATE and (a.no_slim or not _sharded(slim, a.shards)):
        print(f"research queue vqueue: --gate {CHECK_GATE} is the run beside each candidate's check, as one shard or more: "
              "it runs with neither --no-slim nor --shards below 1, and --shards is N (ci.toml's plan) or auto:N (an auto "
              "plan)", file=sys.stderr)
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
        graph2 = tuple(shlex.split(a.graph2))
        world = World(q, labels, Executor(Transport(store, (*graph2, "enqueue"), a.repo)), graph2=graph2,
                      priority_file=a.priority_file, labels_at=labels_at, cache=a.log.parent, on=a.on,
                      frozen_file=a.frozen, deselect_file=a.deselect, wide_file=a.wide_file, pins_file=a.pins_file,
                      keep_file=a.keep_runs, heavy_file=a.heavy_file, heavy_on=a.heavy_on)
        if a.after_landing:
            return _after_landing(world, a.after_landing)
        after = None if a.shadow or a.dry_run else [sys.executable, "-m", "research", "queue", "vqueue", *argv]
        loop = Loop(world, a.log, shadow=a.shadow, dry_run=a.dry_run, batch=a.batch, depth=a.depth, lean=a.lean,
                    pending_s=a.pending_hours * 3600, slim=slim, state=a.state, tree_ejects=a.tree_ejects, gate=a.gate,
                    shards=a.shards, records=None if a.no_records else vqueue_records, graph2_shadow=not a.no_graph2_shadow,
                    after=after)
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


__all__ = ["GRAPH2", "Judged", "Loop", "Past", "Tick", "Transport", "World", "main"]
