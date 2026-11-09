"""`research queue vqueue`: the verified queue's loop (`research.vqueue`).  Each tick, under its own lock (`--lock`, never
drive's), derives a `vqueue.Snapshot` from what is outside it, asks the policy (``decide``, `vqueue.decide`) what to do and
applies its actions in order, stopping the tick at the first that fails: the next tick derives everything again.  Between ticks
it keeps only what a snapshot reads: the refs, the labels, graph2's results and the decision log (`--log`), from which come
the candidates, `history`, each candidate's `tries`, `reruns` and first enqueue, the results a retry or a rerun set aside, the
record of the run beside each check, a ring's, a red's and a flake's once-only and the owners already told.  `--state` is a
report it rewrites each tick: each candidate's verdict beside graph2's and the run beside it's, the jobs its graph lacks, and
the `queue/*` branches it ignores.

The snapshot.  `main` is origin's `main`, fetched into the clone's own refs (`refs/vqueue/`), so drive's clone, its worktree and
its refs are left as drive keeps them.  `ready` is the open PRs drive would land from `next` now: admitted as drive admits them
(`Queue.entries`, reading drive's state, the labels read afresh each tick as drive reads them: `marked_after`,
`signed_verdicts`, a mark carried to a head with the same own change), with a land verdict by a non-owner or under the low-risk
rule (drive.toml's `lowrisk_lane` and `lowrisk_excludes` at `main`), not on `main` yet, ranked by drive's priority file then
mark time, each after the ready PRs it contains; `files` is its change against its merge base with `main`, `lean` drive's
Lean-change test (`queue.lean_change`), and `conflict` whether `Queue._merge` merges it onto `main`, kept by (head, `main`).
`candidates` are the loop's own builds: each of origin's `queue/<sha12>` branches at the sha a build line of the log names
(the window's, else the whole log's, which its other lines are then read from too), with that line's base and members, lowest
first; any other `queue/*` branch is no candidate, and is logged once (`ignore`).  `now` is the clock, `lean` and `pending_s`
the command's (`--lean`, `--pending-hours`).  A candidate's verdict is graph2's results for `ready:<sha>` (`Executor.settle`),
each key's from the first its retry or any candidate's rerun didn't set aside (a task graph2's `verdict` says failed by its
own fault, which it labels no result for, failed), on the merge gate's tasks, those drive's `executor = "on"` path gates on
(none `"gate": false`, of `drive.SLIM_STEPS` or off the gate in policy.toml at the candidate): a failed task an unexpired
allowance of that policy covers passed, as drive reads it (`Driver._task_allowed`); else a failed or blocked task is fail,
each failed task with its inputs (its own, its env's spec files and those of each task it reads), its failing tests
(result/v1's `failed_tests`) and its key and failing tests in its base's graph as graph2 listed it, where there are
(`Failed.base_key`, `Failed.base_tests`), ended once no gating task runs but behind an infra error, and pending while those
are to come: the base's check isn't listed (the loop enqueues it, once a base) or a task's key there waits on what it reads;
else a task whose newest result is graph2's infra error (`detail.infra`; graph2's `tree` where `Failed` has one) is infra,
else every gating task passed is pass, else pending.  No graph while graph2 says its listing failed (`verdict`, else its
`ready_verdict` label), a graph that can't be read, a graph lacking a job `main` waits for (`drive.gating`) and failing
nothing, or no graph of a candidate the log never enqueued is list-failed.  graph2's own verdict, where it
has one, is compared and a disagreement logged.  With ``slim`` (`research.vqueue_slim`), the run beside the check of what
graph2 doesn't check starts once, for the lowest candidate once graph2's verdict reads pass to the policy, so each, on drive's
machines, is one of a candidate about to advance.  Pending until then, it is judged each tick, its record logged whenever it
changes, and combined with the check's verdict: a candidate passes only when both pass.

The actions.  Build merges its members in order onto its base (`Queue._merge`), leaving out one that doesn't merge (`skip`, its
owner told once per head and base), pushes `queue/<sha12>` (or takes as its own one no line names that is the same build made
before, as a tick cut off after its push leaves: `adopted`), logs it and enqueues `ready:<sha>` as graph/v2 at its priority.
Advance pushes `main` to the candidate with a lease on its base and logs it; then what drive does when `main` moves
(`Driver._on_main`) is a step of its own (`on_main`), which the next tick runs again until one is logged done.
Drop cancels the check (graph2's `cancel`, where it has one) and the run beside it while that runs, deletes the branch and logs
its outcome, with its base and its inherited failures, their inputs whole, which `history` reads back.  Eject withdraws the
PR's ready mark at its head (`Queue.eject`, under the clone's queue lock) and tells its owner the failed tasks, their keys and
where their logs are.  Retry enqueues again (graph2 lists a check whose listing failed, reruns what ended in infra errors once
its own verdict is infra, and leaves a pending one as it is), reruns those keys itself while graph2's verdict is another (a
task still runs, or one failed), sets the infra results aside, and starts a run beside it that ended in infra again.  Rerun
runs the keys afresh (graph2's `rerun`), their results set aside, a key of the run beside starting that run again; without
graph2's `rerun` it is logged `absent`, and counts as done.  Ring tells infra's handle (`drive.INFRA`), once a candidate and
why.  Red tells `CI` and the owners of what the base fails (as drive names them: a test file's suite's, else its job's) that it
is red on them, once a base and task.  A test graph2's verdict says flaked (`flaky`: a fail a rerun replaced) is told to its
file's owner in drive's registry and `CI`, once a key: a test that flakes is deleted.  A skip's, ring's, red's or flake's
message is logged as about to be posted (`posting`) before it is, and its record says whether it was (`posted`): one whose
tick stopped in between is told, so no one is told twice, and one whose post failed is told again.  Each action and its
result is one JSON line in the log, whose directory also holds the loop's caches.

`--shadow` logs advance, eject, ring, red, flake and every message with `shadow: true` and does none of them, and builds,
drops, retries, reruns and starts the runs beside: its `main` is the newest candidate it advanced (`refs/vqueue/shadow` in the
clone, never pushed, else its log's newest advance) while origin's `main` is below it, a pass its `main` no longer holds is no
history, and a PR it ejected stays out until marked again.  `--dry-run` prints the snapshot and the actions and writes
nothing: no ref on origin, no log, no enqueue, no label, no message, no run.  The loop never writes drive's state, its
priority file, `next`, `checked` or a label, but an eject's without `--shadow`.

On the control pod it runs beside drive, from a pinned checkout of this tree, in drive's clone (or its own clone of origin):
`research queue vqueue --shadow --lean 3 --log /data/vqueue/decisions.jsonl`, with `--once` for one tick and `--dry-run` to
see one.
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
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

from . import policy, vqueue
from .drive import CI, JOB_OF_KIND, SLIM_STEPS, Driver, cancel_runs, gating, merge_thread_post
from .drive import INFRA as INFRA_HANDLE
from .executor import BLOCKED, Executor, QueueTransport, fold_results, task_key, task_target
from .queue import (PR, WORKDIR, Member, Queue, QueueError, StoreLabels, _git, _out, base_drivers, flags_at, label_verifier,
                    lean_change, lowrisk, merged_tree, priority_list, queue_commit, review_owners, rules_at, unreviewed)
from .vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Action, Advance, Build, Candidate, Drop, Eject, Failed, Outcome,
                     Ready, Red, Rerun, Retry, Ring, Snapshot, Verdict)

#: graph2, infra's executor on the control pod: its argv before the subcommand
GRAPH2 = ("python3.12", "/root/executor/infra/graph2.py", "--db", "/root/executor/graph2.db", "--events", "/root/executor/events")
LOCK, LOG, STATE = Path("/data/vqueue/vqueue.lock"), Path("/data/vqueue/decisions.jsonl"), Path("/data/vqueue/state.json")
#: drive's priority list (its `--priority-file`)
PRIORITY = Path("/data/drive/priority")
#: how far back the decision log is read
WINDOW_S = 48 * 3600
INTERVAL_S = 60
#: the clone's refs the loop fetches into, none of drive's
NS = "refs/vqueue"
MAIN = f"{NS}/main"
#: the shadow's own `main`, never pushed: the newest candidate it advanced
SHADOW = f"{NS}/shadow"
#: no automatic gc or maintenance under drive's own commands in its clone
QUIET = ("-c", "gc.auto=0", "-c", "maintenance.auto=false")
TIME, STAMP_RE = "%Y-%m-%dT%H:%M:%SZ", re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
#: the store label graph2 puts on `commit:<sha>` with its terminal verdict on `ready:<sha>`
READY_VERDICT = "ready_verdict"
#: why a candidate's failures wait on its base (`World._base`) while graph2 hasn't listed the base's check
UNLISTED = "its check isn't listed yet"
#: who ejects, as drive's ejections are by
EJECTOR = "ci"
#: the most first-parent commits a candidate's walk reads above its base
WALK = 64
#: `vqueue.Failed`'s fields: a verdict fills each it has (`_failure`)
FAILED_FIELDS = frozenset(f.name for f in fields(Failed))
#: the PRs judged at once (`World._judge`): a merge onto `main` takes seconds in drive's clone, and a moved `main` judges every
#: ready PR again
JOBS = 8


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime(TIME)


def unstamp(s: object) -> float | None:
    if not (isinstance(s, str) and STAMP_RE.fullmatch(s)):
        return None
    return datetime.strptime(s, TIME).replace(tzinfo=UTC).timestamp()


def branch(sha: str) -> str:
    return f"refs/heads/queue/{sha[:12]}"


def _digest(doc: object) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


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

    def ignored(self) -> set[tuple[str, str]]:
        return {(r.get("ref"), r.get("sha")) for r in self.of("ignore")}

    def flaked(self) -> set[str]:
        return {r.get("key") for r in self.of("flaky") if r.get("done") or r.get("shadow")} | self._posting(
            "flaky", lambda r: {r.get("key")})

    def rung(self) -> set[tuple[str, str]]:
        return {(r["sha"], r.get("why")) for r in self.of("ring") if r.get("done") or r.get("shadow")} | self._posting(
            "ring", lambda r: {(r.get("sha"), r.get("why"))})

    def told(self) -> set[str]:
        def heads(r: dict) -> set[str]:
            return {f"{h}@{r.get('base')}" for h in r.get("heads") or ()}

        return {k for r in self.of("skip") if r.get("done") or r.get("shadow") for k in heads(r)} | self._posting("skip", heads)

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

    def moved(self) -> list[tuple[dict, list[str]]]:
        """Each advance that pushed `main` and logged drive's steps after it as their own (`on_main`) with no record of them
        done, and the stacked PRs' owners their tries told."""
        out = []
        for r in self.of("advance"):
            if not (r.get("on_main") and r.get("done") and r.get("sha")):
                continue
            ons = self.of("on_main", r["sha"])
            if not any(o.get("done") for o in ons):
                out.append((r, list((ons[-1].get("told") if ons else None) or ())))
        return out


@dataclass
class Judged:
    """A candidate's verdict with what it rests on: ``mark`` names the terminal infra or list-failed evidence a retry answers,
    ``graph2`` is graph2's own verdict (``theirs`` its whole JSON), ``lacks`` the jobs `main` waits for (`drive.gating`) that no
    task of the graph runs, which drive's `executor = "on"` path takes as no promotion and this verdict as list-failed,
    ``unlisted`` the base whose unlisted check its failures wait on, and ``slim`` the verdict of the run beside the check
    (`vqueue_slim.judge` on its ``record``; pending until it starts), combined into ``verdict``."""
    verdict: Verdict
    mark: str | None = None
    note: str = ""
    graph2: str | None = None
    lacks: tuple[str, ...] = ()
    #: whether graph2 listed it, or said its listing failed
    listed: bool = True
    theirs: dict | None = None
    slim: Verdict | None = None
    record: dict | None = None
    unlisted: str | None = None


class _Seen:
    """A transport that keeps each key's results as `Executor.settle` reads them (an infra error folds to pending there), each
    key's from its ``since``-th on: a retry or a rerun set the earlier ones aside.  A key of ``faulted`` ({key: its fail}) with
    neither a pass nor a fail among them reads as that fail."""

    def __init__(self, transport, since: dict[str, int] | None = None, faulted: dict[str, dict] | None = None):
        self.transport, self.since, self.faulted, self.got = transport, since or {}, faulted or {}, {}

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        if isinstance(self.transport, Transport):
            got = self.transport.results(keys, self.since)
        else:
            got = {k: v[self.since.get(k, 0):] for k, v in self.transport.results(keys).items()}
        for k in keys:
            if (doc := self.faulted.get(k)) is not None and fold_results(got.get(k) or (), k)[0] == PENDING:
                got[k] = [*(got.get(k) or ()), doc]
        self.got.update(got)
        return got


class Transport(QueueTransport):
    """`QueueTransport`, pulling a key's results from the store's remote again while none of them passed or failed: graph2 labels
    an infra error as a result, and a retry's or a rerun's result comes after it.  With ``since``, each key's from its
    ``since``-th on, so one set aside reads as no result and is pulled again."""

    def results(self, keys: list[str], since: dict[str, int] | None = None) -> dict[str, list[object]]:
        from .queue import PREFETCH_JOBS
        from .store import labels_sync, vocab

        since = since or {}

        def have(k: str) -> list[object]:
            return [lab.value for lab in self.store.labels(task_target(k), key=vocab.TASK_RESULT)][since.get(k, 0):]

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
    drive's files."""

    def __init__(self, q: Queue, labels, tasks: Executor, *, graph2: tuple[str, ...] = (), priority_file: Path | None = None,
                 post: Callable[[str, tuple[str, ...]], bool] | None = None, now: Callable[[], float] = time.time,
                 labels_at: Callable[[], StoreLabels] | None = None, cache: Path | None = None):
        self.q, self.labels, self.tasks, self.graph2 = q, labels, tasks, tuple(graph2)
        self.priority_file, self.now, self.repo, self.labels_at = priority_file, now, q.workdir, labels_at
        self.keys = cache / "issuer-keys.json" if cache is not None else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            q._kept_at = cache / "own-changes.json"
            q._kept = json.loads(q._kept_at.read_text()) if q._kept_at.is_file() else {}
        self.post: Callable[[str, tuple[str, ...]], bool] = post or merge_thread_post
        #: graph2's subcommands it was found to lack (`_graph2`)
        self.absent: set[str] = set()
        #: by (head, `main`): its files (`_changed`), and (lean, conflict) (`_judge`)
        self._files: dict[tuple[str, str], frozenset[str]] = {}
        self._judged: dict[tuple[str, str], tuple[bool, bool]] = {}
        self._prs: dict[int, PR] | None = None
        self._driver: Driver | None = None
        self._slim: Driver | None = None
        self._main: str | None = None
        #: what the last `ready` read: {number: the PR at its ready head}
        self.prs: dict[int, PR] = {}

    def log(self, text: str) -> None:
        self.q.log(text)

    # -- reads

    def fetch(self) -> str:
        """origin's `main`, fetched with origin's `queue/*` branches into `NS`."""
        self._prs, self.q._registry = None, None
        self.q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--prune", "origin",
                       f"+refs/heads/main:{MAIN}", f"+refs/heads/queue/*:{NS}/queue/*")
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
        rules'), authenticated under its `signed_verdicts`, and a head's carried to a new head with the same own change."""
        q, flags = self.q, flags_at(self.repo, main)
        self.labels = labels = self.labels_at()
        labels.since, labels.change = flags.marked_after or q.rules().marked_after, q.own_change
        if flags.signed_verdicts:
            labels.verify = label_verifier(self.repo, MAIN, self.keys, q.log)
        q.ready, q.before_cutoff, q.marked_by, q.marked_at, q.prefetch = (labels.ready, labels.before_cutoff, labels.marked_by,
                                                                          labels.marked_at, labels.prefetch)

    def open(self) -> dict[int, PR]:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs

    def _missing(self, shas: Iterable[str]) -> set[str]:
        if not (shas := list(dict.fromkeys(shas))):
            return set()
        r = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"], cwd=self.repo, check=True,
                           input="".join(f"{s}\n" for s in shas), capture_output=True, text=True)
        return {line.split()[0] for line in r.stdout.splitlines() if line.endswith(" missing")}

    def _ancestor(self, a: str, b: str) -> bool:
        return not _git(self.repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    def ready(self, main: str) -> list[Ready]:
        """The open PRs drive would land from `next` now, as `vqueue.Ready`."""
        q, labels = self.q, self.labels
        prs = [p for p in self.open().values() if p.base == "main" and not p.move]
        if missing := [p for p in prs if p.head in self._missing(p.head for p in prs)]:
            q._remote(self.repo, *QUIET, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "origin",
                      *(f"+refs/pull/{p.number}/head:{NS}/pr/{p.number}" for p in missing))
        gone = self._missing(p.head for p in prs)
        prs = [p for p in prs if p.head not in gone and not self._ancestor(p.head, main)]
        labels.prefetch([(p.number, p.head) for p in prs])
        entries = q.entries(prs, q.load_state(), base=main, landed=main, queued=main, open_numbers=set(self.open()))
        flags = flags_at(self.repo, main)
        excludes = tuple(flags.lowrisk_excludes) if flags.lowrisk_lane else ()
        self._files = {k: v for k, v in self._files.items() if k[1] == main}
        self._judged = {k: v for k, v in self._judged.items() if k[1] == main}
        admitted = [n for n, e in entries.items() if e.state == "ready"]
        with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:
            files = dict(zip(admitted, pool.map(lambda n: self._changed(main, entries[n].pr.head), admitted)))
            took = []
            for n in admitted:
                p, reviews = entries[n].pr, labels.reviews(n, entries[n].pr.head)
                landing = unreviewed(reviews, review_owners(q, labels, n, p.head, p.owner, self.open())) is None
                held = (unreviewed(reviews, ()) or "").startswith("held")
                if landing or not held and lowrisk(files[n], excludes):
                    took.append(n)
            new = any((entries[n].pr.head, main) not in self._judged for n in took)
            with base_drivers(self.repo, main) if new else contextlib.nullcontext(None) as cfg:   # once a tick, not a PR
                chosen = {n: (files[n], *j)
                          for n, j in zip(took, pool.map(lambda n: self._judge(main, entries[n].pr.head, cfg), took))}
        first = {n: i for i, n in enumerate(priority_list(self.priority_file) if self.priority_file else ())}
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
        self.prs = {n: entries[n].pr for n in chosen}
        return [Ready(n, entries[n].pr.head, *chosen[n][:2], rank=i, conflict=chosen[n][2]) for i, n in enumerate(placed)]

    def _changed(self, main: str, head: str) -> frozenset[str]:
        """The files ``head`` changes against its merge base with ``main``."""
        if (key := (head, main)) not in self._files:
            self._files[key] = frozenset(_out(self.repo, "diff", "--name-only", "--no-renames", f"{main}...{head}").splitlines())
        return self._files[key]

    def _judge(self, main: str, head: str, config: tuple[str, ...] | None = None) -> tuple[bool, bool]:
        """(whether ``head`` is a Lean change, whether it doesn't merge onto ``main``), kept by (head, main).  The merge is `git
        merge-tree`'s alone (`merged_tree`, no worktree, with ``main``'s merge drivers: ``config``, else `base_drivers`), so
        judging every ready PR stays cheap; a build's codemod (`Queue._merge`) checks out a worktree, and a member it refuses is
        left out of that build as one that doesn't merge."""
        if (key := (head, main)) not in self._judged:
            self._judged[key] = (lean_change(self.repo, head, _out(self.repo, "merge-base", main, head)),
                                 merged_tree(self.repo, main, head, config=config)[0] is None)
        return self._judged[key]

    def refs(self) -> dict[str, str]:
        """origin's `queue/*` branches as fetched: {name under `queue/`: sha}."""
        out = _out(self.repo, "for-each-ref", "--format=%(refname) %(objectname)", f"{NS}/queue/")
        return {ref.removeprefix(f"{NS}/queue/"): sha for ref, sha in (line.split() for line in out.splitlines())}

    def members(self, sha: str, base: str, prs: Iterable[int], heads: Iterable[str]) -> list[Member]:
        """Candidate ``sha``'s members as `Driver._on_main` reads them, oldest first: the `Queue: #N` merges on its first-parent
        line above ``base``, while they are the PRs ``prs`` at ``heads`` as built; else those, without their merges."""
        prs, heads = list(prs), list(heads)
        if (walked := self._walk(sha, base)) is not None and [(m.pr, m.head) for m in walked] == list(zip(prs, heads)):
            return walked
        self.log(f"research queue vqueue: {sha[:12]}'s first-parent line above {base[:12]} isn't its members' merges as built")
        return [Member(p, h, "", "") for p, h in zip(prs, heads)]

    def _walk(self, sha: str, base: str) -> list[Member] | None:
        """The `Queue: #N` merges on ``sha``'s first-parent line above ``base``, oldest first; None when another commit is."""
        recs = [r.strip() for r in _out(self.repo, "log", "--first-parent", f"-n{WALK}", "--format=%H%x00%P%x00%B%x1e",
                                         f"{base}..{sha}").split("\x1e") if r.strip()]
        walked = [queue_commit(*r.split("\x00", 2)) for r in reversed(recs)]
        return walked if all(walked) else None

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
        ``base_since``."""
        check = f"ready:{sha}"
        theirs = self.graph2_verdict(sha)
        said = theirs.get("state") if isinstance(theirs, dict) else None
        try:
            got = self.tasks.graph_v2(check)
        except ValueError as e:
            return Judged(Verdict(LIST_FAILED), mark=f"unreadable {_digest(str(e))}", note=str(e), graph2=said, theirs=theirs)
        if got is None:
            if said == LIST_FAILED:
                return Judged(Verdict(LIST_FAILED), mark=f"list-failed {_digest(theirs)}", note=str(theirs.get("why") or ""),
                              graph2=said, theirs=theirs)
            return Judged(Verdict(PENDING), note="not listed yet", graph2=said, listed=False, theirs=theirs)
        _, graph = got
        tasks, envs = {t["id"]: t for t in graph["tasks"]}, graph.get("envs") or {}
        if isinstance(pol := policy.at(self.repo, sha), str):
            self.log(f"research queue vqueue: {pol}: every job gates at {sha[:12]}")
            pol = policy.Policy()
        gates = [i for i, t in tasks.items() if t.get("gate", True) is not False
                 and JOB_OF_KIND.get(t["kind"], t["kind"]) not in (*SLIM_STEPS, *pol.off_gate)]
        if not gates:
            return Judged(Verdict(LIST_FAILED), mark=f"empty {_digest(sorted(tasks))}", note="no task on the merge gate",
                          graph2=said, theirs=theirs)
        states, keys, seen = self._settled(graph, since, _faulted(theirs))
        inputs: dict[str, frozenset[str]] = {}

        def reach(i: str) -> frozenset[str]:
            """``i``'s inputs, its env's spec files and those of each task it reads, transitively."""
            if i not in inputs:
                inputs[i] = frozenset()
                t = tasks[i]
                inputs[i] = frozenset(t["inputs"]).union((envs.get(t["env"]) or {}).get("spec") or (),
                                                         *(reach(r["task"]) for r in t["reads"]))
            return inputs[i]

        def under(i: str, seen_: set[str]) -> list[str]:
            """The tasks ``i`` reads, transitively, ``i`` first."""
            if i in seen_:
                return []
            seen_.add(i)
            return [i, *(j for r in tasks[i]["reads"] for j in under(r["task"], seen_))]

        needed = list(dict.fromkeys(j for i in gates for j in under(i, set())))
        allowed = self._allowed(sha, tasks, states, needed)
        bad: list[str] = []
        for i in gates:
            if states[i][0] == FAIL and i not in allowed:
                bad.append(i)
            elif states[i][0] == BLOCKED:
                bad += [j for j in under(i, set()) if states[j][0] == FAIL and j not in allowed] or [i]
        infra = [i for i in needed if states[i][0] == PENDING and _infra(seen.got.get(keys[i]) or (), keys[i])]
        ended = all(states[i][0] != PENDING or set(under(i, set())) & set(infra) for i in gates)
        jobs = {JOB_OF_KIND.get(t["kind"], t["kind"]) for t in tasks.values()}
        lacks = tuple(s for s in gating(self.repo, sha) if s not in jobs and s not in SLIM_STEPS)
        based = self._base(base, set(bad), base_since) if bad else {}
        waits = based if isinstance(based, str) else None
        based = {} if waits is not None else based
        tree = {e.get("task"): bool(e.get("tree")) for e in (theirs or {}).get("infra") or () if isinstance(e, dict)}
        fail = () if waits is not None else tuple(
            _failure(i, keys[i], reach(i), base_key=based.get(i, (None, None))[0], tests=_tests(states[i][1]),
                     base_tests=based.get(i, (None, None))[1]) for i in dict.fromkeys(bad))
        down = tuple(_failure(i, keys[i], reach(i), tree=tree.get(i, False)) for i in infra)
        if fail:
            mine = Verdict(FAIL, fail, down, ended=ended)
        elif lacks:
            mine = Verdict(LIST_FAILED)
        elif waits is not None:
            mine = Verdict(PENDING)
        elif down:
            mine = Verdict(INFRA, (), down)
        elif all(states[i][0] == PASS or i in allowed for i in gates):
            mine = Verdict(PASS)
        else:
            mine = Verdict(PENDING)
        mark = (f"lacks {' '.join(lacks)}" if mine.state == LIST_FAILED else
                f"infra {_digest(sorted((keys[i], len(seen.got.get(keys[i]) or ())) for i in infra))}" if mine.state == INFRA
                else None)
        note = "; ".join([*([f"graph2 says {said}"] if said in vqueue.STATES and said != mine.state else []),
                          *([f"no task of {', '.join(lacks)}, which `main` waits for"] if lacks else []),
                          *([f"waiting for the base's listing: {waits}"] if waits is not None else []),
                          *([f"allowed: {', '.join(sorted(allowed))}"] if allowed else [])])
        return Judged(mine, mark=mark, note=note, graph2=said, lacks=lacks, theirs=theirs,
                      unlisted=base if waits == UNLISTED else None)

    def _allowed(self, sha: str, tasks: dict[str, dict], states: dict[str, tuple], ids: Iterable[str]) -> set[str]:
        """The failed tasks of ``ids`` an allowance of ``sha``'s policy covers, each red record they name, as drive's
        `executor = "on"` path reads them (`Driver._task_units`, `Driver._task_allowed`): drive takes each as passed."""
        out = set()
        for i in ids:
            if states[i][0] == FAIL:
                d, res = self.driver(), states[i][1] or {}
                if all(d._task_allowed(sha, tasks[i], res, row).covered for row in d._task_units(sha, tasks[i], res).values()):
                    out.add(i)
        return out

    def _settled(self, graph: dict, since: dict[str, int] | None = None, faulted: dict[str, dict] | None = None
                 ) -> tuple[dict[str, tuple], dict[str, str], _Seen]:
        """``graph``'s tasks settled on graph2's results (`Executor.settle`), each key's from its ``since``-th on, a key of
        ``faulted`` with none read as its fail (`_Seen`), each one's key, the graph's or one made from its reads ("" while they
        haven't all passed), and the results read."""
        seen = _Seen(self.tasks.transport, since, faulted)
        states = Executor(seen).settle(graph)
        files, envs = graph.get("files") or {}, graph.get("envs") or {}
        keys: dict[str, str] = {}
        for t in graph["tasks"]:
            reads = [states[r["task"]] for r in t["reads"]]
            keys[t["id"]] = t.get("key") or (task_key(t, files, envs[t["env"]]["digest"],
                                                     {r["task"]: s[1]["outputs"] for r, s in zip(t["reads"], reads)})
                                             if states[t["id"]][0] != BLOCKED and all(s[0] == PASS for s in reads) else "")
        return states, keys, seen

    def _base(self, base: str, ids: set[str], since: dict[str, int] | None = None
              ) -> dict[str, tuple[str | None, frozenset[str] | None]] | str:
        """For each of tasks ``ids`` in ``base``'s graph, as graph2 listed it: its key there, where one is made (graph2 keys a
        task by all it reads, so a candidate's task at its base's key fails as it did there: `vqueue.Failed.base_key`), and its
        failing tests there, None unless it failed naming them (`vqueue.Failed.base_tests`).  Why not yet, while that is to
        come: `ready:<base>` isn't listed (`UNLISTED`), or a task's key there waits on what it reads; no failure is the
        candidate's own until then.  A base whose listing failed, or whose graph can't be read, has none (logged)."""
        if not base:
            return {}
        try:
            got = self.tasks.graph_v2(f"ready:{base}")
        except ValueError as e:
            self.log(f"research queue vqueue: {base[:12]}'s graph can't be read, so no failure on it is inherited: {e}")
            return {}
        if got is None:
            theirs = self.graph2_verdict(base)
            if isinstance(theirs, dict) and theirs.get("state") == LIST_FAILED:
                self.log(f"research queue vqueue: ready:{base[:12]}'s listing failed ({str(theirs.get('why') or '')[-200:]}), "
                         "so no failure on it is inherited: each is its candidate's own")
                return {}
            return UNLISTED
        states, keys, seen = self._settled(got[1], since)
        stuck = any(_infra(seen.got.get(k) or (), k) for k in keys.values() if k)      # what they read may never pass
        if not stuck and (waiting := sorted(i for i in ids if i in keys and not keys[i] and states[i][0] == PENDING)):
            return f"{', '.join(waiting)} there still wait on what they read"
        return {i: (keys[i] or None, (_tests(states[i][1]) or None) if states[i][0] == FAIL else None)
                for i in ids if i in keys}

    def counts(self, keys: Iterable[str]) -> dict[str, int]:
        """How many results each of graph2's ``keys`` has (`sha512:…`; a key of the run beside the check has none)."""
        keys = [k for k in dict.fromkeys(keys) if k.startswith("sha512:")]
        return {k: len(v) for k, v in self.tasks.transport.results(keys).items()} if keys else {}

    def rerun(self, sha: str, keys: Iterable[str]) -> str | None:
        """graph2's `rerun` of ``keys`` on `ready:<sha>`, as said; None when graph2 has no `rerun`."""
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

    def graph2_verdict(self, sha: str) -> dict | None:
        """graph2's own verdict on `ready:<sha>`: its `verdict` command's JSON, else its `ready_verdict` label's, else None."""
        if (r := self._graph2("verdict", "--check", f"ready:{sha}")) is not None:
            if r.returncode:
                self.log(f"research queue vqueue: graph2 verdict ready:{sha[:12]} exited {r.returncode}: {r.stderr.strip()[-300:]}")
                return None
            try:
                got = json.loads(r.stdout)
            except ValueError as e:
                self.log(f"research queue vqueue: graph2 verdict ready:{sha[:12]} printed no JSON ({e}): {r.stdout.strip()[-300:]}")
                return None
            return got if isinstance(got, dict) else None
        store = getattr(self.tasks.transport, "store", None)
        if store is None:
            return None
        labs = store.labels(f"commit:{sha}", key=READY_VERDICT)
        if not labs and getattr(store, "remote", None) is not None:
            from .store import labels_sync

            labels_sync.pull_targets(store, [f"commit:{sha}"])
            labs = store.labels(f"commit:{sha}", key=READY_VERDICT)
        if not labs:
            return None
        v = labs[-1].value
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError as e:
                self.log(f"research queue vqueue: commit:{sha[:12]}'s {READY_VERDICT} label holds no JSON ({e})")
                return None
        return {"labelled": labs[-1].ts, **v} if isinstance(v, dict) else None     # each verdict its own, for a retry's mark

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

    def build(self, prs: list[PR], base: str) -> tuple[str, list[Member], list[tuple[PR, str]]]:
        """``prs`` merged in order onto ``base`` (`Queue._merge`): (the tip, the members merged, (PR, why) of each left out)."""
        tip, merged, left = base, [], []
        for p in prs:
            commit, why = self.q._merge(tip, p, merged)
            if commit is None or commit == tip:
                left.append((p, why or f"its head {p.head[:12]} is already in what it merges onto"))
                continue
            merged.append(Member(p.number, p.head, p.owner or "", commit))
            tip = commit
        return tip, merged, left

    def push(self, sha: str, ref: str, lease: str | None = None) -> None:
        """origin's ``ref`` to ``sha`` ("": deleted), with a lease on ``lease`` when given."""
        self.q._remote(self.repo, "push", "-q", *([f"--force-with-lease={ref}:{lease}"] if lease is not None else []), "origin",
                       f"{sha}:{ref}" if sha else f":{ref}")

    def enqueue(self, sha: str, priority: int) -> None:
        try:
            self.tasks.enqueue_graph(f"ready:{sha}", priority)
        except RuntimeError as e:
            raise QueueError(f"enqueue ready:{sha[:12]}: {e}") from None

    def cancel(self, sha: str) -> str:
        """graph2's `cancel` of `ready:<sha>`, as said."""
        if (r := self._graph2("cancel", "--check", f"ready:{sha}")) is None:
            return "no cancel"
        return "cancelled" if r.returncode == 0 else f"cancel exited {r.returncode}: {(r.stderr or r.stdout).strip()[-200:]}"

    def cancel_runs(self, ids: Iterable[str], why: str) -> dict[str, str]:
        """`research cancel` of runs ``ids``, as drive cancels a run beside a check: {run id: why} of each that failed."""
        return cancel_runs(self.repo, list(ids), why)

    def advance(self, sha: str, base: str) -> None:
        """origin's `main` fast-forwarded from ``base`` to ``sha``."""
        self.q._drive_only("main", "advance", False)
        self.push(sha, "refs/heads/main", base)

    def on_main(self, sha: str, base: str, members: list[Member], told: list[str]) -> None:
        """What drive does once ``members`` are on `main` at ``sha`` from ``base`` (`Driver._on_main`); ``told``, the stacked
        PRs whose owners were told, is drive's list it extends as it tells."""
        self.driver()._on_main({"told": told, "stale_closed": self.q.load_state().get("drive", {}).get("stale_closed", {})},
                               members, base, sha, "")

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
        """The handles owning tasks ``ids`` of ``sha``'s graph, as drive names the owner of what failed (`Driver._red_owners`):
        a test file's suite's in the registry, else its job's in ci.toml at ``sha``."""
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

    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = Driver(self.q, on="", post=self.post, now=self.now)
        return self._driver

    def slim_driver(self) -> Driver:
        """The Driver the run beside each check is launched and judged with (`vqueue_slim`), on drive's machines: drive.toml's
        `on` at `main` when first asked.  It outlives the ticks, as `vqueue_slim.judge` expects."""
        if self._slim is None:
            if not (on := flags_at(self.repo, self._main or MAIN).on):
                raise QueueError(f"drive.toml at main {(self._main or MAIN)[:12]} names no `on`, drive's machines, so no run "
                                 "beside a candidate's check can launch: name them (--no-slim runs only with --shadow or --dry-run)")
            self._slim = Driver(self.q, on=on, post=self.post, now=self.now)
        return self._slim


def _suite(t: dict) -> str | None:
    """A test file task's suite, the directory its result's test ids are relative to."""
    return s if t.get("kind") == "test-file" and isinstance(s := (t.get("vars") or {}).get("suite"), str) else None


def _tests(doc: object) -> frozenset[str]:
    """The failing tests a result names (result/v1's `failed_tests`)."""
    got = (doc.get("detail") or {}).get("failed_tests") if isinstance(doc, dict) else None
    return frozenset(x for x in got if isinstance(x, str)) if isinstance(got, list) else frozenset()


def _faulted(theirs: dict | None) -> dict[str, dict]:
    """The tasks graph2's `verdict` says failed by their own fault (an undeclared read, a race: a failed entry with a `why`),
    for which it labels no result: {key: the fail each reads as}.  Not from a `ready_verdict` label, which a rerun leaves
    standing until the check ends again."""
    if not isinstance(theirs, dict) or "labelled" in theirs:
        return {}
    return {k: {"key": k, "task": e.get("task"), "status": FAIL, "detail": {"fault": e["why"]}}
            for e in theirs.get("failed") or () if isinstance(e, dict) and e.get("why") and isinstance(k := e.get("key"), str)}


def _failure(task: str, key: str, inputs: frozenset[str], **kw) -> Failed:
    """A `vqueue.Failed`, each of ``kw`` it has a field for."""
    return Failed(task, key, inputs, **{k: v for k, v in kw.items() if k in FAILED_FIELDS})


def _infra(docs: Iterable[object], key: str) -> bool:
    """Whether ``key``'s newest result is graph2's infra error, its attempts run out."""
    mine = [d for d in docs if isinstance(d, dict) and d.get("key") == key]
    return bool(mine) and mine[-1].get("status") == "error" and (mine[-1].get("detail") or {}).get("infra") is True


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


class Loop:
    """The verified queue's loop on ``world``: `tick` once, or `run` every ``interval`` seconds.  ``slim`` is
    `research.vqueue_slim` (its `start`, `judge` and `combine`), or None to judge each candidate on its graph2 check alone."""

    def __init__(self, world: World, log: Path, *, decide: Callable[[Snapshot], list[Action]] = vqueue.decide,
                 shadow: bool = False, dry_run: bool = False, batch: int = 6, depth: int = 2, lean: int = 1,
                 pending_s: float = 4 * 3600, slim=None, now: Callable[[], float] = time.time,
                 out: Callable[[str], None] = print, state: Path | None = None):
        self.world, self.log_path, self.decide, self.shadow, self.dry_run = world, Path(log), decide, shadow, dry_run
        self.batch, self.depth, self.lean, self.pending_s, self.slim = batch, depth, lean, pending_s, slim
        self.now, self.out, self.state = now, out, state
        #: the last tick this loop ran: a tick that logs nothing still counts
        self.last = 0

    # -- the snapshot

    def snapshot(self, past: Past, n: int) -> Tick:
        w = self.world
        origin = w.fetch()
        main = self._shadow_main(past, origin) if self.shadow else origin
        ejected = {(r.get("number"), (r.get("heads") or [None])[0]): r.get("at") for r in past.of("eject") if r.get("shadow")}
        ready = []
        for x in w.ready(main):
            at = ejected.get((x.number, x.head))
            if at is not None and not (w.labels.marked_at(x.number, x.head) or "") > at:
                continue
            ready.append(x)
        refs, builds, whole = w.refs(), past.builds(), past
        if older := {sha for name, sha in refs.items() if name == sha[:12] and sha not in builds
                     and (name, sha) not in past.ignored()}:                # past the window: read the whole log for them
            whole = Past.read(self.log_path, 0)
            builds |= {sha: r for sha, r in whole.builds().items() if sha in older}
        mine = {sha for name, sha in refs.items() if name == sha[:12] and sha in builds}
        foreign = {name: sha for name, sha in refs.items() if sha not in mine or name != sha[:12]}
        cands, judged, wrote = [], {}, []
        for sha in sorted(mine):
            p, b = whole if sha in older else past, builds[sha]
            base, prs, heads = b.get("base") or "", tuple(b.get("members") or ()), tuple(b.get("heads") or ())
            j = w.verdict(sha, base, p.since(sha), p.since(base))
            if not j.listed and not p.tries(sha):           # pushed, and its enqueue failed or never ran
                j = Judged(Verdict(LIST_FAILED), mark="not enqueued", note="the log holds no enqueue of it", graph2=j.graph2,
                           listed=False, theirs=j.theirs)
            if self.slim is not None and not w._ancestor(sha, main):
                j = self._beside(p, n, main, origin, sha, base, prs, heads, j, wrote)
            judged[sha] = j
            cands.append(Candidate(sha, base, prs, heads, j.verdict, p.tries(sha), reruns=p.reruns(sha),
                                   enqueued=p.enqueued(sha)))
        on = {c.sha: c.base for c in cands}

        def height(sha: str, seen: tuple[str, ...] = ()) -> int:
            return 1 + height(on[sha], (*seen, sha)) if on.get(sha) in on and sha not in seen else 0

        cands.sort(key=lambda c: (height(c.sha), c.sha))       # each after the candidate it is built on
        history = past.history((lambda sha: bool(sha) and w._ancestor(sha, main)) if self.shadow else None)
        snap = Snapshot(main, tuple(sorted(ready, key=lambda x: x.rank)), tuple(cands), history, self.batch, self.depth,
                        lean=self.lean, now=self.now(), pending_s=self.pending_s)
        return Tick(n, main, origin, snap, judged, whole, past.told(), foreign, wrote)

    def _shadow_main(self, past: Past, origin: str) -> str:
        """The shadow's `main`: the newest candidate it advanced (`SHADOW`, else its log's newest advance) while origin's `main`
        is below it; else origin's."""
        w = self.world
        for sha in (w.shadow_main(), next((r["sha"] for r in reversed(past.of("advance")) if r.get("shadow") and r.get("sha")),
                                          None)):
            if sha and not w._missing([sha]) and w._ancestor(origin, sha):
                return sha
        return origin

    def _beside(self, past: Past, n: int, main: str, origin: str, sha: str, base: str, prs: tuple[int, ...],
                heads: tuple[str, ...], j: Judged, wrote: list[dict]) -> Judged:
        """``j`` combined with the verdict of the run beside candidate ``sha``'s check (`vqueue_slim`), pending until it starts:
        here, once, for the lowest candidate (on ``main``) once graph2's verdict reads pass to the policy (`vqueue._reading`:
        an inherited fail once nothing else runs, too), so each run beside a check, on drive's machines, is one of a
        candidate about to advance; again by a retry of its infra or a rerun of its failure.  Judged each tick, its record
        logged (to ``wrote`` too) whenever it changed; with --dry-run neither started nor judged.  A check that failed to list
        stays list-failed beside a run that hasn't ended, which `combine` reads as pending: one that never starts would hold
        it so."""
        started, rec = past.slim(sha)
        line = dict(sha=sha, base=base, members=prs, heads=heads)
        if self.dry_run or not started and (base != main or vqueue._reading(j.verdict) != PASS):
            v = Verdict(PENDING)
        else:
            if not started:
                self.write(got := self._started(n, main, origin, line))
                wrote.append(got)
                rec = got["record"]
            was = json.dumps(rec, sort_keys=True)
            v = self.slim.judge(self.world.slim_driver(), rec)
            if json.dumps(rec, sort_keys=True) != was:
                self.write(got := self._line(n, main, origin, "slim", **line, why=f"judged {v.state}", record=rec, done=True))
                wrote.append(got)
        if j.verdict.state == LIST_FAILED and v.state == PENDING:
            return replace(j, slim=v, record=rec)
        both = self.slim.combine(j.verdict, v)
        if both.state == FAIL:              # `combine` keeps no `ended`: a fail has ended once both checks stopped running
            g = j.verdict
            both = replace(both, ended=(g.state != PENDING and (g.state != FAIL or g.ended)) and v.state != PENDING)
        return replace(j, verdict=both, slim=v, record=rec)

    # -- a tick

    def tick(self) -> list[dict]:
        """One tick: the records it logged (with --dry-run, those it would have)."""
        past = Past.read(self.log_path, self.now() - WINDOW_S)
        self.last = max(past.tick, self.last) + 1
        t = self.snapshot(past, self.last)
        if self.dry_run:
            self.out(self.summary(t))
        actions = list(self.decide(t.snap))
        if self.dry_run:
            self.out("\n".join(f"would {_name(a)}: {a}" for a in actions) or "nothing to do")
            return []
        new = [self.record(t, "ignore", sha=sha, ref=name, why=f"origin's queue/{name} is no build of the loop's: no candidate")
               for name, sha in t.foreign.items() if (name, sha) not in past.ignored()] + self._flaky(t) + self._bases(t)
        new += [self._on_main(t, adv, told) for adv, told in t.past.moved()]
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
            for rec in recs:
                self.write(rec)
            logged += recs
            if failed := next((r["error"] for r in recs if r.get("error")), None):
                self.out(f"research queue vqueue: tick {t.n}: {_name(a)} failed, the tick stops: {failed}")
                break
            if isinstance(a, Advance) and not self.shadow:
                self.write(rec := self._on_main(t, recs[0], []))
                logged.append(rec)
        for sha, j in t.judged.items():
            if j.note:
                self.out(f"research queue vqueue: tick {t.n}: {sha[:12]} {j.verdict.state}: {j.note}")
        if past.bad:
            self.out(f"research queue vqueue: tick {t.n}: {len(past.bad)} line(s) of {self.log_path} read as none: {past.bad[0]}")
        self.keep(t)
        return logged

    def keep(self, t: Tick) -> None:
        """The state file (``state``): this tick, and each candidate's verdict beside graph2's and the jobs its graph lacks."""
        if self.state is None:
            return
        doc = {"at": stamp(self.now()), "tick": t.n, "main": t.main, "origin_main": t.origin,
               "candidates": {sha: {"mine": j.verdict.state, "graph2": j.graph2, "lacks": list(j.lacks), "note": j.note,
                                    "beside": j.slim.state if j.slim is not None else None}
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
                 f"ready {len(s.ready)} (rank order; ! doesn't merge onto main, L a Lean change): "
                 + " ".join(f"#{x.number}" + ("!" if x.conflict else "") + ("L" if x.lean else "") for x in s.ready)]
        for c in s.candidates:
            j = t.judged[c.sha]
            lines.append(f"candidate {c.sha[:12]} on {c.base[:12]}: {' '.join(f'#{n}' for n in c.members)} {c.verdict.state}"
                         f" tries {c.tries} reruns {c.reruns}" + (f" (beside: {j.slim.state})" if j.slim else "")
                         + (f" ({j.note})" if j.note else "")
                         + "".join(f"\n  failed {f.task} {f.key[:19]}" for f in c.verdict.failed)
                         + "".join(f"\n  infra {f.task} {f.key[:19]}" for f in c.verdict.infra))
        lines.append(f"history {len(s.history)}: " + " ".join(o.state for o in s.history[-10:]))
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
        raise QueueError(f"{a}: no such action")

    def _build(self, a: Build, t: Tick) -> list[dict]:
        w = self.world
        if unknown := [n for n in a.members if n not in w.prs]:
            raise QueueError(f"build: {', '.join(f'#{n}' for n in unknown)} not ready")
        if self.slim is not None:
            w.slim_driver()                                 # before anything is pushed
        tip, merged, left = w.build([w.prs[n] for n in a.members], a.base)
        recs = []
        for p, why in left:
            text = None if (key := f"{p.head}@{a.base}") in t.told else (
                f"#{p.number} at {p.head[:12]} doesn't merge onto the verified queue's candidate on {a.base[:12]}: {why}. "
                f"It is left out until it merges: merge `main` into it")
            posts = text is not None and not self.shadow
            done = posts and self._say(t, "skip", text, self.owner(p.number), base=a.base, heads=[p.head], number=p.number)
            if done or (text and self.shadow):
                t.told.add(key)
            recs.append(self.record(t, "skip", base=a.base, members=[p.number], heads=[p.head], number=p.number, why=why,
                                    text=text, done=done, **({"posted": done} if posts else {})))
        if not merged:
            recs.append(self.record(t, "build", base=a.base, priority=a.priority, why="no member merges onto its base"))
            return recs
        if adopted := next((sha for name, sha in sorted(t.foreign.items())
                            if name == sha[:12] and w.rebuilt(sha, tip, a.base, merged)), None):
            tip = adopted                                   # pushed before by a build whose line the log never got
        else:
            w.push(tip, branch(tip))
        line = {"sha": tip, "base": a.base, "members": [m.pr for m in merged], "heads": [m.head for m in merged],
                "priority": a.priority, **({"adopted": True} if adopted else {})}
        try:
            w.enqueue(tip, a.priority)
        except (QueueError, OSError, subprocess.SubprocessError) as e:  # pushed: the next tick reads it as not enqueued
            return [*recs, self.record(t, "build", **line, enqueued=False, done=True, error=str(e)[-600:])]
        return [*recs, self.record(t, "build", **line, enqueued=True, done=True)]

    def _start(self, t: Tick, sha: str, base: str, line: dict, why: str = "started") -> list[dict]:
        """The run beside candidate ``sha``'s check started (`vqueue_slim.start`), its record logged; a start that fails is
        logged with no record, which `vqueue_slim.judge` reads as infra."""
        if self.slim is None:
            return []
        return [self._started(t.n, t.main, t.origin, {**line, "sha": sha, "base": base}, why)]

    def _started(self, n: int, main: str, origin: str, line: dict, why: str = "started") -> dict:
        d = self.world.slim_driver()                       # drive.toml naming no machines stops the tick: no run is infra
        try:
            rec, error = self.slim.start(d, line["sha"], line["base"]), None
        except (QueueError, ValueError, OSError, subprocess.SubprocessError) as e:
            rec, error = None, f"{type(e).__name__}: {e}"[-600:]
        return self._line(n, main, origin, "slim", **line, why=why, record=rec, done=True,
                          **({"start_error": error} if error else {}))

    def _advance(self, c: Candidate, t: Tick) -> dict:
        """Push `main` to ``c`` (with --shadow, the shadow's own `main`).  Logged as soon as it is pushed, before what drive does
        once its members are on `main` (`_on_main`), a step of its own the tick runs next."""
        w = self.world
        if self.shadow:
            w.set_shadow_main(c.sha)
        else:
            w.advance(c.sha, c.base)
        return self.record(t, "advance", sha=c.sha, base=c.base, members=c.members, heads=c.heads,
                           why=f"`main` {c.base[:12]} fast-forwarded to it", done=not self.shadow,
                           **({} if self.shadow else {"on_main": True}))

    def _on_main(self, t: Tick, adv: dict, told: list[str]) -> dict:
        """What drive does once advance ``adv``'s members are on `main` (`World.on_main`), logged as its own step: one that fails
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

    def _drop(self, a: Drop, c: Candidate, t: Tick) -> dict:
        """Cancel ``c``'s check and the run beside it while that runs (it would hold drive's machines for nothing), delete its
        branch and log its outcome."""
        w, j = self.world, t.judged.get(c.sha)
        cancel = w.cancel(c.sha)
        ids = list((j.record or {}).get("ids") or ()) if j is not None and j.slim is not None and j.slim.state == PENDING else []
        failed = w.cancel_runs(ids, f"vqueue: candidate {c.sha[:12]} was dropped: {a.why}") if ids else {}
        w.push("", branch(c.sha), c.sha)
        return self.record(t, "drop", sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why, outcome=a.outcome,
                           failed=c.verdict.failed, cancel=cancel, done=True,
                           **({"beside_cancelled": ids} if ids else {}), **({"beside_cancel_failed": failed} if failed else {}))

    def _eject(self, a: Eject, t: Tick) -> dict:
        w = self.world
        tasks = "; ".join(f"{f.task} (key {f.key})" for f in a.failed) or "none named"
        text = (f"#{a.number} at {a.head[:12]} is ejected from the verified queue: {a.why}. Failed: {tasks}. Each result, its "
                f"log's digest with it: `research store labels task:<key> --key result --history`. Fix it and mark it ready "
                f"again")
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
        said = w.rerun(c.sha, keys) if keys else ""
        if said is None:
            return [self.record(t, "rerun", **line, why="graph2 has no `rerun`: the failure stands", failed=failed,
                                keys=list(a.keys), absent=True, done=False)]
        again = self._start(t, c.sha, c.base, line, "started again: its failure is rerun") if set(a.keys) & beside else []
        return [self.record(t, "rerun", **line, why=f"the lone member's own failures, afresh: {said or 'the run beside'}",
                            failed=failed, keys=list(a.keys), counts=counts, done=True), *again]

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
        """Enqueue `ready:<base>` of each base whose unlisted check a candidate's failures wait on (`Judged.unlisted`), once a
        base, at the candidate's priority: its keys tell which failures the candidate inherits.  A candidate's own check was
        enqueued by its build; one that fails to enqueue is logged and enqueued by the next tick."""
        w, recs, done = self.world, [], {r.get("sha") for r in t.past.of("base") if r.get("enqueued")}
        for c in t.snap.candidates:
            if (b := t.judged[c.sha].unlisted) is None or b in done or t.past.tries(b):
                continue
            done.add(b)
            built = t.past.built(c.sha) or {}
            line = dict(sha=b, candidate=c.sha, priority=built.get("priority") if isinstance(built.get("priority"), int) else 0,
                        why=f"{c.sha[:12]}'s failures wait on its base's listing")
            try:
                w.enqueue(b, line["priority"])
            except (QueueError, OSError, subprocess.SubprocessError) as e:
                recs.append(self.record(t, "base", **line, enqueued=False, done=False, error=str(e)[-600:]))
                continue
            recs.append(self.record(t, "base", **line, enqueued=True, done=True))
        return recs

    def _ring(self, a: Ring, c: Candidate, t: Tick) -> list[dict]:
        if (c.sha, a.why) in t.past.rung():
            return []
        infra = "; ".join(f"{f.task} (key {f.key})" for f in c.verdict.infra)
        text = (f"the verified queue's candidate ready:{c.sha} ({' '.join(f'#{n}' for n in c.members)}) is {c.verdict.state} "
                f"after {c.tries} enqueues: {a.why}" + (f". Infra: {infra}" if infra else ""))
        line = dict(sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why, failed=c.verdict.infra, text=text)
        if self.shadow:
            return [self.record(t, "ring", **line, done=False)]
        if not self._say(t, "ring", text, (INFRA_HANDLE,), sha=c.sha, why=a.why):
            return [self.record(t, "ring", **line, done=False, posted=False,
                                error=f"ringing {INFRA_HANDLE} about {c.sha[:12]} failed: the next tick rings again")]
        return [self.record(t, "ring", **line, done=True, posted=True)]

    def _red(self, a: Red, c: Candidate, t: Tick) -> dict:
        """Tell `CI` and the owners of what failed that ``a.base`` is red on it, once a base and task: a post that fails is told
        again by the next red on that base, and the drop after it goes ahead."""
        w, tasks = self.world, "; ".join(f"{f.task} (key {f.key})" for f in a.failed)
        to = tuple(dict.fromkeys((CI, *w.task_owners(c.sha, (f.task for f in a.failed)))))
        new = {f.task for f in a.failed} - t.past.reds(a.base)
        text = None if not new else (
            f"the verified queue's base {a.base[:12]} is red: {a.why}. Failed there as on {c.sha[:12]}: {tasks}. Each result, its "
            f"log's digest with it: `research store labels task:<key> --key result --history`")
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


def _locked(path: Path):
    """Hold ``path``'s lock, or None when another tick holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def main(argv: list[str]) -> int:
    from .queue import GitHub, credential_args, remote_and_repo
    from .store.local import LocalStore

    ap = argparse.ArgumentParser(prog="research queue vqueue", description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=WORKDIR, help=f"the clone to work in (default drive's, {WORKDIR}); the loop "
                                                              f"fetches into its own refs ({NS}/) and touches none of drive's")
    ap.add_argument("--remote", default=None, help="the repository (default the clone's origin)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--lock", type=Path, default=LOCK, help=f"the loop's own lock (default {LOCK})")
    ap.add_argument("--log", type=Path, default=LOG, help=f"the decision log, one JSON line per action (default {LOG}); its "
                                                          f"directory holds the loop's caches")
    ap.add_argument("--state", type=Path, default=STATE, help=f"each tick's candidates, their verdicts beside graph2's and what "
                                                              f"their graphs lack, rewritten each tick (default {STATE})")
    ap.add_argument("--priority-file", type=Path, default=PRIORITY, help=f"drive's priority list (default {PRIORITY})")
    ap.add_argument("--graph2", default=shlex.join(GRAPH2), help="graph2's argv before its subcommand (default the control pod's)")
    ap.add_argument("--batch", type=int, default=6, help="the most members a candidate holds (default 6)")
    ap.add_argument("--depth", type=int, default=2, help="the most candidates stacked at once (default 2)")
    ap.add_argument("--lean", type=int, default=1, help="the most members of a candidate that change Lean, a lock or a spec "
                                                        "(default 1)")
    ap.add_argument("--pending-hours", type=float, default=4.0,
                    help="hours the lowest candidate's check may stay pending after its first enqueue before it is enqueued "
                         "again; rung at twice that (default 4)")
    ap.add_argument("--no-slim", action="store_true",
                    help="judge each candidate on its graph2 check alone, with no run beside it of lean-agreement and the "
                         "preflights (research.vqueue_slim), which it otherwise launches on drive's machines for the lowest "
                         "candidate once its check passed; only with --shadow or --dry-run, since `main` would move past what "
                         "that run checks")
    ap.add_argument("--shadow", action="store_true", help="log advance, eject, ring, red, flakes and messages, and do none of "
                                                          "them; its own main is refs/vqueue/shadow in the clone")
    ap.add_argument("--dry-run", action="store_true", help="print the snapshot and the actions; write nothing")
    ap.add_argument("--once", action="store_true", help="one tick")
    ap.add_argument("--interval", type=float, default=INTERVAL_S, help=f"seconds between ticks (default {INTERVAL_S})")
    a = ap.parse_args(argv)
    if a.no_slim and not (a.shadow or a.dry_run):
        print("research queue vqueue: --no-slim moves `main` past lean-agreement and the preflights unchecked: it runs only "
              "with --shadow or --dry-run", file=sys.stderr)
        return 1
    if (slim := None if a.no_slim else _slim()) is None and not a.no_slim:
        print("research queue vqueue: research.vqueue_slim isn't in this tree, so no run beside a candidate's check can start: "
              "--no-slim judges each candidate on its graph2 check alone", file=sys.stderr)
        return 1
    lock = None if a.dry_run else _locked(a.lock)
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
        q.branch = "main"
        graph2 = tuple(shlex.split(a.graph2))
        world = World(q, labels, Executor(Transport(store, (*graph2, "enqueue"), a.repo)), graph2=graph2,
                      priority_file=a.priority_file, labels_at=labels_at, cache=a.log.parent)
        loop = Loop(world, a.log, shadow=a.shadow, dry_run=a.dry_run, batch=a.batch, depth=a.depth, lean=a.lean,
                    pending_s=a.pending_hours * 3600, slim=slim, state=a.state)
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
