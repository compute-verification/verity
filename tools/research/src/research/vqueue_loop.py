"""`research queue vqueue`: the verified queue's loop (`research.vqueue`).  Each tick, under its own lock (`--lock`, never
drive's), derives a `vqueue.Snapshot` from what is outside it, asks the policy (``decide``, `vqueue.decide`) what to do and
applies its actions in order, stopping the tick at the first that fails: the next tick derives everything again.  Between ticks
it keeps only what a snapshot reads: the refs, the labels, graph2's results and the decision log (`--log`), from which come
`history`, each candidate's `tries`, a retry's and a ring's once-only and the owners already told.

The snapshot.  `main` is origin's `main`, fetched into the clone's own refs (`refs/vqueue/`), so drive's clone, its worktree and
its refs are left as drive keeps them.  `ready` is the open PRs drive would land from `next` now: admitted as drive admits them
(`Queue.entries`, reading drive's state), with a land verdict by a non-owner or under the low-risk rule (drive.toml's
`lowrisk_lane` and `lowrisk_excludes` at `main`), not on `main` yet, ranked by drive's priority file then mark time, each after
the ready PRs it contains; `files` is its change against its merge base with `main`, `lean` drive's Lean-change test
(`queue.lean_change`), and `conflict` whether `Queue._merge` merges it onto `main`, kept by (head, `main`).  `candidates` are
origin's `queue/<sha12>` branches, each the `Queue: #N` merges on its first-parent line down to `main` or another candidate.
A candidate's verdict is graph2's results for `ready:<sha>` (`Executor.settle`) on the merge gate's tasks, those drive's
`executor = "on"` path gates on (none of `drive.SLIM_STEPS`, none off the gate in policy.toml at the candidate): a failed or
blocked task is fail, else a task whose newest result is graph2's infra error (`detail.infra`) is infra, else every gating task
passed is pass, else pending.  No graph while graph2 says its listing failed (`verdict`, else its `ready_verdict` label), or a
graph that can't be read, is list-failed.  graph2's own verdict, where it has one, is compared and a disagreement logged.

The actions.  Build merges its members in order onto its base (`Queue._merge`), leaving out one that doesn't merge (`skip`, its
owner told once per head and base), pushes `queue/<sha12>` and enqueues `ready:<sha>` as graph/v2 at its priority.  Advance
pushes `main` to the candidate with a lease on its base and does what drive does when `main` moves (`Driver._on_main`).  Drop
cancels the check (graph2's `cancel`, where it has one), deletes the branch and logs its outcome.  Eject withdraws the PR's
ready mark at its head (`Queue.eject`) and tells its owner the failed tasks, their keys and where their logs are.  Retry
enqueues again, once for each terminal infra or list-failed verdict (by its evidence: the infra results there are, graph2's
listing verdict).  Ring tells @infra, once a candidate.  Each action and its result is one JSON line in the log.

`--shadow` logs advance, eject, ring and every message with `shadow: true` and does none of them, and builds, drops and
retries: its `main` is the newest candidate it advanced (logged) while origin's `main` is below it, and a PR it ejected stays
out until marked again.  `--dry-run` prints the snapshot and the actions and writes nothing: no ref on origin, no log, no
enqueue, no label, no message.  The loop never writes drive's state, its priority file, `next`, `checked` or a label, but an
eject's without `--shadow`.

On the control pod it runs beside drive, from a pinned checkout of this tree, in drive's clone (or its own clone of origin):
`research queue vqueue --shadow --log /data/vqueue/decisions.jsonl`, with `--once` for one tick and `--dry-run` to see one.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

from . import policy, vqueue
from .drive import JOB_OF_KIND, SLIM_STEPS, Driver, merge_thread_post
from .executor import BLOCKED, Executor, QueueTransport, fold_results, task_key, task_target
from .queue import (PR, WORKDIR, Member, Queue, QueueError, StoreLabels, _git, _out, flags_at, lean_change, lowrisk,
                    priority_list, queue_commit, review_owners, rules_at, unreviewed)
from .vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Action, Advance, Build, Candidate, Drop, Eject, Failed, Outcome,
                     Ready, Retry, Ring, Snapshot, Verdict)

#: graph2, infra's executor on the control pod: its argv before the subcommand
GRAPH2 = ("python3.12", "/root/executor/infra/graph2.py", "--db", "/root/executor/graph2.db", "--events", "/root/executor/events")
LOCK, LOG = Path("/data/vqueue/vqueue.lock"), Path("/data/vqueue/decisions.jsonl")
#: drive's priority list (its `--priority-file`)
PRIORITY = Path("/data/drive/priority")
#: how far back the decision log is read
WINDOW_S = 48 * 3600
INTERVAL_S = 60
#: the clone's refs the loop fetches into, none of drive's
NS = "refs/vqueue"
MAIN = f"{NS}/main"
#: no automatic gc or maintenance under drive's own commands in its clone
QUIET = ("-c", "gc.auto=0", "-c", "maintenance.auto=false")
TIME = "%Y-%m-%dT%H:%M:%SZ"
#: the store label graph2 puts on `commit:<sha>` with its terminal verdict on `ready:<sha>`
READY_VERDICT = "ready_verdict"
#: who ejects, as drive's ejections are by
EJECTOR = "ci"
INFRA_HANDLE = "@infra"
#: the newest first-parent commits a candidate's walk reads before it gives up on finding its base
WALK = 64
#: the PRs judged at once (`World._judge`): a merge onto `main` takes seconds in drive's clone, and a moved `main` judges every
#: ready PR again
JOBS = 8


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime(TIME)


def unstamp(s: object) -> float | None:
    try:
        return datetime.strptime(str(s), TIME).replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def branch(sha: str) -> str:
    return f"refs/heads/queue/{sha[:12]}"


def _digest(doc: object) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Past:
    """The decision log's window (`WINDOW_S`), oldest first, and the last tick it holds."""
    tick: int = 0
    lines: list[dict] = field(default_factory=list)
    bad: int = 0

    @classmethod
    def read(cls, path: Path, since: float) -> Past:
        if not path.is_file():
            return cls()
        out, tick, bad = [], 0, 0
        for line in reversed(path.read_text().splitlines()):
            try:
                rec = json.loads(line)
            except ValueError:
                rec = None
            if not isinstance(rec, dict):
                bad += 1                                    # a torn last line, as a crash mid-write leaves one
                continue
            tick = tick or int(rec.get("tick") or 0)
            if (at := unstamp(rec.get("at"))) is None or at < since:
                break
            out.append(rec)
        return cls(tick, out[::-1], bad)

    def of(self, action: str, sha: str | None = None) -> list[dict]:
        return [r for r in self.lines if r.get("action") == action and (sha is None or r.get("sha") == sha)]

    def history(self) -> tuple[Outcome, ...]:
        return tuple(Outcome(tuple(o.get("members") or ()), tuple(o.get("heads") or ()), o["state"], tuple(o.get("suspects") or ()))
                     for r in self.of("drop") if isinstance(o := r.get("outcome"), dict) and o.get("state"))

    def tries(self, sha: str) -> int:
        return sum(1 for r in self.lines if r.get("sha") == sha and r.get("action") in ("build", "retry") and r.get("enqueued"))

    def retried(self, sha: str) -> set[str]:
        return {r.get("mark") for r in self.of("retry", sha) if r.get("mark")}

    def rung(self) -> set[str]:
        return {r["sha"] for r in self.of("ring") if r.get("done") or r.get("shadow")}

    def told(self) -> set[str]:
        return {f"{h}@{r.get('base')}" for r in self.of("skip") if r.get("done") or r.get("shadow") for h in r.get("heads") or ()}

    def built(self, sha: str) -> dict | None:
        return next((r for r in reversed(self.of("build", sha)) if r.get("enqueued")), None)


@dataclass
class Judged:
    """A candidate's verdict with what it rests on: ``mark`` names the terminal infra or list-failed evidence a retry answers."""
    verdict: Verdict
    mark: str | None = None
    note: str = ""
    graph2: str | None = None


class _Seen:
    """A transport that keeps each key's results as `Executor.settle` reads them: an infra error folds to pending there."""

    def __init__(self, transport):
        self.transport, self.got = transport, {}

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        got = self.transport.results(keys)
        self.got.update(got)
        return got


class Transport(QueueTransport):
    """`QueueTransport`, pulling a key's results from the store's remote again while none of them passed or failed: graph2 labels
    an infra error as a result, and a retry's result comes after it."""

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        from .queue import PREFETCH_JOBS
        from .store import labels_sync, vocab

        have = {k: [lab.value for lab in self.store.labels(task_target(k), key=vocab.TASK_RESULT)] for k in keys}
        if (todo := [k for k, v in have.items() if fold_results(v, k)[0] == PENDING]) \
                and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [task_target(k) for k in todo], jobs=PREFETCH_JOBS)
            have.update({k: [lab.value for lab in self.store.labels(task_target(k), key=vocab.TASK_RESULT)] for k in todo})
        return have


class _Clone(Queue):
    """The queue in a clone it doesn't bring to the remote's state (drive's, or the loop's own): the loop fetches into `NS`."""

    def _clone(self) -> None:
        return None


class World:
    """What the loop reads and changes: git in the clone and on origin, graph2, the labels, the PRs and the merge thread.  Each
    effect is one method, so a test fakes what it doesn't run for real."""

    def __init__(self, q: Queue, labels, tasks: Executor, *, graph2: tuple[str, ...] = (), priority_file: Path | None = None,
                 post: Callable[[str, tuple[str, ...]], bool] | None = None, now: Callable[[], float] = time.time):
        self.q, self.labels, self.tasks, self.graph2 = q, labels, tasks, tuple(graph2)
        self.priority_file, self.now, self.repo = priority_file, now, q.workdir
        self.post: Callable[[str, tuple[str, ...]], bool] = post or merge_thread_post
        #: graph2's subcommands it was found to lack (`_graph2`)
        self.lacks: set[str] = set()
        #: by (head, `main`): its files (`_changed`), and (lean, conflict) (`_judge`)
        self._files: dict[tuple[str, str], frozenset[str]] = {}
        self._judged: dict[tuple[str, str], tuple[bool, bool]] = {}
        self._prs: dict[int, PR] | None = None
        self._driver: Driver | None = None
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
        return main

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
            chosen = {n: (files[n], *j) for n, j in zip(took, pool.map(lambda n: self._judge(main, entries[n].pr.head), took))}
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

    def _judge(self, main: str, head: str) -> tuple[bool, bool]:
        """(whether ``head`` is a Lean change, whether it doesn't merge onto ``main``), kept by (head, main)."""
        if (key := (head, main)) not in self._judged:
            rec = self.q._recorded(main, head)
            self._judged[key] = (lean_change(self.repo, head, _out(self.repo, "merge-base", main, head)),
                                 not ("tree" in rec or rec.get("merged")))
        return self._judged[key]

    def candidates(self) -> list[str]:
        """The candidates on origin, as fetched: their shas."""
        return list(dict.fromkeys(_out(self.repo, "for-each-ref", "--format=%(objectname)", f"{NS}/queue/").split()))

    def walk(self, sha: str, stops: set[str], main: str) -> tuple[str, list[Member]] | None:
        """(its base, its members oldest first) of candidate ``sha``: the `Queue: #N` merges on its first-parent line down to the
        first commit that is one of ``stops`` or on ``main``; None when that line holds anything else first."""
        recs = [r.strip() for r in _out(self.repo, "log", "--first-parent", f"-n{WALK}", "--format=%H%x00%P%x00%B%x1e",
                                         sha).split("\x1e") if r.strip()]
        members: list[Member] = []
        for i, rec in enumerate(recs):
            c, parents, body = rec.split("\x00", 2)
            if i and (c in stops or self._ancestor(c, main)):
                return c, members[::-1]
            if (m := queue_commit(c, parents, body)) is None:
                return None
            members.append(m)
        return None

    def verdict(self, sha: str) -> Judged:
        """Candidate ``sha``'s verdict from graph2's results (see the module's docstring), beside graph2's own."""
        check = f"ready:{sha}"
        theirs = self.graph2_verdict(sha)
        said = theirs.get("state") if isinstance(theirs, dict) else None
        try:
            got = self.tasks.graph_v2(check)
        except ValueError as e:
            return Judged(Verdict(LIST_FAILED), mark=f"unreadable {_digest(str(e))}", note=str(e), graph2=said)
        if got is None:
            if said == LIST_FAILED:
                return Judged(Verdict(LIST_FAILED), mark=f"list-failed {_digest(theirs)}", note=str(theirs.get("why") or ""),
                              graph2=said)
            return Judged(Verdict(PENDING), note="not listed yet", graph2=said)
        _, graph = got
        tasks = {t["id"]: t for t in graph["tasks"]}
        if isinstance(pol := policy.at(self.repo, sha), str):
            self.log(f"research queue vqueue: {pol}: every job gates at {sha[:12]}")
            pol = policy.Policy()
        gating = [i for i, t in tasks.items() if JOB_OF_KIND.get(t["kind"], t["kind"]) not in (*SLIM_STEPS, *pol.off_gate)]
        if not gating:
            return Judged(Verdict(LIST_FAILED), mark=f"empty {_digest(sorted(tasks))}", note="no task on the merge gate",
                          graph2=said)
        seen = _Seen(self.tasks.transport)
        states = Executor(seen).settle(graph)
        files, envs = graph.get("files") or {}, graph.get("envs") or {}
        keys: dict[str, str] = {}
        for i, t in tasks.items():                          # each key settle read: the graph's, or one made from its reads
            reads = [states[r["task"]] for r in t["reads"]]
            keys[i] = t.get("key") or (task_key(t, files, envs[t["env"]]["digest"],
                                                {r["task"]: s[1]["outputs"] for r, s in zip(t["reads"], reads)})
                                       if states[i][0] != BLOCKED and all(s[0] == PASS for s in reads) else "")
        inputs: dict[str, frozenset[str]] = {}

        def reach(i: str) -> frozenset[str]:
            if i not in inputs:
                inputs[i] = frozenset()
                inputs[i] = frozenset(tasks[i]["inputs"]).union(*(reach(r["task"]) for r in tasks[i]["reads"]))
            return inputs[i]

        def under(i: str, seen_: set[str]) -> list[str]:
            """The tasks ``i`` reads, transitively, ``i`` first."""
            if i in seen_:
                return []
            seen_.add(i)
            return [i, *(j for r in tasks[i]["reads"] for j in under(r["task"], seen_))]

        needed = list(dict.fromkeys(j for i in gating for j in under(i, set())))
        bad: list[str] = []
        for i in gating:
            if states[i][0] == FAIL:
                bad.append(i)
            elif states[i][0] == BLOCKED:
                bad += [j for j in under(i, set()) if states[j][0] == FAIL] or [i]
        infra = [i for i in needed if states[i][0] == PENDING and _infra(seen.got.get(keys[i]) or (), keys[i])]
        fail = tuple(Failed(i, keys[i], reach(i)) for i in dict.fromkeys(bad))
        down = tuple(Failed(i, keys[i], reach(i)) for i in infra)
        if fail:
            mine = Verdict(FAIL, fail, down)
        elif down:
            mine = Verdict(INFRA, (), down)
        elif all(states[i][0] == PASS for i in gating):
            mine = Verdict(PASS)
        else:
            mine = Verdict(PENDING)
        mark = f"infra {_digest(sorted((keys[i], len(seen.got.get(keys[i]) or ())) for i in infra))}" if infra else None
        note = f"graph2 says {said}" if said and said != mine.state and said in vqueue.STATES else ""
        return Judged(mine, mark=mark if mine.state == INFRA else None, note=note, graph2=said)

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
        return v if isinstance(v, dict) else None

    def _graph2(self, *args: str) -> subprocess.CompletedProcess | None:
        """graph2 run with ``args``; None when it has no such subcommand (argparse's `invalid choice`), or there is no graph2."""
        if not self.graph2 or args[0] in self.lacks:
            return None
        r = subprocess.run([*self.graph2, *args], cwd=self.repo, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode == 2 and "invalid choice" in r.stderr:
            self.lacks.add(args[0])
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

    def advance(self, sha: str, base: str, members: list[Member]) -> None:
        """origin's `main` fast-forwarded from ``base`` to ``sha``, and what drive does once ``members`` are on it."""
        self.q._drive_only("main", "advance", False)
        self.push(sha, "refs/heads/main", base)
        self.driver()._on_main({"told": [], "stale_closed": self.q.load_state().get("drive", {}).get("stale_closed", {})},
                               members, base, sha, "")

    def withdraw(self, number: int, head: str, why: str) -> PR:
        return self.q.eject(number, why, EJECTOR, sha=head)

    def handle(self, number: int) -> str | None:
        p = self.prs.get(number) or self.open().get(number)
        return self.q.handle(p) if p is not None else None

    def say(self, text: str, to: tuple[str, ...]) -> bool:
        """``text`` posted to ``to`` in the merge thread, as drive posts."""
        return bool(self.post(f"{' '.join(to)} vqueue: {text}", to))

    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = Driver(self.q, on="", post=self.post, now=self.now)
        return self._driver


def _infra(docs: Iterable[object], key: str) -> bool:
    """Whether ``key``'s newest result is graph2's infra error, its attempts run out."""
    mine = [d for d in docs if isinstance(d, dict) and d.get("key") == key]
    return bool(mine) and mine[-1].get("status") == "error" and (mine[-1].get("detail") or {}).get("infra") is True


@dataclass
class Tick:
    """What one tick saw: its snapshot and what each action needs that the snapshot doesn't hold."""
    n: int
    main: str
    origin: str
    snap: Snapshot
    judged: dict[str, Judged]
    members: dict[str, list[Member]]
    past: Past


class Loop:
    """The verified queue's loop on ``world``: `tick` once, or `run` every ``interval`` seconds."""

    def __init__(self, world: World, log: Path, *, decide: Callable[[Snapshot], list[Action]] = vqueue.decide,
                 shadow: bool = False, dry_run: bool = False, batch: int = 6, depth: int = 2,
                 now: Callable[[], float] = time.time, out: Callable[[str], None] = print):
        self.world, self.log_path, self.decide, self.shadow, self.dry_run = world, Path(log), decide, shadow, dry_run
        self.batch, self.depth, self.now, self.out = batch, depth, now, out
        #: the last tick this loop ran: a tick that logs nothing still counts
        self.last = 0

    # -- the snapshot

    def snapshot(self, past: Past, n: int) -> Tick:
        w = self.world
        origin = w.fetch()
        main = origin
        if self.shadow:                                     # the newest candidate this shadow advanced, while above origin's
            for r in reversed(past.of("advance")):
                if r.get("shadow") and r.get("sha") and not w._missing([r["sha"]]) and w._ancestor(origin, r["sha"]):
                    main = r["sha"]
                    break
        ejected = {(r.get("number"), (r.get("heads") or [None])[0]): r.get("at") for r in past.of("eject") if r.get("shadow")}
        ready = []
        for x in w.ready(main):
            at = ejected.get((x.number, x.head))
            if at is not None and not (w.labels.marked_at(x.number, x.head) or "") > at:
                continue
            ready.append(x)
        shas = w.candidates()
        cands, judged, members = [], {}, {}
        for sha in shas:
            built = past.built(sha)
            walked = None if w._ancestor(sha, main) and built else w.walk(sha, set(shas) - {sha}, main)
            if walked is not None:
                base, ms = walked
            elif built:
                base, ms = built["base"], [Member(p, h, "", "") for p, h in zip(built["members"], built["heads"])]
            else:
                base, ms = "", []
                w.log(f"research queue vqueue: candidate {sha[:12]} holds a commit that is no `Queue: #N` merge above its base")
            judged[sha] = j = w.verdict(sha)
            members[sha] = ms
            cands.append(Candidate(sha, base, tuple(m.pr for m in ms), tuple(m.head for m in ms), j.verdict, past.tries(sha)))
        snap = Snapshot(main, tuple(sorted(ready, key=lambda x: x.rank)), tuple(cands), past.history(), self.batch, self.depth)
        return Tick(n, main, origin, snap, judged, members, past)

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
        logged: list[dict] = []
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
        for sha, j in t.judged.items():
            if j.note:
                self.out(f"research queue vqueue: tick {t.n}: {sha[:12]} {j.verdict.state}: {j.note}")
        return logged

    def run(self, interval: float, sleep: Callable[[float], None] = time.sleep) -> int:
        while True:
            try:
                self.tick()
            except Exception as e:                          # the loop goes on: the next tick derives everything again
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
                         f" tries {c.tries}" + (f" ({j.note})" if j.note else "")
                         + "".join(f"\n  failed {f.task} {f.key[:19]}" for f in c.verdict.failed)
                         + "".join(f"\n  infra {f.task} {f.key[:19]}" for f in c.verdict.infra))
        lines.append(f"history {len(s.history)}: " + " ".join(o.state for o in s.history[-10:]))
        return "\n".join(lines)

    # -- the actions

    def record(self, t: Tick, action: str, *, sha: str | None = None, base: str | None = None, members: Iterable[int] = (),
               heads: Iterable[str] = (), priority: int | None = None, number: int | None = None, why: str = "",
               outcome: Outcome | None = None, failed: Iterable[Failed] = (), **extra) -> dict:
        return {"at": stamp(self.now()), "tick": t.n, "shadow": self.shadow, "main": t.main, "action": action, "sha": sha,
                "base": base, "members": list(members), "heads": list(heads), "priority": priority, "number": number,
                "why": why, "outcome": outcome and {"members": list(outcome.members), "heads": list(outcome.heads),
                                                    "state": outcome.state, "suspects": list(outcome.suspects)},
                "failed": [{"task": f.task, "key": f.key} for f in failed],
                **({"origin_main": t.origin} if t.origin != t.main else {}), **extra}

    def write(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

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
        if isinstance(a, Ring):
            return self._ring(a, c, t)
        raise QueueError(f"{a}: no such action")

    def _build(self, a: Build, t: Tick) -> list[dict]:
        w = self.world
        if unknown := [n for n in a.members if n not in w.prs]:
            raise QueueError(f"build: {', '.join(f'#{n}' for n in unknown)} not ready")
        tip, merged, left = w.build([w.prs[n] for n in a.members], a.base)
        told, recs = t.past.told(), []
        for p, why in left:
            done = False
            if f"{p.head}@{a.base}" not in told and not self.shadow:
                done = w.say(f"#{p.number} at {p.head[:12]} doesn't merge onto the verified queue's candidate on "
                             f"{a.base[:12]}: {why}. It is left out until it merges; merge `main` into it",
                             (f"@{w.handle(p.number)}" if w.handle(p.number) else "@ci",))
            recs.append(self.record(t, "skip", base=a.base, members=[p.number], heads=[p.head], number=p.number, why=why,
                                    done=done))
        if not merged:
            recs.append(self.record(t, "build", base=a.base, priority=a.priority, why="no member merges onto its base"))
            return recs
        w.push(tip, branch(tip))
        w.enqueue(tip, a.priority)
        recs.append(self.record(t, "build", sha=tip, base=a.base, members=[m.pr for m in merged],
                                heads=[m.head for m in merged], priority=a.priority, enqueued=True, done=True))
        return recs

    def _advance(self, c: Candidate, t: Tick) -> dict:
        if not self.shadow:
            self.world.advance(c.sha, c.base, t.members[c.sha])
        return self.record(t, "advance", sha=c.sha, base=c.base, members=c.members, heads=c.heads,
                           why=f"`main` {c.base[:12]} fast-forwarded to it", done=not self.shadow)

    def _drop(self, a: Drop, c: Candidate, t: Tick) -> dict:
        w = self.world
        cancel = w.cancel(c.sha)
        w.push("", branch(c.sha), c.sha)
        return self.record(t, "drop", sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why, outcome=a.outcome,
                           failed=c.verdict.failed, cancel=cancel, done=True)

    def _eject(self, a: Eject, t: Tick) -> dict:
        w = self.world
        tasks = "; ".join(f"{f.task} (key {f.key})" for f in a.failed) or "none named"
        text = (f"#{a.number} at {a.head[:12]} is ejected from the verified queue: {a.why}. Failed: {tasks}. Each result, its "
                f"log's digest with it: `research store labels task:<key> --key result --history`. Fix it and mark it ready "
                f"again")
        if not self.shadow:
            w.withdraw(a.number, a.head, a.why)
            handle = w.handle(a.number)
            w.say(text, (f"@{handle}" if handle else "@ci",))
        return self.record(t, "eject", members=[a.number], heads=[a.head], number=a.number, why=a.why, failed=a.failed,
                           text=text, done=not self.shadow)

    def _retry(self, c: Candidate, t: Tick) -> list[dict]:
        j = t.judged[c.sha]
        why = (f"{c.verdict.state} verdict ({j.mark}) was retried already: its retry's results are pending" if j.mark in
               t.past.retried(c.sha) else None if j.mark else f"its verdict is {c.verdict.state}, not a terminal infra or "
               f"list-failed one")
        if why is not None:
            if any(r.get("why") == why for r in t.past.of("retry", c.sha)):
                return []
            return [self.record(t, "retry", sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=why, mark=j.mark,
                                enqueued=False, done=False)]
        built = t.past.built(c.sha) or {}
        priority = built.get("priority") if isinstance(built.get("priority"), int) else 0
        self.world.enqueue(c.sha, priority)
        return [self.record(t, "retry", sha=c.sha, base=c.base, members=c.members, heads=c.heads, priority=priority,
                            why=f"{c.verdict.state}: {j.note or 'enqueued again'}", failed=c.verdict.infra, mark=j.mark,
                            enqueued=True, done=True)]

    def _ring(self, a: Ring, c: Candidate, t: Tick) -> list[dict]:
        if c.sha in t.past.rung():
            return []
        infra = "; ".join(f"{f.task} (key {f.key})" for f in c.verdict.infra)
        text = (f"the verified queue's candidate ready:{c.sha} ({' '.join(f'#{n}' for n in c.members)}) is {c.verdict.state} "
                f"after {c.tries} enqueues: {a.why}" + (f". Infra: {infra}" if infra else ""))
        done = False
        if not self.shadow:
            if not self.world.say(text, (INFRA_HANDLE,)):
                raise QueueError(f"ringing {INFRA_HANDLE} about {c.sha[:12]} failed: the next tick rings again")
            done = True
        return [self.record(t, "ring", sha=c.sha, base=c.base, members=c.members, heads=c.heads, why=a.why,
                            failed=c.verdict.infra, text=text, done=done)]


def _name(a: Action) -> str:
    return type(a).__name__.lower()


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
    from .queue import credential_args, remote_and_repo
    from .store.local import LocalStore

    ap = argparse.ArgumentParser(prog="research queue vqueue", description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=WORKDIR, help=f"the clone to work in (default drive's, {WORKDIR}); the loop "
                                                              f"fetches into its own refs ({NS}/) and touches none of drive's")
    ap.add_argument("--remote", default=None, help="the repository (default the clone's origin)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--lock", type=Path, default=LOCK, help=f"the loop's own lock (default {LOCK})")
    ap.add_argument("--log", type=Path, default=LOG, help=f"the decision log, one JSON line per action (default {LOG})")
    ap.add_argument("--priority-file", type=Path, default=PRIORITY, help=f"drive's priority list (default {PRIORITY})")
    ap.add_argument("--graph2", default=shlex.join(GRAPH2), help="graph2's argv before its subcommand (default the control pod's)")
    ap.add_argument("--batch", type=int, default=6, help="the most members a candidate holds (default 6)")
    ap.add_argument("--depth", type=int, default=2, help="the most candidates stacked at once (default 2)")
    ap.add_argument("--shadow", action="store_true", help="log advance, eject, ring and messages, and do none of them")
    ap.add_argument("--dry-run", action="store_true", help="print the snapshot and the actions; write nothing")
    ap.add_argument("--once", action="store_true", help="one tick")
    ap.add_argument("--interval", type=float, default=INTERVAL_S, help=f"seconds between ticks (default {INTERVAL_S})")
    a = ap.parse_args(argv)
    lock = None if a.dry_run else _locked(a.lock)
    if not a.dry_run and lock is None:
        print(f"research queue vqueue: another tick holds {a.lock}", file=sys.stderr)
        return 1
    try:
        remote, slug = remote_and_repo(a.remote, None, a.repo)
        store = LocalStore(a.store)
        labels = StoreLabels(store, log=lambda s: print(s, file=sys.stderr))
        from .queue import GitHub

        q = _Clone(a.repo, remote=remote, github=GitHub(slug), ready=labels.ready, store=store,
                   auth=credential_args(a.repo, remote), before_cutoff=labels.before_cutoff, marked_by=labels.marked_by,
                   marked_at=labels.marked_at, prefetch=labels.prefetch, land_into="main")
        q.branch = "main"
        graph2 = tuple(shlex.split(a.graph2))
        world = World(q, labels, Executor(Transport(store, (*graph2, "enqueue"), a.repo)), graph2=graph2,
                      priority_file=a.priority_file)
        loop = Loop(world, a.log, shadow=a.shadow, dry_run=a.dry_run, batch=a.batch, depth=a.depth)
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
