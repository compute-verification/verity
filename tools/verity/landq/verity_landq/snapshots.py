"""Drive a snapshot queue, a pure `decide(snapshot) -> actions` with the verified queue's contract (`research.vqueue`), with
`model`'s events and decisions, so the simulator and the property tests run it.

    from research import vqueue
    from verity_landq import sim, snapshots
    q = snapshots.Driver(vqueue, lean=3)
    sim.simulate(arrivals, world, policy, step=q.step, state=q.start("main"))

The driver stands in for the queue's loop.  It builds each snapshot from what the events said: the ready and reviewed PRs,
the candidates with the verdict their results make, and the outcomes of decided candidates.  It calls `decide` and turns
each action into a `model` decision with the same effect on the world, and calls it again until it asks for nothing new,
as a loop that ticks often would.  A tick that builds waits for its trees before deciding again, as the loop's builds
finish within its tick.  A candidate's verdict is `fail` once a task failed, else `infra` once a task's outcome is an infra
error, though others still run, as the loop reads graph2's, else `pass` once every listed task passed, else `pending`.  An
infra outcome a retry asks for again is set aside.  The driver fills each field the contract has and leaves the rest to
their defaults: a failure's key, its key on the candidate's base, its failing tests and the base's (as the world's results
give them), whether the verdict has ended (every listed task has an outcome) with its infra errors beside its failures, a
candidate's reruns and when its check was enqueued, and the snapshot's time (the simulator's minutes, as seconds).  The
policy gives the snapshot's batch and depth; keyword arguments set its other fields, such as `lean` or `rings`, which
otherwise keep the queue's own defaults.  A `Ring` pages once a candidate and why, and a `Red` once a base and task, as
the loop rings them; a `Rerun` runs the failed keys' tasks afresh.  A task named in `deferred` (`sim.Task.deferred`)
starts once, as the loop starts the run beside a check: on the lowest candidate, while it is on `main`, once the queue's
reading of its other tasks' verdict (the module's `_reading` where it has one) is pass.  The module is a parameter, so
nothing here imports it.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from types import ModuleType

from verity_landq import model as lq

#: what makes a change a Lean one for the snapshot's `lean` flag: Lean, a lock or a Lake file
LEAN = (".lean", "lean-audit.json", "lakefile.toml", "lake-manifest.json", "lean-toolchain")
#: the most decide calls one event leads to
TICKS = 8
#: the whole tree as the contract's inputs say it (a glob whose `**` crosses directories), for the `.` `touches` reads
WHOLE = "**"


@dataclass
class Cand:
    cid: int
    base: str
    members: tuple[lq.Member, ...]
    tree: str | None = None
    tasks: frozenset[str] | None = None
    results: dict[str, lq.Result] = field(default_factory=dict)
    tries: int = 1
    reruns: int = 0
    enqueued: float | None = None
    started: set[str] = field(default_factory=set)


@dataclass
class QState:
    """What the driver knows, mutable and its own: `stack` lists the live candidates, as the simulator reads them, and `now`
    is the minute the simulator last set."""
    main: str
    ready: dict[int, tuple[str, frozenset[str], bool, int]] = field(default_factory=dict)  # pr: head, paths, reviewed, rank
    stack: list[Cand] = field(default_factory=list)
    history: list = field(default_factory=list)
    building: set[int] = field(default_factory=set)
    cid: int = 0
    rank: int = 0
    decides: int = 0
    now: float = 0.0
    rung: set[tuple[str, str]] = field(default_factory=set)  # (candidate, why) rings and (base, task) reds already paged
    dropped: list[Cand] = field(default_factory=list)  # what this tick's actions dropped, for the ejects after each drop


def _fields(cls) -> frozenset[str]:
    return frozenset(getattr(cls, "__dataclass_fields__", {}))


class Driver:
    def __init__(self, vq: ModuleType, *, deferred: Iterable[str] = (), **snapshot: object) -> None:
        unknown = sorted(set(snapshot) - _fields(vq.Snapshot))
        if unknown:
            raise TypeError(f"{getattr(vq, '__name__', 'the queue')}.Snapshot has no field {', '.join(unknown)}")
        self.vq, self.extra, self.deferred = vq, snapshot, frozenset(deferred)
        self.has = {name: _fields(getattr(vq, name)) for name in ("Failed", "Verdict", "Candidate", "Snapshot")}
        self.reading = getattr(vq, "_reading", lambda v: v.state)

    def start(self, main: str) -> QState:
        return QState(main)

    def step(self, s: QState, e: lq.Event, policy: lq.Policy) -> tuple[QState, tuple[lq.Decision, ...]]:
        handler = getattr(self, "on_" + type(e).__name__, None)
        if handler is None:
            raise TypeError(f"not a queue event: {e!r}")
        handler(s, e)
        out: list[lq.Decision] = []
        last = None
        for _ in range(TICKS):
            if s.building:
                break
            actions = list(self.vq.decide(self.snapshot(s, policy)))
            s.decides += 1
            if not actions or actions == last:
                break
            last, s.dropped = actions, []
            for a in actions:
                out += getattr(self, "do_" + type(a).__name__)(s, a)
        return s, tuple(out + self.begin(s))

    def begin(self, s: QState) -> list[lq.Decision]:
        """`Start` the deferred tasks of the lowest candidate, on `main`, once the queue reads its other tasks as passed."""
        c = s.stack[0] if s.stack else None
        if c is None or c.tree is None or c.tasks is None or c.base != s.main:
            return []
        todo = sorted((self.deferred & c.tasks) - c.started)
        if not todo or self.reading(self.verdict(c, skip=self.deferred)) != self.vq.PASS:
            return []
        c.started.update(todo)
        return [lq.Start(c.tree, t) for t in todo]

    # the snapshot ---------------------------------------------------------------------------------------------------------

    def snapshot(self, s: QState, policy: lq.Policy):
        vq, has = self.vq, self.has
        ready = tuple(vq.Ready(pr, head, paths, lean=any(p.endswith(LEAN) for p in paths), rank=rank)
                      for pr, (head, paths, reviewed, rank) in sorted(s.ready.items(), key=lambda kv: kv[1][3]) if reviewed)

        def candidate(c: Cand):
            more = {"reruns": c.reruns} if "reruns" in has["Candidate"] else {}
            if "enqueued" in has["Candidate"] and c.enqueued is not None:
                more["enqueued"] = c.enqueued * 60
            return vq.Candidate(c.tree, c.base, tuple(m.pr for m in c.members), tuple(m.head for m in c.members),
                                self.verdict(c), c.tries, **more)

        fields = {"batch": policy.batch, "depth": policy.depth}
        if "now" in has["Snapshot"]:
            fields["now"] = s.now * 60
        return vq.Snapshot(s.main, ready, tuple(candidate(c) for c in s.stack), tuple(s.history),
                           **{**fields, **self.extra})

    @staticmethod
    def key(c: Cand, task: str, r: lq.Result) -> str:
        return r.key if r.key is not None else f"{task}@{c.tree}"

    def verdict(self, c: Cand, skip: frozenset[str] = frozenset()):
        vq, has = self.vq, self.has
        if c.tree is None or c.tasks is None:
            return vq.Verdict()
        got = {t: c.results.get(t) for t in sorted(c.tasks - skip)}

        def entry(t: str, r: lq.Result):
            inputs = frozenset(WHOLE if i in lq.ROOTS else i for i in r.inputs)
            more = {k: v for k, v in (("base_key", r.base_key), ("tests", r.tests), ("base_tests", r.base_tests))
                    if k in has["Failed"]}
            return vq.Failed(t, self.key(c, t, r), inputs, **more)

        failed = tuple(entry(t, r) for t, r in got.items() if r and r.status == lq.FAIL)
        infra = tuple(entry(t, r) for t, r in got.items() if r and r.status == lq.INFRA)
        ended = all(got.values())
        if failed:
            if "ended" in has["Verdict"]:
                return vq.Verdict(vq.FAIL, failed=failed, infra=infra, ended=ended)
            return vq.Verdict(vq.FAIL, failed=failed)
        if infra:
            return vq.Verdict(vq.INFRA, infra=infra)
        if all(r and r.status == lq.PASS for r in got.values()):
            return vq.Verdict(vq.PASS)
        return vq.Verdict()

    def find(self, s: QState, tree: str) -> int:
        for i, c in enumerate(s.stack):
            if c.tree == tree:
                return i
        raise ValueError(f"decide named a candidate that isn't live: {tree} (live: {[c.tree for c in s.stack]})")

    # events ----------------------------------------------------------------------------------------------------------------

    def on_Ready(self, s: QState, e: lq.Ready) -> None:
        old = s.ready.get(e.pr)
        if old is not None and old[0] == e.head:
            rank = old[3]
        else:
            rank, s.rank = s.rank, s.rank + 1
        s.ready[e.pr] = (e.head, frozenset(e.paths), e.reviewed, rank)

    def on_Withdrawn(self, s: QState, e: lq.Withdrawn) -> None:
        s.ready.pop(e.pr, None)

    def on_Built(self, s: QState, e: lq.Built) -> None:
        for c in s.stack:
            if c.cid == e.cid and c.tree is None:
                c.tree, c.enqueued = e.tree, s.now
        s.building.discard(e.cid)

    def on_Conflict(self, s: QState, e: lq.Conflict) -> None:
        s.building.discard(e.cid)
        s.stack = [c for c in s.stack if c.cid != e.cid]

    def on_Listed(self, s: QState, e: lq.Listed) -> None:
        for c in s.stack:
            if c.tree == e.tree and c.tasks is None:
                c.tasks = frozenset(e.tasks)

    def on_Result(self, s: QState, e: lq.Result) -> None:
        for c in s.stack:
            if c.tree == e.tree:
                old = c.results.get(e.task)
                if old is None or old.status == lq.INFRA:
                    c.results[e.task] = e

    def on_MainMoved(self, s: QState, e: lq.MainMoved) -> None:
        s.main = e.tree

    # actions ---------------------------------------------------------------------------------------------------------------

    def do_Build(self, s: QState, a) -> list[lq.Decision]:
        members = tuple(lq.Member(pr, s.ready[pr][0], s.ready[pr][1]) for pr in a.members if pr in s.ready)
        c = Cand(s.cid, a.base, members)
        s.cid += 1
        s.stack.append(c)
        s.building.add(c.cid)
        return [lq.Build(c.cid, a.base, members)]

    def do_Advance(self, s: QState, a) -> list[lq.Decision]:
        i = self.find(s, a.sha)
        landed = tuple(m for c in s.stack[:i + 1] for m in c.members)
        s.stack = s.stack[i + 1:]
        s.main = a.sha
        for m in landed:
            if s.ready.get(m.pr, ("",))[0] == m.head:
                del s.ready[m.pr]
        return [lq.Advance(a.sha, landed)]

    def do_Drop(self, s: QState, a) -> list[lq.Decision]:
        if a.outcome is not None:
            s.history.append(a.outcome)
        if a.sha == s.main:
            return []
        c = s.stack.pop(self.find(s, a.sha))
        s.dropped.append(c)
        return [lq.Cancel(c.cid, c.tree, a.why)]

    def do_Eject(self, s: QState, a) -> list[lq.Decision]:
        if s.ready.get(a.number, ("",))[0] == a.head:
            del s.ready[a.number]
        tree = next((c.tree for c in [*s.stack, *s.dropped] if any(m.pr == a.number for m in c.members)), "")
        return [lq.Eject(a.number, a.head, tree or "", a.failed[0].task if a.failed else "", a.why)]

    def do_Retry(self, s: QState, a) -> list[lq.Decision]:
        c = s.stack[self.find(s, a.sha)]
        c.tries += 1
        again = sorted(t for t, r in c.results.items() if r.status == lq.INFRA)
        for t in again:
            del c.results[t]
        return [lq.Retry(c.tree, t, c.tries) for t in again]

    def do_Rerun(self, s: QState, a) -> list[lq.Decision]:
        c = s.stack[self.find(s, a.sha)]
        c.reruns += 1
        keys = set(a.keys)
        again = sorted(t for t, r in c.results.items() if r.status == lq.FAIL and self.key(c, t, r) in keys)
        for t in again:
            del c.results[t]
        return [lq.Rerun(c.tree, t) for t in again]

    def do_Ring(self, s: QState, a) -> list[lq.Decision]:
        c = s.stack[self.find(s, a.sha)]
        if (a.sha, a.why) in s.rung:
            return []
        s.rung.add((a.sha, a.why))
        return [lq.Page(c.tree, "", c.tries)]

    def do_Red(self, s: QState, a) -> list[lq.Decision]:
        new = {(a.base, f.task) for f in a.failed} - s.rung
        if not new:
            return []
        s.rung |= new
        return [lq.Page(a.sha, "red: " + ", ".join(sorted(t for _, t in new)), 0)]
