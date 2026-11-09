"""Drive a snapshot queue, a pure `decide(snapshot) -> actions` with the verified queue's contract (`research.vqueue`), with
the model's events and decisions, so the simulator and the property tests run it as they run `model.step`.

    from research import vqueue
    from verity_landq import sim, snapshots
    q = snapshots.Driver(vqueue)
    sim.simulate(arrivals, world, policy, step=q.step, state=q.start("main"))

The driver stands in for the queue's loop.  It builds each snapshot from what the events said: the ready and reviewed PRs,
the candidates with the verdict their results make, and the outcomes of decided candidates.  It calls `decide` and turns
each action into the model's decision with the same effect on the world, and calls it again until it asks for nothing
new, as a loop that ticks often would.  A tick that builds waits for its trees before deciding again, as the loop's builds
finish within its tick.  A candidate's verdict is `fail` once a task failed, `pass` once every listed task passed, `infra`
once every task has an outcome and some only infra ones, else `pending`; a failure carries the task's key and, where the
contract has `Failed.base_key`, its key on the candidate's base, as the world's results give them.  A `Red` is rung as a
page.  The module is a parameter, so nothing here imports it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

from verity_landq import model as lq

#: what makes a change a Lean one for the snapshot's `lean` flag: Lean, a lock or a Lake file
LEAN = (".lean", "lean-audit.json", "lakefile.toml", "lake-manifest.json", "lean-toolchain")
#: the most decide calls one event leads to
TICKS = 8
#: the whole tree as the contract's inputs say it (a glob whose `**` crosses directories), for the model's `.`
WHOLE = "**"


@dataclass
class Cand:
    cid: int
    base: str
    members: tuple[lq.Member, ...]
    tree: str | None = None
    tasks: frozenset[str] | None = None
    results: dict[str, tuple[str, frozenset[str], str | None, str | None]] = field(default_factory=dict)  # status, inputs, keys
    tries: int = 1


@dataclass
class QState:
    """What the driver knows, mutable and its own: `stack` lists the live candidates, as the simulator reads them."""
    main: str
    ready: dict[int, tuple[str, frozenset[str], bool, int]] = field(default_factory=dict)  # pr: head, paths, reviewed, rank
    stack: list[Cand] = field(default_factory=list)
    history: list = field(default_factory=list)
    building: set[int] = field(default_factory=set)
    cid: int = 0
    rank: int = 0
    decides: int = 0


class Driver:
    def __init__(self, vq: ModuleType) -> None:
        self.vq = vq

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
            last = actions
            for a in actions:
                out += getattr(self, "do_" + type(a).__name__)(s, a)
        return s, tuple(out)

    # the snapshot ---------------------------------------------------------------------------------------------------------

    def snapshot(self, s: QState, policy: lq.Policy):
        vq = self.vq
        ready = tuple(vq.Ready(pr, head, paths, lean=any(p.endswith(LEAN) for p in paths), rank=rank)
                      for pr, (head, paths, reviewed, rank) in sorted(s.ready.items(), key=lambda kv: kv[1][3]) if reviewed)
        cands = tuple(vq.Candidate(c.tree, c.base, tuple(m.pr for m in c.members), tuple(m.head for m in c.members),
                                   self.verdict(c), c.tries) for c in s.stack)
        return vq.Snapshot(s.main, ready, cands, tuple(s.history), batch=policy.batch, depth=policy.depth,
                           rings=policy.page_after)

    def verdict(self, c: Cand):
        vq = self.vq
        if c.tree is None or c.tasks is None:
            return vq.Verdict()
        got = {t: c.results.get(t) for t in sorted(c.tasks)}
        based = "base_key" in getattr(vq.Failed, "__dataclass_fields__", {})

        def entry(t: str, o: tuple):
            inputs = frozenset(WHOLE if i in lq.ROOTS else i for i in o[1])
            key = o[2] if o[2] is not None else f"{t}@{c.tree}"
            return vq.Failed(t, key, inputs, o[3]) if based else vq.Failed(t, key, inputs)

        failed = tuple(entry(t, o) for t, o in got.items() if o and o[0] == lq.FAIL)
        if failed:
            return vq.Verdict(vq.FAIL, failed=failed)
        if all(o and o[0] == lq.PASS for o in got.values()):
            return vq.Verdict(vq.PASS)
        if all(o for o in got.values()):
            return vq.Verdict(vq.INFRA, infra=tuple(entry(t, o) for t, o in got.items() if o[0] == lq.INFRA))
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
                c.tree = e.tree
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
                if old is None or old[0] == lq.INFRA:
                    c.results[e.task] = (e.status, frozenset(e.inputs), e.key, e.base_key)

    def on_MainMoved(self, s: QState, e: lq.MainMoved) -> None:
        s.main = e.tree

    # actions ---------------------------------------------------------------------------------------------------------------

    def do_Build(self, s: QState, a) -> list[lq.Decision]:
        members = tuple(lq.Member(pr, s.ready[pr][0], s.ready[pr][1]) for pr in a.members if pr in s.ready)
        c = Cand(s.cid, a.base, members)
        s.cid += 1
        s.stack.append(c)
        s.building.add(c.cid)
        return [lq.Build(c.cid, a.base, members, False)]

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
        return [lq.Cancel(c.cid, c.tree, a.why)]

    def do_Eject(self, s: QState, a) -> list[lq.Decision]:
        if s.ready.get(a.number, ("",))[0] == a.head:
            del s.ready[a.number]
        tree = next((c.tree for c in s.stack if any(m.pr == a.number for m in c.members)), "")
        return [lq.Eject(a.number, a.head, tree or "", a.failed[0].task if a.failed else "", a.why)]

    def do_Retry(self, s: QState, a) -> list[lq.Decision]:
        c = s.stack[self.find(s, a.sha)]
        c.tries += 1
        again = sorted(t for t, o in c.results.items() if o[0] == lq.INFRA)
        for t in again:
            del c.results[t]
        return [lq.Retry(c.tree, t, c.tries) for t in again]

    def do_Ring(self, s: QState, a) -> list[lq.Decision]:
        c = s.stack[self.find(s, a.sha)]
        return [lq.Page(c.tree, "", c.tries)]

    def do_Red(self, s: QState, a) -> list[lq.Decision]:
        return [lq.Page(a.sha, "red: " + ", ".join(f.task for f in a.failed), 0)]
