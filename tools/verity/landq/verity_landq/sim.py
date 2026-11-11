"""A discrete-event world that replays PR traffic through a landing queue, and what the queue would have done.

    from research import vqueue
    from verity_landq import model as lq, sim, snapshots
    q = snapshots.Driver(vqueue)
    world = sim.World(tasks=(sim.Task("pytest:a", frozenset({"a/"}), 12.0),), nodes=4)
    report = sim.simulate([sim.Arrival(7, "<head>", 0.0, frozenset({"a/x.py"}))], world, lq.Policy(batch=4, depth=2),
                          step=q.step, state=q.start("main"))
    sim.summary(report)["p90_h"]

Times are minutes from the trace's start.  The world holds what the queue doesn't decide: when each PR is marked ready and
reviewed, what its change touches and which gating tasks it breaks, the tasks with their inputs and durations, the nodes
that run them, and the faults.  The queue is a `step(state, event, policy)` from a start `state`, as `snapshots.Driver`
gives them for a snapshot queue's `decide`.

- A candidate's tree is built `World.build` minutes after its `Build`, and its tasks are listed `World.listing` minutes
  later.  Every task listed gates.
- A task's key on a tree is the set of PR heads in the tree's history whose change touches the task's inputs, so a candidate
  runs only the tasks its own members touch and reads the rest from the cache, as an executor keyed by content does.  A key
  already running or queued is not run twice.
- A run takes one of `World.nodes` nodes for its task's `minutes`, lower candidates first; a `Task.off_pool` task runs on
  capacity of its own, at once.  A `Task.deferred` task is listed with the others but runs only when the queue decides
  `Start` for it, never from the cache.
- A run fails when its tree fails the task, else falsely with probability `World.flaky`; either fail is cached, as a pass
  is.  A tree fails the tasks its start's tree is red on (`World.red`, cached as failed at the start) and those a PR head
  in its history breaks, less those a later head in it fixes.  With probability `World.infra` a run ends in an infra error
  instead, which is never cached.  Each result carries the task's key on its tree and on the tree's base, and a fail its
  failing tests (`RED` for a task the start is red on, `#N` for one PR N breaks, `flaky:<run>` for a false fail) and the
  base's failing tests of the task.  A `Rerun` runs a task afresh, its cached result set aside.
- An ejected PR comes back `World.fix` minutes later at a new head that breaks nothing.
- An advance is unsound when the tree fails a test `main` didn't.

`summary` reports ready-to-main (from a PR's first ready mark to the first `main` holding its last head) and main's lag (at
each sample, how long the PR the queue has held longest off `main` has waited since it was ready and reviewed).
"""
from __future__ import annotations

import hashlib
import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from verity_landq import model as lq

#: the failing test of a task the start's tree is red on
RED = "red"


@dataclass(frozen=True)
class Task:
    """A gating task: its id, its declared inputs (paths, directories, `.` or globs, as `model.touches` reads them) and how
    long a run takes."""
    name: str
    inputs: frozenset[str]
    minutes: float
    off_pool: bool = False
    deferred: bool = False


@dataclass(frozen=True)
class Arrival:
    """A PR marked ready at `head` at minute `ready`, reviewed at `reviewed` (at `ready` when None), whose change touches
    `paths`, fails the tasks named in `breaks` and makes pass again those named in `fixes` (each one it touches)."""
    pr: int
    head: str
    ready: float
    paths: frozenset[str]
    reviewed: float | None = None
    breaks: frozenset[str] = frozenset()
    fixes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class World:
    tasks: tuple[Task, ...]
    nodes: int = 4
    build: float = 2.0
    listing: float = 3.0
    infra: float = 0.0
    flaky: float = 0.0
    fix: float = 120.0
    seed: int = 0
    warm: bool = True                  # every task's result at the start's tree is cached: a `main` that passed
    red: frozenset[str] = frozenset()  # the tasks the start's tree fails, until a change that fixes them reaches `main`


@dataclass(frozen=True)
class Landing:
    """One arrival: its first ready mark, when it was first ready and reviewed, when `main` first held its last head (None
    if never), and how often it was ejected, at a head that broke nothing (`false_ejects`) or not."""
    pr: int
    head: str
    ready: float
    batchable: float
    main: float | None
    ejects: int
    false_ejects: int


@dataclass(frozen=True)
class Report:
    landings: tuple[Landing, ...]
    held: tuple[tuple[float, float], ...]      # (from, to): a PR head the queue held off main, ready and reviewed
    end: float
    nodes: int
    candidates: int = 0
    retries: int = 0
    reruns: int = 0
    pages: int = 0
    runs: int = 0
    cached: int = 0
    busy: float = 0.0                           # node-minutes of runs
    advances: int = 0
    unsound: tuple[str, ...] = ()               # trees main moved to that fail a test main didn't


Step = Callable[[Any, lq.Event, lq.Policy], tuple[Any, tuple[lq.Decision, ...]]]


def simulate(arrivals: Iterable[Arrival], world: World, policy: lq.Policy, *, step: Step, state: Any, main: str = "main",
             limit: int = 5_000_000, until: float | None = None) -> Report:
    """Replay ``arrivals`` through the queue ``step`` from ``state``, its start with `main` at ``main``, in ``world`` until
    nothing is left to happen, ``limit`` events, or minute ``until``, where the report ends.  The state need only list the
    queue's candidates as `stack`, each with its `tree`; a `now` it has is set to the minute of each event before the event
    is stepped."""
    return _Sim(tuple(arrivals), world, policy, step, main, limit, state).run(until)


def quantile(xs: list[float], f: float) -> float:
    """The `f`-quantile of `xs`: the sorted value at index ⌊f·n⌋, or the last one when that index is past the end."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def summary(r: Report, *, within: float = 120.0, lag: float = 180.0, every: float = 10.0, start: float = 0.0) -> dict:
    """Ready-to-main quantiles (hours) over the arrivals that reached `main`, from the ready mark and from the review
    (``review_*``, which leaves out the wait for a review), the share of all arrivals on `main` within ``within`` minutes of
    their ready mark, main's lag (its worst, and the share of samples every ``every`` minutes from ``start`` at most
    ``lag``), and what the queue spent: candidates, ejects, retries, runs and node use."""
    rows = [x for x in r.landings if x.ready >= start]
    hours = [(x.main - x.ready) / 60 for x in rows if x.main is not None]
    reviewed = [(x.main - x.batchable) / 60 for x in rows if x.main is not None]
    samples = _lag(r.held, start, r.end, every)
    return {"prs": len(rows), "landed": len(hours), "p50_h": quantile(hours, .5), "p90_h": quantile(hours, .9),
            "review_p50_h": quantile(reviewed, .5), "review_p90_h": quantile(reviewed, .9),
            "within": sum(x.main is not None and x.main - x.ready <= within for x in rows) / len(rows) if rows else 0.0,
            "lag_max_h": max(samples, default=0.0) / 60, "lag_ok": sum(s <= lag for s in samples) / len(samples) if samples else 1.0,
            "ejects": sum(x.ejects for x in rows), "false_ejects": sum(x.false_ejects for x in rows),
            "candidates": r.candidates, "advances": r.advances, "retries": r.retries,
            "reruns": r.reruns, "pages": r.pages, "runs": r.runs, "cached": r.cached,
            "utilization": r.busy / (r.nodes * (r.end - start)) if r.end > start else 0.0, "unsound": len(r.unsound)}


def _lag(held: tuple[tuple[float, float], ...], start: float, end: float, every: float) -> list[float]:
    """At each sample, the longest any held interval covering it has run."""
    spans = sorted(held)
    out, i, live = [], 0, []
    t = start
    while t <= end:
        while i < len(spans) and spans[i][0] <= t:
            heapq.heappush(live, spans[i])
            i += 1
        while live and live[0][1] <= t:
            heapq.heappop(live)
        open_ = [a for a, b in live if b > t]
        out.append(t - min(open_) if open_ else 0.0)
        t += every
    return out


class Index:
    """Which of ``tasks`` a change touches, as `model.touches` reads their inputs, without a scan of every input."""

    def __init__(self, tasks: tuple[Task, ...]) -> None:
        self.exact: dict[str, list[int]] = {}
        self.globs: list[tuple[str, int]] = []
        self.roots: list[int] = []
        for i, t in enumerate(tasks):
            for s in t.inputs:
                if s in lq.ROOTS:
                    self.roots.append(i)
                elif lq.GLOB.intersection(s):
                    self.globs.append((s, i))
                else:
                    self.exact.setdefault(s.rstrip("/"), []).append(i)

    def touch(self, paths: frozenset[str]) -> frozenset[int]:
        """The indices of the tasks whose inputs a change to ``paths`` touches."""
        hit: set[int] = set(self.roots) if paths else set()
        for p in paths:
            parts = p.split("/")
            for i in range(1, len(parts) + 1):
                hit.update(self.exact.get("/".join(parts[:i]), ()))
        for g, i in self.globs:
            if i not in hit and lq.touches(paths, frozenset({g})):
                hit.add(i)
        return frozenset(hit)


def _bit(pr: int, head: str) -> int:
    return int.from_bytes(hashlib.sha512(f"{pr}@{head}".encode()).digest()[:8], "big")


@dataclass
class _Run:
    task: int
    key: int
    bad: frozenset[str]                         # the tests its tree fails of the task
    waiters: set[str] = field(default_factory=set)


def _tests(bad: frozenset[tuple[int, str]], i: int) -> frozenset[str]:
    return frozenset(t for j, t in bad if j == i)


class _Sim:
    def __init__(self, arrivals: tuple[Arrival, ...], world: World, policy: lq.Policy, step: Step, main: str, limit: int,
                 state: Any):
        self.world, self.policy, self.step, self.limit = world, policy, step, limit
        self.tasks = world.tasks
        self.names = [t.name for t in self.tasks]
        self.index = {t.name: i for i, t in enumerate(self.tasks)}
        self.touch = Index(self.tasks).touch
        self.rng = random.Random(world.seed)
        self.state = state
        self.now = 0.0
        self.heap: list = []
        self.n = 0
        unknown = sorted(t for t in world.red if t not in self.index)
        if unknown:
            raise ValueError(f"World.red names tasks the world doesn't list: {unknown}")
        red = frozenset(self.index[t] for t in world.red)
        # a tree's keys by task, and the (task, test) pairs it fails
        self.trees: dict[str, tuple[tuple[int, ...], frozenset[tuple[int, str]]]] = {
            main: (tuple(0 for _ in self.tasks), frozenset((i, RED) for i in red))}
        self.base: dict[str, str] = {}
        self.main = main
        self.live: set[str] = set()
        self.cache: dict[tuple[int, int], tuple[str, frozenset[str]]] = (
            {(i, 0): (lq.FAIL, frozenset({RED})) if i in red else (lq.PASS, frozenset()) for i in range(len(self.tasks))}
            if world.warm else {})
        self.runs: dict[int, _Run] = {}
        self.by_key: dict[tuple[int, int], int] = {}
        self.runq: list = []
        self.free = world.nodes
        self.info: dict[tuple[int, str], tuple[frozenset[str], frozenset[int], frozenset[int], frozenset[int]]] = {}
        self.row: dict[tuple[int, str], int] = {}
        self.rows: list[dict] = []
        self.open: dict[int, tuple[str, float]] = {}
        self.held: list[tuple[float, float]] = []
        self.counts = dict(candidates=0, retries=0, reruns=0, pages=0, runs=0, cached=0, advances=0)
        self.busy = 0.0
        self.unsound: list[str] = []
        for a in sorted(arrivals, key=lambda a: (a.ready, a.pr)):
            rid = len(self.rows)
            reviewed = a.ready if a.reviewed is None else max(a.ready, a.reviewed)
            self.rows.append(dict(pr=a.pr, head=a.head, ready=a.ready, batchable=reviewed, main=None, ejects=0, false=0))
            self.know(a.pr, a.head, a.paths, a.breaks, rid, a.fixes)
            self.push(a.ready, "ready", (a.pr, a.head, reviewed <= a.ready))
            if reviewed > a.ready:
                self.push(reviewed, "ready", (a.pr, a.head, True))

    # the world ---------------------------------------------------------------------------------------------------------

    def know(self, pr: int, head: str, paths: frozenset[str], breaks: frozenset[str], rid: int,
             fixes: frozenset[str] = frozenset()) -> None:
        touch = self.touch(paths)
        unknown = sorted(b for b in breaks | fixes if b not in self.index)
        if unknown:
            raise ValueError(f"Arrival #{pr}@{head} breaks or fixes tasks the world doesn't list: {unknown}")

        def touched(names: frozenset[str]) -> frozenset[int]:
            return frozenset(self.index[b] for b in names if self.index[b] in touch)

        self.info[(pr, head)] = (paths, touched(breaks), touch, touched(fixes))
        self.row[(pr, head)] = rid

    def result(self, tree: str, i: int, status: str, tests: frozenset[str] = frozenset()) -> lq.Result:
        base = self.base.get(tree)
        base_tests = None if base is None else _tests(self.trees[base][1], i) or None
        return lq.Result(tree, self.names[i], status, self.tasks[i].inputs, f"{self.trees[tree][0][i]:x}",
                         None if base is None else f"{self.trees[base][0][i]:x}", tests, base_tests)

    def push(self, t: float, kind: str, payload) -> None:
        heapq.heappush(self.heap, (t, self.n, kind, payload))
        self.n += 1

    def run(self, until: float | None = None) -> Report:
        events = 0
        while self.heap and events < self.limit:
            if until is not None and self.heap[0][0] > until:
                self.now = until
                break
            t, _, kind, payload = heapq.heappop(self.heap)
            self.now = t
            events += 1
            getattr(self, "on_" + kind)(*payload)
        end = self.now
        for _, a in self.open.values():
            self.held.append((a, end))
        rows = tuple(Landing(r["pr"], r["head"], r["ready"], r["batchable"], r["main"], r["ejects"], r["false"])
                     for r in self.rows)
        return Report(rows, tuple(self.held), end, self.world.nodes, busy=self.busy, unsound=tuple(self.unsound),
                      **self.counts)

    def feed(self, event: lq.Event) -> None:
        if hasattr(self.state, "now"):
            self.state.now = self.now
        self.state, ds = self.step(self.state, event, self.policy)
        for d in ds:
            getattr(self, "do_" + type(d).__name__)(d)
        if self.live:
            self.live &= {c.tree for c in self.state.stack}

    def hold(self, pr: int, head: str, reviewed: bool) -> None:
        """``head`` is now ``pr``'s: the head it moved from stops counting as held, and a reviewed one is held from now."""
        old = self.open.get(pr)
        if old is not None and old[0] != head:
            self.release(pr, old[0])
        if reviewed:
            self.open.setdefault(pr, (head, self.now))

    def release(self, pr: int, head: str) -> None:
        old = self.open.get(pr)
        if old is not None and old[0] == head:
            del self.open[pr]
            self.held.append((old[1], self.now))

    def on_ready(self, pr: int, head: str, reviewed: bool) -> None:
        self.hold(pr, head, reviewed)
        self.feed(lq.Ready(pr, head, self.info[(pr, head)][0], reviewed))

    def on_built(self, cid: int, tree: str) -> None:
        self.feed(lq.Built(cid, tree))
        if any(c.tree == tree for c in self.state.stack):
            self.push(self.now + self.world.listing, "listed", (tree,))

    def on_listed(self, tree: str) -> None:
        if not any(c.tree == tree for c in self.state.stack):
            return
        self.live.add(tree)
        self.feed(lq.Listed(tree, frozenset(self.names)))
        keys = self.trees[tree][0]
        for i, name in enumerate(self.names):
            if tree not in self.live:
                return
            if self.tasks[i].deferred:
                continue
            got = self.cache.get((i, keys[i]))
            if got is not None:
                self.counts["cached"] += 1
                self.feed(self.result(tree, i, *got))
            else:
                self.want(i, tree)

    def want(self, i: int, tree: str) -> None:
        keys, bad = self.trees[tree]
        k = (i, keys[i])
        rid = self.by_key.get(k)
        if rid is not None:
            self.runs[rid].waiters.add(tree)
            return
        rid = self.n
        self.n += 1
        self.runs[rid] = _Run(i, keys[i], _tests(bad, i), {tree})
        self.by_key[k] = rid
        if self.tasks[i].off_pool:
            self.begin(rid)
        else:
            heapq.heappush(self.runq, (int(tree[1:]) if tree[1:].isdigit() else 0, rid))
            self.dispatch()

    def dispatch(self) -> None:
        while self.free > 0 and self.runq:
            _, rid = heapq.heappop(self.runq)
            run = self.runs.get(rid)
            if run is None:
                continue
            if not run.waiters & self.live:
                del self.runs[rid], self.by_key[(run.task, run.key)]
                continue
            self.free -= 1
            self.busy += self.tasks[run.task].minutes
            self.begin(rid)

    def begin(self, rid: int) -> None:
        self.counts["runs"] += 1
        self.push(self.now + self.tasks[self.runs[rid].task].minutes, "done", (rid,))

    def on_done(self, rid: int) -> None:
        run = self.runs.pop(rid)
        del self.by_key[(run.task, run.key)]
        task = self.tasks[run.task]
        if not task.off_pool:
            self.free += 1
        tests: frozenset[str] = frozenset()
        if self.rng.random() < self.world.infra:
            status = lq.INFRA
        elif run.bad or self.rng.random() < self.world.flaky:
            status, tests = lq.FAIL, run.bad or frozenset({f"flaky:{rid}"})
        else:
            status = lq.PASS
        if status != lq.INFRA:
            self.cache[(run.task, run.key)] = (status, tests)
        for tree in sorted(run.waiters):
            if tree in self.live:
                self.feed(self.result(tree, run.task, status, tests))
        self.dispatch()

    def on_back(self, pr: int, head: str) -> None:
        self.hold(pr, head, True)
        self.feed(lq.Ready(pr, head, self.info[(pr, head)][0], True))

    # the queue's decisions ------------------------------------------------------------------------------------------------

    def do_Build(self, d: lq.Build) -> None:
        keys, bad = self.trees[d.base]
        keys, bad = list(keys), set(bad)
        for m in d.members:
            _, breaks, touch, fixes = self.info[(m.pr, m.head)]
            b = _bit(m.pr, m.head)
            for i in touch:
                keys[i] ^= b
            broken = {(j, f"#{m.pr}") for j in breaks}
            bad = {(j, t) for j, t in bad | broken if j not in fixes}
        tree = f"T{d.cid}"
        self.trees[tree] = (tuple(keys), frozenset(bad))
        self.base[tree] = d.base
        self.counts["candidates"] += 1
        self.push(self.now + self.world.build, "built", (d.cid, tree))

    def do_Cancel(self, d: lq.Cancel) -> None:
        self.live.discard(d.tree)

    def do_Advance(self, d: lq.Advance) -> None:
        self.counts["advances"] += 1
        if self.trees[d.tree][1] - self.trees[self.main][1]:
            self.unsound.append(d.tree)
        self.main = d.tree
        for m in d.landed:
            r = self.rows[self.row[(m.pr, m.head)]]
            if r["main"] is None:
                r["main"] = self.now
            self.release(m.pr, m.head)

    def do_Eject(self, d: lq.Eject) -> None:
        rid = self.row[(d.pr, d.head)]
        r = self.rows[rid]
        r["ejects"] += 1
        paths, breaks, _, fixes = self.info[(d.pr, d.head)]
        r["false"] += not breaks
        self.release(d.pr, d.head)
        head = f"{d.head}+"
        self.know(d.pr, head, paths, frozenset(), rid, frozenset(self.names[i] for i in fixes))
        r["head"] = head
        self.push(self.now + self.world.fix, "back", (d.pr, head))

    def do_Retry(self, d: lq.Retry) -> None:
        self.counts["retries"] += 1
        if d.tree in self.live:
            self.want(self.index[d.task], d.tree)

    def do_Rerun(self, d: lq.Rerun) -> None:
        self.counts["reruns"] += 1
        if d.tree in self.live:
            i = self.index[d.task]
            self.cache.pop((i, self.trees[d.tree][0][i]), None)
            self.want(i, d.tree)

    def do_Start(self, d: lq.Start) -> None:
        if d.tree in self.live:
            self.want(self.index[d.task], d.tree)

    def do_Page(self, d: lq.Page) -> None:
        self.counts["pages"] += 1
