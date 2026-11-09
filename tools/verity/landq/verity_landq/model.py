"""The landing queue's decision logic as a pure reference model: `main` moves only to a candidate whose every gating task
passed on that exact tree.

No I/O, no clock and nothing but the standard library.  `step` is the one function that makes a new `State`: it takes an
`Event` (what the queue heard) and returns the next state and the `Decision`s its driver carries out (build a candidate, move
`main`, eject a PR, retry a task).  A `State` is frozen; every verdict is derived from the results the events delivered
(`verdict`), and a result counts only for the tree it was reported on.  The same events give the same states and decisions
wherever they are replayed.

    from verity_landq import model as lq

    policy = lq.Policy(batch=4, depth=2)
    state = lq.State.start("<main's tip>")
    state, decisions = lq.step(state, lq.Ready(7, "<head>", frozenset({"a/x.py"}), reviewed=True), policy)
    # decisions == (Build(cid=0, base="<main's tip>", members=(Member(7, "<head>", ...),), bisect=False),)

The queue
---------
A PR is queued while it is ready at its head (`Ready`), and batched once it is also reviewed there; a new head is a new
entry at the back of the queue, and `Withdrawn` takes it out.  The queue tests *candidates*: up to `Policy.batch` queued PRs,
in queue order, merged onto `main`'s tip, the next batch merged onto that while it runs, and so on up to `Policy.depth`
candidates in flight (the *stack*, bottom first).  The driver builds each candidate's merge (`Build`), answering with its tree
(`Built`) or the member that doesn't merge (`Conflict`, which sets that PR aside until `main` moves or it has a new head),
lists its gating tasks (`Listed`) and reports each one's outcome on that tree (`Result`): `PASS`, `FAIL` with the task's
declared inputs, or `INFRA` (a timeout, an OOM kill, a lost node, a setup death), which is no verdict.

The rules
---------
1. `main` advances only to a candidate whose every listed gating task passed on its own tree (`Advance`): the highest such
   candidate of the stack, with every candidate below it, which its tree holds.
2. A fail stands: a candidate with a failing gating task never passes, and every candidate above it is cancelled (`Cancel`),
   its members back in the queue in their places.
3. A failing task's suspects are the members of the failing candidate and of each candidate below it, down to the nearest
   one that passed that task, or `main`.  The one suspect whose change touches the task's declared inputs is ejected
   (`Eject`, why `INPUTS`); a lone suspect is ejected whatever it touches (why `ALONE`).  The candidate holding an ejected PR
   and every one above it are cancelled.
4. When no failing task names one PR (none of its suspects touches its inputs, or several do) and every suspect is a member of
   the failing candidate, the candidate is bisected: cancelled, and its members tested as two candidates holding only them, on
   its base, the first half below the second (`Build.bisect`).  Until then (a candidate below it hasn't reported the task) it
   waits at the top of the stack, and nothing is built on it.
5. An infra outcome ejects nothing and fails nothing: the task is retried on its tree (`Retry`), and paged once it has ended
   so `Policy.page_after` times (`Page`).
6. A candidate is built only on a candidate that is built (its tree known) and hasn't failed, so each `Build`'s base is a
   tree.  `MainMoved`, a push of `main` the queue didn't decide, lands the candidates up to the one whose tree it is, or else
   cancels the whole stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import Mapping, Union

PASS, FAIL, INFRA, PENDING = "pass", "fail", "infra", "pending"
#: inputs that name the whole tree, and the characters that make an input a glob
ROOTS, GLOB = frozenset({".", "./"}), frozenset("*?[")
#: why a PR was ejected: its change touches the failing task's declared inputs, or it was the failing task's only suspect
INPUTS, ALONE = "inputs", "alone"
#: why a candidate was cancelled
BELOW_FAILED, EJECTED, BISECTED, NEW_HEAD, UNREVIEWED, WITHDRAWN, CONFLICT, MAIN_MOVED = (
    "a candidate below it failed", "it holds an ejected PR or stands on one that does", "bisected",
    "a member has a new head", "a member is no longer reviewed", "a member was withdrawn", "a member doesn't merge",
    "main moved")


@dataclass(frozen=True)
class Policy:
    """How the queue batches: `batch` PRs a candidate at most (k), `depth` candidates in flight at most (d), and an infra
    outcome paged at its `page_after`-th time on one task of one tree."""
    batch: int = 4
    depth: int = 2
    page_after: int = 3

    def __post_init__(self) -> None:
        for name in ("batch", "depth", "page_after"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Policy.{name} must be a positive int, not {getattr(self, name)!r}")


@dataclass(frozen=True)
class Member:
    """A PR in a candidate: its number, the head merged and the paths its change touches."""
    pr: int
    head: str
    paths: frozenset[str]


@dataclass(frozen=True)
class Queued:
    """A PR ready at `head`: batched once `reviewed` there and not set `aside` (a conflict), in `seq` order."""
    pr: int
    head: str
    paths: frozenset[str]
    reviewed: bool
    seq: int
    aside: bool = False


@dataclass(frozen=True)
class Candidate:
    """A batch in flight: `members` merged onto `base`, its `tree` once built, its gating `tasks` once listed."""
    cid: int
    base: str
    members: tuple[Member, ...]
    bisect: bool = False
    tree: str | None = None
    tasks: frozenset[str] | None = None


@dataclass(frozen=True)
class Outcome:
    """What the results say of one task on one tree: `PASS`, `FAIL` (with the task's declared `inputs`) or `INFRA`, and how
    many infra outcomes it had."""
    status: str
    inputs: frozenset[str] = frozenset()
    infra: int = 0


# --- events --------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ready:
    """PR `pr` is ready at `head`, whose change touches `paths`, and `reviewed` there or not.  At its queued head it updates
    `reviewed` in place (a candidate holding it is cancelled when it is no longer reviewed); at a new head the PR goes to the
    back of the queue; an ejected PR is queued again."""
    pr: int
    head: str
    paths: frozenset[str]
    reviewed: bool = True


@dataclass(frozen=True)
class Withdrawn:
    """PR `pr` is no longer ready: unmarked, closed or held."""
    pr: int


@dataclass(frozen=True)
class Built:
    """Candidate `cid`'s merge is commit `tree`."""
    cid: int
    tree: str


@dataclass(frozen=True)
class Conflict:
    """Candidate `cid`'s member `pr` doesn't merge onto its base and the members before it."""
    cid: int
    pr: int


@dataclass(frozen=True)
class Listed:
    """`tasks` are the gating tasks of `tree`."""
    tree: str
    tasks: frozenset[str]


@dataclass(frozen=True)
class Result:
    """Task `task` ended on `tree` with `status`: `PASS`, `FAIL` (with the task's declared `inputs`) or `INFRA`; `key` is the
    task's key on `tree` and `base_key` its key on the tree's base, `tests` a fail's failing tests and `base_tests` the
    base's failing tests of the task (None when the base passes it), when the world knows them (the model reads none)."""
    tree: str
    task: str
    status: str
    inputs: frozenset[str] = frozenset()
    key: str | None = None
    base_key: str | None = None
    tests: frozenset[str] = frozenset()
    base_tests: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.status not in (PASS, FAIL, INFRA):
            raise ValueError(f"Result.status must be one of {PASS!r}, {FAIL!r}, {INFRA!r}, not {self.status!r}")


@dataclass(frozen=True)
class MainMoved:
    """`main` was pushed to `tree` by something other than this queue's `Advance`."""
    tree: str


Event = Union[Ready, Withdrawn, Built, Conflict, Listed, Result, MainMoved]


# --- decisions -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Build:
    """Merge `members`' heads, in order, onto `base` (`main`'s tip or a candidate's tree), answer `Built` or `Conflict`, and
    put the tree's check on the executor.  `bisect`: the members are half of a failed candidate's, tested alone."""
    cid: int
    base: str
    members: tuple[Member, ...]
    bisect: bool


@dataclass(frozen=True)
class Cancel:
    """Candidate `cid` (tree `tree`, None before it was built) is out of flight, for `why`; its check may stop."""
    cid: int
    tree: str | None
    why: str


@dataclass(frozen=True)
class Advance:
    """Fast-forward `main` to `tree`, landing `landed`."""
    tree: str
    landed: tuple[Member, ...]


@dataclass(frozen=True)
class Eject:
    """Take PR `pr` at `head` out of the queue until its next ready mark: task `task` failed on `tree`, and `why` names it."""
    pr: int
    head: str
    tree: str
    task: str
    why: str


@dataclass(frozen=True)
class Retry:
    """Run task `task` on `tree` again: its `attempt`-th outcome there was infra."""
    tree: str
    task: str
    attempt: int


@dataclass(frozen=True)
class Rerun:
    """Run task `task` on `tree` afresh: the cached result at its key is set aside, so the run doesn't read it."""
    tree: str
    task: str


@dataclass(frozen=True)
class Start:
    """Run task `task` on `tree`, a task that runs only when the queue starts it (`sim.Task.deferred`)."""
    tree: str
    task: str


@dataclass(frozen=True)
class Page:
    """Task `task` on `tree` has ended with an infra outcome `attempts` times."""
    tree: str
    task: str
    attempts: int


@dataclass(frozen=True)
class SetAside:
    """PR `pr` at `head` doesn't merge onto candidate `cid`'s base: it waits until `main` moves or it has a new head."""
    pr: int
    head: str
    cid: int


Decision = Union[Build, Cancel, Advance, Eject, Retry, Rerun, Start, Page, SetAside]


# --- state ---------------------------------------------------------------------------------------------------------------

def _frozen(d: Mapping) -> Mapping:
    return MappingProxyType(dict(d))


@dataclass(frozen=True)
class State:
    """`main`'s tip; the `queue` of ready PRs by number (those in flight among them); the `stack` of candidates in flight,
    bottom first; the bisection `groups` still to build, in order; the `results` by (tree, task) of the stack's trees; the PRs
    `ejected` at their head, until a new ready mark; and the next queue `seq` and candidate id `cid`."""
    main: str
    queue: Mapping[int, Queued] = field(default_factory=dict)
    stack: tuple[Candidate, ...] = ()
    groups: tuple[tuple[Member, ...], ...] = ()
    results: Mapping[tuple[str, str], Outcome] = field(default_factory=dict)
    ejected: Mapping[int, Eject] = field(default_factory=dict)
    seq: int = 0
    cid: int = 0

    def __post_init__(self) -> None:
        for name in ("queue", "results", "ejected"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    @classmethod
    def start(cls, main: str) -> State:
        """The queue before any event: `main` at `main`, nothing ready, nothing in flight."""
        return cls(main)


def verdict(state: State, candidate: Candidate) -> str:
    """`FAIL` once a listed gating task of the candidate failed on its tree, `PASS` once every one passed there, else `PENDING`
    (not built, not listed, or a task without an outcome or with only infra ones).  Only results on the candidate's own tree
    count."""
    return _verdict(state.results, candidate)


def _verdict(results: Mapping[tuple[str, str], Outcome], candidate: Candidate) -> str:
    if candidate.tree is None or candidate.tasks is None:
        return PENDING
    got = [results.get((candidate.tree, t)) for t in candidate.tasks]
    if any(o is not None and o.status == FAIL for o in got):
        return FAIL
    return PASS if all(o is not None and o.status == PASS for o in got) else PENDING


def touches(paths: frozenset[str], inputs: frozenset[str]) -> bool:
    """Whether a change touching `paths` touches `inputs`: a path that is an input, lies under one (a directory; `.` is the
    whole tree), or matches one that is a glob (`fnmatch`'s, whose `*` crosses `/`)."""
    if not paths:
        return False
    if paths & inputs or inputs & ROOTS:
        return True
    globs = [i for i in inputs if GLOB.intersection(i)]
    if globs and any(fnmatchcase(p, g) for g in globs for p in paths):
        return True
    dirs = tuple(i.rstrip("/") + "/" for i in inputs if not GLOB.intersection(i))
    return any(p.startswith(dirs) for p in paths)


def in_flight(state: State) -> frozenset[int]:
    """The PRs some candidate of the stack holds."""
    return frozenset(m.pr for c in state.stack for m in c.members)


def queued(state: State) -> tuple[Queued, ...]:
    """The ready PRs in queue order."""
    return tuple(sorted(state.queue.values(), key=lambda q: q.seq))


def step(state: State, event: Event, policy: Policy) -> tuple[State, tuple[Decision, ...]]:
    """The queue's next state after `event`, and what its driver does now, in order."""
    tx = _Tx(state, policy)
    handler = _HANDLERS.get(type(event))
    if handler is None:
        raise TypeError(f"not a queue event: {event!r}")
    handler(tx, event)
    tx.settle()
    return tx.freeze(), tuple(tx.out)


class _Tx:
    """One `step`'s working copy of a state: the only place state changes, discarded once frozen."""

    def __init__(self, s: State, policy: Policy) -> None:
        self.policy = policy
        self.main = s.main
        self.queue = dict(s.queue)
        self.stack = list(s.stack)
        self.groups = list(s.groups)
        self.results = dict(s.results)
        self.ejected = dict(s.ejected)
        self.seq, self.cid = s.seq, s.cid
        self.out: list[Decision] = []

    def freeze(self) -> State:
        return State(self.main, self.queue, tuple(self.stack), tuple(self.groups), self.results, self.ejected, self.seq,
                     self.cid)

    def verdict(self, c: Candidate) -> str:
        return _verdict(self.results, c)

    # handlers ----------------------------------------------------------------------------------------------------------

    def ready(self, e: Ready) -> None:
        old = self.queue.get(e.pr)
        paths = frozenset(e.paths)
        if old is not None and old.head == e.head:
            self.queue[e.pr] = replace(old, paths=paths, reviewed=e.reviewed)
            if not e.reviewed:
                self.drop(e.pr, UNREVIEWED)
            return
        if old is not None:
            self.drop(e.pr, NEW_HEAD)
        self.ejected.pop(e.pr, None)
        self.queue[e.pr] = Queued(e.pr, e.head, paths, e.reviewed, self.seq)
        self.seq += 1

    def withdrawn(self, e: Withdrawn) -> None:
        if self.queue.pop(e.pr, None) is not None:
            self.drop(e.pr, WITHDRAWN)

    def built(self, e: Built) -> None:
        for i, c in enumerate(self.stack):
            if c.cid == e.cid and c.tree is None:
                self.stack[i] = replace(c, tree=e.tree)

    def conflict(self, e: Conflict) -> None:
        for i, c in enumerate(self.stack):
            if c.cid == e.cid and any(m.pr == e.pr for m in c.members):
                self.cancel_from(i, CONFLICT)
                q = self.queue.get(e.pr)
                head = next(m.head for m in c.members if m.pr == e.pr)
                if q is not None and q.head == head:
                    self.queue[e.pr] = replace(q, aside=True)
                    self.out.append(SetAside(e.pr, head, e.cid))
                return

    def listed(self, e: Listed) -> None:
        for i, c in enumerate(self.stack):
            if c.tree == e.tree and c.tasks is None:
                self.stack[i] = replace(c, tasks=frozenset(e.tasks))

    def result(self, e: Result) -> None:
        if e.tree not in {c.tree for c in self.stack}:
            return
        key = (e.tree, e.task)
        old = self.results.get(key)
        if e.status == INFRA:
            if old is not None and old.status != INFRA:
                return
            n = (old.infra if old else 0) + 1
            self.results[key] = Outcome(INFRA, frozenset(), n)
            self.out.append(Retry(e.tree, e.task, n))
            if n == self.policy.page_after:
                self.out.append(Page(e.tree, e.task, n))
            return
        if old is not None and old.status == FAIL:
            return
        self.results[key] = Outcome(e.status, frozenset(e.inputs), old.infra if old else 0)

    def main_moved(self, e: MainMoved) -> None:
        for i, c in enumerate(self.stack):
            if c.tree == e.tree:
                self.land(i)
                return
        self.cancel_from(0, MAIN_MOVED)
        self.groups.clear()
        self.main = e.tree
        self.unset_aside()

    # the rules ---------------------------------------------------------------------------------------------------------

    def settle(self) -> None:
        while self.advance() or self.fail() or self.form():
            pass

    def advance(self) -> bool:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.verdict(self.stack[i]) == PASS:
                tree, landed = self.land(i)
                self.out.append(Advance(tree, landed))
                return True
        return False

    def land(self, i: int) -> tuple[str, tuple[Member, ...]]:
        landed = tuple(m for c in self.stack[:i + 1] for m in c.members)
        tree = self.stack[i].tree
        assert tree is not None
        del self.stack[:i + 1]
        self.main = tree
        for m in landed:
            self.queue.pop(m.pr, None)
        self.prune()
        self.unset_aside()
        return tree, landed

    def fail(self) -> bool:
        for i, c in enumerate(self.stack):
            if self.verdict(c) == FAIL:
                above = i + 1 < len(self.stack)
                if above:
                    self.cancel_from(i + 1, BELOW_FAILED)
                return self.attribute(i) or above
        return False

    def attribute(self, i: int) -> bool:
        c = self.stack[i]
        assert c.tree is not None and c.tasks is not None
        culprits: dict[int, tuple[Member, str, str]] = {}
        local = True
        for task in sorted(c.tasks):
            o = self.results.get((c.tree, task))
            if o is None or o.status != FAIL:
                continue
            suspects, j = list(c.members), i - 1
            while j >= 0:
                below = self.stack[j]
                b = self.results.get((below.tree, task)) if below.tasks is not None and task in below.tasks else None
                if b is not None and b.status == PASS:
                    break
                suspects[:0] = below.members
                j -= 1
            local = local and j == i - 1
            named = [m for m in suspects if touches(m.paths, o.inputs)]
            pick = (named[0], INPUTS) if len(named) == 1 else (suspects[0], ALONE) if len(suspects) == 1 else None
            if pick is not None and pick[0].pr not in culprits:
                culprits[pick[0].pr] = (pick[0], task, pick[1])
        if culprits:
            low = min(k for k, cand in enumerate(self.stack) if any(m.pr in culprits for m in cand.members))
            self.cancel_from(low, EJECTED)
            for pr in sorted(culprits):
                m, task, why = culprits[pr]
                self.queue.pop(pr, None)
                self.ejected[pr] = Eject(pr, m.head, c.tree, task, why)
                self.out.append(self.ejected[pr])
            return True
        if not local:
            return False
        h = len(c.members) // 2
        self.cancel_from(i, BISECTED)
        self.groups[:0] = [c.members[:h], c.members[h:]]
        return True

    def form(self) -> bool:
        if len(self.stack) >= self.policy.depth:
            return False
        top = self.stack[-1] if self.stack else None
        if top is not None and (top.tree is None or self.verdict(top) == FAIL):
            return False
        base = top.tree if top is not None else self.main
        assert base is not None
        flight = {m.pr for c in self.stack for m in c.members}

        def batchable(pr: int, head: str | None = None) -> bool:
            q = self.queue.get(pr)
            return (q is not None and q.reviewed and not q.aside and pr not in flight
                    and (head is None or q.head == head))

        while self.groups:
            group = self.groups.pop(0)
            members = tuple(m for m in group if batchable(m.pr, m.head))
            if members:
                return self.build(base, members, bisect=True)
        ready = [q for q in sorted(self.queue.values(), key=lambda q: q.seq) if batchable(q.pr)][:self.policy.batch]
        if not ready:
            return False
        return self.build(base, tuple(Member(q.pr, q.head, q.paths) for q in ready), bisect=False)

    # helpers -----------------------------------------------------------------------------------------------------------

    def build(self, base: str, members: tuple[Member, ...], bisect: bool) -> bool:
        self.stack.append(Candidate(self.cid, base, members, bisect))
        self.out.append(Build(self.cid, base, members, bisect))
        self.cid += 1
        return True

    def cancel_from(self, i: int, why: str) -> None:
        for c in self.stack[i:]:
            self.out.append(Cancel(c.cid, c.tree, why))
        del self.stack[i:]
        self.prune()

    def drop(self, pr: int, why: str) -> None:
        """Take `pr` out of flight: cancel the candidate holding it and every one above, and out of any bisection group."""
        for i, c in enumerate(self.stack):
            if any(m.pr == pr for m in c.members):
                self.cancel_from(i, why)
                break
        self.groups = [g for g in (tuple(m for m in grp if m.pr != pr) for grp in self.groups) if g]

    def prune(self) -> None:
        trees = {c.tree for c in self.stack}
        self.results = {k: v for k, v in self.results.items() if k[0] in trees}

    def unset_aside(self) -> None:
        for pr, q in list(self.queue.items()):
            if q.aside:
                self.queue[pr] = replace(q, aside=False)


_HANDLERS = {Ready: _Tx.ready, Withdrawn: _Tx.withdrawn, Built: _Tx.built, Conflict: _Tx.conflict, Listed: _Tx.listed,
             Result: _Tx.result, MainMoved: _Tx.main_moved}
