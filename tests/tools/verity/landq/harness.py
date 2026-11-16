"""A world for a queue's property tests: PRs that break tasks, candidate trees built from merges, and every event the
queue's decisions call for delivered in a random order, with infra outcomes and false fails mixed in.  The queue is a
`step` from a start `state`, as `snapshots.Driver` gives them for a snapshot queue's `decide`.  The checks read only what
the world delivered (its log) and what it knows (which PRs a tree holds), never the queue's state."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from verity_landq import model as lq

MAIN0 = "main0"
#: the log's mark of a `Rerun` decision, as (RERUN, tree, task): the results there before it are set aside
RERUN = "rerun"


@dataclass(frozen=True)
class Spec:
    """One PR: the paths its change touches, the tasks it fails (only tasks whose inputs it touches: tasks are hermetic), and
    what happens to it: reviewed only after its ready mark, withdrawn, or given a second head (the same change)."""
    pr: int
    paths: frozenset[str]
    breaks: frozenset[str] = frozenset()
    late_review: bool = False
    withdraw: bool = False
    rehead: bool = False


@dataclass(frozen=True)
class Scenario:
    tasks: dict[str, frozenset[str]]
    prs: tuple[Spec, ...]
    policy: lq.Policy
    order: int
    infra: float = 0.0
    flaky: int = 0
    interactions: tuple[tuple[int, int, str], ...] = ()


@dataclass
class Run:
    scenario: Scenario
    log: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    trees: dict = field(default_factory=dict)
    parent: dict = field(default_factory=dict)
    mains: list = field(default_factory=list)
    unsound: list = field(default_factory=list)
    state: Any = None
    steps: int = 0
    pending: int = 0


def random_scenario(rng: random.Random, *, bad: float = 0.25, infra: float = 0.0, flaky: int = 0, pairs: int = 0,
                    churn: bool = True) -> Scenario:
    paths = [f"p{i}/f.py" for i in range(rng.randint(2, 8))]
    tasks = {f"t{i}": frozenset(rng.sample(paths, rng.randint(1, min(3, len(paths))))) for i in range(rng.randint(1, 6))}
    prs = []
    for n in range(1, rng.randint(1, 12) + 1):
        mine = frozenset(rng.sample(paths, rng.randint(1, min(3, len(paths)))))
        touched = sorted(t for t, ins in tasks.items() if mine & ins)
        breaks = frozenset(rng.sample(touched, rng.randint(1, len(touched)))) if touched and rng.random() < bad else frozenset()
        prs.append(Spec(n, mine, breaks, late_review=churn and rng.random() < 0.2,
                        withdraw=churn and rng.random() < 0.08, rehead=churn and rng.random() < 0.1))
    inter = []
    for _ in range(pairs):
        if len(prs) >= 2:
            a, b = rng.sample(prs, 2)
            touched = sorted(t for t, ins in tasks.items() if (a.paths | b.paths) & ins)
            if touched:
                inter.append((a.pr, b.pr, rng.choice(touched)))
    policy = lq.Policy(batch=rng.randint(1, 4), depth=rng.randint(1, 3))
    return Scenario(tasks, tuple(prs), policy, rng.randrange(2**32), infra, flaky, tuple(inter))


def run(s: Scenario, step, state, limit: int = 20_000) -> Run:
    """Drive the queue, ``step`` from ``state`` (its start with `main` at `MAIN0`), through the scenario until nothing is
    left to deliver (or `limit` events)."""
    rng = random.Random(s.order)
    r = Run(s, trees={MAIN0: frozenset()}, parent={MAIN0: None}, mains=[MAIN0])
    flaky = [s.flaky]
    pending: list = []
    after: dict[str, list] = {}
    spec = {p.pr: p for p in s.prs}
    for p in s.prs:
        pending.append(lq.Ready(p.pr, f"h{p.pr}a", p.paths, reviewed=not p.late_review))
        if p.late_review:
            pending.append(("late", lq.Ready(p.pr, f"h{p.pr}a", p.paths, reviewed=True)))
        if p.rehead:
            pending.append(("late", lq.Ready(p.pr, f"h{p.pr}b", p.paths, reviewed=True)))
        if p.withdraw:
            pending.append(("late", lq.Withdrawn(p.pr)))
    held: list = [e for e in pending if isinstance(e, tuple)]
    pending = [e for e in pending if not isinstance(e, tuple)]

    def outcome(tree: str, task: str) -> str:
        holds = {pr for pr, _ in r.trees[tree]}
        if s.infra and rng.random() < s.infra:
            return lq.INFRA
        if any(task in spec[pr].breaks for pr in holds) or any(a in holds and b in holds and t == task
                                                               for a, b, t in s.interactions):
            return lq.FAIL
        if flaky[0] and rng.random() < 0.2:
            flaky[0] -= 1
            return lq.FAIL
        return lq.PASS

    def result(tree: str, task: str) -> lq.Result:
        return lq.Result(tree, task, outcome(tree, task), s.tasks[task])

    def on(d) -> None:
        r.decisions.append(d)
        if isinstance(d, lq.Build):
            tree = f"T{len(r.trees)}"
            r.trees[tree] = r.trees[d.base] | {(m.pr, m.head) for m in d.members}
            r.parent[tree] = d.base
            pending.append(lq.Built(d.cid, tree))
            after[tree] = [lq.Listed(tree, frozenset(s.tasks))] + [result(tree, t) for t in sorted(s.tasks)]
        elif isinstance(d, lq.Retry):
            pending.append(result(d.tree, d.task))
        elif isinstance(d, lq.Rerun):
            r.log.append((RERUN, d.tree, d.task))
            pending.append(result(d.tree, d.task))
        elif isinstance(d, lq.Advance):
            check_advance(r, d)
            r.mains.append(d.tree)

    while (pending or held) and r.steps < limit:
        if held and (not pending or rng.random() < 0.15):
            e = held.pop(rng.randrange(len(held)))[1]
        else:
            e = pending.pop(rng.randrange(len(pending)))
        if isinstance(e, lq.Built):
            pending.extend(after.pop(e.tree, []))
        if rng.random() < 0.05 and len(r.log) > 3:
            stale = rng.choice(r.log)
            if isinstance(stale, lq.Result):
                pending.append(stale)
        r.log.append(e)
        state, ds = step(state, e, s.policy)
        r.steps += 1
        for d in ds:
            on(d)
    r.state, r.pending = state, len(pending) + len(held)
    return r


def check_advance(r: Run, d: lq.Advance) -> None:
    """Soundness at one `Advance`, from the log alone: the tree is built on `main` as the world last moved it, it was listed,
    every listed task has a delivered pass on that tree and no delivered fail there since its last `Rerun` there (which
    sets the earlier results aside, a stale delivery of one included), and no PR it holds breaks anything."""
    s, tree, main = r.scenario, d.tree, r.mains[-1]
    why = []
    up, seen = tree, set()
    while up is not None and up != main and up not in seen:
        seen.add(up)
        up = r.parent.get(up)
    if up != main:
        why.append(f"{tree} is not built on main {main}")
    listed = [e.tasks for e in r.log if isinstance(e, lq.Listed) and e.tree == tree]
    if not listed:
        why.append(f"{tree} was never listed")
    for task in sorted(listed[0] if listed else ()):
        since = max((i for i, e in enumerate(r.log) if e == (RERUN, tree, task)), default=-1) + 1
        aside = {id(e) for e in r.log[:since]}  # a stale delivery repeats an earlier result's object
        got = [e.status for e in r.log[since:]
               if isinstance(e, lq.Result) and e.tree == tree and e.task == task and id(e) not in aside]
        if lq.PASS not in got or lq.FAIL in got:
            why.append(f"{task} on {tree}: {got}")
    holds = {pr for pr, _ in r.trees.get(tree, ())}
    spec = {p.pr: p for p in s.prs}
    bad = sorted(pr for pr in holds if spec[pr].breaks)
    bad += [f"{a}+{b}" for a, b, _ in s.interactions if a in holds and b in holds]
    if bad:
        why.append(f"{tree} holds breaking changes {bad}")
    landed = {(m.pr, m.head) for m in d.landed}
    if landed != r.trees.get(tree, frozenset()) - r.trees[main]:
        why.append(f"landed {sorted(landed)} is not what {tree} holds beyond main")
    if why:
        r.unsound.append((d, why))


def final_heads(r: Run) -> dict[int, str]:
    """Each PR's last head the world marked ready, of those still ready and reviewed at the end (not withdrawn)."""
    heads: dict[int, tuple[str, bool]] = {}
    for e in r.log:
        if isinstance(e, lq.Ready):
            heads[e.pr] = (e.head, e.reviewed)
        elif isinstance(e, lq.Withdrawn):
            heads.pop(e.pr, None)
    return {pr: h for pr, (h, ok) in heads.items() if ok}


def outcomes(r: Run) -> tuple[dict[int, str], dict[int, lq.Eject]]:
    """{PR: head landed} and {PR: its last ejection}, from the decisions."""
    landed, ejected = {}, {}
    for d in r.decisions:
        if isinstance(d, lq.Advance):
            for m in d.landed:
                landed[m.pr] = m.head
                ejected.pop(m.pr, None)
        elif isinstance(d, lq.Eject):
            ejected[d.pr] = d
    return landed, ejected
