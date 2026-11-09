"""The verified queue: `main` moves only to a candidate tree whose every gating task passed.

A candidate is `main`'s tip, or the candidate below it, with one `Queue: #N` merge per member on top (built by `Queue._merge`,
as `next` is), pushed to `queue/<sha12>` and checked by graph2 as `ready:<sha>`.  Its verdict decides what happens:

    pass     `main` fast-forwards to it, so its members are on `main` and the candidate above it now sits on `main`
    fail     the members whose changed files meet a failing task's inputs are the suspects: one suspect is ejected (its ready
             mark withdrawn, its owner told which tasks failed), several are bisected; none meeting them makes every member a
             suspect.  The candidates above it are dropped, and the rest go into the next batch
    infra    nothing is blamed and `main` stays: the check is enqueued again, and @infra is rung once it has been `rings` times
    pending  wait

`decide` is the whole policy: a pure function from a `Snapshot` (what the refs, the labels and graph2's results say now) to
the `Action`s that bring the world closer to it.  The loop builds the snapshot, applies the actions and keeps nothing between
ticks but what the next snapshot reads (the refs, the labels, graph2's results, and the outcomes it recorded), so a restart
repeats nothing that matters: a candidate's ref is named by its sha and enqueueing a check twice queues it once.

Only the lowest candidate's verdict is read: one above it is checked on a base that may still fail, so its verdict counts once
the candidates below it are on `main`.  Bisecting keeps no state of its own: a suspect's `cap` (half the suspects of the
newest failed outcome it was in) bounds the batches it goes into until it reaches `main` or is ejected, and a member that
failed alone is checked alone once more, then ejected as a possible flake.
"""
from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .store.tool import glob_re

PENDING, PASS, FAIL, INFRA, LIST_FAILED = "pending", "pass", "fail", "infra", "list-failed"
STATES = (PENDING, PASS, FAIL, INFRA, LIST_FAILED)


@dataclass(frozen=True)
class Ready:
    """A PR the queue may take: marked ready at ``head``, with a land verdict by a non-owner or under the low-risk rule."""
    number: int
    head: str
    files: frozenset[str]          # the paths its change touches, against its merge base with `main`
    lean: bool = False             # it changes Lean, a lock or a spec: a batch holds at most one such member
    rank: int = 0                  # the queue's order, lowest first (drive's priority file, then mark time)
    conflict: bool = False         # it doesn't merge onto `main`: the queue skips it until its head changes


@dataclass(frozen=True)
class Failed:
    """A gating task that failed (or, in `Verdict.infra`, ran out of attempts on infra errors)."""
    task: str
    key: str
    inputs: frozenset[str]         # its declared inputs and those of every task it reads, transitively: paths and globs


@dataclass(frozen=True)
class Verdict:
    state: str = PENDING
    failed: tuple[Failed, ...] = ()
    infra: tuple[Failed, ...] = ()


@dataclass(frozen=True)
class Candidate:
    sha: str
    base: str                      # `main`'s tip when it was built, or the candidate below it
    members: tuple[int, ...]       # in merge order
    heads: tuple[str, ...]         # each member's head when it was merged
    verdict: Verdict = Verdict()
    tries: int = 1                 # how many times `ready:<sha>` was enqueued


@dataclass(frozen=True)
class Outcome:
    """A decided candidate, as the loop recorded it from a `Drop`: what bisecting and the flake check read."""
    members: tuple[int, ...]
    heads: tuple[str, ...]
    state: str                     # pass | fail
    suspects: tuple[int, ...] = () # fail: the members it suspects; () for a pass


@dataclass(frozen=True)
class Snapshot:
    main: str
    ready: tuple[Ready, ...]
    candidates: tuple[Candidate, ...] = ()
    history: tuple[Outcome, ...] = ()   # oldest first
    batch: int = 6                 # the most members a candidate takes
    depth: int = 2                 # the most candidates checked at once, each on the one below
    rings: int = 3                 # infra verdicts on one candidate before @infra is rung


@dataclass(frozen=True)
class Build:
    """Merge ``members`` in order onto ``base``, push the result to `queue/<sha12>` and enqueue `ready:<sha>` at ``priority``.
    A member that doesn't merge onto what is built so far is left out and its owner told: the candidate's members are what
    merged."""
    members: tuple[int, ...]
    base: str
    priority: int


@dataclass(frozen=True)
class Advance:
    """Fast-forward `main` to ``sha``: only while `main` is still the candidate's base."""
    sha: str


@dataclass(frozen=True)
class Drop:
    """Cancel `ready:<sha>`, delete `queue/<sha12>` and record ``outcome`` when it was decided (None when it only went stale)."""
    sha: str
    why: str
    outcome: Outcome | None = None


@dataclass(frozen=True)
class Eject:
    """Withdraw #``number``'s ready mark at ``head`` and tell its owner ``why`` with the tasks that failed."""
    number: int
    head: str
    why: str
    failed: tuple[Failed, ...] = ()


@dataclass(frozen=True)
class Retry:
    """Enqueue `ready:<sha>` again: graph2 reruns only what ended in infra errors, or lists a check whose listing failed."""
    sha: str


@dataclass(frozen=True)
class Ring:
    """Tell @infra that ``sha``'s check keeps ending in infra errors."""
    sha: str
    why: str


Action = Build | Advance | Drop | Eject | Retry | Ring


ON_MAIN = "on main"
BASE_FAILED = "its base failed"
BASE_DROPPED = "its base was dropped"


def decide(s: Snapshot) -> list[Action]:
    """The actions ``s`` calls for, in the order they are applied: the drops of the candidates off main's chain, what the
    lowest candidate's verdict calls for, then, while the chain is shorter than `depth`, one batch built on its top: one a
    tick, since the batch above it would sit on the sha the loop gives this one.

    A candidate already on `main` (the loop advanced to it and stopped before its drop, or `main` was moved to it) is dropped
    `on main`, with a pass outcome when its verdict passed, and its members stay out of the next batch, as does a PR whose
    head a pass outcome holds: it is on `main`."""
    ready = {r.number: r for r in s.ready}
    acts: list[Action] = []
    chain, landed = _chain(s, ready, acts)
    main, ejected = s.main, set()
    if chain:
        c1 = chain[0]
        if c1.verdict.state == PASS:
            acts += [Advance(c1.sha), Drop(c1.sha, ON_MAIN, Outcome(c1.members, c1.heads, PASS))]
            main, landed, chain = c1.sha, landed | set(c1.members), chain[1:]
        elif c1.verdict.state == FAIL:
            ejected = _failed(c1, ready, s.history, acts)
            acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
            chain = []
        elif c1.verdict.state in (INFRA, LIST_FAILED):
            acts.append(Retry(c1.sha))
            if c1.tries >= s.rings:
                acts.append(Ring(c1.sha, _ring_why(c1)))
    if len(chain) < s.depth:
        busy = landed | ejected | {m for c in chain for m in c.members}
        pool = sorted((r for r in ready.values() if r.number not in busy and not r.conflict), key=lambda r: (r.rank, r.number))
        outcomes = [*s.history, *(a.outcome for a in acts if isinstance(a, Drop) and a.outcome is not None)]
        if members := _batch(pool, outcomes, s.batch):
            acts.append(Build(members, chain[-1].sha if chain else main, s.depth - len(chain)))
    return acts


def meets(files: Iterable[str], inputs: Iterable[str]) -> bool:
    """Whether a path of ``files`` meets one of ``inputs``.  An input with `*` or `?` is a glob (`**` crosses directories,
    `*` and `?` don't); any other is a repository path, which covers itself and, as a directory (with or without a trailing
    `/`), everything under it."""
    paths, globs = _inputs(inputs if isinstance(inputs, frozenset) else frozenset(inputs))
    for f in files:
        p = f
        while True:
            if p in paths:
                return True
            cut = p.rfind("/")
            if cut <= 0:
                break
            p = p[:cut]
        if any(g.match(f) for g in globs):
            return True
    return False


_glob = functools.lru_cache(maxsize=4096)(glob_re)


@functools.lru_cache(maxsize=64)
def _inputs(inputs: frozenset[str]) -> tuple[frozenset[str], tuple[re.Pattern[str], ...]]:
    paths, globs = set(), []
    for i in inputs:
        i = i.rstrip("/")
        if "*" in i or "?" in i:
            globs.append(_glob(i))
        elif i:
            paths.add(i)
    return frozenset(paths), tuple(globs)


def _chain(s: Snapshot, ready: dict[int, Ready], acts: list[Action]) -> tuple[list[Candidate], set[int]]:
    """Main's chain, lowest first, and the members of the candidates already on `main`; the drops of the other candidates go
    to ``acts``, each with every candidate above it."""
    by_sha: dict[str, Candidate] = {}
    for c in sorted(s.candidates, key=lambda c: c.sha):
        by_sha.setdefault(c.sha, c)
    above: dict[str, list[Candidate]] = {}
    for c in by_sha.values():
        above.setdefault(c.base, []).append(c)
    done: set[str] = set()

    def drop(c: Candidate, why: str) -> None:
        todo = [(c, why)]
        while todo:
            c, why = todo.pop(0)
            if c.sha not in done:
                done.add(c.sha)
                acts.append(Drop(c.sha, why))
                todo += [(d, BASE_DROPPED) for d in above.get(c.sha, ())]

    on_main: list[Candidate] = []
    c = by_sha.get(s.main)
    while c is not None and c.sha not in done:
        done.add(c.sha)
        on_main.append(c)
        c = by_sha.get(c.base)
    for c in reversed(on_main):
        acts.append(Drop(c.sha, ON_MAIN, Outcome(c.members, c.heads, PASS) if c.verdict.state == PASS else None))

    chain: list[Candidate] = []
    below: set[int] = set()
    top = s.main
    while on := [c for c in above.get(top, ()) if c.sha not in done]:
        keep = min(on, key=lambda c: (_ranks(c, ready), c.sha))
        for c in on:
            if c is not keep:
                drop(c, f"a second candidate on {top[:12]}: {keep.sha[:12]}'s members come first in rank")
        if why := _stale(keep, ready, below):
            drop(keep, why)
            break
        done.add(keep.sha)
        chain.append(keep)
        below.update(keep.members)
        top = keep.sha

    rest = [c for c in by_sha.values() if c.sha not in done]
    left = {c.sha for c in rest}
    for c in rest:
        if c.base not in left:
            drop(c, f"not on main's chain: its base {c.base[:12]} is neither main nor a chain member")
    for c in rest:
        drop(c, "not on main's chain: its bases make a cycle")
    return chain, {m for c in on_main for m in c.members}


def _ranks(c: Candidate, ready: dict[int, Ready]) -> list[tuple[int, int, int]]:
    return sorted((0, ready[m].rank, m) if m in ready else (1, 0, m) for m in c.members)


def _stale(c: Candidate, ready: dict[int, Ready], below: set[int]) -> str | None:
    for m, h in zip(c.members, c.heads):
        r = ready.get(m)
        if r is None:
            return f"#{m}'s ready mark was withdrawn"
        if r.head != h:
            return f"#{m}'s head moved from {h[:12]} to {r.head[:12]}"
        if m in below:
            return f"#{m} is in a candidate below it too"
    return None


def _failed(c1: Candidate, ready: dict[int, Ready], history: tuple[Outcome, ...], acts: list[Action]) -> set[int]:
    """The ejection, if any, and the drop a failed lowest candidate calls for; the PRs it ejects."""
    failed = c1.verdict.failed
    tasks = ", ".join(f.task for f in failed)
    heads = dict(zip(c1.members, c1.heads))
    by_inputs = [m for m in c1.members if any(meets(ready[m].files, f.inputs) for f in failed)]
    eject = None
    if len(by_inputs) == 1:
        n = by_inputs[0]
        hit = ", ".join(f.task for f in failed if meets(ready[n].files, f.inputs))
        eject = Eject(n, heads[n], f"{hit} failed, and its changed files meet their inputs, no other member's", failed)
    elif len(c1.members) == 1 and any(o.state == FAIL and o.members == c1.members and o.heads == c1.heads for o in history):
        n = c1.members[0]
        eject = Eject(n, heads[n], f"failed alone twice at {heads[n][:12]} ({tasks}), and no failing task's inputs meet its "
                                   "files: if a test flaked, delete it (a test that flakes is deleted, not rerun)", failed)
    if eject is not None:
        acts.append(eject)
    suspects = (eject.number,) if eject is not None else tuple(by_inputs) if by_inputs else c1.members
    acts.append(Drop(c1.sha, f"failed: {tasks}", Outcome(c1.members, c1.heads, FAIL, suspects)))
    return set() if eject is None else {eject.number}


def _ring_why(c: Candidate) -> str:
    if c.verdict.state == LIST_FAILED:
        return f"{c.sha[:12]}'s check failed to list, {c.tries} times enqueued"
    tasks = ", ".join(f.task for f in c.verdict.infra)
    return f"{c.sha[:12]}'s check ended in infra errors, {c.tries} times enqueued" + (f": {tasks}" if tasks else "")


def _batch(pool: list[Ready], outcomes: list[Outcome], size: int) -> tuple[int, ...]:
    """The next batch from ``pool`` (in rank order): each PR whose cap allows it, while the batch is under its limit, with at
    most one Lean PR.  A suspect's cap is half the suspects of the newest outcome holding it at its head, at least 1."""
    newest = _newest(outcomes, {r.number: r.head for r in pool})
    batch: list[int] = []
    limit, lean = size, False
    for r in pool:
        if len(batch) >= limit:
            break
        o = newest.get(r.number)
        if (o is not None and o.state == PASS) or (r.lean and lean):
            continue
        cap = max(1, len(o.suspects) // 2) if o is not None and o.state == FAIL and r.number in o.suspects else size
        if len(batch) < cap:
            batch.append(r.number)
            limit, lean = min(limit, cap), lean or r.lean
    return tuple(batch)


def _newest(outcomes: list[Outcome], want: dict[int, str]) -> dict[int, Outcome]:
    """For each PR of ``want`` ({number: head}), the newest of ``outcomes`` that holds it at that head."""
    found: dict[int, Outcome] = {}
    for o in reversed(outcomes):
        if len(found) == len(want):
            break
        for m, h in zip(o.members, o.heads):
            if m not in found and want.get(m) == h:
                found[m] = o
    return found
