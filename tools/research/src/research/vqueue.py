"""The verified queue: `main` moves only to a candidate tree with no new failure: each gating task passed or fails on its base.

A candidate is `main`'s tip, or the candidate below it, with one `Queue: #N` merge per member on top (built by `Queue._merge`,
as `next` is), pushed to `queue/<sha12>` and checked by graph2 as `ready:<sha>`.  Its verdict decides what happens:

    pass     `main` fast-forwards to it, so its members are on `main` and the candidate above it now sits on `main`
    fail     a failure its base has too (at its base's key, with its failing tests among the base's, or as a read outside its
             task's declared inputs that the base's task makes too, whatever files) is inherited and blames nobody: a candidate whose every failure is inherited passes once every other gating task has, so `main`'s
             reds only shrink, and each is rung once (`Red`).  The members whose changed files meet one of its own failures'
             inputs are the suspects: one suspect is ejected (its ready mark withdrawn, its owner told which tasks failed),
             several are bisected; none meeting them makes every member a suspect, and a lone member's failed keys are rerun
             once, afresh, before it is ejected.  The candidates above it are dropped, and the rest go into the next batch.
             A fail that names no failed task is rung, and nothing else happens
    infra    nothing is blamed and `main` stays: the check is enqueued again at every reading, and infra is rung from `rings`
             enqueues on (2 at most), in words that change only with the erring tasks, so the loop tells infra once for each
             set; a listing failure is handled alike.  With `tree_ejects`, infra errors graph2 saw on two or more nodes
             (`tree`) follow the tree: they are bisected like a fail, on the members whose files meet their inputs, else on
             every member, and eject a lone member.  Without it, the default, infra that persists until infra is rung is
             bisected the same way, ejecting nobody: a lone member whose check keeps erring is rung to its owner and set
             aside behind the other PRs while any waits (`infra` outcome; set aside again, nobody is rung), else retried
    pending  wait; a check still pending `pending_s` after its first enqueue is enqueued again, and rung at twice that

`decide` is the whole policy: a pure function from a `Snapshot` (what the refs, the labels and graph2's results say now) to
the `Action`s that bring the world closer to it.  The loop builds the snapshot, applies the actions and keeps nothing between
ticks but what the next snapshot reads (the refs, the labels, graph2's results, and the outcomes it recorded), so a restart
repeats nothing that matters: a candidate's ref is named by its sha and enqueueing a check twice queues it once.

Only the lowest candidate's verdict is read: one above it is checked on a base that may still fail, so its verdict counts once
the candidates below it are on `main`.  Bisecting keeps no state of its own: a suspect's `cap` (half the suspects of the
newest failed outcome it was in) bounds the batches it goes into until it reaches `main` or is ejected.
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
    lean: bool = False             # it changes Lean, a lock or a spec: a batch holds at most `Snapshot.lean` such members
    rank: int = 0                  # the queue's order, lowest first (drive's priority file, then mark time)
    conflict: bool = False         # it doesn't merge onto `main`: the queue skips it until its head changes


@dataclass(frozen=True)
class Failed:
    """A gating task that failed (or, in `Verdict.infra`, ran out of attempts on infra errors)."""
    task: str
    key: str
    inputs: frozenset[str]         # its declared inputs and those of every task it reads, transitively: paths and globs
    base_key: str | None = None    # the same task id's key in the base's listed graph, None when unknown
    tests: frozenset[str] = frozenset()       # its failing test ids, empty when unknown
    base_tests: frozenset[str] | None = None  # the base's failing tests of the same task id: None when it passed or is unknown
    tree: bool = False             # infra: graph2 saw it err on two or more nodes, so the tree breaks the machines


@dataclass(frozen=True)
class Verdict:
    """graph2's results for the candidate's newest enqueue or rerun: pending after a `Retry` or a `Rerun` until it says
    something newer."""
    state: str = PENDING
    failed: tuple[Failed, ...] = ()
    infra: tuple[Failed, ...] = ()
    ended: bool = False            # fail: no gating task still runs: each passed, failed, erred on infra, or reads a failed task


@dataclass(frozen=True)
class Candidate:
    sha: str
    base: str                      # `main`'s tip when it was built, or the candidate below it
    members: tuple[int, ...]       # in merge order
    heads: tuple[str, ...]         # each member's head when it was merged
    verdict: Verdict = Verdict()
    tries: int = 1                 # how many times `ready:<sha>` was enqueued
    reruns: int = 0                # how many times its failed keys were rerun (`Rerun`)
    enqueued: float | None = None  # when `ready:<sha>` was first enqueued, in seconds since the epoch; None when unknown


@dataclass(frozen=True)
class Outcome:
    """A decided candidate, as the loop recorded it from a `Drop`: what bisecting and a red's ring read."""
    members: tuple[int, ...]
    heads: tuple[str, ...]
    state: str                     # pass | fail | infra: a lone member whose check kept erring, set aside behind the rest
    suspects: tuple[int, ...] = () # fail: the members its own failures (or the infra errors it was bisected on) suspect
    base: str | None = None        # the candidate's base: of a fail, and of a pass with inherited failures
    inherited: tuple[Failed, ...] = ()  # its inherited failures, which its base fails too


@dataclass(frozen=True)
class Snapshot:
    main: str
    ready: tuple[Ready, ...]
    candidates: tuple[Candidate, ...] = ()
    history: tuple[Outcome, ...] = ()   # oldest first
    batch: int = 6                 # the most members a candidate takes
    depth: int = 2                 # the most candidates checked at once, each on the one below
    rings: int = 2                 # infra verdicts on one candidate before infra is rung; 2 at most
    lean: int = 1                  # the most members of a batch that change Lean, a lock or a spec
    now: float | None = None       # when the snapshot was taken, in seconds since the epoch; None: nothing times out
    pending_s: float = 4 * 3600    # how long the lowest candidate is pending after its first enqueue before it is retried
    tree_ejects: bool = False      # infra errors marked `tree` are bisected and eject; off while graph2 marks `tree` on any
                                   # error that ran out of attempts on two or more nodes, as an outage's do


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
    """Enqueue `ready:<sha>` again: graph2 reruns only what ended in infra errors, lists a check whose listing failed, or takes
    up again a check that stayed pending."""
    sha: str


@dataclass(frozen=True)
class Rerun:
    """Run ``keys`` on `ready:<sha>` again, afresh, their cached results set aside (the same tree built again has the same
    keys, which graph2 answers from its cache); the candidate counts it in `reruns`."""
    sha: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Ring:
    """Tell infra that ``sha``'s check keeps ending in infra errors or failing to list, or stays pending, or that its verdict is
    malformed; or tell the owner of ``pr``, its lone member, that its check keeps erring."""
    sha: str
    why: str
    pr: int | None = None


@dataclass(frozen=True)
class Red:
    """Tell ci and the owners of ``failed``'s tasks that ``base`` fails them too: `main` stays red on them until a fix lands,
    while the queue lands what fails nothing more.  Said with the outcome that records them, once for each."""
    sha: str
    base: str
    why: str
    failed: tuple[Failed, ...]


Action = Build | Advance | Drop | Eject | Retry | Rerun | Ring | Red


#: the prefix of a task's test for each file it read outside its declared inputs, a fault graph2 names by file
UNDECLARED = "undeclared:"
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
        reading = _reading(c1.verdict)
        if c1.verdict.state == FAIL and not c1.verdict.failed:
            acts.append(Ring(c1.sha, f"{c1.sha[:12]}'s verdict is fail but names no failed task: nothing is blamed or dropped "
                                     "until it names one"))
        elif reading == PASS:
            acts += _red(c1, _outcomes(s, acts))
            acts += [Advance(c1.sha), Drop(c1.sha, ON_MAIN, _passed(c1))]
            main, landed, chain = c1.sha, landed | set(c1.members), chain[1:]
        elif reading == FAIL and (rerun := _rerun(c1, ready)) is not None:
            acts.append(rerun)
        elif reading == FAIL or (reading == INFRA and s.tree_ejects and _tree(c1.verdict)):
            ejected = _failed(c1, ready, _outcomes(s, acts), acts)
            acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
            chain = []
        elif reading == INFRA and (s.tree_ejects or c1.tries < min(s.rings, 2)):
            acts.append(Retry(c1.sha))
            if c1.tries >= min(s.rings, 2):
                acts.append(Ring(c1.sha, _ring_why(c1)))
        elif reading == INFRA and len(c1.members) > 1:
            acts.append(Ring(c1.sha, _ring_why(c1)))
            ejected = _failed(c1, ready, _outcomes(s, acts), acts, tree=False)
            acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
            chain = []
        elif reading == INFRA:
            outcomes = _outcomes(s, acts)
            aside, told = _waiting(s, ready, chain, landed, outcomes), _told(c1, outcomes)
            if not aside:
                acts.append(Retry(c1.sha))
            if not (aside and told):
                acts.append(Ring(c1.sha, _ring_why(c1)))
            if not told:
                acts.append(Ring(c1.sha, _owner_why(c1), c1.members[0]))
            if aside:
                acts.append(Drop(c1.sha, f"{_ring_why(c1)}, with only #{c1.members[0]} merged: it goes behind the PRs waiting, "
                                         "and nobody is ejected for it", Outcome(c1.members, c1.heads, INFRA, c1.members, c1.base)))
                acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
                chain = []
        elif reading == PENDING and s.now is not None and c1.enqueued is not None:
            if s.now - c1.enqueued > s.pending_s and c1.tries == 1:
                acts.append(Retry(c1.sha))
            if s.now - c1.enqueued > 2 * s.pending_s:
                acts.append(Ring(c1.sha, f"{c1.sha[:12]}'s check is still pending {2 * s.pending_s / 3600:g} h after its first "
                                         f"enqueue, {c1.tries} times enqueued"))
    if len(chain) < s.depth:
        base = chain[-1].sha if chain else main
        busy = landed | ejected | {m for c in chain for m in c.members}
        pool = sorted((r for r in ready.values() if r.number not in busy and not r.conflict), key=lambda r: (r.rank, r.number))
        members = _batch(pool, _outcomes(s, acts), s.batch, s.lean)
        if members:
            acts.append(Build(members, base, s.depth - len(chain)))
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
        acts.append(Drop(c.sha, ON_MAIN, _passed(c) if _reading(c.verdict) == PASS else None))

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


def _inherited(f: Failed) -> bool:
    """Whether the base fails ``f`` too, so no member is to blame: at the same key (graph2 keys a task by everything it reads,
    so no member touched it, and a fail stands), with its failing tests among the base's failing tests of it, or, for a task
    failed by reading files outside its declared inputs (tests `UNDECLARED`), when the base fails that task so too, whatever
    files each names, and every other test it fails (what its run failed beside the reads) is among the base's: a task that
    faults stays faulted, so nothing stale passes, and the base stays red on it until its declarations land."""
    if f.base_key is not None and f.key == f.base_key:
        return True
    if not f.tests or f.base_tests is None:
        return False
    rest = {t for t in f.tests if not t.startswith(UNDECLARED)}
    return f.tests <= f.base_tests or (len(rest) < len(f.tests) and rest <= f.base_tests
                                       and any(t.startswith(UNDECLARED) for t in f.base_tests))


def _reading(v: Verdict) -> str:
    """What ``v`` calls for with its inherited failures set aside: a fail whose every failure is inherited is a pass once no
    other gating task runs, infra while one ended in infra errors, and pending until then.  A listing failure reads as infra."""
    if v.state == LIST_FAILED:
        return INFRA
    if v.state != FAIL or not v.failed or not all(_inherited(f) for f in v.failed):
        return v.state
    return INFRA if v.infra else PASS if v.ended else PENDING


def _tree(v: Verdict) -> tuple[Failed, ...]:
    """``v``'s infra errors that follow the tree: graph2 saw each on two or more nodes."""
    return tuple(f for f in v.infra if f.tree)


def _passed(c: Candidate) -> Outcome:
    if inherited := tuple(f for f in c.verdict.failed if _inherited(f)):
        return Outcome(c.members, c.heads, PASS, base=c.base, inherited=inherited)
    return Outcome(c.members, c.heads, PASS)


def _outcomes(s: Snapshot, acts: list[Action]) -> list[Outcome]:
    """The snapshot's history and the outcomes ``acts`` records, oldest first."""
    return [*s.history, *(a.outcome for a in acts if isinstance(a, Drop) and a.outcome is not None)]


def _red(c: Candidate, outcomes: list[Outcome]) -> list[Red]:
    """The ring of ``c``'s inherited failures no outcome records yet: none holds the task as inherited at its key, or at its
    base's key, which is the same red."""
    said = {(f.task, k) for o in outcomes for f in o.inherited for k in (f.key, f.base_key) if k is not None}
    new = tuple(f for f in c.verdict.failed if _inherited(f) and not {(f.task, f.key), (f.task, f.base_key)} & said)
    if not new:
        return []
    return [Red(c.sha, c.base, f"{c.base[:12]} fails {', '.join(f.task for f in new)} too, as {c.sha[:12]} does: `main` stays "
                               "red on them until a fix lands, while the queue lands what fails nothing more", new)]


def _rerun(c1: Candidate, ready: dict[int, Ready]) -> Rerun | None:
    """The rerun of a lone member's own failures that meet none of its files, the first time: built alone again, its tree
    would have the same keys, which graph2 answers from its cache."""
    own = [f for f in c1.verdict.failed if not _inherited(f)]
    keys = tuple(dict.fromkeys(f.key for f in own if f.key))
    if len(c1.members) != 1 or c1.reruns or not keys or any(meets(ready[c1.members[0]].files, f.inputs) for f in own):
        return None
    return Rerun(c1.sha, keys)


def _failed(c1: Candidate, ready: dict[int, Ready], outcomes: list[Outcome], acts: list[Action], tree: bool = True) -> set[int]:
    """What a failed lowest candidate calls for: the ring of the reds it shows, the ejection its own failures make (or, with
    none, its infra errors that follow the tree; with ``tree`` False, every infra error of a check that keeps erring, which
    ejects nobody), if any, and its drop; the PRs it ejects.  Only those suspect anyone."""
    v, heads = c1.verdict, dict(zip(c1.members, c1.heads))
    own = tuple(f for f in v.failed if not _inherited(f))
    by_inputs = [m for m in c1.members if any(meets(ready[m].files, f.inputs) for f in own or (_tree(v) if tree else v.infra))]
    eject = None
    if own and len(by_inputs) == 1:
        n = by_inputs[0]
        hit = ", ".join(f.task for f in own if meets(ready[n].files, f.inputs))
        eject = Eject(n, heads[n], f"{hit} failed, and its changed files meet their inputs, no other member's", own)
    elif own and len(c1.members) == 1:
        n = c1.members[0]
        again = " and again when rerun afresh" if c1.reruns else ""
        eject = Eject(n, heads[n], f"{', '.join(f.task for f in own)} failed alone at {heads[n][:12]}{again}, and no failing "
                                   "task's inputs meet its files: if a test flaked, delete it (a test that flakes is deleted, "
                                   "not rerun)", own)
    elif len(c1.members) == 1 and tree:
        n = c1.members[0]
        eject = Eject(n, heads[n], f"{_tree_why(c1)}, with only it merged on {c1.base[:12]}: a check that keeps erring on a "
                                   "change (an undeclared read, a hang, an untraced program) is the change's to fix", _tree(v))
    suspects = (eject.number,) if eject is not None else tuple(by_inputs) if by_inputs else c1.members
    acts += _red(c1, outcomes)
    if eject is not None:
        acts.append(eject)
    why = (f"failed: {', '.join(f.task for f in v.failed)}" if own else _tree_why(c1) if tree else
           f"{_ring_why(c1)}, again after it was enqueued again: bisected on them, ejecting nobody")
    acts.append(Drop(c1.sha, why, Outcome(c1.members, c1.heads, FAIL, suspects, c1.base,
                                          tuple(f for f in v.failed if _inherited(f)))))
    return set() if eject is None else {eject.number}


def _ring_why(c: Candidate) -> str:
    """Words that change with the tasks ``c``'s check erred on, not with its enqueues: the loop rings each why once."""
    if c.verdict.state == LIST_FAILED:
        return f"{c.sha[:12]}'s check failed to list"
    tasks = ", ".join(sorted({f.task for f in c.verdict.infra}))
    return f"{c.sha[:12]}'s check ended in infra errors" + (f": {tasks}" if tasks else "")


def _owner_why(c: Candidate) -> str:
    erred = "failed to list" if c.verdict.state == LIST_FAILED else "ended in infra errors: " + ", ".join(
        sorted({f.task for f in c.verdict.infra}))
    return (f"#{c.members[0]}'s check, with only it merged on {c.base[:12]}, {erred}, again after it was enqueued again.  "
            "Nobody is ejected for it: it waits behind the other PRs, enqueued again, until its check passes or a push "
            "changes it (a hang, an out-of-memory kill and an undeclared read are a change's to fix)")


def _told(c: Candidate, outcomes: list[Outcome]) -> bool:
    """Whether ``c``'s lone member was set aside at its head before: its owner and infra were rung then."""
    return any(o.state == INFRA and (c.members[0], c.heads[0]) in zip(o.members, o.heads) for o in outcomes)


def _waiting(s: Snapshot, ready: dict[int, Ready], chain: list[Candidate], landed: set[int], outcomes: list[Outcome]) -> bool:
    """Whether another PR waits behind the lowest candidate: in a candidate above it, or in the next batch."""
    busy = landed | set(chain[0].members)
    pool = sorted((r for r in ready.values() if r.number not in busy and not r.conflict), key=lambda r: (r.rank, r.number))
    return len(chain) > 1 or bool(_batch(pool, outcomes, s.batch, s.lean))


def _tree_why(c: Candidate) -> str:
    return f"{c.sha[:12]}'s check ended in infra errors on two or more nodes, which follow its tree: " + ", ".join(
        f.task for f in _tree(c.verdict))


def _batch(pool: list[Ready], outcomes: list[Outcome], size: int, lean: int) -> tuple[int, ...]:
    """The next batch from ``pool`` (in rank order): each PR whose cap allows it, while the batch is under its limit, with at
    most ``lean`` Lean PRs.  A suspect's cap is half the suspects of the newest outcome holding it at its head, at least 1;
    a PR set aside (`infra`) comes after every other, alone, the longest set aside first."""
    newest = _newest(outcomes, {r.number: r.head for r in pool})
    at = {id(o): i for i, o in enumerate(outcomes)}
    batch: list[int] = []
    limit, leans = size, 0
    for r in sorted(pool, key=lambda r: at[id(o)] if (o := newest.get(r.number)) is not None and o.state == INFRA else -1):
        if len(batch) >= limit:
            break
        o = newest.get(r.number)
        if (o is not None and o.state == PASS) or (r.lean and leans >= lean):
            continue
        if o is not None and o.state == INFRA and batch:
            break
        cap = (max(1, len(o.suspects) // 2) if o is not None and o.state == FAIL and r.number in o.suspects else
               1 if o is not None and o.state == INFRA else size)
        if len(batch) < cap:
            batch.append(r.number)
            limit, leans = min(limit, cap), leans + r.lean
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
