"""The verified queue: `main` moves only to a candidate tree with no new failure: each gating task passed or fails on its base.

A candidate is `main`'s tip, or the candidate below it, with one `Queue: #N` merge per member on top (built by `Queue._merge`,
as `next` is), pushed to `queue/<sha12>` and checked by graph2 as `ready:<sha>`.  Its verdict decides what happens:

    pass     `main` fast-forwards to it, so its members are on `main` and the candidate above it now sits on `main`
    fail     a failure its base has too (at its base's key, with its failing tests among the base's, or as a read outside its
             task's declared inputs that the base's task makes too, whatever files) is inherited and blames nobody: a candidate whose every failure is inherited passes once every other gating task has, so `main`'s
             reds only shrink, and each is rung once (`Red`).  The members whose changed files meet one of its own failures'
             inputs are the suspects: one suspect is ejected (its ready mark withdrawn, its owner told which tasks failed),
             several are bisected; none meeting them makes every member a suspect, and a lone member's failed keys are rerun
             once, afresh, before it is ejected.  It and the candidates above it are dropped and, in the same tick, built
             again in order without the suspects (`_restack`), which go into the batches after them.
             A fail that names no failed task is rung, and nothing else happens.
    infra    nothing is blamed and `main` stays: the check is enqueued again at every reading, and infra is rung from `rings`
             enqueues on (2 at most), in words that change only with the erring tasks, so the loop tells infra once for each
             set; a listing failure is handled alike.  With `tree_ejects`, infra errors graph2 saw on two or more nodes
             (`tree`) follow the tree: they are bisected like a fail, on the members whose files meet their inputs, else on
             every member, and eject a lone member.  Without it, the default, infra that persists until infra is rung is
             bisected the same way, ejecting nobody: a lone member whose check keeps erring is rung to its owner and set
             aside behind the other PRs while any waits or it isn't marked at the head it was built at (`infra` outcome;
             set aside again, nobody is rung), else retried.
             A launch that outran its wait or that the pod lost before any run registered (`Failed.launch`) points at the
             machines, not the members: the check is enqueued again and infra rung from `rings` on, and no member is
             bisected out or set aside for it, until `launch_tries` readings; then it is handled as any other infra.  A
             launcher that exited without a run ran the candidate's own check.py, so that is any other infra at once
    pending  wait; a check still pending `pending_s` after its first enqueue is enqueued again, and rung at twice that

A batch with a member that changes Lean, a lock or a spec (`Ready.lean`) has its lock records regenerated before its check, as
drive's lock run regenerated them once such a change landed (`Build.records`): it is staged off main's chain, its members
merged onto the chain's top, and its records run, drive's lock run on that tree, runs there (`Candidate.waits`).  Nothing is
built on a staged batch and it counts toward no `depth`, so the chain goes on filling with batches of PRs with no Lean change
while the run runs; only no second Lean batch is staged meanwhile, and no PR changing a Lean input (`Ready.lean_input`) is
taken until the staged batch is on the chain, so none built beside it changes the Lean inputs of the top it goes onto.  Once
the run passed and the chain has room, the batch goes onto the chain's top (`Commit`): as staged when that top is still its
base, else its members merged again onto it, and checked there with its records included, on a records commit of what the
run regenerated, whatever its records change of what a guarantee says, or as built when it regenerated nothing.  Records
regenerated on one tree go onto another only when the two trees' Lean inputs are the same (`vqueue_slim.lean_key`), which
are all the records depend on; else (a change below it left the chain, or `main` moved elsewhere) the batch is staged again
there, its records run first again.  A run that ended in infra is launched once more (`Regenerate`).  One that failed or
ended in infra twice refuses them: the staged batch is dropped and its Lean members are ejected, and infra is rung of
infra.  A records commit still held `pending_s` after it was first drops its staged batch, whose members are built again.
The tick that stages a batch builds the PRs with no Lean change beside it, and the tick that puts one on the chain builds on
it there (`Build.after`).

`decide` is the whole policy: a pure function from a `Snapshot` (what the refs, the labels and graph2's results say now) to
the `Action`s that bring the world closer to it.  The loop builds the snapshot, applies the actions and keeps nothing between
ticks but what the next snapshot reads (the refs, the labels, graph2's results, and the outcomes it recorded), so a restart
repeats nothing that matters: a candidate's ref is named by its sha and enqueueing a check twice queues it once.

Only the lowest candidate's verdict is read whole: one above it is checked on a base that may still fail, so its pass counts
once the candidates below it are on `main`, in the tick that advances to the one below it too: a tick advances through the
passed candidates at the chain's bottom in order, one `Advance` and drop each, up to the first that hasn't passed.  Its fail
counts at once, as the lowest's would, once its base's word on each of its own failures is known (`Failed.base_tests`), or,
for a failure whose inputs meet its own members and none below it, `main`'s (`Failed.main_tests`, `_said`); until then it waits, since its base may yet fail the same.  The lowest fail
on the chain is the one read, and nothing is built on a candidate that failed.  A fail is read as soon as a gating task has
failed, while others still run (not `Verdict.ended`), unless each of its failures is inherited.  Bisecting keeps no state of
its own: a suspect's `cap` (half the suspects of the newest failed outcome it was in) bounds the batches it goes into until
it reaches `main` or is ejected.

A candidate in flight lands as built: a member whose owner withdrew its mark, pushed a new head or closed it stays in it,
its new head waits for the next batch, and a fail blames it by its files as built without ejecting it.  Only a red or a
member others took from the queue (`Snapshot.pulled`: a hold, a withdrawal or an ejection by another) drops a candidate, with
every one above it, and the same tick builds each again without what the red suspects or without that member alone
(`_restack`): a candidate rebuilt for any cause leaves out a member not ready at the head it was built at.

Lean changes (`Ready.lean`) and changes with none go in batches of their own, so no Lean failure drops a PR with no Lean
change.  A Lean batch opens whenever none is staged, whether or not PRs with no Lean change wait (`_next`): the Lean PRs first
in rank, up to `lean`, so none at 0; or, while a pinned Lean PR (`Snapshot.pinned`) that a batch may take waits, the pinned
ones alone, up to `batch` and under no `lean` cap.  A pin orders a Lean batch or cuts it, and at `lean` 0 it is the only way
into one.  Staged, the batch takes no build from the PRs with no Lean change: the tick that stages it builds theirs beside it.
It keeps those changing a Lean input out for its whole records run, though, so they go first: no Lean batch is staged,
pinned or not, while one of them that a batch may take waits, whatever its rank.  So only a batch staged before such a PR
was ready keeps it out, and it is in a build before the next is staged.  With no records run (`Snapshot.records` off) a
Lean batch takes the tick's build and a place on the chain, so it is built only on `main` or on a top candidate holding no
Lean change, at most every second build; while it waits for such a top the tick's batch is of PRs with no Lean change, and
with none to take nothing is built (`Wait`), as while the chain holds a wide Lean change (`Snapshot.wide`) or, with a records
run, a lone PR set aside before that changes a Lean input, whose next erring check would stage again a batch staged on it.
"""
from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .store.tool import glob_re

PENDING, PASS, FAIL, INFRA, LIST_FAILED = "pending", "pass", "fail", "infra", "list-failed"
STATES = (PENDING, PASS, FAIL, INFRA, LIST_FAILED)


@dataclass(frozen=True)
class Ready:
    """A PR the queue may take: marked ready at ``head``, with a land verdict by a non-owner or under the low-risk rule."""
    number: int
    head: str
    files: frozenset[str]          # the paths its change touches, against its merge base with `main`
    lean: bool = False             # its Lean change compiles (a spec's or a Lean source's, not a lock's alone): it goes in a
                                   # batch of such members only (`_next`), at most `Snapshot.lean` of them, but for a batch
                                   # of pinned ones (`Snapshot.pinned`)
    rank: int = 0                  # the queue's order, lowest first (drive's priority file, then mark time)
    conflict: bool = False         # it doesn't merge onto `main`: the queue skips it until its head changes
    lean_input: bool = False       # a path of ``files`` is a Lean input (`vqueue_slim.lean_inputs`, what `lean_key` hashes),
                                   # Lean change or not: no batch takes it beside a staged batch (`_pool`), and with no Lean
                                   # change no Lean batch is staged while it waits (`_next`)
    move: bool = False             # a move PR, whose move the loop regenerates on what its batch merged below it: a batch
                                   # holds at most one, last, and nothing is built on a candidate holding one (`_moves_last`)
    scripts: bool = False          # it changes `tools/move/`'s scripts (`queue.MOVE_SCRIPTS`): no move is built on or beside it


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
    launch: bool = False           # infra: its check registered no run because its launch outran its wait or the pod lost
                                   # it, not because the launcher, which runs the candidate's own check.py, exited
    reads: frozenset[str] | None = None  # every path it can have read, paths and globs as `meets` reads them (its declared
                                         # inputs, narrowed by what its run traced): None when nothing bounds them
    named: frozenset[str] = frozenset()  # the repository paths and `art:` ids its records name: its failing tests' files,
                                         # its frames, what its report says; what it read only where `reads` holds them
    kind: str | None = None        # what its records say failed (`KINDS`), None when they don't say
    main_tests: frozenset[str] | None = None  # `main`'s failing tests of the same task id, of a candidate above another: none
                                              # when `main` passed it, None when unknown (`_said`)


#: what a failure's records can say failed (`Failed.kind`): the change's own work; the machine (its disk or memory full); a
#: kill (a signal, 128+N); a timeout; a flake (a test that failed and passed on one tree); the state of a store or a build
#: tree the run shares (its build's outputs changed under it, an unreachable store)
KINDS = ("own", "infra", "killed", "timeout", "flaky", "store-state")


@dataclass(frozen=True)
class Verdict:
    """graph2's results for the candidate's newest enqueue or rerun: pending after a `Retry` or a `Rerun` until it says
    something newer."""
    state: str = PENDING
    failed: tuple[Failed, ...] = ()
    infra: tuple[Failed, ...] = ()
    ended: bool = False            # fail: no gating task still runs: each passed, failed, erred on infra, or reads a failed task


@dataclass(frozen=True)
class RecordsRun:
    """A candidate's records run (`Build.records`, `Regenerate`): drive's lock run on its tree, before its check, which
    regenerates the lock records it finds stale.  Pending while it launches or runs, or with ``why`` while its passed run's
    records can't be read yet; pass once it passed (`Commit`); fail when it failed; infra when it registered no run or never
    ran, launched again while ``starts`` is under 2."""
    state: str = PENDING
    why: str = ""
    starts: int = 1                # how many times it was launched


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
    records: RecordsRun | None = None   # its records run, None while none was launched
    repairs: str | None = None     # the staged batch (`waits`) it was built for by `Commit`, which it replaces: with its records,
                                   # on main's chain, or staged again on another tree
    waits: bool = False            # staged off main's chain, its records run first (`Build.records`): its check never started, no
                                   # verdict of it is read and nothing is built on it
    held: float | None = None      # when its records commit was first held (`Commit`), in seconds since the epoch


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
    lean: int = 1                  # the most members of a Lean batch (`_next`); 0: only a pinned Lean PR opens one
    now: float | None = None       # when the snapshot was taken, in seconds since the epoch; None: nothing times out
    pending_s: float = 4 * 3600    # how long the lowest candidate is pending after its first enqueue before it is retried
    tree_ejects: bool = False      # infra errors marked `tree` are bisected and eject; off while graph2 marks `tree` on any
                                   # error that ran out of attempts on two or more nodes, as an outage's do
    launch_tries: int = 3          # readings of launch-only infra (`Failed.launch`) on one candidate before it is handled
                                   # as any other infra: a tree that keeps its own launch from registering is found then
    records: bool = False          # a Lean batch's records are regenerated before its check (`Build.records`); off, they aren't
    left: tuple[tuple[int, str, str], ...] = ()  # each other open PR into `main`, as (number, head, why the queue doesn't
                                   # take it now): what a candidate's drop for a member that left `ready` says
    wide: frozenset[int] = frozenset()  # the PRs whose Lean change is wide: one of them that `lean` counts (`Ready.lean`)
                                   # fills that cap alone, and a batch on a candidate holding one takes none `lean` counts
    pinned: frozenset[int] = frozenset()  # the PRs the pins file names: while a Lean PR of them that a batch may take waits,
                                   # a Lean batch holds pinned Lean PRs only, under no `lean` cap (`_next`)
    pulled: frozenset[int] = frozenset()  # the PRs of `left` taken from the queue by others than their owners, at their
                                   # heads or at the heads candidates in flight hold them at (a mark withdrawn by another,
                                   # as a red's eject is, a reviewer's hold, the freeze, or a PR holding the head of one):
                                   # a candidate holding one is dropped
    flying: tuple[Ready, ...] = ()  # each member of a candidate on the chain not ready at its head as built, at that head:
                                   # the files a fail's blame reads of it


@dataclass(frozen=True)
class Build:
    """Merge ``members`` in order onto ``base``, push the result to `queue/<sha12>` and enqueue `ready:<sha>` at ``priority``;
    with ``records`` (a member changes Lean, a lock or a spec), launch its records run instead: the batch is staged off main's
    chain (`Candidate.waits`) until it goes onto it (`Commit`).  A member that doesn't merge onto what is built so far is left
    out and its owner told, unless what it merges onto or `main` already holds its head: the candidate's members are what
    merged.  With ``after``, a staged batch this tick's `Commit` puts on ``base``, the build is on what that put there (the
    batch as built, its records commit, or its members merged again), and nothing is built when it put nothing there (it is
    staged again, its records commit is held, or no member merges): the next tick decides on what it left.  With
    ``replaces``, it rebuilds that candidate, which the same actions drop, without the members a red or a leave subtracts
    (`decide`): a later Build of them whose ``base`` is that candidate goes on what this one built, or, when no member
    merged, on what this one went on."""
    members: tuple[int, ...]
    base: str
    priority: int
    records: bool = False
    after: str | None = None
    replaces: str | None = None


@dataclass(frozen=True)
class Advance:
    """Fast-forward `main` to ``sha``: only while `main` is still the candidate's base."""
    sha: str


@dataclass(frozen=True)
class Drop:
    """Cancel `ready:<sha>` but where, in shadow, it runs to its verdict to be compared, delete `queue/<sha12>` and record
    ``outcome`` when it was decided (None when it only went stale)."""
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
    malformed, or that its records run's records can't be read or ended in infra twice; or tell the owner of ``pr``, its lone
    member, that its check keeps erring; or tell ``to`` (handles) that its records commit is held."""
    sha: str
    why: str
    pr: int | None = None
    to: tuple[str, ...] = ()


@dataclass(frozen=True)
class Red:
    """Tell ci and the owners of ``failed``'s tasks that ``base`` fails them too: `main` stays red on them until a fix lands,
    while the queue lands what fails nothing more.  Said with the outcome that records them, once for each."""
    sha: str
    base: str
    why: str
    failed: tuple[Failed, ...]


@dataclass(frozen=True)
class Regenerate:
    """Launch ``sha``'s records run (again): drive's lock run on its tree, of every Lean package with a lock."""
    sha: str


@dataclass(frozen=True)
class Commit:
    """Put staged batch ``sha``, whose records run passed, onto main's chain at ``base``, its top, and enqueue its check at
    ``priority``.  On ``base`` as staged, it is checked on ``sha`` as built when the run regenerated nothing, else on a records
    commit of what it regenerated on top of ``sha``, made as drive commits a lock run's.  On another ``base`` its members are
    merged again onto it, and that tree is checked as built or with those records committed on it only when its Lean inputs
    are ``sha``'s (`vqueue_slim.lean_key`); else it is staged again, its records run first.  Whatever is pushed `repairs`
    ``sha``, which the next tick drops.  A records commit that can't be made yet is held, and ``sha`` stays staged."""
    sha: str
    base: str
    priority: int


@dataclass(frozen=True)
class Wait:
    """Build nothing, for ``why``: Lean PRs wait for a chain holding no wide Lean change and no PR set aside before that
    changes a Lean input or, with no records run, for a top candidate holding no Lean change, and no PR with no Lean change
    is ready to take.  The loop logs each ``why`` once."""
    why: str


Action = Build | Advance | Drop | Eject | Retry | Rerun | Ring | Red | Regenerate | Commit | Wait


#: the prefix of a task's test for each file it read outside its declared inputs, a fault graph2 names by file
UNDECLARED = "undeclared:"
ON_MAIN = "on main"
BASE_FAILED = "its base failed"
BASE_DROPPED = "its base was dropped"


def decide(s: Snapshot) -> list[Action]:
    """The actions ``s`` calls for, in the order they are applied: the drops of the candidates off main's chain, what the
    lowest candidate's verdict calls for (a pass advances `main` through it and each passed candidate above it in turn, up to
    the first that hasn't passed), what the lowest fail above it calls for (as the lowest's fail would, once its base's
    word on each of its own failures is known, or `main`'s where only its own members meet the failure: `_said`; until then
    nothing, and nothing is built above it), what each staged batch's records run calls for, the rebuilds of what a red or a
    leave dropped (`_restack`), then the tick's builds on the chain's top: a Lean batch while none is staged (`_next`), and,
    while the chain is shorter than `depth`, one batch of PRs with no Lean change: one a tick, since the batch above it would
    sit on the sha the loop gives this one.  The Lean batch is staged with its records run first (`Build.records`), ahead of
    the batch built beside it; it takes no place in the chain, so it is built on the top of a full chain too, and none is
    staged while a PR changing a Lean input that a batch may take waits: it goes first (`_next`).  With no records run
    (`Snapshot.records` off) it takes a place in the chain and is the tick's one build.  While the chain holds a wide member
    (`Snapshot.wide`), no Lean batch opens: built on it, a Lean change would build again what the wide one built; nor while
    it holds a PR set aside before that changes a Lean input (`_next`); and while Lean PRs wait so with no PR with no Lean
    change to take, nothing is built and `Wait` says why.  A batch holds at most one move PR (`Ready.move`), last, and none
    beside or on a member that changes `tools/move/`'s scripts (`Ready.scripts`, `_moves_last`); nothing is built on a
    candidate holding a move, nor in the tick that advances to one, since the ready PRs were read on the `main` before it and
    may lack its move: no batch, no rebuild (`_restack`) and no `Commit`, and none after a `Commit` that puts one there.

    A candidate already on `main` (the loop advanced to it and stopped before its drop, or `main` was moved to it) is dropped
    `on main`, with a pass outcome when its verdict passed, and its members stay out of the next batch, as does a PR whose
    head a pass outcome holds: it is on `main`.  A staged batch (`Candidate.waits`) is off the chain and has no verdict to read:
    what its records run calls for (`_first`), and nothing is built on it.  Once its run passed it goes onto the chain's top
    (`Commit`) as soon as the chain has room, in a tick that rebuilds nothing, and the tick's builds go on it there
    (`Build.after`), as the next tick's would; one whose member isn't marked at the head it was staged at, or isn't taken
    there (its drop names why: `_untaken`), goes only onto its own base, as built, and is dropped, built again without that
    member, when that is no longer the top.  Beside a staged
    batch, one kept or the one this tick stages, the tick's other batch takes no Lean PR, so one is staged at a time, nor a
    PR changing a Lean input (`Ready.lean_input`), so its records carry onto the top it goes onto, nor a lone PR set aside
    (`_aside`), which goes behind it as behind any PR waiting.

    A red or a member set aside drops its candidate and every one above it, a refused records run drops its staged batch, and
    a member others took from the queue drops its own candidate and every one above it, or its staged batch (`_chain`); in
    the same tick each is built again (`_restack`), in order on what is left of the chain, without the suspects, the ejected,
    the members set aside and the members not ready at the heads they were built at: no other member leaves, and each keeps
    its place.  One with a Lean member is staged again off the chain (`Build.records`) while no other batch stays staged,
    else its members wait for a Lean batch as any Lean PR does, and the rest go on the chain's top."""
    ready = {r.number: r for r in s.ready}
    built = {**ready, **{r.number: r for r in s.flying}}
    acts: list[Action] = []
    chain, landed, staged, rest = _chain(s, ready, acts)
    whole = list(chain)
    main, ejected, out, hold = s.main, set(), set(), False
    c1 = chain[0] if chain else None
    if c1 is not None:
        reading = _reading(c1.verdict)
        if c1.verdict.state == FAIL and not c1.verdict.failed:
            acts.append(Ring(c1.sha, f"{c1.sha[:12]}'s verdict is fail but names no failed task: nothing is blamed or dropped "
                                     "until it names one"))
        elif reading == PASS:
            while chain and _reading(chain[0].verdict) == PASS:
                c, chain = chain[0], chain[1:]
                acts += _red(c, _outcomes(s, acts))
                acts += [Advance(c.sha), Drop(c.sha, ON_MAIN, _passed(c))]
                main, landed, hold = c.sha, landed | set(c.members), hold or _moving(c, built)
        elif reading == FAIL and (rerun := _rerun(c1, built)) is not None:
            acts.append(rerun)
        elif reading == FAIL or (reading == INFRA and s.tree_ejects and _tree(c1.verdict)):
            ejected, out = _failed(c1, ready, _outcomes(s, acts), acts, built=built)
            acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
            chain, rest = [], [*chain, *rest]
        elif reading == INFRA and (s.tree_ejects or c1.tries < min(s.rings, 2)
                                   or (_unlaunched(c1.verdict) and c1.tries < s.launch_tries)):
            acts.append(Retry(c1.sha))
            if c1.tries >= min(s.rings, 2):
                acts.append(Ring(c1.sha, _ring_why(c1)))
        elif reading == INFRA and len(c1.members) > 1:
            acts.append(Ring(c1.sha, _ring_why(c1)))
            ejected, out = _failed(c1, ready, _outcomes(s, acts), acts, tree=False, built=built)
            acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
            chain, rest = [], [*chain, *rest]
        elif reading == INFRA:
            outcomes, m = _outcomes(s, acts), c1.members[0]
            gone = m not in ready or ready[m].head != c1.heads[0]
            aside, told = gone or _waiting(s, ready, chain, landed, outcomes, staged), _told(c1, outcomes)
            if not aside:
                acts.append(Retry(c1.sha))
            if not (aside and told):
                acts.append(Ring(c1.sha, _ring_why(c1)))
            if not told:
                acts.append(Ring(c1.sha, _owner_why(c1), m))
            if aside:
                why = _untaken(c1, m, s) if gone else None
                acts.append(Drop(c1.sha, f"{_ring_why(c1)}, with only #{m} merged"
                                         + (f", which isn't taken at {c1.heads[0][:12]} now ({why})" if why else
                                            f", which isn't marked at {c1.heads[0][:12]} now" if gone else "")
                                         + ": it goes behind the PRs waiting, and nobody is ejected for it",
                                 Outcome(c1.members, c1.heads, INFRA, c1.members, c1.base)))
                acts += [Drop(c.sha, BASE_FAILED) for c in chain[1:]]
                out, chain, rest = set(c1.members), [], [*chain, *rest]
        elif reading == PENDING and s.now is not None and c1.enqueued is not None:
            if s.now - c1.enqueued > s.pending_s and c1.tries == 1:
                acts.append(Retry(c1.sha))
            if s.now - c1.enqueued > 2 * s.pending_s:
                acts.append(Ring(c1.sha, f"{c1.sha[:12]}'s check is still pending {2 * s.pending_s / 3600:g} h after its first "
                                         f"enqueue, {c1.tries} times enqueued"))
    held = False
    for k, c in enumerate(chain):
        if c is c1 or _reading(c.verdict) != FAIL or not c.verdict.failed:
            continue
        under = [built[m] for d in whole[:whole.index(c)] for m in d.members if m in built]
        if held := (said := _said(c, under, built)) is None:
            break
        if _reading(said.verdict) != FAIL:
            continue
        if (rerun := _rerun(said, built)) is not None:
            acts.append(rerun)
        else:
            got, sus = _failed(said, ready, _outcomes(s, acts), acts, red=c.base == s.main, built=built)
            ejected, out = ejected | got, out | sus
            acts += [Drop(d.sha, BASE_FAILED) for d in chain[k + 1:]]
            chain, rest = chain[:k], [*chain[k:], *rest]
        break
    kept, passed = [], []
    for c in staged:
        first, got = _first(s, c, ready)
        acts += first
        if got is None:
            kept.append(c)
            passed += [c] if c.records is not None and c.records.state == PASS and not first else []
        else:
            ejected, out = ejected | got[0], out | got[1]
            rest.append(c)
    n = len(acts)
    kept += _restack(s, ready, chain, rest, landed | ejected | out, main, acts, kept, hold)
    flying = {m for d in chain for m in d.members if m in built}
    if not held and not hold and not any(built[m].move for m in flying):
        base, after = chain[-1].sha if chain else main, None
        if passed and len(chain) < s.depth and len(acts) == n:
            c = passed[0]
            kept.remove(c)
            if c.base == base or (gone := _unmarked(c, ready)) is None:
                acts.append(Commit(c.sha, base, s.depth - len(chain)))
                chain, after = [*chain, c], c.sha
            else:
                cause = (f"#{gone} isn't taken at the head it was staged at ({why})" if (why := _untaken(c, gone, s))
                         else f"#{gone} isn't marked at the head it was staged at")
                acts.append(Drop(c.sha, f"{cause}, so it isn't merged again onto {base[:12]}, main's chain's top now: its "
                                        "other members are built again"))
                kept += _restack(s, ready, chain, [c], landed | ejected | out, main, acts, kept)
                base = chain[-1].sha if chain else main
            flying = {m for d in chain for m in d.members if m in built}
            if any(built[m].move for m in flying):
                return acts
        scripts = any(built[m].scripts for m in flying)
        outcomes, room, priority = _outcomes(s, acts), len(chain) < s.depth, max(s.depth - len(chain), 0)
        busy = landed | ejected | {m for d in (*chain, *kept) for m in d.members}
        pool = [r for r in _pool(ready, busy, bool(kept)) if not (r.move and scripts)]
        lean, why = _next(s, pool, outcomes, chain, built)
        lean = _moves_last(lean, ready)
        builds = [Build(lean, base, priority, s.records, after)] if lean and (s.records or room) else []
        if room and (s.records or not lean):
            beside = bool(kept or lean)
            plain = _batch([r for r in _pool(ready, busy | set(lean), beside) if not (r.move and scripts)], outcomes,
                           s.batch, 0, s.wide)
            plain = _moves_last(plain, ready)
            if plain and not (beside and _aside(plain, ready, outcomes)):
                builds.append(Build(plain, base, priority, after=after))
        acts += builds or ([Wait(why)] if why and room else [])
    return acts


def _pool(ready: dict[int, Ready], busy: set[int], staged: bool) -> list[Ready]:
    """The PRs a batch may take, in rank order: ready, merging onto `main`, and in no candidate on main's chain, on `main` or
    staged (``busy``); beside a staged batch (``staged``: one kept, or the one this tick stages), no Lean PR and no PR
    changing a Lean input (`Ready.lean_input`): one Lean batch is staged at a time, and no batch built beside it changes the
    Lean inputs (`vqueue_slim.lean_key`) of the chain's top it goes onto, so its records carry there rather than its records
    run starting again (`Commit`)."""
    return sorted((r for r in ready.values() if r.number not in busy and not r.conflict
                   and not (staged and (r.lean or r.lean_input))), key=lambda r: (r.rank, r.number))


def _aside(members: tuple[int, ...], ready: dict[int, Ready], outcomes: list[Outcome]) -> bool:
    """Whether ``members`` is a lone PR set aside at its head (an `infra` outcome), which goes behind the PRs waiting, a staged
    batch's among them."""
    newest = _newest(outcomes, {m: ready[m].head for m in members})
    return len(members) == 1 and (o := newest.get(members[0])) is not None and o.state == INFRA


def _unmarked(c: Candidate, ready: dict[int, Ready]) -> int | None:
    """The first member of ``c`` not marked ready at the head it was built at, None when none is."""
    return next((m for m, h in zip(c.members, c.heads) if m not in ready or ready[m].head != h), None)


def _untaken(c: Candidate, m: int, s: Snapshot) -> str | None:
    """Why the loop doesn't take member ``m`` of ``c`` at the head ``c`` holds it at, where `Snapshot.left` gives a cause
    there other than its mark (a move `main` landed that it lacks, `queue.lacked_moves`); None when its mark is withdrawn or
    missing there, or its head moved."""
    head = c.heads[c.members.index(m)]
    why = next((w for n, h, w in s.left if n == m and h == head), None)
    return None if why is None or why.startswith(("it isn't marked", "its ready mark")) else why


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


def _chain(s: Snapshot, ready: dict[int, Ready], acts: list[Action]
           ) -> tuple[list[Candidate], set[int], list[Candidate], list[Candidate]]:
    """Main's chain, lowest first, up to its first candidate with a member that left (`_stale`); the members of the
    candidates already on `main`; the staged batches kept (`Candidate.waits`), first in rank first; and the candidates
    dropped for `_restack` to build again: that first stale candidate with every one above it on the chain, then each
    staged batch with a member that left.  The drops of the other candidates go to ``acts``, each with every candidate above
    it.  A staged batch another candidate `repairs` is dropped for it, whatever their ranks; one is kept whatever its base,
    which the batch leaves once it goes onto the chain (`Commit`), while no member of it is in a candidate on the chain, on
    `main`, in a staged batch first in rank or taken from the queue by others."""
    by_sha: dict[str, Candidate] = {}
    for c in sorted(s.candidates, key=lambda c: c.sha):
        by_sha.setdefault(c.sha, c)
    above: dict[str, list[Candidate]] = {}
    for c in by_sha.values():
        above.setdefault(c.base, []).append(c)
    fixed = {c.repairs: c for c in sorted(by_sha.values(), key=lambda c: c.sha) if c.repairs in by_sha}
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
    for sha, by in fixed.items():
        if sha not in done:
            drop(by_sha[sha], f"its records were regenerated in {by.sha[:12]}, which is checked in its place" if not by.waits
                 else f"it is staged again in {by.sha[:12]}, on {by.base[:12]}, whose Lean inputs aren't those its records "
                      "were regenerated on: its records run first again")

    chain: list[Candidate] = []
    cut: list[Candidate] = []
    below: set[int] = set()
    untaken = {n: (head, why) for n, head, why in s.left}
    top = s.main
    while on := [c for c in above.get(top, ()) if c.sha not in done and not c.waits]:
        keep = min(on, key=lambda c: (_ranks(c, ready), c.sha))
        for c in on:
            if c is not keep:
                drop(c, f"a second candidate on {top[:12]}: {keep.sha[:12]}'s members come first in rank")
        done.add(keep.sha)
        if (why := _stale(keep, below, untaken, s.pulled)) or cut:
            acts.append(Drop(keep.sha, why or BASE_DROPPED))
            cut.append(keep)
        else:
            chain.append(keep)
        below.update(keep.members)
        top = keep.sha

    landed = {m for c in on_main for m in c.members}
    staged: list[Candidate] = []
    for c in sorted((c for c in by_sha.values() if c.sha not in done and c.waits), key=lambda c: (_ranks(c, ready), c.sha)):
        if why := _stale(c, below | landed | {m for k in staged for m in k.members}, untaken, s.pulled,
                         "a candidate on main's chain, on `main` or staged first"):
            drop(c, why)
            cut.append(c)
        else:
            done.add(c.sha)
            staged.append(c)
    rest = [c for c in by_sha.values() if c.sha not in done]
    left = {c.sha for c in rest}
    for c in rest:
        if c.base not in left:
            drop(c, f"not on main's chain: its base {c.base[:12]} is neither main nor a chain member")
    for c in rest:
        drop(c, "not on main's chain: its bases make a cycle")
    return chain, landed, staged, cut


def _ranks(c: Candidate, ready: dict[int, Ready]) -> list[tuple[int, int, int]]:
    return sorted((0, ready[m].rank, m) if m in ready else (1, 0, m) for m in c.members)


def _stale(c: Candidate, below: set[int], untaken: dict[int, tuple[str, str]], pulled: frozenset[int],
           where: str = "a candidate below it") -> str | None:
    """Why ``c`` is built again without a member, naming its first member's cause: a candidate below it (``where``) holding
    it too, or the queue no longer taking it for a cause its owner didn't give (``pulled``, `Snapshot.pulled`; why:
    ``untaken``, by number, is `Snapshot.left`); None while no member has one.  A member whose owner withdrew its mark,
    pushed a new head or closed it lands as built."""
    for m in c.members:
        if m in below:
            return f"#{m} is in {where} too"
        if m in pulled:
            return f"#{m} isn't taken now: {untaken[m][1] if m in untaken else 'taken from the queue'}"
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


def _unlaunched(v: Verdict) -> bool:
    """Whether ``v`` erred only in launches that outran their wait or that the pod lost (`Failed.launch`)."""
    return bool(v.infra) and all(f.launch for f in v.infra)


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


def _files(built: dict[int, Ready], m: int) -> frozenset[str]:
    return built[m].files if m in built else frozenset()


def _rerun(c1: Candidate, built: dict[int, Ready]) -> Rerun | None:
    """The rerun of a lone member's own failures that meet none of its files (``built``: the members as built), the first
    time: built alone again, its tree would have the same keys, which graph2 answers from its cache."""
    own = [f for f in c1.verdict.failed if not _inherited(f)]
    keys = tuple(dict.fromkeys(f.key for f in own if f.key))
    if len(c1.members) != 1 or c1.reruns or not keys or any(meets(_files(built, c1.members[0]), f.inputs) for f in own):
        return None
    return Rerun(c1.sha, keys)


def _first(s: Snapshot, c: Candidate, ready: dict[int, Ready]
           ) -> tuple[list[Action], tuple[set[int], set[int]] | None]:
    """What staged batch ``c`` calls for, with the PRs it ejects and those its rebuild leaves out (`_restack`): None while it
    stays staged.  Its run is launched (`Regenerate`; once more when it ended in infra); while it runs nothing happens, and
    infra is rung while its passed run's records can't be read; once it passed, nothing here: it waits for the chain's room
    (`Commit`, `decide`), and a records commit still held `pending_s` after it was first drops it, its members built again
    in the next batch.  A run that failed or ended in infra twice refuses its records: infra is rung of infra, its Lean
    members are ejected, and it is dropped, an outcome suspecting them.  A member no longer marked at the head it was built
    at (its owner withdrew the mark, pushed or closed it) is suspected whatever its change, and, as `_failed` reads one,
    isn't ejected."""
    r = c.records
    if r is None or (r.state == INFRA and r.starts < 2):
        return [Regenerate(c.sha)], None
    if r.state == PENDING:
        return [Ring(c.sha, f"{c.sha[:12]}'s records run: {r.why}")] if r.why else [], None
    if r.state == PASS and c.held is not None and s.now is not None and s.now - c.held > s.pending_s:
        return [Drop(c.sha, f"its records commit is still held {s.pending_s / 3600:g} h after it was first: its members are "
                            "built again")], (set(), set(c.members))
    if r.state == PASS:
        return [], None
    said = f"refused its records: {r.why}" if r.state == FAIL else f"ended in infra twice: {r.why}"
    heads = dict(zip(c.members, c.heads))
    marked = {m for m in c.members if m in ready and ready[m].head == heads[m]}
    lean = tuple(m for m in c.members if m not in marked or ready[m].lean) or c.members
    out = tuple(n for n in lean if n in marked)
    acts: list[Action] = [Ring(c.sha, f"{c.sha[:12]}'s records run {said}")] if r.state == INFRA else []
    acts += [Eject(n, heads[n], f"its records run {said}") for n in out]
    acts.append(Drop(c.sha, f"its records run {said}", Outcome(c.members, c.heads, FAIL, lean, c.base)))
    return acts, (set(out), set(lean))


def _failed(c1: Candidate, ready: dict[int, Ready], outcomes: list[Outcome], acts: list[Action], tree: bool = True,
            red: bool = True, built: dict[int, Ready] | None = None) -> tuple[set[int], set[int]]:
    """What a failed candidate calls for: the ring of the reds it shows (with ``red``: its base is `main`, which they say is
    red; else none is rung or recorded), the ejection its own failures make (or, with none, its infra errors that follow the
    tree; with ``tree`` False, every infra error of a check that keeps erring, which ejects nobody), if any, and its drop;
    the PRs it ejects, and its suspects.  Only those suspect anyone, by their files as built (``built``, else ``ready``); a
    suspect no longer marked at the head it was built at (its owner withdrew the mark, pushed or closed it) isn't ejected."""
    v, heads, built = c1.verdict, dict(zip(c1.members, c1.heads)), ready if built is None else built
    own = tuple(f for f in v.failed if not _inherited(f))
    by_inputs = [m for m in c1.members
                 if any(meets(_files(built, m), f.inputs) for f in own or (_tree(v) if tree else v.infra))]
    eject = None
    if own and len(by_inputs) == 1:
        n = by_inputs[0]
        hit = ", ".join(f.task for f in own if meets(_files(built, n), f.inputs))
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
    if eject is not None and ((r := ready.get(eject.number)) is None or r.head != eject.head):
        eject = None
    acts += _red(c1, outcomes) if red else []
    if eject is not None:
        acts.append(eject)
    why = (f"failed: {', '.join(f.task for f in v.failed)}" if own else _tree_why(c1) if tree else
           f"{_ring_why(c1)}, again after it was enqueued again: bisected on them, ejecting nobody")
    acts.append(Drop(c1.sha, why, Outcome(c1.members, c1.heads, FAIL, suspects, c1.base,
                                          tuple(f for f in v.failed if _inherited(f)) if red else ())))
    return set() if eject is None else {eject.number}, set(suspects)


def culprit(failed: Iterable[Failed], files: dict[int, frozenset[str]]) -> tuple[int | None, str]:
    """The one PR a red of `main`'s own check names among those landed since it last passed (``files``: each one's change as it
    landed), and why; None and why not unless every failure is a change's own (`Failed.kind` none or `own`) and bounded in what
    it read (`Failed.reads`), and that PR's change meets what each one read while no other's meets any: nothing is reverted on
    a guess."""
    if not (failed := list(failed)):
        return None, "it names no failure"
    if odd := [f for f in failed if f.kind not in (None, "own")]:
        return None, f"{', '.join(f'{f.task} failed as {f.kind}' for f in odd)}, not by a change"
    if unbounded := [f.task for f in failed if f.reads is None]:
        return None, f"nothing bounds what {', '.join(unbounded)} read"
    hit = sorted({n for n, fs in files.items() for f in failed if meets(fs, f.reads)})
    if len(hit) != 1:
        return None, (f"{', '.join(f'#{n}' for n in hit)} each meet what they read" if hit else
                      "no PR landed since meets what they read")
    if missed := [f.task for f in failed if not meets(files[hit[0]], f.reads)]:
        return None, f"#{hit[0]} meets what some read, not what {', '.join(missed)} read"
    return hit[0], f"#{hit[0]}'s change meets what {', '.join(f.task for f in failed)} read, and no other's landed since"


def _said(c: Candidate, under: list[Ready], built: dict[int, Ready]) -> Candidate | None:
    """``c``, a failed candidate above the lowest, with its base's word on each of its own failures, or None while one has
    none, so it waits for its base's check.  Where that check hasn't said (`Failed.base_tests`), `main`'s word stands for it
    (`Failed.main_tests`) when a member of ``c`` meets the failure's inputs and no member of the candidates below it does
    (``under``, as built): those leave them as `main` has them."""
    failed = []
    for f in c.verdict.failed:
        if not _inherited(f) and f.base_tests is None:
            if (f.main_tests is None or any(meets(r.files, f.inputs) for r in under)
                    or not any(meets(_files(built, m), f.inputs) for m in c.members)):
                return None
            f = replace(f, base_tests=f.main_tests)
        failed.append(f)
    return replace(c, verdict=replace(c.verdict, failed=tuple(failed)))


def _restack(s: Snapshot, ready: dict[int, Ready], chain: list[Candidate], rest: list[Candidate], out: set[int], main: str,
             acts: list[Action], staged: list[Candidate], hold: bool = False) -> list[Candidate]:
    """Build ``rest``, the candidates this tick dropped for a red, a refusal, a member set aside or a leave, lowest first,
    again in order on ``chain``'s top (``main`` when it is empty), each `Build.replaces` its own, without the members of
    ``out`` (on `main`, ejected, suspected or set aside), those not ready at the heads they were built at, and those a
    candidate below or a staged batch kept (``staged``) holds; one with none left is not built.  ``chain`` gains each, as a
    candidate at its old sha, which a Build on it names, but for one with a Lean member: that one is staged off the chain,
    its records run first (`Build.records`), and those after it go on the chain's top as before; while another batch stays
    staged it is not built, and its members wait for a Lean batch, as one is staged at a time.  None is built in a tick that
    advanced to a candidate holding a move (``hold``) nor on a top holding one (`decide`): their members wait in the pool
    until it is on `main`.  What it stages."""
    base, below = chain[-1].sha if chain else main, {m for c in [*chain, *staged] for m in c.members}
    built = {**ready, **{r.number: r for r in s.flying}}
    again: list[Candidate] = []
    for c in rest:
        if hold or (chain and _moving(chain[-1], built)):
            break
        keep = [(m, h) for m, h in zip(c.members, c.heads) if m not in out and m not in below and m in ready
                and ready[m].head == h]
        if not keep:
            continue
        members, heads = tuple(m for m, _ in keep), tuple(h for _, h in keep)
        records = s.records and any(ready[m].lean for m in members)
        if records and (staged or again):
            continue
        acts.append(Build(members, base, max(s.depth - len(chain), 0), records, replaces=c.sha))
        below.update(members)
        if records:
            again.append(Candidate(c.sha, base, members, heads, waits=True))
            continue
        chain.append(Candidate(c.sha, base, members, heads))
        base = c.sha
    return again


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


def _waiting(s: Snapshot, ready: dict[int, Ready], chain: list[Candidate], landed: set[int], outcomes: list[Outcome],
             staged: list[Candidate]) -> bool:
    """Whether another PR waits behind the lowest candidate: in a candidate above it, in a staged batch whose records run
    passed, or in a batch `main` alone would get, a Lean one or one of PRs with no Lean change."""
    if len(chain) > 1 or any(c.records is not None and c.records.state == PASS for c in staged):
        return True
    pool = _pool(ready, landed | {m for c in (chain[0], *staged) for m in c.members}, bool(staged))
    return bool(_next(s, pool, outcomes, [], ready)[0] or _batch(pool, outcomes, s.batch, 0, s.wide))


def _tree_why(c: Candidate) -> str:
    return f"{c.sha[:12]}'s check ended in infra errors on two or more nodes, which follow its tree: " + ", ".join(
        f.task for f in _tree(c.verdict))


def _wide(r: Ready, wide: frozenset[int]) -> bool:
    """Whether ``r`` is a wide member: ``wide`` names it and the `lean` cap counts it (`Ready.lean`)."""
    return r.lean and r.number in wide


def _next(s: Snapshot, pool: list[Ready], outcomes: list[Outcome], chain: list[Candidate],
          built: dict[int, Ready]) -> tuple[tuple[int, ...], str]:
    """The Lean batch to open on ``chain``'s top from ``pool`` (in rank order), and why it is none while its PRs wait for a top
    it may go on ('' otherwise; ``built``: the chain's members as built).  It holds Lean PRs only, whether or not a PR with
    no Lean change waits: while a pinned one (`Snapshot.pinned`) that a batch may take (`_takes`) waits, the pinned ones alone,
    up to `batch` and under no `lean` cap; else the first in rank, up to `lean`, so at 0 none but a pinned one set aside.  A
    lone one set aside opens a batch only in its turn, behind every PR of ``pool`` a batch may take and every one set aside
    longer, whatever their change (`_batch`; at `lean` 0, no Lean PR but a pinned one).  With a records run, none opens,
    pinned or not, while ``pool`` holds a PR with no Lean change that changes a Lean input (`Ready.lean_input`) and that a
    batch may take, whatever its rank: staged, the batch would keep that PR out of every build for its whole run, and each
    batch staged after it again, so the PR goes first, in the tick's batch or a later one.  None opens
    while the chain holds a wide Lean change; nor, with a records run, while it holds a lone PR set aside before at its
    head (`_told`) that changes a Lean input: when its check errs again it leaves the chain, and a batch staged on it is
    staged again; nor, with no records run (`Snapshot.records` off: it takes the tick's build and a place on the chain), on
    a top candidate holding a Lean change, so at most every second build is one."""
    newest = _newest(outcomes, {r.number: r.head for r in pool})
    leans = {r.number: r for r in pool if r.lean}
    pinned = [r for r in leans.values() if r.number in s.pinned]
    members = () if any(_takes(r, newest) for r in pinned) else _batch(list(leans.values()), outcomes, s.batch, s.lean, s.wide)
    members = members or _batch(pinned, outcomes, s.batch, s.batch, s.wide)
    if members and not _takes(leans[members[0]], newest) and members != _batch(
            [r for r in pool if not r.lean or s.lean or r in pinned], outcomes, s.batch, s.batch, s.wide):
        return (), ""
    if s.records and members and any(r.lean_input and not r.lean and _takes(r, newest) for r in pool):
        return (), ""
    wide = [m for c in chain for m in c.members if m in built and _wide(built[m], s.wide)]
    aside = [c.members[0] for c in chain if s.records and len(c.members) == 1 and _told(c, outcomes)
             and (r := built.get(c.members[0])) is not None and (r.lean or r.lean_input)]
    top = [m for m in chain[-1].members if m in built and built[m].lean] if chain and not s.records else []
    if not members or not (wide or aside or top):
        return members, ""
    held = (f"the chain holds #{wide[0]}'s wide Lean change" if wide else
            f"the chain holds #{aside[0]}, set aside before, which changes a Lean input" if aside else
            f"the newest candidate, {chain[-1].sha[:12]}, holds #{top[0]}'s Lean change")
    return (), (f"nothing built: with a Lean change, {', '.join(f'#{n}' for n in members)} wait{'s' * (len(members) == 1)} "
                f"while {held}, and no PR with no Lean change is ready to take")


def _moving(c: Candidate, built: dict[int, Ready]) -> bool:
    """Whether candidate ``c`` holds a move PR (`Ready.move`), on which nothing is built (`decide`)."""
    return any(m in built and built[m].move for m in c.members)


def _moves_last(members: tuple[int, ...], ready: dict[int, Ready]) -> tuple[int, ...]:
    """``members`` with their first move PR (`Ready.move`) last, the loop regenerating its move on what the rest merged, and
    any other move left for a later batch; with no move at all beside a member that changes `tools/move/`'s scripts
    (`Ready.scripts`), since a move and the restacks across it run `main`'s.  A candidate holding a move has none built on it
    (`decide`), so no PR lacking the move merges above it."""
    moves = [m for m in members if ready[m].move]
    rest = tuple(m for m in members if not ready[m].move)
    return rest if not moves or any(ready[m].scripts for m in rest) else (*rest, moves[0])


def _takes(r: Ready, newest: dict[int, Outcome]) -> bool:
    """Whether a batch may take ``r`` as any other PR: no pass or infra outcome holds it at its head (``newest``, `_newest`)."""
    return (o := newest.get(r.number)) is None or o.state not in (PASS, INFRA)


def _batch(pool: list[Ready], outcomes: list[Outcome], size: int, lean: int, wide: frozenset[int]) -> tuple[int, ...]:
    """The next batch from ``pool`` (in rank order): each PR whose cap allows it, while the batch is under its limit, with at
    most ``lean`` Lean PRs, and a wide one (`_wide`) only as its first Lean PR, which fills that cap.  A suspect's cap is
    half the suspects of the newest outcome holding it at its head, at least 1; a PR set aside (`infra`) comes after every
    other, alone, the longest set aside first."""
    newest = _newest(outcomes, {r.number: r.head for r in pool})
    at = {id(o): i for i, o in enumerate(outcomes)}
    batch: list[int] = []
    limit, leans = size, 0
    for r in sorted(pool, key=lambda r: at[id(o)] if (o := newest.get(r.number)) is not None and o.state == INFRA else -1):
        if len(batch) >= limit:
            break
        o, w = newest.get(r.number), _wide(r, wide)
        if (o is not None and o.state == PASS) or (r.lean and (leans >= lean or w and leans > 0)):
            continue
        if o is not None and o.state == INFRA and batch:
            break
        cap = (max(1, len(o.suspects) // 2) if o is not None and o.state == FAIL and r.number in o.suspects else
               1 if o is not None and o.state == INFRA else size)
        if len(batch) < cap:
            batch.append(r.number)
            limit, leans = min(limit, cap), lean if w else leans + r.lean
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
