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

from dataclasses import dataclass

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


def decide(s: Snapshot) -> list[Action]:
    """The actions ``s`` calls for, in the order they are applied."""
    raise NotImplementedError("vqueue.decide")
