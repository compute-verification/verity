"""A small snapshot queue with the verified queue's contract (its types' fields as `research.vqueue` names them), for the
snapshot driver's tests: one candidate at a time on `main`, holding the first `batch` ready PRs by rank.  A pass advances
and then drops the candidate, now on `main`, with its outcome; a fail drops the candidate and ejects the members whose
files meet a failing task's inputs, or every member when none does; infra is retried, and rung at `rings` tries."""
from __future__ import annotations

from dataclasses import dataclass

from verity_landq import model as lq

PENDING, PASS, FAIL, INFRA, LIST_FAILED = "pending", "pass", "fail", "infra", "list-failed"


@dataclass(frozen=True)
class Ready:
    number: int
    head: str
    files: frozenset[str]
    lean: bool = False
    rank: int = 0
    conflict: bool = False


@dataclass(frozen=True)
class Failed:
    task: str
    key: str
    inputs: frozenset[str]


@dataclass(frozen=True)
class Verdict:
    state: str = PENDING
    failed: tuple[Failed, ...] = ()
    infra: tuple[Failed, ...] = ()


@dataclass(frozen=True)
class Candidate:
    sha: str
    base: str
    members: tuple[int, ...]
    heads: tuple[str, ...]
    verdict: Verdict = Verdict()
    tries: int = 1


@dataclass(frozen=True)
class Outcome:
    members: tuple[int, ...]
    heads: tuple[str, ...]
    state: str
    suspects: tuple[int, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    main: str
    ready: tuple[Ready, ...]
    candidates: tuple[Candidate, ...] = ()
    history: tuple[Outcome, ...] = ()
    batch: int = 6
    depth: int = 2
    rings: int = 3


@dataclass(frozen=True)
class Build:
    members: tuple[int, ...]
    base: str
    priority: int


@dataclass(frozen=True)
class Advance:
    sha: str


@dataclass(frozen=True)
class Drop:
    sha: str
    why: str
    outcome: Outcome | None = None


@dataclass(frozen=True)
class Eject:
    number: int
    head: str
    why: str
    failed: tuple[Failed, ...] = ()


@dataclass(frozen=True)
class Retry:
    sha: str


@dataclass(frozen=True)
class Ring:
    sha: str
    why: str


def decide(s: Snapshot) -> list:
    if not s.candidates:
        taken = s.ready[:s.batch]
        return [Build(tuple(r.number for r in taken), s.main, 0)] if taken else []
    c = s.candidates[0]
    v = c.verdict
    if v.state == PASS:
        return [Advance(c.sha), Drop(c.sha, "on main", Outcome(c.members, c.heads, PASS))]
    if v.state == FAIL:
        files = {r.number: r.files for r in s.ready}
        named = [n for n in c.members if any(lq.touches(files.get(n, frozenset()), f.inputs) for f in v.failed)]
        culprits = named or list(c.members)
        return [Drop(c.sha, "fail", Outcome(c.members, c.heads, FAIL, tuple(culprits))),
                *(Eject(n, h, "fail", v.failed) for n, h in zip(c.members, c.heads) if n in culprits)]
    if v.state == INFRA:
        return [Retry(c.sha)] + ([Ring(c.sha, "infra")] if c.tries + 1 == s.rings else [])
    return []
