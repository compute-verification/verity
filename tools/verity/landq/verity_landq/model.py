"""The vocabulary the simulator (`sim`) and a queue's driver (`snapshots.Driver`) share: the events the world sends the
queue, the decisions the queue sends back, the queue's batch size and depth, and how a change's paths touch a task's
declared inputs.  No queue lives here: the queue is a snapshot queue's `decide` (`research.vqueue`), which
`snapshots.Driver` runs on these events and turns into these decisions.  No I/O, and nothing but the standard library.

    from research import vqueue
    from verity_landq import model as lq, snapshots
    q = snapshots.Driver(vqueue)
    state, decisions = q.step(q.start("<main's tip>"), lq.Ready(7, "<head>", frozenset({"a/x.py"})), lq.Policy(batch=4))
    # decisions == (Build(cid=0, base="<main's tip>", members=(Member(7, "<head>", ...),)),)

A PR is ready at a head (`Ready`), reviewed there or not, until `Withdrawn`.  The queue tests *candidates*: queued PRs
merged onto `main`'s tip or onto another candidate's tree.  The world builds each candidate's merge (`Build`), answering
with its tree (`Built`) or the member that doesn't merge (`Conflict`), lists its gating tasks (`Listed`) and reports each
one's outcome on that tree (`Result`): `PASS`, `FAIL` with the task's declared inputs, or `INFRA` (a timeout, an OOM kill, a
lost node, a setup death), which is no verdict.  `MainMoved` is a push of `main` the queue didn't decide.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Union

PASS, FAIL, INFRA = "pass", "fail", "infra"
#: inputs that name the whole tree, and the characters that make an input a glob
ROOTS, GLOB = frozenset({".", "./"}), frozenset("*?[")


@dataclass(frozen=True)
class Policy:
    """How the queue batches: `batch` PRs a candidate at most (k) and `depth` candidates in flight at most (d)."""
    batch: int = 4
    depth: int = 2

    def __post_init__(self) -> None:
        for name in ("batch", "depth"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Policy.{name} must be a positive int, not {getattr(self, name)!r}")


@dataclass(frozen=True)
class Member:
    """A PR in a candidate: its number, the head merged and the paths its change touches."""
    pr: int
    head: str
    paths: frozenset[str]


# --- events --------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ready:
    """PR `pr` is ready at `head`, whose change touches `paths`, and `reviewed` there or not."""
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
    base's failing tests of the task (None when the base passes it), when the world knows them."""
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
    """`main` was pushed to `tree` by something other than the queue's `Advance`."""
    tree: str


Event = Union[Ready, Withdrawn, Built, Conflict, Listed, Result, MainMoved]


# --- decisions -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Build:
    """Merge `members`' heads, in order, onto `base` (`main`'s tip or a candidate's tree), answer `Built` or `Conflict`, and
    put the tree's check on the executor."""
    cid: int
    base: str
    members: tuple[Member, ...]


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


Decision = Union[Build, Cancel, Advance, Eject, Retry, Rerun, Start, Page]


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
