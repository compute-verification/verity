"""The schedule check: each pass's plan is a fixed function of the schedule the prover publishes, which `audit.ledger`
requires before it credits an appended chunk (`audit.AuditRun.scheduled`).  Of record in its Lean,
`lean/RemoteAudit/Schedule.lean`; this is held to it by the difftest (`lean/tests/test_schedule.py`, refusals compared
word for word).

Why.  The stacked guarantee (`Accounting.Guarantees.DeclaredCircuitRequirement`) lets a pass's layout follow the codec
and the append coins through an `a`-bit summary (`Accounting.Assumptions.SteeredLayout`) and charges `2^a` for it.
Every layout satisfies that premise at `a` = the coins' own bits, so `a` means something only once the layout is a fixed
public function of something whose bits are bounded whatever the coins (`Accounting.ScheduleDeterministic`).  Here that
is the schedule: the verifier computes each pass's plan from it, refuses any other, and refuses a schedule over the
cap `a` on its bits the prover committed before the session's first coin (`audit.Params.schedule_cap`).  The bits of
the one schedule the prover publishes after the coins bound nothing: a prover whose schedule is 200 bits on most coins
and 300 where it steers takes up to 2^200 + 2^300 layouts over the coins, so only `a` = 301 holds of it, whatever
each session's own schedule's bits are.

The schedule (`Schedule`; JSON `{"registered", "appends", "steps"}`):

- `registered`: `[first, count, free]` runs of the registered store, in store order and disjoint, each freed at step
  `free`, so resident through it; every other registered chunk is resident throughout;
- `appends`: one list a session's append, in order, of its chunks in index order, each `null` (never resident:
  withdrawn, or its encode never landed), `[write, null]` (resident from step `write + 1` on) or `[write, free]` (resident
  from step `write + 1` through `free`, `free > write`), memory-map/v1's writes and frees (`memory_map.schedule`);
- `steps`: the passes' steps, strictly increasing, below 2^64.

A pass at step s reads the schedule's plan at s (`Schedule.plan`): the chunks resident then, as runs in store order,
adjacent runs joined, which is the plan `memory_map.Replay` builds.  The appends are the schedule's, the first at N,
each at the next fresh index (`Schedule.append_runs`).  `bits` is the size of the schedule's encoding (`encode`: LEB128
integers, prefix-free, so fewer than 2^b schedules encode in b bits), at most the cap.

The JSON form (`run_json`) is the Lean's: `{"blocks": N, "cap": a, "schedule", "passes": [[step, [[first, count], ...]],
...], "appends": [[first, count], ...]}` to `{"bits": b}` or `{"refused": why}`."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

STEP_LIMIT = 1 << 64
"""Steps are below 2^64."""

Chunk = tuple[int, int | None] | None
"""An appended chunk: None never resident, or its write's step and its free's, if it is freed."""


class Refused(ValueError):
    """The session's passes or appends are not the schedule's, or the schedule is not one: the Lean's reason, word for
    word."""


class Malformed(ValueError):
    """The JSON form isn't the schedule check's inputs."""


def _resident(b: Chunk, s: int) -> bool:
    if b is None:
        return False
    w, f = b
    return w < s and (f is None or s <= f)


def _varint(n: int) -> bytes:
    out = bytearray()
    while n >= 128:
        out.append(n % 128 + 128)
        n //= 128
    out.append(n)
    return bytes(out)


def _block_code(b: Chunk) -> list[int]:
    if b is None:
        return [0]
    w, f = b
    return [1, w] if f is None else [f - w + 1, w]


@dataclass(frozen=True, slots=True)
class Schedule:
    """A session's schedule: the registered runs it frees, each append's chunks and the passes' steps."""

    registered: tuple[tuple[int, int, int], ...]
    appends: tuple[tuple[Chunk, ...], ...]
    steps: tuple[int, ...]

    def plan(self, blocks: int, s: int) -> tuple[tuple[int, int], ...]:
        """The plan at step `s` in a store of `blocks` registered chunks: the chunks resident then, as runs in store
        order, adjacent runs joined."""
        spans, cur = [], 0
        for f, c, e in self.registered:
            if e < s:
                if cur < f:
                    spans.append((cur, f))
                cur = f + c
        if cur < blocks:
            spans.append((cur, blocks))
        j = blocks
        for a in self.appends:
            for b in a:
                if _resident(b, s):
                    spans.append((j, j + 1))
                j += 1
        out: list[list[int]] = []
        for a, b in spans:
            if out and out[-1][1] == a:
                out[-1][1] = b
            else:
                out.append([a, b])
        return tuple((a, b - a) for a, b in out)

    def append_runs(self, blocks: int) -> tuple[tuple[int, int], ...]:
        """The appends as the verifier numbers them, `(first, count)`, the first at `blocks`."""
        out, first = [], blocks
        for a in self.appends:
            out.append((first, len(a)))
            first += len(a)
        return tuple(out)

    def numbers(self) -> list[int]:
        out = [len(self.registered)] + [x for r in self.registered for x in r] + [len(self.appends)]
        for a in self.appends:
            out += [len(a)] + [x for b in a for x in _block_code(b)]
        out.append(len(self.steps))
        prev = 0
        for s in self.steps:
            out.append(s - prev)
            prev = s
        return out

    def encode(self) -> bytes:
        """LEB128 integers, each list after its length: the registered runs, each append's chunks (a tag, 0 never
        resident, 1 never freed, else free − write + 1, then the write's step) and the steps (the first, then each
        one's rise)."""
        return b"".join(_varint(n) for n in self.numbers())

    @property
    def bits(self) -> int:
        return 8 * len(self.encode())

    def malformed(self, blocks: int) -> str | None:
        """Why the schedule is not one over `blocks` registered chunks: the first rule it breaks."""
        rs = self.registered
        if any(c == 0 or f + c > blocks or e >= STEP_LIMIT or (k > 0 and rs[k - 1][0] + rs[k - 1][1] > f)
               for k, (f, c, e) in enumerate(rs)):
            return ("the registered frees are runs of the registered store, in store order and disjoint, each at a step "
                    "below 2^64")
        for i, a in enumerate(self.appends):
            if not a:
                return f"append {i} has no block"
            for k, b in enumerate(a):
                if b is None:
                    continue
                w, f = b
                if f is None:
                    if w >= STEP_LIMIT:
                        return f"append {i}'s block {k} is written at a step past 2^64"
                elif f >= STEP_LIMIT:
                    return f"append {i}'s block {k} is freed at a step past 2^64"
                elif f <= w:
                    return f"append {i}'s block {k} is freed at step {f}, not after its write at step {w}"
        if any(s >= STEP_LIMIT or (k > 0 and self.steps[k - 1] >= s) for k, s in enumerate(self.steps)):
            return "the passes' steps strictly increase below 2^64"
        return None

    @classmethod
    def of_json(cls, j: object) -> Schedule:
        if not isinstance(j, Mapping) or not {"registered", "appends", "steps"} <= set(j):
            raise Malformed("a schedule is {registered, appends, steps}")
        return cls(tuple(_triple(r) for r in _list(j["registered"], "registered")),
                   tuple(tuple(_block(b) for b in _list(a, "an append")) for a in _list(j["appends"], "appends")),
                   tuple(_nat(s, "a step") for s in _list(j["steps"], "steps")))

    def to_json(self) -> dict:
        return {"appends": [[None if b is None else [b[0], b[1]] for b in a] for a in self.appends],
                "registered": [list(r) for r in self.registered], "steps": list(self.steps)}


def check(schedule: Schedule, blocks: int, cap: int, passes: Sequence[tuple[int, Sequence[tuple[int, int]]]],
          appends: Sequence[tuple[int, int]]) -> int:
    """The schedule's bits, at most `cap`, the cap on them the prover committed before the session's first coin
    (`audit.Params.schedule_cap`), when each of the session's passes `(step, runs)`, its plan's runs as the verifier
    received them, is at the schedule's step and its plan is the schedule's there, and its appends `(first, count)`
    are the schedule's; refused (`Refused`) as the Lean refuses otherwise."""
    if type(cap) is not int or cap < 0:
        raise ValueError(f"a schedule's cap is a nonnegative integer of bits, not {cap!r}")
    if (why := schedule.malformed(blocks)) is not None:
        raise Refused(why)
    if cap < schedule.bits:
        raise Refused(f"the schedule is {schedule.bits} bits, over the cap of {cap} committed before the coins")
    want = schedule.append_runs(blocks)
    if len(appends) != len(want):
        raise Refused(f"the session made {len(appends)} appends, the schedule {len(want)}")
    for i, (got, w) in enumerate(zip(appends, want)):
        if tuple(got) != w:
            raise Refused(f"append {i} is ({got[0]}, {got[1]}), the schedule's ({w[0]}, {w[1]})")
    if len(passes) != len(schedule.steps):
        raise Refused(f"the session has {len(passes)} passes, the schedule {len(schedule.steps)}")
    for k, ((s, runs), t) in enumerate(zip(passes, schedule.steps)):
        if s != t:
            raise Refused(f"pass {k} is at step {s}, the schedule's at {t}")
        if tuple(tuple(r) for r in runs) != schedule.plan(blocks, s):
            raise Refused(f"pass {k}'s plan is not the schedule's at step {s}")
    return schedule.bits


def _nat(x: object, what: str) -> int:
    if type(x) is not int or x < 0:
        raise Malformed(f"{what} is a nonnegative integer, not {x!r}")
    return x


def _list(x: object, what: str) -> list:
    if type(x) is not list:
        raise Malformed(f"{what} is a list, not {x!r}")
    return x


def _pair(x: object, what: str) -> tuple[int, int]:
    if type(x) is not list or len(x) != 2:
        raise Malformed(f"{what} is [first, count], not {x!r}")
    return _nat(x[0], what), _nat(x[1], what)


def _triple(x: object) -> tuple[int, int, int]:
    if type(x) is not list or len(x) != 3:
        raise Malformed(f"a registered run is [first, count, free], not {x!r}")
    return _nat(x[0], "a run's first"), _nat(x[1], "a run's count"), _nat(x[2], "a run's free")


def _block(x: object) -> Chunk:
    if x is None:
        return None
    if type(x) is not list or len(x) != 2:
        raise Malformed(f"an appended block is null or [write, free], not {x!r}")
    return _nat(x[0], "a write's step"), None if x[1] is None else _nat(x[1], "a free's step")


def of_json(j: object) -> tuple[Schedule, int, int, list[tuple[int, tuple[tuple[int, int], ...]]],
                                list[tuple[int, int]]]:
    """The JSON form's schedule, registered chunks, cap, passes and appends (`Malformed` if they aren't)."""
    if not isinstance(j, Mapping) or not {"blocks", "cap", "schedule", "passes", "appends"} <= set(j):
        raise Malformed("the schedule check's inputs are its blocks, cap, schedule, passes and appends")
    passes = []
    for p in _list(j["passes"], "passes"):
        if type(p) is not list or len(p) != 2:
            raise Malformed(f"a pass is [step, runs], not {p!r}")
        passes.append((_nat(p[0], "a pass's step"), tuple(_pair(r, "a run") for r in _list(p[1], "a pass's runs"))))
    return (Schedule.of_json(j["schedule"]), _nat(j["blocks"], "blocks"), _nat(j["cap"], "cap"), passes,
            [_pair(a, "an append") for a in _list(j["appends"], "appends")])


def run_json(j: Mapping) -> dict:
    """The schedule check of the JSON form's inputs: `{"bits": b}`, or `{"refused": why}`."""
    schedule, blocks, cap, passes, appends = of_json(j)
    try:
        return {"bits": check(schedule, blocks, cap, passes, appends)}
    except Refused as e:
        return {"refused": str(e)}
