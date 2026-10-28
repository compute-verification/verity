"""memory-map/v1 (`verity/memory-map/v1`, the program's I3): the per-step record of the KV blocks a serving node wrote
and freed beside its registered store, the remote bandwidth audit's input from serving, and `Replay`, which turns it
into the audit's appends, releases and plan changes.

~~~text
line 1       {"schema": "verity/memory-map/v1", "registration": hex(Registration.digest), "blocks": N,
              "block_bytes": 4096}
line 2, …    {"step": s, "freed": [slot, …], "written": [[slot, j, hex(d_j)], …], "withdrawn": [j, …]}, one per serving
              step that wrote, freed or withdrew a block, steps strictly increasing; "withdrawn" only on a step that
              withdraws one (a map without any reads as before it had the key)
slot         4,096 bytes of the node's memory, named by the registered block it held at registration (vLLM: its
              registry's region block index, a KV pool's pages registered as filler), or from N up for memory outside
              the registered store, below 2^63.  A slot below N starts live, holding registered block `slot`; one at
              or above N is live once written
j            the store index the writer gave the block: the store's next fresh one, N plus every block written or
              withdrawn before in the session, in record order
d_j          codec.digest(payload_j), the sha512/row/v2 digest of the block's 4,096-byte payload, 64 bytes: the device
              encodes the payload under its bound key (codec.binding(d_j, j, key, c_a)) at index j, once the verifier
              has sent its append's coin c_a
~~~

A write puts block j in its slot and releases the block the slot held, if it was live: the write overwrote it.  A
free releases the slot's block and leaves the slot not live, its memory out of the store, until a write.

Frees are structural.  A block is freed in the record of its values' last structural reader: the last step of the
schedule that runs a gate with one of them as an operand, a gather that may select one included, whichever request
the gate is in (a page requests share is freed once, after the last of them reads it).  That step is where Lean's
`live` ends (`Definitions.Accounting.Requirement`, whose `StoreMatches` keeps every block holding a credited value
resident through it), and it is `requirement.Requirement.last` over the circuit's census.  It reads the circuit, its
schedule and the registration, never a committed value.  A free at a value's last use (the last step whose gather
selected it) breaks `StoreMatches` for each candidate no later step selects, and one wrong unit upstream of the
selection moves it.  vLLM's `kv_free` is a free, though the pages' bytes stay as they were: a block left live past its
values' last structural reader would be credited past the circuit's use of it and past the lifetime cap, so the writer
records the free, and the store's filler plan leaves the released blocks out.  A free of a slot below N releases its
registered block, and the writer records one at its values' last structural reader, so no registered block stays
credited past it: weights only the first steps read are freed after the last of them, and a gathered table (an
embedding, the experts a router indexes) stays until the last step whose gather may select any of its rows, whichever
rows the run selected.  That every free but a re-key's (below), of a registered block or a written one, is at its
values' last structural reader is the memory-map unit's to prove (`verity.ml.circuits.requirement_units`,
`PROTOCOL.md` §8 "The requirement's link").  A
step's frees apply before its writes; a step frees a slot at most once, only a live one, and writes a slot at most
once.  A block is written once, complete and encoded (deferred residency):
a KV block the step's tokens are still filling is not in the record until it is complete, and a complete one is not
until the device has encoded it in place under its append's coin (below).  The write of a block completed at step s is
recorded at the step before the first step whose attention reads it encoded: s itself if the coin is back before step
s + 1's reads, later if not.  Until then the block is in no record and uncredited, like a partly written block (the
prover-side registry's dirty blocks, the block its slot held freed at the step a write first reached it), though
serving reads it as plaintext from its slot.

Withdrawn.  A block freed before its encode lands (its values' last structural reader at or before the step its write
would be recorded at) is withdrawn: no write, no free, no credit.  It was never resident encoded, and a block no step
reads while it is resident is in no record, so the line its write would be in, the one open when its encode lands,
lists its index in `withdrawn` instead.  A line's writes and withdrawals take the store's next fresh indices together,
each once and in index order, so a withdrawn index is never written.  Its slot holds nothing in the record from the
step that reached it until a later write, and the writer records no free of it.  The prover releases the index
(`Change.released` covers it) and names it withdrawn (`Change.withdrawn`, `Session.release(released, withdrawn)`), and
the verifier checks that no plan held it, so it was never written.

Lifetime.  No plan holds a written block L windows or more after the window its append joined (`audit.Params.lifetime`,
the windows in 30 s), so every block written leaves the credited store within L.  A block the writer keeps
longer (a prefix cache's) is re-keyed: its payload appended again, the same `d_j` at the store's next fresh index
under a fresh coin, so the device encodes it anew under a fresh bound key.  A re-key's free is the one exception to a
free at its values' last structural reader: the re-key's step frees the slot, releasing the old index, and the new
index is written in the slot when its encode lands (deferred residency, above), so the value is uncredited in
between, and withdrawn if its last structural reader comes first.  A block appended in window w is resident from w
or w + 1, so its re-key, recorded at a step before the first pass of window w + L, falls at age L − 1 at the latest.
At L = 1 that is the window it was appended in, before it may be resident, so a session that defers residency has
L ≥ 2 (`audit.Params.deferred`).

The record is the prover's: `d_j` is a hash of user data, a witness, never public (AGENTS.md's zero-knowledge rule).
The prover's audit side commits the digests of the blocks serving completes in an append tree (`audit.append_tree`,
under fresh salts) and sends its root and size (`Session.append`) once they are complete, and the released blocks'
indices as runs (`Session.release`); the verifier sees those and the plans, never the record.  The verifier replies
to each append with its first index and its coin, and the device encodes the append's blocks in place under the coin
once it arrives.  Residency waits for that encode, so the coin's round trip is off serving's critical path: the
writer records the blocks' writes, in their appends' order and at the indices they took, at the step before the first
step whose attention reads them encoded.  `Replay.apply` gives each step's `Change`: when every coin is back before
the next step's reads, its blocks are the step's own append (`Change.append_tree`); deferred, they are those of the
appends sent before whose writes the step records.  The appends are sent before the root of any pass whose plan
holds them, and that pass's seed binds them (`audit.chained`), so it is computed after they are sent.
That the session's appends, re-keys and frees are this record's, in its order and at its steps, is the memory-map
unit's to prove against the record's append chain and each pass's head (`audit.Record.appends`,
`WindowRecord.heads`, `RequirementLink.mapped`, `PROTOCOL.md` §8 "The memory map's choices"): frees, re-keys and
append order chosen after the seed would otherwise be advice.  So a step's appends are sent after the pass at that
step's root and before the next pass's, and a re-key's old index is below the stale bound of the pass after its step
(`WindowRecord.stale`, a value the unit recomputes inside its proof from the committed appends).

Passes and steps.  Each pass of the audit covers one step `s`, which it names with its root (`Session.receive(...,
step=s)`, public), and the steps of a session's passes strictly increase.  Its plan holds every complete block the
circuit reads at `s`, filler beside them allowed: here the blocks resident at `s`, registered or written at a step
before `s` and not freed or overwritten at one (`Store.resident` and `PlanMatches`, in the Lean module
`Definitions.Accounting.Requirement`).
A step's record takes effect after the step's reads: its writes are of blocks the step completed, or an earlier step
did and the device has since encoded, and its frees are of blocks whose values' last structural reader it is.
`Replay.take(s)` is the plan's change for the pass at `s` since the last pass (`Plan.changed`, the root's
`Session.receive(..., change=...)`), the blocks appended and released between them, of which a block appended and
released in between is in neither; it refuses a step at or before one already applied or taken.  That a pass's plan
holds every complete block the circuit reads at its step is the plan's proof block's to check, against this record and
the circuit's requirement, inside zero knowledge.

The schedule.  `schedule` turns the map into the schedule the prover publishes (each block's write and free, the
appends' extents and the passes' steps, `schedule.py`), from which the verifier computes every pass's plan and which it
requires before it credits an appended block (`audit.Session.scheduled`): the plans `take` gives are that schedule's.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from verity.ml.commitments import MerkleTree, domains

from . import audit
from . import schedule as _schedule
from .codec import BLOCK_BYTES
from .receipts import DIGEST_BYTES

SCHEMA = domains.MEMORY_MAP
SLOT_LIMIT = 1 << 63


class MemoryMapError(ValueError):
    """A memory map that breaks memory-map/v1, with what to fix in the writer."""


@dataclass(frozen=True, slots=True)
class Header:
    """The registered store a memory map is beside: its registration's digest and its N blocks of `BLOCK_BYTES`."""

    registration: bytes
    blocks: int

    def to_json(self) -> dict:
        return {"block_bytes": BLOCK_BYTES, "blocks": self.blocks, "registration": self.registration.hex(),
                "schema": SCHEMA}


@dataclass(frozen=True, slots=True)
class Written:
    """A block a step writes, one it or an earlier step completed and the device has encoded: its slot, the store
    index its writer gave it, and its payload's digest `d_j`."""

    slot: int
    index: int
    digest: bytes


@dataclass(frozen=True, slots=True)
class Step:
    """One serving step's record: the slots it freed, then the blocks it wrote, in index order, and the appended
    indices it withdrew, in index order (none by default)."""

    step: int
    freed: tuple[int, ...]
    written: tuple[Written, ...]
    withdrawn: tuple[int, ...] = ()

    def to_json(self) -> dict:
        out = {"freed": list(self.freed), "step": self.step,
               "written": [[w.slot, w.index, w.digest.hex()] for w in self.written]}
        if self.withdrawn:
            out["withdrawn"] = list(self.withdrawn)
        return out


def _int(value: object, what: str, limit: int) -> int:
    if type(value) is not int or not 0 <= value < limit:
        raise MemoryMapError(f"{what} is an integer from 0 below {limit}, not {value!r}")
    return value


def _digest(value: object, what: str) -> bytes:
    if type(value) is not str or len(value) != 2 * DIGEST_BYTES or value != value.lower():
        raise MemoryMapError(f"{what} is {DIGEST_BYTES} bytes in lowercase hex, not {value!r}")
    try:
        return bytes.fromhex(value)
    except ValueError:
        raise MemoryMapError(f"{what} is {DIGEST_BYTES} bytes in lowercase hex, not {value!r}") from None


def header(doc: object) -> Header:
    if type(doc) is not dict or set(doc) != {"block_bytes", "blocks", "registration", "schema"}:
        raise MemoryMapError("a memory map's first line is {schema, registration, blocks, block_bytes}")
    if doc["schema"] != SCHEMA:
        raise MemoryMapError(f"the schema is {SCHEMA!r}, not {doc['schema']!r}")
    if doc["block_bytes"] != BLOCK_BYTES:
        raise MemoryMapError(f"a block is {BLOCK_BYTES} bytes, not {doc['block_bytes']!r}")
    blocks = _int(doc["blocks"], "the registered store's blocks", 1 << 64)
    if blocks < 1:
        raise MemoryMapError("a registered store is at least one block")
    return Header(_digest(doc["registration"], "the registration's digest"), blocks)


def step(doc: object) -> Step:
    if type(doc) is not dict or not {"freed", "step", "written"} <= set(doc) <= {"freed", "step", "withdrawn", "written"}:
        raise MemoryMapError("a step's line is {step, freed, written}, with withdrawn if it withdraws a block")
    s = _int(doc["step"], "a step", audit.STEP_LIMIT)
    if type(doc["freed"]) is not list or type(doc["written"]) is not list or type(doc.get("withdrawn", [])) is not list:
        raise MemoryMapError(f"step {s}'s freed, written and withdrawn are lists")
    freed = tuple(_int(x, f"step {s}'s freed slot", SLOT_LIMIT) for x in doc["freed"])
    written = []
    for w in doc["written"]:
        if type(w) is not list or len(w) != 3:
            raise MemoryMapError(f"step {s}'s written block is [slot, index, digest], not {w!r}")
        written.append(Written(_int(w[0], f"step {s}'s written slot", SLOT_LIMIT),
                               _int(w[1], f"step {s}'s store index", 1 << 64), _digest(w[2], f"step {s}'s d_j")))
    withdrawn = tuple(_int(j, f"step {s}'s withdrawn index", 1 << 64) for j in doc.get("withdrawn", []))
    return Step(s, freed, tuple(written), withdrawn)


def parse(lines: Iterable[str]) -> tuple[Header, Iterator[Step]]:
    """A memory map's header and its steps, read lazily, one JSON object a line.  Each line is checked against the
    format here; the rules across lines are `Replay`'s."""
    it = iter(lines)
    first = next(it, None)
    if first is None:
        raise MemoryMapError("a memory map starts with its header line")
    return header(json.loads(first)), (step(json.loads(line)) for line in it if line.strip())


def dumps(head: Header, steps: Iterable[Step]) -> str:
    """The memory map as its lines, each a canonical JSON object (sorted keys, no spaces)."""
    lines = [head.to_json()] + [s.to_json() for s in steps]
    return "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in lines)


def _runs(indices: Iterable[int]) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for j in sorted(indices):
        if out and out[-1][0] + out[-1][1] == j:
            out[-1][1] += 1
        else:
            out.append([j, 1])
    return tuple((f, c) for f, c in out)


@dataclass(frozen=True, slots=True)
class Change:
    """What one step does to the audit's store: from `first`, the step's fresh indices, the blocks it writes, with
    digests `digests` (none if empty) at `indices`, and those it withdraws (`withdrawn`, canonical runs); and the store
    indices it releases, freed, overwritten or withdrawn, as canonical runs (`released`, which covers `withdrawn`: the
    prover sends both, `Session.release(released, withdrawn)`)."""

    step: int
    first: int
    digests: tuple[bytes, ...]
    released: tuple[tuple[int, int], ...]
    withdrawn: tuple[tuple[int, int], ...] = ()

    @property
    def count(self) -> int:
        return len(self.digests)

    @property
    def indices(self) -> tuple[int, ...]:
        """The store indices of the blocks the step writes, `digests`' in order: its fresh ones less those withdrawn."""
        gone = {j for f, c in self.withdrawn for j in range(f, f + c)}
        return tuple(j for j in range(self.first, self.first + self.count + len(gone)) if j not in gone)

    def append_tree(self, registration: bytes, salts: Sequence[bytes]) -> MerkleTree | None:
        """The step's append tree under fresh `salts`, a salt a block (`audit.append_tree`), whose root and `count`
        go to `Session.append`; None if the step wrote nothing.  A step that withdraws a block has none: its writes
        are not the whole of the append its blocks went out in when they completed, whose tree is the one sent."""
        if self.withdrawn:
            raise MemoryMapError(f"step {self.step} withdraws a block, so its writes are not one append's whole: the "
                                 "tree is the append made when its blocks completed")
        return audit.append_tree(registration, self.first, self.digests, salts) if self.digests else None


class Replay:
    """The prover's audit side of a memory map: the block each slot holds, each step's `Change` (`apply`), and the
    plan's change for the pass at a step since the last pass (`take`).  Refuses (`MemoryMapError`) a map that breaks
    memory-map/v1's rules across lines: steps out of order, a freed slot not live, a written or withdrawn index that
    is not the store's next fresh one, or a header of another registration; a refused step changes nothing.  And
    refuses a pass whose step is not after every step applied and every pass's step taken.  A withdrawn index joins no
    plan, so `take` neither adds nor removes it."""

    def __init__(self, head: Header, registration: audit.Registration | None = None) -> None:
        if registration is not None and (head.registration != registration.digest or head.blocks != registration.blocks):
            raise MemoryMapError("the memory map is beside another registration: its header names "
                                 f"{head.registration.hex()[:16]}… of {head.blocks} blocks")
        self.header, self.extent, self.last, self.taken = head, head.blocks, -1, -1
        self.moved: dict[int, int | None] = {}
        self._added: set[int] = set()
        self._removed: set[int] = set()

    def holds(self, slot: int) -> int | None:
        """The store index of the block `slot` holds, or None if it isn't live."""
        return self.moved.get(slot, slot if slot < self.header.blocks else None)

    def apply(self, s: Step) -> Change:
        if s.step <= self.last:
            raise MemoryMapError(f"step {s.step} follows step {self.last}: steps strictly increase")
        if s.step < self.taken:
            raise MemoryMapError(f"step {s.step} comes after the pass at step {self.taken} was taken: a pass reads "
                                 "the blocks the steps before it left, so apply each step's record before the pass "
                                 "after it")
        if len(set(s.freed)) != len(s.freed):
            raise MemoryMapError(f"step {s.step} frees a slot twice")
        for slot in s.freed:
            if self.holds(slot) is None:
                raise MemoryMapError(f"step {s.step} frees slot {slot}, which holds no block: a slot at or above N "
                                     "is live once written, and a freed one until it is written again")
        if len({w.slot for w in s.written}) != len(s.written):
            raise MemoryMapError(f"step {s.step} writes a slot twice")
        first, k = self.extent, len(s.written) + len(s.withdrawn)
        if (any(not first <= j < first + k for j in s.withdrawn)
                or any(a >= b for a, b in zip(s.withdrawn, s.withdrawn[1:]))):
            raise MemoryMapError(f"step {s.step} withdraws {list(s.withdrawn)}: a withdrawal takes the index its block's "
                                 f"write would have, among the step's fresh ones from {first}, in increasing order")
        taken = set(s.withdrawn)
        for w, j in zip(s.written, (j for j in range(first, first + k) if j not in taken)):
            if w.index != j:
                raise MemoryMapError(f"step {s.step}'s block at slot {w.slot} has index {w.index}, not the store's next "
                                     f"fresh one, {j}: N plus every block written or withdrawn before, in record order")
        gone = [self.holds(slot) for slot in s.freed]
        self.moved.update((slot, None) for slot in s.freed)
        gone += [j for w in s.written if (j := self.holds(w.slot)) is not None]
        self.moved.update((w.slot, w.index) for w in s.written)
        self.extent = first + k
        self.last = s.step
        for j in gone:
            if j in self._added:
                self._added.discard(j)
            else:
                self._removed.add(j)
        self._added.update(w.index for w in s.written)
        return Change(s.step, first, tuple(w.digest for w in s.written), _runs(gone + list(s.withdrawn)),
                      _runs(s.withdrawn))

    def take(self, step: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        """The plan's change for the pass that covers `step`, since the last `take`, for a prover whose plans hold
        every resident block: (added, removed) runs of store indices, the blocks appended since and those released
        since that the last plan held.  The pass names `step` with its root.  Its plan is the blocks resident at
        `step`, so every step applied is before it, and it follows the last pass's."""
        if type(step) is not int or not max(self.last, self.taken) < step < audit.STEP_LIMIT:
            raise MemoryMapError(f"a pass at step {step!r} follows the steps applied (to {self.last}) and the last "
                                 f"pass's (at {self.taken}): a pass reads the blocks resident at its step, those of "
                                 "the steps before it, so take it before applying its own step's record")
        out = _runs(self._added), _runs(self._removed)
        self._added, self._removed, self.taken = set(), set(), step
        return out


def schedule(head: Header, steps: Iterable[Step], appends: Sequence[tuple[int, int]],
             passes: Sequence[int]) -> _schedule.Schedule:
    """The schedule the prover publishes for `Session.scheduled` (`schedule.Schedule`), from its memory map: each
    registered block's free, each appended block's write and free (a block's free is the step that freed its slot or
    wrote over it), the appended blocks grouped as the session's `appends` (first, count) went out, and the passes'
    steps.  A block the map never writes (withdrawn, or appended and not encoded by the map's end) is never resident.
    The map is held to memory-map/v1 as `Replay` holds it (`MemoryMapError`).  A prover whose plans are `Replay.take`'s
    sends the plans this schedule gives."""
    replay, write, free = Replay(head), {}, {}
    for s in steps:
        for slot in [*s.freed, *(w.slot for w in s.written)]:
            if (j := replay.holds(slot)) is not None:
                free[j] = s.step
        replay.apply(s)
        write.update((w.index, s.step) for w in s.written)
    registered: list[list[int]] = []
    for j in sorted(j for j in free if j < head.blocks):
        if registered and registered[-1][0] + registered[-1][1] == j and registered[-1][2] == free[j]:
            registered[-1][1] += 1
        else:
            registered.append([j, 1, free[j]])
    return _schedule.Schedule(
        tuple((f, c, e) for f, c, e in registered),
        tuple(tuple((write[j], free.get(j)) if j in write else None for j in range(first, first + count))
              for first, count in appends),
        tuple(passes))
