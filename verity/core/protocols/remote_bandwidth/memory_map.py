"""memory-map/v1 (`verity/memory-map/v1`, the program's I3): the per-step record of the KV blocks a serving node wrote
and freed beside its registered store, the remote bandwidth audit's input from serving, and `Replay`, which turns it
into the audit's appends, releases and plan changes.

~~~text
line 1       {"schema": "verity/memory-map/v1", "registration": hex(Registration.digest), "blocks": N,
              "block_bytes": 4096}
line 2, …    {"step": s, "freed": [slot, …], "written": [[slot, j, hex(d_j)], …]}, one per serving step that wrote or
              freed a block, steps strictly increasing
slot         4,096 bytes of the node's memory, named by the registered block it held at registration (vLLM: its
              registry's region block index, a KV pool's pages registered as filler), or from N up for memory outside
              the registered store, below 2^63.  A slot below N starts live, holding registered block `slot`; one at
              or above N is live once written
j            the store index the writer gave the block: the store's next fresh one, N plus every block written before
              in the session, in record order
d_j          codec.digest(payload_j), the sha512/row/v2 digest of the block's 4,096-byte payload, 64 bytes: the device
              encodes the payload under its bound key (codec.binding(d_j, j, key)) at index j
~~~

A write puts block j in its slot and releases the block the slot held, if it was live: the write overwrote it.  A
free releases the slot's block and leaves the slot not live, its memory out of the store, until a write; a slot whose
bytes stay as they were (vLLM's `kv_free`, which returns pages to filler with their last record) is not freed, and its
block stays live until a write overwrites it.  A step's frees apply before its writes; a step frees a slot at most
once, only a live one, and writes a slot at most once.  A block is written once, complete: a KV block the step's tokens
are still filling is not in the record until the step that completes it (the prover-side registry's dirty blocks).

The record is the prover's: `d_j` is a hash of user data, a witness, never public (AGENTS.md's zero-knowledge rule).
The prover's audit side (`Replay`) commits each step's digests in an append tree (`audit.append_tree`, under fresh
salts) and sends its root and size (`Session.append`), and the released blocks' indices as runs (`Session.release`); the
verifier sees those and the plans, never the record.  The appends of a step are sent before the root of any pass
whose plan holds them; `Replay.take` is the plan's change since the last pass (`Plan.changed`, the root's
`Session.receive(..., change=...)`), the blocks appended and released between them, of which a block appended and
released in between is in neither.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from verity.ml.commitments import MerkleTree

from . import audit
from .codec import BLOCK_BYTES
from .receipts import DIGEST_BYTES

SCHEMA = "verity/memory-map/v1"
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
    """A block a step completed: its slot, the store index its writer gave it, and its payload's digest `d_j`."""

    slot: int
    index: int
    digest: bytes


@dataclass(frozen=True, slots=True)
class Step:
    """One serving step's record: the slots it freed, then the blocks it wrote, in index order."""

    step: int
    freed: tuple[int, ...]
    written: tuple[Written, ...]

    def to_json(self) -> dict:
        return {"freed": list(self.freed), "step": self.step,
                "written": [[w.slot, w.index, w.digest.hex()] for w in self.written]}


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
    if type(doc) is not dict or set(doc) != {"freed", "step", "written"}:
        raise MemoryMapError("a step's line is {step, freed, written}")
    s = _int(doc["step"], "a step", 1 << 64)
    if type(doc["freed"]) is not list or type(doc["written"]) is not list:
        raise MemoryMapError(f"step {s}'s freed and written are lists")
    freed = tuple(_int(x, f"step {s}'s freed slot", SLOT_LIMIT) for x in doc["freed"])
    written = []
    for w in doc["written"]:
        if type(w) is not list or len(w) != 3:
            raise MemoryMapError(f"step {s}'s written block is [slot, index, digest], not {w!r}")
        written.append(Written(_int(w[0], f"step {s}'s written slot", SLOT_LIMIT),
                               _int(w[1], f"step {s}'s store index", 1 << 64), _digest(w[2], f"step {s}'s d_j")))
    return Step(s, freed, tuple(written))


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
    """What one step does to the audit's store: blocks `first` … appended with digests `digests` (none if empty), and
    the store indices it released, freed or overwritten, as canonical runs."""

    step: int
    first: int
    digests: tuple[bytes, ...]
    released: tuple[tuple[int, int], ...]

    @property
    def count(self) -> int:
        return len(self.digests)

    def append_tree(self, registration: bytes, salts: Sequence[bytes]) -> MerkleTree | None:
        """The step's append tree under fresh `salts`, a salt a block (`audit.append_tree`), whose root and `count`
        go to `Session.append`; None if the step wrote nothing."""
        return audit.append_tree(registration, self.first, self.digests, salts) if self.digests else None


class Replay:
    """The prover's audit side of a memory map: the block each slot holds, each step's `Change` (`apply`), and the
    plan's change since the last pass (`take`).  Refuses (`MemoryMapError`) a map that breaks memory-map/v1's rules
    across lines: steps out of order, a freed slot not live, an index that is not the store's next fresh one, or a
    header of another registration; a refused step changes nothing."""

    def __init__(self, head: Header, registration: audit.Registration | None = None) -> None:
        if registration is not None and (head.registration != registration.digest or head.blocks != registration.blocks):
            raise MemoryMapError("the memory map is beside another registration: its header names "
                                 f"{head.registration.hex()[:16]}… of {head.blocks} blocks")
        self.header, self.extent, self.last = head, head.blocks, -1
        self.moved: dict[int, int | None] = {}
        self._added: set[int] = set()
        self._removed: set[int] = set()

    def holds(self, slot: int) -> int | None:
        """The store index of the block `slot` holds, or None if it isn't live."""
        return self.moved.get(slot, slot if slot < self.header.blocks else None)

    def apply(self, s: Step) -> Change:
        if s.step <= self.last:
            raise MemoryMapError(f"step {s.step} follows step {self.last}: steps strictly increase")
        if len(set(s.freed)) != len(s.freed):
            raise MemoryMapError(f"step {s.step} frees a slot twice")
        for slot in s.freed:
            if self.holds(slot) is None:
                raise MemoryMapError(f"step {s.step} frees slot {slot}, which holds no block: a slot at or above N "
                                     "is live once written, and a freed one until it is written again")
        if len({w.slot for w in s.written}) != len(s.written):
            raise MemoryMapError(f"step {s.step} writes a slot twice")
        first = self.extent
        for u, w in enumerate(s.written):
            if w.index != first + u:
                raise MemoryMapError(f"step {s.step}'s block at slot {w.slot} has index {w.index}, not the store's next "
                                     f"fresh one, {first + u}: N plus every block written before, in record order")
        gone = [self.holds(slot) for slot in s.freed]
        self.moved.update((slot, None) for slot in s.freed)
        gone += [j for w in s.written if (j := self.holds(w.slot)) is not None]
        self.moved.update((w.slot, w.index) for w in s.written)
        self.extent = first + len(s.written)
        self.last = s.step
        for j in gone:
            if j in self._added:
                self._added.discard(j)
            else:
                self._removed.add(j)
        self._added.update(range(first, self.extent))
        return Change(s.step, first, tuple(w.digest for w in s.written), _runs(gone))

    def take(self) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        """The plan's change since the last `take`, for a prover whose plans hold every live block: (added, removed)
        runs of store indices, the blocks appended since and those released since that the last plan held."""
        out = _runs(self._added), _runs(self._removed)
        self._added, self._removed = set(), set()
        return out
