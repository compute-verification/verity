"""memory-map/v1 (`remote_bandwidth.memory_map`, the program's I3), the prover's record of the blocks each serving step
wrote and freed beside its registered store, as `Lines` builds it: the integration's one builder, which the Commit's map
(`verity_vllm.commit.memory_map.record`, from the circuit) and a `Store`'s (`MemoryMap`, from its KV events) both drive.

~~~text
slot      a block's store index, a `Store`'s in the registry's numbering: region ``first`` plus its region block, a KV
          block of page p, column c in layer l at ``first_l + ids[p, c]``; every region's blocks are 4,096 bytes
j         an appended block's index: N, the store's blocks when the map began, plus every block appended before it
d_j       `codec.digest` of the block's payload (a `Store`'s: its region codec's decoding of its bytes, `Store.payload`)
appended  a block `kv_append` completes, once its step's forward and frees are done (`complete`) if a free has not
          taken it back by then, a step's as one append by slot (``on_append``), which the session sends and the device
          encodes in place once its coin arrives (`encoded`); and a block re-keyed once its j is as old as the lifetime cap
written   an appended block, in the line open when its append lands (`Lines.land`, in append order: its encode, the step
          before the first whose reads see it; deferred residency, remote_bandwidth's ``PROTOCOL.md`` §4), uncredited
          until then.  An append that would write a slot its line writes already waits a line, with every append after it
freed     a slot's block, registered or written, in the line its caller frees it (`Lines.free`), and a written block in
          the line of its last structural reader (`Append.lasts`), or the line after its write if that is no earlier: one
          read last before its write is recorded (a `Store`'s whose page `kv_free` returns first: at the open line's
          step) is written all the same (its j is the session's) and freed a line later (memory-map/v1 withdraws it,
          which `Lines` doesn't emit yet).  A line frees no slot it writes (a write releases the slot's previous block).
          A `Store` frees an appended block (j >= N) of a page `kv_free` returns, though its bytes stay as filler: the
          audit credits only the blocks the circuit keeps live, so the store releases it (dirty, out of every plan) until
          a write completes it again.  A page's untouched blocks, registered before the seed and unwritten since, stay
          resident (`kv_free` leaves them in the filler).  And a live block a step's write reaches, completed or not: its
          bytes are no registered block until the block's write is recorded (`Store`'s dirty); `kv_alloc` emits nothing
re-key    a live written block appended again with its d_j at a fresh j (`Lines.rekey`): its old j freed in the open
          line, its new one written where that append lands
~~~

Steps.  `Store.kv_append(appends, s)` opens step s's line: the engine has scheduled step s, whose forward is next.  A
`kv_free` is recorded in the open line, the step whose forward ran last, which bounds the freed values' last structural
reader (where memory-map/v1 frees a block) from above: vLLM frees a finished request's pages in
`Scheduler.update_from_output` after step s's forward, and a preempted request's in the next `Scheduler.schedule`,
before that step's `kv_append`; an abort between the two frees there too.  Step s is those pages' last structural
reader if their request was scheduled at s (a finished or preempted one was) and its attention reads every row it
holds (dense attention, DSA's top-k network), and earlier if not, so s bounds it from above.  A block written at s and freed before the line closes
was never read by a later step: the line drops the write and frees the block the slot held before, if any.

`end_step` (the binder's `Scheduler.update_from_output`, once the step's forward has written its KV) takes the
digests of the step's completed blocks from the bytes in place, and `complete` (the session, once the engine step
has returned, its finished requests' pages freed) appends those still held; a line not completed by then is at its
close.  The line closes (`flush`) when the next step opens, at the next `kv_append` or `kv_write_reference`, before the
store changes: after the preemptions of that step's `schedule`, and before its pass, whose plan is the blocks resident
at its step (`Replay.take`).  It writes the blocks whose encodes landed while it was open.  Pages freed after the last
line closed go in the next line.

Lifetime.  No plan holds a block written after the codec seed L windows or more after the window its append joined
(`audit.Params.lifetime`, ``lifetime`` here, the session's, or `LIFETIME`).  The map keeps the window each live written
block's j joined (`Live.window`): the window open when its append is made, that of the first pass after it.  The verifier's is the
window of the first root it receives after the append, which is that pass's or, if the pass is discarded at a window
boundary, a later one, so the map's is never later.  Closing a line in window w re-keys every live written block that
joined at w − L or before and is neither freed nor written in the line: the line frees its old j, which so leaves the
plans before the first pass of window w, the first window that could not hold it, and appends it at a fresh j, whose
write waits for its encode.  No margin past L is needed, provided the window of the next pass is open when the line
closes (the binder takes the verifier's messages before `kv_append`).

The header names the remote audit's `Registration.digest`, the one the verifier gives `Store.begin_window` for the
session's first window; without it there is no map.  Its indices are one session's (`audit.Session`: a later session
over the store starts again from N), so a store records one map.

The map is a witness (``d_j`` hashes user data): it goes to ``path``, the prover's own file, and is never published.
"""
from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from verity.core.protocols.remote_bandwidth import audit, codec
from verity.core.protocols.remote_bandwidth.memory_map import Change, Header, MemoryMapError, Replay, Step, Written, dumps, parse

LIFETIME = audit.lifetime_windows(audit.WINDOW_NS, deferred=True)
"""The lifetime cap a map keeps without the session's, in windows: the audit's at its stated window under deferred
residency, 2."""


def record(store: Any, path: str | Path | None = None, registration: bytes | None = None,
           lifetime: int = LIFETIME, on_line: Callable[[Change], None] | None = None,
           on_append: Callable[[int, tuple[int, ...], tuple[bytes, ...], list[bytes]], None] | None = None) -> MemoryMap:
    """Start ``store``'s memory map (`Store.record_memory_map`; ``store`` a `verity_vllm.protocols.remote_bandwidth.registry.Store`)."""
    if registration is None:
        raise ValueError("a memory map names the store's registration: pass the remote audit's Registration.digest, "
                         "the one the session's first window chains from (Store.begin_window)")
    if store.memory_map is not None:
        raise RuntimeError("the store already records a memory map")
    lay = store.layout()
    bad = [(r.name, r.block_bytes) for r in lay.regions if r.block_bytes != codec.BLOCK_BYTES]
    if bad:
        raise ValueError(f"memory-map/v1 slots are {codec.BLOCK_BYTES}-byte blocks; regions {bad[:4]} are not: "
                         f"register them with block_bytes={codec.BLOCK_BYTES}")
    store.memory_map = MemoryMap(store, Header(registration, lay.blocks), path, lifetime, on_line, on_append)
    return store.memory_map


def end(store: Any) -> None:
    """Close ``store``'s map's open line and stop recording: its session is over, and the steps after it are none of its."""
    if store.memory_map is not None:
        store.memory_map.flush()
        store.memory_map = None


class Live(NamedTuple):
    """A written block live as of the last line: its index, its digest, its last structural reader (None: until a free)
    and its append's window."""

    index: int
    digest: bytes
    last: int | None
    window: int


@dataclass(slots=True)
class Append:
    """One append: its first index, its blocks' slots, digests and last structural readers (None: until a free; a caller
    may set one until its write is taken) in index order, and the window open when it was made."""

    first: int
    slots: tuple[int, ...]
    digests: tuple[bytes, ...]
    lasts: list[int | None]
    window: int = 0


class Lines:
    """memory-map/v1's lines beside a registered store of ``blocks`` blocks, from a session's appends and frees in its
    caller's slots, one line a step (the rules above).  ``live`` is each slot's written block as of the last line."""

    def __init__(self, blocks: int) -> None:
        self.next = blocks
        self.live: dict[int, Live] = {}
        self._pending: deque[Append] = deque()
        self._landed: deque[Append] = deque()
        self._freed: set[int] = set()
        self._due: list[tuple[int, int, int]] = []

    @property
    def waiting(self) -> int:
        """Blocks appended whose writes no line has taken: their appends still to land, or landed since the last line."""
        return sum(len(a.slots) for a in (*self._pending, *self._landed))

    @property
    def busy(self) -> bool:
        """Whether a later line has anything to record yet: a landed append, or a free."""
        return bool(self._landed or self._freed or self._due)

    def unwritten(self) -> dict[int, tuple[Append, int]]:
        """Each slot of an append whose write no line has taken: the newest such append, and the block's place in it."""
        return {x: (a, u) for a in (*self._landed, *self._pending) for u, x in enumerate(a.slots)}

    def append(self, slots: Sequence[int], digests: Sequence[bytes], lasts: Sequence[int | None] | None = None,
               window: int = 0) -> Append:
        """Append the blocks ``slots`` at the next indices, each read last at ``lasts`` (None: until a free)."""
        a = Append(self.next, tuple(slots), tuple(digests), [None] * len(slots) if lasts is None else list(lasts), window)
        self.next += len(a.slots)
        self._pending.append(a)
        return a

    def land(self, first: int) -> None:
        """The append from ``first`` has landed: the next line writes it."""
        if not self._pending or self._pending[0].first != first:
            raise RuntimeError(f"the append from block {first} is not the oldest awaiting its encode "
                               f"({self._pending[0].first if self._pending else 'none awaits'}): appends are encoded "
                               "in their order")
        self._landed.append(self._pending.popleft())

    def free(self, slots: Iterable[int]) -> None:
        """The next line frees the blocks ``slots`` hold, registered or written."""
        self._freed.update(slots)

    def held(self, step: int) -> dict[int, Live]:
        """The written blocks line ``step`` keeps: freed by no `free` and read past it."""
        return {x: b for x, b in self.live.items() if x not in self._freed and (b.last is None or b.last > step)}

    def rekey(self, slots: Sequence[int], window: int = 0) -> Append:
        """Free the live blocks ``slots`` in the next line, and append them again with their digests and readers."""
        bs = [self.live[x] for x in slots]
        self.free(slots)
        return self.append(slots, [b.digest for b in bs], [b.last for b in bs], window)

    def close(self, step: int) -> Step | None:
        """Line ``step``: its frees and the landed appends it writes, or None if it records nothing."""
        freed, self._freed = self._freed, set()
        while self._due and self._due[0][0] <= step:
            _, x, j = heapq.heappop(self._due)
            if x in self.live and self.live[x].index == j:
                freed.add(x)
        for x in freed:
            self.live.pop(x, None)
        seen: set[int] = set()
        written = []
        while self._landed and not seen & set(self._landed[0].slots):
            a = self._landed.popleft()
            for u, (x, d, last) in enumerate(zip(a.slots, a.digests, a.lasts)):
                self.live[x] = Live(a.first + u, d, last, a.window)
                if last is not None:
                    heapq.heappush(self._due, (max(last, step + 1), x, a.first + u))
                written.append(Written(x, a.first + u, d))
            seen.update(a.slots)
        freed -= seen
        return Step(step, tuple(sorted(freed)), tuple(written)) if freed or written else None


class MemoryMap:
    """memory-map/v1 of ``store``'s KV pool beside ``header``'s store, built by ``lines``, its lines in ``path`` (appended
    as each closes) or, without one, kept (`text`).  ``replay`` is the map as the audit replays it, which refuses a line that breaks
    memory-map/v1 before it is written.  ``on_append(first, slots, digests, payloads)`` gets each append as it is made (the
    remote session's, `verity_vllm.protocols.remote_bandwidth.remote`, which sends it and calls `encoded` once the device has encoded it
    under its coin); without it, each append's encode lands at once, as when the coin is back before the next step's
    reads.  ``on_line(change)``, if given, gets each line's `Change` as it closes.  ``ahead`` is how many windows past the
    open one the pass after a line may fall in (1 for a serving session whose step can run past a boundary, its pass then
    the next window's first): a line re-keys for that window, so no block it leaves is stale there."""

    def __init__(self, store: Any, header: Header, path: str | Path | None = None, lifetime: int = LIFETIME,
                 on_line: Callable[[Change], None] | None = None,
                 on_append: Callable[[int, tuple[int, ...], tuple[bytes, ...], list[bytes]], None] | None = None) -> None:
        if type(lifetime) is not int or lifetime < 1:
            raise ValueError("the lifetime cap is a positive number of windows")
        self.store, self.header, self.path = store, header, None if path is None else Path(path)
        self.lifetime, self.replay, self.lines = lifetime, Replay(header), Lines(header.blocks)
        self.ahead = 0
        self.last = -1
        self.rekeyed = 0
        self.on_line, self.on_append = on_line, on_append
        self._head = dumps(header, ())
        self._open: tuple[int, set[int], set[int]] | None = None
        self._digests: dict[int, bytes] | None = None
        self._kept: list[str] = []
        if self.path is not None:
            self.path.write_text(self._head)

    @property
    def extent(self) -> int:
        return self.replay.extent

    def slots(self, keys: np.ndarray) -> tuple[int, ...]:
        """The store indices of the pool's blocks ``page * per_page + column``, in every layer region, ascending."""
        if not len(keys):
            return ()
        p = self.store.pool
        ids = p.ids[keys // p.per_page, keys % p.per_page]
        return tuple(sorted(int(x) for rn in p.regions for x in (self.store.region(rn).first + ids).tolist()))

    def free(self, keys: np.ndarray) -> np.ndarray:
        """`kv_free` returned the pool's live blocks ``keys``: those appended (held at a j of the map's, or completed in
        the open line) are freed in the open line, the step whose forward ran last, which bounds their last structural
        reader from above (after the last line closed, in the next); those whose writes no line has taken yet are read
        last there; the rest, registered blocks no write has reached, stay resident.  The mask of those freed."""
        p, lines = self.store.pool, self.lines
        first = self.store.region(p.regions[0]).first
        completed = self._open[1] if self._open is not None else set()
        waiting = lines.unwritten()
        keys = np.asarray(keys)
        now, taken, dead = (np.zeros(len(keys), bool) for _ in range(3))
        for u, k in enumerate(keys.tolist()):
            x = first + int(p.ids[k // p.per_page, k % p.per_page])
            dead[u], taken[u] = x in waiting, x in completed
            now[u] = not dead[u] and not taken[u] and x in lines.live
        lines.free(self.slots(keys[now]))
        if self._open is not None:
            self._open[2].update(self.slots(keys[taken]))
        for x in self.slots(keys[dead]):
            a, u = waiting[x]
            a.lasts[u] = self.last
        return now | taken | dead

    def check(self, step: int) -> None:
        if type(step) is not int or step <= self.last:
            raise MemoryMapError(f"kv_append's step {step!r} follows step {self.last}: a memory map's steps strictly "
                                 "increase, so give each kv_append the next step")

    def open(self, step: int, reached: np.ndarray, completed: np.ndarray) -> None:
        """Open step ``step``'s line, closing the one before: its write reached the pool's live blocks ``reached`` and
        completed ``completed`` (keys ``page * per_page + column``)."""
        self.flush(step)
        self.last = step
        self.lines.free(self.slots(reached))
        self._open, self._digests = (step, set(self.slots(completed)), set()), None

    def end_step(self) -> None:
        """The open step's forward has written its KV: take its completed blocks' digests from the bytes in place."""
        if self._open is not None:
            self._digests = {x: codec.digest(self.store.payload(x)) for x in self._open[1]}

    def complete(self) -> None:
        """The open step's forward and its frees are done: its completed blocks a free has not taken back go out as
        one append (a block completed and freed in the step is never read while resident, so it is no write; the
        block its slot held was freed when the write reached it).  Called again, or with nothing completed, it appends
        nothing."""
        if self._open is None or not self._open[1]:
            return
        _, completed, taken = self._open
        digests = self._digests or {}
        got = sorted(completed - taken)
        completed.clear()
        if got:
            self._send(self.lines.append(got, [digests.get(x) or codec.digest(self.store.payload(x)) for x in got],
                                         window=self.window()))

    def _send(self, a: Append) -> None:
        if self.on_append is None:
            self.encoded(a.first)
        else:
            self.on_append(a.first, a.slots, a.digests, [self.store.payload(x) for x in a.slots])

    def encoded(self, first: int) -> None:
        """The device has encoded the append from ``first`` under its coin: the open line writes its blocks (they are
        resident from the next step on, the first whose reads see them encoded)."""
        self.lines.land(first)

    def window(self) -> int:
        """The window of the store's next pass: the open one, else the next to open."""
        return self.store.window if self.store.window_open else self.store.window + 1

    def flush(self, step: int | None = None) -> list[Step]:
        """Close the open line (none: nothing): append its completed blocks if `complete` has not, free and append its
        re-keys for the window of the next pass, write the blocks whose encodes have landed, and write it.  With
        ``step``, first refuse it unless it follows the last step opened, before the store changes."""
        if step is not None:
            self.check(step)
        if self._open is None:
            return []
        self.complete()
        s, self._open = self._open[0], None
        w = self.window() + self.ahead
        rekey = sorted(x for x, b in self.lines.held(s).items() if b.window <= w - self.lifetime)
        if rekey:
            self.rekeyed += len(rekey)
            self._send(self.lines.rekey(rekey, self.window()))
        line = self.lines.close(s)
        if line is None:
            return []
        change = self.replay.apply(line)
        if self.on_line is not None:
            self.on_line(change)
        text = dumps(self.header, [line])[len(self._head):]
        if self.path is None:
            self._kept.append(text)
        else:
            with self.path.open("a") as f:
                f.write(text)
        return [line]

    def text(self) -> str:
        """The map's lines so far, header first (`memory_map.parse` reads them)."""
        return self.path.read_text() if self.path is not None else self._head + "".join(self._kept)


def agreement(backend: Any) -> dict[str, Any]:
    """The memory map against the session and the store (``backend``, a `remote.StoreBackend`), once the session is done:
    replayed from its header, its live indices are the session's last plan's and its blocks the audit's appends, in order,
    up to those whose writes wait for their encodes (deferred residency); then, its open line closed, every slot it holds
    at an appended index holds bytes of that index's digest in the store."""
    trace, store = backend.trace, backend.store

    def replay() -> tuple[Replay, dict[int, bytes]]:
        head, steps = parse(store.memory_map.text().splitlines(keepends=True))
        r, digests = Replay(head), {}
        for st in steps:
            r.apply(st)
            digests.update((w.index, w.digest) for w in st.written)
        return r, digests

    def resident(r: Replay) -> np.ndarray:
        out = np.zeros(max(r.extent, len(trace.live)), bool)
        out[:r.header.blocks] = True
        out[[x for x in r.moved if x < r.header.blocks]] = False
        out[[j for j in r.moved.values() if j is not None]] = True
        return out

    r, digests = replay()
    n = r.header.blocks
    mine = np.zeros(max(r.extent, len(trace.live)), bool)
    mine[:len(trace.live)] = trace.live
    live = bool(np.array_equal(resident(r), mine))
    appends = sorted(digests) == list(range(n, r.extent)) and trace.digests[:r.extent - n] == [digests[j] for j in sorted(digests)]
    store.memory_map.flush()
    r, digests = replay()
    held = sorted(x for x, j in r.moved.items() if j is not None and j >= n)
    wrong = [x for x in held if not backend.holds(x, r.holds(x), digests[r.holds(x)])]
    return {"live_agrees": live, "appends_agree": appends, "recorded": r.extent - n,
            "unrecorded": len(trace.digests) - (r.extent - n), "held_appended": len(held), "held_wrong": wrong[:16],
            "agrees": live and appends and not wrong}
