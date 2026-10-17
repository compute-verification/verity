"""`MemoryMap`: a `Store`'s KV events as memory-map/v1 (`remote_bandwidth.memory_map`, the program's I3), the prover's
record of the blocks each serving step wrote and freed beside its registered store.

~~~text
slot      the block's store index in the registry's numbering: region ``first`` plus its region block, a KV block of
          page p, column c in layer l at ``first_l + ids[p, c]``; every region's blocks are 4,096 bytes
j         the store's next fresh index: N, the store's blocks when the map began, plus every block written before it,
          in record order (a step's blocks by slot)
d_j       `codec.digest` of the block's payload, its region codec's decoding of the bytes it holds once written
          (`Store.payload`)
written   a block `kv_append` completes, at the step that completes it (a dirty block is in no step), and a block
          re-keyed: written again in its slot, the same payload at a fresh j, once its j is as old as the lifetime cap
freed     a live block of a page `kv_free` returns, though its bytes stay as filler: the audit credits only the blocks
          the circuit keeps live, so the store releases it (dirty, out of every plan) until a write completes it again
          under a new j.  And a live block a step's write reaches without completing it: its bytes are no registered
          block until a write completes it (`Store`'s dirty).  A block is freed once; `kv_alloc` emits nothing
~~~

Steps.  `Store.kv_append(appends, s)` opens step s's line: the engine has scheduled step s, whose forward is next.  A
`kv_free` is recorded in the open line, the step whose forward ran last, which is the last read of the freed values:
vLLM frees a finished request's pages in `Scheduler.update_from_output` after step s's forward, and a preempted
request's in the next `Scheduler.schedule`, before that step's `kv_append`; an abort between the two frees there too.
Each of those pages was last read by step s's forward if its request was scheduled at s (a finished or preempted one
was), and earlier if not, so s bounds the last read from above.  A block written at s and freed before the line closes
was never read by a later step: the line drops the write and frees the block the slot held before, if any.

`end_step` (the binder's `Scheduler.update_from_output`, once the step's forward has written its KV) takes the
digests of the step's writes from the bytes in place.  The line closes (`flush`) when the next step opens, at the next
`kv_append` or `kv_write_reference`, before the store changes: after the preemptions of that step's `schedule`, and
before its pass, whose plan is the blocks resident at its step (`Replay.take`).  Pages freed after the last line
closed go in the next line.

Lifetime.  No plan holds a block written after the codec seed L windows or more after the window its append joined
(`audit.Params.lifetime`, ``lifetime`` here, 60 by default).  The map keeps the window each live written block's j
joined (`joined`): the window open when its line closes, that of the first pass after it.  The verifier's is the window
of the first root it receives after the append, which is that pass's or, if the pass is discarded at a window boundary,
a later one, so the map's is never later.  Closing a line in window w re-keys every live written block that joined at
w − L or before and is neither freed nor written in the line, so its old j leaves the plans before the first pass of
window w, the first window that could not hold it: no margin past L is needed, provided the window of the next pass is
open when the line closes (the binder takes the verifier's messages before `kv_append`).

The header names the remote audit's `Registration.digest`, the one the verifier gives `Store.begin_window` for the
session's first window; without it there is no map.  Its indices are one session's (`audit.Session`: a later session
over the store starts again from N), so a store records one map.

The map is a witness (``d_j`` hashes user data): it goes to ``path``, the prover's own file, and is never published.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth import audit, codec
from verity.core.protocols.remote_bandwidth.memory_map import Change, Header, MemoryMapError, Replay, Step, Written, dumps


def record(store: Any, path: str | Path | None = None, registration: bytes | None = None,
           lifetime: int = audit.LIFETIME, on_line: Callable[[Change, list[bytes]], None] | None = None) -> MemoryMap:
    """Start ``store``'s memory map (`Store.record_memory_map`; ``store`` a `verity_vllm.store.registry.Store`)."""
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
    store.memory_map = MemoryMap(store, Header(registration, lay.blocks), path, lifetime, on_line)
    return store.memory_map


class MemoryMap:
    """memory-map/v1 of ``store``'s KV pool beside ``header``'s store, its lines in ``path`` (appended as each closes)
    or, without one, kept (`text`).  ``replay`` is the map as the audit replays it, which refuses a line that breaks
    memory-map/v1 before it is written; ``on_line(change, payloads)``, if given, gets each line's `Change` and its
    written blocks' payloads as it closes (the remote session's appends, `verity_vllm.store.remote`)."""

    def __init__(self, store: Any, header: Header, path: str | Path | None = None, lifetime: int = audit.LIFETIME,
                 on_line: Callable[[Change, list[bytes]], None] | None = None) -> None:
        if type(lifetime) is not int or lifetime < 1:
            raise ValueError("the lifetime cap is a positive number of windows")
        self.store, self.header, self.path = store, header, None if path is None else Path(path)
        self.lifetime, self.replay = lifetime, Replay(header)
        self.last = -1
        self.joined: dict[int, int] = {}
        self.rekeyed = 0
        self.on_line = on_line
        self._head = dumps(header, ())
        self._open: tuple[int, set[int], set[int]] | None = None
        self._digests: dict[int, bytes] | None = None
        self._later: set[int] = set()
        self._kept: list[str] = []
        if self.path is not None:
            self.path.write_text(self._head)

    @property
    def extent(self) -> int:
        return self.replay.extent

    def slots(self, keys: np.ndarray) -> tuple[int, ...]:
        """The store indices of the pool's blocks ``page * per_page + column``, in every layer region, ascending."""
        p = self.store.pool
        ids = p.ids[keys // p.per_page, keys % p.per_page]
        return tuple(sorted(int(x) for rn in p.regions for x in (self.store.region(rn).first + ids).tolist()))

    def free(self, keys: np.ndarray) -> None:
        """`kv_free` released the pool's live blocks ``keys``: freed in the open line, the step whose forward read them
        last (after the last line closed, in the next)."""
        (self._open[1] if self._open is not None else self._later).update(self.slots(keys))

    def check(self, step: int) -> None:
        if type(step) is not int or step <= self.last:
            raise MemoryMapError(f"kv_append's step {step!r} follows step {self.last}: a memory map's steps strictly "
                                 "increase, so give each kv_append the next step")

    def open(self, step: int, freed: np.ndarray, written: np.ndarray) -> None:
        """Open step ``step``'s line, closing the one before: its write reached the pool's blocks ``freed`` and completed
        ``written`` (keys ``page * per_page + column``)."""
        self.flush(step)
        self.last = step
        self._open = (step, set(self.slots(freed)) | self._later, set(self.slots(written)))
        self._later, self._digests = set(), None

    def end_step(self) -> None:
        """The open step's forward has written its KV: take its blocks' digests from the bytes in place."""
        if self._open is not None:
            self._digests = {x: codec.digest(self.store.payload(x)) for x in self._open[2]}

    def window(self) -> int:
        """The window of the store's next pass: the open one, else the next to open."""
        return self.store.window if self.store.window_open else self.store.window + 1

    def flush(self, step: int | None = None) -> list[Step]:
        """Close the open line (none: nothing), with its re-keys for the window of the next pass, and write it.  With
        ``step``, first refuse it unless it follows the last step opened, before the store changes."""
        if step is not None:
            self.check(step)
        if self._open is None:
            return []
        s, freed, written = self._open
        self._open = None
        for x in written & freed:
            written.discard(x)
            if self.replay.holds(x) is None:
                freed.discard(x)
        w = self.window()
        rekey = {x for x, a in self.joined.items() if a <= w - self.lifetime} - written - freed
        digests = self._digests or {}
        out = []
        if freed or written or rekey:
            first = self.extent
            got = [Written(x, first + u, digests.get(x) or codec.digest(self.store.payload(x)))
                   for u, x in enumerate(sorted(written | rekey))]
            out = [Step(s, tuple(sorted(freed)), tuple(got))]
            change = self.replay.apply(out[0])
            for x in freed:
                self.joined.pop(x, None)
            self.joined.update((g.slot, w) for g in got)
            self.rekeyed += len(rekey)
            if self.on_line is not None:
                self.on_line(change, [self.store.payload(g.slot) for g in got])
            lines = dumps(self.header, out)[len(self._head):]
            if self.path is None:
                self._kept.append(lines)
            else:
                with self.path.open("a") as f:
                    f.write(lines)
        return out

    def text(self) -> str:
        """The map's lines so far, header first (`memory_map.parse` reads them)."""
        return self.path.read_text() if self.path is not None else self._head + "".join(self._kept)
