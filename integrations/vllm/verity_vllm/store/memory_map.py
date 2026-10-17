"""`MemoryMap`: a `Store`'s KV events as memory-map/v1 (`remote_bandwidth.memory_map`, the program's I3), the prover's
record of the blocks each serving step wrote and freed beside its registered store.

~~~text
slot      the block's store index in the registry's numbering: region ``first`` plus its region block, a KV block of
          page p, column c in layer l at ``first_l + ids[p, c]``; every region's blocks are 4,096 bytes
j         the store's next fresh index: N, the store's blocks when the map began, plus every block written before it,
          in record order (a step's blocks by slot)
d_j       `codec.digest` of the block's payload, its region codec's decoding of the bytes it holds once written
          (`Store.payload`)
written   a block `kv_append` completes, at the step that completes it (a dirty block is in no step)
freed     a live (registered) block of a page `kv_free` returns, though its bytes stay as filler: the audit credits
          only the blocks the circuit keeps live, so the store releases it (dirty, out of every plan) until a write
          completes it again under a new j; it is freed at the next `kv_append`'s step, before that step's writes.
          And a live block a step's write reaches without completing it: its bytes are no registered block until a
          write completes it (`Store`'s dirty).  A block is freed once; `kv_alloc` emits nothing
~~~

`Store.kv_append` runs before the step's KV writes land on the device (the binder calls it from
`Scheduler.schedule`), so it records each step's slots, and their indices and digests are taken at `flush`, once the
bytes are in place: at the next `kv_append`, at the binder's `end` (`Scheduler.update_from_output`), or by the driver
after the last step.  `kv_write_reference` writes first and flushes at once.  Pages freed after the last step are in
no step.

The header names the remote audit's `Registration.digest`, the one the verifier gives `Store.begin_window` for the
session's first window; without it there is no map.

The map is a witness (``d_j`` hashes user data): it goes to ``path``, the prover's own file, and is never published.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import Header, MemoryMapError, Step, Written, dumps


def record(store: Any, path: str | Path | None = None, registration: bytes | None = None) -> MemoryMap:
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
    head = Header(registration, lay.blocks)
    store.memory_map = MemoryMap(store, head, path)
    return store.memory_map


class MemoryMap:
    """memory-map/v1 of ``store``'s KV pool beside ``header``'s store, its lines in ``path`` (appended at each `flush`)
    or, without one, kept (`text`)."""

    def __init__(self, store: Any, header: Header, path: str | Path | None = None) -> None:
        self.store, self.header, self.path = store, header, None if path is None else Path(path)
        self.extent, self.last = header.blocks, -1
        self._head = dumps(header, ())
        self._pending: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
        self._kept: list[str] = []
        self._freed: set[int] = set()
        if self.path is not None:
            self.path.write_text(self._head)

    def slots(self, keys: np.ndarray) -> tuple[int, ...]:
        """The store indices of the pool's blocks ``page * per_page + column``, in every layer region, ascending."""
        p = self.store.pool
        ids = p.ids[keys // p.per_page, keys % p.per_page]
        return tuple(sorted(int(x) for rn in p.regions for x in (self.store.region(rn).first + ids).tolist()))

    def free(self, keys: np.ndarray) -> None:
        """`kv_free` released the pool's live blocks ``keys``: freed at the next step."""
        self._freed.update(self.slots(keys))

    def record(self, step: int, freed: np.ndarray, written: np.ndarray) -> None:
        """Step ``step`` freed the pool's blocks ``freed`` (and those released since the last step) and completed
        ``written`` (keys ``page * per_page + column``), left for `flush`, which gives each written block the next
        fresh index, in slot order, and its digest."""
        self.flush(step)
        self.last = step
        gone, self._freed = tuple(sorted(self._freed.union(self.slots(freed)))), set()
        if gone or len(written):
            self._pending.append((step, gone, self.slots(written)))

    def flush(self, step: int | None = None) -> list[Step]:
        """The recorded steps not yet flushed, with each written block's digest from the bytes its slot holds now.
        With ``step``, first refuse it unless it follows the last step recorded, before the store changes."""
        if step is not None and step <= self.last:
            raise MemoryMapError(f"kv_append's step {step} follows step {self.last}: a memory map's steps strictly "
                                 "increase, so give each kv_append the next step")
        out = []
        for s, freed, slots in self._pending:
            first, self.extent = self.extent, self.extent + len(slots)
            out.append(Step(s, freed, tuple(Written(x, first + u, codec.digest(self.store.payload(x)))
                                            for u, x in enumerate(slots))))
        self._pending = []
        if out:
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
