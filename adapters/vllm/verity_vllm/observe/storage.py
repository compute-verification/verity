"""AllocationId adapter (runtime side): stable integer ids for tensor storages, never raw pointers in the log.

An *allocation* is one `(device, storage data pointer, nbytes, generation)`.  The generation increments each time the
same pointer range is observed to be reused by a **new** storage: the previous allocation is then implied freed
(`StorageFree(implied=True)`) before the new `StorageAlloc` is emitted.

How "new storage" is decided (facts, no judgement):

* torch preserves the Python object of a storage (`t.untyped_storage() is t.untyped_storage()`), so `id(storage)`
  identifies the live `StorageImpl`; a `weakref` callback fires when the storage is destroyed (the moment its device
  memory goes back to the caching allocator) and yields a direct `StorageFree(implied=False)`.  Verified on torch
  2.13 by a pod probe: pointer *and* object-id reuse both happen right after a free, so the death callback is
  what makes generations exact.
* Fallbacks (no weakref support / no preservation): same pointer with a different `nbytes`, or a pointer range that
  overlaps another live range, is a new allocation (implied free of the overlapped ones).

Events are queued in `events_pending()` and must be written *before* the `ViewDesc` that references the new
allocation (the observer drains the queue right before every record it writes).  The adapter holds no tensor or
storage references (weakrefs only).
"""

from __future__ import annotations

import bisect
import threading
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any

from .events import StorageAlloc, StorageFree, ViewDesc


@dataclass
class _Live:
    alloc: int
    ptr: int
    nbytes: int
    device: str
    generation: int
    key: int | None            # id(storage PyObject) while alive, None when unknown
    ref: Any = None            # weakref to the storage object (frees), or None


def dtype_name(dtype: Any) -> str:
    """`torch.bfloat16` -> "bfloat16" (the pure side never imports torch)."""
    s = str(dtype)
    return s[6:] if s.startswith("torch.") else s


class StorageAdapter:
    """Assigns `AllocationId`s.  `observe(tensor)` / `view_of(tensor)` are the hot path (called for every tensor
    argument of every dispatcher call); `events_pending()` drains the StorageAlloc/StorageFree records."""

    def __init__(self, *, use_weakrefs: bool | None = None) -> None:
        self._live: dict[str, dict[int, _Live]] = {}      # device -> ptr -> live allocation
        self._starts: dict[str, list[int]] = {}           # device -> sorted live pointers
        self._by_key: dict[int, _Live] = {}               # id(storage) -> live allocation
        self._gen: dict[tuple[str, int], int] = {}        # (device, ptr) -> last generation handed out
        self._null: dict[str, int] = {}                   # device -> alloc id of the empty storage
        self._pending: deque = deque()
        self._lock = threading.RLock()      # re-entrant: a GC-triggered death callback may fire under the lock
        self._next = 1
        self.use_weakrefs = _probe_weakrefs() if use_weakrefs is None else bool(use_weakrefs)
        self.stats = {"allocs": 0, "frees_direct": 0, "frees_implied": 0, "generation_bumps": 0, "observed": 0}

    # ------------------------------------------------------------------ public API

    def observe(self, t: Any, origin: str = "op") -> int:
        """Allocation id of `t`'s storage (queues StorageAlloc / StorageFree as needed)."""
        st = t.untyped_storage()
        nbytes = int(st.nbytes())
        device = str(t.device)
        if nbytes == 0:
            return self._null_alloc(device)
        ptr = int(st.data_ptr())
        key = id(st) if self.use_weakrefs else None
        return self.observe_raw(ptr, nbytes, device, key=key, origin=origin, storage=st if self.use_weakrefs else None)

    def observe_raw(self, ptr: int, nbytes: int, device: str, *, key: int | None = None, origin: str = "op",
                    storage: Any = None) -> int:
        """Pointer-level entry point (testable without torch).  `key` identifies the live storage object when
        known; `storage` (optional) receives a weakref whose death is recorded as a direct StorageFree."""
        self.stats["observed"] += 1
        if nbytes == 0 or ptr == 0:
            return self._null_alloc(device)
        with self._lock:
            table = self._live.setdefault(device, {})
            if key is not None:
                prev = self._by_key.get(key)
                if prev is not None and (prev.ptr != ptr or prev.device != device):
                    self._free_locked(prev, implied=True)     # same storage object, new data pointer (resize_/set_)
            cur = table.get(ptr)
            if cur is not None:
                same_obj = key is None or cur.key is None or cur.key == key
                if cur.nbytes == nbytes and same_obj:
                    if cur.key is None and key is not None:
                        cur.key = key
                        self._by_key[key] = cur
                    return cur.alloc
                self._free_locked(cur, implied=True)          # same pointer, new storage
            for other in self._overlapping_locked(device, ptr, nbytes):
                self._free_locked(other, implied=True)        # range reused by a new allocation
            gen = self._gen.get((device, ptr), -1) + 1
            self._gen[(device, ptr)] = gen
            if gen > 0:
                self.stats["generation_bumps"] += 1
            entry = _Live(self._next, ptr, nbytes, device, gen, key)
            self._next += 1
            table[ptr] = entry
            bisect.insort(self._starts.setdefault(device, []), ptr)
            if key is not None:
                self._by_key[key] = entry
            if storage is not None:
                entry.ref = weakref.ref(storage, self._make_death_callback(entry))
            self.stats["allocs"] += 1
            self._pending.append(StorageAlloc(entry.alloc, ptr, nbytes, device, gen, origin))
            return entry.alloc

    def view_of(self, t: Any, origin: str = "op") -> ViewDesc:
        """Physical view of a tensor: allocation, storage offset (elements), shape, strides, dtype, device."""
        alloc = self.observe(t, origin)
        return ViewDesc(alloc, int(t.storage_offset()), tuple(int(s) for s in t.shape),
                        tuple(int(s) for s in t.stride()), dtype_name(t.dtype), str(t.device))

    def events_pending(self) -> list[StorageAlloc | StorageFree]:
        """Drain the queued StorageAlloc / StorageFree records (in observation order)."""
        out: list = []
        while True:
            try:
                out.append(self._pending.popleft())
            except IndexError:
                return out

    def has_pending(self) -> bool:
        return bool(self._pending)

    def live_count(self) -> int:
        return sum(len(t) for t in self._live.values())

    # ------------------------------------------------------------------ internals

    def _null_alloc(self, device: str) -> int:
        a = self._null.get(device)
        if a is None:
            with self._lock:
                a = self._next
                self._next += 1
                self._null[device] = a
                self._pending.append(StorageAlloc(a, 0, 0, device, 0, "null"))
        return a

    def _overlapping_locked(self, device: str, ptr: int, nbytes: int) -> list[_Live]:
        table = self._live.get(device)
        starts = self._starts.get(device)
        if not table or not starts:
            return []
        end = ptr + nbytes
        out: list[_Live] = []
        i = bisect.bisect_right(starts, ptr) - 1
        if i >= 0:
            e = table[starts[i]]
            if e.ptr < ptr < e.ptr + e.nbytes:
                out.append(e)
        j = bisect.bisect_right(starts, ptr)
        while j < len(starts) and starts[j] < end:
            out.append(table[starts[j]])
            j += 1
        return out

    def _free_locked(self, e: _Live, *, implied: bool) -> None:
        table = self._live.get(e.device)
        if table is not None and table.get(e.ptr) is e:
            del table[e.ptr]
            starts = self._starts[e.device]
            i = bisect.bisect_left(starts, e.ptr)
            if i < len(starts) and starts[i] == e.ptr:
                del starts[i]
        if e.key is not None and self._by_key.get(e.key) is e:
            del self._by_key[e.key]
        e.ref = None
        self.stats["frees_implied" if implied else "frees_direct"] += 1
        self._pending.append(StorageFree(e.alloc, implied))

    def _make_death_callback(self, entry: _Live):
        def _died(_ref, _entry=entry, _self=self) -> None:
            with _self._lock:
                if _entry.ref is not None:          # still live in our table
                    _self._free_locked(_entry, implied=False)
        return _died


def _probe_weakrefs() -> bool:
    """True when this torch preserves storage PyObjects and they support weakrefs (both needed for exact frees)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    try:
        t = torch.empty(2)
        s = t.untyped_storage()
        if s is not t.untyped_storage():
            return False
        weakref.ref(s)
        return True
    except Exception:  # noqa: BLE001
        return False
