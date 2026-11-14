"""Static slots: the values of a step vLLM replays from a CUDA graph, read with no Python inside the replay.

A step replayed from a CUDA graph runs no Python for its captured work, so a value of that work is read from a slot: a
tensor allocated once, into which the capture records a device copy that every replay issues again.

  * `Slots.capture_wrapper` wraps `CudaGraphManager.capture`, through which vLLM captures every graph: around each
    capture's forwards `capturing` is its (mode, padded tokens) and `plans` holds the copies it recorded.
  * `Slots.stage` copies a value into its slot, one per (order, name) with as many rows as the largest capture size
    for a value of one row a token, and one per (order, name, shape) otherwise. A slot is allocated in the uncaptured
    warm-up forward vLLM runs before each capture, outside the graph's memory pool, so the capture records only the
    copy. With ``defer`` the copy is held in `pending` and `flush` issues the held copies together.
  * `Slots.entries` is a replayed step's plan with its live copies, in order, and `Slots.value` cuts an entry to the
    step: the first n rows of a token-row slot (rows n.. are the graph's padding, never a value), the whole slot
    otherwise.

vLLM's memory-profiling captures are discarded, and so are their slots (`end_profiling_capture`).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

__all__ = ["GRAPH_MODES", "WORDS", "Entry", "Slots", "stream_capturing"]

#: the modes of a step vLLM replays from a graph
GRAPH_MODES = ("FULL", "PIECEWISE")
#: the words a flush copies by, widest first
WORDS = ((8, torch.int64), (4, torch.int32), (2, torch.int16), (1, torch.uint8))

#: a staged copy: its order, its name, its slot, and whether the slot holds one row a token
Entry = tuple[tuple[Any, ...], str, torch.Tensor, bool]


def stream_capturing(val: torch.Tensor) -> bool:
    """Whether work on ``val`` issued now is recorded into a CUDA graph."""
    return val.is_cuda and torch.cuda.is_current_stream_capturing()


class Slots:
    def __init__(self, errors: list[str] | None = None, stats: dict[str, Any] | None = None,
                 dropped: Callable[[], None] | None = None) -> None:
        self.errors: list[str] = errors if errors is not None else []
        self.stats: dict[str, Any] = stats if stats is not None else {}
        for k in ("slot_bytes", "slot_resets", "profiling_captures", "capture_flushes", "strided_copies"):
            self.stats.setdefault(k, 0)
        self.dropped = dropped
        self.max_rows = 0
        self.slots: dict[tuple[Any, ...], torch.Tensor] = {}
        self.static: set[tuple[Any, ...]] = set()       # (order, name) captured once with a leading dim other than the padded size
        self.plans: dict[tuple[str, int], list[Entry]] = {}
        self.capturing: tuple[str, int] | None = None
        self.pending: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_wrapper(self, mine: Callable[[Any], bool] | None = None) -> Callable[[Any], Any]:
        """The wrapper of `CudaGraphManager.capture`; with ``mine``, only a manager it accepts is followed."""
        slots = self

        def capture(orig: Any) -> Any:
            def run(manager: Any, create_forward_fn: Any, *a: Any, **kw: Any) -> Any:
                if mine is not None and not mine(manager):
                    return orig(manager, create_forward_fn, *a, **kw)
                descs = [d for ds in getattr(manager, "_capture_descs", {}).values() for d in ds]
                slots.begin_capture(max((int(d.num_tokens) for d in descs), default=0))

                def create(desc: Any, warmup: bool) -> Any:
                    fn = create_forward_fn(desc, warmup=warmup)
                    key = (desc.cg_mode.name, int(desc.num_tokens))

                    def forward_fn(cg_mode: Any) -> Any:
                        slots.plans[key] = []
                        slots.capturing = key
                        try:
                            return fn(cg_mode)
                        finally:
                            slots.capturing = None
                            slots.end_capture(key)
                    return forward_fn
                try:
                    return orig(manager, create, *a, **kw)
                finally:
                    if getattr(manager, "_capture_mem_samples", None) is not None:
                        slots.end_profiling_capture()       # vLLM's memory profiling: its graphs are discarded, so are their slots
            return run
        return capture

    def begin_capture(self, rows: int) -> None:
        if rows > self.max_rows:
            if self.slots:
                self.stats["slot_resets"] += 1
            self.drop()
            self.max_rows = rows

    def end_capture(self, key: tuple[str, int]) -> None:
        if self.pending:
            self.errors.append(f"capture {key[0]}:{key[1]}: {len(self.pending)} captured copies never flushed")
            self.pending = []

    def end_profiling_capture(self) -> None:
        self.drop()
        self.max_rows = 0
        self.stats["profiling_captures"] += 1

    def drop(self) -> None:
        self.slots, self.plans, self.static, self.pending = {}, {}, set(), []
        self.stats["slot_bytes"] = 0
        if self.dropped is not None:
            self.dropped()

    def stage(self, order: tuple[Any, ...], name: str, val: torch.Tensor, n_pad: int, defer: bool = False) -> Entry:
        rows = val.dim() > 0 and int(val.shape[0]) == n_pad and n_pad <= self.max_rows
        if not rows:
            self.static.add((order, name))
        key = (order, name) if rows else (order, name, tuple(val.shape))
        slot = self.slots.get(key)
        if slot is None or slot.dtype != val.dtype or tuple(slot.shape[1:]) != tuple(val.shape[1:]):
            if slot is not None:
                self.errors.append(f"{name}: slot {tuple(slot.shape)}/{slot.dtype} reused for {tuple(val.shape)}/{val.dtype}")
            shape = (self.max_rows, *val.shape[1:]) if rows else tuple(val.shape)
            slot = torch.empty(shape, dtype=val.dtype, device=val.device)
            self.slots[key] = slot
            self.stats["slot_bytes"] += slot.numel() * slot.element_size()
        dst = slot[: val.shape[0]] if rows else slot
        if defer:
            self.pending.append((dst, val))                 # holds `val`, so its storage outlives its owner's `del`
        else:
            dst.copy_(val)
        return order, name, slot, rows

    def flush(self) -> None:
        """Issue the held copies: the dense ones as one multi-tensor copy per word width (the widest that divides a copy's
        length and both its addresses), the strided ones one by one."""
        groups: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for d, v in self.pending:
            if not (d.is_contiguous() and v.is_contiguous()):
                d.copy_(v)
                self.stats["strided_copies"] += 1
                continue
            nb = v.numel() * v.element_size()
            word = next(dt for w, dt in WORDS if nb % w == 0 and d.data_ptr() % w == 0 and v.data_ptr() % w == 0)
            dst, src = groups.setdefault(word, ([], []))
            dst.append(d.reshape(-1).view(word))
            src.append(v.reshape(-1).view(word))
        for dst, src in groups.values():
            torch._foreach_copy_(dst, src)
        self.stats["capture_flushes"] += 1
        self.pending = []

    def entries(self, mode: str, pad: int, live: list[Entry] | None = None) -> list[Entry] | None:
        """The copies of a step replayed in ``mode`` at ``pad`` tokens, with its ``live`` ones, in order; None where no
        capture recorded a plan for it."""
        plan = self.plans.get((mode, pad))
        return None if plan is None else sorted(plan + (live or []), key=lambda e: e[0])

    def value(self, entry: Entry, n: int, pad: int) -> torch.Tensor:
        """An entry's value at a step of ``n`` tokens replayed at ``pad``."""
        order, name, slot, rows = entry
        return slot if not rows else slot[:pad] if (order, name) in self.static else slot[:n]
