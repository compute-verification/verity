"""What a `NativeHostCommitter`'s serving thread hands its worker, in order (each step's `_StepJob`, and with window views on a
`_WindowEnd` behind a detached window's last step), and the window views (`WindowViews`): a Commit window finalized while the next one
runs (`serve_commit.CommitWindows`).

With views on, each window's steps commit into a view of their own: a shallow copy of the committer taken at the window's first step,
after its reset, so it holds that window's per-step dicts, lists and spans while `reset` rebinds the committer's for the next window.
The worker commits each job into its `_StepJob.com`, so a step it absorbs after its window was detached still lands in its own window.
The view's finalize waits for its own window's end (`_WindowEnd`), which the worker reaches after the window's last step: there the
pinned blocks the window's values sit in are lent to the view (until `serve_seal.thaw`) and the pool is reset for the next window's
steps, in the worker's order.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _StepJob:
    step: int
    metas: list                             # the step's `native_host.TensorMeta`s, in tree order
    event: Any
    t_step_end: float = 0.0
    backlog: int = 0
    blocks: list = field(default_factory=list)
    arena: Any = None                       # arena staging: current device block
    cursor: int = 0                         # arena staging: bytes used in `arena`
    overflow: list = field(default_factory=list)   # arena staging: (block, used_bytes) of earlier, filled blocks of this step
    stream: Any = None                      # packed mode: memoryview of the whole packed step stream (pinned host copy)
    stream_handle: Any = None               # fast path: raw cudaStream_t of the compute stream (cached per step)
    arena_ptr: Any = None                   # fast path: data_ptr of job.arena (cached)
    host_pre: Any = None                    # inline-D2H: pinned host block already receiving the packed step (issued on the compute stream)
    acq_s: float = 0.0                      # fast path: serving-thread seconds inside the hooks (acquisition + memcpy issue), this step
    acq_bytes: int = 0
    deferred_dst: list = field(default_factory=list)   # batched acquisition: arena views awaiting the step-end _foreach_copy_
    deferred_src: list = field(default_factory=list)   # ... and their (held) sources
    com: Any = None                         # window views: the view of the step's window the worker commits it into (None: the committer)
    seq: int = -1                           # the step's id among every step the committer began (the collector's key for its windows)


@dataclass
class _WindowEnd:
    """Queued behind a detached window's last step (`WindowViews.detach`): the worker ends the window there."""
    view: Any
    ended: threading.Event


def window_source(src: Any) -> Any:
    """A source as its detached window's finalize and readers see it: a shallow copy with its stats (synced first, then frozen:
    ``stats_frozen``) and its rows (`served_record.SamplerInputsSource`) copied, which the next window's steps and `begin_run` rebind
    or reset."""
    if hasattr(src, "sync_stats"):
        src.sync_stats()
    c = copy.copy(src)
    c.stats_frozen = True
    for k in ("stats", "rows"):
        v = getattr(src, k, None)
        if isinstance(v, dict):
            setattr(c, k, dict(v))
    return c


class WindowViews:
    """`NativeHostCommitter`'s window views (the module's docstring): its serving thread detaches each window, its worker ends it."""

    #: what a view shares with the committer once either has made it: one of each for the process
    SHARED_LAZY = ("_side", "_gpu_ext", "_HG")

    def _init_views(self) -> None:
        # the open window's view, whether the last window was detached, the windows detached (serving thread) and ended by the worker
        # (`_WindowEnd`), whose difference is how far behind the worker is, and the steps begun (`_StepJob.seq`)
        self.window_views = False
        self._view: Any = None
        self._detached = False
        self._windows_detached = self._windows_ended = 0
        self.staged_bytes = 0                       # the bytes the open window's steps staged (their step ends'), serving thread
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq - 1

    def _reset_views(self) -> None:
        """The committer's reset: the pinned pool is recycled here unless the window was detached, whose end resets it in the
        worker's order (`_end_window`)."""
        if not self._detached:
            self.pool.reset()
        self._detached, self._view, self.staged_bytes = False, None, 0

    def enable_window_views(self) -> None:
        """Commit each window into a view of its own (`detach`). The serving thread must take nothing from the pinned pool (the window's
        end resets it on the worker): refused under direct staging and inline D2H, whose step ends take blocks."""
        if self.staging != "arena" or self.inline_d2h:
            raise ValueError(f"window views: the serving thread takes pinned blocks under staging={self.staging!r}"
                             f"{' with inline D2H' if self.inline_d2h else ''}; only arena staging leaves the pool to the worker")
        self.window_views = True

    def _window_view(self) -> Any:
        if not self.window_views:
            return None
        if self._view is None:
            self._view = copy.copy(self)
            self._view.installed = True             # guard: never install()/uninstall() a view
            for k in ("_device_pool_key", "released"):   # the window's own pool pieces and release (`device_pool`)
                self._view.__dict__.pop(k, None)
        return self._view

    def detach(self) -> Any:
        """The open window handed off with its last step ended: its view, holding what the serving thread recorded of it, whose
        `finalize` waits for the worker to absorb the window's steps; None for a window with no step. The next `reset` leaves the
        pinned pool to the window's end."""
        self._end_step()
        view, self._view = self._view, None
        if view is None:
            return None
        self._detach_into(view)
        view._ended = threading.Event()
        self._detached = True
        self._windows_detached += 1
        self._q.put(_WindowEnd(view, view._ended))
        return view

    def _detach_into(self, view: Any) -> None:
        """What the serving thread recorded of the window that the next window's reset rebinds or clears, into its view."""
        view._step_index = self._step_index
        view.staged_bytes = self.staged_bytes
        view.x03_expected, view.x03_committed = self.x03_expected, self.x03_committed
        view.gc_pauses = copy.copy(self.gc_pauses)          # its lists: the reset rebinds the committer's
        view.sources = [window_source(s) for s in self.sources]

    def _absorb_job(self, job: Any) -> None:
        """Worker: end a detached window, or commit a step into its window's view (`_absorb_step`), the handles either made shared."""
        if isinstance(job, _WindowEnd):
            return self._end_window(job)
        com = job.com if job.com is not None else self
        self._share_lazy(self, com)
        try:
            com._absorb_step(job)
        finally:
            self._share_lazy(com, self)

    def _share_lazy(self, src: Any, dst: Any) -> None:
        for k in self.SHARED_LAZY:
            v = getattr(src, k, None)
            if v is not None and getattr(dst, k, None) is None:
                setattr(dst, k, v)

    def _end_window(self, end: _WindowEnd) -> None:
        """Worker: every step of ``end``'s window absorbed. Its pinned blocks go to its view, the pool is reset for the next window."""
        try:
            end.view.lent = self.pool.lend()
            self.pool.reset()
            self._window_ended(end.view)
        finally:
            self._windows_ended += 1
            end.ended.set()

    def _window_ended(self, view: Any) -> None:
        """Worker, at a detached window's end: what else the window's end hands over or resets (the collector's plans)."""

    def wait_absorbed(self) -> float:
        """A detached window's view: block until the worker has ended the window (every one of its steps absorbed); the seconds."""
        ended = getattr(self, "_ended", None)
        if ended is None:
            return 0.0
        t0 = time.perf_counter()
        ended.wait()
        return time.perf_counter() - t0
