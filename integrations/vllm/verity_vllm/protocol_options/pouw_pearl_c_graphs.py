"""The Pearl-C device path's CUDA graphs (``graphs``, `pouw_pearl_c_device.PearlCDeviceRun`): each decode call's steps are
captured once a pipeline and replayed on the caller's stream.  A graph reads its call's keys and pass offsets from its
weight's device slot, which a forward's first call fills for the whole forward in one upload (a call the prediction missed
uploads its own), and scatters its leaves, flags and A root into the pass itself.  The same kernels write the same bytes as
the eager steps, which the e2e's ``graphs`` gate holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pouw_pearl_c_host import _zeros, torch_uint8
from .pouw_pearl_c_screen import copy_kernel, screen_into

#: with ``graphs``, calls of at most this many rows (decode's) replay their steps as one CUDA graph; above it the launches are
#: a small share of the call, and a graph's own y would cost m x n of memory per weight
GRAPH_ROWS = 256
#: with ``graphs``, each weight's slot of per-call inputs its graph reads: dom_a ‖ unit (136 bytes, padded), and the byte
#: offsets of the call's leaves, flags and A root in the pass's buffers (three int64)
SLOT_BYTES = 144


@dataclass
class _Graph:
    """One pipeline's call steps captured as a CUDA graph, over buffers of its own where a call's differ: y, and the call's
    tile leaves, which are copied to the pass's leaf buffer at the call's offset after each replay; with hashing and no filler
    rows, the screen's flags too (``flags``), copied likewise."""
    y: Any
    leaves: Any
    flags: Any = None
    graph: Any = None


class GraphCalls:
    """`PearlCDeviceRun`'s graph path, as its base class: the run's ``graphs``, ``order``, ``key_slots``, ``offs``,
    ``_host_keys``, ``_host_offs``, ``_slots_copied``, ``_predicted`` and ``_graphs``, its `_call_steps`, and its pass."""

    def _graphed(self, m16: int) -> bool:
        """Whether calls of m₁₆ rows replay as graphs (`graphs`, the serial schedule), reading their keys and offsets from
        their weight's slot."""
        return self.graphs and m16 <= GRAPH_ROWS and not self.schedule.side and self.schedule.main_priority != "greatest"

    def _offsets(self, index: int, tiles_at: int, flags_at: int) -> tuple[int, int, int]:
        """A call's byte offsets in the pass's buffers: its leaves, its flags, its A root."""
        return 32 * tiles_at, 2 * flags_at, 32 * index

    def _slot(self, w: int, index: int, m16: int, m: int, stream: Any) -> None:
        """The call's keys and offsets in its weight's slot, which its graph reads.  At a forward's first weight, every
        weight's slot for the rest of the forward is predicted (the next indices, the same m, the pass's running offsets) and
        uploaded in one copy; a call whose slot doesn't hold its own values (another order or m) gets them uploaded alone, so
        the bytes don't depend on the prediction."""
        import numpy as np
        cur, R = self.cur, self.kernel.run
        want = (index, m16, cur.tiles, cur.flag_rows)
        pos = self.order[w]
        if pos == 0:
            if self._slots_copied is not None:
                self._slots_copied.synchronize()
            keys, offs = self._host_keys.numpy(), self._host_offs.numpy()
            tiles_at, flags_at = cur.tiles, cur.flag_rows
            for v, i in sorted(self.order.items(), key=lambda kv: kv[1]):
                keys[i, :136] = np.frombuffer(self.keys(self.roots[v], index + i, m16), np.uint8)
                offs[i] = self._offsets(index + i, tiles_at, flags_at)
                self._predicted[v] = (index + i, m16, tiles_at, flags_at)
                tiles_at += R.cdiv(m16, R.TILE) * (self.weights[v].n // R.TILE)
                flags_at += m
            self.key_slots.copy_(self._host_keys, non_blocking=True)
            self.offs.copy_(self._host_offs, non_blocking=True)
            import torch
            self._slots_copied = self._slots_copied or torch.cuda.Event()
            self._slots_copied.record(stream)
        if self._predicted.get(w) != want:
            self.uploads.put(self.key_slots[pos], self.keys(self.roots[w], index, m16), stream)
            self.uploads.put(self.offs[pos].view(torch_uint8()), np.array(self._offsets(index, cur.tiles, cur.flag_rows),
                                                                             "<i8").tobytes(), stream)
            self._predicted[w] = want

    def _replay(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int) -> tuple[Any, bool]:
        """The call's steps as one CUDA graph on the caller's stream (`graphs`): captured at the pipeline's first call, after
        one eager run of the same steps (their first use syncs: the GEMMs' shared-memory probes), on buffers of the graph's
        own for y and the leaves.  Returns y, and whether the graph screened the call (with hashing, when it has no filler
        rows); the leaves, and the flags when it did, go to the pass's buffers at the call's offsets."""
        import torch
        screen = self.hashing and m == sh.m16
        key = (w, sh.m16, self.omit, self.hashing, self.schedule.name, screen)
        g = self._graphs.get(key)
        if g is None:
            g = _Graph(torch.empty((sh.mp, pipe.n), dtype=torch.bfloat16, device=self.device), _zeros(32 * tiles, self.device))
            if screen:
                g.flags = _zeros(2 * m, self.device)[:2 * m].view(m, 2)
            if self.cur.poison is not None:
                for t in (g.y.view(torch.uint8), g.leaves, g.flags):
                    if t is not None:
                        t.fill_(self.cur.poison)
            pipe.b["y"], pipe.b["leaves"] = g.y.data_ptr(), g.leaves.data_ptr()
            steps = self._call_steps(pipe)
            if screen:
                steps = [*steps, lambda: screen_into(g.flags, sh.xa[:m], sh.b["alpha_a"], sh.b["aux_a"], self.fused_screen)]
            if self.hashing:
                steps = [*steps, self._scatter(w, sh, pipe, g, tiles, m if screen else 0)]
            self.kernel.dev.stream = raw
            try:
                for f in steps:
                    f()
                capture = torch.cuda.Stream(device=self.device)
                capture.wait_stream(self.torch_stream(raw))
                g.graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(capture):
                    g.graph.capture_begin(capture_error_mode="thread_local")
                    self.kernel.dev.stream = capture.cuda_stream
                    try:
                        for f in steps:
                            f()
                    finally:
                        g.graph.capture_end()
                self.torch_stream(raw).wait_stream(capture)
            finally:
                self.kernel.dev.stream = None
            self._graphs[key] = g
        else:
            g.graph.replay()
        return g.y.clone(), screen

    def _scatter(self, w: int, sh: Any, pipe: Any, g: _Graph, tiles: int, flag_rows: int):
        """The graph's last step: the call's leaves, flags (when it screened) and A root into the pass's buffers, at the
        offsets in its weight's slot."""
        size, offs = pipe.size["tree_a"], self.offs[self.order[w]]
        root, flags, copy = sh.b["tree_a"][size - 32:size], self.flag_buf.view(-1), copy_kernel()

        def scatter():
            copy[(-(-32 * tiles // 1024),)](g.leaves, self.leaves, offs, 0, 32 * tiles, BLOCK=1024)
            if flag_rows:
                copy[(-(-2 * flag_rows // 1024),)](g.flags, flags, offs, 1, 2 * flag_rows, BLOCK=1024)
            copy[(1,)](root, self.roots_buf, offs, 2, 32, BLOCK=1024)
        return scatter
