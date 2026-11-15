"""The Pearl-C device path's CUDA graphs (``graphs``, `pouw_pearl_c_device.PearlCDeviceRun`): each decode call's steps are
captured once a pipeline and replayed on the caller's stream.  A graph reads its call's keys and pass offsets from its
weight's device slot, which a forward's first call fills for the whole forward in one upload (a call the prediction missed
uploads its own), and in its last launch screens its rows into the pass's flags and copies its leaves and A root there.  The same kernels write the same bytes as
the eager steps, which the e2e's ``graphs`` gate holds.

Under a schedule with a side lane (`pouw_pearl_c_lanes`), a call of decode's rows keeps its lanes and ring but replays each
lane's steps as one graph (`_lane_graph`), one pair a pipeline and ring set: the main lane's, ending with a scatter of its
flags (and A's root, where A's tree stays there), and the side lane's, ending with a scatter of the leaves; where the tree
runs off the main lane, the main graph records an external event after the per-row keying, and the tree, ending with a
scatter of A's root, is a graph of its own, which the side lane running A's tree (`pouw_pearl_c_lanes.TREE_APART_ROWS`)
replays behind that event.  The main
graph runs ``stats_a`` (and the screen's scatter, where A's tree is aside) on a branch of its own beside the per-row
keying, joined before the forming reads A's scales and at the graph's end.  Every graph reads
the pair's own slot, the call's keys and offsets in one upload, which waits for the side steps of the set's last call, as
the set's other buffers do.  The driver replays them (`_Driver`, on the graphs' executables and the streams' and
events' handles), so the call's host work is a few launches, and the next call's main lane runs beside this one's side
lane, as eagerly.

Inside vLLM's own capture of a whole decode forward, a call launches the same steps into that capture instead
(`pouw_pearl_c_whole_calls`, `GraphCalls`' base `WholeCalls`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verity_vllm.protocols.pouw.pearl_c.ends import _RunState
from verity_vllm.protocols.pouw.pearl_c.host import _event, _Lane, _zeros, torch_uint8
from verity_vllm.protocols.pouw.pearl_c.screen import scatter_into
from verity_vllm.protocols.pouw.pearl_c.whole_calls import WholeCall, WholeCalls

#: with ``graphs``, calls of at most this many rows (decode's) replay their steps as one CUDA graph; above it the launches are
#: a small share of the call, and a graph's own y would cost m x n of memory per weight
GRAPH_ROWS = 256
#: with ``graphs``, each weight's slot of per-call inputs its graph reads: dom_a ‖ unit (168 bytes, padded), and
#: the byte offsets of the call's leaves, flags and A root in the pass's buffers (three int64)
SLOT_BYTES = 176
#: a lane graph's slot (`_lane_graph`): dom_a ‖ unit padded to `SLOT_BYTES`, then the call's three offsets
LANE_SLOT_BYTES = SLOT_BYTES + 24


def _capturing() -> bool:
    import torch
    return torch.cuda.is_current_stream_capturing()


@dataclass
class _Graph:
    """One pipeline's call steps captured as a CUDA graph, over buffers of its own where a call's differ: y, and the call's
    tile leaves, which are copied to the pass's leaf buffer at the call's offset after each replay."""
    y: Any
    leaves: Any
    graph: Any = None
    steps: list = field(default_factory=list)
    #: where a deferred whole capture splits `steps` (`_call_cuts`): the end of A's tree, the start of the tile hashing
    cuts: tuple[int, int] = (0, 0)
    #: where a deferred-a whole capture forks A's tree (`_call_tree`: ``tree_seed_a``, the step before ``cuts[0]``), or None
    #: where the call's pipeline doesn't split A's commitment
    tree_at: int | None = None
    #: where a whole capture forking A's scales takes ``stats_a`` from (`_call_stats`, between the cuts), or None
    stats_at: int | None = None
    #: a lane graph's (`_lane_graph`): the side lane's graph and steps, and the slot both lanes read (dom_a ‖ unit, then the
    #: call's offsets at `SLOT_BYTES`, `offs`; its address `slot_ptr`).  Where A's tree is off the main lane, the main
    #: graph records `ev_tree` (an external event) before step `cut`, and the side steps before `side_cut` (A's tree and
    #: its root's scatter) are a graph of their own, `tree`, for the side lane running A's tree.  `execs` are the graphs'
    #: executables the driver launches (main, tree or 0, side), once captured
    side: Any = None
    side_steps: list = field(default_factory=list)
    slot: Any = None
    offs: Any = None
    slot_ptr: int = 0
    tree: Any = None
    cut: int | None = None
    side_cut: int = 0
    ev_tree: Any = None
    execs: tuple = ()
    #: the byte a poison left in `y` and `leaves`, until a call takes the graph (`PassEnds._poison_graphs`)
    poisoned: int | None = None


class GraphCalls(WholeCalls, _RunState):
    """`PearlCDeviceRun`'s graph path, as its base class: the run's ``graphs``, ``driver``, ``fork``, ``order``, ``nb`` (its
    trees' node bytes), ``key_slots``, ``offs``, ``_host_keys``, ``_host_offs``, ``_slots_copied``, ``_predicted`` and
    ``_graphs``, ``shapes``, its `_call_steps` and `_call_cuts`, and its pass and the pass's buffers (``roots_buf``, ``flag_buf``, whose
    lengths bound it)."""

    def _graph_call(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int, index: int, caller: Any) -> tuple[Any, bool]:
        """A graphed call: inside vLLM's capture of a whole forward, its steps launched into that capture (and in a retained
        pass's, the copies to its stage, once its A tree is done: A's words and tree, and y's m real rows); otherwise its own
        graph replayed after its slot.  vLLM's warm-up forward before each capture runs outside it, and makes the graphs and
        stages."""
        if self._whole is not None and _capturing():
            self._whole.append(WholeCall(w, m, sh.k, sh.n, sh.m16, tiles))
            out = self._into(w, sh, pipe, raw, tiles, m)
            if self._whole_retained:
                self._tree_joined()
                xa, tree, y = self._stages[w, sh.m16]
                for stage, t in zip((xa, tree), self._kept(sh, pipe), strict=True):
                    stage.copy_(t)
                y[:m].copy_(out[0][:m])
            if self._statement_on():
                self.statement.stage(w, sh)
            return out
        if self._whole is not None and self._whole_retained and (w, sh.m16) not in self._stages:
            import torch
            self._stages[w, sh.m16] = (*(t.new_empty(t.shape) for t in self._kept(sh, pipe)),
                                       sh.xa.new_empty((sh.m16, sh.n), dtype=torch.bfloat16))
            self._stages_poisoned = None
        if self._whole is not None and self._statement_on():
            self.statement.prepare(w, sh)
        if self.hashing:
            self._slot(w, index, sh.m16, m, caller)
        return self._replay(w, sh, pipe, raw, tiles, m)

    @staticmethod
    def _kept(sh: Any, pipe: Any) -> tuple[Any, Any]:
        """What a retained call keeps of its shape's buffers once its steps ran: A's words (m₁₆ rows) and A's tree."""
        return sh.xa[:sh.m16], sh.buf("tree_a")[:pipe.size["tree_a"]]

    def _widen_later(self, sh: Any) -> bool:
        """Whether a call of `sh` leaves its A words' widening (`LaneCalls._stage_a`) to its `_into`: inside a whole capture
        forking A's scales, at a shape that replays as a graph, which is where the call goes to `_into`."""
        return self._forks_stats() and self._whole is not None and self._graphed(sh.m16) and _capturing()

    def _graphed(self, m16: int) -> bool:
        """Whether calls of m₁₆ rows replay as graphs (`graphs`, the serial schedule), reading their keys and offsets from
        their weight's slot."""
        return self.graphs and m16 <= GRAPH_ROWS and not self.schedule.side and self.schedule.main_priority != "greatest"

    def _lane_graphed(self, sh: Any, side: list) -> bool:
        """Whether a call of `sh` replays each lane's steps as a graph (`graphs`, a side lane, at most `GRAPH_ROWS` rows,
        the GEMM untimed)."""
        return self.graphs and bool(side) and sh.m16 <= GRAPH_ROWS and not self.time_gemm

    def _lane_graph(self, w: int, sh: Any, pipe: Any, steps: list, side: list, m: int, tiles: int,
                    aside: bool) -> tuple[_Graph, bool]:
        """A call's lane graphs on the ring set it took (`sh.j`), and whether they screen it (`_graph_key`'s rule: no filler
        rows).  Made at their first call, over buffers of their own for y, the leaves and the slot, with `pipe` pointed at
        those (the caller points it at the set's): the main lane's steps, then (when it screens, or A's tree stays on the
        main lane) one scatter of the flags and A's root; the side lane's steps, then one scatter of the leaves; each
        scatter at the offsets in the slot.  Where A's tree is off the main lane (`aside`), the main graph records an
        external event after the per-row keying, and the tree, then a scatter of A's root, is a graph of its own for the
        side lane running A's tree (the pipeline's ``lane_cuts``).  That call runs the steps eagerly, and `_capture_lanes`
        captures them."""
        import torch
        rows = m if m == sh.m16 else 0
        key = ("lanes", w, sh.m16, self.omit, self.hashing, self.schedule.name, rows, aside, sh.j)
        g = self._graphs.get(key)
        if g is None:
            g = _Graph(torch.empty((sh.mp, pipe.n), dtype=torch.bfloat16, device=self.device), _zeros(self.nb * tiles, self.device))
            g.slot = _zeros(LANE_SLOT_BYTES, self.device)
            g.offs, g.slot_ptr = g.slot[SLOT_BYTES:LANE_SLOT_BYTES].view(torch.int64), g.slot.data_ptr()
            if aside:
                g.cut, g.side_cut = pipe.lane_cuts
                g.ev_tree = torch.cuda.Event(external=True)
            if self.cur.poison is not None:
                g.y.view(torch.uint8).fill_(self.cur.poison)
                g.leaves.fill_(self.cur.poison)
            pipe.b["y"], pipe.b["leaves"] = g.y.data_ptr(), g.leaves.data_ptr()
            size, nb = pipe.size["tree_a"], self.nb
            root, flags, none, leaves = sh.buf("tree_a")[size - nb:size], self.flag_buf.view(-1), g.leaves[:0], g.leaves[:nb * tiles]
            xa, alpha, aux = sh.xa, sh.b["alpha_a"], sh.b["aux_a"]
            screen = (lambda: scatter_into(xa, alpha, aux, rows, flags, none, self.leaves, none if aside else root,
                                           self.roots_buf, g.offs)) if rows or not aside else None
            g.steps, shift = self._main_steps(pipe, list(steps), screen, aside)
            if g.cut is not None:
                g.cut += shift
            g.side_steps = [*side, lambda: scatter_into(xa, alpha, aux, 0, flags, leaves, self.leaves, none, self.roots_buf,
                                                        g.offs)]
            if aside:
                g.side_steps.insert(g.side_cut, lambda: scatter_into(xa, alpha, aux, 0, flags, none, self.leaves, root,
                                                                     self.roots_buf, g.offs))
                g.side_cut += 1
            self._graphs[key] = g
        g.poisoned = None
        return g, rows > 0

    def _main_steps(self, pipe: Any, steps: list, screen: Any, aside: bool) -> tuple[list, int]:
        """A lane graph's main steps, and how far they moved the per-row keying's end: `steps` with ``stats_a`` (A's
        scales, which read only A's words) on the run's fork lane beside the steps before it (`_forked`), and with it the
        scatter of the screen (`screen`) where that reads nothing else of the call (A's tree aside); else `screen` last."""
        names = getattr(pipe, "main_names", ())
        if self.driver is None or "stats_a" not in names:
            return [*steps, *([screen] if screen else [])], 0
        at = names.index("stats_a")
        beside = [steps[at], *([screen] if screen and aside else [])]
        out = self._forked([*steps[:at], *steps[at + 1:]], beside, at)
        return [*out, *([screen] if screen and not aside else [])], 1

    def _forked(self, steps: list, beside: list, at: int) -> list:
        """`steps`, launched on whichever stream the kernel's device is on (a lane, or a capture of it), with `beside` on
        the run's fork lane (``fork``, the main lane's priority): it starts behind an event at the start, the lane waits for
        its first step before its own step `at`, and for all of them at its end.  The events go through the driver on raw
        handles, so a capture takes them as the graph's edges and the fork as a branch of it."""
        import torch
        if self.fork is None:
            self.fork = _Lane(self.device, "greatest", -2)
        D, dev, fork = self.driver, self.kernel.dev, self.fork
        assert D is not None
        start, first, end = _event(), _event(), _event()

        def go():
            lane = dev.stream
            D.record(start, lane)
            D.wait(fork.raw, start)
            dev.stream = fork.raw
            try:
                with torch.cuda.stream(fork.stream):
                    beside[0]()
                    D.record(first, fork.raw)
                    for f in beside[1:]:
                        f()
                    D.record(end, fork.raw)
            finally:
                dev.stream = lane
        return [go, *steps[:at], lambda: D.wait(dev.stream, first), *steps[at:], lambda: D.wait(dev.stream, end)]

    def _capture_lanes(self, g: _Graph) -> None:
        """`g`'s lanes' steps captured as its graphs (the main lane's, recording `ev_tree` before step `cut` where A's tree
        is off it, then the tree's, and the side lane's), after the call that made it ran them eagerly (their first use
        syncs: the GEMMs' shared-memory probes), and their executables for the driver."""
        import torch
        capture = torch.cuda.Stream(device=self.device)
        capture.wait_stream(torch.cuda.current_stream(self.device))
        parts = [(g.steps, g.cut)]
        if g.side_cut:
            parts.append((g.side_steps[:g.side_cut], None))
        parts.append((g.side_steps[g.side_cut:], None))
        made = []
        for steps, cut in parts:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(capture):
                graph.capture_begin(capture_error_mode="thread_local")
                self.kernel.dev.stream = capture.cuda_stream
                try:
                    for i, f in enumerate(steps):
                        if i == cut:
                            g.ev_tree.record(capture)
                        f()
                finally:
                    graph.capture_end()
            made.append(graph)
        torch.cuda.current_stream(self.device).wait_stream(capture)
        g.graph, g.side, g.tree = made[0], made[-1], made[1] if g.side_cut else None
        g.execs = (g.graph.raw_cuda_graph_exec(), g.tree.raw_cuda_graph_exec() if g.tree else 0, g.side.raw_cuda_graph_exec())

    def _offsets(self, index: int, tiles_at: int, flags_at: int) -> tuple[int, int, int]:
        """Call `index`'s byte offsets in the pass's buffers: its leaves, its flags, its A root (at its position in the pass,
        the index past the pass's base)."""
        return self.nb * tiles_at, 2 * flags_at, self.nb * (index - self.cur.first)

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
                kv = self.keys(self.roots[v], index + i, m16)
                keys[i, :len(kv)] = np.frombuffer(kv, np.uint8)
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
        own for y and the leaves.  Returns y, and whether the graph screened the call (`_graph_key`); the leaves, and the
        flags when it did, go to the pass's buffers at the call's offsets."""
        import torch
        key, screen = self._graph_key(w, sh, m)
        g = self._graphs.get(key)
        if g is None:
            g = _Graph(torch.empty((sh.mp, pipe.n), dtype=torch.bfloat16, device=self.device), _zeros(self.nb * tiles, self.device))
            if self.cur.poison is not None:
                g.y.view(torch.uint8).fill_(self.cur.poison)
                g.leaves.fill_(self.cur.poison)
            pipe.b["y"], pipe.b["leaves"] = g.y.data_ptr(), g.leaves.data_ptr()
            steps = self._call_steps(pipe)
            g.cuts, g.tree_at, g.stats_at = self._call_cuts(pipe), self._call_tree(pipe), self._call_stats(pipe)
            if self.hashing:
                steps = [*steps, self._scatter(w, sh, pipe, g, tiles, m if screen else 0)]
            g.steps = steps
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
        g.poisoned = None
        return g.y.clone(), screen

    def _graph_key(self, w: int, sh: Any, m: int) -> tuple[tuple, bool]:
        """A call's graph and whether it screens the call's rows (with hashing).  A graph of m₁₆ rows serves every m it rounds
        up from, so it screens only when there are no filler rows, and `_after` screens the rest; inside a whole capture
        (from its warm-up on) vLLM's batch fixes m, so the graph is m's own and screens it, as no `_after` runs at replay.  A
        deferred-a capture's graphs (`_defers_a`) are its own: their pipelines split A's commitment (`_split_whole`), and a
        graph made unsplit would keep A's tree on the main lane."""
        rows = m if self.hashing and (m == sh.m16 or self._whole is not None) else 0
        return (w, sh.m16, self.omit, self.hashing, self.schedule.name, rows, self._defers_a()), rows > 0

    def _scatter(self, w: int, sh: Any, pipe: Any, g: _Graph, tiles: int, flag_rows: int):
        """The graph's last step, one launch: the screen of the call's first `flag_rows` rows (all of them when it screens,
        else none), into the pass's flags, and the call's leaves and A root into the pass's buffers, at the offsets in its
        weight's slot."""
        size, offs = pipe.size["tree_a"], self.offs[self.order[w]]
        root, flags, leaves = sh.b["tree_a"][size - self.nb:size], self.flag_buf.view(-1), g.leaves[:self.nb * tiles]

        def scatter():
            scatter_into(sh.xa, sh.b["alpha_a"], sh.b["aux_a"], flag_rows, flags, leaves, self.leaves, root, self.roots_buf, offs)
        return scatter
