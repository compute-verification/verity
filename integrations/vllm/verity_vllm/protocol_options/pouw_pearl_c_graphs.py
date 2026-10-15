"""The Pearl-C device path's CUDA graphs (``graphs``, `pouw_pearl_c_device.PearlCDeviceRun`): each decode call's steps are
captured once a pipeline and replayed on the caller's stream.  A graph reads its call's keys and pass offsets from its
weight's device slot, which a forward's first call fills for the whole forward in one upload (a call the prediction missed
uploads its own), and in its last launch screens its rows into the pass's flags and copies its leaves and A root there.  The same kernels write the same bytes as
the eager steps, which the e2e's ``graphs`` gate holds.

Inside vLLM's own capture of a whole decode forward (`pouw_pearl_c_whole`, between `whole_capture_begin` and
`whole_capture_end`), a call launches the same steps into vLLM's capture instead, and is recorded; at each replay of that
forward no call's Python runs, so the forward's host work is done at once, around vLLM's launch of the replay: before it,
`whole_replay` uploads its slots; after it, while the GPU runs the forward, `whole_replayed` appends its matmuls, tiles and
screened rows to the pass and stages the next forward's slots, which the next `whole_replay` uploads when the pass is where
they were staged for (else it makes them itself: the same bytes either way).  A retained pass's capture also copies each
call's A words and tree, as its steps leave them, to its weight's stage, from which `whole_replayed` writes the call's record
and files, as `_retained` writes them for a call made alone.

With ``whole_defer`` (and hashing), a whole capture forks each call's tile hashing and scatter, after its GEMM, onto a side
stream of the run's own, and the next call joins it after its own A tree, so the two run side by side in the forward's graph;
a call joins at once when the call before is of its shape (the side steps still read that shape's buffers), and the
forward's last call joins its own before vLLM's capture ends.  The same kernels write the same bytes, so the commitments are
the serial capture's.

With a `statement` enabled (and hashing), a whole capture also copies each call's A words to the statement's stage, and
`whole_replayed` launches its commitment of the forward's rows after the replay; its key says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c_work as PW

from .pouw_pearl_c_host import _bytes, _write, _zeros, torch_uint8
from .pouw_pearl_c_screen import excluded, scatter_into

#: with ``graphs``, calls of at most this many rows (decode's) replay their steps as one CUDA graph; above it the launches are
#: a small share of the call, and a graph's own y would cost m x n of memory per weight
GRAPH_ROWS = 256
#: with ``graphs``, each weight's slot of per-call inputs its graph reads: dom_a ‖ unit (168 bytes, padded), and
#: the byte offsets of the call's leaves, flags and A root in the pass's buffers (three int64)
SLOT_BYTES = 176


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


@dataclass(frozen=True)
class WholeCall:
    """One call of a forward vLLM captured whole: what its replay appends to the pass."""
    w: int
    m: int
    k: int
    n: int
    m16: int
    tiles: int


@dataclass(frozen=True)
class _Slots:
    """A forward's slots, for `calls` from the pass's counters `at` (matmuls, tiles, screened rows): each call's keys and
    offsets by its weight's slot position (``pos``), each weight's prediction as `_slot` reads it, and the counters after."""
    calls: tuple[WholeCall, ...]
    at: tuple[int, int, int]
    pos: Any
    keys: Any
    offs: Any
    predicted: dict
    after: tuple[int, int, int]


class GraphCalls:
    """`PearlCDeviceRun`'s graph path, as its base class: the run's ``graphs``, ``order``, ``nb`` (its trees' node bytes),
    ``key_slots``, ``offs``, ``_host_keys``, ``_host_offs``, ``_slots_copied``, ``_predicted`` and ``_graphs``, its `_call_steps` and `_call_cuts`, and
    its pass and the pass's buffers (``roots_buf``, ``flag_buf``, whose lengths bound it)."""

    #: while vLLM captures forwards (`whole_capture_begin`), the calls made inside its stream capture, in order; None otherwise
    _whole: list[WholeCall] | None = None
    #: whether those forwards are a retained pass's, whose capture copies each call's A words and tree to its stage
    _whole_retained: bool = False
    #: each weight's stage, (A words, A tree) as its call's steps leave them, made at the first retained capture
    _stages: dict[int, tuple[Any, Any]] | None = None
    #: the slots the last `whole_replay` uploaded, and those `whole_replayed` staged for the forward after it
    _uploaded: _Slots | None = None
    _next: _Slots | None = None
    #: whether a whole capture defers each call's tile hashing beside the next call (with hashing)
    whole_defer: bool = False
    #: inside a deferred whole capture, the shape whose call's side steps no call has joined yet, and the side stream
    _deferred: Any = None
    _side_stream: Any = None
    #: a commitment of each forward replayed whole beside the scheme's, while its ``enabled`` (with hashing): inside the
    #: capture, `stage(w, sh)` after each call's steps (and `prepare(w, sh)` outside it, in the warm-up); after each replay,
    #: `after(calls, tiles_at)` on the replaying stream (benchmarks/pouw/served_zk/statement.py's `Statement`)
    statement: Any = None

    def _graph_call(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int, index: int, caller: Any) -> tuple[Any, bool]:
        """A graphed call: inside vLLM's capture of a whole forward, its steps launched into that capture (and in a retained
        pass's, the copies to its stage); otherwise its own graph replayed after its slot.  vLLM's warm-up forward before each
        capture runs outside it, and makes the graphs and stages."""
        if self._whole is not None and _capturing():
            self._whole.append(WholeCall(w, m, sh.k, sh.n, sh.m16, tiles))
            out = self._into(w, sh, pipe, raw, tiles, m)
            if self._whole_retained:
                for stage, t in zip(self._stages[w], self._kept(sh, pipe), strict=True):
                    stage.copy_(t)
            if self._statement_on():
                self.statement.stage(w, sh)
            return out
        if self._whole is not None and self._whole_retained and w not in self._stages:
            self._stages[w] = tuple(t.new_empty(t.shape) for t in self._kept(sh, pipe))
        if self._whole is not None and self._statement_on():
            self.statement.prepare(w, sh)
        if self.hashing:
            self._slot(w, index, sh.m16, m, caller)
        return self._replay(w, sh, pipe, raw, tiles, m)

    @staticmethod
    def _kept(sh: Any, pipe: Any) -> tuple[Any, Any]:
        """What a retained call keeps of its shape's buffers once its steps ran: A's words (m₁₆ rows) and A's tree."""
        return sh.xa[:sh.m16], sh.b["tree_a"][:pipe.size["tree_a"]]

    def _retained(self, cur: Any, mm: Matmul, m: int, flags: Any, xa: Any, tree: Any) -> None:
        """A retained call's record (the rows its screen excluded, from its flags) and files (its rows as BF16, A's tree)."""
        import torch
        cur.records.append(PW.UnitRecord(mm, m, mm.shape.k, excluded(flags[:, 0].bool(), flags[:, 1].bool(), xa[:m]),
                                         self.weights[mm.weight].excluded))
        d = cur.retain / "calls"
        d.mkdir(parents=True, exist_ok=True)
        rows = _bytes(xa.to(torch.bfloat16).view(torch.uint8))
        cur.files.append({"rows": _write(d / f"{mm.index}.rows.bf16", rows, digest=False),
                          "tree": _write(d / f"{mm.index}.tree", _bytes(tree))})

    def whole_key(self, retained: bool | None = None) -> tuple | None:
        """What a forward captured whole depends on (hashing, omitted steps, schedule, with hashing whether its pass is
        retained, which `retained` sets in place of the open pass's, and whether it defers the tile hashing), or None when the
        pass can't replay one: graphs off, or a schedule the graph path doesn't run."""
        if not self.graphs or self.cur is None or self.schedule.side or self.schedule.main_priority == "greatest":
            return None
        retained = self.cur.retain is not None if retained is None else retained
        key = (self.hashing, self.omit, self.schedule.name, self.hashing and retained, self._defers())
        return (*key, "statement") if self._statement_on() else key

    def _defers(self) -> bool:
        """Whether a whole capture forks each call's tile hashing beside the next call (`whole_defer`, with hashing)."""
        return self.whole_defer and self.hashing

    def _statement_on(self) -> bool:
        """Whether forwards replayed whole commit `statement`'s rows too (it is set and enabled, with hashing)."""
        return self.statement is not None and self.statement.enabled and self.hashing

    def whole_capture_begin(self, key: tuple | None = None) -> None:
        """Calls are recorded from here, for the forwards of `key`'s kind (`whole_key`'s; an unretained pass's when None).
        A deferring run's side stream is made here, before vLLM's capture begins."""
        self._whole, self._whole_retained, self._deferred = [], bool(key and key[3]), None
        if self._whole_retained and self._stages is None:
            self._stages = {}
        if self._defers() and self._side_stream is None:
            import torch
            self._side_stream = torch.cuda.Stream(device=self.device)

    def whole_capture_end(self) -> tuple[WholeCall, ...]:
        """The calls captured since `whole_capture_begin`.  Each weight's slot holds one call's keys, so a weight called twice
        in one forward is refused."""
        calls, self._whole, self._whole_retained, self._deferred = tuple(self._whole or ()), None, False, None
        seen = [c.w for c in calls]
        if len(set(seen)) != len(seen):
            raise RuntimeError(f"{self.entry}: a weight is called twice in one captured forward, and its slot holds one call")
        return calls

    def _keep_whole(self, calls: tuple[WholeCall, ...], flags_at: int) -> None:
        """A retained pass's records and files of a forward replayed whole, its first call's flags at row `flags_at` of the
        pass's buffer: each call's from its stage and its flags, as a call made alone keeps them."""
        cur = self.cur
        for mm, c in zip(cur.matmuls[len(cur.matmuls) - len(calls):], calls, strict=True):
            self._retained(cur, mm, c.m, self.flag_buf[flags_at:flags_at + c.m], *self._stages[c.w])
            flags_at += c.m

    def _graph_buffers(self):
        """Every buffer the graphs write, as bytes: each graph's y and leaves, and each stage."""
        import torch
        for g in self._graphs.values():
            yield from (g.y.view(torch.uint8), g.leaves)
        for stage in (self._stages or {}).values():
            yield from (t.view(torch.uint8) for t in stage)

    def _slots(self, calls: tuple[WholeCall, ...], at: tuple[int, int, int]) -> _Slots:
        import numpy as np
        index, tiles_at, flags_at = at
        offs, predicted = np.empty((len(calls), 3), np.int64), {}
        for i, c in enumerate(calls):
            offs[i] = self._offsets(index + i, tiles_at, flags_at)
            predicted[c.w] = (index + i, c.m16, tiles_at, flags_at)
            tiles_at += c.tiles
            flags_at += c.m
        keys = np.frombuffer(b"".join(self.keys(self.roots[c.w], index + i, c.m16) for i, c in enumerate(calls)), np.uint8)
        pos = np.fromiter((self.order[c.w] for c in calls), np.intp, len(calls))
        return _Slots(calls, at, pos, keys.reshape(len(calls), -1), offs, predicted, (index + len(calls), tiles_at, flags_at))

    def whole_replay(self, calls: tuple[WholeCall, ...]) -> bool:
        """Before vLLM replays a forward captured whole, with hashing: every call's keys and offsets (its index, and the
        pass's running tiles and screened rows) into its weight's slot, in one upload on the replaying stream.  Returns
        whether `whole_replayed` had staged them, for these calls from where the pass is."""
        cur = self.cur
        if not self.hashing or not calls:
            return False
        at, s = (len(cur.matmuls), cur.tiles, cur.flag_rows), self._next
        staged = s is not None and s.calls is calls and s.at == at
        if not staged:
            s = self._slots(calls, at)
        if s.after[1] > self.capacity or self.nb * s.after[0] > len(self.roots_buf) or s.after[2] > len(self.flag_buf):
            raise RuntimeError(f"{self.entry}: the pass is over its capacity; end it and begin another")
        if self._slots_copied is not None:
            self._slots_copied.synchronize()
        self._host_keys.numpy()[s.pos, :s.keys.shape[1]] = s.keys
        self._host_offs.numpy()[s.pos] = s.offs
        import torch
        self.key_slots.copy_(self._host_keys, non_blocking=True)
        self.offs.copy_(self._host_offs, non_blocking=True)
        self._slots_copied = self._slots_copied or torch.cuda.Event()
        self._slots_copied.record()
        self._uploaded = s
        return staged

    def whole_replayed(self, calls: tuple[WholeCall, ...]) -> None:
        """After vLLM launches its replay of a forward captured whole: each call's matmul, and with hashing its tiles and
        rows, appended to the pass, as `_after` appends them for a call made alone; in a retained pass, each call's record and
        files; and with hashing, the next forward's slots for the same calls, staged while the GPU runs this one."""
        cur, index = self.cur, len(self.cur.matmuls)
        cur.matmuls.extend(Matmul(index + i, Shape(c.m16, c.k, c.n), c.w) for i, c in enumerate(calls))
        if not self.hashing or not calls:
            return
        s, self._uploaded = self._uploaded, None
        self._predicted = dict(s.predicted)    # the upload rewrote the uncalled weights' slots from the host copy
        cur.tiles, cur.flag_rows = s.after[1], s.after[2]
        if self._statement_on():
            self.statement.after(calls, s.at[1])
        if cur.retain is not None:
            self._keep_whole(calls, s.at[2])
        self._next = self._slots(calls, s.after)

    def _graphed(self, m16: int) -> bool:
        """Whether calls of m₁₆ rows replay as graphs (`graphs`, the serial schedule), reading their keys and offsets from
        their weight's slot."""
        return self.graphs and m16 <= GRAPH_ROWS and not self.schedule.side and self.schedule.main_priority != "greatest"

    def _offsets(self, index: int, tiles_at: int, flags_at: int) -> tuple[int, int, int]:
        """A call's byte offsets in the pass's buffers: its leaves, its flags, its A root."""
        return self.nb * tiles_at, 2 * flags_at, self.nb * index

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
            g.cuts = self._call_cuts(pipe)
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
        return g.y.clone(), screen

    def _graph_key(self, w: int, sh: Any, m: int) -> tuple[tuple, bool]:
        """A call's graph and whether it screens the call's rows (with hashing).  A graph of m₁₆ rows serves every m it rounds
        up from, so it screens only when there are no filler rows, and `_after` screens the rest; inside a whole capture
        (from its warm-up on) vLLM's batch fixes m, so the graph is m's own and screens it, as no `_after` runs at replay."""
        rows = m if self.hashing and (m == sh.m16 or self._whole is not None) else 0
        return (w, sh.m16, self.omit, self.hashing, self.schedule.name, rows), rows > 0

    def _into(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int) -> tuple[Any, bool]:
        """The call's graph steps launched on `raw`, the stream vLLM is capturing, over the graph's own buffers.  The graph
        must exist: made outside any capture (its eager run syncs), by vLLM's warm-up forward.  Returns the graph's own y, not
        a copy: a captured forward calls each weight once (`whole_capture_end`), and reads that call's y, or changes it in
        place, before the weight's next call rewrites it.  Deferred (`_defers`), the call joins the call before's side steps
        after its A tree and forks its own after its GEMM; the forward's last call (every weight called once) joins them."""
        key, screen = self._graph_key(w, sh, m)
        g = self._graphs.get(key)
        if g is None or g.graph is None:
            raise RuntimeError(f"{self.entry}: weight {w}'s call of {sh.m16} rows has no graph yet, and vLLM's capture can't make "
                               f"one: run the forward once outside the capture first")
        pipe.b["y"], pipe.b["leaves"] = g.y.data_ptr(), g.leaves.data_ptr()
        self.kernel.dev.stream = raw
        try:
            if not self._defers():
                for f in g.steps:
                    f()
            else:
                a, b = g.cuts
                for f in g.steps[:a]:
                    f()
                self._join()
                for f in g.steps[a:b]:
                    f()
                self._fork(sh, g.steps[b:], raw)
                if len(self._whole) == len(self.order):
                    self._join()
        finally:
            self.kernel.dev.stream = None
        return g.y, screen

    def _fork(self, sh: Any, steps: list, raw: int) -> None:
        """`steps` on the side stream, after everything the capturing stream has launched so far, with torch's current stream
        the side's too (the scatter is a Triton launch); the next `_join` makes the capturing stream wait for them."""
        import torch
        side = self._side_stream
        side.wait_stream(torch.cuda.current_stream(self.device))
        self.kernel.dev.stream = side.cuda_stream
        try:
            with torch.cuda.stream(side):
                for f in steps:
                    f()
        finally:
            self.kernel.dev.stream = raw
        self._deferred = sh

    def _join(self) -> None:
        """The capturing stream waits for the side steps no call has joined yet, if any."""
        if self._deferred is None:
            return
        import torch
        torch.cuda.current_stream(self.device).wait_stream(self._side_stream)
        self._deferred = None

    def join_before(self, sh: Any) -> None:
        """Before a call writes `sh`'s buffers (its A words first): the side steps of a call of the same shape, which still
        read them, joined."""
        if self._deferred is not None and self._deferred is sh:
            self._join()

    def _scatter(self, w: int, sh: Any, pipe: Any, g: _Graph, tiles: int, flag_rows: int):
        """The graph's last step, one launch: the screen of the call's first `flag_rows` rows (all of them when it screens,
        else none), into the pass's flags, and the call's leaves and A root into the pass's buffers, at the offsets in its
        weight's slot."""
        size, offs = pipe.size["tree_a"], self.offs[self.order[w]]
        root, flags, leaves = sh.b["tree_a"][size - self.nb:size], self.flag_buf.view(-1), g.leaves[:self.nb * tiles]

        def scatter():
            scatter_into(sh.xa, sh.b["alpha_a"], sh.b["aux_a"], flag_rows, flags, leaves, self.leaves, root, self.roots_buf, offs)
        return scatter
