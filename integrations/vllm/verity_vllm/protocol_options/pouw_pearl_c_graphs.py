"""The Pearl-C device path's CUDA graphs (``graphs``, `pouw_pearl_c_device.PearlCDeviceRun`): each decode call's steps are
captured once a pipeline and replayed on the caller's stream.  A graph reads its call's keys and pass offsets from its
weight's device slot, which a forward's first call fills for the whole forward in one upload (a call the prediction missed
uploads its own), and scatters its leaves, flags and A root into the pass itself.  The same kernels write the same bytes as
the eager steps, which the e2e's ``graphs`` gate holds.

Inside vLLM's own capture of a whole decode forward (`pouw_pearl_c_whole`, between `whole_capture_begin` and
`whole_capture_end`), a call launches the same steps into vLLM's capture instead, and is recorded; at each replay of that
forward no call's Python runs, so the forward's host work is done at once, around vLLM's launch of the replay: before it,
`whole_replay` uploads its slots; after it, while the GPU runs the forward, `whole_replayed` appends its matmuls, tiles and
screened rows to the pass and stages the next forward's slots, which the next `whole_replay` uploads when the pass is where
they were staged for (else it makes them itself: the same bytes either way).  A retained pass's capture also copies each
call's A words and tree, as its steps leave them, to its weight's stage, from which `whole_replayed` writes the call's record
and files, as `_retained` writes them for a call made alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verity_pouw.protocol import Matmul, Shape
from verity_pouw.schemes import pearl_c_work as PW

from .pouw_pearl_c_host import _bytes, _write, _zeros, torch_uint8
from .pouw_pearl_c_screen import copy_kernel, excluded, screen_into

#: with ``graphs``, calls of at most this many rows (decode's) replay their steps as one CUDA graph; above it the launches are
#: a small share of the call, and a graph's own y would cost m x n of memory per weight
GRAPH_ROWS = 256
#: with ``graphs``, each weight's slot of per-call inputs its graph reads: dom_a ‖ unit (136 bytes, padded), and the byte
#: offsets of the call's leaves, flags and A root in the pass's buffers (three int64)
SLOT_BYTES = 144


def _capturing() -> bool:
    import torch
    return torch.cuda.is_current_stream_capturing()


@dataclass
class _Graph:
    """One pipeline's call steps captured as a CUDA graph, over buffers of its own where a call's differ: y, and the call's
    tile leaves, which are copied to the pass's leaf buffer at the call's offset after each replay; with hashing and no filler
    rows, the screen's flags too (``flags``), copied likewise."""
    y: Any
    leaves: Any
    flags: Any = None
    graph: Any = None
    steps: list = field(default_factory=list)


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
    """`PearlCDeviceRun`'s graph path, as its base class: the run's ``graphs``, ``order``, ``key_slots``, ``offs``,
    ``_host_keys``, ``_host_offs``, ``_slots_copied``, ``_predicted`` and ``_graphs``, its `_call_steps`, and its pass and the
    pass's buffers (``roots_buf``, ``flag_buf``, whose lengths bound it)."""

    #: while vLLM captures forwards (`whole_capture_begin`), the calls made inside its stream capture, in order; None otherwise
    _whole: list[WholeCall] | None = None
    #: whether those forwards are a retained pass's, whose capture copies each call's A words and tree to its stage
    _whole_retained: bool = False
    #: each weight's stage, (A words, A tree) as its call's steps leave them, made at the first retained capture
    _stages: dict[int, tuple[Any, Any]] | None = None
    #: the slots the last `whole_replay` uploaded, and those `whole_replayed` staged for the forward after it
    _uploaded: _Slots | None = None
    _next: _Slots | None = None

    def _graph_call(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int, index: int, caller: Any) -> tuple[Any, bool]:
        """A graphed call: inside vLLM's capture of a whole forward, its steps launched into that capture (and in a retained
        pass's, the copies to its stage); otherwise its own graph replayed after its slot.  vLLM's warm-up forward before each
        capture runs outside it, and makes the graphs and stages."""
        if self._whole is not None and _capturing():
            if self.hashing and m != sh.m16:
                raise RuntimeError(f"{self.entry}: a call of {m} rows inside vLLM's whole-forward capture, whose screen would "
                                   f"run at offsets fixed at capture: only calls without filler rows are captured whole")
            self._whole.append(WholeCall(w, m, sh.k, sh.n, sh.m16, tiles))
            out = self._into(w, sh, pipe, raw, tiles, m)
            if self._whole_retained:
                for stage, t in zip(self._stages[w], self._kept(sh, pipe), strict=True):
                    stage.copy_(t)
            return out
        if self._whole is not None and self._whole_retained and w not in self._stages:
            self._stages[w] = tuple(t.new_empty(t.shape) for t in self._kept(sh, pipe))
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
        """What a forward captured whole depends on (hashing, omitted steps, schedule, and with hashing whether its pass is
        retained, which `retained` sets in place of the open pass's), or None when the pass can't replay one: graphs off, or a
        schedule the graph path doesn't run."""
        if not self.graphs or self.cur is None or self.schedule.side or self.schedule.main_priority == "greatest":
            return None
        retained = self.cur.retain is not None if retained is None else retained
        return (self.hashing, self.omit, self.schedule.name, self.hashing and retained)

    def whole_capture_begin(self, key: tuple | None = None) -> None:
        """Calls are recorded from here, for the forwards of `key`'s kind (`whole_key`'s; an unretained pass's when None)."""
        self._whole, self._whole_retained = [], bool(key and key[3])
        if self._whole_retained and self._stages is None:
            self._stages = {}

    def whole_capture_end(self) -> tuple[WholeCall, ...]:
        """The calls captured since `whole_capture_begin`.  Each weight's slot holds one call's keys, so a weight called twice
        in one forward is refused."""
        calls, self._whole, self._whole_retained = tuple(self._whole or ()), None, False
        seen = [c.w for c in calls]
        if len(set(seen)) != len(seen):
            raise RuntimeError(f"{self.entry}: a weight is called twice in one captured forward, and its slot holds one call")
        return calls

    def _whole_retained(self, calls: tuple[WholeCall, ...], flags_at: int) -> None:
        """A retained pass's records and files of a forward replayed whole, its first call's flags at row `flags_at` of the
        pass's buffer: each call's from its stage and its flags, as a call made alone keeps them."""
        cur = self.cur
        for mm, c in zip(cur.matmuls[len(cur.matmuls) - len(calls):], calls, strict=True):
            self._retained(cur, mm, c.m, self.flag_buf[flags_at:flags_at + c.m], *self._stages[c.w])
            flags_at += c.m

    def _graph_buffers(self):
        """Every buffer the graphs write, as bytes: each graph's y, leaves and flags, and each stage."""
        import torch
        for g in self._graphs.values():
            yield from (t for t in (g.y.view(torch.uint8), g.leaves, g.flags) if t is not None)
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
        if s.after[1] > self.capacity or 32 * s.after[0] > len(self.roots_buf) or s.after[2] > len(self.flag_buf):
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
        if cur.retain is not None:
            self._whole_retained(calls, s.at[2])
        self._next = self._slots(calls, s.after)

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
        key, screen = self._graph_key(w, sh, m)
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
        screen = self.hashing and m == sh.m16
        return (w, sh.m16, self.omit, self.hashing, self.schedule.name, screen), screen

    def _into(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int) -> tuple[Any, bool]:
        """The call's graph steps launched on `raw`, the stream vLLM is capturing, over the graph's own buffers.  The graph
        must exist: made outside any capture (its eager run syncs), by vLLM's warm-up forward."""
        key, screen = self._graph_key(w, sh, m)
        g = self._graphs.get(key)
        if g is None or g.graph is None:
            raise RuntimeError(f"{self.entry}: weight {w}'s call of {sh.m16} rows has no graph yet, and vLLM's capture can't make "
                               f"one: run the forward once outside the capture first")
        pipe.b["y"], pipe.b["leaves"] = g.y.data_ptr(), g.leaves.data_ptr()
        self.kernel.dev.stream = raw
        try:
            for f in g.steps:
                f()
        finally:
            self.kernel.dev.stream = None
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
