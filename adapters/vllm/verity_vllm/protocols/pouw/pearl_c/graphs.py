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

Inside vLLM's own capture of a whole decode forward (`pouw_pearl_c_whole`, between `whole_capture_begin` and
`whole_capture_end`), a call launches the same steps into vLLM's capture instead, and is recorded; at each replay of that
forward no call's Python runs, so the forward's host work is done at once, around vLLM's launch of the replay: before it,
`whole_replay` uploads its slots; after it, while the GPU runs the forward, `whole_replayed` appends its matmuls, tiles and
screened rows to the pass and stages the next forward's slots, which the next `whole_replay` uploads when the pass is where
they were staged for (else it makes them itself: the same bytes either way).  A retained pass's capture also copies each
call's A words and tree, as its steps leave them, to its stage (its weight's at its m₁₆: vLLM captures a forward at each of
its batch sizes, and a call's A words and tree are sized by m₁₆), from which `whole_replayed` writes the call's record and
files, as `_retained` writes them for a call made alone.

With ``whole_defer`` (and hashing), a whole capture forks each call's tile hashing and scatter, after its GEMM, onto a side
stream of the run's own, and the next call joins it after its own A tree, so the two run side by side in the forward's graph;
a call joins at once when the call before is of its shape (the side steps still read that shape's buffers), and the
forward's last call joins its own before vLLM's capture ends.  The same kernels write the same bytes, so the commitments are
the serial capture's.

With ``whole_defer_a`` too, where the kernel's pipeline splits A's commitment (``rows_a``, then ``tree_seed_a``, A's tree;
`pouw_pearl_c_lanes`' ``deferred-a`` split, with no padding rows' E_A lines), a whole capture forks each call's A tree
after its per-row keying onto a tree stream of the run's own, beside the call's forming and GEMM, which read nothing the
tree writes.  What reads A's root or tree waits for it first (`_tree_joined`): the call's side steps' scatter, which copies
A's root to the pass (the tile hashing before it reads nothing the tree writes, and runs beside it: `TILES_BESIDE_TREE`),
and a retained capture's copy of A's tree to its stage; and so a call joining the side steps of a call of its shape joins
that call's tree too.  The forward's last call joins the tree stream before vLLM's capture ends.
With ``whole_shape_joins`` too, a call doesn't join the call before's side steps: the capturing stream joins a call's side
steps (and through their scatter its A tree) only before the next call of its shape writes the shape's buffers, which they
read (`join_before`, at an event after them where other calls' side steps followed), so they run beside the calls of other
shapes between (a decoder layer's linears are of four shapes); the forward's last call joins them all.  Every step is still
inside the forward's graph, and the same kernels write the same bytes.
With ``whole_tree_sm`` too, the call's GEMMs beside its tree launch one persistent CTA fewer (the pipeline's ``beside``),
since the tree's CTA holds an SM whole and a GEMM CTA left without one would start only when the tree ends; each tile is
computed as before, by another CTA.

With ``whole_fork_stats`` too (deferring), a whole capture forks each call's A scales (``stats_a``, which reads A's words
alone and writes its shape's alpha, beta, ia and aux) onto a stats stream before the call's A commitment, which reads and
writes none of them, and the capturing stream joins it in its place (an event after ``stats_a``), before the forming steps
that read them; nothing past that join is reordered, so the same kernels write the same bytes.  ``stats_a`` is launched
after the commitment's first steps, so the commitment's CTAs, each holding an SM whole, take the SMs the call before's tile
hashing leaves free.  Where those steps read A's words only in pd_commit's mode 1 and it can read the call's BF16 rows instead
(`run.Pipeline.reads_bf16`, widening each as widen_a does), widen_a moves onto the stats stream before ``stats_a``, so A's
commitment starts at the call's x; the forming, the screen and the stages read A's words after the join, as before.  With
``whole_prefetch_b`` the stats stream takes the scales and then B̃'s prefetch into L2 (`run.Pipeline.prefetch_b`, which
writes nothing) before the commitment, which leaves DRAM idle, and the forward's last call joins the stats stream.

With a `statement` enabled (and hashing), a whole capture also copies each call's A words to the statement's stage, and
`whole_replayed` launches its commitment of the forward's rows after the replay; its key says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c_work as PW

from verity_vllm.protocols.pouw.pearl_c.passes import call_index
from verity_vllm.protocols.pouw.pearl_c.host import _bytes, _event, _Lane, _write, _zeros, torch_uint8
from verity_vllm.protocols.pouw.pearl_c.screen import excluded, scatter_into

#: with ``graphs``, calls of at most this many rows (decode's) replay their steps as one CUDA graph; above it the launches are
#: a small share of the call, and a graph's own y would cost m x n of memory per weight
GRAPH_ROWS = 256
#: with ``graphs``, each weight's slot of per-call inputs its graph reads: dom_a ‖ unit (168 bytes, padded), and
#: the byte offsets of the call's leaves, flags and A root in the pass's buffers (three int64)
SLOT_BYTES = 176
#: a lane graph's slot (`_lane_graph`): dom_a ‖ unit padded to `SLOT_BYTES`, then the call's three offsets
LANE_SLOT_BYTES = SLOT_BYTES + 24
#: a deferred-a whole capture's side steps join the A trees forked so far only before the scatter, the one step that reads
#: A's root, so a call's tile hashing (it reads C~ and U alone) runs beside its A tree, PD_G tiles a CTA (the pipeline's
#: ``tiles_group``) so that its CTAs leave pd_commit's and pd_tree's whole SMs free (beside the tree a CTA a tile, and
#: grouped after it, each is slower).  False joins them first (``PEARLC_TILES_BESIDE_TREE=0``)
TILES_BESIDE_TREE = True


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
    """A forward's slots, for `calls` from the pass's counters `at` (matmuls, tiles, screened rows) in the pass whose first
    call has index `base`: each call's keys and offsets by its weight's slot position (``pos``), each weight's prediction as
    `_slot` reads it, and the counters after."""
    calls: tuple[WholeCall, ...]
    at: tuple[int, int, int]
    base: int
    pos: Any
    keys: Any
    offs: Any
    predicted: dict
    after: tuple[int, int, int]


class GraphCalls:
    """`PearlCDeviceRun`'s graph path, as its base class: the run's ``graphs``, ``driver``, ``fork``, ``order``, ``nb`` (its
    trees' node bytes), ``key_slots``, ``offs``, ``_host_keys``, ``_host_offs``, ``_slots_copied``, ``_predicted`` and
    ``_graphs``, its `_call_steps` and `_call_cuts`, and its pass and the pass's buffers (``roots_buf``, ``flag_buf``, whose
    lengths bound it)."""

    #: while vLLM captures forwards (`whole_capture_begin`), the calls made inside its stream capture, in order; None otherwise
    _whole: list[WholeCall] | None = None
    #: whether those forwards are a retained pass's, whose capture copies each call's A words and tree to its stage
    _whole_retained: bool = False
    #: each (weight, m₁₆)'s stage, (A words, A tree) as its call's steps leave them, made at its first retained warm-up
    _stages: dict[tuple[int, int], tuple[Any, Any]] | None = None
    #: the slots the last `whole_replay` uploaded, and those `whole_replayed` staged for the forward after it
    _uploaded: _Slots | None = None
    _next: _Slots | None = None
    #: whether a whole capture defers each call's tile hashing beside the next call (with hashing)
    whole_defer: bool = False
    #: inside a deferred whole capture, the calls whose side steps the capturing stream hasn't joined yet, oldest first: each
    #: one's shape and (``whole_shape_joins``) the side stream's event after them; and the side stream
    _pending: list | None = None
    _side_stream: Any = None
    #: with ``whole_defer``, whether a call joins only the side steps of its shape's last call, not the call before's
    whole_shape_joins: bool = False
    #: with ``whole_defer``, whether a whole capture also forks each call's A tree beside its forming and GEMM (with hashing,
    #: where the call's pipeline splits A's commitment); inside one, whether a call forked a tree the capturing stream hasn't
    #: joined yet, and the tree stream
    whole_defer_a: bool = False
    _tree_open: bool = False
    _tree_stream: Any = None
    #: with ``whole_defer``, whether a whole capture also forks each call's A scales (``stats_a``, which reads A's words
    #: alone) beside its A commitment, joined where the call's forming reads them; and the stats stream
    whole_fork_stats: bool = False
    _stats_stream: Any = None
    #: the call in flight's A words left to its `_into` (`_widen_later`): its shape, BF16 rows' address, rows and filled rows
    _widening: Any = None
    #: with ``whole_defer_a``, whether a call's GEMMs beside its forked A tree leave the tree's CTA its SM (the pipeline's
    #: ``beside``)
    whole_tree_sm: bool = False
    #: with ``whole_defer``, whether a whole capture also prefetches each call's B̃ into L2 on the stats stream after its
    #: scales; inside one, the event after the last call's forked ``stats_a``, and whether a prefetch is unjoined
    whole_prefetch_b: bool = False
    _stats_done: Any = None
    _prefetch_open: bool = False
    #: a commitment of each forward replayed whole beside the scheme's, while its ``enabled`` (with hashing): inside the
    #: capture, `stage(w, sh)` after each call's steps (and `prepare(w, sh)` outside it, in the warm-up); after each replay,
    #: `after(calls, tiles_at)` on the replaying stream (benchmarks/pouw/served_zk/statement.py's `Statement`)
    statement: Any = None

    def _graph_call(self, w: int, sh: Any, pipe: Any, raw: int, tiles: int, m: int, index: int, caller: Any) -> tuple[Any, bool]:
        """A graphed call: inside vLLM's capture of a whole forward, its steps launched into that capture (and in a retained
        pass's, the copies to its stage, once its A tree is done); otherwise its own graph replayed after its slot.  vLLM's
        warm-up forward before each capture runs outside it, and makes the graphs and stages."""
        if self._whole is not None and _capturing():
            self._whole.append(WholeCall(w, m, sh.k, sh.n, sh.m16, tiles))
            out = self._into(w, sh, pipe, raw, tiles, m)
            if self._whole_retained:
                self._tree_joined()
                for stage, t in zip(self._stages[w, sh.m16], self._kept(sh, pipe), strict=True):
                    stage.copy_(t)
            if self._statement_on():
                self.statement.stage(w, sh)
            return out
        if self._whole is not None and self._whole_retained and (w, sh.m16) not in self._stages:
            self._stages[w, sh.m16] = tuple(t.new_empty(t.shape) for t in self._kept(sh, pipe))
        if self._whole is not None and self._statement_on():
            self.statement.prepare(w, sh)
        if self.hashing:
            self._slot(w, index, sh.m16, m, caller)
        return self._replay(w, sh, pipe, raw, tiles, m)

    @staticmethod
    def _kept(sh: Any, pipe: Any) -> tuple[Any, Any]:
        """What a retained call keeps of its shape's buffers once its steps ran: A's words (m₁₆ rows) and A's tree."""
        return sh.xa[:sh.m16], sh.buf("tree_a")[:pipe.size["tree_a"]]

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
        retained, which `retained` sets in place of the open pass's, whether it defers the tile hashing, and after those
        ``a_tree``, ``tree_sm``, ``stats``, ``prefetch_b`` and ``shape_joins`` where it takes those options), or None when the
        pass can't replay one: graphs off, or a schedule the graph path doesn't run."""
        if not self.graphs or self.cur is None or self.schedule.side or self.schedule.main_priority == "greatest":
            return None
        retained = self.cur.retain is not None if retained is None else retained
        key = (self.hashing, self.omit, self.schedule.name, self.hashing and retained, self._defers())
        key = (*key, "a_tree") if self._defers_a() else key
        key = (*key, "tree_sm") if self._defers_a() and self.whole_tree_sm else key
        key = (*key, "stats") if self._forks_stats() else key
        key = (*key, "prefetch_b") if self._prefetches() else key
        key = (*key, "shape_joins") if self._defers() and self.whole_shape_joins else key
        return (*key, "statement") if self._statement_on() else key

    def _defers(self) -> bool:
        """Whether a whole capture forks each call's tile hashing beside the next call (`whole_defer`, with hashing)."""
        return self.whole_defer and self.hashing

    def _defers_a(self) -> bool:
        """Whether a whole capture forks each call's A tree beside its forming and GEMM (`whole_defer_a`, deferring)."""
        return self.whole_defer_a and self._defers()

    def _forks_stats(self) -> bool:
        """Whether a whole capture forks each call's ``stats_a`` beside its A commitment (`whole_fork_stats`, deferring)."""
        return self.whole_fork_stats and self._defers()

    def _prefetches(self) -> bool:
        """Whether a whole capture prefetches each call's B̃ beside its A commitment (`whole_prefetch_b`, deferring)."""
        return self.whole_prefetch_b and self._defers()

    def _widen_later(self, sh: Any) -> bool:
        """Whether a call of `sh` leaves its A words' widening (`LaneCalls._stage_a`) to its `_into`: inside a whole capture
        forking A's scales, at a shape that replays as a graph, which is where the call goes to `_into`."""
        return self._forks_stats() and self._whole is not None and self._graphed(sh.m16) and _capturing()

    def _reads_bf16(self, pipe: Any, upto: int) -> bool:
        """Whether the call's steps before ``stats_a`` (step `upto`) read A's words only in pd_commit's mode 1 (``rows_a``;
        ``tree_seed_a`` reads none), and `pipe` lets that read the call's BF16 rows instead (`run.Pipeline.reads_bf16`)."""
        have = {name for name, _, _ in pipe.steps()}
        before = [x for x in self._call_names(pipe) if x in have][:upto]
        return ("rows_a" in before and set(before) <= {"rows_a", "tree_seed_a"}
                and getattr(pipe, "reads_bf16", lambda: False)())

    def _widen(self, widen: tuple, stream: int, pdl: bool = True) -> None:
        """A call's A words widened from its BF16 rows (`_widening`'s), on `stream`, as `LaneCalls._stage_a` launches it."""
        sh, src, m, filled = widen
        kw = {} if pdl else {"pdl": False}
        self.kernel.run.widen_a(self.kernel.dev, src, sh.xa.data_ptr(), m, filled, sh.k, stream=stream, **kw)

    def _statement_on(self) -> bool:
        """Whether forwards replayed whole commit `statement`'s rows too (it is set and enabled, with hashing)."""
        return self.statement is not None and self.statement.enabled and self.hashing

    def whole_capture_begin(self, key: tuple | None = None) -> None:
        """Calls are recorded from here, for the forwards of `key`'s kind (`whole_key`'s; an unretained pass's when None).
        A deferring run's side stream (and tree stream, deferring A's trees, and stats stream, forking A's scales or
        prefetching B̃) is made here, before vLLM's capture begins."""
        self._whole, self._whole_retained, self._pending, self._tree_open = [], bool(key and key[3]), [], False
        self._prefetch_open = False
        if self._whole_retained and self._stages is None:
            self._stages = {}
        if self._defers() and self._side_stream is None:
            import torch
            self._side_stream = torch.cuda.Stream(device=self.device)
        if self._defers_a() and self._tree_stream is None:
            import torch
            self._tree_stream = torch.cuda.Stream(device=self.device)
        if (self._forks_stats() or self._prefetches()) and self._stats_stream is None:
            import torch
            self._stats_stream = torch.cuda.Stream(device=self.device)

    def whole_capture_end(self) -> tuple[WholeCall, ...]:
        """The calls captured since `whole_capture_begin`.  Each weight's slot holds one call's keys, so a weight called twice
        in one forward is refused."""
        calls, self._whole, self._whole_retained, self._pending = tuple(self._whole or ()), None, False, None
        self._tree_open = self._prefetch_open = False
        seen = [c.w for c in calls]
        if len(set(seen)) != len(seen):
            raise RuntimeError(f"{self.entry}: a weight is called twice in one captured forward, and its slot holds one call")
        return calls

    def _keep_whole(self, calls: tuple[WholeCall, ...], flags_at: int) -> None:
        """A retained pass's records and files of a forward replayed whole, its first call's flags at row `flags_at` of the
        pass's buffer: each call's from its stage and its flags, as a call made alone keeps them."""
        cur = self.cur
        for mm, c in zip(cur.matmuls[len(cur.matmuls) - len(calls):], calls, strict=True):
            self._retained(cur, mm, c.m, self.flag_buf[flags_at:flags_at + c.m], *self._stages[c.w, c.m16])
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
        base = self.cur.base
        index, tiles_at, flags_at = base + at[0], at[1], at[2]
        offs, predicted = np.empty((len(calls), 3), np.int64), {}
        for i, c in enumerate(calls):
            offs[i] = self._offsets(index + i, tiles_at, flags_at)
            predicted[c.w] = (index + i, c.m16, tiles_at, flags_at)
            tiles_at += c.tiles
            flags_at += c.m
        keys = np.frombuffer(b"".join(self.keys(self.roots[c.w], index + i, c.m16) for i, c in enumerate(calls)), np.uint8)
        pos = np.fromiter((self.order[c.w] for c in calls), np.intp, len(calls))
        return _Slots(calls, at, base, pos, keys.reshape(len(calls), -1), offs, predicted, (at[0] + len(calls), tiles_at, flags_at))

    def whole_replay(self, calls: tuple[WholeCall, ...]) -> bool:
        """Before vLLM replays a forward captured whole, with hashing: every call's keys and offsets (its index, and the
        pass's running tiles and screened rows) into its weight's slot, in one upload on the replaying stream.  Returns
        whether `whole_replayed` had staged them, for these calls from where the pass is."""
        cur = self.cur
        call_index(self, len(calls))
        if not self.hashing or not calls:
            return False
        at, s = (len(cur.matmuls), cur.tiles, cur.flag_rows), self._next
        staged = s is not None and s.calls is calls and s.at == at and s.base == cur.base
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
        cur = self.cur
        index = cur.base + len(cur.matmuls)
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
        return self.nb * tiles_at, 2 * flags_at, self.nb * (index - self.cur.base)

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
        after its A tree and forks its own after its GEMM; the forward's last call (every weight called once) joins them.
        Deferring A's trees (`_defers_a`, where the call's graph has one, ``tree_at``), the call forks its tree after its
        per-row keying, then joins the call before's side steps; the last call joins the tree stream too.  Forking A's
        scales (`_forks_stats`, ``stats_at``) or prefetching B̃ (`_prefetches`), the call launches them on the stats stream
        as the module's docstring says and joins ``stats_a`` in its place; the last call joins that stream too.  Joining by
        shape (``whole_shape_joins``), the call doesn't join the call before's side steps; `join_before` joins its shape's.
        The call's A words left to it (`_widening`) are widened before ``stats_a`` where `_reads_bf16`, else first."""
        key, screen = self._graph_key(w, sh, m)
        g = self._graphs.get(key)
        if g is None or g.graph is None:
            raise RuntimeError(f"{self.entry}: weight {w}'s call of {sh.m16} rows has no graph yet, and vLLM's capture can't make "
                               f"one: run the forward once outside the capture first")
        pipe.b["y"], pipe.b["leaves"] = g.y.data_ptr(), g.leaves.data_ptr()
        widen, self._widening = self._widening, None
        if widen is not None and widen[0] is not sh:
            raise RuntimeError(f"{self.entry}: weight {w}'s call has another shape's A words left to widen")
        s = g.stats_at if self._forks_stats() else None
        bf16 = widen is not None and s is not None and self._reads_bf16(pipe, s)
        if widen is not None and not bf16:
            self._widen(widen, raw)
        if bf16:
            pipe.rows_bf16 = widen[1:3]
        self.kernel.dev.stream = raw
        try:
            if not self._defers():
                for f in g.steps:
                    f()
            else:
                a, b = g.cuts
                t = g.tree_at if self._defers_a() else None
                prefetch = getattr(pipe, "prefetch_b", None) if self._prefetches() else None
                fork = (None if s is None else g.steps[s], raw, widen if bf16 else None, prefetch)
                if s is not None or prefetch is not None:
                    self._open_stats()
                if prefetch is not None:
                    self._fork_stats(*fork)
                for f in g.steps[:a if t is None else t]:
                    f()
                if s is not None and prefetch is None:
                    self._fork_stats(*fork)
                if t is not None:
                    self._fork_tree(g.steps[t:a], raw)
                if not self.whole_shape_joins:
                    self._join()
                pipe.beside = int(t is not None and self.whole_tree_sm)
                try:
                    for i in range(a, b):
                        if i == s:
                            self._join_stats()
                        else:
                            g.steps[i]()
                finally:
                    pipe.beside = 0
                pipe.tiles_group = t is not None and TILES_BESIDE_TREE
                try:
                    self._fork(sh, g.steps[b:], raw)
                finally:
                    pipe.tiles_group = False
                if len(self._whole) == len(self.order):
                    self._join()
                    self._join_tree()
                    self._join_prefetch()
        finally:
            self.kernel.dev.stream = None
            if bf16:
                pipe.rows_bf16 = None
        return g.y, screen

    def _fork_tree(self, steps: list, raw: int) -> None:
        """`steps` (the call's A tree) on the tree stream, after everything the capturing stream has launched so far;
        what reads A's root or tree waits for them (`_tree_joined`)."""
        import torch
        tree = self._tree_stream
        tree.wait_stream(torch.cuda.current_stream(self.device))
        self.kernel.dev.stream = tree.cuda_stream
        try:
            with torch.cuda.stream(tree):
                for f in steps:
                    f()
        finally:
            self.kernel.dev.stream = raw
        self._tree_open = True

    def _open_stats(self) -> None:
        """The stats stream waits for everything the capturing stream has launched so far (the call's A words)."""
        import torch
        self._stats_stream.wait_stream(torch.cuda.current_stream(self.device))

    def _fork_stats(self, step: Any, raw: int, widen: tuple | None = None, prefetch: Any = None) -> None:
        """On the stats stream, after `_open_stats`'s wait and whatever the capturing stream launched since: `widen` (the
        call's A words, `_widen`, plainly: the stream's launch before it isn't the one it follows), `step` (``stats_a``) and
        `prefetch` (B̃ into L2), each where given; `_join_stats` joins the first two, `_join_prefetch` the prefetch."""
        import torch
        stats = self._stats_stream
        self.kernel.dev.stream = stats.cuda_stream
        try:
            with torch.cuda.stream(stats):
                if widen is not None:
                    self._widen(widen, stats.cuda_stream, pdl=False)
                if step is not None:
                    step()
                    self._stats_done = torch.cuda.Event()
                    self._stats_done.record(stats)
                if prefetch is not None:
                    prefetch()
                    self._prefetch_open = True
        finally:
            self.kernel.dev.stream = raw

    def _join_stats(self) -> None:
        """The capturing stream waits for the call's ``stats_a``, in its place among the call's steps (not for a prefetch
        behind it)."""
        import torch
        torch.cuda.current_stream(self.device).wait_event(self._stats_done)

    def _join_prefetch(self) -> None:
        """The capturing stream waits for every prefetch forked onto the stats stream, as vLLM's capture must before it ends."""
        if self._prefetch_open:
            import torch
            torch.cuda.current_stream(self.device).wait_stream(self._stats_stream)
        self._prefetch_open = False

    def _tree_joined(self, stream: Any = None) -> None:
        """`stream` (torch's current one when None) waits for the A trees forked so far, if a call forked one: before
        anything reads A's root or tree."""
        if self._tree_open:
            import torch
            (torch.cuda.current_stream(self.device) if stream is None else stream).wait_stream(self._tree_stream)

    def _join_tree(self) -> None:
        """The capturing stream waits for the A trees forked so far, as vLLM's capture must before it ends."""
        if self._tree_open:
            import torch
            torch.cuda.current_stream(self.device).wait_stream(self._tree_stream)
        self._tree_open = False

    def _fork(self, sh: Any, steps: list, raw: int) -> None:
        """`steps` on the side stream, after everything the capturing stream has launched so far, the last (with hashing, the
        scatter, which copies A's root) after the A trees forked so far too (`TILES_BESIDE_TREE`), with torch's current stream
        the side's too (the scatter is a Triton launch); the next `_join` (or `join_before` the next call of `sh`) joins them."""
        import torch
        side = self._side_stream
        side.wait_stream(torch.cuda.current_stream(self.device))
        cut = len(steps) - 1 if TILES_BESIDE_TREE and self.hashing and steps else 0
        if not cut:
            self._tree_joined(side)
        self.kernel.dev.stream = side.cuda_stream
        try:
            with torch.cuda.stream(side):
                for i, f in enumerate(steps):
                    if cut and i == cut:
                        self._tree_joined(side)
                    f()
        finally:
            self.kernel.dev.stream = raw
        self._pending.append((sh, side.record_event() if self.whole_shape_joins else None))

    def _join(self) -> None:
        """The capturing stream waits for the side steps no call has joined yet, if any."""
        if not self._pending:
            return
        import torch
        torch.cuda.current_stream(self.device).wait_stream(self._side_stream)
        self._pending = []

    def join_before(self, sh: Any) -> None:
        """Before a call writes `sh`'s buffers (its A words first): the side steps of its shape's last call, which still read
        them, joined with every call's before them, at the event after them where others followed (``whole_shape_joins``)."""
        pending = self._pending or []
        at = next((i for i in range(len(pending) - 1, -1, -1) if pending[i][0] is sh), None)
        if at is None:
            return
        if at == len(pending) - 1:
            self._join()
            return
        import torch
        torch.cuda.current_stream(self.device).wait_event(pending[at][1])
        del pending[:at + 1]

    def _scatter(self, w: int, sh: Any, pipe: Any, g: _Graph, tiles: int, flag_rows: int):
        """The graph's last step, one launch: the screen of the call's first `flag_rows` rows (all of them when it screens,
        else none), into the pass's flags, and the call's leaves and A root into the pass's buffers, at the offsets in its
        weight's slot."""
        size, offs = pipe.size["tree_a"], self.offs[self.order[w]]
        root, flags, leaves = sh.b["tree_a"][size - self.nb:size], self.flag_buf.view(-1), g.leaves[:self.nb * tiles]

        def scatter():
            scatter_into(sh.xa, sh.b["alpha_a"], sh.b["aux_a"], flag_rows, flags, leaves, self.leaves, root, self.roots_buf, offs)
        return scatter
