"""The Pearl-C device path's whole-forward half (`WholeCalls`, `pouw_pearl_c_graphs.GraphCalls`' base): a decode call
launched into vLLM's capture of a whole forward, over its graph's steps, and the forward's host work at each replay.

Inside vLLM's own capture of a whole decode forward (`pouw_pearl_c_whole`, between `whole_capture_begin` and
`whole_capture_end`), a call launches the same steps into vLLM's capture instead, and is recorded; at each replay of that
forward no call's Python runs, so the forward's host work is done at once, around vLLM's launch of the replay: before it,
`whole_replay` uploads its slots; after it, while the GPU runs the forward, `whole_replayed` appends its matmuls, tiles and
screened rows to the pass and stages the next forward's slots, which the next `whole_replay` uploads when the pass is where
they were staged for (else it makes them itself: the same bytes either way).  A retained pass's capture also copies each
call's A words and tree, as its steps leave them, and its real rows of y, before the forward's next op, to its stage (its
weight's at its m₁₆: vLLM captures a forward at each of its batch sizes, and a call's A words and tree are sized by m₁₆), from
which `whole_replayed` writes the call's record and files, as `_retained` writes them for a call made alone.

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

With ``whole_pd_shared`` (and hashing), a whole capture launches each call's pd_commit and pd_tree in half an SM's shared
memory in place of PD_WHOLE's whole one (the pipeline's ``pd_shared``), so two of their CTAs, or one and a side lane's, share
an SM.  The kernels are the same and use the same shared memory whatever the launch gives, so they write the same bytes.

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

from dataclasses import dataclass
from typing import Any

from verity.protocols.pouw.verifier.protocol import Matmul, Shape

from verity_vllm.protocols.pouw.pearl_c.passes import call_index

#: a deferred-a whole capture's side steps join the A trees forked so far only before the scatter, the one step that reads
#: A's root, so a call's tile hashing (it reads C~ and U alone) runs beside its A tree, PD_G tiles a CTA (the pipeline's
#: ``tiles_group``) so that its CTAs leave pd_commit's and pd_tree's whole SMs free (beside the tree a CTA a tile, and
#: grouped after it, each is slower).  False joins them first (``PEARLC_TILES_BESIDE_TREE=0``)
TILES_BESIDE_TREE = True


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


class WholeCalls:
    """`GraphCalls`' whole-forward half, as its base class: a call made inside vLLM's capture of a whole forward
    (`_into`, which `GraphCalls._graph_call` sends it to, over the call's graph), and the forward's capture and replay
    around it, on the run's attributes `GraphCalls` names."""

    #: while vLLM captures forwards (`whole_capture_begin`), the calls made inside its stream capture, in order; None otherwise
    _whole: list[WholeCall] | None = None
    #: whether those forwards are a retained pass's, whose capture copies each call's A words, A tree and y to its stage
    _whole_retained: bool = False
    #: each (weight, m₁₆)'s stage, (A words, A tree, y's m₁₆ rows) as its call's steps leave them, made at its first retained
    #: warm-up; y is copied in the capture right after the call, ahead of the forward's next op, which may change it in place
    _stages: dict[tuple[int, int], tuple[Any, Any, Any]] | None = None
    #: the byte a poison left in every stage, until a replay or a new stage (`PassEnds._poison_graphs`)
    _stages_poisoned: int | None = None
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
    #: whether a whole capture (with hashing) launches each call's pd_commit and pd_tree in half an SM's shared memory, not
    #: PD_WHOLE's (the pipeline's ``pd_shared``): the same kernels and bytes, two of their CTAs to an SM
    whole_pd_shared: bool = False
    #: with ``whole_defer``, whether a whole capture also prefetches each call's B̃ into L2 on the stats stream after its
    #: scales; inside one, the event after the last call's forked ``stats_a``, and whether a prefetch is unjoined
    whole_prefetch_b: bool = False
    _stats_done: Any = None
    _prefetch_open: bool = False
    #: a commitment of each forward replayed whole beside the scheme's, while its ``enabled`` (with hashing): inside the
    #: capture, `stage(w, sh)` after each call's steps (and `prepare(w, sh)` outside it, in the warm-up); after each replay,
    #: `after(calls, tiles_at)` on the replaying stream (benchmarks/pouw/served_zk/statement.py's `Statement`)
    statement: Any = None

    def whole_key(self, retained: bool | None = None) -> tuple | None:
        """What a forward captured whole depends on (hashing, omitted steps, schedule, with hashing whether its pass is
        retained, which `retained` sets in place of the open pass's, whether it defers the tile hashing, and after those
        ``a_tree``, ``tree_sm``, ``stats``, ``prefetch_b``, ``shape_joins`` and ``pd_shared`` where it takes those options),
        or None when the pass can't replay one: graphs off, or a schedule the graph path doesn't run."""
        if not self.graphs or self.cur is None or self.schedule.side or self.schedule.main_priority == "greatest":
            return None
        retained = self.cur.retain is not None if retained is None else retained
        key = (self.hashing, self.omit, self.schedule.name, self.hashing and retained, self._defers())
        key = (*key, "a_tree") if self._defers_a() else key
        key = (*key, "tree_sm") if self._defers_a() and self.whole_tree_sm else key
        key = (*key, "stats") if self._forks_stats() else key
        key = (*key, "prefetch_b") if self._prefetches() else key
        key = (*key, "shape_joins") if self._defers() and self.whole_shape_joins else key
        key = (*key, "pd_shared") if self._pd_shared() else key
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

    def _pd_shared(self) -> bool:
        """Whether a whole capture launches pd_commit and pd_tree in half an SM's shared memory (`whole_pd_shared`, with
        hashing)."""
        return self.whole_pd_shared and self.hashing

    def _prefetches(self) -> bool:
        """Whether a whole capture prefetches each call's B̃ beside its A commitment (`whole_prefetch_b`, deferring)."""
        return self.whole_prefetch_b and self._defers()

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
        pass's buffer: each call's from its stage (A's words and tree, and its real rows of y) and its flags, as a call made
        alone keeps them."""
        cur = self.cur
        for mm, c in zip(cur.matmuls[len(cur.matmuls) - len(calls):], calls, strict=True):
            xa, tree, y = self._stages[c.w, c.m16]
            self._retained(cur, mm, c.m, self.flag_buf[flags_at:flags_at + c.m], xa, tree, y=y[:c.m])
            flags_at += c.m

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
        self._unpoisoned()
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
        The call's A words left to it (`_widening`) are widened before ``stats_a`` where `_reads_bf16`, else first.  With
        `_pd_shared` every step sees the pipeline's ``pd_shared`` set, so its pd_commit and pd_tree go into the capture in
        half an SM's shared memory."""
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
        shared = self._pd_shared()
        if shared:
            pipe.pd_shared = True
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
            if shared:
                pipe.pd_shared = False
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
