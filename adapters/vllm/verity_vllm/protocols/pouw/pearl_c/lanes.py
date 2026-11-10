"""The Pearl-C device path's schedule (`pouw_pearl_c_device.PearlCDeviceRun`'s lanes): the lane each call step runs on, the
per-call buffers a side lane touches, and a call's eager launches on those lanes.

The schedule (`verity.core.protocols.pouw.serving.Schedule`, by name from `SCHEDULES`; the decode plan's step 2)
places each call's step groups (`STEP_GROUPS`) on lanes.  Under ``deferred``, the measured one, the call's tile
hashing (`SIDE_IO`: the tile leaves) runs on a side lane at the least priority behind an event after the GEMM, while the next
calls go on, and the rest of the call runs on a main lane of the run's own at the greatest priority, joined to the caller's
stream by events on both sides (``deferred-caller`` leaves it on the caller's stream).

Under ``deferred-a`` A's tree goes beside the tile hashing too, where the kernel's pipeline splits A's commitment
(`run.Pipeline`'s ``split_a``: ``rows_a``, pd_commit's mode 1, then ``tree_seed_a``, pd_tree; outside pd, where the pipeline
has it, the same two steps over the generic kernels, A's level 0 among the tree's), with hashing (`LaneCalls._aside`,
`splits_at`); a retained pass's record reads the tree once the call's side lane is done.  The main lane keeps the
per-row keying forming reads (``row_seeds``: each real row's digest, seed, line key and E_A line), and A's tree builds
A's levels, seed_A and line_a off it.  The padding rows (m₁₆ … the kernels' tile, outside the
matmul) are the GEMM's alone: no tile leaf reads them (the leaves stop at m₁₆ rows) and y is served to m, so the split
pipeline writes none of their E_A lines (`run.Pipeline`'s ``fillers``), and the GEMM forms them from whatever lines the
shape's buffer holds.  Every committed and served byte is the serial schedule's; the padding rows' products are not.  So
A's tree runs behind an event right after the main lane's per-row keying, beside the forming and the GEMM, which read
nothing the tree writes, and copies A's root to the pass.  Below a full tile of rows (`TREE_APART_ROWS`) it runs on the
call's side lane, whose tile hashing follows it behind the main lane's end; from a full tile, whose tile hashing takes
longer than the tree, on the run's next side lane after the call's (whose last work is the call before's tile hashing),
so the call's side lane starts the tile hashing at the main lane's end, then waits for the tree, so its end is the call's
side steps' end.  No further stream: past the device's hardware queues (8 by default), a stream shares one with another,
whose work it then waits behind.

Each shape rotates `Schedule.ring` sets of the buffers the side lane touches (`rotated`, with the call's keys, dom_a ‖ unit,
when A's tree moves), and the run has as many side lanes (``sides``), which its calls take in turn whatever their shape, so
one call's side steps run beside the next call's; a call reuses a set only once the side steps of the set's last call are
done (the shape's event for the set, ``done``), and the pass's end, and the next pass's begin, wait for every side lane.
The same kernels read and write the same bytes, so the commitments are the serial schedule's.  A schedule changes only between passes (`begin`), and one whose placement this path, the kernel or the scheme's
format can't run is refused (`schedule_of`).
"""
from __future__ import annotations

import functools
import hashlib
import struct
from contextlib import nullcontext
from typing import Any

from verity.core.protocols.pouw.serving import SCHEDULES, Schedule, StepIO, hash_format, schedule_refusals

from verity_vllm.protocols.pouw.pearl_c.graphs import SLOT_BYTES
from verity_vllm.protocols.pouw.pearl_c.host import _event, _stream

#: each call step's group (`verity.core.protocols.pouw.serving.GROUPS`), which a schedule places on a lane.  The per-row
#: keying forming reads (A's pads, each row's digest, each real row's seed) is ``row_seeds``; A's hm96 leaves, tree and
#: unit seed are ``a_tree``.  ``commit_a`` (`run.Pipeline.FUSED`), which also writes each row's seed and E_A line, is
#: ``row_seeds``, and so is ``rows_a``, its mode 1; ``tree_seed_a`` (A's levels, seed_A and line_a) is ``a_tree``;
#: ``form_ep_a`` (E' and the forming in one launch) is ``forming``, and so is ``form_epw_a`` (those and A'F_B's windows)
STEP_GROUPS = {"pad_a": "row_seeds", "rowdig_a": "row_seeds", "hash_rows_a": "a_tree", "tree_a": "a_tree", "seed_a": "a_tree",
               "seeds_a": "row_seeds", "lines_ea": "forming", "stats_a": "forming", "eprime_a": "forming", "form_a": "forming",
               "af": "cleanup", "pa": "cleanup", "gemm_pearlc": "cleanup", "hash_leaf": "tile_hash", "commit_a": "row_seeds",
               "rows_a": "row_seeds", "tree_seed_a": "a_tree", "form_ep_a": "forming", "form_epw_a": "forming"}
#: the steps this path can run on a side lane, with the per-call buffers they read and write (the constant keys and the pass's
#: leaf buffer aside, which no later call rewrites): each shape rotates these, so a call never overwrites what the side lane
#: still reads.  The tile hashing reads C~ and U; A's tree (``tree_seed_a``, only where the pipeline splits A's commitment,
#: which then writes no padding rows' E_A lines) reads level 0 (outside pd, the row digests it hashes into level 0) and the
#: call's keys and writes the levels, seed_A and line_a.  The screen runs on the main lane or the caller
SIDE_IO = {"hash_leaf": StepIO("tile_hash", frozenset({"c", "u"}), frozenset({"segdig_t"})),
           "tree_seed_a": StepIO("a_tree", frozenset({"tree_a", "rowdig_a", "keys"}), frozenset({"tree_a", "seed_a", "line_a"}))}
SIDE_GROUPS = frozenset(io.group for io in SIDE_IO.values())
#: the rows (m₁₆) from which a call's A tree, where it is aside, runs on the side lane after the call's rather than on the
#: call's own: on the H200 the own lane was the faster at m₁₆ 16 and the other at 64, at 7 of GLM-5.3's 8 dense shapes each
TREE_APART_ROWS = 64


@functools.cache
def rotated(schedule: Schedule) -> tuple[str, ...]:
    """The per-call buffers each shape rotates under `schedule`: those its side-lane steps read or write (``keys``, the
    call's dom_a ‖ unit, among them when A's tree moves)."""
    return tuple(sorted({b for io in SIDE_IO.values() if io.group in schedule.side for b in io.reads | io.writes}))


def splits(kernel: Any) -> bool:
    """Whether `kernel`'s pipeline splits A's commitment (`run.Pipeline.FUSED`'s ``tree_seed_a``), so A's tree may leave
    the GEMM's lane."""
    return "tree_seed_a" in getattr(kernel.run.Pipeline, "FUSED", {})


def splits_at(kernel: Any, m16: int, n: int, k: int) -> bool:
    """Whether `kernel`'s pipeline splits A's commitment at a shape: at pd's (`run.pd_variant`), and at every other where
    it splits the generic kernels too (`run.Pipeline.FUSED_GENERIC`, a prefill's rows among them)."""
    R = kernel.run
    return splits(kernel) and (hasattr(R.Pipeline, "FUSED_GENERIC") or R.pd_variant(m16, n, k) is not None)


def whole_options(entry: str, kernel: Any, whole_defer: bool, whole_defer_a: bool, whole_fork_stats: bool,
                  whole_tree_sm: bool, lag_ends: bool = False, aside_ends: bool = False,
                  whole_prefetch_b: bool = False, whole_shape_joins: bool = False) -> None:
    """Refuse a whole capture's options, and a pass end's, where `kernel` or the others can't run them.  `whole_defer_a`
    (with `whole_defer`, on a kernel whose pipeline splits A's commitment) also defers each call's A tree within a whole
    capture (`pouw_pearl_c_graphs`), and `whole_fork_stats` (with `whole_defer`) runs each call's A scales there beside its
    A commitment; `whole_tree_sm` (with `whole_defer_a`) launches the call's GEMMs beside its A tree with one persistent CTA
    fewer, leaving the tree's CTA its SM; `whole_shape_joins` (with `whole_defer`) joins a call's side steps only before
    the next call of its shape; `aside_ends` (with `lag_ends`) runs a lagged end's tile tree aside;
    `whole_prefetch_b` (with `whole_defer`) prefetches each call's B̃ into L2 beside its A commitment (`run.Pipeline.prefetch_b`,
    which writes nothing)."""
    if whole_defer_a and not (whole_defer and splits(kernel)):
        raise ValueError(f"{entry}: whole_defer_a defers A's tree within a deferred whole capture (whole_defer), on a kernel "
                         f"whose pipeline splits A's commitment (`Pipeline.FUSED` tree_seed_a)")
    if whole_fork_stats and not whole_defer:
        raise ValueError(f"{entry}: whole_fork_stats forks A's scales within a deferred whole capture (whole_defer)")
    if whole_shape_joins and not whole_defer:
        raise ValueError(f"{entry}: whole_shape_joins joins the side steps of a deferred whole capture (whole_defer)")
    if whole_tree_sm and not (whole_defer_a and hasattr(kernel.run.Pipeline, "beside")):
        raise ValueError(f"{entry}: whole_tree_sm leaves a deferred A tree (whole_defer_a) its SM, on a kernel whose "
                         f"pipeline's GEMMs take `Pipeline.beside`")
    if aside_ends and not lag_ends:
        raise ValueError(f"{entry}: aside_ends runs a lagged end's tree aside, and lag_ends is off")
    if whole_prefetch_b and not whole_defer:
        raise ValueError(f"{entry}: whole_prefetch_b prefetches within a deferred whole capture (whole_defer)")
    if whole_prefetch_b and not hasattr(kernel.run.Pipeline, "prefetch_b"):
        raise ValueError(f"{entry}: whole_prefetch_b runs the kernel's B̃ prefetch (`Pipeline.prefetch_b`), which it lacks")


def schedule_of(name: str | Schedule, scheme: Any, kernel: Any = None) -> Schedule:
    """`name`'s schedule (`verity.core.protocols.pouw.serving.SCHEDULES`), refused where the scheme's format, this path or (when
    given) `kernel` can't run it: A's tree leaves the main lane only on a kernel whose pipeline splits A's commitment."""
    s = SCHEDULES[name] if isinstance(name, str) else name
    bad = schedule_refusals(hash_format(scheme), s)
    if set(s.side) - SIDE_GROUPS:
        bad.append(f"{s.name}: this path runs only {', '.join(sorted(SIDE_GROUPS))} on a side lane, not "
                   f"{', '.join(sorted(set(s.side) - SIDE_GROUPS))}")
    if "a_tree" in s.side and kernel is not None and not splits(kernel):
        bad.append(f"{s.name}: this kernel's pipeline doesn't split A's commitment (no `Pipeline.FUSED` tree_seed_a), so "
                   f"A's tree stays on the GEMM's lane")
    if bad:
        raise ValueError(f"{scheme.name}: " + "; ".join(bad))
    return s


def admitted(scheme: Any, kernel: Any) -> frozenset[str]:
    """The schedules' names `schedule_of` admits for `scheme` on `kernel`."""
    ok = set()
    for name in SCHEDULES:
        try:
            schedule_of(name, scheme, kernel)
        except ValueError:
            continue
        ok.add(name)
    return frozenset(ok)


class LaneCalls:
    """`PearlCDeviceRun`'s lanes, as its base class: the run's ``schedule``, ``main``, ``sides``, ``driver``, ``hashing``,
    ``cur``, ``kernel``, ``device``, ``nb``, ``keys``, ``uploads``, ``roots``, ``roots_buf``,
    ``_side_turn`` and ``_side_cache``, and its
    graph path's lane graphs (`GraphCalls._lane_graph`, ``_capture_lanes``, ``_offsets``); where a whole capture cuts a
    call's steps (`_call_cuts`, `_call_tree`, `_call_stats`, from ``entry`` and `_call_names`); and a lagged end's readback
    (`settle`: ``_lagged``, ``commits``) and tree aside (`_end_aside`: ``_end_side``, ``leaves``); and a call's A words
    (`_stage_a`)."""

    def _call_cuts(self, pipe: Any) -> tuple[int, int]:
        """Where a deferred whole capture (`whole_defer`) splits `_call_steps`: after A's tree, beside which the call before's
        tile hashing runs, and before the call's own tile hashing, which it forks."""
        have = {name for name, _, _ in pipe.steps()}
        groups = [STEP_GROUPS[x] for x in self._call_names(pipe) if x in have]
        commit = ("row_seeds", "a_tree")
        a = next((i for i, g in enumerate(groups) if g not in commit), len(groups))
        b = next((i for i, g in enumerate(groups) if g == "tile_hash"), len(groups))
        if any(g in commit for g in groups[a:]) or any(g != "tile_hash" for g in groups[b:]):
            raise RuntimeError(f"{self.entry}: the call's steps {groups} don't start with A's commitment and end with the tile hashing")
        return a, b

    def _call_tree(self, pipe: Any) -> int | None:
        """Where a deferred-a whole capture (`whole_defer_a`) forks A's tree from `_call_steps`: ``tree_seed_a``'s place, the
        last of A's commitment's steps (`_call_cuts`' first cut is after it), or None where the pipeline doesn't split it."""
        have = {name for name, _, _ in pipe.steps()}
        present = [x for x in self._call_names(pipe) if x in have]
        if "tree_seed_a" not in present:
            return None
        at = present.index("tree_seed_a")
        if at + 1 != self._call_cuts(pipe)[0]:
            raise RuntimeError(f"{self.entry}: the call's steps {present} don't end A's commitment with A's tree")
        return at

    def _call_stats(self, pipe: Any) -> int | None:
        """Where a whole capture forking A's scales (`whole_fork_stats`) takes ``stats_a`` from `_call_steps`: its place, a
        forming step, which `_call_cuts` holds between A's commitment and the tile hashing; or None where the call has none."""
        have = {name for name, _, _ in pipe.steps()}
        present = [x for x in self._call_names(pipe) if x in have]
        return present.index("stats_a") if "stats_a" in present else None

    def _split_whole(self, sh: Any) -> bool:
        """Whether a call of `sh` splits A's commitment for a deferred-a whole capture (`whole_defer_a`, deferring): a call
        that replays as a graph, at a shape the kernel's pd launches take (`_aside`'s rule)."""
        return (self._defers_a() and self._graphed(sh.m16)
                and self.kernel.run.pd_variant(sh.m16, sh.n, sh.k) is not None)

    def settle(self) -> None:
        """Complete a lagged end's readback (`end` with ``lag``), waiting for its copies: its commitment, in place."""
        if self._lagged is None:
            return
        (out, done, (roots_a, root, flags), units, count), self._lagged = self._lagged, None
        done.synchronize()
        self._ended(out, roots_a.numpy().tobytes(), root.numpy().tobytes(), flags.tolist(), units, count)

    def _end_aside(self, count: int, roots: Any, flags: Any) -> tuple[int, Any, list[Any]]:
        """A lagged end's tile tree on a stream of its own (`aside_ends`), so the next step needn't wait for it: on the current
        stream the pass's leaves are copied to the end's buffer, which `settle` left free, and A's roots and the flag counts
        start back to the host; the end's stream waits for that.  Returns the copy's address, the stream and the readbacks."""
        import torch
        n = self.nb * count
        if self._end_side is None or self._end_side[1].numel() < n:
            self._end_side = (self._end_side[0] if self._end_side else torch.cuda.Stream(self.device),
                              torch.empty(2 * n, dtype=torch.uint8, device=self.device))
        side, staged = self._end_side
        staged[:n].copy_(self.leaves[:n], non_blocking=True)
        got = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True).copy_(t, non_blocking=True) for t in (roots, flags)]
        side.wait_stream(torch.cuda.current_stream())
        return staged.data_ptr(), side, got

    def _stage_a(self, sh: Any, x2: Any, m: int, filled: int) -> None:
        """A call's A words on the current stream: x2's m rows widened to FP32 into ``sh.xa`` and its rows m to `filled` - 1
        zeroed, in one launch where the kernel has widen_a and x2 is aligned, contiguous BF16; else torch's copy and fill.
        Inside a whole capture forking A's scales (`GraphCalls._widen_later`) that launch is left to the call's `_into`
        (``_widening``), which runs it beside A's commitment where the commitment reads x2 itself, else first."""
        import torch
        dev, k = self.kernel.dev, x2.shape[-1]
        if (self.device.type == "cuda" and x2.dtype == torch.bfloat16 and x2.is_contiguous() and x2.data_ptr() % 16 == 0
                and k % 8 == 0 and getattr(dev, "has", lambda _: False)("widen_a")):
            if self._widen_later(sh):
                self._widening = (sh, x2.data_ptr(), m, max(filled, m))
                return
            self.kernel.run.widen_a(dev, x2.data_ptr(), sh.xa.data_ptr(), m, max(filled, m), k, stream=_stream(self.device))
            return
        sh.xa[:m].copy_(x2)
        if filled > m:
            sh.xa[m:filled].zero_()

    def _ended(self, out: dict[str, Any], roots_a: bytes, root: bytes, flags: list[int], units: int, count: int) -> None:
        out.update(transcript_root=root.hex(), activation_roots=hashlib.sha512(roots_a).hexdigest(),
                   failing_rows=int(flags[0]), unsure_rows=int(flags[1]))
        self.commits.update(transcript_root=out["transcript_root"], units=units, tiles=count)

    def _aside(self, sh: Any) -> bool:
        """Whether a call of `sh` runs A's tree on the side lane (the module's docstring): the schedule moves ``a_tree``,
        hashing is on, and the kernel's pipeline splits A's commitment at this shape (`splits_at`)."""
        return "a_tree" in self.schedule.side and self.hashing and splits_at(self.kernel, sh.m16, sh.n, sh.k)

    def _side_groups(self, sh: Any) -> frozenset[str]:
        """The groups a call of `sh` runs on the side lane, with hashing: the schedule's, but A's tree only `_aside`."""
        key = (self.schedule.name, self.hashing, sh.m16, sh.n, sh.k)
        got = self._side_cache.get(key)
        if got is None:
            got = self._side_cache[key] = frozenset(g for g in self.schedule.side if g != "a_tree" or self._aside(sh)) \
                if self.hashing else frozenset()
        return got

    def _lanes(self, w: int, sh: Any, pipe: Any, steps: list, side: list, raw: int, caller: Any, index: int,
               aside: bool, graph: tuple[int, int] | None = None) -> tuple[Any, bool]:
        """A call's steps launched on the schedule's lanes, over the ring's next set of its shape: the main lane (the caller's
        stream, or the run's own after an event on the caller's), then the run's next side lane (`sides`), which runs the
        side steps behind an event after the main lane; where A's tree is aside (`aside`), the tree runs behind an event
        after the main lane's per-row keying (the pipeline's ``lane_cuts``) and copies A's root to the pass, below
        `TREE_APART_ROWS` on the call's side lane, before the rest, and from it on the side lane after the call's, while the
        call's runs the rest behind the main lane's end, then waits for the tree.  The keys go to the set's own when the
        ring rotates them, once the side steps of the set's last call are done.  With `graph` (the call's m and tiles;
        `_lane_graphed`), each lane replays its steps as graphs of the set's (`_lane_graph`: made, run eagerly here and
        captured at their first call, replayed by the driver after, `_replay_lanes`), whose slot takes the keys and the
        call's offsets in one copy, once the side steps of the set's last call are done; returns the graphs and whether
        they screened the call ((None, False) without)."""
        import torch
        cuda = self.device.type == "cuda"
        main = self.main if self.schedule.main_priority == "greatest" else None
        lane_raw, lane = (main.raw, main.stream) if main is not None else (raw, caller)
        j, rot, to, tl = 0, (), None, None
        if side:
            j, rot = sh.turn % len(sh.sets), rotated(self.schedule)
            sh.turn += 1
            k = self._side_turn % len(self.sides)
            to = self.sides[k]
            tl = (self.sides[(k + 1) % len(self.sides)] if sh.m16 >= TREE_APART_ROWS else to) if aside else None
            self._side_turn += 1
        apart = tl is not None and tl is not to
        sh.j, sh.rot = j, rot
        g, screened = None, False
        if graph is not None:
            g, screened = self._lane_graph(w, sh, pipe, steps, side, *graph, aside)
            if g.execs:
                self._replay_lanes(g, sh, j, to, tl, raw, main, self._slot_data(w, sh, index))
                return g, screened
            steps, side = g.steps, g.side_steps
        pipe.b.update({x: sh.sets[j][x].data_ptr() for x in rot if x != "keys"})
        keys = sh.buf("keys") if g is None else g.slot
        pipe.keys.update(dom_a=keys.data_ptr(), unit=keys.data_ptr() + self.nb)
        waited = False
        if self.hashing:
            if cuda and ("keys" in rot or g is not None) and sh.done[j] is not None:   # the side lane may still read set j's keys
                caller.wait_event(sh.done[j])
                waited = True
            stream = caller if self.uploads.cuda else None
            data = self._slot_data(w, sh, index) if g is not None else self.keys(self.roots[w], index, sh.m16)
            self.uploads.put(keys, data, stream)
        if cuda and main is not None:               # the main lane starts after the caller's x, y and keys
            main.event.record(caller)
            lane.wait_event(main.event)
        if cuda and side and sh.done[j] is not None and not waited:  # and rewrites set j only once its side steps are done
            lane.wait_event(sh.done[j])
        cut, side_cut = (g.cut, g.side_cut) if g is not None else pipe.lane_cuts if aside else (None, 0)
        self._launching = lane
        self.kernel.dev.stream = lane_raw
        try:
            with torch.cuda.stream(lane) if g is not None else nullcontext():
                for i, f in enumerate(steps):
                    if i == cut and cuda:
                        tl.event.record(lane)
                    f()
            if tl is not None:
                self.kernel.dev.stream = tl.raw
                with torch.cuda.stream(tl.stream) if g is not None else nullcontext():
                    if cuda:
                        tl.stream.wait_event(tl.event)
                    for f in side[:side_cut]:
                        f()
                if g is None:
                    size, nb, p = pipe.size["tree_a"], self.nb, index - self.cur.base
                    with torch.cuda.stream(tl.stream) if cuda else nullcontext():
                        self.roots_buf[nb * p:nb * (p + 1)].copy_(sh.buf("tree_a")[size - nb:size])
                if cuda and apart:
                    to.tree_event.record(tl.stream)
            if side:
                self.kernel.dev.stream = to.raw
                with torch.cuda.stream(to.stream) if g is not None else nullcontext():
                    if cuda:
                        to.event.record(lane)
                        to.stream.wait_event(to.event)
                    for f in side[side_cut:]:
                        f()
                    if apart and cuda:
                        to.stream.wait_event(to.tree_event)
                if cuda:
                    sh.done[j] = sh.done[j] or _event()
                    sh.done[j].record(to.stream)
            if cuda and main is not None:           # the caller reads y (and A's root, on the main lane) only after it
                main.event.record(lane)
                caller.wait_event(main.event)
            if g is not None:
                self._capture_lanes(g)
        finally:
            self.kernel.dev.stream = None
            self._launching = None
        return g, screened

    def _slot_data(self, w: int, sh: Any, index: int) -> bytes:
        """A lane graph's slot for the call: its keys padded to `SLOT_BYTES`, then its offsets in the pass's buffers."""
        return (self.keys(self.roots[w], index, sh.m16).ljust(SLOT_BYTES, b"\0")
                + struct.pack("<3q", *self._offsets(index, self.cur.tiles, self.cur.flag_rows)))

    def _replay_lanes(self, g: Any, sh: Any, j: int, to: Any, tl: Any, raw: int, main: Any, data: bytes) -> None:
        """A call's lane graphs replayed by the driver (`_Driver`, on the streams' and events' handles), with `_lanes`'s
        events: the main lane waits for the side steps of set j's last call and uploads the slot, which only the graphs
        read, so the upload is off the chain from the caller's stream, then waits for the caller and replays its graph;
        where A's tree is aside, side lane `tl` (`to`, or from `TREE_APART_ROWS` the one after it) replays the tree graph
        after the main graph's own event past the per-row keying; side lane `to` replays the side graph after the main
        graph, then waits for the tree where `tl` is another; the caller waits for the main lane."""
        D = self.driver
        lane = main.raw if main is not None else raw
        done = sh.done[j]
        if done is not None:
            D.wait(lane, done)
        self.uploads.put_raw(g.slot_ptr, data, lane, D)
        if main is not None:
            D.record(main.event, raw)
            D.wait(lane, main.event)
        main_exec, tree_exec, side_exec = g.execs
        apart = bool(tree_exec) and tl is not to
        D.launch(main_exec, lane)
        D.record(to.event, lane)
        if tree_exec:
            D.wait(tl.raw, g.ev_tree)
            D.launch(tree_exec, tl.raw)
            if apart:
                D.record(to.tree_event, tl.raw)
        D.wait(to.raw, to.event)
        D.launch(side_exec, to.raw)
        if apart:
            D.wait(to.raw, to.tree_event)
        if done is None:
            done = sh.done[j] = _event()
        D.record(done, to.raw)
        if main is not None:
            D.record(main.event, lane)
            D.wait(raw, main.event)
