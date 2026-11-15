"""A Pearl-C call's eager launches on its schedule's lanes (`lanes`), over torch's streams and the kernel's driver: `LaneCalls`,
a part of the device run."""
from __future__ import annotations

import hashlib
import struct
from contextlib import nullcontext
from typing import Any, cast


from verity_torch.pouw.pearl_c.ends import _RunState
from verity_torch.pouw.pearl_c.graphs import SLOT_BYTES
from verity_torch.pouw.pearl_c.host import _Driver, _event, _Lane, _stream
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.lanes import (
    STEP_GROUPS,
    TREE_APART_ROWS,
    rotated,
    splits_at,
)


class LaneCalls(_RunState):
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
        j, to, tl = 0, None, None
        rot: tuple[str, ...] = ()
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
                        cast("_Lane", tl).event.record(lane)    # a cut is of a tree aside, which took its lane above
                    f()
            if tl is not None:
                self.kernel.dev.stream = tl.raw
                with torch.cuda.stream(tl.stream) if g is not None else nullcontext():
                    if cuda:
                        tl.stream.wait_event(tl.event)
                    for f in side[:side_cut]:
                        f()
                if g is None:
                    size, nb, p = pipe.size["tree_a"], self.nb, index - self.cur.first
                    with torch.cuda.stream(tl.stream) if cuda else nullcontext():
                        self.roots_buf[nb * p:nb * (p + 1)].copy_(sh.buf("tree_a")[size - nb:size])
                if cuda and apart:
                    cast("_Lane", to).tree_event.record(tl.stream)   # a tree lane is taken with the call's side lane
            if side:
                to = cast("_Lane", to)              # the call's side lane, taken above with its side steps
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
        D = cast("_Driver", self.driver)    # no driver is unchecked: its first call fails
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


__all__ = ["LaneCalls"]
