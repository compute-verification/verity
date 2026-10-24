"""The Pearl-C device path's schedule (`pouw_pearl_c_device.PearlCDeviceRun`'s lanes): the lane each call step runs on, the
per-call buffers a side lane touches, and a call's eager launches on those lanes.

The schedule (`verity.core.protocols.pouw.serving.Schedule`, by name from `SCHEDULES`; the decode plan's step 2,
bc-b139c29c's) places each call's step groups (`STEP_GROUPS`) on lanes.  Under ``deferred``, the measured one, the call's tile
hashing (`SIDE_IO`: the tile leaves) runs on a side lane at the least priority behind an event after the GEMM, while the next
calls go on, and the rest of the call runs on a main lane of the run's own at the greatest priority, joined to the caller's
stream by events on both sides (``deferred-caller`` leaves it on the caller's stream, #564's first cut).

Under ``deferred-a`` A's tree goes beside the tile hashing too, where the kernel's pipeline splits A's commitment
(`run.Pipeline`'s ``split_a``: ``rows_a``, pd_commit's mode 1, then ``tree_seed_a``, pd_tree), with hashing, outside a
retained pass (its record reads the tree as the call ends; `LaneCalls._aside`).  The main lane keeps the per-row keying
forming reads (``row_seeds``: each real row's digest, seed, line key and E_A line), and the side lane builds A's levels,
seed_A, line_a and the padding rows' E_A lines under it, then copies A's root to the pass.  The padding rows (m₁₆ … the
kernels' tile, outside the matmul) are the GEMM's alone: no tile leaf reads them (the leaves stop at m₁₆ rows) and y is
served to m, so the GEMM, which runs before the side lane, forms them from the lines their set last held.  Every committed
and served byte is the serial schedule's; the padding rows' products are not.

Each shape rotates `Schedule.ring` sets of the buffers the side lane touches (`rotated`, with the call's keys, dom_a ‖ unit,
when A's tree moves), and a call reuses a set only once the side lane has finished with it; the pass's end, and the next
pass's begin, wait for the side lane.  The same kernels read and write the same bytes, so the commitments are the serial
schedule's.  A schedule changes only between passes (`begin`), and one whose placement this path, the kernel or the scheme's
format can't run is refused (`schedule_of`).
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from verity.core.protocols.pouw.serving import SCHEDULES, Schedule, StepIO, hash_format, schedule_refusals

#: each call step's group (`verity.core.protocols.pouw.serving.GROUPS`), which a schedule places on a lane.  The per-row
#: keying forming reads (A's pads, each row's digest, each real row's seed) is ``row_seeds``; A's hm96 leaves, tree and
#: unit seed are ``a_tree``.  ``commit_a`` (`run.Pipeline.FUSED`), which also writes each row's seed and E_A line, is
#: ``row_seeds``, and so is ``rows_a``, its mode 1; ``tree_seed_a`` (A's levels, seed_A and line_a) is ``a_tree``
STEP_GROUPS = {"pad_a": "row_seeds", "rowdig_a": "row_seeds", "hash_rows_a": "a_tree", "tree_a": "a_tree", "seed_a": "a_tree",
               "seeds_a": "row_seeds", "lines_ea": "forming", "stats_a": "forming", "eprime_a": "forming", "form_a": "forming",
               "af": "cleanup", "pa": "cleanup", "gemm_pearlc": "cleanup", "hash_leaf": "tile_hash", "commit_a": "row_seeds",
               "rows_a": "row_seeds", "tree_seed_a": "a_tree"}
#: the steps this path can run on a side lane, with the per-call buffers they read and write (the constant keys and the pass's
#: leaf buffer aside, which no later call rewrites): each shape rotates these, so a call never overwrites what the side lane
#: still reads.  The tile hashing reads C~ and U; A's tree (``tree_seed_a``, only where the pipeline splits A's commitment)
#: reads level 0 and the call's keys and writes the levels, seed_A, line_a and the padding rows' E_A lines.  The screen runs
#: on the caller
SIDE_IO = {"hash_leaf": StepIO("tile_hash", frozenset({"c", "u"}), frozenset({"segdig_t"})),
           "tree_seed_a": StepIO("a_tree", frozenset({"tree_a", "keys"}), frozenset({"tree_a", "seed_a", "line_a", "ea_codes"}))}
SIDE_GROUPS = frozenset(io.group for io in SIDE_IO.values())


def rotated(schedule: Schedule) -> tuple[str, ...]:
    """The per-call buffers each shape rotates under `schedule`: those its side-lane steps read or write (``keys``, the
    call's dom_a ‖ unit, among them when A's tree moves)."""
    return tuple(sorted({b for io in SIDE_IO.values() if io.group in schedule.side for b in io.reads | io.writes}))


def splits(kernel: Any) -> bool:
    """Whether `kernel`'s pipeline splits A's commitment (`run.Pipeline.FUSED`'s ``tree_seed_a``), so A's tree may leave
    the GEMM's lane."""
    return "tree_seed_a" in getattr(kernel.run.Pipeline, "FUSED", {})


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


class LaneCalls:
    """`PearlCDeviceRun`'s lanes, as its base class: the run's ``schedule``, ``main``, ``side``, ``hashing``, ``cur``,
    ``kernel``, ``device``, ``nb``, ``keys``, ``uploads``, ``roots`` and ``roots_buf``."""

    def _aside(self, sh: Any) -> bool:
        """Whether a call of `sh` runs A's tree on the side lane (the module's docstring): the schedule moves ``a_tree``,
        hashing is on, the pass isn't retained, and the kernel's pipeline splits A's commitment at this shape (pd's)."""
        R = self.kernel.run
        return ("a_tree" in self.schedule.side and self.hashing and self.cur is not None and self.cur.retain is None
                and splits(self.kernel) and R.pd_variant(sh.m16, sh.n, sh.k) is not None)

    def _side_groups(self, sh: Any) -> frozenset[str]:
        """The groups a call of `sh` runs on the side lane, with hashing: the schedule's, but A's tree only `_aside`."""
        if not self.hashing:
            return frozenset()
        return frozenset(g for g in self.schedule.side if g != "a_tree" or self._aside(sh))

    def _lanes(self, w: int, sh: Any, pipe: Any, steps: list, side: list, raw: int, caller: Any, index: int,
               aside: bool) -> None:
        """A call's steps launched eagerly on the schedule's lanes: the main lane (the caller's stream, or the run's own after
        an event on the caller's), then, behind an event after it, the side lane on the ring's next set, which ends with A's
        root copied to the pass when A's tree is there (`aside`).  The keys go to the set's own when the ring rotates them,
        once the side lane is done with the set."""
        import torch
        cuda = self.device.type == "cuda"
        main = self.main if self.schedule.main_priority == "greatest" else None
        lane_raw, lane = (main.raw, main.stream) if main is not None else (raw, caller)
        j, rot = 0, ()
        if side:
            j, rot = sh.turn % len(sh.sets), rotated(self.schedule)
            sh.turn += 1
            pipe.b.update({x: sh.sets[j][x].data_ptr() for x in rot if x != "keys"})
        sh.j, sh.rot = j, rot
        keys = sh.buf("keys")
        pipe.keys.update(dom_a=keys.data_ptr(), unit=keys.data_ptr() + self.nb)
        waited = False
        if self.hashing:
            if cuda and "keys" in rot and sh.done[j] is not None:   # the side lane may still read set j's keys
                caller.wait_event(sh.done[j])
                waited = True
            self.uploads.put(keys, self.keys(self.roots[w], index, sh.m16), caller if self.uploads.cuda else None)
        if cuda and main is not None:               # the main lane starts after the caller's x, y and keys
            main.event.record(caller)
            lane.wait_event(main.event)
        if cuda and side and sh.done[j] is not None and not waited:  # and rewrites set j only once the side lane is done
            lane.wait_event(sh.done[j])
        self._launching = lane
        self.kernel.dev.stream = lane_raw
        try:
            for f in steps:
                f()
            if side:
                if cuda:
                    self.side.event.record(lane)
                    self.side.stream.wait_event(self.side.event)
                self.kernel.dev.stream = self.side.raw
                for f in side:
                    f()
                if aside:
                    size, nb = pipe.size["tree_a"], self.nb
                    with torch.cuda.stream(self.side.stream) if cuda else nullcontext():
                        self.roots_buf[nb * index:nb * (index + 1)].copy_(sh.buf("tree_a")[size - nb:size])
                if cuda:
                    sh.done[j] = sh.done[j] or torch.cuda.Event()
                    sh.done[j].record(self.side.stream)
            if cuda and main is not None:           # the caller reads y (and A's root, on the main lane) only after it
                main.event.record(lane)
                caller.wait_event(main.event)
        finally:
            self.kernel.dev.stream = None
            self._launching = None
