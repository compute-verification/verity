"""The Pearl-C device path's schedule (`PearlCDeviceRun`'s lanes): the lane each call step runs on and the per-call buffers a
side lane touches; a call's eager launches on those lanes are the device run's (``LaneCalls``).

The schedule (`verity.protocols.resource_accounting.compute.pouw.verifier.serving.Schedule`, by name from `SCHEDULES`; the decode plan's step 2)
places each call's step groups (`STEP_GROUPS`) on lanes.  Under ``deferred``, the measured one, the call's tile
hashing (`SIDE_IO`: the tile leaves) runs on a side lane at the least priority behind an event after the GEMM, while the next
calls go on, and the rest of the call runs on a main lane of the run's own at the greatest priority, joined to the caller's
stream by events on both sides (``deferred-caller`` leaves it on the caller's stream).

Under ``deferred-a`` A's tree goes beside the tile hashing too, where the kernel's pipeline splits A's commitment
(`run.Pipeline`'s ``split_a``: ``rows_a``, pd_commit's mode 1, then ``tree_seed_a``, pd_tree; outside pd, where the pipeline
has it, the same two steps over the generic kernels, A's level 0 among the tree's), with hashing (``LaneCalls._aside``,
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
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.verifier.serving import (
    SCHEDULES,
    Schedule,
    StepIO,
    hash_format,
    schedule_refusals,
)

__all__ = ["whole_options"]

#: each call step's group (`verity.protocols.resource_accounting.compute.pouw.verifier.serving.GROUPS`), which a schedule places on a lane.  The per-row
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
#: call's own: on the H200 the own lane was the faster at m₁₆ 16 and the other at 64, at 7 of a measured model's 8 dense shapes each
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
                  whole_prefetch_b: bool = False, whole_shape_joins: bool = False, whole_pd_shared: bool = False) -> None:
    """Refuse a whole capture's options, and a pass end's, where `kernel` or the others can't run them.  `whole_defer_a`
    (with `whole_defer`, on a kernel whose pipeline splits A's commitment) also defers each call's A tree within a whole
    capture (`pouw_pearl_c_graphs`), and `whole_fork_stats` (with `whole_defer`) runs each call's A scales there beside its
    A commitment; `whole_tree_sm` (with `whole_defer_a`) launches the call's GEMMs beside its A tree with one persistent CTA
    fewer, leaving the tree's CTA its SM; `whole_shape_joins` (with `whole_defer`) joins a call's side steps only before
    the next call of its shape; `aside_ends` (with `lag_ends`) runs a lagged end's tile tree aside;
    `whole_prefetch_b` (with `whole_defer`) prefetches each call's B̃ into L2 beside its A commitment (`run.Pipeline.prefetch_b`,
    which writes nothing); `whole_pd_shared` launches each call's pd_commit and pd_tree in half an SM's shared memory
    (`run.Pipeline.pd_shared`)."""
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
    if whole_pd_shared and not hasattr(kernel.run.Pipeline, "pd_shared"):
        raise ValueError(f"{entry}: whole_pd_shared launches pd_commit and pd_tree in half an SM's shared memory "
                         f"(`Pipeline.pd_shared`), which the kernel's pipeline lacks")


def schedule_of(name: str | Schedule, scheme: Any, kernel: Any = None) -> Schedule:
    """`name`'s schedule (`verity.protocols.resource_accounting.compute.pouw.verifier.serving.SCHEDULES`), refused where the scheme's format, this path or (when
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
    return frozenset(name for name in SCHEDULES if _admits(name, scheme, kernel))


def _admits(name: str, scheme: Any, kernel: Any) -> bool:
    """Whether `schedule_of` admits schedule `name` (it raises ValueError for one it refuses)."""
    try:
        schedule_of(name, scheme, kernel)
    except ValueError:
        return False
    return True
