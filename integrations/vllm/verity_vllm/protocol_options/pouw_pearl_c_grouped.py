"""Pearl-C's grouped call for a fused-MoE layer on an H100 or H200 (`pouw_pearl_c_device.PearlCDeviceRun` on pearl_c_sm90's
kernels, ``grouped.py`` beside its ``run.py``): every routed expert's rows of one layer in one launch a step, where
`PearlCDeviceRun.gemm` takes a call per expert.

    y = gemm_grouped(run, ws, x, counts, bias=None)

What vLLM hands it, from its expert-parallel dispatch:

* ``x``: the layer's routed rows (BF16, Σ counts × k), gathered by expert in `ws`'s order;
* ``counts``: each expert's rows, host ints (the dispatch's own, after its all-to-all);
* ``ws``: the experts' weight ids, each a weight of the run, all of one (n, k), each in at most one group.

It returns y (BF16, Σ counts × n) in x's row order, plus ``bias`` when given.  Each expert's rows, in blocks of at most
pd_commit's 256 (`grouped.units`), are each a unit: one matmul of the pass (m₁₆ rows, its expert's weight), committed with
its own index, keys, A tree, records and tile leaves, as `PearlCDeviceRun.gemm` commits that block called alone; the indices
follow on in unit order, and an expert with no rows has none.  The counts are read on the host and each call uploads its
units' table, so a layer's call is never captured in a CUDA graph (`graphs` is refused).

The run's schedule places the call's steps on its lanes as it does a dense call's (`pouw_pearl_c_lanes`): the main lane (the
run's own at the greatest priority, joined to the caller's stream by events, or the caller's), and with hashing, the run's
next side lane for the schedule's side groups.  Under ``deferred`` that is the tile leaves, behind an event at the main lane's
end; under ``deferred-a`` A's tree too, where the cubin has pd_tree_g (`grouped.has_tree`): the commitment is then split
(`grouped.SPLIT_STEPS`), and the side lane runs A's tree behind an event right after the per-row keying, beside the forming
and the GEMMs, which read nothing it writes, then gathers A's roots into the pass, then the tile leaves.  Each group rotates
`Schedule.ring` sets of what the side lane reads or writes (`ROTATED`), a set rewritten only once its last call's side steps
are done.  The bytes committed and served are the serial schedule's; under the split, a unit's padding rows (m₁₆ to its
segment's end, outside its matmul) are formed from whatever E_A lines the buffer holds, as a dense call's are.

The experts' weight side (`grouped.STACKED`: B̃, P_B, B's 1/scale, F_B transposed and F_A) is stacked once per group, each
expert's buffer moved into the stack at its position, so a dense call of an expert reads the same bytes.  The call's buffers
are kept at the largest call so far, rounded up to a power of two of rows."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.protocol import Matmul, Shape

from .pouw_pearl_c_host import _event, _put, _zeros
from .pouw_pearl_c_kernel import _load
from .pouw_pearl_c_screen import screen_into

#: the call buffers' least rows
MIN_ROWS = 1024
#: the call's buffers a pass poisons (`PearlCDeviceRun.poison`): all but A's words, y, pd_commit's counters and the table
UNPOISONED = frozenset({"xa", "y", "pd_count", "meta"})
#: the call's buffers a side lane reads or writes, which a group keeps a set of per ring slot: the tile leaves' C~ and U, the
#: table (the units' Args and keys, the roots' nodes), and A's tree with seed_A and line_a
ROTATED = ("c", "u", "meta", "tree_a", "seed_a", "line_a")
#: the side groups a grouped call runs off its main lane, by its steps
SIDE_STEPS = {"tile_hash": ("hash_leaf",), "a_tree": ("tree_seed_a",)}


class _Group:
    """One group's stacked weight side and its call buffers (`grouped.sizes` at ``rows`` and ``count``), with ``sets`` of the
    `ROTATED` ones (the first ``b``'s own), the event after each set's last side steps (``done``) and the next call's set
    (``turn``)."""

    def __init__(self, run: Any, G: Any, ws: tuple[int, ...]):
        import torch
        self.G, self.ws = G, ws
        first, R = run.weights[ws[0]], run.kernel.run
        self.n, self.k = first.n, first.k
        size = R.Pipeline.sizes(R.BM, self.n, self.k, R.BM)
        self.stacked = {}
        for name in G.STACKED:
            parts = [run.weights[w].b[name] for w in ws]
            if {len(p) for p in parts} != {size[name]}:
                raise ValueError(f"{run.entry}: an expert's {name} isn't its {size[name]} bytes, so the stack's stride isn't")
            stack = torch.cat(parts)
            self.stacked[name] = stack
            for w, t in zip(ws, stack.split([len(p) for p in parts]), strict=True):
                run.weights[w].b[name] = t
        self.rows, self.count, self.b, self.grouped, self.poisoned = 0, 0, {}, None, None
        self.sets: list[dict[str, Any]] = []
        self.done: list[Any] = []
        self.turn = 0

    def fit(self, run: Any, rows: int, ring: int = 1) -> None:
        """Call buffers for `rows` rows of A and `ring` sets of the `ROTATED` ones, made anew (poisoned in a poisoned pass) when
        the last ones are smaller or fewer, once the side lanes are done with them."""
        if rows <= self.rows and ring <= len(self.sets):
            return
        G, R = self.G, run.kernel.run
        run.join_sides()
        if self.grouped is not None:
            self.grouped.free()
        self.rows = max(MIN_ROWS, R.pow2(max(rows, self.rows)))
        self.count = self.rows // G.SEG
        size = G.sizes(R, self.rows, self.count, self.n, self.k)
        self.b = {x: _zeros(s, run.device) for x, s in size.items()}
        self.sets = [{x: self.b[x] for x in ROTATED}, *({x: _zeros(size[x], run.device) for x in ROTATED} for _ in range(ring - 1))]
        self.done, self.turn = [None] * ring, 0
        ptrs = {**{x: t.data_ptr() for x, t in self.b.items()}, **{x: t.data_ptr() for x, t in self.stacked.items()}}
        self.grouped = G.Grouped(R, run.kernel.dev, self.n, self.k, len(self.ws), self.rows, self.count, ptrs, run.const_keys)
        self.poisoned = None

    def poison(self, cur: Any) -> None:
        """The call buffers, every set's, at the pass's poison byte, once a pass."""
        if cur.poison is None or self.poisoned is cur:
            return
        for bufs in (self.b, *self.sets[1:]):
            for x, t in bufs.items():
                if x not in UNPOISONED:
                    t.fill_(cur.poison)
        self.poisoned = cur

    def take(self) -> tuple[int, dict[str, Any]]:
        """The next call's set and its buffers (``b`` with the set's `ROTATED`), the table's pointers moved to them."""
        j = self.turn % len(self.sets)
        self.turn += 1
        self.grouped.b.update({x: t.data_ptr() for x, t in self.sets[j].items()})
        return j, {**self.b, **self.sets[j]}


def module(run: Any) -> Any:
    """``grouped.py`` beside the run's kernel's ``run.py``, its digest in the run's details."""
    G, sha = _load("pearl_c_grouped", Path(run.kernel.run.__file__).with_name("grouped.py"))
    run.details["grouped"] = {"source_sha256": sha}
    return G


def group(run: Any, ws: tuple[int, ...]) -> _Group:
    """The run's group of experts `ws`, stacked on first use."""
    groups = run.__dict__.setdefault("_groups", {})
    if ws in groups:
        return groups[ws]
    if run.graphs:
        raise ValueError(f"{run.entry}: a grouped call is never captured, and this run captures calls in graphs")
    if len(set(ws)) != len(ws) or not ws:
        raise ValueError(f"{run.entry}: a group's experts are distinct weights, not {ws}")
    taken = {w for g in groups for w in g} & set(ws)
    if taken:
        raise ValueError(f"{run.entry}: weights {sorted(taken)} are already another group's")
    if len({(run.weights[w].n, run.weights[w].k) for w in ws}) != 1:
        raise ValueError(f"{run.entry}: a group's experts are of one (n, k)")
    G = module(run)
    groups[ws] = _Group(run, G, ws)
    for key in [key for key in run.pipes if key[0] in ws]:
        del run.pipes[key]
    return groups[ws]


def gemm_grouped(run: Any, ws: Any, x: Any, counts: Any, bias: Any = None) -> Any:
    """One fused-MoE layer's call (the module's docstring), under inference mode as vLLM runs it."""
    import torch
    if torch.is_inference_mode_enabled():
        return _gemm_grouped(run, tuple(ws), x, [int(c) for c in counts], bias)
    with torch.inference_mode():
        return _gemm_grouped(run, tuple(ws), x, [int(c) for c in counts], bias)


def _side(run: Any, G: Any) -> frozenset[str]:
    """The steps the call runs on a side lane: with hashing, the schedule's side groups', A's tree only where the cubin has
    pd_tree_g."""
    if not run.hashing:
        return frozenset()
    side = run.schedule.side if "a_tree" not in run.schedule.side or G.has_tree(run.kernel.dev) else \
        run.schedule.side - {"a_tree"}
    return frozenset(x for group in side for x in SIDE_STEPS[group])


def _steps(run: Any, G: Any, split: bool) -> list[str]:
    """The call's step names: `grouped.STEPS` (`grouped.SPLIT_STEPS` when `split`) less the hashing ones without hashing and
    the omitted ones, a fused launch dropped when any step it fuses is omitted."""
    fused = run.kernel.run.Pipeline.FUSED
    return [x for x in (G.SPLIT_STEPS if split else G.STEPS) if (run.hashing or x not in G.HASH_STEPS) and x not in run.omit
            and not (x in fused and run.omit & set(fused[x]))]


def _gemm_grouped(run: Any, ws: tuple[int, ...], x: Any, counts: list[int], bias: Any) -> Any:
    import torch
    from .pouw_pearl_c_device import CALL_CAPACITY
    if set(run.schedule.side) - set(SIDE_STEPS):
        raise ValueError(f"{run.entry}: a grouped call runs only {', '.join(sorted(SIDE_STEPS))} on a side lane, not "
                         f"{run.schedule.name}'s {', '.join(sorted(run.schedule.side))}")
    g, cur, R = group(run, ws), run.cur, run.kernel.run
    G, n, k = g.G, g.n, g.k
    if len(counts) != len(ws) or x.shape != (sum(counts), k) or min(counts, default=0) < 0:
        raise ValueError(f"{run.entry}: x {tuple(x.shape)} isn't Σ counts = {sum(counts)} rows of k = {k} over {len(ws)} experts")
    us = G.units(counts, n, R.PD_MAX_ROWS)
    if not us:
        return x.new_zeros((0, n)) if bias is None else x.new_zeros((0, n)) + bias
    rows, tiles, index, real = sum(u.seg for u in us), sum(u.tiles for u in us), len(cur.matmuls), len(x)
    if run.hashing and (cur.tiles + tiles > run.capacity or index + len(us) > CALL_CAPACITY
                        or cur.flag_rows + real > len(run.flag_buf)):
        raise RuntimeError(f"{run.entry}: the pass is over its capacity ({run.capacity} tiles, {CALL_CAPACITY} calls, "
                           f"{len(run.flag_buf)} screened rows); end it and begin another")
    side = _side(run, G)
    split = "tree_seed_a" in side
    cuda = run.device.type == "cuda"
    raw = run.stream()
    caller = run.torch_stream(raw) if cuda else None
    main = run.main if run.schedule.main_priority == "greatest" else None
    lane_raw, lane = (main.raw, main.stream) if main is not None else (raw, caller)
    g.fit(run, rows, run.schedule.ring if side else 1)
    g.poison(cur)
    j, b = g.take()
    if cuda and g.done[j] is not None:                  # set j's last side steps still read its table, C~, U and tree
        caller.wait_event(g.done[j])
    keys = (b"".join(run.keys(run.roots[ws[u.pos]], index + i, u.m16) for i, u in enumerate(us)) if run.hashing
            else bytes(G.KEY_BYTES * len(us)))
    meta = g.grouped.meta(us, keys, run.leaves.data_ptr() + run.nb * cur.tiles, split)
    _put(b["meta"], meta)
    at = G.Grouped.rows_at(us)
    idx = b["meta"][at:at + 8 * real].view(torch.int64)
    nodes = b["meta"][at + 8 * real:at + 8 * (real + len(us))].view(torch.int64)
    xa = b["xa"][:4 * g.rows * k].view(torch.float32).view(g.rows, k)
    xa[:rows].zero_()
    xa.index_copy_(0, idx, x.float())
    by = dict(g.grouped.steps(us, split))
    names = _steps(run, G, split)
    tree = b["tree_a"][:G.tree_bytes(R) * g.count].view(-1, run.nb)
    roots = run.roots_buf[:run.nb * (index + len(us))].view(-1, run.nb)[index:]
    on_main = [x for x in names if x not in side]
    cut = on_main.index("rows_a") + 1 if "rows_a" in on_main else 0
    to = None
    if side:
        to = run.sides[run._side_turn % len(run.sides)]
        run._side_turn += 1
    if cuda and main is not None:                       # the main lane starts after the caller's x and table
        main.event.record(caller)
        lane.wait_event(main.event)
    dev = run.kernel.dev
    try:
        dev.stream = lane_raw
        for name in on_main[:cut]:
            by[name]()
        if split and cuda:                              # A's tree reads only the per-row keying's output
            to.event.record(lane)
        for name in on_main[cut:]:
            by[name]()
        if side:
            dev.stream = to.raw
            with torch.cuda.stream(to.stream) if cuda else nullcontext():
                if split:
                    if cuda:
                        to.stream.wait_event(to.event)
                    if "tree_seed_a" in names:
                        by["tree_seed_a"]()
                    roots.copy_(tree.index_select(0, nodes))
                if cuda:
                    to.event.record(lane)
                    to.stream.wait_event(to.event)
                for name in names:
                    if name in side and name != "tree_seed_a":
                        by[name]()
            if cuda:
                g.done[j] = g.done[j] or _event()
                g.done[j].record(to.stream)
    finally:
        dev.stream = None
    if cuda and main is not None:                       # the caller reads y, the rows' statistics and A's roots after it
        main.event.record(lane)
        caller.wait_event(main.event)
    mms = [Matmul(index + i, Shape(u.m16, k, n), ws[u.pos]) for i, u in enumerate(us)]
    cur.matmuls.extend(mms)
    if run.hashing:
        if not split:
            roots.copy_(tree.index_select(0, nodes))
        flags = run.flag_buf[cur.flag_rows:cur.flag_rows + real]
        alpha = b["alpha_a"][:4 * g.rows].view(torch.int32).index_select(0, idx).view(torch.uint8)
        aux = b["aux_a"][:8 * g.rows].view(torch.int64).index_select(0, idx).view(torch.uint8)
        rows_in = x if x.is_contiguous() and x.dtype == torch.bfloat16 else xa.index_select(0, idx)
        screen_into(flags, rows_in, alpha, aux, run.fused_screen)
        cur.flag_rows += real
        if cur.retain is not None:
            if split and cuda:                          # A's tree is the side lane's
                caller.wait_event(g.done[j])
            slot = G.tree_bytes(R)
            for i, (u, mm) in enumerate(zip(us, mms, strict=True)):
                run._retained(cur, mm, u.m, flags[u.src:u.src + u.m], xa[u.row:u.row + u.m16],
                              b["tree_a"][slot * i:slot * i + run.nb * (2 * R.pow2(u.m16) - 1)])
        cur.tiles += tiles
    y = b["y"][:2 * g.rows * n].view(torch.bfloat16).view(g.rows, n).index_select(0, idx)
    return y if bias is None else y + bias
