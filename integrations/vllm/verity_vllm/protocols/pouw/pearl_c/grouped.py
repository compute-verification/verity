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
follow on in unit order, and an expert with no rows has none.

The counts are read on the host, so a call's layout (its units, `_Call`) is the host's, and so is its table: each unit's
records and keys (`grouped.Grouped.meta`), the expert map, the gathered rows' rows of A, the units' root nodes, then the
call's byte offsets in the pass's buffers (`OFFS_BYTES`).  A layout's table is made once, and each call fills in its keys,
its leaves' address and its offsets in a pinned host slot (`_Group.table`), uploaded in one copy.  The call stages x as FP32
behind a zero row, and its steps read it from there: A's rows into their segments (one gather by the layout's row sources,
`_Call.src`, a filler row the zero row), the launches, and with hashing one scatter (`pouw_pearl_c_screen.scatter_into`)
that screens the rows, at their rows of A, into the pass's flags and copies the units' A roots into the pass at the table's
offsets.  So no step reads anything of the call's but the table, its layout's row sources and the staged x, and with ``graphs``
each lane's steps are captured as a CUDA graph at a layout's second call and replayed by the driver after (`_replay`),
as a dense call's lane graphs are (`pouw_pearl_c_graphs`).  A layout's graphs serve only calls of the same counts; any
other call runs the same steps eagerly.  vLLM's own capture of a forward can't take a grouped call (its counts are the
host's), and a call inside one is refused.

The run's schedule places the call's steps on its lanes as it does a dense call's (`pouw_pearl_c_lanes`): the main lane (the
run's own at the greatest priority, joined to the caller's stream by events, or the caller's), and with hashing, the run's
next side lane for the schedule's side groups.  Under ``deferred`` that is the tile leaves, behind an event at the main lane's
end; under ``deferred-a`` A's tree too, where the cubin has pd_tree_g (`grouped.has_tree`): the commitment is then split
(`grouped.SPLIT_STEPS`), and the side lane runs A's tree behind an event right after the per-row keying, beside the forming
and the GEMMs, which read nothing it writes, then scatters A's roots into the pass, then the tile leaves.  Each group rotates
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

from verity_vllm.protocols.pouw.pearl_c.passes import call_index
from verity_vllm.protocols.pouw.pearl_c.host import _Driver, _event, _put, _zeros
from verity_vllm.protocols.pouw.pearl_c.kernel import _load
from verity_vllm.protocols.pouw.pearl_c.screen import scatter_into, screen_into

#: the call buffers' least rows
MIN_ROWS = 1024
#: the call's buffers a pass poisons (`PearlCDeviceRun.poison`): all but A's words, y, pd_commit's counters and the table
UNPOISONED = frozenset({"xa", "y", "pd_count", "meta"})
#: the call's buffers a side lane reads or writes, which a group keeps a set of per ring slot: the tile leaves' C~ and U, the
#: table (the units' Args and keys, the roots' nodes, the pass offsets), and A's tree with seed_A, line_a and the gathered roots
ROTATED = ("c", "u", "meta", "tree_a", "seed_a", "line_a", "roots")
#: the side groups a grouped call runs off its main lane, by its steps
SIDE_STEPS = {"tile_hash": ("hash_leaf",), "a_tree": ("tree_seed_a",)}
#: the table's last bytes, after `grouped.Grouped.meta`'s: the call's byte offsets in the pass's buffers its scatter writes at,
#: three int64 as `scatter_into` reads them (A's roots in ``roots_buf``, the flags in ``flag_buf``, and an unused third)
OFFS_BYTES = 24
#: the pinned host slots the tables are uploaded from, each written again only once its last upload is done
HOST_SLOTS = 8
#: the layouts a group keeps (`_Group.call`), with their graphs; past it they are dropped
CALLS_KEPT = 64


class _Call:
    """One call layout of a group over one ring set (`_Group.call`): its table's template (`grouped.Grouped.meta` with no
    keys and its leaves at 0, which `_Group.table` fills in a call at a time), its matmuls' shapes, and its steps by lane:
    ``main`` (A's rows into their segments from the staged x, the main lane's launches, and with hashing the scatter of the
    rows' screen and, unless A's tree is aside, the units' roots), ``tree`` (A's tree, then its roots' gather and scatter)
    and ``side`` (the tile leaves), the main lane's event for ``tree`` before step ``cut``.  ``execs`` are its graphs'
    executables once captured (main, tree or 0, side or 0); off the GPU the scatter takes the call's offsets from ``at``.
    ``src`` is each of A's rows' source in the staged x behind its zero row (0, or 1 + its gathered row)."""

    def __init__(self, run: Any, g: _Group, us: list, split: bool, side: frozenset[str], b: dict[str, Any]):
        import numpy as np
        import torch
        G, R, nb, k, n = g.G, run.kernel.run, run.nb, g.k, g.n
        count, real, rows = len(us), sum(u.m for u in us), sum(u.seg for u in us)
        self.count, self.real, self.tiles = count, real, sum(u.tiles for u in us)
        self.shapes = [(Shape(u.m16, k, n), g.ws[u.pos]) for u in us]
        self.template = np.frombuffer(g.grouped.meta(us, bytes(G.KEY_BYTES * count), 0, split), np.uint8)
        self.leaf_at = (G.REC * np.arange(count) + G.TILES_AT) // 8 + 2
        self.leaf_offs = self.template[:G.REC * count].view("<u8")[self.leaf_at].copy()
        at, meta, t = G.Grouped.rows_at(us), b["meta"], len(self.template)
        self.idx = idx = meta[at:at + 8 * real].view(torch.int64)
        nodes = meta[at + 8 * real:at + 8 * (real + count)].view(torch.int64)
        offs = meta[t:t + OFFS_BYTES].view(torch.int64)
        xa = b["xa"][:4 * g.rows * k].view(torch.float32).view(g.rows, k)
        xz = b["xf"][:4 * (real + 1) * k].view(torch.float32).view(real + 1, k)
        self.xf = xf = xz[1:]
        src = np.zeros(rows, np.int64)
        src[self.template[at:at + 8 * real].view("<i8")] = np.arange(1, real + 1)
        self.src = torch.from_numpy(src).to(run.device)
        self.y = b["y"][:2 * g.rows * n].view(torch.bfloat16).view(g.rows, n)
        self.at, self.seen, self.graphs, self.execs, self.ev_tree = (0, 0), 0, (), (), None

        def rows_a():
            torch.index_select(xz, 0, self.src, out=xa[:rows])
        by = dict(g.grouped.steps(us, split))
        names = _steps(run, G, split)
        on_main = [x for x in names if x not in side]
        self.cut = (on_main.index("rows_a") + 1 if "rows_a" in on_main else 0) + 1 if split else None
        self.main = [rows_a, *(by[x] for x in on_main)]
        self.tree: list = []
        self.side = [by[x] for x in names if x in side and x != "tree_seed_a"]
        if not run.hashing:
            return
        alpha, aux = b["alpha_a"][:4 * g.rows].view(torch.int32), b["aux_a"][:8 * g.rows].view(torch.int64)
        alpha_s, aux_s = b["alpha_s"][:4 * real].view(torch.int32), b["aux_s"][:8 * real].view(torch.int64)
        tree = b["tree_a"][:G.tree_bytes(R) * g.count].view(-1, nb)
        roots = b["roots"][:nb * count].view(count, nb)
        flags_to, roots_to, none = run.flag_buf.view(-1), run.roots_buf, b["roots"][:0]

        def scatter(screened: int, gathered: Any) -> None:
            if run.fused_screen:
                scatter_into(xa, alpha, aux, screened, flags_to, gathered, roots_to, none, roots_to, offs, idx)
                return
            pos, flag_rows = self.at
            if screened:
                torch.index_select(alpha, 0, idx, out=alpha_s)
                torch.index_select(aux, 0, idx, out=aux_s)
                screen_into(run.flag_buf[flag_rows:flag_rows + screened], xf, alpha_s, aux_s, False)
            if gathered.numel():
                roots_to[nb * pos:nb * pos + gathered.numel()].copy_(gathered)

        def screen():
            if not split:
                torch.index_select(tree, 0, nodes, out=roots)
            scatter(real, none if split else roots.view(-1))

        def gather():
            torch.index_select(tree, 0, nodes, out=roots)
            scatter(0, roots.view(-1))
        self.main.append(screen)
        if split:
            self.tree = [*([by["tree_seed_a"]] if "tree_seed_a" in names else []), gather]


class _Group:
    """One group's stacked weight side and its call buffers (`grouped.sizes` at ``rows`` and ``count``, with the staged x
    ``xf`` behind a zero row, the gathered rows' α and (s, ρ) for the unfused screen, and the units' A roots), with ``sets``
    of the `ROTATED` ones (the first ``b``'s own), the event after each set's last side steps (``done``) and the next call's
    set (``turn``); its layouts (``calls``), the pinned host slots their tables go up from (``host``) and the driver that
    uploads them (``driver``, the run's or its own; None off the GPU)."""

    def __init__(self, run: Any, G: Any, ws: tuple[int, ...]):
        import torch
        self.G, self.ws = G, ws
        first, R = run.weights[ws[0]], run.kernel.run
        self.n, self.k, self.max_rows = first.n, first.k, R.PD_MAX_ROWS
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
        self.units_of: dict[tuple[int, ...], list] = {}
        self.calls: dict[tuple, _Call] = {}
        cuda = run.device.type == "cuda"
        self.driver = run.driver if run.driver is not None or not cuda else _Driver(run.kernel.dev.cu)
        self.host, self.host_np, self.host_ptr, self.width, self.host_ev, self.host_turn = None, None, 0, 0, [], 0
        self.replays = 0

    def units(self, counts: tuple[int, ...]) -> list:
        """`grouped.units` of `counts`, kept by counts."""
        got = self.units_of.get(counts)
        if got is None:
            if len(self.units_of) >= CALLS_KEPT:
                self.units_of.clear()
            got = self.units_of[counts] = self.G.units(counts, self.n, self.max_rows)
        return got

    def fit(self, run: Any, rows: int, ring: int = 1) -> None:
        """Call buffers for `rows` rows of A and `ring` sets of the `ROTATED` ones, made anew (poisoned in a poisoned pass) when
        the last ones are smaller or fewer, once the side lanes are done with them, the layouts on them dropped."""
        if rows <= self.rows and ring <= len(self.sets):
            return
        import torch
        G, R = self.G, run.kernel.run
        run.join_sides()
        self.drop(run)
        if self.grouped is not None:
            self.grouped.free()
        self.rows = max(MIN_ROWS, R.pow2(max(rows, self.rows)))
        self.count = self.rows // G.SEG
        size = {**G.sizes(R, self.rows, self.count, self.n, self.k), "xf": 4 * (self.rows + 1) * self.k, "alpha_s": 4 * self.rows,
                "aux_s": 8 * self.rows, "roots": run.nb * self.count}
        size["meta"] += OFFS_BYTES
        self.b = {x: _zeros(s, run.device) for x, s in size.items()}
        self.sets = [{x: self.b[x] for x in ROTATED}, *({x: _zeros(size[x], run.device) for x in ROTATED} for _ in range(ring - 1))]
        self.done, self.turn = [None] * ring, 0
        ptrs = {**{x: t.data_ptr() for x, t in self.b.items()}, **{x: t.data_ptr() for x, t in self.stacked.items()}}
        self.grouped = G.Grouped(R, run.kernel.dev, self.n, self.k, len(self.ws), self.rows, self.count, ptrs, run.const_keys)
        self.poisoned = None
        self.width = -(-size["meta"] // 64) * 64
        self.host = torch.zeros((HOST_SLOTS, self.width), dtype=torch.uint8, pin_memory=run.device.type == "cuda")
        self.host_np, self.host_ptr, self.host_ev = self.host.numpy(), self.host.data_ptr(), [None] * HOST_SLOTS

    def drop(self, run: Any) -> None:
        """The layouts dropped, once the device is done with their graphs."""
        if any(c.graphs for c in self.calls.values()):
            import torch
            torch.cuda.synchronize(run.device)
        self.calls.clear()

    def poison(self, cur: Any) -> None:
        """The call buffers, every set's, at the pass's poison byte, once a pass."""
        if cur.poison is None or self.poisoned is cur:
            return
        for bufs in (self.b, *self.sets[1:]):
            for x, t in bufs.items():
                if x not in UNPOISONED:
                    t.fill_(cur.poison)
        self.b["xf"][:4 * self.k].zero_()
        self.poisoned = cur

    def take(self) -> tuple[int, dict[str, Any]]:
        """The next call's set and its buffers (``b`` with the set's `ROTATED`), the table's pointers moved to them."""
        j = self.turn % len(self.sets)
        self.turn += 1
        self.grouped.b.update({x: t.data_ptr() for x, t in self.sets[j].items()})
        return j, {**self.b, **self.sets[j]}

    def call(self, run: Any, counts: tuple[int, ...], us: list, split: bool, side: frozenset[str], j: int,
             b: dict[str, Any]) -> _Call:
        """The layout of `counts` on set j (`take`'s), made at its first call."""
        key = (counts, split, side, j, run.hashing, run.omit, run.schedule.name, run.fused_screen)
        c = self.calls.get(key)
        if c is None:
            if len(self.calls) >= CALLS_KEPT:
                self.drop(run)
            c = self.calls[key] = _Call(run, self, us, split, side, b)
        return c

    def table(self, run: Any, c: _Call, keys: bytes | None, leaves: int, pos: int, flag_rows: int) -> tuple[int, int]:
        """The call's table in the next pinned host slot, once its last upload is done: `c`'s template with each unit's keys
        (none without hashing), its tile leaves from address `leaves` and the call's offsets (its first unit's position
        `pos` in the pass, its first flag row).  Returns the slot and the table's bytes."""
        import numpy as np
        G, i = self.G, self.host_turn % HOST_SLOTS
        self.host_turn += 1
        if self.host_ev[i] is not None:
            self.host_ev[i].synchronize()
        h, t, recs = self.host_np[i], len(c.template), G.REC * c.count
        h[:t] = c.template
        if keys:
            h[:recs].reshape(c.count, G.REC)[:, G.KEYS_AT:G.KEYS_AT + G.KEY_BYTES] = np.frombuffer(keys, np.uint8).reshape(c.count, -1)
        h[:recs].view("<u8")[c.leaf_at] = c.leaf_offs + np.uint64(leaves)
        h[t:t + OFFS_BYTES].view("<i8")[:] = (run.nb * pos, 2 * flag_rows, 0)
        return i, t + OFFS_BYTES

    def upload(self, b: dict[str, Any], i: int, n: int, stream: int) -> None:
        """Slot i's first `n` bytes into the set's table (``b``'s), on raw stream `stream`: the driver's copy, then the slot's
        event; off the GPU a copy."""
        if self.driver is None:
            _put(b["meta"], self.host_np[i, :n].tobytes())
            return
        self.driver.copy_in(b["meta"].data_ptr(), self.host_ptr + i * self.width, n, stream)
        ev = self.host_ev[i] = self.host_ev[i] or _event()
        self.driver.record(ev, stream)


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
        return _gemm_grouped(run, tuple(ws), x, tuple(int(c) for c in counts), bias)
    with torch.inference_mode():
        return _gemm_grouped(run, tuple(ws), x, tuple(int(c) for c in counts), bias)


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


def _eager(run: Any, g: _Group, c: _Call, j: int, cuda: bool, caller: Any, lane: Any, lane_raw: int, main: Any,
           to: Any) -> None:
    """`c`'s steps launched on the lanes: the main lane's (the event for A's tree recorded before step ``cut``), then on side
    lane `to` A's tree behind that event and the tile leaves behind the main lane's end, set j's event after them."""
    import torch
    dev = run.kernel.dev
    try:
        dev.stream = lane_raw
        with torch.cuda.stream(lane) if cuda else nullcontext():
            for s, f in enumerate(c.main):
                if s == c.cut and cuda:
                    to.event.record(lane)
                f()
        if to is not None:
            dev.stream = to.raw
            with torch.cuda.stream(to.stream) if cuda else nullcontext():
                if c.tree:
                    if cuda:
                        to.stream.wait_event(to.event)
                    for f in c.tree:
                        f()
                if cuda:
                    to.event.record(lane)
                    to.stream.wait_event(to.event)
                for f in c.side:
                    f()
            if cuda:
                g.done[j] = g.done[j] or _event()
                g.done[j].record(to.stream)
    finally:
        dev.stream = None


def _capture(run: Any, c: _Call) -> None:
    """`c`'s lanes' steps captured as its graphs (the main lane's, recording the external event ``ev_tree`` before step
    ``cut`` where A's tree is aside; the tree's; the side lane's), after a call ran them eagerly (their first use syncs: the
    GEMMs' shared-memory probes, the scatter's compile), and their executables for the driver."""
    import torch
    capture = torch.cuda.Stream(device=run.device)
    capture.wait_stream(torch.cuda.current_stream(run.device))
    if c.cut is not None:
        c.ev_tree = torch.cuda.Event(external=True)
    made = []
    for steps, cut in ((c.main, c.cut), (c.tree, None), (c.side, None)):
        if not steps:
            made.append(None)
            continue
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture):
            graph.capture_begin(capture_error_mode="thread_local")
            run.kernel.dev.stream = capture.cuda_stream
            try:
                for s, f in enumerate(steps):
                    if s == cut:
                        c.ev_tree.record(capture)
                    f()
            finally:
                graph.capture_end()
                run.kernel.dev.stream = None
        made.append(graph)
    torch.cuda.current_stream(run.device).wait_stream(capture)
    c.graphs = tuple(made)
    c.execs = tuple(0 if x is None else x.raw_cuda_graph_exec() for x in made)


def _replay(run: Any, g: _Group, c: _Call, j: int, slot: int, nbytes: int, b: dict[str, Any], x: Any, raw: int, main: Any,
            lane_raw: int, to: Any) -> None:
    """`c`'s graphs replayed by the driver with `_eager`'s events: the main lane waits for set j's last side steps and
    uploads the table, waits for the caller's staged x and replays its graph; side lane `to` replays the tree's after the
    main graph's event past the per-row keying, then the side graph after the main graph's end; the caller waits for the
    main lane."""
    D, done = g.driver, g.done[j]
    if done is not None:
        D.wait(lane_raw, done)
    g.upload(b, slot, nbytes, lane_raw)
    c.xf.copy_(x)
    if main is not None:
        D.record(main.event, raw)
        D.wait(lane_raw, main.event)
    main_exec, tree_exec, side_exec = c.execs
    D.launch(main_exec, lane_raw)
    if to is not None:
        D.record(to.event, lane_raw)
        if tree_exec:
            D.wait(to.raw, c.ev_tree)
            D.launch(tree_exec, to.raw)
        D.wait(to.raw, to.event)
        if side_exec:
            D.launch(side_exec, to.raw)
        if done is None:
            done = g.done[j] = _event()
        D.record(done, to.raw)
    if main is not None:
        D.record(main.event, lane_raw)
        D.wait(raw, main.event)
    g.replays += 1


def _gemm_grouped(run: Any, ws: tuple[int, ...], x: Any, counts: tuple[int, ...], bias: Any) -> Any:
    import torch
    from verity_vllm.protocols.pouw.pearl_c import CALL_CAPACITY
    if set(run.schedule.side) - set(SIDE_STEPS):
        raise ValueError(f"{run.entry}: a grouped call runs only {', '.join(sorted(SIDE_STEPS))} on a side lane, not "
                         f"{run.schedule.name}'s {', '.join(sorted(run.schedule.side))}")
    g, cur, R = group(run, ws), run.cur, run.kernel.run
    G, n, k = g.G, g.n, g.k
    if len(counts) != len(ws) or x.shape != (sum(counts), k) or min(counts, default=0) < 0:
        raise ValueError(f"{run.entry}: x {tuple(x.shape)} isn't Σ counts = {sum(counts)} rows of k = {k} over {len(ws)} experts")
    us = g.units(counts)
    if not us:
        return x.new_zeros((0, n)) if bias is None else x.new_zeros((0, n)) + bias
    rows, tiles, pos, real = sum(u.seg for u in us), sum(u.tiles for u in us), len(cur.matmuls), len(x)
    index = call_index(run, len(us))
    if run.hashing and (cur.tiles + tiles > run.capacity or pos + len(us) > CALL_CAPACITY
                        or cur.flag_rows + real > len(run.flag_buf)):
        raise RuntimeError(f"{run.entry}: the pass is over its capacity ({run.capacity} tiles, {CALL_CAPACITY} calls, "
                           f"{len(run.flag_buf)} screened rows); end it and begin another")
    cuda = run.device.type == "cuda"
    if cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{run.entry}: a grouped call's counts are the host's, so a CUDA graph capture can't take it")
    side = _side(run, G)
    split = "tree_seed_a" in side
    raw = run.stream()
    caller = run.torch_stream(raw) if cuda else None
    main = run.main if run.schedule.main_priority == "greatest" else None
    lane_raw, lane = (main.raw, main.stream) if main is not None else (raw, caller)
    g.fit(run, rows, run.schedule.ring if side else 1)
    g.poison(cur)
    j, b = g.take()
    c = g.call(run, counts, us, split, side, j, b)
    keys = b"".join(run.keys(run.roots[ws[u.pos]], index + i, u.m16) for i, u in enumerate(us)) if run.hashing else None
    slot, nbytes = g.table(run, c, keys, run.leaves.data_ptr() + run.nb * cur.tiles, pos, cur.flag_rows)
    c.at = (pos, cur.flag_rows)
    to = None
    if side:
        to = run.sides[run._side_turn % len(run.sides)]
        run._side_turn += 1
    graphed = cuda and run.graphs and run.fused_screen
    if graphed and c.execs:
        _replay(run, g, c, j, slot, nbytes, b, x, raw, main, lane_raw, to)
    else:
        if cuda and g.done[j] is not None:              # set j's last side steps still read its table, C~, U and tree
            caller.wait_event(g.done[j])
        g.upload(b, slot, nbytes, raw)
        c.xf.copy_(x)
        if cuda and main is not None:                   # the main lane starts after the caller's table and staged x
            main.event.record(caller)
            lane.wait_event(main.event)
        _eager(run, g, c, j, cuda, caller, lane, lane_raw, main, to)
        if cuda and main is not None:                   # the caller reads y and the pass's flags after it
            main.event.record(lane)
            caller.wait_event(main.event)
        c.seen += 1
        if graphed and c.seen >= 2:
            _capture(run, c)
    mms = [Matmul(index + i, shape, w) for i, (shape, w) in enumerate(c.shapes)]
    cur.matmuls.extend(mms)
    if run.hashing:
        flags = run.flag_buf[cur.flag_rows:cur.flag_rows + real]
        cur.flag_rows += real
        if cur.retain is not None:
            if split and cuda:                          # A's tree is the side lane's
                caller.wait_event(g.done[j])
            xa, slot_a = b["xa"][:4 * g.rows * k].view(torch.float32).view(g.rows, k), G.tree_bytes(R)
            for i, (u, mm) in enumerate(zip(us, mms, strict=True)):
                run._retained(cur, mm, u.m, flags[u.src:u.src + u.m], xa[u.row:u.row + u.m16],
                              b["tree_a"][slot_a * i:slot_a * i + run.nb * (2 * R.pow2(u.m16) - 1)])
        cur.tiles += tiles
    y = c.y.index_select(0, c.idx)
    return y if bias is None else y + bias
