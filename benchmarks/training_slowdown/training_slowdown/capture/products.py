"""A captured step's matrix products as te_core.io_bound reads them (P-width's dataflow side), for a program recorded
from its start (te_core.capture.script, `Trace.region`) as well as a step captured by capture_step or capture_fn.

Each GEMM of the measured step is a row (`rows`): its dims and token dims, whether each operand depends on θ′, the
width each operand is passed at (`io_bound._passed_width`), the bits a block pays an entry of it at policy k (the rates
`io_bound._Rates` states), the shortest chain that makes an operand entry whole by itself (`io_bound._chain_ks`'s
rule), whether both operands are one array, its partial sums' width (`io_bound.ACC_BITS`), its price, and what the
step's whole-making system (`io_bound.chain_classes`) reads of it: the classes whose outputs reach its operands (`fed`)
and their paid factor (`P`, `_reach`), and whether each operand reads a gather's rows (an embedding's, the same for
every token that shares an id) before any op makes them token-specific (`gathered`, `_gathered`), which the bound drops
(te_core.formalism). The block's `p_match` lists the ops that break P-match, and its `elementwise_products` the
matrix products written with elementwise ops (a broadcast multiplication and a sum, or a gather, a multiplication and
a sum by index), which no row is and whose classes the whole-making system would miss. Identical rows of the step (a layer's GEMMs
repeated over the layers) are one row with their count, so `gathered` puts layer 0's products of the embedding's rows in
rows of their own. A trace keeps the rows at each k of the console's dial too (`rows_on_dial`, `DIAL_K`), since a
row's rates, kind and feeds move with k.

A recorded program reads θ′ through storage versions: a weight's view in the measured step reads the version the
previous step's optimizer wrote, not the attribute's entry value. So the rates and chains here are keyed by the
capture's values (`te_core.capture.structure._Flow`), not by nodes as io_bound's are, and every value the measured step
reads from an earlier step that depends on θ′ (θ′ itself, the optimizer's state) is read at its width with no chain,
as io_bound takes an updated parameter or state; for a step captured by capture_step or capture_fn they are io_bound's.

The summation order of a GEMM's partial sums is its kernel's, which no ATen graph shows (a GEMM is one op), so `order`
is ``unknown`` here; the order belongs to the circuit model, not the trace, and te_core.formalism sums the rows in the
reference circuit's chains (`te_core.formalism.ORDER`), as te_core.gates prices a GEMM."""
from __future__ import annotations

import math
import operator

import torch

from te_core import io_bound
from te_core.capture.structure import _GATHERS, _WALK, _closure, _dtype_name, _Flow, _gemms, _macs, _phases, _pk, _shape

#: the capture's version of these rows (te_core's VERSION when they were last changed), so a trace that has them is
#: distinguishable from one traced before them. A recorded program's trace also carries the te_core VERSION whose code
#: wrote it (`capture_code`, te_core.capture.script's `record`, from 4.12.14 on; a trace without it was written by 4.12.9
#: to 4.12.13's code, which differ only in that 4.12.12's and 4.12.13's list fewer elementwise products)
CAPTURE = "4.12.9"
#: the torch the frozen capture traces under, the trace job's (``torch==2.15.0+cpu``): a step's graph, and so what the
#: capture counts on it, moves with torch's version, so te_core.capture.script refuses to trace under another
TORCH = "2.15.0"
ORDER = "unknown"
#: the console's dial's k levels in bits (te_core.panels' `K_LEVELS`, which a test holds this to): each of its block
#: sizes, the reference's, the served demo's block and the console's default
DIAL_K = (65_536, 80_000, 145_568, 800_000, 8_000_000, 80_000_000, 800_000_000)


class _Values:
    """The measured step's dataflow: the capture's values (`_Flow`), which depend on θ′ (`dep`), and the step whose
    GEMMs are rows (`region`; None: a step captured by capture_step or capture_fn, with io_bound's dependent slots)."""

    def __init__(self, graph) -> None:
        base, varied, _ = graph._inputs
        self.region = base.region
        self.token_dims = {(m.node, m.part): m.token_dims for m in graph.matmuls}
        if self.region is None:
            st = io_bound._Step(graph)
            self.gm, self.nodes, self.flow, self.dep = st.gm, st.nodes, st.flow, st.dep
            return
        gm = self.gm = base.gm
        self.nodes = list(gm.graph.nodes)
        pos = {n: i for i, n in enumerate(self.nodes)}
        vnodes = [list(t.gm.graph.nodes) for t in varied]

        def vshape(n, idx, vs):
            v = (vs[pos[n]] if vs is not None else n).meta.get("val")
            if idx is not None:
                v = v[idx] if isinstance(v, (list, tuple)) and idx < len(v) else None
            return _shape(v)

        def tok(n, idx) -> bool:
            s = vshape(n, idx, None)
            return s is not None and any(vshape(n, idx, vs) != s for vs in vnodes)

        flow = self.flow = _Flow(gm, _phases(gm), tok)
        past = {v for v, k in flow.source.items() if k[0] == "input" and flow.step[v] < self.region}
        self.dep = _closure(flow, past.__contains__)

    def step_of(self, n) -> int:
        return int(n.meta.get("te_step", 0))

    def measured(self, n) -> bool:
        return self.region is None or self.step_of(n) == self.region

    def values(self, n) -> list[int]:
        v = self.flow.val.get(n)
        return [v] if isinstance(v, int) else [x for x in v if isinstance(x, int)] if isinstance(v, list) else []

    def affected(self, n) -> bool:
        return any(self.dep[v] for v in self.values(n))

    def reads_affected(self, n, a) -> bool:
        return any(self.dep[x] for x in self.flow.reads.get((n, a), ()))


def _made(vs: _Values) -> dict:
    """Each node's values: its own and those it wrote in place."""
    out: dict = {}
    for n in vs.nodes:
        out[n] = list(vs.values(n))
    for _, v, n in vs.flow.writes:
        if n is not None and v not in out.setdefault(n, []):
            out[n].append(v)
    return out


def _carried(vs: _Values, v: int, made_by: dict) -> bool:
    """A value the measured step reads as given: a graph source, or one an earlier step made."""
    n = made_by.get(v)
    return n is None or n.op != "call_function" or (vs.region is not None and vs.step_of(n) < vs.region)


def _reads(vs: _Values, n, a) -> frozenset:
    """The values node `n` read through argument `a`: a view's own value, not also the version of its storage it was
    made from (`_Flow` records both), so a repeating view's spread holds; a later version written in place stays."""
    xs = vs.flow.reads.get((n, a), frozenset())
    own = vs.flow.val.get(a)
    if isinstance(own, int) and own in xs and len(xs) > 1:
        return frozenset(x for x in xs if x == own or x not in vs.flow.preds[own])
    return xs


def _rates_and_chains(vs: _Values, k: float) -> tuple[dict, dict, dict]:
    """io_bound's `_Rates` and `_chain_ks` keyed by value: (rate by value, chain by value, the node that made each)."""
    flow, dep = vs.flow, vs.dep
    made = _made(vs)
    made_by = {v: n for n, xs in made.items() for v in xs}
    rate: dict = {}
    chain: dict = {}

    def R(n, a) -> float:
        return max((rate.get(x, 0.0) for x in _reads(vs, n, a)), default=0.0)

    def C(n, a) -> float:
        return max((chain.get(x, math.inf) for x in _reads(vs, n, a)), default=0.0)

    for v in range(len(flow.preds)):
        if _carried(vs, v, made_by):
            rate[v] = float(flow.width[v]) if dep[v] else 0.0
            chain[v] = math.inf if dep[v] else 0.0
    for n in vs.nodes:
        if n.op != "call_function" or not vs.measured(n) or n.target is operator.getitem:
            continue
        outs = [v for v in made.get(n, []) if v not in rate]
        if not outs:
            continue
        if not vs.affected(n) and not any(dep[v] for v in outs):
            for v in outs:
                rate[v], chain[v] = 0.0, 0.0
            continue
        w = io_bound._width(n)
        name, gs = _pk(n.target), _gemms(n)
        if gs:
            r = min(float(w), max(_gemm_rate(vs, n, g, R, k) for g in gs))
            both = all(vs.reads_affected(n, g.operands[0]) and vs.reads_affected(n, g.operands[1]) for g in gs)
            c = float(max(g.dims["k"] for g in gs)) if both else math.inf
        else:
            ins = [a for a in io_bound._tensor_args(n) if vs.reads_affected(n, a)]
            out = max(io_bound._numel(n), 1)
            if name in _GATHERS and len(n.args) > 1 and hasattr(n.args[1], "op") and not vs.reads_affected(n, n.args[1]):
                r = 0.0
            elif not ins:
                r = float(w)
            elif name in _WALK and name not in io_bound._REPEATS and ins[0] is n.args[0]:
                r = R(n, ins[0])
            elif name in io_bound._CONCAT:
                r = sum(R(n, a) * io_bound._numel(a) for a in ins) / out
            else:
                r = max(R(n, a) * io_bound._numel(a) / out for a in ins)
            r = min(float(w), r)
            if name in _GATHERS and len(n.args) > 1 and hasattr(n.args[1], "op") and not vs.reads_affected(n, n.args[1]):
                c = 0.0
            else:
                big = [a for a in ins if name in io_bound._CONCAT or io_bound._numel(a) >= io_bound._numel(n)]
                c = max((C(n, a) for a in big), default=0.0)
        for v in outs:
            rate[v], chain[v] = (r, c) if dep[v] else (0.0, 0.0)
    return rate, chain, made_by


#: ops each of whose output entries is one entry of one argument, or a sum of distinct ones (concatenation, padding,
#: scatter, the embedding's backward): no argument fans out, whatever its size
_LAYOUT = io_bound._CONCAT | {"constant_pad_nd", "pad", "embedding_dense_backward", "embedding_backward", "index_add",
                              "index_add_", "scatter_add", "scatter_add_", "index_put", "index_put_", "_unsafe_index_put",
                              "scatter", "scatter_", "slice_backward", "select_backward", "slice_scatter",
                              "select_scatter"}


def _reach(vs: _Values, made: dict, made_by: dict, rate: dict) -> tuple[dict, dict, dict]:
    """io_bound's whole-making system (`io_bound._whole_making`) keyed by value, and P-match's check.

    A value's reach is {K: A}: the outputs of the class of chain length K (every product with both operands affected,
    `io_bound.chain_classes`) that its entries are made whole from, A entries of it for each. A product with both
    operands affected starts {K: 1}, K its longest part's; one with an operand outside the update passes its affected
    operand's times n/K (m/K for the B side); any other op passes its affected arguments' of at least its size (a
    concatenation's parts whatever their size), only the lightest where some are paid at their width (each such entry
    is needed). Its paid factor P is the largest factor a path into it has for paid entries (n/K through such a
    product). Carried and unaffected values reach nothing.

    P-match breaks at an op other than a layout op (`_LAYOUT`) with no affected argument of its size, whose output is
    made whole from a smaller argument that reaches a class (a repeating view, a broadcast), unless that argument is a
    statistic: a reduction's output (a norm's or softmax's row statistic), or one computed from statistics alone, which
    a block is charged nothing for and whose broadcast fans out no more than its reduction fanned in; and at a gather by
    token data of a value that reaches a class, whose rows repeat. Returns (reach, P, {op: (largest fan-out,
    count)})."""
    flow, dep = vs.flow, vs.dep
    reach: dict = {}
    paid: dict = {}
    stat: dict = {}
    broken: dict = {}

    def got(n, a) -> dict:
        out: dict = {}
        for x in _reads(vs, n, a):
            for K, w in reach.get(x, {}).items():
                out[K] = max(out.get(K, 0.0), w)
        return out

    def got_paid(n, a) -> float:
        return max((paid.get(x, 1.0) for x in _reads(vs, n, a)), default=1.0)

    def elems(a) -> int:
        b = flow.base.get(a)
        size = flow.size.get(b) if not isinstance(b, list) else None
        return min(io_bound._numel(a), size) if size else io_bound._numel(a)

    def mass(x: dict) -> float:
        return sum(w * w / (K * K) for K, w in x.items())

    def breaks(name, fan) -> None:
        f, c = broken.get(name, (0.0, 0))
        broken[name] = (max(f, fan), c + 1)

    for v in range(len(flow.preds)):
        if _carried(vs, v, made_by):
            reach[v], paid[v], stat[v] = {}, 1.0, False
    for n in vs.nodes:
        if n.op != "call_function" or not vs.measured(n) or n.target is operator.getitem:
            continue
        outs = [v for v in made.get(n, []) if v not in reach]
        if not outs:
            continue
        name, gs = _pk(n.target), _gemms(n)
        r: dict = {}
        best, s = 1.0, False
        if not vs.affected(n) and not any(dep[v] for v in outs):
            pass
        elif gs and all(vs.reads_affected(n, g.operands[0]) and vs.reads_affected(n, g.operands[1]) for g in gs):
            r = {float(max(g.dims["k"] for g in gs)): 1.0}
        elif gs:
            for g in gs:
                for a, f in ((g.operands[0], g.dims["n"] / g.dims["k"]), (g.operands[1], g.dims["m"] / g.dims["k"])):
                    if vs.reads_affected(n, a):
                        for K, w in got(n, a).items():
                            r[K] = max(r.get(K, 0.0), w * f)
                        best = max(best, f * got_paid(n, a), f)
        else:
            size = io_bound._numel(n)
            cat = name in io_bound._CONCAT
            aff = [a for a in io_bound._tensor_args(n) if vs.reads_affected(n, a)]
            ins = [a for a in aff if cat or io_bound._numel(a) >= size]
            for a in ins:
                best = max(best, got_paid(n, a))
            whole = [a for a in ins if io_bound.is_paid(max((rate.get(x, 0.0) for x in _reads(vs, n, a)),
                                                            default=0.0), io_bound._width(a))]
            if not cat and len(ins) > 1 and whole:
                ins = [min(whole, key=lambda a: mass(got(n, a)))]
            for a in ins:
                for K, w in got(n, a).items():
                    r[K] = max(r.get(K, 0.0), w)
            reached = {a: any(stat.get(x, False) for x in _reads(vs, n, a)) for a in aff if got(n, a)}
            s = bool(reached) and (all(reached.values()) or
                                   name not in _WALK and name not in _GATHERS and any(elems(a) > size for a in aff))
            if name in _GATHERS and len(n.args) > 1 and hasattr(n.args[1], "op") and \
                    not vs.reads_affected(n, n.args[1]) and n.args[0] in reached:
                breaks(name, size / max(elems(n.args[0]), 1))
            elif name not in _LAYOUT and not ins and any(not st for st in reached.values()):
                breaks(name, max(size / max(elems(a), 1) for a, st in reached.items() if not st))
        for v in outs:
            reach[v], paid[v], stat[v] = (r, best, s) if dep[v] else ({}, 1.0, False)
    return reach, paid, broken


#: ops that take a token-indexed value's entries from a table that isn't token-indexed (a positional table's rows, as
#: RoPE's cos and sin are sliced to the context), that number positions, or that mask by them
_TAKE = {"slice", "select", "narrow", "index", "index_select", "take", "gather", "embedding", "as_strided"}
_POSITIONAL = {"arange", "tril", "triu", "tril_", "triu_"}


def _gathered(vs: _Values, made: dict) -> set:
    """The token-indexed values that carry a gather's rows (an embedding's), the same for every token that shares an
    index, before any op makes them token-specific. A gather by token data (an index that doesn't depend on θ′) of a
    table that isn't token-indexed starts them, keyed by its index, and so does the index itself; an op carries its
    arguments' rows on where its token-indexed arguments are of one key, or the same for every token (a broadcast, a
    constant, a value that isn't token-indexed). Its output is token-specific where one is already, where two keys
    meet (a token's id and its position), where it takes rows of a table that isn't token-indexed by anything but a
    gathered index (RoPE's cos and sin sliced to the context), numbers or masks positions (`_POSITIONAL`, the causal
    mask's tril), or is random (a dropout mask). Any other op carries the rows on, so a product is never kept for want
    of a reason to drop it."""
    from te_core.capture.structure import _RANDOM

    flow = vs.flow
    tok = flow.tok
    key: dict = {}
    state: dict = {}
    for n in vs.nodes:
        if n.op == "call_function" and vs.measured(n) and _pk(n.target) in _GATHERS and len(n.args) > 1 and \
                hasattr(n.args[1], "op") and not vs.reads_affected(n, n.args[1]) and \
                not any(tok[x] for x in _reads(vs, n, n.args[0])) and any(tok[v] for v in made.get(n, [])):
            key[n] = ("g", frozenset(_reads(vs, n, n.args[1])))
            state.update(dict.fromkeys(key[n][1], key[n]))
    made_by = {v: n for n, xs in made.items() for v in xs}
    for v in range(len(flow.preds)):
        if v not in state and _carried(vs, v, made_by):
            state[v] = "x" if tok[v] else "u"
    for n in vs.nodes:
        if n.op != "call_function" or not vs.measured(n) or n.target is operator.getitem:
            continue
        outs = [v for v in made.get(n, []) if v not in state]
        if not outs:
            continue
        name = _pk(n.target)
        args = io_bound._tensor_args(n)
        table = name in _TAKE and bool(args) and args[0] is n.args[0] and \
            not any(tok[x] for x in _reads(vs, n, n.args[0]))
        got = {state.get(x, "u") for a in (args[1:] if table else args) for x in _reads(vs, n, a) if tok[x]} - {"u"}
        if n in key:
            out = key[n]
        elif name in _RANDOM or name in _POSITIONAL or "x" in got or len(got) > 1 or table and not got:
            out = "x"
        else:
            out = next(iter(got), "u")
        for v in outs:
            state[v] = out if tok[v] else "u"
    return {v for v, s in state.items() if isinstance(s, tuple) and tok[v]}


#: elementwise multiplications, by the positions of their two multiplicands
_MULTIPLY = {"mul": (0, 1), "mul_": (0, 1), "div": (0, 1), "div_": (0, 1), "addcmul": (1, 2), "addcmul_": (1, 2),
             "addcdiv": (1, 2), "addcdiv_": (1, 2)}

def _sums_by_index(n) -> bool:
    """Whether node `n` adds its source's entries into its output at indices (index_add, scatter_add, an accumulating
    index_put, a summing or averaging scatter_reduce or index_reduce)."""
    name = _pk(n.target)
    if name in ("index_add", "index_add_", "scatter_add", "scatter_add_"):
        return True
    if name in ("index_put", "index_put_", "_unsafe_index_put", "_index_put_impl_"):
        return bool(n.args[3] if len(n.args) > 3 else n.kwargs.get("accumulate", False))
    if name in ("scatter_reduce", "scatter_reduce_", "index_reduce", "index_reduce_"):
        return (n.args[4] if len(n.args) > 4 else n.kwargs.get("reduce")) in ("sum", "mean")
    return False


def _rows_by_data(n):
    """The table node `n` gathers rows of by the step's data (an embedding by its token ids): a gather of rows
    (embedding, table[ids], index_select along dim 0) whose index is a graph input through views and casts; else
    None."""
    name, a = _pk(n.target), n.args
    if name == "embedding" and len(a) > 1:
        index = a[1]
    elif name in ("index", "_unsafe_index") and len(a) > 1 and isinstance(a[1], (list, tuple)) and a[1] and \
            all(x is None for x in a[1][1:]):
        index = a[1][0]
    elif name == "index_select" and len(a) > 2 and a[1] == 0:
        index = a[2]
    else:
        return None
    if not (hasattr(a[0], "op") and hasattr(index, "op")):
        return None
    return a[0] if getattr(io_bound._root(index), "op", None) == "placeholder" else None


def _lookup(n) -> bool:
    """Whether node `n` looks a module tensor's rows up by the step's data (`_rows_by_data`, its table a module
    tensor through views and casts). Its rows repeat only where the data does, so its entries are its own; an index
    a program computes is no lookup, and a table it builds is one only where its entries are its own (`_sources`)."""
    table = _rows_by_data(n)
    return table is not None and getattr(io_bound._root(table), "op", None) == "get_attr"


def _sources(vs: _Values) -> list:
    """Each value's sources, the values that depend on θ′ whose entries it is computed from (None: it doesn't depend on
    θ′). A graph source, a GEMM's output (a product the rows count) and a lookup's (`_lookup`) are their own sources,
    and so is a gather of rows by the step's data (`_rows_by_data`) from a table the program builds, where each version
    of it the gather reads is its own source (a copy of θ's table, never a broadcast); and so is any value with at
    least as many entries as its sources have; any other value's are its affected predecessors', storage-versioned
    (`_Flow`), so a view, a cast, a map, a multiplication by a value outside the update or a copy in place keeps the
    entries of the value it broadcasts."""
    flow, dep = vs.flow, vs.dep
    own = {n for n in vs.nodes if n.op == "call_function" and (_gemms(n) or _lookup(n))}
    rows = {n: t for n in vs.nodes if n.op == "call_function" and (t := _rows_by_data(n)) is not None}
    size = _sizes(vs)
    src: list = [None] * len(flow.preds)
    for v, ps in enumerate(flow.preds):
        if not dep[v]:
            continue
        aff = [p for p in ps if dep[p]]
        n = flow.node[v]
        whole = n in rows and all(src[x] == frozenset((x,)) for x in _reads(vs, n, rows[n]) if dep[x])
        s = frozenset((v,)) if not aff or n in own or whole else frozenset().union(*(src[p] for p in aff))
        src[v] = s if sum(size[x] for x in s) < size[v] else frozenset((v,))
    return src


def _sizes(vs: _Values) -> list[int]:
    """Each value's entries (a Python scalar's 1)."""
    return [max(b // max(w, 1), 1) for b, w in zip(vs.flow.bits, vs.flow.width)]


#: ops whose output is their first argument's entries, its shape unchanged: casts and copies
_SAME = {"_to_copy", "to", "clone", "contiguous", "detach", "alias", "lift_fresh_copy", "lift_fresh", "type_as"}
#: views that only add or drop dims of size 1, or merge none
_RESHAPE = {"unsqueeze", "squeeze", "view", "_unsafe_view", "reshape", "_reshape_alias", "view_as", "flatten",
            "unflatten", "squeeze_copy", "unsqueeze_copy", "view_copy"}
#: ops that broadcast their first argument to their output's shape
_EXPAND = {"expand", "expand_as", "broadcast_to", "expand_copy"}


def _shape_of(x) -> tuple | None:
    v = x.meta.get("val") if hasattr(x, "meta") else None
    return tuple(int(s) for s in v.shape) if hasattr(v, "shape") else None


def _cuts(shape: tuple) -> list[int]:
    """The sizes of `shape`'s trailing blocks (each dim's entries times every later dim's), from 1 up: dim i is the run
    of an entry's index, in order, from prod(shape[i + 1:]) to prod(shape[i:])."""
    return sorted({math.prod(shape[i:]) for i in range(len(shape) + 1)})


def _cells(runs: list[frozenset], shape: tuple) -> tuple[list[int], list[frozenset]] | None:
    """Sets of runs (lo, hi) of the index of an array of `shape` on the finest grid through their ends and its dims'
    (`_cuts`): the grid and each set's cells (cell j from grid[j] to grid[j + 1]); None where the ends don't nest
    (each a divisor of the next), so no grid has every run as cells (a 2 x 3 and a 3 x 2 share none)."""
    grid = sorted({*_cuts(shape), *(e for s in runs for r in s for e in r)})
    if any(b % a for a, b in zip(grid, grid[1:])):
        return None
    return grid, [frozenset(j for j in range(len(grid) - 1) if any(lo <= grid[j] and grid[j + 1] <= hi for lo, hi in s))
                  for s in runs]


def _moved(runs: frozenset, shape: tuple, out: tuple, to: dict) -> frozenset | None:
    """Runs of an array of `shape` as runs of `out`'s index, each dim i of `shape` moved to dim to[i] of `out` (a
    broadcast's or a permutation's); None where they don't nest with `shape`'s dims."""
    got = _cells([runs], shape)
    if got is None:
        return None
    grid, (cells,) = got
    moved = set()
    for j in cells:
        lo, hi = grid[j], grid[j + 1]
        i = next(i for i in range(len(shape)) if math.prod(shape[i + 1:]) <= lo and hi <= math.prod(shape[i:]))
        below, at = math.prod(shape[i + 1:]), math.prod(out[to[i] + 1:])
        moved.add((lo // below * at, hi // below * at))
    return frozenset(moved)


def _broadcast(runs, shape: tuple, out: tuple) -> frozenset | None:
    """The runs of `out`'s index along which a value of `shape`, flat along `runs`, is, broadcast to it: its own, and
    every dim of `out` it lacks or has size 1 in."""
    if runs is None or shape is None or out is None or len(shape) > len(out):
        return None
    k = len(out) - len(shape)
    got = _moved(runs, shape, out, {i: i + k for i in range(len(shape))})
    return None if got is None else got | {(math.prod(out[j + 1:]), math.prod(out[j:])) for j in range(len(out))
                                           if out[j] > 1 and (j < k or shape[j - k] == 1)}


def _meet(runs: list[frozenset], shape: tuple) -> frozenset | None:
    """The runs of an array of `shape`'s index in every set of `runs`; None where they don't nest."""
    got = _cells(runs, shape)
    return None if got is None else frozenset((got[0][j], got[0][j + 1]) for j in frozenset.intersection(*got[1]))


def _whole(runs: frozenset, shape: tuple) -> bool:
    """Whether `runs` cover an array of `shape`'s whole index, so a value flat along them reads the same entries at
    every position (a scalar's broadcast)."""
    got = _cells([runs], shape)
    return got is not None and len(got[1][0]) == len(got[0]) - 1


def _flats(vs: _Values, src: list) -> list:
    """Each value's flat runs: the runs of its entries' index (`_cuts`'s, a dim or part of one) along which what it
    reads of θ′ doesn't vary, so it is a broadcast along them (None: unknown, or it doesn't depend on θ′). A value
    that is its own source (`_sources`) has none; a map, cast, copy or broadcast is flat where every affected argument
    is, broadcast to its shape; a view keeps its argument's, since it keeps the entries' order, however it regroups
    their dims (a code's levels, flat along the lanes of their byte, merged with the bytes into the group); a
    permutation moves them with their dims; every other op's are unknown."""
    flow, dep = vs.flow, vs.dep
    written = {v for _, v, _ in flow.writes}
    flat: list = [None] * len(flow.preds)
    for v, s in enumerate(src):
        if s is not None and s == frozenset((v,)):
            flat[v] = frozenset()
    for n in vs.nodes:
        v = flow.val.get(n)
        if n.op != "call_function" or not isinstance(v, int) or not dep[v] or flat[v] is not None or v in written:
            continue
        out, name = _shape_of(n), _pk(n.target)
        if out is None:
            continue
        first = n.args[0] if n.args and hasattr(n.args[0], "op") else None
        own = flow.val.get(first) if first is not None else None
        shape = _shape_of(first) if first is not None else None
        f = flat[own] if isinstance(own, int) else None
        if name in _EXPAND or name in _SAME or name in _MULTIPLY or torch.Tag.pointwise in getattr(n.target, "tags", ()):
            got = [_broadcast(flat[x] if isinstance(x := flow.val.get(a), int) else None, _shape_of(a), out)
                   for a in io_bound._tensor_args(n) if vs.reads_affected(n, a)]
            flat[v] = None if not got or None in got else _meet(got, out)
        elif name in _RESHAPE and f is not None and shape is not None and math.prod(shape) == math.prod(out):
            flat[v] = f
        elif name in ("permute", "transpose", "t", "numpy_T", "mT") and f is not None and shape is not None:
            order = list(range(len(shape)))
            if name == "permute":
                order = [int(d) % len(shape) for d in n.args[1]]
            elif len(shape) >= 2:
                d0, d1 = ((int(n.args[1]), int(n.args[2])) if name == "transpose" else (-2, -1))
                d0, d1 = d0 % len(shape), d1 % len(shape)
                order[d0], order[d1] = order[d1], order[d0]
            flat[v] = _moved(f, shape, out, {d: i for i, d in enumerate(order)})
    return flat


def _elementwise_products(vs: _Values) -> list[dict]:
    """The measured step's matrix products written with elementwise ops, which no row is and whose outputs the
    whole-making system would miss: a multiplication of two values that depend on θ′ whose result has more entries
    than the sources (`_sources`) of either have, each broadcast or repeated along a run of the index the other isn't
    (`_flats`; runs it can't tell never exempt, unless the other is flat along the whole index, a scalar's broadcast,
    which mixes with nothing): an outer product, a GEMM with K = 1, or a matrix product once summed, as
    (x.unsqueeze(-1) * W).sum(-2) writes x @ W, whatever maps or copies the broadcast passes through first. Two
    broadcasts along the same dims (a token's code over lanes times the token's scale) are elementwise. The runs are
    compared on the grid both share (`_cells`), so a code's levels whose lanes a view merges into their group, times
    the group's scale, are elementwise, whatever maps and casts follow the view, and two broadcasts each flattened by
    a view are still a product. And a sum by
    index (`_sums_by_index`) of a multiplication of two such values one of which carries a gather's rows (a sparse
    product: gather, mul and index_add), in the forward or the backward. A gather's rows and such a multiplication are
    carried by every view, and by every other op with one affected argument and no fewer entries (a cast, a map, a
    broadcast). A multiplication of two values of its own size (a norm's, a gate's) is elementwise, and so is one with
    an operand that doesn't depend on θ′, whose product is θ's and never counted (P-match checks its broadcast).
    [{"op", "count", "macs"}] by op, the multiplications' MACs."""
    phases = _phases(vs.gm)
    dep = vs.dep
    made = _made(vs)
    src, size = _sources(vs), _sizes(vs)
    flat = _flats(vs, src)
    gathered: set = set()
    product: dict = {}
    found: dict = {}

    def ent(n, a) -> int:
        s = frozenset().union(*(src[x] for x in _reads(vs, n, a) if dep[x]))
        return min(io_bound._numel(a), sum(size[x] for x in s))

    def runs(n, a) -> frozenset | None:
        """Operand `a` of `n` broadcast to `n`'s entries: the runs of their index along which it is flat."""
        own = vs.flow.val.get(a)
        if not isinstance(own, int) or _reads(vs, n, a) - {own}:
            return None
        return _broadcast(flat[own], _shape_of(a), _shape_of(n))

    def mixes(n, a, b) -> bool:
        """Whether each operand is a broadcast along a run the other isn't (unknown: it may be, unless the other is
        flat along the whole index), compared on the grid both share (`_cells`)."""
        out = _shape_of(n)
        fa, fb = runs(n, a), runs(n, b)
        if any(f is not None and _whole(f, out) for f in (fa, fb)):
            return False
        got = None if fa is None or fb is None else _cells([fa, fb], out)
        if got is None:
            return True
        ga, gb = got[1]
        return not (ga <= gb or gb <= ga)

    def add(name: str, macs: int) -> None:
        c, m = found.get(name, (0, 0))
        found[name] = (c + 1, m + macs)

    for n in vs.nodes:
        if n.op != "call_function":
            continue
        name = "getitem" if n.target is operator.getitem else _pk(n.target)
        outs = made.get(n, [])
        aff = [a for a in io_bound._tensor_args(n) if vs.reads_affected(n, a)]
        a, b = (n.args[i] if len(n.args) > i else None for i in _MULTIPLY[name]) if name in _MULTIPLY else (None, None)
        both = a in aff and b in aff
        if name in _GATHERS and n.args and n.args[0] in aff:
            gathered.update(outs)
        elif len(aff) == 1 and not both and not _gemms(n) and (name in _WALK or
                                                                io_bound._numel(n) >= io_bound._numel(aff[0])):
            got = _reads(vs, n, aff[0])
            if gathered & got:
                gathered.update(outs)
            if carried := [product[x] for x in got if x in product]:
                product.update(dict.fromkeys(outs, max(carried)))
        if not vs.measured(n) or phases.get(n, "forward") not in ("forward", "backward"):
            continue
        if both:
            out = io_bound._numel(n)
            if out > ent(n, a) and out > ent(n, b) and mixes(n, a, b):
                add(name, out)
            elif gathered & (_reads(vs, n, a) | _reads(vs, n, b)):
                product.update(dict.fromkeys(outs, out))
        elif _sums_by_index(n):
            for x in io_bound._tensor_args(n)[1:]:
                if carried := [product[v] for v in _reads(vs, n, x) if v in product]:
                    add(name, max(carried))
                    break
    return [{"op": op, "count": c, "macs": m} for op, (c, m) in sorted(found.items())]


def gathered_gemms(graph) -> set[tuple[str, str]]:
    """The (node, part) of each GEMM of the measured step with an operand that reads a gather's rows (`_gathered`)."""
    vs = _Values(graph)
    gathered = _gathered(vs, _made(vs))
    by_name = {n.name: n for n in vs.nodes}
    out = set()
    for m in graph.matmuls:
        n = by_name[m.node]
        g = next(x for x in _gemms(n) if x.part == m.part)
        if any(x in gathered for a in (g.operands[0], g.operands[1]) for x in _reads(vs, n, a)):
            out.add((m.node, m.part))
    return out


def _gemm_rate(vs: _Values, n, g, R, k: float) -> float:
    """io_bound's `_Rates._gemm` with each operand's rate as the node read it."""
    A, B = g.operands[0], g.operands[1]
    d = g.dims
    td = vs.token_dims.get((n.name, g.part), {})
    if not any(x in td for x in ("batch", "m", "n")):
        return math.inf
    ra = R(n, A) if vs.reads_affected(n, A) else 0.0
    rb = R(n, B) if vs.reads_affected(n, B) else 0.0
    if sum(x in td for x in ("m", "k", "n")) >= 2:
        return ra * d["k"] / max(d["n"], 1) + rb * d["k"] / max(d["m"], 1)
    token_a = "m" in td or "n" not in td
    param, r_tok, spread = (B, ra, d["n"]) if token_a else (A, rb, d["m"])
    tok = A if token_a else B
    b_tok, b_par = io_bound._passed_width(vs, n, tok), io_bound._passed_width(vs, n, param)
    if vs.reads_affected(n, param) and (io_bound.is_paid(r_tok, b_tok)
                                        or d["k"] >= 3 ** 1.5 * math.sqrt(k / io_bound.ACC_BITS)):
        return 4 * d["k"] ** 2 * b_tok * b_par / k
    return r_tok * d["k"] / max(spread, 1)


def rows(graph, k: float = io_bound.CONSOLE[1], gemm_dtype: str | None = "bfloat16") -> dict:
    """The measured step's products (module docstring) at policy k: {"capture", "k", "order", "tokens", "gates",
    "rows", "p_match"}, each row
    {"count", "phase", "op", "dims", "token_dims", "macs", "outputs", "affected", "widths", "rates", "paid", "self",
    "b_c", "k_whole", "gates_per_mac", "cell_gates", "fed", "P", "gathered"} (MACs and outputs over its count;
    `k_whole` None: no chain makes an operand entry whole by itself; `fed` the [K, A] pairs of the classes reaching its
    operands; `gathered` whether each operand reads a gather's rows before any op makes them token-specific), `gates`
    the step's |C| at io_bound's prices (`io_bound.step_gates`) for a step captured by capture_step or capture_fn, else
    None (the circuit's count is the trace's), `p_match` [{"op", "fan_out", "count"}] the ops that break P-match
    (`_reach`), and `elementwise_products` the matrix products the step writes with elementwise ops
    (`_elementwise_products`), which te_core.formalism refuses to price."""
    vs = _Values(graph)
    made = _made(vs)
    return _rows(vs, graph, k, gemm_dtype, made, _gathered(vs, made)) | {"elementwise_products": _elementwise_products(vs)}


def rows_on_dial(graph, gemm_dtype: str | None = "bfloat16") -> dict:
    """`rows` at the console's k, and under "at" each other k of the console's dial (`DIAL_K`, by its integer as a
    string) the rows and P-match check there ({"k", "rows", "p_match"}): a product's rates, kind and feeds move with k
    (a block of k bits pays an output made whole at 4K²·b²/k, and makes one whole by itself from K ≥ 3^1.5·√(k/b_c)),
    so the dial's term at a k reads that k's rows, which only the captured step gives. A trace keeps these (te_core.
    capture.script) so the dial needs no retrace."""
    vs = _Values(graph)
    made = _made(vs)
    gathered = _gathered(vs, made)
    out = _rows(vs, graph, io_bound.CONSOLE[1], gemm_dtype, made, gathered)
    out["elementwise_products"] = _elementwise_products(vs)
    out["at"] = {}
    for k in DIAL_K:
        if k != io_bound.CONSOLE[1]:
            b = _rows(vs, graph, float(k), gemm_dtype, made, gathered, gates=False)
            out["at"][str(k)] = {x: b[x] for x in ("k", "rows", "p_match")}
    return out


def _rows(vs: _Values, graph, k: float, gemm_dtype, made: dict, gathered: set, gates: bool = True) -> dict:
    rate, chain, made_by = _rates_and_chains(vs, k)
    reach, paid, broken = _reach(vs, made, made_by, rate)
    by_name = {n.name: n for n in vs.nodes}
    merged: dict = {}
    for m in graph.matmuls:
        n = by_name[m.node]
        g = next(x for x in _gemms(n) if x.part == m.part)
        A, B = g.operands[0], g.operands[1]
        aff = (vs.reads_affected(n, A), vs.reads_affected(n, B))

        def got(a, table):
            return max((table.get(x, 0.0) for x in _reads(vs, n, a)), default=0.0)

        rates = (got(A, rate) if aff[0] else 0.0, got(B, rate) if aff[1] else 0.0)
        widths = (io_bound._passed_width(vs, n, A), io_bound._passed_width(vs, n, B))
        whole = min((v for v in (got(A, chain), got(B, chain)) if v > 0), default=math.inf)
        fed: dict = {}
        P = 1.0
        for a, on in ((A, aff[0]), (B, aff[1])):
            for x in _reads(vs, n, a) if on else ():
                P = max(P, paid.get(x, 1.0))
                for K, w in reach.get(x, {}).items():
                    fed[K] = max(fed.get(K, 0.0), w)
        gpm, cell = io_bound._gemm_price(m.phase, _pk(n.target), _dtype_name(A), gemm_dtype)
        d = g.dims
        row = {"phase": m.phase, "op": _pk(n.target), "dims": {x: d[x] for x in ("batch", "m", "k", "n")},
               "token_dims": sorted(m.token_dims), "macs": _macs(d), "outputs": d["batch"] * d["m"] * d["n"],
               "affected": list(aff), "widths": list(widths), "rates": [round(r, 6) for r in rates],
               "paid": [io_bound.is_paid(r, w) for r, w in zip(rates, widths)],
               "self": io_bound._root(A, scale=True) is io_bound._root(B, scale=True), "b_c": io_bound.ACC_BITS,
               "k_whole": whole if math.isfinite(whole) else None, "gates_per_mac": gpm, "cell_gates": cell,
               "fed": [[K, round(w, 9)] for K, w in sorted(fed.items())], "P": round(P, 9),
               "gathered": [any(x in gathered for x in _reads(vs, n, a)) for a in (A, B)]}
        key = repr(sorted(row.items()))
        if key in merged:
            merged[key]["count"] += 1
        else:
            merged[key] = {"count": 1} | row
    gates = None if not gates else io_bound.step_gates(graph, gemm_dtype) if vs.region is None else \
        (graph.formalism or {}).get("gates")
    return {"capture": CAPTURE, "k": k, "order": ORDER, "tokens": graph.tokens, "gates": gates,
            "rows": list(merged.values()),
            "p_match": [{"op": op, "fan_out": round(f, 6), "count": c} for op, (f, c) in sorted(broken.items())]}
