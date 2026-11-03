"""A captured step's matrix products as te_core.io_bound reads them (P-width's dataflow side), for a program recorded
from its start (te_core.capture.script, `Trace.region`) as well as a step captured by capture_step or capture_fn.

Each GEMM of the measured step is a row (`rows`): its dims and token dims, whether each operand depends on θ′, the
width each operand is passed at (`io_bound._passed_width`), the bits a block pays an entry of it at policy k (the rates
`io_bound._Rates` states), the shortest chain that makes an operand entry whole by itself (`io_bound._chain_ks`'s
rule), whether both operands are one array, its partial sums' width (`io_bound.ACC_BITS`), its price, and what the
step's whole-making system (`io_bound.chain_classes`) reads of it: the classes whose outputs reach its operands (`fed`)
and their paid factor (`P`, `_reach`). The block's `p_match` lists the ops that break P-match. Identical rows of the
step (a layer's GEMMs repeated over the layers) are one row with their count.

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

from te_core import io_bound
from te_core.capture.structure import _GATHERS, _WALK, _closure, _dtype_name, _Flow, _gemms, _macs, _phases, _pk, _shape

#: the capture's version of these rows (te_core's VERSION when they were last changed), so a trace that has them is
#: distinguishable from one traced before them
CAPTURE = "4.12"
ORDER = "unknown"


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
    "b_c", "k_whole", "gates_per_mac", "cell_gates", "fed", "P"} (MACs and outputs over its count; `k_whole` None: no
    chain makes an operand entry whole by itself; `fed` the [K, A] pairs of the classes reaching its operands), `gates`
    the step's |C| at io_bound's prices (`io_bound.step_gates`) for a step captured by capture_step or capture_fn, else
    None (the circuit's count is the trace's), and `p_match` [{"op", "fan_out", "count"}] the ops that break P-match
    (`_reach`)."""
    vs = _Values(graph)
    rate, chain, made_by = _rates_and_chains(vs, k)
    reach, paid, broken = _reach(vs, _made(vs), made_by, rate)
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
               "fed": [[K, round(w, 9)] for K, w in sorted(fed.items())], "P": round(P, 9)}
        key = repr(sorted(row.items()))
        if key in merged:
            merged[key]["count"] += 1
        else:
            merged[key] = {"count": 1} | row
    gates = io_bound.step_gates(graph, gemm_dtype) if vs.region is None else (graph.formalism or {}).get("gates")
    return {"capture": CAPTURE, "k": k, "order": ORDER, "tokens": graph.tokens, "gates": gates,
            "rows": list(merged.values()),
            "p_match": [{"op": op, "fan_out": round(f, 6), "count": c} for op, (f, c) in sorted(broken.items())]}
