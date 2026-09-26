"""`Q_word_v1{X,W}`: a partition of every Call's gates into the largest units S with |out(S)| <= X bits, or one boundary gate of at
most W bits, whose inputs are committed values.

The rule:

  * **Partition.** Every gate belongs to exactly one unit.  A value more than one unit needs is committed as its own unit -- unless
    it is a trivial recompute (at most `TRIVIAL_GATES` gates) of committed values, which is then its own unit but not a commitment.
  * **Width** is the IR's own: `out(S)` is the gates of S read by a gate outside S or returned (`verity.ir.parts`, `layout.py`), and
    |out(S)| is the sum of their declared widths in bits.  A unit passes at |out(S)| <= X (16: where the word allows it), or when
    out(S) is ONE gate of at most W bits (32: float interiors such as a norm scale, attention's max, sum and inverse sum).  No dtype
    is read: when gates become single bits the invariant is unchanged.
  * It refines `Q_module_body_v1` without moving its boundary: a Call's inputs and outputs are the bridge's required values.

The partition of one Call activation (`partition`), on its explicit gates:

  1. The Call's returned gates, in return order, packed into targets of at most X declared bits (a wider gate is a target alone).
  2. Backward over the gates (the canonical layout is topological): a gate belongs to the unit of its consumers when they all
     belong to one unit.  A gate whose consumers belong to different units is needed by more than one unit; those of them read
     by a unit directly are promoted to targets (committed interior values), and the pass repeats until every gate has one owner.
  3. Gates no returned value depends on (dead gates) join the one unit whose gates they read, else form a unit of their own.

The committed interior values it derives are what a tap must acquire: attention's S, P, row max / sum and inverse sum are exactly
such values, and so is a norm's scale.  `unit_rule` summarises one specialization; `check_query` checks a Program and fails by name.
"""
from __future__ import annotations

import json
from array import array
from collections import Counter
from typing import Any, Callable, Iterable

import numpy as np

from verity.ir.defs import PrimitiveDefinition
from verity.ir.layout import WIDE_FAN_IN, resolve
from verity.ir.refs import Affine

QUERY_ID = "Q_word_v1"

#: a shared value computed from committed values by at most this many gates is recomputed by its readers' verifier, not committed
TRIVIAL_GATES = 2

_KINDS = ("output", "committed", "recompute", "literal", "dead", "interior")

#: Calls with more gates than this are not partitioned explicitly (`check_calls` names them)
MAX_GATES = 24_000_000


def query_id(X: int = 16, W: int = 32, extras: int = 0) -> str:
    return f"{QUERY_ID}{{X={int(X)},W={int(W)}" + (f",EXTRAS={int(extras)}" if extras else "") + "}"


def parse(spec: str) -> tuple[int, int, int]:
    """`"16"` / `"16/32"` / `"16/32+2"` -> (X, W, extras); W defaults to max(X, 32)."""
    body, _, e = str(spec).partition("+")
    x, _, w = body.partition("/")
    return int(x), int(w) if w else max(int(x), 32), int(e or 0)


class QueryRuleViolation(ValueError):
    """A Program whose units cannot satisfy `Q_word_v1{X,W}`: named per Definition and module, never widened."""


def specialization(fam: str, statics: dict[str, Any]):
    """The registered Definition `fam` bound to `statics` (a primitive as it is); function-valued statics (`DOT`, rendered
    `{"fn": id}` or as the bare id) resolve to the registered Definition they name."""
    from verity.ir.codec import lookup_primitive
    from verity_vllm.query.program_view import _registry
    reg, _errors = _registry()
    d = reg.defs.get(fam)
    if d is None:
        return lookup_primitive(reg, fam)

    def fn_value(v: Any) -> Any:
        fid = v.get("fn") if isinstance(v, dict) else (json.loads(v).get("fn") if isinstance(v, str) and v[:5] == '{"fn"' else v)
        if not isinstance(fid, str):
            return v
        try:
            return lookup_primitive(reg, fid)
        except KeyError:
            return reg.defs[fid] if fid in reg.defs else v

    return d.bind(**{k: fn_value(v) for k, v in statics.items()}) if hasattr(d, "bind") else d


def _call_scope(fn: Any) -> tuple[Any, Any]:
    """(circuit, activation of `fn`) in a one-Call Program: the Call's parameters are Input gates, its returned leaves designated
    outputs.  A returned leaf a real Program does not read only shrinks out(S), so widths here bound those in any Program."""
    from verity.ir.defs import CompositeDefinition
    from verity.ir.program import Program
    params, ret = fn.signature()
    shell = CompositeDefinition(f"Word[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a), register=False)
    c = Program(shell).circuit
    rs = c.root_scope
    k = next(k for k, n in enumerate(rs.spec.body.nodes) if n.fn is fn)
    return c, rs.child(k)


def _member_refs(a: Any, ax: int | None, m: int) -> Any:
    if ax is None:
        return a.refs
    if ax == 0:
        L = a.type.elem.leaves
        return a.refs.slice(m * L, (m + 1) * L)
    return a.take(ax, m).refs


def _wide_operands(gate: Any, call: Any) -> Iterable[int]:
    """A wide fan-in gate's operands, skipping whole arguments that are the Call's own parameters (Input gates, outside the Call)."""
    s = gate.scope
    body = s.spec.body
    k = body.node_at(gate.index - s.offset)
    node = body.nodes[k]
    m = (gate.index - s.offset - body.offsets[k]) // node.fn.gates if node.form == "batch" else 0
    at_call = s.spec is call.spec and s.offset == call.offset
    for a, ax in zip(node.args, node.axes if node.form == "batch" else (None,) * len(node.args)):
        refs = _member_refs(a, ax, m)
        if at_call and isinstance(refs, Affine) and refs.space == "p":
            continue
        for j in range(len(refs)):
            yield resolve(s, refs[j])


class Graph:
    """The explicit gates of one Call activation (local index = root index - the Call's offset): declared width, primitive and
    activation per gate, whether it reads a Call parameter, and the wires between its gates (`src[e]` -> `dst[e]`, grouped by
    consumer in gate order).  `outputs`: the returned gates in return order."""

    def __init__(self, fn: Any) -> None:
        c, call = _call_scope(fn)
        lo, n = call.offset, fn.gates
        self.n = n
        self.width = np.empty(n, np.int64)
        self.prim = np.empty(n, np.int64)
        self.scope = np.empty(n, np.int64)
        self.external = np.zeros(n, bool)
        self.prims: dict[str, int] = {}
        self.scopes: dict[str, int] = {}
        src, dst = array("q"), array("q")
        for j in range(n):
            g = c.gate(lo + j)
            self.width[j] = g.prim.ret.w
            self.prim[j] = self.prims.setdefault(g.prim.id, len(self.prims))
            self.scope[j] = self.scopes.setdefault(g.scope.spec.id, len(self.scopes))
            ops = g.operands
            ext = False
            if len(ops) > WIDE_FAN_IN:
                ops = _wide_operands(g, call)
                ext = True
            for o in ops:
                o -= lo
                if 0 <= o < n:
                    src.append(o)
                    dst.append(j)
                else:
                    ext = True
            self.external[j] = ext
        self.src = np.frombuffer(src, np.int64) if len(src) else np.zeros(0, np.int64)
        self.dst = np.frombuffer(dst, np.int64) if len(dst) else np.zeros(0, np.int64)
        self.outputs = [g - lo for g in (resolve(call, r) for r in fn.body.ret) if lo <= g < lo + n]
        self.prim_names = {v: k for k, v in self.prims.items()}
        self.scope_names = {v: k for k, v in self.scopes.items()}


def _targets(G: Graph, X: int) -> list[list[int]]:
    """The returned gates (first occurrence, return order) packed into runs of at most X declared bits."""
    seen, out, cur, bits = set(), [], [], 0
    for g in G.outputs:
        if g in seen:
            continue
        seen.add(g)
        w = int(G.width[g])
        if cur and bits + w > X:
            out.append(cur)
            cur, bits = [], 0
        cur.append(g)
        bits += w
    if cur:
        out.append(cur)
    return out


def partition(G: Graph, X: int) -> tuple[np.ndarray, list[str]]:
    """(owner per gate, kind per unit): kinds "output" (returned values), "interior" (a value more than one unit reads), "dead"."""
    n = G.n
    head = np.full(n, -1, np.int64)
    kinds: list[str] = []
    for t in _targets(G, X):
        head[t] = len(kinds)
        kinds.append("output")
    order = np.argsort(G.src, kind="stable")
    cons = array("q", G.dst[order].tobytes())                 # consumers of gate g: cons[start[g]:start[g + 1]]
    start = array("q", np.searchsorted(G.src[order], np.arange(n + 1)).tobytes())
    while True:
        owner = array("q", head.tobytes())                     # -1: no returned value depends on it; -2: several units
        for g in range(n - 1, -1, -1):
            if owner[g] >= 0:
                continue
            o = -1
            for e in range(start[g], start[g + 1]):
                oc = owner[cons[e]]
                if oc == -1 or oc == o:
                    continue
                if o == -1 and oc >= 0:
                    o = oc
                else:
                    o = -2
                    break
            owner[g] = o
        own = np.frombuffer(owner, np.int64)
        s_, d_ = G.src[order], G.dst[order]
        promote = np.unique(s_[(own[s_] == -2) & (own[d_] >= 0)])
        if not len(promote):
            break
        head[promote] = np.arange(len(kinds), len(kinds) + len(promote))
        kinds += ["interior"] * len(promote)
    own = np.frombuffer(owner, np.int64).copy()
    dead = np.flatnonzero(own == -1)
    if len(dead):
        # a dead gate joins the one live unit it reads, taking the dead-only units it reads along; reading several live units, or
        # none, it heads a dead unit
        by_consumer = np.argsort(G.dst, kind="stable")
        dstart = np.searchsorted(G.dst[by_consumer], np.arange(n + 1))
        live = len(kinds)
        parent: dict[int, int] = {}

        def find(u: int) -> int:
            while u in parent:
                u = parent[u]
            return u

        for g in dead:
            us = {find(int(own[G.src[by_consumer[e]]])) for e in range(dstart[g], dstart[g + 1])}
            alive = {u for u in us if u < live}
            if len(alive) == 1:
                (a,) = alive
                for u in us - alive:
                    parent[u] = a
                own[g] = a
            else:
                own[g] = len(kinds)
                kinds.append("dead")
        if parent:
            for g in dead:
                own[g] = find(int(own[g]))
            keep = sorted(set(range(len(kinds))) - set(parent))
            remap = np.full(len(kinds), -1, np.int64)
            remap[keep] = np.arange(len(keep))
            own = remap[own]
            kinds = [kinds[u] for u in keep]
    return own, kinds


def units(G: Graph, X: int, W: int, extras: int = 0) -> dict[str, Any]:
    """The partition of `G` with, per unit: gates, |out(S)| in bits over its boundary gates (read by another unit, or returned),
    the units it reads, and its kind refined -- an interior unit is "literal" (reads nothing), "recompute" (at most
    `TRIVIAL_GATES` gates reading only committed values, and at most one other such value depends on it), else "committed"."""
    owner, kinds = partition(G, X)
    U = len(kinds)
    gates = np.bincount(owner, minlength=U)
    cross = owner[G.src] != owner[G.dst]
    boundary = np.zeros(G.n, bool)
    boundary[G.src[cross]] = True
    boundary[G.outputs] = True
    bo = owner[boundary]
    out_bits = np.bincount(bo, weights=G.width[boundary], minlength=U).astype(np.int64)
    out_gates = np.bincount(bo, minlength=U)
    first = np.full(U, -1, np.int64)
    bidx = np.flatnonzero(boundary)
    uu, pos = np.unique(bo, return_index=True)
    first[uu] = bidx[pos]
    last = np.zeros(U, np.int64)
    np.maximum.at(last, owner, np.arange(G.n))
    external = np.bincount(owner[G.external], minlength=U) > 0
    pairs = np.unique(np.stack([owner[G.dst[cross]], owner[G.src[cross]]], 1), axis=0) if cross.any() else np.zeros((0, 2), np.int64)
    reads: dict[int, list[int]] = {}
    readers: dict[int, list[int]] = {}
    for r, w in pairs.tolist():
        reads.setdefault(r, []).append(w)
        readers.setdefault(w, []).append(r)
    committed = [k == "output" for k in kinds]
    small = gates <= TRIVIAL_GATES
    for u in sorted((u for u in range(U) if kinds[u] == "interior"), key=lambda u: last[u]):
        ins = reads.get(u, [])
        if not ins and not external[u]:
            kinds[u] = "literal"
            committed[u] = True
        elif small[u] and all(committed[v] for v in ins) and \
                sum(1 for r in readers.get(u, ()) if kinds[r] == "interior" and small[r]) <= 1:
            kinds[u] = "recompute"
        else:
            kinds[u] = "committed"
            committed[u] = True
    ok = (out_bits <= X + extras) | ((out_gates == 1) & (out_bits <= W + extras))
    return {"owner": owner, "kinds": kinds, "gates": gates, "out_bits": out_bits, "out_gates": out_gates, "ok": ok,
            "head_prim": np.where(first >= 0, G.prim[np.maximum(first, 0)], -1),
            "head_scope": np.where(first >= 0, G.scope[np.maximum(first, 0)], -1)}


def _separable(fn: Any) -> list[tuple[Any, int]] | None:
    """[(callee, members)] when `fn`'s body nodes read only its parameters and it returns every leaf they produce: no value crosses
    between nodes, so its partition is each node member's own (a Gemm's N coordinates, attention's heads).  None otherwise."""
    from verity.ir.intervals import norm, ref_intervals, total
    from verity.ir.parts import n_members
    body = fn.body
    for n in body.nodes:
        if n.form == "scan":
            return None
        reads: dict = {}
        for a in n.args:
            ref_intervals(a.refs, reads)
        if any(space != "p" for space, _i in reads):
            return None
    ret: dict = {}
    ref_intervals(body.ret, ret)
    for k, n in enumerate(body.nodes):
        if total(norm(ret.get(("n", k), ()))) != n.out.leaves:
            return None
    return [(n.fn, n_members(n)) for n in body.nodes]


def _combine(fid: str, gates: int, parts: list[tuple[dict[str, Any], int]], X: int, W: int, extras: int) -> dict[str, Any]:
    agg: dict[tuple, dict[str, Any]] = {}
    interior = Counter()
    viol: dict[str, dict[str, Any]] = {}
    for r, m in parts:
        for c in r["classes"]:
            k = (c["kind"], c["head"], c["scope"], c["gates"], c["out_bits"], c.get("out_gates", 1))
            a = agg.get(k)
            if a is None:
                a = agg[k] = dict(c, count=0)
            a["count"] += c["count"] * m
        for i in r["committed_interior"]:
            interior[(i["scope"], i["head"])] += i["words"] * m
        for v in r["violations"]:
            e = viol.setdefault(v["class"], dict(v, units=0))
            e["units"] += v.get("units", 1) * m
    classes = sorted(agg.values(), key=lambda c: (-c["count"], c["kind"], str(c["scope"])))
    rule = _finish(fid, gates, classes, [{"scope": s, "head": p, "words": w} for (s, p), w in interior.most_common()], X, W, extras)
    rule["violations"] = [dict(v, definition=fid) for v in viol.values()]
    return rule


def unit_rule(fn: Any, X: int = 16, W: int = 32, extras: int = 0, *, memo: dict | None = None, max_gates: int = MAX_GATES) -> dict[str, Any]:
    """One Definition specialization under `Q_word_v1{X,W}`: {"definition", "query", "gates", "units_per_call", "by_kind",
    "committed_interior_words", "max_out_bits", "classes", "violations"}.  A class groups units by kind, the primitive and the
    activation of their first boundary gate, gates per unit and |out(S)|; `count` is units per Call, `nodes` the primitives of one
    unit.  `committed_interior` lists the committed interior values by (activation, primitive): what a tap must acquire."""
    memo = {} if memo is None else memo
    key = (fn.id, X, W, extras)
    if key in memo:
        return memo[key]
    cap = X + extras
    if isinstance(fn, PrimitiveDefinition):
        bits = fn.ret.w
        classes = [] if not fn.params else [{"kind": "output", "head": fn.id, "scope": fn.id, "gates": 1, "out_bits": bits, "count": 1,
                                             "nodes": {fn.id: 1}, "ok": bits <= cap or bits <= W + extras}]
        rule = _finish(fn.id, 1, classes, [], X, W, extras)
    elif (sep := _separable(fn)) is not None:
        rule = _combine(fn.id, fn.gates, [(unit_rule(c, X, W, extras, memo=memo, max_gates=max_gates), m) for c, m in sep], X, W, extras)
    elif fn.gates > max_gates:
        rule = _finish(fn.id, fn.gates, [], [], X, W, extras)
        rule["violations"] = [{"definition": fn.id, "class": "too-large", "gates": fn.gates, "limit": max_gates}]
    else:
        G = Graph(fn)
        R = units(G, X, W, extras)
        kinds = np.array([_KINDS.index(k) for k in R["kinds"]], np.int64)
        cols = np.stack([kinds, R["head_prim"], R["head_scope"], R["gates"], R["out_bits"], R["out_gates"]], 1)
        keys, first, count = np.unique(cols, axis=0, return_index=True, return_counts=True)
        pick = np.zeros(len(kinds), bool)
        pick[first] = True
        sel = pick[R["owner"]]
        pu, pc = np.unique(np.stack([R["owner"][sel], G.prim[sel]], 1), axis=0, return_counts=True)
        nodes: dict[int, Counter] = {}
        for (u, p), k in zip(pu.tolist(), pc.tolist()):
            nodes.setdefault(u, Counter())[G.prim_names[p]] = k
        classes = []
        for (kd, hp, hs, g, ob, og), u, k in zip(keys.tolist(), first.tolist(), count.tolist()):
            classes.append({"kind": _KINDS[kd], "head": G.prim_names.get(hp), "scope": G.scope_names.get(hs), "gates": g, "out_bits": ob,
                            "out_gates": og, "count": k, "ok": bool(R["ok"][u]), "nodes": dict(nodes.get(u, Counter()).most_common(8))})
        classes.sort(key=lambda c: (-c["count"], c["kind"], str(c["scope"])))
        interior = Counter()
        for c in classes:
            if c["kind"] == "committed":
                interior[(c["scope"], c["head"])] += c["out_gates"] * c["count"]
        rule = _finish(fn.id, G.n, classes, [{"scope": s, "head": p, "words": w} for (s, p), w in interior.most_common()], X, W, extras)
    memo[key] = rule
    return rule


def _finish(fid: str, gates: int, classes: list[dict], interior: list[dict], X: int, W: int, extras: int) -> dict[str, Any]:
    viol = [{"definition": fid, "class": f"{c['kind']}:{c['head']}@{c['scope']}", "out_bits": c["out_bits"], "out_gates": c.get("out_gates", 1),
             "limit": [X + extras, W + extras], "units": c["count"]} for c in classes if not c["ok"]]
    by_kind = Counter()
    for c in classes:
        by_kind[c["kind"]] += c["count"]
    return {"definition": fid, "query": query_id(X, W, extras), "gates": gates, "units_per_call": sum(by_kind.values()),
            "by_kind": dict(by_kind), "committed_interior_words": sum(i["words"] for i in interior), "committed_interior": interior,
            "max_out_bits": max((c["out_bits"] for c in classes), default=0), "classes": classes, "violations": viol}


def check_calls(groups: Iterable[dict[str, Any]], X: int = 16, W: int = 32, extras: int = 0, *, strict: bool = True,
                specialize: Callable[[str, dict], Any] = specialization) -> dict[str, Any]:
    """`Q_word_v1{X,W}` over a Program's Calls, grouped: each group {"family", "statics", "definition", "calls", "where",
    "inputs_committed", "outputs_committed"}.  Per group: its Definition's unit rule and the committed-boundary conditions (every
    Value the Call reads and every Value it outputs is committed, else a unit would cross Calls).  Violations are aggregated per
    (Definition, class) with the Calls they cover and the first modules; `strict` raises `QueryRuleViolation` naming them."""
    memo: dict = {}
    out: list[dict[str, Any]] = []
    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def note(v: dict[str, Any], g: dict[str, Any]) -> None:
        a = agg.setdefault((v["definition"], v["class"]), dict(v, calls=0, where=[]))
        a["calls"] += int(g.get("calls") or 0)
        if g.get("where") is not None and len(a["where"]) < 8 and g["where"] not in a["where"]:
            a["where"].append(g["where"])

    for g in groups:
        try:
            rule = unit_rule(specialize(g["family"], g["statics"]), X, W, extras, memo=memo)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            rule = _finish(g["definition"], 0, [], [], X, W, extras)
            rule["violations"] = [{"definition": g["definition"], "class": "unresolved", "why": f"{type(e).__name__}: {str(e)[:200]}"}]
        v = list(rule["violations"])
        if not g.get("inputs_committed", True):
            v.append({"definition": g["definition"], "class": "input-not-committed",
                      "why": "the Call reads a Value interior to its module body: a unit would cross Calls"})
        if not g.get("outputs_committed", True):
            v.append({"definition": g["definition"], "class": "output-not-committed",
                      "why": "the Call's output is interior to its module body (not a required value): no unit can end at it"})
        for x in v:
            note(x, g)
        k = int(g.get("calls") or 0)
        out.append({"where": g.get("where"), "definition": g["definition"], "calls": k, "units": rule["units_per_call"] * k,
                    "gates": rule["gates"] * k, "committed_interior_words": rule["committed_interior_words"] * k, "rule": rule,
                    "violations": len(v)})
    viol = sorted(agg.values(), key=lambda a: (-a["calls"], a["definition"], a["class"]))
    res = {"query": query_id(X, W, extras), "ok": not viol, "violations": viol, "groups": out,
           **{k: sum(g[k] for g in out) for k in ("calls", "units", "gates", "committed_interior_words")}}
    if strict and viol:
        raise QueryRuleViolation(f"{query_id(X, W, extras)}: {len(viol)} violation(s): " + "; ".join(_name(x) for x in viol[:12])
                                 + (" ..." if len(viol) > 12 else ""))
    return res


def _name(x: dict[str, Any]) -> str:
    what = f" (|out(S)| = {x['out_bits']} b over {x['out_gates']} gate(s); limit {x['limit'][0]} b, or one gate of {x['limit'][1]} b)" \
        if "out_bits" in x else ""
    where = f" under {x['where'][0]}" + (f" (+{len(x['where']) - 1})" if len(x["where"]) > 1 else "") if x.get("where") else ""
    return f"{x['definition']}: {x['class']}{what}, {x['calls']} Call(s){where}"


def call_groups(P: Any, Q: Any) -> list[dict[str, Any]]:
    """The Calls of `P` grouped for `check_calls` under the population `Q` (`required.Population`: `Q_module_body_v1` + its
    policy): committed = the required values, and a literal's output (written in the descriptor)."""
    committed = Q.required.values
    literals = Q.literals
    groups: dict[tuple, dict[str, Any]] = {}
    for c in P.calls:
        if c.id in literals:
            continue
        ins = all(r in committed or r.call in literals for r in c.reads)
        outs = all((c.id, p.name) in committed for p in c.port_table)
        where = Q.program_path(c.id)
        k = (c.definition, where, ins, outs)
        g = groups.get(k)
        if g is None:
            g = groups[k] = {"family": c.family, "statics": c.statics, "definition": c.definition, "where": where, "calls": 0,
                             "inputs_committed": ins, "outputs_committed": outs}
        g["calls"] += 1
    return list(groups.values())


def check_query(P: Any, Q: Any, X: int = 16, W: int = 32, extras: int = 0, *, strict: bool = True) -> dict[str, Any]:
    """Check the Program `P` against `Q_word_v1{X,W}` refining the population `Q` (see `call_groups`); `strict` raises
    `QueryRuleViolation` naming every violating (Definition, class)."""
    return check_calls(call_groups(P, Q), X, W, extras, strict=strict)


__all__ = ["QUERY_ID", "TRIVIAL_GATES", "QueryRuleViolation", "query_id", "parse", "specialization", "Graph", "partition", "units",
           "unit_rule", "check_calls", "call_groups", "check_query"]
