"""`Q_word_v1{X,W,R=no-recompute}`: every Call's circuit cut into proof units whose committed outputs are at most X bits, or one value
of at most W bits, over values serving commits.

The rule (no recomputes):

  * A gate is interior to exactly one unit or on the boundary of one or more units; each gate is **computed in exactly one unit**.
  * For every computation gate exactly one unit certifies it: its proof being valid implies the gate is the correct function of the
    committed values the unit reads.  A value read across units is **committed once at serving**, and every reader reuses it.  There
    are no public inputs: the Call's inputs (weights, tokens, earlier Calls' outputs) are committed, private values.  Constants and
    wiring (`WIRING`: bit rearrangement, such as BF16 -> FP32 zero-extension) are circuit structure: hardwired into each unit's
    statement, never committed, not counted as gates; a unit reads through wiring to its source.
  * **Width** counts committed elements: a unit passes at |out(S)| <= X (16, where the word allows it), or when out(S) is ONE value of at
    most W bits (32: float interiors such as a norm scale, attention's max, sum and inverse sum).  No dtype is read.
  * It refines `Q_module_body_v1` without moving its boundary: a Call's inputs and outputs are the bridge's required values.

The cut of one Call (`partition`), on its computing gates (`Graph`: wiring and constants removed, reads through them re-pointed):

  1. The returned gates, in return order, packed into outputs of at most X bits (a wider value is an output alone).
  2. Backward over the gates (the layout is topological): a gate joins its consumers' unit when they all belong to one; a gate
     several units read is committed as its own unit, and the pass repeats.
  3. A gate no returned value depends on (dead) joins the one unit it reads, else heads a dead unit; any value such a gate reads
     across units is committed too.

`verity.ir.partition.validate_unit_cut` checks every Call's cut; the committed interior values are what serving must acquire (a tap
where it does not today).  `unit_rule` summarises one specialization; `check_query` checks a Program and fails by name.
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

#: primitives that only rearrange bits (each output bit a constant or a copy of one input bit; `tests/query/test_word.py` checks it):
#: wiring, like constants, is computed in no unit, may be read by any, and is never committed
WIRING = frozenset({"Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"})

#: the rule's name in the query id
RULE = "no-recompute"

_KINDS = ("output", "committed", "dead")

#: Calls with more gates than this are not partitioned explicitly (`check_calls` names them)
MAX_GATES = 24_000_000


def query_id(X: int = 16, W: int = 32, extras: int = 0) -> str:
    return f"{QUERY_ID}{{X={int(X)},W={int(W)}" + (f",EXTRAS={int(extras)}" if extras else "") + f",R={RULE}}}"


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
        fid = v.get("fn") if isinstance(v, dict) else (json.loads(v).get("fn") if isinstance(v, str) and v.startswith('{"fn"') else v)
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
    """The explicit gates of one Call activation that COMPUTE (local index), with wiring and constants removed: declared width,
    primitive and activation per gate, whether it reads a Call parameter, and the wires between them (`src[e]` -> `dst[e]`), a read
    through wiring standing for reads of the wiring's source gates.  `outputs`: the returned gates in return order (a returned wiring
    value: its sources; a returned constant: nothing).  `total` / `free` / `free_by_prim`: the Call's gates, and the free ones."""

    def __init__(self, fn: Any) -> None:
        c, call = _call_scope(fn)
        lo, n = call.offset, fn.gates
        width = np.empty(n, np.int64)
        prim = np.empty(n, np.int64)
        scope = np.empty(n, np.int64)
        external = np.zeros(n, bool)
        free = np.zeros(n, bool)
        const = np.zeros(n, bool)
        self.prims: dict[str, int] = {}
        self.scopes: dict[str, int] = {}
        ops: list[list[int]] = []
        inputs: dict[int, int] = {}                                  # the Call's parameter gates it reads (root id -> index)
        wide: set[int] = set()
        for j in range(n):
            g = c.gate(lo + j)
            width[j] = g.prim.ret.w
            prim[j] = self.prims.setdefault(g.prim.id, len(self.prims))
            scope[j] = self.scopes.setdefault(g.scope.spec.id, len(self.scopes))
            const[j] = not g.prim.params
            free[j] = const[j] or g.prim.id in WIRING
            it = g.operands
            ext = False
            if len(it) > WIDE_FAN_IN:
                it = _wide_operands(g, call)
                ext = True
                wide.add(j)
            mine = []
            for o in it:
                if 0 <= o - lo < n:
                    mine.append(o - lo)
                else:
                    ext = True
                    mine.append(-1 - inputs.setdefault(o, len(inputs)))      # an input gate: -1 - its index
            external[j] = ext
            ops.append(mine)
        for j in range(n):                                          # a value computed from constants alone is a constant too
            if not free[j] and not external[j] and ops[j] and all(o >= 0 and const[o] for o in ops[j]):
                const[j] = free[j] = True
        # recomputes: a computation gate with the primitive and the operands (by value: constants, inputs, wiring, earlier canonical
        # gates) of an earlier one computes the same value again; a gate whose wide operands were not enumerated matches nothing
        intern: dict[tuple, int] = {}
        tok = [0] * n
        first: dict[tuple, int] = {}
        dup: list[tuple[int, int]] = []
        for j in range(n):
            key = (int(prim[j]), tuple(intern.setdefault(("i", -1 - o), len(intern)) if o < 0 else tok[o] for o in ops[j]))
            if free[j]:
                tok[j] = intern.setdefault(("s",) + key, len(intern))
            elif j in wide:
                tok[j] = intern.setdefault(("u", j), len(intern))
            elif key in first:
                dup.append((j, first[key]))
                tok[j] = tok[first[key]]
            else:
                first[key] = j
                tok[j] = intern.setdefault(("g", j), len(intern))
        srcs: dict[int, tuple[int, ...]] = {}
        for j in np.flatnonzero(free).tolist():
            s: set[int] = set()
            for o in ops[j]:
                s.update(srcs[o] if o >= 0 and free[o] else (o,))
            srcs[j] = tuple(sorted(s))
        keep = np.flatnonzero(~free)
        idx = np.full(n, -1, np.int64)
        idx[keep] = np.arange(len(keep))
        src, dst = array("q"), array("q")
        self.input_reads: list[tuple[int, int]] = []                 # (input index, computing gate)
        for j in keep.tolist():
            seen: set[int] = set()
            for o in ops[j]:
                for u in (srcs[o] if o >= 0 and free[o] else (o,)):
                    if u in seen:
                        continue
                    seen.add(u)
                    if u < 0:
                        self.input_reads.append((-1 - u, int(idx[j])))
                    else:
                        src.append(int(idx[u]))
                        dst.append(int(idx[j]))
        self.input_ids = sorted(inputs, key=inputs.__getitem__)       # root gate id of each input gate read
        self.total, self.free = n, int(free.sum())
        self.free_by_prim = Counter()
        for k, v in zip(*np.unique(prim[free], return_counts=True)):
            self.free_by_prim[k] = int(v)
        self.n = len(keep)
        self.keep = keep                                           # computation gate -> the Call's gate id
        self.recomputed = [(int(idx[a]), int(idx[b])) for a, b in dup]   # (gate, earlier gate computing the same value)
        self.width, self.prim, self.scope, self.external = width[keep], prim[keep], scope[keep], external[keep]
        self.src = np.frombuffer(src, np.int64) if len(src) else np.zeros(0, np.int64)
        self.dst = np.frombuffer(dst, np.int64) if len(dst) else np.zeros(0, np.int64)
        outs: list[int] = []
        for g in (resolve(call, r) - lo for r in fn.body.ret):
            if 0 <= g < n:
                outs += [int(idx[u]) for u in (srcs[g] if free[g] else (g,)) if u >= 0]
        self.outputs = outs
        self.prim_names = {v: k for k, v in self.prims.items()}
        self.scope_names = {v: k for k, v in self.scopes.items()}
        self.free_by_prim = Counter({self.prim_names[k]: v for k, v in self.free_by_prim.items()})


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


def _owners(G: Graph, head: np.ndarray, kinds: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Owner per gate for the given heads: backward, a gate joins its consumers' unit when they agree; a gate several units read
    directly becomes a head (committed); a gate no returned value depends on (dead) joins the one live unit it reads, else heads a
    dead unit.  Loops until every gate a unit reads from another unit is a head: the committed set is the boundary set."""
    n = G.n
    order = np.argsort(G.src, kind="stable")
    cons = array("q", G.dst[order].tobytes())                 # consumers of gate g: cons[start[g]:start[g + 1]]
    start = array("q", np.searchsorted(G.src[order], np.arange(n + 1)).tobytes())
    by_consumer = np.argsort(G.dst, kind="stable")
    dstart = np.searchsorted(G.dst[by_consumer], np.arange(n + 1))
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
        own = np.frombuffer(owner, np.int64).copy()
        s_, d_ = G.src[order], G.dst[order]
        promote = np.unique(s_[(own[s_] == -2) & (own[d_] >= 0)])
        if not len(promote):
            live = len(kinds)
            for g in np.flatnonzero(own == -1).tolist():              # reads of committed values (heads) cross units freely
                us = {int(own[G.src[e_]]) for e_ in by_consumer[dstart[g]:dstart[g + 1]].tolist() if head[G.src[e_]] < 0}
                alive = {u for u in us if 0 <= u < live}
                own[g] = alive.pop() if len(alive) == 1 and len(us) == 1 else -1
            for g in np.flatnonzero(own == -1).tolist():
                own[g] = len(kinds)
                kinds.append("dead")
            cross = own[G.src] != own[G.dst]
            promote = np.unique(G.src[cross][head[G.src[cross]] < 0])
            if not len(promote):
                return own, head
            del kinds[live:]
        head[promote] = np.arange(len(kinds), len(kinds) + len(promote))
        kinds += ["committed"] * len(promote)


def partition(G: Graph, X: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(owner per gate, kind per unit, head per gate): "output" (returned values), "committed" (a value another unit reads), "dead"
    (read by no returned value); `head[g] >= 0` marks the gates the cut commits -- the returned ones and every value read across units."""
    head = np.full(G.n, -1, np.int64)
    kinds: list[str] = []
    for t in _targets(G, X):
        head[t] = len(kinds)
        kinds.append("output")
    own, head = _owners(G, head, kinds)
    return own, kinds, head


def units(G: Graph, X: int, W: int, extras: int = 0) -> dict[str, Any]:
    """The partition of `G` with, per unit: gates, |out(S)| in bits over its boundary gates (read by another unit, or returned), and
    the partition invariant's check (`verity.ir.partition.validate_unit_cut` over the gates the cut commits; violations by gate id)."""
    from verity.ir.partition import INPUT_UNIT, validate_unit_cut
    owner, kinds, head = partition(G, X)
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
    # the Call's parameters stand in for the input unit: values committed outside the Call (the Program's inputs under their roots, or
    # another Call's committed outputs; `check_calls` requires each to be a required value)
    k = len(G.input_ids)
    reads = list(zip(G.src.tolist(), G.dst.tolist())) + [(G.n + i, q) for i, q in G.input_reads]
    committed = np.flatnonzero(head >= 0).tolist() + list(range(G.n, G.n + k))
    cut = validate_unit_cut(G.total + k, G.free, owner.tolist() + [INPUT_UNIT] * k, reads, G.outputs, committed, range(G.n, G.n + k),
                            G.recomputed)
    if not cut.ok:                                                 # the Call's gate ids (its canonical layout), not the compacted ones
        ids = G.keep.tolist() + [f"input:{g}" for g in G.input_ids]
        for k in cut.codes:
            if k != "gate-count":
                cut.detail[k]["gates"] = [[ids[a] for a in x] if isinstance(x, list) else ids[x] for x in cut.detail[k]["gates"]]
    ok = (out_bits <= X + extras) | ((out_gates == 1) & (out_bits <= W + extras))
    return {"owner": owner, "kinds": kinds, "gates": gates, "out_bits": out_bits, "out_gates": out_gates, "ok": ok, "cut": cut,
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
    interior, ibits = Counter(), Counter()
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
            ibits[(i["scope"], i["head"])] += i.get("bits", 32 * i["words"]) * m
        for v in r["violations"]:
            e = viol.setdefault(v["class"], dict(v, units=0))
            e["units"] += v.get("units", 1) * m
    classes = sorted(agg.values(), key=lambda c: (-c["count"], c["kind"], str(c["scope"])))
    rule = _finish(fid, gates, classes, [{"scope": s, "head": p, "words": w, "bits": ibits[(s, p)]} for (s, p), w in interior.most_common()],
                   X, W, extras)
    rule["violations"] = [dict(v, definition=fid) for v in viol.values()]
    rule["free_gates"] = sum(r.get("free_gates", 0) * m for r, m in parts)
    free = Counter()
    for r, m in parts:
        for k, v in (r.get("free_by_prim") or {}).items():
            free[k] += v * m
    rule["free_by_prim"] = dict(free.most_common(8))
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
    if isinstance(fn, PrimitiveDefinition) and (not fn.params or fn.id in WIRING):
        rule = _finish(fn.id, 1, [], [], X, W, extras)
        rule["free_gates"], rule["free_by_prim"] = 1, {fn.id: 1}
    elif isinstance(fn, PrimitiveDefinition):
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
        cut = R["cut"]
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
        interior, ibits = Counter(), Counter()
        for c in classes:
            if c["kind"] == "committed":
                interior[(c["scope"], c["head"])] += c["out_gates"] * c["count"]
                ibits[(c["scope"], c["head"])] += c["out_bits"] * c["count"]
        rule = _finish(fn.id, G.total, classes, [{"scope": s, "head": p, "words": w, "bits": ibits[(s, p)]} for (s, p), w in interior.most_common()],
                       X, W, extras)
        rule["free_gates"] = G.free
        rule["free_by_prim"] = dict(G.free_by_prim.most_common(8))
        if not cut.ok:
            rule["violations"].append({"definition": fn.id, "class": "cut", "codes": list(cut.codes), "detail": cut.detail})
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


#: where serving commits each committed interior value today, by (activation family, primitive of the value): the FA2 / FA3 hidden
#: stream's fields (`commit/hidden_stream.py` StreamLayout; S after the softcap), else None -- a new tap
_ATTENTION = ("AttnBlock", "AttnBlockSoftcap", "AttnBlockFA3", "DotBf16", "AttentionHead", "AttentionHeadSoftcap")
_STREAM = {"AmpereBF16TcDot16_v1": "S (scores)", "HopperBF16WgmmaDot16_v1": "S (scores)", "MufuTanh_v1": "S (softcapped scores)",
           "MufuEx2Ftz_v1": "P (f32 probabilities) / ROW rescale", "F2fpBf16_v1": "RP (bf16 probabilities)", "F32Max_v1": "ROW row_max",
           "F32MulFtz_v1": "ROW step (max * scale)", "F32AddFtz_v1": "ROW lane sums / FIN sum", "F32FmaFtz_v1": "ROW lane sums",
           "Fa2InvSum_v1": "FIN inv_sum", "Fa3InvSum_v1": "FIN inv_sum"}
#: the kernel a new tap goes in, by the Definition family holding the value
_TAP_KERNEL = {"AttnBlock": "FA2 / FA3 tap build (flash_fwd_kernel / hopper mainloop)", "AttnBlockSoftcap": "FA2 tap build (softcap instantiation)",
               "AttentionHead": "FA2 / FA3 tap build (epilogue)", "AttentionHeadSoftcap": "FA2 tap build (softcap epilogue)",
               "RMSNormTriton": "vLLM batch-invariant Triton _rms_norm_kernel", "RMSNormFusedCuda": "vLLM CUDA fused_add_rms_norm_kernel<bf16, 0, true>",
               "MoeRouterTopK": "vLLM _moe_C.topk_softmax (topkGatingSoftmax)", "MoeRouterTopKNorm": "vLLM _moe_C.topk_softmax (topkGatingSoftmax)"}


def committed_today(scope: str | None, head: str | None) -> str | None:
    """Where serving commits a committed interior value today (a hidden-stream field), or None: the value needs a new tap."""
    fam = (scope or "").split("{")[0].rsplit("_v", 1)[0]
    if fam in _ATTENTION:
        return _STREAM.get(head or "")
    return None


def tap_kernel(scope: str | None) -> str:
    fam = (scope or "").split("{")[0].rsplit("_v", 1)[0]
    return _TAP_KERNEL.get(fam, f"the {fam} kernel")


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


__all__ = ["QUERY_ID", "WIRING", "RULE", "committed_today", "tap_kernel", "QueryRuleViolation", "query_id", "parse", "specialization", "Graph", "partition", "units",
           "unit_rule", "check_calls", "call_groups", "check_query"]
