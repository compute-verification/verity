"""A flat body's dead units, and the body without them; the registered compiler's dead-free form of a body.

A unit is one node of a body: a Call, a primitive, a batch or a scan. It is live when the body's return reads one of its outputs,
or a live unit's arguments do; a unit no live unit or return reads is dead, whatever its gates compute. This is the unit level of
circuit-check's dead-gates property (`verity.ml.circuits.liveness`, transitive), which also counts the gates no output reads inside
a live unit. `without_dead_units` drops the dead units and renumbers the references to the rest, each reference kept in its form
(an `Affine` stays an `Affine`, a `Concat` keeps its parts), so that equal bodies stay equal.

`dead_free` also swaps each Call of a Definition with dead gates inside it for its version without them (`dead_free_versions`,
`registry.dead_free`): the same statics, signature and function, so only the node's gates change. And it narrows each linear
whose output words are read only in part (`narrowed`, `COLUMNS`) to the columns read: the last layer's fused qkv on the prompt
rows before the last, whose queries nothing reads, becomes a K/V-only projection over the weight's K and V rows. The Build's
derive and `predict.build` put the request wrapper alike in that form at one rank (`kept_records`), and at a tensor-parallel rank
under the Build's `--dead-free` (`build_program(dead_free=True)`): the registered compiler's Programs. Rows of record keep the v1s.
"""
from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Any

from verity.ml.circuits.defs import Body, Node, bind
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Refs, Strided


def _read(r: Refs, out: set[int]) -> None:
    if isinstance(r, (Affine, Strided)):
        if r.space == "n" and r.count:
            out.add(r.idx)
    elif isinstance(r, Concat):
        for p in r.parts:
            _read(p, out)
    elif isinstance(r, Explicit):
        out.update(i for s, i, _ in r.items if s == "n")
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")


def live_units(body: Body) -> list[bool]:
    """Per node of `body`, whether the return reads it through live nodes (a node reads only earlier nodes)."""
    live, read = [False] * len(body.nodes), set()
    _read(body.ret, read)
    for k in range(len(body.nodes) - 1, -1, -1):
        if k in read:
            live[k] = True
            for a in body.nodes[k].args:
                _read(a.refs, read)
    return live


def dead_units(body: Body) -> list[int]:
    """The indices of `body`'s dead units, in body order."""
    return [k for k, v in enumerate(live_units(body)) if not v]


def _renumbered(r: Refs, new: list[int], memo: dict[int, Refs]) -> Refs:
    if (hit := memo.get(id(r))) is not None:
        return hit
    if isinstance(r, Affine):
        out = Affine("n", new[r.idx], r.base, r.count) if r.space == "n" and r.count and new[r.idx] != r.idx else r
    elif isinstance(r, Strided):
        out = (Strided("n", new[r.idx], r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
               if r.space == "n" and r.count and new[r.idx] != r.idx else r)
    elif isinstance(r, Concat):
        parts = [_renumbered(p, new, memo) for p in r.parts]
        out = r if all(p is q for p, q in zip(parts, r.parts)) else Concat(parts)
    elif isinstance(r, Explicit):
        items = tuple((s, new[i] if s == "n" else i, j) for s, i, j in r.items)
        out = r if items == r.items else Explicit(items)
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")
    memo[id(r)] = out
    return out


def without_dead_units(body: Body) -> tuple[Body, list[int]]:
    """`body` without its dead units, and the index in `body` of each node kept, in order."""
    kept = [k for k, v in enumerate(live_units(body)) if v]
    if len(kept) == len(body.nodes):
        return body, kept
    new = list(range(len(body.nodes)))
    for i, k in enumerate(kept):
        new[k] = i
    memo: dict[int, Refs] = {}

    def coll(a: Coll) -> Coll:
        r = _renumbered(a.refs, new, memo)
        return a if r is a.refs else Coll(a.type, r)

    nodes, offsets = [], [0]
    for k in kept:
        n = body.nodes[k]
        nodes.append(replace(n, args=tuple(coll(a) for a in n.args)))
        offsets.append(offsets[-1] + n.gates)
    return Body(body.params, body.ret_type, nodes, _renumbered(body.ret, new, memo), offsets), kept


@cache
def dead_free_versions() -> dict[str, Any]:
    """`Name_vN` of each Definition with dead gates the registered compiler emits -> its version without them."""
    from verity.ml.definitions.registry import dead_free as DF
    return {"RMSNormTriton_v1": DF.RMSNormTritonV5, "TokenSelect_v1": DF.TokenSelectV4, "GumbelTopPTokenSelect_v2": DF.GumbelTopPTokenSelectV3,
            "Attention_v5": DF.AttentionV18}


def swapped(body: Body) -> Body:
    """`body` with each node of a Definition `dead_free_versions` names bound to its dead-free version at the same statics."""
    table = dead_free_versions()
    nodes = [replace(n, fn=bind(table[n.fn.base_id], **n.fn.bindings)) if n.form != "primitive" and n.fn.base_id in table else n
             for n in body.nodes]
    if all(a is b for a, b in zip(nodes, body.nodes)):
        return body
    offsets = [0]
    for n in nodes:
        offsets.append(offsets[-1] + n.gates)
    return Body(body.params, body.ret_type, nodes, body.ret, offsets)


#: `Name_vN` of a linear `x[K] @ W[N,K]^T [+ b[N]]` that is one coordinate Definition batched over W's rows, so that its output
#: word j reads only row j of each argument named -> those arguments (W's, and the bias's)
COLUMNS = {"Gemm_v1": (1,), "Gemm_v2": (1,), "GemmBias_v1": (1, 2), "GemmBias_v2": (1, 2)}


def _widen(span: list[int], lo: int, hi: int) -> None:
    span[0], span[1] = min(span[0], lo), max(span[1], hi)


def _reach(r: Refs, spans: dict[int, list[int]], full: dict[int, int], seen: set[int]) -> None:
    """Widen `spans[k]` ([least, past the greatest) leaf read) by the leaves `r` reads of each node k it holds."""
    if id(r) in seen:
        return
    seen.add(id(r))
    if isinstance(r, (Affine, Strided)):
        if r.space != "n" or not r.count or r.idx not in spans:
            return
        if isinstance(r, Affine):
            _widen(spans[r.idx], r.base, r.base + r.count)
        elif min(r.outer_stride, r.inner_stride) < 0:
            _widen(spans[r.idx], 0, full[r.idx])
        else:
            last = [r.count - 1] + ([(r.count - 1) // r.inner * r.inner - 1] if r.count > r.inner else [])
            _widen(spans[r.idx], r.base, max(r[j][2] for j in last) + 1)
    elif isinstance(r, Concat):
        for p in r.parts:
            _reach(p, spans, full, seen)
    elif isinstance(r, Explicit):
        for s, i, j in r.items:
            if s == "n" and i in spans:
                _widen(spans[i], j, j + 1)
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")


def _shifted(r: Refs, by: dict[int, int], memo: dict[int, Refs]) -> Refs:
    """`r` with each leaf of a node k in `by` moved down by `by[k]`, each reference kept in its form."""
    if (hit := memo.get(id(r))) is not None:
        return hit
    if isinstance(r, Affine):
        out = Affine("n", r.idx, r.base - by[r.idx], r.count) if r.space == "n" and r.count and r.idx in by else r
    elif isinstance(r, Strided):
        out = (Strided("n", r.idx, r.base - by[r.idx], r.inner, r.outer_stride, r.inner_stride, r.count)
               if r.space == "n" and r.count and r.idx in by else r)
    elif isinstance(r, Concat):
        parts = [_shifted(p, by, memo) for p in r.parts]
        out = r if all(p is q for p, q in zip(parts, r.parts)) else Concat(parts)
    elif isinstance(r, Explicit):
        items = tuple((s, i, j - by[i]) if s == "n" and i in by else (s, i, j) for s, i, j in r.items)
        out = r if items == r.items else Explicit(items)
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")
    memo[id(r)] = out
    return out


def narrowed(body: Body) -> tuple[Body, dict[int, tuple[int, int]]]:
    """`body` with each Call of a `COLUMNS` linear whose output words are read only in [a, b) ⊊ [0, N) bound at N = b - a over its
    column arguments' rows a..b-1, the references to its outputs moved down by a; and those nodes' [a, b) by node index."""
    cand = {k: n for k, n in enumerate(body.nodes) if n.form == "call" and n.fn.base_id in COLUMNS}
    if not cand:
        return body, {}
    full = {k: n.out.leaves for k, n in cand.items()}
    spans = {k: [full[k], 0] for k in cand}
    seen: set[int] = set()
    for n in body.nodes[min(cand) + 1:]:
        for a in n.args:
            _reach(a.refs, spans, full, seen)
    _reach(body.ret, spans, full, seen)
    cols, by, nodes = {}, {}, list(body.nodes)
    for k, n in cand.items():
        N = n.fn.bindings["N"]
        per = full[k] // N
        a, b = spans[k][0] // per, -(-spans[k][1] // per)
        if b <= a or (a, b) == (0, N):
            continue
        fn = bind(n.fn.defn, **{**n.fn.bindings, "N": b - a})
        params, ret = fn.signature()
        args = tuple(Coll(params[i][1], c.refs.slice(a * (c.type.leaves // N), b * (c.type.leaves // N))) if i in COLUMNS[n.fn.base_id]
                     else c for i, c in enumerate(n.args))
        nodes[k] = Node("call", fn, args, ret, name=n.name)
        cols[k], by[k] = (a, b), a * per
    if not cols:
        return body, {}
    memo: dict[int, Refs] = {}
    for k in range(min(cols) + 1, len(nodes)):
        n = nodes[k]
        args = tuple(Coll(c.type, r) if (r := _shifted(c.refs, by, memo)) is not c.refs else c for c in n.args)
        if any(x is not y for x, y in zip(args, n.args)):
            nodes[k] = replace(n, args=args)
    offsets = [0]
    for n in nodes:
        offsets.append(offsets[-1] + n.gates)
    return Body(body.params, body.ret_type, nodes, _shifted(body.ret, by, memo), offsets), cols


def _dead_free(body: Body) -> tuple[Body, list[int], dict[int, tuple[int, int]]]:
    pruned, kept = without_dead_units(body)
    form, cols = narrowed(swapped(pruned))
    return form, kept, cols


def dead_free(body: Body) -> tuple[Body, list[int]]:
    """`body` in the registered compiler's form: without its dead units, the rest `swapped` and `narrowed`; and the index in `body`
    of each node kept, in order."""
    return _dead_free(body)[:2]


def _cut(slot: dict, lo: int, hi: int) -> dict:
    """A Call record's return or argument slot cut to its leaves [lo, hi)."""
    if slot.get("leaves") is not None:
        return {**slot, "leaves": [slot["leaves"][0] + lo, slot["leaves"][0] + hi]}
    if not slot.get("spans"):
        return slot
    spans, at = [], 0
    for p in slot["spans"]:
        s, e = p["leaves"]
        x, y = max(lo, at), min(hi, at + e - s)
        if y > x:
            spans.append({**p, "leaves": [s + x - at, s + y - at]})
        at += e - s
    return {**slot, "spans": spans}


def kept_records(body: Body, calls: list, hints: dict[int, Any], *, registered: bool) -> tuple[Body, list, dict[int, Any]]:
    """A derivation's flat `body` with its runtime correspondence's `calls` (one per node) and its node `hints` (by node index), in
    the registered compiler's form (`dead_free`) when `registered`. A swapped node keeps its Call record: its arguments and outputs
    are the v1's. A narrowed node's record keeps the columns it computes: its output's and its column arguments' leaves cut to
    them."""
    if not registered:
        return body, calls, dict(hints)
    form, kept, cols = _dead_free(body)
    recs = [calls[k] for k in kept]
    for i, (a, b) in cols.items():
        n, rec = body.nodes[kept[i]], recs[i]
        N, widths = n.fn.bindings["N"], COLUMNS[n.fn.base_id]
        per = n.out.leaves // N
        recs[i] = {**rec, "returns": [_cut(s, a * per, b * per) for s in rec["returns"]],
                   "args": [_cut(s, a * (n.args[j].type.leaves // N), b * (n.args[j].type.leaves // N)) if j in widths else s
                            for j, s in enumerate(rec["args"])]}
    return form, recs, {i: hints[k] for i, k in enumerate(kept) if k in hints}
