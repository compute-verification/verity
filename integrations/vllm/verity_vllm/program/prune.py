"""A flat body's dead units, and the body without them; the registered compiler's dead-free form of a body.

A unit is one node of a body: a Call, a primitive, a batch or a scan. It is live when the body's return reads one of its outputs,
or a live unit's arguments do; a unit no live unit or return reads is dead, whatever its gates compute. This is the unit level of
circuit-check's dead-gates property (`verity.ml.circuits.liveness`, transitive), which also counts the gates no output reads inside
a live unit. `without_dead_units` drops the dead units and renumbers the references to the rest, each reference kept in its form
(an `Affine` stays an `Affine`, a `Concat` keeps its parts, a cut of a `PartLog` stays a cut), so that equal bodies stay equal.

`dead_free` also swaps each Call of a Definition with dead gates inside it for its version without them (`dead_free_versions`,
`registry.dead_free`): the same statics, signature and function, so only the node's gates change. And it narrows each linear
whose output words are read only in part (`narrowed`, `COLUMNS`) to the columns read: the last layer's fused qkv on the prompt
rows before the last, whose queries nothing reads, becomes a K/V-only projection over the weight's K and V rows. The Build's
derive and `predict.build` put the request wrapper alike in that form at one rank (`kept_records`), and at a tensor-parallel rank
under the Build's `--dead-free` (`build_program(dead_free=True)`): the registered compiler's Programs. A Program the Build lifts to
bits (`TargetProfile.ir == "boolean"`) takes the form without the swaps (`BITS`): no Boolean Definition names a dead-free version
as its word, so it keeps the v1s, whose Boolean versions the lift reads.

Class G's request Programs take that form batched (`batched_form`, `BATCHED`; `build_program(batched=True)`, the Build's `--batched`): each run of
consecutive Calls of one Definition that read no output of the run becomes one `batch` node over them, in order. An argument
every member reads alike is shared (axis None); any other is mapped on axis 0, its members' references stacked: one `Affine` where
they are consecutive runs of one node or Input, one `Strided` where they are evenly spaced, else their `concat`. A reference to a
member's output becomes one to the batch's output at the member's leaves, in its form (a cut of a `PartLog` stays a cut; a plain
`Concat` is put back in `concat`'s normal form). The form reads the body as written and nothing else, so equal bodies give equal
forms. Two kinds of Call stay nodes of their own, for the root readers that take only Calls: an attention's, which holds the KV
`commit.memory_map` reads (`holds_kv`), and a collective, which reads peer Inputs (`tp_binding.rank_sites`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import cache
from itertools import accumulate, chain, islice
from typing import Any

from verity.ml.circuits.defs import Body, Node, bind
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Refs, Strided, concat
from verity.ml.circuits.types import Array


def _parts(r: Concat, done: dict[int, list]) -> Iterable[Refs]:
    """The parts of `r` a walk that accumulates (a union, idempotent) has yet to visit: for a `Concat` cut from a `PartLog`, the
    log's parts past the longest cut of it the walk has visited (`done`, by log), then its own. An attention's key sequence is the
    one before it plus a row, so a walk over a layer's cuts visits each of its parts once, not once per row."""
    if r.prefix is None:
        return r.parts
    log, j = r.prefix
    d = done.get(id(log))
    if d is None or d[0] is not log:
        d = done[id(log)] = [log, 0]
    lo, d[1] = d[1], max(d[1], j)
    return chain(islice(log.parts, lo, j), r.parts[j:])


def _mapped(r: Concat, f: Callable[[Refs], Refs], logs: dict[int, list]) -> Refs:
    """`Concat` of `f` of each part of `r`, or `r` itself when `f` changes none. A `Concat` cut from a `PartLog` stays a cut, of the
    log of the log's parts under `f` (`logs`, by log), which maps each part once; a plain `Concat` keeps its parts as they are."""
    if r.prefix is None:
        parts = [f(p) for p in r.parts]
        return r if all(p is q for p, q in zip(parts, r.parts)) else Concat(parts)
    log, j = r.prefix
    st = logs.get(id(log))
    if st is None or st[0] is not log:
        st = logs[id(log)] = [log, PartLog(), None]   # the log, its parts under f, the first part f changes
    sub = st[1]
    for i in range(len(sub), j):
        p = log.parts[i]
        q = f(p)
        if q is not p and st[2] is None:
            st[2] = i
        sub.append(q)
    tail = [f(p) for p in r.parts[j:]]
    if (st[2] is None or st[2] >= j) and all(p is q for p, q in zip(tail, r.parts[j:])):
        return r
    return sub.raw_cut(j, tail)


def _read(r: Refs, out: set[int], done: dict[int, list]) -> None:
    if isinstance(r, (Affine, Strided)):
        if r.space == "n" and r.count:
            out.add(r.idx)
    elif isinstance(r, Concat):
        for p in _parts(r, done):
            _read(p, out, done)
    elif isinstance(r, Explicit):
        out.update(i for s, i, _ in r.items if s == "n")
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")


def live_units(body: Body) -> list[bool]:
    """Per node of `body`, whether the return reads it through live nodes (a node reads only earlier nodes)."""
    live, read, done = [False] * len(body.nodes), set(), {}
    _read(body.ret, read, done)
    for k in range(len(body.nodes) - 1, -1, -1):
        if k in read:
            live[k] = True
            for a in body.nodes[k].args:
                _read(a.refs, read, done)
    return live


def dead_units(body: Body) -> list[int]:
    """The indices of `body`'s dead units, in body order."""
    return [k for k, v in enumerate(live_units(body)) if not v]


def _renumbered(r: Refs, new: list[int], memo: dict[int, Refs], logs: dict[int, list]) -> Refs:
    if (hit := memo.get(id(r))) is not None:
        return hit
    if isinstance(r, Affine):
        out = Affine("n", new[r.idx], r.base, r.count) if r.space == "n" and r.count and new[r.idx] != r.idx else r
    elif isinstance(r, Strided):
        out = (Strided("n", new[r.idx], r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
               if r.space == "n" and r.count and new[r.idx] != r.idx else r)
    elif isinstance(r, Concat):
        out = _mapped(r, lambda p: _renumbered(p, new, memo, logs), logs)
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
    logs: dict[int, list] = {}

    def coll(a: Coll) -> Coll:
        r = _renumbered(a.refs, new, memo, logs)
        return a if r is a.refs else Coll(a.type, r)

    nodes, offsets = [], [0]
    for k in kept:
        n = body.nodes[k]
        nodes.append(replace(n, args=tuple(coll(a) for a in n.args)))
        offsets.append(offsets[-1] + n.gates)
    return Body(body.params, body.ret_type, nodes, _renumbered(body.ret, new, memo, logs), offsets), kept


@cache
def dead_free_versions() -> dict[str, Any]:
    """`Name_vN` of each Definition with dead gates the registered compiler emits -> its version without them."""
    from verity.ml.definitions.registry import dead_free as DF
    return {"RMSNormTriton_v1": DF.RMSNormTritonV5, "TokenSelect_v1": DF.TokenSelectV4, "GumbelTopPTokenSelect_v2": DF.GumbelTopPTokenSelectV3,
            "Attention_v5": DF.AttentionV18}


def swapped(body: Body) -> Body:
    """`body` with each node of a Definition `dead_free_versions` names bound to its dead-free version at the same statics."""
    table = dead_free_versions()
    nodes = [replace(n, fn=bind(table[n.fn.base_id], **n.fn.bindings)) if getattr(n.fn, "base_id", None) in table else n
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


def _reach(r: Refs, spans: dict[int, list[int]], full: dict[int, int], seen: set[int], done: dict[int, list]) -> None:
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
        for p in _parts(r, done):
            _reach(p, spans, full, seen, done)
    elif isinstance(r, Explicit):
        for s, i, j in r.items:
            if s == "n" and i in spans:
                _widen(spans[i], j, j + 1)
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")


def _shifted(r: Refs, by: dict[int, int], memo: dict[int, Refs], logs: dict[int, list]) -> Refs:
    """`r` with each leaf of a node k in `by` moved down by `by[k]`, each reference kept in its form."""
    if (hit := memo.get(id(r))) is not None:
        return hit
    if isinstance(r, Affine):
        out = Affine("n", r.idx, r.base - by[r.idx], r.count) if r.space == "n" and r.count and r.idx in by else r
    elif isinstance(r, Strided):
        out = (Strided("n", r.idx, r.base - by[r.idx], r.inner, r.outer_stride, r.inner_stride, r.count)
               if r.space == "n" and r.count and r.idx in by else r)
    elif isinstance(r, Concat):
        out = _mapped(r, lambda p: _shifted(p, by, memo, logs), logs)
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
    done: dict[int, list] = {}
    for n in body.nodes[min(cand) + 1:]:
        for a in n.args:
            _reach(a.refs, spans, full, seen, done)
    _reach(body.ret, spans, full, seen, done)
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
    logs: dict[int, list] = {}
    for k in range(min(cols) + 1, len(nodes)):
        n = nodes[k]
        args = tuple(Coll(c.type, r) if (r := _shifted(c.refs, by, memo, logs)) is not c.refs else c for c in n.args)
        if any(x is not y for x, y in zip(args, n.args)):
            nodes[k] = replace(n, args=args)
    offsets = [0]
    for n in nodes:
        offsets.append(offsets[-1] + n.gates)
    return Body(body.params, body.ret_type, nodes, _shifted(body.ret, by, memo, logs), offsets), cols


#: `kept_records`' form for a Program the Build lifts to bits: the registered compiler's form without the swaps
BITS = "bits"
#: `kept_records`' form for class G's request Programs: the registered compiler's form, `batched`
BATCHED = "batched"
#: an attention's KV parameters, which `commit.memory_map` reads from one Call (its `KV` and `LATENT`)
KV = ("kc", "vc")
LATENT = {"SparseMla": ("kv",), "SparseMlaSplit": ("kv",)}
#: the root parameters a collective reads (`tp_binding.PEERS`)
PEERS = "peers_"


def holds_kv(fn: Any) -> bool:
    """Whether a Call of `fn` holds an attention's KV (`KV`, `LATENT`)."""
    return set(LATENT.get(getattr(fn, "name", None), KV)) <= {p for p, _ in fn.signature()[0]}


def _top(r: Refs, peers: frozenset[int], done: dict[int, list]) -> tuple[int, bool]:
    """The greatest node `r` reads (-1 for none) and whether it reads a parameter in `peers`. A cut of a `PartLog` reads the
    log's running maxima (`done`, by log), so a layer's cuts visit each of its parts once."""
    if isinstance(r, (Affine, Strided)):
        if not r.count:
            return -1, False
        return (r.idx, False) if r.space == "n" else (-1, r.space == "p" and r.idx in peers)
    if isinstance(r, Explicit):
        return (max((i for s, i, _ in r.items if s == "n"), default=-1), any(s == "p" and i in peers for s, i, _ in r.items))
    if not isinstance(r, Concat):
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")
    hi, peer, parts = -1, False, r.parts
    if r.prefix is not None:
        log, j = r.prefix
        d = done.get(id(log))
        if d is None or d[0] is not log:
            d = done[id(log)] = [log, [-1], [False]]
        tops, read = d[1], d[2]
        while len(tops) <= j:
            k, q = _top(log.parts[len(tops) - 1], peers, done)
            tops.append(max(tops[-1], k))
            read.append(read[-1] or q)
        hi, peer, parts = tops[j], read[j], parts.tail
    for p in parts:
        k, q = _top(p, peers, done)
        hi, peer = max(hi, k), peer or q
    return hi, peer


def _same(a: Refs, b: Refs) -> bool:
    """Whether `a` and `b` are the same references as written."""
    if a is b:
        return True
    if type(a) is not type(b) or a.count != b.count:
        return False
    if isinstance(a, Affine):
        return (a.space, a.idx, a.base) == (b.space, b.idx, b.base)
    if isinstance(a, Strided):
        return (a.space, a.idx, a.base, a.inner, a.outer_stride, a.inner_stride) == (b.space, b.idx, b.base, b.inner, b.outer_stride, b.inner_stride)
    if isinstance(a, Explicit):
        return a.items == b.items
    return len(a.parts) == len(b.parts) and all(_same(p, q) for p, q in zip(a.parts, b.parts))


def _stacked(rs: list[Refs]) -> Refs:
    """The references of a mapped argument whose member m reads `rs[m]`."""
    r0, r1 = rs[0], rs[1]
    if isinstance(r0, Affine) and isinstance(r1, Affine) and r0.count and (s := r1.base - r0.base) > 0 and all(
            isinstance(r, Affine) and (r.space, r.idx, r.base, r.count) == (r0.space, r0.idx, r0.base + m * s, r0.count) for m, r in enumerate(rs)):
        n = len(rs) * r0.count
        return Affine(r0.space, r0.idx, r0.base, n) if s == r0.count else Strided(r0.space, r0.idx, r0.base, r0.count, s, 1, n)
    return concat(rs)


def _joined(parts: list[Refs]) -> list[Refs]:
    """`parts` with each two neighbours that are one Affine run joined: members' outputs a batch made adjacent."""
    out: list[Refs] = []
    for p in parts:
        q = out[-1] if out else None
        if (isinstance(p, Affine) and isinstance(q, Affine) and (p.space, p.idx) == (q.space, q.idx)
                and p.base == q.base + q.count):
            out[-1] = Affine(q.space, q.idx, q.base, q.count + p.count)
        else:
            out.append(p)
    return out


def _moved(r: Refs, at: list[tuple[int, int]], memo: dict[int, Refs], logs: dict[int, list]) -> Refs:
    """`r` with each leaf j of node k moved to leaf `at[k][1]` + j of node `at[k][0]`, each reference kept in its form but a plain
    Concat's, whose neighbouring runs are joined (`_joined`)."""
    if (hit := memo.get(id(r))) is not None:
        return hit
    if isinstance(r, (Affine, Strided)):
        if r.space != "n" or not r.count or at[r.idx] == (r.idx, 0):
            out = r
        else:
            k, o = at[r.idx]
            out = (Affine("n", k, o + r.base, r.count) if isinstance(r, Affine) else
                   Strided("n", k, o + r.base, r.inner, r.outer_stride, r.inner_stride, r.count))
    elif isinstance(r, Concat):
        if r.prefix is None:
            parts = [_moved(p, at, memo, logs) for p in r.parts]
            out = r if all(p is q for p, q in zip(parts, r.parts)) else concat(_joined(parts))
        else:
            out = _mapped(r, lambda p: _moved(p, at, memo, logs), logs)
    elif isinstance(r, Explicit):
        items = tuple(("n", at[i][0], at[i][1] + j) if s == "n" else (s, i, j) for s, i, j in r.items)
        out = r if items == r.items else Explicit(items)
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")
    memo[id(r)] = out
    return out


def batched_form(body: Body) -> tuple[Body, list[int]]:
    """`body` in the batched form (module docstring), and how many of `body`'s nodes each of its nodes holds, in order."""
    nodes, peers = body.nodes, frozenset(i for i, (name, _) in enumerate(body.params) if name.startswith(PEERS))
    at: list[tuple[int, int]] = []
    out: list[Node] = []
    sizes: list[int] = []
    memo: dict[int, Refs] = {}
    logs: dict[int, list] = {}
    done: dict[int, list] = {}
    kv: dict[str, bool] = {}
    k = 0
    while k < len(nodes):
        n, hi = nodes[k], k + 1
        if n.form == "call" and not (kv[n.fn.id] if n.fn.id in kv else kv.setdefault(n.fn.id, holds_kv(n.fn))) and not (
                peers and any(_top(a.refs, peers, done)[1] for a in n.args)):
            while hi < len(nodes) and nodes[hi].form == "call" and nodes[hi].fn.id == n.fn.id and all(
                    (t := _top(a.refs, peers, done))[0] < k and not t[1] for a in nodes[hi].args):
                hi += 1
        j = len(out)
        if hi == k + 1:
            args = tuple(a if (r := _moved(a.refs, at, memo, logs)) is a.refs else Coll(a.type, r) for a in n.args)
            out.append(n if all(x is y for x, y in zip(args, n.args)) else replace(n, args=args))
            at.append((j, 0))
        else:
            members, args, axes = nodes[k:hi], [], []
            for g, a in enumerate(n.args):
                rs = [_moved(m.args[g].refs, at, memo, logs) for m in members]
                shared = all(_same(r, rs[0]) for r in rs[1:])
                args.append(Coll(a.type, rs[0]) if shared else Coll(Array(len(rs), a.type), _stacked(rs)))
                axes.append(None if shared else 0)
            out.append(Node("batch", n.fn, tuple(args), Array(hi - k, n.out), axes=tuple(axes), n=hi - k, name=n.name))
            at.extend((j, m * n.out.leaves) for m in range(hi - k))
        sizes.append(hi - k)
        k = hi
    if len(out) == len(nodes):
        return body, sizes
    return Body(body.params, body.ret_type, out, _moved(body.ret, at, memo, logs), [0, *accumulate(n.gates for n in out)]), sizes


def registered_form(ir: str | None) -> bool | str:
    """The form (`kept_records`' `registered`) a request wrapper the Build derives on `ir` (`TargetProfile.ir`) takes: `BITS` on bits,
    else the registered compiler's."""
    return BITS if ir == "boolean" else True


def _dead_free(body: Body, *, swap: bool = True) -> tuple[Body, list[int], dict[int, tuple[int, int]]]:
    pruned, kept = without_dead_units(body)
    form, cols = narrowed(swapped(pruned) if swap else pruned)
    return form, kept, cols


def dead_free(body: Body, *, swap: bool = True, batched: bool = False) -> tuple[Body, list[int]]:
    """`body` in the registered compiler's form: without its dead units, the rest `swapped` (unless not `swap`) and `narrowed`, then
    `batched_form` where `batched`; and the index in `body` of each node kept, in order (before `batched_form`)."""
    form, kept, _ = _dead_free(body, swap=swap)
    return (batched_form(form)[0] if batched else form), kept


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


def kept_records(body: Body, calls: list | None, hints: dict[int, Any], *,
                 registered: bool | str) -> tuple[Body, list | None, dict[int, Any]]:
    """A derivation's flat `body` with its runtime correspondence's `calls` (one per node; None without a correspondence) and
    its node `hints` (by node index), in the registered compiler's form (`dead_free`) when `registered`, without its swaps when
    `registered` is `BITS`, and `batched_form` when it is `BATCHED`. A swapped node keeps its Call record: its arguments and outputs
    are the v1's. A narrowed node's record keeps the columns it computes: its output's and its column arguments' leaves cut to them.
    A batch's record is its members' (`{"members": [...]}`, in order), its hint its first member's."""
    if not registered:
        return body, calls, dict(hints)
    form, kept, cols = _dead_free(body, swap=registered != BITS)
    kept_hints = {i: hints[k] for i, k in enumerate(kept) if k in hints}
    recs = None if calls is None else [calls[k] for k in kept]
    for i, (a, b) in cols.items() if recs is not None else ():
        n, rec = body.nodes[kept[i]], recs[i]
        N, widths = n.fn.bindings["N"], COLUMNS[n.fn.base_id]
        per = n.out.leaves // N
        recs[i] = {**rec, "returns": [_cut(s, a * per, b * per) for s in rec["returns"]],
                   "args": [_cut(s, a * (n.args[j].type.leaves // N), b * (n.args[j].type.leaves // N)) if j in widths else s
                            for j, s in enumerate(rec["args"])]}
    if registered == BATCHED:
        form, sizes = batched_form(form)
        firsts = list(accumulate([0, *sizes[:-1]]))
        kept_hints = {j: kept_hints[i] for j, i in enumerate(firsts) if i in kept_hints}
        if recs is not None:
            recs = [recs[i] if m == 1 else {"members": recs[i:i + m]} for i, m in zip(firsts, sizes)]
    return form, recs, kept_hints
