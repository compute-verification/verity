"""The greedy token select lowered into the node window (`node_window`) as its served stages (`serving_templates.STAGES`,
`greedy_stages`): a `TokenSelect_v4{V}` node (or the `TokenSelect_v1{V}` row it is the function of) as the units of
`greedy_stages.plan(V)`, the K - 1 chunks `GreedyChunk_v2{N,OFF}`, each its own template, then the combine
`GreedyCombine_v2{V,N}`, which reads the last chunk's lanes itself and each other chunk's result through its parameter
``r<c>``. The node's output is the combine's one field, the token's low and high words, which are its I32's two words.

`TokenSelect_v4` and `TokenSelect_v1` are one function: from (logits[0], 0), lane i is taken iff it is strictly above the
best (``Bf16GtStrict_v1``), so the lowest index of the largest logit wins, a NaN is never taken and the token is 0 when
logits[0] is a NaN; v4 computes it over chunks of `dead_free.CHUNK` and a combine from logits[0], v1 as one scan, the stages
over K chunks and a combine (each module's docstring). `test_node_window_select` evaluates the three on rows of a small V.

**The logits read.** The node's ``logits`` are an all-gather's output (`AllGather_v1`, a copy: its output is its parts in rank
order), so the lowering reads through it to the parts (`through`): the window's own rank's are its LM head's words, the
other ranks' a peer's input. A chunk's ``logits`` port, and the combine's, is cut along its outer elements into rows of R
lanes, R the gcd of the chunk's width, the first level of the gather that holds the rank's own slice (`NodeWindow.chunks`)
and every boundary between the parts, so each row is one rank's: the rank's own is a first-level chunk row, wired (gathered
whole where no chunk holds it), another rank's a free input under its peer's name (`NodeWindow.feed_rows`). A combine
parameter is wired to its chunk's output row.

**The length check.** The request's `StopCheck_v1` node, after the last select, is a unit of `GreedyStopCheck_v1`
(`greedy_stop`): the check over the ids, a row each wired to its select's combine, and that the last id is the last select's
combine recomputed over that combine's own rows. A unit of `StopCheck_v1` alone is no block, nor a gather of the ids (Flock's
composer takes a unit only with some row past 111 bytes, and an id is four); the combine's logits row is. Its bit, one u16
word, is the check the Program returns.

Refused by name (`Unexpressible`): a select row other than `TokenSelect_v4` or `_v1`, a length check whose last id is no
lowered select's token or whose members are not that select's, a stage template the row's
`serving_templates.unit_templates` doesn't list, and a cut past a proof block's rows (`interface.MAX_ROWS`)."""
from __future__ import annotations

import math
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.refs import Refs, concat, runs
from verity.commitments.windows import interface as IF
from verity.definitions.registry import b1
from verity.definitions.registry import greedy_stages as G
from verity.definitions.registry import greedy_stop as GS
from verity.definitions.registry import stop_check as SC
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.query.program_view import spec_base

from .build import Unexpressible
from .node_window import Lowering, NodeWindow, Piece, _family, _Free
from .window_builder import Row

#: the select rows lowered as `TokenSelect_v1`'s stages: each one's function is `TokenSelect_v1{V}`'s
ROWS = ("TokenSelect_v1", "TokenSelect_v4")
GATHER_FAMILY = "AllGather_v1"


def through(w: NodeWindow, refs: Refs) -> Refs:
    """``refs`` with each run of an all-gather's output (`AllGather_v1`, a copy of its parts in rank order) replaced by the
    parts it copies, all-gathers of all-gathers too."""
    out, at = [], 0
    for space, idx, lo, hi in runs(refs):
        sub = refs.slice(at, at + hi - lo)
        at += hi - lo
        node = w.fn.body.nodes[idx] if space == "n" else None
        if node is None or not descriptor_id(node.fn).startswith(GATHER_FAMILY + "{"):
            out.append(sub)
            continue
        L = node.out_leaves // w.members(node)
        m = lo // L
        if hi > (m + 1) * L:
            raise Unexpressible(f"{w.name}: a read of node {idx}'s leaves [{lo}, {hi}) spans two of its members")
        out.append(through(w, w.arg(node, 0, m).slice(lo - m * L, hi - m * L)))
    return out[0] if len(out) == 1 else concat(out)


def _bounds(refs: Refs) -> list[int]:
    out, at = [], 0
    for _s, _i, lo, hi in runs(refs):
        out.append(at)
        at += hi - lo
    return out


def _row(w: NodeWindow, refs: Refs) -> tuple[Row, ...] | _Free:
    """The one writer row holding ``refs``' lanes: a first-level gather chunk that holds them, else the row itself, else a
    gather of them made whole; a `_Free` where they are free."""
    try:
        rows = w.rows(refs)
    except _Free as e:
        return e
    held = w._chunked(rows)
    return (held[0],) if len(held) == 1 else (w._whole([rows])[0],)


def width(w: NodeWindow, refs: Refs, N: int) -> int:
    """The rows' lanes R a chunk's ``logits`` port is cut into (module docstring), for a row of logits ``refs`` (already
    `through`) in chunks of N, the rank's own lanes' writer rows read from its first lane (the first run of a node's)."""
    R = math.gcd(N, *_bounds(refs))
    own = next((at for at, (space, *_r) in zip(_bounds(refs), runs(refs)) if space == "n"), 0)
    try:
        rows = w._chunked(w.rows(refs.slice(own, own + N)))
    except _Free:
        return R
    return math.gcd(R, *(r.words for r in rows))


def stages(V: int) -> list[tuple[G.Unit, Any]]:
    """`greedy_stages.plan(V)`'s units, each with its template as `serving_templates` serves the `TokenSelect_v1{V}` row."""
    row = descriptor_id(bind(b1.TokenSelect, V=V))
    served = {t: ST.template(t).fn for t, _n in ST.unit_templates(row)}
    out = []
    for u in G.plan(V):
        t = descriptor_id(u.defn)
        if t not in served:
            raise Unexpressible(f"{row}: stage {t} is not among the row's unit templates (serving_templates.unit_templates)")
        out.append((u, served[t]))
    return out


def token_select(w: NodeWindow, j: int, node: Any) -> None:
    """A greedy select row as its served stages (module docstring): K - 1 chunks and the combine, each M units (unit t a
    member), the node's output the combine's."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    if spec_base(spec) not in ROWS:
        raise Unexpressible(f"{spec}: a select row the stages' lowering doesn't take (not one of {ROWS})")
    V = int(node.fn.bindings["V"])
    plan = stages(V)
    N = plan[0][0].lanes[1] - plan[0][0].lanes[0]
    logits = [through(w, w.arg(node, 0, t)) for t in range(M)]
    R = width(w, logits[0], N)
    k = N // R
    if k + 1 + len(plan) > IF.MAX_ROWS:
        raise Unexpressible(f"{spec}: chunks of {N} lanes in rows of {R} are {k} rows a port, past a proof block's {IF.MAX_ROWS}")
    at: dict[tuple, tuple[str, int]] = {}
    for u, fn in plan:
        t, a = w.unit(fam, fn, M)
        at[u.key] = (t, a)
        lo = u.lanes[0]
        grids = []
        for i in range(k):
            got = [_row(w, ls.slice(lo + i * R, lo + (i + 1) * R)) for ls in logits]
            free = next((g for g in got if isinstance(g, _Free)), None)
            grids.append(free if free is not None else [g for g in got if not isinstance(g, _Free)])
        w.feed_rows(t, "logits", a, grids, (M,))
        for param, key, _field in u.wires:
            wt, wa = at[key]
            w.feed(t, param, a, [(Row(wt, wa + m, "o0", 3),) for m in range(M)], (M,))
    w.combines[j] = (V, N, M, grids, [(param, *at[key]) for param, key, _field in plan[-1][0].wires])
    t, a = at[plan[-1][0].key]
    w.out(j, node, [Piece(0, 2, Row(t, a, "o0", 2), 1, 1)])


def stop_check(w: NodeWindow, j: int, node: Any) -> None:
    """`StopCheck_v1`, the request's length check over its outputs' ids, as a unit of `GreedyStopCheck_v1` (module
    docstring): the check and the last select's combine over the combine's own rows, its logits and its chunks' results,
    with the ids a row each, its select's token; its bit, one u16 word, is the check the Program returns."""
    spec, M = descriptor_id(node.fn), w.members(node)
    if getattr(node.fn, "defn", None) is not SC.StopCheck:
        raise Unexpressible(f"{spec}: a length check the lowering doesn't take (not {SC.StopCheck.id})")
    space, last, *_ = list(runs(w.arg(node, 0, 0)))[-1]
    if space != "n" or last not in w.combines:
        raise Unexpressible(f"{w.name}: node {j} ({spec}): its last id is no lowered select's token")
    V, N, members, grids, wires = w.combines[last]
    if members != M:
        raise Unexpressible(f"{w.name}: node {j} ({spec}): {M} members, its last select's {members}")
    fn = bind(GS.GreedyStopCheck, V=V, N=N, **{k: node.fn.bindings[k] for k in GS.STATICS[2:]})
    t, a = w.unit(_family(node), fn, M)
    w.feed_rows(t, "logits", a, grids, (M,))
    for param, wt, wa in wires:
        w.feed(t, param, a, [(Row(wt, wa + m, "o0", 3),) for m in range(M)], (M,))
    T, took = int(node.fn.bindings["T"]), w.take(node, 0)
    if isinstance(took, _Free) or any(len(rows) != T or any(r.words != 2 for r in rows) for rows in took):
        raise Unexpressible(f"{w.name}: node {j} ({spec}): its ids are not its {T} selects' tokens, a row each")
    w.feed_rows(t, "ids", a, [[(rows[i],) for rows in took] for i in range(T)], (M,))
    w.out(j, node, [Piece(0, 1, Row(t, a, "o0", 1))])


#: the select's and the length check's lowerings by family, beside `node_window.FAMILIES`
FAMILIES: dict[str, Lowering] = {"TokenSelect": token_select, "StopCheck": stop_check}

__all__ = ["FAMILIES", "ROWS", "stages", "stop_check", "through", "token_select", "width"]
