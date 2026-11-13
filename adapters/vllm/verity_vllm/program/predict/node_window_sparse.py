"""The node window's lowerings (`node_window`) of GLM-5.3's DSA attention and MoE layer, the families `FAMILIES` names.

* `DsaLatentRow_v1` is the row itself, a member a unit: it reads its 576 latent coordinates of q_a whole (through the frame's
  gather), its norm's weights and its position's cos/sin row, and returns the key row the attention reads (`HISTORY`'s writer).
* `BatchedGemm_v1{B, K, N}` is its ``SLICES``: unit (b, n) reads head b's K words of x whole and returns output word (b, n).
  Its weight ``w[b, n, :]`` is a strided column of the weights record (the record holds the field as (B, K, N)), which no
  registered read takes, so the port is left a free input under the weights record.
* `SparseMlaSplit_v1` is its ``CHAINS`` cut in whole key blocks (`sparse_mla_split.split_chain`, `chain_rows.plan`): per head,
  its key blocks in FA3's visit order, each reading the head's q_nope and q_rope whole and its keys as the latent rows that
  hold them (a key a row, the port cut that many ways), the blocks after a chain's first the state the one before returns,
  then the head's end (or, past one split, its column ends and combines). A template's units are place-major, then member,
  then head.

Each reads what the frame hands it (`NodeWindow.take`) and leaves free what is not written in the window.
"""
from __future__ import annotations

from typing import Any

from verity.circuits.codec import descriptor_id
from verity.commitments.windows import interface as IF
from verity.definitions.registry import sparse_mla_split as SMS

from verity_vllm.pipeline import chain_rows as CR
from verity_vllm.query.program_view import spec_base

from .build import Unexpressible
from .node_window import Lowering, NodeWindow, Piece, _family, _Free, _part
from .window_builder import WEIGHTS, Row


def _weight_rows(w: NodeWindow, node: Any, k: int, tid: str, port: str, at: int) -> None:
    """Units ``at`` on, one a member, read argument ``k`` of the weights record as ``port``: each member's one run, the
    members' first words a constant step apart (0 for an argument the batch shares)."""
    words = dict(IF.layout(w.b.fns[tid]).rows)[port]
    M = w.members(node)
    firsts = [w.weight(w.arg(node, k, t), words) for t in range(M)]
    step = firsts[1] - firsts[0] if M > 1 else 0
    if any(f != firsts[0] + t * step for t, f in enumerate(firsts)):
        raise Unexpressible(f"{node.fn.id}: argument {k}'s members read the weights record at {firsts[:4]}..., not a step apart")
    w.b.read(tid, port, at, (M,), firsts[0], (step,))


def _label_at(w: NodeWindow, refs: Any) -> str:
    """The label of the weight the first leaf of ``refs`` lies in."""
    (seg,) = w.segments(refs.slice(0, 1))
    return next(label for label, first, n in w.b.weights if first <= seg[2] < first + n)


def latent_row(w: NodeWindow, j: int, node: Any) -> None:
    """`DsaLatentRow_v1`, the row itself: a member's unit reads its latent coordinates ``kv`` whole, its norm's weights ``w``
    and its position's ``cs`` row, and returns its key row."""
    spec, M = descriptor_id(node.fn), w.members(node)
    fn = _part(spec, spec_base(spec))
    kv = w.take(node, 0, whole=True)
    t, a = w.unit(_family(node), fn, M)
    w.feed(t, "kv", a, kv, (M,))
    _weight_rows(w, node, 1, t, "w", a)
    _weight_rows(w, node, 2, t, "cs", a)
    (_o, W), = IF.layout(fn).outs
    w.out(j, node, [Piece(0, W, Row(t, a, "o0", W))])


def batched_gemm(w: NodeWindow, j: int, node: Any) -> None:
    """`BatchedGemm_v1{B, K, N}` as its ``SLICES``: unit (b, n) of member t, ``a + (b N + n) M + t``, reads head b's K words
    of x whole and ``w[b, n, :]``, and returns output word (b, n). Its ``w`` is read in the record's (B, K, N) order, a strided
    column a unit, which no registered read takes: a free input under the weights record (unless the record's run is the
    port's own order, which a registered read takes)."""
    spec, M = descriptor_id(node.fn), w.members(node)
    B, K, N = (int(node.fn.bindings[x]) for x in ("B", "K", "N"))
    fn = _part(spec, "BatchedGemm_v1", B=1, N=1)
    xs = [w.take(node, 0, b * K, (b + 1) * K, whole=True) for b in range(B)]
    t, a = w.unit(_family(node), fn, B * N * M)
    free = next((x for x in xs if isinstance(x, _Free)), None)
    w.feed(t, "x", a, free or [x[m] for x in xs if not isinstance(x, _Free) for _n in range(N) for m in range(M)], (B, N, M))
    refs = node.args[1].refs
    if len(w.segments(refs.slice(0, K))) == 1:
        w.b.read(t, "w", a, (B, N, M), w.weight(refs, B * N * K), (N * K, K, 0))
    else:
        w.feed(t, "w", a, _Free([("param", WEIGHTS, _label_at(w, refs))]), (B, N, M))
    w.out(j, node, [Piece(0, B * N, Row(t, a, "o0", 1), M, 1)])


#: a ``CHAINS`` place's template family by its kind (`chain_rows.Place.key`), in whole key blocks
_CHAIN = {"block": "SparseMlaHeadBlock_v1", "end": "SparseMlaHeadEnd_v1", "cols": "SparseMlaSplitEndCols_v1",
          "combine": "SparseMlaCombine_v1"}


def _chain_part(spec: str, key: tuple, DV: int) -> Any:
    """The template of a chain place of ``key`` in a row ``spec`` of DV output columns, among the row's unit templates."""
    kind, G = key[0], SMS.combine_cols(DV)
    if kind == "block":
        return _part(spec, _CHAIN[kind], NVIS=key[1], FIRST=key[2])
    if kind == "end":
        return _part(spec, _CHAIN[kind], DV=DV)
    if kind == "cols":
        return _part(spec, _CHAIN[kind], G=G, C=key[1])
    return _part(spec, _CHAIN[kind], NSPLIT=key[1], DV=G)


def _heads(xs: list[Any], M: int, NH: int) -> Any:
    """Per-head takes ``xs`` (each M members' rows, or a `_Free`) as a grid of units (member, head), or the first `_Free`."""
    free = next((x for x in xs if isinstance(x, _Free)), None)
    return free or [xs[h][t] for t in range(M) for h in range(NH)]


def sparse_mla_split(w: NodeWindow, j: int, node: Any) -> None:
    """`SparseMlaSplit_v1` as its ``CHAINS`` cut in whole key blocks (module docstring): each place of a head's chain
    (`chain_rows.plan`) a unit per member and head, unit (place i of its template, member t, head h) ``a + i M NH + t NH + h``;
    a key block reads ``q_nope[h]`` and ``q_rope[h]`` whole, its keys a latent row each, and the reads its place names off the
    units before it; head h's output is its end's (its combines') rows."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    S = node.fn.bindings
    NH, DR, DV = (int(S[x]) for x in ("NH", "DR", "DV"))
    KW, U = DR + DV, M * NH
    places = CR.plan(S)
    qn = _heads([w.take(node, 0, h * DV, (h + 1) * DV, whole=True) for h in range(NH)], M, NH)
    qr = _heads([w.take(node, 1, h * DR, (h + 1) * DR, whole=True) for h in range(NH)], M, NH)
    kv = {p.keys: w.take(node, 2, p.keys[0] * KW, p.keys[1] * KW) for p in places if p.key[0] == "block"}
    of: dict[tuple, list[int]] = {}
    for i, p in enumerate(places):
        of.setdefault(p.key, []).append(i)
    where: dict[int, tuple[str, int, Any]] = {}
    for key, ids in of.items():
        fn = _chain_part(spec, key, DV)
        tid, at = w.unit(fam, fn, len(ids) * U)
        for i, ip in enumerate(ids):
            where[ip] = (tid, at + i * U, fn)
    for ip, p in enumerate(places):
        tid, at, _fn = where[ip]
        if p.key[0] == "block":
            keys = kv[p.keys]
            w.feed(tid, "q_nope", at, qn, (M, NH))
            w.feed(tid, "q_rope", at, qr, (M, NH))
            w.feed(tid, "kv", at, keys if isinstance(keys, _Free) else [keys[t] for t in range(M) for _h in range(NH)], (M, NH))
        reads: dict[str, list[tuple[int, str, Any]]] = {}
        for param, s, out, wp in p.reads:
            reads.setdefault(param, []).append((s or 0, out, wp))
        for param, rs in reads.items():
            rows = []
            for _s, out, wp in sorted(rs):
                wt, wa, wfn = where[wp]
                rows.append((wt, wa, out, dict(IF.layout(wfn).outs)[out]))
            w.feed(tid, param, at, [tuple(Row(wt, wa + t * NH + h, out, n) for wt, wa, out, n in rows)
                                    for t in range(M) for h in range(NH)], (M, NH))
    ends = [ip for ip, p in enumerate(places) if p.key[0] in ("end", "combine")]
    G = DV // len(ends)
    tid, at, _fn = where[ends[0]]
    w.out(j, node, [Piece(h * DV, (h + 1) * DV, Row(tid, at + h, "o0", G), U, NH) for h in range(NH)])


#: the lowerings by family (the node's Definition's name)
FAMILIES: dict[str, Lowering] = {
    "DsaLatentRow": latent_row,
    "BatchedGemm": batched_gemm,
    "SparseMlaSplit": sparse_mla_split,
}

__all__ = ["FAMILIES", "batched_gemm", "latent_row", "sparse_mla_split"]
