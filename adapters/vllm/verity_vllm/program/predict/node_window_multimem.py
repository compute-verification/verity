"""torch's symmetric-memory all-reduce (`tp_allreduce.AllReduceMultimem_v1`, the prefill's all-reduce past FlashInfer's 42
tokens) lowered into a rank's node window (`node_window`) as the TP window lowers its site (`pipeline.tp.grain.lower`, read
here, not copied), so the combine units a reader of both windows meets are the same units:

- At coordinate grain (the rank's partial is a row whose unit is one output word: a block-FP8 linear's GEMM coordinate, the
  embedding shard's column), `grain.lower`'s units for the site: the rank's words gathered into chunks of K words
  (`AllGather_v1{WORLD=K, N=1}`, K = `grain.chunk` of the site's token row R), one combine unit a chunk
  (`tp_combine.row_binding`'s, `AllReduceMultimem_v1{SIZE=K}` where every owner's tree is one), and one
  `AllGather_v1{WORLD=R/K, N=K}` a token row gathering its chunks (``output``), which a whole read of the row reuses
  (`NodeWindow.wholes`). Beside them, one `AllGather_v1{WORLD=G/K, N=K}` a column block of G = the widest halving of R
  within ``grain.norm`` (`node_window._cols`), the norms' apply block: a norm reads a block as one row, as it reads the
  decode's COLS all-reduce (`node_window_dense.cols`), since one template's port is cut one way (`NodeWindow.chunks`).
- At row grain (the rank's partial is a row whose own Definition is its unit: the MoE layer's residual add), as G-column
  blocks of each token row, one combine unit a block (`AllReduceMultimem_v1{SIZE=G}`) reading the rank's block (the
  residual add's own COLS block) and the other ranks'. `grain.lower` reads one combine unit a token row of R words here,
  whose row a norm's G-column block would cut.

A combine unit reads ``parts`` cut WORLD ways (`NodeWindow.feed_rows`), row i rank i's: the window's own rank's (`mine`:
the one segment that reads no ``peers_*`` input) its own chunk or block, wired; another rank's, its ``peers_*`` partial, a
free input listed by name. Where the rank's partial is itself free (its producer has no lowering: the embedding shard),
there is no chunk gather and its row is free too, listed under that family, as `grain.lower`'s
gathers would read rows the window doesn't have. The combine units are in position order (chunk or block i at unit
``first + i``), `grain.lower`'s order.

Refused by name (`Unexpressible`) beyond the frame's: a batch of all-reduces, owners of more than one tree (their row units
differ by row), a partial that is an input or of no grain `grain.lower` takes, and a token row a block or chunk doesn't
divide.
"""
from __future__ import annotations

import weakref
from typing import Any

from verity.circuits.defs import bind
from verity.circuits.refs import runs
from verity.definitions.registry import b1_tp2 as B1T
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry import tp_combine as TC
from verity_vllm.pipeline.tp import grain as GR

from .build import Unexpressible
from .node_window import GATHER, PEERS, Lowering, NodeWindow, Piece, _cols, _Free, _stands
from .node_window_dense import _blocks
from .window_builder import Row

FAMILY = "AllReduceMultimem"
Grid = list[tuple[Row, ...]]

#: each window's reads of each node's output, ``(lo, hi)`` leaves, a batch reader's member by member
_READS: weakref.WeakKeyDictionary[NodeWindow, dict[int, list[tuple[int, int]]]] = weakref.WeakKeyDictionary()


def reads(w: NodeWindow, j: int) -> list[tuple[int, int]]:
    """The leaves ``[lo, hi)`` of node ``j``'s output each reader reads, a batch reader's member by member (what
    `pipeline.tp.wires` hands `grain.lower` as a site's reads)."""
    got = _READS.get(w)
    if got is None:
        got = _READS[w] = {}
        for node in w.fn.body.nodes:
            for k, arg in enumerate(node.args):
                rs = runs(arg.refs)
                if not any(r[0] == "n" for r in rs):
                    continue
                L = len(arg.refs) // node.n if node.form == "batch" and node.axes[k] == 0 else 0
                at = 0
                for space, idx, lo, hi in rs:
                    a = lo
                    while space == "n" and a < hi:
                        e = min(hi, a + L - (at + a - lo) % L) if L else hi
                        got.setdefault(idx, []).append((a, e))
                        a = e
                    at += hi - lo
    return got.get(j, [])


def mine(w: NodeWindow, j: int, node: Any, size: int, world: int) -> int:
    """The window's rank at all-reduce node ``j``: the one segment of ``parts`` (leaves [i·size, (i+1)·size)) that reads no
    ``peers_*`` input, its own partial; every other segment reads only peers. Refused by name otherwise."""
    parts = node.args[0].refs
    own = [i for i in range(world) if not any(s == "p" and w.params[x].startswith(PEERS)
                                              for s, x, _lo, _hi in runs(parts.slice(i * size, (i + 1) * size)))]
    if len(own) != 1:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): segments {own} of its {world} read no peer's partial; "
                            "one is the rank's own")
    return own[0]


def _own(w: NodeWindow, j: int, node: Any, size: int, me: int) -> list[tuple[Any, int, int]]:
    """The rank's own partial (``parts`` leaves [me·size, (me+1)·size)) as `grain.lower` takes it: runs ``(producer
    Definition, lo, hi)`` of a producer member's leaves, in order."""
    out: list[tuple[Any, int, int]] = []
    for space, idx, lo, hi in runs(node.args[0].refs.slice(me * size, (me + 1) * size)):
        if space != "n":
            raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): rank {me}'s partial reads input {w.params[idx]}, not a node")
        L = w.meta[idx][1]
        pfn = w.fn.body.nodes[idx].fn
        while lo < hi:
            m = lo // L
            e = min(hi, (m + 1) * L)
            out.append((pfn, lo - m * L, e - m * L))
            lo = e
    return out


def _chunks(w: NodeWindow, node: Any, rows: tuple[Row, ...], K: int) -> Grid:
    """Rank 0's partial's writer rows ``rows`` as K-word chunks, one entry each."""
    out: Grid = []
    cur: list[Row] = []
    words = 0
    for r in rows:
        cur.append(r)
        words += r.words
        if words == K:
            out.append(tuple(cur))
            cur, words = [], 0
        elif words > K:
            raise Unexpressible(f"{w.name}: a writer row of {r.words} words crosses a {K}-word chunk of {node.fn.id}")
    return out


def multimem(w: NodeWindow, j: int, node: Any) -> None:
    """`AllReduceMultimem_v1{WORLD, SIZE}` at the window's rank (module docstring)."""
    if node.form != "call":
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}) is a {node.form} of all-reduces; the node window lowers a call")
    S = node.fn.bindings
    world, size = int(S["WORLD"]), int(S["SIZE"])
    if len(set(map(str, TA.multimem_trees(S["TREES"], world)))) != 1:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): its owners reduce by more than one tree, so its row units "
                            "differ by row")
    where = f"{w.name}: node {j} ({node.fn.id})"
    me = mine(w, j, node, size, world)
    try:
        low = GR.lower(node.fn, [_own(w, j, node, size, me)] * world, reads(w, j), where)
    except ValueError as e:
        raise Unexpressible(str(e)) from None
    if low.grain == GR.COORD and _wide(w, node, size, low.chunk, me):
        _row(w, j, node, low, size, world, low.row)
    elif low.grain == GR.COORD:
        _coordinate(w, j, node, low, size, world, me)
    else:
        _row(w, j, node, low, size, world)


def _wide(w: NodeWindow, node: Any, size: int, K: int, me: int) -> bool:
    """Whether the rank's own partial is written as rows wider than a chunk of K words (the embedding shard's column
    blocks, `node_window_embed`), which a chunk would cut: then the combine reads those rows as blocks, as at row grain."""
    probe = w.take(node, 0, me * size, (me + 1) * size)
    return not isinstance(probe, _Free) and not _stands(probe[0]) and any(r.words > K for r in probe[0])


def _coordinate(w: NodeWindow, j: int, node: Any, low: GR.Lowering, size: int, world: int, me: int) -> None:
    R, K = low.row, low.chunk
    T, C = size // R, R // K
    G = _cols(R, w.b.grain.norm)
    if size % R or R % K or G % K or len(low.combines) != 1:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): token rows of {R} words, chunks of {K}, blocks of {G}, "
                            f"{len(low.combines)} combine runs; a chunk divides a block and a row, one combine template")
    probe = w.take(node, 0, me * size, (me + 1) * size)
    own: Grid | _Free
    if isinstance(probe, _Free) or _stands(probe[0]):
        own = _blocks(w, node, 0, me * size, K, T * C)
    else:
        chunks = _chunks(w, node, probe[0], K)
        tg, ag = w.unit(GATHER, low.gather, T * C)
        w.feed(tg, "parts", ag, chunks, (T, C))
        own = []
        for p, rows in enumerate(chunks):
            r = Row(tg, ag + p, "o0", K)
            w.wholes.setdefault(rows, r)
            own.append((r,))
    fn, _n = low.combines[0]
    tc, ac = w.unit(FAMILY, fn, T * C)
    w.feed_rows(tc, low.parts, ac, [own if i == me else _blocks(w, node, 0, i * size, K, T * C) for i in range(world)],
                (T, C))
    chunk = [Row(tc, ac + p, "o0", K) for p in range(T * C)]
    if low.output is not None:
        to, ao = w.unit(GATHER, low.output, T)
        rows_of = [tuple(chunk[t * C:(t + 1) * C]) for t in range(T)]
        w.feed(to, "parts", ao, rows_of, (T,))
        for t, rows in enumerate(rows_of):
            w.wholes.setdefault(rows, Row(to, ao + t, "o0", R))
    if G != K:
        k, B = G // K, R // G
        tb, ab = w.unit(GATHER, bind(B1T.AllGather, WORLD=k, N=K), T * B)
        blocks = [tuple(chunk[t * C + b * k:t * C + (b + 1) * k]) for t in range(T) for b in range(B)]
        w.feed(tb, "parts", ab, blocks, (T, B))
        for i, rows in enumerate(blocks):
            w.chunks.setdefault(rows[0], (rows, Row(tb, ab + i, "o0", G)))
    w.out(j, node, [Piece(0, size, chunk[0], 1, 1)])


def _row(w: NodeWindow, j: int, node: Any, low: GR.Lowering, size: int, world: int, R: int | None = None) -> None:
    """Blocks of G columns of each token row of R words (the producers' row at row grain, else the site's token row), one
    combine unit a block reading each rank's block, row i rank i's."""
    R = low.width if R is None else R
    G = _cols(R, w.b.grain.norm)
    if size % R or R % G:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): rows of {R} words in blocks of {G}; a block divides a row")
    n = size // G
    defn, statics = TC.row_binding(node.fn.defn, dict(node.fn.statics), n, 0)
    tc, ac = w.unit(FAMILY, bind(defn, **statics), n)
    w.feed_rows(tc, low.parts, ac, [_blocks(w, node, 0, i * size, G, n) for i in range(world)], (size // R, R // G))
    w.out(j, node, [Piece(0, size, Row(tc, ac, "o0", G), 1, 1)])


#: the lowering by family (the node's Definition's name), beside `node_window.FAMILIES` and `node_window_dense.FAMILIES`
FAMILIES: dict[str, Lowering] = {FAMILY: multimem}

__all__ = ["FAMILIES", "FAMILY", "mine", "multimem", "reads"]
