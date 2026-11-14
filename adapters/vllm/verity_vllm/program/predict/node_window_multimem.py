"""torch's symmetric-memory all-reduce (`tp_allreduce.AllReduceMultimem_v1`, the prefill's all-reduce past FlashInfer's 42
tokens) lowered into rank 0's node window (`node_window`) as the TP window lowers its site (`pipeline.tp.grain.lower`, read
here, not copied), so the combine units a reader of both windows meets are the same units:

- At coordinate grain (rank 0's partial is a row whose unit is one output word: a block-FP8 linear's GEMM coordinate, the
  embedding shard's column), `grain.lower`'s units for the site: rank 0's words gathered into chunks of K words
  (`AllGather_v1{WORLD=K, N=1}`, K = `grain.chunk` of the site's token row R), one combine unit a chunk
  (`tp_combine.row_binding`'s, `AllReduceMultimem_v1{SIZE=K}` where every owner's tree is one), and one
  `AllGather_v1{WORLD=R/K, N=K}` a token row gathering its chunks (``output``), which a whole read of the row reuses
  (`NodeWindow.wholes`). Beside them, one `AllGather_v1{WORLD=G/K, N=K}` a column block of G = the widest halving of R
  within ``grain.norm`` (`node_window._cols`), the norms' apply block: a norm reads a block as one row, as it reads the
  decode's COLS all-reduce (`node_window_dense.cols`), since one template's port is cut one way (`NodeWindow.chunks`).
- At row grain (rank 0's partial is a row whose own Definition is its unit: the MoE layer's residual add), as G-column
  blocks of each token row, one combine unit a block (`AllReduceMultimem_v1{SIZE=G}`) reading rank 0's block (the
  residual add's own COLS block) and the other ranks'. `grain.lower` reads one combine unit a token row of R words here,
  whose row a norm's G-column block would cut.

A combine unit reads ``parts`` cut WORLD ways (`NodeWindow.feed_rows`): row 0, rank 0's own chunk or block, wired; row r,
rank r's, its ``peers_*`` partial, a free input listed by name. Where rank 0's partial is itself free (its producer has no
lowering: the embedding shard), there is no chunk gather and row 0 is free too, listed under that family, as `grain.lower`'s
gathers would read rows the window doesn't have. The combine units are in position order (chunk or block i at unit
``first + i``), `grain.lower`'s order.

Refused by name (`Unexpressible`) beyond the frame's: a batch of all-reduces, owners of more than one tree (their row units
differ by row), a partial that is an input or of no grain `grain.lower` takes, and a token row a block or chunk doesn't
divide.
"""
from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import Any

from verity.circuits.defs import bind
from verity.circuits.refs import runs
from verity.definitions.registry import b1_tp2 as B1T
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry import tp_combine as TC
from verity_vllm.pipeline.tp import grain as GR

from .build import Unexpressible
from .node_window import GATHER, Lowering, NodeWindow, Piece, _cols, _Free, _stands
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


def _own(w: NodeWindow, j: int, node: Any, size: int) -> list[tuple[Any, int, int]]:
    """Rank 0's partial (``parts`` leaves [0, size)) as `grain.lower` takes it: runs ``(producer Definition, lo, hi)`` of a
    producer member's leaves, in order."""
    out: list[tuple[Any, int, int]] = []
    for space, idx, lo, hi in runs(node.args[0].refs.slice(0, size)):
        if space != "n":
            raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): rank 0's partial reads input {w.params[idx]}, not a node")
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
    """`AllReduceMultimem_v1{WORLD, SIZE}` at rank 0 (module docstring)."""
    if node.form != "call":
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}) is a {node.form} of all-reduces; the node window lowers a call")
    S = node.fn.bindings
    world, size = int(S["WORLD"]), int(S["SIZE"])
    if len(set(map(str, TA.multimem_trees(S["TREES"], world)))) != 1:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): its owners reduce by more than one tree, so its row units "
                            "differ by row")
    where = f"{w.name}: node {j} ({node.fn.id})"
    try:
        low = GR.lower(node.fn, [_own(w, j, node, size)] * world, reads(w, j), where)
    except ValueError as e:
        raise Unexpressible(str(e)) from None
    peers = [i * size for i in range(1, world)]
    if low.grain == GR.COORD and _wide(w, node, size, low.chunk):
        _row(w, j, node, low, size, peers, low.row)
    elif low.grain == GR.COORD:
        _coordinate(w, j, node, low, size, peers)
    else:
        _row(w, j, node, low, size, peers)


def _wide(w: NodeWindow, node: Any, size: int, K: int) -> bool:
    """Whether rank 0's partial is written as rows wider than a chunk of K words (the embedding shard's column blocks,
    `node_window_embed`), which a chunk would cut: then the combine reads those rows as blocks, as at row grain."""
    probe = w.take(node, 0, 0, size)
    return not isinstance(probe, _Free) and not _stands(probe[0]) and any(r.words > K for r in probe[0])


def _coordinate(w: NodeWindow, j: int, node: Any, low: GR.Lowering, size: int, peers: Sequence[int]) -> None:
    R, K = low.row, low.chunk
    T, C = size // R, R // K
    G = _cols(R, w.b.grain.norm)
    if size % R or R % K or G % K or len(low.combines) != 1:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): token rows of {R} words, chunks of {K}, blocks of {G}, "
                            f"{len(low.combines)} combine runs; a chunk divides a block and a row, one combine template")
    probe = w.take(node, 0, 0, size)
    mine: Grid | _Free
    if isinstance(probe, _Free) or _stands(probe[0]):
        mine = _blocks(w, node, 0, 0, K, T * C)
    else:
        own = _chunks(w, node, probe[0], K)
        tg, ag = w.unit(GATHER, low.gather, T * C)
        w.feed(tg, "parts", ag, own, (T, C))
        mine = []
        for p, rows in enumerate(own):
            r = Row(tg, ag + p, "o0", K)
            w.wholes.setdefault(rows, r)
            mine.append((r,))
    fn, _n = low.combines[0]
    tc, ac = w.unit(FAMILY, fn, T * C)
    w.feed_rows(tc, low.parts, ac, [mine, *(_blocks(w, node, 0, lo, K, T * C) for lo in peers)], (T, C))
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


def _row(w: NodeWindow, j: int, node: Any, low: GR.Lowering, size: int, peers: Sequence[int], R: int | None = None) -> None:
    """Blocks of G columns of each token row of R words (the producers' row at row grain, else the site's token row), one
    combine unit a block reading rank 0's block and the other ranks'."""
    R = low.width if R is None else R
    G = _cols(R, w.b.grain.norm)
    if size % R or R % G:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): rows of {R} words in blocks of {G}; a block divides a row")
    n = size // G
    defn, statics = TC.row_binding(node.fn.defn, dict(node.fn.statics), n, 0)
    tc, ac = w.unit(FAMILY, bind(defn, **statics), n)
    w.feed_rows(tc, low.parts, ac, [_blocks(w, node, 0, lo, G, n) for lo in (0, *peers)], (size // R, R // G))
    w.out(j, node, [Piece(0, size, Row(tc, ac, "o0", G), 1, 1)])


#: the lowering by family (the node's Definition's name), beside `node_window.FAMILIES` and `node_window_dense.FAMILIES`
FAMILIES: dict[str, Lowering] = {FAMILY: multimem}

__all__ = ["FAMILIES", "FAMILY", "multimem", "reads"]
