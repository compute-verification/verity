"""The dense families' lowerings into the node window (`node_window`), each as the split `serving_templates` registers for it
and over only the templates `serving_templates.unit_templates` lists for its row:

- `silu_mul` (`SiluMul_v2`) and `rope_head` (`RoPEHeadGptJ_v2`): the row itself, a member reading its operand whole,
  gathered from the GEMM coordinates that hold it;
- `cols`, a ``COLS`` row (`ResidualAdd_v1`, the all-reduce `AllReduceMnnvl_v1`, the top-k sum `MoeSum_v1`) as its column
  blocks, W = the widest halving of its N within ``grain.norm``, the width of the norms' apply blocks, so a norm reads
  them with no gather: unit (a, t) reads columns [aW, (a+1)W) of each of a parameter's outer rows (an all-reduce's ranks, a
  top-k sum's experts) as one row of the port cut that many ways, rank 0's own wired and another rank's free
  (`NodeWindow.feed_rows`);
- `allreduce_norm`, the fused all-reduce, residual add and RMSNorm (`AllReduceRmsNormMnnvl_v1`) as its ``PARTS``: the
  all-reduce and residual add in column blocks of G (as `cols`), then inside a decoder layer the sum of squares and the
  apply blocks of G, outside every layer (``FINAL``) the whole norm;
- `lm_head` (`Gemm_v2`) and `all_gather` (`AllGather_v1`) as their ``SLICES``: the LM head's N one-word slices, each
  reading x whole and its weight row, and the all-gather's WORLD shards, rank 0's gathered from the LM head's slices and
  the other ranks' free.

Refused by name (`Unexpressible`) beyond the frame's refusals: a ``Gemm`` family row other than `Gemm_v2`. Not lowered
here, so their rows stay free unless their module's lowering is beside these: the vocabulary shard's embedding
(`EmbeddingShard_v1`, `node_window_embed`), whose ``SLICES`` unit reads a strided column of the table, which a weights
read of contiguous rows doesn't take; the token select (`TokenSelect_v4`), whose one unit's 309,760-byte logits row
passes a proof block, and the stop check (`StopCheck_v1`), both `node_window_select`'s.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.commitments.windows import interface as IF
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.query.program_view import spec_base

from .build import Unexpressible
from .node_window import Lowering, NodeWindow, Piece, _cols, _family, _Free, _part
from .window_builder import Row


def _blocks(w: NodeWindow, node: Any, k: int, lo: int, W: int, A: int) -> list[tuple[Row, ...]] | _Free:
    """Argument ``k``'s leaves [lo + aW, lo + (a+1)W) of each member as one row (gathered where several), block-major
    (entry a·M + t); a `_Free` where any is free, before any gather is made."""
    probe = w.take(node, k, lo, lo + A * W)
    if isinstance(probe, _Free):
        return probe
    grid: list[tuple[Row, ...]] = []
    for a in range(A):
        got = w.take(node, k, lo + a * W, lo + (a + 1) * W, whole=True)
        if isinstance(got, _Free):
            return got
        grid += got
    return grid


def _gathered(w: NodeWindow, row: Row, A: int, M: int) -> list[tuple[Row, ...]]:
    """Each member's row of the A column blocks ``row`` starts (block-major, unit a·M + t) gathered whole."""
    return [(r,) for r in w._whole([tuple(Row(row.tid, row.unit + a * M + t, row.out, row.words) for a in range(A))
                                    for t in range(M)])]


def silu_mul(w: NodeWindow, j: int, node: Any) -> None:
    """SwiGLU's `SiluMul_v2{I}` as the row itself: a member reads its 2I gate and up words whole and returns its I."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    fn, half = _part(spec, spec_base(spec)), node.fn.bindings["I"]
    gu = w.take(node, 0, whole=True)
    t, a = w.unit(fam, fn, M)
    w.feed(t, "gu", a, gu, (M,))
    w.out(j, node, [Piece(0, half, Row(t, a, "o0", half))])


def rope_head(w: NodeWindow, j: int, node: Any) -> None:
    """A head's rotary embedding (`RoPEHeadGptJ_v2{D}`) as the row itself: a member reads its D rope columns whole and its
    position's D cos/sin words of the weights record."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    fn, D = _part(spec, spec_base(spec)), node.fn.bindings["D"]
    x = w.take(node, 0, whole=True)
    cs = [w.weight(w.arg(node, 1, t), D) for t in range(M)]
    step = cs[1] - cs[0] if M > 1 else 0
    t, a = w.unit(fam, fn, M)
    w.feed(t, "x", a, x, (M,))
    if all(c == cs[0] + i * step for i, c in enumerate(cs)):
        w.b.read(t, "cs", a, (M,), cs[0], (step,))
    else:
        for i, c in enumerate(cs):
            w.b.read(t, "cs", a + i, (1,), c, (0,))
    w.out(j, node, [Piece(0, D, Row(t, a, "o0", D))])


def cols(w: NodeWindow, j: int, node: Any) -> None:
    """A ``COLS`` row (module docstring) as N / W column blocks, block-major (unit a·M + t), each returning its W columns."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    base = spec_base(spec)
    axis = ST.COLS[base]
    N = node.fn.bindings[axis]
    W = _cols(N, w.b.grain.norm)
    A = N // W
    fn = _part(spec, base, **{axis: W})
    ports = IF.layout(fn).rows
    grids = [[_blocks(w, node, k, i * N, W, A) for i in range(words // W)] for k, (_p, words) in enumerate(ports)]
    t, a = w.unit(fam, fn, A * M)
    for (port, _), g in zip(ports, grids):
        w.feed_rows(t, port, a, g, (A, M))
    w.out(j, node, [Piece(0, N, Row(t, a, "o0", W), M, 1)])


def allreduce_norm(w: NodeWindow, j: int, node: Any) -> None:
    """`AllReduceRmsNormMnnvl_v1` as its ``PARTS`` (module docstring): N / G units of `AllReduceResidualMnnvl_v1{N=G}`,
    block-major, each reading its columns of the WORLD ranks' partials (``parts`` cut WORLD ways) and of the residual,
    returning r's; inside a decoder layer (its weight's label, `serving_templates.in_layer`), a member's
    `RmsNormMnnvlSumSq_v1` reading r whole and N / G `RmsNormMnnvlApply_v1{G}` reading r's block, its G weights and the
    sum; outside every layer (``FINAL``), a member's `RmsNormMnnvl_v1` reading r whole and the weights. G is the widest
    halving of N within ``grain.norm`` (`node_window._cols`)."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    base, N, world = spec_base(spec), node.fn.bindings["N"], node.fn.bindings["WORLD"]
    G = _cols(N, w.b.grain.norm)
    A = N // G
    wlo, label = w.weight(node.args[2].refs, N), w.label(node.args[2].refs, N)
    (rfam, _, _), *rest = (ST.PARTS if ST.in_layer(label) else ST.FINAL)[base]
    parts = [_blocks(w, node, 0, i * N, G, A) for i in range(world)]
    res = _blocks(w, node, 1, 0, G, A)
    tr, ar = w.unit(fam, _part(spec, rfam, N=G), A * M)
    w.feed_rows(tr, "parts", ar, parts, (A, M))
    w.feed(tr, "res", ar, res, (A, M))
    r = Piece(N, 2 * N, Row(tr, ar, "o0", G), M, 1)
    rs = _gathered(w, r.row, A, M)
    if len(rest) == 1:
        tn, an = w.unit(fam, _part(spec, rest[0][0]), M)
        w.feed(tn, "r", an, rs, (M,))
        w.b.read(tn, "w", an, (M,), wlo, (0,))
        w.out(j, node, [Piece(0, N, Row(tn, an, "o0", N)), r])
        return
    (sfam, _, _), (afam, _, _) = rest
    sumsq, apply = _part(spec, sfam), _part(spec, afam, G=G)
    ts, as_ = w.unit(fam, sumsq, M)
    w.feed(ts, "r", as_, rs, (M,))
    ta, aa = w.unit(fam, apply, A * M)
    w.b.feed(ta, "r", aa, A * M, (r.row,), ((A * M, 1),))
    w.b.read(ta, "w", aa, (A, M), wlo, (G, 0))
    w.b.feed(ta, "rs", aa, A * M, (Row(ts, as_, "o0", dict(IF.layout(apply).rows)["rs"]),), ((A, 0), (M, 1)))
    w.out(j, node, [Piece(0, N, Row(ta, aa, "o0", G), M, 1), r])


def lm_head(w: NodeWindow, j: int, node: Any) -> None:
    """A GEMM of one token row into the weight's N rows (`Gemm_v2{K, N}`, the LM head) as its ``SLICES``: N units of
    `Gemm_v2{K, N=1}`, slice-major (unit n·M + t), each reading x whole and its weight row and returning one word."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    base = spec_base(spec)
    if base != "Gemm_v2":
        raise Unexpressible(f"{spec}: a Gemm row the LM head's lowering doesn't take ({base}, not Gemm_v2)")
    K, N = node.fn.bindings["K"], node.fn.bindings["N"]
    x = w.take(node, 0, whole=True)
    lo = w.weight(node.args[1].refs, N * K)
    t, a = w.unit(fam, _part(spec, base, N=1), N * M)
    w.feed(t, "x", a, x if isinstance(x, _Free) else [x[i % M] for i in range(N * M)], (N, M))
    w.b.read(t, "w", a, (N, M), lo, (K, 0))
    w.out(j, node, [Piece(0, N, Row(t, a, "o0", 1), M, 1)])


def all_gather(w: NodeWindow, j: int, node: Any) -> None:
    """The all-gather of the ranks' shards (`AllGather_v1{WORLD, N}`) as its ``SLICES``: WORLD units of
    `AllGather_v1{WORLD=1, N}`, rank-major (unit r·M + t), unit r reading rank r's shard whole (another rank's free) and
    returning it."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    world, N = node.fn.bindings["WORLD"], node.fn.bindings["N"]
    shards: Sequence[list[tuple[Row, ...]] | _Free] = [w.take(node, 0, r * N, (r + 1) * N, whole=True) for r in range(world)]
    t, a = w.unit(fam, _part(spec, spec_base(spec), WORLD=1), world * M)
    for r, g in enumerate(shards):
        w.feed(t, "parts", a + r * M, g, (M,))
    w.out(j, node, [Piece(0, world * N, Row(t, a, "o0", N), M, 1)])


#: the dense half's lowerings by family (the node's Definition's name), beside `node_window.FAMILIES`
FAMILIES: dict[str, Lowering] = {
    "SiluMul": silu_mul,
    "RoPEHeadGptJ": rope_head,
    "ResidualAdd": cols,
    "AllReduceMnnvl": cols,
    "MoeSum": cols,
    "AllReduceRmsNormMnnvl": allreduce_norm,
    "Gemm": lm_head,
    "AllGather": all_gather,
}

__all__ = ["FAMILIES", "all_gather", "allreduce_norm", "cols", "lm_head", "rope_head", "silu_mul"]
