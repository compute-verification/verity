"""The node window's lowerings (`node_window`) of GLM-5.3's DSA attention and MoE layer, the families `FAMILIES` names.

DSA attention:

* `DsaLatentRow_v1` is the row itself, a member a unit: it reads its 576 latent coordinates of q_a whole (through the frame's
  gather), its norm's weights and its position's cos/sin row, and returns the key row the attention reads (`HISTORY`'s writer).
* `BatchedGemm_v1{B, K, N}` is its ``SLICES``: unit (b, n) reads head b's K words of x whole and returns output word (b, n).
  Its weight ``w[b, n, :]`` is a strided column of the weights record (the record holds the field as (B, K, N)), which no
  read of the record takes, so the port reads the rank's entry of the weight in the port's order (`entry`, ``/bnk``).
* `SparseMlaSplit_v1` is its ``CHAINS`` cut in whole key blocks (`sparse_mla_split.split_chain`, `chain_rows.plan`): per head,
  its key blocks in FA3's visit order, each reading the head's q_nope and q_rope whole and its keys as the latent rows that
  hold them (a key a row, the port cut that many ways), the blocks after a chain's first the state the one before returns,
  then the head's end (or, past one split, its column ends and combines). A template's units are place-major, then member,
  then head.
* The MoE layer: the router gate's GEMM (`CublasSplitKGemmF32Out_v1`, `LlBf16DotprodGemm_v1`) is its ``SLICES`` along N;
  `MoeRouterSigmoidBias_v1` its ``SLOTS``, an output row a slot's weight or expert id; `Fp8GroupQuantUe8m0_v1` the row
  itself; and `MoeExpertFp8Gemm(W)_v1` cut per expert, its blocks apart (`expert_gemm`): block GEMMs reading their block's
  rows and scale row from table reads keyed by the pair's expert id.

Each reads what the frame hands it (`NodeWindow.take`) and leaves free what is not written in the window.
"""
from __future__ import annotations

from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.refs import Affine, runs
from verity.commitments.windows import interface as IF
from verity.definitions.registry import sparse_mla_split as SMS

from verity_vllm.check.weights_of_record import weight_param
from verity_vllm.pipeline import chain_rows as CR
from verity_vllm.pipeline import expert_rows as XR
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.query.program_view import spec_base, spec_statics

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
    of x whole and ``w[b, n, :]``, and returns output word (b, n). Its ``w`` is the weights record's run where the record
    holds the port's own order, else the record's (B, K, N) order, a strided column a unit, which no read of the record
    takes: then the rank's entry of the weight in the port's order (`entry`)."""
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
        w.b.read(t, "w", a, (B, N, M), 0, (N * K, K, 0), param=entry(w, refs, K))
    w.out(j, node, [Piece(0, B * N, Row(t, a, "o0", 1), M, 1)])


def entry(w: NodeWindow, refs: Any, K: int) -> str:
    """The rank's registered entry of the weight ``refs`` reads as rows of ``K`` words in ``refs``' own order, a batched
    GEMM's ``w`` (`batched_gemm`): ``weights/<label>@<K>/bnk`` (`node_window_embed.split` keeps it with the tables), its
    words the record's at ``refs`` in turn (`NodeWindow.entries` keeps them). Refused where ``refs`` leave the one weight,
    or another read gave the name other words."""
    label = _label_at(w, refs)
    first, n = next((f, k) for x, f, k in w.b.weights if x == label)
    rs = runs(refs)
    if any(s != "p" or w.params[i] != WEIGHTS for s, i, _lo, _hi in rs):
        raise Unexpressible(f"{label}: a batched GEMM's weight reads past the weight")
    lo, hi = min(r[2] for r in rs), max(r[3] for r in rs)
    (s,) = w.segments(Affine("p", rs[0][1], lo, hi - lo))
    if not first <= s[2] < s[3] <= first + n:
        raise Unexpressible(f"{label}: a batched GEMM's weight reads past the weight")
    param = f"{weight_param(label, K)}/bnk"
    prev = w.entries.setdefault(param, refs)
    if prev is not refs and runs(prev) != rs:
        raise Unexpressible(f"{param}: read at two layouts of the weights record")
    w.b.params[param] = len(refs)
    return param


def row_gemm(w: NodeWindow, j: int, node: Any) -> None:
    """A GEMM of one token row into W's N rows (`CublasSplitKGemmF32Out_v1`, `LlBf16DotprodGemm_v1`: the MoE router gate) as
    its ``SLICES`` along N: unit (n, t), ``a + n M + t``, reads member t's x whole and W's row n, and returns output element n."""
    spec, M = descriptor_id(node.fn), w.members(node)
    K, N = int(node.fn.bindings["K"]), int(node.fn.bindings["N"])
    fn = _part(spec, spec_base(spec), N=1)
    x = w.take(node, 0, whole=True)
    t, a = w.unit(_family(node), fn, N * M)
    w.feed(t, "x", a, x if isinstance(x, _Free) else [x[m] for _n in range(N) for m in range(M)], (N, M))
    w.b.read(t, "w", a, (N, M), w.weight(node.args[1].refs, N * K), (K, 0))
    (_o, W), = IF.layout(fn).outs
    w.out(j, node, [Piece(0, N * W, Row(t, a, "o0", W), M, 1)])


def router_slots(w: NodeWindow, j: int, node: Any) -> None:
    """`MoeRouterSigmoidBias_v1` as its ``SLOTS`` (`MoeRouterSigmoidBiasSlots_v1`): a member's unit reads its logits whole
    and the bias, and returns each slot's weight and expert id as an output row of its own, so a reader of one slot reads
    that row whole."""
    spec, M = descriptor_id(node.fn), w.members(node)
    fn = _part(spec, ST.SLOTS[spec_base(spec)])
    logits = w.take(node, 0, whole=True)
    t, a = w.unit(_family(node), fn, M)
    w.feed(t, "logits", a, logits, (M,))
    _weight_rows(w, node, 1, t, "bias", a)
    outs, at = [], 0
    for o, n in IF.layout(fn).outs:
        outs.append(Piece(at, at + n, Row(t, a, o, n)))
        at += n
    w.out(j, node, outs)


def group_quant(w: NodeWindow, j: int, node: Any) -> None:
    """`Fp8GroupQuantUe8m0_v1{K, G}` as a node of its own, the row itself: a member's unit reads x whole and returns its
    quantised row q and its scales s."""
    spec, M = descriptor_id(node.fn), w.members(node)
    fn = _part(spec, spec_base(spec))
    x = w.take(node, 0, whole=True)
    t, a = w.unit(_family(node), fn, M)
    w.feed(t, "x", a, x, (M,))
    (_q, q), (_s, s) = IF.layout(fn).outs
    w.out(j, node, [Piece(0, q, Row(t, a, "o0", q)), Piece(q, q + s, Row(t, a, "o1", s))])


#: the most u16 words of a block's rows one read of the per-expert cut returns: a row family's blocks are its block GEMMs of the
#: most rows within it
BLOCK_WORDS = 4096


def _block_rows(spec: str, K: int) -> int:
    """The rows n of a block of the ``EXPERTS`` row ``spec``: the most among its block GEMMs' whose rows, n K / 2 words, fit
    `BLOCK_WORDS`."""
    fam = ST.EXPERTS[spec_base(spec)]
    fit = sorted(int(spec_statics(t)["N"]) for t, _n in ST.unit_templates(spec)
                 if spec_base(t) == fam and int(spec_statics(t)["N"]) * K // 2 <= BLOCK_WORDS)
    if not fit:
        raise Unexpressible(f"{spec}: no {fam} block of rows within {BLOCK_WORDS} words")
    return fit[-1]


def _table_reads(w: NodeWindow, tid: str, at: int, units: int, lo: int, label: str, H: int, E: int, block: int) -> None:
    """Units ``[at, at + units)`` of the read ``tid`` read block ``block`` of weight ``label`` (from word ``lo`` of the weights
    record) as a table of E rows of H words, one an expert: a registered parameter of its own (`expert_rows.table_param`),
    its root and domain id table reads, and its row, salt and path an opening at the key ``tok`` that the verdict ``o1``
    holds. The one place the per-expert cut touches the window's tables, table reads and openings."""
    param = XR.table_param(label, H, block)
    w.b.params[param] = H * E
    w.b.tables.setdefault((lo, H, block), IF.Table(param, param, 0, H, E))
    w.b.roots += [IF.TableRead(tid, "root", param, "root", at, units), IF.TableRead(tid, "domain", param, "domain", at, units)]
    w.b.openings.append(IF.TableOpening(tid, param, "tok", ("row", "salt", "path"), "o1", at, units))


def _slots(rows: list[tuple[Row, ...]] | _Free, M: int) -> tuple[list[Row], int]:
    """A pair row's members as (token, slot), member ``t TOPK + k`` the slot-k row of router unit t, read off each member's
    one router row ``rows``: each slot's row of the first token and the unit step between tokens; none where they are free."""
    if isinstance(rows, _Free):
        return [], 0
    by_out: dict[str, Row] = {}
    for r in rows:
        by_out.setdefault(r[0].out, r[0])
    first, K = list(by_out.values()), len(by_out)
    step = rows[K][0].unit - first[0].unit if M > K else 0
    if M % K or any(rows[m] != (Row(first[0].tid, first[m % K].unit + m // K * step, first[m % K].out, first[0].words),)
                    for m in range(M)):
        raise Unexpressible(f"{first[0].tid}: a pair row's {M} members aren't its tokens' slots in turn")
    return first, step


def expert_gemm(w: NodeWindow, j: int, node: Any) -> None:
    """`MoeExpertFp8Gemm(W)_v1` cut per expert (`serving_templates.EXPERTS`, `expert_rows`), its blocks apart: a pair (member
    ``t TOPK + k``, token t's slot k) is N / n block GEMMs of n rows (`_block_rows`), each reading the pair's xq and sx whole,
    its routed weight g (W), and its block's rows and its G-block's scale row as the ``o0`` of a table read (`READ`) keyed by
    the pair's expert id, one table a block (`_table_reads`). The reads are table-major, unit ``(b TOPK + k) T + t``, so a
    table's declarations are three and a key's wire one a slot; the block GEMMs member-major, ``m B + b``, so a member's
    words are its blocks' outputs in turn."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    S = node.fn.bindings
    E, K, N, G = (int(S[x]) for x in ("E", "K", "N", "G"))
    ie, ig, iw, iws = XR.ARGS[spec_base(spec)]
    n = _block_rows(spec, K)
    if N % G or G % n:
        raise Unexpressible(f"{spec}: N = {N} is not whole G-blocks of {G} rows, blocks of {n}")
    B, NG = N // n, N // G
    ids = w.take(node, ie, whole=True)
    slots, step = _slots(ids, M)
    TOPK = len(slots) or 1
    T = M // TOPK
    got = []
    for H, blocks, arg, words in ((n * K // 2, B, iw, E * N * K // 2), (2 * K // G, NG, iws, E * NG * (K // G) * 2)):
        tid, at = w.unit(fam, _part(spec, ST.READ, V=E, H=H), blocks * M)
        lo, label = w.weight(node.args[arg].refs, words), w.label(node.args[arg].refs, words)
        for b in range(blocks):
            _table_reads(w, tid, at + b * M, M, lo, label, H, E, b)
            if not slots:
                w.feed(tid, "tok", at + b * M, ids, (M,))
            for k, r in enumerate(slots):
                w.b.feed(tid, "tok", at + (b * TOPK + k) * T, T, (r,), ((T, step),))
        got.append((tid, at, H))
    gemm = _part(spec, ST.EXPERTS[spec_base(spec)], N=n)
    tg, ag = w.unit(fam, gemm, B * M)
    for port, arg in (("xq", 0), ("sx", 1)):
        x = w.take(node, arg, whole=True)
        w.feed(tg, port, ag, x if isinstance(x, _Free) else [x[m] for m in range(M) for _b in range(B)], (T, TOPK, B))
    if ig is not None:
        g = w.take(node, ig, whole=True)
        if isinstance(g, _Free):
            w.feed(tg, "g", ag, g, (M, B))
        else:
            for m in range(M):
                w.b.feed(tg, "g", ag + m * B, B, g[m], ((B, 0),))
    (tr, ar, wr), (ts, as_, ws) = got
    w.feed(tg, "w", ag, [(Row(tr, ar + (b * TOPK + k) * T + t, "o0", wr),)
                         for t in range(T) for k in range(TOPK) for b in range(B)], (T, TOPK, B))
    w.feed(tg, "ws", ag, [(Row(ts, as_ + (b * n // G * TOPK + k) * T + t, "o0", ws),)
                          for t in range(T) for k in range(TOPK) for b in range(B)], (T, TOPK, NG, G // n))
    w.out(j, node, [Piece(0, N, Row(tg, ag, "o0", n), 1, B)])


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
        for ip in ids:
            # Each place is a range of its own (`interface.ordered`): two places of one key can be links of one chain, and
            # a range holding both would wait on itself.
            tid, at = w.unit(fam, fn, U)
            where[ip] = (tid, at, fn)
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
    "CublasSplitKGemmF32Out": row_gemm,
    "LlBf16DotprodGemm": row_gemm,
    "MoeRouterSigmoidBias": router_slots,
    "Fp8GroupQuantUe8m0": group_quant,
    "MoeExpertFp8Gemm": expert_gemm,
    "MoeExpertFp8GemmW": expert_gemm,
}

__all__ = ["BLOCK_WORDS", "FAMILIES", "batched_gemm", "expert_gemm", "group_quant", "latent_row", "router_slots", "row_gemm",
           "sparse_mla_split"]
