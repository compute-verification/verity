"""DeepSeek-V4-Pro's FlashMLA sparse attention folds (vLLM d9105ea80): `_flashmla_C.sparse_decode_fwd` and `sparse_prefill_fwd`, each
with the C4 / C128 compress the attention emits before it (`mla_compress`), as the Build's `Dsv4SparseAttnRule` constructs them
(`rules/vllm_dsv4.py`).
"""

from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import re
from dataclasses import dataclass

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.dsv4_cache import _entry_slots, _is, _module, _row, _weights, block_bytes, fp8_rows, runs, state_row, swa_block
from verity_vllm.observe.fold.patterns.geometry import GeometryError, sub_rows
from verity_vllm.observe.fold.patterns.ops import as_float, base_op, bound_args
from verity_vllm.observe.fold.patterns.resolutions import unsupported
from verity_vllm.observe.fold.patterns.target import target_of, vocab_kind
from verity_vllm.observe.fold.resolver import Compute, Const, Gather, Group, PerRow, Pieces, Resolution, Shared, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.program.frontend.dsv4_leaves import FP8_DS_MLA_QB
from verity_vllm.program.frontend.rules.vllm_dsv4 import COMPRESSORS, FLASHMLA_GEOMETRY
from verity.ml.definitions.numerics.flashmla_twin import num_sm_parts, request_splits, sched_meta
from verity.ml.definitions.registry import dsv4_compressor as C

SPARSE_DECODE_OP = "_flashmla_C.sparse_decode_fwd"
SPARSE_PREFILL_OP = "_flashmla_C.sparse_prefill_fwd"
ATTN = re.compile(r"model\.layers\.(\d+)\.attn")
I32_NEG1 = 0xFFFFFFFF


@dataclass(frozen=True)
class SparseLayer:
    """A sparse attention layer's statics as `Dsv4SparseAttnRule` takes them, from the model config: its compress ratio (1: window
    only), window, live and padded heads, FlashMLA geometry and scale, and its sink's NH live entries."""
    layer: str
    R: int
    window: int
    NH: int
    NHP: int
    D: int
    DR: int
    QT: int
    BN: int
    DOT: object
    scale: float
    sink: Shared
    sink_root: str


def sparse_layer(node: TreeNode, ctx: StepContext, D: int, sink: object, scale: float | None, last: int) -> SparseLayer | Unsupported:
    """The statics of the attention layer `node` runs in, reading compressed rows up to position `last`."""
    layer, cfg = _module(node), ctx.config
    m = ATTN.fullmatch(layer)
    if m is None or not all(k in cfg for k in ("COMPRESS_RATIOS", "WINDOW", "NH", "PADDED_HEADS", "QK_ROPE", "INDEX_TOPK")):
        return unsupported(node, "the attention is no layer's, or the model config lacks the sparse attention's sizes", module=layer)
    R, window, NH, NHP, DR = (max(1, int(cfg["COMPRESS_RATIOS"][int(m.group(1))])), int(cfg["WINDOW"]), int(cfg["NH"]),
                              int(cfg["PADDED_HEADS"]), int(cfg["QK_ROPE"]))
    G = FLASHMLA_GEOMETRY.get((D, DR))
    if target_of(ctx)[0].compute_capability != (9, 0) or G is None or (R != 1 and R not in COMPRESSORS):
        return unsupported(node, "no FlashMLA cc 9.0 sparse geometry or compressor at the layer's head, rope and ratio", D=D, DR=DR, ratio=R,
                           geometries=sorted(FLASHMLA_GEOMETRY))
    if NHP not in (64, 128) or not 1 <= NH <= NHP or scale is None:
        return unsupported(node, "FlashMLA's sparse kernels take 64 or 128 padded heads, at least one live, and a static scale", NH=NH, NHP=NHP)
    if R == 4 and (last + 1) // 4 > int(cfg["INDEX_TOPK"]):
        return unsupported(node, "more compressed rows than the indexer's top-k: its long path has no Definition", position=last,
                           topk=cfg["INDEX_TOPK"])
    if not isinstance(sink, ViewDesc) or not _is(sink, (NHP,), "float32") or not (sroots := ctx.roots_of(sink)):
        return unsupported(node, "the sink is not an f32 [NHP] root", sink=repr(sink))
    return SparseLayer(layer, R, window, NH, NHP, D, DR, G["QT"], G["BN"], G["DOT"], scale,
                       Shared(ViewDesc(sink.alloc, sink.offset, (NH,), sink.strides, sink.dtype, sink.device)), sroots[0].name)


def _consts(*runs_: tuple[int, int]) -> Pieces:
    """One row of shared i32 constants: each (bits, count) run, in order.  A run of no copies still emits its constant, as the Build's
    `[i32(v)] * n` does at n = 0 (the decode's -1 when its window and compressed rows fill their blocks)."""
    return Pieces((tuple(Const(32, bits, "idx", shared=True, count=n) for bits, n in runs_),))


def mla_compress(node: TreeNode, ctx: StepContext, sl: SparseLayer) -> list[Compute] | Unsupported:
    """The CuTe DSL compress (`compress_norm_rope_store_cutedsl`, compressor.py:402-454) a head-512 compressor runs after its state
    writes, which no event shows, at each window end p its `SavePartialStates` recorded: `Dsv4MlaCompressC4_v1{ROWS}` over the state rows
    of positions p - 7 .. p at and past 0 (row r reads the D words from (r // 4) D of each half; the kernel skips a row below 0) or
    `Dsv4MlaCompressC128_v1{W,R}` over positions p - 127 .. p, the norm weight and cos_sin[p - R + 1] (the kernel's (p // R) * R), into
    compressed entry p // R of the layer's cache (`kv_cache[<layer>]` of the capture's external buffers) in fp8_ds_mla's block layout.
    The Build emits it in `Dsv4SparseAttnRule`, with the attention reading the rows; so does the fold, since the compressor runs before
    or after the layer's window insert, whose rotary table (the attention's `rotary_emb`, the compressor's too) it reads."""
    pend, tab = ctx.scratch.get("dsv4_compress", {}).pop(sl.layer, None), ctx.scratch.get("dsv4_cos_sin", {}).get(sl.layer)
    if pend is None or tab is None or pend[2] != sl.R:
        return unsupported(node, "the layer's compressor state write or window insert did not resolve before its attention", layer=sl.layer)
    (state, W, R, ends), (cs, croot) = pend, tab
    if not ends:
        return []
    D, QB, eps, kc = W // (2 if R == 4 else 1), FP8_DS_MLA_QB, float(ctx.config["EPS"]), ctx.buffers.get(f"kv_cache[{sl.layer}]")
    try:
        C.check_norm_statics(R, D, QB, eps)
    except ValueError as e:
        return unsupported(node, str(e))
    if len({r[0] for r, _ in ends}) != 1 or not isinstance(kc, ViewDesc):
        return unsupported(node, "the window ends are not one request's, or the capture names no compressed cache for the layer", layer=sl.layer)
    if cs.dtype != "float32" or cs.rank != 2 or cs.shape[1] != QB or ends[-1][1] - R + 1 >= cs.shape[0]:
        return unsupported(node, "the rotary table is not an f32 [., QB] table holding the windows' positions", cs=repr(cs))
    wname = f"{sl.layer}.compressor.norm.weight"
    if isinstance(ws := _weights(node, ctx, (wname,)), Unsupported):
        return ws
    if not _is(ws[0].view, (D,), "bfloat16"):
        return unsupported(node, "the compressor's norm weight is not bf16 [D]", w=repr(ws[0].view))
    req, win = ends[0][0][0], 2 * R if R == 4 else R
    taken = {p: [(r, s) for r, s in enumerate(range(p - win + 1, p + 1)) if s >= 0] for _, p in ends}
    need = sorted({s for t in taken.values() for _, s in t})
    if isinstance(src := _entry_slots(node, ctx, f"{sl.layer}.compressor.state_cache", req, need, state.shape[1]), Unsupported):
        return src
    if isinstance(dst := _entry_slots(node, ctx, sl.layer, req, [p // R for _, p in ends], kc.shape[1] if kc.rank > 1 else 0, ratio=R),
                  Unsupported):
        return dst
    src, n, parts = dict(zip(need, src)), kc.shape[1], []
    for (b, rows_n), lo, hi in runs([(d // n, len(taken[p])) for d, (_, p) in zip(dst, ends)]):
        try:
            nope, scale, rope = swa_block(block_bytes(kc), b, n, D - QB, QB, QB)
        except GeometryError as e:
            return unsupported(node, str(e))
        ps = [p for _, p in ends[lo:hi]]
        spec = (vocab_kind(ctx, COMPRESSORS[4], ROWS=rows_n, D=D, QB=QB, EPS=eps) if R == 4 else
                vocab_kind(ctx, COMPRESSORS[128], W=C.DSV4_COMPRESSOR["W"], R=R // C.DSV4_COMPRESSOR["W"], D=D, QB=QB, EPS=eps))
        kv, score = (Pieces(tuple(tuple(state_row(state, src[s], k, W, r // R * D, D) for r, s in taken[p]) for p in ps)) for k in (0, 1))
        idx = tuple(d % n for d in dst[lo:hi])
        parts.append(Compute(spec, hi - lo, (kv, score, ws[0], Gather(cs, tuple(p - R + 1 for p in ps))),
                             (Gather(nope, idx), Gather(rope, idx), Gather(scale, idx)), (), hint=f"{sl.layer}.compressor/compress",
                             attribution=tuple(r for r, _ in ends[lo:hi]), pattern="dsv4.mla_compress", roots=(None, None, wname, croot)))
    return parts


@dataclass
class SparseDecode:
    """`_flashmla_C.sparse_decode_fwd` (`flash_mla_with_kvcache`, flashmla.py:174-264) for a step's one decode token at position p ->
    the layer's compress (`mla_compress`), then `FlashMlaDecode_v1{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE}` of q[0][:NH] over its window rows
    in `kv` (positions max(p - window + 1, 0) .. p, padded to NW, a multiple of BN, with copies of the first) and its compressed rows in
    `extra_kv` (entries 0 .. (p + 1) / R - 1, padded likewise), the indices 0 then -1 and the lengths as constants, and sink[:NH], at the
    splits FlashMLA's planner gives the decode alone (`flashmla_twin`): the Build's `Dsv4SparseAttnRule` decode."""

    name = "dsv4.sparse_decode"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != SPARSE_DECODE_OP:
            return None
        a, rows = bound_args(node), tuple(ctx.start.rows)
        q, kv, xkv, out = (a.get(k) for k in ("q", "kv", "extra_kv", "out_"))
        if len(rows) != 1:
            return unsupported(node, "a launch of more than one decode token: the batch's FlashMLA schedule is not folded yet", T=len(rows))
        if not all(isinstance(v, ViewDesc) for v in (q, kv, out)) or q.rank != 4 or q.shape[:2] != (1, 1) or q.dtype != "bfloat16" \
                or q.strides[-1] != 1 or not _is(out, q.shape, "bfloat16") or out.strides[-1] != 1:
            return unsupported(node, "q and out are not bf16 [1,1,NHP,D] over one token, or the window cache is no view", q=repr(q), out=repr(out))
        (req, p), D = rows[0], q.shape[3]
        sl = sparse_layer(node, ctx, D, a.get("attn_sink"), as_float(a.get("sm_scale")), p)
        if isinstance(sl, Unsupported):
            return sl
        if q.shape[2] != sl.NHP or a.get("d_v") != D or (sl.R > 1) != isinstance(xkv, ViewDesc):
            return unsupported(node, "the decode is not over the layer's padded heads and D values, with a compressed cache iff it compresses",
                               q=repr(q), d_v=a.get("d_v"), ratio=sl.R, extra_kv=repr(xkv))
        DN, BN, s0 = D - sl.DR, sl.BN, max(p - sl.window + 1, 0)
        swa, nc = p - s0 + 1, (p + 1) // sl.R if sl.R > 1 else 0
        NW, NX = -(-swa // BN) * BN, -(-nc // BN) * BN
        if isinstance(wslots := _entry_slots(node, ctx, f"{sl.layer}.swa_cache", req, range(s0, p + 1), kv.shape[1] if kv.rank > 1 else 0),
                      Unsupported):
            return wslots
        xslots = _entry_slots(node, ctx, sl.layer, req, range(nc), xkv.shape[1] if xkv.rank > 1 else 0, ratio=sl.R) if nc else ()
        if isinstance(xslots, Unsupported):
            return xslots
        try:
            win = fp8_rows(block_bytes(kv), wslots, kv.shape[1], DN, sl.DR, sl.QT)
            comp = fp8_rows(block_bytes(xkv), xslots, xkv.shape[1], DN, sl.DR, sl.QT) if nc else []
        except GeometryError as e:
            return unsupported(node, str(e))
        if isinstance(made := mla_compress(node, ctx, sl) if sl.R > 1 else [], Unsupported):
            return made
        rs = win + win[:1] * (NW - swa) + comp + comp[:1] * (NX - nc)
        metas, ns = sched_meta([swa], [nc] if sl.R > 1 else None, topk=sl.window, extra_topk=int(sl.R > 1),
                               parts=num_sm_parts(sl.NHP, 1, int(target_of(ctx)[0].num_sms)), block_size_n=BN)
        spec = vocab_kind(ctx, "flashmla_decode", NH=sl.NH, SPLITS=request_splits(metas, ns, 0, (NW + NX) // BN), NW=NW, DN=DN, DR=sl.DR,
                          QT=sl.QT, BN=BN, DOT=sl.DOT, SCALE=sl.scale)
        lens = (Const(32, swa, "swa_len", shared=True),) + ((Const(32, nc, "extra_len", shared=True),) if NX else ())
        qv, ov = (ViewDesc(v.alloc, v.offset, (1, sl.NH, D), (v.strides[0], v.strides[2], 1), v.dtype, v.device) for v in (q, out))
        attn = Compute(spec, 1, (PerRow(qv), *(Pieces((tuple(r[k] for r in rs),)) for k in range(3)),
                                 _consts((0, swa), (I32_NEG1, NW - swa), (0, nc), (I32_NEG1, NX - nc)), *lens, sl.sink),
                       (PerRow(ov),), (node.id,), hint=f"{sl.layer}/decode", attribution=rows, pattern=self.name,
                       roots=(None,) * (5 + len(lens)) + (sl.sink_root,))
        return Group((*made, attn), tuple(sorted(node.ids())), hint=sl.layer, pattern=self.name)


@dataclass
class SparsePrefill:
    """`_flashmla_C.sparse_prefill_fwd` (`flash_mla_sparse_fwd`, flashmla.py:266-398) for one request's prefill tokens over the bf16
    workspace `kv` that two gathers no event shows fill (`dequantize_and_gather_k_cache`, the CuTe DSL) -> the layer's compress
    (`mla_compress`); `FlashMlaGatherRow_v1{DN,DR,QT}` of each row they write, the compressed entries 0 .. seq_len / R - 1 to rows 0 ..
    and the last gather_len window positions from row chunk_N = seq_len / R (`get_prefill_chunk_plan` at one request: gather_len =
    T + min(prefix, window - 1)), each from the capture's `kv_cache[<layer>]` / `kv_cache[<layer>.swa_cache_layer]`; then per token at p
    `FlashMlaPrefill_v1{NH,NP,D,BN,DOT,SCALE}` of q[t][:NH] over its compressed rows, then its window rows (positions max(p - window + 1,
    0) .. p), padded to its NP pairs of BN-row blocks with copies of the first, the indices 0 then -1, its length and the workspace's one
    row as constants, and sink[:NH]: the Build's `Dsv4SparseAttnRule` prefill, one Compute per run of tokens with one NP."""

    name = "dsv4.sparse_prefill"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != SPARSE_PREFILL_OP:
            return None
        a, rows = bound_args(node), tuple(ctx.start.rows)
        q, wsv, out = (a.get(k) for k in ("q", "kv", "out_"))
        T, req, pos = len(rows), rows[0][0], [p for _, p in rows]
        if not all(isinstance(v, ViewDesc) for v in (q, wsv, out)) or q.rank != 3 or q.shape[0] != T or q.dtype != "bfloat16" \
                or q.strides[-1] != 1 or not _is(out, q.shape, "bfloat16") or out.strides[-1] != 1:
            return unsupported(node, "q and out are not bf16 [T,NHP,D] over the step's tokens, or the workspace is no view", q=repr(q), out=repr(out))
        if any(r != req for r, _ in rows) or pos != list(range(pos[0], pos[0] + T)):
            return unsupported(node, "the prefill is not one request's consecutive positions: the chunk plan's workspace is folded at one")
        D = q.shape[2]
        sl = sparse_layer(node, ctx, D, a.get("attn_sink"), as_float(a.get("sm_scale")), pos[-1])
        if isinstance(sl, Unsupported):
            return sl
        seq, BN, DN = pos[-1] + 1, sl.BN, D - sl.DR
        gather, N = T + min(pos[0], sl.window - 1), seq // sl.R if sl.R > 1 else 0
        if q.shape[1] != sl.NHP or a.get("d_v") != D or wsv.dtype != "bfloat16" or wsv.rank != 3 or wsv.shape != (N + gather, 1, D) \
                or wsv.strides[1:] != (D, 1):
            return unsupported(node, "the workspace is not the chunk plan's bf16 [chunk_N + gather_len, 1, D] rows over the layer's padded heads",
                               kv=repr(wsv), chunk_N=N, gather_len=gather, q=repr(q))
        caches = [(ctx.buffers.get(f"kv_cache[{sl.layer}.swa_cache_layer]"), f"{sl.layer}.swa_cache", range(seq - gather, seq), 1)] \
            + ([(ctx.buffers.get(f"kv_cache[{sl.layer}]"), sl.layer, range(N), sl.R)] if N else [])
        if not all(isinstance(c, ViewDesc) and c.rank > 1 for c, *_ in caches):
            return unsupported(node, "the capture's external buffers name no window or compressed cache for the layer", layer=sl.layer)
        src = []
        for cache, group, entries, ratio in caches:
            if isinstance(slots := _entry_slots(node, ctx, group, req, entries, cache.shape[1], ratio=ratio), Unsupported):
                return slots
            try:
                src.append(fp8_rows(block_bytes(cache), slots, cache.shape[1], DN, sl.DR, sl.QT))
            except GeometryError as e:
                return unsupported(node, str(e))
        if isinstance(made := mla_compress(node, ctx, sl) if sl.R > 1 else [], Unsupported):
            return made
        src = (src[1] if N else []) + src[0]
        ws = ViewDesc(wsv.alloc, wsv.offset, (N + gather, D), (wsv.strides[0], 1), wsv.dtype, wsv.device)
        made.append(Compute(vocab_kind(ctx, "flashmla_gather_row", DN=DN, DR=sl.DR, QT=sl.QT), len(src),
                            tuple(Pieces(tuple((r[k],) for r in src)) for k in range(3)), (PerRow(ws),), (), hint=f"{sl.layer}/gather",
                            attribution=tuple((req, (j + 1) * sl.R - 1) for j in range(N)) + tuple((req, s) for s in range(seq - gather, seq)),
                            pattern=self.name))
        toks = []
        for p in pos:
            s0, tl = max(p - sl.window + 1, 0), (p + 1) // sl.R if sl.R > 1 else 0
            n = tl + p - s0 + 1
            blocks = -(-n // BN)
            NP = -(-blocks // 2)
            kv = [_row(ws, j) for j in range(tl)] + [_row(ws, N + s - (seq - gather)) for s in range(s0, p + 1)]
            toks.append((NP, kv + kv[:1] * (2 * NP * BN - n), n))
        qv, ov = (ViewDesc(v.alloc, v.offset, (T, sl.NH, D), (v.strides[0], v.strides[1], 1), v.dtype, v.device) for v in (q, out))
        for NP, lo, hi in runs([t[0] for t in toks]):
            spec = vocab_kind(ctx, "flashmla_prefill", NH=sl.NH, NP=NP, D=D, BN=BN, DOT=sl.DOT, SCALE=sl.scale)
            made.append(Compute(spec, hi - lo, (PerRow(sub_rows(qv, lo, hi)), Pieces(tuple(tuple(kv) for _, kv, _ in toks[lo:hi])),
                                                Pieces(tuple(_consts((0, n), (I32_NEG1, 2 * NP * BN - n)).rows[0] for _, _, n in toks[lo:hi])),
                                                Pieces(tuple((Const(32, n, "len", shared=True),) for _, _, n in toks[lo:hi])),
                                                Const(32, 1, "skv", shared=True), sl.sink),
                                (PerRow(sub_rows(ov, lo, hi)),), (node.id,) if lo == 0 else (), hint=f"{sl.layer}/prefill",
                                attribution=rows[lo:hi], pattern=self.name, roots=(None,) * 5 + (sl.sink_root,)))
        return Group(tuple(made), tuple(sorted(node.ids())), hint=sl.layer, pattern=self.name)


__all__ = ["SparseDecode", "SparseLayer", "SparsePrefill", "mla_compress", "sparse_layer"]
