"""Eager Match patterns of the exact-int8 profile (`TargetProfile.int8_exact`): its named ops (`engine.int8_ops`), each bound to
registered Definitions (`program.registry.int8_exact`, `ml.circuits.ops.gemm`), whatever kernels the op launches inside:

    verity_int8::row_quant(x[M,K] bf16) -> (xq[M,K] s8, s_a[M] f32)
        -> Compute(Int8RowQuant_v1{K}, rows=M, (PerRow(x),), (PerRow(xq), PerRow(s_a)))
    verity_int8::scaled_mm(xq[M,K] s8, s_a[M] f32, w[N,K] int8, s_w[N,1] f32) -> y[M,N] bf16
        -> Compute(ScaledMmS8_v1{K,N,DOT=BlackwellS8ImmaDot32_v1}, rows=M, (PerRow(xq), PerRow(s_a), Shared(w), Shared(s_w)), (PerRow(y),))
    vllm.unified_attention_with_output(query, ..., output!, layer) ⊃ one verity_int8::kv_quant_batch(key_cache, value_cache,
            block_table, lens, k_scales) -> (kq, vq, sv, v0), then one verity_int8::window_rows_batch(query, kq, vq, sv, v0, k_scales,
            qsl, lens, scale), copied into output; or, under a credited read (`engine.int8_window`), the layer's
            verity_int8::credited_window(kv_cache, .., k!, v!) and verity_int8::kv_quant_window_batch(k, v, starts, lens, k_scales)
            in kv_quant_batch's place, starts[r] the earlier windows' rows
        -> per request r with a window (s = lens[r], its rows qsl[r] .. qsl[r+1] - 1), per KV head g
           Compute(Int8KvQuant_v1{s,D}, rows=1, (GatherMany(K slots 0..s-1 of head g), GatherMany(V slots), Const32),
                   (Shared(kq[r][g][:s]), Shared(vq[r][g][:s]), Shared(sv[r][g]), Shared(v0[r][g]))), then
           per (row i, head h) Compute(Int8LutAttentionRow_v1{D,pos_i+1,SCALE}, rows=1, (Shared(q[i][h]), Shared(kq[r][g][:pos_i+1]),
                                       Const32, Shared(vq[r][g][:pos_i+1]), Shared(sv[r][g]), Shared(v0[r][g])), (Shared(output[i][h]),))

A credited read's window rows are its KV slots' plaintext (pg_window's contract, bit-checked against the plain read), so
kv_quant_window_batch binds the same gathers of the cache credited_window read as kv_quant_batch does; the step's credited_step
and the layer's credited_window are consumed with the scope.  The offsets and lengths are the ops' host-list arguments, so the
fold reads each request's rows and window from the call itself and holds them to the step's scheduled rows.

The K scales are the run header's (`RunHeader.profile["backends"]["int8_k_scales"]`, read back from the layers by
`kernel_readback.int8_k_scales_of`), never read from device memory; each request's rows write its rows of `output` directly
(the copy out of the op's result is consumed with it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from verity.ml.circuits.defs import bind

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_slot_rows, is_row_major
from verity_vllm.observe.fold.patterns.kv_cache import _kv_moves, kv_layer_of, kv_slots
from verity_vllm.observe.fold.patterns.ops import as_float, bound_args, op_name, op_parts
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.transparent import find_descendants, scope_transparents
from verity_vllm.observe.fold.resolver import Compute, Const, GatherMany, Group, PerRow, Resolution, Shared, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode

LIBRARY = "verity_int8"


def _is_op(node: TreeNode, name: str) -> bool:
    return not node.is_triton and op_parts(node.op)[0] == LIBRARY and op_name(node.op) == name


def header_k_scales(ctx: StepContext) -> dict | None:
    try:
        k = ((ctx.header.profile or {}).get("backends") or {}).get("int8_k_scales")
        return k if isinstance(k, dict) else None
    except (AttributeError, TypeError):
        return None


@dataclass
class Int8RowQuant:
    """`verity_int8::row_quant`: the int8 linear's per-row quantizer on a bf16 `[M, K]` activation, both results materialised."""

    name: str = "int8.row_quant"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not _is_op(node, "row_quant"):
            return None
        from verity.ml.definitions.registry import int8_exact
        x = bound_args(node).get("x")
        outs = list(node.outs or ())
        if not isinstance(x, ViewDesc) or len(outs) != 2 or not all(isinstance(v, ViewDesc) for v in outs):
            return unsupported(node, "verity_int8::row_quant argument / results not bound as expected", outs=len(outs))
        xq, s_a = outs
        if x.rank != 2 or not is_row_major(x):
            return unsupported(node, "x is not a row-major [M, K] view", x=repr(x))
        M, K = x.shape
        if xq.shape != (M, K) or s_a.shape != (M,) or not is_row_major(xq):
            return unsupported(node, "results are not row-major xq [M, K], s_a [M]", xq=repr(xq), s_a=repr(s_a))
        if K % 32:
            return unsupported(node, "ScaledMmS8Coordinate_v1 takes K a multiple of 32 (its IMMA k step)", K=K)
        if not (x.dtype == "bfloat16" and xq.dtype == "int8" and s_a.dtype == "float32"):
            return unsupported(node, "Int8RowQuant_v1 is bf16 -> (int8, f32)", dtypes=[x.dtype, xq.dtype, s_a.dtype])
        return Compute(bind(int8_exact.Int8RowQuant, K=K), M, (PerRow(x),), (PerRow(xq), PerRow(s_a)), tuple(sorted(node.ids())),
                       hint=hint_of(node, "quant"), kernels=("elementwise_kernel", "reduce_kernel"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class Int8ScaledMm:
    """`verity_int8::scaled_mm`: the int8 linear's product of its quantizer's results; `w` and `s_w` must be checkpoint parameters
    (weight Roots)."""

    name: str = "int8.scaled_mm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not _is_op(node, "scaled_mm"):
            return None
        from verity.ml.definitions.registry import int8_exact
        a = bound_args(node)
        xq, s_a, w, sw = a.get("xq"), a.get("s_a"), a.get("w"), a.get("s_w")
        outs = list(node.outs or ()) if isinstance(node.outs, (list, tuple)) else [node.outs]
        if not all(isinstance(v, ViewDesc) for v in (xq, s_a, w, sw)) or len(outs) != 1 or not isinstance(outs[0], ViewDesc):
            return unsupported(node, "verity_int8::scaled_mm arguments / result not bound as expected", args=list(a), outs=len(outs))
        (y,) = outs
        if xq.rank != 2 or not is_row_major(xq):
            return unsupported(node, "xq is not a row-major [M, K] view", xq=repr(xq))
        M, K = xq.shape
        N = y.shape[1] if y.rank == 2 else None
        if N is None or y.shape != (M, N) or s_a.shape != (M,) or not is_row_major(y):
            return unsupported(node, "s_a / y are not [M] / row-major [M, N]", s_a=repr(s_a), y=repr(y))
        if w.shape != (N, K) or not is_row_major(w) or sw.shape not in ((N, 1), (N,)):
            return unsupported(node, "w / s_w are not [N, K] / [N, 1] views", w=repr(w), s_w=repr(sw), N=N, K=K)
        if K % 32:
            return unsupported(node, "ScaledMmS8Coordinate_v1 takes K a multiple of 32 (its IMMA k step)", K=K)
        if not (y.dtype == "bfloat16" and w.dtype == xq.dtype == "int8" and sw.dtype == s_a.dtype == "float32"):
            return unsupported(node, "ScaledMmS8_v1 is (int8, f32) x (int8, f32) -> bf16", dtypes=[xq.dtype, s_a.dtype, w.dtype, sw.dtype, y.dtype])
        wroots, sroots = ctx.roots_of(w), ctx.roots_of(sw)
        if not any(r.role == "weight" for r in wroots) or not any(r.role == "weight" for r in sroots):
            return unsupported(node, "w / s_w are not checkpoint parameters (weight Roots)", w=repr(w), s_w=repr(sw))
        wname, sname = wroots[0].name, sroots[0].name
        return Compute(int8_exact.scaled_mm(K, N), M, (PerRow(xq), PerRow(s_a), Shared(w), Shared(sw)), (PerRow(y),), tuple(sorted(node.ids())),
                       hint=hint_of(node, wname), kernels=("_int_mm", "elementwise_kernel"), attribution=attribution_for(ctx, M, node.phase),
                       pattern=self.name, roots=(None, None, wname, sname))


def _sub(v: ViewDesc, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]) -> ViewDesc:
    return ViewDesc(v.alloc, v.offset + offset, shape, strides, v.dtype, v.device)


def _slab(v: ViewDesc, i: int) -> ViewDesc:
    """v[i] of a view with a leading axis."""
    return _sub(v, i * v.strides[0], v.shape[1:], v.strides[1:])


def _keys(v: ViewDesc, g: int, p: int) -> ViewDesc:
    """The first p keys of KV head g of a [G, s, D] view."""
    return _sub(v, g * v.strides[0], (p, v.shape[2]), v.strides[1:])


def _same(a, b: ViewDesc | None) -> bool:
    return isinstance(a, ViewDesc) and b is not None and (a.alloc, a.offset, a.shape, a.strides) == (b.alloc, b.offset, b.shape, b.strides)


@dataclass
class Int8AttentionScope:
    """`unified_attention_with_output` around one `verity_int8::kv_quant_batch` / `verity_int8::window_rows_batch` pair (the int8
    backend, `engine.int8_ops.paged_attention`).  kv_quant_batch reads each request r's first lens[r] KV slots under the step's
    block table and writes (K', V', s_v, f32(V'(0))) per request and KV head, padded to the longest window; window_rows_batch
    reads them and the scope's query rows (request r's the last of its window, by its qsl), and its result is copied into the
    scope's output.  Per request: one `Int8KvQuant_v1{s,D}` per KV head g, then one `Int8LutAttentionRow_v1{D,pos+1,SCALE}` per
    (row, head) in row then head order, each reading its KV head's K scale as a Const32 -- the order
    `vllm_bindings.int8.Int8UnifiedAttentionRule` emits.  A `Group` over the scope: each Int8KvQuant_v1 claims the kv_quant_batch
    call, whose results the Match capture snapshots under that call's event, not the scope's."""

    name: str = "int8.window_attention"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_attention_with_output":
            return None
        kvs = find_descendants(node, lambda n: _is_op(n, "kv_quant_batch") or _is_op(n, "kv_quant_window_batch"))
        wrs = find_descendants(node, lambda n: _is_op(n, "window_rows_batch"))
        if not kvs and not wrs:
            return None
        cws = find_descendants(node, lambda n: _is_op(n, "credited_window"))
        css = find_descendants(node, lambda n: _is_op(n, "credited_step"))
        from verity.ml.definitions.registry import int8_exact
        sa = bound_args(node)
        query, output = sa.get("query"), sa.get("output")
        layer = kv_layer_of(node) or ""
        if not isinstance(query, ViewDesc) or not isinstance(output, ViewDesc) or query.rank != 3:
            return unsupported(node, "unified_attention_with_output query / output not bound as [T, H, D] views", names=list(sa))
        if len(kvs) != 1 or len(wrs) != 1:
            return unsupported(node, "not one kv_quant_batch and one window_rows_batch in the scope", kv_quant=len(kvs), window_rows=len(wrs))
        T, H, D = query.shape
        out3 = output if output.rank == 3 else (ViewDesc(output.alloc, output.offset, (T, H, D), (output.strides[0], D, 1), output.dtype,
                                                         output.device) if output.rank == 2 and output.shape == (T, H * D) and output.strides[1] == 1 else None)
        if out3 is None or out3.shape != (T, H, D) or out3.strides[1:] != (D, 1) or query.strides[1:] != (D, 1):
            return unsupported(node, "query / output rows are not [H][D] row-major", query=repr(query), output=repr(output))
        ks_all = header_k_scales(ctx)
        if ks_all is None or not isinstance(ks_all.get(layer), list):
            return unsupported(node, "the record does not carry this layer's K scales (RunHeader.profile.backends.int8_k_scales)", layer=layer)
        kbits = tuple(int(b) & 0xFFFFFFFF for b in ks_all[layer])
        copies = {id(w): [c for c in find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "copy_")
                          if _src_of(c) is not None and _out0(w) is not None and _src_of(c).alloc == _out0(w).alloc] for w in wrs}
        rcs = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "reshape_and_cache_flash")
        keep = {w.id for w in kvs + wrs + cws + css} | {c.id for cs in copies.values() for c in cs} | {n.id for n in rcs}
        kvn, w = kvs[0], wrs[0]
        ka, a = bound_args(kvn), bound_args(w)
        qsl, lens = _ints(a.get("qsl")), _ints(a.get("lens"))
        if qsl is None or lens is None or _ints(ka.get("lens")) != lens or len(qsl) != len(lens) + 1 or qsl[0] != 0:
            return unsupported(node, "kv_quant_batch / window_rows_batch offsets and lengths not bound as host lists of one batch",
                               qsl=a.get("qsl"), lens=a.get("lens"), kv_lens=ka.get("lens"))
        if any((qsl[r + 1] > qsl[r]) != (s > 0) or not 0 <= qsl[r + 1] - qsl[r] <= s for r, s in enumerate(lens)):
            return unsupported(node, "a request's rows are not its window's last rows, or a window has no rows", qsl=qsl, lens=lens)
        if _is_op(kvn, "kv_quant_window_batch"):
            credited = _credited_window(node, cws)
            if isinstance(credited, Unsupported):
                return credited
            kc, vc, kbuf, vbuf = credited
            starts, at = _ints(ka.get("starts")), [0]
            for s in lens[:-1]:
                at.append(at[-1] + s)
            if not (_same(ka.get("keys"), kbuf) and _same(ka.get("values"), vbuf)) or starts != at or at[-1] + lens[-1] > kbuf.shape[0]:
                return unsupported(node, "kv_quant_window_batch does not read each request's rows of the credited window in order",
                                   keys=repr(ka.get("keys")), starts=starts, lens=lens, rows=repr(kbuf))
        else:
            kc, vc = ka.get("key_cache"), ka.get("value_cache")
        before, bad = scope_transparents(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        for rc in rcs:
            moves = _kv_moves(rc, ctx, self.name, (), layer=layer)
            if isinstance(moves, Unsupported):
                return moves
            before.extend(moves)
        computes: list[Compute] = []
        hint = hint_of(node, str(layer) or "attn")
        qa, scale, kouts = a.get("query"), as_float(a.get("scale")), list(kvn.outs or ())
        if not all(isinstance(v, ViewDesc) for v in (qa, kc, vc)) or scale is None or len(kouts) != 4:
            return unsupported(node, "kv_quant_batch / window_rows_batch arguments not bound as expected", kv_quant=list(ka), window_rows=list(a))
        if not all(_same(a.get(nm), o) for nm, o in zip(("kq", "vq", "sv", "v0"), kouts)):
            return unsupported(node, "window_rows_batch does not read its kv_quant_batch's results", kv_quant=[repr(o) for o in kouts])
        kqb, vqb, svb, v0b = kouts
        T1 = qsl[-1]
        if qa.alloc != query.alloc or qa.offset != query.offset or qa.shape != (T1, H, D) or qa.strides != query.strides:
            return unsupported(node, "window_rows_batch's query is not the scope's query rows", q=repr(qa), query=repr(query))
        if len(copies[id(w)]) != 1:
            return unsupported(node, "window_rows_batch's result is not copied once into the output", copies=len(copies[id(w)]))
        dsts = copies[id(w)][0].args[0] if copies[id(w)][0].args else None
        if not isinstance(dsts, ViewDesc) or dsts.alloc != out3.alloc or dsts.offset != out3.offset or dsts.shape != (T1, H, D):
            return unsupported(node, "window_rows_batch's result is not copied into the output's rows", dst=repr(dsts), rows=T1)
        B, s_max = len(lens), max(lens)
        if (kqb.shape[0], kqb.shape[2:]) != (B, (s_max, D)) or vqb.shape != kqb.shape or svb.shape != (B, kqb.shape[1], D) or v0b.shape != svb.shape:
            return unsupported(node, "kv_quant_batch's results are not [B, G, s_max, D] / [B, G, D]", kv_quant=[repr(o) for o in kouts], B=B, s_max=s_max)
        for r, s in enumerate(lens):
            if not s:
                continue
            q0, n = qsl[r], qsl[r + 1] - qsl[r]
            q, dst = _sub(qa, q0 * qa.strides[0], (n, H, D), qa.strides), _sub(dsts, q0 * dsts.strides[0], (n, H, D), dsts.strides)
            kq, vq, sv, v0 = (_slab(o, r) for o in kouts)
            rows = ctx.start.rows[q0:q0 + n]
            if len(rows) != n or len({r for r, _ in rows}) != 1 or [int(p) for _, p in rows] != list(range(s - n, s)):
                return unsupported(node, "window_rows_batch's rows are not one request's last rows of its window", q0=q0, n=n, s=s, rows=list(rows))
            req = rows[0][0]
            try:
                kcs, vcs = as_slot_rows(kc), as_slot_rows(vc)
                slots = kv_slots(ctx, req, s, layer=layer)
            except GeometryError as e:
                return unsupported(node, str(e), request=req, s=s, layer=layer)
            G = kcs.shape[1]
            if kcs.shape[2] != D or vcs.shape != kcs.shape or H % G or len(kbits) != G or any(x >= kcs.shape[0] for x in slots):
                return unsupported(node, "KV cache geometry / K scales do not match q", k_cache=repr(kcs), H=H, D=D, k_scales=len(kbits))
            if kq.shape != (G, s_max, D) or sv.shape != (G, D) or not all(is_row_major(v) for v in kouts):
                return unsupported(node, "kv_quant_batch's results are not row-major [B, G, s_max, D] / [B, G, D]", kv_quant=[repr(o) for o in kouts])
            if not (q.dtype == kc.dtype == vc.dtype == out3.dtype == "bfloat16" and kq.dtype == vq.dtype == "int8" and sv.dtype == v0.dtype == "float32"):
                return unsupported(node, "Int8KvQuant_v1 / Int8LutAttentionRow_v1 are bf16 -> (int8, f32) -> bf16",
                                   dtypes=[q.dtype, kc.dtype, vc.dtype, out3.dtype, kq.dtype, vq.dtype, sv.dtype, v0.dtype])
            if not math.isfinite(scale) or scale <= 0:
                return unsupported(node, "softmax scale is not a positive finite value", scale=scale)
            head = lambda c, g: _sub(c, g * c.strides[1], (c.shape[0], D), (c.strides[0], c.strides[2]))  # noqa: E731
            fkv = bind(int8_exact.Int8KvQuant, S=s, D=D)
            for g in range(G):
                computes.append(Compute(fkv, 1, (GatherMany(head(kcs, g), (slots,)), GatherMany(head(vcs, g), (slots,)), Const(32, kbits[g], "k_scale")),
                                        (Shared(_keys(kq, g, s)), Shared(_keys(vq, g, s)), Shared(_slab(sv, g)), Shared(_slab(v0, g))), (kvn.id,),
                                        hint=f"{hint}/kv_quant", kernels=("kv_rows", "elementwise_kernel", "reduce_kernel"), attribution=(rows[-1],),
                                        pattern=self.name))
            for i in range(n):
                p = s - n + i + 1
                fr = bind(int8_exact.Int8LutAttentionRow, D=D, S=p, SCALE=scale)
                qi, di = _slab(q, i), _slab(dst, i)
                for hd in range(H):
                    g = hd // (H // G)
                    computes.append(Compute(fr, 1, (Shared(_slab(qi, hd)), Shared(_keys(kq, g, p)), Const(32, kbits[g], "k_scale"),
                                                    Shared(_keys(vq, g, p)), Shared(_slab(sv, g)), Shared(_slab(v0, g))), (Shared(_slab(di, hd)),), (),
                                            hint=hint, kernels=("lut_rows", "elementwise_kernel", "reduce_kernel"), attribution=(rows[i],),
                                            pattern=self.name))
        return Group((*before, *computes), tuple(sorted(node.ids())), hint=hint, pattern=self.name)


def _credited_window(scope: TreeNode, cws: list[TreeNode]) -> tuple[ViewDesc, ViewDesc, ViewDesc, ViewDesc] | Unsupported:
    """The (key cache, value cache, K window, V window) of the scope's one credited_window: its kv_cache [pages, G, slots, 2D]
    as FlashAttention's [pages, slots, G, D] key and value views, and its K and V [rows, G, D]."""
    if len(cws) != 1:
        return unsupported(scope, "kv_quant_window_batch without exactly one credited_window in the scope", credited_window=len(cws))
    a = bound_args(cws[0])
    kv, k, v = a.get("kv_cache"), a.get("k"), a.get("v")
    if not all(isinstance(x, ViewDesc) for x in (kv, k, v)) or kv.rank != 4 or kv.shape[3] % 2 or k.rank != 3 or k.shape != v.shape:
        return unsupported(scope, "credited_window's kv_cache / k / v are not [pages, G, slots, 2D] / [rows, G, D] views",
                           kv_cache=repr(kv), k=repr(k), v=repr(v))
    P, Gk, S, D2 = kv.shape
    D = D2 // 2
    shape, strides = (P, S, Gk, D), (kv.strides[0], kv.strides[2], kv.strides[1], kv.strides[3])
    if kv.strides[3] != 1 or k.shape[1:] != (Gk, D) or not (is_row_major(k) and is_row_major(v)) or not k.dtype == v.dtype == "bfloat16":
        return unsupported(scope, "credited_window's cache or windows are not laid out as the int8 backend's", kv_cache=repr(kv), k=repr(k))
    return _sub(kv, 0, shape, strides), _sub(kv, D, shape, strides), k, v


def _ints(x) -> list[int] | None:
    """A host list argument (an op's `int[]`) as ints, else None."""
    if isinstance(x, (list, tuple)) and all(isinstance(i, int) and not isinstance(i, bool) for i in x):
        return [int(i) for i in x]
    return None


def _out0(node: TreeNode) -> ViewDesc | None:
    o = node.outs
    o = o[0] if isinstance(o, (list, tuple)) and o else o
    return o if isinstance(o, ViewDesc) else None


def _src_of(copy: TreeNode) -> ViewDesc | None:
    a = copy.args
    return a[1] if len(a) > 1 and isinstance(a[1], ViewDesc) else None


__all__ = ["Int8RowQuant", "Int8ScaledMm", "Int8AttentionScope", "header_k_scales"]
