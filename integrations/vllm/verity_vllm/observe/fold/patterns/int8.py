"""Eager Match patterns of the exact-int8 profile (`TargetProfile.int8_exact`): its named ops (`engine.int8_ops`), each bound to
registered Definitions (`program.registry.int8_exact`, `ml.circuits.ops.gemm`), whatever kernels the op launches inside:

    verity_int8::linear(x[M,K] bf16, w[N,K] int8, s_w[N,1] f32) -> (xq[M,K] s8, s_a[M] f32, y[M,N] bf16)
        -> Compute(Int8RowQuant_v1{K}, rows=M, (PerRow(x),), (PerRow(xq), PerRow(s_a))), then
           Compute(ScaledMmS8_v1{K,N,DOT=BlackwellS8ImmaDot32_v1}, rows=M, (PerRow(xq), PerRow(s_a), Shared(w), Shared(s_w)), (PerRow(y),))
    vllm.unified_attention_with_output(query, ..., output!, layer) ⊃ per request verity_int8::kv_quant(key_cache, value_cache, blocks, s,
            k_scales) -> (kq, vq, sv, v0), then verity_int8::window_rows(q, kq, vq, sv, v0, k_scales, scale), copied into output
        -> per KV head g Compute(Int8KvQuant_v1{s,D}, rows=1, (GatherMany(K slots 0..s-1 of head g), GatherMany(V slots), Const32),
                                 (Shared(kq[g]), Shared(vq[g]), Shared(sv[g]), Shared(v0[g]))), then
           per (row i, head h) Compute(Int8LutAttentionRow_v1{D,pos_i+1,SCALE}, rows=1, (Shared(q[i][h]), Shared(kq[g][:pos_i+1]), Const32,
                                       Shared(vq[g][:pos_i+1]), Shared(sv[g]), Shared(v0[g])), (Shared(output[i][h]),))

The K scales are the run header's (`RunHeader.profile["backends"]["int8_k_scales"]`, read back from the layers by
`kernel_readback.int8_k_scales_of`), never read from device memory; each request's rows write its rows of `output` directly
(the copy out of the op's result is consumed with it).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from verity.ml.circuits.defs import bind

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_slot_rows, is_row_major
from verity_vllm.observe.fold.patterns.kv_cache import _kv_moves, kv_layer_of, kv_slots
from verity_vllm.observe.fold.patterns.ops import as_float, bound_args, op_name, op_parts
from verity_vllm.observe.fold.patterns.resolutions import _group_compute, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.transparent import find_descendants, scope_transparents
from verity_vllm.observe.fold.resolver import Compute, Const, GatherMany, PerRow, Resolution, Shared, StepContext, Unsupported
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
class Int8Linear:
    """`verity_int8::linear`: the plain int8 linear on a bf16 `[M, K]` activation, its three results materialised; `w` and `s_w`
    must be checkpoint parameters (weight Roots)."""

    name: str = "int8.linear"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not _is_op(node, "linear"):
            return None
        from verity_vllm.program.registry import int8_exact
        a = bound_args(node)
        x, w, sw = a.get("x"), a.get("w"), a.get("s_w")
        outs = list(node.outs or ())
        if not all(isinstance(v, ViewDesc) for v in (x, w, sw)) or len(outs) != 3 or not all(isinstance(v, ViewDesc) for v in outs):
            return unsupported(node, "verity_int8::linear arguments / results not bound as expected", args=list(a), outs=len(outs))
        xq, s_a, y = outs
        if x.rank != 2 or not is_row_major(x):
            return unsupported(node, "x is not a row-major [M, K] view", x=repr(x))
        M, K = x.shape
        N = y.shape[1] if y.rank == 2 else None
        if N is None or y.shape != (M, N) or xq.shape != (M, K) or s_a.shape != (M,) or not all(is_row_major(v) for v in (xq, y)):
            return unsupported(node, "results are not row-major xq [M, K], s_a [M], y [M, N]", xq=repr(xq), s_a=repr(s_a), y=repr(y))
        if w.shape != (N, K) or not is_row_major(w) or sw.shape not in ((N, 1), (N,)):
            return unsupported(node, "w / s_w are not [N, K] / [N, 1] views", w=repr(w), s_w=repr(sw), N=N, K=K)
        if K % 32:
            return unsupported(node, "ScaledMmS8Coordinate_v1 takes K a multiple of 32 (its IMMA k step)", K=K)
        if not (x.dtype == y.dtype == "bfloat16" and w.dtype == xq.dtype == "int8" and sw.dtype == s_a.dtype == "float32"):
            return unsupported(node, "Int8Linear_v1 is bf16 x (int8, f32) -> (int8, f32, bf16)",
                               dtypes=[x.dtype, w.dtype, sw.dtype, xq.dtype, s_a.dtype, y.dtype])
        wroots, sroots = ctx.roots_of(w), ctx.roots_of(sw)
        if not any(r.role == "weight" for r in wroots) or not any(r.role == "weight" for r in sroots):
            return unsupported(node, "w / s_w are not checkpoint parameters (weight Roots)", w=repr(w), s_w=repr(sw))
        wname, sname = wroots[0].name, sroots[0].name
        hint, attr = hint_of(node, wname), attribution_for(ctx, M, node.phase)
        quant = Compute(bind(int8_exact.Int8RowQuant, K=K), M, (PerRow(x),), (PerRow(xq), PerRow(s_a)), tuple(sorted(node.ids())),
                        hint=f"{hint}/quant", kernels=("elementwise_kernel", "reduce_kernel"), attribution=attr, pattern=self.name)
        mm = Compute(int8_exact.scaled_mm(K, N), M, (PerRow(xq), PerRow(s_a), Shared(w), Shared(sw)), (PerRow(y),), (),
                     hint=hint, kernels=("_int_mm", "elementwise_kernel"), attribution=attr, pattern=self.name,
                     roots=(None, None, wname, sname))
        return _group_compute(quant, also=(mm,))


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
    """`unified_attention_with_output` around per-request `verity_int8::kv_quant` / `verity_int8::window_rows` pairs (the int8
    backend, `engine.int8_ops.paged_attention`).  A request's kv_quant reads its first `s` KV slots under the step's block table and
    writes (K', V', s_v, f32(V'(0))) per KV head; its window_rows reads them and a row slice of the scope's query (its window's last
    rows), and its result is copied into the same rows of the scope's output.  Per request: one `Int8KvQuant_v1{s,D}` per KV head
    g, then one `Int8LutAttentionRow_v1{D,pos+1,SCALE}` per (row, head) in row then head order, each reading its KV head's K scale
    as a Const32 -- the order `vllm_bindings.int8.Int8UnifiedAttentionRule` emits."""

    name: str = "int8.window_attention"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_attention_with_output":
            return None
        kvs = find_descendants(node, lambda n: _is_op(n, "kv_quant"))
        wrs = find_descendants(node, lambda n: _is_op(n, "window_rows"))
        if not kvs and not wrs:
            return None
        from verity_vllm.program.registry import int8_exact
        sa = bound_args(node)
        query, output = sa.get("query"), sa.get("output")
        layer = kv_layer_of(node) or ""
        if not isinstance(query, ViewDesc) or not isinstance(output, ViewDesc) or query.rank != 3:
            return unsupported(node, "unified_attention_with_output query / output not bound as [T, H, D] views", names=list(sa))
        if len(kvs) != len(wrs):
            return unsupported(node, "kv_quant and window_rows calls do not pair up", kv_quant=len(kvs), window_rows=len(wrs))
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
        keep = {w.id for w in kvs + wrs} | {c.id for cs in copies.values() for c in cs} | {n.id for n in rcs}
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
        for kvn, w in zip(kvs, wrs):
            ka, a = bound_args(kvn), bound_args(w)
            kc, vc, s = ka.get("key_cache"), ka.get("value_cache"), ka.get("s")
            q, scale = a.get("q"), as_float(a.get("scale"))
            kouts = list(kvn.outs or ())
            if not all(isinstance(v, ViewDesc) for v in (q, kc, vc)) or not isinstance(s, int) or scale is None or len(kouts) != 4:
                return unsupported(node, "kv_quant / window_rows arguments not bound as expected", kv_quant=list(ka), window_rows=list(a))
            if not all(_same(a.get(nm), o) for nm, o in zip(("kq", "vq", "sv", "v0"), kouts)):
                return unsupported(node, "window_rows does not read its kv_quant's results", kv_quant=[repr(o) for o in kouts])
            kq, vq, sv, v0 = kouts
            if q.alloc != query.alloc or q.rank != 3 or q.shape[1:] != (H, D) or (q.offset - query.offset) % query.strides[0]:
                return unsupported(node, "window_rows' q is not a row slice of the scope's query", q=repr(q), query=repr(query))
            if len(copies[id(w)]) != 1:
                return unsupported(node, "window_rows' result is not copied once into the output", copies=len(copies[id(w)]))
            dst = copies[id(w)][0].args[0] if copies[id(w)][0].args else None
            n, q0 = q.shape[0], (q.offset - query.offset) // query.strides[0]
            if not isinstance(dst, ViewDesc) or dst.alloc != out3.alloc or dst.offset != out3.offset + q0 * out3.strides[0] or dst.shape != (n, H, D):
                return unsupported(node, "window_rows' result is not copied into its own rows of the output", dst=repr(dst), q0=q0, n=n)
            rows = ctx.start.rows[q0:q0 + n]
            if len(rows) != n or len({r for r, _ in rows}) != 1 or [int(p) for _, p in rows] != list(range(s - n, s)):
                return unsupported(node, "window_rows' rows are not one request's last rows of its window", q0=q0, n=n, s=s, rows=list(rows))
            req = rows[0][0]
            try:
                kcs, vcs = as_slot_rows(kc), as_slot_rows(vc)
                slots = kv_slots(ctx, req, s, layer=layer)
            except GeometryError as e:
                return unsupported(node, str(e), request=req, s=s, layer=layer)
            G = kcs.shape[1]
            if kcs.shape[2] != D or vcs.shape != kcs.shape or H % G or len(kbits) != G or any(x >= kcs.shape[0] for x in slots):
                return unsupported(node, "KV cache geometry / K scales do not match q", k_cache=repr(kcs), H=H, D=D, k_scales=len(kbits))
            if kq.shape != (G, s, D) or vq.shape != (G, s, D) or sv.shape != (G, D) or v0.shape != (G, D) or not all(is_row_major(v) for v in kouts):
                return unsupported(node, "kv_quant's results are not row-major [G, s, D] / [G, D]", kv_quant=[repr(o) for o in kouts])
            if not (q.dtype == kc.dtype == vc.dtype == out3.dtype == "bfloat16" and kq.dtype == vq.dtype == "int8" and sv.dtype == v0.dtype == "float32"):
                return unsupported(node, "Int8KvQuant_v1 / Int8LutAttentionRow_v1 are bf16 -> (int8, f32) -> bf16",
                                   dtypes=[q.dtype, kc.dtype, vc.dtype, out3.dtype, kq.dtype, vq.dtype, sv.dtype, v0.dtype])
            if not math.isfinite(scale) or scale <= 0:
                return unsupported(node, "softmax scale is not a positive finite value", scale=scale)
            head = lambda c, g: _sub(c, g * c.strides[1], (c.shape[0], D), (c.strides[0], c.strides[2]))  # noqa: E731
            fkv = bind(int8_exact.Int8KvQuant, S=s, D=D)
            for g in range(G):
                computes.append(Compute(fkv, 1, (GatherMany(head(kcs, g), (slots,)), GatherMany(head(vcs, g), (slots,)), Const(32, kbits[g], "k_scale")),
                                        (Shared(_slab(kq, g)), Shared(_slab(vq, g)), Shared(_slab(sv, g)), Shared(_slab(v0, g))), (),
                                        hint=f"{hint}/kv_quant", kernels=("elementwise_kernel", "reduce_kernel"), attribution=(rows[-1],),
                                        pattern=self.name))
            for i in range(n):
                p = s - n + i + 1
                fr = bind(int8_exact.Int8LutAttentionRow, D=D, S=p, SCALE=scale)
                qi, di = _slab(q, i), _slab(dst, i)
                for hd in range(H):
                    g = hd // (H // G)
                    computes.append(Compute(fr, 1, (Shared(_slab(qi, hd)), Shared(_keys(kq, g, p)), Const(32, kbits[g], "k_scale"),
                                                    Shared(_keys(vq, g, p)), Shared(_slab(sv, g)), Shared(_slab(v0, g))), (Shared(_slab(di, hd)),), (),
                                            hint=hint, kernels=("_int_mm", "elementwise_kernel", "reduce_kernel"), attribution=(rows[i],),
                                            pattern=self.name))
        primary = dataclasses.replace(computes[0], consumed=tuple(sorted(node.ids())))
        return _group_compute(primary, before=tuple(before), also=tuple(computes[1:]))


def _out0(node: TreeNode) -> ViewDesc | None:
    o = node.outs
    o = o[0] if isinstance(o, (list, tuple)) and o else o
    return o if isinstance(o, ViewDesc) else None


def _src_of(copy: TreeNode) -> ViewDesc | None:
    a = copy.args
    return a[1] if len(a) > 1 and isinstance(a[1], ViewDesc) else None


__all__ = ["Int8Linear", "Int8AttentionScope", "header_k_scales"]
