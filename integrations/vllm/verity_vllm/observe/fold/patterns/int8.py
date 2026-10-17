"""Eager Match patterns of the exact-int8 profile (`TargetProfile.int8_exact`): its two named ops (`engine.int8_ops`), each ONE
registered Definition (`program.registry.int8_exact`), whatever kernels the op launches inside:

    verity_int8::linear(x[M,K] bf16, w[N,K] int8, s_w[N,1] f32) -> (xq[M,K] s8, s_a[M] f32, y[M,N] bf16)
        -> Compute(Int8Linear_v1{K,N}, rows=M, (PerRow(x), Shared(w), Shared(s_w)), (PerRow(xq), PerRow(s_a), PerRow(y)))
    vllm.unified_attention_with_output(query, ..., output!, layer) ⊃ one verity_int8::window_attention(q, key_cache, value_cache,
                                                                      blocks, s, k_scales, scale) per request, each copied into output
        -> per request Compute(Int8LutAttentionWindow_v1{N,S,H,G,D,SCALE}, rows=1, (GatherMany(q rows), GatherMany(K slots 0..s-1),
                               GatherMany(V slots), Const32 x G), (GatherMany(output rows),))

The K scales are the run header's (`RunHeader.profile["backends"]["int8_k_scales"]`, read back from the layers by
`kernel_readback.int8_k_scales_of`), never read from device memory; each request's window writes its rows of `output` directly
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


def _rows(v: ViewDesc, n: int) -> GatherMany:
    return GatherMany(v, (tuple(range(n)),))


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
            return unsupported(node, "Int8Linear_v1's S8Matmul_v1 takes K a multiple of 32", K=K)
        if not (x.dtype == y.dtype == "bfloat16" and w.dtype == xq.dtype == "int8" and sw.dtype == s_a.dtype == "float32"):
            return unsupported(node, "Int8Linear_v1 is bf16 x (int8, f32) -> (int8, f32, bf16)",
                               dtypes=[x.dtype, w.dtype, sw.dtype, xq.dtype, s_a.dtype, y.dtype])
        wroots, sroots = ctx.roots_of(w), ctx.roots_of(sw)
        if not any(r.role == "weight" for r in wroots) or not any(r.role == "weight" for r in sroots):
            return unsupported(node, "w / s_w are not checkpoint parameters (weight Roots)", w=repr(w), s_w=repr(sw))
        wname, sname = wroots[0].name, sroots[0].name
        return Compute(bind(int8_exact.Int8Linear, K=K, N=N), M, (PerRow(x), Shared(w), Shared(sw)), (PerRow(xq), PerRow(s_a), PerRow(y)),
                       tuple(sorted(node.ids())), hint=hint_of(node, wname), kernels=("_int_mm", "elementwise_kernel", "reduce_kernel"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name, roots=(None, wname, sname))


@dataclass
class Int8AttentionScope:
    """`unified_attention_with_output` around `verity_int8::window_attention` calls (the int8 backend, `engine.int8_ops.paged_attention`):
    one Compute per call.  A call's q is a slice of the scope's query rows (one request's: its window's last rows), its window the
    request's first `s` KV slots under the step's block table; its result is copied into the same rows of the scope's output."""

    name: str = "int8.window_attention"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_attention_with_output":
            return None
        wins = find_descendants(node, lambda n: _is_op(n, "window_attention"))
        if not wins:
            return None
        from verity_vllm.program.registry import int8_exact
        sa = bound_args(node)
        query, output = sa.get("query"), sa.get("output")
        layer = kv_layer_of(node) or ""
        if not isinstance(query, ViewDesc) or not isinstance(output, ViewDesc) or query.rank != 3:
            return unsupported(node, "unified_attention_with_output query / output not bound as [T, H, D] views", names=list(sa))
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
                          if _src_of(c) is not None and _out0(w) is not None and _src_of(c).alloc == _out0(w).alloc] for w in wins}
        rcs = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "reshape_and_cache_flash")
        keep = {w.id for w in wins} | {c.id for cs in copies.values() for c in cs} | {n.id for n in rcs}
        before, bad = scope_transparents(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        for rc in rcs:
            moves = _kv_moves(rc, ctx, self.name, (), layer=layer)
            if isinstance(moves, Unsupported):
                return moves
            before.extend(moves)
        computes: list[Compute] = []
        for w in wins:
            a = bound_args(w)
            q, kc, vc, s, scale = a.get("q"), a.get("key_cache"), a.get("value_cache"), a.get("s"), as_float(a.get("scale"))
            if not all(isinstance(v, ViewDesc) for v in (q, kc, vc)) or not isinstance(s, int) or scale is None:
                return unsupported(node, "window_attention arguments not bound as expected", names=list(a))
            if q.alloc != query.alloc or q.rank != 3 or q.shape[1:] != (H, D) or (q.offset - query.offset) % query.strides[0]:
                return unsupported(node, "window_attention's q is not a row slice of the scope's query", q=repr(q), query=repr(query))
            if len(copies[id(w)]) != 1:
                return unsupported(node, "window_attention's result is not copied once into the output", copies=len(copies[id(w)]))
            dst = copies[id(w)][0].args[0] if copies[id(w)][0].args else None
            n, q0 = q.shape[0], (q.offset - query.offset) // query.strides[0]
            if not isinstance(dst, ViewDesc) or dst.alloc != out3.alloc or dst.offset != out3.offset + q0 * out3.strides[0] or dst.shape != (n, H, D):
                return unsupported(node, "window_attention's result is not copied into its own rows of the output", dst=repr(dst), q0=q0, n=n)
            rows = ctx.start.rows[q0:q0 + n]
            if len(rows) != n or len({r for r, _ in rows}) != 1 or [int(p) for _, p in rows] != list(range(s - n, s)):
                return unsupported(node, "window_attention's rows are not one request's last rows of its window", q0=q0, n=n, s=s, rows=list(rows))
            req = rows[0][0]
            try:
                kcs, vcs = as_slot_rows(kc), as_slot_rows(vc)
                slots = kv_slots(ctx, req, s, layer=layer)
            except GeometryError as e:
                return unsupported(node, str(e), request=req, s=s, layer=layer)
            G = kcs.shape[1]
            if kcs.shape[2] != D or vcs.shape != kcs.shape or H % G or len(kbits) != G or any(x >= kcs.shape[0] for x in slots):
                return unsupported(node, "KV cache geometry / K scales do not match q", k_cache=repr(kcs), H=H, D=D, k_scales=len(kbits))
            if not (q.dtype == kc.dtype == vc.dtype == out3.dtype == "bfloat16"):
                return unsupported(node, "Int8LutAttentionWindow_v1 is bf16", dtypes=[q.dtype, kc.dtype, vc.dtype, out3.dtype])
            if not math.isfinite(scale) or scale <= 0:
                return unsupported(node, "softmax scale is not a positive finite value", scale=scale)
            spec = bind(int8_exact.Int8LutAttentionWindow, N=n, S=s, H=H, G=G, D=D, SCALE=scale)
            computes.append(Compute(spec, 1, (_rows(q, n), GatherMany(kcs, (slots,)), GatherMany(vcs, (slots,)), Const(32, kbits, "k_scale")),
                                    (_rows(dst, n),), (), hint=hint_of(node, str(layer) or "attn"), kernels=("elementwise_kernel", "reduce_kernel"),
                                    attribution=(rows[-1],), pattern=self.name))
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
