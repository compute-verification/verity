"""The exact-int8 profile's calls (`TargetProfile.int8_exact`, `engine.int8_ops`): the int8 linear's quantizer and product, and the int8
attention."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc, refuse
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.attention import UnifiedAttentionWithOutputRule
from verity_vllm.program.frontend.rules.vllm_bindings.observations import ATTN_PINS, _OBSERVED, _norm_attn, attn_kind_for
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _row_value
from verity_vllm.program.frontend.torch_frontend import _tensor_type

#: `engine.int8_attention_vllm.Int8AttentionImpl`, by name: the frontend reads the instantiated layers, never the engine
INT8_ATTENTION_IMPL = "Int8AttentionImpl"
#: the FA2 pins (`ATTN_PINS`) the int8 window keeps: causal decoder attention, an unquantized cache, no ALiBi / sinks / context parallelism
INT8_ATTN_PINS = ("attn_type", "kv_cache_dtype", "alibi_slopes", "sinks", "dcp_world_size", "pcp_world_size")


def _int8_op(name: str) -> set:
    try:
        return {getattr(torch.ops.verity_int8, name).default}
    except (AttributeError, RuntimeError):     # the ops exist once `engine.int8_ops` is imported (the int8 method and backend import it)
        return set()


class Int8RowQuantRule(OpRule):
    """`verity_int8::row_quant(x bf16 [M, K]) -> (xq s8 [M, K], s_a f32 [M])` -> per token row `int8_row_quant{K}` (Int8RowQuant_v1).
    Its results are read by the linear's product (`Int8ScaledMmRule`), the next node of the same body: values the two exchange, no
    return slot of the linear."""
    modes = frozenset({"profile"})
    info = RuleInfo("Int8RowQuant[vllm]", 1, "op-binding",
                    applies_to="verity_int8::row_quant(Tensor x) -> (Tensor, Tensor)",
                    preconditions=("x is a bf16 value [M, K], K a multiple of 32",),
                    evidence_domain="structural; numerics: Int8RowQuant_v1 (engine.int8_exact.quantize_rows, bit-equal on the RTX PRO 6000 to its CPU run)")

    def targets(self):
        return _int8_op("row_quant")

    def applies(self, node, ctx):
        if len(node.args) != 1:
            return "unexpected operator arity"
        (x,) = node.args
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if isinstance(ctx.val(x), Alloc):
            return "x is an unwritten allocation"
        xs = shape(ctx, x)
        if len(xs) != 2:
            return f"shape: x {xs} is not [M, K]"
        if xs[1] % 32:
            return f"shape: K={xs[1]}; ScaledMmS8Coordinate_v1 takes K a multiple of 32 (its IMMA k step)"
        return True

    def emit(self, node, ctx):
        (x,) = node.args
        rows, lead = rows_of(ctx.coll(x))
        K = shape(ctx, x)[1]
        fq = ctx.kind("int8_row_quant", node, K=K)
        hint = ctx.hint_name(node)
        qs = [ctx.call(fq, r, node=node, rule=self, name=f"{hint}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, (from_rows([q[0] for q in qs], lead), array_of([q[1] for q in qs])))
        ctx.record(self, node, {"K": K, "rows": len(rows)}, emitted=f"{fq.id} x{len(rows)}")


class Int8ScaledMmRule(OpRule):
    """`verity_int8::scaled_mm(xq s8 [M, K], s_a f32 [M], w int8 [N, K], s_w f32 [N, 1] or [N]) -> y bf16 [M, N]` -> per token row
    `int8_scaled_mm{K,N}` (ScaledMmS8_v1, ScaledMmS8Coordinate_v1 per output), so the query of record cuts one coordinate of the
    product instead of its whole fan-in."""
    modes = frozenset({"profile"})
    info = RuleInfo("Int8ScaledMm[vllm]", 1, "op-binding",
                    applies_to="verity_int8::scaled_mm(Tensor xq, Tensor s_a, Tensor w, Tensor s_w) -> Tensor",
                    preconditions=("xq is an int8 value [M, K], s_a a float32 value [M]", "w is an int8 parameter [N, K]",
                                   "s_w is a float32 parameter [N, 1] or [N]"),
                    evidence_domain="structural; numerics: Int8RowQuant_v1 + ScaledMmS8_v1 = Int8Linear_v1 (engine.int8_exact's plain linear, bit-equal on the RTX PRO 6000 "
                                    "to its CPU run)")

    def targets(self):
        return _int8_op("scaled_mm")

    def applies(self, node, ctx):
        if len(node.args) != 4:
            return "unexpected operator arity"
        xq, sa, w, sw = node.args
        r = _row_value(ctx, xq, "xq", torch.int8) or _row_value(ctx, sa, "s_a", torch.float32)
        if r:
            return r
        if isinstance(ctx.val(xq), Alloc) or isinstance(ctx.val(sa), Alloc):
            return "xq / s_a is an unwritten allocation"
        xs, as_, ws, ss = shape(ctx, xq), shape(ctx, sa), shape(ctx, w), shape(ctx, sw)
        if len(xs) != 2 or ws is None or len(ws) != 2 or ws[1] != xs[1]:
            return f"shape: xq {xs} vs w {ws}"
        if as_ != (xs[0],):
            return f"shape: s_a {as_} vs M={xs[0]}"
        if ss not in ((ws[0], 1), (ws[0],)):
            return f"shape: s_w {ss} vs N={ws[0]}"
        if xs[1] % 32:
            return f"shape: K={xs[1]}; ScaledMmS8Coordinate_v1 takes K a multiple of 32 (its IMMA k step)"
        if not dtype_is(ctx, w, torch.int8) or not dtype_is(ctx, sw, torch.float32):
            return f"numerics_unregistered: w {ctx.dtype_of(w)}, s_w {ctx.dtype_of(sw)}; Int8Linear_v1 reads int8 weights and float32 scales"
        if ctx.weight_of(w) is None or ctx.weight_of(sw) is None:
            return "w / s_w are not parameters; Int8Linear_v1 takes the checkpoint's int8 rows and their scales as weight Inputs"
        return True

    def emit(self, node, ctx):
        xq, sa, w, sw = node.args
        N, K = shape(ctx, w)
        rows, lead = rows_of(ctx.coll(xq))
        scales = ctx.coll(sa).reshape(Array(len(rows), Value(32)))
        wc, sc = ctx.coll(w), ctx.coll(sw).reshape(Array(N, Value(32)))
        fm = ctx.kind("int8_scaled_mm", node, K=K, N=N)
        hint = ctx.hint_name(node)
        ys = [ctx.call(fm, r, scales[i], wc, sc, node=node, rule=self, name=f"{hint}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(ys, lead))
        ctx.bind_weight_role(w, self, "int8_w", K=K, N=N)
        ctx.bind_weight_role(sw, self, "int8_ws", N=N)
        ctx.record(self, node, {"K": K, "N": N, "rows": len(rows)}, emitted=f"{fm.id} x{len(rows)}")


class Int8UnifiedAttentionRule(UnifiedAttentionWithOutputRule):
    """`vllm::unified_attention_with_output` on a layer whose OBSERVED impl is the int8 backend's (`INT8_ATTENTION_IMPL`): the call's
    rows extend the layer's history as in `UnifiedAttentionWithOutput[vllm]`, over the window of its T query rows and the S = P0 + T
    keys so far, as `engine.int8_ops.paged_attention` computes it: per KV head g, ONE `int8_kv_quant{S,D}` (Int8KvQuant_v1) over the
    head's S keys; then per query row i (position P0 + i) and head h, ONE `int8_attention_row{D,P0+i+1,SCALE}` (Int8LutAttentionRow_v1)
    over the first P0 + i + 1 rows of its KV head's Int8KvQuant_v1.  The layer's K scales (the impl's, read back when the layer is
    instantiated: `vllm_meta.attention_impls`' `int8_k_scales`) enter as one Const32 before each Call that reads one, as the Match
    fold emits them (`observe.fold.patterns.int8`).  Under `registered_values` each row is `int8_attention_row_table`
    (Int8LutAttentionRow_v2), reading the registered `exp2_lut127` the rule declares when a layer's observed impl is the int8 one."""
    info = RuleInfo("Int8UnifiedAttentionWithOutput[vllm]", 2, "state",
                    applies_to="vllm.unified_attention_with_output(query, key, value, Tensor! output, layer_name: str, None, None, kv_cache_dummy_dep?)",
                    preconditions=(f"the layer's OBSERVED attention impl is {INT8_ATTENTION_IMPL}, with its K scales (one per KV head) and "
                                   "float softmax scale read back",
                                   "q [T, NH, D], k, v [T, KVH, D] bf16; NH % KVH == 0", "output is an unwritten allocation of q's shape",
                                   "assumed (op contract): causal attention over the history + this call's rows"),
                    evidence_domain="structural; numerics: Int8KvQuant_v1 + Int8LutAttentionRow_v1 (engine.int8_ops' kv_quant_batch and "
                                    "window_rows_batch, bit-equal on the RTX PRO 6000 to its CPU run and to fattn_ref.py)")

    def registered_tables(self, ep, mode):
        ts = self.targets()
        layers = (self._args(n)[4] for n in ep.graph.nodes if n.op == "call_function" and n.target in ts)
        return ("exp2_lut127",) if any(isinstance(ln, str) and self.impl_refusal(ln) is None for ln in layers) else ()

    def impl_refusal(self, layer_name: str) -> str | None:
        obs = (_OBSERVED["attention_impls"] or {}).get(layer_name) or {}
        if not str(obs.get("impl", "")).endswith(f".{INT8_ATTENTION_IMPL}"):
            return f"the layer's observed impl {obs.get('impl')!r} is not {INT8_ATTENTION_IMPL}"
        if not obs.get("int8_k_scales"):
            return "the layer's K scales were not read back"
        if not isinstance(obs.get("scale"), float):
            return f"the layer has no observed float softmax scale ({obs.get('scale')!r})"
        role, _, window = attn_kind_for(obs)
        if role != "attention" or window is not None:
            return f"numerics_unregistered: a soft cap or sliding window on {layer_name!r}; Int8LutAttentionRow_v1 is plain causal"
        for k in INT8_ATTN_PINS:
            if _norm_attn(obs.get(k, "<absent>")) != _norm_attn(ATTN_PINS[k]):
                return f"numerics_unregistered: attention {k}={obs.get(k, '<absent>')!r} on {layer_name!r}, not {ATTN_PINS[k]!r}"
        return None

    def emit(self, node, ctx):
        q, k, v, out, layer_name, _, _, dep = self._args(node)
        h = self.extend_history(node, ctx)
        obs = _OBSERVED["attention_impls"][layer_name]
        bits, scale = [int(b) for b in obs["int8_k_scales"]], obs["scale"]
        if len(bits) != h.KVH:
            refuse("shape", node, h.mp, f"{len(bits)} K scales for {h.KVH} KV heads")
        S, D, rep = h.P0 + h.T, h.D, h.NH // h.KVH
        from verity.ml.definitions.registry.int8_exact import row_refusal

        if why := row_refusal({"D": D, "S": S}):
            refuse("shape", node, h.mp, f"Int8LutAttentionRow_v1 {why}")
        hint = ctx.hint_name(node)
        kv = Array(S, Array(h.KVH, Array(D, Value(16))))
        klog, vlog = Coll(kv, h.klog.cut(S)), Coll(kv, h.vlog.cut(S))

        def k_scale(g, name):
            return ctx.call(ctx.kind("const", node, W=32, BITS=bits[g]), node=node, rule=self, name=f"{name}/k_scale")

        fkv = ctx.kind("int8_kv_quant", node, S=S, D=D)
        kvq = []
        for g in range(h.KVH):
            name = f"{hint}/kv_quant[g={g},s={S}]"
            head = [Coll(Array(S, Array(D, Value(16))), concat([log[j][g].refs for j in range(S)])) for log in (klog, vlog)]
            kvq.append(ctx.call(fkv, head[0], head[1], k_scale(g, name), node=node, rule=self, name=name))
        rows = []
        table = (ctx.table("exp2_lut127", node),) if ctx.registered_values else ()
        for i in range(h.T):
            n = h.P0 + i + 1
            fr = ctx.kind("int8_attention_row_table" if table else "int8_attention_row", node, D=D, S=n, SCALE=scale)
            for hd in range(h.NH):
                g, name = hd // rep, f"{hint}[t={h.P0 + i},h={hd}]"
                c = kvq[g]
                rows.append(ctx.call(fr, h.qrows[i][hd], c[0][0:n], k_scale(g, name), c[1][0:n], c[2], c[3], *table, node=node, rule=self,
                                     name=name))
        y = Coll(_tensor_type(h.qs, 16), concat([r.refs for r in rows]))
        ctx.rebind(out, y, self, what="output")
        ctx.record(self, node, {"layer_name": layer_name, "positions": [h.P0, S - 1], "NH": h.NH, "KVH": h.KVH, "D": D, "scale": scale,
                                "k_scales": [f"0x{b:08x}" for b in bits], "kv_write_consumed": dep is not None, "assumed_causal": True,
                                "impl": obs.get("impl")},
                   emitted=f"{fkv.id} x{h.KVH} + Int8LutAttentionRow_v{2 if table else 1}{{S={h.P0 + 1}..{S}}} x{h.T * h.NH} + Const32 "
                           f"x{h.KVH + h.T * h.NH}")


__all__ = ["Int8RowQuantRule", "Int8ScaledMmRule", "Int8UnifiedAttentionRule", "INT8_ATTENTION_IMPL"]
