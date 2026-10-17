"""The exact-int8 profile's two calls (`TargetProfile.int8_exact`, `engine.int8_ops`): the int8 linear and the int8 attention."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, array_of
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


class Int8LinearRule(OpRule):
    """`verity_int8::linear(x bf16 [M, K], w int8 [N, K], s_w f32 [N, 1] or [N]) -> (xq, s_a, y)` -> `int8_linear{K,N}`
    (Int8Linear_v1) per token row; all three results are values (`xq` s8 [M, K], `s_a` f32 [M], `y` bf16 [M, N])."""
    modes = frozenset({"profile"})
    info = RuleInfo("Int8Linear[vllm]", 1, "op-binding",
                    applies_to="verity_int8::linear(Tensor x, Tensor w, Tensor s_w) -> (Tensor, Tensor, Tensor)",
                    preconditions=("x is a bf16 value [M, K]", "w is an int8 parameter [N, K]", "s_w is a float32 parameter [N, 1] or [N]"),
                    evidence_domain="structural; numerics: Int8Linear_v1 (engine.int8_exact's plain linear, bit-equal on the RTX PRO 6000 "
                                    "to its CPU run)")

    def targets(self):
        return _int8_op("linear")

    def applies(self, node, ctx):
        if len(node.args) != 3:
            return "unexpected operator arity"
        x, w, sw = node.args
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if isinstance(ctx.val(x), Alloc):
            return "x is an unwritten allocation"
        xs, ws, ss = shape(ctx, x), shape(ctx, w), shape(ctx, sw)
        if len(xs) != 2 or ws is None or len(ws) != 2 or ws[1] != xs[1]:
            return f"shape: x {xs} vs w {ws}"
        if ss not in ((ws[0], 1), (ws[0],)):
            return f"shape: s_w {ss} vs N={ws[0]}"
        if xs[1] % 32:
            return f"shape: K={xs[1]}; Int8Linear_v1's S8Matmul_v1 takes K a multiple of 32"
        if not dtype_is(ctx, w, torch.int8) or not dtype_is(ctx, sw, torch.float32):
            return f"numerics_unregistered: w {ctx.dtype_of(w)}, s_w {ctx.dtype_of(sw)}; Int8Linear_v1 reads int8 weights and float32 scales"
        if ctx.weight_of(w) is None or ctx.weight_of(sw) is None:
            return "w / s_w are not parameters; Int8Linear_v1 takes the checkpoint's int8 rows and their scales as weight Inputs"
        return True

    def emit(self, node, ctx):
        x, w, sw = node.args
        N, K = shape(ctx, w)
        rows, lead = rows_of(ctx.coll(x))
        wc, sc = ctx.coll(w), ctx.coll(sw).reshape(Array(N, Value(32)))
        fn = ctx.kind("int8_linear", node, K=K, N=N)
        outs = [ctx.call(fn, r, wc, sc, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, (from_rows([o[0] for o in outs], lead), array_of([o[1] for o in outs]), from_rows([o[2] for o in outs], lead)))
        ctx.bind_weight_role(w, self, "int8_w", K=K, N=N)
        ctx.bind_weight_role(sw, self, "int8_ws", N=N)
        ctx.record(self, node, {"K": K, "N": N, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class Int8UnifiedAttentionRule(UnifiedAttentionWithOutputRule):
    """`vllm::unified_attention_with_output` on a layer whose OBSERVED impl is the int8 backend's (`INT8_ATTENTION_IMPL`): the call's
    rows extend the layer's history as in `UnifiedAttentionWithOutput[vllm]`, and the call is ONE `int8_attention{N,S,H,G,D,SCALE}`
    (Int8LutAttentionWindow_v1) over the window: its T query rows and the S = P0 + T keys so far, with the layer's K scales (the
    impl's, read back when the layer is instantiated: `vllm_meta.attention_impls`' `int8_k_scales`) as one Const32 per KV head."""
    info = RuleInfo("Int8UnifiedAttentionWithOutput[vllm]", 1, "state",
                    applies_to="vllm.unified_attention_with_output(query, key, value, Tensor! output, layer_name: str, None, None, kv_cache_dummy_dep?)",
                    preconditions=(f"the layer's OBSERVED attention impl is {INT8_ATTENTION_IMPL}, with its K scales (one per KV head) and "
                                   "float softmax scale read back",
                                   "q [T, NH, D], k, v [T, KVH, D] bf16; NH % KVH == 0", "output is an unwritten allocation of q's shape",
                                   "assumed (op contract): causal attention over the history + this call's rows"),
                    evidence_domain="structural; numerics: Int8LutAttentionWindow_v1 (engine.int8_attention, bit-equal on the RTX PRO 6000 "
                                    "to its CPU run and to fattn_ref.py)")

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
            return f"numerics_unregistered: a soft cap or sliding window on {layer_name!r}; Int8LutAttentionWindow_v1 is plain causal"
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
        S = h.P0 + h.T
        fn = ctx.kind("int8_attention", node, N=h.T, S=S, H=h.NH, G=h.KVH, D=h.D, SCALE=scale)
        hint = f"{ctx.hint_name(node)}[t={h.P0}..{S - 1}]"
        ks = array_of([ctx.call(ctx.kind("const", node, W=32, BITS=b), node=node, rule=self, name=f"{hint}/k_scale{g}") for g, b in enumerate(bits)])
        kv = Array(S, Array(h.KVH, Array(h.D, Value(16))))
        y = ctx.call(fn, h.qrows, Coll(kv, h.klog.cut(S)), Coll(kv, h.vlog.cut(S)), ks, node=node, rule=self, name=hint)
        ctx.rebind(out, y.reshape(_tensor_type(h.qs, 16)), self, what="output")
        ctx.record(self, node, {"layer_name": layer_name, "positions": [h.P0, S - 1], "NH": h.NH, "KVH": h.KVH, "D": h.D, "scale": scale,
                                "k_scales": [f"0x{b:08x}" for b in bits], "kv_write_consumed": dep is not None, "assumed_causal": True,
                                "impl": obs.get("impl")},
                   emitted=f"{fn.id} x1 (S={S}) + Const32 x{len(bits)}")


__all__ = ["Int8LinearRule", "Int8UnifiedAttentionRule", "INT8_ATTENTION_IMPL"]
