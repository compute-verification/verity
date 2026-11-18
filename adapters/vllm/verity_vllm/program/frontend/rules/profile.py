"""Profile-mode rules for vLLM's ops in the vLLM eager profile (B0/B1): vLLM-style modules -> TA1's registered kinds.

In-place rules bind the kernel interface (`rms_norm`, `fused_add_rms_norm`, `rotary_embedding`, `silu_and_mul`) and rebind the
mutated operands explicitly; the state rule binds `unified_attention` and makes the KV dependencies explicit. The rules on ATen
ops are `verity_torch.frontend.rules.profile`'s, and `PROFILE_RULES` is both lists, the ATen rules first.
"""

from __future__ import annotations

from verity.circuits.refs import Coll, array_of
from verity.circuits.types import Array, Value
from verity_torch.frontend.derive import Alloc, StaticIndex, refuse
from verity_torch.frontend.rules.base import InPlaceRule, RuleInfo, StateRule
from verity_torch.frontend.rules.common import (
    and_mul_operands,
    attention_scale,
    from_rows,
    known,
    kv_logs,
    norm_kind,
    refuse_cached_prefix,
    row_value,
    rows_of,
    shape,
)
from verity_torch.frontend.rules.profile import (
    PROFILE_TORCH_RULES,
    _without_batch_invariance,
)
from verity_torch.frontend.strided_views import _tensor_type
from verity_torch.frontend.value_types import shape_of

from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.targets import UnregisteredTarget

# -- in-place kernel interface --------------------------------------------------------------------------------


class RmsNormRule(InPlaceRule):
    """`rms_norm(out!, x, w, eps)` -> `RMSNormTriton{N,EPS}` per row; `out` (an allocation) is rebound."""
    modes = frozenset({"profile"})
    info = RuleInfo("RmsNorm[vllm]", 1, "in-place-binding", applies_to="_C.rms_norm(Tensor! out, x, weight, eps)",
                    preconditions=("out is an unwritten allocation (or a value) of x's shape", "x is bf16 [.., N], weight is bf16 [N]",
                                   "eps is a Python float (becomes the EPS static)", "weight is not written", "the target is batch-invariant"),
                    evidence_domain="structural; numerics: RMSNormTriton_v1 (vLLM batch-invariant Triton _rms_norm_kernel)")

    def targets(self):
        return targets_for("rms_norm")

    def applies(self, node, ctx):
        if (r := _without_batch_invariance("_C.rms_norm", "the batch-invariant Triton _rms_norm_kernel RMSNormTriton_v1 states")):
            return r
        out, x, w, eps = node.args[:4]
        r = row_value(ctx, x, "x") or row_value(ctx, w, "weight")
        if r:
            return r
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "out is not an allocation/value"
        if shape(ctx, w) != (known(shape(ctx, x))[-1],):
            return f"weight {shape(ctx, w)} vs hidden {known(shape(ctx, x))[-1]}"
        if shape(ctx, out) != shape(ctx, x):
            return "out shape differs from x"
        if not isinstance(eps, float):
            return "eps is not a static float"
        return True

    def emit(self, node, ctx):
        out, x, w, eps = node.args[:4]
        xc, wc = ctx.coll(x), ctx.coll(w)
        rows, lead = rows_of(xc)
        N = rows[0].type.n
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import (
            active_target_profile,  # vllm_bindings imports this module
        )
        fn, extra = norm_kind(ctx, "rmsnorm", node, N, eps, TARGET=active_target_profile(), M=len(rows))
        outs = [ctx.call(fn, r, wc, *extra, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.bind_weight_role(w, self, "rmsnorm.weight", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class FusedAddRmsNormRule(InPlaceRule):
    """`fused_add_rms_norm(x!, res!, w, eps)` -> `RMSNormFusedCuda{N,EPS}` per row: x := normed, res := x + res."""
    modes = frozenset({"profile"})
    info = RuleInfo("FusedAddRmsNorm[vllm]", 1, "in-place-binding", applies_to="_C.fused_add_rms_norm(Tensor! x, Tensor! residual, weight, eps)",
                    preconditions=("x and residual are bf16 values of the same shape [.., N]", "weight is bf16 [N]", "eps is a static float",
                                   "both mutated operands are rebound; their other aliases are invalidated"),
                    evidence_domain="structural; numerics: RMSNormFusedCuda_v2 (vLLM CUDA generic fused_add_rms_norm_kernel)")

    def targets(self):
        return targets_for("fused_add_rms_norm")

    def applies(self, node, ctx):
        x, res, w, eps = node.args[:4]
        r = row_value(ctx, x, "x") or row_value(ctx, res, "residual") or row_value(ctx, w, "weight")
        if r:
            return r
        if isinstance(ctx.val(x), Alloc) or isinstance(ctx.val(res), Alloc):
            return "x/residual is an unwritten allocation"
        if shape(ctx, x) != shape(ctx, res):
            return "x and residual differ in shape"
        if shape(ctx, w) != (known(shape(ctx, x))[-1],):
            return "weight shape vs hidden size"
        if not isinstance(eps, float):
            return "eps is not a static float"
        return True

    def emit(self, node, ctx):
        x, res, w, eps = node.args[:4]
        xr, lead = rows_of(ctx.coll(x))
        rr, _ = rows_of(ctx.coll(res))
        wc = ctx.coll(w)
        N = xr[0].type.n
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import (
            active_target_profile,  # vllm_bindings imports this module
        )
        try:
            fn, extra = norm_kind(ctx, "rmsnorm_fused", node, N, eps, TARGET=active_target_profile(), M=len(xr))
        except UnregisteredTarget as e:
            refuse("numerics_unregistered", node, ctx.module_path(node), f"the declared target has no registered fused_add_rms_norm kind: {e}")
        ns, rs = [], []
        for i, (a, b) in enumerate(zip(xr, rr)):
            o = ctx.call(fn, a, b, wc, *extra, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            ns.append(o[0])
            rs.append(o[1])
        ctx.rebind(x, from_rows(ns, lead), self, what="x")
        ctx.rebind(res, from_rows(rs, lead), self, what="residual")
        ctx.bind_weight_role(w, self, "rmsnorm.weight", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "rows": len(xr)}, emitted=f"{fn.id} x{len(xr)}")


class RotaryEmbeddingRule(InPlaceRule):
    """`rotary_embedding(positions, q!, k!, head_size, cos_sin_cache, is_neox)` -> `RoPE{NHEADS,D}` per row for q
    and for k, with the cos/sin row `cos_sin_cache[positions[t]]` (static positions) as the table Input."""
    modes = frozenset({"profile"})
    info = RuleInfo("RotaryEmbedding[vllm]", 1, "in-place-binding",
                    applies_to="_C.rotary_embedding(positions, Tensor! query, Tensor!? key, head_size, cos_sin_cache, is_neox)",
                    preconditions=("positions is static structure (StaticIndex): the computation never depends on data",
                                   "q, k are bf16 [T, NHEADS*D] (or [T, NHEADS, D]) with D == head_size",
                                   "cos_sin_cache is a bf16 value [MAX_POS, D] — a registered buffer (Input with recorded derivation) or a value",
                                   "is_neox is True (RoPEHead is the NeoX rotation)", "every position < MAX_POS"),
                    evidence_domain="structural; numerics: RoPE_v1 (vLLM _C.rotary_embedding NeoX), cos/sin row as Input (COS_SIN_QUALIFICATION)")

    def targets(self):
        return targets_for("rotary_embedding")

    def applies(self, node, ctx):
        pos, q, k, head_size, cache, is_neox = node.args[:6]
        if not isinstance(ctx.val(pos), StaticIndex):
            return "positions is not static structure (a computed value would make the computation data-dependent)"
        r = row_value(ctx, q, "query") or row_value(ctx, cache, "cos_sin_cache")
        if r:
            return r
        if k is not None:
            r = row_value(ctx, k, "key")
            if r:
                return r
        if not is_neox:
            return "numerics_unregistered: GPT-J (interleaved) rotation has no registered kind"
        cs = known(shape(ctx, cache))
        if len(cs) != 2 or cs[1] != head_size:
            return f"cos_sin_cache {cs} vs head_size {head_size}"
        qs = known(shape(ctx, q))
        if qs[-1] % head_size and (len(qs) < 2 or qs[-1] != head_size):
            return f"query last dim {qs[-1]} not a multiple of head_size"
        T = ctx.val(pos).shape[0] if ctx.val(pos).shape else 1
        if qs[0] != T:
            return f"positions has {T} entries for {qs[0]} rows"
        return True

    def emit(self, node, ctx):
        pos, q, k, head_size, cache, is_neox = node.args[:6]
        positions = ctx.static_index(pos).flat()
        table = ctx.coll(cache)
        MAX_POS = table.type.n
        for p in positions:
            if not (0 <= p < MAX_POS):
                refuse("shape", node, ctx.module_path(node), f"position {p} outside the cos/sin table ({MAX_POS} rows)")
        D = head_size
        w_cache = ctx.weight_of(cache)

        def rotate(t, what):
            c = ctx.coll(t)
            shp, wd = shape_of(c.type)
            T = shp[0]
            nh = (shp[1] * (shp[2] if len(shp) > 2 else 1)) // D if len(shp) > 1 else shp[-1] // D
            flat = c.reshape(Array(T, Array(nh, Array(D, Value(wd)))))
            fn = ctx.kind("rope", node, NHEADS=nh, D=D)
            outs = [ctx.call(fn, flat[i], table[positions[i]], node=node, rule=self, name=f"{ctx.hint_name(node)}/{what}[{i}]") for i in range(T)]
            ctx.rebind(t, array_of(outs).reshape(c.type), self, what=what)
            return fn.id, T, nh

        qi = rotate(q, "query")
        ki = rotate(k, "key") if k is not None else None
        if w_cache is not None:
            ctx.bind_weight_role(cache, self, "rope.cos_sin_cache", D=D, MAX_POS=MAX_POS)
        ctx.record(self, node, {"positions": positions[:64], "n_positions": len(positions), "D": D, "is_neox": True,
                                "cos_sin_is_buffer": w_cache.name if w_cache else None,
                                "cos_sin_derivation": "module buffer computed at init (vLLM _compute_cos_sin_cache); COS_SIN_QUALIFICATION applies"},
                   emitted=f"{qi[0]} x{qi[1]}" + (f" + {ki[0]} x{ki[1]}" if ki else ""))


class SiluAndMulRule(InPlaceRule):
    """`silu_and_mul(out!, x)` -> `SiluMul{I}` per row; `out` (an allocation of [.., I]) is rebound."""
    modes = frozenset({"profile"})
    info = RuleInfo("SiluAndMul[vllm]", 1, "in-place-binding", applies_to="_C.silu_and_mul(Tensor! out, x[.., 2I])",
                    preconditions=("x is bf16 [.., 2I]", "out is an unwritten allocation (or value) [.., I]"),
                    evidence_domain="structural; numerics: the vocabulary's silu_mul kind (b1-eager-v3, dense2-eager-v1: SiluMul_v2; "
                                    "b1-eager, b1-eager-v2: SiluMul_v1; vLLM _C.silu_and_mul)")

    def targets(self):
        return targets_for("silu_and_mul")

    def applies(self, node, ctx):
        return and_mul_operands(node, ctx)

    def emit(self, node, ctx):
        out, x = node.args[:2]
        rows, lead = rows_of(ctx.coll(x))
        INTER = rows[0].type.n // 2
        fn = ctx.kind("silu_mul", node, I=INTER)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"I": INTER, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


# -- state: attention over the request's KV history ------------------------------------------------------------


class UnifiedAttentionRule(StateRule):
    """`unified_attention(q, k, v, layer_name)`: the KV cache is *implicit* state keyed by `layer_name` (vLLM's forward
    context). This rule makes it explicit (design §4.2): the rows `k`, `v` of this call are appended to the layer's
    history for this request, and the attention of the row at global position P reads `K[0:P+1]`, `V[0:P+1]` — the
    values produced by this and earlier calls of the same layer — as `Attention{T=P+1,NH,KVH,D}` arguments.
    Ownership: one history per layer_name; two layers sharing a name, or a name reused across requests, is refused.
    Causality is the op's contract (vLLM decoder attention); it is recorded as an assumed precondition."""
    modes = frozenset({"profile"})
    info = RuleInfo("UnifiedAttention[vllm]", 1, "state", applies_to="vllm.unified_attention(query, key, value, layer_name) -> Tensor",
                    preconditions=("q, k, v are bf16 [T, NH*D] / [T, KVH*D] (or [T, heads, D]) with NH % KVH == 0",
                                   "layer_name is a static string; its KV history is owned by exactly this layer",
                                   "the rows of this call extend the history at positions len(history)..len(history)+T-1",
                                   "assumed (op contract): causal attention over the history + this call's rows"),
                    evidence_domain="structural; numerics: Attention_v1 (FA2 causal GQA) per position")

    def targets(self):
        return targets_for("unified_attention")

    def applies(self, node, ctx):
        q, k, v, layer_name = node.args[:4]
        for nm, a in (("query", q), ("key", k), ("value", v)):
            r = row_value(ctx, a, nm)
            if r:
                return r
            if isinstance(ctx.val(a), Alloc):
                return f"{nm} is an unwritten allocation"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        qs, ks, vs = known(shape(ctx, q)), known(shape(ctx, k)), shape(ctx, v)
        if qs[0] != ks[0] or ks != vs:
            return f"row counts differ q{qs} k{ks} v{vs}"
        return True

    def emit(self, node, ctx):
        refuse_cached_prefix(ctx, node, self.rule_id)
        q, k, v, layer_name = node.args[:4]
        qc, kc, vc = ctx.coll(q), ctx.coll(k), ctx.coll(v)
        D = ctx.config.get("D") or ctx.allowed_dims.get("D")
        qs = shape_of(qc.type)[0]
        ks = shape_of(kc.type)[0]
        T = qs[0]
        if len(qs) == 3:
            NH, D_ = qs[1], qs[2]
        else:
            if D is None:
                refuse("shape", node, ctx.module_path(node), "head dim D is needed to split flat q/k/v rows: pass config={'D': ...} or 3-D q/k/v")
            NH, D_ = qs[1] // D, D
        if D is not None and D_ != D:
            refuse("shape", node, ctx.module_path(node), f"head dim {D_} vs config D={D}")
        D = D_
        KVH = (ks[1] * (ks[2] if len(ks) > 2 else 1)) // D
        if NH % KVH:
            refuse("shape", node, ctx.module_path(node), f"NH={NH} not a multiple of KVH={KVH}")
        qrows = qc.reshape(Array(T, Array(NH, Array(D, Value(16)))))
        krows = kc.reshape(Array(T, Array(KVH, Array(D, Value(16)))))
        vrows = vc.reshape(Array(T, Array(KVH, Array(D, Value(16)))))
        hist = ctx.state.setdefault("kv_history", {})
        owner = ctx.state.setdefault("kv_owner", {})
        mp = ctx.module_path(node)
        if layer_name in owner and owner[layer_name] != mp:
            refuse("side_input", node, mp, f"KV history {layer_name!r} is written from two different modules ({owner[layer_name]} and {mp}): "
                                           f"cache ownership is ambiguous")
        owner.setdefault(layer_name, mp)
        krow_list, vrow_list = hist.setdefault(layer_name, ([], []))
        P0 = len(krow_list)
        for i in range(T):
            krow_list.append(krows[i])
            vrow_list.append(vrows[i])
        klog, vlog = kv_logs(ctx, layer_name, krow_list, vrow_list)
        # `allowed_dims`: the request's row count of this call (prefill T / decode 1) and the context length it reaches
        ctx.check_dim("T", T, node)
        ctx.check_dim("CONTEXT", P0 + T, node)
        outs = []
        scale = attention_scale(ctx, D, node)
        for i in range(T):
            P = P0 + i + 1
            fn = ctx.kind("attention_value" if scale else "attention", node, T=P, NH=NH, KVH=KVH, D=D)
            kcat = Coll(Array(P, Array(KVH, Array(D, Value(16)))), klog.cut(P))
            vcat = Coll(Array(P, Array(KVH, Array(D, Value(16)))), vlog.cut(P))
            outs.append(ctx.call(fn, qrows[i], kcat, vcat, *scale, node=node, rule=self, name=f"{ctx.hint_name(node)}[t={P - 1}]"))
        ctx.define(node, array_of(outs).reshape(_tensor_type(qs, 16)))
        ctx.record(self, node, {"layer_name": layer_name, "positions": [P0, P0 + T - 1], "NH": NH, "KVH": KVH, "D": D, "history_rows_after": P0 + T,
                                "kv_rows_from_this_request": True, "assumed_causal": True}, emitted=f"Attention_v1 x{T} (T={P0 + 1}..{P0 + T})")


PROFILE_VLLM_RULES = [RmsNormRule(), FusedAddRmsNormRule(), RotaryEmbeddingRule(), SiluAndMulRule(), UnifiedAttentionRule()]
PROFILE_RULES = PROFILE_TORCH_RULES + PROFILE_VLLM_RULES

__all__ = ["PROFILE_RULES", "PROFILE_TORCH_RULES", "PROFILE_VLLM_RULES"]
