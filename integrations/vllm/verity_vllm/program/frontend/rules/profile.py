"""Profile-mode rules for the vLLM eager profile (B0/B1): ordinary vLLM-style modules -> TA1's registered kinds.

Op rules bind functional ops (`aten.linear`, `aten.embedding`, `aten.argmax`, `aten.add`); in-place rules bind
the kernel interface (`rms_norm`, `fused_add_rms_norm`, `rotary_embedding`, `silu_and_mul`) and rebind the
mutated operands explicitly; the state rule binds `unified_attention` and makes the KV dependencies explicit.
"""

from __future__ import annotations

import dataclasses

import torch

from verity.ml.circuits.refs import Coll, array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import InPlaceRule, OpRule, RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import (attention_scale, dtype_is, from_rows, kv_logs, norm_kind, refuse_cached_prefix, rows_of,
                                                      shape)
from verity_vllm.program.frontend.rules.reference import LinearRule
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.targets import UnregisteredTarget

aten = torch.ops.aten
BF16 = torch.bfloat16


def _row_tensor(ctx, a, name: str, dtype=BF16) -> str | None:
    """Precondition helper: `a` is a value collection of `dtype` with rank >= 1."""
    if not isinstance(a, torch.fx.Node):
        return f"{name} is not a graph value"
    v = ctx.val(a)
    if isinstance(v, Alloc):
        return None  # allocation: fine as a write target
    if not isinstance(v, Coll):
        return f"{name} is not a value collection ({type(v).__name__})"
    if not dtype_is(ctx, a, dtype):
        return f"numerics_unregistered: {name} has dtype {ctx.dtype_of(a)}; the profile's kinds are {dtype}"
    if not shape(ctx, a):
        return f"{name} is a scalar"
    return None


class EmbeddingRule(OpRule):
    """`aten.embedding(table, ids)` with *computed or input* ids -> one `Embedding{V,H}` gather per token."""
    modes = frozenset({"profile"})
    info = RuleInfo("Embedding[profile]", 1, "op-binding", applies_to="aten.embedding(table[V,H] bf16, ids) with ids a value (Input or TokenSelect output)",
                    preconditions=("table is a bf16 value collection [V, H]", "ids is a value collection whose leaves are 32-bit (I32 token ids)",
                                   "padding_idx / scale_grad / sparse are defaults"),
                    evidence_domain="structural; numerics: Embedding_v1 gather (exact-model-tested)")

    def targets(self):
        return {aten.embedding.default}

    def applies(self, node, ctx):
        table, ids = node.args[0], node.args[1]
        r = _row_tensor(ctx, table, "table")
        if r:
            return r
        if len(shape(ctx, table) or ()) != 2:
            return "table is not 2-D"
        v = ctx.val(ids)
        if isinstance(v, StaticIndex):
            return "ids are static structure (StaticGather applies)"
        if not isinstance(v, Coll):
            return "ids is not a value collection"
        if shape_of(v.type)[1] != 32:
            return (f"numerics_unregistered: ids carry {shape_of(v.type)[1]}-bit leaves; Embedding_v1 takes I32 token ids — feed int32 ids "
                    f"(torch int64 ids would need a narrowing kind)")
        if len(node.args) > 2 and node.args[2] not in (-1, None):
            return "padding_idx is set"
        return True

    def emit(self, node, ctx):
        table, ids = ctx.coll(node.args[0]), ctx.coll(node.args[1])
        V, H = shape_of(table.type)[0]
        shp = shape_of(ids.type)[0]
        n = 1
        for s in shp:
            n *= s
        flat = ids.reshape(Array(n, Value(32)))
        fn = ctx.kind("embedding", node, V=V, H=H)
        outs = [ctx.call(fn, flat[i], table, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i in range(n)]
        ctx.define(node, array_of(outs).reshape(_tensor_type(tuple(shp) + (H,), 16)))
        ctx.bind_weight_role(node.args[0], self, "embedding.table", V=V, H=H)
        ctx.record(self, node, {"V": V, "H": H, "tokens": n}, emitted=f"{fn.id} x{n}")


class ArgmaxTokenSelectRule(OpRule):
    """`aten.argmax(logits, dim=-1)` over bf16 logits -> `TokenSelect{V}` per row (greedy sampling)."""
    modes = frozenset({"profile"})
    info = RuleInfo("ArgmaxTokenSelect", 1, "op-binding", applies_to="aten.argmax(bf16 [.., V], dim=-1|None, keepdim=False)",
                    preconditions=("logits is a bf16 value collection", "reduction over the last axis (or the whole 1-D tensor)",
                                   "torch's int64 result is carried as the kind's I32 token id (values < V)"),
                    evidence_domain="structural; numerics: TokenSelect_v1 (argmax, lowest index on ties = torch.argmax)")

    def targets(self):
        return {aten.argmax.default}

    def applies(self, node, ctx):
        r = _row_tensor(ctx, node.args[0], "logits")
        if r:
            return r
        shp = shape(ctx, node.args[0])
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keep = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        if keep:
            return "keepdim=True"
        if dim is None and len(shp) != 1:
            return "argmax over a flattened multi-dim tensor"
        if dim is not None and dim not in (-1, len(shp) - 1):
            return f"argmax along dim {dim}, not the last axis"
        return True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        rows, lead = rows_of(x)
        V = rows[0].type.n
        fn = ctx.kind("token_select", node, V=V)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        out = array_of(outs).reshape(_tensor_type(lead, 32)) if lead else outs[0]
        ctx.define(node, out)
        ctx.record(self, node, {"V": V, "rows": len(rows), "torch_dtype": str(ctx.dtype_of(node)), "carried_as": "I32"}, emitted=f"{fn.id} x{len(rows)}")


class ResidualAddRule(OpRule):
    """`aten.add(a, b)` of two same-shape bf16 tensors -> `ResidualAdd{N}` per row."""
    modes = frozenset({"profile"})
    info = RuleInfo("ResidualAdd[profile]", 1, "op-binding", applies_to="aten.add.Tensor(bf16, bf16) same shape, alpha=1",
                    preconditions=("both operands are bf16 value collections of identical shape", "alpha == 1"),
                    evidence_domain="structural; numerics: ResidualAdd_v1")

    def targets(self):
        return {aten.add.Tensor}

    def applies(self, node, ctx):
        for nm, a in (("a", node.args[0]), ("b", node.args[1])):
            r = _row_tensor(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return "operands differ in shape (broadcast is not a bound rule)"
        if node.kwargs.get("alpha", 1) != 1:
            return "alpha != 1"
        return True

    def emit(self, node, ctx):
        a, b = ctx.coll(node.args[0]), ctx.coll(node.args[1])
        ra, lead = rows_of(a)
        rb, _ = rows_of(b)
        N = ra[0].type.n
        fn = ctx.kind("residual_add", node, N=N)
        outs = [ctx.call(fn, x, y, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(ra)}, emitted=f"{fn.id} x{len(ra)}")


def _bias_shape(shp: tuple[int, ...] | None) -> int | None:
    """`[N]`, or `[1, .., 1, N]` (the same broadcast as `[N]` under torch's trailing-axis alignment) -> N; anything else -> None."""
    if not shp or any(s != 1 for s in shp[:-1]):
        return None
    return shp[-1]


class BroadcastBiasAddRule(OpRule):
    """[X-04] `aten.add.Tensor(x[.., N], b[N])` — a rank-1 bias broadcast over the trailing axis of a bf16 value collection — ->
    the registered `BiasAdd_v1{N}` (`bias_add` kind: one f32 add + RNE per element, `P.Bf16Add`) per row of `x`.

    This is the general operation, not a layer recogniser: it is what vLLM's batch-invariant linear emits after the persistent
    GEMM whenever `bias is not None` (Qwen2 `QKVParallelLinear`, but also any `y + b` with a 1-D `b`), and what
    `LinearRule("profile", …, "bias_add")` already emits for a fused `aten.linear(x, W, b)`; both orders of the operands are
    accepted because IEEE-754 addition is commutative, so `b + x` and `x + b` are the same kind call.  Numerics: torch computes
    a bf16 `add` in f32 and rounds once (RNE) — exactly `BiasAdd_v1`'s body, so the rule adds no new registered semantics.

    Fail-closed: refused unless BOTH operands are bf16 value collections (a static index, an allocation or a lifted constant is
    not a value), `b` is `[N]` (or `[1,..,1,N]`, the identical broadcast), `x` is >= 2-D with `x.shape[-1] == N`, and
    `alpha == 1`.  Every other broadcast (`[M,1] + [M,N]`, `[N] + [N]` with N == M, leading-axis broadcast, scalar `other` =
    `aten.add.Scalar`) stays unbound: `ResidualAdd[profile]` takes the same-shape case and anything else is an `Unsupported`.
    `aten.add.Tensor` is functional (out-of-place) — no operand is mutated or aliased, so alias/lifetime tracking is untouched."""
    modes = frozenset({"profile"})
    info = RuleInfo("BroadcastBiasAdd[profile]", 1, "op-binding", applies_to="aten.add.Tensor(x[.., N] bf16, b[N] bf16) (either order), alpha == 1",
                    preconditions=("both operands are bf16 value collections", "exactly one operand is a rank-1 [N] (or [1,..,1,N]) bias",
                                   "the other operand is >= 2-D with trailing dim N (per-row broadcast of the bias)", "alpha == 1",
                                   "functional op: no mutation, no aliasing"),
                    evidence_domain="structural; numerics: BiasAdd_v1 (f32 add, one RNE rounding = torch bf16 add; exact-model-tested)")

    def targets(self):
        return {aten.add.Tensor}

    def _split(self, node, ctx):
        """-> (x_node, b_node, N, bias_first) or a refusal string."""
        a0, a1 = node.args[0], node.args[1]
        for nm, a in (("a", a0), ("b", a1)):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{nm} is not a value collection"
            if not dtype_is(ctx, a, BF16):
                return f"numerics_unregistered: {nm} has dtype {ctx.dtype_of(a)}; BiasAdd_v1 is bf16 + bf16 (a mixed-dtype add promotes and is a different numeric)"
        s0, s1 = shape(ctx, a0), shape(ctx, a1)
        if s0 is None or s1 is None:
            return "operand shapes unknown"
        if s0 == s1:
            return "same-shape operands (ResidualAdd, not a bias broadcast)"
        for x, b, sx, sb, bias_first in ((a0, a1, s0, s1, False), (a1, a0, s1, s0, True)):
            N = _bias_shape(sb)
            if N is not None and len(sx) >= 2 and sx[-1] == N and N >= 1:
                return x, b, N, bias_first
        return f"not a [.., N] + [N] bias broadcast ({s0} + {s1}); other broadcast patterns are not bound"

    def applies(self, node, ctx):
        r = self._split(node, ctx)
        if isinstance(r, str):
            return r
        if node.kwargs.get("alpha", 1) != 1:
            return "alpha != 1"
        return True

    def emit(self, node, ctx):
        xn, bn, N, bias_first = self._split(node, ctx)
        x, b = ctx.coll(xn, "x"), ctx.coll(bn, "b")
        if len(shape_of(b.type)[0]) != 1:
            b = b.reshape(_tensor_type((N,), 16))  # [1,..,1,N] -> [N]: the same leaves, the same broadcast
        rows, lead = rows_of(x)
        fn = ctx.kind("bias_add", node, N=N)
        outs = [ctx.call(fn, r, b, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]/bias") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        if ctx.weight_of(bn) is not None:
            ctx.bind_weight_role(bn, self, "linear.bias", N=N)
        ctx.record(self, node, {"N": N, "rows": len(rows), "bias_first": bias_first, "x_shape": list(shape_of(x.type)[0]),
                                "bias_is_parameter": (ctx.weight_of(bn) or None) and ctx.weight_of(bn).name}, emitted=f"{fn.id} x{len(rows)}")


def _without_batch_invariance(op: str, states: str) -> str | None:
    """A refusal when the active target is not batch-invariant, where ``op`` is the kernel vLLM dispatches without batch invariance
    (cuBLAS's GEMM, `layernorm_kernels.cu`'s `rms_norm_kernel` at a token-count-dependent block), not the batch-invariant one the
    rule's kind states."""
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile  # vllm_bindings imports this module
    if active_target_profile().batch_invariant:
        return None
    return f"numerics_unregistered: {op} on a target without batch invariance is the kernel vLLM dispatches there, not {states}"


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
        r = _row_tensor(ctx, x, "x") or _row_tensor(ctx, w, "weight")
        if r:
            return r
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "out is not an allocation/value"
        if shape(ctx, w) != (shape(ctx, x)[-1],):
            return f"weight {shape(ctx, w)} vs hidden {shape(ctx, x)[-1]}"
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
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile  # vllm_bindings imports this module
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
        r = _row_tensor(ctx, x, "x") or _row_tensor(ctx, res, "residual") or _row_tensor(ctx, w, "weight")
        if r:
            return r
        if isinstance(ctx.val(x), Alloc) or isinstance(ctx.val(res), Alloc):
            return "x/residual is an unwritten allocation"
        if shape(ctx, x) != shape(ctx, res):
            return "x and residual differ in shape"
        if shape(ctx, w) != (shape(ctx, x)[-1],):
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
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile  # vllm_bindings imports this module
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
        r = _row_tensor(ctx, q, "query") or _row_tensor(ctx, cache, "cos_sin_cache")
        if r:
            return r
        if k is not None:
            r = _row_tensor(ctx, k, "key")
            if r:
                return r
        if not is_neox:
            return "numerics_unregistered: GPT-J (interleaved) rotation has no registered kind"
        cs = shape(ctx, cache)
        if len(cs) != 2 or cs[1] != head_size:
            return f"cos_sin_cache {cs} vs head_size {head_size}"
        qs = shape(ctx, q)
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
        out, x = node.args[:2]
        r = _row_tensor(ctx, x, "x")
        if r:
            return r
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "out is not an allocation/value"
        xs, os_ = shape(ctx, x), shape(ctx, out)
        if xs[-1] % 2 or os_ != xs[:-1] + (xs[-1] // 2,):
            return f"out {os_} vs x {xs}"
        return True

    def emit(self, node, ctx):
        out, x = node.args[:2]
        rows, lead = rows_of(ctx.coll(x))
        I = rows[0].type.n // 2
        fn = ctx.kind("silu_mul", node, I=I)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"I": I, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


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
            r = _row_tensor(ctx, a, nm)
            if r:
                return r
            if isinstance(ctx.val(a), Alloc):
                return f"{nm} is an unwritten allocation"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        qs, ks, vs = shape(ctx, q), shape(ctx, k), shape(ctx, v)
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


class ProfileLinearRule(LinearRule):
    """`LinearRule` in profile mode, which states the `gemm` kind, the batch-invariant matmul: on a target without batch invariance
    an exported `aten.linear` is the GEMM vLLM's unquantized linear dispatches to, which that kind does not state."""

    def __init__(self) -> None:
        super().__init__("profile", BF16, "gemm", "bias_add")
        self.info = dataclasses.replace(self.info, preconditions=self.info.preconditions + ("the target is batch-invariant",))

    def applies(self, node, ctx):
        return _without_batch_invariance("aten.linear", "the batch-invariant matmul the 'gemm' kind states") or super().applies(node, ctx)


PROFILE_RULES = [ProfileLinearRule(), EmbeddingRule(), ArgmaxTokenSelectRule(), ResidualAddRule(), BroadcastBiasAddRule(),
                 RmsNormRule(), FusedAddRmsNormRule(), RotaryEmbeddingRule(), SiluAndMulRule(), UnifiedAttentionRule()]

__all__ = ["PROFILE_RULES"]
