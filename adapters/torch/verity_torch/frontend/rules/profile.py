"""Profile-mode rules on ATen ops for the eager profile (B0/B1): ordinary modules -> TA1's registered kinds.

Op rules bind functional ops (`aten.linear`, `aten.embedding`, `aten.argmax`, `aten.add`). The `gemm` kind `aten.linear` emits is
the batch-invariant matmul, so the rule asks whether the active target is batch-invariant: the reading an adapter installs with
`read_batch_invariance`, the vLLM adapter's observed target.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
from verity.circuits.refs import Coll, array_of
from verity.circuits.types import Array, Value

from verity_torch.frontend.derive import StaticIndex
from verity_torch.frontend.rules.base import OpRule, RuleInfo
from verity_torch.frontend.rules.common import dtype_is, from_rows, known, row_value, rows_of, shape
from verity_torch.frontend.rules.reference import LinearRule
from verity_torch.frontend.strided_views import _tensor_type
from verity_torch.frontend.value_types import shape_of

aten = torch.ops.aten
BF16 = torch.bfloat16


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
        r = row_value(ctx, table, "table")
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
        r = row_value(ctx, node.args[0], "logits")
        if r:
            return r
        shp = known(shape(ctx, node.args[0]))
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
            r = row_value(ctx, a, nm)
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


def _default_batch_invariance() -> bool:
    return True


_batch_invariant: Callable[[], bool] = _default_batch_invariance


def read_batch_invariance(read: Callable[[], bool]) -> None:
    """Installs the adapter's reading of whether the active target is batch-invariant, which `_without_batch_invariance` asks; until
    an adapter installs one, the target is (the default target is)."""
    global _batch_invariant
    _batch_invariant = read

def _without_batch_invariance(op: str, states: str) -> str | None:
    """A refusal when the active target is not batch-invariant, where ``op`` is the kernel vLLM dispatches without batch invariance
    (cuBLAS's GEMM, `layernorm_kernels.cu`'s `rms_norm_kernel` at a token-count-dependent block), not the batch-invariant one the
    rule's kind states."""
    if _batch_invariant():
        return None
    return f"numerics_unregistered: {op} on a target without batch invariance is the kernel vLLM dispatches there, not {states}"


class ProfileLinearRule(LinearRule):
    """`LinearRule` in profile mode, which states the `gemm` kind, the batch-invariant matmul: on a target without batch invariance
    an exported `aten.linear` is the GEMM vLLM's unquantized linear dispatches to, which that kind does not state."""

    def __init__(self) -> None:
        super().__init__("profile", BF16, "gemm", "bias_add")
        self.info = dataclasses.replace(self.info, preconditions=self.info.preconditions + ("the target is batch-invariant",))

    def applies(self, node, ctx):
        return _without_batch_invariance("aten.linear", "the batch-invariant matmul the 'gemm' kind states") or super().applies(node, ctx)


PROFILE_TORCH_RULES = [ProfileLinearRule(), EmbeddingRule(), ArgmaxTokenSelectRule(), ResidualAddRule(), BroadcastBiasAddRule()]

__all__ = ["PROFILE_TORCH_RULES", "read_batch_invariance"]
