"""Generic elementwise / activation bindings."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.rules.base import InPlaceRule, OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import call_args, dtype_is, from_rows, rows_of, scalar_kind, shape
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.rules.vllm_bindings.operands import BF16, _row_value, aten
from verity.ml.definitions.registry.dsv4_moe import limit_bits


# -- generic elementwise / activation bindings (dense2 vocabulary; none names a model) ----------------------------------------


def _C_op(name: str) -> set:
    from verity_vllm.program.frontend.rules.vllm_iface import real_C
    C = real_C()
    return {getattr(C, name).default} if C is not None and hasattr(C, name) else set()


class ScalarTensorMulRule(OpRule):
    """`aten.mul.Tensor(x bf16 [.., N], s bf16 0-d tensor)` -> `scale_tensor{N}` (Bf16MulScalarTensor_v1) per row, `s` shared.
    Keyed on the observed dtypes and the scalar's rank — the scalar is a graph value (a buffer), never a constant folded in."""
    modes = frozenset({"profile"})
    info = RuleInfo("ScalarTensorMul[aten]", 1, "op-binding", applies_to="aten.mul.Tensor(x bf16 [.., N], s bf16 [])",
                    preconditions=("x is a bf16 value collection of rank >= 1", "s is a bf16 value collection of rank 0 (or [1])"),
                    evidence_domain="structural; numerics: Bf16MulScalarTensor_v1 (bf16 x bf16 scalar tensor, bf16 out)")

    def targets(self):
        return {aten.mul.Tensor}

    @staticmethod
    def _operands(node, ctx):
        """(x, s) with `s` the scalar tensor, whichever order the graph wrote them (`aten.mul` is commutative and the kernel
        computes the same f32 product); None when neither operand is a 0-d / [1] tensor."""
        a, b = node.args
        for x, sc in ((a, b), (b, a)):
            if isinstance(ctx.val(sc), Coll) and shape(ctx, sc) in ((), (1,)) and shape(ctx, x):
                return x, sc
        return None

    def applies(self, node, ctx):
        if len(node.args) != 2 or not all(isinstance(a, torch.fx.Node) for a in node.args):
            return "operands are not both graph values"
        ops = self._operands(node, ctx)
        if ops is None:
            return f"neither operand is a scalar tensor (shapes {shape(ctx, node.args[0])} x {shape(ctx, node.args[1])})"
        x, sc = ops
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if not dtype_is(ctx, sc, BF16):
            # bf16 x f32-tensor promotes (a 0-d tensor does not, but a [1] one does) / f32 x f32: other numerics, refused
            return f"numerics_unregistered: s has dtype {ctx.dtype_of(sc)}; Bf16MulScalarTensor_v1 is the bf16 x bf16-scalar-tensor kernel"
        return True

    def emit(self, node, ctx):
        x, sc = self._operands(node, ctx)
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        scol = ctx.coll(sc).reshape(Array(1, Value(16)))
        fn = ctx.kind("scale_tensor", node, N=N)
        outs = [ctx.call(fn, r, scol, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "scalar_shape": list(shape(ctx, sc))}, emitted=f"{fn.id} x{len(rows)}")


def _and_mul_operands(node, ctx):
    """An `act_and_mul` kernel's `(out!, x[.., 2I])`: x a bf16 row value, out an allocation (or value) of [.., I]."""
    out, x = node.args[:2]
    r = _row_value(ctx, x, "x")
    if r:
        return r
    if not isinstance(ctx.val(out), (Alloc, Coll)):
        return "out is not an allocation/value"
    xs, os_ = shape(ctx, x), shape(ctx, out)
    if xs[-1] % 2 or os_ != xs[:-1] + (xs[-1] // 2,):
        return f"out {os_} vs x {xs}"
    return True


class GeluTanhAndMulRule(InPlaceRule):
    """`_C.gelu_tanh_and_mul(out!, x[.., 2I])` -> `gelu_tanh_mul{I}` (GeluTanhMul_v1) per row; `out` is rebound (mirrors SiluAndMul)."""
    modes = frozenset({"profile"})
    info = RuleInfo("GeluTanhAndMul[vllm]", 1, "in-place-binding", applies_to="_C.gelu_tanh_and_mul(Tensor! out, x[.., 2I])",
                    preconditions=("x is bf16 [.., 2I]", "out is an unwritten allocation (or value) [.., I]"),
                    evidence_domain="structural; numerics: GeluTanhMul_v1 (vLLM _C.gelu_tanh_and_mul)")

    def targets(self):
        return targets_for("gelu_tanh_and_mul")

    def applies(self, node, ctx):
        return _and_mul_operands(node, ctx)

    def emit(self, node, ctx):
        out, x = node.args[:2]
        rows, lead = rows_of(ctx.coll(x))
        I = rows[0].type.n // 2
        fn = ctx.kind("gelu_tanh_mul", node, I=I)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"I": I, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")



class SiluAndMulWithClampRule(InPlaceRule):
    """`_C.silu_and_mul_with_clamp(out!, x[.., 2I], limit, alpha, beta)` -> `silu_mul_clamp{I,LIMIT}` (SiluMulClamp_v1) per row; `out`
    is rebound (mirrors SiluAndMul). SwiGLU-OAI's sigmoid scale and up bias (alpha != 1, beta != 0) have no Definition and are refused."""
    modes = frozenset({"profile"})
    info = RuleInfo("SiluAndMulWithClamp[vllm]", 1, "in-place-binding",
                    applies_to="_C.silu_and_mul_with_clamp(Tensor! out, x[.., 2I], float limit, float alpha=1., float beta=0.)",
                    preconditions=("x is bf16 [.., 2I]", "out is an unwritten allocation (or value) [.., I]",
                                   "limit is a positive finite bf16 value", "alpha = 1.0 and beta = 0.0"),
                    evidence_domain="structural; numerics: SiluMulClamp_v1 (vLLM act_and_mul_kernel with HAS_CLAMP)")

    def targets(self):
        return targets_for("silu_and_mul_with_clamp")

    def applies(self, node, ctx):
        a = call_args(node)
        if (a["alpha"], a["beta"]) != (1.0, 0.0):
            return f"numerics_unregistered: alpha {a['alpha']}, beta {a['beta']}; SiluMulClamp_v1 is the clamp at alpha 1.0, beta 0.0"
        try:
            limit_bits(a["limit"])
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        if not ctx.vocab.has("silu_mul_clamp"):
            return "numerics_unregistered: the vocabulary has no silu_mul_clamp (SiluMulClamp_v1{I,LIMIT})"
        return _and_mul_operands(node, ctx)

    def emit(self, node, ctx):
        a = call_args(node)
        rows, lead = rows_of(ctx.coll(a["input"]))
        I, LIMIT = rows[0].type.n // 2, float(a["limit"])
        fn = ctx.kind("silu_mul_clamp", node, I=I, LIMIT=LIMIT)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(a["result"], from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"I": I, "LIMIT": LIMIT, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")

class Bf16ScalarChainRule(OpRule):
    """bf16 row-wise ops with a static Python-float constant: `aten.div.Tensor(x, C)` -> `bf16_div_scalar{N,C}`,
    `aten.mul.Tensor(x, C)` -> `bf16_mul_scalar{N,C}`, `aten.tanh.default(x)` -> `bf16_tanh{N}`. The constant is part of the kind."""
    modes = frozenset({"profile"})
    info = RuleInfo("Bf16ScalarChain[aten]", 1, "op-binding", applies_to="aten.div.Tensor / aten.mul.Tensor (x bf16, C: float) ; aten.tanh.default(x bf16)",
                    preconditions=("x is a bf16 value collection of rank >= 1", "C is a static Python float (div/mul)"),
                    evidence_domain="structural; numerics: Bf16DivScalar_v1 / Bf16MulScalar_v1 / Bf16Tanh_v1")

    def targets(self):
        return {aten.div.Tensor, aten.mul.Tensor, aten.tanh.default}

    def applies(self, node, ctx):
        x = node.args[0]
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if node.target is aten.tanh.default:
            return True
        if len(node.args) != 2 or not isinstance(node.args[1], (float, int)) or isinstance(node.args[1], bool):
            return "the second operand is not a static Python scalar"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        if node.target is aten.tanh.default:
            (fn, value), det = (ctx.kind("bf16_tanh", node, N=N), ()), {"N": N}
        else:
            role, C = "bf16_div_scalar" if node.target is aten.div.Tensor else "bf16_mul_scalar", float(node.args[1])
            (fn, value), det = scalar_kind(ctx, role, node, N, C), {"N": N, "C": C}
        member = _softcap_member(node, ctx) if node.target is aten.mul.Tensor else None
        outs = [ctx.call(fn, r, *value, node=node, rule=self, name=f"{ctx.hint_name(node)}/{member}[{i}]" if member else f"{ctx.hint_name(node)}[{i}]")
                for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {**det, "rows": len(rows), **({"member": member} if member else {})}, emitted=f"{fn.id} x{len(rows)}")


def _softcap_member(node, ctx) -> str | None:
    """`aten.mul.Tensor(tanh(x / C), C)` — the last op of a logits soft cap `C * tanh(x / C)` — produces the value
    its module returns: the row names carry the member hint `out` (the manifest's `members_for` reads it; the div / tanh rows are the
    chain's interior).  Structural test only: the operand is a `aten.tanh` of a `aten.div.Tensor(_, C)` with the same C, in one module."""
    x = node.args[0]
    if not (isinstance(x, torch.fx.Node) and x.target is aten.tanh.default and isinstance(x.args[0], torch.fx.Node)):
        return None
    d = x.args[0]
    if d.target is not aten.div.Tensor or len(d.args) != 2 or not isinstance(d.args[1], (int, float)) or float(d.args[1]) != float(node.args[1]):
        return None
    if ctx.module_path(d) != ctx.module_path(node):
        return None
    return "out"
