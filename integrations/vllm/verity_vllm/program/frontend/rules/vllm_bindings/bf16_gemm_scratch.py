"""Scratch-only bindings used to probe the Definitions on #1962."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.registry import bf16_gemm_check

aten = torch.ops.aten


class CublasLinearScratchRule(OpRule):
    modes = frozenset({"profile"})
    info = RuleInfo(
        "CublasLinear[scratch]",
        1,
        "op-binding",
        applies_to="unbiased aten.linear on a target without batch invariance",
        preconditions=("bf16 x and W[N,K]", "no bias", "the target is cc 9.0 without batch invariance"),
        evidence_domain="scratch probe of #1962's core Gemm_v2 binding; capture unconfirmed",
    )

    def targets(self):
        return {aten.linear.default}

    def applies(self, node, ctx):
        tp = active_target_profile()
        if tp.batch_invariant or tp.compute_capability != (9, 0):
            return "not the scratch Hopper non-batch-invariant route"
        x, w = node.args[:2]
        bias = node.args[2] if len(node.args) > 2 else None
        if bias is not None:
            return "the scratch rule does not model a biased cuBLAS linear"
        if not isinstance(ctx.val(x), Coll) or not isinstance(ctx.val(w), Coll):
            return "x or W is not a value"
        if ctx.dtype_of(x) != torch.bfloat16 or ctx.dtype_of(w) != torch.bfloat16:
            return "x or W is not bf16"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if xs is None or ws is None or not xs or len(ws) != 2 or xs[-1] != ws[1]:
            return f"x {xs} vs W {ws}"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets

        x, w = node.args[:2]
        N, K = shape(ctx, w)
        DOT = targets.tensor_core_dot(active_target_profile())
        fn = ctx.kind("cublas_linear", node, K=K, N=N, DOT=DOT)
        rows, lead = rows_of(ctx.coll(x))
        outs = [ctx.call(fn, row, ctx.coll(w), node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
                for i, row in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(w, self, "linear.weight", K=K, N=N)
        ctx.record(self, node, {"M": len(rows), "K": K, "N": N, "DOT": DOT.name}, emitted=f"{fn.id} x{len(rows)}")


class LlBf16ScratchRule(OpRule):
    modes = frozenset({"profile"})
    info = RuleInfo(
        "LlBf16Gemm[scratch]",
        1,
        "op-binding",
        applies_to="vllm_iface.ll_bf16_gemm(x, W)",
        preconditions=("x [M,K] and W [N,K] bf16", "cc 9.0", "M <= 16"),
        evidence_domain="scratch probe of #1962's cuteDSL gate Definitions; FMA=1 capture unconfirmed",
    )

    def targets(self):
        return {iface.ll_bf16_gemm.default}

    def applies(self, node, ctx):
        x, w = node.args
        if active_target_profile().compute_capability != (9, 0):
            return "the scratch rule is only for cc 9.0"
        if not isinstance(ctx.val(x), Coll) or not isinstance(ctx.val(w), Coll):
            return "x or W is not a value"
        if ctx.dtype_of(x) != torch.bfloat16 or ctx.dtype_of(w) != torch.bfloat16:
            return "x or W is not bf16"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if xs is None or ws is None or len(xs) != 2 or len(ws) != 2 or xs[1] != ws[1]:
            return f"x {xs} vs W {ws}"
        kind, _ = bf16_gemm_check.dispatch_sm90(xs[0], xs[1], ws[0])
        return True if kind in ("dotprod", "splitk") else f"the shim unexpectedly selected {kind}"

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets

        x, w = node.args
        M, K = shape(ctx, x)
        N = shape(ctx, w)[0]
        kind, statics = bf16_gemm_check.dispatch_sm90(M, K, N)
        if kind == "dotprod":
            fn = ctx.kind("ll_bf16_dotprod", node, N=N, K=K, FMA=1, **statics)
        else:
            DOT = targets.tensor_core_dot(active_target_profile())
            fn = ctx.kind("ll_bf16_splitk", node, N=N, K=K, DOT=DOT, **statics)
        rows, lead = rows_of(ctx.coll(x))
        outs = [ctx.call(fn, row, ctx.coll(w), node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
                for i, row in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(w, self, "moe.gate.weight", K=K, N=N)
        ctx.record(self, node, {"M": M, "K": K, "N": N, "kernel": kind, **statics},
                   emitted=f"{fn.id} x{len(rows)}")


SCRATCH_RULES = [CublasLinearScratchRule(), LlBf16ScratchRule()]
