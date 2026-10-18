"""GLM-5.3's served-default BF16 GEMMs on cc 9.0.

The unbiased linears use cuBLAS's captured per-shape, per-M route.  The MoE gate leaf is emitted only when vLLM dispatches
its cuteDSL ``ll_bf16_gemm`` tier; its larger-token fallback is ``aten.mm.dtype`` and uses cuBLAS's captured f32-output
route.  An absent route is unsupported rather than being interpreted as a non-split launch.
"""
from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _strides, _untransposed
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.registry import bf16_gemm_check

aten = torch.ops.aten


def _h200_default() -> str | None:
    target = active_target_profile()
    if target.compute_capability != (9, 0):
        return f"compute capability {target.compute_capability}, not cc 9.0"
    if target.batch_invariant:
        return "batch invariance selects a different GEMM"
    return None


class CublasLinearRule(OpRule):
    modes = frozenset({"profile"})
    info = RuleInfo(
        "CublasLinear[vllm-default]",
        1,
        "op-binding",
        applies_to="unbiased aten.linear on cc 9.0 without batch invariance",
        preconditions=(
            "bf16 x[...,K] and W[N,K]",
            "the shape and flattened row count have a captured vLLM-default cuBLAS route",
        ),
        evidence_domain="GLM-5.3 BF16 GEMM defaults capture; GO pending its capture-side probe fix",
    )

    def targets(self):
        return {aten.linear.default}

    def applies(self, node, ctx):
        if (why := _h200_default()) is not None:
            return why
        x, w = node.args[:2]
        bias = node.args[2] if len(node.args) > 2 else None
        if bias is not None:
            return "the captured default route is for an unbiased linear"
        if not isinstance(ctx.val(x), Coll) or not isinstance(ctx.val(w), Coll):
            return "x or W is not a value"
        if ctx.dtype_of(x) != torch.bfloat16 or ctx.dtype_of(w) != torch.bfloat16:
            return "x or W is not bf16"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if xs is None or ws is None or not xs or len(ws) != 2 or xs[-1] != ws[1]:
            return f"x {xs} vs W {ws}"
        M, (N, K) = len(rows_of(ctx.coll(x))[0]), ws
        route = bf16_gemm_check.default_cublas_route("linear", M, K, N)
        if route is None:
            return f"no captured vLLM-default cuBLAS linear route for M={M}, K={K}, N={N}"
        role = "cublas_splitk_gemm" if route[0] == "splitk" else "gemm_dot"
        if not ctx.vocab.has(role):
            return f"the vocabulary has no {role}"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets

        x, w = node.args[:2]
        N, K = shape(ctx, w)
        rows, lead = rows_of(ctx.coll(x))
        M = len(rows)
        kernel, statics = bf16_gemm_check.default_cublas_route("linear", M, K, N)
        DOT = targets.tensor_core_dot(active_target_profile())
        role = "cublas_splitk_gemm" if kernel == "splitk" else "gemm_dot"
        fn = ctx.kind(role, node, K=K, N=N, DOT=DOT, **statics)
        outs = [
            ctx.call(fn, row, ctx.coll(w), node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            for i, row in enumerate(rows)
        ]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(w, self, "linear.weight", K=K, N=N)
        ctx.record(
            self,
            node,
            {"M": M, "K": K, "N": N, "kernel": kernel, **statics, "DOT": DOT.name},
            emitted=f"{fn.id} x{M}",
        )


class LlBf16GemmRule(OpRule):
    modes = frozenset({"profile"})
    info = RuleInfo(
        "LlBf16Gemm[vllm-default]",
        1,
        "op-binding",
        applies_to="vllm_iface.ll_bf16_gemm(x, W)",
        preconditions=(
            "bf16 x[M,K] and W[N,K] on cc 9.0",
            "vLLM's default cuteDSL dispatch selects its dotprod or split-K form",
        ),
        evidence_domain="GLM-5.3 BF16 GEMM defaults capture; GO pending its capture-side probe fix",
    )

    def targets(self):
        return {iface.ll_bf16_gemm.default}

    def applies(self, node, ctx):
        if (why := _h200_default()) is not None:
            return why
        x, w = node.args
        if not isinstance(ctx.val(x), Coll) or not isinstance(ctx.val(w), Coll):
            return "x or W is not a value"
        if ctx.dtype_of(x) != torch.bfloat16 or ctx.dtype_of(w) != torch.bfloat16:
            return "x or W is not bf16"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if xs is None or ws is None or len(xs) != 2 or len(ws) != 2 or xs[1] != ws[1]:
            return f"x {xs} vs W {ws}"
        kernel, _ = bf16_gemm_check.dispatch_sm90(xs[0], xs[1], ws[0])
        if kernel not in ("dotprod", "splitk"):
            return f"the leaf unexpectedly names the {kernel} route"
        role = "ll_bf16_dotprod" if kernel == "dotprod" else "ll_bf16_splitk"
        if not ctx.vocab.has(role):
            return f"the vocabulary has no {role}"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets

        x, w = node.args
        M, K = shape(ctx, x)
        N = shape(ctx, w)[0]
        kernel, statics = bf16_gemm_check.dispatch_sm90(M, K, N)
        if kernel == "dotprod":
            fn = ctx.kind(
                "ll_bf16_dotprod",
                node,
                N=N,
                K=K,
                FMA=bf16_gemm_check.DEFAULT_LL_BF16_FMA,
                **statics,
            )
        else:
            DOT = targets.tensor_core_dot(active_target_profile())
            fn = ctx.kind("ll_bf16_splitk", node, N=N, K=K, DOT=DOT, **statics)
        rows, lead = rows_of(ctx.coll(x))
        outs = [
            ctx.call(fn, row, ctx.coll(w), node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            for i, row in enumerate(rows)
        ]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(w, self, "moe.gate.weight", K=K, N=N)
        ctx.record(
            self,
            node,
            {"M": M, "K": K, "N": N, "kernel": kernel, **statics},
            emitted=f"{fn.id} x{M}",
        )


class CublasRouterRule(OpRule):
    modes = frozenset({"profile"})
    info = RuleInfo(
        "CublasRouter[vllm-default]",
        1,
        "op-binding",
        applies_to="aten.mm.dtype(x, W.T, out_dtype=float32), GateLinear's cc 9.0 fallback above 16 tokens",
        preconditions=(
            "x is row-major bf16 [M,K], W.T is the transposed view of row-major bf16 W[N,K], and output is row-major f32 [M,N]",
            "vLLM dispatches the cuBLAS tier and the shape and M have a captured vLLM-default route",
        ),
        evidence_domain="GLM-5.3 BF16 GEMM defaults capture; GO pending its capture-side probe fix",
    )

    def targets(self):
        return {aten.mm.dtype}

    def applies(self, node, ctx):
        if (why := _h200_default()) is not None:
            return why
        if len(node.args) != 3 or node.kwargs or node.args[2] is not torch.float32:
            return f"expected aten.mm.dtype(x, W.T, torch.float32), got args={node.args[2:]} kwargs={node.kwargs}"
        x, wt = node.args[:2]
        if not isinstance(ctx.val(x), Coll) or not isinstance(ctx.val(wt), Coll):
            return "x or W.T is not a value"
        if ctx.dtype_of(x) != torch.bfloat16 or ctx.dtype_of(wt) != torch.bfloat16 or ctx.dtype_of(node) != torch.float32:
            return f"expected bf16 x and W.T with f32 output, got {ctx.dtype_of(x)}, {ctx.dtype_of(wt)} -> {ctx.dtype_of(node)}"
        xs, wts, ys = shape(ctx, x), shape(ctx, wt), shape(ctx, node)
        if xs is None or wts is None or len(xs) != 2 or len(wts) != 2 or xs[1] != wts[0]:
            return f"x {xs} vs W.T {wts}"
        M, K, N = xs[0], xs[1], wts[1]
        if ys != (M, N):
            return f"output {ys}, not {(M, N)}"
        if _strides(x) != (K, 1) or _strides(wt) != (1, K) or _strides(node) != (N, 1):
            return f"noncanonical strides x={_strides(x)}, W.T={_strides(wt)}, output={_strides(node)}"
        kernel, _ = bf16_gemm_check.dispatch_sm90(M, K, N)
        if kernel != "cublas":
            return f"vLLM dispatches the {kernel} tier at M={M}, K={K}, N={N}, not aten.mm.dtype"
        route = bf16_gemm_check.default_cublas_route("router", M, K, N)
        if route is None:
            return f"no captured vLLM-default cuBLAS router route for M={M}, K={K}, N={N}"
        role = "cublas_splitk_gemm_f32out" if route[0] == "splitk" else "cublas_gemm_f32out"
        if not ctx.vocab.has(role):
            return f"the vocabulary has no {role}"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets

        x, wt = node.args[:2]
        M, K = shape(ctx, x)
        N = shape(ctx, wt)[1]
        kernel, statics = bf16_gemm_check.default_cublas_route("router", M, K, N)
        DOT = targets.tensor_core_dot(active_target_profile())
        role = "cublas_splitk_gemm_f32out" if kernel == "splitk" else "cublas_gemm_f32out"
        fn = ctx.kind(role, node, K=K, N=N, DOT=DOT, **statics)
        rows, lead = rows_of(ctx.coll(x))
        w = _untransposed(ctx, wt, K, N, ctx.hint_name(node) + "/W")
        outs = [
            ctx.call(fn, row, w, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            for i, row in enumerate(rows)
        ]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(wt, self, "moe.gate.weight", K=K, N=N)
        ctx.record(
            self,
            node,
            {"M": M, "K": K, "N": N, "kernel": kernel, **statics, "DOT": DOT.name},
            emitted=f"{fn.id} x{M}",
        )


__all__ = ["CublasLinearRule", "CublasRouterRule", "LlBf16GemmRule"]
