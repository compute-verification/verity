"""vLLM's non-batch-invariant CUDA RMSNorm bindings."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.rules.base import InPlaceRule, RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.registry import cuda_rmsnorm as C


BF16 = torch.bfloat16


def _row_tensor(ctx, a, name: str) -> str | None:
    if not isinstance(a, torch.fx.Node):
        return f"{name} is not a graph value"
    if not isinstance(ctx.val(a), (Alloc, Coll)):
        return f"{name} is not an allocation/value"
    if not dtype_is(ctx, a, BF16):
        return f"numerics_unregistered: {name} has dtype {ctx.dtype_of(a)}; the CUDA vector-8 kernels take bf16"
    if not shape(ctx, a):
        return f"{name} is a scalar"
    return None


def _non_batch_invariant() -> str | None:
    return "the target is batch-invariant" if active_target_profile().batch_invariant else None


class CudaRmsNormRule(InPlaceRule):
    """`_C.rms_norm` under vLLM's default non-batch-invariant CUDA dispatch."""

    modes = frozenset({"profile"})
    info = RuleInfo(
        "RmsNorm[vllm-cuda]",
        1,
        "in-place-binding",
        applies_to="_C.rms_norm(Tensor! out, x, weight, eps)",
        preconditions=(
            "the declared target is not batch-invariant",
            "out is an allocation/value of x's shape",
            "x is bf16 [.., N], weight is bf16 [N], and eps is a static float",
            "BLOCK is cuda_rmsnorm.launch_block(number of token rows, N, fused=False)",
        ),
        evidence_domain="structural; numerics: RMSNormCudaVector8_v1 (capture pending in draft #1970)",
    )

    def targets(self):
        return targets_for("rms_norm")

    def applies(self, node, ctx):
        if (r := _non_batch_invariant()):
            return r
        out, x, w, eps = node.args[:4]
        if (r := _row_tensor(ctx, x, "x") or _row_tensor(ctx, w, "weight")):
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
        rows, lead = rows_of(ctx.coll(x))
        wc = ctx.coll(w)
        N = rows[0].type.n
        block = C.launch_block(len(rows), N, fused=False)
        fn = ctx.kind("rmsnorm_cuda", node, N=N, EPS=eps, BLOCK=block)
        outs = [ctx.call(fn, row, wc, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, row in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.bind_weight_role(w, self, "rmsnorm.weight", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "BLOCK": block, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class CudaFusedAddRmsNormRule(InPlaceRule):
    """`_C.fused_add_rms_norm` under vLLM's default non-batch-invariant CUDA dispatch."""

    modes = frozenset({"profile"})
    info = RuleInfo(
        "FusedAddRmsNorm[vllm-cuda]",
        1,
        "in-place-binding",
        applies_to="_C.fused_add_rms_norm(Tensor! x, Tensor! residual, weight, eps)",
        preconditions=(
            "the declared target is not batch-invariant",
            "x and residual are bf16 values of the same shape [.., N]",
            "weight is bf16 [N], eps is a static float, and both mutated operands are rebound",
            "BLOCK is cuda_rmsnorm.launch_block(number of token rows, N, fused=True)",
        ),
        evidence_domain="structural; numerics: RMSNormFusedCudaVector8_v1 (capture pending in draft #1970)",
    )

    def targets(self):
        return targets_for("fused_add_rms_norm")

    def applies(self, node, ctx):
        if (r := _non_batch_invariant()):
            return r
        x, res, w, eps = node.args[:4]
        if (r := _row_tensor(ctx, x, "x") or _row_tensor(ctx, res, "residual") or _row_tensor(ctx, w, "weight")):
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
        block = C.launch_block(len(xr), N, fused=True)
        fn = ctx.kind("rmsnorm_fused_cuda", node, N=N, EPS=eps, BLOCK=block)
        normed, residual = [], []
        for i, (a, b) in enumerate(zip(xr, rr)):
            out = ctx.call(fn, a, b, wc, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            normed.append(out[0])
            residual.append(out[1])
        ctx.rebind(x, from_rows(normed, lead), self, what="x")
        ctx.rebind(res, from_rows(residual, lead), self, what="residual")
        ctx.bind_weight_role(w, self, "rmsnorm.weight", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "BLOCK": block, "rows": len(xr)}, emitted=f"{fn.id} x{len(xr)}")


__all__ = ["CudaFusedAddRmsNormRule", "CudaRmsNormRule"]
