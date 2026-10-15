"""vLLM's served-default BF16 cuBLAS GEMMs on cc 9.0, folded as the Build binds them (`vllm_bindings.bf16_gemm.CublasLinearRule`).

`aten.linear(x [M,K] bf16, W [N,K] bf16 parameter)` without bias and off batch invariance runs cuBLAS on the captured route of
(M, K, N) (`bf16_gemm_check.default_cublas_route`): split-K -> `CublasSplitKGemmWord_v1{K,N,SPLIT_K,DOT}`, non-split ->
`Gemm_v2{K,N,DOT}`, one instance per row, DOT the target's k16 step.  A shape or M without a captured route is refused, never read
as a non-split launch.  A batch-invariant target is the base `GemmLaunch`'s.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.ml.circuits.defs import bind
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_row_major
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import _targets, target_of
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode


@dataclass
class CublasLinear:
    name: str = "gemm.cublas_linear_default"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.linear":
            return None
        tp, _recorded = target_of(ctx)
        if tp.batch_invariant or tuple(tp.compute_capability) != (9, 0):
            return None
        from verity_vllm.program.registry import b1, bf16_gemm_check, cublas_bf16

        a = bound_args(node)
        x, w, bias = a.get("input", a.get("arg0")), a.get("weight", a.get("arg1")), a.get("bias", a.get("arg2"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(w, ViewDesc) or len(outs) != 1:
            return unsupported(node, "aten.linear arguments not bound as (input, weight)", input=repr(x), weight=repr(w))
        if bias is not None:
            return unsupported(node, "the captured default cuBLAS route is for an unbiased linear", bias=repr(bias))
        out = outs[0]
        if x.rank != 2 or w.rank != 2 or out.rank != 2 or not all(is_row_major(v) for v in (x, w, out)):
            return unsupported(node, "aten.linear on cuBLAS takes row-major [M,K] x [N,K] -> [M,N]", x=repr(x), w=repr(w), out=repr(out))
        (M, K), (N, K2) = x.shape, w.shape
        if K2 != K or out.shape != (M, N) or {x.dtype, w.dtype, out.dtype} != {"bfloat16"}:
            return unsupported(node, "aten.linear shapes or dtypes are not bf16 [M,K] x [N,K] -> [M,N]", x=repr(x), w=repr(w), out=repr(out))
        roots = ctx.roots_of(w)
        if not any(r.role == "weight" for r in roots):
            return unsupported(node, "the linear's weight is not a checkpoint parameter (weight Root)", w=repr(w))
        route = bf16_gemm_check.default_cublas_route("linear", M, K, N)
        if route is None:
            return unsupported(node, "no captured vLLM-default cuBLAS linear route", M=M, K=K, N=N)
        T = _targets()
        try:
            dot = T.tensor_core_dot(tp)
        except T.UnregisteredTarget as e:
            return unsupported(node, str(e))
        kernel, statics = route
        spec = (bind(cublas_bf16.CublasSplitKGemmWord, K=K, N=N, SPLIT_K=statics["SPLIT_K"], DOT=dot) if kernel == "splitk"
                else bind(b1.GemmV2, K=K, N=N, DOT=dot))
        return Compute(spec, M, (PerRow(x), Shared(w)), (PerRow(out),), tuple(sorted(node.ids())), hint=hint_of(node, roots[0].name),
                       kernels=("gemm", "splitK", "cutlass", "sm90"), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
