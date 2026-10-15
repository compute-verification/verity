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


def _permuted(v: ViewDesc, order: tuple[int, ...]) -> ViewDesc:
    return ViewDesc(v.alloc, v.offset, tuple(v.shape[i] for i in order), tuple(v.strides[i] for i in order), v.dtype, v.device)


@dataclass
class MlaAbsorbBmm:
    """`aten.bmm(x [B,M,K] bf16, W [B,K,N] bf16 parameter view)` / `aten.bmm.out(..., out=)`, MLA's latent absorptions (`W_UK_T`,
    `W_UV`, views of a parameter's storage) -> `Compute(BatchedGemmWord_v1{B,K,N,DOT}, rows=M, (PerRow(x[:, m, :]), Shared(W^T)),
    (PerRow(out[:, m, :]),))`: the Build's `MlaAbsorbBmmRule`, token for token over all B heads, DOT the target's k16 step."""

    name: str = "gemm.mla_absorb_bmm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.bmm":
            return None
        tp, _recorded = target_of(ctx)
        if tp.batch_invariant:
            return None
        from verity_vllm.program.registry import cublas_bf16

        a = bound_args(node)
        x, w = a.get("self", a.get("arg0")), a.get("mat2", a.get("arg1"))
        outs = [a["out"]] if isinstance(a.get("out"), ViewDesc) else node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(w, ViewDesc) or len(outs) != 1:
            return unsupported(node, "aten.bmm arguments not bound as (self, mat2)", x=repr(x), w=repr(w))
        out = outs[0]
        if x.rank != 3 or w.rank != 3 or out.rank != 3 or {x.dtype, w.dtype, out.dtype} != {"bfloat16"}:
            return unsupported(node, "MLA absorb bmm takes bf16 [B,M,K] x [B,K,N] -> [B,M,N]", x=repr(x), w=repr(w), out=repr(out))
        (B, M, K), (B2, K2, N) = x.shape, w.shape
        if (B2, K2) != (B, K) or out.shape != (B, M, N) or x.strides[2] != 1 or w.strides[1] != 1 or out.strides[2] != 1:
            return unsupported(node, "bmm shapes or inner strides are not [B,M,K] x [B,K,N] -> [B,M,N] with unit inner strides",
                               x=repr(x), w=repr(w), out=repr(out))
        roots = ctx.roots_of(w)
        if not roots:
            return unsupported(node, "the absorbed weight is not a view of a parameter's storage (no Root)", w=repr(w))
        T = _targets()
        try:
            dot = T.tensor_core_dot(tp)
        except T.UnregisteredTarget as e:
            return unsupported(node, str(e))
        spec = bind(cublas_bf16.BatchedGemmWord, B=B, K=K, N=N, DOT=dot)
        return Compute(spec, M, (PerRow(_permuted(x, (1, 0, 2))), Shared(_permuted(w, (0, 2, 1)))), (PerRow(_permuted(out, (1, 0, 2))),),
                       tuple(sorted(node.ids())), hint=hint_of(node, roots[0].name), kernels=("gemm", "cutlass", "sm90"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
