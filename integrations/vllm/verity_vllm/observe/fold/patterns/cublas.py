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
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_weight_view, is_row_major
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import _targets, target_of, vocab_kind
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
        from verity.ml.definitions.registry import b1, bf16_gemm_check, cublas_bf16

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
                       kernels=("gemm", "splitK", "cutlass"), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class CublasMmF32Out:
    """`aten.mm.dtype(x [M,K] bf16, W.T, float32)` over a bf16 [N,K] parameter on cuBLAS's captured f32-output route: the Build's
    `vllm_bindings.bf16_gemm.CublasRouterRule` (`f32out_route`), one instance per row over the untransposed W."""

    name: str = "gemm.cublas_mm_f32out"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or node.op != "aten.mm.dtype":
            return None
        tp, _recorded = target_of(ctx)
        if tp.batch_invariant or tuple(tp.compute_capability) != (9, 0):
            return None
        from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import f32out_route

        a = bound_args(node)
        x, wt, outs = a.get("self", a.get("arg0")), a.get("mat2", a.get("arg1")), node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(wt, ViewDesc) or len(outs) != 1:
            return unsupported(node, "aten.mm.dtype arguments not bound as (self, mat2)", x=repr(x), wt=repr(wt))
        out = outs[0]
        if x.rank != 2 or wt.rank != 2 or x.shape[1] != wt.shape[0]:
            return unsupported(node, "aten.mm.dtype takes [M,K] x [K,N]", x=repr(x), wt=repr(wt))
        (M, K), N = x.shape, wt.shape[1]
        try:
            w = as_weight_view(wt, K, N)
        except GeometryError as e:
            return unsupported(node, f"the mm's mat2 is not W.T of a row-major [N,K] weight: {e}")
        if (x.dtype, w.dtype, out.dtype) != ("bfloat16", "bfloat16", "float32") or not is_row_major(x) or not is_row_major(out) \
                or out.shape != (M, N):
            return unsupported(node, "aten.mm.dtype is not row-major bf16 [M,K] x W.T -> f32 [M,N]", x=repr(x), w=repr(w), out=repr(out))
        roots = [r for r in ctx.roots_of(w) if r.role == "weight"]
        if not roots:
            return unsupported(node, "the mm's weight is not a checkpoint parameter (weight Root)", w=repr(w))
        route = f32out_route(M, K, N)
        if isinstance(route, str):
            return unsupported(node, route)
        _phase, role, statics = route
        T = _targets()
        try:
            dot = T.tensor_core_dot(tp)
        except T.UnregisteredTarget as e:
            return unsupported(node, str(e))
        return Compute(vocab_kind(ctx, role, K=K, N=N, DOT=dot, **statics), M, (PerRow(x), Shared(w)), (PerRow(out),),
                       tuple(sorted(node.ids())), hint=hint_of(node, roots[0].name), kernels=("gemm", "splitK"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name, roots=(None, roots[0].name))


def _permuted(v: ViewDesc, order: tuple[int, ...]) -> ViewDesc:
    return ViewDesc(v.alloc, v.offset, tuple(v.shape[i] for i in order), tuple(v.strides[i] for i in order), v.dtype, v.device)


def _holds(r: ViewDesc, v: ViewDesc) -> bool:
    """`v` is a sub-box of `r`'s axes: its first element is an in-bounds index of `r`, and each of its axes runs in bounds along one of
    `r`'s.  W_UK_T and W_UV interleave in kv_b's one allocation, so the allocation's Roots alone do not say which one a read is."""
    axes = sorted((d for d in range(r.rank) if r.shape[d] > 1), key=lambda d: -r.strides[d])
    coord, rem = {}, v.offset - r.offset
    if rem < 0 or r.dtype != v.dtype:
        return False
    for d in axes:
        coord[d], rem = divmod(rem, r.strides[d])
    if rem or any(coord[d] >= r.shape[d] for d in axes):
        return False
    used: set[int] = set()
    for n, st in zip(v.shape, v.strides):
        if n == 1:
            continue
        d = next((d for d in axes if d not in used and r.strides[d] == st), None)
        if d is None or coord[d] + n > r.shape[d]:
            return False
        used.add(d)
    return True


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
        from verity.ml.definitions.registry import cublas_bf16

        a = bound_args(node)
        x, w = a.get("self", a.get("arg0")), a.get("mat2", a.get("arg1"))
        outs = [a["out"]] if isinstance(a.get("out"), ViewDesc) else node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(w, ViewDesc) or len(outs) != 1:
            return unsupported(node, "aten.bmm arguments not bound as (self, mat2)", x=repr(x), w=repr(w))
        out = outs[0]
        if x.rank != 3 or w.rank != 3 or out.rank != 3 or {x.dtype, w.dtype, out.dtype} != {"bfloat16"}:
            return unsupported(node, "MLA absorb bmm takes bf16 [B,M,K] x [B,K,N] -> [B,M,N]", x=repr(x), w=repr(w), out=repr(out))
        (B, M, K), (B2, K2, N) = x.shape, w.shape
        if (B2, K2) != (B, K) or out.shape != (B, M, N) or x.strides[2] != 1 or out.strides[2] != 1:
            return unsupported(node, "bmm shapes or inner strides are not [B,M,K] x [B,K,N] -> [B,M,N] with unit-stride x and out rows",
                               x=repr(x), w=repr(w), out=repr(out))
        roots = ctx.roots_of(w)
        if not roots:
            return unsupported(node, "the absorbed weight is not a view of a parameter's storage (no Root)", w=repr(w))
        held = [r for r in roots if _holds(r.view, w)]
        if len(roots) > 1 and len(held) != 1:
            return unsupported(node, "the absorbed weight lies in no single Root of its storage", w=repr(w), roots=[r.name for r in roots])
        root = held[0] if held else roots[0]
        T = _targets()
        try:
            dot = T.tensor_core_dot(tp)
        except T.UnregisteredTarget as e:
            return unsupported(node, str(e))
        spec = bind(cublas_bf16.BatchedGemmWord, B=B, K=K, N=N, DOT=dot)
        return Compute(spec, M, (PerRow(_permuted(x, (1, 0, 2))), Shared(_permuted(w, (0, 2, 1)))), (PerRow(_permuted(out, (1, 0, 2))),),
                       tuple(sorted(node.ids())), hint=hint_of(node, root.name), kernels=("gemm", "cutlass"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
