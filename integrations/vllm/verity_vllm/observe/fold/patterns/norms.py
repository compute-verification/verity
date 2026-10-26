"""RMSNorm patterns: the residual-free Triton norm launch and the fused residual-add norm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import value_inputs as VI

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_row_major
from verity_vllm.observe.fold.patterns.ops import as_float, base_op, bound_args, first, op_name, triton_name
from verity_vllm.observe.fold.patterns.resolutions import _check_pins, _fmt, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import _targets, target_of
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, Shared, StepContext, Supplied
from verity_vllm.observe.fold.tree import TreeNode


def eps_input(eps: float) -> Supplied:
    """A norm's epsilon as the registered value `eps`: the f32 word the kernel reads (vLLM passes the Python float, the launch narrows it)."""
    return Supplied("registration", "eps", 32, int(P.bits_of(np.float32(eps))))


@dataclass
class RMSNormTritonLaunch:
    """`TritonLaunch …batch_invariant._rms_norm_kernel` (vLLM `rms_norm_batch_invariant`, the residual-free norm:
    layer 0's input norm) → `Compute(RMSNormTriton_v1{N,EPS}, rows=M, (PerRow(x), Shared(w)), (PerRow(out),))`.
    Pins: `BLOCK_SIZE=1024`, `HAS_WEIGHT=True`, `num_warps=4` (the reduction tree the registered body writes out);
    `eps` must equal the profile's `EPS`."""

    constexprs: dict[str, Any]
    num_warps: int | None = None
    eps: float | None = None
    name: str = "rmsnorm.triton_rms_norm_kernel"
    kernel: str = "_rms_norm_kernel"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or triton_name(node) != self.kernel:
            return None
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        got = {k: first(consts.get(k), a.get(k)) for k in self.constexprs}
        bad = _check_pins(node, got, self.constexprs)
        if bad is not None:
            return bad
        if self.num_warps is not None:
            bad = _check_pins(node, {"num_warps": node.enter.num_warps}, {"num_warps": self.num_warps})
            if bad is not None:
                return bad
        x, w, out = a.get("input_ptr"), a.get("weight_ptr"), a.get("output_ptr")
        n_cols, eps = a.get("n_cols"), as_float(a.get("eps"))
        if not all(isinstance(v, ViewDesc) for v in (x, w, out)) or not isinstance(n_cols, int) or eps is None:
            return unsupported(node, "rms_norm launch arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or x.shape[1] != n_cols or not is_row_major(x) or out.shape != x.shape or not is_row_major(out):
            return unsupported(node, "input/output are not row-major [M,N] views", x=repr(x), out=repr(out), n_cols=n_cols)
        if w.shape != (n_cols,) or w.strides != (1,):
            return unsupported(node, "weight is not a contiguous [N] view", w=repr(w))
        if not (x.dtype == w.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "RMSNormTriton_v1 is bf16", dtypes=[x.dtype, w.dtype, out.dtype])
        if self.eps is not None and eps != self.eps:
            return unsupported(node, "eps differs from the profile's EPS", eps=eps, pinned=self.eps)
        M = x.shape[0]
        if node.enter.grid and node.enter.grid[0] != M:
            return unsupported(node, "grid does not launch one program per row", grid=node.enter.grid, M=M)
        roots = ctx.roots_of(w)
        spec, operands = bind(b1.RMSNormTriton, N=n_cols, EPS=self.eps if self.eps is not None else eps), (PerRow(x), Shared(w))
        if getattr(ctx.profile, "registered_values", False):
            spec, operands = bind(VI.RMSNormTritonV3, N=n_cols), operands + (eps_input(self.eps if self.eps is not None else eps),)
        return Compute(spec, M, operands, (PerRow(out),), (node.id,), hint=hint_of(node, roots[0].name if roots else "rms_norm"),
                       kernels=(self.kernel,), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class FusedAddRMSNorm:
    """`_C.fused_add_rms_norm(input!, residual!, weight, epsilon)` (in place: input ← norm(input+residual),
    residual ← input+residual) → `Compute(RMSNormFusedCuda_v2{N,EPS}, rows=M, (PerRow(x), PerRow(res), Shared(w)),
    (PerRow(x), PerRow(res)))` on a batch-invariant record target (`frontend.targets.rmsnorm_fused_spec`, the selector the Build's
    vocabulary calls; refused by name elsewhere).  The fold reads all operands before writing (the outputs alias the inputs)."""

    eps: float | None = None
    name: str = "rmsnorm.fused_add_rms_norm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "fused_add_rms_norm":
            return None
        a = bound_args(node)
        x, res, w = a.get("input", a.get("arg0")), a.get("residual", a.get("arg1")), a.get("weight", a.get("arg2"))
        eps = as_float(first(a.get("epsilon"), a.get("eps"), a.get("arg3")))
        if not all(isinstance(v, ViewDesc) for v in (x, res, w)) or eps is None:
            return unsupported(node, "fused_add_rms_norm arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or not is_row_major(x) or res.shape != x.shape or not is_row_major(res):
            return unsupported(node, "input/residual are not row-major [M,N] views of one shape", x=repr(x), res=repr(res))
        N = x.shape[1]
        if w.shape != (N,) or w.strides != (1,):
            return unsupported(node, "weight is not a contiguous [N] view", w=repr(w), N=N)
        if not (x.dtype == res.dtype == w.dtype == "bfloat16"):
            return unsupported(node, "RMSNormFusedCuda_v2 is bf16", dtypes=[x.dtype, res.dtype, w.dtype])
        if self.eps is not None and eps != self.eps:
            return unsupported(node, "epsilon differs from the profile's EPS", eps=eps, pinned=self.eps)
        M = x.shape[0]
        roots = ctx.roots_of(w)
        eps = self.eps if self.eps is not None else eps
        registered = getattr(ctx.profile, "registered_values", False)
        try:
            spec = _targets().rmsnorm_fused_spec(target_of(ctx)[0], N, eps, eps_input=registered, M=M)
        except _targets().UnregisteredTarget as e:
            return unsupported(node, f"target has no registered fused_add_rms_norm binding: {e}")
        operands = (PerRow(x), PerRow(res), Shared(w)) + ((eps_input(eps),) if registered else ())
        return Compute(spec, M, operands, (PerRow(x), PerRow(res)), (node.id,),
                       hint=hint_of(node, roots[0].name if roots else "fused_add_rms_norm"), kernels=("fused_add_rms_norm_kernel",),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


def _cuda_norm_args(node: TreeNode, names: tuple[str, ...]) -> tuple | None:
    a = bound_args(node)
    vals = tuple(a.get(n, a.get(f"arg{i}")) for i, n in enumerate(names))
    return vals if all(isinstance(v, ViewDesc) for v in vals[:-1]) and isinstance(vals[-1], float) else None


@dataclass
class CudaRmsNorm:
    """`_C.rms_norm(out! [M,N], x [M,N], weight [N], eps)` off batch invariance (vLLM's default `rms_norm_kernel<bf16, 8, 2, true>`) ->
    `Compute(RMSNormCudaVector8_v1{N,EPS,BLOCK}, rows=M, (PerRow(x), Shared(weight)), (PerRow(out),))`, BLOCK
    `cuda_rmsnorm.launch_block(M, N, fused=False)`: the Build's `CudaRmsNormRule`.  A batch-invariant target is another pattern's."""

    name: str = "norm.rms_norm_cuda"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "_C.rms_norm" or target_of(ctx)[0].batch_invariant:
            return None
        from verity.ml.definitions.registry import cuda_rmsnorm as C

        got = _cuda_norm_args(node, ("result", "input", "weight", "epsilon"))
        if got is None:
            return unsupported(node, "rms_norm arguments not bound as (out, input, weight, eps)")
        out, x, w, eps = got
        if not _rows_ok((out, x)) or out.shape != x.shape or w.shape != (x.shape[1],) or w.dtype != "bfloat16":
            return unsupported(node, "rms_norm off batch invariance takes row-major bf16 [M,N] out / x and a [N] bf16 weight", out=repr(out), x=repr(x), w=repr(w))
        M, N = x.shape
        spec = bind(C.RMSNormCudaVector8, N=N, EPS=eps, BLOCK=C.launch_block(M, N, fused=False))
        roots = ctx.roots_of(w)
        return Compute(spec, M, (PerRow(x), Shared(w)), (PerRow(out),), tuple(sorted(node.ids())), hint=hint_of(node, roots[0].name if roots else "rms_norm"),
                       kernels=("rms_norm_kernel",), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class CudaFusedAddRmsNorm:
    """`_C.fused_add_rms_norm(x! [M,N], residual! [M,N], weight [N], eps)` off batch invariance (`fused_add_rms_norm_kernel<bf16, 8>`) ->
    `Compute(RMSNormFusedCudaVector8_v1{N,EPS,BLOCK}, rows=M, (PerRow(x), PerRow(residual), Shared(weight)), (PerRow(x),
    PerRow(residual)))` in place, BLOCK `cuda_rmsnorm.launch_block(M, N, fused=True)`: the Build's `CudaFusedAddRmsNormRule`."""

    name: str = "norm.fused_add_rms_norm_cuda"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "_C.fused_add_rms_norm" or target_of(ctx)[0].batch_invariant:
            return None
        from verity.ml.definitions.registry import cuda_rmsnorm as C

        got = _cuda_norm_args(node, ("input", "residual", "weight", "epsilon"))
        if got is None:
            return unsupported(node, "fused_add_rms_norm arguments not bound as (input, residual, weight, eps)")
        x, res, w, eps = got
        if not _rows_ok((x, res)) or x.shape != res.shape or x.alloc == res.alloc or w.shape != (x.shape[1],) or w.dtype != "bfloat16":
            return unsupported(node, "fused_add_rms_norm off batch invariance takes distinct row-major bf16 [M,N] x / residual and a [N] bf16 weight",
                               x=repr(x), residual=repr(res), w=repr(w))
        M, N = x.shape
        spec = bind(C.RMSNormFusedCudaVector8, N=N, EPS=eps, BLOCK=C.launch_block(M, N, fused=True))
        roots = ctx.roots_of(w)
        return Compute(spec, M, (PerRow(x), PerRow(res), Shared(w)), (PerRow(x), PerRow(res)), tuple(sorted(node.ids())),
                       hint=hint_of(node, roots[0].name if roots else "fused_add_rms_norm"), kernels=("fused_add_rms_norm_kernel",),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


def _rows_ok(views: tuple) -> bool:
    return all(v.rank == 2 and v.dtype == "bfloat16" and is_row_major(v) for v in views)
