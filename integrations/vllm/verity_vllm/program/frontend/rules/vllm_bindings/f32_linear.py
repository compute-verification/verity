"""The f32 linear on a widened bf16 activation: `nn.Linear(dtype=float32)` on `x.to(float32)` (GLM-4.7-Flash's MoE router gate), which
vLLM's batch-invariant mode leaves to cuBLAS off SM80 with split-K off (`CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUBLASLT_WORKSPACE_SIZE=1`)."""

from __future__ import annotations

import importlib.metadata

import torch

from verity_vllm.program.frontend.torch_frontend import shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.norm_chain import widened_bf16
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.operands import aten

CUBLAS_STACK = {"torch": "2.13.0+cu129", "nvidia-cublas-cu12": "12.9.1.4"}
CUBLAS_CC = (12, 0)

# (K, N) -> ((M_lo, M_hi, kernel, TREE), ...): the kernel cuBLAS launches for x[M, K] @ W[N, K]^T on sm_120 at CUBLAS_STACK with split-K
# off (r20261004-065341-db1b) and the `CublasF32Linear` tree it reduces with (exact LCA queries, r20261004-070541-1622); an M in no
# range is unmeasured
F32_LINEAR_KERNELS = {
    (2048, 64): ((1, 1, "gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false>", "gemv"),
                 (8, 8, "gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false>", "gemv"),
                 (256, 256, "cutlass_80_simt_sgemm_128x32_8x5_tn_align1", "simt")),
}


def cublas_f32_kernel(K: int, N: int, M: int) -> tuple[str, str] | None:
    """(kernel, TREE) cuBLAS runs for M rows of the (K, N) f32 linear, None when unmeasured."""
    for lo, hi, kernel, tree in F32_LINEAR_KERNELS.get((K, N), ()):
        if lo <= M <= hi:
            return kernel, tree
    return None


def installed_stack() -> dict[str, str | None]:
    def version(dist: str) -> str | None:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {"torch": torch.__version__, "nvidia-cublas-cu12": version("nvidia-cublas-cu12")}


class CublasF32LinearRule(OpRule):
    """`aten.linear(x f32 [.., K], W f32 [N, K])` with x the widened bf16 activation -> `CublasF32Linear{K,N,TREE}` per row of x, TREE
    the kernel family cuBLAS runs for x's M rows (`cublas_f32_kernel`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("CublasF32Linear[vllm]", 1, "op-binding", applies_to="aten.linear(x float32, W float32) with no bias",
                    preconditions=("x is an f32 tensor carrying bf16 words (`WidenBf16ToF32`); W a genuine f32 value [N, K], K = x.shape[-1]",
                                   "no bias", f"the target is sm_{CUBLAS_CC[0]}{CUBLAS_CC[1]} under batch invariance (cuBLAS, split-K off)",
                                   f"the derive's stack is {CUBLAS_STACK} (the kernel selection the table records)",
                                   "x's row count M has a measured kernel for (K, N) (`F32_LINEAR_KERNELS`)"),
                    evidence_domain="numerics: CublasF32Linear_v1 (registry.cublas_f32), its trees exact on the sm_120 probe")

    def targets(self):
        return {aten.linear.default}

    def applies(self, node, ctx):
        x, w = node.args[0], node.args[1]
        if len(node.args) > 2 and node.args[2] is not None:
            return "numerics_unregistered: an f32 linear with a bias"
        if widened_bf16(ctx, x) is not True:
            return "x is not a widened bf16 activation (an f32 tensor carrying bf16 words)"
        if widened_bf16(ctx, w) is not False:
            return "W is not a genuine f32 value collection"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if not xs or ws is None or len(ws) != 2 or xs[-1] != ws[1]:
            return f"shape: x {xs} vs W {ws}"
        prof = active_target_profile()
        if prof.compute_capability != CUBLAS_CC or not prof.batch_invariant:
            return (f"numerics_unregistered: cuBLAS's f32 kernels are measured on sm_{CUBLAS_CC[0]}{CUBLAS_CC[1]} under batch invariance; "
                    f"the target is cc {prof.compute_capability}, batch_invariant={prof.batch_invariant}")
        if (got := installed_stack()) != CUBLAS_STACK:
            return f"numerics_unregistered: cuBLAS's f32 kernel table is measured at {CUBLAS_STACK}, this derive runs {got}"
        M = 1
        for d in xs[:-1]:
            M *= d
        if cublas_f32_kernel(ws[1], ws[0], M) is None:
            return f"numerics_unregistered: cuBLAS's f32 kernel at K={ws[1]}, N={ws[0]}, M={M} is unmeasured"
        return True

    def emit(self, node, ctx):
        x, w = ctx.coll(node.args[0], "x"), ctx.coll(node.args[1], "W")
        N, K = shape_of(w.type)[0]
        rows, lead = rows_of(x)
        kernel, tree = cublas_f32_kernel(K, N, len(rows))
        fn = ctx.kind("f32_linear_cublas", node, K=K, N=N, TREE=tree)
        outs = [ctx.call(fn, r, w, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(node.args[1], self, "linear.weight.f32", K=K, N=N)
        ctx.record(self, node, {"K": K, "N": N, "M": len(rows), "kernel": kernel, "TREE": tree},
                   emitted=f"{fn.id} x{len(rows)}")


__all__ = ["CUBLAS_CC", "CUBLAS_STACK", "CublasF32LinearRule", "F32_LINEAR_KERNELS", "cublas_f32_kernel", "installed_stack"]
