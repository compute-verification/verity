"""Activation patterns: the fused SiLU-and-multiply."""

from __future__ import annotations

from dataclasses import dataclass

from verity.ir.defs import bind
from verity_vllm.program.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_row_major
from verity_vllm.observe.fold.patterns.ops import bound_args, first, op_name
from verity_vllm.observe.fold.patterns.resolutions import _fmt, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, StepContext
from verity_vllm.observe.fold.tree import TreeNode


@dataclass
class SiluAndMul:
    """`_C.silu_and_mul(result!, input)` → `Compute(SiluMul_v1{I}, rows=M, (PerRow(gu[M,2I]),), (PerRow(out[M,I]),))`."""

    name: str = "act.silu_and_mul"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "silu_and_mul":
            return None
        a = bound_args(node)
        out = first(a.get("result"), a.get("out"), a.get("arg0"))
        x = first(a.get("input"), a.get("x"), a.get("arg1"))
        if not isinstance(out, ViewDesc) or not isinstance(x, ViewDesc):
            return unsupported(node, "silu_and_mul arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or out.rank != 2 or x.shape[0] != out.shape[0] or x.shape[1] != 2 * out.shape[1] or not is_row_major(x) or not is_row_major(out):
            return unsupported(node, "silu_and_mul views are not row-major [M,2I] -> [M,I]", x=repr(x), out=repr(out))
        if not (x.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "SiluMul_v1 is bf16", dtypes=[x.dtype, out.dtype])
        M, I = out.shape
        return Compute(bind(b1.SiluMul, I=I), M, (PerRow(x),), (PerRow(out),), (node.id,), hint=hint_of(node, "silu_and_mul"),
                       kernels=("act_and_mul_kernel",), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
