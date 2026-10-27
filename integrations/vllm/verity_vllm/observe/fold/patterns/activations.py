"""Activation patterns: the fused SiLU-and-multiply, and its clamped form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from verity_vllm.program.frontend.rules.vocab import vocabulary_for

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_row_major
from verity_vllm.observe.fold.patterns.ops import as_float, bound_args, first, op_name
from verity_vllm.observe.fold.patterns.resolutions import _fmt, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import target_of
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode


@dataclass
class SiluAndMul:
    """`_C.silu_and_mul(result!, input)` → `Compute(<silu_mul>{I}, rows=M, (PerRow(gu[M,2I]),), (PerRow(out[M,I]),))`, where `<silu_mul>`
    is the Definition the record target's `numerics_profile` vocabulary binds the Build's `silu_mul` kind to (`SiluMul_v2` under
    b1-eager-v3)."""

    name: str = "act.silu_and_mul"
    op: ClassVar[str] = "silu_and_mul"
    kind: ClassVar[str] = "silu_mul"

    def statics(self, node: TreeNode, a: dict) -> dict | Unsupported:
        """The kind's statics beyond I that the call's arguments give."""
        return {}

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != self.op:
            return None
        a = bound_args(node)
        out = first(a.get("result"), a.get("out"), a.get("arg0"))
        x = first(a.get("input"), a.get("x"), a.get("arg1"))
        if not isinstance(out, ViewDesc) or not isinstance(x, ViewDesc):
            return unsupported(node, f"{self.op} arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or out.rank != 2 or x.shape[0] != out.shape[0] or x.shape[1] != 2 * out.shape[1] or not is_row_major(x) or not is_row_major(out):
            return unsupported(node, f"{self.op} views are not row-major [M,2I] -> [M,I]", x=repr(x), out=repr(out))
        if not (x.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "SiluMul is bf16", dtypes=[x.dtype, out.dtype])
        statics = self.statics(node, a)
        if isinstance(statics, Unsupported):
            return statics
        M, I = out.shape
        numerics = target_of(ctx)[0].numerics_profile
        try:
            spec = vocabulary_for("profile", numerics).kind(self.kind, I=I, **statics)
        except ValueError as e:
            return unsupported(node, f"the record target's numerics profile has no vocabulary: {e}", numerics_profile=numerics)
        if spec is None:
            return unsupported(node, f"the record target's numerics profile binds no {self.kind} kind", numerics_profile=numerics)
        return Compute(spec, M, (PerRow(x),), (PerRow(out),), (node.id,), hint=hint_of(node, self.op),
                       kernels=("act_and_mul_kernel",), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class SiluAndMulWithClamp(SiluAndMul):
    """`_C.silu_and_mul_with_clamp(result!, input, limit, alpha=1, beta=0)` (DeepSeek-V4's SwiGLU limit) → the same Compute over the
    vocabulary's `silu_mul_clamp{I,LIMIT}` (`SiluMulClamp_v1`), the Build's `SiluAndMulWithClampRule`; SwiGLU-OAI's alpha and beta
    have no Definition."""

    name: str = "act.silu_and_mul_with_clamp"
    op: ClassVar[str] = "silu_and_mul_with_clamp"
    kind: ClassVar[str] = "silu_mul_clamp"

    def statics(self, node: TreeNode, a: dict) -> dict | Unsupported:
        from verity.ml.definitions.registry.dsv4_moe import limit_bits  # the DeepSeek-V4 registry on use
        limit, alpha, beta = as_float(a.get("limit")), as_float(a.get("alpha", 1.0)), as_float(a.get("beta", 0.0))
        if (alpha, beta) != (1.0, 0.0):
            return unsupported(node, "silu_and_mul_with_clamp's alpha and beta are not 1 and 0 (SiluMulClamp_v1 is the clamp alone)",
                               alpha=alpha, beta=beta)
        try:
            limit_bits(limit)
        except (TypeError, ValueError) as e:
            return unsupported(node, f"silu_and_mul_with_clamp's limit: {e}", limit=repr(a.get("limit")))
        return {"LIMIT": float(limit)}
