"""PoUW's executed linears: the host executor's own record of each call, never a device launch.

Under ``pouw:ncp-v2`` each linear layer's ``apply`` is PoUW's executor. It computes the call on the host and returns its served
y as one dispatcher op, ``verity_pouw.ncp_linear(x, weight, bias, entry, w)`` (`protocol_options.pouw.ncp_linear_op`), so an
observer records the op with its operands and output and nothing of the host work inside it. `PouwLinear` folds that record
into the call's served rows, the Definitions the Build binds the same linear to under the same target (`TritonGemmRule`). Every
other GEMM stays the Triton launch's (`gemm.GemmLaunch`). `with_protocols` adds the pattern to a run's profile when its capture
installed a PoUW executor.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_row_major
from verity_vllm.observe.fold.patterns.ops import base_op
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.resolver import Compute, PerRow, Profile, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode
from verity.ml.definitions.registry import pouw_rows

OP = "verity_pouw.ncp_linear"


@dataclass
class PouwLinear:
    """``verity_pouw.ncp_linear(x[M,K], W[N,K], b[N] | None, entry, w) -> y[M,N]`` -> `Compute(NcpLinearRow_v1{K,N}, rows=M,
    (PerRow(x), Shared(W)), (PerRow(y),))`, or `NcpLinearRowBias_v1{K,N}` with `Shared(b)` when the call carries its bias.
    x, W, b and y are bf16 row-major; W and b are parameters."""

    name: str = "pouw.ncp_linear"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != OP:
            return None
        args = list(node.args) + [None] * 3
        x, w, b = args[0], args[1], args[2]
        outs = node.output_views()
        if not (isinstance(x, ViewDesc) and isinstance(w, ViewDesc) and outs) or (b is not None and not isinstance(b, ViewDesc)):
            return unsupported(node, "ncp_linear without tensor x, weight and output views", args=[repr(a) for a in node.args[:3]])
        y = outs[0]
        if x.rank != 2 or w.rank != 2 or y.rank != 2 or x.shape[1] != w.shape[1] or y.shape != (x.shape[0], w.shape[0]):
            return unsupported(node, "ncp_linear shapes are not x[M,K], W[N,K], y[M,N]", x=repr(x), w=repr(w), y=repr(y))
        (M, K), N = x.shape, w.shape[0]
        if b is not None and b.shape != (N,):
            return unsupported(node, "ncp_linear bias is not b[N]", b=repr(b), N=N)
        views = (x, w, y) + ((b,) if b is not None else ())
        if any(v.dtype != "bfloat16" for v in views) or not all(is_row_major(v) for v in views):
            return unsupported(node, "ncp_linear's operands are bf16 row-major", dtypes=[v.dtype for v in views])
        if K % 16:
            return unsupported(node, "K is not a multiple of the 16-wide dot step", K=K)
        params = (w,) + ((b,) if b is not None else ())
        if not all(any(r.role == "weight" for r in ctx.roots_of(p)) for p in params):
            return unsupported(node, "ncp_linear's weight or bias is not a parameter")
        names = tuple(ctx.roots_of(p)[0].name for p in params)
        spec = bind(pouw_rows.NcpLinearRowBias if b is not None else pouw_rows.NcpLinearRow, K=K, N=N)
        return Compute(spec, M, (PerRow(x), *(Shared(p) for p in params)), (PerRow(y),), tuple(sorted(node.ids())),
                       hint=hint_of(node, names[0]), attribution=attribution_for(ctx, M, node.phase), pattern=self.name,
                       roots=(None, *names))


def with_protocols(profile: Profile, header: Any) -> Profile:
    """``profile`` with `PouwLinear` first when the run's header records a PoUW executor (``notes.protocols``, written by the
    capture that installed it: `pipeline.m1_capture`); unchanged otherwise."""
    entries = ((getattr(header, "notes", None) or {}).get("protocols") or ())
    if not any(str(e).partition(":")[0] == "pouw" for e in entries):
        return profile
    return dataclasses.replace(profile, patterns=[PouwLinear(), *profile.patterns])


__all__ = ["OP", "PouwLinear", "with_protocols"]
