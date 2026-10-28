"""Operand helpers the binding rules share."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc, refuse
from verity_vllm.program.frontend.strided_views import _permute
from verity_vllm.program.frontend.rules.common import dtype_is, shape


aten = torch.ops.aten
BF16 = torch.bfloat16


def _row_value(ctx, a, name: str, dtype=BF16) -> str | None:
    if not isinstance(a, torch.fx.Node):
        return f"{name} is not a graph value"
    v = ctx.val(a)
    if isinstance(v, Alloc):
        return None
    if not isinstance(v, Coll):
        return f"{name} is not a value collection ({type(v).__name__})"
    if not dtype_is(ctx, a, dtype):
        return f"numerics_unregistered: {name} has dtype {ctx.dtype_of(a)}; the profile's kinds are {dtype}"
    if not shape(ctx, a):
        return f"{name} is a scalar"
    return None


def _strides(n) -> tuple[int, ...] | None:
    """The exported operand's strides (fx `val` meta), or None."""
    v = n.meta.get("val") if isinstance(n, torch.fx.Node) else None
    return tuple(int(s) for s in v.stride()) if hasattr(v, "stride") else None


def _untransposed(ctx, b, K: int, N: int, hint: str) -> Coll:
    """`b` [K, N] is `W.t()` (strides checked by the caller); the [N, K] matrix it views. Taken from the transpose's operand when
    `b` is literally an `aten.t` / 2-D `transpose` node (the common case: a parameter), else by permuting the references back."""
    if isinstance(b, torch.fx.Node) and b.target in (aten.t.default, aten.transpose.int, aten.numpy_T.default) and shape(ctx, b.args[0]) == (N, K):
        return ctx.coll(b.args[0])
    from verity_vllm.program.frontend.errors import FrontendError
    try:
        return _permute(ctx.coll(b), [1, 0], hint)
    except FrontendError as e:
        refuse("unsupported_view", ctx.current, ctx.module_path(ctx.current), f"b is not a transposed view of a [N, K] matrix: {e.message}")
