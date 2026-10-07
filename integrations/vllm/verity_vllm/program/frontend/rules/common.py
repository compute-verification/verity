"""Helpers shared by op rules: row iteration over leading axes, dtype checks."""

from __future__ import annotations

import math

import numpy as np

import torch

from verity.ml.circuits.refs import Coll, PartLog, array_of
from verity.ml.circuits.types import Array
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of


def kv_logs(ctx, layer_name: str, krow_list: list[Coll], vrow_list: list[Coll]) -> tuple[PartLog, PartLog]:
    """The key and value rows of `layer_name`'s history as `PartLog`s, one row per part, so that the rows an attention call
    reads are `log.cut(P)`. The logs follow the history lists: a row the cache write rebound starts a new log."""
    logs = ctx.state.setdefault("kv_logs", {})
    have = logs.get(layer_name) or (None, None)
    out = []
    for log, rows in zip(have, (krow_list, vrow_list)):
        if log is None or len(log) > len(rows) or any(p is not r.refs for p, r in zip(log.parts, rows)):
            log = PartLog()
        for r in rows[len(log):]:
            log.append(r.refs)
        out.append(log)
    logs[layer_name] = (out[0], out[1])
    return out[0], out[1]


def norm_kind(ctx, role: str, node, N: int, eps: float, **statics) -> tuple[object, tuple[Coll, ...]]:
    """The norm Definition for `role` (`rmsnorm`, `rmsnorm_fused`, `layer_norm`, `layer_norm_aten`; its other `statics`) and its trailing
    operands: `EPS` a static, or under `registered_values` eps the registered value `eps` (the f32 word the kernel reads), the same name
    the fold supplies."""
    if not ctx.registered_values:
        return ctx.kind(role, node, N=N, EPS=eps, **statics), ()
    bits = int(np.float32(eps).view(np.uint32))
    return ctx.kind(f"{role}_value", node, N=N, **statics), (ctx.supplied("eps", 32, bits, node),)


def scalar_kind(ctx, role: str, node, N: int, C: float) -> tuple[object, tuple[Coll, ...]]:
    """The elementwise kind `role` (`bf16_div_scalar`, `bf16_mul_scalar`, `add_scalar_f32`, `add_scalar_bf16`) and its trailing operands:
    `C` a static, or under `registered_values` the f32 word the ATen kernel reads (the division's `RN32(1 / C)`, else `f32(C)`), a
    registered value of its own named by the role (`div_scalar`, `mul_scalar`, `add_scalar_f32`, `add_scalar_bf16`; `_1`, ... for
    another value)."""
    if not ctx.registered_values:
        return ctx.kind(role, node, N=N, C=C), ()
    from verity_vllm.program.registry.dense import reciprocal_bits
    bits = reciprocal_bits(C) if role == "bf16_div_scalar" else int(np.float32(C).view(np.uint32))
    return ctx.kind(f"{role}_value", node, N=N), (ctx.supplied(role.removeprefix("bf16_"), 32, bits, node, distinct=True),)


def attention_scale(ctx, D: int, node, cap: float | None = None) -> tuple[Coll, ...]:
    """The attention chain's trailing operands: none (the scale `D^-0.5 * log2(e)` a constant), or under `registered_values` the
    registered value `scale_log2_d{D}`, the same name the fold supplies; a softcapped chain's (`cap`) are FA2's two words instead,
    `softcap_mul_d{D}` (`softmax_scale / cap`) and `softcap_scale_log2` (`cap * log2(e)`)."""
    if not ctx.registered_values:
        return ()
    if cap is not None:
        from verity_vllm.program.registry.dense import softcap_mul_bits, softcap_scale_log2_bits
        return (ctx.supplied(f"softcap_mul_d{D}", 32, softcap_mul_bits(D, cap), node),
                ctx.supplied("softcap_scale_log2", 32, softcap_scale_log2_bits(cap), node))
    from verity_vllm.program.registry import b1
    return (ctx.supplied(f"scale_log2_d{D}", 32, b1._scale_log2_bits(D), node),)


def rows_of(x: Coll) -> tuple[list[Coll], tuple[int, ...]]:
    """`x[..., K]` as its rows `[K]` in row-major order, plus the leading shape."""
    shape, w = shape_of(x.type)
    if not shape:
        raise ValueError("a scalar has no rows")
    K = shape[-1]
    lead = tuple(shape[:-1])
    n = math.prod(lead) if lead else 1
    flat = x.reshape(Array(n, Array(K, x.type if not isinstance(x.type, Array) else _leaf(x.type))))
    return [flat[i] for i in range(n)], lead


def _leaf(t):
    while isinstance(t, Array):
        t = t.elem
    return t


def from_rows(rows: list[Coll], lead: tuple[int, ...]) -> Coll:
    """Reassemble rows `[N]` into `[*lead, N]`."""
    if not rows:
        raise ValueError("no rows")
    out = array_of(rows)
    N = rows[0].type.n if isinstance(rows[0].type, Array) else 1
    w = shape_of(rows[0].type)[1]
    return out.reshape(_tensor_type(tuple(lead) + (N,), w)) if lead else rows[0]


def dtype_is(ctx, a, *dts: torch.dtype) -> bool:
    dt = ctx.dtype_of(a) if isinstance(a, torch.fx.Node) else None
    return dt in dts


def shape(ctx, a) -> tuple[int, ...] | None:
    return ctx.shape_of_node(a) if isinstance(a, torch.fx.Node) else None


__all__ = ["rows_of", "from_rows", "dtype_is", "shape"]
