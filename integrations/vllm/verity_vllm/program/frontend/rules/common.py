"""Helpers shared by op rules: row iteration over leading axes, dtype checks."""

from __future__ import annotations

import math

import torch

from verity.ir.refs import Coll, PartLog, array_of
from verity.ir.types import Array
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
