"""The training rules for softmax and log-softmax over the last axis of an f32 value, forward and backward, which `training_rules()`
places with the others: SDPA's math path's `_safe_softmax`, `_softmax_backward_data`, `_log_softmax` and
`_log_softmax_backward_data`, in torch CPU's form with the reference vocabulary's sequential sums (`ref_prims`).

    SafeSoftmax, SoftmaxBackward, LogSoftmax, LogSoftmaxBackward
"""

from __future__ import annotations

import torch

from verity.circuits.defs import bind
from verity.definitions.ops import scalar as SC
from verity.circuits.refs import Coll, Strided
from verity.circuits.types import Array
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as R
from verity_vllm.program.frontend.rules.training_emit import (BIT, F32, _batch, _const, _ones, _operand_ok, _per_row, _rows, _shape,
                                                              _TrainRule, aten)
from verity_vllm.program.frontend.rules.training_views import _info
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of


def _last_dim(ctx, node, a, dim) -> str | None:
    S = _shape(ctx, a)
    if not S or dim % len(S) != len(S) - 1:
        return f"over dim {dim} of {list(S)}, not the last"
    return None


class SafeSoftmax(_TrainRule):
    """`_safe_softmax(x, -1)` (SDPA's math path) = `where(all(x == -inf), 0, softmax(x))` per row: `SoftmaxRef_v1{N,f32}` rows,
    `F32Eq_v1` against -inf, a `BitAnd_v1` fold, `SelectF32_v1`."""
    info = _info("SafeSoftmax", "aten._safe_softmax over the last axis of f32", "the operand is an f32 value", "no dtype argument")

    def targets(self):
        return {aten._safe_softmax.default}

    def applies(self, node, ctx):
        why = _operand_ok(ctx, node.args[0])
        if why or not isinstance(ctx.val(node.args[0]), Coll):
            return why or "the operand is not a value"
        if len(node.args) > 2 and node.args[2] is not None or node.kwargs.get("dtype") is not None:
            return "numerics_unregistered: a dtype argument"
        return _last_dim(ctx, node, node.args[0], node.args[1]) or True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        S = shape_of(x.type)[0]
        x2, Rn, N = _rows(x, S)
        sm = _batch(ctx, node, ctx.kind("softmax", node, N=N, DT="f32"), x2, axes=(0,))
        eq = _batch(ctx, node, SC.F32Eq, x.reshape(Array(Rn * N, F32)), _const(ctx, node, 0xFF800000), axes=(0, None))
        r = eq.refs
        acc = Coll(Array(Rn, BIT), Strided(r.space, r.idx, r.base, 1, N, 0, Rn))
        for i in range(1, N):
            acc = _batch(ctx, node, SC.BitAnd, acc, Coll(Array(Rn, BIT), Strided(r.space, r.idx, r.base + i, 1, N, 0, Rn)), axes=(0, 0))
        out = _batch(ctx, node, SC.SelectF32, _per_row(acc, Rn, N), _const(ctx, node, 0), sm.reshape(Array(Rn * N, F32)),
                     axes=(0, None, 0))
        ctx.define(node, out.reshape(_tensor_type(S, 32)))
        ctx.record(self, node, {"rows": Rn, "N": N}, emitted=f"SoftmaxRef_v1{{N={N},DT=f32}} x{Rn} + mask")


class SoftmaxBackward(_TrainRule):
    """`_softmax_backward_data(g, y, -1, f32)` = `y * (g - sum(g * y))` per row, torch CPU's form: `DotRef_v1{N}(g, y)`, `F32Sub_v1`,
    `F32Mul_v1`."""
    info = _info("SoftmaxBackward", "aten._softmax_backward_data over the last axis, f32", "g and y are f32 values",
                 "input_dtype is float32")

    def targets(self):
        return {aten._softmax_backward_data.default}

    def applies(self, node, ctx):
        for a in node.args[:2]:
            why = _operand_ok(ctx, a)
            if why or not isinstance(ctx.val(a), Coll):
                return why or "an operand is not a value"
        if node.args[3] != torch.float32:
            return f"numerics_unregistered: input_dtype {node.args[3]}"
        return _last_dim(ctx, node, node.args[0], node.args[2]) or True

    def emit(self, node, ctx):
        g, y = ctx.coll(node.args[0]), ctx.coll(node.args[1])
        S = shape_of(g.type)[0]
        g2, Rn, N = _rows(g, S)
        y2 = _rows(y, S)[0]
        dot = _batch(ctx, node, bind(R.DotRef, K=N), g2, y2, axes=(0, 0))
        diff = _batch(ctx, node, R.F32Sub, g.reshape(Array(Rn * N, F32)), _per_row(dot, Rn, N), axes=(0, 0))
        out = _batch(ctx, node, P.F32Mul, y.reshape(Array(Rn * N, F32)), diff, axes=(0, 0))
        ctx.define(node, out.reshape(_tensor_type(S, 32)))
        ctx.record(self, node, {"rows": Rn, "N": N}, emitted=f"DotRef_v1{{K={N}}} x{Rn} + F32Sub/F32Mul")


class LogSoftmax(_TrainRule):
    """`_log_softmax(x, -1, False)` of f32 = `(x - max) - log(sum(exp(x - max)))` per row: `LogSoftmaxRef_v1{N,f32}` rows."""
    info = _info("LogSoftmax", "aten._log_softmax over the last axis of f32", "the operand is an f32 value", "half_to_float is false")

    def targets(self):
        return {aten._log_softmax.default}

    def applies(self, node, ctx):
        why = _operand_ok(ctx, node.args[0])
        if why or not isinstance(ctx.val(node.args[0]), Coll):
            return why or "the operand is not a value"
        if node.args[2]:
            return "numerics_unregistered: half_to_float"
        return _last_dim(ctx, node, node.args[0], node.args[1]) or True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        S = shape_of(x.type)[0]
        x2, Rn, N = _rows(x, S)
        out = _batch(ctx, node, bind(R.LogSoftmaxRef, N=N, DT="f32"), x2, axes=(0,))
        ctx.define(node, out.reshape(_tensor_type(S, 32)))
        ctx.record(self, node, {"rows": Rn, "N": N}, emitted=f"LogSoftmaxRef_v1{{N={N},DT=f32}} x{Rn}")


class LogSoftmaxBackward(_TrainRule):
    """`_log_softmax_backward_data(g, y, -1, f32)` = `g - exp(y) * sum(g)` per row, torch CPU's form: `DotRef_v1{N}(g, ones)`,
    `F32ExpRn_v1`, `F32Mul_v1`, `F32Sub_v1`."""
    info = _info("LogSoftmaxBackward", "aten._log_softmax_backward_data over the last axis, f32", "g and y are f32 values",
                 "input_dtype is float32")

    def targets(self):
        return {aten._log_softmax_backward_data.default}

    def applies(self, node, ctx):
        return SoftmaxBackward.applies(self, node, ctx)

    def emit(self, node, ctx):
        g, y = ctx.coll(node.args[0]), ctx.coll(node.args[1])
        S = shape_of(g.type)[0]
        g2, Rn, N = _rows(g, S)
        s = _batch(ctx, node, bind(R.DotRef, K=N), g2, _ones(ctx, node, N), axes=(0, None))
        e = _batch(ctx, node, R.F32ExpRn, y.reshape(Array(Rn * N, F32)), axes=(0,))
        p = _batch(ctx, node, P.F32Mul, e, _per_row(s, Rn, N), axes=(0, 0))
        out = _batch(ctx, node, R.F32Sub, g.reshape(Array(Rn * N, F32)), p, axes=(0, 0))
        ctx.define(node, out.reshape(_tensor_type(S, 32)))
        ctx.record(self, node, {"rows": Rn, "N": N}, emitted=f"DotRef_v1{{K={N}}} x{Rn} + F32ExpRn/F32Mul/F32Sub")


__all__ = ["SafeSoftmax", "SoftmaxBackward", "LogSoftmax", "LogSoftmaxBackward"]
