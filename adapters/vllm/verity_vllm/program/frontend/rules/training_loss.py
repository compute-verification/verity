"""Training rules for a step's reductions, its loss and its embedding's gradient: `sum` and `mean` over axes (`Reduce`), the
negative log-likelihood loss and its backward (`NllLossForward`, `NllLossBackward`) and an f32 embedding's gradient, on the CPU and
at a compute capability (`EmbeddingDenseBackward`). `rules.training` places them in the training rule set."""

from __future__ import annotations

import math

import torch
from verity.circuits.defs import bind
from verity.circuits.refs import Affine, Coll, Strided, broadcast
from verity.circuits.types import Array, Value
from verity.circuits.ops import scalar as SC
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry import ref_prims as R
from verity_vllm.definitions.registry import torch_embedding_f32 as TEF

from verity_vllm.program.frontend.rules.training_emit import (
    BIT,
    F32,
    _batch,
    _bits,
    _call,
    _const,
    _iota,
    _ones,
    _operand,
    _operand_ok,
    _per_row,
    _ref,
    _shape,
    _tokens,
    _TrainRule,
    aten,
)
from verity_vllm.program.frontend.rules.training_views import (
    StaticFloat,
    _info,
    _view_permute,
)
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of


class Reduce(_TrainRule):
    """`sum` / `mean` of an f32 value over `dim` (every axis when `dim` is None or `[]`, as torch reduces them, and for `.default`)
    -> the reduced axes moved last by reference, one batch of `DotRef_v1{N}` against ones (a sequential sum from +0) per kept
    element; `mean` divides the sum by f32(N) (`F32Div_v1`), as torch does."""
    info = _info("Reduce", "aten.sum.dim_IntList / aten.mean.dim / aten.sum.default / aten.mean.default of f32",
                 "the operand is an f32 value of at least one axis", "no dtype argument")

    def targets(self):
        return {aten.sum.dim_IntList, aten.mean.dim, aten.sum.default, aten.mean.default}

    def applies(self, node, ctx):
        why = _operand_ok(ctx, node.args[0])
        if why or not isinstance(ctx.val(node.args[0]), Coll):
            return why or "the operand is not a value"
        if not _shape(ctx, node.args[0]):
            return "a 0-d operand (no axis to reduce)"
        if node.kwargs.get("dtype") is not None or (len(node.args) > 3 and node.args[3] is not None):
            return "numerics_unregistered: a dtype argument"
        return True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        S = shape_of(x.type)[0]
        r = len(S)
        given = node.args[1] if len(node.args) > 1 else None
        dims = sorted({d % r for d in given}) if given else list(range(r))
        keep = [a for a in range(r) if a not in dims]
        perm = tuple(keep + dims)
        xp = _view_permute(x, perm, node.name) if perm != tuple(range(r)) else x
        Rn, N = math.prod(S[a] for a in keep), math.prod(S[a] for a in dims)
        s = _batch(ctx, node, bind(R.DotRef, K=N), xp.reshape(Array(Rn, Array(N, F32))), _ones(ctx, node, N), axes=(0, None))
        if node.target in (aten.mean.dim, aten.mean.default):
            s = _batch(ctx, node, P.F32Div, s, _const(ctx, node, _bits(float(N))), axes=(0, None))
        ctx.define(node, s.reshape(_tensor_type(ctx.shape_of_node(node), 32)))
        ctx.record(self, node, {"dims": dims, "rows": Rn, "N": N}, emitted=f"DotRef_v1{{K={N}}} x{Rn}")


def _nll_ok(ctx, node, logp, target, weight, reduction) -> str | None:
    why = _operand_ok(ctx, logp)
    if why or not isinstance(ctx.val(logp), Coll):
        return why or "the input is not a value"
    if len(_shape(ctx, logp)) != 2:
        return f"input of shape {list(_shape(ctx, logp))} (only [M, V])"
    if not isinstance(target, torch.fx.Node) or not isinstance(ctx.val(target), Coll) or ctx.dtype_of(target) not in (torch.int32, torch.int64):
        return "the target is not an int32/int64 value"
    if weight is not None:
        return "numerics_unregistered: class weights"
    if reduction not in (0, 1):
        return f"numerics_unregistered: reduction {reduction} (only none and mean)"
    return None


class NllLossForward(_TrainRule):
    """`nll_loss_forward(logp[M,V], target[M], None, reduction, ignore_index)` -> (loss, total_weight): the target's entry of each
    row (`GatherF32x{V}_v1`), rows whose target is `ignore_index` zeroed (`I32Eq_v1`, `SelectF32_v1`). With the mean reduction the
    loss is `-sum / count` and total_weight the count, sums sequential (`DotRef_v1{M}`; torch sums in a cascade); with none the loss
    is each row's `-entry` (`F32Neg_v1`: torch's kernels' `-1 * entry`, a sign flip) and total_weight +0, as both kernels set it."""
    info = _info("NllLossForward", "aten.nll_loss_forward, mean or no reduction, no weights, [M, V] input", "logp is an f32 value, "
                 "the target an integer value", "an out-of-range target reads NaN (GatherF32x's extension; torch raises)")

    def targets(self):
        return {aten.nll_loss_forward.default}

    def applies(self, node, ctx):
        return _nll_ok(ctx, node, *node.args[:4]) or True

    def emit(self, node, ctx):
        logp = ctx.coll(node.args[0])
        M, V = _shape(ctx, node.args[0])
        t = _tokens(ctx, node, node.args[1])
        ign = _batch(ctx, node, SC.I32Eq, t, _const(ctx, node, node.args[4] & 0xFFFFFFFF), axes=(0, None))
        picked = _batch(ctx, node, R.gather_f32(V), t, logp.reshape(Array(M, Array(V, F32))), axes=(0, 0))
        zero = _const(ctx, node, 0)
        if node.args[3] == 0:
            neg = _batch(ctx, node, SC.F32Neg, picked, axes=(0,))
            rows = _batch(ctx, node, SC.SelectF32, ign, zero, neg, axes=(0, None, 0))
            ctx.define(node, [rows.reshape(_tensor_type([M], 32)), StaticFloat(torch.zeros(()))])
            ctx.record(self, node, {"M": M, "V": V, "ignore_index": node.args[4], "reduction": "none"},
                       emitted=f"GatherF32x{V}_v1 x{M} + F32Neg_v1/SelectF32_v1 x{M}")
            return
        sel = _batch(ctx, node, SC.SelectF32, ign, zero, picked, axes=(0, None, 0))
        neg = _call(ctx, node, bind(R.DotRef, K=M), sel, Coll(Array(M, F32), broadcast(*_ref(_const(ctx, node, _bits(-1.0))), M)))
        w = _batch(ctx, node, SC.SelectF32, ign, zero, _const(ctx, node, _bits(1.0)), axes=(0, None, None))
        cnt = _call(ctx, node, bind(R.DotRef, K=M), w, _ones(ctx, node, M))
        loss = _call(ctx, node, P.F32Div, neg, cnt)
        ctx.define(node, [loss, cnt])
        ctx.record(self, node, {"M": M, "V": V, "ignore_index": node.args[4]}, emitted=f"GatherF32x{V}_v1 x{M} + DotRef_v1{{K={M}}} x2")


def _one_hot(ctx, node, t: Coll, M: int, V: int) -> Coll:
    """`t[m] == v` for every (m, v), row-major: one `I32Eq_v1` batch."""
    io = _iota(ctx, node, V)
    tr, ir = t.refs, io.refs
    if not isinstance(tr, Affine):
        t = _batch(ctx, node, SC.I32Add, t, _const(ctx, node, 0), axes=(0, None))
        tr = t.refs
    tb = Coll(Array(M * V, Value(32)), Strided(tr.space, tr.idx, tr.base, V, 1, 0, M * V))
    ib = Coll(Array(M * V, Value(32)), Strided(ir.space, ir.idx, ir.base, V, 0, 1, M * V))
    return _batch(ctx, node, SC.I32Eq, tb, ib, axes=(0, 0))


class NllLossBackward(_TrainRule):
    """`nll_loss_backward(go, logp[M,V], target, None, reduction, ignore_index, total_weight)`: with the mean reduction
    `-(go / total_weight)` (`F32Div_v1`, `F32Neg_v1`), with none each row's `-go[m]` (`F32Neg_v1`, torch's kernels' `-1 * go[m]`),
    at each row's target, +0 elsewhere and on the rows whose target is `ignore_index` (per row the gradient or +0 by `I32Eq_v1`
    against `ignore_index`, as `NllLossForward` masks its rows; a one-hot of `I32Eq_v1`; `SelectF32_v1`)."""
    info = _info("NllLossBackward", "aten.nll_loss_backward, mean or no reduction, no weights", "mean: go and total_weight are 0-d "
                 "f32", "none: go is f32 of shape [M]", "an out-of-range target selects nothing (torch raises)")

    def targets(self):
        return {aten.nll_loss_backward.default}

    def applies(self, node, ctx):
        go, logp, target, weight, reduction, _ign, tw = node.args[:7]
        why = _nll_ok(ctx, node, logp, target, weight, reduction) or _operand_ok(ctx, go) or _operand_ok(ctx, tw)
        if why:
            return why
        M = _shape(ctx, logp)[0]
        want = (M,) if reduction == 0 else ()
        return True if _shape(ctx, go) == want else f"go of shape {list(_shape(ctx, go))} (reduction {reduction}: {list(want)})"

    def emit(self, node, ctx):
        go, logp, target, _w, reduction, ignore, tw = node.args[:7]
        M, V = _shape(ctx, logp)
        if reduction == 0:
            g, ax = _operand(ctx, node, go, (M,))
            g, gax = (_batch(ctx, node, SC.F32Neg, g, axes=(0,)), 0) if ax is not None else (_call(ctx, node, SC.F32Neg, g), None)
        else:
            g = _call(ctx, node, SC.F32Neg, _call(ctx, node, P.F32Div, _operand(ctx, node, go, ())[0], _operand(ctx, node, tw, ())[0]))
            gax = None
        t = _tokens(ctx, node, target)
        zero = _const(ctx, node, 0)
        ign = _batch(ctx, node, SC.I32Eq, t, _const(ctx, node, ignore & 0xFFFFFFFF), axes=(0, None))
        row = _batch(ctx, node, SC.SelectF32, ign, zero, g, axes=(0, None, gax))
        hot = _one_hot(ctx, node, t, M, V)
        out = _batch(ctx, node, SC.SelectF32, hot, _per_row(row, M, V), zero, axes=(0, 0, None))
        ctx.define(node, out.reshape(_tensor_type([M, V], 32)))
        ctx.record(self, node, {"M": M, "V": V, "ignore_index": ignore, "reduction": "none" if reduction == 0 else "mean"},
                   emitted=f"I32Eq_v1 + SelectF32_v1 x{M * (V + 1)}")


class EmbeddingDenseBackward(_TrainRule):
    """`embedding_dense_backward(g[..., H], ids, V, -1, False)`: row v of the weight's gradient is the sum of the gradient rows of the
    tokens with id v. Without a `cc` it is torch CPU's per-row `add_` in token order from +0: per token one `SelectF32_v1` of its
    gradient row into the rows whose id matches (a one-hot of `I32Eq_v1`) and one `F32Add_v1` into the accumulator, V x H words
    each. At a compute capability `cc` it is the CUDA kernels' (`torch_embedding_f32.EmbeddingGradF32_v1`): the rows sorted by
    id, each id's tokens summed in its kernel's order and each sum routed to its id's row."""
    info = _info("EmbeddingDenseBackward", "aten.embedding_dense_backward, padding_idx -1, no scale_grad_by_freq",
                 "g is an f32 value, ids an integer value", "an out-of-range id adds nothing")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.embedding_dense_backward.default}

    def applies(self, node, ctx):
        g, ids, V, pad, freq = node.args[:5]
        why = _operand_ok(ctx, g)
        if why or not isinstance(ctx.val(g), Coll):
            return why or "the gradient is not a value"
        if not isinstance(ctx.val(ids), Coll) or ctx.dtype_of(ids) not in (torch.int32, torch.int64):
            return "the ids are not an integer value"
        if pad != -1 or freq:
            return "numerics_unregistered: padding_idx or scale_grad_by_freq"
        if self.cc:
            try:
                bind(TEF.EmbeddingGradF32, M=ctx.val(ids).type.leaves, V=V, H=_shape(ctx, g)[-1])
            except ValueError as e:
                return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        g, ids, V = node.args[:3]
        gc = ctx.coll(g)
        H = _shape(ctx, g)[-1]
        t = _tokens(ctx, node, ids)
        M = t.type.leaves
        if self.cc:
            fn = bind(TEF.EmbeddingGradF32, M=M, V=V, H=H)
            ctx.define(node, _call(ctx, node, fn, t, gc.reshape(Array(M, Array(H, F32)))).reshape(_tensor_type([V, H], 32)))
            ctx.record(self, node, {"tokens": M, "V": V, "H": H}, emitted=f"{fn.id} x1")
            return
        hot = _one_hot(ctx, node, t, M, V)
        gr = gc.refs
        if not isinstance(gr, Affine):
            gc = _batch(ctx, node, P.F32Add, gc.reshape(Array(M * H, F32)), _const(ctx, node, 0), axes=(0, None))
            gr = gc.refs
        hr, zero = hot.refs, _const(ctx, node, 0)
        acc = Coll(Array(V * H, F32), broadcast(*_ref(zero), V * H))
        for m in range(M):
            cond = Coll(Array(V * H, BIT), Strided(hr.space, hr.idx, hr.base + m * V, H, 1, 0, V * H))
            row = Coll(Array(V * H, F32), Strided(gr.space, gr.idx, gr.base + m * H, H, 0, 1, V * H))
            sel = _batch(ctx, node, SC.SelectF32, cond, row, zero, axes=(0, 0, None))
            acc = _batch(ctx, node, P.F32Add, acc, sel, axes=(0, 0))
        ctx.define(node, acc.reshape(_tensor_type([V, H], 32)))
        ctx.record(self, node, {"tokens": M, "V": V, "H": H}, emitted=f"(SelectF32_v1 + F32Add_v1) x{M * V * H}")


__all__ = ["Reduce", "NllLossForward", "NllLossBackward", "EmbeddingDenseBackward"]
