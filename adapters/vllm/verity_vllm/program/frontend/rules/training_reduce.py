"""A whole tensor reduced to one element at a compute capability, as torch 2.14's CUDA `reduce_kernel` (Reduce.cuh) computes it:
`linalg_vector_norm(x, 2)` of a bf16 or f32 value (`NormTwoOps`: ``acc = fma(d, d, acc)``, sqrt.rn, a bf16 result rounded by
cvt.rn.bf16.f32) and `sum` of an f32 value (``acc + d``), in the order `numerics.torch_kernels.reduce_all` states at the launch
`reduce_config` reads off the length on the RTX PRO 6000 (`FullReduce`): its scalar path under 128 elements, its vectorized
path from 128, across CTAs past 130,560.

The kernel reads the tensor in memory order, so a transposed matrix is read as the matrix it is a view of. Each step is a word:
`F32Mul_v2` (an accumulator's first square, ``fma(d, d, +0)``), `F32Fma_v2`, `F32Add_v2` (every NaN `F32_NAN_PTX`),
`PouwF32SqrtRn_v1` and `F2fpBf16_v1`. An add of a lane the kernel leaves at +0 is left out: every partial is a round-to-nearest
sum from +0, so it is never -0, and adding +0 to it changes nothing.

    FullReduce(cc)      the rule (`training_rules` places it before `Reduce`)
"""

from __future__ import annotations

import math

import torch

from verity.definitions.ops.sqrt import PouwF32SqrtRn
from verity.circuits.refs import Coll, concat, strided_view
from verity.circuits.types import Array
from verity.definitions.numerics import torch_kernels as TK
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import prims as P
from verity_vllm.program.frontend.rules.training_emit import _DT, BF16, F32, _batch, _call, _const, _operand_ok, _TrainRule, aten
from verity_vllm.program.frontend.rules.training_views import _info, _view_permute
from verity_vllm.program.frontend.strided_views import _tensor_type

ILP = TK.ILP
_NORM = aten.linalg_vector_norm.default
_SUMS = (aten.sum.default, aten.sum.dim_IntList)


def _memory_order(t) -> tuple[int, ...] | None:
    """The axes of a dense tensor (its elements one block of memory) outermost first, so that the tensor permuted to them and
    flattened is in memory order; None if it isn't dense."""
    axes = sorted((d for d in range(t.dim()) if t.shape[d] != 1), key=lambda d: -t.stride()[d])
    step = 1
    for d in reversed(axes):
        if t.stride()[d] != step:
            return None
        step *= t.shape[d]
    return tuple(axes + [d for d in range(t.dim()) if t.shape[d] == 1])


def _cta_halves(ctx, node, s: Coll, ctas: int, bw: int) -> Coll:
    """Each CTA's `block_x_reduce` over its bw threads' words (s, CTA-major), one batch a level: lane t adds lane t + h."""
    w = bw
    while w > 1:
        h = w // 2
        lo, hi = (Coll(Array(ctas * h, F32), strided_view(s, b, h, w, 1, ctas * h)) for b in (0, h))
        s, w = _batch(ctx, node, P.F32AddV2, lo, hi, axes=(0, 0)), h
    return s


def _prefix(new: Coll, old: Coll, n: int) -> Coll:
    """new's words, then old's from len(new) on: n words."""
    return new if len(new) == n else Coll(Array(n, F32), concat([new.refs, old[len(new):n].refs]))


def _halves(ctx, node, v: Coll, n: int, width: int) -> Coll:
    """`block_x_reduce` over `width` lanes, the first n holding v's words and the rest +0 (left out)."""
    h = width // 2
    while h >= 1:
        if n > h:
            v, n = _prefix(_batch(ctx, node, P.F32AddV2, v[:n - h], v[h:n], axes=(0, 0)), v, h), h
        h //= 2
    return v[0]


def reduce_all(ctx, node, x: Coll, square: bool) -> Coll:
    """`torch_kernels.reduce_all` on the words x (f32, memory order, L >= 2 of them): the f32 word before the projection."""
    L = len(x)
    vec, bw, ctas = TK.reduce_config(L)

    def first(xs: Coll) -> Coll:
        return (_batch(ctx, node, P.F32MulV2, xs, xs, axes=(0, 0)) if square else
                _batch(ctx, node, P.F32AddV2, _const(ctx, node, 0), xs, axes=(None, 0)))

    def step(acc: Coll, xs: Coll) -> Coll:
        return (_batch(ctx, node, P.F32FmaV2, xs, xs, acc, axes=(0, 0, 0)) if square else
                _batch(ctx, node, P.F32AddV2, acc, xs, axes=(0, 0)))

    if not vec:
        a0 = first(x[:bw])
        s = a0 if L == bw else _prefix(_batch(ctx, node, P.F32AddV2, a0[:L - bw], first(x[bw:]), axes=(0, 0)), a0, bw)
        return _halves(ctx, node, s, bw, bw)
    n4, T = L // ILP, bw * ctas
    cols: list[Coll] = []
    for k in range(-(-n4 // T)):
        A = min(T, n4 - k * T)
        xs = Coll(Array(ILP * A, F32), strided_view(x, ILP * k * T, A, 1, ILP, ILP * A))
        out = first(xs) if k == 0 else step(Coll(Array(ILP * A, F32), concat([c[:A].refs for c in cols])), xs)
        parts = [out[i * A:(i + 1) * A] for i in range(ILP)]
        cols = parts if k == 0 else [_prefix(p, c, T) for p, c in zip(parts, cols)]
    if r := L % ILP:
        cols[0] = _prefix(step(cols[0][:r], x[ILP * n4:]), cols[0], T)
    s = cols[0]
    for c in cols[1:]:
        s = _batch(ctx, node, P.F32AddV2, s, c, axes=(0, 0))
    p = _cta_halves(ctx, node, s, ctas, bw)
    if ctas == 1:
        return p[0]
    n = min(ctas, bw)
    w = p[:n]
    for j in range(bw, ctas, bw):
        k = min(bw, ctas - j)
        w = _prefix(_batch(ctx, node, P.F32AddV2, w[:k], p[j:j + k], axes=(0, 0)), w, n)
    return _halves(ctx, node, w, n, bw)


class FullReduce(_TrainRule):
    """`linalg_vector_norm(x, 2)` of a bf16 or f32 value and `sum` of an f32 value, over every element, at a compute capability
    `cc` (`cublas_bf16.TRAIN_CC`, the RTX PRO 6000's, whose multiprocessor count sets the CTAs), as torch 2.14's CUDA
    `reduce_kernel` computes them (module docstring). Refused: without `cc` a norm (torch CPU's order isn't modeled; a sum is
    `Reduce`'s), an operand not dense in memory, from 128 elements an offset not a multiple of 4 (the vectorized path's unaligned
    head), one element (a norm of one element is torch's `abs`), another ord or a dtype argument."""
    info = _info("FullReduce", "aten.linalg_vector_norm (ord 2) of a bf16 or f32 value and aten.sum of an f32 value, to one element",
                 "a compute capability", "the operand dense in memory, at an offset a multiple of 4 elements from 128 of them",
                 "at least 2 elements", "no dtype argument",
                 "words: F32Mul_v2, F32Fma_v2, F32Add_v2, PouwF32SqrtRn_v1, F2fpBf16_v1")

    def __init__(self, cc: tuple[int, int] | None = None):
        if cc is not None and tuple(cc) != CB.TRAIN_CC:
            raise ValueError(f"FullReduce: reduce_kernel's launch is modeled at {CB.TRAIN_CC} (the RTX PRO 6000), not {tuple(cc)}")
        self.cc = cc

    def targets(self):
        return {_NORM, *_SUMS}

    def applies(self, node, ctx):
        norm = node.target is _NORM
        if self.cc is None:
            return ("numerics_unregistered: a vector norm is lowered at a compute capability (training_rules(cc=...)), as the "
                    "GPU's reduce_kernel computes it" if norm else "a sum without a compute capability is Reduce's")
        a = node.args[0]
        if why := _operand_ok(ctx, a, tuple(_DT) if norm else (torch.float32,)):
            return why
        if not isinstance(ctx.val(a), Coll):
            return "the operand is not a value"
        if norm and (node.args[1] if len(node.args) > 1 else node.kwargs.get("ord", 2)) != 2:
            return "numerics_unregistered: a vector norm of ord other than 2"
        if node.kwargs.get("dtype") is not None:
            return "numerics_unregistered: a dtype argument"
        if math.prod(ctx.shape_of_node(node)) != 1:
            return "a reduction to more than one element (Reduce lowers a sum in torch CPU's order)"
        t = ctx.tensor_meta(a)
        if t is None or _memory_order(t) is None:
            return "numerics_unregistered: an operand not dense in memory (reduce_kernel reads it by its strides)"
        if t.numel() < 2:
            return "numerics_unregistered: one element (torch's norm of one element is abs, not reduce_kernel)"
        if TK.reduce_config(t.numel())[0] and t.storage_offset() % ILP:
            return f"numerics_unregistered: an operand at element {t.storage_offset()} (reduce_kernel's unaligned head)"
        return True

    def emit(self, node, ctx):
        a = node.args[0]
        t, norm, dt = ctx.tensor_meta(a), node.target is _NORM, ctx.dtype_of(a)
        L, order = t.numel(), _memory_order(t)
        x = ctx.coll(a)
        if order != tuple(range(t.dim())):
            x = _view_permute(x, order, node.name)
        x = x.reshape(Array(L, BF16 if dt == torch.bfloat16 else F32))
        if dt == torch.bfloat16:
            x = _batch(ctx, node, P.Bf16ToF32, x, axes=(0,))
        y = reduce_all(ctx, node, x, norm)
        if norm:
            y = _call(ctx, node, PouwF32SqrtRn, y)
        if ctx.dtype_of(node) == torch.bfloat16:
            y = _call(ctx, node, P.F2fpBf16, y)
        ctx.define(node, Coll(_tensor_type(ctx.shape_of_node(node), y.type.w), y.refs))
        _, bw, ctas = TK.reduce_config(L)
        ctx.record(self, node, {"op": "norm" if norm else "sum", "dtype": _DT[dt], "L": L, "block_width": bw, "ctas": ctas},
                   emitted=f"reduce_kernel of {L} words in {ctas} CTA(s) of {bw} threads")


__all__ = ["FullReduce", "reduce_all"]
