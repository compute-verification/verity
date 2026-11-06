"""Training rules: the ops a training step adds to a forward (its backward, its loss and its optimizer), lowered in reference mode to
registered Definitions, as a rule set of their own (`training_rules()`): the default rules, with the training rules before and after
them. The default rules are unchanged; a node either kind of rule lowers is lowered as before unless a training rule claims it first,
and the training rules placed first claim only what the defaults lower per row or cannot represent at a step's sizes.

Arithmetic is the reference vocabulary's (`ref_prims`): each element-wise op is the f32 operation torch's CPU kernel performs for it
(true division, IEEE `sqrt`, `pow` 2 and 3 as products, `rsqrt` as `1 / sqrt`, `lerp` as `fma(w, end - start, start)` or
`fma(w - 1, end - start, end)`, `addcmul` as `fma(value * t1, t2, self)`), and each reduction is the vocabulary's sequential one (`DotRef`, ascending from +0, where torch's kernels
sum in blocks). Matrix products and the element-wise ops of `Binary` and `Unary` also take bf16 values: a product is
`LinearRef_v1{DT=bf16}` (operands widened, the f32 dot, its result rounded), and an element-wise op widens its bf16 operands to f32
(`Bf16ToF32_v1`, exact), performs the f32 operation and rounds a bf16 result once (`F32ToBf16Rn_v1`), as torch CPU's bf16 kernels
do; where torch's bf16 kernel rounds otherwise, the rule refuses. At a compute capability `cc`, a bf16 `mm` or `mv` is instead
F.linear's GEMM as that GPU's cuBLAS computes it (`Matmul`): the route its capture fitted (`cublas_bf16.train_route`), each k16 step
the tensor core's dot (`HopperBF16WgmmaDot16_v1`); `add`, `sub`, `mul` and `div` are the GPU's (`Binary`); a vector norm or a sum of a
whole tensor is its `reduce_kernel`'s (`rules.training_reduce.FullReduce`); and the conversions to bf16 and to and from f16 and a
`kthvalue` are its own (`rules.training_masks`). A random op's values are an input of the step, committed with its
generator's seed (`rules.training_draws.Random`). A training op with no Definition to lower it to is refused with the Definition it needs
(`_log_softmax`: a correctly-rounded f32 log, `F32LogRn`).

    training_rules(min_view_leaves, cc, tf32) -> RuleSet     the training rule set

Views and static float tensors are `rules.training_views`'s, matrix products `rules.training_matmul`'s, comparisons, masks,
integer sums, conversions, table lookups and `kthvalue` `rules.training_masks`'s, draws `rules.training_draws`'s and the helpers
the rules emit with `rules.training_emit`'s; a survey of what the rules refuse is `rules.training_survey`'s.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.refs import Affine, Coll, Strided, broadcast
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import ref_prims as R
from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.training_emit import (_DT, BIT, F32, _batch, _bits, _call, _const, _ew, _flat, _iota,
                                                              _operand, _operand_ok, _ones, _per_row, _ref, _rounded, _rows, _shape,
                                                              _tokens, _TrainRule, _widened, aten)
from verity_vllm.program.frontend.rules.training_draws import Random
from verity_vllm.program.frontend.rules.training_masks import Cast, Clamp, Compare, Index, IntSum, KthValue, Logic, Where
from verity_vllm.program.frontend.rules.training_matmul import Matmul
from verity_vllm.program.frontend.rules.training_reduce import FullReduce
from verity_vllm.program.frontend.rules.training_views import BoxPermute, StaticFloat, StaticFloats, _info, _view_permute
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of


# ---------------------------------------------------------------------------------------------------------
# element-wise
# ---------------------------------------------------------------------------------------------------------

_BINARY = {aten.add.Tensor: "add", aten.add.Scalar: "add", aten.sub.Tensor: "sub", aten.sub.Scalar: "sub", aten.mul.Tensor: "mul",
           aten.mul.Scalar: "mul", aten.div.Tensor: "div", aten.div.Scalar: "div"}
#: each operation's f32 Definition: torch CPU's, and the GPU's (every NaN result `F32_NAN_PTX`)
_F32 = {"add": (P.F32Add, P.F32AddV2), "sub": (R.F32Sub, P.F32SubV2), "mul": (P.F32Mul, P.F32MulV2), "div": (P.F32Div, P.F32DivV2),
        "fma": (P.F32Fma, P.F32FmaV2)}


def _alpha(node):
    """`add` / `sub`'s alpha: keyword-only in `.Tensor`, positional in `.Scalar`."""
    return node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)


class Binary(_TrainRule):
    """`add` / `sub` / `mul` / `div` (`.Tensor`, `.Scalar`) of f32, bf16, bool or int64 values, static float tensors and Python
    numbers, broadcast, element by element as torch's CPU kernels compute them: `F32Add_v1`, `F32Sub_v1`, `F32Mul_v1`, `F32Div_v1`
    (true division), and `add` (`sub`) with alpha != 1 one `F32Fma_v1(b, f32(alpha), a)` (`-alpha`). A bf16 operand is widened,
    a bool one is 1.0 or +0 (`SelectF32_v1`), an int64 one of an f32 result rounded to nearest (`I64ToF32Rn_v1`) and a bf16
    result rounded once (`F32ToBf16Rn_v1`). A number is its f32 word, but in a bf16 `add` or `sub` torch CPU first rounds it (and
    a 0-d static) to bf16. At a compute capability `cc` they are the GPU's: the same operations with its NaN (`F32Add_v2`,
    `F32Sub_v2`, `F32Mul_v2`, `F32Div_v2`, `F32Fma_v2`), a bf16 result rounded by `F2fpBf16_v1`, every number its f32 word, and a
    division by a number the product by its reciprocal, `RN(1 / f32(s))`, which torch's CUDA kernel computes on the host.
    Refused: a bf16 add with alpha != 1, an int64 operand of a bf16 result (torch rounds it to bf16 first), a rounding mode, and
    at `cc` a division by a 0-d static tensor (the GPU multiplies by a CPU scalar's reciprocal and divides by a device tensor, and
    the graph doesn't say which)."""
    info = _info("Binary", "aten add/sub/mul/div .Tensor/.Scalar with broadcasting, f32 or bf16", "operands are f32, bf16, bool or "
                 "int64 values, static float tensors or Python numbers", "alpha != 1 only for an f32 add or sub",
                 "int64 operands only of an f32 result", "no rounding_mode", "at cc: no division by a 0-d static tensor",
                 "one batch over the output's elements")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return set(_BINARY)

    def applies(self, node, ctx):
        if ctx.dtype_of(node) not in _DT:
            return f"numerics_unregistered: result dtype {ctx.dtype_of(node)}"
        if node.kwargs.get("rounding_mode") is not None:
            return "numerics_unregistered: a rounding mode"
        if _alpha(node) != 1 and ctx.dtype_of(node) != torch.float32:
            return "numerics_unregistered: a bf16 add with alpha != 1"
        whys = [w for w in (_operand_ok(ctx, a, (*_DT, torch.bool, torch.int64)) for a in node.args[:2]) if w]
        if whys:
            return whys[0]
        if all(not (isinstance(a, torch.fx.Node) and isinstance(ctx.val(a), Coll)) for a in node.args[:2]):
            return "no operand is a value (a static result: StaticFloats)"
        if ctx.dtype_of(node) != torch.float32 and any(isinstance(a, torch.fx.Node) and ctx.dtype_of(a) == torch.int64
                                                       for a in node.args[:2]):
            return "numerics_unregistered: an int64 operand of a bf16 result (torch rounds it to bf16 first)"
        b = node.args[1]
        if (self.cc and _BINARY[node.target] == "div" and isinstance(b, torch.fx.Node) and isinstance(ctx.val(b), StaticFloat)
                and _shape(ctx, b) == ()):
            return ("numerics_unregistered: a division by a 0-d static tensor on the GPU (a CPU scalar's reciprocal or a device "
                    "tensor's quotient)")
        return True

    def _static(self, ctx, node, a):
        """A value operand as it is; a number or static one as the f32 words this op reads (`StaticFloat`): in a bf16 add or sub
        on the CPU, a number or 0-d static rounded to bf16."""
        if isinstance(a, torch.fx.Node) and isinstance(ctx.val(a), Coll):
            return a
        t = torch.as_tensor(a if isinstance(a, (int, float)) else ctx.val(a).tensor, dtype=torch.float32)
        if self.cc is None and ctx.dtype_of(node) == torch.bfloat16 and _BINARY[node.target] in ("add", "sub") and t.numel() == 1:
            t = t.to(torch.bfloat16).to(torch.float32)
        return StaticFloat(t.contiguous())

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        op, gpu = _BINARY[node.target], self.cc is not None
        a, b = node.args[:2]
        if gpu and op == "div" and isinstance(b, (int, float)):
            with np.errstate(all="ignore"):
                b, op = float(np.float32(1.0) / np.float32(b)), "mul"
        a, b = (self._static(ctx, node, x) for x in (a, b))
        alpha = _alpha(node)
        if alpha != 1:
            sign = -1 if op == "sub" else 1
            ops = [_operand(ctx, node, b, S), (_const(ctx, node, _bits(sign * alpha)), None), _operand(ctx, node, a, S)]
            fn = _F32["fma"][gpu]
        else:
            ops, fn = [_operand(ctx, node, x, S) for x in (a, b)], _F32[op][gpu]
        out = _ew(ctx, node, fn, ops, S)
        ctx.define(node, _rounded(ctx, node, out, S, P.F2fpBf16 if gpu else P.F32ToBf16Rn))
        ctx.record(self, node, {"shape": list(S), "dtype": _DT[ctx.dtype_of(node)], "alpha": alpha},
                   emitted=f"{fn.id} x{math.prod(S)}")


class Unary(_TrainRule):
    """`neg`, `abs`, `sqrt`, `rsqrt`, `reciprocal`, `exp`, `sigmoid`, `pow.Tensor_Scalar` of an f32 or bf16 value, as torch's CPU
    kernels compute them: `abs` clears the sign bit (`F32Fabs_v1`, f32 only), `rsqrt` = 1 / sqrt, `sigmoid` = 1 / (1 + exp(-x)),
    `pow` 2 / 3 products, 0.5 sqrt, -0.5 rsqrt, -1 and -2 reciprocals, 1 the operand, 0 one; `exp` is `F32ExpRn_v1` (correctly
    rounded, where torch's vectorized exp is not). `sqrt` is IEEE's (`F32Sqrt_v1`, as CUDA's `sqrt.rn`); torch CPU's (and its `pow`
    0.5) is MKL's, which is not IEEE's on some CPUs. A bf16 operand is widened and the result rounded to bf16, except where torch's
    bf16 kernel rounds in between, which is refused: `pow` 3 and -2 round the square, and `rsqrt` (and `pow` -0.5) the square root
    in its scalar loop; a bf16 `abs` is refused too (torch's CPU kernel keeps a NaN's payload, its CUDA kernel rounds through f32)."""
    info = _info("Unary", "aten neg/abs/sqrt/rsqrt/reciprocal/exp/sigmoid/pow.Tensor_Scalar of f32 or bf16", "the operand is an f32 "
                 "or bf16 value", "pow's exponent is one torch special-cases (0, 0.5, 1, 2, 3, -0.5, -1, -2)",
                 "bf16: not abs or rsqrt, nor pow 3, -2 or -0.5", "one batch per step over the elements")

    def targets(self):
        return {aten.neg.default, aten.abs.default, aten.sqrt.default, aten.rsqrt.default, aten.reciprocal.default,
                aten.exp.default, aten.sigmoid.default, aten.pow.Tensor_Scalar}

    def applies(self, node, ctx):
        why = _operand_ok(ctx, node.args[0], (torch.float32,) if node.target is aten.abs.default else tuple(_DT))
        if why or not isinstance(ctx.val(node.args[0]), Coll):
            return why or "the operand is not a value"
        e = node.args[1] if node.target is aten.pow.Tensor_Scalar else None
        if node.target is aten.pow.Tensor_Scalar and e not in (0, 0.5, 1, 2, 3, -0.5, -1, -2):
            return f"numerics_unregistered: pow with exponent {e} needs an f32 pow Definition"
        if ctx.dtype_of(node) == torch.bfloat16 and (e in (3, -2, -0.5) or node.target is aten.rsqrt.default):
            return ("numerics_unregistered: torch's bf16 kernel rounds in between (pow 3 and -2 the square, rsqrt's scalar loop "
                    "the square root)")
        return True

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        one = (_const(ctx, node, _bits(1.0)), None)

        def ew(fn, *ops):
            return _flat(_ew(ctx, node, fn, list(ops), S), S)

        t = node.target
        e = node.args[1] if t is aten.pow.Tensor_Scalar else None
        if t is aten.pow.Tensor_Scalar and e == 1:
            ctx.define(node, ctx.coll(node.args[0]))
            ctx.record(self, node, {"exponent": e}, emitted="identity (torch copies the operand)")
            return
        x = ctx.coll(node.args[0])
        x = _flat(_widened(ctx, node, x) if ctx.dtype_of(node.args[0]) == torch.bfloat16 else x, S)
        if t is aten.neg.default:
            y = ew(SC.F32Neg, x)
        elif t is aten.abs.default:
            y = ew(SC.F32Fabs, x)
        elif t is aten.sqrt.default or e == 0.5:
            y = ew(R.F32Sqrt, x)
        elif t is aten.rsqrt.default or e == -0.5:
            y = ew(P.F32Div, one, ew(R.F32Sqrt, x))
        elif t is aten.reciprocal.default or e == -1:
            y = ew(P.F32Div, one, x)
        elif t is aten.exp.default:
            y = ew(R.F32ExpRn, x)
        elif t is aten.sigmoid.default:
            y = ew(P.F32Div, one, ew(P.F32Add, ew(R.F32ExpRn, ew(SC.F32Neg, x)), one))
        elif e == 2:
            y = ew(P.F32Mul, x, x)
        elif e == 3:
            y = ew(P.F32Mul, ew(P.F32Mul, x, x), x)
        elif e == -2:
            y = ew(P.F32Div, one, ew(P.F32Mul, x, x))
        else:
            y = ew(P.F32Add, one, (_const(ctx, node, 0), None))
        ctx.define(node, _rounded(ctx, node, y[0].reshape(_tensor_type(S, 32)), S))
        ctx.record(self, node, {"op": str(t), "exponent": e, "dtype": _DT[ctx.dtype_of(node)]}, emitted=f"x{math.prod(S)}")


class Lerp(_TrainRule):
    """`lerp.Scalar(start, end, w)`, torch CPU's vectorized form: `fma(c, end - start, base)` with `c, base = (w, start)` if
    |w| < 0.5 else `(w - 1, end)` (`F32Sub_v1`, `F32Fma_v1`)."""
    info = _info("Lerp", "aten.lerp.Scalar of f32 values", "start and end are f32 values or static", "the weight is a Python number")

    def targets(self):
        return {aten.lerp.Scalar}

    def applies(self, node, ctx):
        whys = [w for w in (_operand_ok(ctx, a) for a in node.args[:2]) if w]
        if whys:
            return whys[0]
        if not isinstance(node.args[2], (int, float)) or isinstance(node.args[2], bool):
            return "the weight is not a Python number"
        return True

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        w = float(np.float32(node.args[2]))
        start, end = (_operand(ctx, node, a, S) for a in node.args[:2])
        diff = _flat(_ew(ctx, node, R.F32Sub, [end, start], S), S)
        small = abs(w) < 0.5
        c = _const(ctx, node, _bits(w if small else float(np.float32(w) - np.float32(1.0))))
        out = _ew(ctx, node, P.F32Fma, [(c, None), diff, start if small else end], S)
        ctx.define(node, out)
        ctx.record(self, node, {"weight": w, "small": small}, emitted=f"F32Sub_v1+F32Fma_v1 x{math.prod(S)}")


class Addcmul(_TrainRule):
    """`addcmul(self, t1, t2, value)` = `fma(value * t1, t2, self)`, as torch CPU's kernel computes it (`F32Mul_v1`, `F32Fma_v1`)."""
    info = _info("Addcmul", "aten.addcmul of f32 values", "operands are f32 values or static", "value is a Python number")

    def targets(self):
        return {aten.addcmul.default}

    def applies(self, node, ctx):
        whys = [w for w in (_operand_ok(ctx, a) for a in node.args[:3]) if w]
        return whys[0] if whys else True

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        s, t1, t2 = (_operand(ctx, node, a, S) for a in node.args[:3])
        v = (_const(ctx, node, _bits(node.kwargs.get("value", node.args[3] if len(node.args) > 3 else 1))), None)
        vt1 = _flat(_ew(ctx, node, P.F32Mul, [v, t1], S), S)
        ctx.define(node, _ew(ctx, node, P.F32Fma, [vt1, t2, s], S))
        ctx.record(self, node, {"shape": list(S)}, emitted=f"F32Mul_v1 + F32Fma_v1 x{math.prod(S)}")


# ---------------------------------------------------------------------------------------------------------
# reductions, softmax and the loss
# ---------------------------------------------------------------------------------------------------------


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
    """`_log_softmax`: refused. Its value is `x - max - log(sum(exp(x - max)))`, and the reference vocabulary has no f32 log: it needs
    a correctly-rounded `F32LogRn` (the twin of `F32ExpRn_v1`), registered with its Boolean version."""
    info = _info("LogSoftmax", "aten._log_softmax (refused: no f32 log Definition)", "never applies")

    def targets(self):
        return {aten._log_softmax.default}

    def applies(self, node, ctx):
        return ("numerics_unregistered: log_softmax needs a correctly-rounded f32 log (F32LogRn, the twin of F32ExpRn_v1) and the "
                "reference vocabulary has none")

    def emit(self, node, ctx):
        raise AssertionError("never applies")


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
    if reduction != 1:
        return f"numerics_unregistered: reduction {reduction} (only mean)"
    return None


class NllLossForward(_TrainRule):
    """`nll_loss_forward(logp[M,V], target[M], None, mean, ignore_index)` -> (loss, total_weight): the target's entry of each row
    (`GatherF32x{V}_v1`), rows whose target is `ignore_index` zeroed (`I32Eq_v1`, `SelectF32_v1`), the loss
    `-sum / count` and the count, sums sequential (`DotRef_v1{M}`; torch sums in a cascade)."""
    info = _info("NllLossForward", "aten.nll_loss_forward, mean reduction, no weights, [M, V] input", "logp is an f32 value, the target "
                 "an integer value", "an out-of-range target reads NaN (GatherF32x's extension; torch raises)")

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
    """`nll_loss_backward(go, logp[M,V], target, None, mean, ignore_index, total_weight)` = `-(go / total_weight)` at each row's target,
    +0 elsewhere and on the rows whose target is `ignore_index` (`F32Div_v1`, `F32Neg_v1`; per row the gradient or +0 by
    `I32Eq_v1` against `ignore_index`, as `NllLossForward` masks its rows; a one-hot of `I32Eq_v1`; `SelectF32_v1`)."""
    info = _info("NllLossBackward", "aten.nll_loss_backward, mean reduction, no weights", "go and total_weight are 0-d f32",
                 "an out-of-range target selects nothing (torch raises)")

    def targets(self):
        return {aten.nll_loss_backward.default}

    def applies(self, node, ctx):
        go, logp, target, weight, reduction, _ign, tw = node.args[:7]
        why = _nll_ok(ctx, node, logp, target, weight, reduction) or _operand_ok(ctx, go) or _operand_ok(ctx, tw)
        return why or True

    def emit(self, node, ctx):
        go, logp, target, _w, _r, ignore, tw = node.args[:7]
        M, V = _shape(ctx, logp)
        g = _call(ctx, node, SC.F32Neg, _call(ctx, node, P.F32Div, _operand(ctx, node, go, ())[0], _operand(ctx, node, tw, ())[0]))
        t = _tokens(ctx, node, target)
        zero = _const(ctx, node, 0)
        ign = _batch(ctx, node, SC.I32Eq, t, _const(ctx, node, ignore & 0xFFFFFFFF), axes=(0, None))
        row = _batch(ctx, node, SC.SelectF32, ign, zero, g, axes=(0, None, None))
        hot = _one_hot(ctx, node, t, M, V)
        out = _batch(ctx, node, SC.SelectF32, hot, _per_row(row, M, V), zero, axes=(0, 0, None))
        ctx.define(node, out.reshape(_tensor_type([M, V], 32)))
        ctx.record(self, node, {"M": M, "V": V, "ignore_index": ignore}, emitted=f"I32Eq_v1 + SelectF32_v1 x{M * (V + 1)}")


class EmbeddingDenseBackward(_TrainRule):
    """`embedding_dense_backward(g[..., H], ids, V, -1, False)`: row v of the weight's gradient is the sum, in token order from +0, of
    the gradient rows of the tokens with id v (torch CPU's per-row `add_`): per token one `SelectF32_v1` of its gradient row into
    the rows whose id matches (a one-hot of `I32Eq_v1`) and one `F32Add_v1` into the accumulator, V x H words each."""
    info = _info("EmbeddingDenseBackward", "aten.embedding_dense_backward, padding_idx -1, no scale_grad_by_freq",
                 "g is an f32 value, ids an integer value", "an out-of-range id adds nothing")

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
        return True

    def emit(self, node, ctx):
        g, ids, V = node.args[:3]
        gc = ctx.coll(g)
        H = _shape(ctx, g)[-1]
        t = _tokens(ctx, node, ids)
        M = t.type.leaves
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


def training_rules(min_view_leaves: int = 0, cc: tuple[int, int] | None = None, tf32: bool = False) -> RuleSet:
    """The default rules with the training rules: views of static floats, `BoxPermute` (views of more than `min_view_leaves` leaves
    that are not one run) and `Matmul` before them, the element-wise, reduction, softmax, loss, embedding, comparison, mask,
    integer, conversion, lookup, order-statistic and draw rules after. `min_view_leaves` is 0 by default: the default views write a
    permuted view of fewer leaves than their limit as a list of its leaves, which the root's replay then substitutes as one
    reference part per leaf.  `cc` (`cublas_bf16.TRAIN_CC`): bf16 `mm`s and `mv`s as that GPU's cuBLAS computes them (`Matmul`),
    every add, sub, mul and div as its kernels do (`Binary`) and the conversions to bf16 and to and from f16 as its kernels do
    (`Cast`), both placed first, before the default rules' CPU kinds, a whole tensor's vector norm or sum as its `reduce_kernel`
    does (`FullReduce`, before `Reduce`) and `kthvalue` as its radix select does (`KthValue`), in the rule set
    ``training-cc<major>.<minor>``.  `tf32`, at a `cc`: the step sets cuda.matmul.allow_tf32, which the traced graph doesn't
    show, so its f32 `mm`s and `bmm`s are cuBLAS's TF32 GEMMs, which `Matmul` refuses (``training-cc<major>.<minor>-tf32``)."""
    from verity_vllm.program.frontend.rules import DEFAULT_RULES

    first = [StaticFloats(), BoxPermute(min_view_leaves), Matmul(cc, tf32)] + ([Binary(cc), Cast(cc)] if cc else [])
    last = ([] if cc else [Binary()]) + [Unary(), Lerp(), Addcmul(), FullReduce(cc), Reduce(), IntSum(), SafeSoftmax(),
                                         SoftmaxBackward(), LogSoftmax(), LogSoftmaxBackward(), NllLossForward(),
                                         NllLossBackward(), EmbeddingDenseBackward(), Compare(), Logic(), Where(), Clamp(cc),
                                         Index(), KthValue(cc), Random()] + ([] if cc else [Cast()])
    name = "training" if cc is None else f"training-cc{cc[0]}.{cc[1]}" + ("-tf32" if tf32 else "")
    return RuleSet(name, first + list(DEFAULT_RULES.rules) + last, list(DEFAULT_RULES.families))


__all__ = ["training_rules", "Matmul", "Binary", "Unary", "Lerp", "Addcmul", "FullReduce", "Reduce", "IntSum", "SafeSoftmax",
           "SoftmaxBackward", "LogSoftmax", "LogSoftmaxBackward", "NllLossForward", "NllLossBackward", "EmbeddingDenseBackward",
           "Compare", "Logic", "Where", "Clamp", "Cast", "Index", "KthValue", "Random"]
