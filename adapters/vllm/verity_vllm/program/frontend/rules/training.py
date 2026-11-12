"""Training rules: the ops a training step adds to a forward (its backward, its loss and its optimizer), lowered in reference mode to
registered Definitions, as a rule set of their own (`training_rules()`): the default rules that target ATen ops (`rules.TORCH_RULES`,
the default without the rules of vLLM's own ops), with the training rules before and after them. Those default rules are unchanged; a
node either kind of rule lowers is lowered as before unless a training rule claims it first, and the training rules placed first
claim only what the defaults lower per row or cannot represent at a step's sizes.

Arithmetic is the reference vocabulary's (`ref_prims`): each element-wise op is the f32 operation torch's CPU kernel performs for
it (true division, IEEE `sqrt`, `pow` 2 and 3 as products, `rsqrt` as `1 / sqrt`, `lerp` as `fma(w, end - start, start)` or
`fma(w - 1, end - start, end)`, `addcmul` as `fma(value * t1, t2, self)`, `addcdiv` as `self + (value * t1) / t2`, `silu_backward`
as `(dy * s) * fma(x, 1 - s, 1)`), and each reduction is the vocabulary's sequential one (`DotRef`, ascending from +0, where torch's
kernels sum in blocks). Matrix products and the element-wise ops of `Binary`, `Unary` and `SiluBackward` also take bf16 values: a
product is `LinearRef_v1{DT=bf16}` (operands widened, the f32 dot, its result rounded), and an element-wise op widens its bf16
operands to f32 (`Bf16ToF32_v1`, exact), performs the f32 operation and rounds a bf16 result once (`F32ToBf16Rn_v1`), as torch
CPU's bf16 kernels do; where torch's bf16 kernel rounds otherwise, the rule refuses. At a compute capability `cc`, a bf16 `mm` or
`mv` is instead F.linear's GEMM as that GPU's cuBLAS computes it (`Matmul`): the route its capture fitted
(`cublas_bf16.train_route`), each k16 step the tensor core's dot (`HopperBF16WgmmaDot16_v1`); `add`, `sub`, `mul` and `div` are
the GPU's (`Binary`), as are the element-wise ops' operations, `silu_backward`'s, `lerp`'s, `addcdiv`'s, `_foreach_add`'s and a
fused AdamW step's (`Unary`, `SiluBackward`, `Lerp`, `Addcdiv`, `ForeachAdd`, `FusedAdamW`, whose `beta^step` is
`exp(step * log(beta))`: CUDA's `powf` is not modeled); `exp`, alone or inside `sigmoid` and `silu_backward`, is CUDA's `expf`
(`NvExpf_v1`); a vector norm or a sum of a whole tensor is its `reduce_kernel`'s (`rules.training_reduce.FullReduce`); an f32
embedding's gradient is its kernels' (`EmbeddingDenseBackward`); and the conversions to bf16, to and from f16 and from f32 to
int64 and uint8, a clamp and a `kthvalue` are its own (`rules.training_masks`), and `log` is CUDA's `logf` (`Log`).

    training_rules(min_view_leaves, cc, tf32) -> RuleSet     the training rule set

Views and static float tensors are `rules.training_views`'s, matrix products `rules.training_matmul`'s, softmax and log-softmax
`rules.training_softmax`'s, sums and means over axes, the loss and the embedding's gradient `rules.training_loss`'s, torch's flash
attention forward at a compute capability `rules.training_attention`'s, comparisons,
masks, integer sums, adds, shifts and ands, conversions, functionalized copies, table lookups, `amax` and `kthvalue`
`rules.training_masks`'s and the helpers the rules emit with `rules.training_emit`'s; a survey of what the rules refuse is
`rules.training_survey`'s.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.ops import scalar as SC
from verity.definitions.registry import int8_exact as IE
from verity.definitions.registry import moe as MOE
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as R
from verity.definitions.registry import sampling as SM

from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.training_attention import (
    FlashSdpa,
    FlashSdpaBackward,
)
from verity_vllm.program.frontend.rules.training_emit import (
    _DT,
    F32,
    _alpha,
    _batch,
    _bits,
    _const,
    _ew,
    _flat,
    _operand,
    _operand_ok,
    _rounded,
    _shape,
    _TrainRule,
    _widened,
    aten,
)
from verity_vllm.program.frontend.rules.training_loss import (
    EmbeddingDenseBackward,
    NllLossBackward,
    NllLossForward,
    Reduce,
)
from verity_vllm.program.frontend.rules.training_masks import (
    AMax,
    Cast,
    Clamp,
    Compare,
    Copy,
    Index,
    IntOps,
    IntSum,
    KthValue,
    Logic,
    Where,
)
from verity_vllm.program.frontend.rules.training_matmul import Matmul
from verity_vllm.program.frontend.rules.training_reduce import FullReduce
from verity_vllm.program.frontend.rules.training_softmax import (
    LogSoftmax,
    LogSoftmaxBackward,
    SafeSoftmax,
    SoftmaxBackward,
)
from verity_vllm.program.frontend.rules.training_views import (
    BoxPermute,
    StaticFloat,
    StaticFloats,
    _info,
)
from verity_vllm.program.frontend.strided_views import _tensor_type

# ---------------------------------------------------------------------------------------------------------
# element-wise
# ---------------------------------------------------------------------------------------------------------

_BINARY = {aten.add.Tensor: "add", aten.add.Scalar: "add", aten.sub.Tensor: "sub", aten.sub.Scalar: "sub", aten.mul.Tensor: "mul",
           aten.mul.Scalar: "mul", aten.div.Tensor: "div", aten.div.Scalar: "div"}
#: each operation's f32 Definition: torch CPU's, and the GPU's (every NaN result `F32_NAN_PTX`)
_F32 = {"add": (P.F32Add, P.F32AddV2), "sub": (R.F32Sub, P.F32SubV2), "mul": (P.F32Mul, P.F32MulV2), "div": (P.F32Div, P.F32DivV2),
        "fma": (P.F32Fma, P.F32FmaV2)}
#: `exp`: the reference vocabulary's (correctly rounded) on the CPU, and CUDA's `expf` (`moe.NvExpf`) on the GPU
_EXP = (R.F32ExpRn, MOE.NvExpf)


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
    """`neg`, `abs`, `floor`, `sqrt`, `rsqrt`, `reciprocal`, `exp`, `sigmoid`, `pow.Tensor_Scalar` of an f32 or bf16 value, as
    torch's CPU kernels compute them: `abs` clears the sign bit (`F32Fabs_v1`, f32 only), `floor` is `F32Rint_v1` less one where
    that rounded up (`F32GtStrict_v1`, `F32Sub_v1`, f32 only; exact on every word, an x rounded up being no integer, so below 2^23
    in magnitude; a NaN quieted), `rsqrt` = 1 / sqrt, `sigmoid` = 1 / (1 + exp(-x)), `pow` 2 / 3 products, 0.5 sqrt, -0.5 rsqrt,
    -1 and -2 reciprocals, 1 the operand, 0 one; `exp` is `F32ExpRn_v1` (correctly rounded, where torch's vectorized exp is not).
    `sqrt` is IEEE's (`F32Sqrt_v1`, as CUDA's `sqrt.rn`); torch CPU's (and its `pow` 0.5) is MKL's, which is not IEEE's on some
    CPUs. A bf16 operand is widened and the result rounded to bf16, except where torch's bf16 kernel rounds in between, which is
    refused: `pow` 3 and -2 round the square, and `rsqrt` (and `pow` -0.5) the square root in its scalar loop; a bf16 `abs` is
    refused too (torch's CPU kernel keeps a NaN's payload, its CUDA kernel rounds through f32). At a compute capability `cc`,
    `exp` is CUDA's `expf` (`NvExpf_v1`, which its CUDA kernel calls), `sigmoid` its CUDA kernel's `1 / (1 + expf(-x))`, the
    divisions, products and sums are the GPU's (`F32Div_v2`, `F32Mul_v2`, `F32Add_v2`) and a bf16 result is rounded by
    `F2fpBf16_v1`."""
    info = _info("Unary", "aten neg/abs/floor/sqrt/rsqrt/reciprocal/exp/sigmoid/pow.Tensor_Scalar of f32 or bf16", "the operand is "
                 "an f32 or bf16 value", "pow's exponent is one torch special-cases (0, 0.5, 1, 2, 3, -0.5, -1, -2)",
                 "bf16: not abs, floor or rsqrt, nor pow 3, -2 or -0.5", "one batch per step over the elements")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.neg.default, aten.abs.default, aten.floor.default, aten.sqrt.default, aten.rsqrt.default,
                aten.reciprocal.default, aten.exp.default, aten.sigmoid.default, aten.pow.Tensor_Scalar}

    def applies(self, node, ctx):
        f32_only = node.target in (aten.abs.default, aten.floor.default)
        why = _operand_ok(ctx, node.args[0], (torch.float32,) if f32_only else tuple(_DT))
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
        gpu = self.cc is not None
        div, mul, add = (_F32[op][gpu] for op in ("div", "mul", "add"))
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
        elif t is aten.floor.default:
            r = ew(IE.F32Rint, x)
            y = ew(SC.SelectF32, _flat(_ew(ctx, node, R.F32GtStrict, [r, x], S, w=1), S, 1), ew(R.F32Sub, r, one), r)
        elif t is aten.sqrt.default or e == 0.5:
            y = ew(R.F32Sqrt, x)
        elif t is aten.rsqrt.default or e == -0.5:
            y = ew(div, one, ew(R.F32Sqrt, x))
        elif t is aten.reciprocal.default or e == -1:
            y = ew(div, one, x)
        elif t is aten.exp.default:
            y = ew(_EXP[gpu], x)
        elif t is aten.sigmoid.default:
            y = ew(div, one, ew(add, ew(_EXP[gpu], ew(SC.F32Neg, x)), one))
        elif e == 2:
            y = ew(mul, x, x)
        elif e == 3:
            y = ew(mul, ew(mul, x, x), x)
        elif e == -2:
            y = ew(div, one, ew(mul, x, x))
        else:
            y = ew(add, one, (_const(ctx, node, 0), None))
        ctx.define(node, _rounded(ctx, node, y[0].reshape(_tensor_type(S, 32)), S, P.F2fpBf16 if gpu else P.F32ToBf16Rn))
        ctx.record(self, node, {"op": str(t), "exponent": e, "dtype": _DT[ctx.dtype_of(node)]}, emitted=f"x{math.prod(S)}")


class Log(_TrainRule):
    """`log` of an f32 or bf16 value at a compute capability `cc`, as torch's CUDA kernel computes it: CUDA's `logf`, which is
    libdevice's ``__nv_logf`` without flush-to-zero (`NvLogf_v4`, every NaN result 0x7FFFFFFF; on bits `NvLogf_v5`), a bf16
    operand widened and the result rounded by `F2fpBf16_v1`. Without `cc` it refuses: torch CPU's vectorized log is not
    modeled."""
    info = _info("Log", "aten.log of f32 or bf16 at a compute capability", "a compute capability",
                 "the operand is an f32 or bf16 value", "one batch per step over the elements")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.log.default}

    def applies(self, node, ctx):
        if self.cc is None:
            return "numerics_unregistered: torch CPU's log is not modeled; at a compute capability it is CUDA's logf"
        why = _operand_ok(ctx, node.args[0], tuple(_DT))
        if why or not isinstance(ctx.val(node.args[0]), Coll):
            return why or "the operand is not a value"
        return True

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        y = _ew(ctx, node, SM.NvLogfV4, [_operand(ctx, node, node.args[0], S)], S)
        ctx.define(node, _rounded(ctx, node, y, S, P.F2fpBf16))
        ctx.record(self, node, {"shape": list(S), "dtype": _DT[ctx.dtype_of(node)]}, emitted=f"NvLogf_v4 x{math.prod(S)}")


class Lerp(_TrainRule):
    """`lerp.Scalar(start, end, w)` and `_foreach_lerp.Scalar` (each pair of its lists, an EMA's update), as torch's kernels
    compute it (`Lerp.h`, the CPU's vectorized form alike): `fma(c, end - start, base)` with `c, base = (w, start)` if |w| < 0.5
    else `(w - 1, end)` (`F32Sub_v1`, `F32Fma_v1`), and at a compute capability `cc` the GPU's operations (`F32Sub_v2`,
    `F32Fma_v2`): its `self + w * (end - self)` and `end - (end - self) * (1 - w)` are each one FFMA."""
    info = _info("Lerp", "aten.lerp.Scalar / _foreach_lerp.Scalar of f32 values", "start and end are f32 values or static, of one "
                 "shape in a foreach pair", "the weight is a Python number")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.lerp.Scalar, aten._foreach_lerp.Scalar}

    def _pairs(self, node):
        if node.target is aten._foreach_lerp.Scalar:
            return list(zip(node.args[0], node.args[1])) if len(node.args[0]) == len(node.args[1]) else None
        return [tuple(node.args[:2])]

    def applies(self, node, ctx):
        pairs = self._pairs(node)
        if pairs is None:
            return "the lists' lengths differ"
        whys = [w for pair in pairs for w in (_operand_ok(ctx, a) for a in pair) if w]
        if whys:
            return whys[0]
        if node.target is aten._foreach_lerp.Scalar and any(_shape(ctx, s) != _shape(ctx, e) for s, e in pairs):
            return "a pair's shapes differ"
        if not isinstance(node.args[2], (int, float)) or isinstance(node.args[2], bool):
            return "the weight is not a Python number"
        return True

    def emit(self, node, ctx):
        gpu = self.cc is not None
        w = float(np.float32(node.args[2]))
        small = abs(w) < 0.5
        c = (_const(ctx, node, _bits(w if small else float(np.float32(w) - np.float32(1.0)))), None)
        outs = []
        for s, e in self._pairs(node):
            S = _shape(ctx, s) if node.target is aten._foreach_lerp.Scalar else ctx.shape_of_node(node)
            start, end = (_operand(ctx, node, a, S) for a in (s, e))
            diff = _flat(_ew(ctx, node, _F32["sub"][gpu], [end, start], S), S)
            outs.append(_ew(ctx, node, _F32["fma"][gpu], [c, diff, start if small else end], S))
        ctx.define(node, outs if node.target is aten._foreach_lerp.Scalar else outs[0])
        sub, fma = _F32["sub"][gpu].id, _F32["fma"][gpu].id
        ctx.record(self, node, {"weight": w, "small": small, "tensors": len(outs)},
                   emitted=f"{sub}+{fma} x{sum(o.type.leaves for o in outs)}")


def _value(node):
    """`addcmul` / `addcdiv`'s value (keyword-only in their schemas)."""
    return node.kwargs.get("value", node.args[3] if len(node.args) > 3 else 1)


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
        v = (_const(ctx, node, _bits(_value(node))), None)
        vt1 = _flat(_ew(ctx, node, P.F32Mul, [v, t1], S), S)
        ctx.define(node, _ew(ctx, node, P.F32Fma, [vt1, t2, s], S))
        ctx.record(self, node, {"shape": list(S)}, emitted=f"F32Mul_v1 + F32Fma_v1 x{math.prod(S)}")


class Addcdiv(_TrainRule):
    """`addcdiv(self, t1, t2, value)` of f32 values as torch's kernels compute it: torch CPU's `self + (value * t1) / t2`
    (`F32Mul_v1`, `F32Div_v1`, `F32Add_v1`), and at a compute capability `cc` the GPU's `fma(value, t1 / t2, self)` (`F32Div_v2`,
    `F32Fma_v2`), the explicit `std::fma` of its `pointwise_op_impl` (`registry.torch_optim`'s step 6)."""
    info = _info("Addcdiv", "aten.addcdiv of f32 values", "operands are f32 values or static", "value is a Python number")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.addcdiv.default}

    def applies(self, node, ctx):
        whys = [w for w in (_operand_ok(ctx, a) for a in node.args[:3]) if w]
        if whys:
            return whys[0]
        return True if type(_value(node)) in (int, float) else "value is not a Python number"

    def emit(self, node, ctx):
        gpu = self.cc is not None
        S = ctx.shape_of_node(node)
        s, t1, t2 = (_operand(ctx, node, a, S) for a in node.args[:3])
        v = (_const(ctx, node, _bits(_value(node))), None)
        mul, div, add, fma = (_F32[k][gpu] for k in ("mul", "div", "add", "fma"))
        if gpu:
            fns = [div, fma]
            y = _ew(ctx, node, fma, [v, _flat(_ew(ctx, node, div, [t1, t2], S), S), s], S)
        else:
            fns = [mul, div, add]
            vt1 = _flat(_ew(ctx, node, mul, [v, t1], S), S)
            y = _ew(ctx, node, add, [s, _flat(_ew(ctx, node, div, [vt1, t2], S), S)], S)
        ctx.define(node, y)
        ctx.record(self, node, {"shape": list(S), "value": _value(node)},
                   emitted=f"{' + '.join(f.id for f in fns)} x{math.prod(S)}")


class SiluBackward(_TrainRule):
    """`silu_backward(dy, x)` = `(dy * s) * fma(x, 1 - s, 1)` with `s = 1 / (1 + exp(-x))`, the form of torch's kernels (its
    CPU kernel's scalar loop exactly; nvcc contracts `1 + x * (1 - s)` to one FFMA): `F32ExpRn_v1` as `Unary`'s `exp`,
    `F32Add_v1`, `F32Div_v1`, `F32Sub_v1`, `F32Fma_v1`, `F32Mul_v1`. bf16 operands are widened and the result rounded once
    (`F32ToBf16Rn_v1`), as both kernels compute in f32. At a compute capability `cc` the operations are the GPU's
    (`F32Add_v2`, ..., `F32Mul_v2`), `exp` CUDA's `expf` (`NvExpf_v1`, as `Unary`'s) and a bf16 result is rounded by
    `F2fpBf16_v1`."""
    info = _info("SiluBackward", "aten.silu_backward of f32 or bf16", "dy and x are values of the result's dtype",
                 "one batch per step over the elements")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.silu_backward.default}

    def applies(self, node, ctx):
        for a in node.args[:2]:
            why = _operand_ok(ctx, a, tuple(_DT))
            if why or not isinstance(ctx.val(a), Coll):
                return why or "an operand is not a value"
            if ctx.dtype_of(a) != ctx.dtype_of(node):
                return f"numerics_unregistered: {a.name} has dtype {ctx.dtype_of(a)}, the result {ctx.dtype_of(node)}"
        return True

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        gpu = self.cc is not None
        one = (_const(ctx, node, _bits(1.0)), None)

        def ew(op, *ops):
            return _flat(_ew(ctx, node, _F32[op][gpu] if isinstance(op, str) else op, list(ops), S), S)

        dy, x = (_operand(ctx, node, a, S) for a in node.args[:2])
        s = ew("div", one, ew("add", one, ew(_EXP[gpu], ew(SC.F32Neg, x))))
        t = ew("fma", x, ew("sub", one, s), one)
        y = _ew(ctx, node, _F32["mul"][gpu], [ew("mul", dy, s), t], S)
        ctx.define(node, _rounded(ctx, node, y, S, P.F2fpBf16 if gpu else P.F32ToBf16Rn))
        ctx.record(self, node, {"shape": list(S), "dtype": _DT[ctx.dtype_of(node)]}, emitted=f"x{math.prod(S)}")


class ForeachAdd(_TrainRule):
    """`_foreach_add.Scalar(tensors, s)`: each f32 tensor plus the number `s` (an optimizer's step counts), element by element,
    `F32Add_v1`, and at a compute capability `cc` its kernel's `F32Add_v2`; the results a list."""
    info = _info("ForeachAdd", "aten._foreach_add.Scalar of f32 values", "each tensor is an f32 value",
                 "the scalar is a Python number")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten._foreach_add.Scalar}

    def applies(self, node, ctx):
        ts, s = node.args[:2]
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            return "the scalar is not a Python number"
        for a in ts:
            why = _operand_ok(ctx, a)
            if why or not isinstance(ctx.val(a), Coll):
                return why or "a tensor is not a value"
        return True

    def emit(self, node, ctx):
        ts, s = node.args[:2]
        fn = _F32["add"][self.cc is not None]
        c = (_const(ctx, node, _bits(s)), None)
        outs = [_ew(ctx, node, fn, [_operand(ctx, node, a, _shape(ctx, a)), c], _shape(ctx, a)) for a in ts]
        ctx.define(node, outs)
        ctx.record(self, node, {"tensors": len(ts), "scalar": s},
                   emitted=f"{fn.id} x{sum(math.prod(_shape(ctx, a)) for a in ts)}")


_ADAMW_KW = ("lr", "beta1", "beta2", "weight_decay", "eps")


class FusedAdamW(_TrainRule):
    """`_fused_adamw` (`_fused_adamw_` functionalized: `torch.optim.AdamW(fused=True)`'s step) of f32 parameters, gradients,
    moments and step counts at a compute capability `cc`, element by element as torch's CUDA kernel computes it in f32
    (`FusedAdamMathFunctor`), each hyperparameter its f32 word and `t = RN(lr * weight_decay)` a constant:

        p1 = fma(-t, p, p)                                    `param -= lr * weight_decay * param`, one FFMA (none at decay 0)
        m1 = fma(beta1, m, fma(-beta1, g, g))
        v1 = fma(beta2, v, fma(-beta2, g * g, g * g))
        p2 = p1 - RN(step_size * m1) / (sqrt(v1) / bc2_sqrt + eps)

    (`F32Fma_v2`, `F32Mul_v2`, `F32Div_v2`, `F32Add_v2`, `F32Sub_v2`; `sqrt` IEEE's, `F32Sqrt_v1`), each tensor's
    `step_size = lr / bc1`, `bc1 = 1 - beta1^step` and `bc2_sqrt = sqrt(1 - beta2^step)` from its step count. The kernel's `beta^step`
    is CUDA's `powf`, which no Definition models: here it is `exp(step * log(beta))`, the product `F32Mul_v2`, the exp
    `F32ExpRn_v1` and `log(beta)` a correctly rounded constant. The gradients are returned as given. Refused without `cc` (torch
    CPU's fused kernel computes in double), and amsgrad, maximize, a gradient scale or inf flag, or a tensor not f32 (the
    mixed-precision kernel is another)."""
    info = _info("FusedAdamW", "aten._fused_adamw of f32 values at a compute capability", "every tensor an f32 value, each step "
                 "count one word", "no amsgrad, maximize, grad_scale or found_inf", "beta^step is exp(step * log(beta)), not "
                 "CUDA's powf")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten._fused_adamw.default}

    def applies(self, node, ctx):
        if self.cc is None:
            return ("numerics_unregistered: torch CPU's fused AdamW computes in double; the GPU's kernel is lowered at a compute "
                    "capability (training_rules(cc=...))")
        kw = node.kwargs
        if kw.get("amsgrad") or kw.get("maximize") or kw.get("grad_scale") is not None or kw.get("found_inf") is not None:
            return "numerics_unregistered: amsgrad, maximize, a gradient scale or an inf flag"
        ps, gs, ms, vs, mx, ss = node.args[:6]
        if mx or not len(ps) == len(gs) == len(ms) == len(vs) == len(ss):
            return "the lists' lengths differ (or max_exp_avg_sqs is not empty)"
        for a in (*ps, *gs, *ms, *vs, *ss):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll) or ctx.dtype_of(a) != torch.float32:
                return "numerics_unregistered: a tensor that is not an f32 value (the mixed-precision kernel is another)"
        for p, g, m, v, s in zip(ps, gs, ms, vs, ss):
            if not _shape(ctx, p) == _shape(ctx, g) == _shape(ctx, m) == _shape(ctx, v) or math.prod(_shape(ctx, s)) != 1:
                return "a parameter's gradient or moments have another shape, or a step count is not one word"
        if not all(isinstance(kw.get(k), (int, float)) for k in _ADAMW_KW):
            return "a hyperparameter is not a Python number"
        return True

    def emit(self, node, ctx):
        ps, gs, ms, vs, _mx, ss = node.args[:6]
        lr, b1, b2, wd, eps = (np.float32(node.kwargs[k]) for k in _ADAMW_KW)
        fma, mul, div, add, sub = (_F32[k][1] for k in ("fma", "mul", "div", "add", "sub"))

        def c(x) -> tuple[Coll, None]:
            return _const(ctx, node, _bits(x)), None

        T = len(ps)
        steps = Coll(Array(T, F32), concat([ctx.coll(s).reshape(Array(1, F32)).refs for s in ss]))

        def one_minus_pow(beta) -> Coll:
            log_beta = _const(ctx, node, R.f32_log_rn_bits(_bits(beta)))
            x = _batch(ctx, node, R.F32ExpRn, _batch(ctx, node, mul, steps, log_beta, axes=(0, None)), axes=(0,))
            return _batch(ctx, node, sub, _const(ctx, node, _bits(1.0)), x, axes=(None, 0))

        step_size = _batch(ctx, node, div, _const(ctx, node, _bits(lr)), one_minus_pow(b1), axes=(None, 0))
        bc2_sqrt = _batch(ctx, node, R.F32Sqrt, one_minus_pow(b2), axes=(0,))
        new_p, new_m, new_v = [], [], []
        for i, (p, g, m, v) in enumerate(zip(ps, gs, ms, vs)):
            S = _shape(ctx, p)

            def op(fn, *xs, S=S):
                return _flat(_ew(ctx, node, fn, list(xs), S), S)

            P_, G_, M_, V_ = (_operand(ctx, node, a, S) for a in (p, g, m, v))
            p1 = op(fma, c(-(lr * wd)), P_, P_) if wd != 0 else P_
            m1 = op(fma, c(b1), M_, op(fma, c(-b1), G_, G_))
            gg = op(mul, G_, G_)
            v1 = op(fma, c(b2), V_, op(fma, c(-b2), gg, gg))
            den = op(add, op(div, op(R.F32Sqrt, v1), (bc2_sqrt[i], None)), c(eps))
            p2 = op(sub, p1, op(div, op(mul, (step_size[i], None), m1), den))
            new_p.append(p2[0].reshape(_tensor_type(S, 32)))
            new_m.append(m1[0].reshape(_tensor_type(S, 32)))
            new_v.append(v1[0].reshape(_tensor_type(S, 32)))
        ctx.define(node, [new_p, [ctx.coll(g) for g in gs], new_m, new_v, []])
        n = sum(math.prod(_shape(ctx, p)) for p in ps)
        ctx.record(self, node, {"tensors": T, "elements": n, **{k: float(x) for k, x in zip(_ADAMW_KW, (lr, b1, b2, wd, eps))},
                                "pow": "exp(step * log(beta)): F32Mul_v2, F32ExpRn_v1 (CUDA's powf is not modeled)"},
                   emitted=f"F32Fma_v2 x{(5 if wd != 0 else 4) * n} + F32Mul_v2/F32Div_v2/F32Add_v2/F32Sub_v2/F32Sqrt_v1 per element")


def training_rules(min_view_leaves: int = 0, cc: tuple[int, int] | None = None, tf32: bool = False,
                   deterministic: bool = False) -> RuleSet:
    """The default rules that target ATen ops (`rules.TORCH_RULES`) and the default families, with the training rules: views of
    static floats, `BoxPermute` (views of more than `min_view_leaves` leaves
    that are not one run) and `Matmul` before them, the element-wise, optimizer-count, reduction, softmax, loss, embedding,
    comparison, mask, integer, conversion, lookup, order-statistic and copy rules after. `min_view_leaves` is 0 by default:
    the default views write a permuted view of fewer leaves than their limit as a list of its leaves, which the root's replay then
    substitutes as one reference part per leaf.  `cc` (`cublas_bf16.TRAIN_CC`): bf16 `mm`s and `mv`s as that GPU's cuBLAS computes
    them (`Matmul`), every add, sub, mul and div as its kernels do (`Binary`; `Unary`, `SiluBackward`, `ForeachAdd`, `Lerp` and
    `Addcdiv` take the same operations, and `Unary`'s and `SiluBackward`'s `exp` is CUDA's `expf`) and the conversions to bf16,
    to and from f16 and from f32 to int64 and uint8 as its kernels do (`Cast`), both placed first, before the default rules' CPU kinds, a whole tensor's vector norm or sum as its `reduce_kernel` does
    (`FullReduce`, before `Reduce`), an f32 embedding's gradient as its kernels do (`EmbeddingDenseBackward`), a fused AdamW step
    and a clamp as their kernels do (`FusedAdamW`, `Clamp`), `kthvalue` as its radix select does (`KthValue`) and torch's flash
    attention forward and backward as its kernels do (`FlashSdpa`, `FlashSdpaBackward`, placed first), in the rule set
    ``training-cc<major>.<minor>``.  `tf32`, at a `cc`: the step sets cuda.matmul.allow_tf32, which the traced graph doesn't
    show, so its f32 `mm`s and `bmm`s are cuBLAS's TF32 GEMMs: `Matmul` lowers an `mm` at a shape the TF32 capture covers
    (`cublas_bf16.TRAIN_TF32`) as cuBLAS's unsplit TF32 GEMM (`GemmTf32_v1`) and refuses the rest
    (``training-cc<major>.<minor>-tf32``).  `deterministic`, at a `cc`: the step runs under deterministic algorithms, which the
    traced graph doesn't show either, so the flash attention backward is its deterministic kernels' and `FlashSdpaBackward` lowers
    it; without it the rule refuses every call (the suffix ``-deterministic``)."""
    from verity_vllm.program.frontend.rules import DEFAULT_FAMILIES, TORCH_RULES

    first = [StaticFloats(cc), BoxPermute(min_view_leaves), Matmul(cc, tf32)] + ([Binary(cc), Cast(cc)] if cc else [])
    last = ([] if cc else [Binary()]) + [Unary(cc), Lerp(cc), Addcmul(), Addcdiv(cc), SiluBackward(cc), ForeachAdd(cc),
                                         FusedAdamW(cc), FullReduce(cc), Reduce(), IntSum(), AMax(), SafeSoftmax(),
                                         SoftmaxBackward(), LogSoftmax(), LogSoftmaxBackward(), NllLossForward(), NllLossBackward(),
                                         EmbeddingDenseBackward(cc), Log(cc), Compare(), Logic(), IntOps(), Where(), Clamp(cc), Index(),
                                         KthValue(cc), Copy(cc)] + ([] if cc else [Cast()])
    if deterministic and cc is None:
        raise ValueError("training_rules: deterministic is a GPU's (its flash attention backward); without a cc no rule reads it")
    name = "training" if cc is None else f"training-cc{cc[0]}.{cc[1]}" + ("-tf32" if tf32 else "") + ("-deterministic" if deterministic else "")
    return RuleSet(name, first + ([FlashSdpa(cc), FlashSdpaBackward(cc, deterministic)] if cc else []) + list(TORCH_RULES) + last,
                   list(DEFAULT_FAMILIES))


__all__ = ["training_rules", "Matmul", "Binary", "Unary", "Lerp", "Addcmul", "Addcdiv", "SiluBackward", "ForeachAdd", "FusedAdamW",
           "FullReduce", "Reduce", "IntSum", "AMax", "SafeSoftmax", "SoftmaxBackward", "LogSoftmax", "LogSoftmaxBackward",
           "NllLossForward", "NllLossBackward", "EmbeddingDenseBackward", "Log", "Compare", "Logic", "IntOps", "Where", "Clamp", "Cast",
           "Index", "KthValue", "Copy"]
