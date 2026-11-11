"""The training rules for masks, integers, conversions and order statistics (`rules.training`): what a quantizer computes beside its
f32 arithmetic. Each is exact and the same word on torch's CPU and CUDA kernels, except where a docstring names the GPU's
(`training_rules(cc=...)`).

    Compare     gt / ge / lt / le / eq / ne of f32 or f16: IEEE, a NaN unordered and -0 equal to +0
    Logic       and / or / not of bool values
    IntOps      + / << / >> / & of int64 values
    Where       where.self with an f32, bf16, f16 or int64 result
    Clamp       clamp by Python numbers of an int64 or f32 value, at `cc` as the GPU's kernel clamps
    Cast        conversions between int64, uint8, bool and f32, and at `cc` the GPU's to bf16, to and from f16 and from f32 to
                int64 and uint8
    Copy        a functionalized copy_: its source by reference, or converted as `Cast` converts (bf16 to f32 too)
    IntSum      sum of a bool or int64 value to int64
    AMax        amax of an f32 value
    Index       a 1-D f32 table (a codebook), static or a value, indexed by an integer value
    KthValue    kthvalue of an f32 value at `cc`, as the GPU's radix select
"""

from __future__ import annotations

import math
import operator

import torch

from verity.definitions.ops import scalar as SC
from verity.circuits.refs import Coll, Strided, broadcast
from verity.circuits.types import Array, Value
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as R
from verity.definitions.registry import torch_words as W
from verity.definitions.registry.dsv4_decode_indices import gather_i32
from verity_vllm.program.frontend.rules.training_emit import (_DT, _alpha, _batch, _bits, _broadcast, _const, _converted, _ew,
                                                              _flat, _from_bits, _operand, _operand_ok, _ref, _shape,
                                                              _static_coll, _tokens, _TrainRule, _tree, _word_ew, _word_ok,
                                                              _word_operand, aten)
from verity_vllm.program.frontend.rules.training_views import StaticFloat, _info, _view_expand, _view_permute
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import WIDTH_OF_DTYPE, shape_of

I32 = Value(32)
M32 = W.M32


def _a_value(ctx, args) -> bool:
    return any(isinstance(a, torch.fx.Node) and isinstance(ctx.val(a), Coll) for a in args)


def _first(whys) -> str | None:
    return next((w for w in whys if w), None)


_CMP = {aten.gt.Tensor: "gt", aten.gt.Scalar: "gt", aten.ge.Tensor: "ge", aten.ge.Scalar: "ge", aten.lt.Tensor: "lt",
        aten.lt.Scalar: "lt", aten.le.Tensor: "le", aten.le.Scalar: "le", aten.eq.Tensor: "eq", aten.eq.Scalar: "eq",
        aten.ne.Tensor: "ne", aten.ne.Scalar: "ne"}


def _f16_ok(ctx, a) -> str | None:
    """None if `a` is an operand of an f16 comparison that widening reads exactly: an f16 value, or a Python number f16 holds;
    else why not."""
    if isinstance(a, torch.fx.Node):
        if isinstance(ctx.val(a), Coll) and ctx.dtype_of(a) == torch.float16:
            return None
        return f"numerics_unregistered: an f16 comparison with {a.name}, not an f16 value (torch promotes one or rounds it)"
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return f"operand {a!r}"
    held = a != a or torch.tensor(float(a), dtype=torch.float16).item() == a
    return None if held else f"numerics_unregistered: a number f16 rounds ({a!r}; torch rounds it to f16 first)"


def _widened_operand(ctx, node, a, S) -> tuple[Coll, int | None]:
    """`_operand`'s f32 words, an f16 value's widened (`F16ToF32_v1`, exact)."""
    if isinstance(a, torch.fx.Node) and ctx.dtype_of(a) == torch.float16:
        return _broadcast(_converted(ctx, node, ctx.coll(a), W.F16ToF32, 32), S, 32, node.name)
    return _operand(ctx, node, a, S)


class Compare(_TrainRule):
    """`gt` / `ge` / `lt` / `le` / `eq` / `ne` (`.Tensor`, `.Scalar`) of f32 values, static float tensors and Python numbers (each
    its f32 word), broadcast, as IEEE compares them: `F32GtStrict_v1` (`lt` with the operands swapped), `F32Eq_v1`, `ge` and `le`
    their `BitOr_v1`, `ne` the `BitNot_v1` of `F32Eq_v1`. A NaN is unordered and -0 equals +0. f16 values compare with f16 values
    and with Python numbers f16 holds, each widened exactly (`F16ToF32_v1`, whose NaN word no comparison reads), as torch's f16
    comparison reads them. Refused: a bf16 operand (torch compares in bf16, rounding a number to bf16 first), and beside an f16
    value another dtype or a number f16 rounds."""
    info = _info("Compare", "aten gt/ge/lt/le/eq/ne .Tensor/.Scalar of f32 or f16", "operands are f32 values, static float "
                 "tensors or Python numbers, or f16 values and Python numbers f16 holds, one a value",
                 "one batch per step over the output's elements")

    def targets(self):
        return set(_CMP)

    def applies(self, node, ctx):
        args = node.args[:2]
        if any(isinstance(a, torch.fx.Node) and ctx.dtype_of(a) == torch.bfloat16 for a in args):
            return "numerics_unregistered: a bf16 comparison (torch compares in bf16, a number rounded to bf16 first)"
        f16 = any(isinstance(a, torch.fx.Node) and ctx.dtype_of(a) == torch.float16 for a in args)
        if why := _first(_f16_ok(ctx, a) if f16 else _operand_ok(ctx, a) for a in args):
            return why
        return True if _a_value(ctx, args) else "no operand is a value"

    def emit(self, node, ctx):
        S, op = ctx.shape_of_node(node), _CMP[node.target]
        a, b = (_widened_operand(ctx, node, x, S) for x in node.args[:2])
        if op in ("lt", "le"):
            a, b = b, a

        def ew(fn, *ops):
            return _flat(_ew(ctx, node, fn, list(ops), S, w=1), S, 1)

        if op in ("gt", "lt"):
            y = ew(R.F32GtStrict, a, b)
        elif op in ("ge", "le"):
            y = ew(SC.BitOr, ew(R.F32GtStrict, a, b), ew(SC.F32Eq, a, b))
        else:
            y = ew(SC.F32Eq, a, b)
            if op == "ne":
                y = ew(SC.BitNot, y)
        ctx.define(node, y[0].reshape(_tensor_type(S, 1)))
        ctx.record(self, node, {"op": op, "shape": list(S)}, emitted=f"{op} x{math.prod(S)}")


_LOGIC = {aten.bitwise_and.Tensor: SC.BitAnd, aten.logical_and.default: SC.BitAnd, aten.__and__.Tensor: SC.BitAnd,
          aten.bitwise_or.Tensor: SC.BitOr, aten.logical_or.default: SC.BitOr, aten.__or__.Tensor: SC.BitOr,
          aten.bitwise_not.default: SC.BitNot, aten.logical_not.default: SC.BitNot}


class Logic(_TrainRule):
    """`and` / `or` / `not` (bitwise or logical) of bool values, static bool tensors and Python bools, broadcast: `BitAnd_v1`,
    `BitOr_v1`, `BitNot_v1`."""
    info = _info("Logic", "aten bitwise_and/or/not, logical_and/or/not and __and__/__or__ of bool", "operands are bool values, "
                 "static bool tensors or Python bools, one a value")

    def targets(self):
        return set(_LOGIC)

    def _args(self, node):
        return node.args[:1 if _LOGIC[node.target] is SC.BitNot else 2]

    def applies(self, node, ctx):
        if ctx.dtype_of(node) != torch.bool:
            return f"numerics_unregistered: result dtype {ctx.dtype_of(node)}"
        args = self._args(node)
        if why := _first(_word_ok(ctx, a, torch.bool) for a in args):
            return why
        return True if _a_value(ctx, args) else "no operand is a value"

    def emit(self, node, ctx):
        S, fn = ctx.shape_of_node(node), _LOGIC[node.target]
        ctx.define(node, _word_ew(ctx, node, fn, self._args(node), S, torch.bool))
        ctx.record(self, node, {"shape": list(S)}, emitted=f"{fn.id} x{math.prod(S)}")


_INT_OPS = {aten.add.Tensor: W.I64Add, aten.add.Scalar: W.I64Add,
            aten.__lshift__.Tensor: W.I64Shl, aten.__lshift__.Scalar: W.I64Shl, aten.bitwise_left_shift.Tensor: W.I64Shl,
            aten.__rshift__.Tensor: W.I64Shr, aten.__rshift__.Scalar: W.I64Shr, aten.bitwise_right_shift.Tensor: W.I64Shr,
            aten.bitwise_and.Tensor: W.I64And, aten.bitwise_and.Scalar: W.I64And, aten.__and__.Tensor: W.I64And,
            aten.__and__.Scalar: W.I64And}


class IntOps(_TrainRule):
    """`+`, `<<`, `>>` and `&` (`add` with alpha 1, `__lshift__`, `__rshift__`, `bitwise_left_shift`, `bitwise_right_shift`,
    `bitwise_and`, `__and__`) of int64 values, static int64 tensors and Python ints, broadcast, as torch's kernels compute them:
    `I64Add_v1` (modulo 2^64), `I64Shl_v1` (0 for an amount below 0 or past 63), `I64Shr_v1` (the sign shifted in, by 63 for an
    amount below 0 or past 62), `I64And_v1`."""
    info = _info("IntOps", "aten add (alpha 1)/__lshift__/__rshift__/bitwise_left_shift/bitwise_right_shift/bitwise_and/__and__ "
                 "of int64", "operands are int64 values, static int64 tensors or Python ints, one a value")

    def targets(self):
        return set(_INT_OPS)

    def applies(self, node, ctx):
        if ctx.dtype_of(node) != torch.int64:
            return f"numerics_unregistered: result dtype {ctx.dtype_of(node)}"
        if _INT_OPS[node.target] is W.I64Add and _alpha(node) != 1:
            return "numerics_unregistered: an int64 add with alpha != 1"
        args = node.args[:2]
        if why := _first(_word_ok(ctx, a, torch.int64) for a in args):
            return why
        return True if _a_value(ctx, args) else "no operand is a value"

    def emit(self, node, ctx):
        S, fn = ctx.shape_of_node(node), _INT_OPS[node.target]
        ctx.define(node, _word_ew(ctx, node, fn, node.args[:2], S, torch.int64))
        ctx.record(self, node, {"shape": list(S)}, emitted=f"{fn.id} x{math.prod(S)}")


_SELECT = {torch.float32: SC.SelectF32, torch.bfloat16: SC.SelectBf16, torch.float16: SC.SelectBf16, torch.int64: W.SelectI64}


class Where(_TrainRule):
    """`where.self(c, a, b)` of a bool condition (a value or static), element by element, broadcast: an f32 result
    `SelectF32_v1` of the operands' f32 words as `Binary` reads them (a bf16 value widened, a bool one 1.0 or +0, an int64 one
    rounded to nearest, as torch converts each to the result's dtype), an int64 result `SelectI64_v1` and a bf16 or f16 one
    `SelectBf16_v1` (a choice of 16-bit words, whatever their format) of operands of the result's dtype (values, static
    tensors)."""
    info = _info("Where", "aten.where.self with an f32, bf16, f16 or int64 result", "the condition is a bool value or static",
                 "f32: operands f32, bf16, bool or int64 values or static floats; int64, bf16, f16: operands of the result's dtype",
                 "one operand a value")

    def targets(self):
        return {aten.where.self}

    def applies(self, node, ctx):
        dt = ctx.dtype_of(node)
        if dt not in _SELECT:
            return f"numerics_unregistered: result dtype {dt}"
        c, a, b = node.args[:3]
        why = _first([_word_ok(ctx, c, torch.bool)] + [
            _operand_ok(ctx, x, (*_DT, torch.bool, torch.int64)) if dt == torch.float32 else _word_ok(ctx, x, dt) for x in (a, b)])
        if why:
            return why
        return True if _a_value(ctx, (c, a, b)) else "no operand is a value (a static result)"

    def emit(self, node, ctx):
        S, dt = ctx.shape_of_node(node), ctx.dtype_of(node)
        ops = [_word_operand(ctx, node, node.args[0], S, torch.bool)]
        ops += [_operand(ctx, node, x, S) if dt == torch.float32 else _word_operand(ctx, node, x, S, dt) for x in node.args[1:3]]
        ctx.define(node, _ew(ctx, node, _SELECT[dt], ops, S, WIDTH_OF_DTYPE[dt]))
        ctx.record(self, node, {"shape": list(S), "dtype": str(dt)}, emitted=f"{_SELECT[dt].id} x{math.prod(S)}")


def _bounds(node) -> tuple:
    t = node.target
    if t is aten.clamp_min.default:
        return node.args[1], None
    if t is aten.clamp_max.default:
        return None, node.args[1]
    return (node.args[1] if len(node.args) > 1 else node.kwargs.get("min"),
            node.args[2] if len(node.args) > 2 else node.kwargs.get("max"))


class Clamp(_TrainRule):
    """`clamp` / `clamp_min` / `clamp_max` by Python numbers, `min(max(x, lo), hi)` as torch computes it: of an int64 value
    (`I64Le_v1`, `SelectI64_v1`); of an f32 value with each bound its f32 word and a NaN kept, torch CPU's in its vector body and
    scalar tail (`F32GtStrict_v1`, `SelectF32_v1`), and at a compute capability `cc` the GPU's `launch_clamp_scalar`, `isnan(x) ?
    x : fminf(fmaxf(x, lo), hi)` (`F32Eq_v1`, `F32Fmaxf_v1`, `F32Fminf_v1`, `SelectF32_v1`), whose `fmaxf` and `fminf` order -0
    below +0. Refused: a NaN bound (torch fills the result with NaN when both bounds are given), a tensor bound and other
    dtypes."""
    info = _info("Clamp", "aten.clamp / clamp_min / clamp_max of an int64 or f32 value by Python numbers", "the operand a value of "
                 "the result's dtype", "f32: no NaN bound")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.clamp.default, aten.clamp_min.default, aten.clamp_max.default}

    def applies(self, node, ctx):
        a, dt = node.args[0], ctx.dtype_of(node)
        if not isinstance(ctx.val(a), Coll) or ctx.dtype_of(a) != dt or dt not in (torch.int64, torch.float32):
            return f"numerics_unregistered: a clamp of {ctx.dtype_of(a)} to {dt} (this rule lowers an int64 or f32 value)"
        bounds = [x for x in _bounds(node) if x is not None]
        if not bounds:
            return "no bound"
        if dt == torch.int64:
            return True if all(type(x) is int for x in bounds) else "numerics_unregistered: a bound that is not a Python int"
        if not all(type(x) in (int, float) for x in bounds):
            return "numerics_unregistered: a bound that is not a Python number"
        if any(x != x for x in bounds):
            return "numerics_unregistered: a NaN bound"
        return True

    def emit(self, node, ctx):
        S, i64 = ctx.shape_of_node(node), ctx.dtype_of(node) == torch.int64
        w = 64 if i64 else 32

        def ew(fn, *ops, w=w):
            return _flat(_ew(ctx, node, fn, list(ops), S, w), S, w)

        def bound(v):
            return _const(ctx, node, v, 64) if i64 else _const(ctx, node, W.f32_rn_of_int(v) if type(v) is int else _bits(v)), None

        lo, hi = _bounds(node)
        x = _word_operand(ctx, node, node.args[0], S, torch.int64) if i64 else _operand(ctx, node, node.args[0], S)
        if self.cc and not i64:
            y = x if lo is None else ew(SC.F32Fmaxf, x, bound(lo))
            y = y if hi is None else ew(SC.F32Fminf, y, bound(hi))
            x = ew(SC.SelectF32, ew(SC.F32Eq, x, x, w=1), y, x)
        else:
            if lo is not None:
                b = bound(lo)
                x = ew(W.SelectI64, ew(W.I64Le, b, x, w=1), x, b) if i64 else ew(SC.SelectF32, ew(R.F32GtStrict, b, x, w=1), b, x)
            if hi is not None:
                b = bound(hi)
                x = ew(W.SelectI64, ew(W.I64Le, x, b, w=1), x, b) if i64 else ew(SC.SelectF32, ew(R.F32GtStrict, x, b, w=1), b, x)
        ctx.define(node, x[0].reshape(_tensor_type(S, w)))
        ctx.record(self, node, {"shape": list(S), "min": lo, "max": hi}, emitted=f"clamp x{math.prod(S)}")


#: conversions exact on the CPU and the GPU alike
_EXACT = {(torch.int64, torch.uint8): W.I64ToU8, (torch.uint8, torch.int64): W.U8ToI64, (torch.int64, torch.float32): W.I64ToF32Rn}
#: the GPU's conversions, lowered at a compute capability: a Definition, or Definitions applied in turn
_GPU = {(torch.float32, torch.bfloat16): P.F2fpBf16, (torch.float32, torch.float16): W.F32ToF16Rn,
        (torch.float16, torch.float32): W.F16ToF32, (torch.float32, torch.int64): W.F32ToI64Rz,
        (torch.float32, torch.uint8): (W.F32ToI64Rz, W.I64ToU8)}


def _lowers(pair, cc) -> bool:
    """Whether `Cast` lowers the conversion `pair` (from, to) of a value."""
    return (pair in _EXACT or pair == (torch.int64, torch.bool)
            or (pair[0] == torch.bool and pair[1] in (torch.float32, torch.int64)) or bool(cc and pair in _GPU))


def _nonzero(ctx, node, x: Coll) -> Coll:
    """An int64 value's words as bool, x != 0: the `BitNot_v1` of `I64Le_v1(x, 0)` `BitAnd_v1` `I64Le_v1(0, x)`, in its shape."""
    S = tuple(shape_of(x.type)[0])
    zero, v = (_const(ctx, node, 0, 64), None), _flat(x, S, 64)

    def ew(fn, *ops):
        return _flat(_ew(ctx, node, fn, list(ops), S, w=1), S, 1)

    return ew(SC.BitNot, ew(SC.BitAnd, ew(W.I64Le, v, zero), ew(W.I64Le, zero, v)))[0].reshape(_tensor_type(S, 1))


class Cast(_TrainRule):
    """`_to_copy` of a value to another dtype, as torch converts it: int64 to uint8 its low byte (`I64ToU8_v1`), uint8 to int64
    zero-extended (`U8ToI64_v1`), int64 to f32 rounded to nearest even (`I64ToF32Rn_v1`), int64 to bool x != 0 (`_nonzero`), bool
    to f32 and to int64 1 or 0 (`SelectF32_v1`, `SelectI64_v1`), each the same on the CPU and the GPU; at a compute capability
    `cc` the GPU's f32 to bf16 (`F2fpBf16_v1`), f32 to f16 (`F32ToF16Rn_v1`) and f16 to f32 (`F16ToF32_v1`), whose NaN words are
    the GPU's, and f32 to int64 truncated toward zero (`F32ToI64Rz_v1`, saturated, a NaN 0) and to uint8 through int64 (its low
    byte, `I64ToU8_v1`), as c10 converts. `training_rules` places it first at `cc`, ahead of the default rules' f32 to bf16 (torch
    CPU's rounding), which it leaves to them without `cc`. Refused without `cc`: the f16 conversions (torch CPU's keep a NaN's
    payload in their vector body and not in their scalar tail) and f32 to int64 and uint8 (torch CPU's is INT64_MIN for a NaN and
    from 2^63 up)."""
    info = _info("Cast", "aten._to_copy of a value between int64, uint8, bool and f32; at cc f32 to bf16, f32 to and from f16 and "
                 "f32 to int64 and uint8", "the operand is a value", "a dtype pair this rule lowers")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten._to_copy.default}

    def applies(self, node, ctx):
        a = node.args[0]
        if not isinstance(ctx.val(a), Coll):
            return "the operand is not a value"
        pair = (ctx.dtype_of(a), ctx.dtype_of(node))
        if _lowers(pair, self.cc):
            return True
        if pair == (torch.float32, torch.bfloat16):
            return "an f32 to bf16 conversion without a compute capability is the default rules' (torch CPU's rounding)"
        if pair in _GPU:
            why = ("torch CPU's f16 conversions keep a NaN's payload in their vector body and not in their scalar tail"
                   if torch.float16 in pair else "torch CPU's truncation is INT64_MIN for a NaN and from 2^63 up")
            return f"numerics_unregistered: {why}; the GPU's are lowered at a compute capability (training_rules(cc=...))"
        return f"not a conversion this rule lowers ({pair[0]} to {pair[1]})"

    def emit(self, node, ctx):
        a = node.args[0]
        src, dst = ctx.dtype_of(a), ctx.dtype_of(node)
        y, fn = _convert(ctx, node, ctx.coll(a), src, dst)
        ctx.define(node, y)
        ctx.record(self, node, {"from": str(src), "to": str(dst)}, emitted=f"{fn} x{math.prod(ctx.shape_of_node(node))}")


def _convert(ctx, node, x: Coll, src, dst) -> tuple[Coll, str]:
    """(the value x converted from `src` to `dst` as torch's kernels convert it, its Definitions): bool by `_from_bits`, int64 to
    bool by `_nonzero`, bf16 to f32 widened exactly (`Bf16ToF32_v1`), or a pair of `_EXACT` or `_GPU` (through int64 between two
    Definitions)."""
    if src == torch.bool:
        return _from_bits(ctx, node, x, dst), "SelectF32_v1" if dst == torch.float32 else "SelectI64_v1"
    if (src, dst) == (torch.int64, torch.bool):
        return _nonzero(ctx, node, x), "I64Le_v1 x2 then BitAnd_v1 then BitNot_v1"
    f = P.Bf16ToF32 if (src, dst) == (torch.bfloat16, torch.float32) else _EXACT.get((src, dst)) or _GPU[(src, dst)]
    fs = f if isinstance(f, tuple) else (f,)
    for g, w in zip(fs, [64] * (len(fs) - 1) + [WIDTH_OF_DTYPE[dst]], strict=True):
        x = _converted(ctx, node, x, g, w)
    return x, " then ".join(g.id for g in fs)


class Copy(_TrainRule):
    """`copy(self, src)` (a functionalized `copy_`): `src`'s values in `self`'s shape, broadcast, whatever `self` holds (an
    allocation such as autograd's `new_empty_strided` for a gradient, which stays unread, or a value); `src` a value or a static
    float tensor of `self`'s dtype, or a value of another dtype converted as torch's copy kernel converts it, with `_to_copy`'s
    casts: bf16 to f32 exactly (`Bf16ToF32_v1`), `Cast`'s exact pairs, and at a compute capability `cc` the GPU's to bf16, to and
    from f16 and from f32 to int64 and uint8. Refused without `cc`: those."""
    info = _info("Copy", "aten.copy of a value or a static float tensor into a tensor", "src has self's dtype, or is a value "
                 "converted as `_to_copy` converts it", "src broadcasts to self's shape", "self's values are not read")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.copy.default}

    def applies(self, node, ctx):
        src = node.args[1]
        if not isinstance(src, torch.fx.Node) or not isinstance(ctx.val(src), (Coll, StaticFloat)):
            return "src is not a value or a static float tensor"
        pair = (ctx.dtype_of(src), ctx.dtype_of(node))
        if pair[0] == pair[1]:
            return True
        if isinstance(ctx.val(src), Coll) and (pair == (torch.bfloat16, torch.float32) or _lowers(pair, self.cc)):
            return True
        return f"numerics_unregistered: a conversion from {pair[0]} to {pair[1]} (lowered at a compute capability, or not at all)"

    def emit(self, node, ctx):
        src, S = node.args[1], tuple(ctx.shape_of_node(node))
        v, fn = ctx.val(src), None
        if isinstance(v, StaticFloat):
            shape, out = tuple(v.tensor.shape), StaticFloat(v.tensor.broadcast_to(S).contiguous())
        else:
            if ctx.dtype_of(src) != ctx.dtype_of(node):
                v, fn = _convert(ctx, node, v, ctx.dtype_of(src), ctx.dtype_of(node))
            shape = tuple(shape_of(v.type)[0])
            out = v if shape == S else _view_expand(v, list(S), node.name)
        ctx.define(node, out)
        ctx.record(self, node, {"shape": list(S), "src_shape": list(shape), "conversion": fn},
                   emitted=f"{fn} x{math.prod(shape)}" if fn else "references")


def _dims(node, r: int) -> list[int]:
    given = node.args[1] if len(node.args) > 1 else None
    return sorted({d % r for d in given}) if given else list(range(r))


def _first_axes(ctx, node, x: Coll, dims: list[int]) -> tuple[Coll, int, int]:
    """x with the axes `dims` moved first (by reference), flat: (words, terms, rows), term j's word r at j * rows + r."""
    S, w = shape_of(x.type)
    keep = [d for d in range(len(S)) if d not in dims]
    perm = tuple(dims + keep)
    if perm != tuple(range(len(S))):
        x = _view_permute(x, perm, node.name)
    n, rows = math.prod(S[d] for d in dims), math.prod(S[d] for d in keep)
    return x.reshape(Array(n * rows, Value(w))), n, rows


class IntSum(_TrainRule):
    """`sum` of a bool or int64 value over `dim` (every axis for `.default`, None or []) to int64, as torch sums it: each bool 1
    or 0 (`SelectI64_v1`) and the terms added pairwise (`I64Add_v1`), which is torch's sum in any order (modulo 2^64)."""
    info = _info("IntSum", "aten.sum.dim_IntList / sum.default of a bool or int64 value", "the result is int64",
                 "the operand has an axis")

    def targets(self):
        return {aten.sum.dim_IntList, aten.sum.default}

    def applies(self, node, ctx):
        a = node.args[0]
        if not isinstance(ctx.val(a), Coll) or ctx.dtype_of(a) not in (torch.bool, torch.int64):
            return "the operand is not a bool or int64 value"
        if ctx.dtype_of(node) != torch.int64:
            return f"numerics_unregistered: result dtype {ctx.dtype_of(node)}"
        return True if _shape(ctx, a) else "a 0-d operand"

    def emit(self, node, ctx):
        a = node.args[0]
        x = ctx.coll(a)
        if ctx.dtype_of(a) == torch.bool:
            x = _from_bits(ctx, node, x, torch.int64)
        v, n, rows = _first_axes(ctx, node, x, _dims(node, len(shape_of(x.type)[0])))
        s = _tree(ctx, node, W.I64Add, v, n, rows, 64)
        ctx.define(node, s.reshape(_tensor_type(ctx.shape_of_node(node), 64)))
        ctx.record(self, node, {"terms": n, "rows": rows}, emitted=f"I64Add_v1 x{(n - 1) * rows}")


class AMax(_TrainRule):
    """`amax` of an f32 value over `dim` (every axis for none), kept or not: `F32MaxRef_v1` folded pairwise. A maximum is one word
    in any order, so this is torch's CPU and CUDA kernels' value, except where a term is a NaN (here the canonical 0x7FC00000,
    there a NaN term's word) or the maximum is zero with both zeros among the terms (either, by the order)."""
    info = _info("AMax", "aten.amax of an f32 value", "the operand is an f32 value with an axis")

    def targets(self):
        return {aten.amax.default}

    def applies(self, node, ctx):
        a = node.args[0]
        if not isinstance(ctx.val(a), Coll) or ctx.dtype_of(a) != torch.float32:
            return "numerics_unregistered: the operand is not an f32 value"
        return True if _shape(ctx, a) else "a 0-d operand"

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        v, n, rows = _first_axes(ctx, node, x, _dims(node, len(shape_of(x.type)[0])))
        m = _tree(ctx, node, R.F32MaxRef, v, n, rows, 32)
        ctx.define(node, m.reshape(_tensor_type(ctx.shape_of_node(node), 32)))
        ctx.record(self, node, {"terms": n, "rows": rows}, emitted=f"F32MaxRef_v1 x{(n - 1) * rows}")


class Index(_TrainRule):
    """`index.Tensor(table, [i])` of a 1-D f32 table (a codebook), static or a value, by an int64 or int32 value: each index
    narrowed to i32 (`I64ToI32_v1`), wrapped once if negative (`I32Le_v1`, `I32Add_v1`, `SelectI32_v1`: torch reads -T <= i < 0
    as i + T) and its entry read from the table's words (`GatherI32x{T}_v1`). An index out of range reads 0xFFFFFFFF, a NaN
    (torch raises)."""
    info = _info("Index", "aten.index.Tensor of a 1-D f32 table, static or a value, by one integer value", "one index tensor, an "
                 "int64 or int32 value", "an index out of range reads 0xFFFFFFFF (torch raises)")

    def targets(self):
        return {aten.index.Tensor}

    def applies(self, node, ctx):
        tab, idx = node.args[:2]
        t = ctx.val(tab)
        static = isinstance(t, StaticFloat) and t.tensor.dim() == 1
        value = isinstance(t, Coll) and ctx.dtype_of(tab) == torch.float32 and len(_shape(ctx, tab)) == 1
        if not (static or value) or ctx.dtype_of(node) != torch.float32:
            return "the table is not a 1-D f32 tensor, static or a value"
        if len(idx) != 1 or not isinstance(idx[0], torch.fx.Node) or not isinstance(ctx.val(idx[0]), Coll):
            return "not one index tensor that is a value"
        if ctx.dtype_of(idx[0]) not in (torch.int64, torch.int32):
            return f"an index of dtype {ctx.dtype_of(idx[0])}"
        return True

    def emit(self, node, ctx):
        t = ctx.val(node.args[0])
        if isinstance(t, StaticFloat):
            T, tab = t.tensor.numel(), _static_coll(ctx, node, t.tensor)
        else:
            T = _shape(ctx, node.args[0])[0]
            tab = ctx.coll(node.args[0]).reshape(Array(T, I32))
        i = _tokens(ctx, node, node.args[1][0])
        neg = _batch(ctx, node, SC.I32Le, i, _const(ctx, node, -1), axes=(0, None))
        up = _batch(ctx, node, SC.I32Add, i, _const(ctx, node, T), axes=(0, None))
        j = _batch(ctx, node, SC.SelectI32, neg, up, i, axes=(0, 0, 0))
        y = _batch(ctx, node, gather_i32(T), j, tab, axes=(0, None))
        ctx.define(node, y.reshape(_tensor_type(ctx.shape_of_node(node), 32)))
        ctx.record(self, node, {"table": T, "indices": len(i)}, emitted=f"GatherI32x{T}_v1 x{len(i)}")


def _kth_args(node) -> tuple[int, int, bool]:
    k = node.args[1]
    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    keepdim = node.args[3] if len(node.args) > 3 else node.kwargs.get("keepdim", False)
    return k, dim, keepdim


class KthValue(_TrainRule):
    """`kthvalue(x, k, dim, keepdim)` of an f32 value at a compute capability `cc`, as torch's CUDA `gatherKthValue` selects it:
    `radixSelect` over `TopKTypeConfig<float>`'s order keys (`F32OrderKey_v1`: -0 before +0, every NaN after +inf). Per slice,
    K, the k-th smallest key, by a binary search over its 32 bits (each step `I32Le_v1` of the keys with their top bit flipped
    against the candidate, a count by `SelectI32_v1` and `I32Add_v1`, `I32Le_v1` against k); the value `OrderKeyF32_v1(K)`, but
    a NaN that is its slice's only one and its k-th is its own word (radixSelect's unique match, `findPattern`), so two NaNs
    give 0x7FFFFFFF. Only the values are lowered: refused if the indices are read, and without `cc` (torch CPU's quickselect
    returns whichever of -0 and +0 lands at k)."""
    info = _info("KthValue", "aten.kthvalue of an f32 value at a compute capability", "1 <= k <= the slice's length",
                 "the indices are not read", "words: F32OrderKey_v1, OrderKeyF32_v1, I32Le_v1, I32Eq_v1, I32Add_v1, SelectI32_v1, "
                 "BitAnd_v1, SelectF32_v1")

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return {aten.kthvalue.default}

    def applies(self, node, ctx):
        if self.cc is None:
            return ("numerics_unregistered: torch CPU's kthvalue (a quickselect) returns whichever of -0 and +0 lands at k; the "
                    "GPU's radix select is lowered at a compute capability (training_rules(cc=...))")
        a = node.args[0]
        if why := _operand_ok(ctx, a):
            return why
        if not isinstance(ctx.val(a), Coll) or not _shape(ctx, a):
            return "the operand is not a value of at least one axis"
        k, dim, _ = _kth_args(node)
        S = _shape(ctx, a)
        if type(k) is not int or not 1 <= k <= S[dim % len(S)]:
            return f"k = {k!r} outside 1..{S[dim % len(S)]}"
        if any(u.target is not operator.getitem or (u.args[1] != 0 and u.users) for u in node.users):
            return "numerics_unregistered: the indices are read (only the values are lowered)"
        return True

    def emit(self, node, ctx):
        k, dim, keepdim = _kth_args(node)
        x = ctx.coll(node.args[0])
        S = list(shape_of(x.type)[0])
        d = dim % len(S)
        xs, N, rows = _first_axes(ctx, node, x, [d])
        keys = _batch(ctx, node, W.F32OrderKey, xs, axes=(0,))
        s = _batch(ctx, node, SC.I32Add, keys, _const(ctx, node, 0x80000000), axes=(0, None))
        one, zero = _const(ctx, node, 1), _const(ctx, node, 0)

        def over_terms(v: Coll) -> Coll:
            r = v.refs
            return Coll(Array(N * rows, I32), Strided(r.space, r.idx, r.base, rows, 0, 1, N * rows))

        def count(bits: Coll) -> Coll:
            return _tree(ctx, node, SC.I32Add, _batch(ctx, node, SC.SelectI32, bits, one, zero, axes=(0, None, None)), N, rows, 32)

        p = Coll(Array(rows, I32), broadcast(*_ref(zero), rows))
        for b in range(31, -1, -1):
            mid = _batch(ctx, node, SC.I32Add, p, _const(ctx, node, ((1 << b) - 1 + 0x80000000) & M32), axes=(0, None))
            cnt = count(_batch(ctx, node, SC.I32Le, s, over_terms(mid), axes=(0, 0)))
            low = _batch(ctx, node, SC.I32Le, _const(ctx, node, k), cnt, axes=(None, 0))
            up = _batch(ctx, node, SC.I32Add, p, _const(ctx, node, 1 << b), axes=(0, None))
            p = _batch(ctx, node, SC.SelectI32, low, p, up, axes=(0, 0, 0))
        nan_key = _batch(ctx, node, SC.I32Eq, keys, _const(ctx, node, M32), axes=(0, None))
        nan_word = _tree(ctx, node, SC.I32Add, _batch(ctx, node, SC.SelectI32, nan_key, xs, zero, axes=(0, 0, None)), N, rows, 32)
        unique = _batch(ctx, node, SC.BitAnd, _batch(ctx, node, SC.I32Eq, p, _const(ctx, node, M32), axes=(0, None)),
                        _batch(ctx, node, SC.I32Eq, cnt, _const(ctx, node, N - 1), axes=(0, None)), axes=(0, 0))
        y = _batch(ctx, node, SC.SelectF32, unique, nan_word, _batch(ctx, node, W.OrderKeyF32, p, axes=(0,)), axes=(0, 0, 0))
        out = S[:d] + [1] * keepdim + S[d + 1:]
        ctx.define(node, [y.reshape(_tensor_type(out, 32)), None])
        ctx.record(self, node, {"k": k, "slice": N, "slices": rows}, emitted=f"radix select of {N} keys x{rows} (32 steps)")


__all__ = ["Compare", "Logic", "IntOps", "Where", "Clamp", "Cast", "Copy", "IntSum", "AMax", "Index", "KthValue"]
