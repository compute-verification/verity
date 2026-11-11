"""The training rules' views and constants (`rules.training`).

A step's tensors are large and its views compose (`t(t(W))`, the transpose of a concatenation): the default views enumerate a view
of a value that is not one run leaf by leaf, up to a limit. `BoxPermute` represents such a value as boxes, each a region of the
tensor mapped affinely onto one producer's leaves, permutes the boxes, and writes the result back as strided references
(`_view_permute` and `_view_expand` do the same for the rules that read a permuted or broadcast operand). `StaticFloats` keeps a
float tensor the step fixes without reading a value (a literal, the causal mask, a fill) as the tensor torch computes on the CPU,
at a compute capability with a division by a number as the GPU's product by its reciprocal (`StaticFloat`); it becomes constant
gates only where an op reads it or the step returns it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from verity.definitions.ops.prims import const
from verity.circuits.refs import Affine, Coll, Concat, Strided, broadcast, concat
from verity.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import StaticIndex, _LiftedConstant, refuse
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.rules.base import RuleInfo, ViewRule
from verity_vllm.program.frontend.rules.patterns import _constant_tensor
from verity_vllm.program.frontend.strided_views import (_EXPLICIT_LIMIT, _constant_run, _expand, _permute, _row_major, _strided_refs,
                                                        _tensor_type)
from verity_vllm.program.frontend.value_types import WIDTH_OF_DTYPE, shape_of

aten = torch.ops.aten
_PARTS = 1 << 18  # reference parts a view or a constant tensor may take
_INT_VIEW = {8: torch.uint8, 16: torch.int16, 32: torch.int32, 64: torch.int64}
_REF = "reference arithmetic: the registered Definitions' (ref_prims); element-wise ops are torch CPU's f32 operations"


def _info(name: str, applies_to: str, *pre: str, kind: str = "op-binding") -> RuleInfo:
    return RuleInfo(f"{name}[train]", 1, kind, applies_to=applies_to, preconditions=pre, evidence_domain=_REF)



# ---------------------------------------------------------------------------------------------------------
# boxes: a tensor's references as regions mapped affinely onto one producer's leaves
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Box:
    """Coordinates lo <= x < hi of a tensor reference leaf `leaf0 + sum((x - lo) * st)` of (space, idx)."""
    space: str
    idx: int
    lo: tuple
    hi: tuple
    leaf0: int
    st: tuple


def _box(space, idx, lo, hi, leaf0, st) -> _Box:
    return _Box(space, idx, tuple(lo), tuple(hi), int(leaf0), tuple(s if h - l > 1 else 0 for l, h, s in zip(lo, hi, st)))


def _spend(budget: list[int], n: int, where: str) -> None:
    budget[0] -= n
    if budget[0] < 0:
        raise FrontendError("E_UNSUPPORTED_VIEW", f"the view needs more than {_PARTS} reference parts", where)


def _range_boxes(a: int, b: int, shape: tuple) -> list[tuple[tuple, tuple]]:
    """The boxes (lo, hi) whose union is the row-major range [a, b) of `shape`: at most two per axis."""
    if a >= b:
        return []
    if not shape:
        return [((), ())]
    rest = tuple(shape[1:])
    inner = math.prod(rest)
    q0, r0 = divmod(a, inner)
    q1, r1 = divmod(b, inner)
    if q0 == q1:
        return [((q0,) + lo, (q0 + 1,) + hi) for lo, hi in _range_boxes(r0, r1, rest)]
    out = []
    if r0:
        out += [((q0,) + lo, (q0 + 1,) + hi) for lo, hi in _range_boxes(r0, inner, rest)]
        q0 += 1
    if q1 > q0:
        out.append(((q0,) + (0,) * len(rest), (q1,) + rest))
    if r1:
        out += [((q1,) + lo, (q1 + 1,) + hi) for lo, hi in _range_boxes(0, r1, rest)]
    return out


def _pieces(r):
    if isinstance(r, Concat):
        for p in r.parts:
            yield from _pieces(p)
    elif r.count:
        yield r


def _boxes(refs, shape, where: str) -> list[_Box]:
    """`refs`, the row-major references of a tensor of `shape`, as merged boxes."""
    shape = tuple(shape)
    rm = _row_major(shape)
    out: list[_Box] = []
    budget = [_PARTS]

    def run(space, idx, leaf, factor, a, b):
        for lo, hi in _range_boxes(a, b, shape):
            f = sum(l * s for l, s in zip(lo, rm))
            out.append(_box(space, idx, lo, hi, leaf + (f - a) * factor, [factor * s for s in rm]))

    pos = 0
    for p in _pieces(refs):
        n = p.count
        if isinstance(p, Affine):
            run(p.space, p.idx, p.base, 1, pos, pos + n)
        elif isinstance(p, Strided):
            I, O, s = p.inner, p.outer_stride, p.inner_stride
            if I == 1:
                run(p.space, p.idx, p.base, O, pos, pos + n)
            elif n <= I or O == I * s:
                run(p.space, p.idx, p.base, s, pos, pos + n)
            else:
                k = next((k for k in range(len(shape)) if math.prod(shape[k:]) == I), None)
                if k is not None and pos % I == 0 and n % I == 0:
                    lead, tail = shape[:k], shape[k:]
                    rl, rt = _row_major(lead), _row_major(tail)
                    for lo, hi in _range_boxes(pos // I, (pos + n) // I, lead):
                        f = sum(l * st for l, st in zip(lo, rl))
                        out.append(_box(p.space, p.idx, lo + (0,) * len(tail), hi + tail, p.base + (f - pos // I) * O,
                                        [O * st for st in rl] + [s * st for st in rt]))
                else:
                    rows = -(-n // I)
                    _spend(budget, rows, where)
                    for j in range(rows):
                        run(p.space, p.idx, p.base + j * O, s, pos + j * I, pos + min((j + 1) * I, n))
        else:
            _spend(budget, n, where)
            for i, (space, idx, leaf) in enumerate(p):
                run(space, idx, leaf, 0, pos + i, pos + i + 1)
        _spend(budget, 1, where)
        pos += n
    return _merge(out)


def _merge(boxes: list[_Box]) -> list[_Box]:
    """Join boxes adjacent along an axis whose leaves continue one affine map (the union is the same set of references)."""
    r = len(boxes[0].lo) if boxes else 0
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for ax in reversed(range(r)):
            def key(b, ax=ax):
                return (b.space, b.idx, b.lo[:ax] + b.lo[ax + 1:], b.hi[:ax] + b.hi[ax + 1:], b.st[:ax] + b.st[ax + 1:])
            boxes.sort(key=lambda b: key(b) + (b.lo[ax],))
            out = [boxes[0]]
            for b in boxes[1:]:
                a = out[-1]
                if a.hi[ax] == b.lo[ax] and key(a) == key(b):
                    la, lb, d = a.hi[ax] - a.lo[ax], b.hi[ax] - b.lo[ax], b.leaf0 - a.leaf0
                    if d % la == 0 and (la == 1 or a.st[ax] == d // la) and (lb == 1 or b.st[ax] == d // la):
                        out[-1] = _box(a.space, a.idx, a.lo, a.hi[:ax] + (b.hi[ax],) + a.hi[ax + 1:], a.leaf0,
                                       a.st[:ax] + (d // la,) + a.st[ax + 1:])
                        changed = True
                        continue
                out.append(b)
            boxes = out
    return boxes


def _refs(shape, boxes: list[_Box], where: str, budget: list[int] | None = None):
    """The row-major references of a tensor of `shape` covered by `boxes`."""
    budget = budget if budget is not None else [_PARTS]
    if len(boxes) == 1:
        b = boxes[0]
        _spend(budget, 1, where)
        return _strided_refs(b.space, b.idx, b.leaf0, [h - l for l, h in zip(b.lo, b.hi)], b.st, where)
    order = sorted(boxes, key=lambda b: b.lo[0])
    cuts = sorted({b.lo[0] for b in boxes} | {b.hi[0] for b in boxes})
    parts, active, i = [], [], 0
    for c0, c1 in zip(cuts, cuts[1:]):
        active = [b for b in active if b.hi[0] > c0]
        while i < len(order) and order[i].lo[0] <= c0:
            active.append(order[i])
            i += 1
        if len(active) == 1:
            b = active[0]
            _spend(budget, 1, where)
            parts.append(_strided_refs(b.space, b.idx, b.leaf0 + (c0 - b.lo[0]) * b.st[0],
                                       [c1 - c0] + [h - l for l, h in zip(b.lo[1:], b.hi[1:])], b.st, where))
            continue
        for c in range(c0, c1):
            parts.append(_refs(shape[1:], [_Box(b.space, b.idx, b.lo[1:], b.hi[1:], b.leaf0 + (c - b.lo[0]) * b.st[0], b.st[1:])
                                           for b in active], where, budget))
    return concat(parts)


def _view_permute(x: Coll, dims, where: str) -> Coll:
    shape, w = shape_of(x.type)
    if isinstance(x.refs, Affine) or _constant_run(x.refs) is not None or x.type.leaves <= _EXPLICIT_LIMIT:
        return _permute(x, dims, where)
    boxes = [_Box(b.space, b.idx, tuple(b.lo[d] for d in dims), tuple(b.hi[d] for d in dims), b.leaf0, tuple(b.st[d] for d in dims))
             for b in _boxes(x.refs, shape, where)]
    out = [shape[d] for d in dims]
    return Coll(_tensor_type(out, w), _refs(out, _merge(boxes), where))


def _view_expand(x: Coll, sizes, where: str) -> Coll:
    shape, w = shape_of(x.type)
    sizes = [n if m == -1 else m for n, m in zip([1] * (len(sizes) - len(shape)) + list(shape), sizes)]
    if isinstance(x.refs, Affine) or _constant_run(x.refs) is not None or math.prod(sizes) <= _EXPLICIT_LIMIT:
        return _expand(x, sizes, where)
    lead = len(sizes) - len(shape)
    boxes = []
    for b in _boxes(x.refs, shape, where):
        lo, hi, st = [0] * lead, list(sizes[:lead]), [0] * lead
        for a, (n, m) in enumerate(zip(shape, sizes[lead:])):
            lo, hi, st = lo + [b.lo[a] if n == m else 0], hi + [b.hi[a] if n == m else m], st + [b.st[a] if n == m else 0]
        boxes.append(_box(b.space, b.idx, lo, hi, b.leaf0, st))
    return Coll(_tensor_type(sizes, w), _refs(sizes, _merge(boxes), where))


def _permute_dims(node, rank: int) -> tuple[int, ...]:
    t = node.target
    if t is aten.permute.default:
        return tuple(d % rank for d in node.args[1])
    if t is aten.t.default:
        return (1, 0) if rank == 2 else tuple(range(rank))
    d0, d1 = node.args[1] % rank, node.args[2] % rank
    dims = list(range(rank))
    dims[d0], dims[d1] = d1, d0
    return tuple(dims)


def _const(ctx, node, bits: int, w: int = 32) -> Coll:
    """One constant gate per distinct word in the derivation (a node a refused rule removed is made again)."""
    cache = ctx.state.setdefault("train_consts", {})
    bits &= (1 << w) - 1
    fn = const(w, bits)
    k = cache.get((w, bits))
    if k is None or k >= len(ctx.B.nodes) or ctx.B.nodes[k].fn is not fn:
        ctx.call(fn, node=node)
        k = cache[(w, bits)] = len(ctx.B.nodes) - 1
    return Coll(Value(w), Affine("n", k, 0, 1))


def _words(t: torch.Tensor, w: int) -> np.ndarray:
    """A static tensor's words, `w` bits wide, as unsigned integers (a bool tensor's 1 or 0)."""
    t = t.detach().contiguous().reshape(-1)
    t = t.to(torch.uint8) if w == 1 else t.view(_INT_VIEW[w])
    return t.numpy().astype(np.uint64) & np.uint64((1 << w) - 1)


def _static_words(ctx, node, bits: np.ndarray, w: int = 32) -> Coll:
    """Static words of `w` bits (unsigned) as references to one constant gate per distinct word, run by run."""
    bits = np.asarray(bits).reshape(-1)
    n = bits.size
    starts = np.concatenate([[0], np.flatnonzero(np.diff(bits)) + 1])
    if starts.size > _PARTS:
        refuse("shape", node, ctx.module_path(node), f"a constant tensor of {starts.size} runs (limit {_PARTS})")
    ends = np.concatenate([starts[1:], [n]])
    parts = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        c = _const(ctx, node, int(bits[a]), w)
        parts.append(broadcast(c.refs.space, c.refs.idx, c.refs.base, b - a))
    return Coll(Array(n, Value(w)), concat(parts))


@dataclass(eq=False)
class StaticFloat:
    """A float tensor the step fixes without reading a value (a literal, the causal mask, a fill), as torch computes it on the CPU;
    it becomes constant gates only where an op reads it or the step returns it. A bf16 one is held as the f32 tensor of its
    values."""
    tensor: torch.Tensor

    def __repr__(self) -> str:
        return f"StaticFloat{list(self.tensor.shape)}"

    def returned(self, ctx, node) -> Coll:
        """The tensor as a value the step returns (a fill of a mutated input, say): its words in `node`'s dtype, a constant gate a
        distinct word. A NaN is refused, as where an op reads one: torch CPU's NaN word is not the GPU's."""
        if bool(torch.isnan(self.tensor).any()):
            refuse("numerics_unregistered", node, ctx.module_path(node), "a returned static NaN (torch CPU's NaN word is not the GPU's)")
        dt = ctx.dtype_of(node)
        w = WIDTH_OF_DTYPE[dt]
        return _static_words(ctx, node, _words(self.tensor.to(dt), w), w).reshape(_tensor_type(list(self.tensor.shape), w))



# ---------------------------------------------------------------------------------------------------------
# views and constants
# ---------------------------------------------------------------------------------------------------------


class BoxPermute(ViewRule):
    """`permute` / `t` / `transpose.int` of a value whose references are not one run, past the default views' leaf-by-leaf limit."""
    info = _info("BoxPermute", "aten.permute / t / transpose.int of a value of more than `min_leaves` leaves that is not one run",
                 "the operand's references are not one run nor one repeated reference", "no gates: references only", kind="view")

    def __init__(self, min_leaves: int = _EXPLICIT_LIMIT) -> None:
        self.min_leaves = min_leaves

    def targets(self):
        return {aten.permute.default, aten.t.default, aten.transpose.int}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return "the operand is not a value"
        if isinstance(v.refs, Affine) or _constant_run(v.refs) is not None or v.type.leaves <= self.min_leaves:
            return "the default view is O(1) or small"
        return True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        shape = shape_of(x.type)[0]
        dims = _permute_dims(node, len(shape))
        if dims == tuple(range(len(shape))):
            out = x
        else:
            boxes = [_Box(b.space, b.idx, tuple(b.lo[d] for d in dims), tuple(b.hi[d] for d in dims), b.leaf0,
                          tuple(b.st[d] for d in dims)) for b in _boxes(x.refs, shape, node.name)]
            out_shape = [shape[d] for d in dims]
            out = Coll(_tensor_type(out_shape, shape_of(x.type)[1]), _refs(out_shape, _merge(boxes), node.name))
        ctx.define(node, out, view_of=node.args[0])
        ctx.record(self, node, {"dims": list(dims), "parts": len(out.refs.parts) if isinstance(out.refs, Concat) else 1})


_VIEWS = {aten.view.default, aten._unsafe_view.default, aten.reshape.default, aten.expand.default, aten.permute.default, aten.t.default,
          aten.transpose.int, aten.slice.Tensor, aten.select.int, aten.clone.default, aten.unsqueeze.default, aten.squeeze.dim,
          aten.alias.default}
_MAKERS = {aten.scalar_tensor.default, aten.ones_like.default, aten.zeros_like.default, aten.full_like.default, aten.full.default,
           aten.ones.default, aten.zeros.default, aten.fill.Scalar, aten.where.self}
_DIV = {aten.div.Tensor, aten.div.Scalar}
_ARITH = {aten.add.Tensor, aten.add.Scalar, aten.sub.Tensor, aten.sub.Scalar, aten.mul.Tensor, aten.mul.Scalar} | _DIV
_ARANGE = {aten.arange.default, aten.arange.start, aten.arange.start_step}
_SCATTER = {aten.select_scatter.default, aten.slice_scatter.default}
_FLOATS = (torch.float32, torch.bfloat16, torch.float16)


class StaticFloats(ViewRule):
    """A float tensor fixed without reading a value: `scalar_tensor`, `ones_like` / `zeros_like` / `full_like` / `fill.Scalar` (of
    any tensor: only its shape is read), `full` / `ones` / `zeros`, an f32 `arange` of integers within 2^24 (every one an f32, so
    exact however a kernel accumulates it), `where.self` of a static condition and static values, a tensor constant the graph
    holds (`lift_fresh_copy` of a table; an integral one is a `StaticIndex`), a static tensor converted to a float dtype
    (`_to_copy`), an f32 `add`, `sub`, `mul` or `div` of static float tensors and numbers, `select_scatter` / `slice_scatter` of
    static float tensors, and views of these, computed by torch on the CPU (`StaticFloat`). A conversion and an f32 add, sub, mul
    or true division are torch CPU's, which are the GPU's on every value but a NaN, so a static NaN converted and a static NaN
    result are refused; a bf16 or f16 result of arithmetic is refused too (torch CPU rounds a number in a bf16 add to bf16 first,
    the GPU doesn't). At a compute capability `cc` a division
    by a number is the product by its reciprocal `RN(1 / f32(s))`, as `Binary` lowers a value's, and a division by a 0-d static
    tensor is refused (the GPU multiplies by a CPU scalar's reciprocal and divides by a device tensor)."""
    info = _info("StaticFloats", "aten float makers, fills, f32 arange, where.self of static operands, tensor constants, "
                 "conversions, f32 add / sub / mul / div and select_scatter / slice_scatter of static operands and views of static "
                 "float tensors", "the result is float32, bfloat16 or float16", "every operand read for its value is static",
                 "arange: f32, integer bounds and step, every value within 2^24", "a conversion of no NaN", "arithmetic: an f32 "
                 "result and no NaN", "div: no rounding mode; at cc no 0-d static divisor", "no gates: constants are made where read",
                 kind="view")
    accepts_constants = True

    def __init__(self, cc: tuple[int, int] | None = None):
        self.cc = cc

    def targets(self):
        return _MAKERS | _VIEWS | _ARITH | _ARANGE | _SCATTER | {aten.lift_fresh_copy.default, aten._to_copy.default}

    def _static(self, ctx, a):
        if isinstance(a, torch.fx.Node):
            v = ctx.val(a)
            if isinstance(v, StaticFloat):
                return v.tensor
            if isinstance(v, StaticIndex):
                return v.tensor
            return None
        return a

    def _arith(self, ctx, node) -> torch.Tensor:
        args = [self._static(ctx, a) if isinstance(a, torch.fx.Node) else a for a in node.args]
        if self.cc is not None and node.target in _DIV and isinstance(args[1], (int, float)):
            inv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(args[1], dtype=torch.float32)
            return torch.as_tensor(args[0], dtype=torch.float32) * inv
        return node.target(*args, **node.kwargs)

    def applies(self, node, ctx):
        t = node.target
        if t is aten.lift_fresh_copy.default:
            c = ctx.val(node.args[0])
            if not isinstance(c, _LiftedConstant) or _constant_tensor(ctx, c) is None:
                return "not a tensor constant whose values the graph holds"
            return True
        if ctx.dtype_of(node) not in _FLOATS:
            return "the result is not float32, bfloat16 or float16"
        if t in _VIEWS:
            return True if isinstance(ctx.val(node.args[0]), StaticFloat) else "the operand is not a static float tensor"
        if t in _ARANGE:
            if ctx.dtype_of(node) != torch.float32 or not all(type(a) is int for a in node.args):
                return "numerics_unregistered: a bf16 or f16 arange, or a fractional bound or step (kernels accumulate it apart)"
            v = torch.arange(*node.args, dtype=torch.float64)
            return True if not v.numel() or float(v.abs().max()) <= 2 ** 24 else "numerics_unregistered: an integer past 2^24"
        if t in _SCATTER and not all(isinstance(ctx.val(a), StaticFloat) for a in node.args[:2]):
            return "an operand is not a static float tensor"
        if t is aten.where.self:
            if not all(isinstance(self._static(ctx, a), (torch.Tensor, int, float)) for a in node.args[:3]):
                return "where.self of a value"
        if t is aten._to_copy.default:
            src = self._static(ctx, node.args[0])
            if not isinstance(src, torch.Tensor):
                return "the operand is not static"
            if ctx.dtype_of(node.args[0]) != ctx.dtype_of(node) and src.is_floating_point() and bool(torch.isnan(src).any()):
                return "numerics_unregistered: a static NaN converted (torch CPU's NaN word is not the GPU's)"
        if t in _ARITH:
            if not all(isinstance(ctx.val(a), StaticFloat) if isinstance(a, torch.fx.Node)
                       else isinstance(a, (int, float)) and not isinstance(a, bool) for a in node.args[:2]):
                return "an operand is a value (or not a static float tensor or a number)"
            if ctx.dtype_of(node) != torch.float32:
                return "numerics_unregistered: a static bf16 or f16 result (torch CPU rounds a number to it first, the GPU doesn't)"
            if t in _DIV:
                if node.kwargs.get("rounding_mode") is not None:
                    return "numerics_unregistered: a rounding mode"
                b = node.args[1]
                if self.cc is not None and isinstance(b, torch.fx.Node) and ctx.val(b).tensor.dim() == 0:
                    return ("numerics_unregistered: a division by a 0-d static tensor on the GPU (a CPU scalar's reciprocal or a "
                            "device tensor's quotient)")
            if bool(torch.isnan(self._arith(ctx, node)).any()):
                return "numerics_unregistered: a static NaN result (torch CPU's NaN word is not the GPU's)"
        return True

    def emit(self, node, ctx):
        t, S = node.target, ctx.shape_of_node(node)
        if t is aten.lift_fresh_copy.default:
            c = _constant_tensor(ctx, ctx.val(node.args[0])).detach().cpu().clone()
            if not c.is_floating_point():
                ctx.define(node, StaticIndex(c))
                ctx.record(self, node, {"shape": list(c.shape), "dtype": str(c.dtype)}, emitted="static")
                return
            out = c
        elif t is aten._to_copy.default:
            out = self._static(ctx, node.args[0])
        elif t in _ARITH:
            out = self._arith(ctx, node)
        elif t in _ARANGE:
            out = torch.arange(*node.args, dtype=torch.float32)
        elif t in _SCATTER:
            out = t(*[self._static(ctx, a) if isinstance(a, torch.fx.Node) else a for a in node.args])
        elif t in _VIEWS:
            args = [self._static(ctx, a) if isinstance(a, torch.fx.Node) else a for a in node.args]
            out = t(*args, **{k: v for k, v in node.kwargs.items() if k != "memory_format"})
        elif t is aten.where.self:
            c, a, b = (self._static(ctx, x) for x in node.args[:3])
            out = torch.where(torch.as_tensor(c, dtype=torch.bool), torch.as_tensor(a, dtype=torch.float32),
                              torch.as_tensor(b, dtype=torch.float32))
        else:
            fill = {aten.scalar_tensor.default: lambda: node.args[0], aten.ones_like.default: lambda: 1.0,
                    aten.zeros_like.default: lambda: 0.0, aten.full_like.default: lambda: node.args[1],
                    aten.full.default: lambda: node.args[1], aten.ones.default: lambda: 1.0, aten.zeros.default: lambda: 0.0,
                    aten.fill.Scalar: lambda: node.args[1]}[t]()
            out = torch.full(S, fill, dtype=torch.float32)
        out = out.to(ctx.dtype_of(node)).to(torch.float32).contiguous()
        if tuple(out.shape) != tuple(S):
            refuse("shape", node, ctx.module_path(node), f"static {list(out.shape)} vs {list(S)}")
        ctx.define(node, StaticFloat(out))
        ctx.record(self, node, {"shape": list(S)}, emitted="static")


__all__ = ["BoxPermute", "StaticFloat", "StaticFloats"]
