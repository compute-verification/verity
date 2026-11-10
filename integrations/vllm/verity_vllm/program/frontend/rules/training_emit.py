"""What the training rules (`rules.training`) emit with: constants, element-wise operands, batches and the rule base.

    _const(ctx, node, bits, w)          one constant gate per distinct word in the derivation (`training_views`', with the static
                                        tensors' words)
    _operand(ctx, node, a, S)           an element-wise operand broadcast to the output shape, as f32 words
    _word_operand(ctx, node, a, S, dt)  an element-wise operand broadcast to the output shape, as words of its own dtype
    _ew(ctx, node, fn, ops, S, w)       `fn` element by element over the output shape
    _word_ew(ctx, node, fn, args, S, dt) `fn` element by element over the output shape of `_word_operand`'s operands, a static
                                        one read along the axes it varies first when its runs need it
    _tree(ctx, node, fn, v, n, rows, w) `fn` folded pairwise over n terms of `rows` words
    _TrainRule                          a rule of reference mode
"""

from __future__ import annotations

import math

import numpy as np
import torch

from verity.definitions.ops import scalar as SC
from verity.circuits.refs import Affine, Coll, Strided, broadcast, concat
from verity.circuits.types import Array, Value
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as R
from verity.definitions.registry import torch_words as W
from verity_vllm.program.frontend.derive import StaticIndex, refuse
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.rules.base import OpRule
from verity_vllm.program.frontend.rules.training_views import (_PARTS, StaticFloat, _const, _static_words, _view_expand,
                                                               _view_permute, _words)
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import WIDTH_OF_DTYPE, shape_of

aten = torch.ops.aten
F32, BF16, BIT, I64 = Value(32), Value(16), Value(1), Value(64)
_DT = {torch.float32: "f32", torch.bfloat16: "bf16"}
_NAMES = {**_DT, torch.bool: "bool", torch.int64: "int64"}


def _bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def _alpha(node):
    """`add` / `sub`'s alpha: keyword-only in `.Tensor`, positional in `.Scalar`."""
    return node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)


def _note(ctx, node, k0: int) -> None:
    for k in range(k0, len(ctx.B.nodes)):
        ctx.node_hints.setdefault(k, ctx.hint_name(node))
        ctx.node_sources.setdefault(k, node.name)
        ctx.node_fx.setdefault(k, node)


def _batch(ctx, node, fn, *args, axes, n=None) -> Coll:
    k0 = len(ctx.B.nodes)
    try:
        out = ctx.B.batch(fn, *args, axes=axes, n=n, name=ctx.hint_name(node))
    except TypeError as e:
        refuse("shape", node, ctx.module_path(node), f"{getattr(fn, 'id', fn)}: {e}")
    _note(ctx, node, k0)
    return out


def _call(ctx, node, fn, *args) -> Coll:
    return ctx.call(fn, *args, node=node)


def _static_coll(ctx, node, t: torch.Tensor) -> Coll:
    """A static f32 tensor as references to one constant gate per distinct word, run by run."""
    return _static_words(ctx, node, _words(t.to(torch.float32), 32))


def _shape(ctx, a) -> tuple[int, ...]:
    if isinstance(a, torch.fx.Node):
        s = ctx.shape_of_node(a)
        if s is not None:
            return s
        v = ctx.val(a)
        if isinstance(v, StaticFloat):
            return tuple(v.tensor.shape)
    if isinstance(a, (int, float)):
        return ()
    raise TypeError(a)


def _operand_ok(ctx, a, dtypes=(torch.float32,)) -> str | None:
    """None if `a` is an element-wise operand (a value of one of `dtypes`, a static float tensor or a Python number), else why
    not."""
    if isinstance(a, bool):
        return "a bool operand"
    if isinstance(a, (int, float)):
        return None
    if not isinstance(a, torch.fx.Node):
        return f"operand {a!r}"
    v = ctx.val(a)
    if isinstance(v, StaticFloat):
        return None
    if not isinstance(v, Coll):
        return f"{a.name} is not a value ({type(v).__name__})"
    if ctx.dtype_of(a) not in dtypes:
        return (f"numerics_unregistered: {a.name} has dtype {ctx.dtype_of(a)} (this rule lowers "
                f"{' and '.join(_NAMES[d] for d in dtypes)})")
    return None


def _word_ok(ctx, a, dtype) -> str | None:
    """None if `a` is read as `dtype`'s words with no conversion: a value or a static tensor of `dtype`, or a Python bool (for
    bool) or int (for an integer dtype); else why not."""
    if not isinstance(a, torch.fx.Node):
        ok = isinstance(a, bool) if dtype == torch.bool else type(a) is int and not dtype.is_floating_point
        return None if ok else f"operand {a!r} of a {dtype} result"
    if not isinstance(ctx.val(a), (Coll, StaticIndex, StaticFloat)):
        return f"{a.name} is not a value ({type(ctx.val(a)).__name__})"
    if ctx.dtype_of(a) != dtype:
        return f"numerics_unregistered: {a.name} has dtype {ctx.dtype_of(a)}, not {dtype} (a conversion in between)"
    return None


def _converted(ctx, node, x: Coll, fn, w: int) -> Coll:
    """`fn` on each word of the value x, in its shape: words `w` bits wide."""
    S, w_in = shape_of(x.type)
    if x.type.leaves == 1:
        return Coll(_tensor_type(S, w), _call(ctx, node, fn, Coll(Value(w_in), x.refs)).refs)
    return _batch(ctx, node, fn, x.reshape(Array(math.prod(S), Value(w_in))), axes=(0,)).reshape(_tensor_type(S, w))


def _widened(ctx, node, x: Coll) -> Coll:
    """A bf16 value's words widened to f32 (`Bf16ToF32_v1`, exact), in its shape."""
    return _converted(ctx, node, x, P.Bf16ToF32, 32)


def _rounded(ctx, node, y: Coll, S: tuple[int, ...], fn=P.F32ToBf16Rn) -> Coll:
    """An f32 result `y` of shape `S` in the node's dtype: `y` for f32, each word rounded to bf16 by `fn` (torch CPU's
    `F32ToBf16Rn_v1`, or the GPU's `F2fpBf16_v1`) for bf16."""
    if ctx.dtype_of(node) != torch.bfloat16:
        return y
    if not S:
        return _call(ctx, node, fn, Coll(F32, y.refs))
    return _batch(ctx, node, fn, y.reshape(Array(math.prod(S), F32)), axes=(0,)).reshape(_tensor_type(S, 16))


def _from_bits(ctx, node, x: Coll, dtype=torch.float32) -> Coll:
    """A bool value's words as f32 (1.0 or +0, `SelectF32_v1`) or int64 (1 or 0, `SelectI64_v1`), in its shape, as torch converts
    a bool."""
    S = shape_of(x.type)[0]
    f32 = dtype == torch.float32
    fn, w = (SC.SelectF32, 32) if f32 else (W.SelectI64, 64)
    one, zero = _const(ctx, node, _bits(1.0) if f32 else 1, w), _const(ctx, node, 0, w)
    if x.type.leaves == 1:
        return Coll(_tensor_type(S, w), _call(ctx, node, fn, Coll(BIT, x.refs), one, zero).refs)
    flat = x.reshape(Array(math.prod(S), BIT))
    return _batch(ctx, node, fn, flat, one, zero, axes=(0, None, None)).reshape(_tensor_type(S, w))


def _broadcast(x: Coll, S: tuple[int, ...], w: int, where: str) -> tuple[Coll, int | None]:
    """(collection, batch axis) of the value x broadcast to the shape `S`, its words `w` bits wide: one shared word, or `S`'s
    elements."""
    if x.type.leaves == 1:
        return Coll(Value(w), x.refs), None
    sx = tuple(shape_of(x.type)[0])
    if sx != tuple(S):
        x = _view_expand(x.reshape(_tensor_type([1] * (len(S) - len(sx)) + list(sx), w)), list(S), where)
    return x.reshape(Array(math.prod(S), Value(w))), 0


def _operand(ctx, node, a, S: tuple[int, ...]) -> tuple[Coll, int | None]:
    """(collection, batch axis) of an element-wise operand broadcast to the output shape `S`: a shared word, or `S`'s elements;
    f32 words, a bf16 value's widened, a bool value's 1.0 or +0 and an int64 value's rounded to nearest (`I64ToF32Rn_v1`). `a` is
    a Python number, a node or a `StaticFloat`."""
    if isinstance(a, (int, float)):
        return _const(ctx, node, _bits(a)), None
    v = a if isinstance(a, StaticFloat) else ctx.val(a)
    if isinstance(v, StaticFloat):
        if v.tensor.numel() == 1:
            return _const(ctx, node, _bits(v.tensor.reshape(()).item())), None
        return _static_coll(ctx, node, v.tensor.broadcast_to(S)), 0
    x = ctx.coll(a)
    dt = ctx.dtype_of(a)
    if dt == torch.bfloat16:
        x = _widened(ctx, node, x)
    elif dt == torch.bool:
        x = _from_bits(ctx, node, x)
    elif dt == torch.int64:
        x = _converted(ctx, node, x, W.I64ToF32Rn, 32)
    return _broadcast(x, S, 32, node.name)


def _word_operand(ctx, node, a, S: tuple[int, ...], dtype) -> tuple[Coll, int | None]:
    """(collection, batch axis) of an operand `_word_ok` accepts, broadcast to `S`, as `dtype`'s words."""
    w = WIDTH_OF_DTYPE[dtype]
    if not isinstance(a, torch.fx.Node):
        return _const(ctx, node, int(a), w), None
    v = ctx.val(a)
    if isinstance(v, (StaticIndex, StaticFloat)):
        t = v.tensor.to(dtype)
        if t.numel() == 1:
            return _const(ctx, node, int(_words(t, w)[0]), w), None
        return _static_words(ctx, node, _words(t.broadcast_to(S), w), w), 0
    return _broadcast(ctx.coll(a), S, w, node.name)


def _ew(ctx, node, fn, ops: list[tuple[Coll, int | None]], S: tuple[int, ...], w: int = 32) -> Coll:
    """`fn` element by element over the output shape `S`, its result words `w` bits wide."""
    n = math.prod(S)
    args, axes = [c for c, _ in ops], [ax for _, ax in ops]
    if n == 1 and all(ax is None for ax in axes):
        out = _call(ctx, node, fn, *args)
        return out if not S else Coll(_tensor_type(S, w), out.refs)
    return _batch(ctx, node, fn, *args, axes=axes, n=n).reshape(_tensor_type(S, w))


def _flat(c: Coll, S, w: int = 32) -> tuple[Coll, int]:
    return c.reshape(Array(math.prod(S), Value(w))), 0


def _static_major(ctx, args, S: tuple[int, ...], dtype) -> tuple[int, ...] | None:
    """`S`'s axes with those a static operand varies along first, when some static operand's words in `S`'s own order take more
    runs than a constant tensor may; else None."""
    w = WIDTH_OF_DTYPE[dtype]
    ts = [v.tensor for a in args if isinstance(a, torch.fx.Node) and isinstance(v := ctx.val(a), (StaticIndex, StaticFloat))
          and v.tensor.numel() > 1]
    if all(np.count_nonzero(np.diff(_words(t.to(dtype).broadcast_to(S), w))) < _PARTS for t in ts):
        return None
    lead = sorted({len(S) - t.dim() + i for t in ts for i, n in enumerate(t.shape) if n > 1})
    dims = tuple(lead) + tuple(d for d in range(len(S)) if d not in lead)
    return dims if dims != tuple(range(len(S))) else None


def _word_ew(ctx, node, fn, args, S: tuple[int, ...], dtype) -> Coll:
    """`fn` element by element over the output shape `S` of operands `_word_ok` accepts, as `dtype`'s words. A static operand
    whose words in `S`'s order would take more runs than a constant tensor may (a row of shift amounts across a wide tensor) is
    read with the axes it varies along outermost, and the result permuted back: the same one batch, its references strided."""
    w = WIDTH_OF_DTYPE[dtype]
    dims = _static_major(ctx, args, S, dtype)
    if dims is None:
        return _ew(ctx, node, fn, [_word_operand(ctx, node, a, S, dtype) for a in args], S, w)
    T = tuple(S[d] for d in dims)
    ops = []
    for a in args:
        v = ctx.val(a) if isinstance(a, torch.fx.Node) else None
        if isinstance(v, (StaticIndex, StaticFloat)) and v.tensor.numel() > 1:
            ops.append((_static_words(ctx, node, _words(v.tensor.to(dtype).broadcast_to(S).permute(dims), w), w), 0))
        elif v is not None and not isinstance(v, (StaticIndex, StaticFloat)) and ctx.coll(a).type.leaves > 1:
            x, _ = _broadcast(ctx.coll(a), S, w, node.name)
            ops.append(_flat(_view_permute(x.reshape(_tensor_type(list(S), w)), dims, node.name), T, w))
        else:
            ops.append(_word_operand(ctx, node, a, T, dtype))
    return _view_permute(_ew(ctx, node, fn, ops, T, w), tuple(dims.index(d) for d in range(len(S))), node.name)


def _tree(ctx, node, fn, v: Coll, n: int, rows: int, w: int) -> Coll:
    """`fn` folded over n terms of `rows` words each (term j's word r at j * rows + r of `v`): the first half of the terms with
    the second, a level a batch, an odd term carried. The `rows` words; any fold of an associative and commutative `fn` (an
    integer add) is this one."""
    while n > 1:
        h = n // 2
        s = _batch(ctx, node, fn, v[:h * rows], v[h * rows:2 * h * rows], axes=(0, 0))
        if n % 2:
            s = Coll(Array((h + 1) * rows, Value(w)), concat([s.refs, v[2 * h * rows:].refs]))
        v, n = s, h + n % 2
    return v


def _rows(x: Coll, S) -> tuple[Coll, int, int]:
    N = S[-1]
    Rn = math.prod(S[:-1])
    return x.reshape(Array(Rn, Array(N, F32))), Rn, N


def _per_row(v: Coll, Rn: int, N: int) -> Coll:
    """Each of a row value's `Rn` words, repeated over its row's `N` elements (`v` one run)."""
    r = v.refs
    if not isinstance(r, Affine):
        raise FrontendError("E_UNSUPPORTED_VIEW", "a per-row value that is not one run", "training")
    return Coll(Array(Rn * N, v.type.elem if isinstance(v.type, Array) else v.type), Strided(r.space, r.idx, r.base, N, 1, 0, Rn * N))


def _ones(ctx, node, N: int) -> Coll:
    c = _const(ctx, node, _bits(1.0))
    return Coll(Array(N, F32), broadcast(c.refs.space, c.refs.idx, c.refs.base, N))


def _iota(ctx, node, V: int) -> Coll:
    """The I32 words 0..V-1 as one run (doubling `I32Add`s, then one `I32Add` of 0 that lays them out contiguously)."""
    cache = ctx.state.setdefault("train_iota", {})
    hit = cache.get(V)
    if hit is not None and hit[0] < len(ctx.B.nodes) and ctx.B.nodes[hit[0]] is hit[1]:
        return Coll(Array(V, Value(32)), Affine("n", hit[0], 0, V))
    parts, have = [_const(ctx, node, 0).refs], 1
    while have < V:
        m = min(have, V - have)
        base = Coll(Array(m, Value(32)), concat(parts).slice(0, m))
        parts.append(_batch(ctx, node, SC.I32Add, base, _const(ctx, node, have), axes=(0, None)).refs)
        have += m
    out = _batch(ctx, node, SC.I32Add, Coll(Array(V, Value(32)), concat(parts)), _const(ctx, node, 0), axes=(0, None))
    k = len(ctx.B.nodes) - 1
    cache[V] = (k, ctx.B.nodes[k])
    return out


def _tokens(ctx, node, a) -> Coll:
    """Integer token ids as I32 words (`I64ToI32`, range-checked, for int64)."""
    t = ctx.coll(a)
    n = t.type.leaves
    if ctx.dtype_of(a) == torch.int64:
        return _batch(ctx, node, R.I64ToI32, t.reshape(Array(n, Value(64))), axes=(0,))
    return t.reshape(Array(n, Value(32)))


def _ref(c: Coll) -> tuple[str, int, int]:
    return c.refs.space, c.refs.idx, c.refs.base


class _TrainRule(OpRule):
    modes = frozenset({"reference"})
