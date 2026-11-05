"""What the training rules (`rules.training`) emit with: constants, element-wise operands, batches and the rule base.

    _const(ctx, node, bits, w)          one constant gate per distinct word in the derivation
    _operand(ctx, node, a, S)           an element-wise operand broadcast to the output shape, as f32 words
    _ew(ctx, node, fn, ops, S)          `fn` element by element over the output shape
    _TrainRule                          a rule of reference mode
"""

from __future__ import annotations

import math

import numpy as np
import torch

from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Affine, Coll, Strided, broadcast, concat
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import ref_prims as R
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.rules.base import OpRule
from verity_vllm.program.frontend.rules.training_views import _PARTS, StaticFloat, _view_expand
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of

aten = torch.ops.aten
F32, BF16, BIT = Value(32), Value(16), Value(1)
_DT = {torch.float32: "f32", torch.bfloat16: "bf16"}


def _bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


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


def _const(ctx, node, bits: int, w: int = 32) -> Coll:
    """One constant gate per distinct word in the derivation (a node a refused rule removed is made again)."""
    cache = ctx.state.setdefault("train_consts", {})
    fn = const(w, bits & ((1 << w) - 1))
    k = cache.get((w, bits))
    if k is None or k >= len(ctx.B.nodes) or ctx.B.nodes[k].fn is not fn:
        _call(ctx, node, fn)
        k = cache[(w, bits)] = len(ctx.B.nodes) - 1
    return Coll(Value(w), Affine("n", k, 0, 1))


def _static_coll(ctx, node, t: torch.Tensor) -> Coll:
    """A static f32 tensor as references to one constant gate per distinct word, run by run."""
    bits = t.contiguous().view(torch.int32).reshape(-1).numpy().astype(np.uint32)
    n = bits.size
    starts = np.concatenate([[0], np.flatnonzero(np.diff(bits)) + 1])
    if starts.size > _PARTS:
        refuse("shape", node, ctx.module_path(node), f"a constant tensor of {starts.size} runs (limit {_PARTS})")
    ends = np.concatenate([starts[1:], [n]])
    parts = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        c = _const(ctx, node, int(bits[a]))
        parts.append(broadcast(c.refs.space, c.refs.idx, c.refs.base, b - a))
    return Coll(Array(n, F32), concat(parts))


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
                f"{' and '.join(_DT[d] for d in dtypes)})")
    return None


def _bf16_held(ctx, a) -> bool:
    """Whether a Python number or static float operand is a bf16 value (its f32 word survives rounding to bf16)."""
    t = torch.as_tensor(a if isinstance(a, (int, float)) else ctx.val(a).tensor, dtype=torch.float32)
    return torch.equal(t.to(torch.bfloat16).to(torch.float32).view(torch.int32), t.view(torch.int32))


def _widened(ctx, node, x: Coll) -> Coll:
    """A bf16 value's words widened to f32 (`Bf16ToF32_v1`, exact), in its shape."""
    S = shape_of(x.type)[0]
    if x.type.leaves == 1:
        return Coll(_tensor_type(S, 32), _call(ctx, node, P.Bf16ToF32, Coll(BF16, x.refs)).refs)
    return _batch(ctx, node, P.Bf16ToF32, x.reshape(Array(math.prod(S), BF16)), axes=(0,)).reshape(_tensor_type(S, 32))


def _rounded(ctx, node, y: Coll, S: tuple[int, ...]) -> Coll:
    """An f32 result `y` of shape `S` in the node's dtype: `y` for f32, each word rounded to bf16 (`F32ToBf16Rn_v1`) for bf16."""
    if ctx.dtype_of(node) != torch.bfloat16:
        return y
    if not S:
        return _call(ctx, node, P.F32ToBf16Rn, Coll(F32, y.refs))
    return _batch(ctx, node, P.F32ToBf16Rn, y.reshape(Array(math.prod(S), F32)), axes=(0,)).reshape(_tensor_type(S, 16))


def _operand(ctx, node, a, S: tuple[int, ...]) -> tuple[Coll, int | None]:
    """(collection, batch axis) of an element-wise operand broadcast to the output shape `S`: a shared word, or `S`'s elements;
    f32 words, a bf16 value's widened."""
    n = math.prod(S)
    if isinstance(a, (int, float)):
        return _const(ctx, node, _bits(a)), None
    v = ctx.val(a)
    if isinstance(v, StaticFloat):
        if v.tensor.numel() == 1:
            return _const(ctx, node, _bits(v.tensor.reshape(()).item())), None
        return _static_coll(ctx, node, v.tensor.broadcast_to(S)), 0
    x = ctx.coll(a)
    if ctx.dtype_of(a) == torch.bfloat16:
        x = _widened(ctx, node, x)
    sx = tuple(shape_of(x.type)[0])
    if x.type.leaves == 1:
        return Coll(F32, x.refs), None
    if sx != tuple(S):
        x = _view_expand(x.reshape(_tensor_type([1] * (len(S) - len(sx)) + list(sx), 32)), list(S), node.name)
    return x.reshape(Array(n, F32)), 0


def _ew(ctx, node, fn, ops: list[tuple[Coll, int | None]], S: tuple[int, ...]) -> Coll:
    """`fn` element by element over the output shape `S`."""
    n = math.prod(S)
    args, axes = [c for c, _ in ops], [ax for _, ax in ops]
    if n == 1 and all(ax is None for ax in axes):
        out = _call(ctx, node, fn, *args)
        return out if not S else Coll(_tensor_type(S, 32), out.refs)
    return _batch(ctx, node, fn, *args, axes=axes, n=n).reshape(_tensor_type(S, 32))


def _flat(c: Coll, S) -> tuple[Coll, int]:
    return c.reshape(Array(math.prod(S), F32)), 0


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
