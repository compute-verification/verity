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
do; where torch's bf16 kernel rounds otherwise, the rule refuses. At a compute capability `cc`, a bf16 `mm` is instead F.linear's
GEMM as that GPU's cuBLAS computes it (`Matmul`): the route its capture fitted (`cublas_bf16.train_route`), each k16 step the tensor
core's dot (`HopperBF16WgmmaDot16_v1`). A training op with no Definition to lower it to is refused with the Definition it needs
(`_log_softmax`: a correctly-rounded f32 log, `F32LogRn`).

    training_rules(min_view_leaves, cc) -> RuleSet     the training rule set

Views and static float tensors are `rules.training_views`'s; a survey of what the rules refuse is `rules.training_survey`'s.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Affine, Coll, Strided, broadcast, concat
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import ref_prims as R
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.rules.base import OpRule, RuleSet
from verity_vllm.program.frontend.rules.training_views import (_PARTS, BoxPermute, StaticFloat, StaticFloats, _info, _view_expand,
                                                               _view_permute)
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of

aten = torch.ops.aten
F32, BF16, BIT = Value(32), Value(16), Value(1)
_DT = {torch.float32: "f32", torch.bfloat16: "bf16"}


# ---------------------------------------------------------------------------------------------------------
# emitting
# ---------------------------------------------------------------------------------------------------------


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


class _TrainRule(OpRule):
    modes = frozenset({"reference"})


# ---------------------------------------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------------------------------------


#: F.linear's GEMM at x (M, K), W (N, K) by its mm's operand layouts (`_layout`) as autograd lays them out: x @ W^T, dY @ W, dY^T @ X
_ROLES = {("row", "col"): "fwd", ("row", "row"): "dX", ("col", "row"): "dW"}
#: F.linear's (M, K, N) from its role's mm(a[Mm, Km], b[Km, Nm])
_LINEAR_DIMS = {"fwd": lambda Mm, Km, Nm: (Mm, Km, Nm), "dX": lambda Mm, Km, Nm: (Mm, Nm, Km), "dW": lambda Mm, Km, Nm: (Km, Nm, Mm)}


def _layout(t) -> str | None:
    """A dense matrix's layout in memory: "row" (row-major), "col" (column-major: a row-major one transposed), else None."""
    if t is None or t.dim() != 2 or t.storage_offset() != 0:
        return None
    rows, cols = t.shape
    return {(cols, 1): "row", (1, rows): "col"}.get(tuple(t.stride()))


class Matmul(_TrainRule):
    """`mm(a[M,K], b[K,N])` / `bmm(a[B,M,K], b[B,K,N])` of f32 or bf16 values -> one `batch` of `LinearRef_v1{K,N,DT}` over the rows
    of `a` (per batch index for `bmm`), with `b` transposed by reference.  At a compute capability ``cc`` whose cuBLAS routes are
    captured (`cublas_bf16.TRAIN_CC`), a bf16 `mm` is one batch of F.linear's row GEMM as its cuBLAS computes it instead: its role
    read off the operands' layouts (`_ROLES`), its route the capture's (`cublas_bf16.train_route`) and its Definition the route's on
    words.  The rule still claims a bf16 product it can't lower that way (a `bmm`, a layout no role has, a shape the capture doesn't
    cover, a route with no Definition on words) and refuses it, so that no other rule lowers it with torch CPU's arithmetic."""
    info = _info("Matmul", "aten.mm / aten.bmm of f32 or bf16 values", "both operands are values of one dtype, f32 or bf16",
                 "one batch of LinearRef rows per matrix")

    def __init__(self, cc: tuple[int, int] | None = None):
        if cc is not None and tuple(cc) != CB.TRAIN_CC:
            raise ValueError(f"Matmul: cc {tuple(cc)} has no captured cuBLAS routes (cublas_bf16.TRAIN_CC is {CB.TRAIN_CC})")
        self.cc = None if cc is None else f"cc{cc[0]}.{cc[1]}"
        if cc is not None:
            self.info = _info(f"Matmul_{self.cc}", "aten.mm / aten.bmm of f32 or bf16 values",
                              "both operands are values of one dtype, f32 or bf16",
                              f"a bf16 mm: one of F.linear's GEMMs, at a shape and route {self.cc}'s capture covers",
                              "one batch of rows per matrix: the cuBLAS route's GEMM for a bf16 mm, else LinearRef")

    def targets(self):
        return {aten.mm.default, aten.bmm.default}

    def applies(self, node, ctx):
        for a in node.args[:2]:
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return "an operand is not a value"
        dts = {ctx.dtype_of(a) for a in node.args[:2]}
        if len(dts) != 1 or not dts <= set(_DT):
            return f"numerics_unregistered: dtypes {sorted(map(str, dts))}"
        return True

    def emit(self, node, ctx):
        if self.cc is not None and ctx.dtype_of(node.args[0]) == torch.bfloat16:
            return self._emit_gpu(node, ctx)
        a, b = ctx.coll(node.args[0]), ctx.coll(node.args[1])
        sa, sb = _shape(ctx, node.args[0]), _shape(ctx, node.args[1])
        nb = sa[0] if len(sa) == 3 else 1
        M, K, N = sa[-2], sa[-1], sb[-1]
        dt = ctx.dtype_of(node.args[0])
        E = BF16 if dt == torch.bfloat16 else F32
        fn = ctx.kind("linear", node, K=K, N=N, DT=_DT[dt], BIAS=False)
        a3 = a.reshape(Array(nb, Array(M, Array(K, E))))
        b3 = b.reshape(Array(nb, Array(K, Array(N, E))))
        outs = []
        for i in range(nb):
            bT = _view_permute(b3[i], (1, 0), node.name)
            outs.append(_batch(ctx, node, fn, a3[i], bT, axes=(0, None)))
        ctx.define(node, Coll(_tensor_type(list(sa[:-1]) + [N], E.w), concat([o.refs for o in outs])))
        ctx.record(self, node, {"batch": nb, "M": M, "K": K, "N": N}, emitted=f"{fn.id} x{nb * M}")

    def _emit_gpu(self, node, ctx):
        def no(why: str):
            refuse("numerics_unregistered", node, ctx.module_path(node), f"a bf16 {node.target} on {self.cc}: {why}")

        if node.target is not aten.mm.default:
            no("only F.linear's GEMMs have captured routes (cublas_bf16.TRAIN_CAPTURE); without a cc it is torch CPU's")
        layouts = tuple(_layout(ctx.tensor_meta(a)) for a in node.args[:2])
        if layouts not in _ROLES:
            no(f"operand layouts {layouts}, none of F.linear's GEMMs' "
               f"({', '.join(f'{r} {a} x {b}' for (a, b), r in _ROLES.items())})")
        role = _ROLES[layouts]
        (Mm, Km), Nm = _shape(ctx, node.args[0]), _shape(ctx, node.args[1])[1]
        M, K, N = _LINEAR_DIMS[role](Mm, Km, Nm)
        try:
            route = CB.train_route(role, M, K, N)
            fn = word_of(CB.route_gemm(route, *CB.TRAIN_LINEAR_ROLES[role][2](M, K, N)))
        except (ValueError, KeyError) as e:
            no(f"F.linear's {role} GEMM at x ({M}, {K}), W ({N}, {K}): {e.args[0]}")
        a = ctx.coll(node.args[0]).reshape(Array(Mm, Array(Km, BF16)))
        bT = _view_permute(ctx.coll(node.args[1]).reshape(Array(Km, Array(Nm, BF16))), (1, 0), node.name)
        out = _batch(ctx, node, fn, a, bT, axes=(0, None))
        ctx.define(node, Coll(_tensor_type([Mm, Nm], 16), out.refs))
        ctx.record(self, node, {"role": role, "M": M, "K": K, "N": N, "route": route}, emitted=f"{fn.id} x{Mm}")


# ---------------------------------------------------------------------------------------------------------
# element-wise
# ---------------------------------------------------------------------------------------------------------

_BINARY = {aten.add.Tensor: P.F32Add, aten.add.Scalar: P.F32Add, aten.sub.Tensor: R.F32Sub, aten.sub.Scalar: R.F32Sub,
           aten.mul.Tensor: P.F32Mul, aten.mul.Scalar: P.F32Mul, aten.div.Tensor: P.F32Div, aten.div.Scalar: P.F32Div}


class Binary(_TrainRule):
    """`add` / `sub` / `mul` / `div` (`.Tensor`, `.Scalar`) of f32 or bf16 values, static float tensors and Python numbers,
    broadcast: `F32Add_v1`, `F32Sub_v1`, `F32Mul_v1`, `F32Div_v1` (true division) element by element; a number is its f32 word, a
    bf16 operand is widened and a bf16 result rounded. A bf16 result's numbers are bf16 values: torch rounds a number to bf16
    for `add` and `sub` and keeps its f32 value for `mul` and `div`, which agree only then."""
    info = _info("Binary", "aten add/sub/mul/div .Tensor/.Scalar with broadcasting, f32 or bf16", "operands are f32 or bf16 values, "
                 "static float tensors or Python numbers", "a bf16 result's numbers and static operands are bf16 values",
                 "alpha == 1, no rounding_mode", "one batch over the output's elements")

    def targets(self):
        return set(_BINARY)

    def applies(self, node, ctx):
        if ctx.dtype_of(node) not in _DT:
            return f"numerics_unregistered: result dtype {ctx.dtype_of(node)}"
        if node.kwargs.get("alpha", 1) != 1 or node.kwargs.get("rounding_mode") is not None:
            return "numerics_unregistered: alpha != 1 or a rounding mode"
        whys = [w for w in (_operand_ok(ctx, a, tuple(_DT)) for a in node.args[:2]) if w]
        if whys:
            return whys[0]
        statics = [a for a in node.args[:2] if not (isinstance(a, torch.fx.Node) and isinstance(ctx.val(a), Coll))]
        if len(statics) == 2:
            return "no operand is a value (a static result: StaticFloats)"
        if ctx.dtype_of(node) == torch.bfloat16 and not all(_bf16_held(ctx, a) for a in statics):
            return "numerics_unregistered: a bf16 op with a number that bf16 doesn't hold"
        return True

    def emit(self, node, ctx):
        S = ctx.shape_of_node(node)
        fn = _BINARY[node.target]
        out = _ew(ctx, node, fn, [_operand(ctx, node, a, S) for a in node.args[:2]], S)
        ctx.define(node, _rounded(ctx, node, out, S))
        ctx.record(self, node, {"shape": list(S), "dtype": _DT[ctx.dtype_of(node)]}, emitted=f"{fn.id} x{math.prod(S)}")


class Unary(_TrainRule):
    """`neg`, `sqrt`, `rsqrt`, `reciprocal`, `exp`, `sigmoid`, `pow.Tensor_Scalar` of an f32 or bf16 value, as torch's CPU kernels
    compute them: `rsqrt` = 1 / sqrt, `sigmoid` = 1 / (1 + exp(-x)), `pow` 2 / 3 products, 0.5 sqrt, -0.5 rsqrt, -1 and -2
    reciprocals, 1 the operand, 0 one; `exp` is `F32ExpRn_v1` (correctly rounded, where torch's vectorized exp is not). `sqrt` is
    IEEE's (`F32Sqrt_v1`, as CUDA's `sqrt.rn`); torch CPU's (and its `pow` 0.5) is MKL's, which is not IEEE's on some CPUs. A bf16
    operand is widened and the result rounded to bf16, except where torch's bf16 kernel rounds in between, which is refused: `pow`
    3 and -2 round the square, and `rsqrt` (and `pow` -0.5) the square root in its scalar loop."""
    info = _info("Unary", "aten neg/sqrt/rsqrt/reciprocal/exp/sigmoid/pow.Tensor_Scalar of f32 or bf16", "the operand is an f32 or "
                 "bf16 value", "pow's exponent is one torch special-cases (0, 0.5, 1, 2, 3, -0.5, -1, -2)",
                 "bf16: not rsqrt, nor pow 3, -2 or -0.5", "one batch per step over the elements")

    def targets(self):
        return {aten.neg.default, aten.sqrt.default, aten.rsqrt.default, aten.reciprocal.default, aten.exp.default,
                aten.sigmoid.default, aten.pow.Tensor_Scalar}

    def applies(self, node, ctx):
        why = _operand_ok(ctx, node.args[0], tuple(_DT))
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


def _ref(c: Coll) -> tuple[str, int, int]:
    return c.refs.space, c.refs.idx, c.refs.base


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


def training_rules(min_view_leaves: int = 0, cc: tuple[int, int] | None = None) -> RuleSet:
    """The default rules with the training rules: views of static floats, `BoxPermute` (views of more than `min_view_leaves` leaves
    that are not one run) and `Matmul` before them, the element-wise, reduction, softmax, loss and embedding rules after.
    `min_view_leaves` is 0 by default: the default views write a permuted view of fewer leaves than their limit as a list of its
    leaves, which the root's replay then substitutes as one reference part per leaf.  `cc` (`cublas_bf16.TRAIN_CC`): bf16 `mm`s as
    that GPU's cuBLAS computes them (`Matmul`), in the rule set ``training-cc<major>.<minor>``."""
    from verity_vllm.program.frontend.rules import DEFAULT_RULES

    first = [StaticFloats(), BoxPermute(min_view_leaves), Matmul(cc)]
    last = [Binary(), Unary(), Lerp(), Addcmul(), Reduce(), SafeSoftmax(), SoftmaxBackward(), LogSoftmax(), LogSoftmaxBackward(),
            NllLossForward(), NllLossBackward(), EmbeddingDenseBackward()]
    name = "training" if cc is None else f"training-cc{cc[0]}.{cc[1]}"
    return RuleSet(name, first + list(DEFAULT_RULES.rules) + last, list(DEFAULT_RULES.families))


__all__ = ["training_rules", "Matmul", "Binary", "Unary", "Lerp", "Addcmul", "Reduce", "SafeSoftmax", "SoftmaxBackward", "LogSoftmax",
           "LogSoftmaxBackward", "NllLossForward", "NllLossBackward", "EmbeddingDenseBackward"]
