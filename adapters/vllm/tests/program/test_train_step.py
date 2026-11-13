"""The training-step frontend (`program.frontend.train_step`, `derive_graph`, the training rules): each training rule's Program run
on small CPU tensors of even and odd lengths, bit for bit against torch where the rule states torch CPU's own f32 operation (in
bf16, its f32 operation on the widened operands rounded once), and where the reference vocabulary's arithmetic is its own
(sequential sums, a correctly rounded exp and log) bit for bit against the rule's stated arithmetic, composed from the evaluators
of the Definitions it names, and to a tolerance against torch; the functional AdamW against `torch.optim.AdamW(foreach=False)`;
`BoxPermute` against torch's permute; the tiny step in f32 and in bf16 (its linears' operands cast, as autocast casts them), and
its survey with a loss term no rule lowers; the bf16 GEMMs on sm_120 (`training_rules(cc=(12, 0))`), each its captured route's
Definition and, on stand-in routes at small shapes, bit for bit that route's model; and the element-wise ops on sm_120 bit for bit
against the GPU's arithmetic as torch CPU's f32 operations compute it on NaN-free words, and `silu_backward` and a fused AdamW
step against their stated forms in the GPU's Definitions."""
from __future__ import annotations

import bisect
import collections
import functools
import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode
from verity.circuits.boolean.trace import word_of
from verity.circuits.boundary import gate_width
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.parts import Part, n_members
from verity.circuits.program import is_input
from verity.circuits.query_ast import GateSet, evaluate, parse
from verity.circuits.refs import runs as ref_runs
from verity.definitions.numerics.serve3_reference import materialize
from verity.definitions.ops import scalar as SC
from verity.definitions.registry import bf16_gemm_check as BC
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import moe as MOE
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as R

from tests.program.train_step_lf import ADAMW, loss_fn, meta_step, square_loss
from verity_vllm.program.frontend.boolean import derived
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.program_identity import check_identity
from verity_vllm.program.frontend.report import DeriveReport
from verity_vllm.program.frontend.rules import TORCH_RULES, VLLM_RULES
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.rules.training_survey import survey, with_holes
from verity_vllm.program.frontend.train_step import (
    AdamW,
    _patched,
    export_step,
    in_bits_beyond_theta,
    step_program,
    step_report,
)

aten = torch.ops.aten


def _program(f, *xs, theta=(), rules=None):
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(*xs)
    names = [f"x{i}" for i in range(len(xs))]
    rep = derive_graph(gm, xs, mode="reference", rules=rules or training_rules(), name="T", input_names=names, theta=theta)
    assert rep.program is not None, rep.unsupported
    return rep.program


def _bits(x: torch.Tensor) -> list[int]:
    x = x.detach().contiguous().reshape(-1)
    if x.dtype == torch.float32:
        return x.view(torch.int32).numpy().astype(np.uint32).tolist()
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16).numpy().astype(np.uint16).tolist()
    return [int(v) & ((1 << (8 * x.element_size())) - 1) for v in x.tolist()]


class _IeeeSqrt(TorchDispatchMode):
    """torch with IEEE f32 `sqrt` (and `pow` 0.5, which torch computes with it), as the rules state it: torch CPU's is MKL's, which
    is not IEEE's on some CPUs."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        x = args[0] if args else None
        if isinstance(x, torch.Tensor) and x.dtype == torch.float32 and (
                func is aten.sqrt.default or (func is aten.pow.Tensor_Scalar and args[1] == 0.5)):
            return torch.from_numpy(np.asarray(np.sqrt(x.detach().numpy())))
        return func(*args, **(kwargs or {}))


def _outputs(prog, *xs) -> torch.Tensor:
    """The Program's outputs on `xs` as one flat f32 tensor (bf16 words widened)."""
    tab = materialize(prog, [_bits(x) for x in xs])
    got = np.array([tab[g] << 32 - gate_width(prog, g) for g in prog.circuit.root_outputs()], dtype=np.uint32).view(np.float32)
    return torch.from_numpy(got.copy())


def _run(f, *xs, rules=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(the Program's outputs, torch's with IEEE sqrt), each as one flat f32 tensor (bf16 words widened)."""
    got = _outputs(_program(f, *xs, rules=rules), *xs)
    with _IeeeSqrt():
        want = torch.cat([t.detach().reshape(-1).float() for t in f(*xs)])
    return got, want


def _same_bits(got: torch.Tensor, want: torch.Tensor) -> None:
    bad = (got.view(torch.int32) != want.view(torch.int32)).nonzero().reshape(-1)
    assert bad.numel() == 0, (f"{bad.numel()} of {got.numel()} words differ, first at {bad[:4].tolist()}: "
                              f"{got[bad[:4]].tolist()} vs torch {want[bad[:4]].tolist()}")


def _data(*shape, positive=False, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 17 * len(shape))
    x = torch.randn(*shape, generator=g)
    x.reshape(-1)[:4] = torch.tensor([0.0, -0.0, 1e-30, 3.0e7])[: x.numel()]
    return x.abs() + 1e-3 if positive else x


def _masked(x: torch.Tensor) -> torch.Tensor:
    """`x` with its first row -inf throughout and its second causal (-inf after its first two elements)."""
    x = x.clone()
    x[0] = float("-inf")
    x[1, 2:] = float("-inf")
    return x


X, Y = _data(4, 16), _data(4, 16, seed=1)
POS = _data(4, 16, positive=True, seed=2)
ROW, COL = _data(16, seed=3), _data(4, 1, seed=4)
#: odd lengths, where torch's vectorized loops end in a scalar tail: 67 elements, and rows of 67
X67, Y67, POS67 = _data(67, seed=8), _data(67, seed=9), _data(67, positive=True, seed=10)
M67, N67, ROW67, COL3 = _data(3, 67, seed=11), _data(3, 67, seed=12), _data(67, seed=13), _data(3, 1, seed=14)
XB, POSB, X67B, M67B = (t.to(torch.bfloat16) for t in (X, POS, X67, M67))


def _elementwise(x, y, pos, m, row, col) -> dict:
    """The element-wise cases on one set of inputs: `pos` positive, `m` a matrix that `row` and `col` broadcast against."""
    return {
        "add": (lambda a, b: (a + b,), (x, y)),
        "sub": (lambda a, b: (a - b,), (x, y)),
        "mul": (lambda a, b: (a * b,), (x, y)),
        "div": (lambda a, b: (a / b,), (x, pos)),
        "add_scalar": (lambda a: (a + 0.1,), (x,)),
        "mul_scalar": (lambda a: (a * 0.3,), (x,)),
        "div_scalar": (lambda a: (a / 3.0,), (x,)),
        "add_alpha": (lambda a, b: (torch.add(a, b, alpha=0.95),), (x, y)),
        "sub_alpha": (lambda a, b: (torch.sub(a, b, alpha=0.3),), (x, y)),
        "add_scalar_alpha": (lambda a: (a.add(0.5, alpha=-0.01),), (x,)),
        "mul_mask": (lambda a, k: (a * k,), (x, y > 0)),
        "mul_mask_row": (lambda a, k: (a * k,), (m, row > 0)),
        "mul_row": (lambda a, r: (a * r,), (m, row)),
        "mul_col": (lambda a, c: (a * c,), (m, col)),
        "neg": (lambda a: (-a,), (x,)),
        "sqrt": (lambda a: (torch.sqrt(a),), (pos,)),
        "rsqrt": (lambda a: (torch.rsqrt(a),), (pos,)),
        "reciprocal": (lambda a: (torch.reciprocal(a),), (pos,)),
        "pow2": (lambda a: (a ** 2,), (x,)),
        "pow3": (lambda a: (a ** 3,), (x,)),
        "pow_half": (lambda a: (a ** 0.5,), (pos,)),
        "pow_m_half": (lambda a: (a ** -0.5,), (pos,)),
        "pow_m1": (lambda a: (a ** -1,), (pos,)),
        "pow_m2": (lambda a: (a ** -2,), (pos,)),
        "lerp_small": (lambda a, b: (torch.lerp(a, b, 0.1),), (x, y)),
        "lerp_large": (lambda a, b: (torch.lerp(a, b, 0.9),), (x, y)),
        "addcmul": (lambda a, b, c: (torch.addcmul(a, b, c, value=0.05),), (x, y, pos)),
        "addcdiv": (lambda a, b, c: (torch.addcdiv(a, b, c, value=-0.05),), (x, y, pos)),
    }


#: a number f32 holds as 1 + 2^-8, a bf16 tie: torch CPU's bf16 add rounds it to 1 (double -> f32 -> bf16; double -> bf16 would
#: give 1 + 2^-7), and its bf16 mul keeps 1 + 2^-8
TIE = 1 + 2 ** -8 + 2 ** -30


def _elementwise_bf16(x, y, pos, m, row, col) -> dict:
    """The element-wise cases of bf16 values (the f32 inputs rounded) that torch's bf16 kernels compute as one f32 operation rounded
    to bf16, with numbers bf16 holds and numbers it doesn't (which torch CPU rounds to bf16 for `add` and `sub` and keeps for `mul`
    and `div`), and of a bf16 and an f32 value, which torch computes in f32."""
    xb, yb, pb, mb, rb, cb = (t.to(torch.bfloat16) for t in (x, y, pos, m, row, col))
    return {
        "add": (lambda a, b: (a + b,), (xb, yb)),
        "sub": (lambda a, b: (a - b,), (xb, yb)),
        "mul": (lambda a, b: (a * b,), (xb, yb)),
        "div": (lambda a, b: (a / b,), (xb, pb)),
        "add_scalar": (lambda a: (a + 0.375,), (xb,)),
        "sub_scalar": (lambda a: (a - 1.5,), (xb,)),
        "mul_scalar": (lambda a: (a * 0.375,), (xb,)),
        "div_scalar": (lambda a: (a / 3.0,), (xb,)),
        "add_number": (lambda a: (a + 0.1,), (xb,)),
        "sub_number": (lambda a: (a - 0.1,), (xb,)),
        "add_tie": (lambda a: (a + TIE,), (xb,)),
        "mul_number": (lambda a: (a * 3.4445,), (xb,)),
        "mul_neg_number": (lambda a: (a * -4.775,), (xb,)),
        "mul_tie": (lambda a: (a * TIE,), (xb,)),
        "div_number": (lambda a: (a / 0.1,), (xb,)),
        "add_mask": (lambda a, k: (a + k,), (xb, y > 0)),
        "one_minus": (lambda a: (torch.empty_like(a).fill_(1.0) - a,), (xb,)),
        "mul_row": (lambda a, r: (a * r,), (mb, rb)),
        "mul_col": (lambda a, c: (a * c,), (mb, cb)),
        "neg": (lambda a: (-a,), (xb,)),
        "sqrt": (lambda a: (torch.sqrt(a),), (pb,)),
        "reciprocal": (lambda a: (torch.reciprocal(a),), (pb,)),
        "pow2": (lambda a: (a ** 2,), (xb,)),
        "pow_half": (lambda a: (a ** 0.5,), (pb,)),
        "pow_m1": (lambda a: (a ** -1,), (pb,)),
        "mul_mixed": (lambda a, b: (a * b,), (xb, y)),
        "add_mixed": (lambda a, b: (a + b,), (x, yb)),
        "mul_row_mixed": (lambda a, r: (a * r,), (mb, row)),
    }


#: the rules torch CPU computes with the same f32 operations: bit for bit
EXACT = {
    **_elementwise(X, Y, POS, X, ROW, COL),
    **{f"{k}_67": case for k, case in _elementwise(X67, Y67, POS67, M67, ROW67, COL3).items()},
    **{f"{k}_bf16": case for k, case in _elementwise_bf16(X, Y, POS, X, ROW, COL).items()},
    **{f"{k}_bf16_67": case for k, case in _elementwise_bf16(X67, Y67, POS67, M67, ROW67, COL3).items()},
    "nll_backward": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 1, -100, tw),),
                     (torch.tensor(1.0), X, torch.tensor([3, 0, 15, 7]), torch.tensor(4.0))),
    "nll_backward_ignored": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 1, 0, tw),),
                             (torch.tensor(1.0), X, torch.tensor([3, 0, 15, 0]), torch.tensor(2.0))),
    "nll_backward_ignored_m100": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 1, -100, tw),),
                                  (torch.tensor(1.0), X, torch.tensor([3, -100, 15, 7]), torch.tensor(3.0))),
    "embedding_backward": (lambda g, i: (aten.embedding_dense_backward(g, i, 6, -1, False),),
                           (X, torch.tensor([2, 5, 2, 0]))),
    "embedding_backward_67": (lambda g, i: (aten.embedding_dense_backward(g, i, 6, -1, False),), (M67, torch.tensor([2, 5, 2]))),
    "permute_of_concat": (lambda x, y: ((torch.cat([x, y], dim=1).t() * 2.0),), (X, Y)),
    "copy_into_allocation": (lambda a, b: (a.new_empty_strided((4, 16), (16, 1)).copy_(b) + a,), (X, Y)),
    "copy_broadcast": (lambda m, r: (torch.empty_like(m).copy_(r) * m,), (X, ROW)),
    "copy_of_static": (lambda a: (torch.empty_like(a).copy_(torch.full((4, 16), 0.5)) * a,), (X,)),
    "static_arithmetic": (lambda a: (a * (torch.full((16,), 3.0) * 0.5 + 1.0 - torch.ones(16)),), (X,)),
    "foreach_add": (lambda a, s: torch._foreach_add([a, s], 1.0), (X67, torch.tensor(7.0))),
    "returned_static": (lambda a: (a * 2.0, torch.zeros_like(a)), (X,)),
    "returned_static_bf16": (lambda a: (a * 2.0, torch.full((3,), 0.375, dtype=torch.bfloat16)), (X,)),
    "foreach_lerp": (lambda a, b, c, d: torch._foreach_lerp([a, c], [b, d], 0.1), (X67, Y67, X, Y)),
    "foreach_lerp_large_weight": (lambda a, b, c, d: torch._foreach_lerp([a, c], [b, d], 0.9), (X67, Y67, X, Y)),
    "static_division": (lambda a: (a * (torch.full((16,), 3.0) / 7.0),), (X,)),
    "copy_widens_bf16": (lambda a, b: (torch.empty_like(a).copy_(b) * a,), (X, XB)),
    "nll_forward_none": (lambda logp, t: aten.nll_loss_forward(logp, t, None, 0, -100), (X, torch.tensor([3, -100, 15, 7]))),
    "nll_backward_none": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 0, -100, tw),),
                          (Y[:, 0].contiguous(), X, torch.tensor([3, -100, 15, 7]), torch.tensor(0.0))),
    "nll_backward_none_static": (lambda logp, t: (aten.nll_loss_backward(torch.full((4,), 0.25), logp, t, None, 0, 0,
                                                                        torch.zeros(())),), (X, torch.tensor([3, 0, 15, 7]))),
}


def _f32(v: float | np.floating) -> int:
    return int(np.float32(v).view(np.uint32))


ONE, NEG_ONE, NEG_INF = _f32(1.0), _f32(-1.0), 0xFF800000


def _rows_of(x: torch.Tensor) -> list[list[int]]:
    return [_bits(r) for r in x.reshape(-1, x.shape[-1])]


def _dot(x: list[int], w: list[int]) -> int:
    """`DotRef_v1{K}`: acc = +0, then acc = `F32Fma_v1(x[k], w[k], acc)` for k ascending."""
    acc = 0
    for a, b in zip(x, w, strict=True):
        acc = P.F32Fma.evaluate(a, b, acc)
    return acc


def _softmax_row(x: list[int]) -> list[int]:
    """`SoftmaxRef_v1{N,f32}`: m an `F32MaxRef_v1` fold from x[0]; e_i = `F32ExpRn_v1(F32Sub_v1(x_i, m))`; s = +0, then
    `F32Add_v1(s, e_i)` ascending; r = `F32Div_v1(1, s)`; `F32Mul_v1(e_i, r)`."""
    m = x[0]
    for v in x[1:]:
        m = R.F32MaxRef.evaluate(m, v)
    e = [R.F32ExpRn.evaluate(R.F32Sub.evaluate(v, m)) for v in x]
    s = 0
    for v in e:
        s = P.F32Add.evaluate(s, v)
    r = P.F32Div.evaluate(ONE, s)
    return [P.F32Mul.evaluate(v, r) for v in e]


def _reduce(dims=None, mean=False):
    """`Reduce`: the reduced axes moved last, a `DotRef` against ones per kept element; `mean` then `F32Div_v1` by f32(N)."""
    def ref(x):
        r = x.dim()
        ds = sorted({d % r for d in dims}) if dims else list(range(r))
        n = math.prod(x.shape[a] for a in ds)
        sums = [_dot(row, [ONE] * n) for row in _rows_of(x.permute([a for a in range(r) if a not in ds] + ds).reshape(-1, n))]
        return [P.F32Div.evaluate(s, _f32(float(n))) for s in sums] if mean else sums
    return ref


def _mm(a, b):
    """`Matmul` in reference mode, `LinearRef_v1{K,N,f32}` per row of `a`: out[m][n] = `DotRef(a[m, :], b[:, n])`."""
    return [_dot(row, col) for row in _rows_of(a) for col in _rows_of(b.t())]


def _exp_scaled(x):
    """`exp(x * 0.01)`: `F32ExpRn_v1(F32Mul_v1(x, f32(0.01)))`."""
    return [R.F32ExpRn.evaluate(P.F32Mul.evaluate(v, _f32(0.01))) for v in _bits(x)]


def _sigmoid(x):
    """`Unary`'s sigmoid: `F32Div_v1(1, F32Add_v1(F32ExpRn_v1(F32Neg_v1(x)), 1))`."""
    return [P.F32Div.evaluate(ONE, P.F32Add.evaluate(R.F32ExpRn.evaluate(SC.F32Neg.evaluate(v)), ONE)) for v in _bits(x)]


def _exp(x):
    """`Unary`'s exp: `F32ExpRn_v1(x)`."""
    return [R.F32ExpRn.evaluate(v) for v in _bits(x)]


def _in_bf16(ref):
    """`ref`'s f32 arithmetic on bf16 operands, as the bf16 rules compute it: each word widened (`Bf16ToF32_v1`), each result rounded
    (`F32ToBf16Rn_v1`), the bf16 word read as `_run` reads it (widened)."""
    def widen(x):
        w = np.array([P.Bf16ToF32.evaluate(v) for v in _bits(x)], dtype=np.uint32)
        return torch.from_numpy(w.view(np.float32).reshape(tuple(x.shape)).copy())
    return lambda *xs: [P.F32ToBf16Rn.evaluate(v) << 16 for v in ref(*map(widen, xs))]


def _safe_softmax(x):
    """`SafeSoftmax`: a row of `SoftmaxRef`, or +0 throughout where its `F32Eq_v1` against -inf, folded by `BitAnd_v1`, holds
    (`SelectF32_v1`)."""
    out = []
    for row in _rows_of(x):
        dead = 1
        for v in row:
            dead = SC.BitAnd.evaluate(dead, SC.F32Eq.evaluate(v, NEG_INF))
        out += [SC.SelectF32.evaluate(dead, 0, w) for w in _softmax_row(row)]
    return out


def _softmax_backward(g, y):
    """`SoftmaxBackward`: per row d = `DotRef(g, y)`, then `F32Mul_v1(y_i, F32Sub_v1(g_i, d))`."""
    out = []
    for gr, yr in zip(_rows_of(g), _rows_of(y), strict=True):
        d = _dot(gr, yr)
        out += [P.F32Mul.evaluate(b, R.F32Sub.evaluate(a, d)) for a, b in zip(gr, yr)]
    return out


def _log_softmax_backward(g, y):
    """`LogSoftmaxBackward`: per row s = `DotRef(g, ones)`, then `F32Sub_v1(g_i, F32Mul_v1(F32ExpRn_v1(y_i), s))`."""
    out = []
    for gr, yr in zip(_rows_of(g), _rows_of(y), strict=True):
        s = _dot(gr, [ONE] * len(gr))
        out += [R.F32Sub.evaluate(a, P.F32Mul.evaluate(R.F32ExpRn.evaluate(b), s)) for a, b in zip(gr, yr)]
    return out


def _log_softmax(x):
    """`LogSoftmax`: per row m an `F32MaxRef_v1` fold from x[0]; d_i = `F32Sub_v1(x_i, m)`; s = +0, then
    `F32Add_v1(s, F32ExpRn_v1(d_i))` ascending; `F32Sub_v1(d_i, F32LogRn_v1(s))`."""
    out = []
    for row in _rows_of(x):
        m = row[0]
        for v in row[1:]:
            m = R.F32MaxRef.evaluate(m, v)
        d = [R.F32Sub.evaluate(v, m) for v in row]
        s = 0
        for v in d:
            s = P.F32Add.evaluate(s, R.F32ExpRn.evaluate(v))
        ls = R.F32LogRn.evaluate(s)
        out += [R.F32Sub.evaluate(v, ls) for v in d]
    return out


def _nll_forward(ignore: int):
    """`NllLossForward`: per row the target (`I64ToI32_v1`) and its entry (`GatherF32x{V}_v1`), +0 where the target is `ignore`
    (`I32Eq_v1`, `SelectF32_v1`), the loss `F32Div_v1(DotRef(entries, -1s), DotRef(weights, ones))` and that count."""
    def ref(logp, t):
        rows = _rows_of(logp)
        toks = [R.I64ToI32.evaluate(k & (1 << 64) - 1) for k in t.tolist()]
        ign = [SC.I32Eq.evaluate(k, ignore & 0xFFFFFFFF) for k in toks]
        picked = [SC.SelectF32.evaluate(i, 0, R.gather_f32(len(r)).evaluate(k, *r)) for i, k, r in zip(ign, toks, rows, strict=True)]
        cnt = _dot([SC.SelectF32.evaluate(i, 0, ONE) for i in ign], [ONE] * len(rows))
        return [P.F32Div.evaluate(_dot(picked, [NEG_ONE] * len(rows)), cnt), cnt]
    return ref


def _nv_expf(v: int) -> int:
    """`NvExpf_v1` of one word, CUDA's `expf`."""
    return int(np.asarray(evaluate_batch(bind(MOE.NvExpf), np.array([v], np.uint64), kernel="reference")).reshape(-1)[0])


def _silu_backward_with(add, div, sub, fma, mul, exp=R.F32ExpRn.evaluate):
    """`SiluBackward` in the given Definitions: s = `div(1, add(1, exp(F32Neg_v1(x))))`, then
    `mul(mul(dy, s), fma(x, sub(1, s), 1))`."""
    def ref(dy, x):
        out = []
        for g, v in zip(_bits(dy), _bits(x), strict=True):
            s = div.evaluate(ONE, add.evaluate(ONE, exp(SC.F32Neg.evaluate(v))))
            out.append(mul.evaluate(mul.evaluate(g, s), fma.evaluate(v, sub.evaluate(ONE, s), ONE)))
        return out
    return ref


_silu_backward = _silu_backward_with(P.F32Add, P.F32Div, R.F32Sub, P.F32Fma, P.F32Mul)


#: the rules whose reference arithmetic is the vocabulary's own (a sequential sum, a correctly rounded exp or log, a dot by fma),
#: each with its stated arithmetic composed from the evaluators of the Definitions it names
CLOSE = {
    "exp": (lambda x: (torch.exp(x * 0.01),), (X,), _exp_scaled),
    "exp_67": (lambda x: (torch.exp(x * 0.01),), (X67,), _exp_scaled),
    "sigmoid": (lambda x: (torch.sigmoid(x),), (X,), _sigmoid),
    "sigmoid_67": (lambda x: (torch.sigmoid(x),), (X67,), _sigmoid),
    "sum_last": (lambda x: (x.sum(-1),), (X,), _reduce([-1])),
    "sum_last_67": (lambda x: (x.sum(-1),), (M67,), _reduce([-1])),
    "mean_last": (lambda x: (x.mean(-1, keepdim=True),), (X,), _reduce([-1], mean=True)),
    "mean_last_67": (lambda x: (x.mean(-1, keepdim=True),), (_data(16, 67, seed=19),), _reduce([-1], mean=True)),
    "sum_first": (lambda x: (x.sum(0),), (X,), _reduce([0])),
    "sum_first_67": (lambda x: (x.sum(0),), (M67,), _reduce([0])),
    "sum_no_dims": (lambda x: (aten.sum.dim_IntList(x, []),), (X,), _reduce()),
    "sum_all": (lambda x: (x.sum(),), (X,), _reduce()),
    "sum_all_67": (lambda x: (x.sum(),), (X67,), _reduce()),
    "mean_all": (lambda x: (x.mean(),), (X,), _reduce(mean=True)),
    "mean_all_67": (lambda x: (x.mean(),), (M67,), _reduce(mean=True)),
    "mm": (lambda a, b: (a @ b,), (X, _data(16, 8, seed=5)), _mm),
    "mm_67": (lambda a, b: (a @ b,), (M67, _data(67, 5, seed=15)), _mm),
    "bmm": (lambda a, b: (torch.bmm(a, b),), (_data(2, 4, 16, seed=6), _data(2, 16, 4, seed=7)),
            lambda a, b: [w for i in range(a.shape[0]) for w in _mm(a[i], b[i])]),
    "bmm_67": (lambda a, b: (torch.bmm(a, b),), (_data(2, 3, 67, seed=16), _data(2, 67, 5, seed=17)),
               lambda a, b: [w for i in range(a.shape[0]) for w in _mm(a[i], b[i])]),
    "safe_softmax": (lambda x: (aten._safe_softmax(x, -1),), (X,), _safe_softmax),
    "safe_softmax_masked": (lambda x: (aten._safe_softmax(x, -1),), (_masked(X),), _safe_softmax),
    "safe_softmax_masked_67": (lambda x: (aten._safe_softmax(x, -1),), (_masked(M67),), _safe_softmax),
    "softmax_backward": (lambda g, y: (aten._softmax_backward_data(g, y, -1, torch.float32),), (Y, torch.softmax(X, -1)),
                         _softmax_backward),
    "softmax_backward_67": (lambda g, y: (aten._softmax_backward_data(g, y, -1, torch.float32),), (N67, torch.softmax(M67, -1)),
                            _softmax_backward),
    "log_softmax": (lambda x: (torch.log_softmax(x, -1),), (X,), _log_softmax),
    "log_softmax_67": (lambda x: (torch.log_softmax(x, -1),), (M67,), _log_softmax),
    # y of flipped rows, so that `_data`'s 3e7 is not at one place of g and y: there the output is 3e7 less the row's sum, which
    # the sums' orders decide past any relative tolerance
    "log_softmax_backward": (lambda g, y: (aten._log_softmax_backward_data(g, y, -1, torch.float32),),
                             (Y, torch.log_softmax(X.flip(-1), -1)), _log_softmax_backward),
    "log_softmax_backward_67": (lambda g, y: (aten._log_softmax_backward_data(g, y, -1, torch.float32),),
                                (N67, torch.log_softmax(M67.flip(-1), -1)), _log_softmax_backward),
    "nll_forward": (lambda logp, t: aten.nll_loss_forward(logp, t, None, 1, -100), (X, torch.tensor([3, 0, 15, 7])),
                    _nll_forward(-100)),
    "nll_forward_ignored": (lambda logp, t: aten.nll_loss_forward(logp, t, None, 1, 0), (X, torch.tensor([3, 0, 15, 0])),
                            _nll_forward(0)),
    "nll_forward_67": (lambda logp, t: aten.nll_loss_forward(logp, t, None, 1, -100),
                       (_data(5, 67, seed=18), torch.tensor([3, -100, 66, 7, 0])), _nll_forward(-100)),
    "sigmoid_bf16": (lambda x: (torch.sigmoid(x),), (XB,), _in_bf16(_sigmoid)),
    "sigmoid_bf16_67": (lambda x: (torch.sigmoid(x),), (X67B,), _in_bf16(_sigmoid)),
    "exp_bf16": (lambda x: (torch.exp(x),), (XB,), _in_bf16(_exp)),
    "mm_bf16": (lambda a, b: (a @ b,), (XB, _data(16, 8, seed=5).to(torch.bfloat16)), _in_bf16(_mm)),
    "mm_bf16_67": (lambda a, b: (a @ b,), (M67B, _data(67, 5, seed=15).to(torch.bfloat16)), _in_bf16(_mm)),
    "bmm_bf16": (lambda a, b: (torch.bmm(a, b),),
                 (_data(2, 4, 16, seed=6).to(torch.bfloat16), _data(2, 16, 4, seed=7).to(torch.bfloat16)),
                 _in_bf16(lambda a, b: [w for i in range(a.shape[0]) for w in _mm(a[i], b[i])])),
    "silu_backward": (lambda g, x: (aten.silu_backward(g, x),), (Y, X), _silu_backward),
    "silu_backward_67": (lambda g, x: (aten.silu_backward(g, x),), (Y67, X67), _silu_backward),
    "silu_backward_bf16": (lambda g, x: (aten.silu_backward(g, x),), (Y.to(torch.bfloat16), XB), _in_bf16(_silu_backward)),
}


@pytest.mark.parametrize("name", sorted(EXACT))
def test_rule_bit_exact(name: str) -> None:
    f, xs = EXACT[name]
    _same_bits(*_run(f, *xs))


@functools.cache
def _close(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    f, xs, _ref = CLOSE[name]
    return _run(f, *xs)


@pytest.mark.parametrize("name", sorted(CLOSE))
def test_rule_is_its_stated_arithmetic(name: str) -> None:
    """A rule of the vocabulary's own arithmetic is that arithmetic bit for bit: its stated formula, composed from the evaluators
    of the Definitions it names in its stated order. A tolerance against torch can't tell a summation order, an fma from a
    multiply and an add, a reciprocal's product from a division, or a dropped mask, from the stated arithmetic."""
    _f, xs, ref = CLOSE[name]
    want = torch.from_numpy(np.array(ref(*xs), dtype=np.uint32).view(np.float32).copy())
    _same_bits(_close(name)[0], want)


@pytest.mark.parametrize("name", sorted(CLOSE))
def test_rule_close(name: str) -> None:
    """Against torch: f32 to a few ulps; bf16 to about one bf16 ulp, where the f32 results' difference crosses a rounding."""
    got, want = _close(name)
    torch.testing.assert_close(got, want, **(dict(rtol=1e-2, atol=1e-5) if "_bf16" in name else dict(rtol=2e-6, atol=1e-6)))


@pytest.mark.parametrize("f, x, why", [
    (lambda a: (a ** 3,), XB, "rounds in between"), (lambda a: (a ** -2,), POSB, "rounds in between"),
    (lambda a: (torch.rsqrt(a),), POSB, "rounds in between"), (lambda a: (a ** -0.5,), POSB, "rounds in between")])
def test_bf16_rounding_in_between_is_refused(f, x, why: str) -> None:
    """Where torch's bf16 kernel rounds otherwise than once after one f32 operation, the rules refuse: `pow` 3 and -2 round the
    square to bf16, and `rsqrt`'s scalar loop the square root."""
    gm = make_fx(f, tracing_mode="fake")(x)
    u = derive_graph(gm, (x,), mode="reference", rules=training_rules(), name="T").unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and why in u.hint, u


@pytest.mark.parametrize("f, x, why", [
    (lambda a: (torch.empty_like(a, dtype=torch.bfloat16).copy_(a),), X, "a conversion from torch.float32 to torch.bfloat16"),
    (lambda a: (a * (torch.full((16,), float("inf")) * 0.0),), X, "a static NaN result"),
    (lambda a: (a * 2.0, torch.full((4,), float("nan"))), X, "a returned static NaN"),
    (lambda a: (a * (torch.full((16,), 3.0, dtype=torch.bfloat16) * 0.1),), XB, "a static bf16 or f16 result")])
def test_copy_and_static_arithmetic_refusals(f, x, why: str) -> None:
    """A copy that rounds f32 to bf16 is the GPU's rounding, which `Copy` takes only at a compute capability; a static NaN result is
    refused (torch CPU's NaN word is not the GPU's), as is a returned static NaN, and so is a static bf16 result (torch CPU rounds
    a number in a bf16 add to bf16 first)."""
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(x)
    u = derive_graph(gm, (x,), mode="reference", rules=training_rules(), name="T").unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and why in u.hint, u


def test_an_allocation_read_before_a_copy_writes_it_is_refused() -> None:
    gm = make_fx(torch.func.functionalize(lambda a: (a.new_empty_strided((4, 16), (16, 1)) + a,), remove="mutations"),
                 tracing_mode="fake")(X)
    u = derive_graph(gm, (X,), mode="reference", rules=training_rules(), name="T").unsupported
    assert isinstance(u, Unsupported) and "Alloc" in u.hint, u


def _f32_of(s: float) -> torch.Tensor:
    return torch.tensor(s, dtype=torch.float32)


def _recip(s: float) -> torch.Tensor:
    return torch.tensor(np.float32(1.0) / np.float32(s))


#: the GPU's arithmetic (`training_rules(cc=...)`), each case with that arithmetic in torch CPU's f32 operations, which on NaN-free
#: words are the GPU's (its NaN words are the `_v2` Definitions'): a number is its f32 word, which a bf16 add doesn't round first,
#: and a division by a number is the product by its f32 reciprocal; an add with alpha is one fma, as on the CPU; addcdiv is
#: fma(value, t1 / t2, self), as torch CPU's addcmul by a value of 1 computes it
GPU: dict[str, tuple[Callable[..., object], tuple[torch.Tensor, ...], Callable[..., torch.Tensor]]] = {
    "add_number_bf16": (lambda a: (a + 0.1,), (XB,), lambda a: (a.float() + _f32_of(0.1)).bfloat16()),
    "sub_tie_bf16": (lambda a: (a - TIE,), (X67B,), lambda a: (a.float() - _f32_of(TIE)).bfloat16()),
    "mul_number_bf16": (lambda a: (a * 3.4445,), (XB,), lambda a: (a.float() * _f32_of(3.4445)).bfloat16()),
    "div_number_bf16": (lambda a: (a / 0.1,), (X67B,), lambda a: (a.float() * _recip(0.1)).bfloat16()),
    "div_number": (lambda a: (a / 3.0,), (X,), lambda a: a * _recip(3.0)),
    "div_static_vector": (lambda a: (a / torch.full((16,), 3.0),), (X,), lambda a: a / 3.0),
    "div_value_bf16": (lambda a, b: (a / b,), (XB, POSB[0, :1].reshape(())), lambda a, b: (a.float() / b.float()).bfloat16()),
    "add_alpha": (lambda a, b: (torch.add(a, b, alpha=0.95),), (X67, Y67), lambda a, b: torch.add(a, b, alpha=0.95)),
    "mul_mask": (lambda a, k: (a * k,), (M67, ROW67 > 0), lambda a, k: a * k),
    "foreach_add": (lambda a, s: torch._foreach_add([a, s], 1.0), (X67, torch.tensor(7.0)),
                    lambda a, s: torch.cat([(a + 1.0).reshape(-1), (s + 1.0).reshape(-1)])),
    "foreach_lerp": (lambda a, b, c, d: torch._foreach_lerp([a, c], [b, d], 0.1), (X67, Y67, X, Y),
                     lambda a, b, c, d: torch.cat([torch.lerp(a, b, 0.1).reshape(-1), torch.lerp(c, d, 0.1).reshape(-1)])),
    "lerp_large_weight": (lambda a, b: (torch.lerp(a, b, 0.9),), (X67, Y67), lambda a, b: torch.lerp(a, b, 0.9)),
    "addcdiv": (lambda a, b, c: (torch.addcdiv(a, b, c, value=-0.05),), (X67, Y67, POS67),
                lambda a, b, c: torch.addcmul(a, b / c, torch.full_like(a, -0.05))),
    "static_div_number": (lambda a: (a * (torch.full((16,), 3.0) / 7.0),), (X,),
                          lambda a: a * (torch.full((16,), 3.0) * _recip(7.0))),
    "copy_rounds_to_bf16": (lambda a, b: (torch.empty_like(b).copy_(a) * b,), (X, XB),
                            lambda a, b: (a.bfloat16().float() * b.float()).bfloat16()),
}


@pytest.mark.parametrize("name", sorted(GPU))
def test_gpu_binary_is_the_gpus_arithmetic(name: str) -> None:
    f, xs, gpu = GPU[name]
    got = _outputs(_program(f, *xs, rules=training_rules(cc=CB.TRAIN_CC)), *xs)
    _same_bits(got, gpu(*xs).float().reshape(-1))


def test_gpu_and_cpu_numbers_differ() -> None:
    """The cases where the GPU's arithmetic is not the CPU's give different words on their data: a bf16 add of a number bf16
    doesn't hold, and an f32 division by a number, of a value or of a static tensor (a bf16 one rounds both to the same word)."""
    for name in ("add_number_bf16", "sub_tie_bf16", "div_number", "static_div_number"):
        f, xs, _gpu = GPU[name]
        cpu, gpu = (_outputs(_program(f, *xs, rules=r), *xs) for r in (None, training_rules(cc=CB.TRAIN_CC)))
        assert not torch.equal(cpu.view(torch.int32), gpu.view(torch.int32)), name


@pytest.mark.parametrize("n", [1, 7, 33, 1000])
def test_add_alpha_is_one_fma_at_every_length(n: int) -> None:
    """An f32 add with alpha is one fma at every element, torch CPU's vectorized body and its scalar tail alike (lengths below,
    between and above its vector widths); on the GPU the kernel's `a + alpha * b` is one FFMA, which a capture checks."""
    a, b = _data(n, seed=30), _data(n, seed=31)
    for rules in (None, training_rules(cc=CB.TRAIN_CC)):
        _same_bits(*_run(lambda x, y: (torch.add(x, y, alpha=-0.0123),), a, b, rules=rules))


def test_gpu_binary_definitions_and_refusals() -> None:
    """At `cc` every element-wise add, sub, mul and div is the GPU's Definition (`_v2`, NaN `F32_NAN_PTX`; a bf16 result
    `F2fpBf16_v1`), a same-shape product included, which without `cc` the default rules' `MulRef_v1` computes; a division by a
    0-d static tensor is refused at `cc`, and a bf16 add with alpha everywhere."""
    def f(a, b, c, d):
        return a * b, a * 2.0 + 1.0, torch.add(a, b, alpha=0.5), a - b, c / d

    ids = [{nd.fn.id.split("{")[0] for nd in _program(f, X, Y, XB, POSB, rules=r).root.body.nodes}
           for r in (None, training_rules(cc=CB.TRAIN_CC))]
    assert {"MulRef_v1", "F32Mul_v1", "F32Add_v1", "F32Fma_v1", "F32Sub_v1", "F32Div_v1", "F32ToBf16Rn_v1"} <= ids[0], ids[0]
    assert {"F32Mul_v2", "F32Add_v2", "F32Fma_v2", "F32Sub_v2", "F32Div_v2", "F2fpBf16_v1"} <= ids[1], ids[1]
    assert not {"MulRef_v1", "F32Mul_v1", "F32Add_v1", "F32Fma_v1", "F32Sub_v1", "F32Div_v1", "F32ToBf16Rn_v1"} & ids[1], ids[1]
    assert "0-d static" in _gpu_refusal(lambda a: (a / torch.full((), 3.0),), X).hint
    gm = make_fx(lambda a, b: (torch.add(a, b, alpha=0.5),), tracing_mode="fake")(XB, XB)
    u = derive_graph(gm, (XB, XB), mode="reference", rules=training_rules(), name="T").unsupported
    assert isinstance(u, Unsupported) and "alpha != 1" in u.hint, u


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_gpu_silu_backward_is_its_stated_arithmetic(dt: torch.dtype) -> None:
    """At `cc` silu_backward is its stated form in the GPU's Definitions (`_v2`; `NvExpf_v1`, `Unary`'s exp; a bf16 result
    `F2fpBf16_v1`), bit for bit at an odd length."""
    dy, x = Y67.to(dt), X67.to(dt)
    prog = _program(lambda g, v: (aten.silu_backward(g, v),), dy, x, rules=training_rules(cc=CB.TRAIN_CC))
    ids = {nd.fn.id.split("{")[0] for nd in prog.root.body.nodes}
    assert {"F32Add_v2", "F32Div_v2", "F32Sub_v2", "F32Fma_v2", "F32Mul_v2", "NvExpf_v1"} <= ids, ids
    assert not {"F32Add_v1", "F32Div_v1", "F32Sub_v1", "F32Fma_v1", "F32Mul_v1", "F32ToBf16Rn_v1", "F32ExpRn_v1"} & ids, ids
    want = _silu_backward_with(P.F32AddV2, P.F32DivV2, P.F32SubV2, P.F32FmaV2, P.F32MulV2, _nv_expf)(dy.float(), x.float())
    if dt == torch.bfloat16:
        want = [P.F2fpBf16.evaluate(w) << 16 for w in want]
    _same_bits(_outputs(prog, dy, x), torch.from_numpy(np.array(want, dtype=np.uint32).view(np.float32).copy()))


#: words where `expf`'s edges are: ±0, ±inf, NaN, a subnormal, the overflow and underflow edges and their neighbours
EXP_EDGES = torch.from_numpy(np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x42B17217, 0x42B17218,
                                       0xC2CFF1B4, 0xC2CFF1B5, 0xC2AEAC4F, 0xC2AEAC50, 0x3F800000, 0xBF800000],
                                      dtype=np.uint32).view(np.float32).copy())


def _unary(op):
    """`op` as a one-input step (`make_fx` counts a defaulted parameter as an input)."""
    return lambda v: (op(v),)


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_gpu_unary_is_its_stated_arithmetic(dt: torch.dtype) -> None:
    """At `cc` exp is `NvExpf_v1`, sigmoid `F32Div_v2(1, F32Add_v2(NvExpf_v1(-x), 1))` and reciprocal `F32Div_v2(1, x)`, a bf16
    result `F2fpBf16_v1`, bit for bit at an odd length and at expf's edges; without `cc` exp stays `F32ExpRn_v1`."""
    x = torch.cat([X67, EXP_EDGES]).to(dt)
    xs = [P.Bf16ToF32.evaluate(v) for v in _bits(x)] if dt == torch.bfloat16 else _bits(x)
    cases = {"exp": (torch.exp, _nv_expf),
             "sigmoid": (torch.sigmoid, lambda v: P.F32DivV2.evaluate(ONE, P.F32AddV2.evaluate(_nv_expf(SC.F32Neg.evaluate(v)), ONE))),
             "reciprocal": (torch.reciprocal, lambda v: P.F32DivV2.evaluate(ONE, v))}
    for name, (op, word) in cases.items():
        prog = _program(_unary(op), x, rules=training_rules(cc=CB.TRAIN_CC))
        ids = {nd.fn.id.split("{")[0] for nd in prog.root.body.nodes}
        assert not {"F32Div_v1", "F32Add_v1", "F32ExpRn_v1", "F32ToBf16Rn_v1"} & ids, (name, ids)
        want = [word(v) for v in xs]
        if dt == torch.bfloat16:
            want = [P.F2fpBf16.evaluate(w) << 16 for w in want]
        _same_bits(_outputs(prog, x), torch.from_numpy(np.array(want, dtype=np.uint32).view(np.float32).copy()))
    cpu = {nd.fn.id.split("{")[0] for nd in _program(lambda v: (torch.exp(v),), x).root.body.nodes}
    assert "F32ExpRn_v1" in cpu and "NvExpf_v1" not in cpu, cpu


FUSED_ADAMW = dict(lr=1e-3, beta1=0.9, beta2=0.95, weight_decay=0.1, eps=1e-8, amsgrad=False, maximize=False)


def _fused_adamw_step(**kw):
    """`_fused_adamw` over two tensors: the new parameters, first moments and second moments. The step takes positional
    arguments only, since tracing a function with `**kwargs` behind functionalize's wrapper miscounts its inputs."""

    def step(p0, p1, g0, g1, m0, m1, v0, v1, s0, s1):
        out = aten._fused_adamw([p0, p1], [g0, g1], [m0, m1], [v0, v1], [], [s0, s1], **(FUSED_ADAMW | kw))
        return [*out[0], *out[2], *out[3]]

    return step


def _fused_adamw_arithmetic(ps, gs, ms, vs, steps, lr, beta1, beta2, weight_decay, eps, **_):
    """`FusedAdamW`: per tensor `1 - beta^step` = `F32Sub_v2(1, F32ExpRn_v1(F32Mul_v2(step, F32LogRn_v1(beta))))`, the step size
    `F32Div_v2(lr, 1 - beta1^step)` and `F32Sqrt_v1(1 - beta2^step)`; per element p1 = `F32Fma_v2(-(lr * wd), p, p)`, m1 =
    `F32Fma_v2(beta1, m, F32Fma_v2(-beta1, g, g))`, v1 the same of `F32Mul_v2(g, g)` with beta2, and `p1 - (step size * m1) /
    (sqrt(v1) / sqrt(1 - beta2^step) + eps)` in `F32Sub_v2`, `F32Div_v2`, `F32Mul_v2`, `F32Sqrt_v1` and `F32Add_v2`."""
    lr, b1, b2, wd, eps = (np.float32(x) for x in (lr, beta1, beta2, weight_decay, eps))

    def one_minus_pow(beta, s):
        return P.F32SubV2.evaluate(ONE, R.F32ExpRn.evaluate(P.F32MulV2.evaluate(s, R.f32_log_rn_bits(_f32(beta)))))

    new_p, new_m, new_v = [], [], []
    for p, g, m, v, s in zip(ps, gs, ms, vs, steps, strict=True):
        s = _bits(s)[0]
        size, bc2 = P.F32DivV2.evaluate(_f32(lr), one_minus_pow(b1, s)), R.F32Sqrt.evaluate(one_minus_pow(b2, s))
        for pi, gi, mi, vi in zip(_bits(p), _bits(g), _bits(m), _bits(v), strict=True):
            p1 = P.F32FmaV2.evaluate(_f32(-(lr * wd)), pi, pi)
            m1 = P.F32FmaV2.evaluate(_f32(b1), mi, P.F32FmaV2.evaluate(_f32(-b1), gi, gi))
            gg = P.F32MulV2.evaluate(gi, gi)
            v1 = P.F32FmaV2.evaluate(_f32(b2), vi, P.F32FmaV2.evaluate(_f32(-b2), gg, gg))
            den = P.F32AddV2.evaluate(P.F32DivV2.evaluate(R.F32Sqrt.evaluate(v1), bc2), _f32(eps))
            new_p.append(P.F32SubV2.evaluate(p1, P.F32DivV2.evaluate(P.F32MulV2.evaluate(size, m1), den)))
            new_m.append(m1)
            new_v.append(v1)
    return new_p + new_m + new_v


def test_gpu_fused_adamw_is_its_kernels_arithmetic() -> None:
    """At `cc` a fused AdamW step is its CUDA kernel's arithmetic in the GPU's Definitions, with `beta^step` as
    `exp(step * log(beta))` (CUDA's `powf` is not modeled), bit for bit over two tensors of different lengths and step counts, and
    close to torch's own fused AdamW; it is refused without `cc` (torch CPU's kernel is another arithmetic) and with maximize, as
    with amsgrad, a gradient scale or an inf flag."""
    ps, gs, ms = ([_data(67, seed=40 + k), _data(4, 16, seed=43 + k)] for k in range(3))
    vs = [_data(67, positive=True, seed=46) * 1e-3, _data(4, 16, positive=True, seed=47) * 1e-3]
    steps = [torch.tensor(3.0), torch.tensor(1.0)]
    xs = (*ps, *gs, *ms, *vs, *steps)
    prog = _program(_fused_adamw_step(), *xs, rules=training_rules(cc=CC))
    ids = {nd.fn.id.split("{")[0] for nd in prog.root.body.nodes}
    assert {"F32Fma_v2", "F32Mul_v2", "F32Div_v2", "F32Add_v2", "F32Sub_v2", "F32Sqrt_v1", "F32ExpRn_v1"} <= ids, ids
    got = _outputs(prog, *xs)
    want = _fused_adamw_arithmetic(ps, gs, ms, vs, steps, **FUSED_ADAMW)
    _same_bits(got, torch.from_numpy(np.array(want, dtype=np.uint32).view(np.float32).copy()))
    torch.testing.assert_close(got, torch.cat([t.reshape(-1) for t in _fused_adamw_step()(*xs)]), rtol=1e-5, atol=1e-6)
    for rules, kw, why in ((training_rules(), {}, "computes in double"), (training_rules(cc=CC), {"maximize": True}, "maximize")):
        gm = make_fx(torch.func.functionalize(_fused_adamw_step(**kw), remove="mutations"), tracing_mode="fake")(*xs)
        u = derive_graph(gm, xs, mode="reference", rules=rules, name="T").unsupported
        assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and why in u.hint, u


def _adamw_inputs(seed: int = 0, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    p, grad, m = (torch.randn(n, generator=g) for _ in range(3))
    v = torch.rand(n, generator=g) * 1e-3
    return p, grad, m, v


def test_functional_adamw_is_torch_adamw() -> None:
    opt_args = dict(lr=ADAMW["lr"], betas=ADAMW["betas"], eps=ADAMW["eps"], weight_decay=ADAMW["weight_decay"])
    p, grad, m, v = _adamw_inputs()
    t = torch.nn.Parameter(p.clone())
    opt = torch.optim.AdamW([t], foreach=False, **opt_args)
    f = AdamW.from_torch(opt)
    pp, mm, vv = p.clone(), m.clone(), v.clone()
    for step in (1, 2, 3):
        t.grad = grad * step
        if t not in opt.state:
            opt.state[t] = {"step": torch.tensor(0.0), "exp_avg": m.clone(), "exp_avg_sq": v.clone()}
        opt.step()
        s = {k: torch.tensor(x, dtype=torch.float32) for k, x in f.scalars(step).items()}
        pp, mm, vv = f.update(pp, grad * step, mm, vv, s["decay"], s["neg_step_size"], s["bc2_sqrt"])
        for got, want in ((pp, t.detach()), (mm, opt.state[t]["exp_avg"]), (vv, opt.state[t]["exp_avg_sq"])):
            _same_bits(got, want)


@pytest.mark.parametrize("n", [64, 67])
def test_adamw_update_program_bit_exact(n: int) -> None:
    """The derived optimizer update (decoupled decay, lerp, addcmul, sqrt, division, addcdiv) is torch's (with IEEE sqrt), word for
    word."""
    f = AdamW(**ADAMW)
    p, grad, m, v = _adamw_inputs(1, n)
    s = [torch.tensor(x, dtype=torch.float32) for x in f.scalars(1).values()]
    _same_bits(*_run(lambda *a: f.update(*a), p, grad, m, v, *s))


def test_derive_graph_names_inputs_and_theta() -> None:
    prog = _program(lambda w, x: (x * w + 1.0,), ROW, X, theta=("x0",))
    assert prog.theta == ("x0",)
    assert [n for n, _ in prog.fn.signature()[0]] == ["x0", "x1"]
    assert len(prog.root.theta) == 1 and prog.annotations["theta"] == sorted(prog.root.theta)
    gm = make_fx(lambda w, x: (x * w,), tracing_mode="fake")(ROW, X)
    with pytest.raises(ValueError, match="theta names inputs"):
        derive_graph(gm, (ROW, X), mode="reference", rules=training_rules(), input_names=["w", "x"], theta=("v",))
    with pytest.raises(ValueError, match="theta must name each input once"):
        derive_graph(gm, (ROW, X), mode="reference", rules=training_rules(), input_names=["w", "x"], theta=("w", "w"))
    with pytest.raises(ValueError, match="Python identifiers"):
        derive_graph(gm, (ROW, X), mode="reference", rules=training_rules(), input_names=["w", "class"])


def _z_loss(model, x):
    """The cross-entropy with a z-loss, each row's `logsumexp` squared: a loss with one op no rule lowers (`logsumexp`, whose
    backward's ops the rules lower)."""
    lg = model(x[:, :-1]).float()
    lg = lg.reshape(-1, lg.shape[-1])
    return torch.nn.functional.cross_entropy(lg, x[:, 1:].reshape(-1)) + 1e-4 * torch.logsumexp(lg, -1).square().mean()


@functools.cache
def _tiny_survey():
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64)
    opt = AdamW(**ADAMW)
    ex = export_step(model, _z_loss, opt, batch)
    return model, ex[2], step_report(model, _z_loss, opt, batch, survey=True, exported=ex)


@functools.cache
def _tiny_program():
    """The tiny step with `square_loss`: a Program, every op derived."""
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64)
    prog = step_program(model, square_loss, AdamW(**ADAMW), batch)
    assert not isinstance(prog, Unsupported), prog
    return model, prog


def test_tiny_step_survey_lists_its_one_hole_and_holds_no_program() -> None:
    model, layout, s = _tiny_survey()
    assert [(h["op"], h["reason"]) for h in s.holes] == [("aten.logsumexp.default", "numerics_unregistered")]
    assert s.holes[0]["fields"] == [f"hole0.0:{s.holes[0]['node']}"]
    assert (s.unsupported.reason, s.unsupported.node) == ("numerics_unregistered", f"{s.holes[0]['node']}: aten.logsumexp.default")
    assert not hasattr(s, "program") and not hasattr(s, "result") and s.refused is None
    assert s.census["totals"]["gates"] > 0 and "F32Add_v1" in s.primitives
    assert s.theta["names"] == [f"param_{n}".replace(".", "_") for n in layout.params]
    assert s.theta["bits"] == 32 * sum(p.numel() for p in model.parameters())


def test_tiny_bf16_step() -> None:
    """The tiny step with its linears' operands cast to bf16, as bf16 autocast casts them (`LM(matmul_dtype=torch.bfloat16)`). Every
    op derives with either loss, each bf16 matrix product is `LinearRef_v1` rows of bf16 words (attention's are f32: torch's math
    path computes it from upcast operands), and the casts are among the primitives."""
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=torch.bfloat16)
    opt = AdamW(**ADAMW)
    ex = export_step(model, square_loss, opt, batch)
    prog = step_report(model, square_loss, opt, batch, exported=ex).result
    assert not isinstance(prog, Unsupported), prog
    bf16 = ["DT='bf16'" in nd.fn.id for nd in prog.root.body.nodes if nd.fn.id.startswith("LinearRef_v1")]
    mms = sum(n.target is aten.mm.default and n.meta["val"].dtype == torch.bfloat16 for n in ex[0].graph.nodes)
    assert sum(bf16) == mms > 0 and not all(bf16), (sum(bf16), len(bf16), mms)
    s = step_report(model, loss_fn, opt, batch, survey=True, exported=export_step(model, loss_fn, opt, batch))
    assert s.unsupported is None and s.holes == [], (s.unsupported, s.holes)
    assert {P.Bf16ToF32.id, P.F32ToBf16Rn.id} <= set(s.primitives)


CC = CB.TRAIN_CC


@pytest.mark.parametrize("kw", [{}, {"cc": CC}, {"cc": CC, "tf32": True}], ids=["cpu", "cc", "cc-tf32"])
def test_training_rules_build_on_the_torch_only_list(kw: dict) -> None:
    """`verity_vllm.program.frontend.rules.training.training_rules` holds every rule of `rules.TORCH_RULES`, in that list's order,
    and none of `rules.VLLM_RULES`: a training step's rule set names no op of vLLM's."""
    rules = training_rules(**kw).rules
    assert [r for r in rules if any(r is t for t in TORCH_RULES)] == list(TORCH_RULES)
    assert not {r.rule_id for r in rules} & {r.rule_id for r in VLLM_RULES}


def _linear_gemms(x, w, dy):
    """F.linear's three GEMMs at x (M, K), W (N, K), laid out as autograd lays them out: x @ W^T, dY @ W and dY^T @ X."""
    return x @ w.t(), dy @ w, dy.t() @ x


def _meta(*shapes) -> list[torch.Tensor]:
    return [torch.empty(*s, dtype=torch.bfloat16, device="meta") for s in shapes]


def _gpu_refusal(f, *xs) -> Unsupported:
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(*xs)
    u = derive_graph(gm, xs, mode="reference", rules=training_rules(cc=CC), name="T").unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered", u
    return u


@pytest.mark.parametrize("K, N", CB.LF_LINEARS)
def test_gpu_gemms_are_the_captured_routes(K: int, N: int) -> None:
    """On sm_120 the lf model's s step's linears at B 2 x S 1024: each of F.linear's GEMMs, its role read off its operands'
    layouts, is one batch over its rows of its captured route's Definition on words."""
    M = CB.LF_M[0]
    prog = _program(_linear_gemms, *_meta((M, K), (N, K), (M, N)), rules=training_rules(cc=CC))
    calls = [(nd.fn.id, nd.n) for nd in prog.root.body.nodes[3:]]
    assert calls == [(word_of(CB.train_linear(r, M, K, N)).id, rows) for r, rows in (("fwd", M), ("dX", M), ("dW", N))]


#: stand-in routes at M 32, K 64, N 48, each a kind the capture fitted somewhere: (role, its route)
TINY_ROUTES = {"fwd": {"kernel": "nosplit"}, "dX": {"kernel": "serialk", "SPLIT_K": 3},
               "dW": {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"}}
#: the grid.z kinds at the same shapes, a chunk that isn't whole k-tiles taking its residue words, then constant-0 words to a
#: whole tile: an f32 workspace over chunks of [24, 24, 16] at k-tile 16, none over [24, 24], and a bf16 workspace over
#: [16, 16] at k-tile 32
TINY_GRID_Z_ROUTES = {"fwd": {"kernel": "splitk", "SPLIT_K": 3, "partial": "f32", "align": 8, "tile_k": 16},
                      "dX": {"kernel": "serialk", "SPLIT_K": 2, "align": 8, "tile_k": 16},
                      "dW": {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16", "align": 8, "tile_k": 32}}


@pytest.mark.parametrize("routes", [TINY_ROUTES, TINY_GRID_Z_ROUTES], ids=["split", "grid-z"])
def test_gpu_gemms_are_their_routes_bit_for_bit(monkeypatch, routes) -> None:
    """At stand-in routes, the derived GEMMs evaluated on words are each route's model (`bf16_gemm_check.cublas_route_rows`),
    the one the capture's words hold to sm_120's (`test_cublas_train`), and dX's and dW's are not the unsplit chain's (an f32
    workspace's words differ from it only in f32 low bits, which the bf16 rounding hides on these words, and
    `test_cublas_train`'s cancelling words tell apart)."""
    M, K, N = 32, 64, 48
    for role, route in routes.items():
        monkeypatch.setitem(CB.TRAIN_ROUTES, (M, K, N, role), route)
    xs = [_data(*s, seed=20 + i).to(torch.bfloat16) for i, s in enumerate(((M, K), (N, K), (M, N)))]
    prog = _program(_linear_gemms, *xs, rules=training_rules(cc=CC))
    tab = materialize(prog, [_bits(x) for x in xs])
    got = np.array([tab[g] for g in prog.circuit.root_outputs()], dtype=np.uint16)
    x, w, dy = (t.view(torch.int16).numpy().astype(np.uint16) for t in xs)
    operands = {"fwd": (x, w), "dX": (dy, w.T), "dW": (dy.T, x.T)}
    want = {role: BC.cublas_route_rows(routes[role], *operands[role]) for role in ("fwd", "dX", "dW")}
    assert np.array_equal(got, np.concatenate([want[role].reshape(-1) for role in ("fwd", "dX", "dW")]))
    for role in ("dX", "dW"):
        assert not np.array_equal(BC.linear_rows(*operands[role]), want[role]), f"the routes differ at {role}"


#: stand-in routes for the tiny step's linears at M 32: unsplit, but the attention projections' dW (split-K over 2 slices with a
#: bf16 workspace) and the MLP up projections' dX (serial split-K over 3)
TINY_STEP_ROUTES = {**{(32, K, N, r): {"kernel": "nosplit"} for K, N in ((64, 64), (64, 192), (192, 64)) for r in ("fwd", "dX", "dW")},
                    (32, 64, 64, "dW"): {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"},
                    (32, 64, 192, "dX"): {"kernel": "serialk", "SPLIT_K": 3}}


def test_tiny_bf16_step_on_sm120(monkeypatch) -> None:
    """The tiny bf16 step on sm_120 at stand-in routes (`TINY_STEP_ROUTES`): each linear's three GEMMs are their routes'
    Definitions on words, read off the layouts autograd gives them (nine linears at (K, N) = (64, 64), the lm_head among them,
    four at (64, 192), two at (192, 64)), and attention's f32 products are `LinearRef_v1` rows as without `cc`."""
    for key, route in TINY_STEP_ROUTES.items():
        monkeypatch.setitem(CB.TRAIN_ROUTES, key, route)
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=torch.bfloat16)
    prog = step_program(model, square_loss, AdamW(**ADAMW), batch, cc=CC)
    assert not isinstance(prog, Unsupported), prog
    got = collections.Counter(nd.fn.id for nd in prog.root.body.nodes if nd.fn.id.split("{")[0] in (
        "Gemm_v2", "CublasSplitKGemmBf16Partials_v1", "CublasSerialSplitKGemm_v1", "LinearRef_v1"))
    want: collections.Counter[str] = collections.Counter()
    for (K, N), linears in {(64, 64): 9, (64, 192): 4, (192, 64): 2}.items():
        for role in ("fwd", "dX", "dW"):
            want[word_of(CB.train_linear(role, 32, K, N)).id] += linears
    assert {k: v for k, v in got.items() if not k.startswith("LinearRef_v1")} == want
    assert all("DT='f32'" in k for k in got if k.startswith("LinearRef_v1")), got


def test_gpu_lowers_smollm2s_grid_z_dw_to_its_route() -> None:
    """smollm2's dW at B 8 x S 512, sliced over grid.z, is one batch over its rows of its route's Definition on words."""
    prog = _program(lambda dy, x: (dy.t() @ x,), *_meta((4096, 576), (4096, 576)), rules=training_rules(cc=CC))
    assert [(nd.fn.id, nd.n) for nd in prog.root.body.nodes[2:]] == [(word_of(CB.train_linear("dW", 4096, 576, 576)).id, 576)]
    assert word_of(CB.train_linear("dW", 4096, 576, 576)).id.startswith("CublasSplitKGemmBf16PartialsAligned_v1{K=4096,N=576,")


def test_gpu_refuses_what_its_capture_does_not_cover(monkeypatch) -> None:
    """On sm_120 a bf16 product is refused, with its fix, where no captured route lowers it: a shape the capture doesn't cover, a
    layout none of F.linear's GEMMs has, a bmm, and a route with no Definition (split-K over `split_bounds`' chunks with an f32
    workspace, which no cc 12.0 capture fitted); f32 products are as without `cc`, and a `cc` without captured routes is
    refused."""
    assert "is not captured" in _gpu_refusal(_linear_gemms, *_meta((32, 64), (48, 64), (32, 48))).hint
    assert "operand layouts ('col', 'col')" in _gpu_refusal(lambda a, b: (a.t() @ b.t(),), *_meta((64, 32), (48, 64))).hint
    assert "only F.linear's GEMMs" in _gpu_refusal(lambda a, b: (torch.bmm(a, b),), *_meta((2, 16, 32), (2, 32, 16))).hint
    monkeypatch.setitem(CB.TRAIN_ROUTES, (4096, 576, 576, "dW"), {"kernel": "splitk", "SPLIT_K": 2})
    assert "has no Definition on bits" in _gpu_refusal(lambda dy, x: (dy.t() @ x,), *_meta((4096, 576), (4096, 576))).hint
    b = _data(16, 8, seed=5)
    ids = [[nd.fn.id for nd in _program(lambda x, y: (x @ y,), X, b, rules=r).root.body.nodes[2:]] for r in (None, training_rules(cc=CC))]
    assert ids[0] == ids[1] and ids[0][0].startswith("LinearRef_v1") and "DT='f32'" in ids[0][0], ids
    with pytest.raises(ValueError, match="no captured cuBLAS routes"):
        training_rules(cc=(9, 0))
    with pytest.raises(ValueError, match="rules or cc"):
        step_report(None, None, None, None, rules=training_rules(), cc=CC)


class _DeviceBranch(torch.nn.Module):
    """A product whose operands' dtype the device decides, as Newton-Schulz's `x = g.bfloat16() if g.is_cuda else g.float()`."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(32, 64, device="meta"))

    def forward(self, x):
        x, w = (x.bfloat16(), self.w.bfloat16()) if x.is_cuda else (x.float(), self.w.float())
        return x @ w.t()


def _sum_of_squares(model, batch):
    return model(batch).float().square().sum()


def test_gpu_export_takes_the_steps_cuda_branches() -> None:
    """`export_step(..., gpu=True)` traces the step as the GPU runs it: the step's own `is_cuda` reads True, so a product it
    computes in bf16 on a GPU and in f32 elsewhere is bf16, forward and backward; without `gpu` it is f32; and afterwards
    `Tensor.is_cuda` is torch's again."""
    batch = torch.empty(4, 64, device="meta")

    def dtypes(gpu: bool) -> set:
        gm = export_step(_DeviceBranch(), _sum_of_squares, AdamW(**ADAMW), batch, gpu=gpu)[0]
        return {n.args[0].meta["val"].dtype for n in gm.graph.nodes if n.target is aten.mm.default}

    assert dtypes(False) == {torch.float32} and dtypes(True) == {torch.bfloat16}
    assert not isinstance(torch.Tensor.__dict__.get("is_cuda"), property) and torch.zeros(1).is_cuda is False


def test_a_patch_is_put_back_as_the_owner_held_it() -> None:
    """The translator's own patch (`train_step._patched`, which the GPU trace puts on `Tensor.is_cuda`): inside the block the
    owner answers with the value; afterwards an attribute of its own is that attribute again and one it inherited is inherited
    again, also when the block raises."""
    class Base:
        inherited = "the base's"

    class Owner(Base):
        own = "the owner's"

    with _patched(Owner, "own", "patched"), _patched(Owner, "inherited", "patched"):
        assert (Owner.own, Owner.inherited) == ("patched", "patched")
    assert vars(Owner)["own"] == "the owner's" and "inherited" not in vars(Owner) and Owner.inherited == "the base's"
    with pytest.raises(RuntimeError, match="inside the block"), _patched(Owner, "inherited", "patched"):
        raise RuntimeError("inside the block")
    assert "inherited" not in vars(Owner) and Owner.inherited == "the base's"


def test_gpu_export_reads_the_steps_tf32(monkeypatch) -> None:
    """At a cc the step's cuda.matmul.allow_tf32 is the rules' `tf32`, as `export_step(..., gpu=True)` reads it: under it an f32
    linear's product is cuBLAS's TF32 GEMM, which `Matmul` refuses; a step that changes it between its products is refused, as
    is an export traced without `gpu`."""
    model, batch, opt = torch.nn.Linear(64, 32, bias=False, device="meta"), torch.empty(4, 64, device="meta"), AdamW(**ADAMW)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    ex = export_step(model, _sum_of_squares, opt, batch, gpu=True)
    assert ex[2].gpu and ex[2].tf32
    u = step_report(model, _sum_of_squares, opt, batch, exported=ex, cc=CC).unsupported
    assert u is not None and "TF32 GEMM" in u.hint, u
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    assert not export_step(model, _sum_of_squares, opt, batch, gpu=True)[2].tf32

    def toggles(m, b):
        y = m(b)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return (m(b) + y).square().sum()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False

    with pytest.raises(ValueError, match="changes cuda.matmul.allow_tf32"):
        export_step(model, toggles, opt, batch, gpu=True)
    with pytest.raises(ValueError, match=r"export_step\(\.\.\., gpu=True\)"):
        step_report(model, _sum_of_squares, opt, batch, exported=export_step(model, _sum_of_squares, opt, batch), cc=CC)


def test_tiny_step_theta_is_the_parameters() -> None:
    model, prog = _tiny_program()
    layout = prog.annotations["train_step"]
    assert prog.theta == tuple(f"param_{n}".replace(".", "_") for n in layout["params"])
    nodes = prog.root.body.nodes
    assert sum(nodes[i].n * nodes[i].fn.ret.w for i in prog.root.theta) == 32 * sum(p.numel() for p in model.parameters())


def test_a_program_with_holes_is_no_statement() -> None:
    """A survey's holes are fields of the `run` input that no gate checks: its Program is marked with them, `check_identity`
    refuses it (and so the Boolean lowering does), and a derivation that leaves holes returns no Program."""
    gm = make_fx(lambda x: (torch.cos(x) * 2.0,), tracing_mode="fake")(X)
    hs, prog = survey(gm, (X,), mode="reference", rules=training_rules(), name="T")
    assert [(h["op"], h["fields"]) for h in hs] == [("aten.cos.default", ["hole0.0:cos"])]
    assert prog.annotations["survey_holes"] == hs
    with pytest.raises(FrontendError, match="E_SURVEY_HOLES"):
        check_identity(prog)
    with pytest.raises(FrontendError, match="E_SURVEY_HOLES"):
        derived(DeriveReport(prog, None, {"result": {}}))
    rep = derive_graph(gm, (X,), mode="reference", rules=with_holes(training_rules()), name="T")
    assert rep.program is None and rep.unsupported.reason == "side_input" and "E_SURVEY_HOLES" in rep.unsupported.hint, rep.unsupported


def _sources(node) -> set[int]:
    return {k for a in node.args for _s, k, _a, _b in ref_runs(a.refs)}


def _walk_in_bits(prog, S: GateSet, structure: set[int]) -> int:
    """`in_bits_beyond_theta` by brute force: every operand of every gate of S, less S, theta's input nodes and `structure`."""
    offs = prog.root.body.offsets
    skip = structure | set(prog.root.theta)
    reads = {o for lo, hi in S.intervals() for g in range(lo, hi) for o in prog.circuit.gate(g).operands if o not in S}
    return sum(gate_width(prog, o) for o in reads if bisect.bisect_right(offs, o) - 1 not in skip)


def test_witness_block_in_bits_beyond_theta() -> None:
    """A witness block is a GateSet over the step Program (`query_ast`): its gates are `size()`, and `in_bits_beyond_theta` is its
    reads less the start's weights and structure (constants and what is computed from them alone), as a walk over its gates'
    operands finds them. A forward projection's row reads its input row and not its weight; the optimizer's update of one parameter
    reads the gradient, the two moments and the step's three scalars, and not the parameter, the constants or its own values."""
    prog = _tiny_program()[1]
    nodes = prog.root.body.nodes
    structure: set[int] = set()
    while more := {j for j, nd in enumerate(nodes) if j not in structure and not is_input(nd.fn) and _sources(nd) <= structure}:
        structure |= more
    lin = evaluate(prog, parse('instances("//LinearRef_v1{*}")')).by_index(0)
    assert in_bits_beyond_theta(prog, lin) == _walk_in_bits(prog, lin, structure) == 64 * 32
    names = [n for n, _ in prog.fn.signature()[0]]
    p, decay = names.index("param_norm"), names.index("decay")
    first = [j for j, nd in enumerate(nodes) if nd.fn.id == "F32Mul_v1" and len(_sources(nd) & set(prog.root.theta)) == 1
             and decay in _sources(nd)]
    start = next(j for j in first if p in _sources(nodes[j]))
    end = min([j for j in first if j > start] + [len(nodes)])
    root = prog.circuit.root_scope
    update = GateSet((Part(root, frozenset((j, m) for j in range(start, end) for m in range(n_members(nodes[j])))),))
    assert in_bits_beyond_theta(prog, update) == _walk_in_bits(prog, update, structure) == 3 * 64 * 32 + 3 * 32
    iota = {j for j in structure if _sources(nodes[j])}
    reader = next(j for j, nd in enumerate(nodes) if j not in structure and _sources(nd) & iota)
    one = GateSet((Part(root, frozenset((reader, m) for m in range(n_members(nodes[reader])))),))
    assert in_bits_beyond_theta(prog, one) == _walk_in_bits(prog, one, structure)


@pytest.mark.slow
def test_s_step_reference_mode() -> None:
    """The brief's step: the lf model's s size (d 512, 6 layers, 8 heads, SwiGLU 1408, tied embeddings, vocabulary 50,304, context
    1024), batch 2, plain AdamW, in reference mode, with the z-loss (`_z_loss`). Every op of its forward and its cross-entropy
    derives, and the step is refused at the z-loss's `logsumexp`, which no rule lowers, before its backward."""
    model, batch = meta_step("s", 2)
    out = step_program(model, _z_loss, AdamW(**ADAMW), batch, mode="reference")
    assert isinstance(out, Unsupported), out
    assert (out.reason, out.node) == ("numerics_unregistered", "logsumexp: aten.logsumexp.default"), out
