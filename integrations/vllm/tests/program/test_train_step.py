"""The training-step frontend (`program.frontend.train_step`, `derive_graph`, the training rules): each training rule's Program run
on small CPU tensors of even and odd lengths, bit for bit against torch where the rule states torch CPU's own f32 operation (in
bf16, its f32 operation on the widened operands rounded once), and where the reference vocabulary's arithmetic is its own
(sequential sums, a correctly rounded exp) bit for bit against the rule's stated arithmetic, composed from the evaluators of the
Definitions it names, and to a tolerance against torch; the functional AdamW against `torch.optim.AdamW(foreach=False)`;
`BoxPermute` against torch's permute; the tiny step in f32 and in bf16 (its linears' operands cast, as autocast casts them); and
the lf model's s step derived in reference mode, which is refused at its one op with no Definition (`_log_softmax`: `F32LogRn`)."""
from __future__ import annotations

import bisect
import functools
import math

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

from tests.program.train_step_lf import ADAMW, loss_fn, meta_step, square_loss
from verity.ml.circuits.boundary import gate_width
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.parts import Part, n_members
from verity.ml.circuits.program import is_input
from verity.ml.circuits.query_ast import GateSet, evaluate, parse
from verity.ml.circuits.refs import runs as ref_runs
from verity.ml.definitions.numerics.serve3_reference import materialize
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import ref_prims as R
from verity_vllm.program.frontend.boolean import derived
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.program_identity import check_identity
from verity_vllm.program.frontend.report import DeriveReport
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.rules.training_survey import survey, with_holes
from verity_vllm.program.frontend.train_step import AdamW, export_step, in_bits_beyond_theta, step_program, step_report

aten = torch.ops.aten


def _program(f, *xs, theta=()):
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(*xs)
    names = [f"x{i}" for i in range(len(xs))]
    rep = derive_graph(gm, xs, mode="reference", rules=training_rules(), name="T", input_names=names, theta=theta)
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


def _run(f, *xs) -> tuple[torch.Tensor, torch.Tensor]:
    """(the Program's outputs, torch's with IEEE sqrt), each as one flat f32 tensor (bf16 words widened)."""
    prog = _program(f, *xs)
    tab = materialize(prog, [_bits(x) for x in xs])
    got = np.array([tab[g] << 32 - gate_width(prog, g) for g in prog.circuit.root_outputs()], dtype=np.uint32).view(np.float32)
    with _IeeeSqrt():
        want = torch.cat([t.detach().reshape(-1).float() for t in f(*xs)])
    return torch.from_numpy(got.copy()), want


def _same_bits(got: torch.Tensor, want: torch.Tensor) -> None:
    bad = (got.view(torch.int32) != want.view(torch.int32)).nonzero().reshape(-1)
    assert bad.numel() == 0, f"{bad.numel()} of {got.numel()} words differ, first at {bad[:4].tolist()}: " \
                             f"{got[bad[:4]].tolist()} vs torch {want[bad[:4]].tolist()}"


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
    }


def _elementwise_bf16(x, y, pos, m, row, col) -> dict:
    """The element-wise cases of bf16 values (the f32 inputs rounded) that torch's bf16 kernels compute as one f32 operation rounded
    to bf16, numbers bf16 holds among them, and of a bf16 and an f32 value, which torch computes in f32."""
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
}


def _f32(v: float) -> int:
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


#: the rules whose reference arithmetic is the vocabulary's own (a sequential sum, a correctly rounded exp, a dot by fma), each with
#: its stated arithmetic composed from the evaluators of the Definitions it names
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


def test_log_softmax_refused_with_the_definition_it_needs() -> None:
    gm = make_fx(lambda x: (torch.log_softmax(x, -1),), tracing_mode="fake")(X)
    rep = derive_graph(gm, (X,), mode="reference", rules=training_rules(), name="T")
    u = rep.unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and "F32LogRn" in u.hint, u


@pytest.mark.parametrize("f, x, why", [
    (lambda a: (a ** 3,), XB, "rounds in between"), (lambda a: (a ** -2,), POSB, "rounds in between"),
    (lambda a: (torch.rsqrt(a),), POSB, "rounds in between"), (lambda a: (a ** -0.5,), POSB, "rounds in between"),
    (lambda a: (a + 0.1,), XB, "bf16 doesn't hold"), (lambda a: (a * 0.1,), XB, "bf16 doesn't hold")])
def test_bf16_rounding_in_between_is_refused(f, x, why: str) -> None:
    """Where torch's bf16 kernel rounds otherwise than once after one f32 operation, the rules refuse: `pow` 3 and -2 round the
    square to bf16, `rsqrt`'s scalar loop the square root, and `add` rounds a number to bf16 where `mul` keeps its f32 value."""
    gm = make_fx(f, tracing_mode="fake")(x)
    u = derive_graph(gm, (x,), mode="reference", rules=training_rules(), name="T").unsupported
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
        st = opt.state[t] if t in opt.state else None
        if st is None:
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


@functools.cache
def _tiny_survey():
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64)
    opt = AdamW(**ADAMW)
    ex = export_step(model, loss_fn, opt, batch)
    return model, ex[2], step_report(model, loss_fn, opt, batch, survey=True, exported=ex)


@functools.cache
def _tiny_program():
    """The tiny step with a loss that has no log (`square_loss`): a Program, every op derived."""
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64)
    prog = step_program(model, square_loss, AdamW(**ADAMW), batch)
    assert not isinstance(prog, Unsupported), prog
    return model, prog


def test_tiny_step_survey_lists_its_one_hole_and_holds_no_program() -> None:
    model, layout, s = _tiny_survey()
    assert [(h["op"], h["reason"]) for h in s.holes] == [("aten._log_softmax.default", "numerics_unregistered")]
    assert s.holes[0]["fields"] == [f"hole0.0:{s.holes[0]['node']}"]
    assert (s.unsupported.reason, s.unsupported.node) == ("numerics_unregistered", f"{s.holes[0]['node']}: aten._log_softmax.default")
    assert not hasattr(s, "program") and not hasattr(s, "result") and s.refused is None
    assert s.census["totals"]["gates"] > 0 and "F32Add_v1" in s.primitives
    assert s.theta["names"] == [f"param_{n}".replace(".", "_") for n in layout.params]
    assert s.theta["bits"] == 32 * sum(p.numel() for p in model.parameters())


def test_tiny_bf16_step() -> None:
    """The tiny step with its linears' operands cast to bf16, as bf16 autocast casts them (`LM(matmul_dtype=torch.bfloat16)`). With
    the log-free loss every op derives, and each bf16 matrix product is `LinearRef_v1` rows of bf16 words (attention's are f32:
    torch's math path computes it from upcast operands); with the cross-entropy the survey's one hole is `_log_softmax`, as in f32,
    and the casts are among the primitives."""
    model, batch = meta_step("tiny", 2, ctx=16, vocab=64, matmul_dtype=torch.bfloat16)
    opt = AdamW(**ADAMW)
    ex = export_step(model, square_loss, opt, batch)
    prog = step_report(model, square_loss, opt, batch, exported=ex).result
    assert not isinstance(prog, Unsupported), prog
    bf16 = ["DT='bf16'" in nd.fn.id for nd in prog.root.body.nodes if nd.fn.id.startswith("LinearRef_v1")]
    mms = sum(n.target is aten.mm.default and n.meta["val"].dtype == torch.bfloat16 for n in ex[0].graph.nodes)
    assert sum(bf16) == mms > 0 and not all(bf16), (sum(bf16), len(bf16), mms)
    s = step_report(model, loss_fn, opt, batch, survey=True, exported=export_step(model, loss_fn, opt, batch))
    assert [(h["op"], h["reason"]) for h in s.holes] == [("aten._log_softmax.default", "numerics_unregistered")]
    assert {P.Bf16ToF32.id, P.F32ToBf16Rn.id} <= set(s.primitives)


def test_tiny_step_theta_is_the_parameters() -> None:
    model, prog = _tiny_program()
    layout = prog.annotations["train_step"]
    assert prog.theta == tuple(f"param_{n}".replace(".", "_") for n in layout["params"])
    nodes = prog.root.body.nodes
    assert sum(nodes[i].n * nodes[i].fn.ret.w for i in prog.root.theta) == 32 * sum(p.numel() for p in model.parameters())


def test_a_program_with_holes_is_no_statement() -> None:
    """A survey's holes are fields of the `run` input that no gate checks: its Program is marked with them, `check_identity`
    refuses it (and so the Boolean lowering does), and a derivation that leaves holes returns no Program."""
    gm = make_fx(lambda x: (torch.log_softmax(x, -1) * 2.0,), tracing_mode="fake")(X)
    hs, prog = survey(gm, (X,), mode="reference", rules=training_rules(), name="T")
    assert [(h["op"], h["fields"]) for h in hs] == [("aten._log_softmax.default", ["hole0.0:_log_softmax"])]
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


def test_s_step_reference_mode() -> None:
    """The brief's step: the lf model's s size (d 512, 6 layers, 8 heads, SwiGLU 1408, tied embeddings, vocabulary 50,304, context
    1024), batch 2, plain AdamW, in reference mode. Every op of its forward derives; the step is refused at `_log_softmax`, the one
    op whose Definition (a correctly rounded f32 log) the reference vocabulary lacks. About 15 s on CPU."""
    model, batch = meta_step("s", 2)
    out = step_program(model, loss_fn, AdamW(**ADAMW), batch, mode="reference")
    assert isinstance(out, Unsupported), out
    assert (out.reason, out.node) == ("numerics_unregistered", "_log_softmax: aten._log_softmax.default"), out
    assert "F32LogRn" in out.hint
