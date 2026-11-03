"""The training-step frontend (`program.frontend.train_step`, `derive_graph`, the training rules): each training rule's Program run
against torch on small CPU tensors, bit for bit where the rule states torch CPU's own f32 operation and to a tolerance where the
reference vocabulary's arithmetic is its own (sequential sums, a correctly rounded exp); the functional AdamW against
`torch.optim.AdamW(foreach=False)`; `BoxPermute` against torch's permute; and the lf model's s step derived in reference mode,
which is refused at its one op with no Definition (`_log_softmax`: `F32LogRn`)."""
from __future__ import annotations

import bisect
import functools

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

from tests.program.train_step_lf import ADAMW, loss_fn, meta_step, square_loss
from verity.ml.circuits.boundary import gate_width
from verity.ml.circuits.parts import Part, n_members
from verity.ml.circuits.program import is_input
from verity.ml.circuits.query_ast import GateSet, evaluate, parse
from verity.ml.circuits.refs import runs as ref_runs
from verity.ml.definitions.numerics.serve3_reference import materialize
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
    rep = derive_graph(gm, xs, mode="reference", rules=training_rules(min_view_leaves=0), name="T", input_names=names, theta=theta)
    assert rep.program is not None, rep.unsupported
    return rep.program


def _bits(x: torch.Tensor) -> list[int]:
    x = x.detach().contiguous().reshape(-1)
    if x.dtype == torch.float32:
        return x.view(torch.int32).numpy().astype(np.uint32).tolist()
    return [int(v) & ((1 << (8 * x.element_size())) - 1) for v in x.tolist()]


class _IeeeSqrt(TorchDispatchMode):
    """torch with IEEE f32 `sqrt` (and `pow` 0.5, which torch computes with it), as the rules state it: torch CPU's is MKL's, which
    on AMD EPYC Genoa (the CPU nodes) is one ulp off on 11 of `POS`'s 64 words, and IEEE's on an Intel Xeon."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        x = args[0] if args else None
        if isinstance(x, torch.Tensor) and x.dtype == torch.float32 and (
                func is aten.sqrt.default or (func is aten.pow.Tensor_Scalar and args[1] == 0.5)):
            return torch.from_numpy(np.asarray(np.sqrt(x.detach().numpy())))
        return func(*args, **(kwargs or {}))


def _run(f, *xs) -> tuple[torch.Tensor, torch.Tensor]:
    """(the Program's outputs, torch's with IEEE sqrt), each as one flat f32 tensor."""
    prog = _program(f, *xs)
    tab = materialize(prog, [_bits(x) for x in xs])
    got = np.array([tab[g] for g in prog.circuit.root_outputs()], dtype=np.uint32).view(np.float32)
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


X, Y = _data(4, 16), _data(4, 16, seed=1)
POS = _data(4, 16, positive=True, seed=2)
ROW, COL = _data(16, seed=3), _data(4, 1, seed=4)

#: the element-wise rules torch CPU computes with the same f32 operations: bit for bit
EXACT = {
    "add": (lambda x, y: (x + y,), (X, Y)),
    "sub": (lambda x, y: (x - y,), (X, Y)),
    "mul": (lambda x, y: (x * y,), (X, Y)),
    "div": (lambda x, y: (x / y,), (X, POS)),
    "add_scalar": (lambda x: (x + 0.1,), (X,)),
    "mul_scalar": (lambda x: (x * 0.3,), (X,)),
    "div_scalar": (lambda x: (x / 3.0,), (X,)),
    "mul_row": (lambda x, r: (x * r,), (X, ROW)),
    "mul_col": (lambda x, c: (x * c,), (X, COL)),
    "neg": (lambda x: (-x,), (X,)),
    "sqrt": (lambda x: (torch.sqrt(x),), (POS,)),
    "rsqrt": (lambda x: (torch.rsqrt(x),), (POS,)),
    "reciprocal": (lambda x: (torch.reciprocal(x),), (POS,)),
    "pow2": (lambda x: (x ** 2,), (X,)),
    "pow3": (lambda x: (x ** 3,), (X,)),
    "pow_half": (lambda x: (x ** 0.5,), (POS,)),
    "pow_m_half": (lambda x: (x ** -0.5,), (POS,)),
    "pow_m1": (lambda x: (x ** -1,), (POS,)),
    "pow_m2": (lambda x: (x ** -2,), (POS,)),
    "lerp_small": (lambda x, y: (torch.lerp(x, y, 0.1),), (X, Y)),
    "lerp_large": (lambda x, y: (torch.lerp(x, y, 0.9),), (X, Y)),
    "addcmul": (lambda x, y, z: (torch.addcmul(x, y, z, value=0.05),), (X, Y, POS)),
    "nll_backward": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 1, -100, tw),),
                     (torch.tensor(1.0), X, torch.tensor([3, 0, 15, 7]), torch.tensor(4.0))),
    "nll_backward_ignored": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 1, 0, tw),),
                             (torch.tensor(1.0), X, torch.tensor([3, 0, 15, 0]), torch.tensor(2.0))),
    "nll_backward_ignored_m100": (lambda go, logp, t, tw: (aten.nll_loss_backward(go, logp, t, None, 1, -100, tw),),
                                  (torch.tensor(1.0), X, torch.tensor([3, -100, 15, 7]), torch.tensor(3.0))),
    "embedding_backward": (lambda g, i: (aten.embedding_dense_backward(g, i, 6, -1, False),),
                           (X, torch.tensor([2, 5, 2, 0]))),
    "permute_of_concat": (lambda x, y: ((torch.cat([x, y], dim=1).t() * 2.0),), (X, Y)),
}

#: the rules whose reference arithmetic is the vocabulary's own (a sequential sum, a correctly rounded exp, a dot by fma)
CLOSE = {
    "exp": (lambda x: (torch.exp(x * 0.01),), (X,)),
    "sigmoid": (lambda x: (torch.sigmoid(x),), (X,)),
    "sum_last": (lambda x: (x.sum(-1),), (X,)),
    "mean_last": (lambda x: (x.mean(-1, keepdim=True),), (X,)),
    "sum_first": (lambda x: (x.sum(0),), (X,)),
    "mm": (lambda a, b: (a @ b,), (X, _data(16, 8, seed=5))),
    "bmm": (lambda a, b: (torch.bmm(a, b),), (_data(2, 4, 16, seed=6), _data(2, 16, 4, seed=7))),
    "safe_softmax": (lambda x: (aten._safe_softmax(x, -1),), (X,)),
    "softmax_backward": (lambda g, y: (aten._softmax_backward_data(g, y, -1, torch.float32),), (Y, torch.softmax(X, -1))),
    "log_softmax_backward": (lambda g, y: (aten._log_softmax_backward_data(g, y, -1, torch.float32),),
                             (Y, torch.log_softmax(X, -1))),
    "nll_forward": (lambda logp, t: aten.nll_loss_forward(logp, t, None, 1, -100), (X, torch.tensor([3, 0, 15, 7]))),
}


@pytest.mark.parametrize("name", sorted(EXACT))
def test_rule_bit_exact(name: str) -> None:
    f, xs = EXACT[name]
    _same_bits(*_run(f, *xs))


@pytest.mark.parametrize("name", sorted(CLOSE))
def test_rule_close(name: str) -> None:
    f, xs = CLOSE[name]
    got, want = _run(f, *xs)
    torch.testing.assert_close(got, want, rtol=2e-6, atol=1e-6)


def test_log_softmax_refused_with_the_definition_it_needs() -> None:
    gm = make_fx(lambda x: (torch.log_softmax(x, -1),), tracing_mode="fake")(X)
    rep = derive_graph(gm, (X,), mode="reference", rules=training_rules(), name="T")
    u = rep.unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and "F32LogRn" in u.hint, u


def _adamw_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    p, grad, m = (torch.randn(64, generator=g) for _ in range(3))
    v = torch.rand(64, generator=g) * 1e-3
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


def test_adamw_update_program_bit_exact() -> None:
    """The derived optimizer update (decoupled decay, lerp, addcmul, sqrt, division, addcdiv) is torch's (with IEEE sqrt), word for
    word."""
    f = AdamW(**ADAMW)
    p, grad, m, v = _adamw_inputs(1)
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
