"""A training circuit's work read off its Program (`circuit_properties.work`, which reads the circuit alone) beside this
benchmark's reading of the same loop body from a trace (`capture_step`'s rows, `formalism.he_of_rows` and `he_figure`),
on the lf model's tiny step (`adapters/vllm/tests/program/train_step_lf.py`, loaded by its path) under AdamW, at each
k of `products.DIAL_K` at which (H) states β(k).

The Program is the training-step frontend's step (`step_program`) as the body of a scan over iterations: the carry the
parameters and AdamW's two moments, each iteration's inputs its three scalars and its batch, the rope tables shared. Its
work an iteration is what a third iteration adds to two, the iteration a trace's step stands for (its weights an update
away from θ). Both readings are in one unit: the trace's rows priced at f32, the Program's dtype, and |C| in Boolean
gates on both sides. The Program's reading is never above the trace's: every product it charges is one the trace's M
holds, with no more MACs and no larger β on either operand, and its blocks at the trace's F are at most the trace
figure's. Where they differ, the Program's is below, each for one cause:

- Rates: each product's operands are paid at most what the trace pays any product of its shape. Attention's scores
  multiply two activations, which `gemm_rate` prices at the least of attention's form and each single token side's,
  since a Program doesn't name a token's axis; every rate downstream follows.
- Gathered rows: every product of two operands that depend on θ′ reads the embedding's rows, which no op a Program
  states can be shown to make position-specific (the rope tables are inputs of unstated values, the causal mask a
  constant laid along an axis the Program doesn't name), so the Program charges nothing on this step; the trace's M
  holds its products after the rope and the mask.
- The dots only the Program writes as products: the softmax's backward, two `DotRef`s a row, carries the same rows,
  and the sums and means the frontend writes as a product with a repeated constant are `θ`'s.
- The head's products and attention's: from `HEAD_K` the Program gives no bound, since it doesn't say which products
  are the head's, where the trace's figure keeps the weight gradients; a product of two activations, a weight
  gradient or attention's, is left out, where the trace keeps the weight gradients.
- β: every operand at most min(16, β(k)), where the trace's weights are at their width, 32.
- |C|: the products' gates differ by the MACs only the Program writes as products (those dots and sums) and by the
  MAC's Boolean Definition: the Program's `F32Fma_v1`, whose Boolean version `F32Fma_v3` keeps a NaN's payload rule,
  against `F32Fma_v4`, the PTX FFMA the trace prices a MAC at. The rest of |C| is every other primitive at its Boolean
  version on the Program's side, and on the trace's only the ops `training_slowdown.flops` counts, each MAC at one
  `F32Fma_v4`: the exponentials, square roots, plain adds and subtractions, selections, maxima and comparisons are in
  the Program's |C| and in no MAC of the trace's.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from training_slowdown import formalism as F
from training_slowdown import gates as G
from training_slowdown.capture import capture_step, products

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "benchmarks" / "circuit_properties"))
from circuit_properties import work as WK  # noqa: E402

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.circuits.evaluation.bits import compile as compile_bits  # noqa: E402
from verity.circuits.program import Program, is_input  # noqa: E402
from verity.circuits.refs import Coll, concat, tuple_of  # noqa: E402
from verity.circuits.types import Array, Tuple  # noqa: E402
from verity.definitions.boolean import boolean_version  # noqa: E402
from verity.definitions.registry import prims as PRIMS  # noqa: E402
from verity_vllm.program.frontend.train_step import AdamW, step_program  # noqa: E402

LF_PATH = ROOT / "adapters" / "vllm" / "tests" / "program" / "train_step_lf.py"
#: the step's batch as rows of tokens, and the model's context (the rope tables' rows)
BATCH, CTX = (2, 17), 34
#: the policy's c in gates, and the k |C| is read at
C, K0 = 1e13, 65536.0
MC = WK.min_cut()
#: the k of the dial at which (H) states β(k)
KS = tuple(float(k) for k in products.DIAL_K if MC.beta(k) is not None)
#: the primitives no MAC of the trace's prices: every counted op is a multiply, a division, a fused multiply-add, a
#: scaled add or a gather
UNCOUNTED = ("F32ExpRn", "F32Sqrt", "F32Add", "F32Sub", "SelectF32", "F32MaxRef", "F32Eq", "I32Eq", "I32Add",
             "F32Neg", "I64ToI32", "BitAnd")


def _lf():
    spec = importlib.util.spec_from_file_location("training_slowdown_tests_program_lf", LF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LF = _lf()


def loop(step: Program, n: int) -> Program:
    """``step`` (`step_program`'s: θ, the two moments, three scalars, the buffers, the batch) as the body of a scan of
    ``n`` iterations."""
    params, ret = step.fn.signature()
    P = len(step.theta)
    state, xs, shared = params[:3 * P], params[3 * P:3 * P + 3] + params[-1:], params[3 * P + 3:-1]
    carry_t, loss_t = Tuple(*(t for _, t in state)), ret.fields[0]

    def body(B, S, carry, *rest):
        x, sh = rest[:len(xs)], rest[len(xs):]
        out = B.call(step.fn, *(carry[i] for i in range(len(state))), *x[:3], *sh, x[3])
        return tuple_of(Coll(carry_t, concat([out[1 + i].refs for i in range(len(state))])), out[0])

    it = CompositeDefinition("Iteration", 1, (), lambda S: ((("carry", carry_t), *xs, *shared), Tuple(carry_t, loss_t)),
                             body, register=False)
    sig = ((*state, *((name, Array(n, t)) for name, t in xs), *shared), Tuple(carry_t, Array(n, loss_t)))

    def run(B, S, *a):
        init = Coll(carry_t, concat([c.refs for c in a[:len(state)]]))
        return B.scan(bind(it), init, xs=list(a[len(state):len(state) + len(xs)]), shared=list(a[len(state) + len(xs):]))

    return Program(bind(CompositeDefinition("Loop", 1, (), lambda S: sig, run, register=False)), theta=step.theta)


def _key(g: WK.Group) -> tuple:
    return (g.product, g.widths, g.weight, g.out_width, g.k_dim, g.dims, g.rates, g.bits, g.beta, g.reason)


def iteration(step: Program, k: float) -> WK.Work:
    """The work a third iteration adds to two, at k."""
    two, three = WK.read(loop(step, 2), k), WK.read(loop(step, 3), k)
    before = {_key(g): g for g in two.products}
    out = []
    for g in three.products:
        b = before.get(_key(g))
        g = dataclasses.replace(g, macs=g.macs - (b.macs if b else 0), wires=g.wires - (b.wires if b else 0))
        if g.macs:
            out.append(g)
    return WK.Work(k, three.gates - two.gates, out, three.r_step, three.keep)


def primitives(step: Program) -> Counter:
    """The step's primitives' activations by (inside a product, primitive), counted once a Definition and multiplied
    through batches and scans."""
    reader, memo = WK.Reader(step, K0), {}
    by_id: dict = {}

    def walk(fn, inside: bool) -> Counter:
        if (fn.id, inside) not in memo:
            c: Counter = Counter()
            for nd in fn.body.nodes:
                m = nd.n if nd.form in ("batch", "scan") else 1
                if isinstance(nd.fn, PrimitiveDefinition):
                    if not is_input(nd.fn):
                        by_id[nd.fn.id] = nd.fn
                        c[(inside, nd.fn.id)] += m
                else:
                    for k, v in walk(nd.fn, inside or reader.product(nd.fn) is not None).items():
                        c[k] += m * v
            memo[(fn.id, inside)] = c
        return memo[(fn.id, inside)]

    counts = walk(step.root, False)
    return Counter({k: n * sum(compile_bits(boolean_version(by_id[k[1]])).counts.values()) for k, n in counts.items()})


@dataclasses.dataclass
class Reading:
    step: Program
    #: by k: the Program's work an iteration, the trace's priced rows, `he_of_rows`' M and `he_figure`'s figure
    work: dict
    rows: dict
    he: dict
    fig: dict
    size: float


@pytest.fixture(scope="module")
def readings() -> Reading:
    with torch.device("meta"):
        model = LF.LM(LF.SIZES["tiny"], ctx=CTX, vocab=64)
    step = step_program(model, LF.square_loss, AdamW(**LF.ADAMW), torch.empty(*BATCH, dtype=torch.long, device="meta"))
    torch.manual_seed(0)
    model = LF.LM(LF.SIZES["tiny"], ctx=CTX, vocab=64)
    g = capture_step(model, torch.optim.AdamW(model.parameters(), **LF.ADAMW), LF.square_loss,
                     torch.zeros(*BATCH, dtype=torch.long))
    raw = products.raw_on_dial(g)
    size = products.step_gates(g, None)
    r = Reading(step, {}, {}, {}, {}, size)
    for k in KS:
        blk = raw if k == raw["k"] else raw["at"][str(int(k))]
        r.rows[k] = [products.priced(row, None) for row in blk["rows"]]
        r.he[k] = F.he_of_rows(r.rows[k], k, None, g.tokens)
        r.fig[k] = F.he_figure(r.he[k], C, size)
        r.work[k] = iteration(step, k)
    return r


def _ordered(beta, weight) -> tuple:
    """An operand pair's β, the weight's first, two of one kind the larger first."""
    a, b = beta
    if weight[0] != weight[1]:
        return (b, a) if weight[1] else (a, b)
    return tuple(sorted((a, b), reverse=True))


def held(g: WK.Group, he: dict) -> bool:
    """Whether the trace's M holds a group of ``g``'s kinds and contraction with at least its MACs and β."""
    mine = _ordered(g.beta, g.weight)
    return any(t["weight"] == list(g.weight) and t["k_dim"] == g.k_dim and t["macs"] >= g.macs and
               all(m <= th + 1e-9 for m, th in zip(mine, _ordered(t["beta"], t["weight"]))) for t in he["groups"])


def trace_blocks(fig: dict, k: float, size: float) -> float:
    return MC.bound(C, k, N=fig["N"], O=fig["O"], size=size, beta_w=fig["beta_w"], beta_x=fig["beta_x"],
                    b=fig["b_y"], F=fig["F"]).blocks if fig["N"] else 0.0


@pytest.mark.slow
def test_the_programs_bound_is_never_above_the_traces(readings):
    """Derives the step, captures it and reads both at each k (about a minute): what it catches is a Program reading
    that charges work the trace's M doesn't hold, or a β above the trace's, at any k of the dial."""
    r = readings
    for k in KS:
        it, he, fig = r.work[k], r.he[k], r.fig[k]
        assert all(held(g, he) for g in it.groups)
        theirs = trace_blocks(fig, k, r.size)
        at_theirs, own = WK.bound(it, C, F=fig["F"]), WK.bound(it, C)
        assert (at_theirs.bound.blocks if at_theirs else 0.0) <= theirs
        assert (own.bound.blocks if own else 0.0) <= theirs
        assert theirs > 0


@pytest.mark.slow
def test_every_product_pays_at_most_what_the_trace_pays_its_shape(readings):
    """Each of the Program's products (m, K, n) is a shape the trace has a product of, each operand's rate at most
    the least the trace's products of that shape pay it, and every product with two θ′ operands carries the
    embedding's rows, so the Program charges nothing."""
    r = readings
    traced = sum(row["count"] * row["macs"] for row in r.rows[KS[0]])
    for k in KS:
        it = r.work[k]
        lins = [g for g in it.products if g.product.startswith("LinearRef")]
        assert sum(g.macs for g in lins) == traced == r.he[k]["candidates_macs"]
        rows = [row for row in r.rows[k] if all(row["affected"])]
        for g in lins:
            _b, m, kk, n = g.dims
            least = [math.inf, math.inf]
            for row in rows:
                d = row["dims"]
                for (dm, dn), (ra, rb) in (((d["m"], d["n"]), row["rates"]), ((d["n"], d["m"]), row["rates"][::-1])):
                    if (dm, d["k"], dn) == (m, kk, n):
                        least = [min(least[0], ra), min(least[1], rb)]
            assert math.isfinite(least[0]), g.dims
            assert g.rates[0] <= least[0] + 1e-9 and g.rates[1] <= least[1] + 1e-9, (g.dims, g.rates, least)
        assert it.groups == [] and it.r_step is None
        assert it.left_out["self"] == 0 and it.left_out["activations"] == 0
        dots = [g for g in it.products if g.product.startswith("DotRef")]
        assert {g.reason for g in lins} == {"gathered"} and {g.reason for g in dots} == {"gathered", "theta"}


@pytest.mark.slow
def test_both_sizes_are_boolean_gates_and_differ_by_what_each_prices(readings):
    """|C| on both sides in Boolean gates: the products' differ by the MACs only the Program writes as products and by
    their Definition's gates, and the rest of the Program's |C| is the trace's and the primitives no MAC of the
    trace's prices, less what the trace prices its elementwise MACs at beyond the Program's multiplies and divisions."""
    r = readings
    step, it, rows, size = r.step, r.work[K0], r.rows[K0], r.size
    assert it.gates == WK.boolean_gates(step)
    split = primitives(step)
    assert sum(split.values()) == it.gates
    inside = {fid for (ins, fid), g in split.items() if ins and g}
    assert inside == {"F32Fma_v1"}
    fma_v1 = sum(compile_bits(boolean_version(PRIMS.F32Fma)).counts.values())
    traced = sum(row["count"] * row["macs"] for row in rows)
    mine_products = sum(g for (ins, _f), g in split.items() if ins)
    assert mine_products == sum(g.macs for g in it.products) * fma_v1
    assert sum(g.macs for g in it.products if g.product.startswith("LinearRef")) == traced
    theirs_products = sum(row["count"] * row["macs"] * row["gates_per_mac"] for row in rows)
    assert theirs_products == traced * G.gates("F32Fma_v4") and G.gates("F32Fma_v4") > fma_v1
    mine_rest, theirs_rest = it.gates - mine_products, size - theirs_products
    uncounted = sum(g for (ins, f), g in split.items() if not ins and f.rsplit("_v", 1)[0] in UNCOUNTED)
    assert mine_rest - uncounted <= theirs_rest <= mine_rest


def test_the_readings_constants_are_the_benchmarks():
    assert WK.ACC_BITS == products.ACC_BITS and WK.SHARED_PREFIX == F.SHARED_PREFIX and WK.HEAD_K == F.HEAD_K
    for w in (1.0, 16.0, 32.0):
        for rate in (0.5 * w, WK.PAID * w, w):
            assert WK.paid(rate, w) == F.is_paid(rate, w)
    for k in products.DIAL_K:
        assert MC.beta(k) == F.beta_k(k)
        for w, rank in ((32.0, 4.0), (16.0, 64.0)):
            assert WK.k_star(k, w) == pytest.approx(F.k_star(k, w))
            assert WK.beta_eff(w, rank, k) == pytest.approx(F.beta_eff(w, rank, k))
