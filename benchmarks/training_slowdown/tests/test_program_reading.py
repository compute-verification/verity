"""A training circuit's work read off its Program (`circuit_properties.work`, which reads the circuit alone) beside this
benchmark's reading of the same loop body from a trace (`capture_step`'s rows, `formalism.he_of_rows` and `he_figure`),
on the lf model's tiny step (`adapters/vllm/tests/program/train_step_lf.py`, loaded by its path) under AdamW.

The Program is the training-step frontend's step (`step_program`) as the body of a scan over iterations: the carry the
parameters and AdamW's two moments, each iteration's inputs its three scalars and its batch, the rope tables shared. Its
work an iteration is what a third iteration adds to two, the iteration a trace's step stands for (its weights an update
away from θ). Both readings are in one unit: the trace's rows priced at f32, the Program's dtype, and |C| in Boolean
gates on both sides. Where they differ, each for one cause:

- N: the Program's dot products the trace has no row for (the softmax's backward, two `DotRef`s a row, which torch writes
  as a multiply and a sum) are charged.
- O: those dots' outputs.
- |C|: the products' gates differ by the MACs only the Program writes as products (those dots, and the sums and means
  the frontend writes as a product with a repeated constant, which torch writes as reductions and the reading leaves
  out of N as θ's) and by the MAC's Boolean Definition: the Program's `F32Fma_v1`, whose
  Boolean version `F32Fma_v3` keeps a NaN's payload rule, against `F32Fma_v4`, the PTX FFMA the trace prices a MAC at.
  The rest of |C| is every other primitive at its Boolean version on the Program's side, and on the trace's only the
  ops `training_slowdown.flops` counts, each MAC at one `F32Fma_v4`: the exponentials, square roots, plain adds and
  subtractions, selections, maxima and comparisons are in the Program's |C| and in no MAC of the trace's.
- The MACs the slowdown is asked of: the trace's figure keeps the products whose operands its rates pay at the most
  (`he_of_rows` charges an operand whose rate, the bits a block pays an entry of it from θ′, is below its width at
  w·ρ/(w + ρ), and `he_figure` keeps the groups at its best threshold of β_w·β_x); the Program reading caps a β at the
  source bits a value holds, and in the third iteration every operand's are its width, so it keeps every product with
  a weight. Both are (H)'s premise that a value costs at most the bits it holds; they count those bits differently:
  the trace gives a gather's rows none (they repeat over the tokens that share an id) and the batch none, the Program
  reading a gathered row its table's bits and the batch its width.
- F: the trace's 2/√3 wherever its figure holds a backward product; the Program reading's share at the most arrays its
  sub-work multiplies pairwise, two among the weight products.
- β: both slowdowns are priced at the calculator's defaults (a weight's and an output's bf16 width, an activation's
  β(k)), so the ratio of the slowdowns is the ratio of the blocks, from N, O and F alone, times the ratio of |C|.
"""
from __future__ import annotations

import dataclasses
import importlib.util
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
#: the policy both readings are priced at: c in gates and k in bits, where neither is at its floor
POLICY = (1e13, 65536.0)
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


def iteration(step: Program) -> WK.Work:
    """The work a third iteration adds to two."""
    two, three = WK.read(loop(step, 2)), WK.read(loop(step, 3))
    key = lambda g: (g.product, g.widths, g.weight, g.out_width, g.k_dim, g.bits)  # noqa: E731
    before = {key(g): g for g in two.groups}
    groups = []
    for g in three.groups:
        b = before.get(key(g))
        g = dataclasses.replace(g, macs=g.macs - (b.macs if b else 0), wires=g.wires - (b.wires if b else 0))
        if g.macs:
            groups.append(g)
    return WK.Work(three.gates - two.gates, groups, {k: three.left_out[k] - two.left_out[k] for k in three.left_out})


def primitives(step: Program) -> Counter:
    """The step's primitives' activations by (inside a product, primitive), counted once a Definition and multiplied
    through batches and scans."""
    reader, memo = WK.Reader(step), {}
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


@pytest.fixture(scope="module")
def readings():
    with torch.device("meta"):
        model = LF.LM(LF.SIZES["tiny"], ctx=CTX, vocab=64)
    step = step_program(model, LF.square_loss, AdamW(**LF.ADAMW), torch.empty(*BATCH, dtype=torch.long, device="meta"))
    torch.manual_seed(0)
    model = LF.LM(LF.SIZES["tiny"], ctx=CTX, vocab=64)
    g = capture_step(model, torch.optim.AdamW(model.parameters(), **LF.ADAMW), LF.square_loss,
                     torch.zeros(*BATCH, dtype=torch.long))
    raw = products.raw_on_dial(g)
    rows = [products.priced(r, None) for r in raw["rows"]]
    return step, iteration(step), rows, F.he_of_rows(rows, raw["k"], None, g.tokens), products.step_gates(g, None)


@pytest.mark.slow
def test_the_programs_work_is_the_traces_with_each_difference_explained(readings):
    """Derives the step and captures it (seconds each): what it catches is a Program reading that drifts from the
    products, kinds and sharing the benchmark finds in a trace of the same step."""
    _step, it, rows, he, _size = readings
    traced = sum(r["count"] * r["macs"] for r in rows)
    assert he["candidates_macs"] == traced and he["self_macs"] == 0
    dots = [g for g in it.groups if g.product.startswith("DotRef")]
    assert dots and all(not any(g.weight) for g in dots)
    assert it.N == traced + sum(g.macs for g in dots)
    assert it.O == sum(r["count"] * r["outputs"] for r in rows) * 32 + sum(g.wires for g in dots)
    assert it.left_out["lookup"] == 0 and it.left_out["self"] == 0
    assert {g.weight for g in it.groups} == {(False, False), (False, True)}
    assert {w for g in it.groups for w in g.widths} == {w for g in he["groups"] for w in g["width"]} == {32}
    assert {g.bits for g in it.groups} == {(32.0, 32.0)}
    assert WK.clique(e for g in it.groups for e in g.edges) == 3


@pytest.mark.slow
def test_both_sizes_are_boolean_gates_and_differ_by_what_each_prices(readings):
    """|C| on both sides in Boolean gates: the products' differ by the MACs only the Program writes as products and by
    their Definition's gates, and the rest of the Program's |C| is the trace's and the primitives no MAC of the
    trace's prices, less what the trace prices its elementwise MACs at beyond the Program's multiplies and divisions."""
    step, it, rows, _he, size = readings
    assert it.gates == WK.boolean_gates(step)
    split = primitives(step)
    assert sum(split.values()) == it.gates
    inside = {fid for (ins, fid), g in split.items() if ins and g}
    assert inside == {"F32Fma_v1"}
    fma_v1 = sum(compile_bits(boolean_version(PRIMS.F32Fma)).counts.values())
    traced = sum(r["count"] * r["macs"] for r in rows)
    dots = sum(g.macs for g in it.groups if g.product.startswith("DotRef"))
    mine_products = sum(g for (ins, _f), g in split.items() if ins)
    assert mine_products == (traced + dots + it.left_out["theta"]) * fma_v1
    theirs_products = sum(r["count"] * r["macs"] * r["gates_per_mac"] for r in rows)
    assert theirs_products == traced * G.gates("F32Fma_v4") and G.gates("F32Fma_v4") > fma_v1
    mine_rest, theirs_rest = it.gates - mine_products, size - theirs_products
    uncounted = sum(g for (ins, f), g in split.items() if not ins and f.rsplit("_v", 1)[0] in UNCOUNTED)
    assert mine_rest - uncounted <= theirs_rest <= mine_rest


@pytest.mark.slow
def test_both_slowdowns_are_the_calculators_at_its_defaults(readings):
    """Each slowdown is `min_cut.bound`'s at the calculator's defaults on its own reading's work: the trace's on the
    figure `he_figure` keeps, the Program's on the sub-work `circuit_properties.work.bound` picks."""
    _step, it, _rows, he, size = readings
    c, k = POLICY
    mc = WK.min_cut()
    fig = F.he_figure(he, c, size)
    theirs, mine = mc.bound(c, k, N=fig["N"], O=fig["O"], size=size, F=fig["F"]), WK.bound(it, c, k)
    assert (mine.bound.beta_w, mine.bound.beta_x, mine.bound.b) == (theirs.beta_w, theirs.beta_x, theirs.b) == \
        (mc.BF16, mc.beta(k), mc.BF16)
    assert theirs.slowdown > 1 and mine.bound.slowdown > 1
    assert mine.bound.slowdown == pytest.approx(mine.bound.blocks * c / it.gates)
    kept = [g for g in he["groups"] if g["beta"] == [fig["beta_x"], fig["beta_w"]] and g["weight"] == [False, True]]
    assert fig["N"] == sum(g["macs"] for g in kept) < sum(g["macs"] for g in he["groups"])
    assert mine.N == sum(g.macs for g in it.groups if any(g.weight)) and mine.omega == 2
    assert fig["F"] == pytest.approx(mc.share(3)) and mine.bound.F == 1
    ratio = (mine.bound.blocks / theirs.blocks) * (size / it.gates)
    assert mine.bound.slowdown / theirs.slowdown == pytest.approx(ratio)
