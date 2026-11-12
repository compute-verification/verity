"""A training circuit's work read off its Program (`circuit_properties.work`, which reads the circuit alone) beside this
benchmark's reading of the same loop body from a trace (`library.bound` on `capture_step`'s rows), on the lf model's tiny
step (`adapters/vllm/tests/program/train_step_lf.py`, loaded by its path) under AdamW.

The Program is the training-step frontend's step (`step_program`) as the body of a scan over iterations: the carry the
parameters and AdamW's two moments, each iteration's inputs its three scalars and its batch, the rope tables shared. Its
work an iteration is what a third iteration adds to two, the iteration a trace's step stands for (its weights an update
away from θ). Where the two readings differ, each for one cause:

- N: the Program's dot products the trace has no row for (the softmax's backward, two `DotRef`s a row, which torch writes
  as a multiply and a sum) are charged; and the trace leaves out of its M what the Program reading charges: products
  that read an embedding's rows (charged from the second iteration, where the table is the carry's, not θ's),
  attention's, and 1/seq of a token-indexed product's MACs (`SHARED_PREFIX`).
- O: the dots' outputs, and the trace's outputs at bf16's 16 bits where the Program's products are f32.
- |C|: the Program's gates, a primitive one, where the trace prices each counted op at its Definition's Boolean gates.
- β: a weight at its width and an activation at min(width, β(k)) where the trace prices each operand at the rate the
  update reaches it with, at most bf16's 16 bits.
- F: the same, share(3), on the whole work: the forward, input-gradient and weight-gradient products share arrays.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from training_slowdown import library
from training_slowdown import formalism as F
from training_slowdown.capture import capture_step, products

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "benchmarks" / "circuit_properties"))
from circuit_properties import work as WK  # noqa: E402

from verity.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.circuits.program import Program  # noqa: E402
from verity.circuits.refs import Coll, concat, tuple_of  # noqa: E402
from verity.circuits.types import Array, Tuple  # noqa: E402
from verity_vllm.program.frontend.train_step import AdamW, step_program  # noqa: E402

LF_PATH = ROOT / "adapters" / "vllm" / "tests" / "program" / "train_step_lf.py"
#: the step's batch as rows of tokens, and the model's context (the rope tables' rows)
BATCH, CTX = (2, 17), 34
#: the policy both readings are priced at: c in gates and k in bits, where neither is at its floor
POLICY = (1e12, 65536.0)


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
    before = {(g.product, g.widths, g.weight, g.out_width, g.k_dim): g for g in two.groups}
    groups = []
    for g in three.groups:
        b = before.get((g.product, g.widths, g.weight, g.out_width, g.k_dim))
        g = dataclasses.replace(g, macs=g.macs - (b.macs if b else 0), wires=g.wires - (b.wires if b else 0))
        if g.macs:
            groups.append(g)
    return WK.Work(three.gates - two.gates, groups, {k: three.left_out[k] - two.left_out[k] for k in three.left_out})


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
    trace = {k: raw[k] for k in ("capture", "rows", "k", "p_match", "at")} | {
        "tokens": g.tokens, "circuit": {"gates": products.step_gates(g)}}
    return step, iteration(step), trace


@pytest.mark.slow
def test_the_programs_work_is_the_traces_with_each_difference_explained(readings):
    """Derives the step and captures it (seconds each): what it catches is a Program reading that drifts from the
    products, kinds and sharing the benchmark finds in a trace of the same step."""
    step, it, trace = readings
    rows = [products.priced(r) for r in trace["rows"]]
    he = F.he_of_rows(rows, trace["k"], None, trace["tokens"])
    traced = sum(r["count"] * r["macs"] for r in rows)
    assert he["candidates_macs"] == traced and he["self_macs"] == 0
    dots = [g for g in it.groups if g.product.startswith("DotRef")]
    assert dots and all(not any(g.weight) for g in dots)
    assert it.N == traced + sum(g.macs for g in dots)
    assert it.O == sum(r["count"] * r["outputs"] for r in rows) * 32 + sum(g.wires for g in dots)
    assert sum(g["macs"] for g in he["groups"]) < traced - he["gathered_macs"] - he["left_out_macs"]
    assert it.left_out["lookup"] == 0 and it.left_out["self"] == 0
    assert it.gates == step.fn.gates + 3 + BATCH[0] * BATCH[1]
    assert {g.weight for g in it.groups} == {(False, False), (False, True)}
    assert {w for g in it.groups for w in g.widths} == {32} and {w for g in he["groups"] for w in g["width"]} == {16}
    assert WK.clique(e for g in it.groups for e in g.edges) == 3


@pytest.mark.slow
def test_both_slowdowns_are_the_calculators_at_one_policy(readings):
    """Each slowdown is `min_cut.bound`'s on its own reading's work: the trace's at the figure `library.bound` reads,
    the Program's at the sub-work `circuit_properties.work.bound` picks. They differ by |C| (the trace's Boolean gates
    over the Program's own) and by the blocks, which the β, N, O, b and F of each work give."""
    _step, it, trace = readings
    c, k = POLICY
    theirs, mine = library.bound(trace, c, k), WK.bound(it, c, k)
    mc = WK.min_cut()
    fig = theirs["figure"]
    again = mc.bound(c, k, N=fig["N"], O=fig["O"], size=trace["circuit"]["gates"], beta_w=fig["beta_w"],
                     beta_x=fig["beta_x"], b=fig["b_y"], F=fig["F"])
    assert theirs["slowdown"] > 1 and theirs["slowdown"] == pytest.approx(again.slowdown)
    assert mine.bound.slowdown > 1 and mine.bound.slowdown == pytest.approx(mine.bound.blocks * c / it.gates)
    assert fig["F"] == pytest.approx(mc.share(3))
    ratio = (mine.bound.blocks / again.blocks) * (trace["circuit"]["gates"] / it.gates)
    assert mine.bound.slowdown / theirs["slowdown"] == pytest.approx(ratio)
