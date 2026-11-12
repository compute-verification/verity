"""A training circuit's work read off its Program (`circuit_properties.work`, which reads the circuit alone) beside this
benchmark's reading of the same loop body from a trace (`capture_step`'s rows, `work.he_of_rows` and `calculator.slowdown`),
on the lf model's tiny step (`adapters/vllm/tests/program/train_step_lf.py`, loaded by its path) under AdamW, at each
k of `products.DIAL_K` at which (H) states β(k).

The Program is the training-step frontend's step (`step_program`) as the body of a scan over iterations: the carry the
parameters and AdamW's two moments, each iteration's inputs its three scalars and its batch, the rope tables shared. Its
work an iteration is what a third iteration adds to two, the iteration a trace's step stands for (its weights an update
away from θ). Both readings are in one unit: the trace's rows priced at f32, the Program's dtype, and |C| in Boolean
gates on both sides. The Program is read from the circuit alone, with nothing stated beside it, and its reading is
never above the trace's: every product it charges is one the trace's M holds, with no more MACs and no larger β on
either operand, and its blocks at the trace's F, and at its own, are at most what the trace's M gives under the same
choice of sub-work. A dense step with no table lookup (`_dense`) is charged nothing, since its batch's rows may repeat.
Where the readings differ, the Program's is below, each for one cause:

- Rates: each product's operands are paid at most what the trace pays any product of its shape. Attention's scores
  multiply two activations, which `gemm_rate` prices at the least of attention's form and each single token side's,
  since a Program doesn't name a token's axis; every rate downstream follows.
- Gathered rows: every product of two operands that depend on θ′ reads the embedding's rows. From the circuit alone
  they become specific only at a softmax over the causal mask, whose rows and columns the mask makes differ
  (`test_value_reading` holds that rule to the values), so the Program charges the projections downstream of the
  softmax and its P·V, its rows' and columns', and leaves those through the rope tables (inputs of unstated values)
  and the softmax's backward, dQ·Wq and dK·Wk; the trace's M holds its products after the rope and the mask.
- The dots only the Program writes as products: the softmax's backward, two `DotRef`s a row, carries the same rows,
  and the sums and means the frontend writes as a product with a repeated constant are `θ`'s.
- The head's products and attention's: from `HEAD_K` the Program gives no bound, since it doesn't say which products
  are the head's, where the trace's figure keeps the weight gradients; a product of two activations, a weight
  gradient or attention's, is left out, where the trace keeps the weight gradients.
- β and b: every operand at most min(16, β(k)) and every output at most 16 bits, where the trace's weights and
  outputs are at their Definition's width, 32.
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

from training_slowdown import calculator
from training_slowdown import work as TW
from training_slowdown import gates as G
from training_slowdown.capture import capture_step, products

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "benchmarks" / "circuit_properties"))
from circuit_properties import points as PT  # noqa: E402
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
    #: by k: the Program's work an iteration, the trace's priced rows, `work.he_of_rows`' M and `calculator.slowdown`
    #: on them
    work: dict
    rows: dict
    he: dict
    fig: dict
    size: float


def read(make, loss, batch: torch.Tensor) -> Reading:
    """The step of the model ``make()`` builds, on ``batch`` under ``loss`` and AdamW, as a Program and as a trace at
    each k."""
    with torch.device("meta"):
        model = make()
    step = step_program(model, loss, AdamW(**LF.ADAMW), torch.empty_like(batch, device="meta"))
    torch.manual_seed(0)
    model = make()
    g = capture_step(model, torch.optim.AdamW(model.parameters(), **LF.ADAMW), loss, batch)
    raw = products.raw_on_dial(g)
    size = products.step_gates(g)
    r = Reading(step, {}, {}, {}, {}, size)
    for k in KS:
        blk = raw if k == raw["k"] else raw["at"][str(int(k))]
        r.rows[k] = [products.priced(row, raw["capture"]) for row in blk["rows"]]
        record = TW.products_of_trace({"k": k, "rows": r.rows[k], "capture": raw["capture"], "p_match": blk["p_match"],
                                       "tokens": g.tokens})
        r.he[k] = record["he"]
        r.fig[k] = calculator.slowdown({"c_gates": C, "k_bits": k}, record, {"gates": size})
        r.work[k] = iteration(step, k)
    return r


@pytest.fixture(scope="module")
def readings() -> Reading:
    return read(lambda: LF.LM(LF.SIZES["tiny"], ctx=CTX, vocab=64), LF.square_loss,
                torch.zeros(*BATCH, dtype=torch.long))


#: a dense step: no table lookup, a float batch of 32 rows into a linear layer whose contraction, 1024, a block of
#: up to k = 800000 bits can't make its output's entries below their width from
DENSE = (32, 1024)


def _dense() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(DENSE[1], 192, bias=False), torch.nn.SiLU(),
                               torch.nn.Linear(192, 64, bias=False), torch.nn.SiLU(), torch.nn.Linear(64, 64, bias=False))


def _square(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).square().sum()


def _ordered(beta, weight) -> tuple:
    """An operand pair's β, the weight's first, two of one kind the larger first."""
    a, b = beta
    if weight[0] != weight[1]:
        return (b, a) if weight[1] else (a, b)
    return tuple(sorted((a, b), reverse=True))


def fits(work: WK.Work, he: dict) -> bool:
    """Whether the trace's M holds every group the Program charges: its MACs (a token-indexed group's at the work's
    `keep`, as M's are) go into M's groups of its kinds and contraction whose β is at least its own on each operand,
    none past its MACs (a flow from the charged groups to M's)."""
    mine = [(g, g.macs * (1.0 if all(g.weight) else work.keep)) for g in work.groups]
    theirs = he["groups"]
    ok = [[j for j, t in enumerate(theirs) if t["weight"] == list(g.weight) and t["k_dim"] == g.k_dim and
           all(m <= th + 1e-9 for m, th in zip(_ordered(g.beta, g.weight), _ordered(t["beta"], t["weight"])))]
          for g, _m in mine]
    left = [m for _g, m in mine]
    room = [t["macs"] for t in theirs]
    flow = {}
    while True:
        prev, queue, seen = {}, [("s", None)], {("s", None)}
        end = None
        while queue and end is None:
            node = queue.pop(0)
            if node[0] == "s":
                nxt = [("g", i) for i, m in enumerate(left) if m > 1e-6]
            elif node[0] == "g":
                nxt = [("t", j) for j in ok[node[1]]]
            else:
                nxt = [("g", i) for i in range(len(mine)) if flow.get((i, node[1]), 0) > 1e-6]
                if room[node[1]] > 1e-6:
                    end = node
                    break
            for v in nxt:
                if v not in seen:
                    seen.add(v)
                    prev[v] = node
                    queue.append(v)
        if end is None:
            return all(m <= 1e-6 * max(1.0, sum(left)) for m in left)
        path, v = [], end
        while v != ("s", None):
            path.append(v)
            v = prev[v]
        path.reverse()
        amount = min([room[end[1]], left[path[0][1]]] +
                     [flow[(b[1], a[1])] for a, b in zip(path, path[1:]) if a[0] == "t"])
        for a, b in zip(path, path[1:]):
            if a[0] == "g":
                flow[(a[1], b[1])] = flow.get((a[1], b[1]), 0) + amount
            else:
                flow[(b[1], a[1])] -= amount
        left[path[0][1]] -= amount
        room[end[1]] -= amount


def trace_blocks(got: dict) -> float:
    """The blocks of the calculator's choice of the trace's work."""
    return got["bound"]["blocks"] if got["bound"]["available"] else 0.0


def blocks(work: WK.Work, F_: float | None = None) -> float:
    b = WK.bound(work, C, F=F_)
    return b.bound.blocks if b else 0.0


def trace_work(r: Reading, k: float) -> WK.Work:
    """The trace's M as a work, each activation at most β(k) as the calculator charges it, its MACs already at M's share
    a sequence's start leaves."""
    bk = MC.beta(k)
    return WK.Work(k, r.size, [WK.Group("trace", tuple(g["width"]), tuple(g["weight"]), g["b_y"], g["k_dim"], g["macs"],
                                        g["wires"], frozenset(),
                                        beta=tuple(b if w else min(b, bk) for b, w in zip(g["beta"], g["weight"])))
                               for g in r.he[k]["groups"]])


def never_above(r: Reading) -> None:
    """Each of ``r``'s works at each k: every group it charges fits the trace's M, and its blocks, at the calculator's
    F on the trace and at its own, are at most what the trace's M gives under the same choice of sub-work."""
    for k, it in r.work.items():
        assert fits(it, r.he[k]), k
        own = WK.bound(it, C)
        theirs = (r.fig[k]["figure"] or {}).get("F")
        for F_ in {F_ for F_ in (theirs, own.bound.F if own else 1.0) if F_ is not None}:
            assert blocks(it, F_) <= blocks(trace_work(r, k), F_), (k, F_)


@pytest.mark.slow
def test_the_programs_bound_is_never_above_the_traces(readings):
    """Derives the step, captures it and reads it at each k (about a minute): what it catches is a Program reading that
    charges work the trace's M doesn't hold, a β above the trace's, or more blocks than the trace's M gives under the
    same choice of sub-work and F, at any k of the dial. It charges weights' products with activations, and leaves out
    as gathered more than the trace does: what its rule doesn't reach."""
    r = readings
    never_above(r)
    for k in KS:
        assert trace_blocks(r.fig[k]) > 0
        it = r.work[k]
        assert it.groups and all(g.weight == (False, True) for g in it.groups)
        assert it.left_out["gathered"] > r.he[k]["gathered_macs"]


@pytest.mark.slow
def test_a_dense_steps_rows_may_repeat_so_its_circuit_alone_charges_nothing():
    """A step with no table lookup (`_dense`): its batch is rows of values, two of which may
    be equal, so every product of its hidden rows by a shared weight carries rows that may repeat and is left out, where
    the trace, whose contraction a block can't make the hidden rows below their width up to k = 800000, charges them."""
    r = read(_dense, _square, torch.randn(*DENSE))
    never_above(r)
    for k in KS:
        it = r.work[k]
        assert it.groups == [] and it.left_out["gathered"] > 0 and it.left_out["activations"] == 0
        assert (trace_blocks(r.fig[k]) > 0) == (k <= 800_000), k


@pytest.mark.slow
def test_every_product_pays_at_most_what_the_trace_pays_its_shape(readings):
    """Each of the Program's products (m, K, n) is a shape the trace has a product of, each operand's rate at most
    the least the trace's products of that shape pay it."""
    r = readings
    traced = sum(row["count"] * row["macs"] for row in r.rows[KS[0]])
    for k, it in r.work.items():
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
            assert g.rates[0] <= least[0] + 1e-9 and g.rates[1] <= least[1] + 1e-9, (k, g.dims, g.rates, least)
        assert it.r_step is None and it.left_out["self"] == 0


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


def test_the_point_evaluates_each_iteration_it_runs_once(readings):
    """The reading's test at a point (`points.check`) passes the step's loop, its charged claims among them, and
    costs about one evaluation of the step: the two iterations the reader reads as distinct contexts, of two and of
    three, each the step's primitive words once, at one point, none by the reference a word at a time."""
    memo: dict = {}

    def words(fn) -> int:
        if fn.id not in memo:
            memo[fn.id] = sum((nd.out.leaves if isinstance(nd.fn, PrimitiveDefinition) else
                               (nd.n if nd.form in ("batch", "scan") else 1) * words(nd.fn))
                              for nd in fn.body.nodes if not is_input(nd.fn))
        return memo[fn.id]

    for n in (2, 3):
        p = loop(readings.step, n)
        _w, R, root, reasons = WK._read(p, K0, WK.PRODUCTS, WK.LOOKUPS)
        t = PT.check(p, R, root, reasons)
        assert (t.words, t.points, t.claims, dict(t.reference)) == (2 * words(readings.step.root), 1, 23, {})


def test_the_readings_constants_are_the_benchmarks():
    assert WK.ACC_BITS == products.ACC_BITS and WK.SHARED_PREFIX == TW.SHARED_PREFIX and WK.HEAD_K == TW.HEAD_K
    for w in (1.0, 16.0, 32.0):
        for rate in (0.5 * w, WK.PAID * w, w):
            assert WK.paid(rate, w) == TW.is_paid(rate, w)
    assert MC.BETA == calculator.beta_rows()
    for k in products.DIAL_K:
        for w, rank in ((32.0, 4.0), (16.0, 64.0)):
            assert WK.k_star(k, w) == pytest.approx(TW.k_star(k, w))
            assert WK.beta_eff(w, rank, k) == pytest.approx(TW.beta_eff(w, rank, k))
