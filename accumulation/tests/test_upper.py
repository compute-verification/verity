"""Tests for the constructive upper bound (``accumulation/bounds/upper.py``).

Run with ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_upper.py -q``.
"""

from __future__ import annotations

import time

import pytest

from accumulation.algorithms import registry as R
from accumulation.bounds.upper import Plan, check_plan, upper_bound
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.graph.opgraph import Edge, Op, OpGraph

TINY_WL = Workload(tokens=8, seq=4, population=2, rank=2, local_steps=2, gen_tokens=2)
TINY_MODELS = ("tiny2", "tiny-moe")
X_GRID = (64, 256, 4096, 1 << 20)
F_GRID = (1 << 12, 1 << 40)
BIG = 1 << 60


# ---------------------------------------------------------------------------------------------------------
# hand-built micro graph
# ---------------------------------------------------------------------------------------------------------

def micro_matmul(M: int, N: int, K: int, *, a_role: str = "accumulated", b_role: str = "accumulated",
                 width: int = 16) -> OpGraph:
    """One matmul ``C = A . B^T`` on root parameters ``A`` (``M x K``) and ``B`` (``N x K``)."""
    g = OpGraph()
    A = g.add_tensor(kind="param", name="A", leaves=M * K, width=width, role=a_role)
    B = g.add_tensor(kind="param", name="B", leaves=N * K, width=width, role=b_role)
    g.params = {"A": A.id, "B": B.id}
    op = Op(id=0, key=((), 0), fn_id="AccMatmulT_v1", kind="matmul", statics={"M": M, "N": N, "K": K, "CH": 1},
            work_per_copy=M * N * K, copies=1, out=-1)
    out = g.add_tensor(kind="op", name="matmul#0", leaves=M * N, width=width, producer=0)
    op.out = out.id
    op.inputs = [Edge(A.id, M * K), Edge(B.id, N * K)]
    g.ops.append(op)
    g.outputs = [out.id]
    return g


def test_micro_matmul_expected_tile_cost():
    # A, B: 4x4 at 16 bit = 32 B each.  With X = 40 B and a whole-K tile (k = K = 4) the best grid is
    # a = b = 2 (imports 16 + 16 = 32 <= 40): A is read twice and B is read twice -> 128 B.  Every other
    # legal grid costs more: (a=1, b=4) and (a=4, b=1) read one operand 4 times (160 B); splitting K
    # (k = 2, a = b = 4) saves nothing on the operands and adds 4 B x 2 partials per output (192 B).
    g = micro_matmul(4, 4, 4)
    ub = upper_bound(g, F=BIG, X=40)
    check_plan(ub.plan, g, BIG, 40)
    assert ub.total == 128, ub.plan.units
    assert set(u.kind for u in ub.plan.units) == {"tile"}
    tile = ub.plan.units[0]
    # (a=3, b=2) is a ragged equivalent of (a=2, b=2): both read each operand twice
    assert tile.detail["k"] == 4 and tile.detail["nk"] == 1 and tile.count == 4 and tile.imports_max <= 40
    assert ub.per_op == {0: 128}


def test_micro_matmul_single_ru_equals_nonfixed_input_bytes():
    g = micro_matmul(4, 4, 4)
    ub = upper_bound(g, F=BIG, X=64)
    check_plan(ub.plan, g, BIG, 64)
    assert ub.total == 64 == g.tensors[0].bytes + g.tensors[1].bytes
    assert ub.n_units == 1
    # and with an enormous X the answer is the same
    assert upper_bound(g, F=BIG, X=1 << 40).total == 64


def test_micro_matmul_fixed_operand_is_free():
    g = micro_matmul(8, 6, 4, b_role="fixed")
    # only the A rows are ever imported: one row (8 B) fits X = 8 and every row is read exactly once
    for X in (8, 16, 64, 1 << 20):
        ub = upper_bound(g, F=BIG, X=X)
        check_plan(ub.plan, g, BIG, X)
        assert ub.total == 8 * 4 * 2, (X, ub.plan.units)
    # X = 7 still works by splitting K down to single columns with chained 4-byte accumulators
    # (one A element 2 B + one accumulator 4 B); below that nothing is legal
    ub7 = upper_bound(g, F=BIG, X=7)
    check_plan(ub7.plan, g, BIG, 7)
    assert ub7.total > 8 * 4 * 2
    with pytest.raises(ValueError):
        upper_bound(g, F=BIG, X=5)


def test_micro_matmul_partial_sums_are_reduced_and_checked():
    # K large, X small: the contraction must be split, so partial accumulators appear and get reduced
    g = micro_matmul(2, 2, 64)
    X = 40
    ub = upper_bound(g, F=BIG, X=X)
    check_plan(ub.plan, g, BIG, X)
    kinds = {u.kind for u in ub.plan.units}
    tile = next(u for u in ub.plan.units if u.kind == "tile")
    assert tile.detail["nk"] > 1
    if tile.detail["mode"] == "star":
        assert "reduce" in kinds
        emitted = sum(u.partial_out.get(0, 0) for u in ub.plan.units)
        consumed = sum(u.partial_in.get(0, 0) for u in ub.plan.units)
        assert emitted == consumed > 0
    # tampering with the plan must be caught
    bad = Plan(units=[u for u in ub.plan.units if u.kind != "reduce"], F=BIG, X=X, recompute=True, f_mode="upstream")
    if bad.units != ub.plan.units:
        with pytest.raises(AssertionError):
            check_plan(bad, g, BIG, X)
    tile.imports_max = X + 1
    with pytest.raises(AssertionError):
        check_plan(ub.plan, g, BIG, X)


def test_micro_matmul_f_limits_tile_work():
    g = micro_matmul(4, 4, 8)
    # upstream rule: a gate's chain segment must be <= F, so k <= F - 1 (the rounding gate counts 1)
    ub = upper_bound(g, F=4, X=1 << 20)
    check_plan(ub.plan, g, 4, 1 << 20)
    assert all(u.up_max <= 4 for u in ub.plan.units)
    # the cruder work rule: the whole RU work must be <= F
    ub2 = upper_bound(g, F=4, X=1 << 20, f_mode="work")
    check_plan(ub2.plan, g, 4, 1 << 20)
    assert all(u.work_max <= 4 for u in ub2.plan.units)
    assert ub2.total >= ub.total
    # F = 1: single-MAC chain segments (this micro op has no extra rounding work per output)
    ub1 = upper_bound(g, F=1, X=1 << 20)
    check_plan(ub1.plan, g, 1, 1 << 20)
    assert all(u.detail.get("k", 1) == 1 for u in ub1.plan.units if u.kind == "tile")
    with pytest.raises(ValueError):
        upper_bound(g, F=0, X=1 << 20)


# ---------------------------------------------------------------------------------------------------------
# registered algorithms on tiny graphs
# ---------------------------------------------------------------------------------------------------------

def _tiny_cases():
    out = []
    for name, alg in R.ALGORITHMS.items():
        for model in TINY_MODELS:
            if alg.supports(MODELS[model], TINY_WL) is None:
                out.append((name, model))
    return out


_GRAPH_CACHE: dict[tuple[str, str], tuple] = {}


def _tiny_graph(name: str, model: str):
    key = (name, model)
    if key not in _GRAPH_CACHE:
        bp = R.build(name, MODELS[model], TINY_WL)
        _GRAPH_CACHE[key] = (extract(bp), bp)
    return _GRAPH_CACHE[key]


@pytest.mark.parametrize("name,model", _tiny_cases())
def test_tiny_graphs_all_x_f(name, model):
    g, bp = _tiny_graph(name, model)
    for X in X_GRID:
        for F in F_GRID:
            ub = upper_bound(g, F, X, program=bp.program)
            check_plan(ub.plan, g, F, X)
            assert ub.total == ub.plan.total > 0
            assert ub.n_units == sum(u.count for u in ub.plan.units)
            assert sum(ub.per_op.values()) == ub.total
            assert ub.kappa >= 0


@pytest.mark.parametrize("name,model", _tiny_cases())
def test_tiny_graphs_variants(name, model):
    """Literal partitions (no recompute), the cruder F rule, and the un-fused construction stay legal."""
    g, bp = _tiny_graph(name, model)
    X, F = 4096, 1 << 40
    for kw in (dict(recompute=False), dict(f_mode="work"), dict(fuse_attention=False, group_operands=False,
                                                                 fuse_epilogue=False, fuse_opsets=False),
               dict(program=None)):
        ub = upper_bound(g, F, X, **({"program": bp.program} | kw))
        check_plan(ub.plan, g, F, X)
        if not kw.get("recompute", True):
            assert all(u.kind != "reduce" for u in ub.plan.units)


@pytest.mark.parametrize("name,model", _tiny_cases())
def test_u_monotone_in_x_and_f(name, model):
    g, bp = _tiny_graph(name, model)
    xs = (64, 128, 256, 1024, 4096, 1 << 16, 1 << 20)
    fs = (1 << 10, 1 << 12, 1 << 16, 1 << 24, 1 << 40)
    for F in F_GRID:
        prev = None
        for X in xs:
            u = upper_bound(g, F, X, program=bp.program).total
            assert prev is None or u <= prev, (name, model, F, X, u, prev)
            prev = u
    for X in (256, 4096, 1 << 20):
        prev = None
        for F in fs:
            u = upper_bound(g, F, X, program=bp.program).total
            assert prev is None or u <= prev, (name, model, X, F, u, prev)
            prev = u


def test_large_x_reaches_source_bytes():
    """With X and F unbounded the whole graph is one RU: U equals the non-fixed root bytes read."""
    for name, model in _tiny_cases():
        g, bp = _tiny_graph(name, model)
        ub = upper_bound(g, 1 << 40, 1 << 30, program=bp.program)
        read = {e.src for o in g.ops for e in o.inputs}
        src = sum(t.bytes for t in g.tensors if t.kind == "param" and t.role != "fixed" and t.id in read)
        assert ub.total == src, (name, model, ub.total, src)
        assert ub.n_units == 1


# ---------------------------------------------------------------------------------------------------------
# against the lower bound
# ---------------------------------------------------------------------------------------------------------

def test_upper_ge_lower_tiny():
    lower = pytest.importorskip("accumulation.bounds.lower")
    for name, model in _tiny_cases():
        g, bp = _tiny_graph(name, model)
        for X in X_GRID:
            for F in F_GRID:
                U = upper_bound(g, F, X, program=bp.program).total
                L = lower.lower_bound(g, F, X).total
                assert U >= L, (name, model, X, F, U, L)


# ---------------------------------------------------------------------------------------------------------
# llama3-8b
# ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llama8b_pretrain():
    bp = R.build("pretrain-dense", MODELS["llama3-8b"], Workload(tokens=4096, seq=4096))
    return extract(bp), bp


def test_llama8b_pretrain_dense(llama8b_pretrain):
    g, bp = llama8b_pretrain
    F, X = int(1e12), 5_000_000
    t0 = time.time()
    ub = upper_bound(g, F, X, program=bp.program)
    t1 = time.time()
    check_plan(ub.plan, g, F, X)
    t2 = time.time()
    assert t2 - t0 < 60, f"upper_bound {t1 - t0:.1f}s + check_plan {t2 - t1:.1f}s"
    tokens = 4096
    print(f"\nllama3-8b pretrain-dense F=1e12 X=5MB: U={ub.total / 1e9:.1f} GB  U/token={ub.total / tokens / 1e6:.1f} MB"
          f"  n_units={ub.n_units}  kappa={ub.kappa:.1f}  ({t1 - t0:.1f}s + check {t2 - t1:.1f}s)")
    print("by class (GB):", {k: round(v / 1e9, 2) for k, v in sorted(ub.by_class.items(), key=lambda kv: -kv[1])})
    assert ub.kappa > 0
    assert all(u.imports_max <= X and u.up_max <= F for u in ub.plan.units)


def test_llama8b_upper_ge_lower(llama8b_pretrain):
    lower = pytest.importorskip("accumulation.bounds.lower")
    g, bp = llama8b_pretrain
    F, X = int(1e12), 5_000_000
    U = upper_bound(g, F, X, program=bp.program).total
    L = lower.lower_bound(g, F, X).total
    print(f"\nllama3-8b pretrain-dense: U={U / 1e9:.1f} GB  L={L / 1e9:.1f} GB  U/L={U / L:.2f}")
    assert U >= L
