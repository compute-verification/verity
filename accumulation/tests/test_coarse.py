"""Tests for the coarse constructive upper bound (``accumulation/bounds/coarse.py``).

Run with ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_coarse.py -q``.

Policy calibration: ``F = F_hat * fwd``, ``G = G_hat * fwd`` with ``fwd`` the work of one honest inference session.
Legality counts every primitive's ``.work`` (softmax, rmsnorm, ... included), so the sanity condition "the whole
honest session is one legal RU at ``G_hat = 1``" holds by construction only when ``fwd = g.total_work()`` of the
inference circuit.  ``adw(g).all_macs`` (the SPEC's literal calibration) is 0.7 % smaller at 8B and 2.3x smaller
at tiny2; ``test_inference_sanity_all_macs_calibration`` records where that calibration breaks the condition.
"""

from __future__ import annotations

import pytest

from accumulation.algorithms import registry as R
from accumulation.bounds.adw import adw
from accumulation.bounds.coarse import (CoarsePlan, CoarseUnit, check_coarse_plan, gate_level_up,
                                        plan_gate_assignment, upper_coarse)
from accumulation.configs import MODELS, Workload
from accumulation.exact.solve import flatten, is_legal, partition_cost
from accumulation.graph import extract

Q_INF = 16          # tiny2 "Q_inf"
SEQ = 4
F_HATS = (1, 1.5, 2)
G_HATS = (1, 4, 16, 100)


def _build(name: str, model: str, wl: Workload):
    bp = R.build(name, MODELS[model], wl)
    return bp, extract(bp)


@pytest.fixture(scope="module")
def inference():
    bp, g = _build("inference-dense", "tiny2", Workload(tokens=Q_INF, seq=Q_INF))
    return bp, g, g.total_work()


# ---------------------------------------------------------------------------------------------------------
# inference sanity
# ---------------------------------------------------------------------------------------------------------

def test_inference_dense_one_unit_at_every_policy_point(inference):
    bp, g, fwd = inference
    for fh in F_HATS:
        for gh in G_HATS:
            plan = upper_coarse(g, int(fh * fwd), int(gh * fwd), program=bp.program)
            check_coarse_plan(g, plan, int(fh * fwd), int(gh * fwd), program=bp.program)
            assert plan.n_units == 1 and plan.detail["family"] == "single", (fh, gh, plan.notes)
            assert plan.total == 4 * Q_INF, (fh, gh, plan.total, plan.by_class)
            assert plan.by_class == {"tokens": 4 * Q_INF}
            u = plan.units[0]
            assert u.work_max == fwd and u.up_max <= u.work_max <= int(fh * fwd)


def test_inference_sanity_all_macs_calibration(inference):
    """With ``fwd = all_macs`` the whole session has ``work > G`` at ``G_hat = 1`` and ``Up > F`` at every
    ``F_hat <= 2`` on tiny2 (non-matmul work is 2.3x the matmul work there; 0.7 % at 8B), so the planner has to
    split and the 4 B/token sanity value is not reproduced.  Once ``F, G >= total_work`` it is."""
    bp, g, W = inference
    fwd = adw(g).all_macs
    assert W > fwd
    plan = upper_coarse(g, int(1.5 * fwd), fwd, program=bp.program)
    assert plan.detail["family"] != "single" and plan.total > 4 * Q_INF
    plan = upper_coarse(g, W, W, program=bp.program)
    assert plan.detail["family"] == "single" and plan.total == 4 * Q_INF


def test_inference_split_when_G_small(inference):
    bp, g, fwd = inference
    F, G = fwd, fwd // 2
    plan = upper_coarse(g, F, G, program=bp.program)
    check_coarse_plan(g, plan, F, G, program=bp.program)
    assert plan.n_units > 1 and plan.total > 4 * Q_INF
    assert max(u.work_max for u in plan.units) <= G


# ---------------------------------------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------------------------------------

def test_checker_rejects_work_over_G(inference):
    bp, g, fwd = inference
    members = tuple((o.id, "whole", 0, 0) for o in g.ops)
    plan = CoarsePlan([CoarseUnit(members, "all")], F=fwd, G=fwd - 1)
    with pytest.raises(AssertionError, match="work .* > G"):
        check_coarse_plan(g, plan, fwd, fwd - 1, program=bp.program)


def test_checker_rejects_up_over_F(inference):
    bp, g, fwd = inference
    members = tuple((o.id, "whole", 0, 0) for o in g.ops)
    plan = CoarsePlan([CoarseUnit(members, "all")], F=fwd - 1, G=None)
    with pytest.raises(AssertionError, match="Up bound .* > F"):
        check_coarse_plan(g, plan, fwd - 1, None, program=bp.program)
    # the same unit is legal at F = work (Up of the last gate is the whole unit)
    check_coarse_plan(g, plan, fwd, None, program=bp.program)
    assert plan.units[0].up_max <= fwd


def test_checker_rejects_bad_coverage(inference):
    bp, g, fwd = inference
    ops = [o.id for o in g.ops]
    missing = CoarsePlan([CoarseUnit(tuple((o, "whole", 0, 0) for o in ops[:-1]), "partial")], F=fwd, G=None)
    with pytest.raises(AssertionError):
        check_coarse_plan(g, missing, fwd, None, program=bp.program)
    twice = CoarsePlan([CoarseUnit(tuple((o, "whole", 0, 0) for o in ops), "a"),
                        CoarseUnit(((ops[0], "whole", 0, 0),), "b")], F=fwd, G=None)
    with pytest.raises(AssertionError):
        check_coarse_plan(g, twice, fwd, None, program=bp.program)


def test_checker_recomputes_from_graph(inference):
    """The checker does not trust the planner: tampering with a unit's statistics is overwritten."""
    bp, g, fwd = inference
    plan = upper_coarse(g, fwd, fwd, program=bp.program)
    plan.units[0].imports_total = 0
    check_coarse_plan(g, plan, fwd, fwd, program=bp.program)
    assert plan.total == 4 * Q_INF


# ---------------------------------------------------------------------------------------------------------
# training: multi-step local-sgd
# ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sgd():
    out = {}
    for K in (1, 2, 3):
        wl = Workload(tokens=K * Q_INF, seq=SEQ, local_steps=K)
        out[K] = _build("local-sgd", "tiny2", wl)
    return out


def _weights_bytes(g) -> int:
    return sum(t.leaves * t.width // 8 for t in g.tensors
               if t.kind == "param" and t.role in ("accumulated", "carried"))


def test_training_marginal_positive_and_consistent(inference, sgd):
    """``U(K=3) - U(K=2) > 0`` at every calibrated point, for the free search and for fixed ``(m, s)`` choices."""
    _, _, fwd = inference
    F = int(1.5 * fwd)
    for gh in (1, 4, 16):
        G = gh * fwd
        U = {}
        for K in (2, 3):
            bp, g = sgd[K]
            plan = upper_coarse(g, F, G, program=bp.program)
            check_coarse_plan(g, plan, F, G, program=bp.program)
            assert plan.detail["family"] == "convex-block", plan.notes
            assert plan.detail["acyclic"] is True
            # the search's memoised estimate must be the checked total (a structural-key collision would
            # make the search pick a plan on a wrong cost)
            assert plan.detail["planner_cost"] == plan.total, (gh, K, plan.detail["planner_cost"], plan.total)
            U[K] = plan.total
        assert U[3] > U[2] > 0, (gh, U)
        # fixed (m, s, kf = 1, T = 0): the per-step convex block family; the marginal is one step's traffic
        for m in (1, 2):
            for s in (1, 2, 4):
                Ums = {}
                for K in (2, 3):
                    bp, g = sgd[K]
                    plan = upper_coarse(g, F, G, program=bp.program, search=dict(m=[m], s=[s], kf=[1], T=[0]))
                    check_coarse_plan(g, plan, F, G, program=bp.program)
                    if plan.detail["family"] == "convex-block":
                        assert plan.detail["m"] == m and plan.detail["s"] == s and plan.detail["T"] == 0
                    Ums[K] = plan.total
                assert Ums[3] > Ums[2], (gh, m, s, Ums)
                assert Ums[3] >= U[3] and Ums[2] >= U[2]      # the free search is never worse


def test_pretrain_weights_enter_once(inference, sgd):
    """K = 1.  When the whole step is one legal RU, ``W_0`` enters exactly once.  Under ``F = 1.5 fwd`` the step
    (~3 fwd of work) must be cut; convex units import a layer's weights in its forward unit and again in its
    backward unit (a joint fwd+bwd chunk would be a cycle with the units above it), so at most twice."""
    _, _, fwd = inference
    bp, g = sgd[1]
    W = _weights_bytes(g)
    plan = upper_coarse(g, 10 * fwd, 10 * fwd, program=bp.program)
    check_coarse_plan(g, plan, 10 * fwd, 10 * fwd, program=bp.program)
    assert plan.by_class["weights"] == W
    F, G = int(1.5 * fwd), 4 * fwd
    plan = upper_coarse(g, F, G, program=bp.program)
    check_coarse_plan(g, plan, F, G, program=bp.program)
    assert W < plan.by_class["weights"] <= 2 * W, plan.by_class
    assert plan.by_class["tokens"] == 4 * Q_INF + 4 * Q_INF     # toks + targets


def test_checker_rejects_cyclic_joint_plan(inference, sgd):
    """The superseded ``joint`` fwd+bwd chunk units form 2-cycles with the units above them (SPEC §0):
    the checker names the cycle and rejects the plan; the convex plan of the same point passes.  The cyclic
    family stays reachable for comparison only (``search=dict(acyclic=False)``) and is flagged as such."""
    _, _, fwd = inference
    bp, g = sgd[2]
    F, G = int(0.5 * fwd), int(0.5 * fwd)
    cyc = upper_coarse(g, F, G, program=bp.program, search=dict(acyclic=False, fb=["joint"], kf=[1], m=[1], s=[2]))
    assert cyc.detail["family"] == "block-cyclic" and cyc.detail["acyclic"] is False and cyc.detail["checked"] is None
    assert any("CYCLIC" in n for n in cyc.notes)
    with pytest.raises(AssertionError, match="RU quotient is cyclic"):
        check_coarse_plan(g, cyc, F, G, program=bp.program)
    # kf > 1 chunk fusion across steps is a cycle as well (step k+1's forward of chunk c reads W_{k+1}, produced
    # by the update that closes the chunk's dW chain in a later sequence unit of the same fused unit family)
    cyc2 = upper_coarse(g, int(1.5 * fwd), int(1.5 * fwd), program=bp.program,
                        search=dict(acyclic=False, fb=["joint"], kf=[2], m=[1], s=[2]))
    if cyc2.detail["family"] == "block-cyclic":
        assert cyc2.detail["acyclic"] is False
        with pytest.raises(AssertionError, match="RU quotient is cyclic"):
            check_coarse_plan(g, cyc2, int(1.5 * fwd), int(1.5 * fwd), program=bp.program)
    # the default (convex) plan at the same point passes the same checker
    plan = upper_coarse(g, F, G, program=bp.program)
    check_coarse_plan(g, plan, F, G, program=bp.program)
    assert plan.detail["acyclic"] is True and plan.detail["checked"]["acyclic"] is True
    assert plan.detail["family"] == "convex-block"


def test_checker_finds_cycle_in_hand_built_plan(inference):
    """Two units that read each other's outputs: rejected with the cycle spelled out, whatever their work."""
    bp, g, fwd = inference
    ops = [o.id for o in g.ops]
    # unit A: even ops, unit B: odd ops of the forward chain -> A -> B -> A
    a = CoarseUnit(tuple((o, "whole", 0, 0) for o in ops[0::2]), "even")
    b = CoarseUnit(tuple((o, "whole", 0, 0) for o in ops[1::2]), "odd")
    plan = CoarsePlan([a, b], 10 * fwd, 10 * fwd)
    with pytest.raises(AssertionError, match=r"RU quotient is cyclic .*even.*odd|RU quotient is cyclic .*odd.*even"):
        check_coarse_plan(g, plan, 10 * fwd, 10 * fwd, program=bp.program)
    # contiguous bands in topological order are convex: accepted
    half = len(ops) // 2
    plan = CoarsePlan([CoarseUnit(tuple((o, "whole", 0, 0) for o in ops[:half]), "lo"),
                       CoarseUnit(tuple((o, "whole", 0, 0) for o in ops[half:]), "hi")], 10 * fwd, 10 * fwd)
    check_coarse_plan(g, plan, 10 * fwd, 10 * fwd, program=bp.program)
    assert plan.detail["acyclic"] is True


def test_forward_only_plans_are_acyclic(inference):
    """Inference / forward-nonfixed: every plan the planner returns passes the acyclicity check."""
    bp, g, fwd = inference
    for frac in (1.0, 0.3, 0.1):
        plan = upper_coarse(g, int(1.5 * frac * fwd), int(frac * fwd), program=bp.program)
        assert plan.detail["acyclic"] is True
    bp, g = _build("forward-nonfixed", "tiny2", Workload(tokens=Q_INF, seq=SEQ))
    W = g.total_work()
    for frac in (2.0, 0.5, 0.1):
        plan = upper_coarse(g, int(frac * W), int(frac * W), program=bp.program)
        check_coarse_plan(g, plan, int(frac * W), int(frac * W), program=bp.program)
        assert plan.detail["acyclic"] is True


def test_training_gate_level_cross_check(inference, sgd):
    """The coarse plan, mapped to a gate assignment, is legal and costs exactly ``U`` under the exact solver's
    accounting (dense circuits: no inexact edges).  The exact module's ``is_legal`` allocates ~20 KB per gate
    (2.3 GB at K = 3's 125k gates, above the 2.5 GB process cap), so the gate-level ``G`` check runs at K = 2
    only; K = 3 is re-costed and relies on the coarse checker for legality."""
    _, _, fwd = inference
    for K in (2, 3):
        bp, g = sgd[K]
        flat = flatten(bp)
        for gh in (0.5, 2):
            F, G = int(1.5 * fwd), int(gh * fwd)
            plan = upper_coarse(g, F, G, program=bp.program)
            check_coarse_plan(g, plan, F, G, program=bp.program)
            asg = plan_gate_assignment(g, plan, bp.program)
            assert all(x in asg for x in flat.gates)
            if K == 2:
                ok, why = is_legal(flat, asg, None, None, G)
                assert ok, why
            assert partition_cost(flat, asg) == plan.total


# ---------------------------------------------------------------------------------------------------------
# MoE forward (dynamic rollout): layer recognition, expert-row slicing, exact cross-check
# ---------------------------------------------------------------------------------------------------------

MOE_Q_INF = 8       # tiny-moe "Q_inf" (one sequence of 8 tokens)


@pytest.fixture(scope="module")
def moe_inference():
    bp, g = _build("inference-moe", "tiny-moe", Workload(tokens=MOE_Q_INF, seq=MOE_Q_INF))
    return bp, g, g.total_work()


def test_inference_moe_one_unit_at_default_point(moe_inference):
    """The MoE forward with fixed weights is ONE legal RU at the default policy point (and every calibrated
    point): imports = tokens only, 4 B/token."""
    bp, g, fwd = moe_inference
    for fh in F_HATS:
        for gh in G_HATS:
            plan = upper_coarse(g, int(fh * fwd), int(gh * fwd), program=bp.program)
            check_coarse_plan(g, plan, int(fh * fwd), int(gh * fwd), program=bp.program)
            assert plan.n_units == 1 and plan.detail["family"] == "single", (fh, gh, plan.notes)
            assert plan.by_class == {"tokens": 4 * MOE_Q_INF}
            assert plan.detail["anatomy"]["dominant"] == "whole-model"


def test_forward_nonfixed_moe_block_family(moe_inference):
    """Dynamic MoE rollout at the default point: the inlined MoE layers are recognised (no band / fine
    fallback), expert kernels are cut by instance rows (``irows``), every plan is checker-legal and acyclic,
    weights enter at most once per (layer chunk x token chunk), and the anatomy reports active experts."""
    _, _, fwd = moe_inference
    F, G = int(1.5 * fwd), int(4 * fwd)
    for Q in (16, 32, 128):
        bp, g = _build("forward-nonfixed-moe", "tiny-moe", Workload(tokens=Q, seq=MOE_Q_INF))
        plan = upper_coarse(g, F, G, program=bp.program, fwd=fwd)
        check_coarse_plan(g, plan, F, G, program=bp.program)
        assert plan.detail["acyclic"] is True
        assert plan.detail["family"] in ("convex-block", "single", "bands"), plan.notes
        assert plan.detail["family"] != "fine" and not plan.detail.get("fallback"), plan.notes
        assert plan.detail["structure"]["layers"] == [2]                # dense layer 0 + inlined MoE layer 1
        W = _weights_bytes(g)
        assert plan.by_class["tokens"] == 4 * Q
        assert W <= plan.by_class["weights"] <= W * max(1, Q // MOE_Q_INF)
        an = plan.detail["anatomy"]
        assert an["n_ru"] == plan.n_units and an["work_max_fwd"] <= 4.0 + 1e-9
        assert "active_experts_per_ru" in an and an["active_experts_per_ru"]["max"] <= 4
        if plan.detail["family"] == "convex-block" and plan.detail["s_f"] < Q // MOE_Q_INF:
            assert any(m[1] == "irows" for u in plan.units for m in u.members)
        # gate-level cross-check: the plan mapped to gates costs exactly U and is legal under the exact accounting
        flat = flatten(bp)
        asg = plan_gate_assignment(g, plan, bp.program)
        assert all(x in asg for x in flat.gates)
        assert partition_cost(flat, asg) == plan.total
        ok, why = is_legal(flat, asg, None, None, G)
        assert ok, why


def test_forward_nonfixed_moe_matches_exact_when_one_ru_fits(moe_inference):
    """Q = 4 Q_inf at ``G = 4 fwd``: the whole rollout fits one RU (per-sequence ``Up`` = 1 fwd <= F), so
    ``U`` must equal the exact optimum tokens + weights once."""
    _, _, fwd = moe_inference
    F, G = int(1.5 * fwd), int(4 * fwd)
    Q = 4 * MOE_Q_INF
    bp, g = _build("forward-nonfixed-moe", "tiny-moe", Workload(tokens=Q, seq=MOE_Q_INF))
    plan = upper_coarse(g, F, G, program=bp.program)
    check_coarse_plan(g, plan, F, G, program=bp.program)
    assert plan.n_units == 1
    assert plan.total == 4 * Q + _weights_bytes(g)
    assert plan.units[0].up_max <= fwd < plan.units[0].work_max <= G


def test_checker_rejects_irows_outside_range(moe_inference):
    """``irows`` members are validated (row range inside ``mult x rows``, kernel batched with a row model)."""
    bp, g, fwd = moe_inference
    from accumulation.bounds.coarse import _Ctx
    ctx = _Ctx(g, bp.program)
    expert = next(o.id for o in g.ops if ctx.info[o.id].mult > 1 and ctx.info[o.id].rm.kind == "rows")
    info = ctx.info[expert]
    others = [(o.id, "whole", 0, 0) for o in g.ops if o.id != expert]
    bad = CoarsePlan([CoarseUnit(tuple(others + [(expert, "irows", 0, info.mult * info.rm.rows + 1)]), "bad")], F=fwd, G=None)
    with pytest.raises(AssertionError):
        check_coarse_plan(g, bad, 4 * fwd, None, program=bp.program)
    dense = next(o.id for o in g.ops if ctx.info[o.id].mult == 1 and ctx.info[o.id].rm.kind == "rows")
    others = [(o.id, "whole", 0, 0) for o in g.ops if o.id != dense]
    bad = CoarsePlan([CoarseUnit(tuple(others + [(dense, "irows", 0, 1)]), "bad")], F=fwd, G=None)
    with pytest.raises(AssertionError):
        check_coarse_plan(g, bad, 4 * fwd, None, program=bp.program)


# ---------------------------------------------------------------------------------------------------------
# other registry circuits go through the checker
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,model", [
    ("pretrain-dense", "tiny2"), ("es", "tiny2"), ("forward-nonfixed", "tiny2"),
    ("rl-policy-gradient", "tiny2"), ("pretrain-moe", "tiny-moe"), ("inference-moe", "tiny-moe"),
    ("forward-nonfixed-moe", "tiny-moe"), ("lora", "tiny2"), ("inference-decode", "tiny2"),
])
def test_registry_circuits_checked(name, model):
    wl = Workload(tokens=Q_INF, seq=SEQ, population=2, rank=2, gen_tokens=2, local_steps=1)
    bp, g = _build(name, model, wl)
    W = g.total_work()
    for frac in (2.0, 0.5, 0.1):
        F = G = int(frac * W)
        plan = upper_coarse(g, F, G, program=bp.program)
        check_coarse_plan(g, plan, F, G, program=bp.program)
        assert plan.total > 0 and plan.n_units >= 1
        assert plan.detail["acyclic"] is True
        assert max(u.work_max for u in plan.units) <= G
        assert max(u.up_max for u in plan.units) <= F


@pytest.mark.parametrize("name,model,Q,seq,fh,gh", [
    ("forward-nonfixed", "tiny2", 64, 4, 1.5, 4.0), ("forward-nonfixed", "tiny2", 64, 4, 1.0, 1.0),
    ("forward-nonfixed-moe", "tiny-moe", 32, 8, 1.5, 4.0), ("local-sgd", "tiny2", 32, 4, 1.5, 1.0),
])
def test_up_independent_gate_level(name, model, Q, seq, fh, gh):
    """The checker's member-level ``Up`` dominates the gate-level ``max_g work(Up_R(g))`` computed by BFS over
    the flattened circuit's real operand wires (exact module), and its ``Work`` is >= the gate sum
    (``results/up_independence.md``)."""
    inf = "inference-moe" if model == "tiny-moe" else "inference-dense"
    qi = MOE_Q_INF if model == "tiny-moe" else Q_INF
    _, gi = _build(inf, model, Workload(tokens=qi, seq=qi))
    fwd = gi.total_work()
    bp, g = _build(name, model, Workload(tokens=Q, seq=seq, local_steps=2))
    F, G = int(fh * fwd), int(gh * fwd)
    plan = upper_coarse(g, F, G, program=bp.program)
    for ui in range(plan.n_units) if plan.n_units <= 4 else [max(range(plan.n_units), key=lambda i: plan.units[i].up_max)]:
        r = gate_level_up(g, plan, bp.program, unit=ui)
        assert r["sound"], r
        assert r["up_checker"] >= r["up_gate_level"] > 0
        assert r["work_checker"] >= r["work_gate_level"]
        # never absurdly loose on per-sequence units: within the whole unit's work
        assert r["up_checker"] <= r["work_checker"]
