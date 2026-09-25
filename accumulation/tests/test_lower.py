"""Tests for :mod:`accumulation.bounds.lower` (run: ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_lower.py -q``)."""

from __future__ import annotations

import itertools
import math
import time

import pytest

from accumulation.algorithms import registry as R
from accumulation.bounds.adw import adw
from accumulation.bounds.lower import (
    GAMMA_FULL,
    INFEASIBLE,
    THETA_GAMMA,
    LowerBound,
    lower_bound,
    matmul_charge,
    min_cost_per_mac,
)
from accumulation.bounds.upper import check_plan, upper_bound
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.graph.opgraph import Edge, Op, OpGraph

GAMMA = THETA_GAMMA * GAMMA_FULL          # partial-sum charge used by the bound (2.0)
TINY_WL = Workload(tokens=8, seq=4, population=2, rank=2, local_steps=2, gen_tokens=2)
BIG_F = 1 << 40


# ---------------------------------------------------------------------------------------------------------
# micro-graph helpers (no program: pass assume_in_order=True so row views come from leaf counts)
# ---------------------------------------------------------------------------------------------------------

def _param(g: OpGraph, name: str, leaves: int, role: str, width: int = 16) -> int:
    return g.add_tensor(kind="param", name=name, leaves=leaves, width=width, role=role).id


def _op(g: OpGraph, kind: str, statics: dict, inputs: list[tuple[int, int]], out_leaves: int, copies: int = 1,
        work: int = 1, width: int = 16) -> Op:
    oid = len(g.ops)
    t = g.add_tensor(kind="op", name=f"{kind}#{oid}", leaves=out_leaves, width=width, producer=oid)
    op = Op(id=oid, key=((), oid), fn_id=kind, kind=kind, statics=statics, work_per_copy=work, copies=copies,
            out=t.id, inputs=[Edge(src=s, leaves_per_copy=c) for s, c in inputs])
    g.ops.append(op)
    return op


def _matmul(g: OpGraph, A: int, B: int, M: int, N: int, K: int, copies: int = 1) -> Op:
    return _op(g, "matmul", {"M": M, "N": N, "K": K, "CH": 1}, [(A, M * K), (B, N * K)], M * N * copies,
               copies=copies, work=M * N * K)


def _lb(g: OpGraph, X: int, **kw) -> LowerBound:
    return lower_bound(g, BIG_F, X, assume_in_order=True, **kw)


def _brute_int(alpha: float, beta: float, Q: int, N: int, K: int, X: float, gamma: float = GAMMA) -> float:
    best = math.inf
    for a, b, k in itertools.product(range(1, Q + 1), range(1, N + 1), range(1, K + 1)):
        if k * (alpha * a + beta * b) <= X:
            best = min(best, alpha / b + beta / a + gamma * (1 / k - 1 / K))
    return best


def _brute_real(alpha: float, beta: float, Q: int, N: int, K: int, X: float, n: int = 60, gamma: float = GAMMA) -> float:
    def geo(lo: float, hi: float) -> list[float]:
        if hi <= lo:
            return [lo]
        r = (hi / lo) ** (1 / (n - 1))
        return [lo * r ** i for i in range(n)]
    best = math.inf
    for k in geo(1, K):
        for a in geo(1, Q):
            for b in geo(1, N):
                if k * (alpha * a + beta * b) <= X:
                    best = min(best, alpha / b + beta / a + gamma * (1 / k - 1 / K))
    return best


# ---------------------------------------------------------------------------------------------------------
# per-op optimiser
# ---------------------------------------------------------------------------------------------------------

def test_weight_only_charge_is_one_weight_load():
    # alpha = 0: a -> Q, one load of the N x K weights at beta bytes each, when a K-deep row fits X
    L, d = matmul_charge(0.0, 2.0, 8, 8, 8, X=1 << 20)
    assert L == 2 * 8 * 8 and d["a"] == 8 and d["k"] == 8
    # X too small for a full contraction: k = X / beta = 4, partial sums cost gamma (1/4 - 1/8) per MAC
    L, d = matmul_charge(0.0, 2.0, 8, 8, 8, X=8)
    assert L == 8 * 8 * 8 * (2 / 8 + GAMMA * (1 / 4 - 1 / 8))
    assert d["k"] == 4


def test_balanced_interior_optimum_is_analytic():
    # alpha = beta = 1, gamma = 4, Q = N = K = 64, X = 64: k* = sqrt(gamma X / (4 alpha beta)) = 8, a = b = 4
    v, d = min_cost_per_mac(1.0, 1.0, 4.0, 64, 64, 64, 64.0)
    assert abs(v - 0.9375) < 1e-12
    assert abs(d["k"] - 8) < 1e-9 and abs(d["a"] - 4) < 1e-9 and abs(d["b"] - 4) < 1e-9
    assert matmul_charge(1.0, 1.0, 64, 64, 64, 64.0, gamma=4.0)[0] == 245760


def test_infeasible_when_one_mac_exceeds_X():
    assert matmul_charge(3.0, 3.0, 8, 8, 8, X=5)[0] == INFEASIBLE
    assert matmul_charge(0.0, 3.0, 8, 8, 8, X=2)[0] == INFEASIBLE
    assert matmul_charge(0.0, 3.0, 8, 8, 8, X=3)[0] == math.floor(8 * 8 * 8 * (3 / 8 + GAMMA * (1 - 1 / 8)))


@pytest.mark.parametrize("seed", range(4))
def test_optimiser_never_exceeds_brute_force(seed):
    import random
    rng = random.Random(seed)
    for _ in range(40):
        Q, N, K = (rng.choice([1, 2, 3, 5, 8, 12]) for _ in range(3))
        alpha = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0])
        beta = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0])
        X = rng.choice([2, 4, 8, 16, 64, 256, 4096])
        v, _ = min_cost_per_mac(alpha, beta, GAMMA, Q, N, K, float(X))
        bi = _brute_int(alpha, beta, Q, N, K, X)
        br = _brute_real(alpha, beta, Q, N, K, X, n=40)
        if math.isinf(v):
            assert math.isinf(bi) and math.isinf(br)
            continue
        # the real relaxation is the sound quantity: never above any feasible integer or real tile ...
        assert v <= bi + 1e-9
        assert v <= br + 1e-9
        # ... and the grid (resolution ~ (K)^(1/39)) is never far above the exact optimum
        if br > 1e-9:
            assert v >= br * 0.9


def _brute_real_G(alpha: float, beta: float, Q: int, N: int, K: int, X: float, G: float, n: int = 40,
                  gamma: float = GAMMA) -> float:
    def geo(lo: float, hi: float) -> list[float]:
        if hi <= lo:
            return [lo]
        r = (hi / lo) ** (1 / (n - 1))
        return [lo * r ** i for i in range(n)]
    best = math.inf
    for k in geo(1, K):
        for a in geo(1, Q):
            for b in geo(1, N):
                if k * (alpha * a + beta * b) <= X and a * b * k <= G:
                    best = min(best, alpha / b + beta / a + gamma * (1 / k - 1 / K))
    return best


def test_work_cap_interior_is_am_gm():
    # boxes and X slack: the G-limited exposure per MAC is 3 (alpha beta gamma / G)^(1/3) (THEORY §0.1);
    # the certified value is never above it and within the branch-and-bound tolerance
    for alpha, beta, gamma, G in [(1.0, 1.0, 2.0, 1e6), (0.5, 2.0, 2.0, 1e5), (0.1, 2.0, 1.0, 1e7)]:
        v, d = min_cost_per_mac(alpha, beta, gamma, 10 ** 6, 10 ** 6, 10 ** 6, 1e12, G=G)
        amgm = 3 * (alpha * beta * gamma / G) ** (1 / 3)
        assert v <= amgm + 1e-12
        assert v >= amgm * (1 - 1e-3)
        assert d["binding"] == "G" and abs(d["work"] - G) <= 1e-6 * G


def test_work_cap_none_equals_unbinding_and_monotone():
    args = (0.3, 2.0, GAMMA, 4096, 4096, 4096, 5e6)
    v0, d0 = min_cost_per_mac(*args)
    assert d0["exact"] and d0["binding"] == "X"
    assert min_cost_per_mac(*args, G=4096 ** 3)[0] == v0          # G >= Q N K cannot bind: exact path
    assert min_cost_per_mac(*args, G=10 ** 30)[0] == v0
    prev = v0
    for G in (1e9, 1e6, 1e4, 1e2, 1.0):
        v, d = min_cost_per_mac(*args, G=G)
        assert v >= prev - 1e-12
        assert d["work"] <= G * (1 + 1e-9) and d["k"] * (0.3 * d["a"] + 2.0 * d["b"]) <= 5e6 * (1 + 1e-9)
        prev = v
    assert prev > v0
    assert math.isinf(min_cost_per_mac(*args, G=0.5)[0])


@pytest.mark.parametrize("seed", range(3))
def test_work_cap_optimiser_is_sound(seed):
    import random
    rng = random.Random(seed)
    for _ in range(30):
        Q, N, K = (rng.choice([1, 2, 3, 5, 8, 12, 40]) for _ in range(3))
        alpha = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0])
        beta = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0])
        X = rng.choice([2, 4, 8, 16, 64, 256, 4096])
        G = rng.choice([1, 2, 5, 17, 64, 300, 5000])
        v, d = min_cost_per_mac(alpha, beta, GAMMA, Q, N, K, float(X), G=float(G))
        v_none, _ = min_cost_per_mac(alpha, beta, GAMMA, Q, N, K, float(X))
        br = _brute_real_G(alpha, beta, Q, N, K, X, G, n=24)
        if math.isinf(v):
            assert math.isinf(br)
            continue
        assert v <= br + 1e-9                     # never above any feasible tile
        assert v >= v_none - 1e-12                # G only tightens
        if alpha == 0.0 and beta == 0.0:
            assert v == 0.0                       # convention: uncharged operands -> no charge at all
            continue
        if br > 1e-9:
            assert v >= br * 0.85
        if alpha > 0 or beta > 0:
            a, b, k = d["a"], d["b"], d["k"]     # the reported tile is legal
            assert a * b * k <= G * (1 + 1e-9) and k * (alpha * a + beta * b) <= X * (1 + 1e-9)


# ---------------------------------------------------------------------------------------------------------
# hand-built micro graphs
# ---------------------------------------------------------------------------------------------------------

def test_micro_matmul_token_A_accumulated_B():
    g = OpGraph()
    A = _param(g, "A", 64, "token")           # 8 x 8, free
    B = _param(g, "B", 64, "accumulated")     # 8 x 8, share 1 -> beta = 2
    o = _matmul(g, A, B, 8, 8, 8)
    lb = _lb(g, 1 << 20)
    assert lb.gen[A] and not lb.gen[B]
    assert lb.share[B] == 1
    assert lb.alpha_eff[A] == 0.0 and lb.alpha_eff[B] == 2.0
    assert lb.welt_bytes[o.out] == 8 * 2                # one K-segment of B; Wfree = whole B
    assert lb.wfree_bytes[o.out] == 64 * 2 and lb.gen[o.out]
    assert lb.per_op[o.id] == 2 * 8 * 8                 # one weight load
    assert lb.cap == 128
    assert lb.source == 128 + 128 and lb.state_floor == 128 and lb.token_seed == 128
    # single RU reading everything is legal at this X and costs exactly source: L must not exceed it
    assert lb.total == max(lb.source, lb.cap + lb.token_seed) == 256
    assert lb.kappa_op[o.id] == pytest.approx(512 / 128)
    # X below one K-deep weight row: partial sums are forced
    lb2 = _lb(g, 8)
    assert lb2.per_op[o.id] == 8 * 8 * 8 * (2 / 8 + GAMMA * (1 / 4 - 1 / 8)) and lb2.total == lb2.per_op[o.id] + 128


def test_micro_matmul_A_charged_matches_brute_force():
    g = OpGraph()
    A = _param(g, "A", 64, "carried")          # alpha = 2 (share 1)
    B = _param(g, "B", 64, "accumulated")      # beta = 2
    o = _matmul(g, A, B, 8, 8, 8)
    for X in (4, 8, 16, 32, 64, 128, 1 << 20):
        lb = _lb(g, X)
        exact = min_cost_per_mac(2.0, 2.0, GAMMA, 8, 8, 8, float(X))[0]
        assert lb.per_op[o.id] == math.floor(512 * exact)
        assert lb.per_op[o.id] <= 512 * _brute_int(2.0, 2.0, 8, 8, 8, X) + 1e-6
        assert lb.per_op[o.id] <= 512 * _brute_real(2.0, 2.0, 8, 8, 8, X) + 1e-6
    lb = _lb(g, 1 << 20)
    assert lb.per_op[o.id] == 256                      # alpha/N + beta/Q per MAC = A bytes + B bytes
    assert lb.total == 256 == lb.source


def _closure_chain(X: int, theta_grid=(1.0,)):
    """tok (4 x 64) -> t = tok . B^T (B: 8 x 64 accumulated, 1 KB) -> n = rmsnorm(t) -> u = n . C^T (C: 2 x 8).

    Hand computation: one B row is 128 B.  Welt(t) = one K-segment of B = 128 B; Wfree(t) = Welt(n) = all of
    B = 1024 B.  With X < 128 no RU can hold a full B row (n_B = 0), so every element of a row of t present in
    an RU is imported (finished: 2 B, or a 4-byte partial of which (1 - theta_gamma) 4 = 2 B is credit):
    kappa(t) = 2, P(t) = 8 * 2 = 16 B per row, kappa(n) = min(2, P(t) / share(t) / 8) = 2 = w,
    alpha_eff(n) = theta_n * 2."""
    g = OpGraph()
    tok = _param(g, "tok", 4 * 64, "token")
    B = _param(g, "B", 8 * 64, "accumulated")
    t = _matmul(g, tok, B, 4, 8, 64)
    n = _op(g, "rmsnorm", {"Q": 4, "K": 8}, [(t.out, 32)], 32)
    C = _param(g, "C", 2 * 8, "accumulated")
    u = _matmul(g, n.out, C, 4, 2, 8)
    return g, _lb(g, X, theta_grid=theta_grid), t, n, u, B, C


def test_micro_closure_exceeds_X_gives_full_alpha():
    X = 96
    _g, lb, t, n, u, _B, _C = _closure_chain(X)
    assert lb.welt_bytes[t.out] == 128 > X and not lb.gen[t.out]
    assert lb.wfree_bytes[t.out] == 1024 and lb.welt_bytes[n.out] == 1024 and not lb.gen[n.out]
    assert lb.kappa[t.out] == 2.0 and lb.pool[t.out] == 16.0        # 8 columns x 2 B forced imports per row
    assert lb.kappa[n.out] == 2.0 and lb.alpha_eff[n.out] == 2.0    # = w / share with theta = 1
    assert lb.alpha_eff[t.out] == 0.0                               # t has no matmul reader
    assert lb.op_params[u.id]["alpha"] == 2.0 and lb.op_params[u.id]["beta"] == 2.0
    assert lb.per_op[u.id] == matmul_charge(2.0, 2.0, 4, 2, 8, X)[0]
    assert lb.per_op[t.id] == matmul_charge(0.0, 2.0, 4, 8, 64, X)[0]
    assert lb.source == 4 * 64 * 2 + 1024 + 32 and lb.token_seed == 512
    assert lb.total == max(lb.source, lb.cap + 512)
    # with theta = 1/2 half of the credit is claimed directly
    lb2 = _closure_chain(X, theta_grid=(0.5,))[1]
    assert lb2.alpha_eff[n.out] == 1.0 and lb2.kappa[n.out] == 2.0


def test_micro_closure_within_X_gives_zero_alpha():
    X = 2048                                                        # all of B fits: n is generatable
    _g, lb, t, n, u, _B, _C = _closure_chain(X)
    assert lb.gen[t.out] and lb.gen[n.out]
    assert lb.welt_bytes[n.out] == 1024 <= X
    assert lb.kappa[n.out] == 0.0 and lb.alpha_eff[n.out] == 0.0
    assert lb.op_params[u.id]["alpha"] == 0.0
    assert lb.per_op[u.id] == 2 * 2 * 8                             # one load of C
    assert lb.per_op[t.id] == 1024                                  # one load of B
    assert lb.total == lb.source                                    # one RU holding everything is legal


def test_micro_gen_flips_exactly_at_closure_size():
    for X in (1023, 1024, 1025):
        _g, lb, _t, n, _u, _B, _C = _closure_chain(X)
        assert lb.gen[n.out] == (X >= 1024)
        assert (lb.alpha_eff[n.out] > 0) == (X < 1024)


def test_micro_credit_is_split_by_share():
    # the rmsnorm output is read by two matmuls: share = 2 -> each claims theta * kappa / 2
    g = OpGraph()
    tok = _param(g, "tok", 4 * 64, "token")
    B = _param(g, "B", 8 * 64, "accumulated")
    t = _matmul(g, tok, B, 4, 8, 64)
    n = _op(g, "rmsnorm", {"Q": 4, "K": 8}, [(t.out, 32)], 32)
    C1 = _param(g, "C1", 16, "accumulated")
    C2 = _param(g, "C2", 16, "accumulated")
    u1 = _matmul(g, n.out, C1, 4, 2, 8)
    u2 = _matmul(g, n.out, C2, 4, 2, 8)
    lb = _lb(g, 96, theta_grid=(1.0,))
    assert lb.share[n.out] == 2 and lb.kappa[n.out] == 2.0 and lb.alpha_eff[n.out] == 1.0
    assert lb.op_params[u1.id]["alpha"] == lb.op_params[u2.id]["alpha"] == 1.0


def test_micro_matmul_credit_budget_rule():
    # x = rmsnorm(t) with kappa(x) = 2 (chain above, theta_x = 0 so everything is inherited), then
    # y = x . D^T with D 16 x 8 accumulated: kappa(y) = min(w', inh(x) K^2 w_D / X) = 2 * 64 * 2 / X  (SPEC),
    # and P(y) = (N - n_B)^+ w' + min(n_B, N) kappa(y) with n_B = floor(X / (K w_D)) full D rows per RU.
    g = OpGraph()
    tok = _param(g, "tok", 4 * 64, "token")
    B = _param(g, "B", 8 * 64, "accumulated")
    t = _matmul(g, tok, B, 4, 8, 64)
    n = _op(g, "rmsnorm", {"Q": 4, "K": 8}, [(t.out, 32)], 32)
    D = _param(g, "D", 16 * 8, "accumulated")
    y = _matmul(g, n.out, D, 4, 16, 8)
    E = _param(g, "E", 16, "accumulated")
    z = _matmul(g, y.out, E, 4, 1, 16)
    X = 96
    lb = _lb(g, X, theta_grid=(0.0,))
    assert lb.kappa[n.out] == 2.0 and lb.theta[n.out] == 0.0
    nB = X // (8 * 2)                                               # 6 rows of D per RU
    assert lb.kappa[y.out] == pytest.approx(min(2.0, 2.0 * 8 * 8 * 2 / X))
    assert lb.pool[y.out] == pytest.approx((16 - nB) * 2.0 + nB * lb.kappa[y.out])
    assert not lb.gen[y.out]
    # theta = 0 claims nothing: alpha = 0 everywhere
    assert lb.op_params[z.id]["alpha"] == 0.0
    # theta = 1 hands nothing down: y inherits nothing (only the forced imports remain in its row pool)
    lb1 = _lb(g, X, theta_grid=(1.0,))
    assert lb1.kappa[y.out] == 0.0 and lb1.pool[y.out] == (16 - nB) * 2.0 and lb1.op_params[z.id]["alpha"] == 0.0
    # theta = 1/2: inh(n) = 1, kappa(y) = min(2, 1 * 64 * 2 / X), and z claims theta * kappa(y)
    lb5 = _lb(g, X, theta_grid=(0.5,))
    assert lb5.kappa[y.out] == pytest.approx(min(2.0, 1.0 * 8 * 8 * 2 / X))
    assert lb5.op_params[z.id]["alpha"] == pytest.approx(0.5 * lb5.kappa[y.out]) and lb5.kappa[y.out] > 0


def test_aligned_and_broadcast_inputs():
    g = OpGraph()
    x = _param(g, "x", 32, "token")                 # 8 x 4
    gain = _param(g, "g", 4, "accumulated")         # broadcast over rows
    w = _param(g, "w", 32, "carried")               # 8 x 4 aligned
    n = _op(g, "rmsnorm", {"Q": 8, "K": 4}, [(x, 32), (gain, 4)], 32)
    s = _op(g, "add", {"Q": 8, "K": 4}, [(x, 32), (w, 32)], 32)
    e = _op(g, "embed", {"Q": 8, "V": 8, "D": 4}, [(x, 8), (w, 32)], 32)   # gather: no credit
    B = _param(g, "B", 8, "accumulated")            # 2 x 4
    _matmul(g, n.out, B, 8, 2, 4)
    _matmul(g, s.out, B, 8, 2, 4)
    _matmul(g, e.out, B, 8, 2, 4)
    lb = _lb(g, 1, theta_grid=(1.0,))               # X = 1: nothing is generatable
    assert not lb.gen[n.out] and not lb.gen[s.out] and not lb.gen[e.out]
    assert lb.alpha_eff[n.out] == 0.0               # the gain cannot back per-element charges
    assert lb.share[w] == 2 and lb.alpha_eff[w] == 1.0
    assert lb.kappa[s.out] == 1.0                   # add inherits w's slot / share(w)
    assert lb.alpha_eff[s.out] == 1.0
    assert lb.alpha_eff[e.out] == 0.0               # gather with a free index
    assert lb.infeasible                            # X = 1 < beta: no MAC fits (sentinel)


def test_shared_A_across_copies_falls_back_to_zero_alpha():
    g = OpGraph()
    A = _param(g, "A", 16, "carried")               # 4 x 4 rows, read by every copy
    B = _param(g, "B", 32, "accumulated")           # 2 copies x (4 x 4)
    o = _op(g, "matmul", {"M": 4, "N": 4, "K": 4, "CH": 1}, [(A, 16), (B, 16)], 32, copies=2)
    lb = _lb(g, 1 << 20)
    assert any("shared between copies" in n for n in lb.notes)
    assert lb.tiles[o.id]["alpha"] == 0.0
    assert lb.per_op[o.id] == 4 * 4 * 2           # one load of the (merged) weights at beta = 2


def test_theta_choice_is_reported_and_each_is_sound():
    _g, lb, _t, _n, _u, _B, _C = _closure_chain(96, theta_grid=(0.25, 0.5, 0.75, 1.0))
    assert lb.theta_act in (0.25, 0.5, 0.75, 1.0)
    assert lb.theta_kind == {"rmsnorm": lb.theta_act}
    for th in (0.25, 0.5, 0.75, 1.0):
        assert _closure_chain(96, theta_grid=(th,))[1].total <= lb.total


# ---------------------------------------------------------------------------------------------------------
# registry graphs
# ---------------------------------------------------------------------------------------------------------

def _tiny(name: str, wl: Workload = TINY_WL):
    cfg = MODELS["tiny-moe" if "moe" in name else "tiny2"]
    assert R.ALGORITHMS[name].supports(cfg, wl) is None
    bp = R.build(name, cfg, wl)
    return extract(bp), bp


def _check_invariants(lb: LowerBound) -> None:
    assert lb.total >= lb.source
    if not lb.infeasible:
        assert sum(lb.per_op.values()) == lb.cap
        assert lb.total == max(lb.source, lb.cap + lb.token_seed)
    assert lb.source == lb.state_floor + lb.token_seed
    assert all(v >= 0 for v in lb.alpha_eff.values())
    assert all(v >= 0 for v in lb.per_op.values())
    for tid, k in lb.kappa.items():
        assert 0 <= k <= 4.0 + 1e-9
        assert not (lb.gen[tid] and k > 0), "gen tensors carry no credit"
        assert lb.alpha_eff[tid] <= k / lb.share[tid] + 1e-12 or lb.alpha_eff[tid] <= 4.0 / lb.share[tid] + 1e-12
    assert all(s >= 1 for s in lb.share.values())


@pytest.mark.parametrize("name", ["pretrain-dense", "lora"])
def test_monotone_in_X(name):
    g, bp = _tiny(name)
    Xs = [1, 4, 16, 64, 256, 4096, 1 << 20]
    Ls = [lower_bound(g, BIG_F, X, program=bp.program, theta_by_kind=False).total for X in Xs]
    for a, b in itertools.pairwise(Ls):
        assert a >= b
    assert Ls[0] > Ls[-1]


def test_monotone_in_tokens():
    prev = None
    for Q in (4, 8, 16, 32):
        g, bp = _tiny("pretrain-dense", Workload(tokens=Q, seq=4))
        for X in (64, 4096):
            L = lower_bound(g, BIG_F, X, program=bp.program, theta_by_kind=False).total
            if prev is not None and X in prev:
                assert L >= prev[X]
            prev = {**(prev or {}), X: L}


@pytest.mark.parametrize("name", sorted(R.ALGORITHMS))
@pytest.mark.parametrize("X", [64, 256, 4096, 1 << 20])
def test_all_algorithms_tiny(name, X):
    g, bp = _tiny(name)
    lb = lower_bound(g, BIG_F, X, adw=adw(g), program=bp.program)
    _check_invariants(lb)
    assert not lb.infeasible
    assert lb.per_op.keys() == {o.id for o in g.ops}
    assert lb.alpha_eff.keys() == lb.kappa.keys() == lb.gen.keys() == lb.share.keys() == {t.id for t in g.tensors}
    # non-matmul ops are never charged
    assert all(lb.per_op[o.id] == 0 for o in g.ops if o.kind != "matmul")
    # share is never below the graph's (it is the max with the inclusive-range reading of the sweep)
    assert all(lb.share[t.id] >= max(1, g.share(t.id)) for t in g.tensors if g.consumers(t.id))


@pytest.mark.parametrize("name", sorted(R.ALGORITHMS))
@pytest.mark.parametrize("X", [64, 256, 4096, 1 << 20])
def test_lower_below_upper_tiny(name, X):
    g, bp = _tiny(name)
    F = 1 << 40
    try:
        ub = upper_bound(g, F, X, program=bp.program)
    except ValueError as e:      # pre-existing: upper.py cannot plan the 3-edge MoE expert matmuls
        pytest.xfail(f"upper_bound cannot plan {name}: {e}")
    check_plan(ub.plan, g, F, X)
    lb = lower_bound(g, F, X, program=bp.program)
    assert not lb.infeasible
    assert lb.total <= ub.total, (lb.total, ub.total)
    # each theta grid point on its own is a certified bound too
    for th in (0.25, 0.5, 1.0):
        assert lower_bound(g, F, X, program=bp.program, theta_grid=(th,)).total <= ub.total


@pytest.mark.parametrize("name", ["pretrain-dense", "lora", "inference-dense"])
def test_work_cap_monotone_tiny(name):
    g, bp = _tiny(name)
    for X in (64, 4096):
        base = lower_bound(g, BIG_F, X, program=bp.program, theta_by_kind=False)
        huge = lower_bound(g, BIG_F, X, G=1 << 60, program=bp.program, theta_by_kind=False)
        assert huge.total == base.total and huge.cap == base.cap        # L(G=None) == L(G unbounded)
        prev = base.total
        for G in (1 << 20, 4096, 512, 64, 8, 1):
            lb = lower_bound(g, BIG_F, X, G=G, program=bp.program, theta_by_kind=False)
            _check_invariants(lb)
            assert lb.G == G
            assert lb.total >= prev, (X, G, lb.total, prev)          # non-decreasing as G decreases
            prev = lb.total
        if base.cap > 0:                          # inference on fixed weights charges nothing at any G
            assert prev > base.total


@pytest.mark.parametrize("name", ["pretrain-dense", "lora", "es", "inference-dense", "pretrain-moe"])
@pytest.mark.parametrize("G", [1 << 20, 512, 128])
def test_lower_below_upper_tiny_with_work_cap(name, G):
    g, bp = _tiny(name)
    F = 1 << 40
    for X in (256, 4096):
        try:
            ub = upper_bound(g, F, X, G=G, program=bp.program)
        except ValueError as e:      # upper.py: some row op has no legal tile under this G
            pytest.xfail(f"upper_bound cannot plan {name} at G={G}: {e}")
        check_plan(ub.plan, g, F, X, G)
        lb = lower_bound(g, F, X, G=G, program=bp.program)
        assert not lb.infeasible
        assert lb.total <= ub.total, (X, G, lb.total, ub.total)


def test_worst_ru_diagnostic_tiny():
    g, bp = _tiny("pretrain-dense")
    for G in (None, 256):
        lb = lower_bound(g, BIG_F, 256, G=G, program=bp.program)
        wr = lb.worst_ru
        assert set(wr) >= {"worst", "deficit_top", "uncharged", "best_price", "binding_counts", "n_target_ops"}
        w = wr["worst"]
        assert set(w) >= {"op_id", "class", "shape", "input_bytes", "work", "up_work", "target_macs", "bytes_per_mac",
                          "operands", "binding", "theta", "gen", "kappa", "price"}
        assert g.ops[w["op_id"]].kind == "matmul" and w["target"]
        a, b, k = w["shape"]
        prm = lb.op_params[w["op_id"]]
        assert 1 <= a <= prm["Q"] + 1e-9 and 1 <= b <= prm["N"] + 1e-9 and 1 <= k <= prm["K"] + 1e-9
        assert k * (prm["alpha"] * a + prm["beta"] * b) <= 256 * (1 + 1e-9)          # the RU is legal
        if G is not None:
            assert w["work"] <= G * (1 + 1e-9)
            assert all(t["binding"] in ("X", "G", "X+G", "box", "none") for t in lb.tiles.values())
        assert w["binding"] in ("X", "G", "X+G", "box")
        assert len(w["operands"]) == 2 and w["input_bytes"] >= w["claimed_bytes"] - 1e-9
        # the worst op has the lowest realised price among charged target ops
        charged = [o.id for o in g.ops if o.kind == "matmul" and lb.per_op[o.id] > 0
                   and lb.op_params.get(o.id) and w["price"] > 0]
        assert all(lb.per_op[oid] / g.ops[oid].macs() >= w["price"] - 1e-12 for oid in charged)
        assert 1 <= len(wr["deficit_top"]) <= 5
        assert all(gr["deficit"] >= -1e-9 for gr in wr["deficit_top"])
        assert wr["deficit_top"] == sorted(wr["deficit_top"], key=lambda gr: -gr["deficit"])


def test_kappa_L_uses_credited_or_all_macs():
    g, bp = _tiny("pretrain-dense")
    a = adw(g)
    lb = lower_bound(g, BIG_F, 4096, adw=a, program=bp.program)
    assert lb.target_macs == a.credited_macs > 0 and lb.kappa_L == pytest.approx(a.credited_macs / lb.total)
    g, bp = _tiny("inference-dense")
    a = adw(g)
    lb = lower_bound(g, BIG_F, 4096, adw=a, program=bp.program)
    assert a.credited_macs == 0 and lb.target_macs == a.all_macs > 0
    assert lb.kappa_L == pytest.approx(a.all_macs / lb.total)


def test_source_is_state_plus_token_seed_tiny():
    g, bp = _tiny("pretrain-dense")
    lb = lower_bound(g, BIG_F, 4096, program=bp.program)
    st = sum(t.leaves * t.width // 8 for t in g.tensors if t.kind == "param" and t.role in ("accumulated", "carried"))
    assert lb.state_floor == st
    assert lb.token_seed == 8 * 4 + 8 * 4 + 4


def test_gen_pattern_tiny_dense():
    # at X below the smallest weight matrix nothing but the layer-0 first projections is generatable;
    # at X above the whole parameter set everything is
    g, bp = _tiny("pretrain-dense")
    lb_small = lower_bound(g, BIG_F, 16, program=bp.program)
    lb_big = lower_bound(g, BIG_F, 1 << 20, program=bp.program)
    acts = [t for t in g.tensors if t.kind == "op"]
    # everything is generatable except the root-like updated weights (sgdupdate: one W element per output)
    assert all(lb_big.gen[t.id] or g.ops[t.producer].kind == "sgdupdate" for t in acts)
    assert any(g.ops[t.producer].kind == "sgdupdate" and not lb_big.gen[t.id] and lb_big.kappa[t.id] > 0
               for t in acts)
    assert lb_big.total == lb_big.source
    assert 0 < sum(lb_small.gen[t.id] for t in acts) < len(acts)
    assert lb_small.total > lb_small.source


def test_llama3_8b_pretrain_dense():
    cfg = MODELS["llama3-8b"]
    wl = Workload(tokens=4096, seq=4096)
    t0 = time.time()
    bp = R.build("pretrain-dense", cfg, wl)
    g = extract(bp)
    a = adw(g)
    lb = lower_bound(g, 10 ** 12, 5_000_000, adw=a, program=bp.program)
    dt = time.time() - t0
    print(f"\nllama3-8b pretrain-dense X=5MB: L={lb.total:,} B  L/token={lb.total / wl.tokens:,.0f} B  "
          f"source={lb.source:,} cap={lb.cap:,} kappa_L={lb.kappa_L:,.1f} MAC/B  ({dt:.1f}s)")
    assert dt < 60
    _check_invariants(lb)
    assert not lb.infeasible
    assert lb.cap > lb.source                                     # the closure-budget model bites
    assert lb.total / wl.tokens > 10_000_000                      # tens of MB per token
    assert lb.state_floor == sum(t.leaves * t.width // 8 for t in g.tensors
                                 if t.kind == "param" and t.role in ("accumulated", "carried"))
    assert lb.kappa_L > 0
    wq = next(t for t in g.tensors if t.kind == "param" and t.name == "Wq")
    assert g.share(wq.id) == 4 == lb.share[wq.id]
    # gen: only the first layer's pre-Wo activations; everything after the first Wo has |Welt| >> X
    gen_acts = [t for t in g.tensors if t.kind == "op" and lb.gen[t.id]]
    assert 0 < len(gen_acts) < 40
    assert all(lb.welt_bytes[t.id] <= 5_000_000 for t in gen_acts)
