"""The corrected policy model in the exact solver (SPEC §1 / THEORY §0.1): ``X = inf`` (``X=None``), legality is
``work(R) <= G`` and ``work(Up_R(g)) <= F`` only; ``F=None`` = no serial cap.

* ``X=None`` equals a huge ``X``; ``F=None`` equals ``F >= total work``.
* The single-RU fast path (any circuit size) agrees with the brute/MILP solvers where both apply and settles the
  registry ``inference-dense`` circuits at ``F, G >= fwd``: ``I* =`` token bytes (4 B/token).
* The multi-step micro suite (``exact/micro.py``): gate counts, work == operator-graph work, hand values.
"""

from __future__ import annotations

import pytest

from accumulation.exact import micro
from accumulation.exact.solve import Infeasible, exact_min_input, flatten, is_legal, partition_cost
from accumulation.graph import extract
from accumulation.redteam.circuits import Circuit, Op, Param, build

BIG = 1 << 40


def _mm222():
    c = Circuit(params=[Param("x", (2, 2), "token"), Param("W", (2, 2), "accumulated")],
                ops=[Op("matmul", (("p", "x"), ("p", "W")))], name="mm222")
    return flatten(build(c))


@pytest.mark.parametrize("method", ["brute", "milp", "milp-assign"])
def test_X_none_equals_unbounded(method):
    flat = _mm222()
    for F, G in [(None, None), (None, 6), (None, 3), (2, None), (3, 6), (2, 3)]:
        a = exact_min_input(flat, F, None, G, method=method, limit=100)
        b = exact_min_input(flat, BIG if F is None else F, BIG, G, method=method, limit=100)
        assert a.optimal and b.optimal and a.total == b.total, (F, G, a.total, b.total)
        assert is_legal(flat, a.assignment, F, None, G)[0]
        assert partition_cost(flat, a.assignment) == a.total
    with pytest.raises(Infeasible):
        exact_min_input(flat, None, None, 0, method=method, limit=100)


def test_is_legal_none_caps():
    flat = _mm222()
    one = {g: 0 for g in flat.gates}
    assert is_legal(flat, one, None, None, None)[0]
    assert not is_legal(flat, one, None, None, 11)[0]
    assert not is_legal(flat, one, 2, None, None)[0]
    assert not is_legal(flat, one, None, 15, None)[0]


def test_one_ru_fast_path_matches_solvers():
    """Whenever the single RU is legal the fast path fires; its value equals the brute/MILP optimum."""
    for name, bp in micro.micro_programs(max_gates=32):
        flat = flatten(bp)
        W = flat.total_work()
        for F, G in [(None, None), (W, W), (W - 1, None), (None, W - 1)]:
            r = exact_min_input(flat, F, None, G, limit=100)
            ok = is_legal(flat, {g: 0 for g in flat.gates}, F, None, G)[0]
            assert (r.method == "one-ru") == ok, (name, F, G, r.method, ok)
            if ok:
                brute = exact_min_input(flat, F, None, G, method="brute", limit=100)
                assert brute.total == r.total == sum(flat.bytes_of(v) for v in flat.nonfixed_inputs)


def test_registry_inference_is_token_bytes_at_fwd():
    """THEORY §0.1 expected shape: tiny inference at ``F, G >= fwd`` costs exactly the token bytes (4 B/token)."""
    from accumulation.algorithms.registry import ALGORITHMS
    from accumulation.configs import MODELS, Workload
    for m in ("tiny", "tiny2"):
        for q in (2, 3):
            bp = ALGORITHMS["inference-dense"].build(MODELS[m], Workload(tokens=q, seq=q, chunk=1, population=1, rank=1, local_steps=1))
            flat = flatten(bp)
            fwd = flat.total_work()
            r = exact_min_input(flat, fwd, None, fwd, limit=40)
            assert r.optimal and r.method == "one-ru"
            assert r.total == 4 * q, (m, q, r.total)
            g = extract(bp)
            assert sum(o.work_per_copy * o.copies for o in g.ops) == fwd      # solver work == OpGraph work


@pytest.mark.parametrize("factory,kw,gates,work", [
    (micro.micro_inference, dict(q=1, d=2), 8, 6),
    (micro.micro_inference, dict(q=2, d=2), 16, 12),
    (micro.local_sgd_chain, dict(K_steps=1, q=1, d=2), 30, 24),
    (micro.local_sgd_chain, dict(K_steps=2, q=1, d=2), 60, 48),
    (micro.local_sgd_chain, dict(K_steps=1, q=2, d=2), 44, 36),
    (micro.fanout_shared_weight, dict(n=4, q=1, d=2), 40, 32),
    (micro.deep_chain, dict(depth=4, q=1, d=2), 32, 24),
])
def test_multistep_suite_sizes_and_work_measure(factory, kw, gates, work):
    bp = factory(**kw)
    flat = flatten(bp)
    assert flat.n_gates == gates
    assert flat.total_work() == work
    g = extract(bp)
    assert sum(o.work_per_copy * o.copies for o in g.ops) == work


def test_multistep_suite_hand_values():
    fwd = 6
    # inference: token bytes at any F, G >= fwd
    flat = flatten(micro.micro_inference(1))
    for F, G in [(fwd, fwd), (fwd, None), (None, fwd)]:
        assert exact_min_input(flat, F, None, G).total == 4
    # local-sgd K=1: one step fuses at F >= 8 (Up of a W' element), G >= 24 -> the 20 charged root bytes
    flat = flatten(micro.local_sgd_chain(1))
    assert exact_min_input(flat, 9, None, 24, limit=100).total == 20
    assert exact_min_input(flat, 8, None, None, limit=100).total == 20
    assert exact_min_input(flat, 7, None, None, limit=100).total > 20         # F below one step's Up forces a cut
    assert exact_min_input(flat, None, None, 23, limit=100).total > 20        # G below one step's work forces a cut
    # deep chain (fixed weights): fused, the last layer's Round has Up = 3 + 6*(depth-1) = 21 (its own chain plus
    # every earlier layer); F below that forces activation imports (4 B per cut)
    flat = flatten(micro.deep_chain(4))
    assert exact_min_input(flat, 21, None, None, limit=100).total == 4
    r = exact_min_input(flat, 12, None, None, limit=100, time_limit=120)     # two layers per RU: 1 cut
    assert r.total == 8 and r.n_units == 2
    r = exact_min_input(flat, 6, None, None, limit=100, time_limit=120)      # every layer boundary cut: 3 cuts
    assert r.total == 4 + 3 * 4                                              # (RU count is not unique: interleaving is free)
    # fan-out: with G >= 8 + 4*6 = 32 everything is one RU; G = 24 forces two branches out -> W' (8 B) re-imported
    flat = flatten(micro.fanout_shared_weight(4))
    assert exact_min_input(flat, None, None, 32, limit=100).total == 36
    assert exact_min_input(flat, None, None, 24, limit=100, time_limit=120).total == 36 + 8
    # d=1 accumulation chain (hand: SPEC §1 semantics) -- I* grows with K, marginal shrinks as F/G grow
    # cyclic-quotient model (pre-2026-09-15): the K >= 2 optima interleave step k's update with step k+1's forward
    vals = {K: exact_min_input(flatten(micro.local_sgd_chain(K, q=1, d=1)), 6, None, 8, limit=100, acyclic=False).total
            for K in (1, 2, 3)}
    assert vals == {1: 14, 2: 22, 3: 30}
    # acyclic RU quotient (SPEC §0): those interleavings are 2-cycles; I* rises to 26 / 40 (K=1 unchanged)
    vals = {K: exact_min_input(flatten(micro.local_sgd_chain(K, q=1, d=1)), 6, None, 8, limit=100).total for K in (1, 2, 3)}
    assert vals == {1: 14, 2: 26, 3: 40}
    vals_inf = {K: exact_min_input(flatten(micro.local_sgd_chain(K, q=1, d=1)), None, None, None).total for K in (1, 2, 3)}
    assert vals_inf == {1: 10, 2: 14, 3: 18}                                # one RU: charged roots (+4 B tokens/step)
