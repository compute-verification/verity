"""Acyclic RU quotient (SPEC §0, 2026-09-15) in the exact solver: hand case, solver agreement, monotonicity
``I*_acyclic >= I*_cyclic``, and the safety-net check on returned assignments."""
from __future__ import annotations

import random

import pytest

from accumulation.exact import micro
from accumulation.exact.solve import (Infeasible, exact_min_input, flatten, is_legal, make_flat, partition_cost,
                                      quotient_cycle)


def _diamond():
    """x(fixed) -> a -> b -> c, a -> c.  work a=1, b=3, c=1; every output 2 B; G=3.
    cyclic optimum: {a, c} + {b} = 4 B (RU{a,c} imports b.out, RU{b} imports a.out) -- its quotient is a 2-cycle;
    acyclic optimum: {a} + {b} + {c} = 6 B ({a,b} and {b,c} have work 4 > G)."""
    #         0:x      1:a         2:b        3:c
    work = [0, 1, 3, 1]
    width = [16, 16, 16, 16]
    ops = [[], [0], [1], [1, 2]]
    roles = ["fixed", None, None, None]
    return make_flat(work, width, ops, roles, outputs=[3])


def test_hand_diamond_cyclic_vs_acyclic():
    flat = _diamond()
    cyc = exact_min_input(flat, None, None, 3, acyclic=False, method="brute")
    acy = exact_min_input(flat, None, None, 3, acyclic=True, method="brute")
    assert cyc.total == 4 and acy.total == 6
    assert quotient_cycle(flat, cyc.assignment) is not None
    assert quotient_cycle(flat, acy.assignment) is None
    assert acy.n_units == 3
    # is_legal flags the cyclic witness under the default model and accepts it under acyclic=False
    ok, why = is_legal(flat, cyc.assignment, None, None, 3)
    assert not ok and "cyclic" in why
    assert is_legal(flat, cyc.assignment, None, None, 3, acyclic=False)[0]
    # every method agrees
    for m in ("milp", "milp-assign"):
        r = exact_min_input(flat, None, None, 3, acyclic=True, method=m)
        assert r.total == 6 and r.optimal and quotient_cycle(flat, r.assignment) is None


def test_quotient_cycle_witness():
    flat = _diamond()
    w = quotient_cycle(flat, {1: 0, 2: 1, 3: 0})
    assert w == (1, 2, 3)
    assert quotient_cycle(flat, {1: 0, 2: 0, 3: 1}) is None
    assert quotient_cycle(flat, {1: 0, 2: 1, 3: 1}) is None


@pytest.mark.parametrize("bp,F,G", [
    (micro.two_matmul_chain(2, 2, 2, 2), 4, None), (micro.two_matmul_chain(2, 2, 2, 2), None, 6),
    (micro.matmul_add(2, 2, 2), 3, 4), (micro.fixed_then_accum(2, 2, 2), 6, 8),
    (micro.local_sgd_chain(K_steps=2, q=1, d=1), 4, 6),
    (micro.fanout_shared_weight(n=3, q=1, d=1), None, 4), (micro.deep_chain(depth=3, q=1, d=2), 6, None),
])
def test_micro_monotone_and_methods_agree(bp, F, G):
    flat = flatten(bp)
    try:
        cyc = exact_min_input(flat, F, None, G, acyclic=False, limit=80)
        acy = exact_min_input(flat, F, None, G, acyclic=True, limit=80)
    except Infeasible:
        pytest.skip("cell infeasible")
    assert acy.total >= cyc.total
    assert quotient_cycle(flat, acy.assignment) is None
    assert is_legal(flat, acy.assignment, F, None, G)[0]
    assert partition_cost(flat, acy.assignment) == acy.total
    if flat.n_gates <= 24:
        m = exact_min_input(flat, F, None, G, acyclic=True, method="milp", limit=80)
        assert m.total == acy.total
    # lower_hint = cyclic optimum never changes the answer
    h = exact_min_input(flat, F, None, G, acyclic=True, limit=80, lower_hint=cyc.total)
    assert h.total == acy.total


def test_random_small_dags_brute_equals_milp():
    """Random small gate DAGs: brute force and the pairwise MILP agree under the acyclic model, and the acyclic
    optimum is never below the cyclic one."""
    rng = random.Random(7)
    for _ in range(25):
        n_in, n_g = rng.randint(1, 2), rng.randint(3, 7)
        n = n_in + n_g
        work, width, ops, roles = [], [], [], []
        for i in range(n):
            if i < n_in:
                work.append(0); width.append(16); ops.append([]); roles.append(rng.choice(["fixed", "token", "accumulated"]))
            else:
                work.append(rng.randint(1, 3)); width.append(rng.choice([16, 32]))
                k = rng.randint(1, min(3, i))
                ops.append(rng.sample(range(i), k)); roles.append(None)
        flat = make_flat(work, width, ops, roles)
        W = flat.total_work()
        F = rng.choice([None, max(1, W // 2), max(1, W // 3)])
        G = rng.choice([None, max(1, W // 2), max(1, W // 3)])
        try:
            acy = exact_min_input(flat, F, None, G, acyclic=True, method="brute")
        except Infeasible:
            try:
                exact_min_input(flat, F, None, G, acyclic=False, method="brute")
            except Infeasible:
                continue
            # cyclic feasible but acyclic infeasible is possible (e.g. every legal partition has a 2-cycle)
            continue
        cyc = exact_min_input(flat, F, None, G, acyclic=False, method="brute")
        assert acy.total >= cyc.total
        assert quotient_cycle(flat, acy.assignment) is None
        m = exact_min_input(flat, F, None, G, acyclic=True, method="milp")
        assert m.total == acy.total, (acy.total, m.total)
