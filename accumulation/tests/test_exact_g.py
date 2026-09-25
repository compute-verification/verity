"""The per-RU total-work cap ``G`` in the exact solver (SPEC §1, §4): ``I*(P; F, G, X)``.

* ``G=None`` (and any ``G >= total work``) reproduces the ``(F, X)`` optimum.
* ``I*(G)`` is non-increasing in ``G`` (fewer legal partitions when ``G`` shrinks).
* The three solvers agree with each other and with :func:`is_legal` / :func:`partition_cost`.
* One ``2x2x2`` matmul is checked by hand: the forced split is what the SPEC §1 definition says it must be.
"""

from __future__ import annotations

import pytest

from accumulation.exact.solve import Infeasible, exact_min_input, flatten, is_legal, partition_cost, ru_work
from accumulation.redteam.circuits import Circuit, Op, Param, build

BIG = 1 << 40


def _mm222():
    """``y = x . W^T`` with ``x`` a 2x2 token block and ``W`` a 2x2 accumulated weight (16 gates, work 12).

    Per output element: ``Zero32`` (work 0, absorbed by presolve) -> ``Mac`` (1) -> ``Mac`` (1) -> ``Round16``
    (1); each chain reads one row of ``x`` (4 B) and one row of ``W`` (4 B); ``x`` and ``W`` are 8 B each."""
    c = Circuit(params=[Param("x", (2, 2), "token"), Param("W", (2, 2), "accumulated")],
                ops=[Op("matmul", (("p", "x"), ("p", "W")))], name="mm222")
    bp = build(c)
    return bp, flatten(bp)


# Hand derivation (see the docstring of ``_mm222``; every RU must satisfy ``work(R) <= G``):
#   G >= 12 : everything in one RU ->  I* = |x| + |W| = 16.
#   G = 11  : one gate must leave; cheapest is a ``Round16`` alone (imports its 32-bit accumulator, 4 B) -> 20.
#   G = 6   : two 3-work chains per RU; group by x-row: each RU reads x[i,:] (4) + all of W (8) = 12 -> 24
#             (grouping by W-row gives the same 24).
#   G = 3   : one chain per RU: x row (4) + W row (4) = 8, four RUs -> 32.
#   G = 1   : every gate alone: Mac1 reads x_ik, W_jk (4); Mac2 reads acc (4) + x, W (4) = 8; Round reads acc
#             (4): 16 per output, four outputs -> 64.
#   G = 0   : a ``Mac`` gate alone has work 1 > 0 -> infeasible.
HAND_MM222 = {None: 16, 12: 16, 11: 20, 6: 24, 3: 32, 1: 64}


@pytest.mark.parametrize("method", ["brute", "milp", "milp-assign"])
def test_mm222_forced_split_by_hand(method):
    bp, flat = _mm222()
    assert flat.total_work() == 12
    for G, hand in HAND_MM222.items():
        res = exact_min_input(flat, BIG, BIG, G, method=method, limit=100)
        assert res.optimal
        assert res.total == hand, (G, res.total, hand)
        assert partition_cost(flat, res.assignment) == hand
        assert is_legal(flat, res.assignment, BIG, BIG, G)[0]
        if G is not None:
            assert max(ru_work(flat, res.assignment).values()) <= G
    with pytest.raises(Infeasible):
        exact_min_input(flat, BIG, BIG, 0, method=method, limit=100)


def test_G_none_equals_unbounded_and_calling_forms():
    bp, flat = _mm222()
    base = exact_min_input(flat, BIG, BIG)
    assert exact_min_input(flat, BIG, BIG, G=None).total == base.total
    assert exact_min_input(flat, BIG, BIG, flat.total_work()).total == base.total
    assert exact_min_input(flat, BIG, BIG, 10 * flat.total_work()).total == base.total
    # SPEC §5 form and positional/keyword G
    assert exact_min_input(bp.program, bp.roles, BIG, BIG, 6).total == 24
    assert exact_min_input(bp.program, bp.roles, BIG, BIG, G=6).total == 24
    assert exact_min_input(bp, BIG, BIG, G=6).total == 24
    with pytest.raises(TypeError):
        exact_min_input(bp, BIG, BIG, 6, G=3)


def _circuits():
    yield "mm222", Circuit(params=[Param("x", (2, 2), "token"), Param("W", (2, 2), "accumulated")],
                           ops=[Op("matmul", (("p", "x"), ("p", "W")))])
    yield "chain", Circuit(params=[Param("x", (1, 2), "token"), Param("W1", (2, 2), "accumulated"),
                                   Param("W2", (2, 2), "accumulated")],
                           ops=[Op("matmul", (("p", "x"), ("p", "W1"))), Op("matmul", (("o", 0), ("p", "W2")))])
    yield "add_shared", Circuit(params=[Param("x", (2, 2), "token"), Param("W", (2, 2), "fixed")],
                                ops=[Op("matmul", (("p", "x"), ("p", "W"))), Op("add", (("p", "x"), ("o", 0)))])
    yield "wgrad", Circuit(params=[Param("dy", (2, 2), "token"), Param("x", (2, 2), "token")],
                           ops=[Op("matmul_tt", (("p", "dy"), ("p", "x")))])


@pytest.mark.parametrize("name,circ", list(_circuits()), ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("F,X", [(BIG, BIG), (BIG, 12), (4, BIG), (3, 10)])
def test_monotone_in_G_and_solvers_agree(name, circ, F, X):
    bp = build(circ)
    flat = flatten(bp)
    W = flat.total_work()
    prev = None
    grid = [None, W, W - 1, max(1, W // 2), max(1, W // 3), 2, 1]
    for G in grid:
        vals = {}
        for m in ("brute", "milp"):
            try:
                r = exact_min_input(flat, F, X, G, method=m, limit=100)
            except Infeasible:
                vals[m] = None
                continue
            assert r.optimal
            assert is_legal(flat, r.assignment, F, X, G)[0]
            assert partition_cost(flat, r.assignment) == r.total
            vals[m] = r.total
        assert vals["brute"] == vals["milp"], (name, F, X, G, vals)
        v = vals["brute"]
        if prev is not None and v is not None:
            assert v >= prev, (name, F, X, G, v, prev)          # I* non-increasing in G
        if v is None:
            assert prev is None or G is not None                # once infeasible, smaller G stays infeasible
        prev = v if v is not None else prev
    # G >= total work never changes the answer
    assert exact_min_input(flat, F, X, W, limit=100).total == exact_min_input(flat, F, X, limit=100).total


def test_is_legal_reports_G():
    bp, flat = _mm222()
    one = {g: 0 for g in flat.gates}
    assert is_legal(flat, one, BIG, BIG)[0]
    assert is_legal(flat, one, BIG, BIG, 12)[0]
    ok, why = is_legal(flat, one, BIG, BIG, 11)
    assert not ok and "G=11" in why
