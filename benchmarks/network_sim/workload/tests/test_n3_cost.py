"""`n3_cost`: a window's ANDs counted from its parts are its traced size, and the fit and tree arithmetic it prices
served windows with."""

import pytest

from workload import n3_cost as C
from workload import n3_unit as U


def test_the_frames_tree_and_leaf_blocks():
    assert [C.nodes(N, 3) for N in (1, 2, 5, 8)] == [3, 3, 6, 7]
    assert C.leaf_blocks(16) == 1 and C.leaf_blocks(4096) == 33


def test_the_fit_is_exact_on_a_line():
    rows = [{"N": n, "ands": 7 + 5 * n} for n in (1, 2, 10)]
    fit = C.fit(rows)
    assert (fit["fixed"], fit["per_frame"], fit["points"]) == (7, 5, 3) and fit["max_relative_residual"] < 1e-12


@pytest.mark.slow
def test_a_window_composed_from_its_parts_is_its_traced_size():
    """`composed` against `size` at the demo's format (N = 1, 2 and 4; at 4 the demo window's 11,326,212) and salted at
    B = 128 (N = 2 and 3). It is the one test of the part counts the served rows' costs are composed from, which no
    cheaper test traces. About 60 s alone, most of it tracing SHA-512's compression once; after the demo window's slow
    test in the same process, about a second."""
    f, rec = U.demo().layout.fmt, U.Shape(4, 2, 2)
    rows = C.checked(f, rec, [1, 2, 4], 2, False, {}) + C.checked(f._replace(B=128), rec, [2, 3], 2, True, {})
    assert [r["ands"] for r in rows] == [3_557_733, 6_108_595, 11_326_212, 4_018_392, 7_027_870]
