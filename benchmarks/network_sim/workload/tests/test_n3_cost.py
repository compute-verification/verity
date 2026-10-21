"""`n3_cost`: a window's ANDs counted from its parts are its traced size, and the fit and tree arithmetic it prices
served windows with."""

import pytest

from workload import n3_cost as C
from workload import n3_unit as U
from workload import n3_v5 as V5


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


@pytest.mark.slow
def test_a_v4_window_composed_from_its_parts_is_its_traced_size():
    """`composed_v4` against `n3_v4.size` at the demo's format (N = 1, 2 and 4 in R = 2 runs, at 4 the demo's v4 window;
    N = 3 in one run; one segment a frame at N = 1 and 2, a run of one read and of two, with no compaction stage and
    one) and salted at B = 128 (N = 2 and 3). It is the one test of the run parts' counts that the served rows' v4
    costs are composed from. About 60 s alone, most of it tracing SHA-512's compression once."""
    f, rec = U.demo().layout.fmt, U.Shape(4, 2, 2)
    memo: dict = {}
    rows = (C.checked(f, rec, [1, 2, 4], 2, False, memo, R=2) + C.checked(f, rec, [3], 2, False, memo, R=1)
            + C.checked(f, rec, [1, 2], 1, False, memo, R=1)
            + C.checked(f._replace(B=128), rec, [2, 3], 2, True, memo, R=2))
    assert [r["ands"] for r in rows] == [3_915_344, 5_646_217, 8_382_770, 6_469_426, 2_848_566, 4_354_265, 4_376_003,
                                         6_565_492]


def test_the_served_templates_fit_the_served_format():
    assert [V5.template(t, C.SERVED["B"])[:3] for t in C.SERVED_TEMPLATES.values()] == [(U.B64, 256, 4032),
                                                                                          (U.RAW, 256, 4032)]


@pytest.mark.slow
def test_a_v5_window_composed_from_its_parts_is_its_traced_size():
    """`composed_v5` against `n3_v5.size` at v5's demo format, on v4's test's shapes (at N = 4 in R = 2 runs the demo's
    v5 window), and on the canonical demo's two ports with their templates (N = 2 and 3), on rows of B and on rows of
    11 under frames of 16 (and a salted window on rows of 96 under frames of 128). It is the one test of the parts the
    levers' v5 costs are composed from. About 60 s alone, most of it tracing SHA-512's compression once."""
    f, rec = V5.prepare(U.demo().setup).layout.fmt, U.Shape(4, 2, 2)
    memo: dict = {}
    rows = (C.checked(f, rec, [1, 2, 4], 2, False, memo, R=2, v5=()) + C.checked(f, rec, [3], 2, False, memo, R=1, v5=())
            + C.checked(f, rec, [1, 2], 1, False, memo, R=1, v5=())
            + C.checked(f._replace(B=128), rec, [2, 3], 2, True, memo, R=2, v5=()))
    c = V5.canonical_demo()
    fc, narrow = V5.prepare(c.setup).layout.fmt, V5.prepare(c.setup, 11).layout.fmt
    for port in ("p0", "p1"):
        rows += C.checked(fc, rec, [2, 3], 2, False, memo, R=2, v5=c.templates[port])
    for port in ("p0", "p1"):
        rows += C.checked(narrow, rec, [2, 3], 2, False, memo, R=2, v5=c.templates[port], FB=fc.B)
    rows += C.checked(f._replace(B=96, P=1, Q=1), rec, [3], 2, True, memo, R=2, v5=c.templates["p0"], FB=128)
    assert [r["ands"] for r in rows] == [4_682_146, 6_414_577, 9_152_142, 6_853_072, 3_400_517, 4_737_407, 5_153_557,
                                         7_239_462, 6_074_576, 7_417_091, 6_074_962, 7_417_670, 7_080_356, 8_760_324,
                                         7_080_744, 8_760_906, 6_837_418]
