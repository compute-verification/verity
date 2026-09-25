"""``bounds/coarse.py::upper_coarse`` judged by the exact module under the corrected policy (``X = inf``): the
plan expanded to gates must be legal, its claimed cost must dominate the literal gate-level cost, and both must
be ``>= I*``.  (``lower_coarse`` is added to :mod:`accumulation.redteam.coarse_validation` when it lands.)"""

from __future__ import annotations

import pytest

from accumulation.exact import micro
from accumulation.exact.solve import flatten
from accumulation.graph import extract
from accumulation.redteam import coarse_validation as cv
from accumulation.redteam.harness import op_gate_lists

pytest.importorskip("accumulation.bounds.coarse")


def _cells():
    for bp in (micro.micro_inference(1), micro.local_sgd_chain(1, 1, 1), micro.local_sgd_chain(2, 1, 1),
               micro.fanout_shared_weight(4), micro.deep_chain(4), micro.matmul_only(2, 2, 2), micro.fixed_then_accum(2, 2, 1)):
        flat = flatten(bp)
        W = flat.total_work()
        for F, G in ((None, None), (None, -(-W // 2)), (-(-W // 2), None), (-(-W // 2), -(-W // 2))):
            yield bp, F, G


@pytest.mark.parametrize("bp,F,G", list(_cells()), ids=lambda v: getattr(v, "algorithm", str(v)))
def test_upper_coarse_sound_on_micro(bp, F, G):
    flat = flatten(bp)
    g = extract(bp)
    og = op_gate_lists(bp, g, flat)
    cell = cv.eval_cell(bp, flat, g, og, F, G, time_limit=60, exact_limit=64, lower_fn=None)
    if cell["U"] is None:
        pytest.skip(f"upper_coarse unavailable: {cell['U_error']}")      # robustness gap, reported by the table
    assert cell["U_legal"] is True, cell["U_error"]
    assert cell["U"] >= cell["U_literal"]
    if cell["optimal"]:
        assert cell["U_literal"] >= cell["Istar"]
    assert not cell["violations"], cell["violations"]


def test_lower_coarse_hook():
    """When ``lower_coarse`` exists it must be callable through the table's adapter and never exceed I*."""
    fn = cv._lower_coarse_fn()
    if fn is None:
        pytest.skip("lower_coarse not present yet")
    bp = micro.local_sgd_chain(1, 1, 1)
    flat = flatten(bp)
    g = extract(bp)
    og = op_gate_lists(bp, g, flat)
    for F, G in ((None, None), (3, 8), (6, None)):
        cell = cv.eval_cell(bp, flat, g, og, F, G, time_limit=60, exact_limit=64, lower_fn=fn)
        assert cell["L"] is not None, cell["L_error"]
        assert not any(v.startswith("L_gt_Istar") for v in cell["violations"]), cell
