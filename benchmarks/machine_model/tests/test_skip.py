"""`skip`: a design's ε_skip as made, the most over its cells of the skipped work over ♭'s bound."""

from __future__ import annotations

import pytest

from verity_machine_model import skip as K
from verity_machine_model.eps import design
from verity_machine_model.model import device

CELLS = [(m, c, ph) for m in K.MODELS for c in K.CONTEXTS for ph in K.PHASES]


def test_a_designs_skip_is_the_most_over_its_cells():
    """`design_skip` is `floor_eps` at the model, context and phase where it is largest, and names that cell."""
    d = design("dither-c-peel-fx1")
    q = K.design_skip(d)
    by_cell = {cell: K.floor_eps(*cell, d)[0] for cell in CELLS}
    assert q.eps_time == max(by_cell.values()) == by_cell[q.model, q.ctx, q.phase]
    assert (q.design, q.credit) == (d.name, "floor") and 0 < q.eps_time < 0.01


def test_off_sm_120_a_cell_has_no_mac_share_or_overhead():
    """`parts`' honest-time model is sm_120's, so a device directory's cell carries ε_skip alone, on its own vector."""
    data = device("sm_90")
    d = design("dither-2s-licpw-redux-p1-l32", data)
    sheet, measured = (K.design_skip(d, data=data, vector=v) for v in ("DS-90", "DC-90-ri"))
    assert (sheet.eps_mac, sheet.gamma, sheet.rows, sheet.credit) == (None, None, [], "floor")
    assert sheet.eps_time != measured.eps_time and K.prices(d.name, data, "DS-90").vector == "DS-90"
    current = K.prices(d.name, data).vector
    assert K.design_skip(d, data=data).eps_time == K.design_skip(d, data=data, vector=current).eps_time


def test_floor_eps_is_the_skipped_work_over_the_bound_with_it():
    """Corollary D3's ε_skip is n^s_♭ / B_♭ with B_♭ the words plus n^s_♭, each component's piece over the same
    bound, and a served output that is checked downstream never leaves the cheater more to skip."""
    d = design("dither-c-peel-fx1")
    for m, c, ph in CELLS:
        words, free, checked = K.floor_skip(m, c, d, phase=ph)
        eps, split = K.floor_eps(m, c, ph, d)
        assert eps == sum(free.values()) / (words + sum(free.values())) == pytest.approx(sum(split.values()))
        assert K.floor_eps(m, c, ph, d, checked=True)[0] <= eps and all(checked[k] <= free[k] for k in free)


def test_a_leaf_with_no_price_is_refused():
    with pytest.raises(ValueError, match="no price for nope on linears"):
        K.products("nope", "linears", 128)
