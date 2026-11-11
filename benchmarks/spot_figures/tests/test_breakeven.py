"""The break-even planner (`breakeven/`) on Figure 1's own path: the proof server's line builds from the figure's
cards and planner, and its bound falls as β grows."""
from math import isfinite

import pytest


@pytest.mark.slow
def test_the_proof_servers_line_prices_on_figure_1s_path():
    """`P5+poolwin600`, the proof server's line at its last step, builds from the cards (fused per-expert reads, the
    server's per-type prices, the column blocks, the 128k workload) and its world-year bound is finite, positive and
    smaller at 10% of compute than at 1%: a change to figure1, total_cost or the cards that breaks the planner fails
    here rather than when a plan is next regenerated.  Importing the planner builds every Figure 1 line, so it is slow."""
    from breakeven import sweep

    case = sweep.cases()["P5+poolwin600"]
    line = sweep.build(case)
    at1, at10 = (sweep.bound(line, beta, "world year", case.scale) for beta in (0.01, 0.10))
    assert isfinite(at1) and isfinite(at10) and 0 < at10 < at1
