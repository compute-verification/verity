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


#: `reprice.LINEARS` without the served cut's MoE GEMMs
OLD_LINEARS = ("Fused[", "MoeExpertFp8BlockGemm", "FlashInferFp8Linear", "ScaledMmFp8", "BatchedGemm", "CublasSplitKGemm",
               "LlBf16DotprodGemm", "Gemm_v2")


@pytest.mark.slow
def test_the_served_cuts_128k_indexer_is_stated_against_every_linear(monkeypatch):
    """The served cut's line (`tokens`) is the per-expert line's model cut another way, so at the 128k mix its indexer
    factor and served MACs a token are the per-expert line's: k = 283.53 and 79.25e9 MACs.  With `LINEARS` missing the
    served cut's MoE GEMMs (`MoeExpertFp8CoordinateChunk_v1`, `MoeExpertFp8GemmW_v1`) it read k = 156.0 and 66.6e9
    MACs, and so 19% more requests; such a list is refused by those blocks' names.  Slow for the planner's import."""
    from breakeven import levers, reprice

    served, per_expert = (reprice.workload(levers.make_line(name))[7] for name in ("tokens", "per_expert"))
    assert served["indexer_factor"] == pytest.approx(283.5265, rel=1e-6)
    assert served["indexer_factor"] == pytest.approx(per_expert["indexer_factor"], rel=1e-12)
    assert served["served_macs"] == pytest.approx(per_expert["served_macs"], rel=1e-12)
    monkeypatch.setattr(reprice, "LINEARS", OLD_LINEARS)
    with pytest.raises(ValueError, match="MoE expert blocks MoeExpertFp8CoordinateChunk_v1, MoeExpertFp8GemmW_v1 match no"):
        reprice.workload(levers.make_line("tokens"))
