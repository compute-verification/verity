"""The break-even planner (`breakeven/`) on Figure 1's own path: the proof server's line builds from the figure's
cards and planner, and its bound falls as β grows."""
from math import isfinite

import pytest


def test_a_cases_serve_tax_is_the_sum_of_the_cards_named_parts():
    """The planner's serve tax is the serve_tax card's (A), part by part: a case with no lever pays every part and so
    the card's total; one whose replay reads the edge's state live pays every part but the edge snapshot; one with its
    own replay-block length pays the edge's pause over that length; and the planner reads every part the card names,
    so a part added to the card can't be left out of a case's tax."""
    from breakeven import sweep

    parts, tc = sweep.PARTS, sweep.tc
    assert set(parts) == {*sweep.SERVED, tc.EDGE} and len(sweep.SERVED) == len(set(sweep.SERVED))
    assert {"step record", "two-stage boundary commitment", "history tables"} <= set(sweep.SERVED)
    assert sweep.Case().tax() == pytest.approx(sum(parts.values()), rel=1e-12)
    assert sweep.Case().tax() == pytest.approx(sweep.TAXES["tokens"]["tax"], rel=1e-12)
    served = sum(parts[p] for p in sweep.SERVED)
    assert sweep.Case(live=True).tax() == pytest.approx(served, rel=1e-12)
    assert parts[tc.EDGE] == pytest.approx(sweep.EDGE_PAUSE / sweep.WINDOW, rel=1e-12)
    assert sweep.Case(edge_s=60.0).tax() == pytest.approx(served + sweep.EDGE_PAUSE / 60.0, rel=1e-12)
    assert sweep.Case(form="one-stage").tax() == sweep.TAX_ONE_STAGE
    for name, case in sweep.cases().items():
        if case.form != "one-stage":
            assert case.tax() >= served, name


@pytest.mark.slow
def test_the_proof_servers_line_prices_on_figure_1s_path():
    """`P5+poolwin600`, the proof server's line at its last step, builds from the cards (fused per-expert reads, the
    server's per-type prices, the column blocks, the 128k workload) and its world-year bound at a datacenter's
    verification phase is finite, positive and smaller at 10% of compute than at 1%: a change to figure1, total_cost
    or the cards that breaks the planner fails here rather than when a plan is next regenerated.  At one machine's
    verification phase 10% of compute pays for no sampled request, one proof block of each block type proved, so
    there the bound is everything the scope serves.  Importing the planner builds every Figure 1 line, so it is slow."""
    from dataclasses import replace

    from breakeven import sweep

    case = sweep.cases()["P5+poolwin600"]
    line = sweep.build(replace(case, registration="datacenter"))
    at1, at10 = (sweep.bound(line, beta, "world year", case.scale) for beta in (0.01, 0.10))
    assert isfinite(at1) and isfinite(at10) and 0 < at10 < at1
    y, pt = sweep.bound(sweep.build(case), 0.10, "world year", case.scale, full=True)
    assert pt["registration_sample"] == 0 and pt["plan"] is None and pt["sample_budgets"] > 1 and y > at1


@pytest.mark.slow
def test_a_steady_state_row_prices_its_type_and_no_other(tmp_path):
    """A type a steady-state rows file holds costs its row's GPU-seconds a block in the row's statements, with no
    lever on it but the verifier's seconds and the row's record as its basis; every other type keeps its price and
    names the table that set it; the line still plans.  A row that lacks a field is refused.  Slow as the test above."""
    import json
    from dataclasses import replace

    from breakeven import sweep

    reprice, case = sweep.reprice, sweep.cases()["P5+poolwin600"]
    strata = reprice.price_fused(sweep.fuse_reads(sweep.priced_strata(case.line), fixed="sum"), case.fused)
    block = next(x["block"] for x in strata if x["block"].startswith("SparseMlaSplitPart_v1{T=64,"))
    row = {"block": block, "statement_seconds": 3.2, "blocks": 128, "provers": 1, "record": "the-run"}
    path = tmp_path / "steady.json"
    path.write_text(json.dumps({"rows": [row]}))
    rows = reprice.steady(path)

    levers = dict(prover=2.0, k4=True, k_gain="mps_types", front=True, plain=True, cold=0.5)
    before = reprice.apply_server(strata, verify=1.0, **levers)
    after = reprice.apply_server(strata, verify=1.0, steady=rows, **levers)
    bare = reprice.apply_server(strata, steady=rows, **levers)
    for b, a, x in zip(before, after, bare, strict=True):
        if a["block"] != block:
            assert a == b and a["basis"].startswith("reprice.")
            continue
        assert (a["priced"], a["basis"], a["statement_blocks"]) == ("steady", "steady: the-run", 128)
        assert a["seconds_per_block"] == x["seconds_per_block"] == 3.2 / 128
        verifier = b["fixed_seconds"] - _fixed(reprice, strata, block, levers)
        assert x["fixed_seconds"] == 0.0 and a["fixed_seconds"] == pytest.approx(verifier)
    assert {x["basis"] for x in before} >= {"reprice.DEFAULT", "reprice.SERVER", "reprice.FUSED_SERVER"}

    line = sweep.build(replace(case, steady=str(path)))
    shares = reprice.basis_shares(line[1]["partition"].value["strata"])
    assert "steady: the-run" in shares and abs(sum(shares.values()) - 1) < 1e-9
    assert isfinite(at := sweep.bound(line, 0.10, "world year", case.scale)) and at > 0

    path.write_text(json.dumps({"rows": [{k: v for k, v in row.items() if k != "record"}]}))
    with pytest.raises(ValueError, match="lacks record"):
        reprice.steady(path)


def _fixed(reprice, strata, block, levers) -> float:
    """The fixed seconds a statement of `block` proves in with no steady-state row and no verifier."""
    return next(x["fixed_seconds"] for x in reprice.apply_server(strata, **levers) if x["block"] == block)


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
