"""Pins for the lookup-argument cost models (``verity_numerical.explore.lookups``): baseline reproduction, column names,
the table-width trade and the killing table."""

from __future__ import annotations

import pytest
from verity_numerical.bench import contract
from verity_numerical.explore import lookups as L
from verity_numerical.security import accounting as acc
from verity_numerical.security import census as cen

P128, P80 = L.POINTS["2^-128"], L.POINTS["2^-80"]


def test_baseline_reproduces_protocol_md_figures():
    """PROTOCOL.md A 9.4 / 5.4: committed 181.35M (A) and 779.6M (B) at B=4096; depth 228 / 264 / 381; 1 589 slots."""
    a = {B: L.cost_a("logup16", B, P128) for B in L.BATCHES}
    b = {B: L.cost_b("logup16", B, P128) for B in L.BATCHES}
    assert a[4096]["census.committed_elements"] == 181_353_221
    assert b[4096]["census.committed_elements"] == 779_620_604
    assert [a[B]["rounds.sequential_depth"] for B in L.BATCHES] == [228, 264, 381]
    assert a[4096]["rounds.protocol"] == 1589
    assert all(b[B]["rounds.sequential_depth"] == 3 for B in L.BATCHES)
    # identical to calling the accountant directly on the census tables
    direct = acc.census_estimate_a(acc.CandidateA(K=1536, B=4096, trace_field=L.GL, challenge_field=L.GL.extension(3),
                                                  ligero=P128.ligero))
    for k, v in direct.items():
        assert a[4096][k] == v


def test_column_names_are_the_contract_census_columns():
    for opt in L.OPTIONS_A:
        r = L.cost_a(opt, 64, P128)
        for c in L.CENSUS_COLUMNS + L.EXTRA_COLUMNS:
            assert c in r, (opt, c)
        assert not [k for k in r if k.startswith("census.") and k not in L.CENSUS_COLUMNS]
    for opt in L.OPTIONS_B:
        r = L.cost_b(opt, 64, P128)
        assert set(L.CENSUS_COLUMNS + L.EXTRA_COLUMNS) <= set(r)
    # the census / rounds / transcript names are contract measurement names with units
    for c in L.CENSUS_COLUMNS + L.EXTRA_COLUMNS[:3]:
        assert contract.unit_of(c) is not None, c
    assert not contract.validate_measurements([contract.measurement(c, int(L.cost_a("logup16", 1, P128)[c]))
                                               for c in L.CENSUS_COLUMNS + L.EXTRA_COLUMNS[:3]])


def test_range_plans_from_the_census():
    base = L.range_plan(L.RangeScheme(16))
    assert base.queries == {"T_range16": 284} and base.chunk_columns_delta == 0 and base.splits_delta == 0
    assert base.queries["T_range16"] == cen.lookups_by_table(acc.CHECKER_UNIT)["T_range16"]
    small = L.range_plan(L.RangeScheme(16, dedicated_small=True))
    assert small.queries == {"T_range7": 36, "T_range8": 66, "T_range9": 26, "T_range16": 156}
    assert small.rows["T_range9"] == 512 and small.chunk_columns_delta == 0
    eight = L.range_plan(L.RangeScheme(8))
    assert eight.queries == {"T_range8": 459} and eight.chunk_columns_delta == 175 == eight.splits_delta
    twelve = L.range_plan(L.RangeScheme(12))
    assert twelve.queries == {"T_range12": 411}
    assert len(L.tables_for()) == 5 and L.tables_for()[-4:] == L.FIXED_TABLES


def test_dedicated_small_tables_cut_a_depth_by_one_hypercube_dimension_at_b4096():
    base, small = L.cost_a("logup16", 4096, P128), L.cost_a("logup16+small", 4096, P128)
    assert (base["lookup.hypercube_dim"], small["lookup.hypercube_dim"]) == (27, 26)
    assert (base["rounds.sequential_depth"], small["rounds.sequential_depth"]) == (381, 354)
    assert 0 < small["census.committed_elements"] - base["census.committed_elements"] < 10_000
    # under sharding + 2 variables per message the checker chain is the floor for both
    d = L.DEPTH_PROFILES["c2s5"]
    assert L.cost_a("logup16", 4096, P128, d)["rounds.sequential_depth"] == 158
    assert L.cost_a("logup16+small", 4096, P128, d)["rounds.sequential_depth"] == 158


def test_narrow_chunks_lose_at_scale_and_win_at_b1():
    big = {opt: L.cost_a(opt, 4096, P128) for opt in ("logup16", "logup8", "logup12")}
    assert big["logup8"]["rounds.sequential_depth"] == big["logup12"]["rounds.sequential_depth"] == 409
    assert big["logup8"]["rounds.sequential_depth"] > big["logup16"]["rounds.sequential_depth"] == 381
    assert big["logup8"]["census.committed_elements"] > big["logup12"]["census.committed_elements"] > big["logup16"]["census.committed_elements"]
    bb = {opt: L.cost_b(opt, 4096, P128) for opt in ("logup16", "logup8", "logup12")}
    assert bb["logup8"]["census.committed_elements"] > bb["logup12"]["census.committed_elements"] > bb["logup16"]["census.committed_elements"]
    one = {opt: L.cost_b(opt, 1, P128) for opt in ("logup16", "logup8")}
    assert one["logup8"]["census.committed_elements"] < one["logup16"]["census.committed_elements"]
    assert 2 < L.crossover_batch(8, 16) < 4 and 3 < L.crossover_batch(12, 16) < 5
    sweep = {r["W"]: r for r in L.range_width_sweep(4096)}
    assert sweep[16]["A.committed_delta"] == 0 and sweep[16]["B.cells_delta"] == 0
    assert min(sweep, key=lambda W: (sweep[W]["B.lookup_cells"])) == 16
    assert all(sweep[W]["queries_per_unit"] == 284 for W in range(16, 23))   # 16 < w <= 32 needs two chunks anyway


def test_lasso_and_shout_cost_committed_elements_not_depth():
    N = L._range_N(4096)
    lasso = L.lasso_extra(N, 16, 2, 3)
    assert lasso.committed == 3 * 2 * N + 2 * 256 and lasso.hypercube_dim == 27 and lasso.chain == 378
    shout = L.shout_extra(N, 16, 8, 3)
    assert shout.committed == 8 * N * 4 and shout.chain == 27 + 16 + 1
    a = L.cost_a("lasso16_c2", 4096, P128)
    assert a["rounds.sequential_depth"] == 381 and a["census.committed_elements"] > 4 * 181_353_221
    s = L.cost_a("shout16_d8", 4096, P128)
    assert s["rounds.sequential_depth"] == 300 and s["census.committed_elements"] > 20 * 181_353_221
    assert s["lookup.chain"] == 44 and 297 == acc.rounds_and_depth_a(L.candidate_a(4096, P128, L.FIXED_TABLES))["depth.checker_chain"]
    pl = L.cost_a("plookup_gp", 4096, P128)
    assert pl["rounds.sequential_depth"] == 381 and pl["lookup.committed_extra"] == N + 65536
    with pytest.raises(acc.AccountingError):
        L.cost_b("shout16_d4", 64, P128)


def test_merged_logup_over_all_tables_changes_nothing_but_pads():
    base, merged = L.cost_a("logup16", 4096, P128), L.cost_a("logup_merged", 4096, P128)
    assert merged["lookup.hypercube_dim"] == 27 and merged["rounds.sequential_depth"] == 381
    assert 0 < base["census.committed_elements"] - merged["census.committed_elements"] < 10_000
    assert L.cost_b("logup_merged", 4096, P128)["census.committed_elements"] == L.cost_b("logup16", 4096, P128)["census.committed_elements"]


def test_killing_table_no_option_beats_logup_on_both_axes_for_a():
    for point in (P128, P80):
        for dp in L.DEPTH_PROFILES:
            k = L.killing_table(point, dp)
            for B, rows in k.items():
                assert not any(r["beats_A_both"] for r in rows.values()), (point.name, dp, B)
            if dp == "c1s0":
                assert k["4096"]["logup16+small"]["beats_A_depth_within_0.1pct_committed"]
                assert k["1"]["logup8"]["beats_B_committed"] and not k["64"]["logup8"]["beats_B_committed"]
                assert not any(r["beats_B_committed"] for r in k["4096"].values())


def test_every_logup_option_meets_its_operating_point():
    for point in (P128, P80):
        for B in L.BATCHES:
            for opt in L.OPTIONS_A:
                assert L.cost_a(opt, B, point)["soundness.meets"] == 1.0, (point.name, B, opt)
            for opt in L.OPTIONS_B:
                assert L.cost_b(opt, B, point)["soundness.meets"] == 1.0, (point.name, B, opt)


def test_report_shape():
    r = L.report()
    assert set(r["options"]) == {"2^-128", "2^-80"} and set(r["killing"]["2^-128"]) == set(L.DEPTH_PROFILES)
    assert set(r["citations"]) >= {"Hab22", "PH23", "STW23", "AST23", "Tha25", "GW20", "EFG22"}
    assert r["options"]["2^-128"]["c1s0"]["A"]["4096"]["logup16"]["census.committed_elements"] == 181_353_221
