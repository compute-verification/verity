"""`verity bench prove`'s totals on small tables: a row an art holds, a type no census holds, a single session at
another size, the report's shares and range, a partial total, and SPOT at a sampling plan with or without its own
prices. `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from prove_helpers import (
    CENSUS,
    CUT,
    GEMM32,
    GEMM64,
    NORM,
    SESSIONS,
    _outside_a_research_run,  # noqa: F401  (autouse)
    plan,
    steady,
)
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import prove_records as R
from verity_bench import prove_render


def test_a_row_an_art_holds_is_its_runs_and_names_the_art():
    """A row an art holds in place of its run's own is ordered as that run's, and the type's row names the art as its
    record and the run that measured it."""

    census = P.Census.of(CENSUS)
    gemm, norm = P.definition_id(GEMM64), P.definition_id(NORM)
    art = "art:" + "ab" * 32
    rows = (
        steady("gemm64", gemm, "r20260101-000000-0001", 8, 16.0),
        steady("gemm64", gemm, "r20260102-000000-0002", 8, 8.0, art=art),                      # the newer run's, in an art
        steady("norm", norm, "r20260101-000000-0001", 8, 24.0, k_log=22),
        steady("norm", norm, "r20260102-000000-0002", 8, None, k_log=22, art=art, refused="its row lacks `flat`"),
    )
    cost = {r.block: r.cost for r in PF.Flock(rows, SESSIONS, "art:card").table(census).rows}
    assert (cost[GEMM64].source, cost[GEMM64].seconds, cost[GEMM64].record) == (P.STEADY, 1.0, art)
    assert cost[GEMM64].note == (f"`gemm64` (`{gemm}`), 1 prover, full statements (rate.json of art:abababababab, measured in "
                                 "r20260102-000000-0002)")
    # the art's row of the norm isn't a steady-state measurement, so the older run's own prices it
    assert (cost[NORM].source, cost[NORM].seconds, cost[NORM].record) == (P.STEADY, 3.0, "r20260101-000000-0001")
    tried = {r.block: r.cost for r in PF.Flock(rows[3:], SESSIONS, "art:card").table(census).rows}[NORM]
    assert tried.source == P.MODELED and tried.note == (
        "steady state tried as `norm` at 1 prover in r20260102-000000-0002 (rate.json of art:abababababab): its row lacks `flat`")


def test_a_steady_row_of_no_block_type_of_the_census_is_unread_not_dropped():
    """A steady-state row whose staged Definition the census doesn't hold, or holds no longer under its cut, is on no
    block type's row: the table lists it as unread, at whatever prover count and cores, with its figure where it is a
    steady-state measurement and why it isn't one otherwise. It enters no figure: no row, no total, and not the
    range's other end, whose median is over the census's steady-state types alone."""

    census = P.Census.of(CENSUS)
    norm, art = P.definition_id(NORM), "art:" + "cd" * 32
    priced = steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0)
    strays = (
        steady("norm", norm, "r20260102-000000-0002", 8, 24.0, k_log=22),                     # a block type the cut replaces
        steady("old-linear", "Linear_v0{K=64}", "r20260103-000000-0003", 4, 6.0, provers=2, art=art, cores=24),
        steady("old-linear", "Linear_v0{K=64}", "r20260103-000000-0003", 8, None, refused="void: out of device memory"),
        steady("idle", "Idle_v1{N=1}", "r20260101-000000-0001", 8, 2.0, cores=6),            # a type no step ran
    )
    flock = PF.Flock((priced, *strays), SESSIONS, "art:card")
    whole = flock.table(census)
    assert whole.unread == strays[1:] and {r.block: r.cost.source for r in whole.rows}[NORM] == P.STEADY
    recut = flock.table(census.recut(CUT))
    assert recut.unread == strays
    out = P.report(recut, 1.0)
    assert out["unread"] == [
        {"record": "r20260102-000000-0002", "run": "r20260102-000000-0002", "type": "norm", "definition": norm,
         "provers": 1, "cores": 10, "seconds": 3.0, "why": None},
        {"record": art, "run": "r20260103-000000-0003", "type": "old-linear", "definition": "Linear_v0{K=64}",
         "provers": 2, "cores": 24, "seconds": 1.5, "why": None},
        {"record": "r20260103-000000-0003", "run": "r20260103-000000-0003", "type": "old-linear",
         "definition": "Linear_v0{K=64}", "provers": 1, "cores": 10, "seconds": None, "why": "void: out of device memory"},
        {"record": "r20260101-000000-0001", "run": "r20260101-000000-0001", "type": "idle", "definition": "Idle_v1{N=1}",
         "provers": 1, "cores": 6, "seconds": 0.25, "why": None}]
    alone = P.report(PF.Flock((priced,), SESSIONS, "art:card").table(census.recut(CUT)), 1.0)
    assert alone["unread"] == [] and {k: v for k, v in out.items() if k != "unread"} == {k: v for k, v in alone.items() if k != "unread"}
    # a sampling plan takes the prover's table, its unread rows with it
    cells = [{"block": b.name, "type": "", "taken": 1.0} for b in census.blocks]
    assert P.Spot(flock, plan(cells), P.Serving(0.5, 4000.0, "stated")).table(census).unread == strays[1:]

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.9, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert text.endswith(
        "\nUnread (labelled steady-state rows whose staged Definition is no block type of this census under its cut; "
        "they are in no figure above):\n\n"
        f"- r20260102-000000-0002: `norm` at 1 prover, `{norm}`: measured at steady state, 3,000 ms per block\n"
        "- art:cdcdcdcdcdcd, run r20260103-000000-0003: `old-linear` at 2 provers, `Linear_v0{K=64}`: measured at steady "
        "state, 1,500 ms per block\n"
        "- r20260103-000000-0003: `old-linear` at 1 prover, `Linear_v0{K=64}`: not a steady-state measurement: void: out "
        "of device memory\n"
        "- r20260101-000000-0001: `idle` at 1 prover, `Idle_v1{N=1}`: measured at steady state, 250 ms per block\n")
    assert "Unread" not in R.render(page | alone)


def test_a_single_session_prices_its_definition_and_k_at_another_size():
    """A block type the card doesn't measure by name takes its Definition and K's measurement at 2^Δk the seconds and
    2^-Δk the statement, Δk between the two k_logs as the census holds them."""

    doc = {"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        "Gemm_v1{K=64,N=1}": {"k_log": 21, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=4}": {"k_log": 23, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=0}": {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0}}}
    # the card staged N=1 at k_log 20; the census holds it at 21, which is what Δk is taken from
    card = (PF.Session("Gemm_v1{K=64,N=1}", 20, 0.5, 2.0, 1 << 15),)
    cost = {r.block: r.cost for r in PF.Flock((), card, "art:card").table(P.Census.of(doc)).rows}
    assert cost["Gemm_v1{K=64,N=1}"].seconds == 0.5 + 2.0 / (1 << 15)
    assert cost["Gemm_v1{K=64,N=4}"].seconds == 0.5 * 4 + 2.0 / (1 << 13) and "at 2^2 its seconds" in cost["Gemm_v1{K=64,N=4}"].note
    assert cost["Gemm_v1{K=64,N=0}"].seconds == 0.5 / 2 + 2.0 / (1 << 16)
    assert {c.source for c in cost.values()} == {P.SINGLE}


def test_the_report_gives_each_sources_share_and_the_totals_with_their_range():
    census = P.Census.of(CENSUS).recut(CUT)
    table = PF.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),), SESSIONS,
                    "art:card").table(census)
    serving = P.Serving(0.5, 4000.0, "stated")
    assert serving.gpu_seconds_per_token(census) == 2 * 1000 / (0.5 * 4000.0)
    doc = P.report(table, 1.0)
    total = table.gpu_seconds_per_token()
    low, high = table.gpu_seconds_range()
    t = doc["totals"]
    assert t["gpu_seconds_per_token"] == total and t["tokens_per_gpu_second"] == 1 / total
    assert t["gpu_seconds_per_token_range"] == [low, high] and t["tokens_per_gpu_second_range"] == [1 / high, 1 / low]
    assert t["gpus_per_serving_gpu"] == total and t["blocks_per_token"] == 4.25
    # the cut's bare block type has no figure, so the totals are partial and name it; this cut holds no MACs
    assert t["status"] == "partial" and t["steady_time_share"] == pytest.approx(2.0 / total)
    assert t["missing"] == [{"block": "Bare_v1{N=4}", "per_token": 0.25, "block_share": 0.25 / 4.25, "mac_share": None}]
    assert t["verifier_seconds_per_token"] == {"value": 1.0, "block_types": 1, "block_share": 2.0 / 4.25}
    assert t["proof_bytes_per_token"] is None
    by = {s["source"]: s for s in doc["sources"]}
    assert list(by) == list(P.SOURCES)
    assert (by[P.STEADY]["block_types"], by[P.STEADY]["per_token"], by[P.STEADY]["gpu_seconds_per_token"]) == (1, 2.0, 2.0)
    assert (by[P.MODELED]["block_types"], by[P.MISSING]["block_types"], by[P.SINGLE]["block_types"]) == (2, 1, 0)
    assert by[P.MISSING]["gpu_seconds_per_token"] == 0 and by[P.MISSING]["block_share"] == 0.25 / 4.25
    assert sum(s["time_share"] for s in doc["sources"]) == pytest.approx(1.0)
    assert [r["block"] for r in doc["rows"]][-1] == "Bare_v1{N=4}" and doc["rows"][-1]["time_share"] is None
    assert doc["rows"][0]["time_share"] == max(r["time_share"] or 0 for r in doc["rows"])

    # measured only when every block type is measured at steady state; modeled when one is priced another way
    whole = P.Census.of({**CENSUS, "definitions": {GEMM64: CENSUS["definitions"][GEMM64]}})
    unverified = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),)
    measured = P.report(PF.Flock(unverified, SESSIONS, "art:card").table(whole), 1.0)["totals"]
    assert measured["status"] == "measured" and measured["verifier_seconds_per_token"] is None and measured["missing"] == []
    modeled = P.report(PF.Flock(unverified, SESSIONS, "art:card").table(P.Census.of(CENSUS)), 1.0)["totals"]
    assert modeled["status"] == "modeled" and modeled["missing"] == [] and modeled["steady_time_share"] < 1.0


def test_a_total_whose_every_type_is_steady_or_warm_is_modeled_never_measured():
    """Only steady-state and warm-session rows, every type priced from a run of its own: the total is still modeled,
    since a statement proved from warm sessions on one card is not the steady state."""

    both = P.Census.of({**CENSUS, "definitions": {k: CENSUS["definitions"][k] for k in (GEMM64, GEMM32)}})
    rows = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),)
    warm = (PF.Warm(P.definition_id(GEMM32), 19, 4, 4, 2.0, 1, 4, "r20261115-000000-0001", "gemm32"),)
    doc = P.report(PF.Flock(rows, SESSIONS, "art:card", warm=warm).table(both), 1.0)
    assert sorted(r["source"] for r in doc["rows"]) == [P.STEADY, P.WARM]
    assert doc["totals"]["status"] == "modeled" and doc["totals"]["missing"] == []
    assert doc["totals"]["steady_time_share"] < 1.0


def test_a_total_that_leaves_a_block_type_out_is_partial_and_names_it(monkeypatch):
    """One steady-state type and one type with no figure at three quarters of the blocks: steady-state measurements
    cover all of the time the total prices, and the total is still partial, never measured. It names the type it leaves
    out with its share of the blocks, and of the MACs where the census holds them."""

    doc: dict[str, Any] = {"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 100, "macs": 80}, "definitions": {
        GEMM64: {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0, "macs": 60},
        "Unpriced_v1{N=1}": {"of": {"Unpriced_v1": "row"}, "prefill": 3, "decode": 0, "macs": 5},
        "Rare_v1{N=1}": {"of": {"Rare_v1": "row"}, "prefill": 1, "decode": 0, "macs": 5}}}
    reports = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),)
    out = P.report(PF.Flock(reports, SESSIONS, "art:card").table(P.Census.of(doc)), 1.0)
    t = out["totals"]
    assert {s["source"]: s["block_types"] for s in out["sources"]} == {P.STEADY: 1, P.WARM: 0, P.FAMILY: 0, P.SINGLE: 0, P.MODELED: 0, P.MISSING: 2}
    assert (t["steady_time_share"], t["gpu_seconds_per_token"], t["status"]) == (1.0, 1.0, "partial")
    assert t["missing"] == [
        {"block": "Unpriced_v1{N=1}", "per_token": 3.0, "block_share": 0.6, "mac_share": 0.1875},
        {"block": "Rare_v1{N=1}", "per_token": 1.0, "block_share": 0.2, "mac_share": 0.0625}]

    # a census that holds no MACs for one of its types gives the share of the blocks alone
    del doc["definitions"][GEMM64]["macs"]
    bare = P.report(PF.Flock(reports, SESSIONS, "art:card").table(P.Census.of(doc)), 1.0)["totals"]
    assert bare["status"] == "partial" and [(m["block_share"], m["mac_share"]) for m in bare["missing"]] == [(0.6, None), (0.2, None)]
    # at a sampling plan the shares are of the blocks the plan samples
    cells = [{"block": name, "type": "", "taken": 1.0} for name in doc["definitions"]]
    sampled = P.report(P.Spot(PF.Flock(reports, SESSIONS, "art:card"), plan(cells), P.Serving(0.5, 400.0, "stated")).table(
        P.Census.of(doc)), 1.0)["totals"]
    assert sampled["status"] == "partial" and [m["block_share"] for m in sampled["missing"]] == pytest.approx([1 / 3, 1 / 3])

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.8, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert "- prover GPUs per serving GPU: 1, partial: steady-state measurements cover 100.0% of the proving time it prices" in text
    assert ("- partial: 2 block type(s) with no figure, whose proving every total here leaves out: `Unpriced_v1{N=1}` "
            "(60.00% of the blocks, 18.75% of the blocks' MACs); `Rare_v1{N=1}` (20.00% of the blocks, 6.25% of the "
            "blocks' MACs)\n") in text
    assert "not measured at that size, and partial as that ratio is:" in text
    monkeypatch.setattr(prove_render, "MISSING_NAMED", 1)
    assert "(60.00% of the blocks, 18.75% of the blocks' MACs); and 1 more, which `--json` names\n" in R.render(page)
    whole = R.render(page | {"totals": t | {"status": "measured", "missing": []}})
    assert "measured: steady-state measurements cover 100.0% of its proving time" in whole and "partial" not in whole


def test_spot_prices_the_blocks_its_plan_samples_and_refuses_another_census_plan():
    census = P.Census.of(CENSUS)
    flock = PF.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
    serving = P.Serving(0.5, 4000.0, "stated")                    # 1 GPU-second a token: 800 tokens in the plan's period
    cells = [{"block": "Fused[its own name of the K=64 GEMM]", "type": "gemm64", "taken": 4.0},
             {"block": GEMM32, "type": "", "taken": 2.0}, {"block": NORM, "type": "norm", "taken": 0.8}]
    got = plan(cells)
    assert got.sampled == {cells[0]["block"]: 40.0, GEMM32: 20.0, NORM: 8.0}
    assert got.staged == {cells[0]["block"]: "gemm64", NORM: "norm"} and (got.name, got.q) == ("toy", 0.01)

    inner, table = flock.table(census), P.Spot(flock, got, serving).table(census)
    assert table.system == "spot[flock]" and table.range_basis == inner.range_basis
    # a cell is a row's by name, else by the staged type that measured the row; the costs are the prover's own
    assert [(r.block, r.per_token) for r in table.rows] == [(GEMM64, 40 / 800), (GEMM32, 20 / 800), (NORM, 8 / 800)]
    assert [r.cost for r in table.rows] == [r.cost for r in inner.rows]
    # the prover GPUs a serving GPU are the plan's proving over its serving, whatever the serving rate
    proving = sum(n * r.cost.seconds for n, r in zip((40.0, 20.0, 8.0), inner.rows))
    assert P.report(table, 1.0)["totals"]["gpus_per_serving_gpu"] == pytest.approx(proving / 800.0)
    half = P.Serving(0.25, 4000.0, "stated")
    slow = P.report(P.Spot(flock, got, half).table(census), half.gpu_seconds_per_token(census))["totals"]
    assert slow["gpus_per_serving_gpu"] == pytest.approx(proving / 800.0)
    assert slow["tokens_per_gpu_second"] == pytest.approx(400 / proving)

    with pytest.raises(ValueError, match=r"art:plan and the census don't cover each other.*no cell for 1 block type\(s\) "
                                         r"\(Norm_v1.*no block type for 0 cell"):
        P.Spot(flock, plan(cells[:2]), serving).table(census)
    with pytest.raises(ValueError, match=r"no cell for 0 block type\(s\), and the census no block type for 1 cell\(s\) "
                                         r"\(Stray_v1"):
        P.Spot(flock, plan([*cells, {"block": "Stray_v1{N=1}", "type": "", "taken": 1.0}]), serving).table(census)

    # a newer, fuller row staged under another name prices the type; the plan's cell still finds it by the old name
    renamed = PF.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 4, 8.0),
                       steady("gemm-k64", P.definition_id(GEMM64), "r20260103-000000-0003", 8, 4.0)), SESSIONS, "art:card")
    rows = P.Spot(renamed, got, serving).table(census).rows
    assert (rows[0].block, rows[0].per_token, rows[0].cost.staged, rows[0].cost.seconds) == (GEMM64, 40 / 800, "gemm-k64", 0.5)
    assert rows[0].staged_as == ("gemm-k64", "gemm64")
    # the old name's only row superseded: it prices nothing, and still ties the plan's cell to the type
    aliased = PF.Flock((steady("gemm-k64", P.definition_id(GEMM64), "r20260103-000000-0003", 8, 4.0),), SESSIONS, "art:card",
                      aliases=((P.definition_id(GEMM64), "gemm64"),))
    rows = P.Spot(aliased, got, serving).table(census).rows
    assert (rows[0].block, rows[0].cost.staged, rows[0].cost.seconds) == (GEMM64, "gemm-k64", 0.5)
    with pytest.raises(ValueError, match="don't cover each other"):
        P.Spot(replace(aliased, aliases=()), got, serving).table(census)


def test_a_plan_that_states_its_own_prices_gets_both_totals_and_where_they_differ():
    """A plan's sample fits its planner's prices, and the table prices the same blocks from its own sources. Where the
    plan states its prices the report gives the total at each, each source's proving time at both and the block type
    that holds most of the difference; the table's own total doesn't move. A plan that prices none of its cells has
    the one total, and one that prices only some is refused."""

    census = P.Census.of(CENSUS)
    flock = PF.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
    serving = P.Serving(0.5, 4000.0, "stated")                    # 800 tokens in the plan's period, as above
    cells = [{"block": GEMM64, "type": "gemm64", "taken": 4.0, "prove_s": 2.0},
             {"block": GEMM32, "type": "", "taken": 2.0, "prove_s": 0.5}, {"block": NORM, "type": "norm", "taken": 0.8, "prove_s": 0.7}]
    got = plan(cells)
    assert got.priced == pytest.approx({GEMM64: 0.5, GEMM32: 0.25, NORM: 0.875})     # a cell's seconds over its blocks
    table = P.Spot(flock, got, serving).table(census)
    assert [r.planned for r in table.rows] == pytest.approx([0.5, 0.25, 0.875])
    out = P.report(table, 1.0)
    t, sampled = out["totals"], (40.0, 20.0, 8.0)
    own = [r.cost.seconds for r in table.rows]
    assert own[0] == 1.0 and t["gpus_per_serving_gpu"] == pytest.approx(sum(n * x for n, x in zip(sampled, own)) / 800)
    its = t["planned"]
    assert its["gpus_per_serving_gpu"] == its["gpu_seconds_per_token"] == pytest.approx((20 + 5 + 7) / 800)
    # no session measured the K=32 GEMM or the norm: one steady-state type, two modeled from k_log
    assert [(x["source"], x["block_types"]) for x in its["sources"]] == [(P.STEADY, 1), (P.MODELED, 2)]
    assert [x["gpu_seconds_per_token"] for x in its["sources"]] == pytest.approx([40 / 800, (20 * own[1] + 8 * own[2]) / 800])
    assert [x["planned_gpu_seconds_per_token"] for x in its["sources"]] == pytest.approx([20 / 800, (5 + 7) / 800])
    gaps = [n * (x - y) for n, x, y in zip(sampled, own, (0.5, 0.25, 0.875))]
    top = max(range(3), key=lambda i: abs(gaps[i]))
    assert its["largest"] == {"block": table.rows[top].block, "seconds": own[top], "planned_seconds": table.rows[top].planned,
                              "difference_share": pytest.approx(gaps[top] / sum(gaps))}
    assert {r["block"]: r["planned_seconds"] for r in out["rows"]} == got.priced

    page = {"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.9, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert "| ms per block | ms per block in the plan | source |" in text
    assert "| 1,000 | 500 | measured at steady state |" in text
    assert "- the same blocks at the sampling plan's own prices: 0.04 prover GPUs per serving GPU. Its planner sized" in text
    assert ("- where the two differ, this table's proving time over the plan's by this table's source: measured at steady "
            f"state 2x (1 block types), modeled from k_log {(20 * own[1] + 8 * own[2]) / 12:.3g}x (2 block types); `") in text
    assert f"alone is {gaps[top] / sum(gaps):.1%} of the difference" in text
    assert text.count(" --- |") == 9 + 6 + 5 + 6                          # the rows' table has the plan's column

    bare = [{k: v for k, v in c.items() if k != "prove_s"} for c in cells]
    one = P.report(P.Spot(flock, plan(bare), serving).table(census), 1.0)
    assert one["totals"]["planned"] is None and one["totals"]["gpus_per_serving_gpu"] == t["gpus_per_serving_gpu"]
    assert "in the plan" not in (plain := R.render(page | one)) and "own prices" not in plain
    assert P.report(flock.table(census), 1.0)["totals"]["planned"] is None
    with pytest.raises(ValueError, match=r"art:plan states its planner's seconds \(`prove_s`\) for 2 cell\(s\) and not for 1 "
                                         r"\(Norm_v1.*prices every cell or none"):
        plan([*cells[:2], bare[2]])
    # a priced cell that takes no block has no seconds a block, and doesn't void the others'
    assert plan([*cells, {"block": "Idle_v1{N=1}", "type": "", "taken": 0.0, "prove_s": 0.0}]).priced["Idle_v1{N=1}"] == 0.0


def test_a_priced_plans_totals_where_the_table_has_no_price_a_higher_one_or_the_same():
    """The edges of a plan's own prices beside the table's. A block type the table has no figure for is in the plan's
    total and in neither side of the comparison. The block type that holds most of the difference is the largest in
    size, whichever price is the higher. Where the two prices agree, or the table prices nothing, no block type is
    named; and a source the plan prices at 0 has no ratio."""

    census = P.Census.of(CENSUS).recut(CUT)
    half, bare = "Half_v1{N=4}", "Bare_v1{N=4}"
    flock = PF.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
    serving = P.Serving(0.5, 4000.0, "stated")                    # 800 tokens in the plan's period
    cells = [{"block": GEMM64, "type": "gemm64", "taken": 4.0, "prove_s": 2.0}, {"block": GEMM32, "type": "", "taken": 2.0, "prove_s": 0.25},
             {"block": half, "type": "", "taken": 1.0, "prove_s": 0.5}, {"block": bare, "type": "", "taken": 0.8, "prove_s": 0.2}]
    got = plan(cells)
    assert got.priced == pytest.approx({GEMM64: 0.5, GEMM32: 0.125, half: 0.5, bare: 0.25})
    own = {r.block: r.cost.seconds for r in flock.table(census).rows}
    assert own[GEMM64] == 1.0 and own[bare] is None and 0.25 < own[GEMM32] < 0.2501 and 1.0 < own[half] < 1.001

    def page(out: dict) -> str:
        return R.render({"workload": "toy", "model": "Toy", "provers": 1, "cores": 10, "inputs": [], "refused": [], "macs_in_blocks": 0.9,
                         **out, "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
                         "periods": {"gates_basis": "stated", "only": None, "rows": []}})

    def priced(at: dict[str, float]) -> dict:
        return P.report(P.Spot(flock, replace(got, priced=at), serving).table(census), 1.0)

    # a block type with no figure: the plan's total holds it, the table's leaves it out, and the comparison is over
    # the three the table prices
    out = P.report(P.Spot(flock, got, serving).table(census), 1.0)
    t, its = out["totals"], out["totals"]["planned"]
    assert t["status"] == "partial" and [m["block"] for m in t["missing"]] == [bare]
    assert t["gpus_per_serving_gpu"] == pytest.approx((40 + 20 * own[GEMM32] + 10 * own[half]) / 800)
    assert its["gpus_per_serving_gpu"] == pytest.approx((20 + 2.5 + 5 + 2) / 800)
    assert [(x["source"], x["block_types"]) for x in its["sources"]] == [(P.STEADY, 1), (P.MODELED, 2)]
    assert sum(x["planned_gpu_seconds_per_token"] for x in its["sources"]) == pytest.approx((20 + 2.5 + 5) / 800)
    gaps = {GEMM64: 40 * 0.5, GEMM32: 20 * (own[GEMM32] - 0.125), half: 10 * (own[half] - 0.5)}
    assert its["largest"] == {"block": GEMM64, "seconds": 1.0, "planned_seconds": 0.5,
                              "difference_share": pytest.approx(20 / sum(gaps.values()))}
    text = page(out)
    assert "| `Bare_v1{N=4}` | — | 0.01 | — | 250 | missing |" in text          # no ms per block here, the plan's beside it
    assert "- the same blocks at the sampling plan's own prices: 0.0369 prover GPUs per serving GPU." in text

    # the plan prices the K=64 GEMM higher than the table and the rest lower: it is still the block type named, by
    # the size of its difference, and its share is of the net difference, which the others offset
    its = (higher := priced(got.priced | {GEMM64: 3.0}))["totals"]["planned"]
    net = 40 * (1.0 - 3.0) + gaps[GEMM32] + gaps[half]
    assert its["gpus_per_serving_gpu"] > higher["totals"]["gpus_per_serving_gpu"]
    assert its["largest"] == {"block": GEMM64, "seconds": 1.0, "planned_seconds": 3.0,
                              "difference_share": pytest.approx(-80 / net)}
    assert its["largest"]["difference_share"] > 1
    text = page(higher)
    assert "measured at steady state 0.333x (1 block types), modeled from k_log" in text
    assert f"alone is {-80 / net:.1%} of the difference, 1,000 ms per block here and 3,000 ms in the plan" in text

    # the plan's prices are the table's: both totals, and no block type holds a difference
    same = priced({block: 0.25 if seconds is None else seconds for block, seconds in own.items()})
    its = same["totals"]["planned"]
    assert its["largest"] is None
    assert [x["gpu_seconds_per_token"] for x in its["sources"]] == [x["planned_gpu_seconds_per_token"] for x in its["sources"]]
    assert its["gpus_per_serving_gpu"] == pytest.approx(same["totals"]["gpus_per_serving_gpu"] + 8 * 0.25 / 800)
    assert "sampling plan's own prices" in (text := page(same)) and "where the two differ" not in text

    # a table that prices no block type has the plan's total alone
    its = P.report(P.Spot(PF.Flock(), got, serving).table(census), 1.0)["totals"]["planned"]
    assert its == {"gpu_seconds_per_token": pytest.approx(29.5 / 800), "gpus_per_serving_gpu": pytest.approx(29.5 / 800),
                   "sources": [], "largest": None}

    # a source whose block types the plan prices at 0 has no ratio; the others keep theirs
    free = P.report(P.Spot(flock, plan([c | {"prove_s": 0.0} if c["block"] in (GEMM32, half) else c for c in cells]),
                           serving).table(census), 1.0)
    its = free["totals"]["planned"]
    assert [(x["source"], x["planned_gpu_seconds_per_token"]) for x in its["sources"]] == [(P.STEADY, 20 / 800), (P.MODELED, 0.0)]
    text = page(free)
    assert "by this table's source: measured at steady state 2x (1 block types); `" in text
    assert "modeled from k_log" not in text[text.index("- where the two differ"):].split("\n")[0]
