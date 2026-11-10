"""`verity bench prove` with no GPU and no node: the rules on small tables (a census and its cut, each source of a
block type's proving time, the totals and their range, SPOT over a sampling plan), then the store side on a real store with
a directory remote: a workload's records as `fixtures/prove/` holds them (each cut to the fields the command reads;
`records.json` names what each was cut from), the hand computation they reproduce, every refusal by name, and the
command.

`fixtures/prove/` is the reader's input, not a result: the store side parses the bytes the producers wrote (a block
census, the cards, a sampling plan, the proof servers' reports), so its fixtures are those bytes, never regenerated
here, and synthetic ones would test a format the producers don't write. The figures pinned below hold this code's
arithmetic to a computation done independently of it on the same records; they publish no measurement, and what the
command says of a workload is what it prints from the store's records of record."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

from verity_bench import prove as P
from verity_bench import prove_records as R
from verity_bench.cli import main

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "prove"
W = R.WORKLOADS["glm53"]

# ---- the rules, on small tables ------------------------------------------------------------------------------------------------

#: one request of 4 tokens: two block types cut from a GEMM row, one from a norm row, and one no step ran
CENSUS = {
    "tokens": {"prefill": 3, "decode": 1}, "per_request": {"circuit_macs": 4000, "macs": 3600},
    "definitions": {
        'Gemm_v1{K=64,N=1,DOT={"fn":"Dot_v1"}}': {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 6, "decode": 2},
        "Gemm_v1{K=32,N=1}": {"k_log": 19, "of": {"Linear_v1": "row"}, "prefill": 3, "decode": 1},
        'Norm_v1{N=8,EPS={"f64":"0x1.0p-3"},FMA=true,TREE=[0,[1,2]]}': {"k_log": 22, "of": {"Norm_v1": "row", "Add_v1": "part"},
                                                                    "prefill": 1, "decode": 1},
        "Idle_v1{N=1}": {"k_log": 18, "of": {"Idle_v1": "row"}, "prefill": 0, "decode": 0},
    },
}
GEMM64, GEMM32, NORM = (name for name in CENSUS["definitions"] if not name.startswith("Idle"))
CUT = {"name": "split", "label": "a toy cut", "replaces": ["Add_v1, Norm_v1"],
       "strata": [{"block": "Half_v1{N=4}", "kind": "Half_v1", "per_token": 1.0, "k_log": 21},
                  {"block": "Bare_v1{N=4}", "kind": "Bare_v1", "per_token": 0.25},
                  {"block": "Unused_v1{N=4}", "kind": "Unused_v1", "per_token": 0, "k_log": 21}]}
#: single sessions: the K=64 GEMM's own, and another Definition's, both at 2 s a statement's fixed cost
SESSIONS = (P.Session(GEMM64, 20, 0.5, 2.0, 1 << 15), P.Session("Other_v1{N=2}", 22, 2.0, 2.0, 1 << 13))


def steady(type_: str, definition: str, run: str, blocks: int, gpu: float | None, *, full: int = 8, k_log: int = 20,
           verify: float | None = None, refused: str | None = None) -> P.Steady:
    return P.Steady(type_, definition, k_log, blocks, full, run, f"out/{type_}/server-k1", gpu, verify, refused)


def test_a_census_counts_a_request_over_its_tokens_and_a_cut_replaces_its_kinds():
    census = P.Census.of(CENSUS)
    assert [(b.name, b.kind, b.per_token, b.k_log) for b in census.blocks] == [
        (GEMM64, "Linear_v1", 2.0, 20), (GEMM32, "Linear_v1", 1.0, 19), (NORM, "Add_v1, Norm_v1", 0.5, 22)]
    assert (census.served_macs_per_token, census.block_macs_per_token, census.macs_in_blocks, census.cut) == (1000, 900, 0.9, None)

    recut = census.recut(CUT)
    assert [b.name for b in recut.blocks] == [GEMM64, GEMM32, "Half_v1{N=4}", "Bare_v1{N=4}"]
    assert recut.cut == "split" and recut.blocks[-1].k_log is None and recut.macs_in_blocks == 0.9
    with pytest.raises(ValueError, match="replaces Router_v1, which the census doesn't hold"):
        census.recut(CUT | {"replaces": ["Router_v1"]})


def test_a_census_name_is_spelled_as_the_definition_id_a_staged_statement_records():
    assert P.definition_id(GEMM64) == "Gemm_v1{K=64,N=1,DOT=Dot_v1}"
    assert P.definition_id(NORM) == "Norm_v1{N=8,EPS=0.125,FMA=True,TREE=(0,(1,2))}"
    assert P.definition_id('Read_v1{READ={"fn":"Row_v1{V=2,H=3}"},SWAP=false}') == "Read_v1{READ=Row_v1{V=2,H=3},SWAP=False}"


def test_a_block_type_takes_the_best_source_that_holds_it():
    """Steady state, then the card's single session (its own block's, else its Definition and K's scaled), then the k_log
    model, then nothing; and a steady-state attempt that didn't stand is named on the row it would have priced."""

    census = P.Census.of(CENSUS).recut(CUT)
    reports = (
        steady("gemm64", P.definition_id(GEMM64), "r20260101-000000-0001", 8, 16.0, verify=4.0),
        steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),       # the newest full one
        steady("gemm64", P.definition_id(GEMM64), "r20260103-000000-0003", 4, 6.0, verify=2.0),       # newer, but half full
        steady("gemm64", P.definition_id(GEMM64), "r20260104-000000-0004", 8, None, refused="timed out"),
        steady("half", "Half_v1{N=4}", "r20260101-000000-0001", 8, 80.0, refused="timed out"),
        steady("half", "Half_v1{N=4}", "r20260103-000000-0003", 4, 80.0, refused="stopped at the host's memory limit"),
    )
    table = P.Flock(reports, SESSIONS, "art:card").table(census)
    by = {r.block: r for r in table.rows}
    assert table.system == "flock" and list(by) == [b.name for b in census.blocks]

    gemm = by[GEMM64].cost
    assert (gemm.source, gemm.seconds, gemm.record, gemm.staged) == (P.STEADY, 1.0, "r20260102-000000-0002", "gemm64")
    assert gemm.verifier_seconds == 0.5 and gemm.at_steady_rate is None and "full statements" in gemm.note

    # the card measures the K=64 GEMM alone: K=32 is another type, so it and the cut's block are modeled from k_log, at
    # the card's types' median seconds a witness bit and their fixed seconds' line in k_log (flat here, at 2 s)
    rate = (0.5 / 2**20 + 2.0 / 2**22) / 2
    assert by[GEMM32].cost.source == P.MODELED and by[GEMM32].cost.record == "art:card"
    assert by[GEMM32].cost.seconds == pytest.approx(rate * 2**19 + 2.0 / 2**16)
    half = by["Half_v1{N=4}"].cost
    assert half.source == P.MODELED and half.seconds == pytest.approx(rate * 2**21 + 2.0 / 2**14)
    assert half.staged == "half" and "tried as `half` in r20260103-000000-0003" in half.note and "memory limit" in half.note
    assert by["Bare_v1{N=4}"].cost == P.Cost(P.MISSING)

    # the range's other end: every type off the steady state at the full-statement steady types' median rate
    assert by[GEMM32].cost.at_steady_rate == 1.0 / 2**20 * 2**19
    assert "9.54e-07 GPU-seconds a witness bit" in table.range_basis
    total = 2.0 * 1.0 + 1.0 * by[GEMM32].cost.seconds + 1.0 * half.seconds
    assert table.gpu_seconds_per_token() == pytest.approx(total)
    assert table.gpu_seconds_range() == pytest.approx((total, 2.0 + 0.5 + 2.0))

    # with no steady-state report the K=64 GEMM takes its own single session, at a full statement's share of the fixed
    # seconds, and no row has another end
    alone = {r.block: r.cost for r in P.Flock((), SESSIONS, "art:card").table(census).rows}
    assert (alone[GEMM64].source, alone[GEMM64].seconds) == (P.SINGLE, 0.5 + 2.0 / (1 << 15))
    assert all(c.at_steady_rate is None for c in alone.values())
    # with no card either, nothing is priced
    assert {r.cost.source for r in P.Flock().table(census).rows} == {P.MISSING}


def test_a_single_session_prices_its_definition_and_k_at_another_size():
    """A block type the card doesn't measure by name takes its Definition and K's measurement at 2^Δk the seconds and
    2^-Δk the statement, Δk between the two k_logs as the census holds them."""

    doc = {"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 1, "macs": 1}, "definitions": {
        "Gemm_v1{K=64,N=1}": {"k_log": 21, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=4}": {"k_log": 23, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0},
        "Gemm_v1{K=64,N=0}": {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0}}}
    # the card staged N=1 at k_log 20; the census holds it at 21, which is what Δk is taken from
    card = (P.Session("Gemm_v1{K=64,N=1}", 20, 0.5, 2.0, 1 << 15),)
    cost = {r.block: r.cost for r in P.Flock((), card, "art:card").table(P.Census.of(doc)).rows}
    assert cost["Gemm_v1{K=64,N=1}"].seconds == 0.5 + 2.0 / (1 << 15)
    assert cost["Gemm_v1{K=64,N=4}"].seconds == 0.5 * 4 + 2.0 / (1 << 13) and "at 2^2 its seconds" in cost["Gemm_v1{K=64,N=4}"].note
    assert cost["Gemm_v1{K=64,N=0}"].seconds == 0.5 / 2 + 2.0 / (1 << 16)
    assert {c.source for c in cost.values()} == {P.SINGLE}


def test_the_report_gives_each_sources_share_and_the_totals_with_their_range():
    census = P.Census.of(CENSUS).recut(CUT)
    table = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0, verify=4.0),), SESSIONS,
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
    measured = P.report(P.Flock(unverified, SESSIONS, "art:card").table(whole), 1.0)["totals"]
    assert measured["status"] == "measured" and measured["verifier_seconds_per_token"] is None and measured["missing"] == []
    modeled = P.report(P.Flock(unverified, SESSIONS, "art:card").table(P.Census.of(CENSUS)), 1.0)["totals"]
    assert modeled["status"] == "modeled" and modeled["missing"] == [] and modeled["steady_time_share"] < 1.0


def test_a_total_that_leaves_a_block_type_out_is_partial_and_names_it(monkeypatch):
    """One steady-state type and one type with no figure at three quarters of the blocks: steady-state measurements
    cover all of the time the total prices, and the total is still partial, never measured. It names the type it leaves
    out with its share of the blocks, and of the MACs where the census holds them."""

    doc = {"tokens": {"prefill": 1, "decode": 0}, "per_request": {"circuit_macs": 100, "macs": 80}, "definitions": {
        GEMM64: {"k_log": 20, "of": {"Linear_v1": "row"}, "prefill": 1, "decode": 0, "macs": 60},
        "Unpriced_v1{N=1}": {"of": {"Unpriced_v1": "row"}, "prefill": 3, "decode": 0, "macs": 5},
        "Rare_v1{N=1}": {"of": {"Rare_v1": "row"}, "prefill": 1, "decode": 0, "macs": 5}}}
    reports = (steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),)
    out = P.report(P.Flock(reports, SESSIONS, "art:card").table(P.Census.of(doc)), 1.0)
    t = out["totals"]
    assert {s["source"]: s["block_types"] for s in out["sources"]} == {P.STEADY: 1, P.SINGLE: 0, P.MODELED: 0, P.MISSING: 2}
    assert (t["steady_time_share"], t["gpu_seconds_per_token"], t["status"]) == (1.0, 1.0, "partial")
    assert t["missing"] == [
        {"block": "Unpriced_v1{N=1}", "per_token": 3.0, "block_share": 0.6, "mac_share": 0.1875},
        {"block": "Rare_v1{N=1}", "per_token": 1.0, "block_share": 0.2, "mac_share": 0.0625}]

    # a census that holds no MACs for one of its types gives the share of the blocks alone
    del doc["definitions"][GEMM64]["macs"]
    bare = P.report(P.Flock(reports, SESSIONS, "art:card").table(P.Census.of(doc)), 1.0)["totals"]
    assert bare["status"] == "partial" and [(m["block_share"], m["mac_share"]) for m in bare["missing"]] == [(0.6, None), (0.2, None)]
    # at a sampling plan the shares are of the blocks the plan samples
    cells = [{"block": name, "type": "", "taken": 1.0} for name in doc["definitions"]]
    sampled = P.report(P.Spot(P.Flock(reports, SESSIONS, "art:card"), plan(cells), P.Serving(0.5, 400.0, "stated")).table(
        P.Census.of(doc)), 1.0)["totals"]
    assert sampled["status"] == "partial" and [m["block_share"] for m in sampled["missing"]] == pytest.approx([1 / 3, 1 / 3])

    page = {"workload": "toy", "model": "Toy", "inputs": [], "refused": [], "macs_in_blocks": 0.8, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert "- prover GPUs per serving GPU: 1, partial: steady-state measurements cover 100.0% of the proving time it prices" in text
    assert ("- partial: 2 block type(s) with no figure, whose proving every total here leaves out: `Unpriced_v1{N=1}` "
            "(60.00% of the blocks, 18.75% of the blocks' MACs); `Rare_v1{N=1}` (20.00% of the blocks, 6.25% of the "
            "blocks' MACs)\n") in text
    assert "not measured at that size, and partial as that ratio is:" in text
    monkeypatch.setattr(R, "MISSING_NAMED", 1)
    assert "(60.00% of the blocks, 18.75% of the blocks' MACs); and 1 more, which `--json` names\n" in R.render(page)
    whole = R.render(page | {"totals": t | {"status": "measured", "missing": []}})
    assert "measured: steady-state measurements cover 100.0% of its proving time" in whole and "partial" not in whole


def plan(cells: list[dict], **over) -> P.SamplingPlan:
    return P.SamplingPlan.of({"case": "toy", "scope": "world year", "period_s": 100.0, "q": 0.01, "gpu_seconds": 800.0,
                              "replay_sample": 10.0, "cells": cells} | over, "art:plan")


def test_spot_prices_the_blocks_its_plan_samples_and_refuses_another_census_plan():
    census = P.Census.of(CENSUS)
    flock = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
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


def test_a_plan_that_states_its_own_prices_gets_both_totals_and_where_they_differ():
    """A plan's sample fits its planner's prices, and the table prices the same blocks from its own sources. Where the
    plan states its prices the report gives the total at each, each source's proving time at both and the block type
    that holds most of the difference; the table's own total doesn't move. A plan that prices none of its cells has
    the one total, and one that prices only some is refused."""

    census = P.Census.of(CENSUS)
    flock = P.Flock((steady("gemm64", P.definition_id(GEMM64), "r20260102-000000-0002", 8, 8.0),), SESSIONS, "art:card")
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

    page = {"workload": "toy", "model": "Toy", "inputs": [], "refused": [], "macs_in_blocks": 0.9, **out,
            "serving": {"gpu_seconds_per_token": 1.0, "status": "modeled", "basis": "stated"},
            "periods": {"gates_basis": "stated", "only": None, "rows": []}}
    text = R.render(page)
    assert "| ms per block | ms per block in the plan | source |" in text
    assert "| 1,000 | 500 | measured at steady state |" in text
    assert "- the same blocks at the sampling plan's own prices: 0.04 prover GPUs per serving GPU. Its planner sized" in text
    assert ("- where the two differ, this table's proving time over the plan's by this table's source: measured at steady "
            f"state 2x (1 block types), modeled from k_log {(20 * own[1] + 8 * own[2]) / 12:.3g}x (2 block types); `") in text
    assert f"alone is {gaps[top] / sum(gaps):.1%} of the difference" in text
    assert text.count(" --- |") == 9 + 6 + 6                              # the rows' table has the plan's column

    bare = [{k: v for k, v in c.items() if k != "prove_s"} for c in cells]
    one = P.report(P.Spot(flock, plan(bare), serving).table(census), 1.0)
    assert one["totals"]["planned"] is None and one["totals"]["gpus_per_serving_gpu"] == t["gpus_per_serving_gpu"]
    assert "in the plan" not in (plain := R.render(page | one)) and "own prices" not in plain
    assert P.report(flock.table(census), 1.0)["totals"]["planned"] is None
    with pytest.raises(ValueError, match=r"art:plan states its planner's seconds \(`prove_s`\) for 2 cell\(s\) and not for 1 "
                                         r"\(Norm_v1.*prices every cell or none"):
        plan([*cells[:2], bare[2]])


@pytest.mark.parametrize(("change", "why"), [
    ({}, None),
    ({"host_limit": {"stopped_at_s": 310.3}, "rc": [-15], "accepted": False}, "stopped at the host's memory limit"),
    ({"timed_out": True, "rc": [-15], "accepted": False}, "timed out"),
    ({"rc": [0, 1]}, "its provers exited [0, 1]"),
    ({"rc": []}, "its provers exited []"),
    ({"accepted": False}, "not every session was accepted"),
    ({"flat": False}, "its statement cache's hit rate wasn't flat"),
])
def test_a_report_is_a_steady_state_measurement_only_when_accepted_and_flat(change, why):
    report = {"host_limit": {"stopped_at_s": None}, "timed_out": False, "rc": [0], "accepted": True, "flat": True} | change
    assert R.report_why(report) == why


# ---- the store side ------------------------------------------------------------------------------------------------------------


class Store:
    """A machine holding a workload's records: a local store with a directory remote, each art and run on it."""

    def __init__(self, tmp: Path):
        from research.store.local import LocalStore

        self.tmp, self.remote = tmp, tmp / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp / "producer", config_path=self.config)
        self._ts = 0

    def fresh(self, name: str = "fresh"):
        from research.store.local import LocalStore

        return LocalStore(self.tmp / name, config_path=self.config)

    def art(self, kind: str, meta: dict, *, file: Path | None = None, tree: Path | None = None) -> str:
        art = self.store.put_artifact(kind, meta, file=file, tree=tree)
        self.store.push(art)
        return art

    def _scratch(self) -> Path:
        self._ts += 1
        path = self.tmp / "docs" / str(self._ts)
        path.mkdir(parents=True)
        return path

    def doc(self, kind: str, name: str, doc: dict, meta: dict | None = None) -> str:
        """A file art holding `doc`."""

        path = self._scratch() / name
        path.write_text(json.dumps(doc))
        return self.art(kind, meta or {}, file=path)

    def tree(self, kind: str, name: str, doc: dict, meta: dict | None = None) -> str:
        """A tree art holding `doc` as its file `name`."""

        root = self._scratch()
        (root / name).write_text(json.dumps(doc))
        return self.art(kind, meta or {}, tree=root)

    def ran(self, run: str, files: Path | None, *, rc: int = 0) -> None:
        """A run's attempt on the remote, its files a run record (`files`: None for a run that kept none)."""

        from research.store import canon

        record = self.art("run-record/v1", {}, tree=files) if files else None
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "bash", "version": None},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T00:00:00Z"}, "cost": None,
            "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": None},
            "outputs": {"result": None, "run_record": record}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))

    def label(self, target: str, key: str, value: str) -> None:
        self._ts += 1
        self.store.add_label(target, key, value, by="benchmarks", ts=f"2026-01-01T00:{self._ts // 60:02d}:{self._ts % 60:02d}Z")
        self.store.push_labels([target])

    def record(self, target: str, slot: str) -> str:
        self.label(target, R.RUN_OF_RECORD, W.record(slot))
        return target


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


RUNS = fixture("records.json")[R.STEADY]


def run_files(tmp: Path, run: str) -> Path:
    """A fixture run's files as its run record holds them: its jobs' reports under `out/`."""

    files = tmp / "run-files" / run
    if not files.exists():
        shutil.copytree(FIXTURES / "steady" / run, files / "out")
    return files


def census_art(s: Store, **meta) -> str:
    return s.tree(R.CENSUS_KIND, R.CENSUS_FILE, fixture("census.json"), {"model": W.model, "run": "r20260101-000000-0000"} | meta)


def plan_art(s: Store, **change) -> str:
    return s.tree("evidence/v1", R.PLAN_FILE, fixture("mix.json") | change)


def recorded(tmp: Path, **plan_change) -> tuple[Store, dict[str, str]]:
    """The fixtures' records in a store, each labelled its slot's record: the census, its cut, the single-session card,
    the three proof server runs and the sampling plan (with `plan_change` over its fields)."""

    s = Store(tmp)
    arts = {R.CENSUS: s.record(census_art(s), R.CENSUS),
            R.CUT: s.record(s.doc(R.CARD_SCHEMA, "partition.json", fixture("partition.json")), R.CUT),
            R.SINGLE: s.record(s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json")), R.SINGLE),
            R.SAMPLING: s.record(plan_art(s, **plan_change), R.SAMPLING)}
    for run in RUNS:
        s.ran(run, run_files(tmp, run), rc=0 if run == RUNS[-1] else 1)
        s.record(run, R.STEADY)
    return s, arts


def test_the_hand_computation_is_reproduced_from_the_records_it_read(tmp_path):
    """The figures a hand computation gave on these records, to the last digit it kept: with every block proved, the
    blocks and the prover GPU-seconds a token, the tokens a prover GPU proves a second and each source's share; at the
    sampling plan, the served tokens a prover GPU covers a second and the prover GPUs a serving GPU, with the range's
    other end. A fresh machine gets them from the store's remote alone."""

    s, arts = recorded(tmp_path)
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    t = doc["totals"]
    assert doc["cut"] == "per-expert-fused" and len(doc["rows"]) == 150 and doc["refused"] == []
    assert doc["macs_in_blocks"] == pytest.approx(0.98755, abs=5e-6)
    assert t["blocks_per_token"] == pytest.approx(19415356.404181186, rel=1e-12)
    assert t["gpu_seconds_per_token"] == pytest.approx(88303.08784095926, rel=1e-12)
    assert t["tokens_per_gpu_second"] == pytest.approx(1.1324632291467292e-05, rel=1e-12)
    assert t["gpu_seconds_per_token_range"][0] == pytest.approx(27643.734264603525, rel=1e-12)
    assert t["status"] == "modeled" and t["proof_bytes_per_token"] is None
    by = {x["source"]: x for x in doc["sources"]}
    assert [(k, x["block_types"]) for k, x in by.items()] == [(P.STEADY, 8), (P.SINGLE, 86), (P.MODELED, 56), (P.MISSING, 0)]
    assert by[P.STEADY]["per_token"] == pytest.approx(10380036.348432057) and by[P.STEADY]["time_share"] == pytest.approx(
        0.23324002390678747)
    assert by[P.SINGLE]["per_token"] == pytest.approx(6132874.675958165) and by[P.SINGLE]["time_share"] == pytest.approx(
        0.49572246283915494)
    assert by[P.MODELED]["per_token"] == pytest.approx(2902445.3797909454) and by[P.MODELED]["time_share"] == pytest.approx(
        0.27103751325405523)
    assert doc["serving"]["gpu_seconds_per_token"] == pytest.approx(0.00012950804344665488, rel=1e-12)

    # each steady-state type is the newest run's, at the statement it finished; the two killed before any statement are
    # named on the rows the card prices
    rows = {r["block"]: r for r in doc["rows"]}
    steady_rows = [r for r in doc["rows"] if r["source"] == P.STEADY]
    assert {r["record"] for r in steady_rows} == {RUNS[-1]}
    assert sorted(r["note"].split(",")[0] for r in steady_rows) == [
        "`ar-multimem`", "`dsa-logits-v3`", "`fr-w13`", "`fr-w2`", "`residual-add`", "`rmsnorm-v8`", "`sm-fp8-k2048`",
        "`sm-fp8-k6144`"]
    assert sum("statements 128/256 full" in r["note"] for r in steady_rows) == 2
    top = doc["rows"][0]
    assert top["block"] == "FlashInferFp8Linear_v1{K=2048,N=1}" and top["source"] == P.SINGLE and top["record"] == arts[R.SINGLE]
    assert top["time_share"] == pytest.approx(0.4016, abs=5e-5) and 1e3 * top["seconds"] == pytest.approx(56.611, abs=5e-4)
    assert f"steady state tried as `fi-linear-k2048` in {RUNS[1]}" in top["note"] and "host's memory limit" in top["note"]
    assert "tried as `gq-k6144`" in rows["Fp8GroupQuantUe8m0_v1{K=6144,G=128}"]["note"]
    assert [i["targets"] for i in doc["inputs"]] == [[arts[R.CENSUS]], [arts[R.CUT]], RUNS[::-1], [arts[R.SINGLE]]]

    spot = R.build("spot", "glm53", sampling=arts[R.SAMPLING], store=s.fresh("second"), root=REPO)
    assert spot == R.build("spot", "glm53", store=s.fresh("third"), root=REPO)          # the plan of record is the default
    t = spot["totals"]
    assert spot["system"] == "spot[flock]" and spot["inputs"][-1]["targets"] == [arts[R.SAMPLING]]
    assert t["tokens_per_gpu_second"] == pytest.approx(14273, abs=0.5)
    assert t["gpus_per_serving_gpu"] == pytest.approx(0.5409739676190373, rel=1e-12)
    assert t["gpus_per_serving_gpu_range"] == pytest.approx([0.1860208609869626, 0.5409739676190373], rel=1e-12)
    assert t["tokens_per_gpu_second_range"][1] == pytest.approx(41509, abs=0.5)
    assert t["gpu_seconds_per_token"] == pytest.approx(7.006e-05, rel=1e-4)
    # the plan covers one compute and period; every block proved is the same ratio at any
    assert [(x["compute"], x["period"]) for x in spot["periods"]["rows"]] == [("world year", "year")]
    assert spot["periods"]["rows"][0]["prover_gpus"] == pytest.approx(0.5409739676190373 * 374960000.0)
    assert [(x["compute"].split()[0], x["period"]) for x in doc["periods"]["rows"]] == [
        (c, p) for c in ("datacenter", "world") for p in ("week", "month", "year")]
    world = doc["periods"]["rows"][-1]
    assert world["serving_gpus"] == pytest.approx(374960000.0, rel=1e-4)
    assert world["prover_gpu_seconds"] == pytest.approx(doc["totals"]["gpus_per_serving_gpu"] * world["serving_gpus"] * 31536000)

    text = R.render(doc, rows=3)
    assert "| `FlashInferFp8Linear_v1{K=2048,N=1}` | 24 | 626,474 | 56.6 | measured in a single session |" in text
    assert "| 147 more block types |" in text and "- proof bytes per token: not measured" in text
    assert "- served MACs inside a proof block: 98.8%; the rest are served rows no cut fits" in text
    assert "- prover GPU-seconds per token: 88,303 (range 27,644 to 88,303)" in text
    assert "the proof server's own verifier" in text and "Refused" not in text
    assert f"- {R.STEADY}: {', '.join(RUNS[::-1])} (`prove-glm53-flock-steady`): 8 staged types measured" in text
    assert "(modeled, a stated input, not a record)" in text and "| measured at steady state | 8 |" in text
    text = R.render(spot)
    assert "- prover GPUs per serving GPU: 0.541 (range 0.186 to 0.541), modeled" in text
    assert "- served tokens covered per second per prover GPU: 14,273 (range 14,273 to 41,509)" in text
    assert "| world year | year |" in text and "no plan of record covers another period, or a datacenter." in text
    assert text.count("\n| `") == 150


def test_the_plan_of_records_own_prices_are_read_from_its_record(tmp_path):
    """A plan of record whose cells state their planner's seconds: the table's total is the one its sources give, and
    the plan's own is beside it, on every row and in the totals."""

    mix = fixture("mix.json")
    s, _ = recorded(tmp_path, cells=[c | {"prove_s": 0.002 * c["taken"]} for c in mix["cells"]])
    spot = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    t = spot["totals"]
    assert t["gpus_per_serving_gpu"] == pytest.approx(0.5409739676190373, rel=1e-12)
    blocks = mix["replay_sample"] * sum(c["taken"] for c in mix["cells"])
    assert t["planned"]["gpus_per_serving_gpu"] == pytest.approx(0.002 * blocks / mix["gpu_seconds"], rel=1e-9)
    assert sum(x["block_types"] for x in t["planned"]["sources"]) == 150
    assert [r["planned_seconds"] for r in spot["rows"]] == pytest.approx([0.002] * 150)
    text = R.render(spot, rows=2)
    assert "- prover GPUs per serving GPU: 0.541 (range 0.186 to 0.541), modeled" in text
    assert f"- the same blocks at the sampling plan's own prices: {R._n(t['planned']['gpus_per_serving_gpu'])} prover GPUs" in text
    # the plan's column is on every line of the rows' table, the rest's row included
    table = text[text.index("| block type |"):].split("\n\n")[0].splitlines()
    assert table[0].count("|") == 10 and "| ms per block in the plan |" in table[0] and len(table) == 2 + 2 + 1
    assert table[-1].startswith("| 148 more block types | | ") and {line.count("|") for line in table} == {10}


def test_an_unlabelled_or_superseded_input_is_refused_by_name(tmp_path):
    s = Store(tmp_path)
    census = census_art(s)
    cut = s.doc(R.CARD_SCHEMA, "partition.json", fixture("partition.json"))
    single = s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json"))
    mix = plan_art(s)

    def build(system: str = "flock", **kw) -> dict:
        return R.build(system, "glm53", store=s.store, root=REPO, **kw)

    # nothing is labelled: each slot in turn is named with the label that would fix it
    for target, slot, what in ((census, R.CENSUS, "block census"), (cut, R.CUT, "cut"), (single, R.SINGLE, "single-session card")):
        with pytest.raises(ValueError, match=f"no {what} of record: nothing in the store carries "
                                             f"`run_of_record={W.record(slot)}`.*research data label TARGET"):
            build()
        s.record(target, slot)
    doc = build()
    assert doc["inputs"][2] == {"slot": R.STEADY, "record": W.record(R.STEADY), "targets": [],
                                "what": "none of record: no block type is measured at steady state"}
    assert {x["source"]: x["block_types"] for x in doc["sources"]}[P.STEADY] == 0
    assert doc["totals"]["range_basis"] is None and "(range" not in R.render(doc, rows=1)

    # SPOT's plan: none of record, one given that isn't labelled, and --sampling without spot
    with pytest.raises(ValueError, match=f"no sampling plan of record.*`run_of_record={W.record(R.SAMPLING)}`"):
        build("spot")
    with pytest.raises(ValueError, match=f"{mix} isn't a sampling plan of record: it doesn't carry"):
        build("spot", sampling=mix)
    with pytest.raises(ValueError, match="--sampling is SPOT's"):
        build(sampling=mix)
    s.record(mix, R.SAMPLING)
    # the plan names the two fused block types by their staged types, which only a steady-state report ties to a Definition
    with pytest.raises(ValueError, match=rf"the sampling plan {mix} and the census don't cover each other: the plan has no "
                                         r"cell for 2 block type\(s\) \(MoeExpertFp8ReadGemm_v1.*no block type for 2 "
                                         r"cell\(s\) \(Fused\[.*or no steady-state report of record names"):
        build("spot")
    s.ran(RUNS[-1], run_files(tmp_path, RUNS[-1]))
    s.record(RUNS[-1], R.STEADY)
    other = s.record(plan_art(s, case="other"), R.SAMPLING)
    with pytest.raises(ValueError, match=f"2 targets carry `run_of_record={W.record(R.SAMPLING)}` and none is superseded: "
                                         f"{mix}, {other}"):
        build("spot")
    assert build("spot", sampling=other)["inputs"][-1]["what"].startswith("`other`")     # naming one of them chooses it
    s.label(mix, R.SUPERSEDED_BY, other)
    assert build("spot")["inputs"][-1]["targets"] == [other]
    with pytest.raises(ValueError, match=f"the sampling plan of record is superseded: {mix} by {other}"):
        build("spot", sampling=mix)

    # a record of the wrong kind, model or slot
    wrong = s.record(census_art(s, model="Llama-3.1-8B"), R.CENSUS)
    s.label(census, R.SUPERSEDED_BY, wrong)
    with pytest.raises(ValueError, match=f"{wrong} isn't a block census of GLM-5.3 .*: it is a block-census/v1 of Llama-3.1-8B"):
        build()
    s.label(wrong, R.SUPERSEDED_BY, census)
    with pytest.raises(ValueError, match=f"the block census of record is superseded: {census} by {wrong}, {wrong} by {census}"):
        build()
    s.record(census_art(s, run="r20260102-000000-0000"), R.CENSUS)
    misfiled = s.record(s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json"), {"filed": "cut"}), R.CUT)
    s.label(cut, R.SUPERSEDED_BY, misfiled)
    with pytest.raises(ValueError, match=f"{misfiled} isn't a `partition` card of GLM-5.3 .*'slot': 'proof_overhead'"):
        build()


def test_a_steady_state_run_that_cant_be_read_is_left_out_and_listed(tmp_path):
    s, _ = recorded(tmp_path)
    newest, older, oldest = RUNS[::-1]
    empty = tmp_path / "empty-run"
    (empty / "out").mkdir(parents=True)
    (empty / "out" / "job.log").write_text("no report")
    s.ran("r20261109-000000-0001", empty)
    s.ran("r20261109-000000-0002", None)
    for run in ("r20261109-000000-0001", "r20261109-000000-0002", "r20261109-000000-0003"):
        s.record(run, R.STEADY)
    s.label(newest, R.SUPERSEDED_BY, "r20261109-000000-0001")
    doc = R.build("flock", "glm53", store=s.fresh(), root=REPO)
    assert {x["target"]: x["why"] for x in doc["refused"]} == {
        "r20261109-000000-0003": "not in the store or on its remote", "r20261109-000000-0002": "kept no run files",
        "r20261109-000000-0001": "holds no proof server report (`*/server-k*/server.json` with its `types.json`)",
        newest: "superseded by r20261109-000000-0001"}
    assert doc["inputs"][2]["targets"] == [older, oldest]
    # without the newest run each type takes the fullest statement the others measured: the six light types' full
    # statements in the oldest run, not the quarter ones after it, and the two heavy types' half statements
    steady_rows = [r for r in doc["rows"] if r["source"] == P.STEADY]
    assert sorted((r["record"], "full statements" in r["note"]) for r in steady_rows) == [(oldest, True)] * 6 + [(older, False)] * 2
    text = R.render(doc, rows=0)
    assert "Refused (not read):" in text and f"- {newest} (`prove-glm53-flock-steady`): superseded by" in text
    assert "| 150 more block types |" in text

    # an art that names its runs stands for them
    s2 = Store(tmp_path / "second")
    for slot, name in ((R.CUT, "partition.json"), (R.SINGLE, "proof_overhead.json")):
        s2.record(s2.doc(R.CARD_SCHEMA, name, fixture(name)), slot)
    s2.record(census_art(s2), R.CENSUS)
    s2.ran(newest, run_files(tmp_path, newest))
    s2.record(s2.doc("evidence/v1", "steady.json", {}, {"runs": [newest]}), R.STEADY)
    doc = R.build("flock", "glm53", store=s2.fresh(), root=REPO)
    assert doc["inputs"][2]["targets"] == [newest] and doc["totals"]["gpu_seconds_per_token"] == pytest.approx(88303.08784095926)


# ---- the command ---------------------------------------------------------------------------------------------------------------


def test_verity_bench_prove_is_the_commands_runner(tmp_path, monkeypatch, capsys):
    """`verity bench prove` runs benchmarks/prove/run.py, which is `verity-bench prove`: the table from the store alone."""

    s, arts = recorded(tmp_path)
    shutil.rmtree(tmp_path / "producer")
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))

    def run(*argv: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(REPO / "benchmarks" / "prove" / "run.py"), run_name="__main__")
        out = capsys.readouterr()
        return done.value.code, out.out, out.err

    code, out, _ = run("--system", "flock", "--workload", "glm53", "--rows", "2")
    assert code == 0 and out.startswith("# verity bench prove: flock on glm53 (GLM-5.3)\n")
    assert "## Every block proved" in out and "| 148 more block types |" in out
    code, out, _ = run("--system", "spot", "--workload", "glm53", "--sampling", arts[R.SAMPLING], "--json")
    doc = json.loads(out)
    assert code == 0 and doc["schema"] == P.SCHEMA and doc["totals"]["gpus_per_serving_gpu"] == pytest.approx(0.5409739676190373)
    code, _, err = run("--system", "spot", "--workload", "glm53", "--sampling", arts[R.CENSUS], "--local-only")
    assert code == 2 and f"{arts[R.CENSUS]} isn't a sampling plan of record" in err
    with pytest.raises(SystemExit):
        main(["prove", "--system", "sp1", "--workload", "glm53"])
    assert "invalid choice: 'sp1'" in capsys.readouterr().err
