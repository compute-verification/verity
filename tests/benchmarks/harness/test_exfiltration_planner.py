"""`verity bench exfiltration`'s planner: the rule that reads a scope's rows off at a share of total compute and the
rule that reads how its proving is priced from the row's plan, on small rows; then the figures a hand computation gave
on the fixture's rows."""

from __future__ import annotations

import pytest
from exfiltration_helpers import (
    CUT_ROW,
    PLAN,
    PLANNED,
    REPO,
    ROWS,
    YEAR_GB,
    Store,
    as_row,
    mine,
    plan,
    record,
    recorded,
    row,
    rows,
    term,
)
from verity_bench import exfiltration as E

#: the total is 0.02 + 2β and the bound 1/β, so at 10% β is 0.04 and the bound 25 bits, counted twice at λ = 2; the
#: rows are priced over 3 / 0.5 = 6 requests, where their `requests` field says 7
TOY = [row(0.02, 0.06, 50.0), row(0.03, 0.08, 100 / 3), row(0.05, 0.12, 20.0)]


def test_a_scope_is_read_at_the_budget_where_its_total_is_the_share():
    at = E.at_share("world year", "r1", TOY[::-1], 0.10)
    assert at["beta"] == pytest.approx(0.04) and at["inside"] and at["residual"] < 1e-15
    assert at["bound"]["bits"] == pytest.approx(25.0) and at["lambda"] == 2.0 and at["bits"] == pytest.approx(50.0)
    assert at["GB"] == pytest.approx(50.0 / 8e9) and (at["compute"], at["period"]) == ("world", "year")
    assert at["parts"] == pytest.approx({"serve": 0.01, "prove": 0.09})
    assert (at["delta"], at["output_bits"], at["gpu_seconds"], at["weights_bits"]) == (0.01, 8.0, 4.0, 100.0)
    assert at["pinned"] == [0.02, 0.03, 0.05] and at["priced"] is None
    # the requests are the count the rows are priced over, not the field that names another request's
    assert at["requests"] == 6.0
    # at a pinned budget it is that row's bound, and past the last the line through the nearest two, said to be so
    assert E.at_share("world year", "r1", TOY, 0.08)["bound"]["bits"] == pytest.approx(100 / 3)
    past = E.at_share("world year", "r1", TOY, 0.14)
    assert past["beta"] == pytest.approx(0.06) and not past["inside"] and past["bound"]["bits"] == pytest.approx(100 / 6)

    with pytest.raises(ValueError, match="r1's world year rows pin 1 audit budgets, 1 distinct"):
        E.at_share("world year", "r1", TOY[:1], 0.10)
    with pytest.raises(ValueError, match="pin 3 audit budgets, 2 distinct"):
        E.at_share("world year", "r1", [*TOY[:2], TOY[0]], 0.10)
    with pytest.raises(ValueError, match="rows differ in delta, gpu_seconds, which one plan holds fixed"):
        E.at_share("world year", "r1", [*TOY[:2], TOY[2] | {"delta_seen": 0.02, "gpu_seconds": 5.0}], 0.10)
    with pytest.raises(ValueError, match="rows differ in requests, which one plan holds fixed"):
        E.at_share("world year", "r1", [*TOY[:2], TOY[2] | {"q": 0.25}], 0.10)
    with pytest.raises(ValueError, match="rows don't say what requests they are priced over \\(`replay_sample / q`\\)"):
        E.at_share("world year", "r1", [{k: v for k, v in r.items() if k != "q"} for r in TOY], 0.10)
    with pytest.raises(ValueError, match="r1's world year rows are 2 variants of the row \\(h1lin, h8lin\\), each on its own "
                                         "premises: a run of record holds one"):
        E.at_share("world year", "r1", [r | {"h8": h} for r in TOY for h in ("h8lin", "h1lin")], 0.10)
    with pytest.raises(ValueError, match="spend 2.00% of the compute before any audit: no audit budget brings the total to 1.00%"):
        E.at_share("world year", "r1", TOY, 0.01)


def test_proving_time_is_priced_by_the_rows_plan_not_by_the_rows_own_shares():
    """The shares are the plan's cells', summed by block family; how a family is priced is the rows' account, whose own
    shares, written before the row's levers multiply its blocks, are not read. A plan is held to its rows."""

    split = {"families": {"A_v1": {"priced": ["measured"], "share": 0.97}, "B_v1": {"priced": ["computed"], "share": 0.01},
                          "C_v1": {"priced": ["computed", "measured"], "share": 0.01},
                          "D_v1": {"priced": ["measured"], "share": 0.01}}}
    rows = [r | {"split": split} for r in TOY]
    plan = {"scope": "world year", "beta": 0.04, "q": 0.25, "replay_sample": 1.5, "bound_bits": 25.0,
            "cells": [{"block": "B_v1{K=1}", "share": 0.5}, {"block": "A_v1{K=1}", "share": 0.125},
                      {"block": "C_v1{K=2}", "share": 0.125}, {"block": "B_v1{K=2}", "share": 0.0625},
                      {"block": "D_v1", "share": 0.0625}, {"block": "E_v1{K=1}", "share": 0.0625}],
            "left_out": [{"block": "Copy[8 bits]", "share": 0.0625}]}
    priced = E.at_share("world year", "r1", rows, 0.10, ("r2", plan))["priced"]
    assert (priced["of"], priced["run"], priced["beta"]) == ("plan", "r2", 0.04)
    assert priced["ways"] == {
        "computed": {"share": 0.5625, "families": {"B_v1": 0.5625}},
        "measured": {"share": 0.1875, "families": {"A_v1": 0.125, "D_v1": 0.0625}},
        "measured and computed": {"share": 0.125, "families": {"C_v1": 0.125}},
        E.UNSTATED: {"share": 0.125, "families": {"E_v1": 0.0625, "Copy[8 bits]": 0.0625}}}
    assert sum(w["share"] for w in priced["ways"].values()) == 1.0
    assert E.at_share("world year", "r1", rows, 0.10)["priced"] is None

    with pytest.raises(ValueError, match="r2's plan of the world year is priced over 12 requests and r1's rows over 6: it is "
                                         "not these rows' plan"):
        E.at_share("world year", "r1", rows, 0.10, ("r2", plan | {"q": 0.125}))
    with pytest.raises(ValueError, match="r2's plan of the world year bounds 0.0 GB at β = 4.0000%, where r1's rows give "
                                         "0.0 GB: it is not these rows' plan"):
        E.at_share("world year", "r1", rows, 0.10, ("r2", plan | {"bound_bits": 50.0}))
    assert E.at_share("world year", "r1", rows, 0.10, ("r2", plan | {"bound_bits": 25.0 * (1 + 5e-4)}))["priced"]

    # rows whose `split` says it is at the row's q and sampled replays hold their own plans: the one at the pinned
    # budget nearest is read, shares and all; a plan of record is read before it; a `split` at another q is no plan
    own = [r | {"split": split | {"q": r["q"], "replay_sample": r["replay_sample"] * (1 + 1e-12)}} for r in TOY]
    priced = E.at_share("world year", "r1", own, 0.10)["priced"]
    assert (priced["of"], priced["run"], priced["beta"]) == ("rows", "r1", 0.03)
    assert {k: (w["share"], list(w["families"])) for k, w in priced["ways"].items()} == {
        "measured": (0.98, ["A_v1", "D_v1"]), "computed": (0.01, ["B_v1"]), "measured and computed": (0.01, ["C_v1"])}
    assert E.at_share("world year", "r1", own, 0.115)["priced"]["beta"] == 0.05
    assert E.at_share("world year", "r1", own, 0.10, ("r2", plan))["priced"]["of"] == "plan"
    other = [r | {"split": split | {"q": r["q"] / 2, "replay_sample": r["replay_sample"]}} for r in TOY]
    assert E.at_share("world year", "r1", other, 0.10)["priced"] is None


def test_a_family_priced_several_ways_is_split_by_its_ways_and_only_an_own_row_is_measured():
    """A family whose split gives its share by way (`ways`: a table's own row, the Lean verifier at the driver's
    modeled price, a family's row) has its plan's share split in their proportion, not printed under one joined way;
    the Inputs mark a block type's own steady-state row measured, and a family's row and the Lean verifier modeled."""

    split = {"families": {"A_v1": {"priced": ["modeled", "steady"], "share": 0.9, "ways": {"steady": 0.6, "modeled": 0.3}},
                          "B_v1": {"priced": ["family"], "share": 0.1, "ways": {"family": 0.1}}}}
    own = [r | {"split": split | {"q": r["q"], "replay_sample": r["replay_sample"]}} for r in TOY]
    plan = {"scope": "world year", "beta": 0.04, "q": 0.5, "replay_sample": 3.0, "bound_bits": 25.0,
            "cells": [{"block": "A_v1{K=1}", "share": 0.75}, {"block": "B_v1", "share": 0.25}]}
    for at in (E.at_share("world year", "r1", own, 0.10, ("r2", plan)), E.at_share("world year", "r1", own, 0.10)):
        ways = at["priced"]["ways"]
        assert set(ways) == {"steady", "modeled", "family"}
        assert sum(w["share"] for w in ways.values()) == pytest.approx(1.0, abs=1e-12)
    ways = E.at_share("world year", "r1", own, 0.10, ("r2", plan))["priced"]["ways"]
    assert {k: (round(w["share"], 12), w["families"]) for k, w in ways.items()} == {
        "steady": (0.5, {"A_v1": pytest.approx(0.5)}), "modeled": (0.25, {"A_v1": pytest.approx(0.25)}),
        "family": (0.25, {"B_v1": 0.25})}
    point = E.at_share("world year", "r1", own, 0.10, ("r2", plan))
    marks = {i["input"]: i["mark"] for i in E._inputs({"world year": point}, ["r1"], {"inputs": {}},
                                                       {"at_zero_in": [], "GB": None}, 0.10, 0.003, E.ROW,
                                                       E.served_share(None))}
    assert {k: v for k, v in marks.items() if k.startswith("proving priced")} == {
        "proving priced steady": "measured", "proving priced modeled": "modeled", "proving priced family": "modeled"}


def test_the_hand_computation_is_reproduced_from_the_rows_it_read(tmp_path):
    """The figures a hand computation gave on these rows, to the digits it kept: the world year's audit budget at 10%, its
    bound at λ = 1 and at the rows' λ, each part of the 10%, the figure with the network check's own proving taken off the
    budget, and the datacenter year's. A fresh machine gets them from the store's remote alone."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert (doc["schema"], doc["unit"], doc["status"], doc["runs"], doc["refused"]) == (E.SCHEMA, "GB", "modeled", [PLAN, ROWS], [])
    by = {p["scope"]: p for p in doc["compute_audit"]}
    assert list(by) == ["datacenter year", "world year"]
    year = by["world year"]
    assert doc["value"] == year["GB"] == pytest.approx(YEAR_GB, rel=1e-12)
    assert year["beta"] == pytest.approx(0.05542238712065398, rel=1e-12) and year["inside"] and year["residual"] < 1e-9
    assert year["bound"]["GB"] == pytest.approx(13561.74961510603, rel=1e-12) and year["lambda"] == pytest.approx(1.05)
    assert (year["run"], year["mix"], year["delta"], year["output_bits"]) == (ROWS, "B128", 0.01, 28000.0)
    assert year["step"].startswith(E.STEP) and year["pinned"] == [0.05, 0.055, 0.06, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0]
    assert {k: round(100 * v, 3) for k, v in year["parts"].items()} == {
        "serve": 4.339, "wait": 2.293, "prove": 5.372, "replay": 0.17, "pool": 0.07, "edge": 0.044, "compile": 0.005}
    assert sum(v for k, v in year["parts"].items() if k != "wait") == pytest.approx(0.10, abs=1e-5)
    other = by["datacenter year"]
    assert other["GB"] == pytest.approx(13707.861447630507, rel=1e-12) and other["run"] == ROWS and other["priced"] is None
    assert other["beta"] == pytest.approx(0.055420109073442365, rel=1e-12)
    net = doc["network_check"]
    assert net["share"] == 0.003 and net["GB"] == pytest.approx(15053.889111429447, rel=1e-12) and net["inside"]
    assert net["beta"] == pytest.approx(0.052425317582956925, rel=1e-12)
    assert E.build(share=0.097, network_check=0, store=recorded(tmp_path / "again").fresh(), root=REPO)["value"] == net["GB"]


def test_the_rows_requests_are_the_count_they_are_priced_over(tmp_path):
    """A row is priced over its sampled replays / the probability a request is sampled, which is the planner's count and
    the network attester's part's, and not the row's `requests` field. Rows priced over another count are refused: the
    fixture's variant of the row whose indexer is stated against fewer linears is."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    year = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    field = {r["requests"] for r in mine("world year")}
    assert field == {2.02900114160274e16} and year["requests"] == pytest.approx(1.7784008649596718e16, rel=1e-12)
    assert year["requests"] == pytest.approx(doc["periods"][-1]["requests"], rel=1e-12)
    assert year["requests"] == pytest.approx(term().plan()["world_requests"].value, rel=1e-12)
    assert "The row is priced over 1.778e+16 requests, the count the network attester's part is over." in E.render(doc)
    assert "\n- the part is λ times the bound where the row's total is 10% of compute" in E.render(doc)

    s = Store(tmp_path / "other")
    s.label(s.ran(ROWS, [as_row(r) for r in rows() if (r["row"], r["h8"]) == (CUT_ROW, "h8")]))
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 2.11575e\\+16 requests on "
                                         "the compute audit's side and 1.7784e\\+16 on the network attester's"):
        E.build(store=s.store, root=REPO)
    # the run as the planner's driver wrote it, several variants of the row in one scope, is refused by name, and
    # read at the variant named
    both = Store(tmp_path / "both")
    both.label(both.ran(ROWS, [as_row(r) for r in rows()]))
    with pytest.raises(ValueError, match=f"{ROWS}'s world year rows are 2 variants of the row \\(h8, h8lin\\), each on its own "
                                         "premises: a run of record holds one, or --variant names the one meant"):
        E.build(store=both.store, root=REPO)
    named = E.build(store=both.store, root=REPO, variant="h8lin")
    assert named["value"] == pytest.approx(YEAR_GB, rel=1e-12) and named["variant"] == "h8lin" and doc["variant"] is None
    assert "in its variant `h8lin`, read at the audit budget" in E.render(named)


def test_how_the_proving_is_priced_is_the_plans_account(tmp_path):
    """The world year's proving time by how its block families are priced, from the row's plan of record: the sparse
    attention's split part, priced by computation, is over half of it, where the rows' own shares, written before the
    row's levers multiply its blocks, put it under a thousandth and 98.6% in families priced from measured statements.
    Without a plan of record the command says none is read and marks no proving measured."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    priced = next(p for p in doc["compute_audit"] if p["scope"] == "world year")["priced"]
    said = mine("world year")[0]["split"]["families"]
    assert sum(f["share"] for f in said.values() if f["priced"] == ["measured"]) == pytest.approx(0.9857, abs=1e-4)
    assert said["SparseMlaSplitPart_v1"]["share"] < 1e-3 and said["SparseMlaSplitPart_v1"]["priced"] == ["computed"]
    assert (priced["run"], priced["beta"]) == (PLAN, plan()["beta"]) and priced["beta"] == pytest.approx(0.0554224, abs=1e-7)
    ways = priced["ways"]
    assert sum(w["share"] for w in ways.values()) == pytest.approx(1.0, abs=1e-9)
    assert {k: round(w["share"], 4) for k, w in ways.items()} == {
        "computed": 0.5236, "measured": 0.4713, "measured and computed": 0.0051, E.UNSTATED: 0.0}
    assert list(ways["computed"]["families"])[0] == "SparseMlaSplitPart_v1"
    assert ways["computed"]["families"]["SparseMlaSplitPart_v1"] == pytest.approx(0.5229, abs=5e-4)
    mla = sum(s for w in ways.values() for f, s in w["families"].items() if f.startswith("SparseMla"))
    assert mla == pytest.approx(0.5826, abs=5e-4)
    by = {i["input"]: i for i in doc["inputs"]}
    assert (by["proving priced measured"]["mark"], by["proving priced measured"]["value"]) == ("modeled", ways["measured"]["share"])
    assert by["proving priced measured"]["source"].endswith(
        "; each is timed alone, one session on a card, and the rows don't record how the plan prices it with its "
        "provers sharing a GPU (no `split.basis`)")
    assert (by["proving priced computed"]["mark"], by["proving priced measured and computed"]["mark"]) == ("modeled", "modeled")
    assert f"({PLAN}) `out/mix.json`: the row's plan at β = 5.542%" in by["proving priced computed"]["source"]
    text = E.render(doc)
    assert (f"- of the row's proving time, by its plan, β = 5.542% ({PLAN}), the share in block families priced each way is "
            "52.36% computed, 47.13% measured, 0.5086% measured and computed, 0.00179% in no stated way: the largest "
            "families are SparseMlaSplitPart_v1 (52.29%, computed), MoeExpertFp8CoordinateChunk_v1 (27.19%, measured), "
            "MoeExpertFp8GemmW_v1 (13%, measured).") in text
    assert f", and the world year's plan ({PLAN})" in text
    assert f"| world year | {PLAN} | 5.542% | computed | 52.358% | 7 | SparseMlaSplitPart_v1 52.291%, " in text and "98.6" not in text

    # a plan of another row's, or of the row on other premises, is not these rows' plan
    for change, why in (({"q": plan()["q"] / 2}, (f"{PLAN}'s plan of the world year is priced over 3.5568e\\+16 requests and "
                                                   f"{ROWS}'s rows over 1.7784e\\+16: it is not these rows' plan")),
                        ({"bound_bits": plan()["bound_bits"] / 2}, (f"{PLAN}'s plan of the world year bounds 6,78\\d.\\d GB at "
                                                                   f"β = 5.5422%, where {ROWS}'s rows give 13,56\\d.\\d GB"))):
        s = Store(tmp_path / next(iter(change)))
        s.label(s.ran(ROWS, record()))
        s.label(s.ran(PLAN, None, plan=plan() | change))
        with pytest.raises(ValueError, match=why):
            E.build(store=s.store, root=REPO)

    bare = E.build(store=recorded(tmp_path / "bare", planned=False).fresh(), root=REPO)
    assert bare["value"] == doc["value"] and bare["runs"] == [ROWS]
    assert all(p["priced"] is None for p in bare["compute_audit"])
    assert not [i for i in bare["inputs"] if i["input"].startswith("proving priced")]
    assert [i["input"] for i in bare["inputs"] if i["mark"] == "measured"] == ["weights"]
    text = E.render(bare)
    assert ("- how the row's proving time is priced is not read, and none of it is marked measured: the rows' `split` is not "
            "their own plan (it doesn't say the row's `q` and `replay_sample`: it is written before the row's levers "
            "multiply its blocks), and no run of record holds the row's plan (`out/mix.json`).") in text
    assert "1 is measured" in text and "block families priced" not in text and "| priced |" not in text


def test_rows_that_hold_their_own_plans_are_read_with_them(tmp_path):
    """A run whose rows each hold their own plan (a `split` at the row's q and sampled replays) needs no plan beside
    it: the figures a hand computation gave on those rows over the world's year and week, each scope's proving priced
    by its row at the pinned budget nearest, and the rows' `requests` the count they are priced over."""

    s = Store(tmp_path)
    s.label(s.ran(PLANNED, [as_row(r) for r in rows(PLANNED)]))
    doc = E.build(store=s.fresh(), root=REPO)
    week, year = doc["compute_audit"]
    assert (doc["runs"], doc["refused"], week["scope"], year["scope"]) == ([PLANNED], [], "world week", "world year")
    assert doc["value"] == year["GB"] == pytest.approx(14228.634535615152, rel=1e-12)
    assert year["beta"] == pytest.approx(0.055422344454158994, rel=1e-12)
    assert week["GB"] == pytest.approx(13961.63619929689, rel=1e-12)
    assert doc["network_check"]["GB"] == pytest.approx(15042.046132203845, rel=1e-12)
    assert {r["requests"] for r in rows(PLANNED) if r["scope"] == "world year"} == {year["requests"]}
    assert year["requests"] == pytest.approx(doc["periods"][-1]["requests"], rel=1e-12)
    for p, computed, measured in ((week, 0.5346, 0.4613), (year, 0.5240, 0.4717)):
        priced = p["priced"]
        assert (priced["of"], priced["run"], priced["beta"]) == ("rows", PLANNED, 0.055)
        assert {k: round(w["share"], 4) for k, w in priced["ways"].items()} == {
            "computed": computed, "measured": measured, "measured and computed": pytest.approx(0.0042, abs=2e-4), "modeled": 0.0}
        assert sum(w["share"] for w in priced["ways"].values()) == pytest.approx(1.0, abs=1e-9)
        assert list(priced["ways"]["computed"]["families"])[0] == "SparseMlaSplitPart_v1"
    by = {i["input"]: i for i in doc["inputs"]}
    assert (by["proving priced measured"]["mark"], by["proving priced computed"]["mark"]) == ("modeled", "modeled")
    assert (f"({PLANNED}) `split`: the row's own plan at the pinned budget nearest, β = 5.500%"
            in by["proving priced measured"]["source"])
    assert [i["input"] for i in doc["inputs"] if i["mark"] == "measured"] == ["weights"]
    text = E.render(doc)
    assert "- **the compute audit's part: 14,229 GB** (modeled), 37.78 copies of the weights" in text
    assert (f"- of the row's proving time, by its own plan at the pinned budget nearest, β = 5.500% ({PLANNED}), the share in "
            "block families priced each way is 52.4% computed, 47.17% measured, 0.4277% measured and computed, 0.001791% "
            "modeled: the largest families are SparseMlaSplitPart_v1 (52.33%, computed)") in text
    assert f"| world week | {PLANNED} (rows) | 5.500% | computed | 53.457% | 7 | SparseMlaSplitPart_v1 53.390%, " in text
    # no run carries the stop class's record, so the design's stop class is charged its whole schedule: its columns
    # are those of the design before its stop block, and `all three`'s hold its schedules
    assert ("| world | week | 2.268e+14 | 3.411e+14 | 13,962 | 0.3275 | 0.01 | 973,357 | 506,156 | 506,156 | 8.259e-20 | "
            "1.829e-08 | 987,319 | 520,117 | 520,117 | 0.01 | 468,962 |") in text
    assert ("| world | week | 468,962 | 482,924 | 937,924 | 470,722 | 470,722 | 951,885 | 484,684 | 484,684 | "
            "8.259e-20 | 1.829e-08 | 0.01 |") in text
    assert "'s plan (" not in text


def test_a_family_timed_alone_is_measured_only_where_no_proving_is_at_the_assumed_default(tmp_path):
    """A block family priced from measured statements is timed alone, and the plan prices it with its provers sharing
    a GPU through the planner's factor. Its share of the proving is marked measured only where the row's plan records
    what prices its proving (`split.basis`) and none of it is at the planner's default, which is assumed. Where some
    is, the share is modeled, its source says in words how much rests on the default, the share at the default is an
    input marked assumed, and the print says so beside the shares; no GB moves."""

    def built(basis: dict | None) -> dict:
        s = Store(tmp_path / str(len(list(tmp_path.iterdir()))))
        s.label(s.ran(PLANNED, [as_row(r) | {"split": r["split"] | ({} if basis is None else {"basis": basis})}
                                for r in rows(PLANNED)]))
        return E.build(store=s.fresh(), root=REPO)

    docs = {k: built(b) for k, b in (("none", None), ("timed", {"steady: a row": 0.9, "reprice.CACHE": 0.1}),
                                     ("default", {E.DEFAULT_BASIS: 0.75, "reprice.CACHE": 0.25}))}
    assert len({d["value"] for d in docs.values()}) == 1
    by = {k: {i["input"]: i for i in d["inputs"]} for k, d in docs.items()}
    share = by["none"]["proving priced measured"]["value"]
    assert share == pytest.approx(0.4717, abs=1e-4)
    assert {k: b["proving priced measured"]["mark"] for k, b in by.items()} == {
        "none": "modeled", "timed": "measured", "default": "modeled"}
    assert by["timed"]["proving priced measured"]["source"].endswith(
        "; each is timed alone, one session on a card, and the plan prices none of its proving at the planner's "
        "assumed default")
    assert by["default"]["proving priced measured"]["source"].endswith(
        "; each is timed alone, one session on a card, and with its provers sharing a GPU the plan prices 75.000% of its "
        f"proving at the planner's assumed default (`reprice.DEFAULT`): at least {100 * (share - 0.25):.3f}% of the "
        "proving time, in these families")
    name = "proving priced at the planner's assumed default"
    assert [name in b for b in by.values()] == [False, False, True]
    assert (by["default"][name]["value"], by["default"][name]["mark"], by["default"][name]["unit"]) == (
        0.75, "assumed", "of proving time")
    assert {k: [i["input"] for i in d["inputs"] if i["mark"] == "measured"] for k, d in docs.items()} == {
        "none": ["weights"], "timed": ["proving priced measured", "weights"], "default": ["weights"]}
    text = E.render(docs["default"])
    assert ("Of the families priced measured, each is timed alone, one session on a card, and with its provers sharing a "
            "GPU the plan prices 75.000% of its proving at the planner's assumed default (`reprice.DEFAULT`)") in text
