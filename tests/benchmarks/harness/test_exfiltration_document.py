"""`verity bench exfiltration`'s document: every input's mark, each refusal by name, the plan of record's row and row
(i) under its label, and what prices each row's proving."""

from __future__ import annotations

import json

import pytest
from exfiltration_helpers import (
    ASSUMED,
    PLAN,
    PLANNED,
    REPO,
    ROWS,
    YEAR_GB,
    Store,
    as_assumed,
    as_row,
    mine,
    plan,
    record,
    recorded,
    rows,
)
from verity_bench import exfiltration as E


def test_every_input_is_marked_measured_modeled_simulated_or_assumed_with_its_source(tmp_path):
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    by = {i["input"]: i for i in doc["inputs"]}
    assert all(i["mark"] in E.MARKS and i["source"] and i["part"] in ("compute audit", "network attester", "both")
               for i in doc["inputs"])
    assert {k: (by[k]["part"], by[k]["mark"], by[k]["value"]) for k in (
        "planner_rows", "compute_share", "delta", "output_bits", "network_check_share", "advice")} == {
        "planner_rows": ("compute audit", "modeled", 18), "compute_share": ("compute audit", "assumed", 0.1),
        "delta": ("compute audit", "assumed", 0.01), "output_bits": ("compute audit", "assumed", 28000.0),
        "network_check_share": ("compute audit", "assumed", 0.003), "advice": ("compute audit", "assumed", 0)}
    assert by["lambda"]["mark"] == "assumed" and by["lambda"]["value"] == pytest.approx(1.05)
    assert f"`{E.RECORD}` ({PLAN}, {ROWS})" in by["planner_rows"]["source"] and "2 scopes" in by["planner_rows"]["source"]
    # the world's requests are the sampling plan's, as the network attester's calculator reads them from the planner
    requests = by["world_requests"]
    assert (requests["part"], requests["mark"]) == ("both", "modeled") and "benchmarks/spot_figures/breakeven" in requests["source"]
    assert (by["request_count"]["part"], by["request_count"]["value"], by["request_count"]["mark"]) == ("both", "plan", "assumed")
    assert (by["weights"]["part"], by["weights"]["mark"]) == ("both", "measured")
    assert (by["response_ids"]["part"], by["response_ids"]["value"], by["response_ids"]["mark"]) == (
        "network attester", 1024, "assumed")
    assert (by["mfu_serve"]["part"], by["mfu_serve"]["mark"], by["mfu_serve"]["value"]) == ("both", "assumed", 0.4)
    # a request is the plan's, not the census's
    assert (by["request_tokens"]["value"], by["request_tokens"]["mark"]) == (3320.5, "modeled")
    assert by["token_macs"]["mark"] == "modeled" and by["token_macs"]["value"] == pytest.approx(7.925e10, rel=1e-3)
    assert "128k serve plan" in by["request_tokens"]["source"] and "reprice.workload" in by["token_macs"]["source"]
    world = by["world_machines"]["value"] * by["machine_gpus"]["value"] * 365 * 86400
    assert requests["value"] == pytest.approx(world * by["mfu_serve"]["value"] * by["peak_flops"]["value"]
                                              / (by["request_tokens"]["value"] * 2 * by["token_macs"]["value"]), rel=1e-9)

    text = E.render(doc)
    assert text.startswith("# verity bench exfiltration: GLM-5.3's exfiltration capacity over a monitoring period, at 10% of "
                           "total compute\n")
    assert "- **the compute audit's part: 14,240 GB** (modeled), 37.81 copies of the weights (GLM-5.3 in FP4, 376.7 GB)" in text
    assert "β = 5.542%: its bound, 13,562 GB, times λ = 1.05." in text
    assert ("- **the network attester's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids a response) "
            "as the scan's check charges it today") in text
    assert ("24,467,252 GB with the timing taken once and the entries charged through the compute audit, its stop class "
            "charged its whole schedule; and at D_j = 1, ") in text
    assert "the audit budget is β = 5.243% and the compute audit's part 15,054 GB." in text
    assert f"- compute-audit: {PLAN}, {ROWS} (`{E.RECORD}`): the planner's rows, 18 of row `(ii) honest`" in text
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in E.MARKS}
    assert marks["measured"] == 1
    assert f"of {len(doc['inputs'])} inputs, 1 is measured, {marks['modeled']} modeled, 1 simulated and {marks['assumed']} assumed" in text
    assert "| world year | " + ROWS + " | 5.542% | 13,562 | 14,240 | 37.81 | 1.778e+16 | 4.339% | 2.293% | 5.372% |" in text
    assert "| world | year | 1.182e+16 | 1.778e+16 | 14,240 | 0.006406 | 0.01 | 50,753,639 | 26,392,406 | " in text
    assert "| world | week | 2.268e+14 | 3.411e+14 | no rows of record | none | " in text
    assert "Refused" not in text and text.count("\n| datacenter | ") == 6


def test_an_unlabelled_superseded_or_ambiguous_record_is_refused_by_name(tmp_path):
    s = Store(tmp_path)
    held, planned = s.ran(ROWS, record()), s.ran(PLAN, [as_row(r) for r in rows(PLAN)], plan=plan())
    year = s.ran("r20261111-000000-0000", mine("world year"))

    def build(**kw) -> dict:
        return E.build(store=s.store, root=REPO, **kw)

    with pytest.raises(ValueError, match=f"no planner rows of record: nothing in the store carries `run_of_record={E.RECORD}`"
                                         f".*research data label RUN run_of_record {E.RECORD} --by LANE"):
        build()
    # a plan alone is no figure: its own row, at its budget, is not read off
    s.label(planned)
    with pytest.raises(ValueError, match=f"no run of record holds the planner's rows: {PLAN} hold a plan alone"):
        build()
    other = s.label(s.ran("r20261111-000000-0009", mine("datacenter year")))
    with pytest.raises(ValueError, match=f"no run of record holds the world year's rows \\({other} holds datacenter year\\)"):
        build()
    s.label(other, E.SUPERSEDED_BY, held)
    s.label(held)
    assert build()["value"] == pytest.approx(YEAR_GB)

    # a second run holding the world's year: refused until one is superseded, then the newer one is read
    with pytest.raises(ValueError, match=f"{ROWS} and {year} both hold the world year's rows and neither is superseded: label "
                                         "the one the other replaces .*superseded_by NEWER"):
        s.label(year) and build()
    s.label(year, E.SUPERSEDED_BY, held)
    newer = s.label(s.ran("r20261112-000000-0001", [r | {"bound_bits": 2 * r["bound_bits"]} for r in mine("world year")]))
    s.label(planned, E.SUPERSEDED_BY, newer)
    s.label(held, E.SUPERSEDED_BY, newer)
    doc = build()
    assert doc["value"] == pytest.approx(2 * YEAR_GB) and doc["runs"] == [newer]
    assert {x["target"]: x["why"] for x in doc["refused"]}[ROWS] == f"superseded by {newer}"
    assert f"- {ROWS} (`{E.RECORD}`): superseded by {newer}" in E.render(doc)
    # a second plan of the scope is refused the same way, and a plan of a scope with no rows is left out and listed
    a = s.label(s.ran("r20261112-000000-000a", None, plan=plan() | {"bound_bits": 2 * plan()["bound_bits"]}))
    assert build()["compute_audit"][0]["priced"]["run"] == a
    b = s.label(s.ran("r20261112-000000-000b", None, plan=plan()))
    with pytest.raises(ValueError, match=f"{b} and {a} both hold the world year's plan and neither is superseded"):
        build()
    s.label(b, E.SUPERSEDED_BY, a)
    c = s.label(s.ran("r20261112-000000-000c", None, plan=plan() | {"scope": "world week"}))
    assert {x["target"]: x["why"] for x in build()["refused"]}[c] == (
        "holds the world week's plan, and no run of record holds that scope's rows")

    # runs of record that can't be read are left out and listed
    s.label(s.ran("r20261112-000000-0002", None))
    s.label(s.ran("r20261112-000000-0003", [{}], name="out/other.jsonl"))
    s.label(s.ran("r20261112-000000-0004", [r | {"error": "timed out"} for r in mine("world year")]))
    s.label("r20261112-000000-0005")
    assert {x["target"][-1]: x["why"] for x in build()["refused"] if x["target"].startswith("r20261112") and x["target"][-1] < "a"} == {
        "2": "kept no run files", "3": "holds no planner rows (`out/rows.jsonl`, `out/three/rows.jsonl`) and no plan "
                                                "(`out/mix.json`)",
        "4": "holds no row `(ii) honest` at a step `+ the verifier as measured...` of the world `ours`",
        "5": "not in the store or on its remote"}

    # another row of the plan, a step two steps begin with, and a row no run holds
    both = recorded(tmp_path / "both").store
    third = E.build(store=both, root=REPO, row="(iii) honest")
    assert third["runs"] == [ROWS] and third["refused"][0]["target"] == PLAN and third["value"] == pytest.approx(2459.2, abs=0.05)
    assert third["compute_audit"][0]["priced"] is None
    steps = Store(tmp_path / "steps")
    steps.label(steps.ran(ROWS, [r | {"step": r["step"] + more} for r in mine("world year") for more in ("", ", again")]))
    with pytest.raises(ValueError, match=f"`\\+ ` begins 2 steps of {ROWS}'s row `\\(ii\\) honest`: give more of the one meant"):
        E.build(store=steps.store, root=REPO, step="+ ")
    with pytest.raises(ValueError, match="no run of record holds the planner's rows: .*holds no row `\\(iv\\)`"):
        build(row="(iv)")
    with pytest.raises(ValueError, match="the share of total compute is in \\(0, 1\\) and the network verifier's own below it"):
        build(share=0.1, network_check=0.1)


def test_the_two_parts_must_cover_the_same_compute_and_weights(tmp_path):
    def build(name: str, **change) -> dict:
        s = Store(tmp_path / name)
        s.label(s.ran(ROWS, [r | change for r in mine("world year")]))
        return E.build(store=s.store, root=REPO)

    assert build("same")["value"] == pytest.approx(YEAR_GB)
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 1e\\+16 GPU-seconds on the "
                                         "compute audit's side and 1.18247e\\+16 on the network attester's"):
        build("compute", gpu_seconds=1e16)
    with pytest.raises(ValueError, match="the two parts cover different compute over the world year: 2e\\+16 requests on the "
                                         "compute audit's side and 1.7784e\\+16 on the network attester's"):
        build("requests", q=1e-7, replay_sample=2e9)
    with pytest.raises(ValueError, match="the two parts are shares of different weights: the rows of record hold 3,000,000,000,000"):
        build("weights", weights_bits=3_000_000_000_000, target_bits=3_000_000_000_000 / 1.05)


def test_the_figure_is_the_plan_of_records_row_and_row_i_is_only_under_its_label(tmp_path):
    """The figure is the plan of record's row, and `together` sums it. Row (i), whose proving the planner prices at its
    assumed prices, is read over the world's year beside it and printed only under its label, and the document holds
    it as a field of its own; as the row read, the figure carries the label. A run holds its rows in two files, a
    case's over a scope read from the first that holds them, and a later file that holds other ones leaves the run
    out. A row (i) no run of record holds, or that covers other compute, is said to be unread, and the figure stands."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    head, year = next(p for p in doc["compute_audit"] if p["scope"] == "world year"), doc["periods"][-1]
    assert (E.ROW, E.ASSUMED_ROW, doc["row"], doc["label"]) == ("(ii) honest", "(i) honest", "(ii) honest", None)
    assert doc["value"] == head["GB"] == year["compute_audit"]["GB"] == pytest.approx(YEAR_GB, rel=1e-12)
    for t in year["together"].values():
        assert t["GB"] == pytest.approx(t["attester_GB"] + YEAR_GB) and t["with_set_time_card"]["GB"] == pytest.approx(
            t["with_set_time_card"]["attester_GB"] + YEAR_GB)
    a = doc["at_assumed_prices"]
    assert (a["row"], a["label"], a["run"], a["delta"], a["inside"], a["why"], a["sources"]) == (
        E.ASSUMED_ROW, E.ASSUMED_LABEL, ROWS, 0.01, True, None, None)
    assert a["GB"] == pytest.approx(ASSUMED * YEAR_GB, rel=1e-12) and a["beta"] == pytest.approx(head["beta"], rel=1e-12)
    assert json.loads(json.dumps(doc))["at_assumed_prices"]["GB"] == a["GB"] != doc["value"]

    text = E.render(doc)
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in text
    assert "It is the planner's row `(ii) honest`, the plan of record's row, with every modeled lever at its measured value" in text
    line = (f"- row `(i) honest`, at the planner's assumed prices: 17,800 GB, at β = 5.542% ({ROWS}), failing with at most "
            "δ = 0.01 over the monitoring period. It is not the figure. What prices its proving time: own row not recorded, "
            "family row not recorded, planner's price not recorded; at the planner's assumed default (`reprice.DEFAULT`) "
            "not recorded.")
    assert [x for x in text.splitlines() if "(i) honest" in x or "17,8" in x] == [line]

    own = E.build(store=recorded(tmp_path / "own").fresh(), root=REPO, row=E.ASSUMED_ROW)
    assert (own["label"], own["at_assumed_prices"]) == (E.ASSUMED_LABEL, None)
    assert own["value"] == pytest.approx(ASSUMED * YEAR_GB, rel=1e-12)
    text = E.render(own)
    assert "- **the compute audit's part, at the planner's assumed prices: 17,800 GB** (modeled)" in text
    assert "- row `(i) honest`, at" not in text

    s = Store(tmp_path / "alone")
    s.label(s.ran(ROWS, mine()))
    alone = E.build(store=s.store, root=REPO)
    assert alone["value"] == pytest.approx(YEAR_GB) and alone["at_assumed_prices"]["GB"] is None
    assert (f"- row `(i) honest`, at the planner's assumed prices: not read: no run of record holds the planner's rows: "
            f"{ROWS} holds no row `(i) honest` at a step") in E.render(alone)
    files = record()
    s = Store(tmp_path / "other")
    s.label(s.ran(ROWS, files | {E.ROWS_FILES[0]: [r | {"gpu_seconds": 1e16} for r in files[E.ROWS_FILES[0]]
                                                   if r["row"] == E.ASSUMED_ROW]}))
    other = E.build(store=s.store, root=REPO)
    assert other["value"] == pytest.approx(YEAR_GB)
    assert other["at_assumed_prices"]["why"] == "its rows differ from the figure's in gpu_seconds"

    # the driver writes row (i)'s world year to both files: the same rows, in any order, are read once
    twice = [r for r in files[E.ROWS_FILES[0]] if r["row"] == E.ASSUMED_ROW and r["scope"] == "world year"]
    s = Store(tmp_path / "same")
    s.label(s.ran(ROWS, files | {E.ROWS_FILES[1]: files[E.ROWS_FILES[1]] + twice[::-1]}))
    assert E.build(store=s.store, root=REPO)["at_assumed_prices"]["GB"] == pytest.approx(a["GB"], rel=1e-12)
    s = Store(tmp_path / "differ")
    s.label(s.ran(ROWS, files | {E.ROWS_FILES[1]: files[E.ROWS_FILES[1]] + [r | {"bound_bits": 2 * r["bound_bits"]}
                                                                           for r in twice]}))
    with pytest.raises(ValueError, match=f"no run of record holds the planner's rows: {ROWS} holds row `\\(i\\) honest` over "
                                         "the world year in `out/rows.jsonl` and in `out/three/rows.jsonl`, and the two "
                                         "differ"):
        E.build(store=s.store, root=REPO)


def test_a_note_the_figure_s_rows_carry_is_printed_beside_it_where_it_has_no_label(tmp_path):
    """The planner's driver writes on a row what a reader must know beside its figure (`note`): the figure's line
    prints the note of the row at the pinned budget nearest its β, and the document holds it on the scope. Row (i)'s
    label is its rows' note, read itself or beside the figure; else the share the table's rows price; else, for rows
    that record no table's row, "at the planner's assumed prices". Rows that carry none print as they did."""

    said = "row `(ii) honest` with the Lean verifier's seconds scaled x0.25 and nothing else"
    s = Store(tmp_path)
    s.label(s.ran(ROWS, [r | {"note": said} for r in mine()]))
    doc = E.build(store=s.store, root=REPO)
    head = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    assert head["note"] == said and doc["value"] == pytest.approx(YEAR_GB)
    assert f"- **the compute audit's part, {said}: 14,240 GB** (modeled)" in E.render(doc)

    s = Store(tmp_path / "labelled")
    planner = "at the planner's prices, not the table's"
    s.label(s.ran(ROWS, [as_assumed(r) | {"note": planner} for r in mine()]))
    own = E.build(store=s.store, root=REPO, row=E.ASSUMED_ROW)
    assert own["label"] == planner and f"- **the compute audit's part, {planner}: 17,800 GB** (modeled)" in E.render(own)

    # row (i)'s label is what its own rows say of their pricing, beside the figure as when read itself
    under = "under half priced by the table's rows, 43.2% of its proving (by source: planner's price 44%, own row 43%)"
    files = record()
    s = Store(tmp_path / "under")
    s.label(s.ran(ROWS, {f: [r | {"note": under} if r["row"] == E.ASSUMED_ROW else r for r in rs]
                         for f, rs in files.items()}))
    doc = E.build(store=s.store, root=REPO)
    assert doc["at_assumed_prices"]["label"] == under and f"- row `(i) honest`, {under}: 17,800 GB" in E.render(doc)
    priced = {"priced": {"sources": {"by": {"own row": 0.3, "family row": 0.1, "planner's price": 0.6}}}}
    assert E._row_label(priced) == "40% of its proving priced by the table's rows"
    assert E._row_label({"priced": None}) == E._row_label({}) == E.ASSUMED_LABEL

    bare = E.build(store=recorded(tmp_path / "bare").fresh(), root=REPO)
    assert not any("note" in p for p in bare["compute_audit"])
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in E.render(bare)


def test_what_prices_the_rows_proving_is_printed_by_source_or_not_recorded(tmp_path):
    """Beside each row printed, the share of its proving each source prices, as its own plan records it: its `sources`,
    else its `basis` by source, each of own row, family row and the planner's price at 0 where it prices none, and the
    share at the planner's assumed default. A row whose plan records neither, or that is read with a plan of record,
    prints "not recorded" for each."""

    assert E._sources({"families": {}}) is None
    basis = {"reprice.DEFAULT": 0.25, "steady": 0.5, "steady: art:00": 0.125, "family": 0.0625, "reprice.FAMILY": 0.03,
             "reprice.CACHE": 0.0124, "headline.copy_stratum": 0.0201}
    assert E._sources({"basis": basis}) == {
        "by": {"own row": 0.625, "family row": 0.0625, "planner's price": pytest.approx(0.2924), "headline.copy_stratum": 0.0201},
        "default": 0.25}
    sources = {"own row": 0.99, "planner's price": 0.01}
    assert E._sources({"sources": sources, "basis": basis}) == {
        "by": {"own row": 0.99, "family row": 0.0, "planner's price": 0.01}, "default": 0.25}
    assert E._sources({"sources": sources})["default"] is None

    s = Store(tmp_path)
    s.label(s.ran(PLANNED, [as_row(r) | {"split": r["split"] | {"basis": basis}} for r in rows(PLANNED)]
                  + [as_assumed(r) | {"split": r["split"] | {"basis": {"reprice.DEFAULT": 0.979, "reprice.CACHE": 0.021}}}
                     for r in rows(PLANNED)]))
    doc = E.build(store=s.fresh(), root=REPO)
    assert doc["at_assumed_prices"]["sources"] == {
        "by": {"own row": 0.0, "family row": 0.0, "planner's price": pytest.approx(1.0)}, "default": 0.979}
    text = E.render(doc)
    assert (f"- what prices the row's proving time, as its rows record it (its own plan at β = 5.500%, {PLANNED}): own row "
            "62.5%, family row 6.25%, planner's price 29.24%, headline.copy_stratum 2.01%; at the planner's assumed default "
            "(`reprice.DEFAULT`) 25%.\n") in text
    assert ("What prices its proving time: own row 0%, family row 0%, planner's price 100%; at the planner's assumed default "
            "(`reprice.DEFAULT`) 97.9%.\n") in text

    # the fixture's own plans record no source, and a plan of record none either
    none = ("own row not recorded, family row not recorded, planner's price not recorded; at the planner's assumed default "
            "(`reprice.DEFAULT`) not recorded.\n")
    s = Store(tmp_path / "bare")
    s.label(s.ran(PLANNED, [as_row(r) for r in rows(PLANNED)]))
    bare = E.build(store=s.store, root=REPO)
    assert next(p for p in bare["compute_audit"] if p["scope"] == "world year")["priced"]["sources"] is None
    assert "- what prices the row's proving time, as its rows record it: " + none in E.render(bare)
    planned = E.build(store=recorded(tmp_path / "planned").fresh(), root=REPO)
    assert planned["compute_audit"][1]["priced"]["of"] == "plan"
    assert "- what prices the row's proving time, as its rows record it: " + none in E.render(planned)
