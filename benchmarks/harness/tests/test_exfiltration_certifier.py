"""`verity bench exfiltration`'s network attester's part: beside the figures at the requests its calculator counts;
its figures of record at two release schedules; what rests on the set-time card; the output count beside the sums that
take it at 0; and every figure's error at the scans' delta the term states, on term documents with made-up errors."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import pytest
from exfiltration_helpers import (
    HOLDER,
    REPO,
    ROWS,
    YEAR_GB,
    delta,
    held_phase,
    printed,
    reader,
    recorded,
    term,
    with_verdicts,
)
from verity_bench import exfiltration as E


def test_the_network_attesters_part_is_beside_it_at_the_requests_its_calculator_counts(tmp_path, monkeypatch):
    """Both parts count the sampling plan's requests, which the network attester's calculator counts by default and
    which are fewer than the census's, over the same GPU-seconds. The command holds no count of its own: the count and
    what its request is are the calculator's inputs as it marks them, none of them given, and where the calculator
    counts other requests than the rows of record are priced over the command refuses. The network attester's rows are its own
    calculator's for the class the command reads; and the two are summed under each of its rows by the form: its term
    at each jitter bound, plus the compute audit's part or the entries where the row charges them and they are fewer."""

    lb = term()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert [(p["compute"], p["period"]) for p in doc["periods"]] == [
        (c, p) for c in ("datacenter", "world") for p in ("week", "month", "year")]
    year = doc["periods"][-1]
    assert year["requests"] == pytest.approx(1.7784e16, rel=1e-4) and year["gpu_seconds"] == pytest.approx(1.182473856e16)
    counted = lb.inputs()
    assert year["requests"] == pytest.approx(counted["world_requests"].value, rel=1e-12)
    assert counted["world_requests"].value < lb.inputs(requests="census")["world_requests"].value  # the census's request
    by = {i["input"]: i for i in doc["inputs"]}
    for name in ("request_count", "world_requests", "request_tokens", "token_macs"):
        assert (by[name]["part"], by[name]["value"], by[name]["mark"], by[name]["source"]) == (
            "both", counted[name].value, counted[name].mark, counted[name].source)
    assert by["request_count"]["value"] == "plan" and by["request_tokens"]["value"] == 3320.5
    assert not [i["input"] for i in doc["inputs"] if i["value"] == "given" or "--world-requests" in i["source"]]

    theirs = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS, scan_delta=delta(lb)), only=("world",), periods=("year",))
    assert doc["certifier"]["rows"] == theirs["rows"] and [r["row"] for r in theirs["rows"]][-2:] == [E.OUTPUT_COUNT, "design"]
    assert (doc["certifier"]["schema"], doc["certifier"]["request_count"], doc["certifier"]["response_ids"]) == (
        E.TERM_SCHEMA, "plan", 1024)
    assert (doc["certifier"]["of_record"], doc["certifier"]["set_time"]) == ("2", "1")
    scan = doc["certifier"]["rows"][0]
    assert scan["row"] == "scan" and scan["term"]["1"]["bits"] == pytest.approx(22 * year["requests"])

    audit, together = year["compute_audit"], year["together"]
    assert audit["GB"] == doc["value"] and audit["bits_a_request"] == pytest.approx(audit["bits"] / year["requests"])
    assert audit["of_weights"] == pytest.approx(doc["value"] / doc["weights"]["GB"]) == pytest.approx(37.805, abs=5e-4)
    assert list(together) == list(E.TOGETHER) == ["scan", "all three", "design, no stop block", "design"]
    rows = {r["row"]: r for r in theirs["rows"]}
    for name, t in together.items():
        card = t["with_set_time_card"]
        assert (t["certifier_GB"], card["certifier_GB"]) == (rows[name]["term"]["2"]["GB"], rows[name]["term"]["1"]["GB"])
        assert t["GB"] == pytest.approx(t["certifier_GB"] + audit["GB"])
        assert card["GB"] == pytest.approx(card["certifier_GB"] + audit["GB"])
    # the entries stand in for the compute audit's part where the scan charges them and they are fewer
    assert scan["entries"]["bits"] > audit["bits"]
    dc = doc["periods"][2]
    assert dc["compute_audit"]["GB"] == pytest.approx(13707.9, abs=0.05) and dc["compute_audit"]["run"] == ROWS
    assert dc["requests"] == pytest.approx(year["requests"] * dc["gpu_seconds"] / year["gpu_seconds"], rel=1e-12)
    # a scope no run of record holds has the network attester's part alone
    week = doc["periods"][0]
    assert week["compute_audit"] is None and week["together"]["design"]["GB"] is None and week["together"]["design"]["certifier_GB"] > 0
    assert week["together"]["design"]["with_set_time_card"]["GB"] is None

    # the count read is the calculator's: were its plan another, the rows of record would be priced over other requests
    plan = lb.plan()
    monkeypatch.setattr(lb, "plan", lambda: plan | {"world_requests": plan["world_requests"]._replace(
        value=plan["world_requests"].value / 2)})
    with pytest.raises(ValueError, match="the two parts cover different compute over the datacenter year: 4.74291e\\+12 "
                                         "requests on the compute audit's side and 2.37145e\\+12 on the network attester's"):
        E.build(store=recorded(tmp_path / "halved").fresh(), root=REPO)


def test_the_scan_rows_mark_is_this_commands_mark_of_that_name():
    """The calculator marks the `scan` row's accept `run, not proved` by a string of its own, since it cannot read
    this command's: the two are one mark, and no other row of the calculator's carries one."""

    lb = term()
    assert lb.RUN_NOT_PROVED == E.RUN_NOT_PROVED and E.RUN_NOT_PROVED in E.MARKS
    assert {name: k.mark for name, k in lb.ROWS.items() if k.mark} == {"scan": E.RUN_NOT_PROVED}


def test_the_network_attesters_figures_of_record_are_at_two_release_schedules(tmp_path, monkeypatch):
    """The network attester's figures of record are its document's at D_j = 2, held here to that document: the scan's
    over the world's year is the document's `value`, on the class its calculator assumes by default and on the class
    the command reads, and the design's is each class's own: with no run carrying the stop class's record its stop
    class is charged its whole schedule, so it is the jitter, the count and each connection's schedule, the figure of
    the design before its stop block. The headline and the sums are those figures,
    and the calculator's own figure is printed beside the command's where their classes differ. A document of
    another schema is refused by name."""

    lb = term()
    theirs = lb.document(lb.inputs())

    def of_year(document: dict, name: str) -> dict:
        return next(r for r in document["rows"] if (r["scope"], r["period"], r["row"]) == ("world", "year", name))

    design = of_year(theirs, "design")
    assert (theirs["schema"], theirs["request_count"], lb.OF_RECORD) == (E.TERM_SCHEMA, "plan", 2)
    assert theirs["value"] == pytest.approx(59_645_644, abs=0.5) and design["term"]["2"]["GB"] == pytest.approx(30_838_408, abs=0.5)
    assert design["term"] == of_year(theirs, E.OUTPUT_COUNT)["term"]

    wide = E.build(store=recorded(tmp_path / "wide").fresh(), root=REPO, response_ids=lb.RESPONSE_IDS)
    sums = wide["periods"][-1]["together"]
    assert (sums["scan"]["certifier_GB"], sums["design"]["certifier_GB"]) == (theirs["value"], design["term"]["2"]["GB"])
    assert sums["scan"]["with_set_time_card"]["certifier_GB"] == theirs["with_set_time_card"]["value"]
    assert wide["certifier"]["default_class"] == {"response_ids": 4096, "eps": 2 ** -20, "value": theirs["value"], "unit": "GB"}
    text = E.render(wide)
    assert ("- **the network attester's part: 59,645,644 GB** (modeled, on an assumed class of at most 4,096 ids a response) "
            "as the scan's check charges it today, 26.83 bits a request, and 30,838,408 GB by the design no run has behind it, ") in text
    assert ("30,838,408 GB by the design no run has behind it, its stop class charged its whole schedule, each at D_j = 2, "
            "its term of record, and each failing with at most ε = 9.537e-07 (2^-20) over the world's year, at a scan's δ of "
            "8.259e-20 (2^-63.39).\n") in text
    assert "assumes by default" not in text

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    narrow = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))
    sums = doc["periods"][-1]["together"]
    assert E.RESPONSE_IDS == 1024 and sums["scan"]["certifier_GB"] == narrow["value"] == pytest.approx(50_753_639, abs=0.5)
    assert sums["design"]["certifier_GB"] == of_year(narrow, "design")["term"]["2"]["GB"] == pytest.approx(26_392_406, abs=0.5)
    assert sums["design"]["certifier_GB"] == sums[E.OUTPUT_COUNT]["certifier_GB"]
    assert sums["scan"]["GB"] == pytest.approx(narrow["value"] + YEAR_GB) and sums["design"]["GB"] == pytest.approx(26_406_646, abs=0.5)
    assert doc["certifier"]["default_class"] == wide["certifier"]["default_class"]
    text = E.render(doc)
    assert ("- **the network attester's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids a response) "
            "as the scan's check charges it today, 22.83 bits a request, and "
            "26,392,406 GB by the design no run has behind it, before a served audit has sampled a stop block, 26,392,406 GB by "
            "the design no run has behind it, its stop class charged its whole schedule, each at D_j = 2, its term of "
            "record, and each failing with at most ε = 9.537e-07 (2^-20) over the world's year, at a scan's δ of 8.259e-20 "
            "(2^-63.39). On the class of at most 4,096 ids a response that `verity bench certifier_term` assumes by "
            "default, the scan's is 59,645,644 GB.\n") in text
    assert ("at D_j = 2, with no advice priced and with a response's output count taken at 0 under no row: "
            "50,767,879 GB as the scan's check charges it today; 26,406,646 GB by the "
            "design no run has behind it, before a served audit has sampled a stop block; 26,406,646 GB by the design no run "
            "has behind it, its stop class charged its whole schedule; each failing with at most 0.01, the two parts' "
            "errors summed.\n") in text
    assert ("| world | year | 1.182e+16 | 1.778e+16 | 14,240 | 0.006406 | 0.01 | 50,753,639 | 26,392,406 | 26,392,406 | "
            "8.259e-20 | 9.537e-07 | 50,767,879 | 26,406,646 | 26,406,646 | 0.01 | 24,453,012 |") in text
    assert "| scan | 24,453,012 | 24,453,012 | 50,753,639 | 134,745 | 48,906,024 | 4.108e+10 | 8.259e-20 | 9.537e-07 |" in text

    document = lb.document
    monkeypatch.setattr(lb, "document", lambda *a, **kw: document(*a, **kw) | {"schema": "certifier-term/v1"})
    with pytest.raises(ValueError, match="the network attester's document is `certifier-term/v1` and this command reads "
                                         "`certifier-term/v2`"):
        E.build(store=recorded(tmp_path / "older").fresh(), root=REPO)


def test_what_rests_on_the_set_time_card_is_printed_only_under_the_calculators_label(tmp_path, monkeypatch):
    """A figure at D_j = 1, and every figure of a row whose changes include the timing's, is what a network attester
    with a set-time card would be charged, and is printed only under the label the calculator's document gives it: in
    a line, a column or a row that carries the label. The label is the document's, read from the row's own field and
    from its top: the command's code doesn't spell it, and prints another where the document says another."""

    lb = term()
    label = lb.SET_TIME
    assert label and not any(label in p.read_text() for p in Path(E.__file__).parent.glob("exfiltration*.py"))
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    theirs = {(r["scope"], r["period"], r["row"]): r for r in lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    for p in doc["periods"]:
        for name, t in p["together"].items():
            assert t["label"] == theirs[p["compute"], p["period"], name]["with_set_time_card"]
            assert t["label"] == (label if name == "all three" else None) and t["with_set_time_card"]["label"] == label
    assert {r["row"] for r in doc["certifier"]["rows"] if r["with_set_time_card"] == label} == {"timing", "all three"}

    text = E.render(doc)
    shown = printed(text)
    at_one = [(under, cell) for under, cell in shown if "D_j = 1" in under]
    timed = [(under, cell) for under, cell in shown
             if under.rpartition(" | ")[2].startswith(("timing", "all three")) or ", all three" in under.partition(" | ")[0]]
    assert (len(at_one), len(timed)) == (47, 30) and all(label in under for under, _ in at_one + timed)
    # the scan's term and its two sums at D_j = 1, and the sum under `all three`
    # with no run carrying the stop class's record, `all three` and the design are charged their whole schedules
    for figure, least in (("48,906,024", 5), ("48,920,264", 2), ("24,467,252 GB with the timing taken once", 1)):
        under = [under for under, cell in shown if figure in cell]
        assert len(under) >= least and all(label in u for u in under), figure
    assert ("- the network attester's part with a simulated set-time card: 24,453,012 GB with the timing taken once and "
            "the entries charged through the compute audit, its stop class charged its whole schedule; and at D_j = 1, ") in text
    assert ("- together, with a simulated set-time card: 24,467,252 GB with the timing taken once and the entries "
            "charged through the compute audit, its stop class charged its whole schedule; and at D_j = 1, 48,920,264 GB as "
            "the scan's check charges it today; 24,559,030 GB by the design no run has "
            "behind it, before a served audit has sampled a stop block; 24,559,030 GB by the design no run has behind it, its "
            "stop class charged its whole schedule; error 0.01.\n") in text
    assert ("| all three, with a simulated set-time card | 24,453,012 | 0 | 24,453,012 | 64,920 | 24,453,012 | the compute "
            "side's | 8.259e-20 | 9.537e-07 |") in text
    assert "\n## Each monitoring period over all compute, with a simulated set-time card: in GB\n" in text
    assert ("| world | year | 24,453,012 | 24,467,252 | 48,906,024 | 24,544,790 | 24,544,790 | 48,920,264 | "
            "24,559,030 | 24,559,030 | 8.259e-20 | 9.537e-07 | 0.01 |") in text
    assert "with the lengths held by a stop block" not in text and "once a served audit has sampled a stop block" not in text
    # no figure of record is under the label: the headline's and the table of record's lines don't carry it
    assert not [line for line in text.splitlines() if label in line and ("its term of record, " in line or "at D_j = 2, with no" in line)]
    assert label not in text.split("## Each monitoring period over all compute: the two parts side by side")[1].split("\n## ")[0]

    other = "under a label no code of the command holds"
    monkeypatch.setattr(lb, "SET_TIME", other)
    again = E.render(E.build(store=recorded(tmp_path / "again").fresh(), root=REPO))
    assert label not in again and again.count(other) == text.count(label) and again == text.replace(label, other)


def test_the_output_count_is_beside_the_sums_that_take_it_at_zero(tmp_path, listed):
    """Until a served audit has sampled a stop block the server chooses a response's output count. The network
    attester's row for the design before its stop block charges it, as its schedules, and the command reads it from
    there, in every monitoring period. The sums that take it at 0 are named from the stop verdicts' reader's answer,
    never from a schedule that reads 0: with no run carrying the stop class's record no sum does, the rows that
    would hold it as their schedules, the inputs list no output count and the print says so. The reader's answer
    holds no execution phase whatever a run's files hold, since the reader asks the network attester's rule
    without a schedule and the rule charges no class below a schedule it was not handed: with a class's four
    verdicts in the run of record, in the reader's own format and with its formats listed, no row is named, no sum
    takes the count at 0, and the print's lines say the schedule was not handed."""

    lb = term()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    theirs = {(r["scope"], r["period"], r["row"]): r for r in lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    for p in doc["periods"]:
        count, row = p["output_count"], theirs[p["compute"], p["period"], E.OUTPUT_COUNT]
        assert (count["row"], count["mark"], count["at_zero_in"]) == ("design, no stop block", "assumed", [])
        assert (count["GB"], count["bits"]) == (row["schedules"]["GB"], row["schedules"]["bits"])
        assert count["bits_a_request"] == pytest.approx(11) and count["bits"] == pytest.approx(11 * p["requests"])
        sums = p["together"]
        assert sums[E.OUTPUT_COUNT]["certifier_GB"] == sums["design"]["certifier_GB"]
        assert all(theirs[p["compute"], p["period"], name]["schedules"]["bits"] == count["bits"] for name in ("all three", "design"))
    year = doc["periods"][-1]
    assert year["output_count"]["GB"] == pytest.approx(24_453_012, abs=0.5)
    assert year["together"][E.OUTPUT_COUNT]["GB"] == year["together"]["design"]["GB"]
    assert doc["certifier"]["stop"] == {"rows": ["lengths", "all three", "design"], "held_phases": 0, "below": False,
                                        "why": list(lb.NO_RECORDS.why), "verdicts": [], "runs": [], "superseded": []}
    assert [i for i in doc["inputs"] if i["input"] == "output_count"] == []
    text = E.render(doc)
    assert "with no advice priced and with a response's output count taken at 0 under no row: " in text
    assert ("- the output count: 24,453,012 GB (assumed), 11 bits a request. Until a served audit has sampled a stop block the "
            "server chooses a response's output count, and the network attester's row `design, no stop block` charges it "
            "as its schedules. No sum takes it at 0: those under `all three` and `design` hold it too, their stop class "
            "charged its whole schedule.\n") in text
    assert ("- the stop class of the network attester's rows `lengths`, `all three` and `design` is charged its whole "
            "schedule, as the calculator charges it with no record read: no run carries the stop class's record, so the "
            "stop verdicts' reader (`stop_verdicts.of_runs`) was not asked. A sum takes the output count at 0 "
            "only on an execution phase the reader's answer holds, never because its row's schedules read 0:\n"
            "  - no verdict record was read, so each connection is charged its whole schedule: the stop record, the "
            "response tie, the certified read, the compute audit are each missing\n") in text
    assert "| together, design | together's error | output count (assumed) |\n" in text
    assert "what a response's output count is charged while the server chooses it (assumed): no sum of this table takes it at 0.\n" in text
    assert "he sum under" not in text and "| output_count |" not in text
    # on the class the calculator assumes by default a connection's count is 13 bits
    wide = E.build(store=recorded(tmp_path / "wide").fresh(), root=REPO, response_ids=lb.RESPONSE_IDS)
    count = wide["periods"][-1]["output_count"]
    assert count["bits_a_request"] == pytest.approx(13) and count["GB"] == pytest.approx(28_899_014, abs=0.5)

    # the run of record holds a class's four verdicts in the reader's own format, each accepted and complete, with
    # its controls and its rates, and the reader lists their formats. The reader asks the rule without a schedule,
    # so its answer holds no execution phase: no row is named, and no sum takes the count at 0
    store = with_verdicts(tmp_path / "held", [], held=held_phase(listed)).store
    held = E.build(store=store, root=REPO)
    answer = held["certifier"]["stop"]
    assert (answer["rows"], answer["held_phases"], answer["below"], answer["runs"], answer["superseded"]) == (
        ["lengths", "all three", "design"], 0, False, [HOLDER], [])
    assert len(answer["verdicts"]) == 4 and {v["mark"] for v in answer["verdicts"]} == {E.RUN_NOT_PROVED}
    # the document is the calculator's on the reader's own answer on the run that carries the record, and no other's
    asked = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS), stop=reader().of_runs(store, [HOLDER], []))
    assert answer == {k: asked["stop"][k] for k in ("rows", "held_phases", "below", "why", "verdicts")} | {
        "runs": [HOLDER], "superseded": []}
    assert answer["why"] == [
        ("each connection is charged its whole schedule: the schedule was not handed for 1 of the 1 class(es) that hold "
        "a row in the 1 execution phase(s) read, so the rule charges none of those below it, and 0 of the 1 lack a "
        "verdict: this reader is handed no schedule and asks the rule without one"),
        reader().SERVE_NAMED,
        ("the link responses, execution phase 3, class 0: charged its whole schedule on each of its 4 rows: the "
        "schedule was not handed"),
        (f"`out/registrations.jsonl` of {HOLDER} says its registrations were checked under the network certifier's key "
        f"whose SHA-512 is {'5a' * 64}, in the session {'c3' * 32}: the reader holds neither, and compares them with "
        "nothing")]
    bare = E.build(store=recorded(tmp_path / "bare", planned=False).store, root=REPO)
    assert held["periods"] == bare["periods"] == doc["periods"] and held["inputs"] == bare["inputs"]
    assert {tuple(p["output_count"]["at_zero_in"]) for p in held["periods"]} == {()}
    text = E.render(held)
    assert "with no advice priced and with a response's output count taken at 0 under no row: " in text
    assert ("as its schedules. No sum takes it at 0: those under `all three` and `design` hold it too, their stop class "
            "charged its whole schedule.") in text
    assert ("- the stop class of the network attester's rows `lengths`, `all three` and `design` is charged its whole "
            "schedule, by the answer of the stop verdicts' reader (`stop_verdicts.of_runs`) on the runs that carry "
            f"`{E.VERDICTS}` ({HOLDER}). A sum takes the output count at 0 only on an execution phase the reader's answer "
            "holds, never because its row's schedules read 0:\n"
            "  - each connection is charged its whole schedule: the schedule was not handed for 1 of the 1 class(es) ") in text
    assert [line[4:] for line in text.splitlines() if line.startswith("  - ")] == answer["why"]
    assert "what a response's output count is charged while the server chooses it (assumed): no sum of this table takes it at 0.\n" in text
    assert "he sum under" not in text and "| output_count |" not in text and "execution phase(s) of a link" not in text
    # the same files with no format listed in the reader hold nothing either, each verdict missing: as with no record
    reader().FORMATS.clear()
    unlisted = E.build(store=with_verdicts(tmp_path / "unlisted", [], held=held_phase(listed)).store, root=REPO)
    assert (unlisted["certifier"]["stop"]["held_phases"], unlisted["certifier"]["stop"]["runs"]) == (0, [HOLDER])
    assert unlisted["certifier"]["stop"]["verdicts"] == [] and unlisted["certifier"]["stop"]["why"][2].endswith(
        "the compute audit is missing; the schedule was not handed")
    assert unlisted["periods"] == held["periods"] and unlisted["inputs"] == held["inputs"]


def test_every_figure_carries_its_error_at_the_scans_delta_the_term_states(tmp_path):
    """Given no δ, the command gives the calculator the one at which the row it prints that sums the most execution
    phases, the world's year, carries `PERIOD_ERROR`, so that row's ε is that error and every other row's less. Each
    scope's monitoring period carries its rows' own ε, their execution phases times that δ, the compute audit's δ
    beside it, and their sum for the two together. The δ is the one the term's document states among its inputs,
    marked as the calculator marks it, with the command's reason as its source. No amount moves with it."""

    lb = term()
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    given = delta(lb)
    assert Fraction(1, 2 ** 20) == E.PERIOD_ERROR and given == Fraction(1, 12108532285440000000) != lb.SCAN_DELTA
    theirs = lb.document(lb.inputs(response_ids=E.RESPONSE_IDS, scan_delta=given), only=E.SCOPES)
    said, by = doc["certifier"]["scan_delta"], {i["input"]: i for i in doc["inputs"]}
    assert (said["value"], said["mark"]) == (theirs["inputs"]["scan_delta"]["value"], "assumed") and said["value"] == float(given)
    assert said["source"] == (
        "each execution phase's scan's δ, this command's, in place of the calculator's default: 9.537e-07 (2^-20), the error "
        "a monitoring period's figure is to carry at most, over the 1.155e+13 execution phases of the world's year, the row "
        "printed that sums the most, so that every row printed carries that error or less. The network certifier's result "
        "of record takes its δ so (benchmarks/certifier_term/README.md), and no term depends on it")
    assert by["scan_delta"] == {"part": "network certifier", "input": "scan_delta", **said}

    rows = {(r["scope"], r["period"]): r for r in theirs["rows"]}
    for p in doc["periods"]:
        r, audit = rows[p["compute"], p["period"]], p["compute_audit"]
        assert p["eps"] == r["eps"] == float(Fraction(r["phases"]) * given) and p["phases"] == r["phases"]
        assert p["error"] == (None if audit is None else p["eps"] + audit["delta"])
    dc, year = doc["periods"][2], doc["periods"][-1]
    assert year["eps"] == 2 ** -20 == max(p["eps"] for p in doc["periods"]) and year["phases"] == 11547596250000
    assert (year["compute_audit"]["delta"], year["error"]) == (0.01, 0.01 + 2 ** -20)
    assert dc["eps"] == pytest.approx(2.543e-10, rel=1e-3) and dc["error"] == 0.01 + dc["eps"]
    assert doc["periods"][0]["error"] is None  # a scope no run of record holds has no sum, and so no error of one
    # the amounts are the calculator's at its own default δ, where the world's year's ε is 2^-20 and every other row's is less
    default = {(r["scope"], r["period"], r["row"]): r for r in lb.document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    for p in doc["periods"]:
        for name, t in p["together"].items():
            r = default[p["compute"], p["period"], name]
            assert r["eps"] <= 2 ** -20 and t["certifier_GB"] == r["term"]["2"]["GB"]
    assert default["world", "year", "scan"]["eps"] == 2 ** -20

    text = E.render(doc)
    assert ("- the compute audit's part fails with at most δ = 0.01 over the monitoring period. The network attester's ε "
            "over the world's year is each execution phase's scan's δ (assumed), 8.259e-20 (2^-63.39), summed over every "
            "link's execution phases, 1.155e+13 of them: 9.537e-07 (2^-20). The two together fail with at most the sum of "
            "both, 0.01. A figure whose error, summed over everything it covers, is 1 or more is printed as \"no bound at "
            "this period\", with that error and no amount.\n") in text
    assert ("- the network attester's part with a simulated set-time card: 24,453,012 GB with the timing taken once and "
            "the entries charged through the compute audit, its stop class charged its whole schedule; and at D_j = 1, ") in text
    assert ("24,544,790 GB by the design no run has behind it, its stop class charged its whole schedule; "
            "ε = 9.537e-07 (2^-20).\n") in text
    assert ("| compute | period | GPU-seconds | requests | compute audit | bits a request | compute audit's δ | "
            "network attester, scan | network attester, design, no stop block | network attester, design | scan's δ | "
            "network attester's ε | together, scan | "
            "together, design, no stop block | together, design | together's error | output count (assumed) |\n") in text
    assert ("| datacenter | year | 3.154e+12 | 4.743e+12 | 13,708 | 23.12 | 0.01 | 13,536 | 7,039 | 7,039 | 8.259e-20 | "
            "2.543e-10 | 27,244 | 20,747 | 20,747 | 0.01 | 6,521 |") in text
    assert ("| datacenter | week | 6.048e+10 | 9.096e+10 | no rows of record | none | none | 259.6 | 135 | 135 | "
            "8.259e-20 | 4.878e-12 | none | none | none | none | 125.1 |") in text
    shown = printed(text)
    assert {cell for under, cell in shown if under.startswith("scan's δ | ")} == {"8.259e-20"}
    assert len([cell for under, cell in shown if under.startswith("scan's δ | ")]) == 7 + 6 + 6
    assert E.NO_BOUND not in text.replace(f"\"{E.NO_BOUND}\"", "")


def made_up(document, eps: Mapping[tuple[str, str], float], scan_delta: float | None = None, **row: float):
    """The calculator's ``document`` with made-up errors: each scope's monitoring period of ``eps`` at that ε on
    every row, but for the rows ``row`` names, which over the world's year are each at its own; and with
    ``scan_delta`` the δ the document states."""

    def made(*a, **kw) -> dict:
        doc = document(*a, **kw)
        stated = doc["inputs"]["scan_delta"] | ({} if scan_delta is None else {"value": scan_delta})
        rows = [r | {"eps": eps.get((r["scope"], r["period"]), r["eps"])} for r in doc["rows"]]
        return doc | {"inputs": doc["inputs"] | {"scan_delta": stated},
                      "rows": [r | {"eps": row.get(r["row"], r["eps"])} if (r["scope"], r["period"]) == ("world", "year")
                               else r for r in rows]}

    return made


def test_a_figure_whose_error_is_one_or_more_prints_no_bound_and_no_amount(tmp_path, monkeypatch):
    """On a term document with made-up errors. A scope's monitoring period whose ε is under 1 prints the network
    attester's amounts with that ε; one whose ε is 1 or more prints "no bound at this period" with its ε and no
    amount, in the document and in every line, column and row of the print, the output count among them. The two
    together carry the sum of both parts' errors, and print no amount where that is 1 or more though the network
    attester's own is under. The headline is the world's year's, and follows it. The compute audit's part keeps
    its amounts. The δ printed is the one the document states, whatever the command gave the calculator."""

    lb = term()
    document = lb.document
    theirs = {(r["scope"], r["period"], r["row"]): r for r in document(lb.inputs(response_ids=E.RESPONSE_IDS))["rows"]}
    monkeypatch.setattr(lb, "document", made_up(document, {
        ("datacenter", "week"): 0.25, ("datacenter", "year"): 0.995, ("world", "week"): 1.0, ("world", "year"): 3.5},
        scan_delta=0.125))
    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    week, _, dc, none, _, year = doc["periods"]
    assert [(p["eps"], p["error"]) for p in (week, dc, none, year)] == [(0.25, None), (0.995, 1.005), (1.0, None), (3.5, 3.51)]
    assert doc["certifier"]["scan_delta"]["value"] == 0.125 != float(delta(lb))
    assert doc["value"] == pytest.approx(YEAR_GB) and year["compute_audit"]["GB"] == doc["value"]

    def amounts(p: dict) -> list:
        """A scope's monitoring period's figures of the network attester, then of the two together."""

        both = [t for s in p["together"].values() for t in (s, s["with_set_time_card"])]
        count = p["output_count"]
        return [[t["certifier_GB"] for t in both] + [count["GB"], count["bits"], count["bits_a_request"]], [t["GB"] for t in both]]

    # under 1: the network attester's amounts, the calculator's; with no rows of record there is nothing to sum
    for p in (week, dc):
        assert amounts(p)[0][:2] == [theirs[p["compute"], p["period"], "scan"]["term"][dj]["GB"] for dj in ("2", "1")]
        assert None not in amounts(p)[0] and set(amounts(p)[1]) == {None}
    assert dc["compute_audit"]["GB"] == pytest.approx(13707.9, abs=0.05)  # 0.995 + 0.01 is 1 or more: no sum
    # 1 or more, 1 itself among them: no amount of the network attester's, the output count's among them, and no sum
    for p in (none, year):
        assert set(amounts(p)[0]) == set(amounts(p)[1]) == {None}
        assert p["output_count"]["at_zero_in"] == [] and p["gpu_seconds"] and p["requests"]
    assert [sorted(r.keys() & set(E.AMOUNTS)) for r in doc["certifier"]["rows"]] == [[]] * 7
    assert {r["eps"] for r in doc["certifier"]["rows"]} == {3.5} and doc["certifier"]["rows"][0]["row"] == "scan"
    assert doc["certifier"]["default_class"] == {"response_ids": 4096, "eps": 3.5, "value": None, "unit": "GB"}

    text = E.render(doc)
    assert "- **the compute audit's part: 14,240 GB** (modeled), 37.81 copies of the weights" in text
    assert ("- **the network attester's part: no bound at this period** (modeled, on an assumed class of at most 1,024 ids a "
            "response). Its ε over the world's year is 3.5, and at 1 or more it bounds nothing, so no amount is printed under "
            "any of its rows.\n") in text
    assert ("- **together: no bound at this period**. The two parts' errors over the world's year sum to 3.51, and at 1 or "
            "more the sum bounds nothing, so no amount is printed for the two together.\n") in text
    assert "- the output count: no bound at this period (assumed), under the network attester's ε. Until a served audit" in text
    assert ("The network attester's ε over the world's year is each execution phase's scan's δ (assumed), 0.125 (2^-3), "
            "summed over every link's execution phases, 1.155e+13 of them: 3.5. The two together fail with at most the sum "
            "of both, 3.51.") in text
    head = text.split("\n## Inputs")[0]
    assert " GB" not in "".join(line for line in head.splitlines()
                                if line.startswith(("- **the network attester's part", "- **together", "- the output count")))
    assert "with a set-time card" not in head and "its term of record" not in head
    # none of the world year's amounts of the network attester is printed, at either jitter bound or summed; the
    # design's are those of the design before its stop block, its stop class charged its whole schedule
    for figure in ("50,753,639", "26,392,406", "48,906,024", "24,453,012", "24,544,790", "50,767,879",
                   "26,406,646", "24,467,252", "59,645,644", "22.83 bits a request", "11 bits a request"):
        assert figure not in text, figure
    assert "| scan | " + "no bound at this period | " * 6 + "0.125 | 3.5 |" in text
    assert text.count("| " + "no bound at this period | " * 6 + "0.125 | 3.5 |\n") == 7
    no = "no bound at this period | "
    assert ("| world | year | 1.182e+16 | 1.778e+16 | 14,240 | 0.006406 | 0.01 | " + no * 3 + "0.125 | 3.5 | " + no * 3
            + "3.51 | no bound at this period |") in text
    assert "| world | year | " + no * 8 + "0.125 | 3.5 | 3.51 |" in text
    assert ("| world | week | 2.268e+14 | 3.411e+14 | no rows of record | none | none | " + no * 3 + "0.125 | 1 | "
            + "none | " * 3 + "none | no bound at this period |") in text
    assert "| world | week | no bound at this period | none | " + no * 3 + "none | " * 3 + "0.125 | 1 | none |" in text
    assert ("| datacenter | year | 3.154e+12 | 4.743e+12 | 13,708 | 23.12 | 0.01 | 13,536 | 7,039 | 7,039 | 0.125 | 0.995 | "
            + no * 3 + "1.005 | 6,521 |") in text
    assert ("| datacenter | year | 6,521 | no bound at this period | 13,043 | 6,546 | 6,546 | " + no * 3
            + "0.125 | 0.995 | 1.005 |") in text
    assert ("| datacenter | week | 6.048e+10 | 9.096e+10 | no rows of record | none | none | 259.6 | 135 | 135 | 0.125 | "
            "0.25 | none | none | none | none | 125.1 |") in text
    # the δ printed is the document's own on every line that carries one, and the one the command gave is nowhere
    shown = printed(text)
    assert {cell for under, cell in shown if under.startswith("scan's δ | ")} == {"0.125"} and "8.259e-20" not in text
    # every cell that is no bound is on a line that carries the error that says so
    lines = [line for line in text.splitlines() if line.startswith("|") and E.NO_BOUND in line]
    assert len(lines) == 7 + 3 + 3 and all(any(f"| {e} |" in line for e in ("3.5", "1", "1.005")) for line in lines)

    # the network attester's part is bounded and the sum is not: its amounts are printed and the sum's are not
    monkeypatch.setattr(lb, "document", made_up(document, {("world", "year"): 0.995}))
    text = E.render(E.build(store=recorded(tmp_path / "sum").fresh(), root=REPO))
    assert "- **the network attester's part: 50,753,639 GB** (modeled, on an assumed class of at most 1,024 ids" in text
    assert "its term of record, and each failing with at most ε = 0.995 (2^-0.007232) over the world's year" in text
    assert "- **together: no bound at this period**. The two parts' errors over the world's year sum to 1.005" in text
    assert "- the output count: 24,453,012 GB (assumed), 11 bits a request." in text and "- together, " not in text
    # one ε a scope and monitoring period: rows of one that carry two are refused by name
    monkeypatch.setattr(lb, "document", made_up(document, {("world", "year"): 0.5}, design=2.0))
    with pytest.raises(ValueError, match="the network attester's rows over the world year carry 2 different ε \\(0.5, 2\\), and "
                                         "this command prints one a scope and monitoring period"):
        E.build(store=recorded(tmp_path / "two").fresh(), root=REPO)
