"""`verity bench exfiltration`'s stop class's verdicts: a run not proved, a class complete, and each record refused
by name."""

from __future__ import annotations

import hashlib
import json
import re

import pytest
from exfiltration_helpers import (
    HOLDER,
    REPO,
    held_phase,
    reader,
    recorded,
    term,
    with_verdicts,
)
from verity_bench import exfiltration as E

#: the sha256 of the print and of the `--json` document on the fixtures' runs with no verdicts of record: the command's
#: output where no run carries the stop class's record, the calculator's rows as with no record read. A change
#: elsewhere that moves the print on purpose pins the new digests here
NO_VERDICTS = {"text": "8c0ac638eab787f4959a4316f351920b6d69b046025cfed4b4092496a313200c",
               "json": "94412a9c27cf341d6c45c78575cb39ae23ffe4ff6eccfa4eece613a12104d2d6"}
#: made-up verdicts: a verifier, what stands between its verdicts and proved, the run that holds the records, and each
#: verdict's own run
VERIFIER = {"module": "made_up.stop_verifier", "commit": "0123456789ab" * 3 + "0123"}
OPEN = "the two-round form's soundness is not proved"
ALL = "the stop record, the response tie, the request tie and the compute audit's second stage"
#: the made-up class the honest verdicts are on: its boundary's registration, the serve it came from, its name and its
#: rate; and another a control names, which changes a committed value and so its registration, and which is not read
CLASS = {"registration": "ab" * 64, "serve": "r20261113-000000-5e7d", "class": "w0/p4-o2", "rate": "1/1"}
CHANGED = {"registration": "cd" * 64, "serve": "r20261113-000000-ffff", "class": "w1/p4-o2-nostop", "rate": "1/2",
           "stream_registration": "ef" * 64, "stream_names_boundary": False}
ON = f"complete on class `w0/p4-o2` of serve {CLASS['serve']} at rate 1/1"


def made_up_verdict(piece: str, control: str | None = None, **more) -> dict:
    """A made-up verdict's record: the honest run's on ``piece`` accepted on `CLASS`, its stream's registration its
    own and named the boundary, or a ``control`` on it refused, naming `CHANGED`; each by `VERIFIER` and proved by
    nothing."""

    n = list(E.PIECES).index(piece) + (0 if control is None else 4 + list(E.CONTROLS).index(control) * 4)
    named = CHANGED if control else CLASS | {"stream_registration": f"{n + 1:02x}" * 64, "stream_names_boundary": True}
    return {"piece": piece, "control": control, "run": f"r20261113-000000-a{n + 1:03d}", "verifier": VERIFIER,
            "verdict": "accept" if control is None else "refuse", "proved_by": None, "open": OPEN} | named | more


def complete() -> list[dict]:
    """The four pieces accepted, and a changed id refused on the response tie and a changed `max_tokens` on the request
    tie, each by the verifier that accepted that piece."""

    return [*(made_up_verdict(p) for p in E.PIECES), made_up_verdict("response_tie", "id"),
            made_up_verdict("request_tie", "max_tokens")]


def digests(doc: dict) -> dict:
    return {"text": hashlib.sha256(E.render(doc).encode()).hexdigest(),
            "json": hashlib.sha256(json.dumps(doc, indent=1, ensure_ascii=False).encode()).hexdigest()}


def test_with_no_verdicts_of_record_the_print_is_as_it_was_byte_for_byte(tmp_path):
    """With nothing in the store carrying the stop class's record, the print and the document are byte for byte those
    pinned (`NO_VERDICTS`): the stop verdicts' reader is not asked and the calculator's rows are as with no record
    read, the design as the design before its stop block and the network attester's figure in the headline the
    `scan` row's. The output count is assumed as it was, and no sum takes it at 0, so the inputs list none."""

    doc = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert "stop_class" not in doc and E.VERDICTS not in E.render(doc)
    assert digests(doc) == NO_VERDICTS, (
        f"the print with no verdicts of record moved: {digests(doc)}; where a change elsewhere moved it on purpose, pin "
        "these in NO_VERDICTS")
    year = doc["periods"][-1]
    assert {p["output_count"]["mark"] for p in doc["periods"]} == {"assumed"}
    assert [i for i in doc["inputs"] if i["input"] == "output_count"] == [] and year["output_count"]["at_zero_in"] == []
    assert (doc["attester"]["stop"]["runs"], doc["attester"]["stop"]["why"]) == ([], list(term().NO_RECORDS.why))
    assert year["together"]["design"] == year["together"][E.OUTPUT_COUNT]
    assert doc["attester"]["rows"][0]["row"] == "scan" and doc["attester"]["rows"][0]["term"]["2"]["GB"] == pytest.approx(
        term().document(term().inputs(response_ids=E.RESPONSE_IDS))["value"])


def test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured(tmp_path, listed):
    """Every piece accepted and every control refused by the verifier that accepted its piece, none of them proved: the
    class prints `complete on class C of serve S at rate 1/1 (run, not proved: ...; Python verifier for ...)`, with
    what is open in words and no issue's id, and no line calls it proved. The controls name another class, which is
    not read.

    The output count's 0 would carry that mark, printed "run (Python verifier, not proved)" with each verdict's run
    and verifier and counted apart from measured, only where the stop verdicts' reader's answer holds an execution
    phase. It holds none: the run of record holds a class's four verdicts in the reader's own format beside these
    (`held_phase`), the reader asks the network attester's rule without a schedule, and the rule charges no class
    below a schedule it was not handed. So no sum takes the count at 0, no input carries the mark and no figure
    moves: the periods and the inputs are those of the store with no verdict. The same verdicts in a run that holds
    nothing the reader reads say the same of the class."""

    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    doc = E.build(store=with_verdicts(tmp_path, complete(), held=held_phase(listed)).store, root=REPO)
    stop = doc["stop_class"]
    said = f"{ON} (run, not proved: {OPEN}; Python verifier for {ALL})"
    assert (stop["record"], stop["runs"], stop["complete"], stop["mark"], stop["class"], stop["lacks"], stop["said"]) == (
        E.VERDICTS, [HOLDER], True, E.RUN_NOT_PROVED, CLASS, [], said)
    assert [(v["piece"], v["control"]) for v in stop["verdicts"]] == [
        ("stop_record", None), ("response_tie", None), ("response_tie", "id"), ("request_tie", None),
        ("request_tie", "max_tokens"), ("second_stage", None)]
    assert [sorted(v.keys() & set(E.CLASS_FIELDS)) for v in stop["verdicts"] if v["control"]] == [[], []]
    # the reader read the class's four verdicts and holds no execution phase: the schedule was not handed. No row is
    # named, so no input is the output count's and nothing of a monitoring period moves
    answer = doc["attester"]["stop"]
    assert (answer["held_phases"], answer["below"], answer["runs"], answer["superseded"]) == (0, False, [HOLDER], [])
    assert len(answer["verdicts"]) == 4 and answer["why"][0].endswith(reader().NO_SCHEDULE)
    assert answer["why"][2].endswith("class 0: charged its whole schedule on each of its 4 rows: the schedule was not handed")
    assert doc["inputs"] == base["inputs"] and "output_count" not in {i["input"] for i in base["inputs"]}
    assert doc["periods"] == base["periods"] and {tuple(p["output_count"]["at_zero_in"]) for p in doc["periods"]} == {()}
    assert doc["value"] == base["value"]

    text = E.render(doc)
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in E.MARKS}
    assert (marks[E.RUN_NOT_PROVED], marks["measured"], len(doc["inputs"])) == (0, 1, len(base["inputs"]))
    assert (f"of {len(doc['inputs'])} inputs, 1 is measured, {marks['modeled']} modeled, 1 simulated and "
            f"{marks['assumed']} assumed.\n") in text
    assert "run (Python verifier" not in text and "| output_count |" not in text
    assert ("No sum takes it at 0: those under `all three` and `design` hold it too, their stop class charged its whole "
            f"schedule. The stop class's verdicts of record (`{E.VERDICTS}`, {HOLDER}): {said}.\n") in text
    assert f"- the stop class: {HOLDER} (`{E.VERDICTS}`): 6 verdicts, {said}\n" in text
    assert [line[4:] for line in text.splitlines() if line.startswith("  - ")] == answer["why"]
    assert [line for line in text.splitlines() if "proved" in line.replace("not proved", "")] == []

    # the same verdicts in a run that holds nothing the reader reads: the class is complete, the reader's answer
    # holds no execution phase, and so no sum takes the count at 0 and no input carries the mark
    alone = E.build(store=with_verdicts(tmp_path / "alone", complete()).store, root=REPO)
    assert (alone["stop_class"]["complete"], alone["stop_class"]["mark"]) == (True, E.RUN_NOT_PROVED)
    assert alone["inputs"] == base["inputs"] and alone["periods"] == base["periods"]
    assert alone["attester"]["stop"] == base["attester"]["stop"] | {"runs": [HOLDER], "why": [
        *term().NO_RECORDS.why,
        (f"{HOLDER} is not read: it holds none of `out/*-verdicts.jsonl`, `out/registrations.jsonl`, "
        "`out/*-controls.jsonl`, `out/serve-registrations.jsonl`")]}
    text = E.render(alone)
    assert ("No sum takes it at 0: those under `all three` and `design` hold it too, their stop class charged its whole "
            f"schedule. The stop class's verdicts of record (`{E.VERDICTS}`, {HOLDER}): {said}.\n") in text
    assert ("is charged its whole schedule, by the answer of the stop verdicts' reader (`stop_verdicts.of_runs`) on the "
            f"runs that carry `{E.VERDICTS}` ({HOLDER}). ") in text
    assert "run (Python verifier" not in text and "| output_count |" not in text

    # a piece proved and the others not: still run, not proved, and the Python verifier is the others'
    mixed = [v | {"proved_by": "Made.Up.StopRecordSound", "open": None} if v["piece"] == "stop_record" else v
             for v in complete()]
    stop = E.build(store=with_verdicts(tmp_path / "mixed", mixed).store, root=REPO)["stop_class"]
    assert (stop["mark"], stop["said"]) == (E.RUN_NOT_PROVED, (f"{ON} (run, not proved: {OPEN}; Python verifier "
                                            "for the response tie, the request tie and the compute audit's second stage)"))


def test_proved_by_on_every_piece_makes_it_measured(tmp_path, listed):
    """Only where every piece's record names the Lean guarantee that proves its verifier is the class's mark
    measured, and then nothing of the command's prints as not proved but the scan's accept, in the `scan` row's own
    line, and the reader's own line that the serve its verdicts name is run, not proved, which the command prints
    as the reader says it. The output count's 0 would be measured with the class, and counted with the measured
    inputs, only where the stop verdicts' reader's answer holds an execution phase. It holds none, the schedule
    not handed, though the run of record holds a class's four verdicts in the reader's own format (`held_phase`):
    so no input is the output count's and nothing is measured more."""

    proved = [v | {"proved_by": f"Made.Up.{v['piece']}", "open": None} for v in complete()]
    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    doc = E.build(store=with_verdicts(tmp_path, proved, held=held_phase(listed)).store, root=REPO)
    stop = doc["stop_class"]
    assert (stop["complete"], stop["mark"], stop["said"]) == (True, "measured", (
        f"{ON} (measured: proved by `Made.Up.stop_record`, `Made.Up.response_tie`, `Made.Up.request_tie` and "
        "`Made.Up.second_stage`)"))
    answer = doc["attester"]["stop"]
    assert (answer["held_phases"], len(answer["verdicts"])) == (0, 4) and answer["why"][0].endswith(reader().NO_SCHEDULE)
    assert (doc["inputs"], doc["periods"]) == (base["inputs"], base["periods"])
    measured = sum(i["mark"] == "measured" for i in base["inputs"])
    text = E.render(doc)
    assert measured == 1 and f"- of {len(base['inputs'])} inputs, 1 is measured, " in text
    assert [line for line in text.splitlines() if "not proved" in line and term().SCAN_ACCEPT not in line] == [
        f"  - {reader().SERVE_NAMED}"]
    assert "| output_count |" not in text and "taken at 0 in the sums" not in text
    # with nothing in the reader's format the class is as proved and no sum takes the count at 0: nothing is measured more
    alone = E.build(store=with_verdicts(tmp_path / "alone", proved).store, root=REPO)
    assert (alone["stop_class"]["mark"], alone["inputs"], alone["periods"]) == ("measured", base["inputs"], base["periods"])


@pytest.mark.parametrize("field", E.VERDICT_FIELDS)
def test_a_verdicts_record_missing_a_field_is_refused_by_its_name(tmp_path, field):
    lacking = {k: v for k, v in made_up_verdict("stop_record").items() if k != field}
    with pytest.raises(ValueError, match=f"^here lacks `{field}`: a verdict's record holds `piece`, `control`, `run`, "
                                         "`verifier`, `verdict`, `proved_by`, `open`"):
        E.verdict(lacking, "here")
    s = with_verdicts(tmp_path, [*complete()[:3], lacking])
    with pytest.raises(ValueError, match=f"^{HOLDER}'s `{E.VERDICTS_FILE}` line 4 lacks `{field}`"):
        E.build(store=s.store, root=REPO)


def test_a_verdicts_record_whose_field_holds_what_it_cant_is_refused(tmp_path):
    v = made_up_verdict("stop_record")
    with pytest.raises(ValueError, match="^here lacks `verifier.commit`: a verdict's verifier is its module and the commit"):
        E.verdict(v | {"verifier": {"module": "m"}}, "here")
    with pytest.raises(ValueError, match="^here lacks `verifier.module`, `verifier.commit`"):
        E.verdict(v | {"verifier": "m"}, "here")
    for change, why in (({"piece": "stop"}, "`piece` is one of stop_record, response_tie, request_tie, second_stage"),
                        ({"control": "length"}, "`control` is null on the honest run, else one of id, max_tokens"),
                        ({"run": ""}, "`run` is the id of the run that produced the verdict"),
                        ({"verdict": "pass"}, "`verdict` is one of accept, not complete, refuse, withheld"),
                        ({"verdict": "withheld", "proved_by": "G", "open": None},
                         "`open` states what the verifier withheld its accept for, on a `withheld` verdict"),
                        ({"verdict": "withheld", "open": "PRF-456"}, "`open` names an issue, .*"),
                        ({"proved_by": ""}, "`proved_by` is null, or the Lean guarantee whose proof covers the verifier"),
                        ({"open": None}, ("`open` states what stands between the verdict and proved, while `proved_by` is "
                                         "null")),
                        *(({"open": issue}, ("`open` names an issue, and the print carries no issue's id: state what is open "
                                            "in words \\(e.g. `the two-round form's soundness is not proved`\\), and put the "
                                            "issue on the board")) for issue in ("PRF-419", "waits on PRF-419", "NET-386 and more"))):
        with pytest.raises(ValueError, match=f"^here holds .*: {why}$"):
            E.verdict(v | change, "here")
    assert E.verdict(v | {"proved_by": "G", "open": None, "verifier": VERIFIER | {"more": 1}}, "here")["verifier"] == VERIFIER
    assert E.verdict(v | {"open": "SHA-512's collision resistance is assumed"}, "here")["open"].startswith("SHA-512")
    with pytest.raises(ValueError, match="^here is a verdict's record, one JSON object, not list"):
        E.verdict([v], "here")
    s = with_verdicts(tmp_path, json.dumps(v) + "\n\n{not json\n")
    with pytest.raises(ValueError, match=f"^{HOLDER}'s `{E.VERDICTS_FILE}` line 3 is no JSON object"):
        E.build(store=s.store, root=REPO)


def test_without_both_negative_controls_refused_the_class_is_not_complete_and_says_what_is_missing(tmp_path):
    """The class is complete only with every piece accepted, a changed id and a changed `max_tokens` each refused, and
    each refusal by the verifier that accepted the honest run's verdict on its piece. Short of that the inputs print as
    they do with no verdicts, and the output count's line names what is missing. A `withheld` verdict is neither an
    accept nor a refusal: a withheld piece or control keeps the class short, printed `withheld: <open>`, and a control
    on a withheld piece refused by that piece's verifier counts as refused and prints nothing more."""

    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    honest = complete()[:4]
    other = VERIFIER | {"commit": "f" * 40}
    for name, verdicts, lacks in (
            ("none", honest, ["no negative control with a changed id", "no negative control with a changed `max_tokens`"]),
            ("one", [*honest, made_up_verdict("response_tie", "id")],
             ["no negative control with a changed `max_tokens`"]),
            ("accepted", [*complete()[:5], made_up_verdict("request_tie", "max_tokens", verdict="accept")],
             [("the control with a changed `max_tokens` on the request tie is `accept` on r20261113-000000-a011, not "
              "refused")]),
            ("other", [*complete()[:5], made_up_verdict("request_tie", "max_tokens", verifier=other)],
             [("the control with a changed `max_tokens` is refused on r20261113-000000-a011 by `made_up.stop_verifier` at "
              f"{'f' * 40}, not by a verifier that accepted the honest run's verdict on the request tie")]),
            ("unaccepted", [*complete()[:2], made_up_verdict("request_tie", verdict="not complete"), *complete()[3:]],
             ["the request tie is `not complete` on r20261113-000000-a003",
              ("the control with a changed `max_tokens` is refused on r20261113-000000-a011 by `made_up.stop_verifier` at "
              f"{VERIFIER['commit']}, not by a verifier that accepted the honest run's verdict on the request tie")]),
            ("withheld", [*complete()[:3], made_up_verdict("second_stage", verdict="withheld"), *complete()[4:]],
             [f"the compute audit's second stage on r20261113-000000-a004, withheld: {OPEN}"]),
            ("withheld control", [*complete()[:5], made_up_verdict("request_tie", "max_tokens", verdict="withheld")],
             [(f"the control with a changed `max_tokens` on the request tie on r20261113-000000-a011, withheld: {OPEN}, "
              "not refused")]),
            ("withheld, its controls refused", [*complete()[:3], made_up_verdict("second_stage", verdict="withheld"),
                                                made_up_verdict("second_stage", "id"),
                                                made_up_verdict("second_stage", "max_tokens")],
             [f"the compute audit's second stage on r20261113-000000-a004, withheld: {OPEN}"]),
            ("withheld, a control refused by another", [*complete()[:3], made_up_verdict("second_stage", verdict="withheld"),
                                                        made_up_verdict("second_stage", "id", verifier=other),
                                                        made_up_verdict("second_stage", "max_tokens")],
             [f"the compute audit's second stage on r20261113-000000-a004, withheld: {OPEN}",
              ("the control with a changed id is refused on r20261113-000000-a008 by `made_up.stop_verifier` at "
              f"{'f' * 40}, not by a verifier that accepted the honest run's verdict on the compute audit's second stage")]),
            ("missing", complete()[1:], ["no verdict on the stop record"])):
        doc = E.build(store=with_verdicts(tmp_path / name, verdicts).store, root=REPO)
        stop = doc["stop_class"]
        assert (stop["complete"], stop["mark"], stop["lacks"]) == (False, None, lacks), name
        assert doc["inputs"] == base["inputs"], name
        text = E.render(doc)
        said = "not complete: " + "; ".join(lacks)
        assert stop["said"] == said and f"(`{E.VERDICTS}`, {HOLDER}): {said}.\n" in text, name
        assert "complete (run" not in text and "run (Python verifier" not in text, name
        assert text.split("\n## Inputs")[1] == E.render(base).split("\n## Inputs")[1], name


def test_four_verdicts_complete_a_class_only_on_one_class_at_the_rows_rate(tmp_path):
    """The four honest verdicts complete a class only where each names it with its verifier's check that its stream's
    registration names the boundary recorded true, and they agree on the registration, the serve, the class and the
    rate, at the row's 1/1. Otherwise the print names the verdict that lacks a field or records a failed check, or the
    field they disagree on with each value's pieces, and claims nothing: the inputs print as with no verdicts. A
    control is held to its piece and its verifier alone: one naming another class, or none, leaves the class
    complete."""

    base = E.build(store=recorded(tmp_path / "base", planned=False).store, root=REPO)
    ab, cd = "ab" * 64, "cd" * 64

    def honest(change: dict[str, dict], drop: dict[str, tuple[str, ...]] = {}) -> list[dict]:  # noqa: B006
        return [{k: x for k, x in (v | change.get(v["piece"], {})).items() if k not in drop.get(v["piece"], ())}
                if v["control"] is None else v for v in complete()]

    three = "the stop record, the response tie and the request tie"
    for name, verdicts, lacks in (
            ("lacks", honest({}, {"stop_record": ("registration", "serve"), "request_tie": ("stream_names_boundary",)}),
             ["the stop record's verdict on r20261113-000000-a001 lacks `registration`, `serve`",
              "the request tie's verdict on r20261113-000000-a003 lacks `stream_names_boundary`"]),
            ("unchecked", honest({"response_tie": {"stream_names_boundary": False}}),
             [(f"the response tie's verdict on r20261113-000000-a002 records that its stream's registration, `{'02' * 64}`, "
              "doesn't name the class's boundary (`stream_names_boundary` false)")]),
            ("registration", honest({"second_stage": {"registration": cd}}),
             [(f"the honest verdicts disagree on `registration`: `{ab}` ({three}) and `{cd}` (the compute audit's second "
              "stage)")]),
            ("serve", honest({"stop_record": {"serve": "r20261113-000000-ffff"}}),
             [("the honest verdicts disagree on `serve`: `r20261113-000000-ffff` (the stop record) and "
              "`r20261113-000000-5e7d` (the response tie, the request tie and the compute audit's second stage)")]),
            ("class", honest({"request_tie": {"class": "w0/p4-o2-nostop"}, "second_stage": {"class": "w1/p4-o2"}}),
             [("the honest verdicts disagree on `class`: `w0/p4-o2` (the stop record and the response tie), "
              "`w0/p4-o2-nostop` (the request tie) and `w1/p4-o2` (the compute audit's second stage)")]),
            ("rates", honest({"second_stage": {"rate": "1/2"}}),
             [f"the honest verdicts disagree on `rate`: `1/1` ({three}) and `1/2` (the compute audit's second stage)"]),
            ("rate", honest({p: {"rate": "1/2"} for p in E.PIECES}),
             ["the honest verdicts are at rate 1/2, and the row assumes 1/1: every stop and tie block proved"])):
        doc = E.build(store=with_verdicts(tmp_path / name, verdicts).store, root=REPO)
        stop = doc["stop_class"]
        assert (stop["complete"], stop["mark"], stop["class"], stop["lacks"]) == (False, None, None, lacks), name
        assert doc["inputs"] == base["inputs"], name
        said = "not complete: " + "; ".join(lacks)
        assert stop["said"] == said and f"(`{E.VERDICTS}`, {HOLDER}): {said}.\n" in E.render(doc), name

    bare = [v if v["control"] is None else {k: x for k, x in v.items() if k not in E.CLASS_FIELDS} for v in complete()]
    odd = [v if v["control"] is None else v | {"class": "neither", "rate": "2/4", "registration": ""} for v in complete()]
    for name, verdicts in (("bare controls", bare), ("odd controls", odd)):
        stop = E.build(store=with_verdicts(tmp_path / name, verdicts).store, root=REPO)["stop_class"]
        assert (stop["complete"], stop["class"], stop["lacks"]) == (True, CLASS, []), name


def test_an_honest_verdicts_class_field_holding_what_it_cant_is_refused(tmp_path):
    """A class field an honest verdict holds is refused by its name where it holds what that field can't: a digest or a
    serve that is no text, a check that is not true or false, a class not `w<window>/p<P>-o<O>[-nostop]`, a rate not
    `NUM/DEN` in lowest terms. A control's are not read, so not refused."""

    v = made_up_verdict("stop_record")
    for change, why in (({"registration": ""}, "the digest of the class's boundary record"),
                        ({"stream_registration": None}, "the digest of its own stream's registration"),
                        ({"stream_names_boundary": "true"}, ("true or false, the verifier's recorded check that its "
                                                            "stream's registration names the boundary")),
                        ({"serve": 7}, "the store id of the served run the class came from"),
                        *(({"class": c}, "`w<window>/p<P>-o<O>`, with `-nostop` for a class with no stop block")
                          for c in ("p4-o2", "w0/p4-o2-stop", "wA/p4-o2", "w0/p4")),
                        *(({"rate": r}, "`NUM/DEN` in lowest terms, the rate the registration record states")
                          for r in ("2/4", "1", "0/1", "1/0", 1, "1/1 "))):
        k, = change
        with pytest.raises(ValueError, match=f"^here holds `{k}` {re.escape(repr(change[k]))}: `{k}` is {re.escape(why)}$"):
            E.verdict(v | change, "here")
    assert E.verdict(v | {"class": "w12/p4-o2-nostop", "rate": "3/7"}, "here")["rate"] == "3/7"
    control = made_up_verdict("request_tie", "max_tokens") | {"class": "p4", "rate": "2/4"}
    assert E.verdict(control, "here").keys() == set(E.VERDICT_FIELDS)
    with pytest.raises(ValueError, match=f"^{HOLDER}'s `{E.VERDICTS_FILE}` line 1 holds `rate` '2/4'"):
        E.build(store=with_verdicts(tmp_path, [v | {"rate": "2/4"}, *complete()[1:]]).store, root=REPO)


def test_a_verdict_two_runs_hold_or_a_superseded_run_is_refused_by_name(tmp_path, monkeypatch):
    """The runs the stop verdicts' reader is handed are the record's runs as this command lists them, each one the
    print names as read or as refused, and its superseded ones are those this command leaves out as superseded: one
    list, read once."""

    handed, of_runs = [], reader().of_runs
    monkeypatch.setattr(reader(), "of_runs", lambda store, runs, superseded=(): handed.append(
        (list(runs), list(superseded))) or of_runs(store, runs, superseded))
    s = with_verdicts(tmp_path, complete())
    again = s.label(s.ran("r20261113-000000-a100", complete()[:1], name=E.VERDICTS_FILE), value=E.VERDICTS)
    with pytest.raises(ValueError, match=f"^{HOLDER} and {again} both hold a verdict on the stop record and neither is "
                                         "superseded: label the one the other replaces .*superseded_by NEWER"):
        E.build(store=s.store, root=REPO)
    assert handed == []
    s.label(again, E.SUPERSEDED_BY, HOLDER)
    doc = E.build(store=s.store, root=REPO)
    assert doc["stop_class"]["complete"] and {x["target"]: x["why"] for x in doc["refused"]} == {
        again: f"superseded by {HOLDER}"}
    assert f"- {again} (`{E.VERDICTS}`): superseded by {HOLDER}" in E.render(doc)
    answer = doc["attester"]["stop"]
    assert handed == [([HOLDER, again], [again])] == [(answer["runs"], answer["superseded"])]
    assert answer["runs"] == sorted([*doc["stop_class"]["runs"], *(x["target"] for x in doc["refused"])])
    assert (f"on the runs that carry `{E.VERDICTS}` ({HOLDER}, {again}, of which superseded: {again}). ") in E.render(doc)
    with pytest.raises(ValueError, match=f"^{HOLDER} holds two verdicts on the request tie, the control with a changed "
                                         "`max_tokens`"):
        E.build(store=with_verdicts(tmp_path / "twice", [*complete(), complete()[-1]]).store, root=REPO)
    empty = recorded(tmp_path / "empty", planned=False)
    empty.label(empty.ran(HOLDER, [{}], name="out/other.jsonl"), value=E.VERDICTS)
    doc = E.build(store=empty.store, root=REPO)
    assert doc["refused"] == [{"target": HOLDER, "record": E.VERDICTS, "why": f"holds no verdicts (`{E.VERDICTS_FILE}`)"}]
    assert doc["stop_class"]["runs"] == [] and not doc["stop_class"]["complete"]
    assert f"(`{E.VERDICTS}`, no run read): not complete: no verdict on the stop record; " in E.render(doc)
    # a run this command reads no verdict from is handed to the reader all the same: it carries the record
    assert handed[1:] == [([HOLDER], [])] == [(doc["attester"]["stop"]["runs"], doc["attester"]["stop"]["superseded"])]
