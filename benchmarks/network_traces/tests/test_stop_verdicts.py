"""The stop class's verdict records as the calculator's rows read them (`stop_verdicts`): a head is the
registration's own, held to the digest its certificate names; a record is its verifier's, and which of the four
verdicts it is is its verdict's format; the answer counts the execution phases held, those of a head read in which
every class that holds a row has its four verdicts by the network certifier's rule, each on that head; every other
execution phase is charged its whole schedule, so one left out of a run's files charges what it would refused;
a registration that doesn't read, or no record at all, holds none, with a line that says why; each verdict read is
handed on under the mark of a run that is no proof; and the run files are read from the store as the harness reads
it."""

from __future__ import annotations

import fnmatch
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import connection_rows_run as CR  # noqa: E402
import leak_bounds as LB  # noqa: E402
import stop_verdicts as SV  # noqa: E402
from verity.protocols.certifier.verifier import class_charge as CC  # noqa: E402
from verity.protocols.certifier.verifier import connection_rows as C  # noqa: E402
from verity.protocols.certifier.verifier import content_rows as CT  # noqa: E402
from verity.protocols.certifier.verifier.certificate import Certificate, Direction  # noqa: E402
from verity.protocols.certifier.verifier.warden import WardenParams  # noqa: E402

LINK = "responses"
ZEROS = bytes(64)
TIE = "verity-network-certifier/response-tie-verdict/v1"
#: a format for each of the four verdicts: the response tie's own, and a stand-in for each verdict whose verifier's
#: format the reader doesn't list
FORMATS = {kind: TIE if kind == CC.RESPONSE_TIE else f"a format of the {kind}" for kind in CC.VERDICTS}


@pytest.fixture(autouse=True)
def listed(monkeypatch):
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in FORMATS.items()})


def schedule() -> WardenParams:
    """The lengths rows' cost runner's own schedule, with its content classes."""
    args = CR.parser().parse_args(["cost", "--out", "x"])
    return WardenParams(cell_bytes=args.cell_bytes, slots=args.slots, lanes=args.lanes, window=args.window,
                        every=args.every, delay=args.delay, ingress_buckets=args.delay,
                        ingress_cells=args.ingress_cells, egress_cells=args.egress_cells, lengths=1,
                        content=tuple(args.classes))


def head(window: int, *counts: int, parameters: bytes = ZEROS) -> C.Head:
    listed = [("lengths", 4, 8), ("fails", 4, 8)] + [(name, n, 16) for i, n in enumerate(counts) for name in CT.names(i)]
    return C.Head(LINK, window, parameters, tuple(C.Entry(name, rows, size, ZEROS, ZEROS) for name, rows, size in listed))


def registration(of: C.Head) -> dict:
    """A registration, in the verifier's format, of a head, with a certificate that names the head's digest."""
    return {"format": C.REGISTRATION, "link": of.link, "window": of.window, "parameters": of.parameters.hex(),
            "word_bits": C.WORD_BITS, "row_schema": C.SCHEMA,
            "tables": [{"name": e.name, "rows": e.rows, "words": e.row_bytes * 8 // C.WORD_BITS,
                        "domain": e.domain.hex(), "root": e.root.hex()} for e in of.tables],
            "certificate": {"key": "", "digest": of.digest().hex(), "signature": ""}}


def four(of: C.Head, cls: int = 0, proved_by: object = None, **over: object) -> list[dict]:
    """The four verdicts' records on a class of a head, as their verifiers write them: each accepted, complete and
    naming the head, but for `over`, the verdict's fields."""
    return [{CC.RUN: f"run of the {kind}", CC.VERIFIER: {"module": f"the {kind}'s verifier", "commit": "a commit"},
             CC.VERDICT: {SV.FORMAT: FORMATS[kind], CC.LINK: of.link, CC.WINDOW: of.window, CC.CLASS: cls,
                          CC.HEAD: of.digest().hex(), "registration": "a digest", CC.ACCEPTED: True,
                          CC.COMPLETE: True, "codes": []} | over,
             CC.PROVED_BY: proved_by} for kind in CC.VERDICTS]


def by_verdict(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(SV.FORMATS[r[CC.VERDICT][SV.FORMAT]], []).append(r)
    return out


def test_a_head_is_the_registrations_own_and_held_to_the_digest_its_certificate_names():
    params = schedule()
    signed = C.Head(LINK, 7, params.digest, tuple(C.Entry(name, rows, size, ZEROS, bytes([i]) * 64)
                                                  for i, (name, rows, size) in enumerate(C.tables_of(params))))
    cert = Certificate(b"a key", LINK, Direction.EGRESS, 7, signed.digest(), b"a signature")
    doc = json.loads(json.dumps(C.registration(signed, cert, params)))
    assert SV.head_of(doc) == signed and SV.classes(signed) == list(range(len(CR.CLASSES)))
    assert SV.head_of(registration(head(3, 5, 0))) == head(3, 5, 0)
    forged = doc | {"tables": [t | {"rows": t["rows"] + 1} if t["name"] == "request.0" else t for t in doc["tables"]]}
    with pytest.raises(ValueError, match="the registration's certificate doesn't name its head's digest"):
        SV.head_of(forged)
    for off in (doc | {"format": "another"}, {k: v for k, v in doc.items() if k != "certificate"},
                doc | {"parameters": "zz"}, doc | {"tables": [{}]}, [], None):
        with pytest.raises(ValueError, match="a registration of connection rows doesn't read"):
            SV.head_of(off)


#: the answer's first line where some execution phase read is held
EVERY_OTHER = (": each class that holds a row in one of them has its 4 verdicts, accepted and complete. Every other "
               "execution phase of a monitoring period is charged its whole schedule, one no registration was read "
               "for among them")


def test_an_execution_phase_is_held_only_with_the_verdicts_of_every_class_that_holds_a_row_in_it():
    first, second = head(3, 5, 0), head(4, 2, 0)
    both = [registration(first), registration(second)]
    got = SV.stop(both, four(first) + four(second))
    assert got.held == 2 and got.why == (
        f"2 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}",
        f"the link {LINK}, execution phase 3, {CC.charge(first, 0, by_verdict(four(first))).why()}",
        f"the link {LINK}, execution phase 3, class 1: a count of 0, so no connection is charged and no verdict is owed",
        f"the link {LINK}, execution phase 4, {CC.charge(second, 0, by_verdict(four(second))).why()}",
        f"the link {LINK}, execution phase 4, class 1: a count of 0, so no connection is charged and no verdict is owed")
    assert [v["verdict"] for v in got.verdicts] == list(CC.VERDICTS) * 2
    assert all(v["mark"] == LB.RUN_NOT_PROVED == "run, not proved" and
               set(v) == {"verdict", *CC.READ, CC.RUN, CC.VERIFIER, CC.PROVED_BY, "mark"} for v in got.verdicts)
    tie = got.verdicts[1]
    assert (tie[CC.RUN], tie[CC.VERIFIER], tie[CC.HEAD], tie[CC.ACCEPTED], tie[CC.COMPLETE], tie[CC.WINDOW]) == (
        "run of the response tie", {"module": "the response tie's verifier", "commit": "a commit"},
        first.digest().hex(), True, True, 3)
    # a proof named in a record changes no mark here
    proved = SV.stop(both[:1], four(first, proved_by="a theorem"))
    assert proved.held == 1 and {v["mark"] for v in proved.verdicts} == {LB.RUN_NOT_PROVED}
    # an execution phase without its tie stays at its whole schedule, and the other is held
    for records, lack in ((four(first) + four(second)[:1] + four(second)[2:], "the response tie is missing"),
                          (four(first) + four(second, complete=False)[1:2] + four(second)[:1] + four(second)[2:],
                           "the response tie is not complete (run of the response tie)"),
                          (four(first), "the stop record is missing; the response tie is missing; the certified read "
                                        "is missing; the compute audit is missing")):
        got = SV.stop(both, records)
        assert got.held == 1 and got.why[0] == (
            f"1 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}")
        assert got.why[3] == (f"the link {LINK}, execution phase 4, class 0: charged its whole schedule on each of its "
                              f"2 rows: {lack}")
    # with neither held, the line counts the classes that lack a verdict
    neither = SV.stop(both, four(first)[1:] + four(second)[:1])
    assert neither.held == 0 and neither.why[0] == (
        "each connection is charged its whole schedule: 2 of the 2 class(es) that hold a row in the 2 execution "
        "phase(s) read lack a verdict, one in each execution phase that holds a row")
    # a second class that holds a row owes its own four
    wide = head(3, 5, 1)
    assert SV.stop([registration(wide)], four(wide)).held == 0
    assert SV.stop([registration(wide)], four(wide) + four(wide, 1)).held == 1
    # verdicts decided under another serve's head of the same execution phase and class count for nothing
    theirs = head(3, 5, 0, parameters=bytes([1]) * 64)
    other = SV.stop(both[:1], four(theirs))
    assert other.held == 0 and other.verdicts == () and other.why[1].endswith(
        "the stop record is not on this head (run of the stop record); the response tie is not on this head (run of "
        "the response tie); the certified read is not on this head (run of the certified read); the compute audit is "
        "not on this head (run of the compute audit)")
    # records on no head read are counted in a line and count for nothing
    stray = SV.stop(both[:1], four(first) + four(head(9, 5)) + four(theirs))
    assert stray.held == 1 and len(stray.verdicts) == 4 and stray.why[-1] == (
        "8 verdict record(s) read are on no head read, or on no class of one that holds a row, and count for nothing")
    # what was not read is carried
    assert SV.stop(both[:1], four(first), ["r9 is not read"]).why[-1] == "r9 is not read"


def test_an_execution_phase_left_out_of_a_runs_files_charges_what_it_would_refused():
    """A row's execution phases are its monitoring period's, never the registrations a run kept: three execution
    phases with their verdicts hold three; with the middle one's registration left out, with or without its
    verdicts, the answer holds two, as it does with that execution phase there and its tie refused, and the
    calculator's rows are the same either way, at every scope, monitoring period and jitter bound. With every
    registration left out they are the rows of no record at all. A registration nobody signed, with no verdict on
    its head, holds nothing, whatever counts it states."""
    heads = [head(window, 5, 0) for window in (3, 4, 5)]
    registrations = [registration(h) for h in heads]
    records = [r for h in heads for r in four(h)]
    middle = heads[1]
    assert SV.stop(registrations, records).held == 3
    left_out = SV.stop(registrations[:1] + registrations[2:], records)
    without = SV.stop(registrations[:1] + registrations[2:], four(heads[0]) + four(heads[2]))
    refused = SV.stop(registrations, four(heads[0]) + four(heads[2]) + [
        r | {CC.VERDICT: r[CC.VERDICT] | {CC.ACCEPTED: False}} if r[CC.RUN] == "run of the response tie" else r
        for r in four(middle)])
    assert left_out.held == without.held == refused.held == 2
    assert left_out.why[0] == f"2 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}"
    assert refused.why[0] == f"2 of the 3 execution phase(s) read are charged below their schedule{EVERY_OTHER}"
    assert refused.why[3] == (f"the link {LINK}, execution phase 4, class 0: charged its whole schedule on each of its "
                              "5 rows: the response tie is refused (run of the response tie)")
    inp = LB.inputs()
    whole, held = LB.table(inp), LB.table(inp, left_out)
    assert held == LB.table(inp, without) == LB.table(inp, refused) != whole
    assert LB.document(inp, stop=left_out)["rows"] == LB.document(inp, stop=refused)["rows"]
    by = {(r.scope.name, r.period, r.name): r for r in whole}
    for r in held:
        bare = by[r.scope.name, r.period, r.name]
        for dj in LB.JITTERS:
            share = 1 - Fraction(2) / r.phases if LB.ROWS[r.name].verdicts else 1
            assert r.terms[dj] == bare.terms[dj]._replace(schedules=share * bare.terms[dj].schedules)
    # every registration left out is no record at all, as far as a row is charged
    assert LB.table(inp, SV.stop([], records)) == whole
    # an unsigned registration with no verdict on its head holds nothing: not with counts, and not with none
    forged = [registration(head(window, *counts)) for window, counts in ((6, (5, 0)), (7, (0, 0)), (8, ()))]
    assert SV.stop(registrations + forged, records).held == 3 and SV.stop(forged, records).held == 0
    assert LB.table(inp, SV.stop(forged, [])) == whole


def test_a_verdict_whose_format_is_not_listed_reads_as_missing(monkeypatch):
    monkeypatch.undo()
    assert SV.FORMATS == {TIE: CC.RESPONSE_TIE}
    first = head(3, 5, 0)
    got = SV.stop([registration(first)], four(first))
    assert got.held == 0 and [v["verdict"] for v in got.verdicts] == [CC.RESPONSE_TIE]
    assert got.why[1] == (f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each of its 5 "
                          "rows: the stop record is missing; the certified read is missing; the compute audit is "
                          "missing")
    assert got.why[-1] == "3 verdict record(s) read are of a format listed for none of the 4 verdicts, and count for nothing"
    for form in (..., None, 7, ["a list"]):
        tie = four(first)[1]
        tie[CC.VERDICT] = {k: v for k, v in (tie[CC.VERDICT] | {SV.FORMAT: form}).items() if v is not ...}
        assert SV.stop([registration(first)], [tie]).verdicts == ()


def test_anything_less_is_the_whole_schedule_with_a_line_that_says_why():
    assert SV.stop([], []) == LB.NO_RECORDS
    assert SV.stop([], [], ["r9 is not read"]) == LB.Stop(0, (*LB.NO_RECORDS.why, "r9 is not read"))
    whole = "each connection is charged its whole schedule: "
    first = head(3, 5)
    for registrations, records, why in (
            ([], four(first), "4 verdict record(s) were read and no registration (`out/registrations.jsonl`), so no "
                              "class's count is known"),
            ([registration(head(3, 0, 0))], four(head(3, 0, 0)), "no class holds a row in the 1 execution phase(s) read"),
            ([registration(first), registration(head(3, 6))], four(first),
             f"two heads are given for execution phase 3 of the link {LINK}"),
            ([{"format": C.REGISTRATION}], four(first), "a registration of connection rows doesn't read: "
                                                        "KeyError('word_bits')"),
            ([registration(first)], [*four(first), "accept"],
             f"a verdict record is an object with the fields {', '.join(CC.FIELDS)}, its verdict an object, not "
             "'accept'"),
            ([registration(first)], [four(first)[0] | {CC.VERDICT: "accept"}],
             f"a verdict record is an object with the fields {', '.join(CC.FIELDS)}, its verdict an object, not "
             f"{four(first)[0] | {CC.VERDICT: 'accept'}!r}")):
        got = SV.stop(registrations, records)
        assert (got.held, got.verdicts, got.why[0]) == (0, (), whole + why), why
    tampered = registration(first)
    tampered["tables"][2]["rows"] = 0
    got = SV.stop([tampered], four(first))
    assert got.held == 0 and got.why[0] == whole + "the registration's certificate doesn't name its head's digest"
    uneven = C.Head(LINK, 3, ZEROS, tuple(C.Entry(e.name, 6 if e.name == "request.0" else e.rows, e.row_bytes, e.domain,
                                                  e.root) for e in first.tables))
    got = SV.stop([registration(uneven)], four(uneven))
    assert got.held == 0 and got.why[0].startswith(whole + "the head states [5, 6] rows for class 0's three tables")
    # the same head given twice is one head, and one execution phase
    assert SV.stop([registration(first), registration(first)], four(first)).held == 1


class Store:
    """The evidence store as `read` uses it: each run's files, by their path in the run."""

    def __init__(self, tmp: Path, runs: dict[str, dict[str, str] | None]):
        self.tmp, self.runs = tmp, runs

    def get_attempt(self, run: str) -> dict | None:
        if run not in self.runs:
            return None
        return {"outputs": {"run_files": run} if self.runs[run] is not None else {}}

    def listing(self, files: str, paths: list[str]) -> list[dict]:
        return [{"path": p} for p in self.runs[files] if any(fnmatch.fnmatchcase(p, pat) for pat in paths)]

    def fetch(self, files: str, paths: list[str]) -> str:
        at = self.tmp / files
        for name in paths:
            (at / name).parent.mkdir(parents=True, exist_ok=True)
            (at / name).write_text(self.runs[files][name])
        return str(at)


def jsonl(lines: list) -> str:
    return "".join(json.dumps(x) + "\n" for x in lines)


def test_the_run_files_are_read_from_the_store_and_no_run_is_the_whole_schedule(tmp_path):
    assert (SV.RECORD, SV.RECORDS_FILES, SV.REGISTRATIONS_FILE) == (
        "exfiltration-glm53-stop-verdicts", "out/*-verdicts.jsonl", "out/registrations.jsonl")
    first = head(3, 5, 0)
    registrations, records = [registration(first)], four(first)
    stop_record, tie, read, audit = records
    store = Store(tmp_path, {
        "r1": {"out/stop-verdicts.jsonl": jsonl([stop_record]) + "\n", "out/certified-read-verdicts.jsonl": jsonl([read]),
               "out/compute-audit-verdicts.jsonl": jsonl([audit]), "verdicts.jsonl": "not read"},
        "r2": {"out/response-tie-verdicts.jsonl": jsonl([tie]), SV.REGISTRATIONS_FILE: jsonl(registrations),
               "out/another.json": "{}"},
        "r3": {"out/another.json": "{}"}, "r4": None})
    assert SV.read(store, ["r1", "r2"]) == (registrations, [read, audit, stop_record, tie], [])
    neither = "holds neither `out/*-verdicts.jsonl` nor `out/registrations.jsonl`"
    got = SV.of_runs(store, ["r1", "r2", "r3", "r4", "r5"])
    assert got.held == 1 and len(got.verdicts) == 4 and got == SV.stop(registrations, records, [
        f"r3 is not read: it {neither}", "r4 is not read: it kept no run files",
        "r5 is not read: it is not in the store or on its remote"])
    # the tie's run alone: one verdict of the four, so the whole schedule
    assert SV.of_runs(store, ["r2"]).held == 0
    assert SV.of_runs(store, ["r1"]).why[0].endswith("3 verdict record(s) were read and no registration "
                                                      "(`out/registrations.jsonl`), so no class's count is known")
    none = SV.of_runs(store, [])
    assert none == LB.Stop(0, (*LB.NO_RECORDS.why, "no run in the store carries "
                               "`run_of_record=exfiltration-glm53-stop-verdicts`"))
    assert SV.of_runs(store, ["r3"]) == LB.Stop(0, (*LB.NO_RECORDS.why, f"r3 is not read: it {neither}"))
    broken = Store(tmp_path, {"r6": {"out/response-tie-verdicts.jsonl": "{not json\n"}})
    got = SV.of_runs(broken, ["r6"])
    assert got.held == 0 and got.why[0].startswith("each connection is charged its whole schedule: the records of r6 "
                                                   "don't read: ")
    # the calculator's rows follow the answer: the one execution phase held is one of a node's week, the rest of
    # the week is charged its whole schedule, and what the row would be with every one held is held apart
    doc = LB.document(LB.inputs(), stop=SV.of_runs(store, ["r1", "r2"]), only=["node"], periods=["week"])
    by = {r["row"]: r for r in doc["rows"]}
    assert (doc["stop"]["held_phases"], doc["stop"]["below"]) == (1, False)
    phases, bare = by["design"]["phases"], by["design, no stop block"]["schedules"]["bits"]
    assert by["design"]["schedules"]["bits"] == pytest.approx(bare * (1 - 1 / phases), rel=1e-12) and phases > 1
    apart = {w["row"]: w for w in doc["stop"]["with_verdicts"]}
    assert set(apart) == set(doc["stop"]["rows"]) and apart["design"]["schedules"]["bits"] == 0
    whole = LB.document(LB.inputs(), stop=SV.of_runs(store, ["r2"]), only=["node"], periods=["week"])
    at = {r["row"]: r for r in whole["rows"]}
    assert at["design"]["term"] == at["design, no stop block"]["term"] and len(whole["stop"]["verdicts"]) == 1
    assert (whole["stop"]["held_phases"], whole["stop"]["with_verdicts"]) == (0, doc["stop"]["with_verdicts"])
    assert "response tie, run, not proved: accepted, complete, on class 0 in execution phase 3 of the link responses, by " \
           "run run of the response tie under the verifier the response tie's verifier at commit a commit, proved by " \
           "nothing" in LB.render(whole)
