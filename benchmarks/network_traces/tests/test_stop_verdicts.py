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


def head(window: int, *counts: int, parameters: bytes = ZEROS, domain: bytes = ZEROS) -> C.Head:
    """A head of an execution phase with a class of each count; ``domain`` is each table's, which is a session's."""
    listed = [("lengths", 4, 8), ("fails", 4, 8)] + [(name, n, 16) for i, n in enumerate(counts) for name in CT.names(i)]
    return C.Head(LINK, window, parameters, tuple(C.Entry(name, rows, size, domain, ZEROS) for name, rows, size in listed))


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
    return [{CC.RUN: f"run of the {kind}", CC.SERVE: "the serve", CC.REGISTRATION: "its registration's digest",
             CC.VERIFIER: {"module": f"the {kind}'s verifier", "commit": "a commit"},
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
    assert all(v["mark"] == LB.RUN_NOT_PROVED == "run, not proved" and set(v) == {
        "verdict", *CC.READ, CC.RUN, CC.SERVE, CC.REGISTRATION, CC.VERIFIER, CC.PROVED_BY, "mark"} for v in got.verdicts)
    tie = got.verdicts[1]
    assert (tie[CC.RUN], tie[CC.VERIFIER], tie[CC.HEAD], tie[CC.ACCEPTED], tie[CC.COMPLETE], tie[CC.WINDOW]) == (
        "run of the response tie", {"module": "the response tie's verifier", "commit": "a commit"},
        first.digest().hex(), True, True, 3)
    assert (tie[CC.SERVE], tie[CC.REGISTRATION]) == ("the serve", "its registration's digest")
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
    # with neither held and a verdict that no record read is of, the line counts the classes that lack a verdict
    neither = SV.stop(both, four(first)[1:] + four(second)[1:])
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


def not_one(phases: int, differ: str, how: str = "") -> str:
    """The answer's first line where each of the four verdicts was read and they are not one class's."""
    return (f"each connection is charged its whole schedule: not one class: unchecked: each of the 4 verdicts was "
            f"read, and no class that holds a row in the {phases} execution phase(s) read has all of them on its "
            f"head{' ' + how if how else ''}: {differ}")


NO_ONE = "they name no one class of one execution phase on one head"
A_CLASS, A_PHASE = "the class (`class`)", "the execution phase (`link`, `window`)"
A_HEAD = "the signed head (`head`) of the network certifier's rows"
A_SERVE, A_REGISTRATION = "the serve (`serve`)", "the registration (`registration`)"
LACK = ("each connection is charged its whole schedule: 1 of the 1 class(es) that hold a row in the 1 execution "
        "phase(s) read lack a verdict, one in each execution phase that holds a row")


def test_four_verdicts_of_two_serves_or_under_two_registrations_hold_nothing_and_the_first_line_names_which():
    """Four accepts on one head of one class, one of them of another serve or under another registration, are not
    one class's: the answer holds nothing, its first line says so and names the field, the class's line names the
    two, and the calculator's rows are the rows of no record at all."""
    first, whole = head(3, 5, 0), LB.table(LB.inputs())
    assert SV.stop([registration(first)], four(first)).held == 1
    for key, what, other in ((CC.SERVE, A_SERVE, "another serve"),
                             (CC.REGISTRATION, A_REGISTRATION, "another's digest")):
        ours = four(first)[0][key]
        for i in range(len(CC.VERDICTS)):
            records = [r | {key: other} if j == i else r for j, r in enumerate(four(first))]
            got = SV.stop([registration(first)], records)
            assert got.held == 0 and len(got.verdicts) == 4 and got.why[:2] == (
                not_one(1, f"they differ in {what}", "as one serve's under one registration"),
                f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each of its 5 rows: its "
                f"verdicts on this head name 2 `{key}`s, not one ({', '.join(sorted((ours, other)))})"), (key, i)
            assert LB.table(LB.inputs(), got) == whole
            assert {v[key] for v in got.verdicts} == {ours, other}
        # all four of the other serve, or under the other registration, are one class's
        assert SV.stop([registration(first)], [r | {key: other} for r in four(first)]).held == 1
    # one of another serve and one under another registration: both are named
    records = [r | change for r, change in zip(four(first), (
        {CC.SERVE: "another serve"}, {CC.REGISTRATION: "another's digest"}, {}, {}))]
    got = SV.stop([registration(first)], records)
    assert got.held == 0 and got.why[0] == not_one(1, f"they differ in {A_SERVE} and in {A_REGISTRATION}",
                                                   "as one serve's under one registration")
    # two execution phases, each with its four of its own serve, are each held: the names are held to one a class
    second = head(4, 2, 0)
    theirs = [r | {CC.SERVE: "another serve", CC.REGISTRATION: "another's digest"} for r in four(second)]
    assert SV.stop([registration(first), registration(second)], four(first) + theirs).held == 2
    # and a held execution phase beside one whose four are of two serves is still held: the line is the count
    mixed = [r | {CC.SERVE: "another serve"} if j == 0 else r for j, r in enumerate(four(second))]
    got = SV.stop([registration(first), registration(second)], four(first) + mixed)
    assert got.held == 1 and got.why[0].startswith("1 of the 2 execution phase(s) read are charged below")
    assert got.why[3].endswith(": its verdicts on this head name 2 `serve`s, not one (another serve, the serve)")


def test_four_verdicts_that_are_not_one_classs_hold_nothing_and_the_first_line_names_what_they_differ_in():
    """The four verdicts count only as one class's: of one class, in one execution phase, on one signed head. Each
    way four accepts can be of no one class on one head holds nothing, and the answer's first line says so and
    names the field. The calculator's rows are then the rows of no record at all."""
    assert SV.NOT_ONE_CLASS == "not one class: unchecked"
    whole = LB.table(LB.inputs())
    # another class: the stop record is of class 0 and the other three of class 1, on a head whose two classes
    # each hold a row
    wide = head(3, 5, 2)
    got = SV.stop([registration(wide)], four(wide)[:1] + four(wide, 1)[1:])
    assert got.held == 0 and got.why[:3] == (
        not_one(1, f"{NO_ONE}, and differ in {A_CLASS}"),
        f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each of its 5 rows: the "
        "response tie is missing; the certified read is missing; the compute audit is missing",
        f"the link {LINK}, execution phase 3, class 1: charged its whole schedule on each of its 2 rows: the stop "
        "record is missing")
    assert LB.table(LB.inputs(), got) == whole
    assert SV.stop([registration(wide)], four(wide) + four(wide, 1)).held == 1
    # another serve: two of the four are on the head of another session's serve of the same link, execution phase
    # and class, whose tables' domains are that session's
    first, theirs = head(3, 5, 0), head(3, 5, 0, domain=bytes([1]) * 64)
    assert first.digest() != theirs.digest() and SV.stop([registration(theirs)], four(theirs)).held == 1
    got = SV.stop([registration(first)], four(first)[:2] + four(theirs)[2:])
    assert got.held == 0 and len(got.verdicts) == 2 and got.why[0] == not_one(1, f"{NO_ONE}, and differ in {A_HEAD}")
    assert got.why[1].endswith(": the certified read is not on this head (run of the certified read); the compute "
                               "audit is not on this head (run of the compute audit)")
    assert LB.table(LB.inputs(), got) == whole
    # and where the other serve's execution phase is read too, under another number, in the execution phase as well
    later = head(4, 5, 0, domain=bytes([1]) * 64)
    got = SV.stop([registration(first), registration(later)], four(first)[:2] + four(later)[2:])
    assert got.held == 0 and got.why[0] == not_one(2, f"{NO_ONE}, and differ in {A_PHASE} and in {A_HEAD}")
    # another registration: the four name one head, and the registration read for the execution phase is of
    # another, or none was read for it
    for other in (head(3, 6, 0), head(9, 5, 0)):
        got = SV.stop([registration(first)], four(other))
        assert (got.held, got.verdicts) == (0, ()) and got.why[0] == not_one(
            1, "they name one class on one head, which is no class that holds a row on a head read")
        assert LB.table(LB.inputs(), got) == whole
    # each field with a value all four name, and no one place: the stop record's two records are of class 0 on
    # another head and of class 1 on this one, and the other three are of class 0 on this one
    across = [four(head(3, 6, 2))[0], four(wide, 1)[0], *four(wide)[1:]]
    got = SV.stop([registration(wide)], across)
    assert got.held == 0 and got.why[0] == not_one(1, NO_ONE)
    # a verdict that no record read is of is a verdict lacking, and four on one class with one refused is a
    # refusal: neither is this
    refused = [r | {CC.VERDICT: r[CC.VERDICT] | {CC.ACCEPTED: False}} if i == 1 else r
               for i, r in enumerate(four(first))]
    for records in (four(first)[1:], four(first)[:1] + four(theirs)[2:], refused):
        assert SV.stop([registration(first)], records).why[0] == LACK


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


NO_RECORD = ("a verdict record is an object with the fields run, serve, registration, verifier, verdict, proved_by, "
             "its verdict an object, its run, serve, registration named, its verifier by its module and commit, and "
             "its proved_by a name or null; of one read, ")


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
            ([registration(first)], [*four(first), "accept"], f"{NO_RECORD}it is no object: 'accept'"),
            ([registration(first)], [four(first)[0] | {CC.VERDICT: "accept"}],
             f"{NO_RECORD}its `verdict` is no object: {four(first)[0] | {CC.VERDICT: 'accept'}!r}")):
        got = SV.stop(registrations, records)
        assert (got.held, got.verdicts, got.why[0]) == (0, (), whole + why), why
    # four verdicts that each accept on the head hold it, and hold nothing once one of them names no run, no serve,
    # no registration, or no verifier by its module and commit: a verdict nobody can trace to what produced it and
    # to what it is of, and the line names the field. So does one of a format listed for no verdict, which would
    # otherwise count for nothing
    assert SV.stop([registration(first)], four(first)).held == 1
    tie, unlisted = four(first)[1], four(first)[1] | {CC.VERDICT: four(first)[1][CC.VERDICT] | {SV.FORMAT: "another's"}}
    for of in (tie, unlisted):
        for off, lack in (
                *(({k: v for k, v in of.items() if k != field}, f"it has no `{field}`")
                  for field in (CC.RUN, CC.SERVE, CC.REGISTRATION, CC.VERIFIER, CC.PROVED_BY)),
                (of | {CC.RUN: ""}, "its `run` is no name"), (of | {CC.SERVE: ""}, "its `serve` is no name"),
                (of | {CC.REGISTRATION: None}, "its `registration` is no name"),
                (of | {CC.VERIFIER: None}, "its `verifier` names no `module`, `commit`"),
                (of | {CC.VERIFIER: {"module": of[CC.VERIFIER]["module"]}}, "its `verifier` names no `commit`"),
                (of | {CC.VERIFIER: of[CC.VERIFIER] | {"commit": ""}}, "its `verifier` names no `commit`"),
                ({CC.VERDICT: of[CC.VERDICT]}, "it has no `run`, `serve`, `registration`, `verifier`, `proved_by`")):
            got = SV.stop([registration(first)], [*four(first)[:1], off, *four(first)[2:]])
            assert (got.held, got.verdicts, got.why[0]) == (0, (), f"{whole}{NO_RECORD}{lack}: {off!r}"), off
    assert SV.stop([registration(first)], [*four(first), unlisted]).held == 1
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


def held_files(of: C.Head, records: list) -> dict[str, str]:
    """A run's files: the registration of a head and verdict records."""
    return {SV.REGISTRATIONS_FILE: jsonl([registration(of)]), "out/the-verdicts.jsonl": jsonl(records)}


def tie_but(of: C.Head, **over: object) -> list[dict]:
    """The four verdicts' records on a head, the tie's verdict with ``over``."""
    return [r | {CC.VERDICT: r[CC.VERDICT] | over} if i == 1 else r for i, r in enumerate(four(of))]


def dropped(n: int, *runs: str) -> str:
    return (f"{n} verdict record(s) of the superseded run(s) {', '.join(runs)} are accepted and complete, and count "
            "for nothing: a superseded run adds no accept, and what it refused stands")


def test_a_superseded_run_adds_no_accept(tmp_path):
    """The reader is handed every run that carries the record, with which of them another run replaces. A
    superseded run's accepts count for nothing: not alone, not to fill in a verdict a later run lacks, and not as a
    copy in a later run's files."""
    first = head(3, 5, 0)
    records = four(first)
    copies = [r | {CC.RUN: "r1"} for r in records]
    store = Store(tmp_path, {"r1": held_files(first, records), "r2": {"out/another.json": "{}"},
                             "r3": held_files(first, records[:1] + records[2:]), "r4": held_files(first, copies)})
    neither = "holds neither `out/*-verdicts.jsonl` nor `out/registrations.jsonl`"
    assert SV.of_runs(store, ["r1", "r2"]).held == SV.of_runs(store, ["r1", "r2"], []).held == 1
    # the accepting run superseded by one that holds nothing: its four count for nothing, and its registration is
    # still read, so the class is named with what it lacks
    got = SV.of_runs(store, ["r1", "r2"], ["r1"])
    assert (got.held, got.verdicts) == (0, ()) and got.why[0] == LACK
    assert got.why[1].endswith(": the stop record is missing; the response tie is missing; the certified read is "
                               "missing; the compute audit is missing")
    assert got.why[-2:] == (f"r2 is not read: it {neither}", dropped(4, "r1"))
    assert LB.table(LB.inputs(), got) == LB.table(LB.inputs())
    # superseding the run that holds nothing changes nothing
    assert SV.of_runs(store, ["r1", "r2"], ["r2"]).held == 1
    # a superseded run's tie fills in no tie a later run lacks
    assert SV.of_runs(store, ["r1", "r3"]).held == 1
    got = SV.of_runs(store, ["r1", "r3"], ["r1"])
    assert got.held == 0 and len(got.verdicts) == 3 and got.why[1].endswith(": the response tie is missing")
    assert got.why[-1] == dropped(4, "r1")
    # a record that names a superseded run as the run that produced it is that run's, whichever run's files hold it
    assert SV.of_runs(store, ["r4"]).held == 1
    got = SV.of_runs(store, ["r1", "r4"], ["r1"])
    assert (got.held, got.verdicts, got.why[-1]) == (0, (), dropped(8, "r1"))
    # `read` leaves the accepts out and keeps the registrations
    assert SV.read(store, ["r1", "r3"], ["r1"]) == ([registration(first)] * 2, records[:1] + records[2:],
                                                    [dropped(4, "r1")])
    assert SV.read(store, ["r1", "r3"]) == ([registration(first)] * 2, records + records[:1] + records[2:], [])


def test_a_superseded_run_takes_away_no_refusal(tmp_path):
    """What a superseded run refused on a head stands, whatever a later run accepted on it: a refused verdict, an
    accept that is not complete, a record that doesn't read, and a registration that gives the execution phase
    another head. The run must be among the runs handed for that to be read, so a superseded run that is not
    holds everything at its whole schedule. And superseding a run never raises what is held."""
    first = head(3, 5, 0)
    accepts = held_files(first, four(first))
    for old, lack in ((tie_but(first, accepted=False), "the response tie is refused (run of the response tie)"),
                      (tie_but(first, complete=False), "the response tie is not complete (run of the response tie)")):
        store = Store(tmp_path / lack, {"old": held_files(first, old), "new": accepts})
        for superseded in ((), ["old"]):
            got = SV.of_runs(store, ["old", "new"], superseded)
            assert got.held == 0 and got.why[0] == LACK, (lack, superseded)
            assert got.why[1] == (f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each "
                                  f"of its 5 rows: {lack}")
        assert got.why[-1] == dropped(3, "old")
        # the later run alone holds the execution phase, which is why the superseded one is to be handed
        assert SV.of_runs(store, ["new"]).held == 1
        unread = SV.of_runs(store, ["new"], ["old"])
        assert unread == LB.Stop(0, (
            "each connection is charged its whole schedule: the superseded run(s) old are not among the runs "
            "handed, so what they refused was not read: a superseded run adds no accept and takes away no refusal",))
    # a superseded run's record that doesn't read holds nothing, as any run's does
    store = Store(tmp_path / "unread", {"old": held_files(first, [*four(first), "accept"]), "new": accepts})
    for superseded in ((), ["old"]):
        got = SV.of_runs(store, ["old", "new"], superseded)
        assert got.held == 0 and got.why[0].endswith(f"{NO_RECORD}it is no object: 'accept'")
    # and the registration it kept is read as any run's: another head for the execution phase holds nothing
    other = head(3, 6, 0)
    store = Store(tmp_path / "heads", {"old": held_files(other, four(other)), "new": accepts})
    for superseded in ((), ["old"]):
        got = SV.of_runs(store, ["old", "new"], superseded)
        assert got.held == 0 and got.why[0] == (
            f"each connection is charged its whole schedule: two heads are given for execution phase 3 of the link {LINK}")
    # superseding a run never raises what is held, whatever the two runs hold
    second = head(4, 2, 0)
    kept = {"the four": accepts, "a tie refused": held_files(first, tie_but(first, accepted=False)),
            "a tie not complete": held_files(first, tie_but(first, complete=False)),
            "three verdicts": held_files(first, four(first)[1:]), "another head": held_files(other, four(other)),
            "another execution phase": held_files(second, four(second)), "no record": held_files(first, []),
            "no registration": {"out/the-verdicts.jsonl": jsonl(four(first))}, "nothing": {"out/another.json": "{}"},
            "a record that doesn't read": held_files(first, [{CC.VERDICT: {}}])}
    seen = lowered = most = 0
    for i, (a, old) in enumerate(kept.items()):
        for j, (b, new) in enumerate(kept.items()):
            store = Store(tmp_path / f"pair-{i}-{j}", {"old": old, "new": new})
            both = SV.of_runs(store, ["old", "new"]).held
            for superseded in (["old"], ["new"], ["old", "new"]):
                got = SV.of_runs(store, ["old", "new"], superseded).held
                assert got <= both, (a, b, superseded)
                seen, lowered, most = seen + 1, lowered + (got < both), max(most, both)
    assert (seen, most) == (300, 2) and lowered
