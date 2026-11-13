"""The stop class's verdict records as the calculator's rows read them (`stop_verdicts`): a head is the
registration's own, held to the digest its certificate names; a record is its verifier's, and which of the four
verdicts it is is its verdict's format; the answer counts the execution phases held, those of a head read in which
every class that holds a row has its four verdicts by the network certifier's rule, each on that head; every other
execution phase is charged its whole schedule, so one left out of a run's files charges what it would refused;
a registration that doesn't read, or no record at all, holds none, with a line that says why; an execution phase the
rule charges below is held only where each verifier that accepted refused its negative control at the same commit
and the serve's registration registers every block of both sampled populations, as each of their verdicts says it
ran; each verdict read is handed on under the mark of a run that is no proof; and the run files are read from the
store as the harness reads it."""

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
from verity.protocols.spot.verifier.one_stage import registration as R  # noqa: E402

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


#: a serve's registration that registers both sampled populations at every block, and another serve's; each with
#: the digest a record names it by
SERVED = {"format": "a serve's registration", "rates": {"stop_record": "1/1", "response_tie": "1/1"}}
OTHERS = SERVED | {"window": "another serve's"}
REGISTERED, OTHER = SV.digest_of(SERVED), SV.digest_of(OTHERS)


def four(of: C.Head, cls: int = 0, proved_by: object = None, **over: object) -> list[dict]:
    """The four verdicts' records on a class of a head, as their verifiers write them: each accepted, complete and
    naming the head, but for `over`, the verdict's fields; of one serve under `SERVED`, the stop record and the
    response tie at the rate it registers."""
    return [{CC.RUN: f"run of the {kind}", CC.SERVE: "the serve", CC.REGISTRATION: REGISTERED,
             CC.VERIFIER: {"module": f"the {kind}'s verifier", "commit": "a commit"},
             CC.VERDICT: {SV.FORMAT: FORMATS[kind], CC.LINK: of.link, CC.WINDOW: of.window, CC.CLASS: cls,
                          CC.HEAD: of.digest().hex(), "registration": "a digest", CC.ACCEPTED: True,
                          CC.COMPLETE: True, "codes": []} | over,
             CC.PROVED_BY: proved_by} | ({SV.RATE: "1/1"} if kind in SV.RATES else {}) for kind in CC.VERDICTS]


def the_controls() -> list[dict]:
    """The negative control each of the four verifiers owes, as `four` names the verifiers: each refused, for the
    control's word."""
    return [{CC.RUN: f"control run of the {kind}",
             CC.VERIFIER: {"module": f"the {kind}'s verifier", "commit": "a commit"},
             SV.CONTROL: SV.CONTROLS[kind][0],
             CC.VERDICT: {SV.FORMAT: FORMATS[kind], CC.ACCEPTED: False, "codes": ["a refusal"]},
             SV.REASON: SV.REASONS[kind]}
            for kind in CC.VERDICTS]


def refused(*kinds: str) -> str:
    """What a class's line says of the controls of `the_controls` that its verifiers refused: all four's, or
    those of ``kinds``."""
    return "; its verifiers refused their negative controls: " + "; ".join(
        f"the {kind}'s `{SV.CONTROLS[kind][0]}` for \"{SV.REASONS[kind]}\"" for kind in kinds or CC.VERDICTS)


def stop(registrations: list, records: list, left: list[str] | tuple = (), controls: list | None = None,
         served: list | None = None, spent: list | tuple = ()) -> LB.Stop:
    """`SV.stop` with, unless they are given, the four verifiers' controls and the two serves' registrations."""
    return SV.stop(registrations, records, left, the_controls() if controls is None else controls,
                   [SERVED, OTHERS] if served is None else served, spent)


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
EVERY_OTHER = (": each class that holds a row in one of them has its 4 verdicts, accepted and complete, at the "
               "registered rate `1/1`, each by a verifier that refused its negative control. Every other "
               "execution phase of a monitoring period is charged its whole schedule, one no registration was read "
               "for among them")


def test_an_execution_phase_is_held_only_with_the_verdicts_of_every_class_that_holds_a_row_in_it():
    first, second = head(3, 5, 0), head(4, 2, 0)
    both = [registration(first), registration(second)]
    got = stop(both, four(first) + four(second))
    assert got.held == 2 and got.why == (
        f"2 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}",
        "an execution phase is one serve's, and a head's serve is the one its verdict records name, which is run, "
        "not proved",
        f"the link {LINK}, execution phase 3, {CC.charge(first, 0, by_verdict(four(first))).why()}{refused()}",
        f"the link {LINK}, execution phase 3, class 1: a count of 0, so no connection is charged and no verdict is owed",
        f"the link {LINK}, execution phase 4, {CC.charge(second, 0, by_verdict(four(second))).why()}{refused()}",
        f"the link {LINK}, execution phase 4, class 1: a count of 0, so no connection is charged and no verdict is owed")
    assert got.why[1] == SV.SERVE_NAMED
    assert [v["verdict"] for v in got.verdicts] == list(CC.VERDICTS) * 2
    assert all(v["mark"] == LB.RUN_NOT_PROVED == "run, not proved" and set(v) == {
        "verdict", *CC.READ, CC.RUN, CC.SERVE, CC.REGISTRATION, CC.VERIFIER, CC.PROVED_BY, SV.RATE, "mark"}
        for v in got.verdicts)
    tie = got.verdicts[1]
    assert (tie[CC.RUN], tie[CC.VERIFIER], tie[CC.HEAD], tie[CC.ACCEPTED], tie[CC.COMPLETE], tie[CC.WINDOW]) == (
        "run of the response tie", {"module": "the response tie's verifier", "commit": "a commit"},
        first.digest().hex(), True, True, 3)
    assert (tie[CC.SERVE], tie[CC.REGISTRATION], tie[SV.RATE]) == ("the serve", REGISTERED, "1/1")
    # the stop record and the response tie say the rate they ran at; the other two are over no sampled population
    assert [v[SV.RATE] for v in got.verdicts[:4]] == ["1/1", "1/1", None, None]
    # a proof named in a record changes no mark here
    proved = stop(both[:1], four(first, proved_by="a theorem"))
    assert proved.held == 1 and {v["mark"] for v in proved.verdicts} == {LB.RUN_NOT_PROVED}
    # an execution phase without its tie stays at its whole schedule, and the other is held
    for records, lack in ((four(first) + four(second)[:1] + four(second)[2:], "the response tie is missing"),
                          (four(first) + four(second, complete=False)[1:2] + four(second)[:1] + four(second)[2:],
                           "the response tie is not complete (run of the response tie)"),
                          (four(first), "the stop record is missing; the response tie is missing; the certified read "
                                        "is missing; the compute audit is missing")):
        got = stop(both, records)
        assert got.held == 1 and got.why[0] == (
            f"1 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}")
        assert got.why[4] == (f"the link {LINK}, execution phase 4, class 0: charged its whole schedule on each of its "
                              f"2 rows: {lack}")
    # with neither held and a verdict that no record read is of, the line counts the classes that lack a verdict
    neither = stop(both, four(first)[1:] + four(second)[1:])
    assert neither.held == 0 and neither.why[0] == (
        "each connection is charged its whole schedule: 2 of the 2 class(es) that hold a row in the 2 execution "
        "phase(s) read lack a verdict, one in each execution phase that holds a row")
    # a second class that holds a row owes its own four
    wide = head(3, 5, 1)
    assert stop([registration(wide)], four(wide)).held == 0
    assert stop([registration(wide)], four(wide) + four(wide, 1)).held == 1
    # verdicts decided under another serve's head of the same execution phase and class count for nothing
    theirs = head(3, 5, 0, parameters=bytes([1]) * 64)
    other = stop(both[:1], four(theirs))
    assert other.held == 0 and other.verdicts == () and other.why[2].endswith(
        "the stop record is not on this head (run of the stop record); the response tie is not on this head (run of "
        "the response tie); the certified read is not on this head (run of the certified read); the compute audit is "
        "not on this head (run of the compute audit)")
    # records on no head read are counted in a line and count for nothing
    stray = stop(both[:1], four(first) + four(head(9, 5)) + four(theirs))
    assert stray.held == 1 and len(stray.verdicts) == 4 and stray.why[-1] == (
        "8 verdict record(s) read are on no head read, or on no class of one that holds a row, and count for nothing")
    # what was not read is carried
    assert stop(both[:1], four(first), ["r9 is not read"]).why[-1] == "r9 is not read"


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
    assert stop([registration(first)], four(first)).held == 1
    for key, what, other in ((CC.SERVE, A_SERVE, "another serve"), (CC.REGISTRATION, A_REGISTRATION, OTHER)):
        ours = four(first)[0][key]
        for i in range(len(CC.VERDICTS)):
            records = [r | {key: other} if j == i else r for j, r in enumerate(four(first))]
            got = stop([registration(first)], records)
            assert got.held == 0 and len(got.verdicts) == 4 and got.why[:3] == (
                not_one(1, f"they differ in {what}", "as one serve's under one registration"), SV.SERVE_NAMED,
                f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each of its 5 rows: its "
                f"verdicts on this head name 2 `{key}`s, not one ({', '.join(sorted((ours, other)))})"), (key, i)
            assert LB.table(LB.inputs(), got) == whole
            assert {v[key] for v in got.verdicts} == {ours, other}
        # all four of the other serve, or under the other registration, are one class's
        assert stop([registration(first)], [r | {key: other} for r in four(first)]).held == 1
    # one of another serve and one under another registration: both are named
    records = [r | change for r, change in zip(four(first), (
        {CC.SERVE: "another serve"}, {CC.REGISTRATION: OTHER}, {}, {}))]
    got = stop([registration(first)], records)
    assert got.held == 0 and got.why[0] == not_one(1, f"they differ in {A_SERVE} and in {A_REGISTRATION}",
                                                   "as one serve's under one registration")
    # two execution phases, each with its four of its own serve, are each held: the names are held to one a class
    second = head(4, 2, 0)
    theirs = [r | {CC.SERVE: "another serve", CC.REGISTRATION: OTHER} for r in four(second)]
    assert stop([registration(first), registration(second)], four(first) + theirs).held == 2
    # and a held execution phase beside one whose four are of two serves is still held: the line is the count
    mixed = [r | {CC.SERVE: "another serve"} if j == 0 else r for j, r in enumerate(four(second))]
    got = stop([registration(first), registration(second)], four(first) + mixed)
    assert got.held == 1 and got.why[0].startswith("1 of the 2 execution phase(s) read are charged below")
    assert got.why[4].endswith(": its verdicts on this head name 2 `serve`s, not one (another serve, the serve)")


def test_four_verdicts_that_are_not_one_classs_hold_nothing_and_the_first_line_names_what_they_differ_in():
    """The four verdicts count only as one class's: of one class, in one execution phase, on one signed head. Each
    way four accepts can be of no one class on one head holds nothing, and the answer's first line says so and
    names the field. The calculator's rows are then the rows of no record at all."""
    assert SV.NOT_ONE_CLASS == "not one class: unchecked"
    whole = LB.table(LB.inputs())
    # another class: the stop record is of class 0 and the other three of class 1, on a head whose two classes
    # each hold a row
    wide = head(3, 5, 2)
    got = stop([registration(wide)], four(wide)[:1] + four(wide, 1)[1:])
    assert got.held == 0 and got.why[:4] == (
        not_one(1, f"{NO_ONE}, and differ in {A_CLASS}"), SV.SERVE_NAMED,
        f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each of its 5 rows: the "
        "response tie is missing; the certified read is missing; the compute audit is missing",
        f"the link {LINK}, execution phase 3, class 1: charged its whole schedule on each of its 2 rows: the stop "
        "record is missing")
    assert LB.table(LB.inputs(), got) == whole
    assert stop([registration(wide)], four(wide) + four(wide, 1)).held == 1
    # another serve: two of the four are on the head of another session's serve of the same link, execution phase
    # and class, whose tables' domains are that session's
    first, theirs = head(3, 5, 0), head(3, 5, 0, domain=bytes([1]) * 64)
    assert first.digest() != theirs.digest() and stop([registration(theirs)], four(theirs)).held == 1
    got = stop([registration(first)], four(first)[:2] + four(theirs)[2:])
    assert got.held == 0 and len(got.verdicts) == 2 and got.why[0] == not_one(1, f"{NO_ONE}, and differ in {A_HEAD}")
    assert got.why[2].endswith(": the certified read is not on this head (run of the certified read); the compute "
                               "audit is not on this head (run of the compute audit)")
    assert LB.table(LB.inputs(), got) == whole
    # and where the other serve's execution phase is read too, under another number, in the execution phase as well
    later = head(4, 5, 0, domain=bytes([1]) * 64)
    got = stop([registration(first), registration(later)], four(first)[:2] + four(later)[2:])
    assert got.held == 0 and got.why[0] == not_one(2, f"{NO_ONE}, and differ in {A_PHASE} and in {A_HEAD}")
    # another registration: the four name one head, and the registration read for the execution phase is of
    # another, or none was read for it
    for other in (head(3, 6, 0), head(9, 5, 0)):
        got = stop([registration(first)], four(other))
        assert (got.held, got.verdicts) == (0, ()) and got.why[0] == not_one(
            1, "they name one class on one head, which is no class that holds a row on a head read")
        assert LB.table(LB.inputs(), got) == whole
    # each field with a value all four name, and no one place: the stop record's two records are of class 0 on
    # another head and of class 1 on this one, and the other three are of class 0 on this one
    across = [four(head(3, 6, 2))[0], four(wide, 1)[0], *four(wide)[1:]]
    got = stop([registration(wide)], across)
    assert got.held == 0 and got.why[0] == not_one(1, NO_ONE)
    # a verdict that no record read is of is a verdict lacking, and four on one class with one refused is a
    # refusal: neither is this
    refused = [r | {CC.VERDICT: r[CC.VERDICT] | {CC.ACCEPTED: False}} if i == 1 else r
               for i, r in enumerate(four(first))]
    for records in (four(first)[1:], four(first)[:1] + four(theirs)[2:], refused):
        assert stop([registration(first)], records).why[0] == LACK


def held_back(ruled: int, phases: int, rate: int = 0, control: int = 0, serve: int = 0, unread: int = 0,
              head: int = 0) -> str:
    """The answer's first line where execution phases have their verdicts and none of them is held."""
    back = ([f"{serve} by its serve, one named by all of a head's verdict records"] if serve else []) + (
        [f"{rate} by the rate, `1/1` as registered and as run"] if rate else []) + (
        [f"{control} by a negative control, one refused by each verifier that accepted, at its commit, for the "
         "control's word"] if control else []) + (
        [f"{head} by a verdict refused or not complete under another head of the execution phase"] if head else []) + (
        [f"{unread} by a superseded run's record that doesn't read, of a run its serve was not launched after"]
        if unread else [])
    return (f"each connection is charged its whole schedule: {ruled} of the {phases} execution phase(s) read have the "
            f"4 verdicts of each class that holds a row, and none of them is held: {' and '.join(back)}")


def no_control(kind: str, commit: str = "a commit", module: str | None = None) -> str:
    (name, what), module = SV.CONTROLS[kind], module or f"the {kind}'s verifier"
    return (f"the {kind}'s verifier {module} at {commit} has no negative control of record at that commit "
            f"(`{name}`: {what})")


def test_an_execution_phase_is_held_only_where_each_verifier_that_accepted_refused_its_control_at_its_commit():
    """Four accepts hold an execution phase only where each verifier that accepted has, at the same commit, a
    control of record that it refused: the one that verifier owes. Without one the four are read and handed on
    and hold nothing, and the line names the verifier. One control a verifier a commit serves every execution
    phase under that commit; one at another commit, by another verifier or of another thing serves none; and one
    the verifier did not refuse, or refused for another reason than the control's word, leaves it without,
    whatever other it refused."""
    assert SV.CONTROLS == {CC.STOP_RECORD: ("stop_word", "a stop word changed, the count off by one"),
                           CC.RESPONSE_TIE: ("id", "a released id changed"),
                           CC.CERTIFIED_READ: ("max_tokens", "a request word changed, `max_tokens`"),
                           CC.COMPUTE_AUDIT: ("output", "a committed output changed")}
    assert SV.REASONS == {CC.STOP_RECORD: "stop_word", CC.RESPONSE_TIE: "id", CC.CERTIFIED_READ: "max_tokens",
                          CC.COMPUTE_AUDIT: "output"}
    assert SV.CONTROL_FIELDS == ("run", "verifier", "control", "verdict", "reason") and SV.REASON == "reason"
    first, second, whole = head(3, 5, 0), head(4, 2, 0), LB.table(LB.inputs())
    one, both = [registration(first)], [registration(first), registration(second)]
    at = f"the link {LINK}, execution phase 3: held to its whole schedule by a control: "
    below = (f"the link {LINK}, execution phase 3, class 0: charged below its schedule, its 4 verdicts (stop record, "
             "response tie, certified read, compute audit) each accepted and complete on this head")
    # with each verifier's control, the class's line gives each verifier's own word for why it refused
    got = stop(one, four(first))
    assert got.held == 1 and got.why[2] == below + (
        "; its verifiers refused their negative controls: the stop record's `stop_word` for \"stop_word\"; the "
        "response tie's `id` for \"id\"; the certified read's `max_tokens` for \"max_tokens\"; the compute audit's "
        "`output` for \"output\"")
    # no control at all: the four are read and handed on, and hold nothing; the line names each verifier
    got = stop(one, four(first), controls=[])
    assert got.held == 0 and len(got.verdicts) == 4 and got.why[0] == held_back(1, 1, control=1)
    assert got.why[2] == below
    assert got.why[4] == at + "; ".join(no_control(kind) for kind in CC.VERDICTS) and len(got.why) == 5
    assert LB.table(LB.inputs(), got) == whole
    # each verifier owes its own: with one's control left out, the line names that verifier alone, and the class's
    # line gives the other three's words
    for i, kind in enumerate(CC.VERDICTS):
        got = stop(one, four(first), controls=the_controls()[:i] + the_controls()[i + 1:])
        assert got.held == 0 and got.why[4] == at + no_control(kind), kind
        assert got.why[2] == below + refused(*CC.VERDICTS[:i], *CC.VERDICTS[i + 1:]), kind
    # a control at another commit, by another verifier, or of a thing this verifier doesn't owe is none of its
    tie = the_controls()[1]
    strays = ("1 control record(s) read are of a format listed for none of the 4 verdicts, or are not the control "
              "their verdict's verifier owes, and count for nothing")
    for off, stray in ((tie | {CC.VERIFIER: tie[CC.VERIFIER] | {"commit": "another commit"}}, False),
                       (tie | {CC.VERIFIER: tie[CC.VERIFIER] | {"module": "another verifier"}}, False),
                       (tie | {SV.CONTROL: "max_tokens"}, True),
                       (tie | {CC.VERDICT: tie[CC.VERDICT] | {SV.FORMAT: FORMATS[CC.STOP_RECORD]}}, True),
                       (tie | {CC.VERDICT: tie[CC.VERDICT] | {SV.FORMAT: "another's"}}, True)):
        got = stop(one, four(first), controls=[off if c == tie else c for c in the_controls()])
        assert got.held == 0 and got.why[4] == at + no_control(CC.RESPONSE_TIE), off
        assert (got.why[-1] == strays) == stray and len(got.why) == 5 + stray, off
    # a control its verifier did not refuse leaves it without one, whatever other control it refused; it owes no
    # word for a refusal, and the class's line gives none for that verifier
    not_refused = at + ("the response tie's verifier the response tie's verifier at a commit did not refuse its "
                        "negative control (`id`: a released id changed) on ")
    others = refused(CC.STOP_RECORD, CC.CERTIFIED_READ, CC.COMPUTE_AUDIT)
    for accepted in (True, None, "false", 0):
        for reason in ("id", None, ""):
            controls = [c | {SV.REASON: reason} if i == 1 else c for i, c in enumerate(control_but(accepted=accepted))]
            got = stop(one, four(first), controls=controls)
            assert got.held == 0 and got.why[4] == not_refused + "control run of the response tie", (accepted, reason)
            assert got.why[2] == below + others, (accepted, reason)
    bare = [c | {CC.VERDICT: {SV.FORMAT: TIE}} if i == 1 else c for i, c in enumerate(the_controls())]
    assert stop(one, four(first), controls=bare).why[4] == not_refused + "control run of the response tie"
    again = control_but(accepted=True)[1] | {CC.RUN: "a second control run"}
    got = stop(one, four(first), controls=the_controls() + [again])
    assert got.held == 0 and got.why[4] == not_refused + "a second control run" and got.why[2] == below + others
    # a control counts only where it is refused for the reason its change aims at, the control's word: one refused
    # for another reason leaves its verifier without its control, the line names the reason given and the word,
    # and the class's line gives the other three's
    for i, kind in enumerate(CC.VERDICTS):
        name = SV.CONTROLS[kind][0]
        for reason in ("parse", name.upper(), f"{name} ", "refused"):
            controls = [c | {SV.REASON: reason} if j == i else c for j, c in enumerate(the_controls())]
            got = stop(one, four(first), controls=controls)
            assert got.held == 0 and len(got.verdicts) == 4 and got.why[0] == held_back(1, 1, control=1), (kind, reason)
            assert got.why[4] == at + (f"the {kind}'s verifier the {kind}'s verifier at a commit: control {name} "
                                       f"refused for {reason!r}, not {name!r}, on control run of the {kind}"), reason
            assert got.why[2] == below + refused(*CC.VERDICTS[:i], *CC.VERDICTS[i + 1:]) and len(got.why) == 5
    assert stop(one, four(first), controls=[c | {SV.REASON: "parse"} if j == 0 else c
                                            for j, c in enumerate(the_controls())]).why[4] == at + (
        "the stop record's verifier the stop record's verifier at a commit: control stop_word refused for 'parse', "
        "not 'stop_word', on control run of the stop record")
    # so does one refused for the control's word beside one refused for another reason, whichever is read first,
    # and the line names each control that was refused for another
    again = tie | {CC.RUN: "a second control run", SV.REASON: "parse"}
    third = tie | {CC.RUN: "a third control run", SV.REASON: "no head"}
    wrong = at + ("the response tie's verifier the response tie's verifier at a commit: control id refused for "
                  "'parse', not 'id', on a second control run")
    for controls in (the_controls() + [again], [again] + the_controls()):
        got = stop(one, four(first), controls=controls)
        assert got.held == 0 and got.why[4] == wrong and got.why[2] == below + others
    got = stop(one, four(first), controls=the_controls() + [again, third])
    assert got.held == 0 and got.why[4] == at + (
        "the response tie's verifier the response tie's verifier at a commit: control id refused for 'no head', not "
        "'id', on a third control run; control id refused for 'parse', not 'id', on a second control run")
    # two controls of one verifier, each refused for the control's word, serve it
    again = tie | {CC.RUN: "a second control run"}
    got = stop(one, four(first), controls=the_controls() + [again])
    assert got.held == 1 and got.why[2] == below + refused()
    # a control with more fields reads, and none of them is read: not as a head, a class, a serve or a rate
    more = {"registration": OTHER, "serve": "another serve", "class": "w3/p0-o0", "rate": "1/8", "piece": "tie"}
    got = stop(one, four(first), controls=[c | more for c in the_controls()])
    assert (got.held, got.why) == (1, stop(one, four(first)).why)
    # one control a verifier a commit serves every execution phase checked under that commit
    assert stop(both, four(first) + four(second)).held == 2
    # where the second execution phase's tie was accepted at another commit, it owes a control at that commit
    moved = [r | {CC.VERIFIER: r[CC.VERIFIER] | {"commit": "another commit"}} if i == 1 else r
             for i, r in enumerate(four(second))]
    got = stop(both, four(first) + moved)
    assert got.held == 1 and got.why[0].startswith("1 of the 2 execution phase(s) read are charged below")
    assert got.why[6] == (f"the link {LINK}, execution phase 4: held to its whole schedule by a control: "
                          + no_control(CC.RESPONSE_TIE, "another commit")) and len(got.why) == 7
    there = tie | {CC.VERIFIER: tie[CC.VERIFIER] | {"commit": "another commit"}}
    assert stop(both, four(first) + moved, controls=the_controls() + [there]).held == 2
    # each class that holds a row owes its own verifiers' controls
    wide = head(3, 5, 2)
    wider = four(wide) + [r | {CC.VERIFIER: r[CC.VERIFIER] | {"commit": "another commit"}} if i == 1 else r
                          for i, r in enumerate(four(wide, 1))]
    got = stop([registration(wide)], wider)
    assert got.held == 0 and got.why[4] == at + no_control(CC.RESPONSE_TIE, "another commit")
    assert stop([registration(wide)], wider, controls=the_controls() + [there]).held == 1
    # two records of one verdict, each accepted by its own verifier: each verifier owes its control
    twice = four(first) + [four(first)[1] | {CC.RUN: "a second tie's run", CC.VERIFIER: {
        "module": "a second tie's verifier", "commit": "a commit"}}]
    got = stop(one, twice)
    assert got.held == 0 and len(got.verdicts) == 5 and got.why[4] == at + no_control(
        CC.RESPONSE_TIE, module="a second tie's verifier")
    # a class short of a verdict is at its whole schedule by the rule, and no control is asked of it
    got = stop(one, four(first)[1:], controls=[])
    assert got.why[0] == LACK and len(got.why) == 4
    # a control record that doesn't read holds nothing at all, and the line names the field: one without its
    # verifier's word among them, and a refused one whose word is no text
    a_control = ("each connection is charged its whole schedule: a control record is an object with the fields run, "
                 "verifier, control, verdict, reason, its run and its control named, its verifier by its module and "
                 "commit, its verdict an object, and its reason its verifier's word for a refusal; of one read, ")
    no_word = "its `reason` is no text, though its verdict is refused"
    assert {k: v for k, v in tie.items() if k != "reason"} == {k: tie[k] for k in ("run", "verifier", "control", "verdict")}
    for off, lack in (("refused", "it is no object"),
                      *(({k: v for k, v in tie.items() if k != field}, f"it has no `{field}`")
                        for field in SV.CONTROL_FIELDS),
                      (tie | {CC.RUN: ""}, "its `run` is no name"), (tie | {SV.CONTROL: None}, "its `control` is no name"),
                      (tie | {CC.VERIFIER: {"module": "a module"}}, "its `verifier` names no `commit`"),
                      (tie | {CC.VERIFIER: "a verifier"}, "its `verifier` names no `module`, `commit`"),
                      (tie | {CC.VERDICT: "refused"}, "its `verdict` is no object"),
                      (tie | {SV.REASON: None}, no_word), (tie | {SV.REASON: ""}, no_word),
                      (tie | {SV.REASON: ["a word"]}, no_word), (tie | {SV.REASON: 7}, no_word)):
        got = stop(one, four(first), controls=the_controls() + [off])
        assert (got.held, got.verdicts, got.why) == (0, (), (f"{a_control}{lack}: {off!r}",)), off


def under(doc: dict, records: list[dict]) -> list[dict]:
    """The records, naming ``doc`` as the registration their serve registered."""
    return [r | {CC.REGISTRATION: SV.digest_of(doc)} for r in records]


def at_rate(kind: str, rate: object, records: list[dict]) -> list[dict]:
    """The records, the one of ``kind`` saying it ran at ``rate``, or naming no rate where it is `...`."""
    i = CC.VERDICTS.index(kind)
    return [({k: v for k, v in r.items() if k != SV.RATE} if rate is ... else r | {SV.RATE: rate}) if j == i else r
            for j, r in enumerate(records)]


def test_a_class_is_held_only_at_the_rate_its_registration_registers_for_both_populations():
    """The rate is the registration's: the reader is handed the serve's registration, finds it by the digest the
    class's verdicts name, and holds the class only where it registers every block of the stop record's and the
    response tie's populations and each of those verdicts says it ran at that. A registration that was not handed
    or states no rate, another rate registered, and a verdict that names no rate or another than the registered one
    are each refused by name: among them a verdict that states `1/1` under a registration that says
    otherwise."""
    assert (SV.RATE, SV.EVERY_BLOCK) == ("rate", "1/1")
    assert SV.RATES == {CC.STOP_RECORD: ("rates", "stop_record"), CC.RESPONSE_TIE: ("rates", "response_tie")}
    assert REGISTERED == R.digest(SERVED) != OTHER and len(REGISTERED) == 128
    first, second, whole = head(3, 5, 0), head(4, 2, 0), LB.table(LB.inputs())
    one = [registration(first)]
    at = f"the link {LINK}, execution phase 3, class 0: held to its whole schedule by the rate: "
    assert stop(one, four(first)).held == 1
    # the registration is found by the digest the verdicts name: one that was not handed, or another serve's handed
    # in its place, leaves the rate unknown, whatever rate the verdicts state
    for served in ([], [OTHERS]):
        got = stop(one, four(first), served=served)
        assert got.held == 0 and len(got.verdicts) == 4 and got.why[0] == held_back(1, 1, rate=1)
        assert got.why[4] == at + (f"the registration its verdicts name, `{REGISTERED}`, is not among the serves' "
                                   "registrations read (`out/serve-registrations.jsonl`), so the rate it registers "
                                   "is not known") and len(got.why) == 5
        assert LB.table(LB.inputs(), got) == whole
    # a registration that states no rate for a population is refused by the name of where it would state it
    for key, kind in (("stop_record", CC.STOP_RECORD), ("response_tie", CC.RESPONSE_TIE)):
        for rates in ({k: v for k, v in SERVED["rates"].items() if k != key}, SERVED["rates"] | {key: 1},
                      SERVED["rates"] | {key: ""}, SERVED["rates"] | {key: None}):
            doc = SERVED | {"rates": rates}
            got = stop(one, under(doc, four(first)), served=[doc])
            assert got.held == 0 and got.why[4] == at + (
                f"the registration `{SV.digest_of(doc)}` states no rate for the {kind} (`rates.{key}`)"), (key, rates)
    for doc in ({k: v for k, v in SERVED.items() if k != "rates"}, SERVED | {"rates": "1/1"},
                SERVED | {"rates": ["1/1"] * 2}):
        got, name = stop(one, under(doc, four(first)), served=[doc]), SV.digest_of(doc)
        assert got.held == 0 and got.why[4] == at + (
            f"the registration `{name}` states no rate for the stop record (`rates.stop_record`); the registration "
            f"`{name}` states no rate for the response tie (`rates.response_tie`)"), doc
    for key, kind in (("stop_record", CC.STOP_RECORD), ("response_tie", CC.RESPONSE_TIE)):
        # a registration that registers another rate holds nothing, and a verdict that states `1/1` under it
        # is refused by name
        doc = SERVED | {"rates": SERVED["rates"] | {key: "1/8"}}
        got = stop(one, under(doc, four(first)), served=[doc])
        assert got.held == 0 and got.why[0] == held_back(1, 1, rate=1) and got.why[4] == at + (
            f"the registration registers the {kind} at `1/8`, not `1/1`; the {kind} of run of the {kind} "
            "states `1/1` under a registration that registers `1/8`"), kind
        # a verdict that says it ran at the registered rate is not refused, and its class is not held
        got = stop(one, at_rate(kind, "1/8", under(doc, four(first))), served=[doc])
        assert got.held == 0 and got.why[4] == at + (
            f"the registration registers the {kind} at `1/8`, not `1/1`"), kind
        # under a registration at `1/1`, a verdict that states another rate, or none, is refused by name
        for rate, says in (("1/8", "states `1/8`"), ("2/2", "states `2/2`"), ("rate:1/1", "states `rate:1/1`"),
                           (None, "names no rate (`rate`)"), (1, "names no rate (`rate`)"),
                           ("", "names no rate (`rate`)"), (..., "names no rate (`rate`)")):
            got = stop(one, at_rate(kind, rate, four(first)))
            assert got.held == 0 and got.why[4] == at + (
                f"the {kind} of run of the {kind} {says} under a registration that registers `1/1`"), (kind, rate)
        # a rate's text is `NUM/DEN` in lowest terms with no prefix: a registration that states every block in
        # another text registers another rate, whatever its verdicts state
        for text in ("2/2", "rate:1/1", "1/1 ", "1 / 1", "01/1", "1"):
            doc = SERVED | {"rates": SERVED["rates"] | {key: text}}
            for records in (under(doc, four(first)), at_rate(kind, text, under(doc, four(first)))):
                got = stop(one, records, served=[doc])
                assert got.held == 0 and got.why[4].startswith(at + (
                    f"the registration registers the {kind} at `{text}`, not `1/1`")), (kind, text)
    # the certified read and the compute audit are over no sampled population: a rate on their records is not read
    assert stop(one, [r | {SV.RATE: "1/8"} if i >= 2 else r for i, r in enumerate(four(first))]).held == 1
    # `complete` holds no class to a rate: four complete accepts under a registration at an eighth hold nothing
    eighth = OTHERS | {"rates": {"stop_record": "1/1", "response_tie": "1/8"}}
    theirs = at_rate(CC.RESPONSE_TIE, "1/8", under(eighth, four(second)))
    assert all(r[CC.VERDICT][CC.COMPLETE] is True for r in theirs)
    # two execution phases under two registrations: the one whose serve registered every block is held
    got = stop(one + [registration(second)], four(first) + theirs, served=[SERVED, eighth])
    assert got.held == 1 and got.why[0].startswith("1 of the 2 execution phase(s) read are charged below")
    assert got.why[6] == (f"the link {LINK}, execution phase 4, class 0: held to its whole schedule by the rate: the "
                          "registration registers the response tie at `1/8`, not `1/1`") and len(got.why) == 7
    # each class that holds a row is held to the rate of its own registration
    wide = head(3, 5, 2)
    wider = four(wide) + at_rate(CC.RESPONSE_TIE, "1/8", under(eighth, four(wide, 1)))
    got = stop([registration(wide)], wider, served=[SERVED, eighth])
    assert got.held == 0 and got.why[4] == (
        f"the link {LINK}, execution phase 3, class 1: held to its whole schedule by the rate: the registration "
        "registers the response tie at `1/8`, not `1/1`")
    assert stop([registration(wide)], four(wide) + four(wide, 1)).held == 1
    # held back by the rate and by a control: the first line counts each
    got = stop(one, four(first), controls=[], served=[])
    assert got.held == 0 and got.why[0] == held_back(1, 1, rate=1, control=1) and len(got.why) == 6
    # a serve's registration that doesn't read holds nothing at all
    got = stop(one, four(first), served=[SERVED, "a registration"])
    assert (got.held, got.verdicts, got.why) == (0, (), (
        "each connection is charged its whole schedule: a serve's registration doesn't read: it is no object: "
        "'a registration'",))


def of_serve(serve: str, records: list[dict]) -> list[dict]:
    """The records, naming ``serve`` as the served run their class came from."""
    return [r | {CC.SERVE: serve} for r in records]


def unlisted(record: dict) -> dict:
    """The record, its verdict of a format listed for none of the four."""
    return record | {CC.VERDICT: record[CC.VERDICT] | {SV.FORMAT: "another's"}}


BELOW = ("class 0: charged below its schedule, its 4 verdicts (stop record, response tie, certified read, compute "
         "audit) each accepted and complete on this head")
TWO_HEADS = f"each connection is charged its whole schedule: two heads are given for execution phase 3 of the link {LINK}: "
ONE_SERVE = "both of the serve the serve, as their verdict records name it"
NO_SERVE = "no verdict record names the serve of one of them, so they are not known to be two serves'"


def one_domain(table: str) -> str:
    return (f"their table `{table}` has one domain, so they are one session's execution phase, whatever serves their "
            "verdict records name")


def test_an_execution_phase_is_one_serves_and_a_heads_serve_is_the_one_its_records_name():
    """An execution phase is a serve's, a link's and a number's. A registration names no serve, so a head's serve
    is the one its verdict records name, and the answer's second line says that this is a run's word. Two heads
    for one number of a link are two execution phases only as two serves': where a serve is named for both, where
    no record names one for one of them, and where a table has one domain in both, they are one execution phase
    given two heads, which holds nothing. A head whose records name two serves is no one serve's."""
    assert SV.SERVE_NAMED == ("an execution phase is one serve's, and a head's serve is the one its verdict records "
                              "name, which is run, not proved")
    ours, theirs = head(3, 5, 0), head(3, 5, 0, domain=bytes([1]) * 64)
    both, other, whole = [registration(ours), registration(theirs)], of_serve("another serve", four(theirs)), LB.table(LB.inputs())
    # two serves' heads of one number of a link, each with its four: two execution phases, each held, and each
    # line names its serve
    got = stop(both, four(ours) + other)
    assert got.held == 2 and len(got.verdicts) == 8 and got.why == (
        f"2 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}", SV.SERVE_NAMED,
        f"the link {LINK}, execution phase 3 of the serve another serve, {BELOW}{refused()}",
        f"the link {LINK}, execution phase 3 of the serve another serve, class 1: a count of 0, so no connection is "
        "charged and no verdict is owed",
        f"the link {LINK}, execution phase 3 of the serve the serve, {BELOW}{refused()}",
        f"the link {LINK}, execution phase 3 of the serve the serve, class 1: a count of 0, so no connection is "
        "charged and no verdict is owed")
    assert stop(both[::-1], four(ours) + other).why == got.why
    # one of the two short of a verdict is at its whole schedule within its own serve, and the other is held; the
    # rule reads the other serve's stop record, of the same link, number and class, as one on another head
    got = stop(both, four(ours) + other[1:])
    assert got.held == 1 and got.why[2] == (
        f"the link {LINK}, execution phase 3 of the serve another serve, class 0: charged its whole schedule on each "
        "of its 5 rows: the stop record is not on this head (run of the stop record)")
    # the second line is there whatever is held, and a head alone at its number is named without its serve
    alone = stop(both[:1], four(ours)[1:])
    assert alone.why[1] == SV.SERVE_NAMED and alone.why[2].startswith(f"the link {LINK}, execution phase 3, class 0: ")
    # a serve named for both: one execution phase given two heads, which holds nothing at all; one record on the
    # second head that names the first's serve is enough, whichever verdict it is of and however it went, counted
    # or not
    assert stop(both, four(ours) + four(theirs)).why == (TWO_HEADS + ONE_SERVE,)
    for extra in (four(theirs)[0], four(theirs, accepted=False)[2], four(theirs, 1)[3], unlisted(four(theirs)[1])):
        for registrations in (both, both[::-1]):
            got = stop(registrations, four(ours) + other + [extra])
            assert (got.held, got.verdicts, got.why) == (0, (), (TWO_HEADS + ONE_SERVE,)), extra
    # no record names a serve for one of them, or for either: nothing says they are two serves'
    for records in (four(ours), other, [], four(head(9, 5, 0))):
        got = stop(both, records, ["r9 is not read"])
        assert (got.held, got.verdicts, got.why) == (0, (), (TWO_HEADS + NO_SERVE, "r9 is not read")), records
    # a record of a format listed for no verdict still names its head's serve: the other head is held
    got = stop(both, four(ours) + [unlisted(other[0])])
    assert got.held == 1 and got.why[2].startswith(f"the link {LINK}, execution phase 3 of the serve another serve, ")
    # a table of one name with one domain in both heads: one session's execution phase, whatever serves are named
    same = head(3, 6, 0)
    assert same.digest() != ours.digest() and same.tables[0].domain == ours.tables[0].domain
    got = stop([registration(ours), registration(same)], four(ours) + of_serve("another serve", four(same)))
    assert (got.held, got.verdicts, got.why) == (0, (), (TWO_HEADS + one_domain("end.0"),))
    # one such table is enough: the others' domains are another session's
    for table in ("lengths", "request.0", "end.1"):
        mixed = C.Head(LINK, 3, ZEROS, tuple(C.Entry(e.name, e.rows, e.row_bytes, ZEROS if e.name == table else e.domain,
                                                     e.root) for e in theirs.tables))
        for registrations in ([registration(ours), registration(mixed)], [registration(mixed), registration(ours)]):
            got = stop(registrations, four(ours) + of_serve("another serve", four(mixed)))
            assert (got.held, got.why) == (0, (TWO_HEADS + one_domain(table),)), table
    # and a domain shared by tables of two names is no table's in both
    crossed = C.Head(LINK, 3, ZEROS, tuple(C.Entry(e.name, e.rows, e.row_bytes, bytes([i + 1]) * 64, e.root)
                                           for i, e in enumerate(ours.tables)))
    turned = C.Head(LINK, 3, ZEROS, tuple(C.Entry(e.name, e.rows, e.row_bytes, bytes([len(ours.tables) - i]) * 64, e.root)
                                          for i, e in enumerate(ours.tables)))
    assert {e.domain for e in crossed.tables} == {e.domain for e in turned.tables} and len(ours.tables) % 2 == 0
    assert stop([registration(crossed), registration(turned)],
                four(crossed) + of_serve("another serve", four(turned))).held == 2
    # three heads of one number: each pair is held to the same
    third = head(3, 5, 0, domain=bytes([2]) * 64)
    three = both + [registration(third)]
    assert stop(three, four(ours) + other + of_serve("a third serve", four(third))).held == 3
    assert stop(three, four(ours) + other + of_serve("another serve", four(third))).why == (
        f"{TWO_HEADS}both of the serve another serve, as their verdict records name it",)
    # a head whose records name two serves is no one serve's execution phase: each class's four of its own serve
    wide = head(4, 5, 2)
    got = stop([registration(wide)], four(wide) + of_serve("another serve", four(wide, 1)))
    assert got.held == 0 and len(got.verdicts) == 8 and got.why[0] == held_back(1, 1, serve=1)
    assert got.why[4] == (f"the link {LINK}, execution phase 4: held to its whole schedule by its serve: its verdict "
                          "records name 2 serves, not one (another serve, the serve), and an execution phase is one "
                          "serve's") and len(got.why) == 5
    assert LB.table(LB.inputs(), got) == whole
    assert stop([registration(wide)], four(wide) + four(wide, 1)).held == 1
    # so is one on which a record that counts for nothing names another serve
    for extra in (unlisted(of_serve("another serve", four(wide))[0]), of_serve("another serve", four(wide, 7))[0]):
        got = stop([registration(wide)], four(wide) + four(wide, 1) + [extra])
        assert got.held == 0 and got.why[0] == held_back(1, 1, serve=1), extra
    # held back by its serve, the rate and a control: the first line counts each
    got = stop([registration(wide)], four(wide) + of_serve("another serve", four(wide, 1)), controls=[], served=[])
    assert got.held == 0 and got.why[0] == held_back(1, 1, rate=1, control=1, serve=1)


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
    assert stop(registrations, records).held == 3
    left_out = stop(registrations[:1] + registrations[2:], records)
    without = stop(registrations[:1] + registrations[2:], four(heads[0]) + four(heads[2]))
    refused = stop(registrations, four(heads[0]) + four(heads[2]) + [
        r | {CC.VERDICT: r[CC.VERDICT] | {CC.ACCEPTED: False}} if r[CC.RUN] == "run of the response tie" else r
        for r in four(middle)])
    assert left_out.held == without.held == refused.held == 2
    assert left_out.why[0] == f"2 of the 2 execution phase(s) read are charged below their schedule{EVERY_OTHER}"
    assert refused.why[0] == f"2 of the 3 execution phase(s) read are charged below their schedule{EVERY_OTHER}"
    assert refused.why[4] == (f"the link {LINK}, execution phase 4, class 0: charged its whole schedule on each of its "
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
    assert LB.table(inp, stop([], records)) == whole
    # an unsigned registration with no verdict on its head holds nothing: not with counts, and not with none
    forged = [registration(head(window, *counts)) for window, counts in ((6, (5, 0)), (7, (0, 0)), (8, ()))]
    assert stop(registrations + forged, records).held == 3 and stop(forged, records).held == 0
    assert LB.table(inp, stop(forged, [])) == whole


def test_a_verdict_whose_format_is_not_listed_reads_as_missing(monkeypatch):
    monkeypatch.undo()
    assert SV.FORMATS == {TIE: CC.RESPONSE_TIE}
    first = head(3, 5, 0)
    got = stop([registration(first)], four(first))
    assert got.held == 0 and [v["verdict"] for v in got.verdicts] == [CC.RESPONSE_TIE]
    assert got.why[2] == (f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each of its 5 "
                          "rows: the stop record is missing; the certified read is missing; the compute audit is "
                          "missing")
    assert got.why[-2:] == (
        "3 verdict record(s) read are of a format listed for none of the 4 verdicts, and count for nothing",
        "3 control record(s) read are of a format listed for none of the 4 verdicts, or are not the control their "
        "verdict's verifier owes, and count for nothing")
    for form in (..., None, 7, ["a list"]):
        tie = four(first)[1]
        tie[CC.VERDICT] = {k: v for k, v in (tie[CC.VERDICT] | {SV.FORMAT: form}).items() if v is not ...}
        assert stop([registration(first)], [tie]).verdicts == ()


NO_RECORD = ("a verdict record is an object with the fields run, serve, registration, verifier, verdict, proved_by, "
             "its verdict an object, its run, serve, registration named, its verifier by its module and commit, and "
             "its proved_by a name or null; of one read, ")


def test_anything_less_is_the_whole_schedule_with_a_line_that_says_why():
    assert stop([], []) == LB.NO_RECORDS
    assert stop([], [], ["r9 is not read"]) == LB.Stop(0, (*LB.NO_RECORDS.why, "r9 is not read"))
    whole = "each connection is charged its whole schedule: "
    first = head(3, 5)
    for registrations, records, why in (
            ([], four(first), "4 verdict record(s) were read and no registration (`out/registrations.jsonl`), so no "
                              "class's count is known"),
            ([registration(head(3, 0, 0))], four(head(3, 0, 0)), "no class holds a row in the 1 execution phase(s) read"),
            ([registration(first), registration(head(3, 6))], four(first),
             f"two heads are given for execution phase 3 of the link {LINK}: their table `end.0` has one domain, so "
             "they are one session's execution phase, whatever serves their verdict records name"),
            ([{"format": C.REGISTRATION}], four(first), "a registration of connection rows doesn't read: "
                                                        "KeyError('word_bits')"),
            ([registration(first)], [*four(first), "accept"], f"{NO_RECORD}it is no object: 'accept'"),
            ([registration(first)], [four(first)[0] | {CC.VERDICT: "accept"}],
             f"{NO_RECORD}its `verdict` is no object: {four(first)[0] | {CC.VERDICT: 'accept'}!r}")):
        got = stop(registrations, records)
        assert (got.held, got.verdicts, got.why[0]) == (0, (), whole + why), why
    # four verdicts that each accept on the head hold it, and hold nothing once one of them names no run, no serve,
    # no registration, or no verifier by its module and commit: a verdict nobody can trace to what produced it and
    # to what it is of, and the line names the field. So does one of a format listed for no verdict, which would
    # otherwise count for nothing
    assert stop([registration(first)], four(first)).held == 1
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
            got = stop([registration(first)], [*four(first)[:1], off, *four(first)[2:]])
            assert (got.held, got.verdicts, got.why[0]) == (0, (), f"{whole}{NO_RECORD}{lack}: {off!r}"), off
    assert stop([registration(first)], [*four(first), unlisted]).held == 1
    tampered = registration(first)
    tampered["tables"][2]["rows"] = 0
    got = stop([tampered], four(first))
    assert got.held == 0 and got.why[0] == whole + "the registration's certificate doesn't name its head's digest"
    uneven = C.Head(LINK, 3, ZEROS, tuple(C.Entry(e.name, 6 if e.name == "request.0" else e.rows, e.row_bytes, e.domain,
                                                  e.root) for e in first.tables))
    got = stop([registration(uneven)], four(uneven))
    assert got.held == 0 and got.why[0].startswith(whole + "the head states [5, 6] rows for class 0's three tables")
    # the same head given twice is one head, and one execution phase
    assert stop([registration(first), registration(first)], four(first)).held == 1
    # a superseded run's accept that is no verdict record holds nothing at all, as any record that is none does
    got = stop([registration(first)], four(first), spent=["accept"])
    assert (got.held, got.verdicts, got.why) == (0, (), (f"{whole}{NO_RECORD}it is no object: 'accept'",))


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
    assert (SV.RECORD, SV.RECORDS_FILES, SV.REGISTRATIONS_FILE, SV.CONTROLS_FILES, SV.SERVES_FILE) == (
        "exfiltration-glm53-stop-verdicts", "out/*-verdicts.jsonl", "out/registrations.jsonl",
        "out/*-controls.jsonl", "out/serve-registrations.jsonl")
    first = head(3, 5, 0)
    registrations, records = [registration(first)], four(first)
    stop_record, tie, read, audit = records
    of_stop, of_tie, of_read, of_audit = the_controls()
    store = Store(tmp_path, {
        "r1": {"out/stop-verdicts.jsonl": jsonl([stop_record]) + "\n", "out/certified-read-verdicts.jsonl": jsonl([read]),
               "out/compute-audit-verdicts.jsonl": jsonl([audit]), "verdicts.jsonl": "not read",
               "out/stop-controls.jsonl": jsonl([of_stop, of_read, of_audit]), "controls.jsonl": "not read"},
        "r2": {"out/response-tie-verdicts.jsonl": jsonl([tie]), SV.REGISTRATIONS_FILE: jsonl(registrations),
               "out/response-tie-controls.jsonl": jsonl([of_tie]), SV.SERVES_FILE: jsonl([SERVED]),
               "out/another.json": "{}"},
        "r3": {"out/another.json": "{}"}, "r4": None})
    assert SV.read(store, ["r1", "r2"]) == (registrations, [read, audit, stop_record, tie], [],
                                            [of_stop, of_read, of_audit, of_tie], [SERVED], [], [])
    neither = ("holds none of `out/*-verdicts.jsonl`, `out/registrations.jsonl`, `out/*-controls.jsonl`, "
               "`out/serve-registrations.jsonl`")
    got = SV.of_runs(store, ["r1", "r2", "r3", "r4", "r5"])
    assert got.held == 1 and len(got.verdicts) == 4 and got == stop(registrations, records, [
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
    # a line that is no JSON, in a run that is not superseded, holds nothing at all: the line names every run
    # handed, then the line, its file and its run
    broken = Store(tmp_path, {"r6": {"out/response-tie-verdicts.jsonl": jsonl([tie]) + "\n{not json\n"},
                              "r1": store.runs["r1"], "r2": store.runs["r2"]})
    got = SV.of_runs(broken, ["r1", "r2", "r6"])
    assert got.held == 0 and len(got.why) == 1 and got.why[0].startswith(
        "each connection is charged its whole schedule: the records of r1, r2, r6 don't read: line 3 of "
        "`out/response-tie-verdicts.jsonl` of r6 is no JSON: ")
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


def held_files(of: C.Head, records: list, controls: list | None = None, served: list | None = None) -> dict[str, str]:
    """A run's files: the registration of a head, verdict records, and, unless they are given, the four verifiers'
    controls and the serve's registration."""
    return {SV.REGISTRATIONS_FILE: jsonl([registration(of)]), "out/the-verdicts.jsonl": jsonl(records),
            "out/the-controls.jsonl": jsonl(the_controls() if controls is None else controls),
            SV.SERVES_FILE: jsonl([SERVED] if served is None else served)}


def tie_but(of: C.Head, **over: object) -> list[dict]:
    """The four verdicts' records on a head, the tie's verdict with ``over``."""
    return [r | {CC.VERDICT: r[CC.VERDICT] | over} if i == 1 else r for i, r in enumerate(four(of))]


def control_but(**over: object) -> list[dict]:
    """The four verifiers' controls, the tie's verdict with ``over``."""
    return [r | {CC.VERDICT: r[CC.VERDICT] | over} if i == 1 else r for i, r in enumerate(the_controls())]


def reason_but(reason: object) -> list[dict]:
    """The four verifiers' controls, the tie's refused for ``reason``."""
    return [r | {SV.REASON: reason} if i == 1 else r for i, r in enumerate(the_controls())]


def by_another(records: list[dict]) -> list[dict]:
    """The four verdicts' records, the tie's by a verifier that has no control of record."""
    return [r | {CC.VERIFIER: {"module": "a second tie's verifier", "commit": "a commit"}} if i == 1 else r
            for i, r in enumerate(records)]


def dropped(n: int, *runs: str) -> str:
    return (f"{n} verdict record(s) of the superseded run(s) {', '.join(runs)} are accepted and complete, and fill "
            "no verdict: a superseded run adds no accept, and what it refused stands")


def uncounted(n: int, *runs: str) -> str:
    return (f"{n} control record(s) of the superseded run(s) {', '.join(runs)} are refused for their control's word, "
            "and count for nothing: a superseded run adds no control, and one it did not refuse, or refused for "
            "another reason, stands")


def dont_read(n: int, *runs: str, where: str) -> str:
    return (f"{n} record(s) of the superseded run(s) {', '.join(runs)} don't read, and keep at its whole schedule "
            "each execution phase of a serve that was not launched after their run: such a record could have been "
            f"a refusal. The first is {where}")


def later(n: int, *runs: str) -> str:
    return (f"{n} verdict record(s) of {', '.join(runs)} name a serve launched after their own run, are accepted and "
            "complete, and fill no verdict: a run has no accept on a serve launched after it, and what such a "
            "record refused stands")


def kept_whole(at: str, newest: str, n: int = 1) -> str:
    return (f"{at}: held to its whole schedule by a record that doesn't read: its serve was not launched after "
            f"{newest}, the newest of the superseded runs that hold one ({n} such record(s))")


def test_a_superseded_run_adds_no_accept(tmp_path):
    """The reader is handed every run that carries the record, with which of them another run replaces. A
    superseded run's accepts count for nothing: not alone, not to fill in a verdict a later run lacks, and not as a
    copy in a later run's files."""
    first = head(3, 5, 0)
    records = four(first)
    copies = [r | {CC.RUN: "r1"} for r in records]
    store = Store(tmp_path, {"r1": held_files(first, records), "r2": {"out/another.json": "{}"},
                             "r3": held_files(first, records[:1] + records[2:]), "r4": held_files(first, copies)})
    neither = ("holds none of `out/*-verdicts.jsonl`, `out/registrations.jsonl`, `out/*-controls.jsonl`, "
               "`out/serve-registrations.jsonl`")
    assert SV.of_runs(store, ["r1", "r2"]).held == SV.of_runs(store, ["r1", "r2"], []).held == 1
    # the accepting run superseded by one that holds nothing: its four count for nothing, and its registration is
    # still read, so the class is named with what it lacks
    got = SV.of_runs(store, ["r1", "r2"], ["r1"])
    assert (got.held, got.verdicts) == (0, ()) and got.why[0] == LACK
    assert got.why[2].endswith(": the stop record is missing; the response tie is missing; the certified read is "
                               "missing; the compute audit is missing")
    assert got.why[-3:] == (f"r2 is not read: it {neither}", dropped(4, "r1"), uncounted(4, "r1"))
    assert LB.table(LB.inputs(), got) == LB.table(LB.inputs())
    # superseding the run that holds nothing changes nothing
    assert SV.of_runs(store, ["r1", "r2"], ["r2"]).held == 1
    # a superseded run's tie fills in no tie a later run lacks
    assert SV.of_runs(store, ["r1", "r3"]).held == 1
    got = SV.of_runs(store, ["r1", "r3"], ["r1"])
    assert got.held == 0 and len(got.verdicts) == 3 and got.why[2].endswith(": the response tie is missing")
    assert got.why[-2:] == (dropped(4, "r1"), uncounted(4, "r1"))
    # a record that names a superseded run as the run that produced it is that run's, whichever run's files hold it
    assert SV.of_runs(store, ["r4"]).held == 1
    got = SV.of_runs(store, ["r1", "r4"], ["r1"])
    assert (got.held, got.verdicts, got.why[-2:]) == (0, (), (dropped(8, "r1"), uncounted(4, "r1")))
    # `read` hands the accepts apart, leaves the refused controls out and keeps the registrations
    assert SV.read(store, ["r1", "r3"], ["r1"]) == (
        [registration(first)] * 2, records[:1] + records[2:], [dropped(4, "r1"), uncounted(4, "r1")],
        the_controls(), [SERVED] * 2, records, [])
    assert SV.read(store, ["r1", "r3"]) == ([registration(first)] * 2, records + records[:1] + records[2:], [],
                                            the_controls() * 2, [SERVED] * 2, [], [])
    # a superseded run's refused control backs no verifier: the four of a later run are then held back by a control
    alone = Store(tmp_path / "controls", {"old": held_files(first, []), "new": held_files(first, records, controls=[])})
    assert SV.of_runs(alone, ["old", "new"]).held == 1
    got = SV.of_runs(alone, ["old", "new"], ["old"])
    assert got.held == 0 and len(got.verdicts) == 4 and got.why[-1] == uncounted(4, "old")
    assert got.why[0].endswith("and none of them is held: 1 by a negative control, one refused by each verifier that "
                               "accepted, at its commit, for the control's word")
    # nor does a copy of it in a later run's files, which names the superseded run as the one that produced it
    copied = [c | {CC.RUN: "old"} for c in the_controls()]
    alone = Store(tmp_path / "copies", {"old": held_files(first, [], controls=[]),
                                        "new": held_files(first, records, controls=copied)})
    assert SV.of_runs(alone, ["old", "new"]).held == 1
    assert SV.of_runs(alone, ["old", "new"], ["old"]).held == 0
    # a control its verifier doesn't owe counts for nothing in any run: a superseded run's is handed on as such
    stray = [c | {SV.CONTROL: "max_tokens"} if i == 1 else c for i, c in enumerate(the_controls())]
    alone = Store(tmp_path / "stray", {"old": held_files(first, [], controls=stray),
                                       "new": held_files(first, records, controls=[])})
    assert SV.of_runs(alone, ["old", "new"], ["old"]).why[-2:] == (
        "1 control record(s) read are of a format listed for none of the 4 verdicts, or are not the control their "
        "verdict's verifier owes, and count for nothing", uncounted(3, "old"))
    # a control names no serve: one that carries a `serve` naming a superseded run is not that run's
    carried = [c | {CC.SERVE: "old"} for c in the_controls()]
    alone = Store(tmp_path / "carried", {"old": held_files(first, [], controls=[]),
                                         "new": held_files(first, records, controls=carried)})
    assert SV.of_runs(alone, ["old", "new"]).held == SV.of_runs(alone, ["old", "new"], ["old"]).held == 1
    # an accept that names a superseded run as its serve is that run's, whichever run produced it and holds it
    kept = {SV.REGISTRATIONS_FILE: jsonl([registration(first)]), SV.SERVES_FILE: jsonl([SERVED])}
    judged = {"out/the-verdicts.jsonl": jsonl(of_serve("s1", records)), "out/the-controls.jsonl": jsonl(the_controls())}
    served = Store(tmp_path / "serve", {"s1": kept, "v1": judged})
    assert SV.of_runs(served, ["s1", "v1"]).held == 1
    got = SV.of_runs(served, ["s1", "v1"], ["s1"])
    assert (got.held, got.verdicts) == (0, ()) and got.why[0] == LACK and got.why[-1] == dropped(4, "s1")
    assert SV.read(served, ["s1", "v1"], ["s1"]) == (
        [registration(first)], [], [dropped(4, "s1")], the_controls(), [SERVED], of_serve("s1", records), [])
    # a superseded serve's head refuses only within its own serve: a later serve's head of the same number of the
    # link, another session's, is held, and the superseded one is named with what it lacks
    theirs = head(3, 5, 0, domain=bytes([1]) * 64)
    two = Store(tmp_path / "serves", {
        "s1": kept, "v1": judged, "s2": {SV.REGISTRATIONS_FILE: jsonl([registration(theirs)])},
        "v2": {"out/the-verdicts.jsonl": jsonl(of_serve("s2", four(theirs)))}})
    runs = ["s1", "v1", "s2", "v2"]
    assert SV.of_runs(two, runs).held == 2
    got = SV.of_runs(two, runs, ["s1"])
    assert got.held == 1 and len(got.verdicts) == 4 and got.why[-1] == dropped(4, "s1")
    assert got.why[2] == (f"the link {LINK}, execution phase 3 of the serve s1, class 0: charged its whole schedule "
                          "on each of its 5 rows: " + "; ".join(
                              f"the {kind} is not on this head (run of the {kind})" for kind in CC.VERDICTS))
    assert got.why[4] == f"the link {LINK}, execution phase 3 of the serve s2, {BELOW}{refused()}"
    assert SV.of_runs(two, runs, ["s2"]).held == 1 and SV.of_runs(two, runs, ["s1", "s2"]).held == 0


def run_files() -> dict[str, dict[str, str]]:
    """A run's files of each kind these tests hand the reader, by what they hold: the four verdicts' records of a
    class on a head with the verifiers' controls and the serve's registration, and each way of being less."""
    first, other, second = head(3, 5, 0), head(3, 6, 0), head(4, 2, 0)
    theirs = head(3, 5, 0, domain=bytes([1]) * 64)
    return {"the four": held_files(first, four(first)),
            "a tie refused": held_files(first, tie_but(first, accepted=False)),
            "a tie not complete": held_files(first, tie_but(first, complete=False)),
            "three verdicts": held_files(first, four(first)[1:]), "another head": held_files(other, four(other)),
            "another execution phase": held_files(second, four(second)), "no record": held_files(first, []),
            "no registration": {"out/the-verdicts.jsonl": jsonl(four(first))}, "nothing": {"out/another.json": "{}"},
            "a record that doesn't read": held_files(first, [{CC.VERDICT: {}}]),
            "the four and no control": held_files(first, four(first), controls=[]),
            "the four and a control accepted": held_files(first, four(first), controls=control_but(accepted=True)),
            "the four and a control refused for another reason": held_files(first, four(first), controls=reason_but("parse")),
            "the four and a control that doesn't read": held_files(first, four(first), controls=[{CC.VERDICT: {}}]),
            "the four and no serve's registration": held_files(first, four(first), served=[]),
            "the four of another serve": held_files(first, of_serve("another serve", four(first))),
            "the four under another registration": held_files(first, under(OTHERS, four(first)), served=[OTHERS]),
            "the four, the tie at another rate": held_files(first, at_rate(CC.RESPONSE_TIE, "1/8", four(first))),
            "the four, the tie by a verifier with no control": held_files(first, by_another(four(first))),
            "the four of the run old as their serve": held_files(first, of_serve("old", four(first))),
            "the four of the run new as their serve": held_files(first, of_serve("new", four(first))),
            "another serve's head of the number": held_files(theirs, of_serve("another serve", four(theirs))),
            "a record of no listed format, of another serve": held_files(
                first, [unlisted(of_serve("another serve", four(first))[0])]),
            "the four and a registration that doesn't read": held_files(first, four(first)) | {
                SV.REGISTRATIONS_FILE: jsonl([registration(first), {"format": "another's"}])},
            "the four and a serve's registration that doesn't read": held_files(
                first, four(first), served=[SERVED, "no object"])}


def test_a_superseded_run_takes_away_no_refusal(tmp_path):
    """What a superseded run refused on a head stands, whatever a later run accepted on it: a refused verdict, an
    accept that is not complete, a record that doesn't read, and a registration that gives the execution phase
    another head. The run must be among the runs handed for that to be read, so a superseded run that is not
    holds everything at its whole schedule. And superseding a run never raises what is held, where no serve is
    launched after it: here every run and serve is named in words, and no word is a run id."""
    first = head(3, 5, 0)
    accepts = held_files(first, four(first))
    for old, lack in ((tie_but(first, accepted=False), "the response tie is refused (run of the response tie)"),
                      (tie_but(first, complete=False), "the response tie is not complete (run of the response tie)")):
        store = Store(tmp_path / lack, {"old": held_files(first, old), "new": accepts})
        for superseded in ((), ["old"]):
            got = SV.of_runs(store, ["old", "new"], superseded)
            assert got.held == 0 and got.why[0] == LACK, (lack, superseded)
            assert got.why[2] == (f"the link {LINK}, execution phase 3, class 0: charged its whole schedule on each "
                                  f"of its 5 rows: {lack}")
        assert got.why[-2:] == (dropped(3, "old"), uncounted(4, "old"))
        # the later run alone holds the execution phase, which is why the superseded one is to be handed
        assert SV.of_runs(store, ["new"]).held == 1
        unread = SV.of_runs(store, ["new"], ["old"])
        assert unread == LB.Stop(0, (
            "each connection is charged its whole schedule: the superseded run(s) old are not among the runs "
            "handed, so what they refused was not read: a superseded run adds no accept and takes away no refusal",))
    # a superseded run's record that doesn't read holds nothing, as any run's does, where the serve is not known
    # to be launched after that run: these runs and this serve are named in words, and no word is a run id
    store = Store(tmp_path / "unread", {"old": held_files(first, [*four(first), "accept"]), "new": accepts})
    got = SV.of_runs(store, ["old", "new"])
    assert got.held == 0 and got.why[0].endswith(f"{NO_RECORD}it is no object: 'accept'")
    got = SV.of_runs(store, ["old", "new"], ["old"])
    assert got.held == 0 and got.why[0] == held_back(1, 1, unread=1) and got.why[-1] == dont_read(
        1, "old", where="line 5 of `out/the-verdicts.jsonl` of old: it is no object")
    assert kept_whole(f"the link {LINK}, execution phase 3", "old") in got.why
    # it is handed apart, as its run alone, and is none of the run's accepts
    handed = SV.read(store, ["old", "new"], ["old"])
    assert handed[1] == four(first) and handed[5] == four(first) and handed[2][0] == dropped(4, "old")
    assert handed[6] == ["old"]
    # and the registration it kept is read as any run's: another head for the execution phase holds nothing
    other = head(3, 6, 0)
    store = Store(tmp_path / "heads", {"old": held_files(other, four(other)), "new": accepts})
    for superseded in ((), ["old"]):
        got = SV.of_runs(store, ["old", "new"], superseded)
        assert got.held == 0 and got.why[0] == (
            f"each connection is charged its whole schedule: two heads are given for execution phase 3 of the link "
            f"{LINK}: both of the serve the serve, as their verdict records name it")
    # a control a superseded run's verifier did not refuse stands: the later run's four, by that verifier at that
    # commit, are held back by it whether the run is superseded or not, and whatever control the later run refused
    store = Store(tmp_path / "accepted", {"old": held_files(first, [], controls=control_but(accepted=True)),
                                          "new": accepts})
    for superseded in ((), ["old"]):
        got = SV.of_runs(store, ["old", "new"], superseded)
        assert got.held == 0 and got.why[4] == (
            f"the link {LINK}, execution phase 3: held to its whole schedule by a control: the response tie's "
            "verifier the response tie's verifier at a commit did not refuse its negative control (`id`: a released "
            "id changed) on control run of the response tie")
    assert got.why[-1] == uncounted(3, "old")
    # so does a control a superseded run's verifier refused for another reason than the control's word
    store = Store(tmp_path / "reason", {"old": held_files(first, [], controls=reason_but("parse")), "new": accepts})
    for superseded in ((), ["old"]):
        got = SV.of_runs(store, ["old", "new"], superseded)
        assert got.held == 0 and got.why[4] == (
            f"the link {LINK}, execution phase 3: held to its whole schedule by a control: the response tie's "
            "verifier the response tie's verifier at a commit: control id refused for 'parse', not 'id', on control "
            "run of the response tie")
    assert got.why[-1] == uncounted(3, "old")
    # an accept of a superseded run fills no verdict and still stands against the others on its head: the serve
    # and the registration it names, the rate it states and its verifier are held to what holds any record's, so
    # what kept the class at its whole schedule before its run was superseded keeps it there after
    at = f"the link {LINK}, execution phase 3"
    stands = {
        "a serve": (of_serve("another serve", four(first)),
                    f"{at}, class 0: charged its whole schedule on each of its 5 rows: its verdicts on this head name "
                    "2 `serve`s, not one (another serve, the serve)"),
        "a registration": (under(OTHERS, four(first)),
                           f"{at}, class 0: charged its whole schedule on each of its 5 rows: its verdicts on this "
                           f"head name 2 `registration`s, not one ({', '.join(sorted((REGISTERED, OTHER)))})"),
        "a rate": (at_rate(CC.RESPONSE_TIE, "1/8", four(first)),
                   f"{at}, class 0: held to its whole schedule by the rate: the response tie of run of the response "
                   "tie states `1/8` under a registration that registers `1/1`"),
        "a verifier": (by_another(four(first)),
                       f"{at}: held to its whole schedule by a control: "
                       + no_control(CC.RESPONSE_TIE, module="a second tie's verifier"))}
    for name, (old, line) in stands.items():
        store = Store(tmp_path / name, {"old": held_files(first, old, served=[SERVED, OTHERS]), "new": accepts})
        for superseded in ((), ["old"]):
            got = SV.of_runs(store, ["old", "new"], superseded)
            assert got.held == 0 and line in got.why, (name, superseded)
        assert len(got.verdicts) == 4 and got.why[-2] == dropped(4, "old"), name
        assert SV.of_runs(store, ["new"]).held == 1
    # superseding a run never raises what is held, whatever the two runs hold
    kept = run_files()
    seen = lowered = most = 0
    for i, (a, old) in enumerate(kept.items()):
        for j, (b, new) in enumerate(kept.items()):
            store = Store(tmp_path / f"pair-{i}-{j}", {"old": old, "new": new})
            both = SV.of_runs(store, ["old", "new"]).held
            for superseded in (["old"], ["new"], ["old", "new"]):
                got = SV.of_runs(store, ["old", "new"], superseded).held
                assert got <= both, (a, b, superseded)
                seen, lowered, most = seen + 1, lowered + (got < both), max(most, both)
    assert (seen, most) == (3 * len(kept) ** 2, 2) == (1875, 2) and lowered


#: run ids in the order of their launch; `RUN` and `TWIN` state one second
EARLY, RUN, TWIN, LATE, LAST = ("r20260102-030405-0a0a", "r20260102-030406-0b0b", "r20260102-030406-ffff",
                                "r20260102-030407-0000", "r20270101-000000-0c0c")


def test_the_order_of_runs_is_read_from_their_ids():
    """One run is launched after another only where both names are run ids and the first states a later second:
    never where one of them is no run id, which is the side that keeps a schedule whole."""
    ids = (EARLY, RUN, TWIN, LATE, LAST)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            assert SV.launched_after(a, b) is (i > j and (a, b) != (TWIN, RUN)), (a, b)
    for word in ("the serve", "", "old", LAST.upper(), LAST[1:], f"{LAST}-2", f"x{LAST}", LAST[:-1], f"{LAST}\n",
                 LAST.replace("-", "_"), "r2027010a-000000-0c0c", None, 20270101, [LAST], {"run": LAST}):
        for other in (EARLY, word):
            assert not SV.launched_after(word, other) and not SV.launched_after(other, word), word


def why_not(read, x: object) -> str:
    try:
        read(x)
    except ValueError as e:
        return str(e)
    raise AssertionError(x)


def test_a_superseded_runs_record_that_doesnt_read_keeps_whole_each_serve_not_launched_after_it(tmp_path):
    """A record of a superseded run that doesn't read could have been a refusal, so it keeps at its whole schedule
    each execution phase of a serve that was not launched after that run, with a line that names the run, and
    charges nothing on a serve launched after it: a bad file is mended by a new serve, never by a second run over
    the old one. It is one rule for every line of the run's files that doesn't read: a verdict record, a control
    record, a registration of connection rows, a serve's registration, and a line of any of them that is no JSON.
    In a run that is not superseded such a line holds no execution phase at all, whatever the serve."""
    first = head(3, 5, 0)
    at = f"the link {LINK}, execution phase 3"
    record, text = {CC.VERDICT: {}}, "{no json"
    no_json = "it is no JSON: " + why_not(json.loads, text)
    lines = {"a verdict record": ("out/a-verdicts.jsonl", jsonl([record]), CC.lacks(record)),
             "a control record": ("out/a-controls.jsonl", jsonl([record]), SV.control_lacks(record)),
             "a registration of connection rows": (SV.REGISTRATIONS_FILE, jsonl([{"format": "another's"}]),
                                                   why_not(SV.head_of, {"format": "another's"})),
             "a serve's registration": (SV.SERVES_FILE, jsonl(["no object"]), why_not(SV.digest_of, "no object")),
             **{f"no JSON in {path}": (path, text + "\n", no_json) for path in (
                 "out/a-verdicts.jsonl", "out/a-controls.jsonl", SV.REGISTRATIONS_FILE, SV.SERVES_FILE)}}
    assert len(lines) == 8 and all(lack for _, _, lack in lines.values())
    after = 0
    for i, (name, (path, bad, lack)) in enumerate(lines.items()):
        last = dont_read(1, RUN, where=f"line 1 of `{path}` of {RUN}: {lack}")
        for j, serve in enumerate((EARLY, RUN, TWIN, LATE, "the serve")):
            store = Store(tmp_path / f"{i}-{j}", {RUN: {path: bad}, "old": {path: bad},
                                                  LAST: held_files(first, of_serve(serve, four(first)))})
            assert SV.of_runs(store, [LAST]).held == 1
            # in a run that is not superseded: nothing is held, and the one line says what doesn't read
            live = SV.of_runs(store, [RUN, LAST])
            assert live.held == 0 and len(live.why) == 1 and lack.removeprefix("it is ") in live.why[0], (name, serve)
            got = SV.of_runs(store, [RUN, LAST], [RUN])
            if serve == RUN:
                # the superseded run as the serve: the accepts that name it fill no verdict, as built
                assert (got.held, got.verdicts, got.why[-2:]) == (0, (), (dropped(4, RUN), last)), name
                continue
            # superseded: held only where the serve was launched after the run
            assert got.held == (serve == LATE) and got.why[-1] == last and len(got.verdicts) == 4, (name, serve)
            assert (kept_whole(at, RUN) in got.why) == (serve != LATE), (name, serve)
            assert got.why[0] == (f"1 of the 1 execution phase(s) read are charged below their schedule{EVERY_OTHER}"
                                  if serve == LATE else held_back(1, 1, unread=1)), (name, serve)
            after += got.held
            # a superseded run whose name is no run id: no serve is launched after it
            worded = SV.of_runs(store, ["old", LAST], ["old"])
            assert worded.held == 0 and kept_whole(at, "old") in worded.why, (name, serve)
            assert worded.why[-1] == dont_read(1, "old", where=f"line 1 of `{path}` of old: {lack}")
    assert after == len(lines)
    # an execution phase is held only where its serve was launched after each such run, and the line names those
    # it was not launched after
    bad = {"out/a-verdicts.jsonl": "\n\n" + jsonl(["accept", record])}
    where = f"line 3 of `out/a-verdicts.jsonl` of {EARLY}: it is no object"
    for serve, held in ((TWIN, 0), (LAST, 1)):
        store = Store(tmp_path / f"two-{held}", {EARLY: bad, LATE: bad,
                                                 LAST: held_files(first, of_serve(serve, four(first)))})
        got = SV.of_runs(store, [EARLY, LATE, LAST], [EARLY, LATE])
        assert got.held == held and got.why[-1] == dont_read(4, EARLY, LATE, where=where)
        assert (kept_whole(at, LATE, 4) in got.why) == (not held)
        assert SV.read(store, [EARLY, LATE, LAST], [EARLY, LATE])[6] == [EARLY, EARLY, LATE, LATE]
        # each alone superseded, the other's record holds nothing at all
        for one in (EARLY, LATE):
            assert SV.of_runs(store, [EARLY, LATE, LAST], [one]).held == 0
        # a name that is no run id is the newest: no serve is launched after it
        store = Store(tmp_path / f"word-{held}", {"old": bad, LATE: bad, LAST: store.runs[LAST]})
        got = SV.of_runs(store, ["old", LATE, LAST], ["old", LATE])
        assert got.held == 0 and kept_whole(at, "old", 4) in got.why
    # two serves' execution phases: the one launched after the run is held, the one launched before it is not
    second = head(4, 2, 0)
    both = {SV.REGISTRATIONS_FILE: jsonl([registration(first), registration(second)]),
            "out/the-verdicts.jsonl": jsonl(of_serve(EARLY, four(first)) + of_serve(LATE, four(second))),
            "out/the-controls.jsonl": jsonl(the_controls()), SV.SERVES_FILE: jsonl([SERVED])}
    store = Store(tmp_path / "serves", {RUN: bad, LAST: both})
    assert SV.of_runs(store, [LAST]).held == 2 and SV.of_runs(store, [RUN, LAST]).held == 0
    got = SV.of_runs(store, [RUN, LAST], [RUN])
    assert got.held == 1 and got.why[0].startswith("1 of the 2 execution phase(s) read are charged below")
    assert kept_whole(at, RUN, 2) in got.why and not any("execution phase 4: held" in line for line in got.why)
    # the first line counts the execution phases kept whole beside those another rule holds back
    store = Store(tmp_path / "rate", {RUN: bad, LAST: both | {
        "out/the-verdicts.jsonl": jsonl(of_serve(EARLY, four(first))
                                        + at_rate(CC.RESPONSE_TIE, "1/8", of_serve(LATE, four(second))))}})
    assert SV.of_runs(store, [RUN, LAST], [RUN]).why[0] == held_back(2, 2, rate=1, unread=1)
    # a record that doesn't read in the files of a run that is not superseded holds nothing, whichever run it names
    store = Store(tmp_path / "named", {RUN: {"out/another.json": "{}"},
                                       LAST: held_files(first, [*of_serve(LATE, four(first)), {CC.RUN: RUN}])})
    got = SV.of_runs(store, [RUN, LAST], [RUN])
    assert got.held == 0 and len(got.why) == 2 and got.why[1].startswith(f"{RUN} is not read: ")


def test_a_run_has_no_accept_on_a_serve_launched_after_it(tmp_path):
    """A verdict record that names a serve launched after its own run, the one whose files hold it or the one it
    names, fills no verdict where it is accepted and complete, with a line that names the run; what such a record
    refused stands, and its accept is still held to what holds any record on its head."""
    first = head(3, 5, 0)

    def of(name: str, holder: str, serve: str, run: str | None = None, only: int | None = None) -> Store:
        records = of_serve(serve, four(first))
        records = [r | {CC.RUN: run} if run and only in (None, i) else r for i, r in enumerate(records)]
        return Store(tmp_path / name, {holder: held_files(first, records)})

    # a serve launched before its verifiers' run, in its second, or named by no run id: its accepts count
    for i, (holder, serve) in enumerate(((LATE, EARLY), (RUN, TWIN), (TWIN, RUN), (RUN, RUN), (EARLY, "the serve"),
                                         ("a run", LATE))):
        got = SV.of_runs(of(f"counts-{i}", holder, serve), [holder])
        assert got.held == 1 and not any("launched after" in line for line in got.why), (holder, serve)
    # a serve launched after the run whose files hold the record
    store = of("holder", EARLY, LATE)
    got = SV.of_runs(store, [EARLY])
    assert (got.held, got.verdicts, got.why[0], got.why[-1]) == (0, (), LACK, later(4, EARLY))
    assert SV.read(store, [EARLY])[1::2] == ([], the_controls(), of_serve(LATE, four(first)))
    # a serve launched after the run a record names as the one that produced it, whichever run's files hold it
    got = SV.of_runs(of("named", LAST, LATE, run=EARLY), [LAST])
    assert (got.held, got.verdicts, got.why[0], got.why[-1]) == (0, (), LACK, later(4, EARLY))
    got = SV.of_runs(of("one", LAST, LATE, run=EARLY, only=1), [LAST])
    assert got.held == 0 and len(got.verdicts) == 3 and got.why[-1] == later(1, EARLY)
    assert got.why[2].endswith(": the response tie is missing")
    # after both: counted once, under the run whose files hold it
    assert SV.of_runs(of("both", EARLY, LATE, run=RUN), [EARLY]).why[-1] == later(4, EARLY)
    # a superseded run's accept is counted as that run's alone
    got = SV.of_runs(of("old", EARLY, LATE), [EARLY], [EARLY])
    assert got.held == 0 and got.why[-2:] == (dropped(4, EARLY), uncounted(4, EARLY))
    # what such a record refused stands against a later run's accepts on its head
    accepts = held_files(first, of_serve(LATE, four(first)))
    store = Store(tmp_path / "refused", {EARLY: held_files(first, of_serve(LATE, tie_but(first, accepted=False))[1:2]),
                                         LAST: accepts})
    got = SV.of_runs(store, [EARLY, LAST])
    assert got.held == 0 and SV.of_runs(store, [LAST]).held == 1
    assert got.why[2].endswith(": the response tie is refused (run of the response tie)")
    assert not any("launched after" in line for line in got.why)
    # and its accept is still read on its head: the serve it names is held to the one the others name
    store = Store(tmp_path / "serve", {EARLY: accepts, LAST: held_files(first, of_serve(EARLY, four(first)))})
    got = SV.of_runs(store, [EARLY, LAST])
    assert got.held == 0 and SV.of_runs(store, [LAST]).held == 1 and got.why[-1] == later(4, EARLY)
    assert got.why[2].endswith(f"its verdicts on this head name 2 `serve`s, not one ({EARLY}, {LATE})")


def test_a_verdict_that_doesnt_count_counts_against_its_execution_phase_whatever_head_it_names(tmp_path):
    """Refused under one head, registered again under another and accepted would be a second try at a verdict. So
    a verdict that is refused, or accepted and not complete, counts against the execution phase it names by its
    serve, its link and its number, whatever head it names, where that phase's head was read: the phase is
    charged its whole schedule, every class of it, with a line that names the verdict, the class it names and its
    run. It holds whatever class it names, one with a count of 0 under the head read too: two heads need not sort
    a phase's rows into the same classes. An accept under another head counts for nothing, as does a verdict that
    names another serve, link or number."""
    first, second, other = head(3, 5, 0), head(4, 2, 0), head(3, 6, 0)
    registrations, records = [registration(first), registration(second)], four(first) + four(second)
    assert stop(registrations, records).held == 2
    at = f"the link {LINK}, execution phase 3"
    by = "held to its whole schedule, every class of it, by a verdict under another head: "
    nothing = ("1 verdict record(s) read are on no head read, or on no class of one that holds a row, and count for "
               "nothing")
    tie = four(other)[1] | {CC.RUN: "an earlier run"}
    refusal = tie | {CC.VERDICT: tie[CC.VERDICT] | {CC.ACCEPTED: False}}

    def on(**verdict: object) -> dict:
        return refusal | {CC.VERDICT: refusal[CC.VERDICT] | verdict}

    for word, over in ((CC.REFUSED, {CC.ACCEPTED: False}), (CC.REFUSED, {CC.ACCEPTED: None}),
                       (CC.REFUSED, {CC.ACCEPTED: "true"}), (CC.NOT_COMPLETE, {CC.COMPLETE: False})):
        got = stop(registrations, [*records, tie | {CC.VERDICT: tie[CC.VERDICT] | over}])
        assert got.held == 1 and len(got.verdicts) == 8 and nothing not in got.why, over
        assert f"{at}: {by}the response tie is {word} on class 0 (an earlier run)" in got.why, over
        assert not any("execution phase 4: held" in line for line in got.why)
    # whatever class it names: one with a count of 0 under the head read, and one that head doesn't list
    for cls in (1, 2):
        got = stop(registrations, [*records, on(**{CC.CLASS: cls})])
        assert got.held == 1 and nothing not in got.why, cls
        assert f"{at}: {by}the response tie is refused on class {cls} (an earlier run)" in got.why, cls
    # so it keeps every class of the phase whole: two classes that hold a row, each with its four, and a refusal
    # under another head on a third, whose count under the head read is 0
    wide = head(3, 5, 4, 0)
    both = four(wide) + four(wide, cls=1)
    assert stop([registration(wide)], both).held == 1
    got = stop([registration(wide)], [*both, on(**{CC.CLASS: 2})])
    assert got.held == 0 and got.why[0] == held_back(1, 1, head=1)
    assert got.why[2:6] == (
        f"{at}, {BELOW}{refused()}", f"{at}, {BELOW.replace('class 0', 'class 1')}{refused()}",
        f"{at}, class 2: a count of 0, so no connection is charged and no verdict is owed",
        f"{at}: {by}the response tie is refused on class 2 (an earlier run)")
    # an accept under another head counts for nothing
    got = stop(registrations, [*records, tie])
    assert got.held == 2 and nothing in got.why
    # nor does a refusal of another serve, link or number, or of a format listed for none of the four
    for record, verdict in (({CC.SERVE: "another serve"}, {}), ({}, {CC.LINK: "requests"}), ({}, {CC.WINDOW: 5}),
                            ({}, {SV.FORMAT: "another's"})):
        got = stop(registrations, [*records, on(**verdict) | record])
        assert got.held == 2 and not any("under another head" in line for line in got.why), (record, verdict)
        assert (nothing in got.why) == (SV.FORMAT not in verdict), (record, verdict)
    # it counts against the phase it names and no other
    got = stop(registrations, [*records, on(**{CC.WINDOW: 4})])
    assert got.held == 1 and any(line.startswith(f"the link {LINK}, execution phase 4: {by}") for line in got.why)
    # the first line counts it, beside what another rule holds back
    assert stop(registrations[:1], [*four(first), refusal]).why[0] == held_back(1, 1, head=1)
    got = stop(registrations, [*four(first), *at_rate(CC.RESPONSE_TIE, "1/8", four(second)), refusal])
    assert got.why[0] == held_back(2, 2, rate=1, head=1)
    # an execution phase that lacks a verdict is named with it too
    got = stop(registrations[:1], [*four(first)[1:], refusal])
    assert got.held == 0 and got.why[0] == LACK and nothing not in got.why and got.why[4] == (
        f"{at}: {by}the response tie is refused on class 0 (an earlier run)")
    # with the other head's registration handed too, the phase is given two heads, which holds nothing at all
    got = stop([*registrations, registration(other)], [*records, refusal])
    assert got.held == 0 and got.why[0] == TWO_HEADS + ONE_SERVE
    # a superseded run's refusal stands under another head as it does on the one read
    store = Store(tmp_path, {"old": {"out/the-verdicts.jsonl": jsonl([refusal])},
                             "new": held_files(first, four(first))})
    assert SV.of_runs(store, ["new"]).held == 1
    for superseded in ((), ["old"]):
        assert SV.of_runs(store, ["old", "new"], superseded).held == 0


def renamed(files: dict[str, str], names: dict[str, str]) -> dict[str, str]:
    """A run's files with each word for a serve replaced by a name."""
    out = {}
    for path, text in files.items():
        for word, name in names.items():
            text = text.replace(json.dumps(word), json.dumps(name))
        out[path] = text
    return out


def test_superseding_a_run_raises_what_is_held_only_on_a_serve_launched_after_it(tmp_path):
    """Labelling a run superseded raises what is held only on serves that did not exist when that run was
    launched, and only where the run holds a record that doesn't read. Over each pair of the kinds of run files,
    the runs named by run ids: with every serve launched before both runs, superseding the one, the other or both
    never raises what is held; with a serve launched between them it does, and each time the superseded run holds
    such a record and a serve the records name was launched after it."""
    first = head(3, 5, 0)
    kinds = run_files() | {"the four and a line that is no JSON": held_files(first, four(first)) | {
        "out/more-verdicts.jsonl": "{no json\n"}}
    raised = {}
    for serve in (EARLY, LATE):
        files = {kind: renamed(its, {"the serve": serve, "another serve": "r20260101-000000-0d0d"})
                 for kind, its in kinds.items()}
        raised[serve] = seen = 0
        for i, (a, old) in enumerate(files.items()):
            for j, (b, new) in enumerate(files.items()):
                store = Store(tmp_path / f"{serve}-{i}-{j}", {RUN: old, LAST: new})
                both = SV.of_runs(store, [RUN, LAST]).held
                for superseded in ([RUN], [LAST], [RUN, LAST]):
                    seen += 1
                    if SV.of_runs(store, [RUN, LAST], superseded).held <= both:
                        continue
                    handed = SV.read(store, [RUN, LAST], superseded)
                    serves = {r[CC.SERVE] for r in handed[1] + handed[5] if CC.reads(r)}
                    assert any(SV.launched_after(s, run) for run in handed[6] for s in serves), (a, b, superseded)
                    raised[serve] += 1
        assert seen == 3 * len(kinds) ** 2 == 2028
    assert raised[EARLY] == 0 and raised[LATE]


def things(files: dict[str, str]) -> list[tuple[str, object]]:
    """What a run's files hold that the reader reads, each with the kind of file that holds it."""
    kinds = ((SV.REGISTRATIONS_FILE, "registration"), (SV.SERVES_FILE, "serve"), (SV.CONTROLS_FILES, "control"),
             (SV.RECORDS_FILES, "verdict"))
    return [(kind, json.loads(line)) for path, text in files.items() for pattern, kind in kinds
            if fnmatch.fnmatchcase(path, pattern) for line in text.splitlines()]


def less_by_one(files: dict[str, str]):
    """Each way of leaving one thing out of a run's files, with what is left: a whole file, and one record of one."""
    for path, text in files.items():
        yield f"the file {path}", {p: t for p, t in files.items() if p != path}
        lines = text.splitlines(keepends=True)
        for i in range(len(lines)):
            yield f"record {i} of {path}", files | {path: "".join(lines[:i] + lines[i + 1:])}


def _head(doc: object) -> C.Head | None:
    """The head a registration of connection rows lists, or None where the reader says it doesn't read."""
    return None if SV._lack(SV.REGISTRATIONS_FILE, doc) else SV.head_of(doc)


def _counts(control: dict) -> bool:
    kind = SV.FORMATS[control[CC.VERDICT][SV.FORMAT]]
    return control[CC.VERDICT].get(CC.ACCEPTED) is False and control[SV.REASON] == SV.REASONS[kind]


#: the ways a thing the reader is handed stands against another: each is a refusal by the reader's rules, so
#: leaving such a thing out may raise what is held, and leaving out anything else never does
STANDS = ("a registration of connection rows that doesn't read", "a second head of an execution phase",
          "a serve's registration that doesn't read", "a control record that doesn't read",
          "a control that was not refused for its control's word", "a verdict record that doesn't read",
          "a verdict that is refused or not complete",
          "an accept of another serve, or under another registration, than a record that stays on its head",
          "an accept at another rate than every block", "an accept by a verifier without its control")


def stands_against(kind: str, x: object, stays: list[tuple[str, object]], controls: list[object]) -> str | None:
    """How a thing of a ``kind`` of file stands against what ``stays`` when it is left out, one of `STANDS`, or
    None where it refuses nothing; ``controls`` are every control record handed."""
    if kind == "registration":
        if (head := _head(x)) is None:
            return STANDS[0]
        others = [h for k, y in stays if k == kind and (h := _head(y))]
        second = any((h.link, h.window) == (head.link, head.window) and h.digest() != head.digest() for h in others)
        return STANDS[1] if second else None
    if kind == "serve":
        try:
            SV.digest_of(x)
        except ValueError:
            return STANDS[2]
        return None
    if kind == "control":
        if SV.control_lacks(x):
            return STANDS[3]
        of = SV.FORMATS.get(x[CC.VERDICT].get(SV.FORMAT))
        return STANDS[4] if of is not None and x[SV.CONTROL] == SV.CONTROLS[of][0] and not _counts(x) else None
    if CC.lacks(x):
        return STANDS[5]
    verdict = x[CC.VERDICT]
    if verdict.get(CC.ACCEPTED) is not True or verdict.get(CC.COMPLETE) is not True:
        return STANDS[6]
    if any(k == kind and not CC.lacks(y) and y[CC.VERDICT].get(CC.HEAD) == verdict.get(CC.HEAD)
           and (y[CC.SERVE], y[CC.REGISTRATION]) != (x[CC.SERVE], x[CC.REGISTRATION]) for k, y in stays):
        return STANDS[7]
    if (of := SV.FORMATS.get(verdict.get(SV.FORMAT))) is None:
        return None
    if of in SV.RATES and x.get(SV.RATE) != SV.EVERY_BLOCK:
        return STANDS[8]
    its = [c for c in controls if not SV.control_lacks(c) and SV.FORMATS.get(c[CC.VERDICT].get(SV.FORMAT)) == of
           and c[SV.CONTROL] == SV.CONTROLS[of][0] and c[CC.VERIFIER] == x[CC.VERIFIER]]
    return None if its and all(_counts(c) for c in its) else STANDS[9]


def left_out(tmp: Path, olds: dict[str, dict[str, str]], news: dict[str, dict[str, str]]
             ) -> tuple[int, set[str], int]:
    """Leaves one thing at a time out of what the reader is handed of a run `old` beside a run `new`, over each
    pair of their files and with neither, the one or the other superseded: one record of each file, each whole
    file, and the run itself, of the runs handed. Asserts that what is held goes up only where a thing left out
    stands against one that stays (`stands_against`), and that such things are the cause: with them alone put
    back, of a file or a run left out, what is held is not raised. Returns how many cases it saw, the ways those
    that raised it stood, and how many files and runs raised it. The runs' names are symmetric, so leaving out of
    `old` covers both."""
    seen, ways, files = 0, set(), 0
    for a, old in olds.items():
        for b, new in news.items():
            runs = {"old": old, "new": new}
            controls = [x for kind, x in things(old) + things(new) if kind == "control"]
            for superseded in ([], ["old"], ["new"]):
                before = SV.of_runs(Store(tmp, runs), ["old", "new"], superseded).held
                for what, less in (("the run, of the runs handed", None), *less_by_one(old)):
                    handed = Store(tmp, runs if less is None else runs | {"old": less})
                    after = SV.of_runs(handed, ["new"] if less is None else ["old", "new"], superseded).held
                    seen += 1
                    if after <= before:
                        continue
                    stays, gone = things(less or {}) + things(new), things(old)
                    for thing in things(less or {}):
                        gone.remove(thing)
                    stood = {why for kind, x in gone if (why := stands_against(kind, x, stays, controls))}
                    assert stood, (a, b, superseded, what, before, after)
                    ways |= stood
                    # of a file or a run left out, the records that stand against what stays, alone put back
                    if whole := old.keys() - (less or {}).keys():
                        back = {path: "".join(line for line in old[path].splitlines(keepends=True)
                                              for kind, x in things({path: line})
                                              if stands_against(kind, x, stays, controls)) for path in whole}
                        only = SV.of_runs(Store(tmp, runs | {"old": (less or {}) | back}), ["old", "new"], superseded)
                        assert only.held <= before, (a, b, superseded, what, before, only.held)
                        files += 1
    return seen, ways, files


def test_leaving_a_thing_out_raises_what_is_held_only_where_it_stood_against_one_that_stays(tmp_path):
    """Whatever is left out of what the reader is handed (a verdict record, a control record, a registration of
    connection rows, a serve's registration, a whole file, a run), what is held never goes up, unless the thing
    left out is itself a refusal by the reader's rules (`STANDS`): the reader can't read what it is not handed, so
    a refusal counts only while it is. Each kind of run files is tried alone and beside the four verdicts of a
    class, and each way of standing against is met."""
    kinds = run_files()
    # the things one run's files can be left without: the run, each file and each record, over the kinds
    cases = sum(1 + len(list(less_by_one(files))) for files in kinds.values())
    assert (len(kinds), cases) == (25, 337)
    seen, ways, files = left_out(tmp_path, kinds, {name: kinds[name] for name in ("nothing", "the four")})
    assert (seen, ways) == (2 * 3 * cases, set(STANDS)) and files


@pytest.mark.slow
def test_leaving_a_thing_out_of_any_two_runs_raises_what_is_held_only_where_it_stood_against_one_that_stays(tmp_path):
    """The same over every pair of the kinds of run files, with neither, the one or the other superseded."""
    kinds = run_files()
    seen, ways, files = left_out(tmp_path, kinds, kinds)
    assert (seen, ways) == (3 * 337 * len(kinds), set(STANDS)) and files
