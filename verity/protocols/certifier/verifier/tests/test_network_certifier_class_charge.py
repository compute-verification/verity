"""How an execution phase's class is charged (`class_charge.charge`): below its schedule only with a count above 0
and all four verdicts, each accepted, complete and on the head's serve; with one missing, refused, not complete or
of another serve, its whole schedule on every row, that verdict named; and a class of no row owes none."""

from __future__ import annotations

import pytest

from verity.protocols.certifier.verifier import class_charge as K
from verity.protocols.certifier.verifier import connection_rows as C
from verity.protocols.certifier.verifier import content_rows as CT

LINK, WINDOW, SERVE = "responses", 3, "the serve"
ZEROS = bytes(64)


def head(*counts: int, window: int = WINDOW) -> C.Head:
    """A head of the lengths and fails tables and a class's three tables for each count given."""
    tables = [C.Entry(name, 4, 8, ZEROS, ZEROS) for name in ("lengths", "fails")]
    tables += [C.Entry(name, n, 16, ZEROS, ZEROS) for i, n in enumerate(counts) for name in CT.names(i)]
    return C.Head(LINK, window, ZEROS, tuple(tables))


def record(kind: str, cls: int = 0, **over: object) -> dict:
    """A verdict record of `kind` on the class: accepted, complete and on the head's serve, but for `over`; a
    field given as `...` is left out."""
    doc = {K.KIND: kind, K.SERVE: SERVE, K.LINK: LINK, K.WINDOW: WINDOW, K.CLASS: cls, K.RUN: f"run of the {kind}",
           K.VERIFIER: "a verifier", K.VERDICT: K.ACCEPT, K.COMPLETE: True, K.PROVED_BY: None} | over
    return {k: v for k, v in doc.items() if v is not ...}


def four(cls: int = 0, **over: object) -> list[dict]:
    return [record(kind, cls, **over) for kind in K.VERDICTS]


def but(kind: str, *instead: dict) -> list[dict]:
    """All four verdicts' records, with the one of `kind` replaced by `instead`."""
    return [r for r in four() if r[K.KIND] != kind] + list(instead)


def test_all_four_verdicts_charge_the_class_below_its_schedule():
    got = K.charge(head(5), 0, SERVE, four())
    assert (got.below, got.count, got.schedules, got.owed) == (True, 5, 0, ())
    assert [r[K.KIND] for r in got.read] == list(K.VERDICTS)
    assert got.why() == ("class 0: charged below its schedule, its 4 verdicts (stop record, response tie, certified "
                         "read, compute audit) each accepted and complete on the head's serve")
    # the records' order is not read, and another class's, execution phase's or link's records change nothing
    others = four(1) + four(window=WINDOW + 1) + four(link="requests") + [{K.KIND: "another kind"}]
    assert K.charge(head(5, 2), 0, SERVE, others + four()[::-1]) == got
    # two records of one verdict count where both accept and are complete
    assert K.charge(head(5), 0, SERVE, four() + [record(K.RESPONSE_TIE, run="a second run")]).below


def test_a_class_with_a_count_and_no_tie_is_charged_its_whole_schedule():
    got = K.charge(head(5), 0, SERVE, but(K.RESPONSE_TIE))
    assert (got.below, got.schedules, got.owed) == (False, 5, ((K.RESPONSE_TIE, K.MISSING, ()),))
    assert len(got.read) == 3
    assert got.why() == "class 0: charged its whole schedule on each of its 5 rows: the response tie is missing"
    # with no record at all, each of the four is named, and three accepted verdicts are no part of a lower charge
    none = K.charge(head(5), 0, SERVE, [])
    assert (none.below, none.schedules) == (False, 5)
    assert none.owed == tuple((kind, K.MISSING, ()) for kind in K.VERDICTS)
    # a tie on another class, in another execution phase or on another link is not this class's tie
    for other in (record(K.RESPONSE_TIE, 1), record(K.RESPONSE_TIE, window=WINDOW + 1),
                  record(K.RESPONSE_TIE, link="requests"), record(K.RESPONSE_TIE, window=float(WINDOW)),
                  record(K.RESPONSE_TIE, **{K.CLASS: False})):
        assert K.charge(head(5, 2), 0, SERVE, but(K.RESPONSE_TIE, other)).owed == got.owed


def test_a_tie_accepted_and_not_complete_is_charged_its_whole_schedule():
    for complete in (False, ..., None, 1, "true"):
        tie = record(K.RESPONSE_TIE, complete=complete)
        got = K.charge(head(5), 0, SERVE, but(K.RESPONSE_TIE, tie))
        assert (got.below, got.schedules) == (False, 5), complete
        assert got.owed == ((K.RESPONSE_TIE, K.NOT_COMPLETE, ("run of the response tie",)),), complete
        assert tie in got.read
    assert got.why() == ("class 0: charged its whole schedule on each of its 5 rows: the response tie is not "
                         "complete (run of the response tie)")
    # a refused verdict is named as refused whether it is complete or not, and a later accept doesn't undo it
    for verdict in ("reject", ..., True, None):
        refused = record(K.STOP_RECORD, verdict=verdict, run="the first")
        got = K.charge(head(5), 0, SERVE, four() + [refused])
        assert (got.below, got.owed) == (False, ((K.STOP_RECORD, K.REFUSED, ("the first",)),)), verdict
    # every verdict that doesn't count is named, in the four's order
    got = K.charge(head(5), 0, SERVE, [record(K.COMPUTE_AUDIT, complete=False), record(K.STOP_RECORD, verdict="reject")])
    assert [(kind, lack) for kind, lack, _ in got.owed] == [
        (K.STOP_RECORD, K.REFUSED), (K.RESPONSE_TIE, K.MISSING), (K.CERTIFIED_READ, K.MISSING),
        (K.COMPUTE_AUDIT, K.NOT_COMPLETE)]


def test_a_verdict_from_a_run_on_another_serve_is_charged_its_whole_schedule():
    for serve in ("another serve", ..., None, 7):
        tie = record(K.RESPONSE_TIE, serve=serve, run="its run")
        got = K.charge(head(5), 0, SERVE, but(K.RESPONSE_TIE, tie))
        assert (got.below, got.schedules) == (False, 5), serve
        assert got.owed == ((K.RESPONSE_TIE, K.ANOTHER_SERVE, ("its run",)),), serve
        assert tie not in got.read and len(got.read) == 3
    assert got.why() == ("class 0: charged its whole schedule on each of its 5 rows: the response tie is of another "
                         "serve (its run)")
    # four verdicts that are each accepted and complete, on a serve the head is not of, charge nothing less
    assert K.charge(head(5), 0, "another serve", four()).owed == tuple(
        (kind, K.ANOTHER_SERVE, (f"run of the {kind}",)) for kind in K.VERDICTS)
    # and beside this serve's own verdict, another serve's is not read
    assert K.charge(head(5), 0, SERVE, four() + [record(K.RESPONSE_TIE, serve="another serve", verdict="reject")]).below


def test_a_class_with_a_count_of_0_owes_no_verdict():
    got = K.charge(head(0, 5), 0, SERVE, [])
    assert (got.below, got.count, got.schedules, got.owed, got.read) == (False, 0, 0, (), ())
    assert got.why() == "class 0: a count of 0, so no connection is charged and no verdict is owed"
    assert K.charge(head(0, 5), 0, SERVE, four(verdict="reject")) == got
    assert K.charge(head(0, 5), 1, SERVE, []).schedules == 5


def test_a_head_without_the_class_and_a_record_that_is_no_object_are_refused():
    with pytest.raises(ValueError, match="not class 1's request.1, response.1, end.1"):
        K.charge(head(5), 1, SERVE, four(1))
    for cls in (-1, True, "0", None):
        with pytest.raises(ValueError, match="a class is its index in the schedule"):
            K.charge(head(5), cls, SERVE, [])
    uneven = head(5)
    tables = tuple(C.Entry(e.name, 6, e.row_bytes, e.domain, e.root) if e.name == "end.0" else e for e in uneven.tables)
    with pytest.raises(ValueError, match=r"states \[5, 6\] rows for class 0's three tables"):
        K.charge(C.Head(LINK, WINDOW, ZEROS, tables), 0, SERVE, four())
    for serve in ("", None, 3):
        with pytest.raises(ValueError, match="the head's serve is its name"):
            K.charge(head(5), 0, serve, four())
    with pytest.raises(ValueError, match="a verdict record is an object with the fields kind, serve, link, window"):
        K.charge(head(5), 0, SERVE, [*four(), "accept"])
