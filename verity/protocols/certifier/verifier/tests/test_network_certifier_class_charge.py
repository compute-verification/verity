"""How an execution phase's class is charged (`class_charge.charge`): below its schedule only with a count above 0
and all four verdicts, each accepted, complete and on the signed head; with one missing, refused, not complete or
decided under another head, its whole schedule on every row, that verdict named; and a class of no row owes none."""

from __future__ import annotations

import pytest

from verity.protocols.certifier.verifier import class_charge as K
from verity.protocols.certifier.verifier import connection_rows as C
from verity.protocols.certifier.verifier import content_rows as CT

LINK, WINDOW = "responses", 3
ZEROS = bytes(64)


def head(*counts: int, window: int = WINDOW, link: str = LINK) -> C.Head:
    """A head of the lengths and fails tables and a class's three tables for each count given."""
    tables = [C.Entry(name, 4, 8, ZEROS, ZEROS) for name in ("lengths", "fails")]
    tables += [C.Entry(name, n, 16, ZEROS, ZEROS) for i, n in enumerate(counts) for name in CT.names(i)]
    return C.Head(link, window, ZEROS, tuple(tables))


HEAD = head(5)


def record(kind: str, cls: int = 0, of: C.Head = HEAD, run: str | None = None, **over: object) -> dict:
    """A verdict record of the verdict `kind` on the class under the head `of`: accepted and complete, but for
    `over`, the verdict's fields; one given as `...` is left out."""
    verdict = {"format": f"the {kind}'s", K.LINK: of.link, K.WINDOW: of.window, K.CLASS: cls,
               K.HEAD: of.digest().hex(), K.ACCEPTED: True, K.COMPLETE: True, "codes": []} | over
    return {K.RUN: run or f"run of the {kind}", K.VERIFIER: {"module": "a.verifier", "commit": "a commit"},
            K.VERDICT: {k: v for k, v in verdict.items() if v is not ...}, K.PROVED_BY: None}


def four(cls: int = 0, of: C.Head = HEAD, **over: object) -> dict[str, list[dict]]:
    return {kind: [record(kind, cls, of, **over)] for kind in K.VERDICTS}


def but(kind: str, *instead: dict, of: C.Head = HEAD) -> dict[str, list[dict]]:
    """All four verdicts' records, with those of `kind` replaced by `instead`."""
    return four(of=of) | {kind: list(instead)}


def test_all_four_verdicts_charge_the_class_below_its_schedule():
    got = K.charge(HEAD, 0, four())
    assert (got.below, got.count, got.schedules, got.owed) == (True, 5, 0, ())
    assert [kind for kind, _ in got.read] == list(K.VERDICTS)
    assert got.why() == ("class 0: charged below its schedule, its 4 verdicts (stop record, response tie, certified "
                         "read, compute audit) each accepted and complete on this head")
    # another class's and another execution phase's records beside them change nothing, nor does the map's order
    wide, later = head(5, 2), head(5, 2, window=WINDOW + 1)
    mixed = {kind: [*four(1, wide)[kind], *four(of=later)[kind], record(kind, of=wide)] for kind in K.VERDICTS[::-1]}
    held = K.charge(wide, 0, mixed)
    assert (held.below, held.owed, len(held.read)) == (True, (), 4)
    # two records of one verdict count where both accept and are complete
    assert K.charge(HEAD, 0, but(K.RESPONSE_TIE, record(K.RESPONSE_TIE), record(K.RESPONSE_TIE, run="a second"))).below


def test_a_class_with_a_count_and_no_tie_is_charged_its_whole_schedule():
    for records in (but(K.RESPONSE_TIE), {k: v for k, v in four().items() if k != K.RESPONSE_TIE}):
        got = K.charge(HEAD, 0, records)
        assert (got.below, got.schedules, got.owed) == (False, 5, ((K.RESPONSE_TIE, K.MISSING, ()),))
        assert len(got.read) == 3
    assert got.why() == "class 0: charged its whole schedule on each of its 5 rows: the response tie is missing"
    # with no record at all, each of the four is named, and three accepted verdicts are no part of a lower charge
    none = K.charge(HEAD, 0, {})
    assert (none.below, none.schedules) == (False, 5)
    assert none.owed == tuple((kind, K.MISSING, ()) for kind in K.VERDICTS)
    # a tie on another class, in another execution phase or on another link is not this class's tie
    wide = head(5, 2)
    for other in (record(K.RESPONSE_TIE, 1, wide), record(K.RESPONSE_TIE, of=wide, window=WINDOW + 1),
                  record(K.RESPONSE_TIE, of=wide, link="requests"), record(K.RESPONSE_TIE, of=wide, window=float(WINDOW)),
                  record(K.RESPONSE_TIE, of=wide, **{K.CLASS: False}), record(K.RESPONSE_TIE, of=wide, link=...)):
        assert K.charge(wide, 0, but(K.RESPONSE_TIE, other, of=wide)).owed == got.owed


def test_a_tie_accepted_and_not_complete_is_charged_its_whole_schedule():
    for complete in (False, ..., None, 1, "true"):
        tie = record(K.RESPONSE_TIE, complete=complete)
        got = K.charge(HEAD, 0, but(K.RESPONSE_TIE, tie))
        assert (got.below, got.schedules) == (False, 5), complete
        assert got.owed == ((K.RESPONSE_TIE, K.NOT_COMPLETE, ("run of the response tie",)),), complete
        assert (K.RESPONSE_TIE, tie) in got.read
    assert got.why() == ("class 0: charged its whole schedule on each of its 5 rows: the response tie is not "
                         "complete (run of the response tie)")
    # a refused verdict is named as refused whether it is complete or not, and an accept beside it doesn't undo it
    for accepted in (False, ..., None, 1, "accept"):
        refused = record(K.STOP_RECORD, accepted=accepted, run="the first")
        got = K.charge(HEAD, 0, but(K.STOP_RECORD, refused, record(K.STOP_RECORD)))
        assert (got.below, got.owed) == (False, ((K.STOP_RECORD, K.REFUSED, ("the first",)),)), accepted
    # every verdict that doesn't count is named, in the four's order
    got = K.charge(HEAD, 0, {K.COMPUTE_AUDIT: [record(K.COMPUTE_AUDIT, complete=False)],
                             K.STOP_RECORD: [record(K.STOP_RECORD, accepted=False)]})
    assert [(kind, lack) for kind, lack, _ in got.owed] == [
        (K.STOP_RECORD, K.REFUSED), (K.RESPONSE_TIE, K.MISSING), (K.CERTIFIED_READ, K.MISSING),
        (K.COMPUTE_AUDIT, K.NOT_COMPLETE)]


def test_a_verdict_from_a_run_on_another_serve_is_charged_its_whole_schedule():
    """A verdict is held to the serve by the signed head it names: another serve's head of the same link, execution
    phase and class, or no head, is another digest."""
    theirs = C.Head(HEAD.link, HEAD.window, bytes([1]) * 64, HEAD.tables)
    for named in (theirs.digest().hex(), ..., None, HEAD.digest(), HEAD.digest().hex().upper()):
        tie = record(K.RESPONSE_TIE, head=named, run="its run")
        got = K.charge(HEAD, 0, but(K.RESPONSE_TIE, tie))
        assert (got.below, got.schedules) == (False, 5), named
        assert got.owed == ((K.RESPONSE_TIE, K.ANOTHER_HEAD, ("its run",)),), named
        assert (K.RESPONSE_TIE, tie) not in got.read and len(got.read) == 3
    assert got.why() == ("class 0: charged its whole schedule on each of its 5 rows: the response tie is not on this "
                         "head (its run)")
    # four verdicts that are each accepted and complete, under a head this one is not, charge nothing less
    assert K.charge(theirs, 0, four()).owed == tuple(
        (kind, K.ANOTHER_HEAD, (f"run of the {kind}",)) for kind in K.VERDICTS)
    # and beside this head's own verdict, another head's is not read
    other = record(K.RESPONSE_TIE, head=theirs.digest().hex(), accepted=False)
    assert K.charge(HEAD, 0, but(K.RESPONSE_TIE, record(K.RESPONSE_TIE), other)).below


def test_a_class_with_a_count_of_0_owes_no_verdict():
    idle = head(0, 5)
    got = K.charge(idle, 0, {})
    assert (got.below, got.count, got.schedules, got.owed, got.read) == (False, 0, 0, (), ())
    assert got.why() == "class 0: a count of 0, so no connection is charged and no verdict is owed"
    assert K.charge(idle, 0, four(of=idle, accepted=False)) == got
    assert K.charge(idle, 1, {}).schedules == 5


def test_a_head_without_the_class_and_records_that_do_not_read_are_refused():
    with pytest.raises(ValueError, match="not class 1's request.1, response.1, end.1"):
        K.charge(HEAD, 1, four(1))
    for cls in (-1, True, "0", None):
        with pytest.raises(ValueError, match="a class is its index in the schedule"):
            K.charge(HEAD, cls, {})
    tables = tuple(C.Entry(e.name, 6, e.row_bytes, e.domain, e.root) if e.name == "end.0" else e for e in HEAD.tables)
    with pytest.raises(ValueError, match=r"states \[5, 6\] rows for class 0's three tables"):
        K.charge(C.Head(LINK, WINDOW, ZEROS, tables), 0, four())
    for records in ([record(K.RESPONSE_TIE)], four() | {"a fifth verdict": []}):
        with pytest.raises(ValueError, match="the records are given by verdict, each one of stop record, response tie"):
            K.charge(HEAD, 0, records)
    for off in ("accept", {K.RUN: "a run"}, record(K.RESPONSE_TIE) | {K.VERDICT: "accept"}):
        with pytest.raises(ValueError, match="a verdict record is an object with the fields run, verifier, verdict, "
                                             "proved_by, its verdict an object with link, window, class, head"):
            K.charge(HEAD, 0, but(K.RESPONSE_TIE, off))
