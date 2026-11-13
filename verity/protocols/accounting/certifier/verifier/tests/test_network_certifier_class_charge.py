"""How an execution phase's class is charged (`class_charge.charge`): below its schedule only with a count above 0
and all four verdicts, each accepted, complete and on the signed head; with one missing, refused, not complete or
decided under another head, its whole schedule on every row, that verdict named; and a class of no row owes none."""

from __future__ import annotations

import pytest

from verity.protocols.accounting.certifier.verifier import class_charge as K
from verity.protocols.accounting.certifier.verifier import connection_rows as C
from verity.protocols.accounting.certifier.verifier import content_rows as CT

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
               K.HEAD: of.digest().hex(), K.ACCEPTED: True, K.COMPLETE: True, "withheld": None, "codes": []} | over
    return {K.RUN: run or f"run of the {kind}", K.SERVE: "the serve", K.REGISTRATION: "its registration's digest",
            K.VERIFIER: {"module": "a.verifier", "commit": "a commit"},
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


def test_a_verdict_withheld_is_charged_as_one_refused_and_named_withheld_for_its_word():
    """A verdict states what its verifier held its accept back for (`withheld`, null or a word) and the
    verifier's own code words (`codes`, a list). Neither moves a charge: a verdict that is not accepted leaves its
    class at its whole schedule whatever the two say, and the line says "withheld" with the word where it states
    one and "refused" where `withheld` is null. A verdict that leaves either out, a `withheld` that is neither
    null nor a word, `codes` that are no list of words, and an accept that states a word or a code are each a
    record that doesn't read."""
    assert K.SAID == (K.WITHHELD, K.CODES) == ("withheld", "codes")
    refused = K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, record(K.COMPUTE_AUDIT, accepted=False)))
    assert refused.owed == ((K.COMPUTE_AUDIT, K.REFUSED, ("run of the compute audit",)),)
    for word in ("two-stage-form", "another word"):
        for accepted in (False, ..., None, 1, "accept"):
            audit = record(K.COMPUTE_AUDIT, accepted=accepted, withheld=word, codes=[word])
            got = K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, audit))
            assert (got.below, got.count, got.schedules, got.differ) == (False, 5, 5, ()), (word, accepted)
            assert got.owed == ((K.COMPUTE_AUDIT, f"withheld for `{word}`", ("run of the compute audit",)),)
            assert (K.COMPUTE_AUDIT, audit) in got.read and K.not_accepted(audit[K.VERDICT]) == f"withheld for `{word}`"
    assert got.why() == ("class 0: charged its whole schedule on each of its 5 rows: the compute audit is withheld "
                         "for `another word` (run of the compute audit)")
    # null, with codes or without, it is refused, as it was
    for over in ({K.WITHHELD: None}, {K.CODES: []}, {K.WITHHELD: None, K.CODES: ["roots", "registered"]}):
        audit = record(K.COMPUTE_AUDIT, accepted=False, **over)
        assert K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, audit)).owed == refused.owed, over
        assert K.not_accepted(audit[K.VERDICT]) == K.REFUSED
    # a verdict some records refused and others withheld is named once for each, refused first and the words in
    # order, and an accept beside them undoes none
    several = [record(K.COMPUTE_AUDIT, accepted=False, withheld="two-stage-form", run="a third"),
               record(K.COMPUTE_AUDIT), record(K.COMPUTE_AUDIT, accepted=False, withheld="another word", run="a second"),
               record(K.COMPUTE_AUDIT, accepted=None, run="a first"),
               record(K.COMPUTE_AUDIT, accepted=False, withheld="two-stage-form", run="a fourth")]
    got = K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, *several))
    assert (got.below, got.schedules) == (False, 5) and got.owed == (
        (K.COMPUTE_AUDIT, K.REFUSED, ("a first",)), (K.COMPUTE_AUDIT, "withheld for `another word`", ("a second",)),
        (K.COMPUTE_AUDIT, "withheld for `two-stage-form`", ("a fourth", "a third")))
    # all four accepted charge the class below where each states a null `withheld` and no code
    assert K.charge(HEAD, 0, four(**{K.WITHHELD: None, K.CODES: []})).below
    # what doesn't read: a verdict that leaves one of the two out, accepted or not, complete or not
    tie = record(K.RESPONSE_TIE)
    add = ": a verdict with nothing to say states `withheld` null and `codes` empty"
    off = [(accepted, over | more, f"its verdict states no {said}{add}")
           for accepted in (True, False, ...) for more in ({}, {K.COMPLETE: False}, {"format": "another's"})
           for over, said in (({K.WITHHELD: ...}, "`withheld`"), ({K.CODES: ...}, "`codes`"),
                              ({K.WITHHELD: ..., K.CODES: ...}, "`withheld` and no `codes`"),
                              ({K.WITHHELD: ..., K.CODES: 0}, "`withheld`"),
                              ({K.WITHHELD: 0, K.CODES: ...}, "`codes`"))]
    # a `withheld` that is no word, `codes` that are no list of words, on any verdict
    off += [(accepted, {K.WITHHELD: value}, "its verdict's `withheld` is neither a word nor null")
           for accepted in (True, False) for value in ("", 0, False, True, ["two-stage-form"], {})]
    off += [(accepted, {K.CODES: value}, "its verdict's `codes` is no list of words")
            for accepted in (True, False) for value in (None, "roots", {}, [1], [None], ["roots", 2], ("roots",))]
    # and an accept, complete or not, that states a word or a code
    two = "an accept that also says why it is no accept says two things"
    off += [(True, over | more, f"its verdict is accepted and states {said}: {two}")
            for more in ({}, {K.COMPLETE: False})
            for over, said in (({K.WITHHELD: "two-stage-form"}, "`withheld`"), ({K.CODES: ["roots"]}, "`codes`"),
                               ({K.WITHHELD: "two-stage-form", K.CODES: ["roots"]}, "`withheld` and `codes`"))]
    for accepted, over, lack in off:
        bad = record(K.RESPONSE_TIE, accepted=accepted, **over)
        assert K.lacks(bad) == lack and not K.reads(bad), (accepted, over)
        for records in (but(K.RESPONSE_TIE, bad), but(K.RESPONSE_TIE, tie, bad)):
            with pytest.raises(ValueError) as e:
                K.charge(HEAD, 0, records)
            assert str(e.value) == f"{K.A_RECORD}; of the response tie's, {lack}: {bad!r}"
    # a record that names no run is refused for that before its verdict is read
    assert K.lacks(record(K.RESPONSE_TIE, withheld=0) | {K.RUN: ""}) == "its `run` is no name"


def test_a_verdict_of_a_command_that_stopped_is_charged_as_one_refused_and_named_stopped_with_its_word():
    """A verdict may state the word of the error its verifier's command stopped with (`stopped`, null or a word,
    absent as null). It moves no charge: such a verdict is not accepted and leaves its class at its whole schedule,
    and the line says "stopped" with the word. It states `withheld` null and `codes` empty, as every verdict with
    nothing to say there does. A `stopped` that is neither null nor a word, an accept that states one, and a
    verdict that states a `withheld` word or a code too, or leaves either field out, are each a record that
    doesn't read."""
    assert K.STOPPED == "stopped" and K.STOPPED not in K.SAID
    refused = K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, record(K.COMPUTE_AUDIT, accepted=False)))
    for word in ("session", "another word"):
        for accepted in (False, ..., None, 1):
            for more in ({}, {K.COMPLETE: False}, {K.WITHHELD: None, K.CODES: []}):
                audit = record(K.COMPUTE_AUDIT, accepted=accepted, stopped=word, **more)
                got = K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, audit))
                assert (got.below, got.count, got.schedules, got.differ) == (False, 5, 5, ()), (word, accepted)
                assert got.owed == ((K.COMPUTE_AUDIT, f"stopped with `{word}`", ("run of the compute audit",)),)
                assert (K.COMPUTE_AUDIT, audit) in got.read
                assert K.not_accepted(audit[K.VERDICT]) == f"stopped with `{word}`"
    assert got.why() == ("class 0: charged its whole schedule on each of its 5 rows: the compute audit is stopped "
                         "with `another word` (run of the compute audit)")
    # null or absent it is refused, as it was, and an accept that states null is an accept
    for over in ({K.STOPPED: None}, {}):
        audit = record(K.COMPUTE_AUDIT, accepted=False, **over)
        assert K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, audit)).owed == refused.owed, over
        assert K.not_accepted(audit[K.VERDICT]) == K.REFUSED
    assert K.charge(HEAD, 0, four(stopped=None)).below
    # one verdict refused, withheld and stopped by three runs is named once for each: refused, stopped, withheld
    several = [record(K.COMPUTE_AUDIT, accepted=False, withheld="two-stage-form", run="a third"),
               record(K.COMPUTE_AUDIT, accepted=False, stopped="session", run="a second"),
               record(K.COMPUTE_AUDIT, accepted=False, run="a first"),
               record(K.COMPUTE_AUDIT, accepted=False, stopped="another word", run="a fourth")]
    assert K.charge(HEAD, 0, but(K.COMPUTE_AUDIT, *several)).owed == (
        (K.COMPUTE_AUDIT, K.REFUSED, ("a first",)), (K.COMPUTE_AUDIT, "stopped with `another word`", ("a fourth",)),
        (K.COMPUTE_AUDIT, "stopped with `session`", ("a second",)),
        (K.COMPUTE_AUDIT, "withheld for `two-stage-form`", ("a third",)))
    # what doesn't read
    off = [(accepted, {K.STOPPED: value}, "its verdict's `stopped` is neither a word nor null")
           for accepted in (True, False) for value in ("", 0, False, True, ["session"], {})]
    off += [(True, over | more, f"its verdict is accepted and states {said}: an accept that also says why it is no "
                                "accept says two things")
            for more in ({}, {K.COMPLETE: False})
            for over, said in (({K.STOPPED: "session"}, "`stopped`"),
                               ({K.STOPPED: "session", K.CODES: ["a code"]}, "`codes` and `stopped`"),
                               ({K.STOPPED: "session", K.WITHHELD: "two-stage-form"}, "`withheld` and `stopped`"))]
    # a command that stopped reached no verdict: it withheld none and gives no code
    none = "a command that stopped reached no verdict, so it states `withheld` null and `codes` empty"
    off += [(accepted, {K.STOPPED: "session"} | over, f"its verdict states `stopped` and {said}: {none}")
            for accepted in (False, None, ...)
            for over, said in (({K.WITHHELD: "two-stage-form"}, "`withheld`"), ({K.CODES: ["a code"]}, "`codes`"),
                               ({K.WITHHELD: "two-stage-form", K.CODES: ["a code", "another"]},
                                "`withheld` and `codes`"))]
    # and it states both fields, as every verdict does
    off += [(accepted, {K.STOPPED: "session"} | over, f"its verdict states no {said}: a verdict with nothing to say "
                                                       "states `withheld` null and `codes` empty")
            for accepted in (False, None, ...)
            for over, said in (({K.WITHHELD: ...}, "`withheld`"), ({K.CODES: ...}, "`codes`"),
                               ({K.WITHHELD: ..., K.CODES: ...}, "`withheld` and no `codes`"))]
    for accepted, over, lack in off:
        bad = record(K.RESPONSE_TIE, accepted=accepted, **over)
        assert K.lacks(bad) == lack and not K.reads(bad), (accepted, over)
        with pytest.raises(ValueError) as e:
            K.charge(HEAD, 0, but(K.RESPONSE_TIE, record(K.RESPONSE_TIE), bad))
        assert str(e.value) == f"{K.A_RECORD}; of the response tie's, {lack}: {bad!r}"


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
    assert K.A_RECORD == (
        "a verdict record is an object with the fields run, serve, registration, verifier, verdict, proved_by, its "
        "verdict an object, its run, serve, registration named, its verifier by its module and commit, and its "
        "proved_by a name or null")
    for off, lack in (("accept", "it is no object"),
                      ({K.RUN: "a run"}, "it has no `serve`, `registration`, `verifier`, `verdict`, `proved_by`"),
                      (record(K.RESPONSE_TIE) | {K.VERDICT: "accept"}, "its `verdict` is no object")):
        assert K.lacks(off) == lack
        with pytest.raises(ValueError) as e:
            K.charge(HEAD, 0, but(K.RESPONSE_TIE, off))
        assert str(e.value) == f"{K.A_RECORD}; of the response tie's, {lack}: {off!r}"


def test_a_record_that_names_no_run_serve_registration_or_verifier_is_refused_by_the_fields_name():
    """Each record below holds an accepted, complete verdict on the head: what it lacks is who produced or judged
    it, the serve its class came from, what that serve registered, or the field that says what proves the verifier.
    Each is refused, and the refusal names the field."""
    tie = record(K.RESPONSE_TIE)
    assert K.FIELDS == ("run", "serve", "registration", "verifier", "verdict", "proved_by")
    assert K.lacks(tie) is None and K.reads(tie) and K.reads(tie | {K.PROVED_BY: "A.Guarantee"})
    assert K.charge(HEAD, 0, but(K.RESPONSE_TIE, tie)).below
    verifier = tie[K.VERIFIER]
    lacking = [({k: v for k, v in tie.items() if k != field}, f"it has no `{field}`") for field in K.FIELDS]
    lacking += [(tie | {field: value}, f"its `{field}` is no name")
                for field in (K.RUN, K.SERVE, K.REGISTRATION) for value in (None, "", 7, ["a name"])]
    lacking += [(tie | {K.VERIFIER: value}, f"its `verifier` names no `{names}`") for value, names in (
        (None, "module`, `commit"), ("a.verifier", "module`, `commit"), ({}, "module`, `commit"),
        ({K.MODULE: verifier[K.MODULE]}, "commit"), ({K.COMMIT: verifier[K.COMMIT]}, "module"),
        (verifier | {K.MODULE: ""}, "module"), (verifier | {K.COMMIT: None}, "commit"))]
    lacking += [(tie | {K.PROVED_BY: value}, "its `proved_by` is neither a name nor null")
                for value in ("", False, ["A.Guarantee"])]
    for off, lack in lacking:
        assert K.lacks(off) == lack and not K.reads(off), off
        with pytest.raises(ValueError) as e:
            K.charge(HEAD, 0, but(K.RESPONSE_TIE, off))
        assert str(e.value) == f"{K.A_RECORD}; of the response tie's, {lack}: {off!r}"
        # beside a record that reads, and on a class that owes nothing, it is refused the same
        for at, cls, records in ((HEAD, 0, but(K.RESPONSE_TIE, tie, off)), (head(0, 5), 0, {K.RESPONSE_TIE: [off]})):
            with pytest.raises(ValueError, match="a verdict record is an object with the fields"):
                K.charge(at, cls, records)


def test_four_verdicts_of_two_serves_or_under_two_registrations_are_charged_the_whole_schedule():
    """The four are one class's only where their records name one serve and one registration. The head holds the
    network certifier's rows to a serve and says nothing of the values the stop record and the compute audit
    judge, so four accepts on one head of one class, one of them of another serve's values or under another
    registration, charge nothing less, and the line says which name differs."""
    assert K.ONE == (K.SERVE, K.REGISTRATION) == ("serve", "registration")
    assert K.charge(HEAD, 0, four()).differ == ()
    for key, other in ((K.SERVE, "another serve"), (K.REGISTRATION, "another's digest")):
        ours = record(K.COMPUTE_AUDIT)[key]
        for kind in K.VERDICTS:
            got = K.charge(HEAD, 0, but(kind, record(kind) | {key: other}))
            assert (got.below, got.schedules, got.owed, len(got.read)) == (False, 5, (), 4), (key, kind)
            assert got.differ == ((key, tuple(sorted((ours, other)))),)
        assert got.why() == (f"class 0: charged its whole schedule on each of its 5 rows: its verdicts on this head "
                             f"name 2 `{key}`s, not one ({', '.join(sorted((ours, other)))})")
        # all four of the other serve, or under the other registration, are one class's
        assert K.charge(HEAD, 0, {kind: [record(kind) | {key: other}] for kind in K.VERDICTS}).below
        # a second record of a verdict that names another holds the class too, whether or not it accepts
        for accepted in (True, False):
            second = record(K.RESPONSE_TIE, accepted=accepted, run="a second") | {key: other}
            got = K.charge(HEAD, 0, but(K.RESPONSE_TIE, record(K.RESPONSE_TIE), second))
            assert not got.below and got.differ == ((key, tuple(sorted((ours, other)))),)
        # another class's, another execution phase's and another head's records name what they will
        wide = head(5, 2)
        beside = {kind: [record(kind, of=wide), record(kind, 1, wide) | {key: other},
                         record(kind, of=head(5, 2, window=WINDOW + 1)) | {key: other},
                         record(kind, of=wide, head="another head's digest") | {key: other}] for kind in K.VERDICTS}
        assert K.charge(wide, 0, beside).below and K.charge(wide, 1, beside).below
    # both differ, and a verdict lacking beside them: each is named
    both = but(K.STOP_RECORD, record(K.STOP_RECORD) | {K.SERVE: "another serve", K.REGISTRATION: "another's digest"})
    got = K.charge(HEAD, 0, {k: v for k, v in both.items() if k != K.RESPONSE_TIE})
    assert [key for key, _ in got.differ] == list(K.ONE) and got.owed == ((K.RESPONSE_TIE, K.MISSING, ()),)
    assert got.why() == (
        "class 0: charged its whole schedule on each of its 5 rows: the response tie is missing; its verdicts on "
        "this head name 2 `serve`s, not one (another serve, the serve); its verdicts on this head name 2 "
        "`registration`s, not one (another's digest, its registration's digest)")
    # a class of no row owes nothing, whatever its records name
    idle = head(0, 5)
    assert K.charge(idle, 0, {K.STOP_RECORD: [record(K.STOP_RECORD, of=idle),
                                              record(K.STOP_RECORD, of=idle) | {K.SERVE: "another serve"}]}).differ == ()
