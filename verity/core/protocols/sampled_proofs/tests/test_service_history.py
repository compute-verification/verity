"""A stream's registrations as its history grows: each one's check reads the same few records of the log whatever the
stream's length, no round reads the log or the stream's entries whole, and the verdicts, the log and the records are the
ones the whole log gives (a stream that hands its whole log to every check, `_Whole`)."""

from __future__ import annotations

from typing import Any

from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R, verifier as V

from test_registered import (BINDING, BOUND, LAW, MIXED, Q, SCHEME, SERVED, _mixed_case, _program, _registered,
                             _record as _registered_record, _seeded_case)
from test_service import DELTA, PAIR, RUN, SESSION, Clock, _lean, _ports, _record, _spec


class _Whole(S.Stream):
    """A stream whose every check reads its whole log, as a stream's did before it kept what R6 and R7 read."""

    def _earlier(self, record, supplied=None):
        return self._log


class _Counted(list):
    """A list that counts the times it is read whole."""

    def __init__(self) -> None:
        super().__init__()
        self.walks = 0

    def __iter__(self):
        self.walks += 1
        return super().__iter__()


def _stream(cls=S.Stream, spec=None, budget=10 ** 6, clock=None) -> S.Stream:
    return cls(spec or _spec(proof_deadline=60.0), PAIR, budget=budget, delta=DELTA, challenger="challenger-test",
               coins=lambda n: b"\x07" * n, clock=clock or Clock())


def _refusal(e: S.Refused) -> tuple:
    return "refused", e.verdict.ok, e.verdict.codes, repr(e.verdict.detail), repr(e.verdict.verifiers)


def test_a_registration_reads_the_same_records_whatever_the_history(monkeypatch):
    """Over 30 rounds, accepted and rejected, every registration's check gets one record of the log (the last, which R6's
    `previous` names), and no round iterates the log or the stream's entries, the failure count included; the stream's
    rules decide each op on the one entry it names, or none. The whole-log stream, counted the same way, hands its checks
    a log as long as its history and walks it at every registration."""
    check, given, rules, held = V.check, [], V.stream, []

    def spy(*a: Any, **kw: Any) -> V.Verdict:
        given.append(len(a[8]))
        return check(*a, **kw)

    def rules_spy(params: Any, state: Any, *a: Any) -> Any:
        held.append(len(state["entries"]))
        return rules(params, state, *a)

    monkeypatch.setattr(S.V, "check", spy)
    monkeypatch.setattr(S.V, "stream", rules_spy)
    for cls, rounds in ((S.Stream, 30), (_Whole, 6)):
        given.clear()
        stream = _stream(cls)
        stream._log, stream._entries, stream._rule_entries = _Counted(), _Counted(), _Counted()
        prev = None
        for i in range(rounds):
            reg = stream.register(_record(row=i, previous=prev), _ports(6))
            draw = stream.select(reg)
            assert stream.deliver(draw, [SESSION]) is None
            stream.outcome(draw, _lean(accepted=i % 3 != 1), {"statement": "stub"})
            prev = reg.digest
        assert len(stream._log) == rounds and stream.failures == rounds // 3 and not stream.stopped
        if cls is S.Stream:
            assert given == [0] + [1] * (rounds - 1)
            assert stream._log.walks == stream._entries.walks == stream._rule_entries.walks == 0
            assert max(held) == 1 and len(held) > 4 * rounds
        else:
            assert given == list(range(rounds)) and stream._log.walks >= rounds


def _one_stage(cls) -> tuple[list, tuple, tuple, tuple, S.Stream]:
    """A fixed stream of 40 rounds: accepted, rejected and late outcomes, and registrations refused for a window logged
    before, a wrong previous, and windows that one verifier finds equal to an earlier one and the other doesn't, so they
    disagree: a boolean against its integer (Python's `==` finds them equal, the Lean's `jeq` doesn't), and a list
    against a tuple, which the Lean reads as a list. Windows that aren't plain JSON are logged: those tuples, and in the
    second to last round a float, after which the Lean refuses every registration."""
    rounds, clock = 40, Clock()
    stream = _stream(cls, clock=clock)
    calls, prev = [], None
    for i in range(rounds):
        clock.now = 100.0 * i
        window = {"kind": "served", "run": RUN, "row": i}
        rec = _record(row=i, previous=prev)
        if i % 9 == 4:
            rec = _record(row=i - 3, previous=prev)
        elif i % 11 == 6:
            rec = _record(row=i, previous="ab" * 64)
        elif i % 13 == 7:
            rec = {**rec, "window": {**window, "flag": True}}
        elif i % 13 == 8:
            rec = {**rec, "window": {**window, "row": i - 1, "flag": 1}}
        elif i % 10 == 3:
            rec = {**rec, "window": {**window, "tags": ("t", i)}}
        elif i % 10 == 5:
            rec = {**rec, "window": {**window, "row": i - 2, "tags": ["t", i - 2]}}
        elif i == rounds - 2:
            rec = {**rec, "window": {**window, "row": float(i)}}
        try:
            reg = stream.register(rec, _ports(6))
        except S.Refused as e:
            calls.append((*_refusal(e), i))
            continue
        draw = stream.select(reg)
        calls += [("registered", reg.order, reg.digest, reg.receipt), ("draw", draw.draw)]
        if i % 7 == 3:
            clock.now += 61.0
            out = stream.outcome(draw, _lean(), {"statement": "stub"})
        else:
            assert stream.deliver(draw, [SESSION]) is None
            out = stream.outcome(draw, _lean(accepted=i % 5 != 2), {"statement": "stub"})
        calls.append(("outcome", out.accepted, out.verdict.codes, out.record, stream.failures, stream.stopped))
        prev = reg.digest
    return calls, stream.log, stream.records, stream.symbols, stream


class _Forgetful(S.Stream):
    """A stream that leaves the records whose window has no key out of its checks."""

    def _earlier(self, record, supplied=None):
        irregular, self._irregular = self._irregular, []
        try:
            return super()._earlier(record, supplied)
        finally:
            self._irregular = irregular


def test_the_verdicts_log_and_records_are_the_whole_logs():
    """On one fixed stream, every call's result or refusal, the log, the records and the symbols are the whole-log
    stream's; the stream ran into each refusal, logged windows that aren't plain JSON, and counted its failures. A stream
    that leaves those windows' records out of its checks gives other verdicts: the Lean's R6 reads them (round 25's
    window is round 23's, a tuple, as a list)."""
    *got, stream = _one_stage(S.Stream)
    *want, _ = _one_stage(_Whole)
    assert got == want
    calls, _, _, symbols = got
    refused = {c[5]: c for c in calls if c[0] == "refused"}
    assert {i: c[2] for i, c in refused.items()} == {
        **dict.fromkeys((4, 13, 22), ("R6-window",)), **dict.fromkeys((6, 17, 28), ("R6-previous",)),
        **dict.fromkeys((5, 8, 21, 25, 34, 39), ("disagree",))}
    assert all("'lean': {'ok': False, 'codes': ['R6-window']}" in refused[i][3] for i in (5, 25))
    assert all("'lean': {'ok': True" in refused[i][3] for i in (8, 21, 34)) and "non-integer" in refused[39][3]
    assert [stream.log[j]["window"].get("tags", "float") for j in stream._irregular] == [("t", 3), ("t", 23), "float"]
    assert set(symbols) == {"accept", "reject", "late"} and stream.failures == sum(s != "accept" for s in symbols)
    forgetful = _one_stage(_Forgetful)[0]
    assert 25 not in {c[5] for c in forgetful if c[0] == "refused"} and forgetful != want[0]


def test_r7_reads_each_values_first_registration():
    """Registered values: a log of 24 records, two in three registering `W` under its first root and the rest v1. A
    candidate's check gets the last record and the first registration, and the verifier gives it the verdict the
    whole log gives: the held root, a fresh root (R7-held), a window logged before, a wrong previous, a v1 record."""
    prog = _program()
    reg = _registered(prog)
    first, ports = _registered_record(prog, reg)
    stream = _stream()

    def at(row: int, previous: Any, entry: Any, **change: Any) -> dict:
        return {**R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": row}, previous=previous,
                           registered=[entry] if entry else ()), **change}

    stream._append(stream._registration(first, 0))
    for i in range(1, 24):
        stream._append(stream._registration(at(SERVED["row"] + i, R.digest(stream._log[-1]),
                                                reg.entry() if i % 3 else None), i))
    last = R.digest(stream._log[-1])
    row = SERVED["row"] + 24
    for cand, registered, codes in (
            (at(row, last, reg.entry()), {"W": 2}, ()),
            (at(row, last, _registered(prog).entry()), {"W": 2}, ("R7-held",)),
            (at(SERVED["row"] + 5, last, reg.entry()), {"W": 2}, ("R6-window",)),
            (at(row, "ab" * 64, reg.entry()), {"W": 2}, ("R6-previous",)),
            (at(row, last, None), None, ()),
            (at(SERVED["row"], last, None), None, ("R6-window",))):
        earlier = stream._earlier(cand)
        assert earlier == [first, stream._log[-1]] or (cand["window"]["row"] != row and len(earlier) == 3)
        held = ["W"] if registered else None
        part = V.check(cand, prog, Q, LAW, SCHEME, ports, None, (), earlier, None, None, registered, held=held)
        whole = V.check(cand, prog, Q, LAW, SCHEME, ports, None, (), stream.log, None, None, registered, held=held)
        assert part == whole and part.codes == codes and part.verifiers["lean"]["codes"] == list(codes)


def test_a_supplied_position_reads_the_whole_log():
    """W-a: a supplied position follows every proof block of the log (`registered.log_base`), so a check whose supplied values
    read one gets the whole log and its verdict. On the few records R6 and R7 read, the honest record's positions would
    follow fewer proof blocks and it would be refused (`R7-supplied`)."""
    base = _seeded_case()
    stream, log = _stream(), []
    for i in range(6):
        rec, _, _ = _mixed_case(base, log=log, window={**BOUND, "row": SERVED["row"] + i})
        stream._append(stream._registration(rec, i))
        log.append(rec)
    cand, check, _ = _mixed_case(base, log=log, window={**BOUND, "row": SERVED["row"] + 6})
    assert stream._earlier(cand, MIXED) is stream._log and len(few := stream._earlier(cand)) == 2
    args = (cand, check["program"], check["query"], LAW, SCHEME, check["ports"], None, ())
    assert V.check(*args, stream.log, None, None, check["registered"], MIXED, binding=BINDING, held=["W"]).ok
    assert V.check(*args, few, None, None, check["registered"], MIXED, binding=BINDING, held=["W"]).codes == ("R7-supplied",)


def test_windows_equal_to_either_verifier_share_a_key():
    """Python's `==` and the Lean's `jeq` (equal canonical text) find equal only windows with one key; a window that isn't
    plain JSON has none, and its record is read by every check."""
    k = S._window_key
    assert k({"a": 1, "b": [True, None, "s"]}) == k({"b": [1, None, "s"], "a": True}) == k({"a": True, "b": [1, None, "s"]})
    assert len({k({"row": 1}), k({"row": 2}), k({"row": "1"}), k([1]), k({"row": [1]}), k({"row": None})}) == 6
    for irregular in ({"row": 1.0}, ("served", 1), {1: "a"}, b"w", [{"row": float("nan")}], {"row": (1,)}):
        assert k(irregular) is None
