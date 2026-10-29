"""The fleet's halt (R5-all, `service.Fleet`): at the failure that brings any stream of an operator to its budget, every
stream of the operator halts, each of its outcomes not yet final ends as `halted`, never `accept`, outcomes already final
stay, and nothing of the operator is registered or accepted again. Each decision is the verifier of record's (`Fleet.run`
and `Stream.run`'s `halt`)."""

from __future__ import annotations

import pytest

from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R, verifier as V

import test_service_sessions as T
from test_service import SESSION, Clock, _lean, _ports, _record, _refused, _round, _stream


def _open(stream: S.Stream, n: int = 6) -> S.Registration:
    """A registration the stream receives and leaves open."""
    previous = R.digest(stream._entries[-1].registration.record) if stream._entries else None
    return stream.register(_record(n, row=len(stream._entries), previous=previous), _ports(n))


def _fleet(*operators: str, budget: int = 1, clock: Clock | None = None) -> tuple[S.Fleet, list[S.Stream]]:
    fleet = S.Fleet()
    streams = [_stream(budget=budget, clock=clock) for _ in operators]
    assert [fleet.attach(s, o) for s, o in zip(streams, operators)] == list(range(len(operators)))
    return fleet, streams


def test_a_budget_of_one_halts_the_operator_at_its_first_failure():
    """b = 1: one reject halts every stream of the operator. The other stream's sample, delivered and pending its verdicts,
    ends `halted` with no failure of its own, and its accepted verdicts come too late; every call on either stream is
    refused, so is another stream for the operator, and another operator's stream runs on."""
    fleet, (a, b, c) = _fleet("o", "o", "p")
    da, db = a.select(_open(a)), b.select(_open(b))
    assert b.deliver(db, [SESSION]) is None and a.deliver(da, [SESSION]) is None
    out = a.outcome(da, _lean(False), {"statement": "stub"})
    assert not out.accepted and out.verdict.codes == ("reject",)
    cause = {"operator": "o", "stream": 0, "at": 2}
    assert fleet.halted == {"o": cause} and a.halted == cause and b.halted == cause and c.halted is None
    assert a.symbols == ("reject",) and b.symbols == ("halted",) and (a.failures, b.failures) == (1, 0)
    body = b.records[0]
    assert body["symbol"] == "halted" and body["halt"] == cause and body["audit"] is None
    assert body["draw"] == S._sha512(dict(db.draw)) and body["sessions"] == [SESSION]
    assert b._entries[0].outcome.verdict.detail == {"halted": cause} and b._entries[0].outcome.profile is None
    assert _refused(lambda: b.outcome(db, _lean(), {"statement": "stub"})) == ("halted",)
    for s, d in ((a, da), (b, db)):
        for call in (s.ready, s.salt, lambda: _open(s), lambda: s.select(d.registration), lambda: s.session([d]),
                     lambda: s.deliver(d, [SESSION]), lambda: s.outcome(d, _lean(), {})):
            assert _refused(call) == ("halted",)
    assert _refused(lambda: fleet.attach(_stream(), "o")) == ("halted",)
    assert len(a.records) == len(b.records) == 1 and a.halted == cause
    assert _round(c).accepted and c.symbols == ("accept",) and fleet.size == 3 and fleet.halted == {"o": cause}


def test_a_budget_above_one_tolerates_one_failure_fewer():
    """b = 3: a stream's two failures, and another's one, halt nothing: both keep accepting. The halt comes at the third
    failure of one stream, and ends the other's registration, not yet drawn, as `halted`."""
    fleet, (a, b) = _fleet("o", "o", budget=3)
    assert not _round(a, accepted=False).accepted and not _round(a, accepted=False).accepted
    assert not _round(b, accepted=False).accepted
    assert fleet.halted == {} and a.halted is None and b.halted is None and (a.failures, b.failures) == (2, 1)
    assert _round(b).accepted and _round(a).accepted
    rb = _open(b)
    assert not _round(a, accepted=False).accepted and a.stopped
    cause = {"operator": "o", "stream": 0, "at": 7}
    assert fleet.halted == {"o": cause} and a.halted == cause and b.halted == cause
    assert a.symbols == ("reject", "reject", "accept", "reject") and b.symbols == ("reject", "accept", "halted")
    assert b.records[-1]["draw"] is None and b.records[-1]["order"] == rb.order and b.failures == 1
    assert _refused(lambda: b.select(rb)) == ("halted",)


def test_an_honest_late_counts_toward_the_budget():
    """A proof that misses its deadline, delivered late or never, fails as a reject does: at b = 2 two of them halt the
    operator, though the prover may only have been slow."""
    clock = Clock()
    fleet, (a, b) = _fleet("o", "o", budget=2, clock=clock)
    da = a.select(_open(a))
    clock.now = 60.5
    assert a.deliver(da, [SESSION]).verdict.codes == ("late",) and fleet.halted == {} and a.failures == 1
    _open(b)
    da = a.select(_open(a))
    clock.now = 121.0
    assert a.outcome(da, _lean(), {"statement": "stub"}).verdict.codes == ("late",)
    assert fleet.halted == {"o": {"operator": "o", "stream": 0, "at": 3}}
    assert a.symbols == ("late", "late") and b.symbols == ("halted",) and b.records[0]["draw"] is None


def test_a_late_session_counts_each_round_it_covers():
    """A session late for its two rounds is two failures at once: at b = 2 the operator halts after the session's
    outcomes, which stay `late`, and the operator's other stream halts with it."""
    clock = Clock()
    fleet = S.Fleet()
    s, t = T._stream(rounds=2, budget=2, clock=clock), _stream(clock=clock)
    assert (fleet.attach(s, "o"), fleet.attach(t, "o")) == (0, 1)
    d0, d1 = T._drawn(s, 2)
    session = s.session([d0, d1])
    _open(t)
    clock.now = 61.0
    assert [o.verdict.codes for o in s.deliver(session, [SESSION])] == [("late",), ("late",)]
    assert fleet.halted == {"o": {"operator": "o", "stream": 0, "at": 3}}
    assert s.symbols == ("late", "late") and t.symbols == ("halted",) and s.halted == t.halted


def test_a_stream_is_attached_once_before_its_first_registration():
    fleet, registered, attached = S.Fleet(), _stream(), _stream()
    _open(registered)
    with pytest.raises(ValueError, match="before its first registration"):
        fleet.attach(registered, "o")
    assert fleet.attach(attached, "o") == 0
    with pytest.raises(ValueError, match="attached to a fleet already"):
        S.Fleet().attach(attached, "p")
    with pytest.raises(ValueError, match="an operator is a string"):
        fleet.attach(_stream(), 7)
    assert fleet.attach(_stream(budget=1), "o") == 1 and fleet._rules["streams"] == [
        {"operator": "o", "budget": 2}, {"operator": "o", "budget": 1}]


def test_a_fleet_rules_disagreement_stops_the_fleet(monkeypatch):
    """Where the Lean decides a fleet's op otherwise, the op is refused (`disagree`) and so is every op after it: the
    stream's registration is refused with it, and nothing is registered."""
    fleet, (a,) = _fleet("o")
    real = V.lean
    monkeypatch.setattr(V, "lean", lambda f, value: {**real(f, value), "decision": {"ok": 5}} if f == "fleet" else
                        real(f, value))
    v = _refused(lambda: _open(a))
    assert v == (V.DISAGREE,) and a.log == () and fleet.size == 0
    monkeypatch.undo()
    assert _refused(lambda: _open(a)) == (V.DISAGREE,) and _refused(lambda: fleet.attach(_stream(), "p")) == (V.DISAGREE,)
