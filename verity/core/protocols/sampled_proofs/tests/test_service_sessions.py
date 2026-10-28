"""One proof session over several rounds (`Stream.session`): a stream of `rounds` R holds up to R registrations open, a
session opens once after its last covered draw, the first covered round's proof deadline spans it, every covered draw's
outcome binds the same delivered session, and each round's audit record reads only its own proof blocks and roots."""

from __future__ import annotations

import pytest

from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

from test_service import (DELTA, PAIR, QUERY, RUN, SESSION, Clock, _lean, _ports, _refused, _spec)

N = 6


def _units(*draws):
    """A sampler that draws `draws[i]` at its i-th call: each round its own proof blocks."""
    it = iter(draws)
    return lambda law, n: next(it)


def _stream(rounds=2, budget=4, clock=None, sampler=None, spec=None) -> S.Stream:
    return S.Stream(spec or _spec(), PAIR, budget=budget, delta=DELTA, challenger="challenger-test",
                    coins=lambda n: b"\x07" * n, clock=clock or Clock(),
                    sampler=sampler or _units([0, 1], [2, 3], [4, 5], [0, 5], [1, 2]), rounds=rounds)


def _roots(row: int) -> dict[str, str]:
    """Round `row`'s own salted roots."""
    return {"x": f"{2 * row + 1:02x}" * 64, "out": f"{2 * row + 2:02x}" * 64}


def _record(row: int, previous: str | None) -> dict:
    ports = _ports(N)
    roots = [{"port": p, "root": _roots(row)[p], **ports[p]} for p in ("x", "out")]
    return R.record(P.population_program(PAIR, N), QUERY, {"law": "subset", "k": 2}, S.SCHEME, roots,
                    {"kind": "served", "run": RUN, "row": row}, previous=previous)


def _register(stream: S.Stream) -> S.WindowCommitmentRecord:
    row = len(stream._entries)
    previous = stream._entries[-1].registration.digest if stream._entries else None
    return stream.register(_record(row, previous), _ports(N))


def _drawn(stream: S.Stream, k: int) -> list[S.Draw]:
    out = []
    for _ in range(k):
        out.append(stream.select(_register(stream)))
    return out


def _outcome(stream: S.Stream, draw: S.Draw, roots_of: int | None = None, units=None) -> S.Outcome:
    row = draw.registration.order if roots_of is None else roots_of
    return stream.outcome(draw, _lean(), {"statement": "stub", "table": row}, proved_units=units,
                          session_roots=_roots(row))


def test_a_two_round_session_is_accepted():
    clock = Clock()
    stream = _stream(clock=clock)
    d0 = stream.select(_register(stream))
    clock.now = 12.6
    d1 = stream.select(_register(stream))
    assert (d0.units, d1.units) == ([0, 1], [2, 3])
    session = stream.session([d0, d1])
    assert session.orders == [0, 1] and session.stream == stream.digest
    clock.now = 50.0
    assert stream.deliver(session, [SESSION]) is None
    outs = [_outcome(stream, d) for d in (d0, d1)]
    assert all(o.accepted for o in outs) and stream.symbols == ("accept", "accept") and stream.failures == 0
    rounds = [{"order": d.registration.order, "registration": d.registration.digest, "draw": S._sha512(dict(d.draw))}
              for d in (d0, d1)]
    for body, d in zip(stream.records, (d0, d1)):
        assert body["sessions"] == [SESSION] and body["session"] == {"rounds": rounds}
        audit = body["audit"]
        assert audit["draw"] == d.draw and audit["proved_units"] == d.units and audit["accepted"]
        assert audit["registration_digest"] == d.registration.digest
        assert audit["integrity_profile"]["evidence"]["sessions"] == [SESSION]
    assert stream.register(_record(2, d1.registration.digest), _ports(N)).order == 2


def test_each_rounds_audit_record_reads_only_its_own_units_and_roots():
    stream = _stream()
    d0, d1 = _drawn(stream, 2)
    stream.deliver(stream.session([d0, d1]), [SESSION])
    wrong_roots = _outcome(stream, d1, roots_of=0)
    assert not wrong_roots.accepted and wrong_roots.verdict.detail["reject"]["checks"] == ["roots"]
    wrong_units = _outcome(stream, d0, units=d1.units)
    assert not wrong_units.accepted and wrong_units.verdict.detail["reject"]["checks"] == ["proved"]
    assert stream.symbols == ("reject", "reject") and stream.failures == 2


def test_a_session_opened_before_the_last_rounds_coin_is_refused():
    stream = _stream(rounds=3)
    d0 = stream.select(_register(stream))
    r1 = _register(stream)
    early = S.Draw(r1, d0.law, {**d0.draw, "units": [2, 3]})
    assert _refused(lambda: stream.session([d0, early])) == ("not-drawn",)
    d1 = stream.select(r1)
    assert _refused(lambda: stream.session([d1, d0])) == ("order",)
    assert _refused(lambda: stream.session([])) == ("rounds",)
    d2 = stream.select(_register(stream))
    assert _refused(lambda: stream.session([d0, d2])) == ("order",)
    session = stream.session([d0, d1])
    assert _refused(lambda: stream.session([d1, d2])) == ("covered",)
    assert _refused(lambda: stream.deliver(d0, [SESSION])) == ("covered",)
    assert stream.session([d2]).orders == [2]
    stream.deliver(session, [SESSION])
    assert _refused(lambda: stream.deliver(session, [SESSION])) == ("delivered",)
    assert _outcome(stream, d0).accepted and _outcome(stream, d1).accepted
    assert _refused(lambda: stream.session([d0])) == ("final",)


def test_a_session_past_the_first_rounds_deadline_is_a_reject():
    """The session's deadline is its first round's: a proof 61 s after round 0's draw is `late` for both rounds, though
    it is 21 s after round 1's; one delivered by round 0's deadline is accepted."""
    for deliver_at, symbols in ((59.5, ("accept", "accept")), (61.0, ("late", "late"))):
        clock = Clock()
        stream = _stream(clock=clock)
        d0 = stream.select(_register(stream))
        clock.now = 40.0
        d1 = stream.select(_register(stream))
        session = stream.session([d0, d1])
        clock.now = deliver_at
        out = stream.deliver(session, [SESSION])
        if out is None:
            out = tuple(_outcome(stream, d) for d in (d0, d1))
        assert tuple(o.verdict.codes[0] if o.verdict.codes else "accept" for o in out) == symbols
        assert stream.symbols == symbols
        assert all(b["session"]["rounds"][0]["order"] == 0 for b in stream.records)
    assert stream.failures == 2 and all(b["audit"] is None and b["sessions"] == [SESSION] for b in stream.records)
    assert _refused(lambda: stream.deliver(session, [SESSION])) == ("final",)


def test_an_undelivered_session_ends_late_at_its_first_rounds_deadline():
    clock = Clock()
    stream = _stream(clock=clock)
    d0 = stream.select(_register(stream))
    clock.now = 30.0
    d1 = stream.select(_register(stream))
    stream.session([d0, d1])
    clock.now = 60.0
    with pytest.raises(S.Pending):
        stream.outcome(d1, _lean(), {})
    clock.now = 60.5
    assert stream.outcome(d1, _lean(), {}).verdict.codes == ("late",)
    assert stream.outcome(d0, _lean(), {}).verdict.codes == ("late",)
    assert [b["order"] for b in stream.records] == [1, 0] and stream.failures == 2


def test_open_registrations_are_bounded_by_rounds_and_the_budget():
    """Never more than `rounds` registrations open, nor more than the budget's failures left, so a failed session
    never takes a stream past its budget."""
    stream = _stream(rounds=2, budget=4)
    _drawn(stream, 2)
    assert _refused(lambda: _register(stream)) == ("open",)

    clock = Clock()
    stream = _stream(rounds=4, budget=2, clock=clock)
    d0, d1 = _drawn(stream, 2)
    assert _refused(lambda: _register(stream)) == ("open",)
    clock.now = 61.0
    assert len(stream.deliver(stream.session([d0, d1]), [SESSION])) == 2
    assert stream.stopped and stream.failures == 2 == stream.budget
    assert _refused(lambda: _register(stream)) == ("stopped",)

    stream = _stream(rounds=4, budget=3, clock=Clock())
    d0, d1, d2 = _drawn(stream, 3)
    assert _refused(lambda: _register(stream)) == ("open",)
    stream.deliver(d0, [SESSION])
    assert not _outcome(stream, d0, roots_of=1).accepted
    assert stream.failures == 1 and _refused(lambda: _register(stream)) == ("open",)
    stream.deliver(d1, [SESSION])
    assert _outcome(stream, d1).accepted and _register(stream).order == 3


def test_a_stream_of_rounds_states_them_and_runs_one_round_as_before():
    for rounds in (0, 5, 2.0, True):
        with pytest.raises(ValueError, match="rounds"):
            _stream(rounds=rounds)
    one = S.Stream(_spec(), PAIR, budget=4, delta=DELTA, challenger="challenger-test")
    assert one.digest == S._sha512({"format": S.STREAM, "spec": _spec().digest(), "budget": 4, "delta": DELTA,
                                    "challenger": "challenger-test"})
    assert _stream(rounds=1).digest == one.digest != _stream(rounds=2).digest
    stream = _stream(rounds=1)
    d0 = stream.select(_register(stream))
    assert _refused(lambda: _register(stream)) == ("open",)
    session = stream.session([d0])
    assert session == S.ProofRun(stream.digest, (d0,)) and stream.deliver(session, [SESSION]) is None
    assert _outcome(stream, d0).accepted and stream.records[0]["session"]["rounds"][0]["order"] == 0
    d1 = stream.select(_register(stream))
    assert stream.deliver(d1, [SESSION]) is None and _outcome(stream, d1).accepted
    assert "session" not in stream.records[1]


def test_a_session_is_the_streams_own():
    stream, other = _stream(), _stream(budget=3)
    assert other.digest != stream.digest
    d0, d1 = _drawn(stream, 2)
    session = stream.session([d0, d1])
    for forged in (S.ProofRun(other.digest, session.draws), S.ProofRun(stream.digest, ()),
                   S.ProofRun(stream.digest, (d0,))):
        assert _refused(lambda: stream.deliver(forged, [SESSION])) == ("not-session",)
    assert _refused(lambda: stream.deliver(session, [])) == ("sessions",)
