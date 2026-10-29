"""One window in flight (`Pace.in_flight`): at the default, two, a paced session is as it always was, its record and
what it makes public unchanged; at one an honest session is accepted, each coin leaving after the root before it, the
prover idle a round trip between windows; and a root after its window's boundary, a boundary before its window's last
root, a record that names one in flight with a coin out before the root before it, and a challenger log that shows one
are each refused."""
from __future__ import annotations

import dataclasses

import pytest

from test_ledger import DOWN_NS, PASS_NS, UP_NS, paced
from remote_bandwidth_fakes import Clock, Entropy
from test_remote_bandwidth import A, N, challenger, finish, params, store  # noqa: F401
from verity.core.protocols.remote_bandwidth import audit, coins
from verity.core.protocols.remote_bandwidth.audit import Pace, Session, SessionError


def logged(s: Session) -> tuple[coins.LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """The session's coin commitment and the roots its in-process challenger opened each coin after, by family."""
    return s.coins.commitment, {label: s.coins.afters(label) for label, _ in s.coins.commitment.families}


def honest(store, k: int, f: int, windows: int = 5) -> tuple[Session, audit.Record]:
    s, prover, passes, clock, _ = paced(store, params(windows=windows, samples=N, window_ns=1, pace=Pace(k, 1000, f)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    return s, s.record()


def test_two_in_flight_is_the_default_and_names_nothing(store) -> None:
    assert Pace(1).in_flight == Pace(1, 1000).in_flight == 2
    assert Pace(1, 1000).to_json() == Pace(1, 1000, 2).to_json() == {"k": 1, "window_ns": 1000}
    assert Pace(1, 1000, 1).to_json() == {"k": 1, "window_ns": 1000, "in_flight": 1}
    for bad in (0, 3, 1.0, True, None):
        with pytest.raises(ValueError, match="in_flight is 1 or 2"):
            Pace(1, 1000, bad)
    s, r = honest(store, 1, 2)
    assert r.public()["pace"] == {"k": 1, "window_ns": 1000}
    assert [w.released_ns for w in r.windows] == [0, 1, 350, 650, 950]
    assert s.released_reason(*logged(s)) is None


def test_an_honest_session_at_one_in_flight_is_accepted_each_coin_after_the_root_before_it(store) -> None:
    """At k = 1 each window's boundary leaves once its own root is in, so the prover waits a round trip, `DOWN_NS` +
    `UP_NS`, between passes: a pass every 350 ns, not 300.  Every challenge after window 0's is opened after the window
    before's root, and the record names one in flight."""
    s, r = honest(store, 1, 1)
    assert [w.released_ns for w in r.windows] == [0, 350, 700, 1050, 1400]
    assert [w.arrived_ns for w in r.windows] == [(350,), (700,), (1050,), (1400,), (1750,)]
    assert all(r.windows[t + 1].released_ns >= max(w.arrived_ns) for t, w in enumerate(r.windows[:-1]))
    assert PASS_NS + DOWN_NS + UP_NS == 350
    assert s.coins.afters(coins.CHALLENGE)[1:] == tuple(w.roots[-1] for w in s.windows[:-1])
    assert s.released_reason(*logged(s)) is None
    assert r.public()["pace"] == {"k": 1, "window_ns": 1000, "in_flight": 1}
    s, r = honest(store, 3, 1, windows=3)
    assert [w.passes for w in r.windows] == [3, 3, 3]
    assert all(r.windows[t + 1].released_ns >= max(w.arrived_ns) for t, w in enumerate(r.windows[:-1]))


def test_the_boundary_is_due_at_the_windows_kth_root_at_one_in_flight(store) -> None:
    clock = Clock()
    p = params(windows=3, window_ns=5, pace=Pace(2, 1000, 1))
    s = Session(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    coin = s.start()
    assert s.next_boundary_ns() == 2 * A, "not due before its k roots: at the latest k allowances after its coin"
    prover = audit.Prover(store, None, entropy=Entropy(b"receipt salts"))
    clock.now = 100
    ps = prover.run(coin, 0, store.registration.digest, None)
    assert s.receive(0, 0, ps.root, step=0) is None
    assert s.next_boundary_ns() == 2 * A, "at k − 1 roots it is still not due"
    clock.now = 200
    ps = prover.run(coin, 1, ps.root, None)
    assert s.receive(0, 1, ps.root, step=1) is None
    assert s.next_boundary_ns() == 200, "due at its k-th root, when that root arrived"


def two_roots_across_a_boundary(store, f: int, *, back: bool = False):
    """Window 0's first root, its boundary (sooner than due), then its second root; with `back`, the boundary on a
    clock read before the first root arrived."""
    clock = Clock()
    p = params(windows=3, window_ns=5, pace=Pace(2, 1000, f))
    s = Session(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    coin = s.start()
    prover = audit.Prover(store, None, entropy=Entropy(b"receipt salts"))
    clock.now = 100
    ps = prover.run(coin, 0, store.registration.digest, None)
    assert s.receive(0, 0, ps.root, step=0) is None
    clock.now = 50 if back else 150
    s.boundary()
    clock.now = 200
    return s.receive(0, 1, prover.run(coin, 1, ps.root, None).root, step=1)


def test_a_root_after_its_windows_boundary_is_out_of_turn_at_one_in_flight(store) -> None:
    """At two in flight window 0's second root may follow its boundary, up to its deadline; at one it is `order`."""
    assert two_roots_across_a_boundary(store, 2) is None
    assert two_roots_across_a_boundary(store, 1) == audit.Verdict(0, False, "order")


def test_a_boundary_before_its_windows_last_root_arrived_is_refused_at_one_in_flight(store) -> None:
    assert two_roots_across_a_boundary(store, 2, back=True) is None
    with pytest.raises(SessionError, match="after its last root"):
        two_roots_across_a_boundary(store, 1, back=True)


def test_a_record_naming_one_in_flight_with_a_coin_before_the_root_before_it_is_refused(store) -> None:
    """At two in flight and k = 1, window t + 1's challenge leaves while window t's pass runs, before its root: that
    record, renamed to one in flight, is refused, and the honest one-in-flight record renamed to two is not."""
    _, r2 = honest(store, 1, 2)
    with pytest.raises(ValueError, match="one window in flight"):
        dataclasses.replace(r2, pace=Pace(1, 1000, 1))
    _, r1 = honest(store, 1, 1)
    assert dataclasses.replace(r1, pace=Pace(1, 1000)).pace.in_flight == 2


def test_a_challenger_log_with_a_coin_before_the_root_before_it_is_refused_at_one_in_flight(store) -> None:
    """The two-in-flight session's log opens σ_1 after R, before window 0's root: held to one in flight it is refused."""
    s, _ = honest(store, 1, 2)
    signed, received = logged(s)
    windows, appends = [w.roots for w in s.windows], [a[2].root for a in s._appends]
    assert coins.released_reason(s.coins, windows, appends, signed, received) is None
    assert coins.released_reason(s.coins, windows, appends, signed, received, 1) == (
        "σ_1 was opened before window 0's last root, at one window in flight")
