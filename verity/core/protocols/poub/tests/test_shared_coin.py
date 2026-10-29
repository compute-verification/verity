"""One openings coin over g consecutive windows (`Params.groups_per_coin`): at the default, g = 1, a session draws each
window from its own coin, as it always has; at g = 4 an honest session is accepted, each draw closed at its last window
and its windows verified on their shares; a prover that skips the reads of one window of a draw is refused there; a
window short of passes fails `few` wherever it sits in its draw; and a session's coins through a challenger service
hold to what it signed, with ⌈W/g⌉ openings coins, each released after its draw's last root."""
from __future__ import annotations

import copy
import dataclasses
from fractions import Fraction

import pytest

from test_poub import N, SCHEDULE, Entropy, drive, params, store  # noqa: F401
from test_ledger import node
from test_poub_coins import APPENDS, Service, sign  # noqa: F401
from verity.core.protocols.poub import audit, codec, coins, ledger, timing
from verity.core.protocols.poub.audit import Session, SessionError

SCHEDULE8 = timing.honest(1000, 8, 300, gap_ns=10, down_ns=20, up_ns=30, jitter_ns=5)
"""`SCHEDULE` over eight windows: passes (4, 3, 3, 3, 4, 3, 3, 3), so each draw of four holds 13 passes."""


def shared(store, g: int, windows: int = 8, **kw) -> tuple[audit.Params, coins.Coins]:
    """Parameters of `windows` windows drawn `g` a coin, and a challenger in this process committed to their coins."""
    p = params(windows=windows, groups_per_coin=g, **kw)
    return p, coins.local(store.registration.digest, 0, windows, Entropy(b"coins"), appends=APPENDS, per_coin=g)


def finish_draws(s: Session, prover: audit.Prover, passes, clock, open_=None) -> list[audit.Verdict]:
    """`finish` a draw at a time: past the last deadline, each draw closed at its last window, then each of its windows
    verified on its own share of the draws, in order, up to the first rejection; a window whose share is none keeps
    the verdict its close gave it."""
    clock.now = max(clock.now, s.deadline(len(s.windows) - 1) + 1)
    out = []
    for t in range(s.params.windows):
        group = s.params.draw_of(t)
        if t != group[-1]:
            continue
        drawn = s.close(t)
        if isinstance(drawn, audit.Verdict):
            return out + [drawn]
        for u in group:
            if s.windows[u].verdict is not None:
                out.append(s.windows[u].verdict)
                continue
            out.append(s.verify(u, [(open_ or prover.open)(passes[u, i], p) for i, p in s.windows[u].draws]))
            if not out[-1].accepted:
                return out
    return out


def outcomes(verdicts: list[audit.Verdict]) -> list[tuple[int, bool, str | None]]:
    return [(v.window, v.accepted, v.failure) for v in verdicts]


def test_a_coin_a_window_is_the_default_and_draws_each_window_alone(store) -> None:
    p = params()
    assert p.groups_per_coin == 1 and p.coins == 2 * p.windows
    assert [list(p.draw_of(t)) for t in range(p.windows)] == [[0], [1], [2]]
    assert coins.families(3, APPENDS) == coins.families(3, APPENDS, 1) == (
        (coins.CHALLENGE, 3), (coins.OPENINGS, 3), (coins.APPEND, APPENDS))
    s, prover, passes, clock = drive(store, p, SCHEDULE)
    assert all(v.accepted for v in finish_draws(s, prover, passes, clock))
    assert s.coins.afters(coins.OPENINGS) == tuple(w.roots[-1] for w in s.windows)
    r = s.record()
    assert r.groups_per_coin == 1 and "groups_per_coin" not in r.public()
    assert audit.pooled_windows(r)[0] == [(w.receipts, len(v.draws)) for w, v in zip(r.windows, s.windows)]


def test_groups_per_coin_is_a_positive_integer_and_the_coins_are_the_sessions() -> None:
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="groups_per_coin"):
            params(groups_per_coin=bad)
        with pytest.raises(ValueError, match="groups_per_coin"):
            coins.openings_coins(3, bad)
    assert [coins.openings_coins(w, g) for w, g in ((8, 4), (8, 3), (3, 2), (3, 8), (1, 1))] == [2, 3, 2, 1, 1]
    p = params(windows=5, groups_per_coin=2)
    assert p.coins == 5 + 3 and [list(p.draw_of(t)) for t in (0, 1, 2, 4)] == [[0, 1], [0, 1], [2, 3], [4]]


def test_a_session_refuses_coins_drawn_another_number_of_windows_a_coin(store) -> None:
    p, c = shared(store, 4)
    with pytest.raises(ValueError, match="4 a draw"):
        Session(p, store.registration, None, coins.local(store.registration.digest, 0, 8, Entropy(b"coins"),
                                                         appends=APPENDS), not_before_ns=0, clock=lambda: 0)
    with pytest.raises(ValueError, match="1 a draw"):
        Session(params(windows=8), store.registration, None, c, not_before_ns=0, clock=lambda: 0)
    assert coins.layout_reason(c.commitment, 0, 8, 4) is None
    assert "openings 8" in coins.layout_reason(c.commitment, 0, 8)
    with pytest.raises(coins.CoinError, match="openings 3"):
        coins.Coins(c.challenger, store.registration.digest, 0, 8, per_coin=3)


def test_an_honest_session_drawn_four_windows_a_coin_is_accepted(store) -> None:
    """Eight windows, two draws: windows 0 … 2 close with window 3, the draw's `samples` receipts uniform over its
    windows' receipts together, each window verified on its share; the openings coins leave after windows 3's and 7's
    last roots; the record names g, and the pooled bound reads each draw as one window."""
    p, c = shared(store, 4)
    assert p.coins == 10 and c.commitment.families[:2] == ((coins.CHALLENGE, 8), (coins.OPENINGS, 2))
    s, prover, passes, clock = drive(store, p, SCHEDULE8, coins_=c)
    clock.now = s.deadline(7) + 1
    for t in (0, 1, 2):
        with pytest.raises(SessionError, match="drawn with its draw's last window, 3"):
            s.close(t)
    with pytest.raises(SessionError, match="no draws awaiting"):
        s.verify(0, [])
    verdicts = finish_draws(s, prover, passes, clock)
    assert outcomes(verdicts) == [(t, True, None) for t in range(8)] and s.outcome == "accepted"
    assert s.coins.afters(coins.OPENINGS) == (s.windows[3].roots[-1], s.windows[7].roots[-1])
    assert [sum(len(w.draws) for w in s.windows[a:a + 4]) for a in (0, 4)] == [p.samples, p.samples]
    assert len({w.opened_ns for w in s.windows[:4]}) == 1
    r = s.record()
    assert r.public()["groups_per_coin"] == 4 and r.public()["passes"] == list(SCHEDULE8.passes)
    us = [w.receipts for w in r.windows]
    assert us == [n * N for n in SCHEDULE8.passes]
    assert audit.pooled_windows(r)[0] == [(sum(us[:4]), p.samples), (sum(us[4:]), p.samples)]
    fs, floors = audit.pooled_bound(r).floors, audit.floors(r)
    for t, u in enumerate(us):
        draw = sum(us[t // 4 * 4:t // 4 * 4 + 4])
        assert Fraction(floors[t]) >= min(Fraction(1), fs[t // 4] * draw / u) > fs[t // 4]


def test_a_window_the_draw_misses_is_accepted_at_its_close_and_refused_a_verify(store) -> None:
    """Three draws over four windows leave at least one window of each draw with none: its close accepts it, a verify
    of it is refused, and its record counts none drawn, the reading of its own share."""
    p, c = shared(store, 4, samples=3)
    s, prover, passes, clock = drive(store, p, SCHEDULE8, coins_=c)
    clock.now = s.deadline(7) + 1
    s.close(3), s.close(7)
    missed = [u for u in range(8) if s.windows[u].draws == ()]
    assert {u // 4 for u in missed} == {0, 1} and all(s.windows[u].verdict == audit.Verdict(u, True) for u in missed)
    for u in range(8):
        if u in missed:
            with pytest.raises(SessionError, match="no draws awaiting"):
                s.verify(u, [])
        else:
            assert s.windows[u].verdict is None
            assert s.verify(u, [prover.open(passes[u, i], q) for i, q in s.windows[u].draws]).accepted
    assert s.outcome == "accepted"
    r = s.record()
    assert [w.profile.drawn["receipt"] for w in r.windows] == [len(w.draws) for w in s.windows]
    assert audit.pooled_windows(r)[0] == [(sum(w.receipts for w in r.windows[a:a + 4]), 3) for a in (0, 4)]


def test_the_last_draw_may_hold_fewer_windows(store) -> None:
    """Three windows two a coin: windows 0 and 1 are one draw, window 2 the other alone.  The record's `samples` is
    each draw's request, so the memory line's δ charges the draw's rejection at two draws' `samples`, not three, and
    the receipts' binding term at two draws of at most twice a window's receipts each, not three windows."""
    p, c = shared(store, 2, windows=3)
    s, prover, passes, clock = drive(store, p, SCHEDULE, coins_=c)
    assert all(v.accepted for v in finish_draws(s, prover, passes, clock))
    assert s.coins.afters(coins.OPENINGS) == (s.windows[1].roots[-1], s.windows[2].roots[-1])
    assert len(s.windows[0].draws) + len(s.windows[1].draws) == len(s.windows[2].draws) == p.samples
    r = s.record()
    assert audit.pooled_windows(r)[0] == [(7 * N, p.samples), (3 * N, p.samples)]
    assert r.samples == p.samples and ledger.rejection(r) == 2 * p.samples * 2.0 ** -ledger.REJECTION_LOG2
    n = node(squarings=1e9)
    L = ledger.window_receipts(r.pace, r.blocks, r.append_coins, r.codec.max_blocks)
    q = n.sketch_outside * ledger.compressions(r, n)
    assert r.append_coins > 0 and L == (audit.STEP_LIMIT - 1) * r.codec.max_blocks
    per_draw, per_window = ledger.binding_over(2, 2 * L, p.samples, q), ledger.binding_over(3, L, p.samples, q)
    assert ledger.binding(r, n) == per_draw != per_window
    with pytest.raises(ValueError, match=r"the draw of windows 0…1 drew 7 of its 140 receipts, not min\(6, 140\)"):
        dataclasses.replace(r, samples=p.samples - 1)


def skipping(window: int):
    """A prover that reads nothing in `window`: its receipts there are of zeros, the blocks it would hold had it
    dropped the store, and every other window's are honest. It opens with the honest blocks, fetched again."""
    def run(prover, t, i, challenges, previous, plan):
        if t != window:
            return prover.run(challenges[t], i, previous, plan)
        held = copy.copy(prover.store)
        held.blocks = [bytes(codec.BLOCK_BYTES)] * N
        return audit.Prover(held, None, prover.commitment, entropy=prover.entropy).run(challenges[t], i, previous, plan)
    return run


def test_a_prover_that_skips_one_window_of_a_draw_is_refused_there(store) -> None:
    """Under one coin over four windows a prover may gamble that the draw misses the window it skipped, here window 2
    of draw 0's 260 receipts, 60 of them; at 33 draws the draw reaches it, and its first wrong receipt rejects."""
    p, c = shared(store, 4, samples=33)
    s, prover, passes, clock = drive(store, p, SCHEDULE8, run=skipping(2), coins_=c)
    verdicts = finish_draws(s, prover, passes, clock)
    assert s.windows[2].draws
    assert outcomes(verdicts) == [(0, True, None), (1, True, None), (2, False, "receipt")]
    with pytest.raises(SessionError):
        s.record()


def test_a_window_of_too_few_passes_fails_in_its_draw(store) -> None:
    """At four passes a window, windows 1 … 3 hold three: draw 0's close fails `few` at window 1, the first of them."""
    p, c = shared(store, 4, min_passes=4)
    s, prover, passes, clock = drive(store, p, SCHEDULE8, coins_=c)
    assert outcomes(finish_draws(s, prover, passes, clock)) == [(1, False, "few")]


def test_a_session_drawn_two_windows_a_coin_through_a_challenger_service_holds_to_what_it_signed(
        store, tmp_path, sign) -> None:
    p = params(groups_per_coin=2)
    service = Service(tmp_path / "s-0", sign, "s-0", 0, p.windows, b"service", per_coin=2)
    s, prover, passes, clock = drive(store, p, SCHEDULE, coins_=service.coins(store, 0, p.windows, per_coin=2))
    assert all(v.accepted for v in finish_draws(s, prover, passes, clock)) and service.close() is None
    signed, received = service.signed()
    assert signed == s.commitment and s.released_reason(signed, received) is None
    assert [f for f in signed.families if f[0] != coins.APPEND] == [(coins.CHALLENGE, 3), (coins.OPENINGS, 2)]
    assert received[coins.OPENINGS] == (s.windows[1].roots[-1], s.windows[2].roots[-1])
    assert coins.layout_reason(signed, 0, p.windows, 2) is None and "openings 3" in coins.layout_reason(signed, 0, 3)
