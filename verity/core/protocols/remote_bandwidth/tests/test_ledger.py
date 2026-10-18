"""The declared outputs (`ledger`) and pacing (`audit.Pace`) on the small store: a paced session releases a live coin
every k passes and records each root's arrival, γ is `audit.ledger`'s at both R and per stretch when paced, and the
memory line's forms are the Lean statements' conclusions read at the record, the paced ones refused for an unpaced
session or a pass that leaves a registered block out."""

from __future__ import annotations

import dataclasses
import math

import pytest

from test_remote_bandwidth import A, N, SCHEDULE, Clock, Entropy, challenger, drive, finish, params, partial, store  # noqa: F401
from verity.core.protocols.remote_bandwidth import audit, codec, ledger, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import Pace, Session

PASS_NS, DOWN_NS, UP_NS = 300, 20, 30
B = codec.BLOCK_BYTES


def node(*, chip_blocks: int = 1, squarings: float = 0.0, disk: float = 0.0, link: float = 0.0) -> ledger.Node:
    """A node sized to the small store: `timing`'s R, `chip_blocks` blocks on chip, and the given ceilings."""
    return ledger.Node(timing.SPEC_BYTES_PER_S, timing.READ_BYTES_PER_S, chip_blocks * B,
                       {codec.FALLBACK.field.name: squarings}, disk_bytes_per_s=disk, link_bytes_per_s=link)


def paced(store, p: audit.Params, plan=None):
    """A paced session driven on its own schedule: each pass takes `PASS_NS` from its coin's arrival or the pass before,
    each message `DOWN_NS` to the prover and `UP_NS` back; the prover runs at most k passes under a coin, moves to the
    next coin once it has arrived, and stops at the close; the verifier sends each boundary message at
    `Session.next_boundary_ns`, recomputed after every root.  Returns the session, prover, passes, clock and each
    root's arrival."""
    clock = Clock()
    s = Session(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    prover = audit.Prover(store, None, s.commitment, entropy=Entropy(b"receipt salts"))
    challenges, coin_at, closed_at = {0: s.start()}, {0: DOWN_NS}, None
    passes, arrivals, previous = {}, {}, store.registration.digest
    t = i = ready = step = 0
    pending = None
    while True:
        if pending is None:
            if i and t + 1 in coin_at and (coin_at[t + 1] <= ready or i == p.pace.k):
                t, i = t + 1, 0
            stopped = closed_at is not None and closed_at <= ready
            if not stopped and i < p.pace.k and t in coin_at:
                start = max(ready, coin_at[t])
                ps = prover.run(challenges[t], i, previous, plan and plan(t, i))
                pending, ready = (start + PASS_NS + UP_NS, t, i, ps), start + PASS_NS
        b = s.next_boundary_ns()
        if b is not None and (pending is None or b <= pending[0]):
            clock.now = max(clock.now, b)
            coin = s.boundary()
            if coin is None:
                closed_at = clock.now + DOWN_NS
            else:
                challenges[len(s.windows) - 1], coin_at[len(s.windows) - 1] = coin, clock.now + DOWN_NS
            continue
        if pending is None:
            break
        clock.now, u, j, ps = pending
        assert s.receive(u, j, ps.root, ps.plan.runs, step=step) is None
        passes[u, j], arrivals[u, j], previous, pending, step, i = ps, clock.now, ps.root, None, step + 1, i + 1
    return s, prover, passes, clock, arrivals


@pytest.fixture(scope="module")
def k1(store):
    """k = 1, two groups a stretch, two stretches and the closing window, every receipt drawn: no receipt is wrong."""
    s, prover, passes, clock, arrivals = paced(store, params(windows=5, samples=N, window_ns=1, pace=Pace(1, 2)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    return s.record(), arrivals


def test_a_paced_session_releases_a_coin_every_k_passes_and_records_arrivals(store, k1) -> None:
    """At k = 1 each window's boundary leaves once the window before has its root (window 0's at once), so the honest
    prover never waits on a coin, and each window holds one pass, its root's arrival on the verifier's clock."""
    r, arrivals = k1
    assert r.pace == Pace(1, 2) and r.public()["pace"] == {"groups": 2, "k": 1}
    assert [w.released_ns for w in r.windows] == [0, 1, 350, 650, 950]
    assert [w.arrived_ns for w in r.windows] == [(350,), (650,), (950,), (1250,), (1550,)]
    assert {(t, 0): a for t, (a,) in enumerate(w.arrived_ns for w in r.windows)} == arrivals
    assert [w.deadline_ns for w in r.windows] == [1 + A, 350 + A, 650 + A, 950 + A, 1250 + A]
    s, prover, passes, clock, _ = paced(store, params(windows=3, samples=N, window_ns=1, pace=Pace(3, 2)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    assert all(1 <= w.passes <= 3 for w in s.record().windows)


def test_next_boundary_is_the_stated_window_unpaced_and_the_pace_when_paced(store) -> None:
    clock = Clock()
    p = params()
    s = Session(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    assert s.next_boundary_ns() is None
    s.start()
    assert s.next_boundary_ns() == p.window_ns
    p = params(windows=3, window_ns=5, pace=Pace(2, 2))
    s = Session(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    coin = s.start()
    assert s.next_boundary_ns() == A, "not due before its first root: at the latest one allowance after its coin"
    prover = audit.Prover(store, None, s.commitment, entropy=Entropy(b"receipt salts"))
    clock.now = 100
    ps = prover.run(coin, 0, store.registration.digest, None)
    assert s.receive(0, 0, ps.root, step=0) is None
    assert s.next_boundary_ns() == 5, "due at k − 1 = 1 root: as soon as the stated window allows"
    for i in (1, 2):
        ps = prover.run(coin, i, ps.root, None)
        v = s.receive(0, i, ps.root, step=i)
    assert v == audit.Verdict(0, False, "order"), "a window's (k + 1)-th root is out of turn"


def test_params_refuse_a_pace_without_whole_stretches_and_a_closing_window() -> None:
    assert params(windows=3, pace=Pace(1, 2)).pace == Pace(1, 2)
    for w in (2, 4):
        with pytest.raises(ValueError, match="m·groups"):
            params(windows=w, pace=Pace(1, 2))
    with pytest.raises(ValueError, match="positive"):
        Pace(0, 1)


def test_arrivals_are_one_a_pass_and_out_of_order_ones_have_no_time_held_line(k1) -> None:
    """A record takes any clock's arrivals, one integer a pass (the session's clock is whatever its verifier's is);
    the time-held line reads them only in order from each window's challenge to its deadline."""
    r = k1[0]
    w = r.windows[0]
    for bad in ((349, 350), (350.0,)):
        with pytest.raises(ValueError, match="arrivals"):
            dataclasses.replace(w, arrived_ns=bad)
    for bad in ((w.released_ns - 1,), (w.deadline_ns + 1,)):
        odd = dataclasses.replace(r, windows=(dataclasses.replace(w, arrived_ns=bad), *r.windows[1:]))
        assert "not in order" in ledger.memory(odd, node())[1].refused


def test_gamma_is_the_ledger_at_both_r_and_per_window(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r, n = s.record(), node(squarings=1e9)
    out = ledger.declared(r, n)
    sk = ledger.sketch(r, n)
    assert 0 <= sk < 2.0 ** -40
    for g, cap in ((out.spec, n.spec_bytes_per_s), (out.measured, n.read_bytes_per_s)):
        want = audit.ledger(r, capacity_bytes_per_s=cap, chip_bytes=n.chip_bytes, rebuild=n.rebuild(r.codec))
        assert g.ledger == want and g.windows == want.windows and g.capacity_bytes_per_s == cap
        assert (g.error, g.error_horizon) == (want.error + sk, want.error_horizon + sk)
    assert out.to_json()["gamma"]["spec"]["gamma"] == out.spec.gamma


def test_a_paced_session_reads_gamma_per_stretch(k1) -> None:
    """Each stretch's terms are its windows' bytes over its own span, from its first challenge to its last deadline."""
    r, _ = k1
    n = node()
    g = ledger.gamma(r, n, n.spec_bytes_per_s)
    assert [x.window for x in g.windows] == [0, 1, 2] and len(g.ledger.windows) == 5
    first = g.windows[0]
    assert first.span_ns == r.windows[1].deadline_ns - r.windows[0].released_ns
    whole = n.spec_bytes_per_s * first.span_ns / 1e9
    assert first.timing == pytest.approx(1 - 2 * N * B / whole, rel=1e-12)
    assert first.chip == pytest.approx(2 * n.chip_bytes / whole, rel=1e-12)
    assert first.counting == pytest.approx(2 * (audit.named_bits(N) + audit.SLACK_LOG2) / 8 / whole, rel=1e-12)
    assert g.windows[2] == dataclasses.replace(g.ledger.windows[4], window=2)


def test_the_snapshot_is_helds_conclusion(store) -> None:
    """`planned − W − X·span − P·S` blocks over the passes, X the rebuilds' and the disks' rate: here every pass reads
    the whole store, so ε is (W/P + X·span/P + S)/N."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    n = node(chip_blocks=2, squarings=2e9 * codec.FALLBACK.encode_squarings, disk=B * 1e9)
    snap, held, links = ledger.memory(r, n)
    W, X, P = audit.pooled(r)[0], 3.0, r.passes
    assert snap.lost_blocks == pytest.approx(W / P + X * r.span_ns / P + 2, rel=1e-12)
    assert snap.epsilon == min(1, snap.lost_blocks / N) and snap.held_bytes == (1 - snap.epsilon) * N * B
    assert snap.delta == r.windows[0].profile.delta + P * 2.0 ** -audit.SLACK_LOG2 + ledger.sketch(r, n)
    assert "unpaced" in held.refused and links.refused == held.refused and held.epsilon is None


def test_the_time_held_line_is_held_over_times_conclusion(k1) -> None:
    """On `k1`'s schedule: s = (0, 1, 350, 650, 950), the arrivals 300 later each, so e_0 = 350, L = 650 − 1 and
    Σ = 950 − 350; no receipt is wrong, so the line charges the chip and 2L·X·(Σ + 2L) over Σ."""
    r, _ = k1
    X = 1e-3
    n = node(chip_blocks=1, squarings=X * codec.FALLBACK.encode_squarings * 1e9)
    snap, held, links = ledger.memory(r, n)
    L, span = 649, 600
    assert audit.pooled(r)[0] == 0
    assert held.lost_blocks == pytest.approx(1 + 2 * L * X * (span + 2 * L) / span, rel=1e-12)
    assert links.lost_blocks == held.lost_blocks
    more = ledger.memory(r, node(chip_blocks=1, squarings=X * codec.FALLBACK.encode_squarings * 1e9, link=B * 1e6))[2]
    assert more.lost_blocks == pytest.approx(1 + 2 * L * (X + 1e-3) * (span + 2 * L) / span, rel=1e-12)
    assert snap.lost_blocks == pytest.approx(X * r.span_ns / r.passes + 1, rel=1e-12)


def test_wrong_receipts_fill_the_groups_at_their_rates(store) -> None:
    """With 7 of 20 receipts drawn a window, the pooled wrong receipts W are spread over the groups at
    v = (1, 349, 300, 300, L) a wrong read, highest first, at most N a group of one pass."""
    s, prover, passes, clock, _ = paced(store, params(windows=5, samples=7, window_ns=1, pace=Pace(1, 2)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    W, L, span = audit.pooled(r)[0], 649, 600
    assert N < W < 2 * N
    fill = N * L + (W - N) * 349
    assert ledger.memory(r, node(chip_blocks=0))[1].lost_blocks == pytest.approx(fill / span, rel=1e-12)


def test_a_pass_that_leaves_a_block_out_has_no_time_held_line(store) -> None:
    s, prover, passes, clock, _ = paced(store, params(windows=3, samples=N, window_ns=1, pace=Pace(1, 2)),
                                        plan=lambda t, i: partial(t, i) if t == 1 else None)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    snap, held, links = ledger.memory(s.record(), node())
    assert "read 15 of the 20" in held.refused and links.refused == held.refused and snap.refused is None


def test_declared_outputs_are_the_verdict_gamma_and_the_line(k1) -> None:
    r, _ = k1
    out = ledger.declared(r, node()).to_json()
    assert set(out) == {"accepted", "gamma", "memory", "windows"} and out["windows"] == 5
    assert set(out["gamma"]) == {"spec", "measured"} and set(out["memory"]) == set(ledger.FORMS)
    assert [w["window"] for w in out["gamma"]["spec"]["windows"]] == [0, 1, 2]
    assert all(math.isfinite(x["epsilon"]) for x in out["memory"].values())
    assert r.receipt == receipts.SKETCH_SHA512
