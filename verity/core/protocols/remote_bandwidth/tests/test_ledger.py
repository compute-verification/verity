"""The declared outputs (`ledger`) and pacing (`audit.Pace`) on the small store: a paced session releases a live coin
every k passes and records each root's arrival, γ is `audit.ledger`'s at both R and, paced, per fixed window of the
stated T, and the memory line's forms are the Lean statements' conclusions read at the record, bursts and all, the
paced ones refused for an unpaced session, a record of one window or a pass that leaves a registered block out."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import pytest

from test_remote_bandwidth import A, N, SCHEDULE, Clock, Entropy, challenger, drive, finish, params, partial, store  # noqa: F401
from verity.core import claims
from verity.core.protocols.remote_bandwidth import audit, codec, ledger, line, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import AuditRun, Pace

PASS_NS, DOWN_NS, UP_NS = 300, 20, 30
B = codec.BLOCK_BYTES


def node(*, chip_blocks: int = 1, squarings: float = 0.0, disk: float = 0.0, link: float = 0.0,
         bursts: tuple[int, int, int] = (0, 0, 0)) -> ledger.Node:
    """A node sized to the small store: `timing`'s R, `chip_blocks` blocks on chip, the given ceilings, and bursts of
    the given blocks (rebuild, disk, link)."""
    return ledger.Node(timing.SPEC_BYTES_PER_S, timing.READ_BYTES_PER_S, chip_blocks * B,
                       {codec.FALLBACK.field.name: squarings}, disk_bytes_per_s=disk, link_bytes_per_s=link,
                       rebuild_burst_bytes=bursts[0] * B, disk_burst_bytes=bursts[1] * B,
                       link_burst_bytes=bursts[2] * B)


def paced(store, p: audit.Params, plan=None):
    """A paced session driven on its own schedule: each pass takes `PASS_NS` from its coin's arrival or the pass before,
    each message `DOWN_NS` to the prover and `UP_NS` back; the prover runs at most k passes under a coin, moves to the
    next coin once it has arrived, and stops at the close; the verifier sends each boundary message at
    `AuditRun.next_boundary_ns`, recomputed after every root.  Returns the session, prover, passes, clock and each
    root's arrival."""
    clock = Clock()
    s = AuditRun(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    prover = audit.Prover(store, None, entropy=Entropy(b"receipt salts"))
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
    """k = 1, five windows, the last the closing window, γ per fixed window of 1,000 ns, every receipt drawn: no
    receipt is wrong."""
    s, prover, passes, clock, arrivals = paced(store, params(windows=5, samples=N, window_ns=1, pace=Pace(1, 1000)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    return s.record(), arrivals


def test_a_paced_session_releases_a_coin_every_k_passes_and_records_arrivals(store, k1) -> None:
    """At k = 1 each window's boundary leaves once the window before has its root (window 0's at once), so the honest
    prover never waits on a coin, and each window holds one pass, its root's arrival on the verifier's clock."""
    r, arrivals = k1
    assert r.pace == Pace(1, 1000) and r.public()["pace"] == {"k": 1, "window_ns": 1000}
    assert [w.released_ns for w in r.windows] == [0, 1, 350, 650, 950]
    assert [w.arrived_ns for w in r.windows] == [(350,), (650,), (950,), (1250,), (1550,)]
    assert {(t, 0): a for t, (a,) in enumerate(w.arrived_ns for w in r.windows)} == arrivals
    assert [w.deadline_ns for w in r.windows] == [1 + A, 350 + A, 650 + A, 950 + A, 1250 + A]
    s, prover, passes, clock, _ = paced(store, params(windows=3, samples=N, window_ns=1, pace=Pace(3)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    assert all(1 <= w.passes <= 3 for w in s.record().windows)


def test_next_boundary_is_the_stated_window_unpaced_and_the_pace_when_paced(store) -> None:
    clock = Clock()
    p = params()
    s = AuditRun(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    assert s.next_boundary_ns() is None
    s.start()
    assert s.next_boundary_ns() == p.window_ns
    p = params(windows=3, window_ns=5, pace=Pace(2))
    s = AuditRun(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    coin = s.start()
    assert s.next_boundary_ns() == A, "not due before its first root: at the latest one allowance after its coin"
    prover = audit.Prover(store, None, entropy=Entropy(b"receipt salts"))
    clock.now = 100
    ps = prover.run(coin, 0, store.registration.digest, None)
    assert s.receive(0, 0, ps.root, step=0) is None
    assert s.next_boundary_ns() == 5, "due at k − 1 = 1 root: as soon as the stated window allows"
    for i in (1, 2):
        ps = prover.run(coin, i, ps.root, None)
        v = s.receive(0, i, ps.root, step=i)
    assert v == audit.Verdict(0, False, "order"), "a window's (k + 1)-th root is out of turn"


def test_a_pace_has_a_closing_window_and_a_record_of_one_window_has_no_time_held_line(k1) -> None:
    """A paced session has at least one window before its closing one; a record of one window, which no paced session
    makes, has the snapshot only, and γ per fixed window."""
    assert params(windows=2, pace=Pace(1)).pace == Pace(1, audit.WINDOW_NS)
    with pytest.raises(ValueError, match="closing window"):
        params(windows=1, pace=Pace(1))
    for bad in ((0,), (1, 0), (1, 1.5)):
        with pytest.raises(ValueError, match="positive"):
            Pace(*bad)
    r = k1[0]
    one = dataclasses.replace(r, windows=r.windows[:1])
    snap, held, links = ledger.memory(one, node())
    assert "one window" in held.refused and links.refused == held.refused and snap.refused is None
    assert [w.window for w in ledger.gamma(one, node(), timing.SPEC_BYTES_PER_S).windows] == [0]


def test_arrivals_are_one_a_pass_and_out_of_order_ones_have_no_time_held_line(k1) -> None:
    """A record takes any clock's arrivals, one integer a pass (the session's clock is whatever its verifier's is);
    the time-held line and γ per fixed window read them only in order from each window's challenge to its deadline,
    and a paced record without them has neither."""
    r = k1[0]
    w = r.windows[0]
    for bad in ((349, 350), (350.0,)):
        with pytest.raises(ValueError, match="arrivals"):
            dataclasses.replace(w, arrived_ns=bad)
    for bad in ((w.released_ns - 1,), (w.deadline_ns + 1,)):
        odd = dataclasses.replace(r, windows=(dataclasses.replace(w, arrived_ns=bad), *r.windows[1:]))
        assert "not in order" in ledger.memory(odd, node())[1].refused
        with pytest.raises(ValueError, match="not in order"):
            ledger.gamma(odd, node(), timing.SPEC_BYTES_PER_S)
    bare = dataclasses.replace(r, windows=tuple(dataclasses.replace(x, arrived_ns=()) for x in r.windows))
    assert "has none" in ledger.memory(bare, node())[1].refused
    with pytest.raises(ValueError, match="has none"):
        ledger.gamma(bare, node(), timing.SPEC_BYTES_PER_S)


def test_gamma_is_the_ledger_at_both_r_and_per_window(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r, n = s.record(), node(squarings=1e9)
    out = ledger.declared(r, n)
    sk, d = ledger.sketch(r, n), ledger.draw(r, n)
    assert 0 <= sk < 2.0 ** -40
    for g, cap in ((out.spec, n.spec_bytes_per_s), (out.measured, n.read_bytes_per_s)):
        want = audit.ledger(r, capacity_bytes_per_s=cap, chip_bytes=n.chip_bytes, rebuild=n.rebuild(r.codec))
        assert g.ledger == want and g.windows == want.windows and g.capacity_bytes_per_s == cap
        assert g.draw == d and (g.error, g.error_horizon) == (want.error + sk + d.total,
                                                              want.error_horizon + sk + d.total)
    assert out.to_json()["gamma"]["spec"]["gamma"] == out.spec.gamma
    assert out.to_json()["gamma"]["spec"]["error"]["draw"] == d.to_json()


def test_a_paced_session_reads_gamma_per_fixed_window(k1) -> None:
    """`k1` spans [0, 1620) (its last deadline, 1250 + A), so at T = 1,000 its fixed windows are [0, 1000) and
    [1000, 1620), whatever its coins' times.  A pass is in the one that holds it whole, from its challenge to its
    root: windows 0 to 2's ([0, 350], [1, 650], [350, 950]) in the first, while windows 3's and 4's ([650, 1250] and
    [950, 1550]) straddle 1,000 and are in neither, so the second credits nothing."""
    r, _ = k1
    n = node(chip_blocks=2, squarings=1e9)
    assert ledger.windows(r) == [(0, 1000), (1000, 1620)]
    g = ledger.gamma(r, n, n.spec_bytes_per_s)
    assert [x.window for x in g.windows] == [0, 1] and len(g.ledger.windows) == 5
    first, second = g.windows
    assert (first.span_ns, second.span_ns) == (1000, 620)
    whole = n.spec_bytes_per_s * 1000 / 1e9
    assert first.timing == pytest.approx(1 - 3 * N * B / whole, rel=1e-12)
    assert first.chip == pytest.approx(3 * n.chip_bytes / whole, rel=1e-12)
    assert first.counting == pytest.approx(3 * (audit.named_bits(N) + audit.SLACK_LOG2) / 8 / whole, rel=1e-12)
    assert (second.timing, second.chip, second.counting, second.sample, second.horizon) == (1, 0, 0, 0, 0)
    assert first.rebuild == second.rebuild == g.ledger.rebuild > 0
    assert g.ledger == audit.ledger(r, capacity_bytes_per_s=n.spec_bytes_per_s, chip_bytes=n.chip_bytes,
                                    rebuild=n.rebuild(r.codec)), "the session's γ credits every pass"


def test_the_snapshot_is_helds_conclusion(store) -> None:
    """`planned − W − (X·span + B) − P·S` blocks over the passes, X the rebuilds' and the disks' rate and B their
    bursts: here every pass reads the whole store, so ε is (W/P + (X·span + B)/P + S)/N."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    n = node(chip_blocks=2, squarings=2e9 * codec.FALLBACK.encode_squarings, disk=B * 1e9, bursts=(3, 5, 7))
    snap, held, links = ledger.memory(r, n)
    W, X, P = audit.pooled(r)[0], 3.0, r.passes
    assert snap.lost_blocks == pytest.approx(W / P + (X * r.span_ns + 3 + 5) / P + 2, rel=1e-12)
    assert snap.epsilon == min(1, snap.lost_blocks / N) and snap.held_bytes == (1 - snap.epsilon) * N * B
    draws = len(r.windows) * r.samples
    assert draws == sum(w.profile.drawn["receipt"] for w in r.windows) == 21, "each window holds more than 7 receipts"
    assert snap.delta == (r.windows[0].profile.delta + P * 2.0 ** -audit.SLACK_LOG2 + ledger.hiding(r)
                          + ledger.binding(r, n) + ledger.sketch(r, n) + draws * 2.0 ** -64)
    d = ledger.delta_parts(r, n)
    assert d == {"profile": r.windows[0].profile.delta, "slack": P * 2.0 ** -audit.SLACK_LOG2,
                 "hiding": ledger.hiding(r), "binding": ledger.binding(r, n), "sketch": ledger.sketch(r, n),
                 "rejection": draws / 2 ** 64}
    assert tuple(d) == ledger.DELTA_PARTS
    parts = d["profile"] + d["slack"] + d["hiding"] + d["binding"] + d["sketch"] + d["rejection"]
    assert snap.delta == parts and draws > 0
    assert Fraction(d["rejection"]) == Fraction(draws, 2 ** ledger.REJECTION_LOG2)
    assert "unpaced" in held.refused and links.refused == held.refused and held.epsilon is None


def test_the_coin_commitments_hiding_is_charged(store) -> None:
    """`ledger.hiding` is `LineHeld`'s hiding terms under hm96-sha512's pinned key: two distances a window, each
    within 2^-192, n·2^-191 over the record's n windows, which the line's δ carries (`PooledDraw` reads none)."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert len(r.windows) > 1 and ledger.hiding(r) == pytest.approx(len(r.windows) * 2.0 ** -191, rel=1e-12)
    assert ledger.hiding(r) == ledger.hiding_over(len(r.windows))


def test_the_lines_prf_advantages_are_named_by_a_registered_claim_and_in_no_part_of_delta() -> None:
    """`LineHeld`'s Σ_t Adv(prf t) has no numeric value, so the ledger names it by the claim that prices it
    (`PRF_CLAIM`, an assumption of the registry) and puts it in no part of δ."""
    assert claims.get(ledger.PRF_CLAIM).kind == "assumption" and ledger.PRF_CLAIM not in ledger.DELTA_PARTS


def test_the_draws_terms_are_charged_by_name(store) -> None:
    """`ledger.draw` is `PooledDraw`'s terms beyond δ over the record's windows: the rejection Σ_t k_t/2^64 at the
    record's `samples` and the receipts' binding term under `cr/sha-512`, Σ_t √(|L_t|·s)·q_t/2^256.5
    at the session's bounds, not the record's U_t and s'_t: |L_t| the most receipts a window can have (unpaced, fewer
    than 2^64 passes of the codec's `max_blocks`, since the session reserved append coins), s the draw cap, and
    q_t = 2·F·C + 2·s·69 for the fork finder that runs the prover twice; the PRF advantages are named with their claim
    and in no total.  No hiding term: the prover gets the coin commitment only at the session's end.  γ's errors carry
    the two; the line's δ carries the binding term beside its own."""
    s, prover, passes, clock = drive(store, params(samples=7), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r, n = s.record(), node(squarings=1e9)
    drawn = [w.profile.drawn["receipt"] for w in r.windows]
    assert len(drawn) > 1 and max(drawn) == 7 and r.samples == 7 and r.append_coins > 0 and r.pace is None
    d = ledger.draw(r, n)
    assert d.rejection == len(r.windows) * r.samples / 2.0 ** 64 and not hasattr(d, "hiding")
    fc = n.sketch_outside * ledger.compressions(r, n)
    L = (2 ** 64 - 1) * r.codec.max_blocks
    want = len(r.windows) * math.sqrt(L * 7) * (2 * fc + 2 * 7 * (64 + 5)) / 2 ** 256.5
    realized = sum(math.sqrt(w.receipts * s) * (2 * fc + 2 * s * (64 + 5)) for w, s in zip(r.windows, drawn)) / 2 ** 256.5
    assert d.binding == ledger.binding(r, n) == pytest.approx(want, rel=1e-12) and realized < d.binding < 2.0 ** -100
    paced = dataclasses.replace(r, pace=audit.Pace(1), append_coins=0)
    assert ledger.binding(paced, n) == pytest.approx(
        len(r.windows) * math.sqrt(r.blocks * 7) * (2 * fc + 2 * 7 * (64 + 5)) / 2 ** 256.5, rel=1e-12)
    with pytest.raises(ValueError, match="session's bounds"):
        ledger.binding(dataclasses.replace(r, samples=None), n)
    assert d.total == d.rejection + d.binding
    assert d.to_json() == {"binding": d.binding, "rejection": d.rejection,
                           "named": {"prf_advantage": {"claim": "prf/sha-512", "term": "Σ_t Adv(prf t)",
                                                       "value": None, "in_total": False}}}
    g = ledger.gamma(r, n, n.spec_bytes_per_s)
    assert g.draw == d and g.error == g.ledger.error + g.sketch + d.total


def test_delta_charges_the_draws_a_window_requested_where_it_opens_all_its_receipts(store) -> None:
    """`LineHeld`'s error is E = δs + δ + Σ_t k_t/2^64 + Σ_t hide_t + bind + Σ_t Adv(prf t), k_t the draws window t
    requested: a constant, since a window's U_t is the prover's.  A window of U_t ≤ k_t receipts opens them all, so the
    record's draws there are U_t, and a δ charging them would fall (k_t − U_t)/2^64 short of E less the advantages, far
    more than AuditRO's slack covers.  At k = 70, windows of 80, 60 and 60 receipts draw 70, 60 and 60: the ledger
    charges 3·70 at the record's `samples`, so D − (E − Σ Adv) is the slack P·2^-SLACK_LOG2, up to its five additions'
    rounding.  A record whose draws aren't min(samples, U_t) is refused, and one that states no samples has no δ."""
    s, prover, passes, clock = drive(store, params(samples=70), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r, n = s.record(), node(squarings=1e9)
    k = [r.samples] * len(r.windows)
    U, drawn = [w.receipts for w in r.windows], [w.profile.drawn["receipt"] for w in r.windows]
    assert k == [70] * 3 and U == [80, 60, 60] and drawn == [70, 60, 60] and r.public()["samples"] == 70
    deltas = {line.delta for line in ledger.memory(r, n)}
    assert len(deltas) == 1
    D = Fraction(deltas.pop())
    delta_s, delta = Fraction(ledger.sketch(r, n)), Fraction(r.windows[0].profile.delta)
    assert delta_s > 0 and Fraction(ledger.delta_parts(r, n)["rejection"]) == Fraction(sum(k), 2 ** 64)
    held = (delta_s + delta + Fraction(sum(k), 2 ** ledger.REJECTION_LOG2) + Fraction(ledger.hiding(r))
            + Fraction(ledger.binding(r, n)))
    slack, rounding = Fraction(r.passes, 2 ** audit.SLACK_LOG2), 5 * D / 2 ** 53
    assert D >= held and abs(D - held - slack) <= rounding < slack
    assert D - Fraction(sum(k) - sum(drawn), 2 ** ledger.REJECTION_LOG2) < held, "the draws made fall short of E"
    with pytest.raises(ValueError, match=r"window 0 drew 70 of its 80 receipts, not min\(60, 80\)"):
        dataclasses.replace(r, samples=60)
    with pytest.raises(ValueError, match="states no samples|a record no session made doesn't state"):
        ledger.memory(dataclasses.replace(r, samples=None), n)


def test_the_time_held_line_is_held_over_times_conclusion(k1) -> None:
    """On `k1`'s schedule: s = (0, 1, 350, 650, 950), the arrivals 300 later each, so e_0 = 350, L = 650 − 1 and
    Σ = 950 − 350; no receipt is wrong, so the line charges the chip and 2L·(X·(Σ + 2L) + B) over Σ, B the rebuild
    and disk bursts and, link-charged, the link window too."""
    r, _ = k1
    X = 1e-3
    n = node(chip_blocks=1, squarings=X * codec.FALLBACK.encode_squarings * 1e9, bursts=(2, 3, 0))
    snap, held, links = ledger.memory(r, n)
    L, span = 649, 600
    assert audit.pooled(r)[0] == 0
    assert held.lost_blocks == pytest.approx(1 + 2 * L * (X * (span + 2 * L) + 2 + 3) / span, rel=1e-12)
    assert links.lost_blocks == held.lost_blocks
    more = ledger.memory(r, node(chip_blocks=1, squarings=X * codec.FALLBACK.encode_squarings * 1e9, link=B * 1e6,
                                 bursts=(2, 3, 4)))[2]
    assert more.lost_blocks == pytest.approx(1 + 2 * L * ((X + 1e-3) * (span + 2 * L) + 2 + 3 + 4) / span, rel=1e-12)
    assert snap.lost_blocks == pytest.approx((X * r.span_ns + 2 + 3) / r.passes + 1, rel=1e-12)


def test_the_stated_node_has_a_burst_for_each_way_back() -> None:
    """`Node.stated`: a rebuild batch is at most S_chip, a disk read a second of the disks' ceiling, and a link window
    `timing.LINK_WINDOW_BYTES` when the links are charged; each may be stated instead."""
    n = ledger.Node.stated(disk_bytes_per_s=4.32e9, link_bytes_per_s=50e9)
    assert (n.rebuild_burst_bytes, n.disk_burst_bytes, n.link_burst_bytes) == (
        timing.CHIP_BYTES, 4.32e9, timing.LINK_WINDOW_BYTES)
    assert (ledger.Node.stated().disk_burst_bytes, ledger.Node.stated().link_burst_bytes) == (0, 0)
    m = ledger.Node.stated(disk_bytes_per_s=4.32e9, disk_burst_bytes=4 << 20, link_burst_bytes=0)
    assert (m.disk_burst_bytes, m.link_burst_bytes) == (4 << 20, 0) and m.to_json()["disk_burst_bytes"] == 4 << 20
    with pytest.raises(ValueError, match="link_burst_bytes"):
        dataclasses.replace(n, link_burst_bytes=-1.0)


def test_the_sketch_charges_the_outside_factor_the_node_states(store) -> None:
    """`ledger.sketch` charges Q = F·C, C the node's compressions over the span (`ledger.compressions`) and F its
    `sketch_outside`, `timing.SKETCH_OUTSIDE` (2^20) unless stated, which `to_json` records; F is a positive
    integer."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r, n = s.record(), node(squarings=1e9)
    assert n.sketch_outside == ledger.Node.stated().sketch_outside == timing.SKETCH_OUTSIDE == 2 ** 20
    assert n.to_json()["sketch_outside"] == 2 ** 20
    c = ledger.compressions(r, n)
    assert c == math.ceil(n.sha512_per_s * r.span_ns / 1e9)
    assert ledger.sketch(r, n) == float(timing.sketch_failure(2 ** 20 * c, r.passes, r.rows, blocks=r.blocks,
                                                              outside=1))
    assert ledger.Node.stated(sketch_outside=1).to_json()["sketch_outside"] == 1
    near = dataclasses.replace(n, sketch_outside=1)
    assert ledger.sketch(r, near) == float(timing.sketch_failure(c, r.passes, r.rows, blocks=r.blocks, outside=1))
    assert ledger.memory(r, n)[0].delta - ledger.memory(r, near)[0].delta == pytest.approx(
        ledger.sketch(r, n) - ledger.sketch(r, near))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError, match="sketch_outside"):
            dataclasses.replace(n, sketch_outside=bad)


def test_the_node_carries_a_write_ceiling_per_tier() -> None:
    """Beside its read ceilings a node has one GPU's and the host DRAM's write ceilings, `timing`'s in `Node.stated`
    and by default, each a positive finite rate, and no DPU's."""
    for n in (ledger.Node.stated(), ledger.Node.stated(reported_read_bytes_per_s=2e13), node()):
        assert (n.gpu_write_bytes_per_s, n.host_write_bytes_per_s) == (
            timing.GPU_WRITE_BYTES_PER_S, timing.HOST_WRITE_BYTES_PER_S) == (1.597e12, 819.2e9)
        assert (n.to_json()["gpu_write_bytes_per_s"], n.to_json()["host_write_bytes_per_s"]) == (1.597e12, 819.2e9)
    assert not [f.name for f in dataclasses.fields(ledger.Node) if "dpu" in f.name]
    for name in ("gpu_write_bytes_per_s", "host_write_bytes_per_s"):
        for bad in (0.0, -1.0, math.inf):
            with pytest.raises(ValueError, match=name):
                dataclasses.replace(ledger.Node.stated(), **{name: bad})


def test_wrong_receipts_fill_the_groups_at_their_rates(store) -> None:
    """With 7 of 20 receipts drawn a window, the pooled wrong receipts W are spread over the groups at
    v = (1, 349, 300, 300, L) a wrong read, highest first, at most N a group of one pass."""
    s, prover, passes, clock, _ = paced(store, params(windows=5, samples=7, window_ns=1, pace=Pace(1, 1000)))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    W, L, span = audit.pooled(r)[0], 649, 600
    assert N < W < 2 * N
    fill = N * L + (W - N) * 349
    assert ledger.memory(r, node(chip_blocks=0))[1].lost_blocks == pytest.approx(fill / span, rel=1e-12)


def test_the_paced_forms_are_the_lines_stretch_over_the_whole_session_in_floats(store) -> None:
    """The paced forms are `line`'s stretch held over time with the whole session one stretch, in floats
    (`line.session_lost`): within rounding of the exact line's stretch 0 at `per` = C once groups follow the closing
    one, and pinned to the bit as the declared outputs publish them, with no receipt wrong and with some."""
    n = node(chip_blocks=3, squarings=1.5e11, disk=4.32e9, link=5e10, bursts=(1, 2, 6))
    pins = {N: (8.168255770738538, 32.39970707097953, 88.1677192237573),
            7: (13.799709289395285, 50.24511225287638, 106.01312440565415)}
    for samples, pinned in pins.items():
        s, prover, passes, clock, _ = paced(store, params(windows=5, samples=samples, window_ns=1, pace=Pace(2, 1000)))
        assert all(v.accepted for v in finish(s, prover, passes, clock))
        r = s.record()
        out = ledger.memory(r, n)
        assert tuple(x.lost_blocks for x in out) == pinned
        rates = line.Rates(1.5e11, r.codec.encode_squarings, 4.32e9, 5e10, B, 2 * B, 6 * B)
        C, last = len(r.windows) - 1, r.windows[-1].arrived_ns[-1]
        gs = [line.Group(w.released_ns, w.arrived_ns[-1], w.passes, N, 0, w.passes * N, 0) for w in r.windows]
        gs += [line.Group(last, last, 1, N, 0, N, 0)] * C
        for f, x in zip(line.FORMS, out[1:]):
            exact = line.stretches(gs, per=C, blocks=N, block_bytes=B, chip_bytes=n.chip_bytes,
                                   wrong=audit.pooled(r)[0], charge=line.charges(rates, B)[f])
            assert [(y.lo, y.hi) for y in exact] == [(0, C)]
            assert x.lost_blocks == pytest.approx(float(exact[0].time.lost_blocks), rel=1e-12)


def test_a_pass_that_leaves_a_block_out_has_no_time_held_line(store) -> None:
    s, prover, passes, clock, _ = paced(store, params(windows=3, samples=N, window_ns=1, pace=Pace(1, 1000)),
                                        plan=lambda t, i: partial(t, i) if t == 1 else None)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    snap, held, links = ledger.memory(s.record(), node())
    assert "read 15 of the 20" in held.refused and links.refused == held.refused and snap.refused is None


def test_declared_outputs_are_the_verdict_gamma_and_the_line(k1) -> None:
    r, _ = k1
    out = ledger.declared(r, node()).to_json()
    assert set(out) == {"accepted", "gamma", "memory", "windows"} and out["windows"] == 5
    assert set(out["gamma"]) == {"spec", "measured"} and set(out["memory"]) == set(ledger.FORMS)
    assert [w["window"] for w in out["gamma"]["spec"]["windows"]] == [0, 1]
    assert all(math.isfinite(x["epsilon"]) for x in out["memory"].values())
    assert r.receipt == receipts.SKETCH_SHA512
