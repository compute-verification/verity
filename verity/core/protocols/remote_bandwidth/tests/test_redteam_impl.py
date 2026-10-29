"""The red team's exploits against the remote bandwidth audit as implemented: each test asserts the property the
audit's claim needs, and was marked xfail with its finding id while the reference broke it; RB-1, RB-2 and RB-3 are
fixed and pass.  The exploits run on `test_remote_bandwidth`'s store and drivers."""

from __future__ import annotations

import pytest

from remote_bandwidth_fakes import Clock, Entropy
from test_remote_bandwidth import N, SCHEDULE, drive, finish, params, store  # noqa: F401
from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import Session

BLOCK = 4096
CAPACITY = N * BLOCK / 300e-9    # a pass of the store fills its 300 ns at the node's capacity


def window_shortfall(w: audit.WindowRecord) -> float:
    """1 − the window's receipts over its capacity across [its challenge's release, its deadline]."""
    return 1 - w.receipts * BLOCK / (CAPACITY * (w.deadline_ns - w.released_ns) / 1e9)


def test_rb1_the_ledger_bounds_every_window_s_shortfall(store) -> None:  # noqa: F811
    """The claim is the hybrid: short by under 0.15% over the horizon and by under 1% in every window.  `Ledger.floor` is
    the every-window half, but it is only the openings' bound on a window's wrong receipts.  A prover that reads one
    block a pass through window 5 (every receipt correct, every one opened) is accepted with floor 0; the window's
    certified reads are 3.3% of its capacity, and the session's γ moves by about 1/W of that, 8% at W = 12 (0.17% at
    600 windows of 1 s).  Fixed: the ledger reads each window over its own span, and names window 5 the worst."""
    windows, idle = 12, 5
    schedule = timing.honest(1000, windows, 300, gap_ns=10, down_ns=20, up_ns=30, jitter_ns=5)
    p = params(windows=windows, samples=N * 4)
    s, prover, passes, clock = drive(store, p, schedule,
                                     plan=lambda t, i: receipts.Plan(((0, 1),), N) if t == idle else None)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    led = audit.ledger(r, capacity_bytes_per_s=CAPACITY, chip_bytes=0, rebuild=audit.Rebuild(r.codec, 0.0))
    short = [window_shortfall(w) for w in r.windows]
    assert short[idle] > 0.96 and max(x for t, x in enumerate(short) if t != idle) < 0.35
    assert led.floor == 0
    per_window = led.to_json().get("windows")
    assert per_window is not None and per_window[idle]["gamma"] >= short[idle]
    assert led.worst.window == led.worst_horizon.window == led.to_json()["worst"]["window"] == idle


def test_rb2_the_record_names_its_codec(store) -> None:  # noqa: F811
    """`Record.public()` repeats the receipt kind and `bind` "since what its receipts rest on depends on them", and
    `ledger` checks `link` against `bind`.  μ_rebuild rests on the codec (field, k, mask), and the record omitted it: a
    session under `p512/k2`, unmasked and resting on the unmeasured assumption 7, had a public record of the same shape
    as one under the stated `p1024/k3/masked`, and `ledger` takes any `rebuild` for either.  Fixed: the record names
    the codec and its parameters, and `ledger` prices μ_rebuild from a model of the record's codec, refusing
    another's."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    assert store.registration.codec.name == "p1024/k2/masked"
    assert store.registration.codec.name in s.record().public().values()
    assert s.record().public()["codec_params"] == {"c": 105, "field": "p1024", "masked": True, "n": 1024, "rounds": 2}
    stated = timing.SQUARINGS_PER_S["p1024"]
    with pytest.raises(ValueError, match="record's codec p1024/k2/masked"):
        audit.ledger(s.record(), capacity_bytes_per_s=CAPACITY, chip_bytes=0,
                     rebuild=audit.Rebuild(codec.DEFAULT, stated))
    with pytest.raises(ValueError, match="not a fraction"):
        audit.ledger(s.record(), capacity_bytes_per_s=CAPACITY, chip_bytes=0, rebuild=0.0)
    led = audit.ledger(s.record(), capacity_bytes_per_s=CAPACITY, chip_bytes=0,
                       rebuild=audit.Rebuild(codec.FALLBACK, stated))
    assert led.rebuild == pytest.approx(stated * BLOCK / (65_344 * CAPACITY))


def joint_pass(store: audit.Store, window: int, i: int, seeds: tuple[bytes, bytes], reads: list[int],
               entropy) -> tuple[audit.Pass, audit.Pass]:
    """Pass i of two sessions' chains at once: each block is read once and serves a receipt of each."""
    plan = receipts.Plan.whole(N)
    rs = [receipts.receipter(store.registration.receipt, sd) for sd in seeds]
    ys: tuple[list[bytes], list[bytes]] = ([], [])
    for j in plan:
        block = store.blocks[j]
        reads[0] += 1
        for k in (0, 1):
            ys[k].append(rs[k](j, block))
    out = []
    for k in (0, 1):
        salts = tuple(entropy(receipts.SALT_BYTES) for _ in range(receipts.leaves(N)))
        out.append(audit.Pass(window, i, seeds[k], plan, tuple(ys[k]), salts,
                              receipts.tree(seeds[k], plan, ys[k], salts)))
    return out[0], out[1]


def test_rb3_a_session_refuses_to_start_while_another_reads_the_store(store) -> None:  # noqa: F811
    """`Session(not_before_ns=…)` is the only thing that keeps two sessions over one store from sharing reads
    (`PROTOCOL.md` §5), and it was optional: left out, it failed open.  Two sessions run at once over one clock, both
    told they may start at 0, as only a verifier that runs them at once would; each block is read once a pass and
    serves a receipt of each chain.  Both are accepted with every receipt opened, and the two records together certify
    twice the reads made.  Fixed: every session names its `not_before_ns`, and one that names the last record's end
    waits it out."""
    p = params(windows=2, samples=N * 3, window_ns=910)
    clock = Clock()
    sessions = [Session(p, store.registration, None,
                        coins.local(store.registration.digest, k, p.windows, Entropy(b"coins %d" % k), appends=1),
                        not_before_ns=0, clock=clock) for k in (0, 1)]
    provers = [audit.Prover(store, None, s.commitment, entropy=Entropy(b"unused")) for s in sessions]
    reads, entropy = [0], Entropy(b"joint salts")
    challenges = [s.start() for s in sessions]
    previous = [store.registration.digest] * 2
    passes: list[dict] = [{}, {}]
    for t in range(2):
        for i in range(3):
            clock.now += 300
            seeds = tuple(receipts.seed(coins.coin(challenges[k]), i, previous[k]) for k in (0, 1))
            pair = joint_pass(store, t, i, seeds, reads, entropy)
            for k in (0, 1):
                passes[k][t, i], previous[k] = pair[k], pair[k].root
                assert sessions[k].receive(t, i, pair[k].root, pair[k].plan.runs, step=3 * t + i) is None
        clock.now += 10
        challenges = [s.boundary() for s in sessions]
    for k in (0, 1):
        verdicts = finish(sessions[k], provers[k], passes[k], clock)
        assert [v.accepted for v in verdicts] == [True, True]
    credited = sum(s.record().receipts for s in sessions)
    assert credited == 2 * reads[0] == 2 * 6 * N
    with pytest.raises((TypeError, ValueError)):
        Session(p, store.registration, None,
                coins.local(store.registration.digest, 2, p.windows, Entropy(b"coins 2"), appends=1), clock=clock)
    end = max(s.record().end_ns for s in sessions)
    nxt = Session(p, store.registration, None,
                  coins.local(store.registration.digest, 3, p.windows, Entropy(b"coins 3"), appends=1),
                  not_before_ns=end, clock=clock)
    clock.now = end - 1
    with pytest.raises(audit.SessionError, match="may still be reading"):
        nxt.start()
