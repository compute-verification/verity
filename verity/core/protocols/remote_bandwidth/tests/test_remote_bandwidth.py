"""The remote bandwidth audit end to end on a small store: an honest session replayed from `timing.honest` is accepted
and recorded, and each way a session goes wrong is rejected with its failure."""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest

from verity.core.protocols.remote_bandwidth import audit, codec, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import Params, Session, SessionError

N = 20
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def payload(j: int) -> bytes:
    if j % 3 == 0:
        return bytes(codec.BLOCK_BYTES)
    return b"".join(hashlib.sha512(b"%d/%d" % (j, w)).digest()[:63] + b"\0" for w in range(codec.BLOCK_WORDS))


@pytest.fixture(scope="module")
def store() -> audit.Store:
    payloads = [payload(j) for j in range(N)]
    reg = audit.register(SOURCE, NODE, N, audit.payload_tree(NODE, payloads).commitment.root, nonce=bytes(32))
    return audit.Store(reg, payloads)


def params(**kw) -> Params:
    base = dict(blocks=N, window_ns=1000, rtt_allowance_ns=50, tail_ns=310, windows=3, samples=7, delta=1e-6)
    return Params(**{**base, **kw})


SCHEDULE = timing.honest(1000, 3, 300, gap_ns=10, down_ns=20, up_ns=30)


def nonces():
    count = iter(range(1 << 16))
    return lambda: next(count).to_bytes(32, "big")


def drive(store, p: Params, schedule: timing.Schedule, *, tiles=None, verifier_tiles=None, run=None):
    """Replay `schedule` against a session, the prover's passes computed as their roots arrive.  `run(prover, t, σ, i,
    previous)` replaces the honest pass."""
    tiles = tiles or audit.NoiseTiles(96)
    prover = audit.Prover(store, tiles)
    clock = Clock()
    s = Session(p, SOURCE, store.registration, verifier_tiles or tiles, clock=clock, nonce=nonces())
    events = sorted([(at, 1, t, i) for t, i, at in schedule.arrivals] + [(b, 0, t, 0) for t, b in enumerate(schedule.boundaries)])
    challenges, previous, passes = {0: s.start()}, store.registration.digest, {}
    for at, kind, t, i in events:
        clock.now = at
        if kind == 0:
            nxt = s.boundary()
            if nxt is not None:
                challenges[t + 1] = nxt
            continue
        ps = (run or (lambda pr, *a: pr.run(*a)))(prover, t, challenges[t], i, previous)
        passes[t, i] = ps
        previous = ps.root
        if s.receive(t, i, ps.root) is not None:
            break
    return s, prover, passes, clock


def finish(s: Session, prover: audit.Prover, passes, clock: Clock, open_=None):
    clock.now = max(clock.now, s.deadline(len(s.windows) - 1) + 1)
    out = []
    for t in range(len(s.windows)):
        draws = s.close(t)
        if isinstance(draws, audit.Verdict):
            return out + [draws]
        out.append(s.verify(t, [(open_ or prover.open)(passes[t, i], j) for i, j in draws]))
        if not out[-1].accepted:
            return out
    return out


def test_an_honest_session_is_accepted_and_recorded(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert r.public() == {"block_bytes": 4096, "blocks": N, "passes": [4, 3, 3], "protocol": audit.PROTOCOL_ID,
                          "registration": store.registration.digest.hex(), "span_ns": SCHEDULE.span_ns}
    w0 = r.profile()["windows"][0]
    assert w0["passes"] == 4 and w0["released_ns"] == 0 and w0["deadline_ns"] == 1000 + 50 + 310
    assert w0["profile"]["levels"] == [{"name": "pass", "query": None, "size": 4, "draw": None},
                                       {"name": "receipt", "query": None, "size": N, "draw": {"subset": 7}}]
    assert r.windows[1].released_ns == SCHEDULE.boundaries[0]


def test_the_ledger_counts_every_gamma_term(store) -> None:
    s, prover, passes, clock = drive(store, params(samples=1000), SCHEDULE)
    finish(s, prover, passes, clock)
    r = s.record()
    capacity = N * 4096 / 300e-9
    g = audit.ledger(r, capacity_bytes_per_s=capacity, chip_bytes=4096, host=0.002)
    whole = capacity * r.span_ns / 1e9
    assert g.timing == pytest.approx(1 - SCHEDULE.utilization)
    assert g.sample == 0                                  # every receipt opened
    assert g.chip == pytest.approx(r.passes * 4096 / whole)
    assert g.counting == pytest.approx((64 * r.passes * N + 41 * r.passes) / 8 / whole)
    assert g.gamma == pytest.approx(g.timing + g.chip + g.counting + 0.002)
    assert set(g.to_json()["terms"]) == set(audit.LEDGER_SOURCES)
    with pytest.raises(ValueError, match="understated"):
        audit.ledger(r, capacity_bytes_per_s=capacity / 2, chip_bytes=0)


def test_codec_round_trips_and_refuses_one_round(store) -> None:
    k = store.registration.key
    for j in (0, 1, N - 1):
        assert codec.decode(store.blocks[j], j, k) == store.payloads[j]
    assert store.blocks[0] != bytes(codec.BLOCK_BYTES)            # zero weights encode to a store that is not zeros
    assert codec.decode(b"\xff" * codec.BLOCK_BYTES, 0, k) is None
    with pytest.raises(ValueError, match="at least 2 rounds"):
        codec.key(bytes(32), 1)
    with pytest.raises(ValueError, match="at least 2 rounds"):
        params(rounds=1)


def test_the_codec_seed_is_drawn_after_and_bound_to_the_payload_root() -> None:
    root = hashlib.sha512(b"payloads").digest()
    a = audit.register(SOURCE, NODE, N, root, nonce=bytes(32))
    assert a == audit.register(SOURCE, NODE, N, root, nonce=bytes(32))
    others = [audit.register(SOURCE, NODE, N, hashlib.sha512(b"other").digest(), nonce=bytes(32)),
              audit.register(SOURCE, b"another node", N, root, nonce=bytes(32)),
              audit.register(SOURCE, NODE, N + 1, root, nonce=bytes(32)),
              audit.register(SOURCE, NODE, N, root, nonce=bytes([1]) * 32),
              audit.register(b"another verifier", NODE, N, root, nonce=bytes(32))]
    assert len({a.codec_seed, *(o.codec_seed for o in others)}) == 6
    assert len({a.digest, *(o.digest for o in others)}) == 6


def test_a_receipt_is_seed_first_and_binds_z_elementwise(store) -> None:
    seed = hashlib.sha512(b"a pass seed").digest()
    pre = receipts.prefix(seed)
    assert len(pre) == 128 and seed in pre[:128]
    z = bytes(range(96))
    y = receipts.Receipter(seed)(5, store.blocks[5], z)
    mixed = bytes(e ^ s for e, s in zip(store.blocks[5], (z * 43)[:4096]))
    assert y == hashlib.sha512(pre + (5).to_bytes(8, "big") + mixed).digest()
    for u in (0, 47, 95):
        flipped = z[:u] + bytes([z[u] ^ 1]) + z[u + 1:]
        assert receipts.Receipter(seed)(5, store.blocks[5], flipped) != y
    long = bytes(range(256)) * 20
    folded = receipts.spread(long)
    assert len(folded) == 4096 and folded[:1024] == bytes(a ^ b for a, b in zip(long[:1024], long[4096:5120]))


def test_seeds_chain_from_the_registration_and_across_windows(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert passes[0, 0].seed == receipts.seed(s.windows[0].challenge, 0, store.registration.digest)
    assert passes[0, 2].seed == receipts.seed(s.windows[0].challenge, 2, passes[0, 1].root)
    assert passes[1, 0].seed == receipts.seed(s.windows[1].challenge, 0, passes[0, 3].root)
    assert len({w.challenge for w in s.windows}) == 3


def test_unchained_passes_are_rejected(store) -> None:
    """A prover that computes every pass's receipts from the registration's digest, so one read of each block could
    serve every pass, fails on the first opening in a later pass."""
    def unchained(pr, t, challenge, i, previous):
        return pr.run(t, challenge, i, store.registration.digest)
    s, prover, passes, clock = drive(store, params(samples=1000), SCHEDULE, run=unchained)
    verdicts = finish(s, prover, passes, clock)
    assert verdicts[-1] == audit.Verdict(0, False, "receipt-path", (1, 0))


def test_a_root_after_the_deadline_rejects_the_session(store) -> None:
    late = dataclasses.replace(SCHEDULE, arrivals=tuple((t, i, at + 200 if (t, i) == (1, 2) else at)
                                                         for t, i, at in SCHEDULE.arrivals))
    s, *_ = drive(store, params(), late)
    assert s.failed == audit.Verdict(1, False, "late")
    with pytest.raises(SessionError, match="rejected"):
        s.boundary()


def test_the_honest_prover_is_late_without_a_tail_allowance() -> None:
    with pytest.raises(ValueError, match="after the deadline"):
        timing.honest(1000, 3, 300, gap_ns=10, down_ns=20, up_ns=30, tail_ns=0)


@pytest.mark.parametrize("window, index, failure", [(0, 1, "order"), (1, 0, "order")])
def test_roots_out_of_order_reject(store, window, index, failure) -> None:
    clock = Clock()
    s = Session(params(), SOURCE, store.registration, audit.NoiseTiles(96), clock=clock, nonce=nonces())
    s.start()
    assert s.receive(window, index, bytes(64)) == audit.Verdict(0, False, failure)


def test_a_window_with_too_few_passes_is_rejected(store) -> None:
    s, prover, passes, clock = drive(store, params(min_passes=4), SCHEDULE)
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, True), audit.Verdict(1, False, "few")]


def test_draws_wait_for_final_roots(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    s.windows[2].final = False
    clock.now = s.deadline(2)
    with pytest.raises(SessionError, match="not final"):
        s.close(2)
    with pytest.raises(SessionError, match="in order"):
        s.close(1)
    assert isinstance(s.close(0), tuple)                    # window 1's first root made window 0's final


def _tampered(field_: str, value):
    def open_(p, j, *, prover):
        return dataclasses.replace(prover.open(p, j), **{field_: value(prover.open(p, j))})
    return open_


@pytest.mark.parametrize("field_, value, failure", [
    ("leaf", lambda o: bytes([o.leaf[0] ^ 1]) + o.leaf[1:], "receipt-path"),
    ("receipt_path", lambda o: o.receipt_path[:-1], "receipt-path"),
    ("leaf", lambda o: o.leaf[:-64], "malformed"),
    ("block", lambda o: b"\xff" * codec.BLOCK_BYTES, "codec"),
    ("block", lambda o: bytes([o.block[0] ^ 1]) + o.block[1:], "payload-path"),
    ("payload_path", lambda o: tuple(reversed(o.payload_path)), "payload-path"),
    ("tile", lambda o: b"not PoUW's evidence", "tile"),
    ("block_index", lambda o: o.block_index + 1, "malformed"),
])
def test_a_bad_opening_rejects(store, field_, value, failure) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    open_ = _tampered(field_, value)
    verdicts = finish(s, prover, passes, clock, open_=lambda p, j: open_(p, j, prover=prover))
    assert verdicts[0].failure == failure and not verdicts[0].accepted
    with pytest.raises(SessionError):
        s.record()


def test_a_receipt_without_the_tiles_output_rejects(store) -> None:
    """Receipts made with a z PoUW's check does not recompute (here, zeros) open with good paths and fail the receipt."""
    class Zeros(audit.NoiseTiles):
        def output(self, seed, j):
            return bytes(self.z_bytes)
    s, prover, passes, clock = drive(store, params(), SCHEDULE, tiles=Zeros(96), verifier_tiles=audit.NoiseTiles(96))
    assert finish(s, prover, passes, clock)[0].failure == "receipt"


def test_openings_are_drawn_from_the_verifiers_secret(store) -> None:
    def draws(source: bytes):
        clock = Clock()
        s = Session(params(windows=1), source, store.registration, audit.NoiseTiles(96), clock=clock, nonce=nonces())
        s.start()
        s.receive(0, 0, bytes(64))
        s.boundary()
        clock.now = s.deadline(0) + 1
        return s.close(0)
    assert draws(SOURCE) == draws(SOURCE) != draws(b"another secret")


def test_opening_size_matches_the_timing_model(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    o = prover.open(passes[0, 0], 0)
    assert o.size == timing.opening_bytes(N)
    assert timing.wrong_bound(10 ** 6, timing.samples_for(1e-3, 2.0 ** -40), 2.0 ** -40) <= 10 ** 3


@pytest.mark.parametrize("rtt_ms, window_ms, pass_ms", [(1, 100, 20), (50, 100, 20), (20, 1000, 150), (50, 10000, 37)])
def test_honest_schedules_meet_their_deadlines_and_fill_the_window(rtt_ms, window_ms, pass_ms) -> None:
    ms = timing.MS
    one = timing.honest(window_ms * ms, 1, pass_ms * ms, gap_ns=100_000, down_ns=rtt_ms * ms // 2, up_ns=rtt_ms * ms // 2)
    many = timing.honest(window_ms * ms, 50, pass_ms * ms, gap_ns=100_000, down_ns=rtt_ms * ms // 2,
                         up_ns=rtt_ms * ms // 2)
    period = pass_ms * ms + 100_000
    assert one.passes[0] == math.ceil((window_ms * ms) / period)
    assert 1 - many.utilization < 1 - one.utilization
    assert 1 - many.utilization >= 100_000 / period            # the chaining gap stays
