"""The remote bandwidth audit end to end on a small store: an honest session replayed from `timing.honest` is accepted
and recorded, and each way a session goes wrong is rejected with its failure."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import Params, Session, SessionError

N = 20
P = 4
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"
STREAM = timing.honest(12, 300, 300, gap_ns=10, down_ns=20, up_ns=30, jitter_ns=5)


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Entropy:
    """A deterministic stand-in for the operating system's generator: SHA-512 in counter mode under `label`."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(j: int) -> bytes:
    if j % 3 == 0:
        return bytes(codec.BLOCK_BYTES)
    return b"".join(hashlib.sha512(b"%d/%d" % (j, w)).digest()[:63] + b"\0" for w in range(codec.BLOCK_WORDS))


@pytest.fixture(scope="module")
def store() -> audit.Store:
    payloads = [payload(j) for j in range(N)]
    salts = [Entropy(b"payload salts")(receipts.SALT_BYTES * N)[receipts.SALT_BYTES * j:receipts.SALT_BYTES * (j + 1)]
             for j in range(N)]
    reg = audit.register(SOURCE, NODE, N, audit.payload_tree(NODE, payloads, salts).commitment.root, nonce=bytes(32))
    return audit.Store(reg, payloads, salts)


def params(**kw) -> Params:
    base = dict(blocks=N, passes=P, interval_ns=300, deadline_ns=STREAM.deadline_ns, windows=3, samples=7, delta=1e-6)
    return Params(**{**base, **kw})


def challenger(store, p: Params, label: bytes = b"coins") -> coins.Challenger:
    return coins.Challenger(store.registration.digest, 0, p.coins, entropy=Entropy(label))


def drive(store, p: Params, stream: timing.Stream, *, tiles=None, verifier_tiles=None, run=None, plan=None, sent=None,
          skip=(), label=b"coins"):
    """Replay `stream` against a session: each coin released on schedule, each pass computed as its root arrives.
    `plan(k)` is pass k's plan (None: the whole store); `run(prover, coin, plan)` replaces the honest pass, `sent(ps)`
    the runs sent with its root, and the roots of passes in `skip` are never sent."""
    tiles = tiles or audit.NoiseTiles(96)
    clock = Clock()
    s = Session(p, store.registration, verifier_tiles or tiles, challenger(store, p, label), clock=clock)
    prover = audit.Prover(store, tiles, s.commitment, entropy=Entropy(b"receipt salts"))
    events = sorted([(r, 0, k) for k, r in enumerate(stream.released)] + [(a, 1, k) for k, a in enumerate(stream.arrivals)])
    released, passes = {}, {}
    for at, kind, k in events:
        clock.now = at
        if kind == 0:
            released[k] = s.release()
            continue
        ps = (run or (lambda pr, coin, plan_: pr.run(coin, plan_)))(prover, released[k], plan and plan(k))
        passes[k] = ps
        if k not in skip and s.receive(k, ps.root, (sent or (lambda q: q.plan.runs))(ps)) is not None:
            break
    return s, prover, passes, clock


def finish(s: Session, prover: audit.Prover, passes, clock: Clock, open_=None):
    clock.now = max(clock.now, max(s.deadline(k) for k in range(len(s.released))) + 1)
    out = []
    for t in range(s.params.windows):
        draws = s.close(t)
        if isinstance(draws, audit.Verdict):
            return out + [draws]
        out.append(s.verify(t, [(open_ or prover.open)(passes[k], p) for k, p in draws]))
        if not out[-1].accepted:
            return out
    return out


def test_an_honest_session_is_accepted_and_recorded(store) -> None:
    s, prover, passes, clock = drive(store, params(), STREAM)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert r.public() == {"block_bytes": 4096, "blocks": N, "coins": s.commitment.root.hex(), "passes": [P] * 3,
                          "protocol": audit.PROTOCOL_ID, "receipts": [P * N] * 3,
                          "registration": store.registration.digest.hex(), "span_ns": STREAM.span_ns}
    w1 = r.profile()["windows"][1]
    assert w1["lengths"] == [N] * P and w1["released"] == [1200, 1500, 1800, 2100]
    assert w1["deadlines"] == [k + STREAM.deadline_ns for k in w1["released"]]
    assert w1["profile"]["levels"] == [{"name": "receipt", "query": None, "size": P * N, "draw": {"subset": 7}}]
    assert STREAM.deadline_ns == 20 + 30 + 2 * 5 + 300 + 10


def partial(k: int) -> receipts.Plan:
    """A pass that reads blocks 5 … 19, then 0 … 2 backwards, skipping 3 and 4, and one more block each later pass of
    its window."""
    return receipts.Plan.of([*range(5, N), 2, 1, 0][:15 + k % P], N)


def test_a_session_of_partial_plans_is_accepted_and_certifies_its_receipts(store) -> None:
    s, prover, passes, clock = drive(store, params(samples=1000), STREAM, plan=partial)
    assert passes[1].plan.runs == ((5, 15), (2, 1)) and passes[3].plan.runs == ((5, 15), (2, 1), (1, 1), (0, 1))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert [w.lengths for w in r.windows] == [(15, 16, 17, 18)] * 3 and r.public()["receipts"] == [66] * 3
    g = audit.ledger(r, capacity_bytes_per_s=N * 4096 / 300e-9, chip_bytes=0)
    assert g.timing == pytest.approx(1 - 198 * 4096 / (N * 4096 / 300e-9 * r.span_ns / 1e9))


def test_a_plan_maps_positions_to_blocks_and_is_canonical() -> None:
    plan = receipts.Plan(((7, 3), (0, 2), (12, 1)), N)
    assert list(plan) == [7, 8, 9, 0, 1, 12] and [plan.block(p) for p in range(6)] == list(plan)
    assert receipts.Plan.of(list(plan), N) == plan and plan.length == 6
    assert plan.encode() == bytes.fromhex("00000003" + "%016x%016x" * 3 % (7, 3, 0, 2, 12, 1))
    assert plan.digest == hashlib.sha512(receipts.PLAN_DOMAIN + plan.encode()).digest()
    assert receipts.Plan.whole(N).runs == ((0, N),)
    with pytest.raises(IndexError):
        plan.block(6)


def test_a_receipt_depends_on_its_block_not_its_position(store) -> None:
    """The plan's order is only the tree's: the same blocks listed in another order give each block the same receipt
    (z is the tile before the block's read), and a different root."""
    prover = audit.Prover(store, audit.NoiseTiles(96), challenger(store, params()).commitment, entropy=Entropy(b"s"))
    seed = receipts.seed(bytes(range(32)), 0)
    a = prover.pass_under(0, seed, receipts.Plan(((0, 10),), N))
    b = prover.pass_under(0, seed, receipts.Plan.of(list(reversed(range(10))), N))
    assert a.receipts == tuple(reversed(b.receipts)) and a.root != b.root


@pytest.mark.parametrize("runs", [
    ((0, 5), (3, 4)),            # block 3 twice
    ((10, 5), (0, 11)),          # block 10 twice, in a later run
    ((15, 6),),                  # past the store
    ((0, 0),),                   # an empty run
    ((0, 3), (3, 2)),            # one run written as two
    (),                          # no blocks
    ([0, 5],),                   # not a run
    ((0, True),),
])
def test_a_plan_that_repeats_a_block_or_is_not_canonical_rejects(store, runs) -> None:
    s = Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=Clock())
    s.release()
    assert s.receive(0, bytes(64), runs) == audit.Verdict(0, False, "plan")


def test_receipts_committed_under_another_plan_reject(store) -> None:
    """A prover whose trees commit every block but which sends a plan of the first 16 (one whole leaf, so every opened
    leaf has the size the sent plan gives it) fails on the receipt path: the tree's binding is the plan's digest."""
    s, prover, passes, clock = drive(store, params(), STREAM, sent=lambda ps: ((0, 16),))
    assert finish(s, prover, passes, clock)[0].failure == "receipt-path"


def test_the_ledger_counts_every_gamma_term(store) -> None:
    """At Δ = 300 ns and A = 370 ns consecutive passes' windows overlap by 70 ns with no trusted floor, and by none
    with a 70 ns one."""
    s, prover, passes, clock = drive(store, params(samples=1000), STREAM)
    finish(s, prover, passes, clock)
    r = s.record()
    capacity = N * 4096 / 300e-9
    g = audit.ledger(r, capacity_bytes_per_s=capacity, chip_bytes=4096, host=0.002)
    whole = capacity * r.span_ns / 1e9
    assert r.span_ns == 11 * 300 + 370
    assert g.timing == pytest.approx(1 - 12 * 300 / r.span_ns)
    assert g.overlap == pytest.approx(11 * 70 / r.span_ns)
    assert g.sample == 0                                  # every receipt opened
    assert g.chip == pytest.approx(r.passes * 4096 / whole)
    assert g.counting == pytest.approx((64 * r.receipts + 41 * r.passes) / 8 / whole) and r.receipts == r.passes * N
    assert g.gamma == pytest.approx(g.timing + g.overlap + g.chip + g.counting + 0.002)
    assert set(g.to_json()["terms"]) == set(audit.LEDGER_SOURCES)
    assert audit.ledger(r, capacity_bytes_per_s=capacity, chip_bytes=0, rtt_floor_ns=70).overlap == 0
    with pytest.raises(ValueError, match="understated"):
        audit.ledger(r, capacity_bytes_per_s=capacity / 2, chip_bytes=0)


def test_overlap_is_the_time_reads_may_serve_two_passes() -> None:
    """Σ|I_k| − |∪ I_k|: windows [0, 10], [4, 12] and [6, 9] are 21 ns long and cover 12, and a window no longer than
    the floor is empty."""
    assert audit.overlap_ns([0, 4, 6], [10, 12, 9], 0) == 10 + 8 + 3 - 12
    assert audit.overlap_ns([0, 4, 6], [10, 12, 9], 3) == (7 + 5 + 0) - 9
    assert audit.overlap_ns([0, 100], [50, 150], 0) == 0


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


def test_payload_leaves_hide_the_payload_under_their_salts(store) -> None:
    """Equal payloads (blocks 0 and 3 are zeros) commit to different leaves under their salts, as hm96-sha512/row/v2."""
    from verity.ml.commitments import hm96
    from verity.ml.commitments.rowleaf import sha512_row_v2_digest
    x = sha512_row_v2_digest(store.payloads[0], 8 * codec.BLOCK_BYTES)
    assert store.payloads[0] == store.payloads[3] and store.payload_tree.open(0).value != store.payload_tree.open(3).value
    assert store.payload_tree.open(0).value == hm96.SHA512.tree_leaf(
        hm96.SHA512.default_key, hm96.SHA512.commit_string(hm96.SHA512.default_key, x, store.salts[0]))
    assert audit.PAYLOAD_SCHEMA == receipts.SCHEMA == "hm96-sha512/row/v2"


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


def test_each_pass_is_seeded_by_its_own_live_coin(store) -> None:
    s, prover, passes, clock = drive(store, params(), STREAM)
    assert all(passes[k].seed == receipts.seed(s.coins[k], k) for k in range(12))
    assert len(set(s.coins)) == 12
    c = challenger(store, params())
    first = c.open(0)
    assert coins.coin_reason(c.commitment, first, 0) is None and first.coin == s.coins[0]
    assert coins.coin_reason(c.commitment, first, 1) == "not coin 1"
    forged = dataclasses.replace(first, coin=bytes([first.coin[0] ^ 1]) + first.coin[1:])
    assert coins.coin_reason(c.commitment, forged, 0) == "the coin does not open against the session's coin commitment"
    with pytest.raises(ValueError, match="does not open"):
        prover.run(forged)
    with pytest.raises(ValueError, match="already open"):
        c.open(0)


def test_a_pass_receipted_before_its_coin_is_rejected(store) -> None:
    """A prover that makes every pass's receipts under the session's first coin, so one read of each block could serve
    every pass, fails on the first opening in a later pass."""
    first = {}

    def early(pr, coin, plan):
        first.setdefault("coin", coin.coin)
        return pr.pass_under(coin.index, receipts.seed(first["coin"], coin.index), plan)
    s, prover, passes, clock = drive(store, params(samples=1000), STREAM, run=early)
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, False, "receipt-path", (1, 0))]


def test_a_root_after_its_deadline_rejects_the_session(store) -> None:
    late = dataclasses.replace(STREAM, arrivals=tuple(a + 200 if k == 6 else a for k, a in enumerate(STREAM.arrivals)))
    s, *_ = drive(store, params(), late)
    assert s.failed == audit.Verdict(1, False, "late")
    with pytest.raises(SessionError, match="rejected"):
        s.release()


def test_the_honest_prover_is_late_with_less_than_the_least_deadline() -> None:
    least = timing.deadline(300, gap_ns=10, down_ns=20, up_ns=30)
    st = timing.honest(12, 300, 300, gap_ns=10, down_ns=20, up_ns=30)
    assert max(a - r for r, a in zip(st.released, st.arrivals)) == least
    with pytest.raises(ValueError, match="after its deadline"):
        timing.honest(12, 300, 300, gap_ns=10, down_ns=20, up_ns=30, deadline_ns=least - 1)


def test_a_root_before_its_coin_or_twice_rejects(store) -> None:
    s = Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=Clock())
    s.release()
    assert s.receive(1, bytes(64)) == audit.Verdict(0, False, "early")
    s = Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=Clock())
    s.release()
    assert s.receive(0, bytes(64)) is None and s.receive(0, bytes(64)) == audit.Verdict(0, False, "duplicate")


def test_coins_are_released_on_schedule(store) -> None:
    clock = Clock()
    s = Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=clock)
    s.release()
    clock.now = 299
    with pytest.raises(SessionError, match="due at 300"):
        s.release()
    clock.now = 300
    assert s.release().index == 1


def test_a_missing_root_rejects_its_window(store) -> None:
    s, prover, passes, clock = drive(store, params(), STREAM, skip={5})
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, True), audit.Verdict(1, False, "missing")]


def test_draws_wait_for_the_windows_roots(store) -> None:
    s, prover, passes, clock = drive(store, params(), STREAM, skip={11})
    clock.now = s.deadline(11)
    assert isinstance(s.close(0), tuple)
    with pytest.raises(SessionError, match="not open to draw"):
        s.close(0)
    with pytest.raises(SessionError, match="not in before its last deadline"):
        s.close(2)
    clock.now += 1
    assert s.close(2) == audit.Verdict(2, False, "missing")


def _tampered(field_: str, value):
    def open_(ps, p, *, prover):
        return dataclasses.replace(prover.open(ps, p), **{field_: value(prover.open(ps, p))})
    return open_


@pytest.mark.parametrize("field_, value, failure", [
    ("leaf", lambda o: bytes([o.leaf[0] ^ 1]) + o.leaf[1:], "receipt-path"),
    ("leaf_salt", lambda o: bytes([o.leaf_salt[0] ^ 1]) + o.leaf_salt[1:], "receipt-path"),
    ("receipt_path", lambda o: o.receipt_path[:-1], "receipt-path"),
    ("leaf", lambda o: o.leaf[:-64], "malformed"),
    ("leaf_salt", lambda o: o.leaf_salt[:-1], "malformed"),
    ("block", lambda o: b"\xff" * codec.BLOCK_BYTES, "codec"),
    ("block", lambda o: bytes([o.block[0] ^ 1]) + o.block[1:], "payload-path"),
    ("payload_salt", lambda o: bytes([o.payload_salt[0] ^ 1]) + o.payload_salt[1:], "payload-path"),
    ("payload_path", lambda o: tuple(reversed(o.payload_path)), "payload-path"),
    ("tile", lambda o: b"not PoUW's evidence", "tile"),
    ("position", lambda o: o.position + 1, "malformed"),
])
def test_a_bad_opening_rejects(store, field_, value, failure) -> None:
    s, prover, passes, clock = drive(store, params(), STREAM, plan=partial)
    open_ = _tampered(field_, value)
    verdicts = finish(s, prover, passes, clock, open_=lambda ps, p: open_(ps, p, prover=prover))
    assert verdicts[0].failure == failure and not verdicts[0].accepted
    with pytest.raises(SessionError):
        s.record()


def test_a_receipt_without_the_tiles_output_rejects(store) -> None:
    """Receipts made with a z PoUW's check does not recompute (here, zeros) open with good paths and fail the receipt."""
    class Zeros(audit.NoiseTiles):
        def output(self, seed, p):
            return bytes(self.z_bytes)
    s, prover, passes, clock = drive(store, params(), STREAM, tiles=Zeros(96), verifier_tiles=audit.NoiseTiles(96))
    assert finish(s, prover, passes, clock)[0].failure == "receipt"


def test_openings_are_drawn_from_the_windows_live_coin(store) -> None:
    def draws(label: bytes):
        s, *_ = drive(store, params(), STREAM, label=label)
        out = s.close(0)
        with pytest.raises(ValueError, match="already open"):
            s.challenger.open(3 * P)                       # window 0's openings coin, after the pass coins
        return out
    assert draws(b"coins") == draws(b"coins") != draws(b"other coins")


def test_opening_size_matches_the_timing_model(store) -> None:
    s, prover, passes, clock = drive(store, params(), STREAM)
    assert prover.open(passes[0], 0).size == timing.opening_bytes(N)
    assert timing.wrong_bound(10 ** 6, timing.samples_for(1e-3, 2.0 ** -40), 2.0 ** -40) <= 10 ** 3


@pytest.mark.parametrize("pass_ms, rtt_ms, jitter_ms, floor", [(20, 1, 0.05, 0.5), (20, 0.2, 0, 0), (50, 5, 0.5, 0.9),
                                                               (37, 20, 1, 0.5)])
def test_honest_streams_meet_their_deadlines_and_cost_what_the_model_says(pass_ms, rtt_ms, jitter_ms, floor) -> None:
    """At Δ* consecutive passes' windows just don't overlap and μ is (g + U)/(p + g + U), the session's ends aside; at
    Δ = p the overlap is charged and μ is (g + U)/p.  The stream doesn't stop at window boundaries."""
    ms, gap = timing.MS, 50_000
    kw = dict(gap_ns=gap, rtt_ns=round(rtt_ms * ms), jitter_ns=round(jitter_ms * ms), rtt_floor_ns=round(floor * rtt_ms * ms))
    u = kw["rtt_ns"] + 2 * kw["jitter_ns"] - kw["rtt_floor_ns"]
    star = timing.best_interval(pass_ms * ms, **kw)
    assert timing.mu(star, pass_ms * ms, **kw) == pytest.approx((gap + u) / (pass_ms * ms + gap + u))
    assert timing.mu(pass_ms * ms, pass_ms * ms, **kw) == pytest.approx((gap + u) / (pass_ms * ms))
    streams = {delta: timing.honest(400, delta, pass_ms * ms, gap_ns=gap, down_ns=kw["rtt_ns"] // 2,
                                    up_ns=kw["rtt_ns"] - kw["rtt_ns"] // 2, jitter_ns=kw["jitter_ns"])
               for delta in (star, pass_ms * ms)}
    for delta, st in streams.items():
        ends = 3 * st.deadline_ns / st.span_ns
        assert st.mu(kw["rtt_floor_ns"]) == pytest.approx(timing.mu(delta, pass_ms * ms, **kw), abs=ends)
    assert audit.overlap_ns(streams[star].released, streams[star].deadlines, kw["rtt_floor_ns"]) == 0
    st = streams[pass_ms * ms]
    assert audit.overlap_ns(st.released, st.deadlines, kw["rtt_floor_ns"]) == 399 * (st.deadline_ns - kw["rtt_floor_ns"]
                                                                                     - pass_ms * ms)
