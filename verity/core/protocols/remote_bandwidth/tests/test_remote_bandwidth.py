"""The remote bandwidth audit end to end on a small store: an honest session replayed from `timing.honest` is accepted
and recorded, and each way a session goes wrong is rejected with its failure."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import Params, Session, SessionError

N = 20
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"
SCHEDULE = timing.honest(1000, 3, 300, gap_ns=10, down_ns=20, up_ns=30, jitter_ns=5)
A = 20 + 30 + 2 * 5 + 300 + 10


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
    base = dict(blocks=N, deadline_ns=A, windows=3, samples=7, delta=1e-6)
    return Params(**{**base, **kw})


def challenger(store, p: Params, label: bytes = b"coins") -> coins.Challenger:
    return coins.Challenger(store.registration.digest, 0, p.coins, entropy=Entropy(label))


def honest(prover, t, i, challenges, previous, plan):
    return prover.run(challenges[t], i, previous, plan)


def drive(store, p: Params, schedule: timing.Schedule, *, tiles=None, verifier_tiles=None, run=honest, plan=None,
          sent=None, label=b"coins"):
    """Replay `schedule` against a session: `σ_0` released at 0 and each boundary message on time, each pass computed
    as its root arrives, chained from the pass before.  `plan(t, i)` is pass (t, i)'s plan (None: the whole store);
    `run(prover, t, i, challenges, previous, plan)` replaces the honest pass and `sent(ps)` the runs sent with its
    root."""
    tiles = tiles or audit.NoiseTiles(96)
    clock = Clock()
    s = Session(p, store.registration, verifier_tiles or tiles, challenger(store, p, label), clock=clock)
    prover = audit.Prover(store, tiles, s.commitment, entropy=Entropy(b"receipt salts"))
    events = sorted([(0, 0, -1)] + [(b, 0, t) for t, b in enumerate(schedule.boundaries)]
                    + [(a, 1, (t, i)) for t, i, a in schedule.arrivals])
    challenges, passes, previous = {}, {}, store.registration.digest
    for at, kind, x in events:
        clock.now = at
        if kind == 0:
            coin = s.start() if x < 0 else s.boundary()
            if coin is not None:
                challenges[x + 1] = coin
            continue
        t, i = x
        ps = run(prover, t, i, challenges, previous, plan and plan(t, i))
        passes[t, i], previous = ps, ps.root
        if s.receive(t, i, ps.root, (sent or (lambda q: q.plan.runs))(ps)) is not None:
            break
    return s, prover, passes, clock


def finish(s: Session, prover: audit.Prover, passes, clock: Clock, open_=None):
    clock.now = max(clock.now, s.deadline(len(s.windows) - 1) + 1)
    out = []
    for t in range(s.params.windows):
        draws = s.close(t)
        if isinstance(draws, audit.Verdict):
            return out + [draws]
        out.append(s.verify(t, [(open_ or prover.open)(passes[t, i], p) for i, p in draws]))
        if not out[-1].accepted:
            return out
    return out


def test_an_honest_session_is_accepted_and_recorded(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert SCHEDULE.passes == (4, 3, 3) and SCHEDULE.deadlines == (1000 + A, 2000 + A, 3000 + A)
    assert r.public() == {"block_bytes": 4096, "blocks": N, "coins": s.commitment.root.hex(), "passes": [4, 3, 3],
                          "protocol": audit.PROTOCOL_ID, "receipts": [4 * N, 3 * N, 3 * N],
                          "registration": store.registration.digest.hex(), "span_ns": 3000 + A}
    w1 = r.profile()["windows"][1]
    assert w1["lengths"] == [N] * 3 and w1["released_ns"] == 1000 and w1["deadline_ns"] == 2000 + A
    assert w1["opened_ns"] == 3000 + A + 1
    assert w1["profile"]["levels"] == [{"name": "receipt", "query": None, "size": 3 * N, "draw": {"subset": 7}}]
    assert r.end_ns == SCHEDULE.span_ns and A == timing.deadline(300, gap_ns=10, down_ns=20, up_ns=30, jitter_ns=5)


def partial(t: int, i: int) -> receipts.Plan:
    """A pass that reads blocks 5 … 19, then 0 … 2 backwards, skipping 3 and 4, and one more block each later pass of
    its window."""
    return receipts.Plan.of([*range(5, N), 2, 1, 0][:15 + i], N)


def test_a_session_of_partial_plans_is_accepted_and_certifies_its_receipts(store) -> None:
    s, prover, passes, clock = drive(store, params(samples=1000), SCHEDULE, plan=partial)
    assert passes[0, 1].plan.runs == ((5, 15), (2, 1)) and passes[0, 3].plan.runs == ((5, 15), (2, 1), (1, 1), (0, 1))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert [w.lengths for w in r.windows] == [(15, 16, 17, 18), (15, 16, 17), (15, 16, 17)]
    assert r.public()["receipts"] == [66, 48, 48]
    g = audit.ledger(r, capacity_bytes_per_s=N * 4096 / 300e-9, chip_bytes=0)
    assert g.timing == pytest.approx(1 - 162 * 4096 / (N * 4096 / 300e-9 * r.span_ns / 1e9))


def test_plans_sent_as_arrays_are_received_as_their_runs_are(store) -> None:
    """The verifier takes a plan as the store holds it, an (R, 2) array, and checks it in numpy (`Plan.from_array`)."""
    s, prover, passes, clock = drive(store, params(samples=1000), SCHEDULE, plan=partial,
                                     sent=lambda ps: np.array(ps.plan.runs, np.int64))
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    assert [w.lengths for w in s.record().windows] == [(15, 16, 17, 18), (15, 16, 17), (15, 16, 17)]


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
    seed = receipts.seed(bytes(range(32)), 0, store.registration.digest)
    a = prover.pass_under(0, 0, seed, receipts.Plan(((0, 10),), N))
    b = prover.pass_under(0, 0, seed, receipts.Plan.of(list(reversed(range(10))), N))
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
    np.array([[0, 5], [3, 4]]),  # as arrays, `Plan.from_array`
    np.array([[0, 3], [3, 2]]),
    np.zeros((0, 2), np.int64),
    np.array([[0.0, 5.0]]),
])
def test_a_plan_that_repeats_a_block_or_is_not_canonical_rejects(store, runs) -> None:
    s = Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=Clock())
    s.start()
    assert s.receive(0, 0, bytes(64), runs) == audit.Verdict(0, False, "plan")


def test_receipts_committed_under_another_plan_reject(store) -> None:
    """A prover whose trees commit every block but which sends a plan of the first 16 (one whole leaf, so every opened
    leaf has the size the sent plan gives it) fails on the receipt path: the tree's binding is the plan's digest."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE, sent=lambda ps: ((0, 16),))
    assert finish(s, prover, passes, clock)[0].failure == "receipt-path"


def test_the_ledger_counts_every_gamma_term(store) -> None:
    """A chained session's span is its passes, a gap g after each, and its two ends: `σ_0`'s trip down before the first
    pass, and after the last boundary the pass in flight's slack and its root's trip up."""
    s, prover, passes, clock = drive(store, params(samples=1000), SCHEDULE)
    finish(s, prover, passes, clock)
    r = s.record()
    capacity = N * 4096 / 300e-9
    g = audit.ledger(r, capacity_bytes_per_s=capacity, chip_bytes=4096, host=0.002)
    whole = capacity * r.span_ns / 1e9
    assert r.span_ns == 3000 + A and r.passes == 10
    assert g.timing == pytest.approx(1 - 10 * 300 / r.span_ns)
    assert g.sample == 0                                  # every receipt opened
    assert g.chip == pytest.approx(r.passes * 4096 / whole)
    assert g.counting == pytest.approx((64 * r.receipts + 41 * r.passes) / 8 / whole) and r.receipts == r.passes * N
    assert g.gamma == pytest.approx(g.timing + g.chip + g.counting + 0.002)
    assert set(g.to_json()["terms"]) == set(audit.LEDGER_SOURCES) and "overlap" not in audit.LEDGER_SOURCES
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


def test_the_seed_is_the_windows_challenge_the_pass_index_and_the_root_before() -> None:
    sigma, root = bytes(range(32)), hashlib.sha512(b"a root").digest()
    assert receipts.seed(sigma, 3, root) == hashlib.sha512(receipts.SEED_DOMAIN + sigma + bytes([0, 0, 0, 3]) + root).digest()
    assert len({receipts.seed(sigma, 3, root), receipts.seed(sigma, 4, root), receipts.seed(bytes(32), 3, root),
                receipts.seed(sigma, 3, bytes(64))}) == 4
    for bad in ((sigma[:31], 0, root), (sigma, -1, root), (sigma, 1 << 32, root), (sigma, 0, root[:63])):
        with pytest.raises(ValueError):
            receipts.seed(*bad)


def test_each_pass_is_seeded_by_the_chain_from_its_windows_coin(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    order = sorted(passes)
    previous = [store.registration.digest] + [passes[k].root for k in order[:-1]]
    assert all(passes[t, i].seed == receipts.seed(s.windows[t].challenge, i, b) for (t, i), b in zip(order, previous))
    assert len({w.challenge for w in s.windows}) == 3
    c = challenger(store, params())
    first = c.open(0)
    assert coins.coin_reason(c.commitment, first, 0) is None and first.coin == s.windows[0].challenge
    assert coins.coin_reason(c.commitment, first, 1) == "not coin 1"
    forged = dataclasses.replace(first, coin=bytes([first.coin[0] ^ 1]) + first.coin[1:])
    assert coins.coin_reason(c.commitment, forged, 0) == "the coin does not open against the session's coin commitment"
    with pytest.raises(ValueError, match="does not open"):
        prover.run(forged, 0, store.registration.digest)
    with pytest.raises(ValueError, match="openings coin"):
        prover.run(c.open(3), 0, store.registration.digest)
    with pytest.raises(ValueError, match="already open"):
        c.open(0)


def unchained(prover, t, i, challenges, previous, plan):
    """Every pass seeded after the registration, so a window's passes could all run at once off one read."""
    return prover.pass_under(t, i, receipts.seed(challenges[t].coin, i, prover.store.registration.digest), plan)


def detached(prover, t, i, challenges, previous, plan):
    """Window 1's chain started over from the registration, so it need not wait on window 0's last root."""
    return prover.run(challenges[t], i, prover.store.registration.digest if (t, i) == (1, 0) else previous, plan)


def early(prover, t, i, challenges, previous, plan):
    """Window 1's passes under window 0's challenge, so they could start before window 1's challenge is out."""
    return prover.pass_under(t, i, receipts.seed(challenges[0 if t == 1 else t].coin, i, previous), plan)


@pytest.mark.parametrize("run, verdicts", [
    (unchained, [audit.Verdict(0, False, "receipt-path", (1, 0))]),
    (detached, [audit.Verdict(0, True), audit.Verdict(1, False, "receipt-path", (0, 0))]),
    (early, [audit.Verdict(0, True), audit.Verdict(1, False, "receipt-path", (0, 0))]),
])
def test_passes_off_the_chain_are_rejected(store, run, verdicts) -> None:
    """Each opens with good paths under the seeds it used, and fails the first opening whose seed is not the chain's."""
    s, prover, passes, clock = drive(store, params(samples=1000), SCHEDULE, run=run)
    assert finish(s, prover, passes, clock) == verdicts


def test_a_root_after_its_deadline_rejects_the_session(store) -> None:
    late = dataclasses.replace(SCHEDULE, arrivals=tuple((t, i, a + 100 if (t, i) == (0, 3) else a)
                                                        for t, i, a in SCHEDULE.arrivals))
    assert SCHEDULE.arrivals[3] == (0, 3, 1293) and 1293 + 100 > SCHEDULE.deadlines[0]
    s, *_ = drive(store, params(), late)
    assert s.failed == audit.Verdict(0, False, "late")
    with pytest.raises(SessionError, match="rejected"):
        s.boundary()


def test_the_honest_prover_is_never_late_and_the_allowance_is_tight() -> None:
    """With T − 1 a multiple of p + g, window 0's last pass starts the nanosecond before the boundary message arrives,
    so its root is one nanosecond short of `A` after the boundary; an allowance a nanosecond less makes it late."""
    a = timing.deadline(300, gap_ns=10, down_ns=20, up_ns=30)
    st = timing.honest(4 * 310 + 1, 3, 300, gap_ns=10, down_ns=20, up_ns=30)
    assert max(at - st.boundaries[t] for t, _, at in st.arrivals) == a - 1
    assert timing.honest(4 * 310 + 1, 3, 300, gap_ns=10, down_ns=20, up_ns=30, deadline_ns=a - 1).passes == st.passes
    with pytest.raises(ValueError, match="after the deadline"):
        timing.honest(4 * 310 + 1, 3, 300, gap_ns=10, down_ns=20, up_ns=30, deadline_ns=a - 2)


def test_roots_out_of_order_reject(store) -> None:
    def session() -> Session:
        return Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=Clock())
    s = session()
    with pytest.raises(SessionError, match="not started"):
        s.receive(0, 0, bytes(64))
    s.start()
    assert s.receive(0, 1, bytes(64)) == audit.Verdict(0, False, "order")
    s = session()
    s.start()
    assert s.receive(0, 0, bytes(64)) is None and s.receive(0, 0, bytes(64)) == audit.Verdict(0, False, "order")
    s = session()
    s.start()
    assert s.receive(1, 0, bytes(64)) == audit.Verdict(0, False, "order")         # window 1's challenge isn't out
    s = session()
    s.start()
    s.boundary()
    assert s.receive(1, 0, bytes(64)) is None and s.receive(0, 0, bytes(64)) == audit.Verdict(1, False, "order")
    s = session()
    s.start()
    assert s.receive(0, 0, bytes(63)) == audit.Verdict(0, False, "malformed")
    s = session()
    s.start()
    with pytest.raises(SessionError, match="starts once"):
        s.start()


def test_a_window_of_too_few_passes_is_rejected(store) -> None:
    s, prover, passes, clock = drive(store, params(min_passes=4), SCHEDULE)
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, True), audit.Verdict(1, False, "few")]


def test_draws_wait_for_the_windows_roots(store) -> None:
    clock = Clock()
    s = Session(params(), store.registration, audit.NoiseTiles(96), challenger(store, params()), clock=clock)
    s.start()
    assert s.receive(0, 0, bytes(64)) is None
    with pytest.raises(SessionError, match="not final"):
        s.close(0)
    clock.now = 1000
    s.boundary()
    clock.now = 1000 + A
    with pytest.raises(SessionError, match="not final"):
        s.close(0)
    assert s.receive(1, 0, bytes(64)) is None                  # window 1's first root makes window 0's final
    with pytest.raises(SessionError, match="in order"):
        s.close(1)
    assert isinstance(s.close(0), tuple)
    with pytest.raises(SessionError, match="not open to draw"):
        s.close(0)


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
    s, prover, passes, clock = drive(store, params(), SCHEDULE, plan=partial)
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
    s, prover, passes, clock = drive(store, params(), SCHEDULE, tiles=Zeros(96), verifier_tiles=audit.NoiseTiles(96))
    assert finish(s, prover, passes, clock)[0].failure == "receipt"


def test_openings_are_drawn_from_the_windows_live_coin(store) -> None:
    def draws(label: bytes):
        s, _, _, clock = drive(store, params(), SCHEDULE, label=label)
        clock.now = s.deadline(2) + 1
        out = s.close(0)
        with pytest.raises(ValueError, match="already open"):
            s.challenger.open(s.params.windows)            # window 0's openings coin, after the challenges
        return out
    assert draws(b"coins") == draws(b"coins") != draws(b"other coins")


def test_a_session_over_the_store_starts_after_the_one_before_it_ends(store) -> None:
    """Sessions over one store are not chained to each other, so the next one's first coin waits out the last one's
    deadline (less the trusted round-trip floor): no read serves receipts of both."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    finish(s, prover, passes, clock)
    end = s.record().end_ns
    clock = Clock()
    nxt = Session(params(), store.registration, audit.NoiseTiles(96),
                  coins.Challenger(store.registration.digest, 1, params().coins, entropy=Entropy(b"next")),
                  clock=clock, not_before_ns=end)
    clock.now = end - 1
    with pytest.raises(SessionError, match="may still be reading"):
        nxt.start()
    clock.now = end
    assert nxt.start().index == 0


def test_opening_size_matches_the_timing_model(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert prover.open(passes[0, 0], 0).size == timing.opening_bytes(N)
    assert timing.wrong_bound(10 ** 6, timing.samples_for(1e-3, 2.0 ** -40), 2.0 ** -40) <= 10 ** 3


@pytest.mark.parametrize("pass_ms, rtt_ms, window_ms", [(20, 1, 300), (20, 50, 1000), (50, 5, 1000), (37, 20, 300)])
def test_honest_schedules_fill_their_windows_to_the_chained_steady_state(pass_ms, rtt_ms, window_ms) -> None:
    """Every window certifies at least `⌊(T − p − g)/(p + g)⌋` passes, and the span the passes leave uncovered is `g`
    after each pass and the session's two ends, which come to at most `A + J`."""
    ms, gap, jitter = timing.MS, 50_000, 50_000
    kw = dict(gap_ns=gap, down_ns=rtt_ms * ms // 2, up_ns=rtt_ms * ms - rtt_ms * ms // 2, jitter_ns=jitter)
    st = timing.honest(window_ms * ms, 40, pass_ms * ms, **kw)
    assert min(st.passes) >= (window_ms * ms - pass_ms * ms - gap) // (pass_ms * ms + gap)
    a = timing.deadline(pass_ms * ms, **kw)
    mu = 1 - st.utilization
    steady = timing.chained(pass_ms * ms, gap)
    assert steady <= mu <= steady + (a + jitter) / st.span_ns


@pytest.mark.parametrize("pass_ms, rtt_ms, jitter_ms, floor", [(20, 1, 0.05, 0.5), (20, 0.2, 0, 0), (50, 5, 0.5, 0.9),
                                                               (37, 20, 1, 0.5)])
def test_the_live_coin_alternative_costs_what_the_appendix_says(pass_ms, rtt_ms, jitter_ms, floor) -> None:
    """At Δ* consecutive passes' read windows just don't overlap and μ is (g + U)/(p + g + U); at Δ = p the overlap is
    charged and μ is (g + U)/p; coins faster than passes end are refused."""
    ms, gap = timing.MS, 50_000
    kw = dict(gap_ns=gap, rtt_ns=round(rtt_ms * ms), jitter_ns=round(jitter_ms * ms), rtt_floor_ns=round(floor * rtt_ms * ms))
    u = kw["rtt_ns"] + 2 * kw["jitter_ns"] - kw["rtt_floor_ns"]
    star = timing.live_best_interval(pass_ms * ms, **kw)
    assert star == pass_ms * ms + gap + u
    assert timing.live_mu(star, pass_ms * ms, **kw) == pytest.approx((gap + u) / (pass_ms * ms + gap + u))
    assert timing.live_mu(pass_ms * ms, pass_ms * ms, **kw) == pytest.approx((gap + u) / (pass_ms * ms))
    assert timing.live_mu(star, pass_ms * ms, **kw) > timing.chained(pass_ms * ms, gap)
    with pytest.raises(ValueError, match="faster"):
        timing.live_mu(pass_ms * ms - 1, pass_ms * ms, **kw)
