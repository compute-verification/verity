"""The remote bandwidth audit end to end on a small store: an honest session replayed from `timing.honest` is accepted
and recorded, and each way a session goes wrong is rejected with its failure."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from remote_bandwidth_fakes import Clock, Entropy
from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts, timing
from verity.core.protocols.remote_bandwidth.audit import Params, Session, SessionError
from verity.ml.randomness import frame

N = 20
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"
SCHEDULE = timing.honest(1000, 3, 300, gap_ns=10, down_ns=20, up_ns=30, jitter_ns=5)
A = 20 + 30 + 2 * 5 + 300 + 10
APPENDS = 8
"""The append coins a test session's challenger commits to: these sessions append none, so a few keep it quick."""


def payload(j: int) -> bytes:
    """64-byte pieces with their top byte cleared, so every word is below p at either field."""
    if j % 3 == 0:
        return bytes(codec.BLOCK_BYTES)
    return b"".join(hashlib.sha512(b"%d/%d" % (j, u)).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


@pytest.fixture(scope="module")
def store() -> audit.Store:
    """The store under the fallback codec, p1024 masked at k = 2: about 4 s to encode where the default's k = 3 takes 6.
    Nothing here depends on k; the vectors' session runs at the default."""
    payloads = [payload(j) for j in range(N)]
    salts = [Entropy(b"payload salts")(receipts.SALT_BYTES * N)[receipts.SALT_BYTES * j:receipts.SALT_BYTES * (j + 1)]
             for j in range(N)]
    root = audit.payload_tree(NODE, payloads, salts).commitment.root
    return audit.Store(audit.register(SOURCE, NODE, N, root, codec.FALLBACK, nonce=bytes(32)), payloads, salts)


def registered(store: audit.Store, receipt: str, bind: bool, rows: int | None = None) -> audit.Store:
    """`store` registered under `receipt`, `bind` and `rows` with the same nonce: none is in the codec seed's context,
    so its blocks are `store`'s, and only the registration and its digest differ."""
    reg = audit.register(SOURCE, NODE, N, store.registration.payload_root, codec.FALLBACK, receipt, bind=bind,
                         rows=rows, nonce=bytes(32))
    assert reg.key == store.registration.key
    out = copy.copy(store)
    out.registration = reg
    return out


@pytest.fixture(scope="module")
def stores(store) -> dict[tuple[str, bool], audit.Store]:
    """`store` under each receipt kind, with its receipts of the block alone and binding z: four registrations, four
    digests, one set of blocks."""
    out = {(kind, bind): registered(store, kind, bind) for kind in receipts.KINDS for bind in (False, True)}
    assert len({st.registration.digest for st in out.values()}) == 4
    return out


@pytest.fixture(scope="module")
def sketch_store(stores) -> audit.Store:
    """The default registration's store, named by its kind for the tests that are about it."""
    return stores[receipts.SKETCH_SHA512, False]


@pytest.fixture(scope="module")
def block_store(stores) -> audit.Store:
    """block-sha512, the kind kept for comparison."""
    return stores[receipts.BLOCK_SHA512, False]


@pytest.fixture(scope="module")
def rows64_store(store) -> audit.Store:
    """sketch-sha512 at 64 rows, of the block alone."""
    return registered(store, receipts.SKETCH_SHA512, False, 64)


def params(**kw) -> Params:
    base = dict(blocks=N, deadline_ns=A, windows=3, samples=7, delta=1e-6, window_ns=1000, codec=codec.FALLBACK)
    return Params(**{**base, **kw})


def no_rebuild(r: audit.Record) -> audit.Rebuild:
    """A rebuild model of the record's codec at a ceiling of 0, for the tests of the other terms."""
    return audit.Rebuild(r.codec, 0.0)


def params_of(st: audit.Store, **kw) -> Params:
    """`params` under `st`'s receipt kind, `bind` and rows."""
    return params(receipt=st.registration.receipt, bind=st.registration.bind, rows=st.registration.rows, **kw)


def challenger(store, p: Params, label: bytes = b"coins", session: int = 0) -> coins.Coins:
    return coins.local(store.registration.digest, session, p.windows, Entropy(label), appends=APPENDS)


def honest(prover, t, i, challenges, previous, plan):
    return prover.run(challenges[t], i, previous, plan)


def drive(store, p: Params, schedule: timing.Schedule, *, tiles=None, verifier_tiles=None, run=honest, plan=None,
          sent=None, label=b"coins", coins_=None):
    """Replay `schedule` against a session: `σ_0` released at 0 and each boundary message on time, each pass computed
    as its root arrives, chained from the pass before.  `plan(t, i)` is pass (t, i)'s plan (None: the whole store);
    `run(prover, t, i, challenges, previous, plan)` replaces the honest pass and `sent(ps)` the runs sent with its
    root.  `tiles` is the stub of PoUW's tiles under a registration that binds z, and None under one that does not.
    The session's coins are `coins_`, or a challenger in this process drawing under `label`."""
    tiles = tiles or (audit.NoiseTiles(96) if store.registration.bind else None)
    clock = Clock()
    s = Session(p, store.registration, verifier_tiles or tiles, coins_ or challenger(store, p, label), not_before_ns=0,
                clock=clock)
    prover = audit.Prover(store, tiles, s.commitment, entropy=Entropy(b"receipt salts"))
    events = sorted([(0, 0, -1)] + [(b, 0, t) for t, b in enumerate(schedule.boundaries)]
                    + [(a, 1, (t, i)) for t, i, a in schedule.arrivals])
    challenges, passes, previous = {}, {}, store.registration.digest
    for step, (at, kind, x) in enumerate(events):
        clock.now = at
        if kind == 0:
            coin = s.start() if x < 0 else s.boundary()
            if coin is not None:
                challenges[x + 1] = coin
            continue
        t, i = x
        ps = run(prover, t, i, challenges, previous, plan and plan(t, i))
        passes[t, i], previous = ps, ps.root
        if s.receive(t, i, ps.root, (sent or (lambda q: q.plan.runs))(ps), step=step) is not None:
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
    assert r.public() == {"appended": 0, "appends": None, "bind": False, "block_bytes": 4096, "blocks": N,
                          "codec": "p1024/k2/masked",
                          "codec_params": {"c": 105, "field": "p1024", "masked": True, "n": 1024, "rounds": 2},
                          "coins": s.commitment.root.hex(), "heads": [[None] * 4, [None] * 3, [None] * 3],
                          "layout": "store-order/v0", "lifetime": audit.lifetime_windows(1000), "pace": None,
                          "passes": [4, 3, 3],
                          "post_seed": [0, 0, 0], "protocol": audit.PROTOCOL_ID,
                          "receipt": "sketch-sha512", "receipts": [4 * N, 3 * N, 3 * N],
                          "registration": store.registration.digest.hex(), "released": 0, "rows": 16,
                          "samples": 7, "span_ns": 3000 + A}
    assert r.codec == store.registration.codec == codec.FALLBACK
    w1 = r.profile()["windows"][1]
    assert w1["lengths"] == [N] * 3 and w1["released_ns"] == 1000 and w1["deadline_ns"] == 2000 + A
    assert all(a < b for w in r.windows for a, b in zip(w.steps, w.steps[1:])) and len(w1["steps"]) == 3
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
    g = audit.ledger(r, capacity_bytes_per_s=N * 4096 / 300e-9, chip_bytes=0, rebuild=no_rebuild(r))
    assert g.timing == pytest.approx(1 - 162 * 4096 / (N * 4096 / 300e-9 * r.span_ns / 1e9))


def test_plans_sent_as_arrays_are_received_as_their_runs_are(store) -> None:
    """The verifier takes a plan as the store holds it, an (R, 2) array, and checks it from the array's buffer
    (`Plan.from_array`)."""
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


def store_order(blocks: list[int]) -> np.ndarray:
    """The byte map of vectors that are `blocks`' bytes in store order: row v, byte x is 4096·blocks[v] + x."""
    return np.array(blocks, np.int64)[:, None] * 4096 + np.arange(4096, dtype=np.int64)


def fragments(blocks: int) -> np.ndarray:
    """A tensor-core fragment cut of `blocks` blocks (RB-D1's `gemm_blocks`): vector v holds 32-byte sector v mod 4 of
    every 128-byte codec word of blocks 4·(v // 4) … 4·(v // 4) + 3, so each word is split across 4 vectors."""
    words = np.arange(blocks * 32, dtype=np.int64).reshape(-1, 1, 128, 1)        # (group of 4 blocks, 1, word, 1)
    sectors = np.arange(4, dtype=np.int64).reshape(1, 4, 1, 1) * 32 + np.arange(32, dtype=np.int64)
    return (words * 128 + sectors).reshape(-1, 4096)


def test_a_kernels_vectors_are_credited_only_as_whole_registered_blocks() -> None:
    """`plan_of`: under store-order/v0 a kernel's sketch vector is one registered block's 4,096 bytes in store order
    (32 whole codec words at p1024), and the plan it credits is those blocks.  A fragment cut that splits each word
    across 4 vectors, padding, a byte past the store, a block twice and a layout the verifier can't map are refused."""
    assert receipts.plan_of(store_order([3, 4, 5, 0]), N) == receipts.Plan(((3, 3), (0, 1)), N)
    assert receipts.plan_of(store_order([3, 4, 5, 0]).tolist(), N).runs == ((3, 3), (0, 1))
    assert receipts.plan_of(store_order(list(range(N))), N) == receipts.Plan.whole(N)
    cut = fragments(8)
    assert sorted(cut.ravel().tolist()) == list(range(8 * 4096))                  # every stored byte once, no padding
    assert {len({int(v) for v in np.nonzero((cut // 128) == w)[0]}) for w in range(8 * 32)} == {4}
    padded = store_order([0, 1])
    padded[1, 4000:] = -1                                                         # a tail past the stored bytes
    permuted = store_order([2])
    permuted[0, 128:384] = np.roll(permuted[0, 128:384], 128)                     # whole words, out of store order
    for vectors, match in [(cut, "not block 0's 4096 bytes in store order"), (padded, "vector 1 is not block 1's"),
                           (store_order([N]), "does not start at a registered block"),
                           (np.full((1, 4096), -1, np.int64), "does not start at a registered block"),
                           (permuted, "not block 2's"), (store_order([1, 2, 1]), "repeat a registered block"),
                           (store_order([0])[:, :4095], "is not 4096 bytes"), ([], "one or more vectors"),
                           (store_order([0]).astype(np.float64), "does not start at a registered block")]:
        with pytest.raises(ValueError, match=match):
            receipts.plan_of(vectors, N)
    with pytest.raises(ValueError, match="a block layout is one of"):
        receipts.plan_of(store_order([0]), N, "gemm-a-fragment/v0")


@pytest.mark.parametrize("bind", [False, True])
def test_a_receipt_depends_on_its_block_not_its_position(stores, bind) -> None:
    """The plan's order is only the tree's: the same blocks listed in another order give each block the same receipt
    (z, when the receipts bind it, is the tile before the block's read), and a different root."""
    store = stores[receipts.DEFAULT, bind]
    prover = audit.Prover(store, audit.NoiseTiles(96) if bind else None, challenger(store, params()).commitment,
                          entropy=Entropy(b"s"))
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
    s = Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=Clock())
    s.start()
    assert s.receive(0, 0, bytes(64), runs, step=0) == audit.Verdict(0, False, "plan")


def test_a_pass_that_would_take_its_window_to_the_receipt_limit_rejects(store) -> None:
    """A window's receipts stay below `RECEIPT_LIMIT`: over a store appended to its codec's `max_blocks`, every pass of
    all of it is kept until the one that would take window 0's receipts to the limit, which rejects with `plan`."""
    s = Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=Clock())
    s.start()
    top = s.params.codec.max_blocks
    s.append(bytes(64), top - s.extent)
    last = -(-audit.RECEIPT_LIMIT // top) - 1
    for i in range(last):
        assert s.receive(0, i, bytes([1]) * 64, ((0, top),), step=i) is None
    assert sum(p.length for p in s.windows[0].plans) == last * top < audit.RECEIPT_LIMIT
    assert s.receive(0, last, bytes([1]) * 64, ((0, top),), step=last) == audit.Verdict(0, False, "plan")


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
    rebuild = audit.Rebuild(r.codec, timing.SQUARINGS_PER_S["p1024"])
    g = audit.ledger(r, capacity_bytes_per_s=capacity, chip_bytes=4096, rebuild=rebuild, host=0.002)
    whole = capacity * r.span_ns / 1e9
    assert r.span_ns == 3000 + A and r.passes == 10
    assert g.timing == pytest.approx(1 - 10 * 300 / r.span_ns)
    assert g.sample == 0                                  # every receipt opened
    assert g.chip == pytest.approx(r.passes * 4096 / whole)
    named = [audit.named_bits(k) + audit.SLACK_LOG2 for w in r.windows for k in w.lengths]
    assert g.counting == pytest.approx(sum(named) / 8 / whole) and r.receipts == r.passes * N
    assert sum(named) == r.passes * (audit.named_bits(N) + audit.SLACK_LOG2)
    assert sum(named) < 64 * r.receipts + audit.SLACK_LOG2 * r.passes
    assert g.rebuild == pytest.approx(timing.SQUARINGS_PER_S["p1024"] * 4096 / (65_344 * capacity))
    assert g.gamma == pytest.approx(g.timing + g.chip + g.counting + g.rebuild + 0.002)
    assert g.horizon == g.floor == 0 and g.gamma_horizon == pytest.approx(g.gamma)
    assert set(g.to_json()["terms"]) == set(audit.LEDGER_SOURCES) and "overlap" not in audit.LEDGER_SOURCES
    assert [w.span_ns for w in g.windows] == [w.span_ns for w in r.windows] == [1000 + A] * 3
    assert [w.timing for w in g.windows] == pytest.approx(SCHEDULE.window_timing)
    assert g.windows[0].timing == pytest.approx(1 - 4 * 300 / (1000 + A))
    for w, x in zip(g.windows, r.windows):
        whole_t = capacity * x.span_ns / 1e9
        assert w.chip == pytest.approx(x.passes * 4096 / whole_t) and w.sample == w.horizon == w.floor == 0
        assert w.counting == pytest.approx(sum(audit.named_bits(k) + audit.SLACK_LOG2 for k in x.lengths) / 8 / whole_t)
        assert w.rebuild == g.rebuild
        assert w.gamma == pytest.approx(w.timing + w.chip + w.counting + w.rebuild + 0.002)
    assert g.worst.gamma == max(w.gamma for w in g.windows) > g.gamma
    assert g.to_json()["worst"]["window"] == g.worst.window == 1 and g.to_json()["windows"][1] == g.windows[1].to_json()
    with pytest.raises(ValueError, match="understated"):
        audit.ledger(r, capacity_bytes_per_s=capacity / 2, chip_bytes=0, rebuild=no_rebuild(r))
    with pytest.raises(ValueError, match="link term"):                 # an unbound receipt is of the block alone
        audit.ledger(r, capacity_bytes_per_s=capacity, chip_bytes=0, rebuild=no_rebuild(r), link=0.001)
    assert audit.ledger(dataclasses.replace(r, bind=True), capacity_bytes_per_s=capacity, chip_bytes=0,
                        rebuild=no_rebuild(r), link=0.001).link == 0.001


def test_the_rebuild_term_is_the_record_s_codec_at_the_stated_ceiling() -> None:
    """μ_rebuild is the stated squaring ceiling over the record's codec's encode, `C_sq · 4096 / (E_sq · R)`: at the
    node's ceiling and measured R 0.065% under p1024/k3/masked, 0.097% at k = 2 and 0.31% under p512/k2.  A model
    of another codec than the record's, a bare fraction, or a ceiling that isn't a finite nonnegative rate is
    refused."""
    p512 = codec.Codec(codec.P512, 2, False)
    assert [c.encode_squarings for c in (codec.DEFAULT, codec.FALLBACK, p512)] == [32 * 3 * 1021, 32 * 2 * 1021,
                                                                                 64 * 2 * 509]
    assert [c.encode_squarings for c in (codec.DEFAULT, codec.FALLBACK, p512)] == [98_016, 65_344, 65_152]
    for c in (codec.DEFAULT, p512):
        assert ((c.field.p + 1) // 4).bit_length() - 1 == c.field.n - 3
    fractions = [audit.Rebuild(c, timing.SQUARINGS_PER_S[c.field.name]).fraction(timing.READ_BYTES_PER_S)
                 for c in (codec.DEFAULT, codec.FALLBACK, p512)]
    assert fractions == pytest.approx([6.4781e-4, 9.7172e-4, 3.1296e-3], rel=1e-4)
    r = synthetic((100, 100, 60), 5, 1e-6)
    assert r.codec == codec.DEFAULT
    rebuild = audit.Rebuild(codec.DEFAULT, timing.SQUARINGS_PER_S["p1024"])
    g = audit.ledger(r, capacity_bytes_per_s=100 * 4096, chip_bytes=0, rebuild=rebuild)
    assert g.rebuild == rebuild.fraction(100 * 4096) and {w.rebuild for w in g.windows} == {g.rebuild}
    with pytest.raises(ValueError, match="p1024/k2/masked .* the record's codec p1024/k3/masked"):
        audit.ledger(r, capacity_bytes_per_s=100 * 4096, chip_bytes=0, rebuild=audit.Rebuild(codec.FALLBACK, 1e9))
    with pytest.raises(ValueError, match="not a fraction"):
        audit.ledger(r, capacity_bytes_per_s=100 * 4096, chip_bytes=0, rebuild=0.001)
    with pytest.raises(TypeError):
        audit.ledger(r, capacity_bytes_per_s=100 * 4096, chip_bytes=0)
    for bad in (-1.0, math.inf, math.nan, True, "1e9"):
        with pytest.raises(ValueError, match="finite nonnegative"):
            audit.Rebuild(codec.DEFAULT, bad)
    with pytest.raises(ValueError, match="of a Codec"):
        audit.Rebuild("p1024/k3/masked", 1e9)


def synthetic(lengths: tuple[int, ...], drawn: int, delta: float) -> audit.Record:
    """A record of windows of one pass each, `lengths[t]` receipts and `drawn` openings, a second apiece."""
    windows = tuple(audit.WindowRecord(t, (u,), t * 10**9, (t + 1) * 10**9, (t + 1) * 10**9, IntegrityProfile(
        ("receipt",), (u,), {"receipt": Subset(drawn)}, delta, drawn={"receipt": drawn})) for t, u in enumerate(lengths))
    return audit.Record(bytes(64), bytes(64), max(lengths), 4096, windows)


def test_the_ledger_reads_the_openings_per_window_and_over_the_horizon() -> None:
    """The hybrid claim at 2^-40, as `Accounting.Guarantees.RateSessionDraw` reads it: 7 windows of 2,773
    openings each (d·n ≥ 18,470) are short by under 0.15% over the horizon, and by under 1% in every window, together.
    Read per window, each window's own bound at δ is about 1% of its receipts, with 7·δ in all."""
    n, d, u, delta = 7, 2773, 100_000, 2.0 ** -40
    r = synthetic((u,) * n, d, delta)
    g = audit.ledger(r, capacity_bytes_per_s=u * 4096, chip_bytes=0, rebuild=no_rebuild(r))
    dbar = 1 - delta ** (1 / (n * d))
    assert g.timing == pytest.approx(0, abs=1e-12)
    assert g.horizon == pytest.approx(dbar) and 0.0014 < g.horizon < 0.0015
    assert g.floor == pytest.approx(n * dbar) and (1 - g.floor / n) ** (n * d) == pytest.approx(delta)
    assert 0.0099 < g.floor < 0.01
    assert g.sample == pytest.approx(0.00981) and g.gamma - g.gamma_horizon == pytest.approx(g.sample - g.horizon)
    assert g.to_json()["horizon"] == {"floor": g.floor, "gamma": g.gamma_horizon, "sample": g.horizon}
    # Every window's own, from the record: the pooled event leaves each window's μ_sample under 1%, every one at once;
    # its γ adds λ, log2 of q^k/k! for its pass of 10^5 receipts, 48.8 bits a receipt or 0.149% of a 4 KiB block, so
    # the window's γ is 1.15%.
    assert audit.floors(synthetic((u,) * n, d, delta)) == pytest.approx((g.floor,) * n)
    assert all(w.horizon == pytest.approx(g.floor) and w.sample == pytest.approx(g.sample) for w in g.windows)
    w = g.worst_horizon
    assert w.horizon < 0.01 and w.counting == pytest.approx((audit.named_bits(u) + audit.SLACK_LOG2) / 8 / (u * 4096))
    assert 48.8 < audit.named_bits(u) / u < 48.9
    assert w.gamma_horizon == pytest.approx(w.horizon + w.counting) and 0.0114 < w.gamma_horizon < 0.0115
    assert g.to_json()["worst"]["horizon"] == {"gamma": w.gamma_horizon, "window": w.window}


def test_the_ledger_refuses_a_window_its_capacity_cannot_have_read() -> None:
    """Window 0 certifies 100 blocks in its second, where the capacity reads 60: understated, though the session's two
    seconds hold its 110."""
    r = synthetic((100, 10), 5, 1e-6)
    with pytest.raises(ValueError, match="window 0 .* understated"):
        audit.ledger(r, capacity_bytes_per_s=60 * 4096, chip_bytes=0, rebuild=no_rebuild(r))
    assert audit.ledger(r, capacity_bytes_per_s=100 * 4096, chip_bytes=0,
                        rebuild=no_rebuild(r)).windows[0].timing == 0


def test_the_horizon_puts_the_shortfall_where_it_hides_most() -> None:
    """Windows of unequal size: the pooled bound's shortfall goes to the largest windows first, each capped where its
    draws would have found it (s distinct draws found s correct receipts), and a window opened whole has none."""
    lengths, s, delta = (5000, 1000, 1000, 40), 40, 1e-6
    wrong, floor = audit.pooled(synthetic(lengths, s, delta))
    budget = -math.expm1(math.log(delta) / (4 * s)) * 4 * s
    assert wrong == pytest.approx(budget / s * 5000) and floor == pytest.approx(budget / s)
    wrong, floor = audit.pooled(synthetic((12, 1000), 10, delta))
    budget = -math.expm1(math.log(delta) / 20) * 20
    assert budget / 10 > 1 - 10 / 1000 and floor == pytest.approx(1 - 10 / 1000)
    assert wrong == pytest.approx(990 + (budget - 9.9) / 10 * 12)
    assert audit.pooled(synthetic((40,) * 3, 40, delta)) == (0.0, 0.0)


def test_codec_round_trips_and_refuses_one_round(store) -> None:
    k = store.registration.key
    for j in (0, 1, N - 1):
        assert codec.decode(store.blocks[j], j, k) == store.payloads[j]
    assert store.blocks[0] != bytes(codec.BLOCK_BYTES)            # zero weights encode to a store that is not zeros
    assert codec.decode(b"\xff" * codec.BLOCK_BYTES, 0, k) is None
    with pytest.raises(ValueError, match="at least 2 rounds"):
        codec.Codec(codec.P1024, 1, True)
    with pytest.raises(ValueError, match="a Codec"):
        params(codec="p1024/k3/masked")


def test_the_codecs_are_named_and_each_decodes_its_own_blocks() -> None:
    """The default is p1024 masked at k = 3, the fallback k = 2 at the same width, and p512 the upgrade path: each a
    block of 128- or 64-byte words, each its own store, and a block under one is no other's."""
    p512 = codec.Codec(codec.P512, 2, False)
    assert [c.name for c in (codec.DEFAULT, codec.FALLBACK, p512)] == ["p1024/k3/masked", "p1024/k2/masked", "p512/k2"]
    assert (codec.P1024.word_bytes, codec.P1024.block_words, codec.P512.word_bytes, codec.P512.block_words) == (128, 32, 64, 64)
    assert (codec.DEFAULT.max_blocks, p512.max_blocks) == (1 << 59, 1 << 58)
    seed, m, j = hashlib.sha256(b"codec seed").digest(), payload(1), 3
    keys = [codec.key(seed, c) for c in (codec.DEFAULT, codec.FALLBACK, codec.Codec(codec.P1024, 3, False), p512)]
    blocks = [codec.encode(m, j, k) for k in keys]
    assert len(set(blocks)) == 4 and all(codec.decode(b, j, k) == m for b, k in zip(blocks, keys))
    assert all(codec.decode(blocks[0], j, k) != m for k in keys[1:]) and codec.decode(blocks[0], j + 1, keys[0]) != m
    assert keys[0].c[:2] == keys[1].c and keys[0].mask == keys[1].mask and keys[2].mask is None and keys[3].mask is None
    assert codec.decode((codec.P1024.p + 1).to_bytes(128, "little") + blocks[0][128:], j, keys[0]) is None
    with pytest.raises(ValueError, match="below 576460752303423488"):
        codec.encode(m, 1 << 59, keys[0])
    with pytest.raises(ValueError, match="one of"):
        codec.Codec(codec.Field("p1024", 1024, 107), 3, True)


def test_the_codec_seed_is_drawn_after_and_bound_to_the_payload_root() -> None:
    root = hashlib.sha512(b"payloads").digest()
    a = audit.register(SOURCE, NODE, N, root, nonce=bytes(32))
    assert a == audit.register(SOURCE, NODE, N, root, nonce=bytes(32)) and a.codec == codec.DEFAULT
    others = [audit.register(SOURCE, NODE, N, hashlib.sha512(b"other").digest(), nonce=bytes(32)),
              audit.register(SOURCE, b"another node", N, root, nonce=bytes(32)),
              audit.register(SOURCE, NODE, N + 1, root, nonce=bytes(32)),
              audit.register(SOURCE, NODE, N, root, nonce=bytes([1]) * 32),
              audit.register(b"another verifier", NODE, N, root, nonce=bytes(32)),
              audit.register(SOURCE, NODE, N, root, codec.FALLBACK, nonce=bytes(32))]
    assert len({a.codec_seed, *(o.codec_seed for o in others)}) == 7
    assert len({a.digest, *(o.digest for o in others)}) == 7
    assert dataclasses.replace(a, codec=codec.FALLBACK).digest != a.digest     # the digest binds the codec's name
    assert a.receipt == receipts.DEFAULT == receipts.SKETCH_SHA512 == receipts.KINDS[0] and a.bind is False
    blocked = audit.register(SOURCE, NODE, N, root, codec.DEFAULT, receipts.BLOCK_SHA512, nonce=bytes(32))
    assert blocked == dataclasses.replace(a, receipt=receipts.BLOCK_SHA512, rows=0)   # the same codec seed and store
    assert blocked.digest not in {a.digest, *(o.digest for o in others)}       # the digest binds the receipt kind
    bound = audit.register(SOURCE, NODE, N, root, bind=True, nonce=bytes(32))
    assert bound == dataclasses.replace(a, bind=True)                          # nor is bind in the codec seed's context
    assert bound.digest not in {a.digest, blocked.digest, *(o.digest for o in others)}   # the digest binds it
    assert a.digest == hashlib.sha512(audit.REGISTRATION_DOMAIN + frame(NODE) + N.to_bytes(8, "big") + root
                                      + a.codec_seed + frame(a.codec.name) + frame(a.receipt) + b"\0"
                                      + frame("store-order/v0") + (16).to_bytes(4, "big")).digest()
    assert bound.digest == hashlib.sha512(audit.REGISTRATION_DOMAIN + frame(NODE) + N.to_bytes(8, "big") + root
                                          + a.codec_seed + frame(a.codec.name) + frame(a.receipt) + b"\1"
                                          + frame("store-order/v0") + (16).to_bytes(4, "big")).digest()
    assert a.rows == receipts.SKETCH_ROWS == receipts.ROWS[0] == 16 and blocked.rows == 0
    wide = audit.register(SOURCE, NODE, N, root, rows=64, nonce=bytes(32))
    assert wide == dataclasses.replace(a, rows=64)                             # nor are the rows in its context
    assert wide.digest == hashlib.sha512(audit.REGISTRATION_DOMAIN + frame(NODE) + N.to_bytes(8, "big") + root
                                         + a.codec_seed + frame(a.codec.name) + frame(a.receipt) + b"\0"
                                         + frame("store-order/v0") + (64).to_bytes(4, "big")).digest()
    assert wide.digest not in {a.digest, blocked.digest, bound.digest, *(o.digest for o in others)}
    assert audit.register(SOURCE, NODE, N, root, rows=16, nonce=bytes(32)) == a
    assert blocked == audit.register(SOURCE, NODE, N, root, codec.DEFAULT, receipts.BLOCK_SHA512, rows=0,
                                     nonce=bytes(32))
    sketch, block = receipts.SKETCH_SHA512, receipts.BLOCK_SHA512
    for kind, rows, match in ((sketch, 32, "rows are one of"), (sketch, 0, "one of"), (sketch, True, "one of"),
                              (sketch, 64.0, "one of"), (block, 16, "sketches nothing"), (block, False, "nothing")):
        with pytest.raises(ValueError, match=match):
            audit.register(SOURCE, NODE, N, root, codec.DEFAULT, kind, rows=rows, nonce=bytes(32))
        with pytest.raises(ValueError, match=match):
            params(receipt=kind, rows=rows)
    assert a.layout == receipts.LAYOUT == receipts.STORE_ORDER == receipts.LAYOUTS[0]
    assert dataclasses.replace(a, layout="gemm-a-fragment/v0").digest != a.digest   # the digest binds the layout
    with pytest.raises(ValueError, match="a block layout is one of"):                # one the verifier can't map
        audit.register(SOURCE, NODE, N, root, layout="gemm-a-fragment/v0", nonce=bytes(32))
    with pytest.raises(ValueError, match="a block layout is one of"):
        params(layout="gemm-a-fragment/v0")
    with pytest.raises(ValueError, match="a receipt kind is one of"):
        audit.register(SOURCE, NODE, N, root, codec.DEFAULT, "sha512", nonce=bytes(32))
    with pytest.raises(ValueError, match="bind is True or False"):
        audit.register(SOURCE, NODE, N, root, bind=1, nonce=bytes(32))
    p512 = codec.Codec(codec.P512, 2, False)
    assert audit.register(SOURCE, NODE, 1 << 58, root, p512, nonce=bytes(32)).blocks == 1 << 58
    with pytest.raises(ValueError, match="at most"):
        audit.register(SOURCE, NODE, (1 << 58) + 1, root, p512, nonce=bytes(32))
    with pytest.raises(ValueError, match="names a Codec"):
        audit.register(SOURCE, NODE, N, root, "p1024/k3/masked", nonce=bytes(32))


def test_a_session_refuses_parameters_that_are_not_the_registrations(store, stores, block_store, rows64_store) -> None:
    bound = stores[receipts.DEFAULT, True]
    for st, p in ((store, params(codec=codec.DEFAULT)), (store, params(blocks=N + 1)),
                  (store, params(receipt=receipts.BLOCK_SHA512)), (block_store, params()),
                  (store, params(bind=True)), (bound, params()), (store, params(rows=64)), (rows64_store, params()),
                  (rows64_store, params(rows=16))):
        with pytest.raises(ValueError, match="not the registration's"):
            Session(p, st.registration, audit.NoiseTiles(96) if p.bind else None, challenger(st, p), not_before_ns=0,
                    clock=Clock())
    other = copy.copy(store)
    other.registration = dataclasses.replace(store.registration, layout="gemm-a-fragment/v0")
    with pytest.raises(ValueError, match="not the registration's"):                # a layout the session can't map
        Session(params(), other.registration, None, challenger(other, params()), not_before_ns=0, clock=Clock())
    with pytest.raises(ValueError, match="a receipt kind is one of"):
        params(receipt="sha512")
    with pytest.raises(ValueError, match="bind is True or False"):
        params(bind=0)


def test_tiles_are_taken_only_by_a_registration_that_binds_z(store, stores) -> None:
    """A receipt of the block alone has no tile, so passing PoUW's tiles to it is refused, as is leaving them out of
    one that binds z."""
    bound = stores[receipts.DEFAULT, True]
    for st, tiles in ((store, audit.NoiseTiles(96)), (bound, None)):
        with pytest.raises(ValueError, match="binds z takes PoUW's tiles"):
            Session(params_of(st), st.registration, tiles, challenger(st, params()), not_before_ns=0, clock=Clock())
        with pytest.raises(ValueError, match="binds z takes PoUW's tiles"):
            audit.Prover(st, tiles, challenger(st, params()).commitment)


def test_payload_leaves_hide_the_payload_under_their_salts(store) -> None:
    """Equal payloads (blocks 0 and 3 are zeros) commit to different leaves under their salts, as hm96-sha512/row/v2."""
    from verity.ml.commitments import hm96
    from verity.ml.commitments.rowleaf import sha512_row_v2_digest
    x = sha512_row_v2_digest(store.payloads[0], 8 * codec.BLOCK_BYTES)
    assert store.payloads[0] == store.payloads[3] and store.payload_tree.open(0).value != store.payload_tree.open(3).value
    assert store.payload_tree.open(0).value == hm96.SHA512.tree_leaf(
        hm96.SHA512.default_key, hm96.SHA512.commit_string(hm96.SHA512.default_key, x, store.salts[0]))
    assert audit.PAYLOAD_SCHEMA == receipts.SCHEMA == "hm96-sha512/row/v2"


def test_a_receipt_is_seed_first_of_the_block_alone_or_binds_z_elementwise(store) -> None:
    """block-sha512 hashes the block after the seed's block; a receipt that binds z hashes the block ⊕ spread(z), so
    every byte of z counts, and a z of 4,096 zeros gives the block alone's."""
    seed = hashlib.sha512(b"a pass seed").digest()
    pre = receipts.prefix(seed)
    assert len(pre) == 128 and seed in pre[:128]
    alone = receipts.Receipter(seed)(5, store.blocks[5])
    assert alone == hashlib.sha512(pre + (5).to_bytes(8, "big") + store.blocks[5]).digest()
    assert alone == receipts.Receipter(seed)(5, store.blocks[5], bytes(codec.BLOCK_BYTES))
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


def test_a_sketch_receipt_hashes_the_passs_signed_sums_of_the_block_seed_first(store) -> None:
    """σ_j is the pass's ±1 signs times the block as int8, or the block ⊕ spread(z) when the receipt binds z (here
    numpy's product, in `receipts` bit-plane sums), 16 int32 hashed after the seed's block.  A one-bit change of the
    block moves every row, and another seed's signs give another receipt."""
    seed, z, j, block = hashlib.sha512(b"a pass seed").digest(), bytes(range(96)), 5, store.blocks[5]
    r = receipts.SketchReceipter(seed)
    bits = np.frombuffer(b"".join(receipts.sign_bits(seed)), np.uint8).reshape(16, 512)
    signs = 1 - 2 * np.unpackbits(bits, axis=1, bitorder="little").astype(np.int64)
    pre = receipts.sketch_prefix(seed)
    assert len(pre) == 128 and seed in pre and pre != receipts.prefix(seed)
    for zz, e in ((None, block), (z, bytes(a ^ b for a, b in zip(block, (z * 43)[:4096])))):
        sigma = r.sketch(block, zz)
        assert sigma == (signs @ np.frombuffer(e, np.int8).astype(np.int64)).astype("<i4").tobytes()
        y = r(j, block, zz)
        assert y == hashlib.sha512(pre + j.to_bytes(8, "big") + sigma).digest() != receipts.Receipter(seed)(j, block, zz)
        for u in (0, 47, 4095):
            flipped = block[:u] + bytes([block[u] ^ 1]) + block[u + 1:]
            assert (np.frombuffer(r.sketch(flipped, zz), "<i4") != np.frombuffer(sigma, "<i4")).all()
        other = hashlib.sha512(b"the next pass's seed").digest()
        assert receipts.sign_bits(other) != receipts.sign_bits(seed) and receipts.SketchReceipter(other)(j, block, zz) != y
    assert r(j, block) == r(j, block, bytes(codec.BLOCK_BYTES)) != r(j, block, z)
    for bad in ((-1, block, z), (j, block[:-1], z), (j, block, b""), (-1, block, None), (j, block[:-1], None)):
        with pytest.raises(ValueError):
            r(*bad)
    with pytest.raises(ValueError, match="a receipt kind is one of"):
        receipts.receipter("sha512", seed)


def test_a_64_row_sketch_is_its_own_signs_and_hashes_256_bytes(store) -> None:
    """At k_s = 64 the signs' key binds k, so its first 16 rows are not the 16-row signs, and σ_j is 64 int32, 256
    bytes, three SHA-512 compressions after the seed's block where 16 rows take one."""
    seed, z, j, block = hashlib.sha512(b"a pass seed").digest(), bytes(range(96)), 5, store.blocks[5]
    r = receipts.SketchReceipter(seed, 64)
    assert r.rows == 64 and receipts.SketchReceipter(seed).rows == 16
    assert receipts.receipter("sketch-sha512", seed).rows == 16
    bits = np.frombuffer(b"".join(receipts.sign_bits(seed, 64)), np.uint8).reshape(64, 512)
    signs = 1 - 2 * np.unpackbits(bits, axis=1, bitorder="little").astype(np.int64)
    assert receipts.sign_bits(seed, 64)[:16] != receipts.sign_bits(seed)
    pre = receipts.sketch_prefix(seed)
    for zz, e in ((None, block), (z, bytes(a ^ b for a, b in zip(block, (z * 43)[:4096])))):
        sigma = r.sketch(block, zz)
        assert len(sigma) == 256
        assert sigma == (signs @ np.frombuffer(e, np.int8).astype(np.int64)).astype("<i4").tobytes()
        y = r(j, block, zz)
        assert y == hashlib.sha512(pre + j.to_bytes(8, "big") + sigma).digest() == receipts.receipter(
            "sketch-sha512", seed, 64)(j, block, zz) != receipts.SketchReceipter(seed)(j, block, zz)
        assert -(-(len(pre) + 8 + len(sigma) + 17) // 128) == 4                  # the seed's block and three
    for rows in (0, 16.0, 32, True):
        with pytest.raises(ValueError, match="one of"):
            receipts.SketchReceipter(seed, rows)
        with pytest.raises(ValueError, match="one of"):
            receipts.sign_bits(seed, rows)
    assert type(receipts.receipter("block-sha512", seed)) is receipts.Receipter
    assert type(receipts.receipter("block-sha512", seed, 0)) is receipts.Receipter
    with pytest.raises(ValueError, match="sketches nothing"):
        receipts.receipter("block-sha512", seed, 64)


def test_an_honest_64_row_session_is_accepted_and_names_its_rows(rows64_store) -> None:
    """The verifier recomputes each opened receipt at the registration's 64 rows, and the record names them; receipts
    made at 16 rows under the 64-row registration fail the first draw."""
    st = rows64_store
    s, prover, passes, clock = drive(st, params_of(st), SCHEDULE, plan=partial)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    assert s.record().public()["rows"] == s.record().rows == 64
    ps = passes[1, 2]
    assert ps.receipts[0] == receipts.SketchReceipter(ps.seed, 64)(ps.plan.block(0), st.blocks[ps.plan.block(0)])
    s, prover, passes, clock = drive(st, params_of(st), SCHEDULE,
                                     run=committed_with(lambda seed, prover: receipts.SketchReceipter(seed)))
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, False, "receipt", s.windows[0].draws[0])]


@pytest.mark.parametrize("kind", receipts.KINDS)
def test_an_honest_session_of_either_kind_is_accepted_and_names_it(stores, kind) -> None:
    """The schedule of partial plans under each kind: the verifier recomputes each opened receipt from the block (under
    sketch-sha512, under signs of each pass's own), and the record names the kind."""
    st = stores[kind, False]
    s, prover, passes, clock = drive(st, params_of(st), SCHEDULE, plan=partial)
    assert all(v.accepted for v in finish(s, prover, passes, clock))
    r = s.record()
    assert r.public()["receipt"] == kind and r.registration == st.registration.digest
    ps = passes[1, 2]
    j = ps.plan.block(0)
    assert ps.receipts[0] == receipts.receipter(kind, ps.seed)(j, st.blocks[j])
    if kind == receipts.SKETCH_SHA512:
        assert len({receipts.sign_bits(ps.seed) for ps in passes.values()}) == len(passes) == 10


def committed_with(receipter):
    """Passes on the chain, committed under its seeds, with the receipts `receipter(seed, prover)` makes."""
    def run(prover, t, i, challenges, previous, plan):
        seed = receipts.seed(coins.coin(challenges[t]), i, previous)
        plan = plan or receipts.Plan.whole(N)
        r, tiles = receipter(seed, prover), prover.tiles
        ys = tuple(r(j, prover.store.blocks[j], None if tiles is None else tiles.output(seed, j)) for j in plan)
        salts = tuple(prover.entropy(receipts.SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return audit.Pass(t, i, seed, plan, ys, salts, receipts.tree(seed, plan, ys, salts))
    return run


@pytest.mark.parametrize("receipter", [
    lambda seed, prover: receipts.SketchReceipter(prover.store.registration.digest),
    lambda seed, prover: receipts.Receipter(seed),
], ids=["signs-known-before-the-session", "block-sha512"])
def test_sketch_receipts_not_under_the_passs_signs_reject(sketch_store, receipter) -> None:
    """Sketches under signs fixed before the session, or block-sha512 receipts, committed on the chain: each opening's
    paths and decoding check, and the first draw fails the receipt."""
    s, prover, passes, clock = drive(sketch_store, params(receipt=receipts.SKETCH_SHA512), SCHEDULE,
                                     run=committed_with(receipter))
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, False, "receipt", s.windows[0].draws[0])]


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
    first = c.challenge(0, c.window)
    assert coins.coin_reason(c.commitment, first, coins.CHALLENGE, 0) is None
    assert coins.coin(first) == s.windows[0].challenge
    assert (coins.coin_reason(c.commitment, first, coins.CHALLENGE, 1)
            == "coin is the epoch's challenge coin 0, not its challenge coin 1")
    forged = dataclasses.replace(first, coin=bytes([first.coin[0] ^ 1]) + first.coin[1:])
    assert (coins.coin_reason(c.commitment, forged, coins.CHALLENGE, 0)
            == "coin does not open against the epoch's coin commitment")
    with pytest.raises(ValueError, match="does not open"):
        prover.run(forged, 0, store.registration.digest)
    with pytest.raises(ValueError, match="openings coin"):
        prover.run(c.openings(0, first.after), 0, store.registration.digest)
    with pytest.raises(ValueError, match="already open"):
        c.challenger.open(coins.CHALLENGE, 0, c.window)


def unchained(prover, t, i, challenges, previous, plan):
    """Every pass seeded after the registration, so a window's passes could all run at once off one read."""
    return prover.pass_under(t, i, receipts.seed(coins.coin(challenges[t]), i, prover.store.registration.digest), plan)


def detached(prover, t, i, challenges, previous, plan):
    """Window 1's chain started over from the registration, so it need not wait on window 0's last root."""
    return prover.run(challenges[t], i, prover.store.registration.digest if (t, i) == (1, 0) else previous, plan)


def early(prover, t, i, challenges, previous, plan):
    """Window 1's passes under window 0's challenge, so they could start before window 1's challenge is out."""
    return prover.pass_under(t, i, receipts.seed(coins.coin(challenges[0 if t == 1 else t]), i, previous), plan)


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
        return Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=Clock())
    s = session()
    with pytest.raises(SessionError, match="not started"):
        s.receive(0, 0, bytes(64), step=0)
    s.start()
    assert s.receive(0, 1, bytes(64), step=0) == audit.Verdict(0, False, "order")
    s = session()
    s.start()
    assert s.receive(0, 0, bytes(64), step=0) is None
    assert s.receive(0, 0, bytes(64), step=1) == audit.Verdict(0, False, "order")
    s = session()
    s.start()
    assert s.receive(1, 0, bytes(64), step=0) == audit.Verdict(0, False, "order")  # window 1's challenge isn't out
    clock = Clock()
    s = Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=clock)
    s.start()
    clock.now = 1000
    s.boundary()
    assert s.receive(1, 0, bytes(64), step=0) is None
    assert s.receive(0, 0, bytes(64), step=1) == audit.Verdict(1, False, "order")
    s = session()
    s.start()
    assert s.receive(0, 0, bytes(63), step=0) == audit.Verdict(0, False, "malformed")
    s = session()
    s.start()
    with pytest.raises(SessionError, match="starts once"):
        s.start()


@pytest.mark.parametrize("steps, failure", [
    ((0, 5, 9), None),
    ((None,), "step"),
    ((3, 3), "step"),
    ((4, 2), "step"),
    ((-1,), "step"),
    ((True,), "step"),
    ((audit.STEP_LIMIT,), "step"),
    ((0, 1, "2"), "step"),
], ids=["increasing", "none", "repeated", "decreasing", "negative", "bool", "limit", "str"])
def test_each_pass_names_a_step_above_the_previous_passs(store, steps, failure) -> None:
    """A pass covers one step of the served circuit, named with its root: a root with no step, or one not above the
    previous pass's, across windows too, rejects the session (`step`), and an accepted window records them."""
    clock = Clock()
    s = Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=clock)
    s.start()
    out = None
    for k, step in enumerate(steps):
        if k == 2:
            clock.now = 1000
            s.boundary()
        out = s.receive(0 if k < 2 else 1, k if k < 2 else 0, bytes(64), step=step)
        if out is not None:
            break
    assert out == (None if failure is None else audit.Verdict(0 if k < 2 else 1, False, failure))
    if failure is None:
        assert [w.steps for w in s.windows] == [[0, 5], [9]]


def test_a_window_of_too_few_passes_is_rejected(store) -> None:
    s, prover, passes, clock = drive(store, params(min_passes=4), SCHEDULE)
    assert finish(s, prover, passes, clock) == [audit.Verdict(0, True), audit.Verdict(1, False, "few")]


def test_draws_wait_for_the_windows_roots(store) -> None:
    clock = Clock()
    s = Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=clock)
    s.start()
    assert s.receive(0, 0, bytes(64), step=0) is None
    with pytest.raises(SessionError, match="not final"):
        s.close(0)
    clock.now = 1000
    s.boundary()
    clock.now = 1000 + A
    with pytest.raises(SessionError, match="not final"):
        s.close(0)
    assert s.receive(1, 0, bytes(64), step=1) is None          # window 1's first root makes window 0's final
    with pytest.raises(SessionError, match="in order"):
        s.close(1)
    assert isinstance(s.close(0), tuple)
    with pytest.raises(SessionError, match="not open to draw"):
        s.close(0)


def test_a_paced_windows_roots_are_final_at_its_k_th_root(store) -> None:
    """A paced window takes at most k roots, so its k-th makes them final once its boundary has left: it is drawn then,
    with no wait for the next window's first root or its deadline, and the openings coin's request names that root on
    the challenger's log."""
    clock = Clock()
    p = params(pace=audit.Pace(2))
    s = Session(p, store.registration, None, challenger(store, p), not_before_ns=0, clock=clock)
    s.start()
    assert s.receive(0, 0, bytes(64), step=0) is None
    with pytest.raises(SessionError, match="not final"):
        s.close(0)
    last = hashlib.sha512(b"window 0's last root").digest()
    assert s.receive(0, 1, last, step=1) is None
    with pytest.raises(SessionError, match="not final"):
        s.close(0)
    clock.now = 1000
    s.boundary()
    assert isinstance(s.close(0), tuple)
    assert s.coins.afters(coins.OPENINGS) == (last,)
    assert s.deadline(0) == 1000 + A
    assert s.receive(0, 2, bytes(64), step=2) == audit.Verdict(0, False, "order")


def test_a_boundary_waits_out_the_stated_window(store) -> None:
    """The window T is stated, 60 s with 33,273 openings at 2^-40 unless the params say otherwise: no boundary message
    leaves sooner than `window_ns` after its window's challenge, and a refused one leaves the window open."""
    p = Params(blocks=N, deadline_ns=A, windows=3)
    assert (p.window_ns, p.samples, p.delta) == (audit.WINDOW_NS, audit.SAMPLES, audit.DELTA) == (60 * 10**9, 33_273, 2.0**-40)
    clock = Clock()
    s = Session(params(), store.registration, None, challenger(store, params()), not_before_ns=0, clock=clock)
    s.start()
    clock.now = 999
    with pytest.raises(SessionError, match="may leave at 1000 ns, not 999"):
        s.boundary()
    clock.now = 1000
    assert s.boundary().index == 1 and s.windows[0].boundary_ns == 1000
    clock.now = 1999
    with pytest.raises(SessionError, match="window 1 is 1000 ns"):
        s.boundary()
    clock.now = 2500
    assert s.boundary().index == 2 and s.windows[1].boundary_ns == 2500
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="window_ns"):
            params(window_ns=bad)


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
    ("tile", lambda o: b"not PoUW's evidence", "malformed"),   # a receipt of the block alone has no tile
    ("position", lambda o: o.position + 1, "malformed"),
])
def test_a_bad_opening_rejects(store, field_, value, failure) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE, plan=partial)
    open_ = _tampered(field_, value)
    verdicts = finish(s, prover, passes, clock, open_=lambda ps, p: open_(ps, p, prover=prover))
    assert verdicts[0].failure == failure and not verdicts[0].accepted
    with pytest.raises(SessionError):
        s.record()


@pytest.mark.parametrize("kind", receipts.KINDS)
def test_a_session_that_binds_z_recomputes_each_tile(stores, kind) -> None:
    """PoUW's opt-in join, under either kind: an honest session is accepted and its record says it binds z; each
    receipt is of the block and the tile's z, not of the block alone; an opening without PoUW's evidence fails the tile;
    and receipts made with a z PoUW's check does not recompute (here, zeros) open with good paths and fail the receipt."""
    st, tiles = stores[kind, True], audit.NoiseTiles(96)
    s, prover, passes, clock = drive(st, params_of(st), SCHEDULE, plan=partial)
    assert all(v.accepted for v in finish(s, prover, passes, clock)) and s.record().public()["bind"] is True
    ps = passes[1, 2]
    j, r = ps.plan.block(0), receipts.receipter(kind, ps.seed)
    assert ps.receipts[0] == r(j, st.blocks[j], tiles.output(ps.seed, j)) != r(j, st.blocks[j])
    s, prover, passes, clock = drive(st, params_of(st), SCHEDULE)
    bad = _tampered("tile", lambda o: b"not PoUW's evidence")
    assert finish(s, prover, passes, clock, open_=lambda ps, p: bad(ps, p, prover=prover))[0].failure == "tile"

    class Zeros(audit.NoiseTiles):
        def output(self, seed, p):
            return bytes(self.z_bytes)
    s, prover, passes, clock = drive(st, params_of(st), SCHEDULE, tiles=Zeros(96), verifier_tiles=tiles)
    assert finish(s, prover, passes, clock)[0].failure == "receipt"


def test_openings_are_drawn_from_the_windows_live_coin(store) -> None:
    def draws(label: bytes):
        s, _, _, clock = drive(store, params(), SCHEDULE, label=label)
        clock.now = s.deadline(2) + 1
        out = s.close(0)
        with pytest.raises(ValueError, match="already open"):
            s.coins.challenger.open(coins.OPENINGS, 0, s.windows[0].roots[-1])
        assert s.coins.afters(coins.OPENINGS) == (s.windows[0].roots[-1],)
        return out
    assert draws(b"coins") == draws(b"coins") != draws(b"other coins")


def test_a_session_over_the_store_starts_after_the_one_before_it_ends(store) -> None:
    """Sessions over one store are not chained to each other, so the next one's first coin waits out the last one's
    deadline (less the trusted round-trip floor): no read serves receipts of both."""
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    finish(s, prover, passes, clock)
    end = s.record().end_ns
    clock = Clock()
    nxt = Session(params(), store.registration, None, challenger(store, params(), b"next", 1), clock=clock,
                  not_before_ns=end)
    clock.now = end - 1
    with pytest.raises(SessionError, match="may still be reading"):
        nxt.start()
    clock.now = end
    assert nxt.start().index == 0
    later = challenger(store, params(), b"later", 2)
    with pytest.raises(TypeError, match="not_before_ns"):           # every session names when it may start
        Session(params(), store.registration, None, later, clock=clock)
    for bad in (None, -1, 1.5, True):
        with pytest.raises(ValueError, match="not_before_ns"):
            Session(params(), store.registration, None, later, not_before_ns=bad, clock=clock)


def test_opening_size_matches_the_timing_model(store) -> None:
    s, prover, passes, clock = drive(store, params(), SCHEDULE)
    assert prover.open(passes[0, 0], 0).size == timing.opening_bytes(N)
    assert timing.wrong_bound(10 ** 6, timing.samples_for(1e-3, 2.0 ** -40), 2.0 ** -40) <= 10 ** 3


def test_the_stated_claim_holds_in_every_window_and_for_the_horizons_sampling_term() -> None:
    """`PROTOCOL.md` §1 and §10's reading through the ledger, on one node whose store spans every tier, 94 GB of each
    GPU's HBM and host DRAM's share, at the node's R, S_chip and squaring ceiling.  At the stated T = 60 s and 33,273
    openings a window (555 a second), a 10 min session's worst window has its whole γ_t at 0.896–0.927%; over the
    horizon the pooled μ_sample is under 0.01% and the whole γ about 0.67%, over 0.15%, which μ_chip exceeds alone and
    λ (0.116%) with μ_rebuild.  At spec every reading is `1 − (1 − γ)·R/R_spec`, 5.8%: R cancels in what is certified.
    The same 94 GB a GPU in HBM alone leaves host DRAM's 4.7% of R idle, 5.57–5.60%.  The 20 s option at 11,091
    openings has μ_sample under 0.25% and a whole γ_t over 1%, and the 1 s option's μ_sample alone is under 1%, at
    2,773."""
    s, store = audit.SAMPLES, timing.spanning(94 * 10 ** 9)
    assert 789.0e9 < store < 789.1e9
    assert (timing.samples_for(0.0025, audit.DELTA), timing.samples_for(0.01, audit.DELTA)) == (11_091, 2_773)
    assert s == 3 * 11_091 and audit.WINDOW_NS == 60 * timing.S
    idle = timing.HOST_READ_BYTES_PER_S / timing.READ_BYTES_PER_S
    for rtt in (1, 20):
        g = timing.stated(store, audit.WINDOW_NS, 10, rtt_ns=rtt * timing.MS, samples=s)
        assert g.worst.sample < 0.001 and 0.00895 < g.worst.gamma < 0.00928
        assert 0.00115 < g.counting < 0.00116 and g.counting + g.rebuild > 0.0015 and g.chip > 0.0015
        assert g.horizon < 1e-4 and 0.0066 < g.gamma_horizon < 0.0068
        spec = timing.stated(store, audit.WINDOW_NS, 10, rtt_ns=rtt * timing.MS, samples=s,
                             capacity_bytes_per_s=timing.SPEC_BYTES_PER_S).worst
        assert spec.gamma == pytest.approx(1 - (1 - g.worst.gamma) * timing.READ_BYTES_PER_S / timing.SPEC_BYTES_PER_S)
        assert 0.058 < spec.gamma < 0.0586
        hbm = timing.stated(timing.GPUS * 94 * 10 ** 9, audit.WINDOW_NS, 10, rtt_ns=rtt * timing.MS, samples=s,
                            read_bytes_per_s=timing.GPUS * timing.GPU_READ_BYTES_PER_S).worst
        assert hbm.timing > idle and 0.0557 < hbm.gamma < 0.0561
        twenty = timing.stated(store, 20 * timing.S, 30, rtt_ns=rtt * timing.MS, samples=11_091).worst
        assert twenty.sample < 0.0025 and 0.012 < twenty.gamma < 0.0137
        one = timing.stated(store, timing.S, 600, rtt_ns=rtt * timing.MS, samples=2_773).worst
        assert one.sample < 0.01 < 0.09 < one.gamma


def test_the_node_constants_are_the_sums_over_its_devices() -> None:
    """R, S_chip, the squaring ceiling and the SHA-512 rate are the node's (MEM-26): its 8 GPUs' and its host's, every
    tier, each GPU's value kept apart for the benches that read one GPU."""
    assert timing.GPUS == 8
    assert timing.READ_BYTES_PER_S == 8 * 1.539e12 + 606.8e9 and timing.SPEC_BYTES_PER_S == 8 * 1.597e12 + 819.2e9
    assert timing.CHIP_BYTES == 8 * 263_163_904 + 1_000_341_504 == 3_105_652_736
    assert timing.HOST_CHIP_BYTES == 2 * (336 * 2 ** 20 + 2 * 2 ** 20 + 64 * (2 * 2 ** 20 + (48 + 64 + 64) * 2 ** 10))
    assert timing.SQUARINGS_PER_S == {w: 8 * r for w, r in timing.GPU_SQUARINGS_PER_S.items()}
    assert timing.SHA512_PER_S == 8 * timing.GPU_SHA512_PER_S
    gpu = 94 * 10 ** 9
    host = timing.spanning(gpu) - 8 * gpu
    assert host / timing.HOST_READ_BYTES_PER_S == pytest.approx(gpu / timing.GPU_READ_BYTES_PER_S)


def test_the_write_ceilings_are_the_sheets_above_every_write_measured() -> None:
    """One GPU's and the host DRAM's write ceilings (MEM-45) are their data sheets' rates, which bound every prover
    where a measurement bounds only the kernels tried, each above every write its run measured: the card's 1,528.98
    GB/s on the device and 56.39 streamed in (r20261020-211937-b9c0), the host's 207.69 (r20261020-212232-6ba5).  Each
    docstring names its runs; a fill of derived codec labels is bounded by the labeller, about 1.05 GB/s a GPU under
    the default codec, and no DPU's is stated."""
    assert timing.GPU_WRITE_BYTES_PER_S == timing.GPU_SPEC_BYTES_PER_S == 1.597e12 > 1528.98e9 > 56.39e9
    assert timing.HOST_WRITE_BYTES_PER_S == timing.HOST_SPEC_BYTES_PER_S == 819.2e9 > 207.69e9
    body = ast.parse(Path(timing.__file__).read_text()).body
    docs = {a.targets[0].id: b.value.value for a, b in zip(body, body[1:])
            if isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Name) and isinstance(b, ast.Expr)
            and isinstance(b.value, ast.Constant)}
    gpu, host = docs["GPU_WRITE_BYTES_PER_S"], docs["HOST_WRITE_BYTES_PER_S"]
    assert "r20261020-211937-b9c0" in gpu and "SQUARINGS_PER_S[width]" in gpu
    assert "r20261020-212232-6ba5" in host and "r20261020-211937-b9c0" in host and "No DPU" in host
    labels = timing.GPU_SQUARINGS_PER_S[codec.DEFAULT.field.name] * codec.BLOCK_BYTES / codec.DEFAULT.encode_squarings
    assert labels == pytest.approx(1.05e9, rel=0.01)
    assert not [name for name in vars(timing) if name.startswith("DPU_")]


def test_the_ledgers_count_bounds_auditros_binomial() -> None:
    """`named_bits`, λ's count a pass (`PROTOCOL.md` §10), against `Accounting.Guarantees.Remote.AuditRO`'s last term
    C(q, k)/2^λ, λ a natural number: charged `named_bits(k) + SLACK_LOG2`, a pass's term is at most 2^-SLACK_LOG2,
    checked in exact integers against `math.comb`.  It is within 2 bits of log2(q^k/k!) (exact factorials up to 10^5),
    never above k·log2 q, and at q = 2^40 at most the 40·k AuditRO's docstring reads.  At 94 GB of 4 KiB receipts and
    q = 2^64 it is 40.99 bits a receipt, at the node's 789 GB over every tier 37.92, and the grid's sizing reads the
    same count."""
    cases = [(q_log2, k) for q_log2 in (1, 10, 40) for k in (0, 1, 2, 3, 7, 2 ** q_log2 // 2, 2 ** q_log2)
             if k <= 2 ** q_log2 and k <= 4096] + [(20, 1000), (64, 3), (64, 3000), (64, 10 ** 4), (64, 10 ** 5)]
    for q_log2, k in cases:
        named = audit.named_bits(k, q_log2)
        slack = audit.SLACK_LOG2
        assert math.comb(2 ** q_log2, k) * 2 ** slack <= 2 ** (named + slack) and named <= k * q_log2
        assert named - (k * q_log2 - math.log2(math.factorial(k))) < 2
        if q_log2 == 40:
            assert named <= 40 * k
        assert timing._named_bits(k, q_log2, True) == named and timing._named_bits(k, q_log2, False) == k * q_log2
    k = 94 * 10 ** 9 // codec.BLOCK_BYTES
    assert 40.98 < audit.named_bits(k) / k < 41.0 and audit.named_bits(k) < 64 * k
    k = timing.spanning(94 * 10 ** 9) // codec.BLOCK_BYTES
    assert 37.92 < audit.named_bits(k) / k < 37.93
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError, match="nonnegative integers"):
            audit.named_bits(bad)
    with pytest.raises(ValueError, match="nonnegative integers"):
        audit.named_bits(1, -1)


def test_auditros_slack_adds_at_most_2_to_the_minus_50_over_an_hour_of_the_fastest_passes() -> None:
    """`SLACK_LOG2`'s reason: an hour of the node's passes over eight GPUs' 62 GiB shards, 130,023,424 blocks read at
    `timing.READ_BYTES_PER_S`, is under 2^17 passes, and their slack adds at most 2^-50 to γ's error and the memory
    line's δ, 2^-10 of `DELTA`."""
    pass_s = 130_023_424 * codec.BLOCK_BYTES / timing.READ_BYTES_PER_S
    passes = math.ceil(3600 / pass_s)
    assert 41.2e-3 < pass_s < 41.3e-3 and passes == 87_326 < 2 ** 17
    assert 2 ** 17 * 2.0 ** -audit.SLACK_LOG2 <= 2.0 ** -50 == audit.DELTA / 2 ** 10


def test_below_the_store_floor_no_window_meets_one_percent() -> None:
    """`store_floor`'s closed form against the ledger: a single 20 min window at 555 openings a second, whose μ_sample
    and ends are under 0.01%, is over 1% just below the floor at g = 50 µs and under it just above.  At g = 0 the floor
    is μ_chip, λ and μ_rebuild alone, about 380 GB over the node's every tier."""
    assert 380.4e9 < timing.store_floor() < 380.6e9 and 458.3e9 < timing.store_floor(50 * timing.US) < 458.5e9
    for scale, over in ((0.98, True), (1.02, False)):
        g = timing.stated(round(scale * timing.store_floor(50 * timing.US)), 1200 * timing.S, 1, rtt_ns=timing.MS,
                          samples=555 * 1200)
        assert (g.worst.gamma > 0.01) is over


@pytest.mark.parametrize("pass_ms, rtt_ms, window_ms", [(20, 1, 300), (20, 50, 1000), (50, 5, 1000), (37, 20, 300)])
def test_honest_schedules_fill_their_windows_to_the_chained_steady_state(pass_ms, rtt_ms, window_ms) -> None:
    """Every window certifies at least `⌊(T − p − g)/(p + g)⌋` passes, and the span the passes leave uncovered is `g`
    after each pass and the session's two ends, which come to at most `A + J`.  A window over its own span `T + A` is
    charged its ends too, and the rest of the pass in flight at its challenge: at most `A + J + p` more."""
    ms, gap, jitter = timing.MS, 50_000, 50_000
    kw = dict(gap_ns=gap, down_ns=rtt_ms * ms // 2, up_ns=rtt_ms * ms - rtt_ms * ms // 2, jitter_ns=jitter)
    st = timing.honest(window_ms * ms, 40, pass_ms * ms, **kw)
    assert min(st.passes) >= (window_ms * ms - pass_ms * ms - gap) // (pass_ms * ms + gap)
    a = timing.deadline(pass_ms * ms, **kw)
    mu = 1 - st.utilization
    steady = timing.chained(pass_ms * ms, gap)
    assert steady <= mu <= steady + (a + jitter) / st.span_ns
    assert steady <= min(st.window_timing) and max(st.window_timing) <= steady + (a + jitter + pass_ms * ms) / (
        window_ms * ms + a)


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
