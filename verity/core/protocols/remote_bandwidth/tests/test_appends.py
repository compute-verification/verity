"""Blocks written after the codec seed, the KV pages of a serving node: appended mid-session under their bound keys,
read by plans sent as changes, released, opened with the digests their keys come from, and credited only on the
property's verdict."""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib

import pytest

from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts, schedule
from verity.core.protocols.remote_bandwidth.audit import Params, Session
from verity.core.protocols.remote_bandwidth.receipts import Plan

N = 8
CAP = 256
"""The cap on its schedule's bits that a test session commits before its coins (`Params.schedule_cap`): above
`HONEST_SCHEDULE`'s 200, so a ledger crediting the schedule's own bits instead would show."""
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"
P512 = codec.Codec(codec.P512, 2, False)


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Entropy:
    """SHA-512 in counter mode under `label`, in place of the operating system's generator."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(label: bytes) -> bytes:
    """64-byte pieces with their top byte cleared, so every word is below p."""
    return b"".join(hashlib.sha512(label + b"/%d" % u).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


def salts(label: bytes, n: int) -> list[bytes]:
    entropy = Entropy(label)
    return [entropy(receipts.SALT_BYTES) for _ in range(n)]


@functools.cache
def registered() -> audit.Store:
    payloads = [payload(b"weights %d" % j) for j in range(N)]
    ys = salts(b"payload salts", N)
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    return audit.Store(audit.register(SOURCE, NODE, N, root, P512, nonce=bytes(32)), payloads, ys)


def store() -> audit.Store:
    """A store of N registered blocks under p512/k2 with no appends yet, its own to append to."""
    out = copy.copy(registered())
    out.blocks, out.appends, out.pending = list(out.blocks), [], {}
    return out


def pages(first: int, count: int) -> list[bytes]:
    return [payload(b"kv page %d" % j) for j in range(first, first + count)]


class Run:
    """One session over `store`, the verifier's side and the prover's in step, every receipt drawn by default.  The
    prover computes its passes from `prover_store` (`store` by default) and commits `store`'s appends, each encoded
    under the coin the verifier replies with (`coins`, by first index), each pass's seed binding every append it sent
    before the pass's root (`head`).  With a `registry`, the session is its next over the store's registration.  Its
    parameters commit `schedule_cap` (`CAP` by default) on its schedule's bits."""

    def __init__(self, store: audit.Store, prover_store: audit.Store | None = None, windows: int = 2,
                 samples: int = 10**6, lifetime: int | None = None, registry: audit.Registry | None = None,
                 deferred: bool = False, appends: int = coins.APPENDS, schedule_cap: int | None = CAP) -> None:
        reg = store.registration
        self.p = Params(blocks=reg.blocks, deadline_ns=10**6, windows=windows, samples=samples, delta=1e-6,
                        window_ns=10, codec=reg.codec, lifetime=lifetime, deferred=deferred,
                        schedule_cap=schedule_cap)
        self.clock = Clock()
        if registry is None:
            self.s = Session(self.p, reg, None, coins.local(reg.digest, 0, self.p.windows, Entropy(b"coins"),
                                                            appends=appends), not_before_ns=0, clock=self.clock)
        else:
            self.s = registry.session(self.p, reg, None, not_before_ns=0, clock=self.clock, entropy=Entropy(b"coins"))
        self.store = store
        self.prover = audit.Prover(prover_store or store, None, self.s.commitment, entropy=Entropy(b"receipt salts"))
        self.challenges = [self.s.start()]
        self.t, self.i, self.previous, self.plan, self.step = 0, 0, reg.digest, Plan.whole(reg.blocks), -1
        self.head: bytes | None = None
        self.passes: dict[tuple[int, int], audit.Pass] = {}
        self.coins: dict[int, bytes] = {}
        self.received: list[tuple[int, int, bytes]] = []

    def commit(self, root: bytes, count: int) -> int:
        """An append's root sent (`Session.append`): the index the verifier assigns, its coin kept, the append as the
        verifier received it, (first, count, root), kept in `received` and, if the store awaits that append's coin,
        its blocks encoded under it."""
        first, coin = self.s.append(root, count)
        self.coins[first] = coin
        self.received.append((first, count, root))
        self.head = audit.append_head(self.head, first, count, root)
        if first in self.store.pending:
            self.store.encode(first, coin)
        return first

    def append(self, count: int) -> audit.Append:
        a = self.store.append(pages(self.store.extent, count), salts(b"page salts %d" % self.store.extent, count))
        assert self.commit(a.root, a.count) == a.first
        return a

    def run(self, plan: Plan | None = None, change: tuple | None = None, sent=None, step: int | None = None
            ) -> audit.Verdict | None:
        """One pass, its plan sent whole (None: the registered store) or as its `change` (added, removed), covering
        `step` (None: the step after the last pass's)."""
        if change is not None:
            plan = self.plan.changed(*change, self.store.extent)
        ps = self.prover.run(self.challenges[self.t], self.i, self.previous, plan, head=self.head)
        self.passes[self.t, self.i], self.previous, self.plan = ps, ps.root, ps.plan
        self.i += 1
        self.step = self.step + 1 if step is None else step
        if sent is not None:
            return self.s.receive(self.t, self.i - 1, ps.root, sent, step=self.step)
        if change is not None:
            return self.s.receive(self.t, self.i - 1, ps.root, change=change, step=self.step)
        return self.s.receive(self.t, self.i - 1, ps.root, None if plan is None else plan.runs, step=self.step)

    def boundary(self) -> None:
        self.clock.now += self.p.window_ns
        coin = self.s.boundary()
        if coin is not None:
            self.challenges.append(coin)
        self.t, self.i = self.t + 1, 0

    def finish(self, open_=None) -> list[audit.Verdict]:
        self.clock.now = self.s.deadline(len(self.s.windows) - 1) + 1
        out = []
        for t in range(self.p.windows):
            draws = self.s.close(t)
            out.append(self.s.verify(t, [(open_ or self.prover.open)(self.passes[t, i], q) for i, q in draws]))
            if not out[-1].accepted:
                break
        return out


def honest_session(schedule_cap: int | None = CAP) -> Run:
    """Window 0: a pass over the registered store; 4 pages appended (8 … 11) and read; one of them and a registered
    block released and dropped from the plan.  Window 1: 2 more pages (12, 13) read with the rest, then a pass sent
    whole."""
    r = Run(store(), schedule_cap=schedule_cap)
    assert r.run() is None
    r.append(4)
    assert r.run(change=(((N, 4),), ())) is None
    assert r.s.release(((2, 1),)) is None and r.s.release(((9, 1),)) is None
    assert r.run(change=((), ((2, 1), (9, 1)))) is None
    r.boundary()
    r.append(2)
    assert r.run(change=(((12, 2),), ())) is None
    assert r.run(Plan(((0, 2), (3, 6), (10, 4)), r.store.extent)) is None
    r.boundary()
    return r


HONEST_SCHEDULE = schedule.Schedule(((2, 1, 1),), (((0, None), (0, 1), (0, None), (0, None)), ((2, None), (2, None))),
                                    (0, 1, 2, 3, 4))
"""`honest_session`'s schedule: registered block 2 freed at step 1; pages 8 … 11 written at step 0, page 9 freed at step
1; pages 12 and 13 written at step 2; passes at steps 0 … 4."""


@pytest.fixture(scope="module")
def honest() -> tuple[Run, list[audit.Verdict]]:
    r = honest_session()
    return r, r.finish()


def test_an_honest_session_over_appended_and_released_pages_is_accepted(honest) -> None:
    """Every receipt is drawn, the appended pages' among them, each opened with its payload's digest and its path to
    its append's root, and every window is accepted.  The record counts each pass's post-seed reads, the blocks
    appended and those released."""
    r, verdicts = honest
    assert verdicts == [audit.Verdict(0, True), audit.Verdict(1, True)]
    assert [p.plan.runs for p in r.passes.values()] == [((0, 8),), ((0, 12),), ((0, 2), (3, 6), (10, 2)),
                                                        ((0, 2), (3, 6), (10, 4)), ((0, 2), (3, 6), (10, 4))]
    rec = r.s.record()
    assert [w.lengths for w in rec.windows] == [(8, 12, 10), (12, 12)]
    assert [w.post_seed for w in rec.windows] == [(0, 4, 3), (5, 5)]
    assert (rec.appended, rec.released) == (6, 2)
    pub = rec.public()
    assert (pub["appended"], pub["released"], pub["post_seed"], pub["receipts"]) == (6, 2, [7, 10], [30, 24])
    ps = r.passes[0, 1]
    o, registered = r.prover.open(ps, 9), r.prover.open(ps, 3)
    assert ps.plan.block(9) == 9 and o.digest == codec.digest(pages(9, 1)[0]) and len(o.digest) == 64
    assert registered.digest == b""
    assert o.size == registered.size + 64 + 64 * (len(o.payload_path) - len(registered.payload_path))
    k = r.store.registration.key
    assert r.store.blocks[9] == codec.encode(pages(9, 1)[0], 9, k, bound=True, coin=r.coins[8])
    assert r.store.blocks[9] != codec.encode(pages(9, 1)[0], 9, k, bound=True)


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The property's verdict on the post-seed blocks' writers: whether it holds, and its carrier bytes."""
    holds: bool
    carrier: float = 0.0

    def carrier_bytes(self, read_bytes: int) -> float:
        return self.carrier


def test_the_ledger_credits_post_seed_receipts_only_on_the_property_s_verdict(honest) -> None:
    """Without a verdict (None), the session's 17 post-seed receipts are certified but not credited: μ_timing counts
    only the registered blocks' 37.  On a verdict that holds, all 54 are credited, and μ_carrier charges each pass the
    least of the verdict's carrier bytes and its post-seed bytes, with the session's schedule check beside it, whose
    cap it reports as the steering: the one committed before the coins, not the 200 bits of the schedule published
    after them.  A verdict that fails is refused."""
    rec = honest[0].s.record()
    scheduled = honest[0].s.scheduled(HONEST_SCHEDULE)
    capacity = 4096 * 10**9 / 10
    common = dict(capacity_bytes_per_s=capacity, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    whole = capacity * rec.span_ns / 1e9
    none = audit.ledger(rec, **common)
    assert none.timing == pytest.approx(1 - 37 * 4096 / whole) and none.carrier == 0
    assert set(none.to_json()["terms"]) == set(audit.LEDGER_SOURCES) and "carrier" in audit.LEDGER_SOURCES
    assert none.steering is None and none.to_json()["steering_bits"] is None
    held = audit.ledger(rec, **common, credit_post_seed=Verdict(True, 3 * 4096), schedule=scheduled)
    assert HONEST_SCHEDULE.bits == 200 and held.steering == scheduled.cap == CAP == held.to_json()["steering_bits"]
    assert held.timing == pytest.approx(1 - 54 * 4096 / whole)
    assert held.carrier == pytest.approx((0 + 3 + 3 + 3 + 3) * 4096 / whole)
    assert held.sample == none.sample and held.counting == none.counting
    whole_0 = capacity * rec.windows[0].span_ns / 1e9
    assert held.windows[0].carrier == pytest.approx(6 * 4096 / whole_0)
    assert none.windows[0].timing - held.windows[0].timing == pytest.approx(7 * 4096 / whole_0)
    assert held.gamma == pytest.approx(none.gamma - 17 * 4096 / whole + 12 * 4096 / whole)
    with pytest.raises(ValueError, match="fails"):
        audit.ledger(rec, **common, credit_post_seed=Verdict(False, 0.0), schedule=scheduled)
    with pytest.raises(ValueError, match="finite nonnegative"):
        audit.ledger(rec, **common, credit_post_seed=Verdict(True, float("inf")), schedule=scheduled)


@dataclasses.dataclass(frozen=True)
class Link:
    """A requirement link that holds or not, the circuit reading `reads[step]` bits from complete blocks at a step
    (none otherwise), its memory-map unit's verdict `maps` on the records it is asked about, each kept in `asked`, and
    its units' draw leaving `epsilon` of its credit unproved except with `delta` (every unit proved by default)."""
    holds: bool
    reads: dict
    maps: object = True
    epsilon: object = 0.0
    delta: object = 0.0
    asked: list = dataclasses.field(default_factory=list)

    def read_bits(self, step: int):
        return self.reads.get(step, 0)

    def mapped(self, record: audit.Record):
        self.asked.append(record)
        return self.maps


def planned(record: audit.Record) -> dict:
    """Each pass's step, the circuit reading every bit of its plan's blocks there."""
    return {s: n * 8 * record.block_bytes for w in record.windows for n, s in zip(w.lengths, w.steps)}


def units_mapped(record: audit.Record, appends, rekeys=()) -> bool:
    """The memory-map unit's verdict as the record fixes it, for a map whose appends are `appends`, (step, first,
    count, root) in the map's order, and whose re-keys are `rekeys`, (step, old index): the chain over all of them is
    the record's, each pass's head is the chain through those of the steps below its own, and each re-key's old index
    is below the stale bound of the first pass after its step, which the unit recomputes from its appends
    (`stale_of`) rather than reads."""

    def chain(below: int) -> bytes | None:
        head = None
        for s, first, count, root in appends:
            if s < below:
                head = audit.append_head(head, first, count, root)
        return head

    steps = [s for w in record.windows for s in w.steps]
    if record.appends != chain(audit.STEP_LIMIT) or any(
            h != chain(s) for w in record.windows for s, h in zip(w.steps, w.heads, strict=True)):
        return False
    bounds = [b for w in stale_of(record, [(f, c, root) for _, f, c, root in appends]) for b in w]
    return all(next((b for s, b in zip(steps, bounds) if s > r), -1) > old for r, old in rekeys)


def stale_of(record: audit.Record, received) -> list[tuple[int, ...]]:
    """Each pass's stale bound as the memory-map unit recomputes it inside its proof, from the appends, (first, count,
    root) in the order the verifier received them, and the record's `blocks`, `lifetime` and heads, never its
    `stale`: an append joined the window of the first pass whose head chains it, and a pass of window t is bounded at
    `blocks` plus the counts of the appends that joined window t − `lifetime` or before."""
    chain, through = None, {None: 0}
    for n, (first, count, root) in enumerate(received, 1):
        chain = audit.append_head(chain, first, count, root)
        through[chain] = n
    joined: dict[int, int] = {}
    for w in record.windows:
        for h in w.heads:
            for a in range(through[h]):
                joined.setdefault(a, w.window)
    return [tuple(record.blocks + sum(c for a, (_, c, _) in enumerate(received)
                                      if joined[a] <= w.window - record.lifetime) for _ in w.heads)
            for w in record.windows]


@dataclasses.dataclass(frozen=True)
class Units(Link):
    """A link whose memory-map unit makes `appends` and `rekeys` (`units_mapped`'s), its verdict on a record whether
    the record is that map's."""
    appends: tuple = ()
    rekeys: tuple = ()

    def mapped(self, record: audit.Record):
        self.asked.append(record)
        return units_mapped(record, self.appends, self.rekeys)


def test_on_a_requirement_link_a_pass_is_credited_the_circuit_s_reads_at_its_step(honest) -> None:
    """`RequirementMoved`: a pass whose credited receipts cover its plan is credited the circuit's reads from complete
    blocks at its step, and one short of its plan those less a block for each planned block it credits no receipt of.
    The honest session's passes are steps 0 … 4, plans of (8, 12, 10) and (12, 12) blocks, of which (0, 4, 3) and
    (5, 5) are post-seed.  A link reading every bit of each plan changes nothing but `linked`, with the property's
    verdict or without it (each uncredited post-seed block then costs its block).  A sparse link, its plans topped up
    with filler that carries none of its reads (the fewest bits a planned block carries, D, is 0): step 0 reads 3
    blocks and 100 bits of its 8, all credited; step 1 5 blocks of its 12, 4 of them uncredited, so 1 block, or all 5
    on the verdict; step 2 20,000 bits of its 10, 3 uncredited, so nothing, or 20,000 bits on the verdict; steps 3 and
    4 nothing.  A link that doesn't hold, reads past its plan's blocks or that aren't a count, or a record whose passes
    name no step are refused."""
    rec = honest[0].s.record()
    scheduled = honest[0].s.scheduled(HONEST_SCHEDULE)
    assert [w.steps for w in rec.windows] == [(0, 1, 2), (3, 4)]
    assert [w.lengths for w in rec.windows] == [(8, 12, 10), (12, 12)]
    capacity, B = 4096 * 10**9 / 10, 8 * 4096
    common = dict(capacity_bytes_per_s=capacity, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    whole, whole_0 = capacity * rec.span_ns / 1e9, capacity * rec.windows[0].span_ns / 1e9
    for verdict in ({}, {"credit_post_seed": Verdict(True), "schedule": scheduled}):
        store_ = audit.ledger(rec, **common, **verdict)
        full = audit.ledger(rec, **common, **verdict, requirement=Link(True, planned(rec)))
        assert full.timing == pytest.approx(store_.timing) and full.linked and not store_.linked
        assert full.to_json()["linked"] is True and store_.to_json()["linked"] is False
    sparse = Link(True, {0: 3 * B + 100, 1: 5 * B, 2: 20_000})
    assert [audit.linked_bits(w, sparse, 4096, False) for w in rec.windows] == [(3 * B + 100, B, 0), (0, 0)]
    assert [audit.linked_bits(w, sparse, 4096, True) for w in rec.windows] == [(3 * B + 100, 5 * B, 20_000), (0, 0)]
    bare = audit.ledger(rec, **common, requirement=sparse)
    held = audit.ledger(rec, **common, credit_post_seed=Verdict(True), requirement=sparse, schedule=scheduled)
    assert bare.timing == pytest.approx(1 - (4 * B + 100) / 8 / whole)
    assert held.timing == pytest.approx(1 - (8 * B + 20_100) / 8 / whole)
    assert held.windows[0].timing == pytest.approx(1 - (8 * B + 20_100) / 8 / whole_0)
    assert held.windows[1].timing == pytest.approx(1)
    assert bare.sample == store_.sample and bare.counting == store_.counting
    for link, match in ((Link(False, {}), "don't hold"), (Link(True, {1: 12 * B + 1}), "more than its pass's plan"),
                        (Link(True, {2: -1}), "nonnegative integer"), (Link(True, {2: 1.5}), "nonnegative integer"),
                        (Link(True, {2: True}), "nonnegative integer")):
        with pytest.raises(ValueError, match=match):
            audit.ledger(rec, **common, requirement=link)
    bare = synthetic((10, 12), ())
    with pytest.raises(ValueError, match="name none"):
        audit.ledger(bare, **dict(common, rebuild=audit.Rebuild(bare.codec, 0.0)), requirement=Link(True, {}))


def test_a_registered_session_with_filler_planned_is_credited_the_circuit_s_reads() -> None:
    """A session whose link covers the weights while KV isn't registered.  The store's 8 registered blocks are
    4 of weights (the last half full), an embedding table of 16 rows in 2, and 2 of filler, and every pass plans the
    whole store (PlanShard at GAPS = 0).  Each of the 5 steps reads every weight, the table row its token selects and
    the KV its earlier steps wrote, which no registered block holds and which is never complete here.  So each plan
    holds every complete block its step reads (`PlanMatches`), beside filler and the unselected table block, which
    carry none of its reads: D is 0, and a credit of D bits a receipt would be nothing.  The ledger credits each pass
    its step's reads from complete blocks, 14 weights of 8,192 bits and a row of 4,096, 118,784 bits of the plan's
    262,144; the KV reads `Requirement.read_bits` adds from step 1 on are not the link's (`read_bits` is
    `creditedReadBits`)."""
    from verity.ml.circuits import requirement as RQ

    steps, weights, rows, tokens = 5, range(14), range(100, 116), (3, 3, 10, 12, 0)
    kv = {200 + t: t for t in range(steps)}
    width = {**{v: 8192 for v in weights}, **{v: 4096 for v in rows}, **{v: 2048 for v in kv}}
    req = RQ.Requirement(width, frozenset((*weights, *rows)), {**{v: -1 for v in (*weights, *rows)}, **kv},
                         {v: steps - 1 for v in width},
                         {t: frozenset((*weights, 100 + tokens[t], *(200 + u for u in range(t)))) for t in range(steps)},
                         steps, {})
    layout = RQ.Layout({**{v: v // 4 for v in weights}, **{v: 4 + (v - 100) // 8 for v in rows},
                        **{v: 100 + t for v, t in kv.items()}},
                       {**{v: v % 4 * 8192 for v in weights}, **{v: (v - 100) % 8 * 4096 for v in rows},
                        **{v: 0 for v in kv}},
                       {100 + t: audit.STEP_LIMIT for t in kv.values()})
    r = Run(store())
    for s in range(steps):
        assert r.run() is None
        if s == 2:
            r.boundary()
    r.boundary()
    assert [v.accepted for v in r.finish()] == [True, True]
    rec = r.s.record()
    assert [w.lengths for w in rec.windows] == [(8, 8, 8), (8, 8)] and [w.steps for w in rec.windows] == [(0, 1, 2),
                                                                                                          (3, 4)]
    credited = {s: req.bits(RQ.credited_reads(req, layout, s)) for s in range(steps)}
    for s, p in enumerate(r.passes.values()):
        assert RQ.plan_matches(req, layout, p.plan, RQ.BLOCK_BITS, s)
        assert min(layout.bits_in(req, RQ.credited_reads(req, layout, s), j) for j in p.plan) == 0
        assert RQ.read_blocks(req, layout, s) != frozenset(p.plan)
        assert req.read_bits(s) == credited[s] + 2048 * s
    assert credited == {s: 14 * 8192 + 4096 for s in range(steps)}
    capacity = 4096 * 10**9 / 10
    common = dict(capacity_bytes_per_s=capacity, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    whole = capacity * rec.span_ns / 1e9
    linked = audit.ledger(rec, **common, requirement=Link(True, credited))
    assert linked.linked and linked.timing == pytest.approx(1 - steps * (14 * 8192 + 4096) / 8 / whole)
    assert audit.ledger(rec, **common).timing == pytest.approx(1 - 40 * 4096 / whole)


def test_a_requirement_link_caps_nothing_unless_its_memory_map_unit_makes_the_record_s_appends(honest) -> None:
    """Frees, re-keys and append order chosen after the seed cost no encode, so the advice they give is zero only if
    W's memory map fixes them, which the memory-map unit proves.  The record carries the chain over the session's
    appends (public: a hash over hiding roots and their indices), the link's `mapped` is asked about the record, and a
    link whose unit doesn't make those appends is refused."""
    r, _ = honest
    rec = r.s.record()
    assert r.head is not None and rec.appends == r.head and rec.public()["appends"] == r.head.hex()
    common = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    link = Link(True, {})
    assert audit.ledger(rec, **common, requirement=link).linked and link.asked == [rec]
    for maps in (False, None, 1):
        with pytest.raises(ValueError, match="memory-map unit"):
            audit.ledger(rec, **common, requirement=Link(True, {}, maps))
    none = Link(True, {})
    audit.ledger(dataclasses.replace(rec, appends=None), **common, requirement=none)
    assert [x.appends for x in none.asked] == [None]
    other = [(0, a.first, a.count, a.root) for a in r.store.appends]
    with pytest.raises(ValueError, match="memory-map unit"):
        audit.ledger(rec, **common, requirement=Units(True, {}, appends=tuple(other)))


def test_a_record_carries_each_pass_s_head_and_stale_bound(honest) -> None:
    """Each pass's head is the chain over the appends received before its root, which its seed binds: none before
    pages 8 … 11, then theirs through window 0, then theirs and 12 … 13's through window 1.  Its stale bound is N
    throughout, since no block reaches L (30 s of 10 ns windows), and the received appends give the same.  The public record lists the
    heads per window (hiding commitments), the verifier's profile heads and bounds both, and the honest session's
    map, step 0 appending 8 … 11 and step 2 12 … 13, is the units' at every pass.  A link reads the heads, not the
    bounds, which its unit recomputes: a record without heads is refused one, and one without bounds is not."""
    r, _ = honest
    rec = r.s.record()
    first, second = r.store.appends
    h1 = audit.append_head(None, first.first, first.count, first.root)
    h2 = audit.append_head(h1, second.first, second.count, second.root)
    assert [w.heads for w in rec.windows] == [(None, h1, h1), (h2, h2)] and h2 == rec.appends
    assert [w.stale for w in rec.windows] == [(N, N, N), (N, N)] == stale_of(rec, r.received)
    assert rec.public()["heads"] == [[None, h1.hex(), h1.hex()], [h2.hex(), h2.hex()]]
    assert rec.profile()["windows"][1]["heads"] == [h2.hex(), h2.hex()]
    assert rec.profile()["windows"][1]["stale"] == [N, N]
    assert units_mapped(rec, [(0, first.first, first.count, first.root), (2, second.first, second.count, second.root)])
    assert not units_mapped(rec, [(1, first.first, first.count, first.root),
                                  (2, second.first, second.count, second.root)])
    common = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))

    def shorn(**field) -> audit.Record:
        return dataclasses.replace(rec, windows=(dataclasses.replace(rec.windows[0], **field), rec.windows[1]))

    with pytest.raises(ValueError, match="name none"):
        audit.ledger(shorn(heads=()), **common, requirement=Link(True, {}))
    assert audit.ledger(shorn(stale=()), **common, requirement=Link(True, {})).linked


def test_the_public_record_has_no_stale_bound(honest) -> None:
    """A stale bound sums the counts of the appends each window joined, which follow the workload, so it is a value:
    `public()` has no `stale`, and is the same whatever the windows' bounds are."""
    rec = honest[0].s.record()
    assert "stale" not in rec.public() and all(w.stale for w in rec.windows)
    moved = dataclasses.replace(rec, windows=tuple(
        dataclasses.replace(w, stale=tuple(b + 1 for b in w.stale)) for w in rec.windows))
    assert moved.public() == rec.public() and moved.profile() != rec.profile()


def test_a_stale_bound_is_recomputed_from_the_received_appends_the_heads_and_l() -> None:
    """L = 1, two passes a window.  Pages 8 and 9 join window 0; 10 and then 11, 12 join window 1, between its passes;
    13 joins window 2 between its passes.  Window 1 is bounded past 8 and 9 (N + 2), window 2 past 8 … 12 (N + 5),
    whether its pass's root comes before 13's append (where the session takes the store's extent) or after (13's first
    index).  Recomputed as the memory-map unit does, from the appends as the verifier received them and the record's
    `blocks`, `lifetime` and heads, the same bounds."""
    r = Run(store(), windows=3, lifetime=1)
    assert r.run() is None
    r.append(2)
    assert r.run(change=(((N, 2),), ())) is None
    r.boundary()
    r.append(1)
    assert r.s.release(((N, 2),)) is None
    assert r.run(change=(((N + 2, 1),), ((N, 2),))) is None
    r.append(2)
    assert r.run(change=(((N + 3, 2),), ())) is None
    r.boundary()
    assert r.s.release(((N + 2, 3),)) is None
    assert r.run(change=((), ((N + 2, 3),))) is None
    r.append(1)
    assert r.run(change=(((N + 5, 1),), ())) is None
    r.boundary()
    assert [v.accepted for v in r.finish()] == [True] * 3
    rec = r.s.record()
    assert [w.stale for w in rec.windows] == [(N, N), (N + 2, N + 2), (N + 5, N + 5)] == stale_of(rec, r.received)


def appended_at_step_one(early: bool) -> Run:
    """Passes at steps 0, 1 and 2 over one window, step 1 appending pages 8 and 9, which step 2's pass reads: sent
    after step 1's root, or `early`, before it."""
    r = Run(store(), windows=1)
    assert r.run(step=0) is None
    if early:
        r.append(2)
    assert r.run(step=1) is None
    if not early:
        r.append(2)
    assert r.run(change=(((N, 2),), ()), step=2) is None
    r.boundary()
    assert r.finish() == [audit.Verdict(0, True)]
    return r


def test_mapped_refuses_an_append_sent_before_its_step_s_pass() -> None:
    """An early append: step 1's append sent before step 1's root binds it into that pass's seed.  The
    session accepts either order and the chain over the appends is the same, so the chain alone can't tell them
    apart; step 1's head can, and a units link whose map appends at step 1 refuses the early session.  An append the
    map dates at step 0 isn't the honest session's either: the heads pin each append between two passes."""
    common = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    honest, early = appended_at_step_one(False), appended_at_step_one(True)
    a = honest.store.appends[0]
    h = audit.append_head(None, a.first, a.count, a.root)
    assert early.store.appends[0].root == a.root and honest.s.record().appends == early.s.record().appends == h
    assert [w.heads for w in honest.s.record().windows] == [(None, None, h)]
    assert [w.heads for w in early.s.record().windows] == [(None, h, h)]
    units = ((1, a.first, a.count, a.root),)
    assert audit.ledger(honest.s.record(), **common, requirement=Units(True, {}, appends=units)).linked
    with pytest.raises(ValueError, match="memory-map unit"):
        audit.ledger(early.s.record(), **common, requirement=Units(True, {}, appends=units))
    assert not units_mapped(honest.s.record(), ((0, a.first, a.count, a.root),))
    assert units_mapped(early.s.record(), ((0, a.first, a.count, a.root),))


@pytest.mark.parametrize("deferred", [False, True], ids=["resident", "deferred"])
def test_a_re_key_s_old_index_is_below_the_stale_bound_of_the_pass_after_its_step(deferred) -> None:
    """L = 2, a pass a window at steps 1 … 4.  Step 0 appends pages 8 and 9, which join window 0, and step 1 page 10,
    which joins window 1; step 2 re-keys 8 and 9 (released, their payloads appended again at 11 and 12, read from
    window 2, or, deferred, from window 3) and step 3 frees 10.  The stale bounds are N, N, N + 2 and N + 3: step 2's
    re-key is at the cap, its old indices below the bound of step 3's pass, and so is one of 10 at step 3, but one of
    10 at step 2 is a window early (10 is not below N + 2), which the units refuse though the session accepts it."""
    r = Run(store(), windows=4, lifetime=2, deferred=deferred)
    r.append(2)
    assert r.run(change=(((N, 2),), ()), step=1) is None
    r.boundary()
    r.append(1)
    assert r.run(change=(((N + 2, 1),), ()), step=2) is None
    r.boundary()
    a = r.store.append(pages(N, 2), salts(b"re-keyed", 2))
    assert r.commit(a.root, a.count) == N + 3 and r.s.release(((N, 2),)) is None
    landed, late = (((N + 3, 2),), ()) if not deferred else ((), ((N + 3, 2),))
    assert r.run(change=(landed, ((N, 2),)), step=3) is None
    r.boundary()
    assert r.s.release(((N + 2, 1),)) is None
    assert r.run(change=(late, ((N + 2, 1),)), step=4) is None
    r.boundary()
    assert [v.accepted for v in r.finish()] == [True] * 4
    rec = r.s.record()
    assert [w.stale for w in rec.windows] == [(N,), (N,), (N + 2,), (N + 3,)] == stale_of(rec, r.received)
    appends = [(s, x.first, x.count, x.root) for s, x in zip((0, 1, 2), r.store.appends)]
    assert units_mapped(rec, appends) and units_mapped(rec, appends, ((2, N), (2, N + 1)))
    assert units_mapped(rec, appends, ((2, N), (2, N + 1), (3, N + 2)))
    assert not units_mapped(rec, appends, ((2, N), (2, N + 1), (2, N + 2)))
    common = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    at_cap = Units(True, {}, appends=tuple(appends), rekeys=((2, N), (2, N + 1)))
    assert audit.ledger(rec, **common, requirement=at_cap).linked
    with pytest.raises(ValueError, match="memory-map unit"):
        audit.ledger(rec, **common, requirement=dataclasses.replace(at_cap, rekeys=((2, N), (2, N + 1), (2, N + 2))))


def test_a_link_credits_what_its_units_draw_proved_and_adds_its_delta_to_the_error(honest) -> None:
    """The plan and memory-map units are sampled: a link whose draw leaves ε of its credit unproved, except with δ,
    credits 1 − ε of the session's linked credit, 37 blocks (each plan read whole, less its uncredited post-seed
    blocks), and each window its own (23 and 14) less ε of the session's,
    since all of the unproved share may be one window's.  Its δ adds to both readings' error, which is otherwise every
    window's δ (or the horizon's one) and 2^-slack_log2 a pass.  ε = 1 credits nothing; an ε outside 0 … 1 or a δ
    outside [0, 1) is refused."""
    rec = honest[0].s.record()
    capacity = 4096 * 10**9 / 10
    common = dict(capacity_bytes_per_s=capacity, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    whole, b = capacity * rec.span_ns / 1e9, 4096
    proved = audit.ledger(rec, **common, requirement=Link(True, planned(rec)))
    sampled = audit.ledger(rec, **common, requirement=Link(True, planned(rec), epsilon=0.25, delta=2.0 ** -30))
    assert 1 - proved.timing == pytest.approx(37 * b / whole)
    assert 1 - sampled.timing == pytest.approx(0.75 * 37 * b / whole)
    for w, own in zip(sampled.windows, (23, 14)):
        assert 1 - w.timing == pytest.approx((own - 0.25 * 37) * b / (capacity * w.span_ns / 1e9))
    assert sampled.sample == proved.sample and sampled.counting == proved.counting
    beside = rec.passes * 2.0 ** -audit.SLACK_LOG2
    assert proved.error == pytest.approx(sum(w.profile.delta for w in rec.windows) + beside, rel=1e-12)
    assert proved.error_horizon == pytest.approx(rec.windows[0].profile.delta + beside, rel=1e-12)
    assert audit.ledger(rec, **common).error == proved.error
    assert sampled.error - proved.error == pytest.approx(2.0 ** -30, rel=1e-9)
    assert sampled.error_horizon - proved.error_horizon == pytest.approx(2.0 ** -30, rel=1e-9)
    assert sampled.to_json()["error"] == {"windows": sampled.error, "horizon": sampled.error_horizon}
    nothing = audit.ledger(rec, **common, requirement=Link(True, planned(rec), epsilon=1))
    assert nothing.timing == pytest.approx(1) and all(w.timing == pytest.approx(1) for w in nothing.windows)
    for epsilon, delta, match in ((-0.01, 0.0, "epsilon"), (1.5, 0.0, "epsilon"), (float("nan"), 0.0, "epsilon"),
                                  (True, 0.0, "epsilon"), ("0.1", 0.0, "epsilon"), (0.0, 1.0, "delta"),
                                  (0.0, -1e-9, "delta"), (0.0, float("nan"), "delta"), (0.0, None, "delta")):
        with pytest.raises(ValueError, match=match):
            audit.ledger(rec, **common, requirement=Link(True, planned(rec), epsilon=epsilon, delta=delta))


def synthetic(lengths: tuple[int, ...], post: tuple[int, ...], steps: tuple[int, ...] = (), heads: tuple = (),
              stale: tuple = ()) -> audit.Record:
    w = audit.WindowRecord(0, lengths, 0, 10**9, 10**9, IntegrityProfile(
        ("receipt",), (sum(lengths),), {"receipt": Subset(1)}, 1e-6, drawn={"receipt": 1}), post, steps, heads, stale)
    return audit.Record(bytes(64), bytes(64), 10, 4096, (w,))


def test_a_window_record_s_heads_and_stale_bounds_are_each_pass_s() -> None:
    h = hashlib.sha512(b"a chain").digest()
    rec = synthetic((10, 12), (), (3, 8), (None, h), (10, 12))
    assert rec.windows[0].to_json()["heads"] == [None, h.hex()] and rec.windows[0].to_json()["stale"] == [10, 12]
    assert rec.public()["heads"] == [[None, h.hex()]] and "stale" not in rec.public()
    assert synthetic((10, 12), ()).windows[0].heads == () and synthetic((10, 12), ()).public()["heads"] == [[]]
    for bad in ((h,), (h, h, h), (None, h[:63]), (None, h.hex()), (None, bytearray(h))):
        with pytest.raises(ValueError, match="heads"):
            synthetic((10, 12), (), heads=bad)
    for bad in ((10,), (10, -1), (10, True), (10, 10.0)):
        with pytest.raises(ValueError, match="stale"):
            synthetic((10, 12), (), stale=bad)


def test_a_window_record_s_steps_are_each_pass_s_strictly_increasing() -> None:
    assert synthetic((10, 12), (), (3, 8)).windows[0].to_json()["steps"] == [3, 8]
    assert synthetic((10, 12), ()).windows[0].steps == ()
    for bad in ((3,), (3, 3), (8, 3), (-1, 3), (3, True), (3, audit.STEP_LIMIT)):
        with pytest.raises(ValueError, match="steps"):
            synthetic((10, 12), (), bad)


def test_a_window_record_s_post_seed_reads_are_each_pass_s_at_most_its_length() -> None:
    assert synthetic((10, 12), ()).windows[0].post_seed == (0, 0)
    assert synthetic((10, 12), (0, 2)).windows[0].credited(False) == (10, 10)
    assert synthetic((10, 12), (0, 2)).windows[0].credited(True) == (10, 12)
    for bad in ((0,), (11, 0), (-1, 0), (0, 1.0)):
        with pytest.raises(ValueError, match="post-seed"):
            synthetic((10, 12), bad)


class Dropped:
    """A prover's memory with the appended pages dropped: what it holds of them is zeros."""

    def __init__(self, blocks: list[bytes]) -> None:
        self.blocks = blocks

    def __getitem__(self, j: int) -> bytes:
        return self.blocks[j] if j < N else bytes(codec.BLOCK_BYTES)


@pytest.mark.parametrize("opens", ["what-it-holds", "re-encoded"])
def test_a_prover_that_drops_appended_pages_is_rejected(opens) -> None:
    """The prover commits its appends, lists the pages in its plans and then keeps only the registered blocks, its
    receipts of the pages made from what it holds.  Opened as held, a page doesn't decode under the digest committed
    for it (`codec`); re-encoded from its payload for the opening, it no longer gives the committed receipt
    (`receipt`)."""
    honest_store = store()
    cheat = copy.copy(honest_store)
    cheat.blocks = Dropped(honest_store.blocks)
    r = Run(honest_store, cheat, windows=1)
    r.append(3)
    assert r.run(change=(((N, 3),), ())) is None
    r.boundary()

    def reencoded(ps, q):
        return dataclasses.replace(r.prover.open(ps, q), block=honest_store.blocks[ps.plan.block(q)])

    verdict = r.finish(None if opens == "what-it-holds" else reencoded)[0]
    assert not verdict.accepted and verdict.failure == ("codec" if opens == "what-it-holds" else "receipt")
    assert r.passes[0, 0].plan.block(verdict.at[1]) >= N


@pytest.mark.parametrize("appended, digest, failure", [
    (True, b"", "malformed"), (True, codec.digest(payload(b"kv page 99")), "codec"),
    (False, codec.digest(payload(b"kv page 9")), "malformed")], ids=["no-digest", "another-digest", "registered"])
def test_an_appended_block_opens_only_with_its_digest(appended, digest, failure) -> None:
    """An appended block's opening without its digest is malformed, and with another payload's digest it doesn't
    decode.  A registered block's opening with a digest is malformed."""
    r = honest_session()
    r.clock.now = r.s.deadline(1) + 1
    draws = r.s.close(0)
    n = next(n for n, (i, q) in enumerate(draws) if (r.passes[0, i].plan.block(q) >= N) == appended)
    opened = [r.prover.open(r.passes[0, i], q) for i, q in draws]
    opened[n] = dataclasses.replace(opened[n], digest=digest)
    assert r.s.verify(0, opened) == audit.Verdict(0, False, failure, draws[n])


@pytest.mark.parametrize("opened", [0, 1], ids=["no-openings", "an-opening"])
def test_a_window_with_no_draws_is_malformed_whatever_its_openings(opened) -> None:
    """No session draws nothing (`Params` refuses `samples` or `min_passes` under 1), but a window whose draws are
    empty is `malformed` at `verify`, with or without openings, as the Lean `Session.verify` has it: a frame that
    doesn't decode, read as no openings, is refused on every window."""
    r = Run(store(), windows=1)
    assert r.run() is None
    r.boundary()
    r.clock.now = r.s.deadline(0) + 1
    draws = r.s.close(0)
    openings = [r.prover.open(r.passes[0, i], q) for i, q in draws][:opened]
    r.s.windows[0].draws = ()
    assert r.s.verify(0, openings) == audit.Verdict(0, False, "malformed") == r.s.failed


def test_an_append_s_leaves_bind_the_indices_the_verifier_assigns() -> None:
    """A prover that commits its append's tree one index past the one the verifier assigns, and opens its pages with
    that tree's paths (the same shape), fails the payload path."""
    st = store()
    r = Run(st, windows=1)
    ys = salts(b"page salts", 2)
    a = st.append(pages(N, 2), ys)
    shifted = audit.append_tree(st.registration.digest, N + 1, a.digests, ys)
    assert r.commit(shifted.commitment.root, 2) == N
    assert len(shifted.open(N + 1).path) == len(a.tree.open(N).path)
    assert r.run(change=(((N, 2),), ())) is None
    r.boundary()

    def opens(ps, q):
        o, j = r.prover.open(ps, q), ps.plan.block(q)
        return o if j < N else dataclasses.replace(o, payload_path=shifted.open(j + 1).path)

    assert r.finish(opens)[0] == audit.Verdict(0, False, "payload-path", (0, N))


def test_a_plan_holding_a_block_not_appended_or_released_is_refused() -> None:
    """A plan is over the store's blocks so far: one holding a block before the append that commits it, or after its
    release, is refused (`plan`), whole or as a change."""
    r = Run(store())
    assert r.s.receive(0, 0, bytes(64), ((0, N + 1),), step=0).failure == "plan"

    r = Run(store())
    r.append(2)
    assert r.s.release(((N, 1),)) is None
    assert r.run(Plan(((0, N + 2),), r.store.extent)).failure == "plan"

    r = Run(store())
    r.append(2)
    assert r.run(change=(((N, 2),), ())) is None
    assert r.s.release(((N + 1, 1),)) is None
    assert r.s.receive(0, 1, bytes(64), change=((), ()), step=1).failure == "plan"

    r = Run(store())
    assert r.s.receive(0, 0, bytes(64), ((0, N),), change=((), ()), step=0).failure == "plan"


def two_windows_under_a_lifetime_of_two() -> Run:
    """L = 2.  Pages 8 and 9 appended before window 0's first root, which they join, and read; page 10 appended after
    window 0's roots but before window 1's first, which it joins, and read."""
    r = Run(store(), windows=4, lifetime=2)
    r.append(2)
    assert r.run(change=(((N, 2),), ())) is None
    r.boundary()
    r.append(1)
    assert r.run(change=(((N + 2, 1),), ())) is None
    r.boundary()
    return r


def test_an_appended_block_leaves_every_plan_within_the_lifetime_unless_re_keyed() -> None:
    """Window 2's plan may not hold pages 8 and 9, which joined window 0 (`lifetime`).  Re-keyed instead (released,
    and their payloads appended again at 11 and 12, under fresh bound keys) they are read in windows 2 and 3, and page
    10, which joined window 1, in window 2 but not 3: every receipt opens and the session is accepted, its record
    naming L."""
    r = two_windows_under_a_lifetime_of_two()
    v = r.run(change=((), ()))
    assert v is not None and v.failure == "lifetime" and v.window == 2

    r = two_windows_under_a_lifetime_of_two()
    a = r.store.append(pages(N, 2), salts(b"re-keyed", 2))
    assert r.commit(a.root, a.count) == N + 3 and r.s.release(((N, 2),)) is None
    assert r.run(change=(((N + 3, 2),), ((N, 2),))) is None
    r.boundary()
    v = copy.deepcopy(r).run(change=((), ()))
    assert v is not None and v.failure == "lifetime" and v.window == 3
    assert r.s.release(((N + 2, 1),)) is None
    assert r.run(change=((), ((N + 2, 1),))) is None
    r.boundary()
    assert r.store.blocks[N + 3] != r.store.blocks[N] and r.store.committed(N + 3)[2] == codec.digest(pages(N, 1)[0])
    assert [v.accepted for v in r.finish()] == [True] * 4
    rec = r.s.record()
    assert [w.post_seed for w in rec.windows] == [(2,), (3,), (3,), (2,)]
    assert rec.lifetime == 2 and rec.public()["lifetime"] == 2
    for bad in (0, -1, True, 1.0):
        with pytest.raises(ValueError, match="lifetime"):
            Params(blocks=N, deadline_ns=1, windows=1, lifetime=bad)


def test_a_session_that_defers_residency_has_a_lifetime_of_at_least_two() -> None:
    """Deferred, a block appended in window w is resident from w or w + 1, so its re-key falls at age L − 1: at L = 1,
    before it may be resident.  The verifier refuses the session's parameters, not only its passes."""
    assert Params(blocks=N, deadline_ns=1, windows=1).deferred is False
    assert Params(blocks=N, deadline_ns=1, windows=1, lifetime=1).lifetime == 1
    assert Params(blocks=N, deadline_ns=1, windows=1, lifetime=2, deferred=True).deferred is True
    with pytest.raises(ValueError, match="defers residency has a lifetime of at least 2 windows, not 1"):
        Params(blocks=N, deadline_ns=1, windows=1, lifetime=1, deferred=True)
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="deferred is a bool"):
            Params(blocks=N, deadline_ns=1, windows=1, deferred=bad)


def test_a_session_s_schedule_cap_is_none_or_bits() -> None:
    """The cap a session commits on its schedule's bits is None (the default: none, so no appended block is credited)
    or a nonnegative integer."""
    assert Params(blocks=N, deadline_ns=1, windows=1).schedule_cap is None
    assert Params(blocks=N, deadline_ns=1, windows=1, schedule_cap=0).schedule_cap == 0
    for bad in (-1, True, 8.0, "200"):
        with pytest.raises(ValueError, match="schedule's cap"):
            Params(blocks=N, deadline_ns=1, windows=1, schedule_cap=bad)


def test_the_lifetime_cap_is_the_whole_windows_of_t_in_30_seconds() -> None:
    """L is a duration's (`LIFETIME_NS`, 30 s) whole windows, at least 1, or 2 deferred: a carrier's life stays about
    30 s whatever T is, where a count of windows was an hour at 60 s and a minute at 1 s.  A paced session's default
    counts its groups as stated windows, the fewest the life allows (the verifier states the groups of their nominal
    length); a stated L stays."""
    assert audit.LIFETIME_NS == 30 * 10**9
    assert [audit.lifetime_windows(t * 10**9) for t in (60, 30, 20, 7, 1)] == [1, 1, 1, 4, 30]
    assert [audit.lifetime_windows(t * 10**9, deferred=True) for t in (60, 20, 15, 1)] == [2, 2, 2, 30]
    assert audit.lifetime_windows(85 * 10**6) == 352 and audit.lifetime_windows(10**9, life_ns=120 * 10**9) == 120
    assert Params(blocks=N, deadline_ns=1, windows=1).lifetime == 1
    assert Params(blocks=N, deadline_ns=1, windows=1, deferred=True).lifetime == 2
    assert Params(blocks=N, deadline_ns=1, windows=1, window_ns=10**9).lifetime == 30
    assert Params(blocks=N, deadline_ns=1, windows=1, window_ns=10**9, lifetime=7).lifetime == 7
    assert Params(blocks=N, deadline_ns=1, windows=2, window_ns=1, pace=audit.Pace(1)).lifetime == 1
    assert Params(blocks=N, deadline_ns=1, windows=2, window_ns=1, pace=audit.Pace(1, 10**9)).lifetime == 30
    for bad in (0, -1, 1.0, None):
        with pytest.raises(ValueError, match="lifetime's window_ns"):
            audit.lifetime_windows(bad)


def withdrawing() -> Run:
    """Pages 8, 9 and 10 appended; a pass holds 8 and 10, whose encodes landed, and not 9."""
    r = Run(store())
    r.append(3)
    assert r.run(change=(((N, 1), (N + 2, 1)), ())) is None
    return r


def test_a_withdrawn_block_is_released_and_was_in_no_plan() -> None:
    """A release names the appended blocks among its runs that were freed before their encode landed (memory-map/v1's
    `withdrawn`): they leave the store like any released block, and the session is accepted, counting them released."""
    r = withdrawing()
    assert r.s.release(((N + 1, 1),), ((N + 1, 1),)) is None
    assert r.run(change=((), ())) is None
    r.boundary()
    assert r.run() is None
    r.boundary()
    assert [v.accepted for v in r.finish()] == [True, True]
    assert (r.s.record().appended, r.s.record().released) == (3, 1)

    r = Run(store())
    r.append(3)
    assert r.s.release(((N + 1, 2), (N, 1)), ((N, 3),)) is None


@pytest.mark.parametrize("runs, withdrawn", [
    (((N, 1),), ((N, 1),)),
    (((2, 1),), ((2, 1),)),
    (((2, 1),), ((N + 1, 1),)),
    (((N + 1, 1),), ((N + 1, 0),)),
    (((N + 1, 1),), "x"),
], ids=["planned", "registered", "outside-the-release", "zero-count", "malformed"])
def test_a_withdrawal_of_a_block_a_plan_held_or_outside_its_release_rejects_the_session(runs, withdrawn) -> None:
    """The verifier checks a withdrawn block was never written: appended (at N or above), among the release's own
    runs, and in no plan received before, however the plan came."""
    r = withdrawing()
    assert r.s.release(runs, withdrawn).failure == "release"


def test_a_block_any_earlier_plan_held_cannot_be_withdrawn() -> None:
    """Page 9 was in the first pass's plan (sent whole) and left it at the next (a change): it was written, and its
    withdrawal is refused."""
    r = Run(store())
    r.append(3)
    assert r.run(Plan(((0, N + 2),), r.store.extent)) is None
    assert r.run(change=((), ((N, 2),))) is None
    assert r.s.release(((N + 1, 1),), ((N + 1, 1),)).failure == "release"
    r = Run(store())
    r.append(3)
    assert r.run(Plan(((0, N + 2),), r.store.extent)) is None
    assert r.s.release(((N + 2, 1),), ((N + 2, 1),)) is None


def test_a_malformed_append_or_release_rejects_the_session() -> None:
    for bad in ((bytes(63), 1), (bytes(64), 0), (bytes(64), "1"), (bytes(64), P512.max_blocks)):
        assert Run(store()).s.append(*bad).failure == "append"
    r = Run(store())
    assert r.s.release(((0, 1),)) is None
    assert r.s.release(((0, 2),)).failure == "release"
    for bad in (((N, 1),), ((0, 1), (1, 1)), ((3, 0),), "x"):
        assert Run(store()).s.release(bad).failure == "release"
    r = Run(store())
    assert r.s.release(((N, 1),)).failure == "release"
    with pytest.raises(audit.SessionError, match="ended"):
        r.s.append(bytes(64), 1)


def test_a_store_appends_at_fresh_indices_under_bound_keys() -> None:
    """An append commits its blocks' digests and leaves them unencoded until its coin arrives (`Store.encode`), which
    encodes them once: a second coin for it, or a coin for no pending append, is refused, and a malformed coin
    leaves it pending."""
    st = store()
    a = st.append(pages(N, 2), salts(b"a", 2))
    b = st.append(pages(N + 2, 1), salts(b"b", 1))
    assert (a.first, a.count, b.first, b.count, st.extent) == (N, 2, N + 2, 1, N + 3)
    assert st.blocks[N:] == [None] * 3 and set(st.pending) == {N, N + 2}
    ca, cb = (hashlib.sha512(label).digest()[:codec.COIN_BYTES] for label in (b"coin a", b"coin b"))
    with pytest.raises(ValueError, match="32 bytes"):
        st.encode(N, ca[:31])
    assert set(st.pending) == {N, N + 2}
    st.encode(N, ca)
    st.encode(N + 2, cb)
    for first in (N, N + 1, N + 2):
        with pytest.raises(ValueError, match="awaits its coin"):
            st.encode(first, ca)
    k = st.registration.key
    for j, coin in ((N, ca), (N + 1, ca), (N + 2, cb)):
        salt, path, d = st.committed(j)
        assert d == codec.digest(pages(j, 1)[0]) and codec.decode(st.blocks[j], j, k, d, coin) == pages(j, 1)[0]
        assert codec.decode(st.blocks[j], j, k, d) is None
    assert st.committed(N + 2)[1] == () and st.committed(0)[2] == b""
    assert a.tree.open(N + 1).value == receipts.hidden(pages(N + 1, 1)[0], a.salts[1])
    with pytest.raises(ValueError, match="salt per payload"):
        st.append(pages(0, 2), salts(b"c", 1))
    with pytest.raises(ValueError, match="one or more"):
        audit.append_tree(st.registration.digest, N, [], [])
    with pytest.raises(ValueError, match="64 bytes each"):
        audit.append_tree(st.registration.digest, N, [bytes(63)], salts(b"d", 1))
    with pytest.raises(ValueError, match="registration's digest"):
        audit.append_domain(bytes(32), N, 1)


def test_a_plan_s_change_is_the_next_plan_in_store_order() -> None:
    p = Plan(((5, 3), (0, 2)), 10)
    assert p.changed((), (), 10).runs == ((0, 2), (5, 3))
    assert p.changed(((2, 3),), (), 12).runs == ((0, 8),)
    assert p.changed(((10, 2), (8, 1)), ((6, 1), (0, 1)), 12).runs == ((1, 1), (5, 1), (7, 2), (10, 2))
    assert p.changed((), ((5, 3),), 10).runs == ((0, 2),)
    assert p.changed(((8, 2),), (), 10).digest == Plan(((0, 2), (5, 5)), 10).digest
    for added, removed, blocks, match in ((((3, 1),), (), 9, "only grows"),
                                          ((), ((2, 1),), 10, "does not hold"),
                                          ((), ((4, 2),), 10, "does not hold"),
                                          ((), ((7, 2),), 10, "does not hold"),
                                          (((6, 1),), (), 10, "holds"),
                                          (((2, 2), (3, 1)), (), 10, "disjoint"),
                                          (((9, 2),), (), 10, "inside the store"),
                                          ((), ((0, 2), (5, 3)), 10, "1 to")):
        with pytest.raises(ValueError, match=match):
            p.changed(added, removed, blocks)
