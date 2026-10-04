"""Secure erasure's verifier (`verity_pous.erasure`): an honest prover accepted, at both deadlines exactly; a prover that
under-fills, or misses either deadline, rejected; seeded prover bugs rejected; the refusals that keep it fail-closed; and
the fill's byte formats.  The verifier's clock is injected throughout."""

from __future__ import annotations

import hashlib
import random

import pytest
from verity.randomness import Key, derive
from verity_pous.erasure import (
    DIGEST_PRIME,
    FILL_DOMAIN,
    PAGE_BYTES,
    PAGE_DOMAIN,
    SPOT_DOMAIN,
    DigestReply,
    ErasureVerifier,
    FillChallenge,
    KeyedFill,
    ReadChallenge,
    ReadReply,
    fill_key,
    page_keccak_f,
    run_erasure,
    spot_indices,
    spot_key,
)
from verity_pous.primitives import KECCAK_F_NS

M, S, PAGE = 4096, 64, 256
T_FILL = 10**9
T_READ = page_keccak_f(PAGE) * KECCAK_F_NS - 1
"""The longest read deadline a 256-byte page allows: 449 ns."""


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class Prover:
    """Honest: fills its memory with `F_K` from the challenge, returns the digest of its memory, and answers each read
    from it, `fill_latency` and `read_latency` after the send.  Subclasses are the bugged ones."""

    def __init__(self, fill_latency=0, read_latency=0, seed=0):
        self.fill_latency, self.read_latency, self.rng = fill_latency, read_latency, random.Random(seed)

    def replies(self, msg):
        if isinstance(msg, FillChallenge):
            self.f = KeyedFill(Key(msg.key), msg.size, msg.page_bytes)
            self.memory = self.store()
            return [(self.fill_latency, DigestReply(self.digest()))]
        return [(self.read_latency, ReadReply(msg.j, self.read(msg.j, msg.index)))]

    def store(self) -> bytearray:
        return bytearray(b"".join(self.f.chunks()))

    def digest(self) -> bytes:
        return self.f.digest_of([bytes(self.memory)])

    def read(self, j, i) -> int:
        return self.memory[i]


class Wire:
    """A transport on the injected clock.  `send(msg)` schedules the prover's replies, each `latency` after the send;
    `receive(until)` delivers the earliest one in by `until` (any, if `eager`), moving the clock to its arrival, else
    moves the clock past `until` and returns None."""

    def __init__(self, clock, prover, eager=False):
        self.clock, self.prover, self.eager, self.sent, self.queue = clock, prover, eager, [], []

    def send(self, msg):
        self.sent.append((msg, self.clock.now))
        self.queue += [(self.clock.now + latency, reply) for latency, reply in self.prover.replies(msg)]

    def receive(self, until):
        self.queue.sort(key=lambda q: q[0])
        if self.queue and (self.eager or self.queue[0][0] <= until):
            t, reply = self.queue.pop(0)
            self.clock.now = max(self.clock.now, t)
            return reply
        self.clock.now = max(self.clock.now, until + 1)
        return None


def _verifier(clock, source=b"audit", **kw):
    kw = {"fill_deadline_ns": T_FILL, "read_deadline_ns": T_READ, "page_bytes": PAGE, **kw}
    return ErasureVerifier(M, S, source=source, clock=clock, **kw)


def _run(prover, source=b"audit", eager=False, **kw):
    clock = Clock()
    v = _verifier(clock, source, **kw)
    wire = Wire(clock, prover, eager)
    return v, run_erasure(v, wire.send, wire.receive), wire


# The honest prover.

@pytest.mark.parametrize("eager", [False, True])
def test_an_honest_prover_is_accepted_at_both_deadlines(eager):
    """Digest exactly `T_fill` and every read exactly `T_read` after its send: accepted, the `s` indices revealed one
    at a time in the order `spot_indices` draws them, each only once the last is closed."""
    v, result, wire = _run(Prover(T_FILL, T_READ), eager=eager)
    assert v.accepted and result.accepted and v.done and not v.rejected
    assert result.fill.passed and result.fill.elapsed_ns == T_FILL
    indices = spot_indices(spot_key(b"audit", v.key), S, M)
    assert [(r.j, r.index, r.elapsed_ns) for r in result.reads] == [(j, i, T_READ) for j, i in enumerate(indices)]
    reads = [(m, t) for m, t in wire.sent if isinstance(m, ReadChallenge)]
    assert [m for m, _ in reads] == [ReadChallenge(j, i) for j, i in enumerate(indices)]
    assert [t for _, t in reads] == [T_FILL + j * T_READ for j in range(S)]
    with pytest.raises(RuntimeError, match=f"all {S} reads are closed"):
        v.challenge()


def test_the_indices_are_fresh_distinct_and_shuffled():
    """`s` distinct indices in range from the spot key, which is bound to `K` and never sent: the fill challenge holds
    `K`, `M` and the page size only.  The order is shuffled, not the subset's sorted one."""
    clock = Clock()
    v = _verifier(clock)
    challenge = v.fill_challenge()
    assert challenge == FillChallenge(fill_key(b"audit", M, S).value, M, PAGE)
    key = spot_key(b"audit", v.key)
    assert key == derive(b"audit", SPOT_DOMAIN, {"fill": v.key.value}) and key != v.key
    indices = spot_indices(key, S, M)
    assert len(set(indices)) == S and all(0 <= i < M for i in indices)
    assert sorted(indices) == list(key.subset(S, M, "indices")) and list(indices) != sorted(indices)


def test_k_is_fresh_and_bound_to_the_audit():
    """Without a source, each verifier draws its own secret; `K` is bound to `M`, `s` and the caller's context."""
    clock = Clock()
    assert ErasureVerifier(M, S, fill_deadline_ns=T_FILL, read_deadline_ns=T_READ, page_bytes=PAGE, clock=clock).key != \
        ErasureVerifier(M, S, fill_deadline_ns=T_FILL, read_deadline_ns=T_READ, page_bytes=PAGE, clock=clock).key
    k = fill_key(b"s", M, S)
    assert k == derive(b"s", FILL_DOMAIN, {"memory": M, "spot-reads": S})
    assert len({k, fill_key(b"s", M + 1, S), fill_key(b"s", M, S + 1), fill_key(b"s", M, S, {"device": "gpu0"})}) == 4
    with pytest.raises(ValueError, match="the audit's own"):
        fill_key(b"s", M, S, {"memory": 1})


# Under-filling.

class UnderFill(Prover):
    """Streams the whole fill for an honest digest, but keeps only the bytes outside `missing`, and answers a read of
    a missing byte with a guess."""

    def __init__(self, missing, **kw):
        super().__init__(**kw)
        self.missing = missing

    def digest(self):
        return self.f.digest()

    def read(self, j, i):
        return self.rng.randrange(256) if i in self.missing else self.memory[i]


def _random_bytes(rng, fraction):
    return set(rng.sample(range(M), int(fraction * M)))


def _range_of_bytes(rng, fraction):
    start = rng.randrange(M)
    return {(start + k) % M for k in range(int(fraction * M))}


@pytest.mark.parametrize("pick", [_random_bytes, _range_of_bytes], ids=["scattered", "contiguous"])
def test_a_prover_that_under_fills_by_more_than_f_is_rejected(pick):
    """`f = 1/4`: a prover missing 30% of `M`, scattered or in one run, passes each read with probability at most
    `0.7 + 0.3/256`, so all 64 with about `10^-10`.  Over 40 audits its digest passes and a read rejects every one;
    holding all but those bytes, it is accepted."""
    for seed in range(40):
        rng = random.Random(seed)
        missing = pick(rng, 0.3)
        v, result, _ = _run(UnderFill(missing, seed=seed), source=b"under-%d" % seed)
        assert result.fill.passed and v.rejected and not v.accepted, seed
        assert not result.reads[-1].passed and result.reads[-1].index in missing, seed
        _, honest, _ = _run(UnderFill(set(), seed=seed), source=b"under-%d" % seed)
        assert honest.accepted, seed


class Recompute(Prover):
    """Holds nothing: streams the fill for its digest, then recomputes each read's page, which takes the page's
    `recompute_ns`."""

    def digest(self):
        return self.f.digest()

    def replies(self, msg):
        out = super().replies(msg)
        if isinstance(msg, FillChallenge):
            self.memory = bytearray(M)
            return out
        return [(self.f.recompute_ns, ReadReply(msg.j, self.f.at(msg.index)))]


@pytest.mark.parametrize("eager", [False, True])
def test_a_prover_that_recomputes_each_read_misses_its_deadline(eager):
    """Every byte right, but each read waits on its page's recompute: `T_read` stays below it, so the first read is
    late and rejects the audit."""
    v, result, _ = _run(Recompute(), eager=eager)
    assert result.fill.passed and len(result.reads) == 1 and not v.accepted
    assert not result.reads[0].on_time and result.reads[0].answered == eager


def test_a_read_deadline_that_reaches_a_recompute_is_refused():
    """`T_read` at a page's recompute time is refused, for the default 1 MiB pages (1.7352 ms) as for small ones."""
    assert page_keccak_f(PAGE_BYTES) == 34_704 and KeyedFill(Key(bytes(32)), 1).recompute_ns == 1_735_200
    for page, limit in [(PAGE, T_READ + 1), (PAGE_BYTES, 1_735_200)]:
        with pytest.raises(ValueError, match="a read would not show the byte was held"):
            ErasureVerifier(M, S, fill_deadline_ns=T_FILL, read_deadline_ns=limit, page_bytes=page, clock=Clock())
        ErasureVerifier(M, S, fill_deadline_ns=T_FILL, read_deadline_ns=limit - 1, page_bytes=page, clock=Clock())


# Deadlines, fail-closed.

@pytest.mark.parametrize("eager", [False, True])
def test_a_late_digest_is_rejected_and_no_index_is_revealed(eager):
    """The digest one nanosecond past `T_fill`: rejected, compared or not, and no read is ever revealed."""
    v, result, wire = _run(Prover(T_FILL + 1), eager=eager)
    assert not result.fill.on_time and not result.fill.correct and result.fill.answered == eager
    assert result.fill.elapsed_ns == T_FILL + 1 and result.reads == () and not v.accepted and v.done
    assert not any(isinstance(m, ReadChallenge) for m, _ in wire.sent)
    with pytest.raises(RuntimeError, match="the audit is rejected"):
        v.challenge()


@pytest.mark.parametrize("eager", [False, True])
def test_a_late_read_is_rejected_and_ends_the_audit(eager):
    """Read 5 one nanosecond past `T_read`, the rest on time: rejected at read 5, and no index after it revealed."""

    class Late(Prover):
        def replies(self, msg):
            out = super().replies(msg)
            return [(T_READ + 1, out[0][1])] if isinstance(msg, ReadChallenge) and msg.j == 5 else out

    v, result, wire = _run(Late(), eager=eager)
    assert len(result.reads) == 6 and all(r.passed for r in result.reads[:5]) and not v.accepted
    assert not result.reads[5].on_time and result.reads[5].elapsed_ns == T_READ + 1
    assert sum(isinstance(m, ReadChallenge) for m, _ in wire.sent) == 6


class Silent(Prover):
    """Answers until read `quiet`, then nothing (`quiet = -1`: not even the digest)."""

    def __init__(self, quiet, **kw):
        super().__init__(**kw)
        self.quiet = quiet

    def replies(self, msg):
        j = msg.j if isinstance(msg, ReadChallenge) else -1
        return [] if j == self.quiet else super().replies(msg)


@pytest.mark.parametrize("quiet", [-1, 0, 7, S - 1])
def test_a_silent_prover_cannot_hold_the_audit_open(quiet):
    """No answer to one step: the driver closes it by the verifier's clock one nanosecond past its deadline, unanswered,
    and the audit is rejected and over."""
    v, result, _ = _run(Silent(quiet))
    closed = result.fill if quiet == -1 else result.reads[-1]
    assert not closed.answered and not closed.on_time and not v.accepted and v.done and v.pending is None
    assert closed.elapsed_ns == (T_FILL if quiet == -1 else T_READ) + 1
    assert len(result.reads) == quiet + 1


def test_a_transport_that_gives_up_early_fails_closed():
    """`receive` returning None within the deadline breaks its contract: the driver stops with the step open, not
    accepted, rather than closing it early or waiting forever."""
    clock = Clock()
    v = _verifier(clock)
    with pytest.raises(RuntimeError, match="within its deadline"):
        run_erasure(v, lambda msg: None, lambda until: None)
    assert v.closes_at == T_FILL and not v.accepted and not v.done


def test_replies_for_other_steps_are_dropped():
    """A stale reply for a closed read and a reply for one not yet revealed arrive before each right answer: the
    driver drops them, and only the answer naming the open read counts."""

    class Noisy(Prover):
        def replies(self, msg):
            out = super().replies(msg)
            if isinstance(msg, ReadChallenge):
                out = [(0, ReadReply(msg.j - 1, 0)), (0, ReadReply(msg.j + 1, 0)), (1, out[0][1])]
            return out

    v, result, _ = _run(Noisy())
    assert v.accepted and [r.elapsed_ns for r in result.reads] == [1] * S


def test_the_verifier_refuses_steps_out_of_order():
    """Each step in its turn: the fill first and once, a read only once the fill passed, an answer only for the open
    step, `expire()` only past the deadline, and nothing after the audit is over."""
    clock = Clock()
    v = _verifier(clock)
    for call, match in [(v.challenge, "the fill comes first"), (lambda: v.receive_digest(b""), "the fill is not open"),
                        (lambda: v.respond(0, 0), "no step is open"), (v.expire, "no step is open")]:
        with pytest.raises(RuntimeError, match=match):
            call()
    challenge = v.fill_challenge()
    with pytest.raises(RuntimeError, match="is sent"):
        v.fill_challenge()
    with pytest.raises(RuntimeError, match="the open step is unanswered"):
        v.challenge()
    with pytest.raises(RuntimeError, match="read 0 is not revealed"):
        v.respond(0, 0)
    clock.now = T_FILL
    with pytest.raises(RuntimeError, match="within its deadline"):
        v.expire()
    assert v.receive_digest(KeyedFill(Key(challenge.key), M, PAGE).digest()).passed
    j, i = v.challenge()
    with pytest.raises(RuntimeError, match="the fill is not open"):
        v.receive_digest(b"")
    assert v.respond(j, v.fill.at(i)).passed
    v.challenge()
    for k, state in [(0, "closed"), (2, "not revealed")]:
        with pytest.raises(RuntimeError, match=f"read {k} is {state}"):
            v.respond(k, 0)
    clock.now += T_READ + 1
    with pytest.raises(RuntimeError, match="the audit is rejected"):
        v.challenge()
    assert v.reads[-1].answered is False and v.done and not v.accepted


def test_answers_of_the_wrong_type_are_wrong():
    """A digest or a byte that is not `bytes` or an `int` (a `bool` included) is wrong, never compared by value."""
    for bad in [True, 1.0, "0", None]:
        clock = Clock()
        v = _verifier(clock)
        v.fill_challenge()
        v.receive_digest(v.fill.digest())
        j, _ = v.challenge()
        assert not v.respond(j, bad).correct
    clock = Clock()
    v = _verifier(clock)
    v.fill_challenge()
    assert not v.receive_digest(bytearray(v.fill.digest())).correct


def test_bad_parameters_are_refused():
    for (size, s), kw, match in [((0, 1), {}, "from 1 to M"), ((M, 0), {}, "from 1 to M"),
                                 ((M, M + 1), {}, "from 1 to M"), ((M, S), {"fill_deadline_ns": -1}, "nonnegative"),
                                 ((M, S), {"read_deadline_ns": -1}, "nonnegative"),
                                 ((M, S), {"page_bytes": 3}, "even number of bytes")]:
        with pytest.raises(ValueError, match=match):
            ErasureVerifier(size, s, **{"fill_deadline_ns": T_FILL, "read_deadline_ns": T_READ, "page_bytes": PAGE,
                                        **kw}, source=b"x", clock=Clock())


# Seeded prover bugs.

def _other_key(f):
    return KeyedFill(Key(bytes([f.key.value[0] ^ 1]) + f.key.value[1:]), f.size, f.page_bytes)


BUGS = {
    "digest of M - 1 bytes": lambda p: KeyedFill(p.f.key, M - 1, PAGE).digest(),
    "digest under another key": lambda p: _other_key(p.f).digest(),
    "digest big-endian": lambda p: p.f.digest()[::-1],
    "fill from another key": lambda p: setattr(p, "memory", bytearray(b"".join(_other_key(p.f).chunks()))),
    "pages reversed": lambda p: setattr(p, "memory", bytearray(b"".join(page[::-1] for page in p.f.chunks()))),
    "read off by one": lambda p, j, i: p.memory[(i + 1) % M],
    "read the j-th byte": lambda p, j, i: p.memory[j],
    "read plus one": lambda p, j, i: (p.memory[i] + 1) % 256,
    "slow fill": T_FILL + 1,
    "slow reads": T_READ + 1,
}
"""A prover's bugs: its digest, its memory, its reads, or its latency."""


class Bugged(Prover):
    def __init__(self, name, seed):
        bug = BUGS[name]
        super().__init__(bug if name == "slow fill" else 0, bug if name == "slow reads" else 0, seed)
        self.name, self.bug = name, bug

    def digest(self):
        if self.name.startswith("digest"):
            return self.bug(self)
        if self.name.startswith("fill") or self.name.startswith("pages"):
            honest = self.f.digest()
            self.bug(self)
            return honest
        return super().digest()

    def read(self, j, i):
        return self.bug(self, j, i) if self.name.startswith("read") else super().read(j, i)


@pytest.mark.parametrize("name", BUGS)
def test_every_seeded_prover_bug_is_rejected(name):
    """Each bug, in 8 audits, is rejected every time, by the step it breaks; the honest prover in the same audits is
    accepted."""
    for seed in range(8):
        source = b"bug-%d" % seed
        v, result, _ = _run(Bugged(name, seed), source=source)
        assert not v.accepted and v.done, (name, seed)
        fill_breaks = name.startswith("digest") or name == "slow fill"
        assert result.fill.passed != fill_breaks and (fill_breaks or not result.reads[-1].passed), (name, seed)
        assert _run(Prover(seed=seed), source=source)[1].accepted


# Byte formats.

def _feistel_page(key: bytes, p: int, page_bytes: int) -> bytes:
    """A fill page from `KeyedFill`'s docstring and `primitives.Feistel10`'s: ten rounds of `L, R ← R, L ⊕
    SHAKE256(0x45 ‖ u8(i) ‖ K ‖ u64(p) ‖ R)[:half]` on a zero page."""
    half = page_bytes // 2
    left, right = bytes(half), bytes(half)
    tweak = key + p.to_bytes(8, "little")
    for i in range(1, 11):
        f = hashlib.shake_256(bytes([PAGE_DOMAIN, i]) + tweak + right).digest(half)
        left, right = right, bytes(a ^ b for a, b in zip(left, f))
    return left + right


def test_the_fill_and_its_digest_follow_the_spec():
    """Each page is the Feistel of the spec, the last cut at `M`, and the digest is the polynomial of the spec in
    `r = K.uniform(2^127 − 1, "digest")` over 15-byte words, the last zero-padded."""
    key = fill_key(b"spec", 1000, 8)
    f = KeyedFill(key, 1000, 64)
    pages = [_feistel_page(key.value, p, 64) for p in range(16)]
    data = b"".join(pages)[:1000]
    assert b"".join(f.chunks()) == data and [f.at(i) for i in range(1000)] == list(data)
    words = [int.from_bytes(data[k:k + 15], "little") for k in range(0, 1000, 15)]
    r = key.uniform(DIGEST_PRIME, "digest")
    h = sum(w * pow(r, len(words) - j, DIGEST_PRIME) for j, w in enumerate(words)) % DIGEST_PRIME
    assert f.digest() == h.to_bytes(16, "little") == f.digest_of([data[:7], data[7:500], b"", data[500:]])
    for n in (999, 1001):
        with pytest.raises(ValueError, match="the digest is over 1000 bytes"):
            f.digest_of([bytes(n)])


def test_pinned_vectors():
    """`K`, the fill, its digest and the indices for one secret, pinned: a second implementation must match them."""
    key = fill_key(b"vector", 1000, 8)
    f = KeyedFill(key, 1000, 64)
    assert key.value.hex() == "14d2cc7dfba794e706534bce4a4df190154a11cf0a288b328a38d3c59a248827"
    assert hashlib.sha256(b"".join(f.chunks())).hexdigest() == \
        "5e27541aad42ca2c6a71dd86f89927153e0892050d43bc5f6cedd40149c425da"
    assert f.digest().hex() == "143354e0f115a8c14c0c6e6dc3186955"
    assert spot_indices(spot_key(b"vector", key), 8, 1000) == (209, 866, 228, 687, 735, 808, 655, 621)
