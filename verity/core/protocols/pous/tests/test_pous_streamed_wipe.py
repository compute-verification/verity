"""The streamed-label wipe (`streamed_wipe`): the label format, an honest session accepted at batch size 1 and refused
above it, which the guarantee doesn't cover, a wrong or late batch that ends the session with no retry, the fill before
any challenge, the tiers the fill covers, a verdict that fails closed, and the guarantee's bound at the deployed instance
and against an exhaustive count on a tiny one.  Every clock is injected."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math

import pytest

import pous_test_signer as ts
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.ml.randomness import derive

SECRET, SERVER = b"auditor secret", b"server-7"
SESSION = hashlib.sha512(b"wipe session").digest()[:32]
N, K, LIMIT = 256, 40, 1_000_000


class Clock:
    """Time advances only when a responder spends it."""

    def __init__(self) -> None:
        self.now = 1_000_000

    def __call__(self) -> int:
        return self.now


def params(batch: int = 1, limit: int = LIMIT) -> sw.WipeParams:
    return sw.WipeParams(N, K, batch, limit, (("gpu", 128 * sw.LABEL_BYTES), ("host-dram", 128 * sw.LABEL_BYTES)))


def challenger(p: sw.WipeParams | None = None, *, clock=None, signer=None, log=None) -> sw.Challenger:
    return sw.Challenger(SECRET, SERVER, SESSION, p or params(), signer or ts.TestSigner(b"wipe challenger", height=2),
                         log=log or ts.session_log(), clock=clock or Clock(), coin_secret=b"coins")


def run(c: sw.Challenger, respond=None, *, clock: Clock | None = None, spend=lambda b: 20_000) -> sw.SignedWipe:
    prover = sw.Prover(c.params)
    for start, data in c.fill(100):
        prover.receive(start, data)
    b = 0

    def answer(idx):
        nonlocal b
        if clock is not None:
            clock.now += spend(b)
        b += 1
        return (respond or prover.answer)(idx)

    c.run(answer)
    return c.close(egress_bits=1234)


def test_label_format():
    key = sw.wipe_key(SECRET, SERVER, SESSION)
    assert key == derive(SECRET, sw.KEY_DOMAIN, {"server": SERVER, "session": SESSION}).value
    assert len(sw.LABEL_TAG) + len(key) + 8 == 75 < 112  # one SHA-512 block a label
    assert sw.label(key, 5) == hashlib.sha512(b"verity-pous/streamed-wipe/label/v1\0" + key + (5).to_bytes(8, "big")).digest()
    assert sw.labels(key, 3, 4) == b"".join(sw.label(key, i) for i in range(3, 7))
    assert sw.wipe_key(SECRET, b"server-8", SESSION) != key
    assert sw.wipe_key(SECRET, SERVER, bytes(32)) != key
    with pytest.raises(ValueError):
        sw.label(key, 1 << 64)


def test_draws_are_uniform_indices_from_the_coin_secret():
    idx = sw.draw_indices(b"coins", SERVER, SESSION, N, K)
    key = derive(b"coins", sw.COIN_DOMAIN, {"server": SERVER, "session": SESSION, "n": N})
    assert idx == tuple(key.uniform(N, j) for j in range(K))
    assert all(0 <= i < N for i in idx) and idx != sw.draw_indices(b"other", SERVER, SESSION, N, K)


@pytest.mark.parametrize("batch", [1, 2, 8, K])
def test_honest_session_is_accepted_at_batch_1_and_refused_above(batch):
    """An honest session runs to completion at any batch size, but only the sequential reveal is accepted: the guarantee
    doesn't cover a larger batch, so the verdict refuses it by name."""
    clock = Clock()
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(params(batch), clock=clock, signer=signer)
    signed = run(c, clock=clock)
    assert signed.ended == sw.COMPLETE and len(signed.batches) == -(-K // batch)
    assert sum(r.count for r in signed.batches) == K and all(r.wrong == 0 for r in signed.batches)
    v = sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES)
    uncovered = f"batch size {batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"
    assert v.reasons == (() if batch == 1 else (uncovered,))
    assert v.public() == {"accepted": batch == 1, "egress_mib": 1}
    assert c.coins() == sw.draw_indices(b"coins", SERVER, SESSION, N, K)
    assert c.params.covered == (batch == 1)


def test_wrong_answer_ends_the_session():
    clock = Clock()
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    key = sw.wipe_key(SECRET, SERVER, SESSION)
    seen = []

    def respond(idx):
        seen.append(idx)
        return bytes(64) if len(seen) == 5 else b"".join(sw.label(key, i) for i in idx)

    signed = run(c, respond, clock=clock)
    assert signed.ended == sw.WRONG and len(signed.batches) == 5 and signed.batches[-1].wrong == 1
    with pytest.raises(RuntimeError, match="no batch is retried"):
        c.challenge()
    v = sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES)
    assert not v.accepted and v.public() == {"accepted": False, "egress_mib": 1}


def test_late_batch_ends_the_session():
    clock = Clock()
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    signed = run(c, clock=clock, spend=lambda b: LIMIT + 1 if b == 3 else 1000)
    assert signed.ended == sw.LATE and len(signed.batches) == 4
    assert not sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES).accepted


def test_challenges_follow_the_whole_fill_once():
    c = challenger()
    with pytest.raises(RuntimeError, match="once the fill is complete"):
        c.challenge()
    stream = c.fill(100)
    next(stream)
    with pytest.raises(RuntimeError, match="once the fill is complete"):
        c.challenge()
    list(stream)
    assert c.filled == N
    with pytest.raises(RuntimeError, match="streams once"):
        next(c.fill(100))
    c.challenge()
    with pytest.raises(RuntimeError, match="unanswered"):
        c.challenge()
    with pytest.raises(ValueError, match="answered with"):
        c.receive(bytes(63))


def test_one_session_an_id():
    log = ts.session_log()
    challenger(log=log)
    with pytest.raises(mc.SessionLogError):
        challenger(log=log)


def test_the_fill_covers_every_declared_tier():
    with pytest.raises(ValueError, match="covers every declared tier"):
        sw.WipeParams(N, K, 1, LIMIT, (("gpu", 128 * sw.LABEL_BYTES),))
    with pytest.raises(ValueError, match="each named once"):
        sw.WipeParams(N, K, 1, LIMIT, (("gpu", 128 * sw.LABEL_BYTES), ("gpu", 128 * sw.LABEL_BYTES)))
    with pytest.raises(ValueError):
        sw.WipeParams(N, K, K + 1, LIMIT, (("gpu", N * sw.LABEL_BYTES),))


def test_verdict_fails_closed():
    clock = Clock()
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    signed = run(c, clock=clock)
    trusted = {signer.public_key}
    assert not sw.verdict(signed, c.commitment, trusted=set(), schemes=ts.SCHEMES).accepted
    assert not sw.verdict(signed, c.commitment, trusted=trusted).accepted  # a test scheme outside a test's mapping
    other = dataclasses.replace(c.commitment, params=params(limit=LIMIT + 1))
    assert not sw.verdict(signed, other, trusted=trusted, schemes=ts.SCHEMES).accepted
    r = signed.batches[0]
    forged = dataclasses.replace(signed, batches=(dataclasses.replace(r, receipt_ns=r.receipt_ns + 1), *signed.batches[1:]))
    assert "the signature does not verify" in sw.verdict(forged, c.commitment, trusted=trusted, schemes=ts.SCHEMES).reasons
    more = dataclasses.replace(signed, egress_bits=0)
    assert not sw.verdict(more, c.commitment, trusted=trusted, schemes=ts.SCHEMES).accepted


def test_public_egress_is_15_bits_of_mib_rounded_up():
    """The bound reads the egress `E`, so the verdict publishes it, rounded up to whole MiB and saturating at 15 bits:
    with the verdict, 16 bits, the least public size class."""
    mib = sw.EGRESS_UNIT_BITS
    assert [sw.egress_units(e) for e in (0, 1, mib, mib + 1)] == [0, 1, 1, 2]
    assert sw.egress_units(sw.EGRESS_MAX_UNITS * mib) == sw.egress_units(10**15) == sw.EGRESS_MAX_UNITS == 2**15 - 1


def test_deployed_bound():
    n, k = sw.DEPLOYED_LABELS, sw.DEPLOYED_K
    assert n == 40_625_000_000
    q, g = sw.retained_bits(n, k, n * sw.LABEL_BITS, 0, -40)
    assert n - g == 15_598_671 and q == 8_186_012_770  # 0.998 GB of labels missing; 1.0233 GB of prior content kept
    assert sw.log2_error(n, k, g, n * sw.LABEL_BITS, 0, q) <= -40
    assert sw.log2_error(n, k, g, n * sw.LABEL_BITS, 0, q - 64) > -40
    q_e, _ = sw.retained_bits(n, k, n * sw.LABEL_BITS, 10**9, -40)
    assert q_e == q + 10**9  # egress adds to what can be kept, bit for bit
    assert sw.challenges_for(n, 10**9 // 64, -40) == 72_074
    # The Lean corollary's explicit point (`Pous.Guarantees.StreamedWipeMeets2600GB`): D = 15,610,000, its crude count.
    d = 15_610_000
    assert k * math.log2(1 - d / n) < -41
    assert math.log2(k) + 36 * (d - 1) + (d - 1) * sw.LABEL_BITS + 41 <= 8_600_000_000


@pytest.mark.parametrize("n", [1 << 19, (1 << 19) + 7, 1_610_612_736, 2_203_125_000, 1 << 50])
def test_bound_at_2152_reads(n: int) -> None:
    """The Lean family's point (`Pous.Guarantees.StreamedWipeMeetsFamily`, `⌈M/19⌉ + E` at `2^-128` with 2152 reads):
    `D = ⌈n/24⌉` labels missed, its crude count `C(n, r) ≤ 2^n`, each term `2^-129`; the exact count needs less."""
    k, m = 2152, n * sw.LABEL_BITS
    d = -(-n // 24)
    assert k * d / n >= 129 * 0.6931471808 and k * math.log2(1 - d / n) < -129
    assert 12 + n + (d - 1) * sw.LABEL_BITS + 129 <= -(-m // 19)
    q, _ = sw.retained_bits(n, k, m, 0, -128)
    assert q <= -(-m // 19)
    assert 18**k * 2**128 <= 19**k  # (18/19)^2152 ≤ 2^-128, though the proof's sampling is at 23/24


def reads_threshold(n: int, r: int, k: int, reserve_bits: int = 0) -> int:
    """`Pous.Guarantees.StreamedWipeMeetsReads`'s threshold before the egress: `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R`."""
    return sw.LABEL_BITS * (-(-n // r) - 1) + n + (k.bit_length() - 1) + 130 + reserve_bits


def least_reads(r: int) -> int:
    """The least `k` with `(r − 1)^k · 2^129 ≤ r^k`, the statement's sampling condition."""
    k = math.floor(129 / -math.log2(1 - 1 / r))
    while (r - 1) ** k * 2**129 > r**k:
        k += 1
    while (r - 1) ** (k - 1) * 2**129 <= r ** (k - 1):
        k -= 1
    return k


def test_h200_at_5GB() -> None:
    """`StreamedWipeMeets141GB5`: at 141·10^9 bytes with 2.3 GB of driver reserve, `r = 59` is the least ratio whose
    threshold is at most 5 GB, `k = 5231` the least read count for it, and the threshold `39,721,768,854` bits; the
    family's own point, `r = 24` and `k = 2152`, is `512·(⌈n/24⌉ − 1) + n + 141`."""
    n, reserve, cap = 2_203_125_000, 23 * 10**8 * 8, 5 * 10**9 * 8
    assert least_reads(59) == 5231 and least_reads(24) == 2101
    assert reads_threshold(n, 59, 5231, reserve) == 39_721_768_854 <= cap
    assert reads_threshold(n, 59, 5231) == 21_321_768_854 <= cap - reserve
    assert reads_threshold(n, 58, least_reads(58), reserve) > cap
    assert reads_threshold(n, 24, 2152) == sw.LABEL_BITS * (-(-n // 24) - 1) + n + 141
    q, _ = sw.retained_bits(n, 5231, n * sw.LABEL_BITS + reserve, 0, -128)
    assert q <= reads_threshold(n, 59, 5231, reserve)  # the exact count needs less


def test_bound_against_an_exhaustive_count():
    """N = 3 labels of 2 bits, k = 2, q = 2 bits of prior content, M = 4 bits inside, no egress: a prover that keeps the
    prior content and label 0, and answers 0 for every other label, wins exactly as often as counted, and the bound
    (minimized over g, in exact arithmetic) is above it."""
    n, b, k, q, m = 3, 2, 2, 2, 4
    wins = total = 0
    for z, labels, c in itertools.product(range(4), itertools.product(range(4), repeat=n),
                                          itertools.product(range(n), repeat=k)):
        total += 1
        sigma = (z, labels[0])
        answers = [sigma[1] if i == 0 else 0 for i in c]
        wins += all(a == labels[i] for a, i in zip(answers, c)) and sigma[0] == z
    pr = wins / total
    bound = min((g / n) ** k + k * math.comb(n, g + 1) * 2 ** (m - q - (g + 1) * b) for g in range(n + 1))
    assert pr <= bound
    assert pr == pytest.approx((1 + 4 / 4 + 2 / 4 + 2 / 16) / 9)  # c by the distinct labels it asks beyond label 0
