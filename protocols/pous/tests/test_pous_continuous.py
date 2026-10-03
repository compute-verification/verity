"""The continuous audit's verifier (`verity_pous.continuous`): each of the continuous game's three requirements on the
protocol (no pipelining, a hard deadline, every late answer counted) enforced on `audit.TimedVerifier` over a stream
of challenges, on a small band with the verifier's clock injected."""

from __future__ import annotations

import dataclasses
import random

import pytest

from verity_pous import band, dense
from verity_pous.audit import DEADLINE_NS, LATE_ROUNDS, TimedVerifier, challenge_key
from verity_pous.codec import encode_segments
from verity_pous.continuous import ContinuousVerifier, run_continuous
from verity_pous.verifier import VerifierKey

PARAMS = band.BandParams(16, 64, 12, 2)
SALT = bytes(range(24))
LATE = 2_000_000
"""`Δ_late`: 8 calls of 241.5 µs, inside the cap."""


@pytest.fixture(scope="module")
def world():
    prims = band.reference(PARAMS)
    w = random.Random(3).randbytes(PARAMS.w_bytes)
    blocks = [b for seg in encode_segments(PARAMS.codec, PARAMS, prims, w, SALT) for b in seg]
    assert len(set(blocks)) == len(blocks)
    return blocks, VerifierKey.from_segments(PARAMS, SALT, [blocks[:16], blocks[16:]])


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def _verifier(vk, n, clock, source=b"audit", **kw):
    kw.setdefault("late_deadline_ns", LATE)
    return ContinuousVerifier(vk, challenge_key(source, vk), n, clock=clock, **kw)


def _wrong(blocks, i):
    return blocks[(i + 1) % len(blocks)]


def test_honest_stream(world):
    """Arrivals far apart, each answered within Δ: accepted, every index the single audit's draw, and the stream is not
    extended past `n`."""
    blocks, vk = world
    clock, gaps, waited = Clock(), random.Random(5), []

    def wait(j):
        waited.append(j)
        clock.now += gaps.randrange(10**6, 10**9)

    def respond(i):
        clock.now += DEADLINE_NS
        return blocks[i]

    v = _verifier(vk, 50, clock)
    result = run_continuous(v, respond, wait=wait)
    key = challenge_key(b"audit", vk)
    assert v.accepted and result.misses == 0 and waited == list(range(50))
    assert [(a.j, a.index) for a in result.answers] == [(j, key.uniform(PARAMS.blocks, j)) for j in range(50)]
    with pytest.raises(RuntimeError, match="all 50 challenges are answered"):
        v.challenge()


# 1. No pipelining.

def test_no_challenge_before_the_answer_or_the_hard_deadline(world):
    """While challenge 0 is open and within its hard deadline, late or not, challenge 1 is refused; once answer 0 is
    in, it is revealed at once."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 3, clock)
    j, i = v.challenge()
    for t in (0, DEADLINE_NS, DEADLINE_NS + 1, LATE):
        clock.now = t
        with pytest.raises(RuntimeError, match="the previous challenge is unanswered"):
            v.challenge()
    assert v.answers == ()
    a = v.respond(j, blocks[i])
    assert a.correct and not a.on_time and not a.rejected
    assert v.challenge()[0] == 1


def test_a_pipelined_answer_is_refused(world):
    """An answer to a challenge not yet revealed is refused, even when it is the right block, and records nothing."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 3, clock)
    _, i0 = v.challenge()
    i1 = challenge_key(b"audit", vk).uniform(PARAMS.blocks, 1)
    for j in (1, 2):
        with pytest.raises(RuntimeError, match=f"challenge {j} is not revealed"):
            v.respond(j, blocks[i1])
    assert v.answers == ()
    assert v.respond(0, blocks[i0]).correct
    with pytest.raises(RuntimeError, match="challenge 1 is not revealed"):
        v.respond(1, blocks[i1])
    assert len(v.answers) == 1


def test_deadlines_run_from_the_verifiers_send(world):
    """Lateness is the verifier's own clock from its reveal: time before the arrival never counts, and a latency the
    responder reports is kept but never read."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 2, clock)
    clock.now = 10**9
    j, i = v.challenge()
    clock.now += DEADLINE_NS + 1
    a = v.respond(j, blocks[i], report={"elapsed_ns": 1})
    assert a.correct and not a.on_time and a.elapsed_ns == DEADLINE_NS + 1 and a.report == {"elapsed_ns": 1}
    clock.now += 10**9
    j, i = v.challenge()
    clock.now += DEADLINE_NS
    assert v.respond(j, blocks[i]).on_time and v.misses == 1


def _same_block_twice(vk):
    for n in range(1000):
        key = challenge_key(b"audit-%d" % n, vk)
        if key.uniform(PARAMS.blocks, 0) == key.uniform(PARAMS.blocks, 1):
            return b"audit-%d" % n
    raise AssertionError("no key draws the same block twice in a row")


def test_a_stale_answer_is_not_credited_to_the_next_challenge(world):
    """Challenge 0 expires, challenge 1 asks for the same block, and answer 0 arrives: it is refused, so challenge 0
    stays a miss and challenge 1 waits for its own answer."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 2, clock, _same_block_twice(vk))
    _, i = v.challenge()
    clock.now = LATE + 1
    assert v.challenge() == (1, i)
    assert v.answers[0].rejected and v.misses == 1
    with pytest.raises(RuntimeError, match="challenge 0 is closed"):
        v.respond(0, blocks[i])
    assert len(v.answers) == 1 and v.timed.pending == 1
    assert v.respond(1, blocks[i]).on_time and v.misses == 1


# 2. A hard deadline.

@pytest.mark.parametrize("rtt", [0, 40_000])
def test_an_answer_past_the_hard_deadline_is_rejected_unchecked(world, monkeypatch, rtt):
    """At `Δ_late + RTT` a right block is late but counted correct; one nanosecond later it is rejected, and its block
    is never checked."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 2, clock, rtt_allowance_ns=rtt)
    assert v.late_limit_ns == LATE + rtt
    j, i = v.challenge()
    clock.now = LATE + rtt
    a = v.respond(j, blocks[i])
    assert a.correct and not a.on_time and not a.rejected
    j, i = v.challenge()
    clock.now += LATE + rtt + 1
    with monkeypatch.context() as m:
        m.setattr(VerifierKey, "check", lambda *_: pytest.fail("a rejected answer's block was checked"))
        a = v.respond(j, blocks[i])
    assert a.rejected and not a.correct and not a.on_time and a.elapsed_ns == LATE + rtt + 1


def test_expire(world):
    """`expire()` closes the open challenge only past its hard deadline, as a rejected answer the hook sees."""
    _, vk = world
    clock, seen = Clock(), []
    v = _verifier(vk, 2, clock, on_answer=seen.append)
    with pytest.raises(RuntimeError, match="no challenge is open"):
        v.expire()
    v.challenge()
    clock.now = LATE
    with pytest.raises(RuntimeError, match="within its hard deadline"):
        v.expire()
    clock.now = LATE + 1
    a = v.expire()
    assert a.rejected and a.elapsed_ns == LATE + 1 and seen == [a] and v.misses == 1
    with pytest.raises(RuntimeError, match="no challenge is open"):
        v.expire()


def test_the_hard_deadline_is_capped(world):
    """`Δ_late + RTT` must fit at most `D_late = 12` calls of 241.5 µs (under 3,139,500 ns) and at most the band's `d`,
    and the continuous verifier does not run without one."""
    _, vk = world
    key = challenge_key(b"audit", vk)
    assert PARAMS.call_ns == 241_500 and LATE_ROUNDS == 12
    edge = (LATE_ROUNDS + 1) * PARAMS.call_ns - 1
    wide = dataclasses.replace(vk, params=band.BandParams(16, 64, 16, 2))
    flat = dataclasses.replace(vk, params=dense.DenseParams(16, 64, 2))
    for target in (vk, wide, flat):
        assert ContinuousVerifier(target, key, 1, late_deadline_ns=edge).late_limit_ns == 3_139_499
        ContinuousVerifier(target, key, 1, late_deadline_ns=edge - 40_000, rtt_allowance_ns=40_000)
        for late, rtt in ((edge + 1, 0), (edge - 39_999, 40_000)):
            with pytest.raises(ValueError, match="fits D_late = 13 calls of 241500 ns, past the cap of 12"):
                ContinuousVerifier(target, key, 1, late_deadline_ns=late, rtt_allowance_ns=rtt)
    narrow = dataclasses.replace(vk, params=band.BandParams(16, 64, 4, 2))
    ContinuousVerifier(narrow, key, 1, late_deadline_ns=5 * PARAMS.call_ns - 1)
    with pytest.raises(ValueError, match="exceeds the band's in-degree d = 4"):
        ContinuousVerifier(narrow, key, 1, late_deadline_ns=5 * PARAMS.call_ns)
    with pytest.raises(ValueError, match="the hard deadline is at least the deadline"):
        ContinuousVerifier(vk, key, 1, late_deadline_ns=DEADLINE_NS - 1)
    with pytest.raises(ValueError, match="needs a hard deadline"):
        ContinuousVerifier(vk, key, 1, late_deadline_ns=None)


def test_without_a_hard_deadline_the_single_audit_waits(world):
    """`TimedVerifier` without `late_deadline_ns` is unchanged: it waits for each answer, however late."""
    blocks, vk = world
    clock = Clock()
    v = TimedVerifier(vk, challenge_key(b"audit", vk), k=1, clock=clock)
    i = v.challenge()
    clock.now = 10**12
    with pytest.raises(RuntimeError, match="the previous challenge is unanswered"):
        v.challenge()
    with pytest.raises(RuntimeError, match="no hard deadline"):
        v.expire()
    a = v.respond(blocks[i])
    assert v.late_limit_ns is None and a.correct and not a.on_time and not a.rejected and not v.accepted


# 3. Every late answer counted.

def test_every_late_answer_is_a_miss(world):
    """On time, late, past the hard deadline, never (closed by `expire()`), wrong, never (closed by the next
    challenge), on time: five misses, four of them late, three rejected; each challenge recorded once, in order."""
    blocks, vk = world
    clock, seen = Clock(), []
    v = _verifier(vk, 7, clock, max_misses=5, on_answer=seen.append)

    def answer(latency, right=True):
        j, i = v.challenge()
        clock.now += latency
        v.respond(j, blocks[i] if right else _wrong(blocks, i))

    answer(DEADLINE_NS)
    answer(DEADLINE_NS + 1)
    answer(LATE + 1)
    v.challenge()
    clock.now += LATE + 1
    v.expire()
    answer(0, right=False)
    v.challenge()
    clock.now += LATE + 1
    assert v.misses == 4
    answer(0)
    r = v.result()
    assert [(a.correct, a.on_time, a.rejected) for a in r.answers] == [
        (True, True, False), (True, False, False), (False, False, True), (False, False, True),
        (False, True, False), (False, False, True), (True, True, False)]
    assert (r.misses, r.late) == (5, 4) and seen == list(r.answers) and [a.j for a in seen] == list(range(7))
    assert v.accepted and not v.exceeded


@pytest.mark.parametrize("max_misses", [0, 1, 2, 3])
def test_the_stream_accepts_at_most_max_misses(world, max_misses):
    """Three late answers of five: the stream is exceeded from the miss past `max_misses` on, and accepts only when
    `max_misses ≥ 3`."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 5, clock, max_misses=max_misses)
    exceeded = []
    for latency in (DEADLINE_NS + 1, 0, DEADLINE_NS + 1, 0, DEADLINE_NS + 1):
        j, i = v.challenge()
        clock.now += latency
        v.respond(j, blocks[i])
        exceeded.append(v.exceeded)
    misses = [1, 1, 2, 2, 3]
    assert exceeded == [m > max_misses for m in misses] and v.accepted == (max_misses >= 3)


def test_the_miss_count_is_every_wrong_or_late_answer(world):
    """A seeded stream of 400 challenges over every outcome (on time, at and past each limit, wrong, silent and closed
    by `expire()` or by the next challenge): the record matches the script answer for answer, and the miss and late
    counts are the script's."""
    blocks, vk = world
    rng, clock, seen, expected = random.Random(7), Clock(), [], []
    n = 400
    v = _verifier(vk, n, clock, max_misses=n, rtt_allowance_ns=40_000, on_answer=seen.append)
    limit, hard = v.limit_ns, v.late_limit_ns
    for j in range(n):
        clock.now += rng.randrange(10**8)
        opened, i = v.challenge()
        assert opened == j
        if rng.randrange(4):
            latency = rng.choice([0, limit, limit + 1, hard, hard + 1, rng.randrange(2 * hard)])
            right = rng.random() < 0.9
            clock.now += latency
            v.respond(j, blocks[i] if right else _wrong(blocks, i))
            expected.append((right and latency <= hard, latency <= limit, latency > hard))
        else:
            clock.now += hard + 1 + rng.randrange(10**6)
            if j == n - 1 or rng.randrange(2):
                v.expire()
            expected.append((False, False, True))
    r = v.result()
    assert [(a.correct, a.on_time, a.rejected) for a in r.answers] == expected
    assert seen == list(r.answers) and [a.j for a in r.answers] == list(range(n))
    assert r.misses == sum(not (c and t) for c, t, _ in expected) > 0
    assert r.late == sum(not t for _, t, _ in expected) > 0
    assert v.done and v.accepted
