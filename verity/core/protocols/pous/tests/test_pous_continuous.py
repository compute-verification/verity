"""The continuous audit's memory challenger (`verity.core.protocols.pous.continuous`): each of the continuous game's three requirements on the
protocol (no pipelining, a hard deadline, every late answer counted) enforced on `audit.UnsignedRounds` over a stream
of challenges, on a small band with the memory challenger's clock injected."""

from __future__ import annotations

import dataclasses
import random

import pytest

from verity.core.protocols.pous import band, dense
from verity.core.protocols.pous.audit import AUDIT_CAP, DEADLINE_NS, LATE_ROUNDS, UnsignedRounds, challenge_key, run_timed
from verity.core.protocols.pous.codec import encode_segments
from verity.core.protocols.pous.continuous import ContinuousMemoryChallenger, Reply, run_continuous
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey

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
    return ContinuousMemoryChallenger(vk, challenge_key(source, vk), n, clock=clock, **kw)


def _wrong(blocks, i):
    return blocks[(i + 1) % len(blocks)]


class Wire:
    """A transport on the injected clock.  `send(j, i)` records the send and schedules what `replies(j, i)` returns,
    `(latency, Reply)` pairs timed from the send.  `receive(until)` delivers the earliest reply in by `until`, moving
    the clock to its arrival, else moves the clock past `until` and returns None."""

    def __init__(self, clock, replies):
        self.clock, self.replies, self.sent, self.queue = clock, replies, [], []

    def send(self, j, i):
        self.sent.append((j, i, self.clock.now))
        self.queue += [(self.clock.now + latency, reply) for latency, reply in self.replies(j, i)]

    def receive(self, until):
        self.queue.sort(key=lambda q: q[0])
        if self.queue and self.queue[0][0] <= until:
            t, reply = self.queue.pop(0)
            self.clock.now = max(self.clock.now, t)
            return reply
        self.clock.now = max(self.clock.now, until + 1)
        return None


def _run(v, clock, replies, wait=None):
    wire = Wire(clock, replies)
    return run_continuous(v, wire.send, wire.receive, wait), wire


@pytest.mark.parametrize("root_only", [False, True])
def test_honest_stream(world, root_only):
    """Arrivals far apart, each answered within Δ: accepted, every index the single audit's sample sent at its arrival,
    and the stream is not extended past `n`."""
    blocks, vk = world
    clock, gaps, waited, tree = Clock(), random.Random(5), [], MerkleTree(vk.leaves)

    def wait(j):
        clock.now += gaps.randrange(10**6, 10**9)
        waited.append((j, clock.now))

    def replies(j, i):
        return [(DEADLINE_NS, Reply(j, blocks[i], tree.path(i) if root_only else None))]

    v = _verifier(vk, 50, clock, root_only=root_only)
    result, wire = _run(v, clock, replies, wait)
    key = challenge_key(b"audit", vk)
    assert v.accepted and result.misses == 0 and [j for j, _ in waited] == list(range(50))
    assert [(a.j, a.index) for a in result.answers] == [(j, key.uniform(PARAMS.blocks, j)) for j in range(50)]
    assert [(j, t) for j, _, t in wire.sent] == waited
    with pytest.raises(RuntimeError, match="all 50 challenges are answered"):
        v.challenge()


# Liveness: the driver closes every challenge by the memory challenger's clock.

def test_a_silent_responder_cannot_hang_the_stream(world):
    """No reply ever: `run_continuous` still ends, each challenge rejected just past its hard deadline and counted
    late, the last one included."""
    _, vk = world
    clock, seen = Clock(), []
    v = _verifier(vk, 6, clock, max_misses=6, rtt_allowance_ns=40_000, on_answer=seen.append)
    result, wire = _run(v, clock, lambda j, i: [])
    assert v.done and v.pending is None and len(wire.sent) == 6
    assert all(a.rejected and not a.correct and not a.on_time for a in result.answers)
    assert [a.elapsed_ns for a in result.answers] == [LATE + 40_000 + 1] * 6
    assert result.misses == result.late == 6 and seen == list(result.answers) and v.accepted


def test_a_slow_answer_is_closed_by_the_clock_and_never_credited(world):
    """Answer 0 is on its way past the hard deadline: the driver expires challenge 0 at the deadline without waiting
    for it, sends challenge 1 for the same label, and drops answer 0 when it lands, so only answer 1 is credited to
    challenge 1."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 2, clock, _same_block_twice(vk))
    latency = {0: LATE + 100_000, 1: DEADLINE_NS}
    result, wire = _run(v, clock, lambda j, i: [(latency[j], Reply(j, blocks[i]))])
    (_, i, _), (_, i1, t1) = wire.sent
    assert i == i1 and t1 == LATE + 1 and not wire.queue
    assert [(a.correct, a.on_time, a.rejected, a.elapsed_ns) for a in result.answers] == [
        (False, False, True, LATE + 1), (True, True, False, DEADLINE_NS)]
    assert result.misses == 1


def test_replies_for_other_challenges_cannot_hold_one_open(world):
    """A transport with a reply always waiting, each naming another challenge, revealed or not: the driver drops them
    and still closes every challenge at its hard deadline."""
    blocks, vk = world
    clock, sent = Clock(), []
    v = _verifier(vk, 3, clock)

    def receive(until):
        clock.now += 300_000
        j, i = sent[-1]
        return Reply(j + 1 if clock.now % 600_000 else j - 1, blocks[i])

    result = run_continuous(v, lambda j, i: sent.append((j, i)), receive)
    assert [(a.rejected, a.elapsed_ns) for a in result.answers] == [(True, 7 * 300_000)] * 3


def test_a_transport_that_gives_up_early_fails_closed(world):
    """`receive` returning None within the hard deadline breaks its contract: the driver stops with the challenge
    open, recording nothing, rather than closing it early or waiting forever."""
    _, vk = world
    clock = Clock()
    v = _verifier(vk, 2, clock)
    with pytest.raises(RuntimeError, match="within its hard deadline"):
        run_continuous(v, lambda j, i: None, lambda until: None)
    assert v.answers == () and v.pending == 0 and v.closes_at == LATE


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
    """An answer to a challenge not yet revealed is refused, even when it is the right label, and records nothing."""
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


def test_deadlines_run_from_the_memory_challengers_send(world):
    """Lateness is the memory challenger's own clock from its reveal: time before the arrival never counts, and a
    latency the responder reports is kept but never read."""
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
    """Challenge 0 expires, challenge 1 asks for the same label, and answer 0 arrives: it is refused, so challenge 0
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


def test_a_bare_memory_challenger_would_credit_it(world):
    """Why the continuous audit runs only on `ContinuousMemoryChallenger`: the same sequence on a bare `UnsignedRounds` with a
    hard deadline credits answer 0, which names no challenge, to challenge 1 as a hit."""
    blocks, vk = world
    clock = Clock()
    v = UnsignedRounds(vk, challenge_key(_same_block_twice(vk), vk), 2, clock=clock, late_deadline_ns=LATE)
    i = v.challenge()
    clock.now = LATE + 1
    assert v.challenge() == i
    a = v.respond(blocks[i])
    assert (a.j, a.correct, a.on_time) == (1, True, True)


# 2. A hard deadline.

@pytest.mark.parametrize("rtt", [0, 40_000])
def test_an_answer_past_the_hard_deadline_is_rejected_unchecked(world, monkeypatch, rtt):
    """At `Δ_late + RTT` a right label is late but counted correct; one nanosecond later it is rejected, and its label
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


def test_challenge_past_the_last_hard_deadline_closes_it_then_refuses(world):
    """`challenge()` with the last challenge open past its hard deadline closes it as rejected, then refuses to reveal
    past `n`; the close stands.  Within the deadline it changes nothing."""
    _, vk = world
    clock = Clock()
    v = _verifier(vk, 1, clock)
    v.challenge()
    clock.now = LATE
    with pytest.raises(RuntimeError, match="the previous challenge is unanswered"):
        v.challenge()
    assert v.pending == 0 and v.answers == ()
    clock.now = LATE + 1
    with pytest.raises(RuntimeError, match="all 1 challenges are answered"):
        v.challenge()
    assert v.done and v.pending is None and v.answers[0].rejected and v.misses == 1


def test_the_hard_deadline_is_capped(world):
    """`Δ_late + RTT` must fit at most `D_late = 12` calls of 241.5 µs (under 3,139,500 ns) and at most the band's `d`,
    and the continuous audit's memory challenger does not run without one."""
    _, vk = world
    key = challenge_key(b"audit", vk)
    assert PARAMS.call_ns == 241_500 and LATE_ROUNDS == 12
    edge = (LATE_ROUNDS + 1) * PARAMS.call_ns - 1
    wide = dataclasses.replace(vk, params=band.BandParams(16, 64, 16, 2))
    flat = dataclasses.replace(vk, params=dense.DenseParams(16, 64, 2))
    for target in (vk, wide, flat):
        assert ContinuousMemoryChallenger(target, key, 1, late_deadline_ns=edge).late_limit_ns == 3_139_499
        ContinuousMemoryChallenger(target, key, 1, late_deadline_ns=edge - 40_000, rtt_allowance_ns=40_000)
        for late, rtt in ((edge + 1, 0), (edge - 39_999, 40_000)):
            with pytest.raises(ValueError, match="fits D_late = 13 calls of 241500 ns, past the cap of 12"):
                ContinuousMemoryChallenger(target, key, 1, late_deadline_ns=late, rtt_allowance_ns=rtt)
    narrow = dataclasses.replace(vk, params=band.BandParams(16, 64, 4, 2))
    ContinuousMemoryChallenger(narrow, key, 1, late_deadline_ns=5 * PARAMS.call_ns - 1)
    with pytest.raises(ValueError, match="exceeds the band's in-degree d = 4"):
        ContinuousMemoryChallenger(narrow, key, 1, late_deadline_ns=5 * PARAMS.call_ns)
    with pytest.raises(ValueError, match="the hard deadline is at least the deadline"):
        ContinuousMemoryChallenger(vk, key, 1, late_deadline_ns=DEADLINE_NS - 1)
    with pytest.raises(ValueError, match="needs a hard deadline"):
        ContinuousMemoryChallenger(vk, key, 1, late_deadline_ns=None)


def test_without_a_hard_deadline_the_single_audit_waits(world):
    """`UnsignedRounds` without `late_deadline_ns` is unchanged: it waits for each answer, however late."""
    blocks, vk = world
    clock = Clock()
    v = UnsignedRounds(vk, challenge_key(b"audit", vk), k=1, clock=clock)
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


# 4. The audit's duration cap.

CAP_NS = AUDIT_CAP * PARAMS.call_ns
"""1,505 calls of 241.5 µs: 363,457,500 ns."""


def _answers(blocks, latency):
    """A responder answering every challenge right, `latency(j)` after its send."""
    return lambda j, i: [(latency(j), Reply(j, blocks[i]))]


def test_the_cap_is_1505_calls_of_call_ns(world):
    """`AUDIT_CAP` is Lean's `auditCap = 1505`, counted in the parameters' `call_ns` as `D` and `D_late` are; a cap
    from 1 to it is accepted, and the cap is off unless given."""
    _, vk = world
    key = challenge_key(b"audit", vk)
    assert AUDIT_CAP == 1505 and PARAMS.call_ns == 241_500 and CAP_NS == 363_457_500
    flat = dataclasses.replace(vk, params=dense.DenseParams(16, 64, 2))
    for target in (vk, flat):
        v = ContinuousMemoryChallenger(target, key, 1, late_deadline_ns=LATE, audit_cap=AUDIT_CAP)
        assert v.timed.cap_ns == AUDIT_CAP * target.params.call_ns and v.ends_at is None
    assert UnsignedRounds(vk, key, audit_cap=1).cap_ns == PARAMS.call_ns
    for cap in (0, AUDIT_CAP + 1):
        with pytest.raises(ValueError, match="an audit's cap is from 1 to 1505 calls"):
            ContinuousMemoryChallenger(vk, key, 1, late_deadline_ns=LATE, audit_cap=cap)
    assert ContinuousMemoryChallenger(vk, key, 1, late_deadline_ns=LATE).timed.cap_ns is None
    assert UnsignedRounds(vk, key).cap_ns is None


def test_an_honest_audit_under_the_cap_accepts(world):
    """243 back-to-back challenges, the graded audit's longest row, each answered within Δ: accepted with the cap,
    which runs from the first send and leaves every challenge its full hard deadline."""
    blocks, vk = world
    clock = Clock()
    clock.now = 10**9
    v = _verifier(vk, 243, clock, audit_cap=AUDIT_CAP)
    result, wire = _run(v, clock, _answers(blocks, lambda j: DEADLINE_NS))
    assert v.accepted and result.accepted and not result.aborted and result.misses == 0
    assert v.ends_at == 10**9 + CAP_NS and clock.now - 10**9 == 243 * DEADLINE_NS < CAP_NS
    assert len(wire.sent) == 243


@pytest.mark.parametrize("over", [0, 1])
def test_the_cap_holds_from_the_first_send_to_the_last_receipt(world, over):
    """The last answer received exactly `1,505 · call_ns` after the first send is within the cap.  One nanosecond
    later the driver closes its challenge at the cap, rejected, and the audit is aborted, though the answer on its way
    is right and on time."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 2, clock, audit_cap=AUDIT_CAP)
    result, _ = _run(v, clock, _answers(blocks, lambda j: DEADLINE_NS),
                     wait=lambda j: setattr(clock, "now", j * (CAP_NS + over - DEADLINE_NS)))
    last = result.answers[-1]
    assert clock.now == CAP_NS + over and v.done and len(result.answers) == 2
    assert (last.correct, last.on_time, last.rejected, v.aborted, v.accepted) == (
        (True, True, False, False, True) if over == 0 else (False, False, True, True, False))


def test_a_stalling_prover_is_rejected_not_retried(world):
    """Every answer right and within its hard deadline, but each one late: with `max_misses` allowing them all the
    stream accepts without the cap, and with it the cap closes challenge 181 before its hard deadline, rejected, and
    aborts the audit.  No other challenge is sent, and none can be."""
    blocks, vk = world
    n, slow = 243, _answers(blocks, lambda j: LATE)
    clock = Clock()
    free = _verifier(vk, n, clock, max_misses=n)
    assert _run(free, clock, slow)[0].misses == n and free.accepted and not free.aborted
    clock = Clock()
    v = _verifier(vk, n, clock, max_misses=n, audit_cap=AUDIT_CAP)
    result, wire = _run(v, clock, slow)
    assert len(wire.sent) == len(result.answers) == 182 and CAP_NS // LATE == 181
    assert all(a.correct and not a.on_time for a in result.answers[:-1])
    cut = result.answers[-1]
    assert cut.rejected and cut.elapsed_ns == CAP_NS + 1 - 181 * LATE < LATE and clock.now == CAP_NS + 1
    assert v.aborted and v.done and result.aborted and not v.accepted and not result.accepted and not v.exceeded
    with pytest.raises(RuntimeError, match="the audit is aborted: it lasted past its cap of 1505 calls"):
        v.challenge()
    with pytest.raises(RuntimeError, match="no challenge is open"):
        v.expire()


def test_the_gaps_between_challenges_count_toward_the_cap(world):
    """Answers on time, but the memory challenger's own gaps reach past the cap: the challenge sent past it closes at
    once, rejected, and the audit is aborted."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 3, clock, audit_cap=AUDIT_CAP)
    result, wire = _run(v, clock, _answers(blocks, lambda j: DEADLINE_NS),
                        wait=lambda j: setattr(clock, "now", j * 200_000_000))
    assert [(a.on_time, a.rejected, a.elapsed_ns) for a in result.answers] == [
        (True, False, DEADLINE_NS), (True, False, DEADLINE_NS), (False, True, 0)]
    assert v.aborted and not v.accepted and [t for _, _, t in wire.sent] == [0, 200_000_000, 400_000_000]


def test_an_answer_past_the_cap_is_rejected_unchecked(world, monkeypatch):
    """Within its hard deadline but received past the cap: rejected without its label checked, the audit aborted; and
    `expire()` closes a challenge at the cap even with no hard deadline."""
    blocks, vk = world
    clock = Clock()
    v = _verifier(vk, 3, clock, audit_cap=10)
    j, i = v.challenge()
    clock.now = DEADLINE_NS
    v.respond(j, blocks[i])
    j, i = v.challenge()
    assert v.closes_at == v.ends_at == 10 * PARAMS.call_ns < DEADLINE_NS + LATE
    clock.now = v.ends_at + 1
    with monkeypatch.context() as m:
        m.setattr(VerifierKey, "check", lambda *_: pytest.fail("an answer past the cap was checked"))
        a = v.respond(j, blocks[i])
    assert a.rejected and a.elapsed_ns < LATE and v.aborted and len(v.answers) == 2
    bare = UnsignedRounds(vk, challenge_key(b"audit", vk), k=2, clock=clock, audit_cap=10)
    clock.now = 0
    bare.challenge()
    clock.now = bare.ends_at
    with pytest.raises(RuntimeError, match="within its hard deadline"):
        bare.expire()
    clock.now += 1
    assert bare.expire().rejected and bare.aborted and bare.done and not bare.accepted


def test_the_memory_challenger_caps_a_graded_audit(world):
    """`run_timed` over a `UnsignedRounds` with a hard deadline and the cap, the graded audit's memory challenger:
    answers late but within the hard deadline end at the cap, rejected."""
    blocks, vk = world
    clock = Clock()

    def respond(i):
        clock.now += LATE
        return blocks[i]

    v = UnsignedRounds(vk, challenge_key(b"audit", vk), k=243, clock=clock, late_deadline_ns=LATE, audit_cap=AUDIT_CAP)
    result = run_timed(v, respond)
    assert result.aborted and not v.accepted and len(result.answers) == 182 and result.answers[-1].rejected


def test_the_cap_composes_with_the_miss_count(world):
    """Wrong, late and silent answers count as misses under the cap as without it, and the challenge the cap closes is
    one more, rejected and late; `exceeded` and `aborted` are separate verdicts, and the stream accepts only with
    neither."""
    blocks, vk = world
    script = [(DEADLINE_NS, True), (DEADLINE_NS + 1, True), (0, False), (None, True), (DEADLINE_NS, True)]

    def stream(gap, cap):
        clock, seen = Clock(), []
        v = _verifier(vk, len(script), clock, max_misses=4, audit_cap=cap, on_answer=seen.append)

        def replies(j, i):
            latency, right = script[j]
            return [] if latency is None else [(latency, Reply(j, blocks[i] if right else _wrong(blocks, i)))]

        result, _ = _run(v, clock, replies, wait=lambda j: setattr(clock, "now", max(clock.now, j * gap)))
        assert seen == list(result.answers)
        return v, result

    v, r = stream(10**6, AUDIT_CAP)
    assert (r.misses, r.late, v.exceeded, v.aborted, v.accepted) == (3, 2, False, False, True)
    v, r = stream(CAP_NS // 4, AUDIT_CAP)
    assert [(a.correct, a.on_time, a.rejected) for a in r.answers] == [
        (True, True, False), (True, False, False), (False, True, False), (False, False, True), (False, False, True)]
    assert (r.misses, r.late, v.exceeded, v.aborted, v.accepted) == (4, 3, False, True, False)
    v, r = stream(CAP_NS // 4, None)
    assert (r.misses, r.late, v.exceeded, v.aborted, v.accepted) == (3, 2, False, False, True)
