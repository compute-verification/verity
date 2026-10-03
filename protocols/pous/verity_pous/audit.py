"""The timed audit with sequential reveal (Lean `seqAccepts`), strict: every answer correct and on time.

The challenger draws `k` indices uniformly with replacement from `0..B-1` through `verity.randomness`, keyed by the
verifier's secret and bound to `vk`'s root, and reveals index `j` only once answer `j - 1` is in.

The deadline is `Δ = deadline_ns` (0.5 ms: `docs/p3-gpu-measurements.md` found 300 µs fails about 1.5% of honest
132-answer audits on the host clock) plus `rtt_allowance_ns`, the verifier's network round trip and its jitter, which
the verifier cannot tell apart from the responder's time and so also grants the adversary.

Two clocks.  `TimedVerifier` is the deployed verifier: it measures each answer's wall-clock latency from the reveal and
rejects any over the deadline.  `simulate` is the model's: the responder answers each index with a `Prog` (Lean
`SeqResponder.step`, state kept inside the responder), `oracle.run` counts the answer's rounds, and any answer over the
round budget or `work` queries is late.  The budget is the deadline over `call_ns`, the adversary's fastest oracle call,
unless `deadline_rounds` gives it directly.  `on_answer` is the timing hook for both.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from verity.randomness import Key, derive

from .band import check_deployment
from .oracle import Cost, Prog, pure, run
from .primitives import Primitives
from .scheme import Encoding
from .verifier import MerkleTree, VerifierKey

CHALLENGE_DOMAIN = "verity/pous/audit-challenge/v1"

DEADLINE_NS = 500_000
"""Δ: the strict per-answer deadline, 0.5 ms (`docs/p3-gpu-measurements.md` §1)."""

RTT_ALLOWANCE_NS = 0
"""The default network round-trip allowance added to Δ: none, until the verifier's RTT is measured."""

CALL_NS = 242_000
"""The adversary's fastest oracle call: `P_T`'s Feistel-SHAKE256 floor at 64 KB, 242 µs; a `Π₂` call is at least 480 µs
(`docs/p3-instantiation.md` §3)."""

CHALLENGES = 132
"""P3's `k` for 1% at §2c's point (`γ = 0.0183`); parameters with a `challenges` attribute (the dense scheme's 111)
override it."""

CERTIFIED_ROUNDS = 10
"""The most oracle rounds per answer that P3 is certified for at the operating point, counted in `P` and `Π₂` calls:
`D = 10` on `B(220, 12)` (`Pous.SecurityProofs.P3ConcreteMeets64`, `P3ConcreteMeets14GB`), and 12 on the fallback
`B(220, 16)` (`P3ConcreteMeets64D12`), for Lean's `h_0`, not yet the code's (`schemes.p3`).  The column-game certificate
`band_twoWay_220_12` covers 10 game levels, which the instantiation's rule of 3 calls per level reads as 28 rounds
(`docs/p3-instantiation.md` §1); no `Meets` statement covers that reading."""

WORK = 1 << 14
"""`Q`: queries per answer (§2c)."""

LATE_ROUNDS = 12
"""The most sequential calls a late answer may take, `D_late ≤ 12` (`Pous.SecurityProofs.BandGradedMeetsFamily`, with
`D_late ≤ d`): at 241.5 µs per call the hard deadline needs `Δ_late + RTT < 13 × 241.5 µs ≈ 3.14` ms."""


def _challenges(vk: VerifierKey) -> int:
    return getattr(vk.params, "challenges", CHALLENGES)


def challenge_key(source: bytes, vk: VerifierKey) -> Key:
    """The audit's key: `source` is the verifier's secret or a beacon round, never the prover's root."""
    return derive(source, CHALLENGE_DOMAIN, {"root": vk.root, "salt": vk.salt})


class Responder(Protocol):
    def answer(self, j: int, i: int) -> Prog[int]:
        """Round `j`'s answer to index `i`, block `i` as an `m`-bit label, as an oracle program."""
        ...


@dataclass(frozen=True, slots=True)
class HonestResponder:
    """Answers from `C`, with no queries; `path` gives a root-only verifier the audit path."""

    encoding: Encoding
    tree: MerkleTree | None = None

    def block(self, i: int) -> bytes:
        return self.encoding.block(i)

    def path(self, i: int) -> tuple[bytes, ...]:
        if self.tree is None:
            raise ValueError("build the responder with the encoding's tree to give paths")
        return self.tree.path(i)

    def answer(self, j: int, i: int) -> Prog[int]:
        return pure(self.encoding.blocks[i])


@dataclass(frozen=True, slots=True)
class Answer:
    """One answer: its index, verdict, and either its oracle cost (`simulate`) or its latency (`TimedVerifier`), with
    the responder's own `report` (for example its device and server nanoseconds) beside it.  `rejected` is an answer
    past the hard deadline, or none by then: its block is never checked, and it is neither correct nor on time."""

    j: int
    index: int
    correct: bool
    on_time: bool
    cost: Cost | None = None
    elapsed_ns: int | None = None
    report: object = None
    rejected: bool = False


@dataclass(frozen=True, slots=True)
class AuditResult:
    answers: tuple[Answer, ...]

    @property
    def accepted(self) -> bool:
        return all(a.correct and a.on_time for a in self.answers)

    @property
    def late(self) -> int:
        """The answers past `Δ + RTT`, the rejected ones included."""
        return sum(not a.on_time for a in self.answers)

    @property
    def misses(self) -> int:
        """The answers wrong or late: the continuous audit's miss count."""
        return sum(not (a.correct and a.on_time) for a in self.answers)

    @property
    def rounds(self) -> tuple[int, ...]:
        return tuple(a.cost.depth for a in self.answers if a.cost is not None)


def deadline_rounds(deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS,
                    call_ns: int = CALL_NS) -> int:
    """The oracle rounds an adversary finishes within the deadline."""
    if deadline_ns < 0 or rtt_allowance_ns < 0 or call_ns <= 0:
        raise ValueError("deadlines and allowances are nonnegative, call times positive")
    return (deadline_ns + rtt_allowance_ns) // call_ns


def simulate(vk: VerifierKey, responder: Responder, prims: Primitives, key: Key, k: int | None = None,
             deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS, call_ns: int | None = None,
             rounds: int | None = None, work: int = WORK,
             on_answer: Callable[[Answer], None] | None = None) -> AuditResult:
    """The audit in the oracle model: rounds are the clock, `rounds` per answer if given, else
    `deadline_rounds(deadline_ns, rtt_allowance_ns, call_ns)`.  `k` and `call_ns` default to the parameters' own
    (`challenges`, `call_ns`), else P3's."""
    k = _challenges(vk) if k is None else k
    call_ns = getattr(vk.params, "call_ns", CALL_NS) if call_ns is None else call_ns
    budget = deadline_rounds(deadline_ns, rtt_allowance_ns, call_ns) if rounds is None else rounds
    m, mb = vk.params.m, vk.params.block_bytes
    answers = []
    for j in range(k):
        i = key.uniform(vk.params.blocks, j)
        value, cost = run(responder.answer(j, i), prims)
        correct = type(value) is int and 0 <= value < 1 << m and vk.check(i, value.to_bytes(mb, "little"))
        a = Answer(j, i, correct, cost.depth <= budget and cost.work <= work, cost=cost)
        if on_answer is not None:
            on_answer(a)
        answers.append(a)
    return AuditResult(tuple(answers))


class TimedVerifier:
    """Verifier-driven sequential reveal with a strict wall-clock deadline per answer.

    For the band codec it refuses a deadline past the cliff (`band.check_deployment`: the effective `D` must not
    exceed `d`).  `challenge()` reveals the next index and starts its clock; `respond(block)` stops it, is on time within
    `deadline_ns + rtt_allowance_ns`, and checks the block against
    `vk`'s leaf digests, or with `path` against its root when `root_only`.  A second challenge before the answer, or
    more than `k`, is refused.

    `late_deadline_ns` (`Δ_late`, at least `Δ`) adds a hard deadline: it is refused when `Δ_late + RTT` fits more than
    `LATE_ROUNDS` calls of the parameters' `call_ns`, or more than the band's `d`.  An answer past `Δ_late + RTT` is
    rejected without checking its block.  Once that deadline has passed with no answer, `expire()` closes the challenge
    as rejected, and so does the next `challenge()` before it reveals another index.  Without it, the verifier waits for
    each answer however late it is.
    """

    def __init__(self, vk: VerifierKey, key: Key, k: int | None = None, deadline_ns: int = DEADLINE_NS,
                 rtt_allowance_ns: int = RTT_ALLOWANCE_NS, clock: Callable[[], int] = time.perf_counter_ns,
                 on_answer: Callable[[Answer], None] | None = None, root_only: bool = False,
                 late_deadline_ns: int | None = None) -> None:
        if deadline_ns < 0 or rtt_allowance_ns < 0:
            raise ValueError("deadlines and allowances are nonnegative")
        check_deployment(vk.params, deadline_ns, rtt_allowance_ns)
        if late_deadline_ns is not None:
            if late_deadline_ns < deadline_ns:
                raise ValueError("the hard deadline is at least the deadline")
            call_ns = getattr(vk.params, "call_ns", CALL_NS)
            late = deadline_rounds(late_deadline_ns, rtt_allowance_ns, call_ns)
            if late > LATE_ROUNDS:
                raise ValueError(f"Δ_late + RTT = {late_deadline_ns + rtt_allowance_ns} ns fits D_late = {late} "
                                 f"calls of {call_ns} ns, past the cap of {LATE_ROUNDS}")
            check_deployment(vk.params, late_deadline_ns, rtt_allowance_ns)
        self.vk, self.key, self.k, self.root_only = vk, key, _challenges(vk) if k is None else k, root_only
        self.deadline_ns, self.rtt_allowance_ns, self.late_deadline_ns = deadline_ns, rtt_allowance_ns, late_deadline_ns
        self.clock, self.on_answer = clock, on_answer
        self.answers: list[Answer] = []
        self._open: tuple[int, int] | None = None

    @property
    def limit_ns(self) -> int:
        """Δ plus the round-trip allowance."""
        return self.deadline_ns + self.rtt_allowance_ns

    @property
    def late_limit_ns(self) -> int | None:
        """Δ_late plus the round-trip allowance, or None without a hard deadline."""
        return None if self.late_deadline_ns is None else self.late_deadline_ns + self.rtt_allowance_ns

    @property
    def pending(self) -> int | None:
        """The open challenge's `j`, or None."""
        return None if self._open is None else len(self.answers)

    @property
    def sent_ns(self) -> int | None:
        """The open challenge's send time on the verifier's clock, or None."""
        return None if self._open is None else self._open[1]

    @property
    def done(self) -> bool:
        return len(self.answers) == self.k

    @property
    def accepted(self) -> bool:
        return self.done and AuditResult(tuple(self.answers)).accepted

    def challenge(self) -> int:
        """Reveal the next index and start its clock.  A challenge open within its hard deadline, or with none, is
        refused.  One open past it is first closed as rejected, as by `expire()`, and that close stands even when the
        call then raises because it was the last of the `k`.  `continuous.run_continuous` closes such a challenge with
        `expire()` and calls this only with none open."""
        if self._open is not None and self._expire(self.clock()) is None:
            raise RuntimeError("the previous challenge is unanswered")
        if self.done:
            raise RuntimeError(f"all {self.k} challenges are answered")
        i = self.key.uniform(self.vk.params.blocks, len(self.answers))
        self._open = (i, self.clock())
        return i

    def respond(self, block: bytes, path: tuple[bytes, ...] | None = None, report: object = None) -> Answer:
        stop = self.clock()
        if self._open is None:
            raise RuntimeError("no challenge is open")
        i, start = self._open
        self._open = None
        elapsed = stop - start
        if self.late_limit_ns is not None and elapsed > self.late_limit_ns:
            return self._record(Answer(len(self.answers), i, False, False, elapsed_ns=elapsed, report=report,
                                       rejected=True))
        correct = self.vk.check_path(i, block, path or ()) if self.root_only else self.vk.check(i, block)
        return self._record(Answer(len(self.answers), i, correct, elapsed <= self.limit_ns, elapsed_ns=elapsed,
                                   report=report))

    def expire(self) -> Answer:
        """Close the open challenge unanswered, once its hard deadline has passed: rejected."""
        if self._open is None:
            raise RuntimeError("no challenge is open")
        if self.late_limit_ns is None:
            raise RuntimeError("no hard deadline: the verifier waits for the answer")
        a = self._expire(self.clock())
        if a is None:
            raise RuntimeError("the open challenge is within its hard deadline")
        return a

    def _expire(self, now: int) -> Answer | None:
        i, start = self._open
        if self.late_limit_ns is None or now - start <= self.late_limit_ns:
            return None
        self._open = None
        return self._record(Answer(len(self.answers), i, False, False, elapsed_ns=now - start, rejected=True))

    def _record(self, a: Answer) -> Answer:
        self.answers.append(a)
        if self.on_answer is not None:
            self.on_answer(a)
        return a

    def result(self) -> AuditResult:
        return AuditResult(tuple(self.answers))


def run_timed(verifier: TimedVerifier, respond: Callable[[int], bytes | tuple[bytes, object]],
              path: Callable[[int], tuple[bytes, ...]] | None = None) -> AuditResult:
    """Drive `verifier` against `respond(index)` until all `k` are answered.  `respond` returns the block, or
    `(block, report)` to keep the responder's own record in the `Answer`; `path(index)` serves a root-only verifier."""
    while not verifier.done:
        i = verifier.challenge()
        out = respond(i)
        block, report = out if isinstance(out, tuple) else (out, None)
        verifier.respond(block, path(i) if path is not None else None, report)
    return verifier.result()
