"""The timed audit with sequential reveal (Lean `seqAccepts`), strict: every answer correct and on time.

The memory challenger draws `k` indices uniformly with replacement from `0..B-1` through `verity.ml.randomness`,
keyed by its own secret and bound to `vk`'s root, and reveals index `j` only once answer `j - 1` is in.

The deadline is `Δ = deadline_ns` (0.5 ms: `docs/p3-gpu-measurements.md` found 300 µs fails about 1.5% of honest
132-answer audits on the host clock) plus `rtt_allowance_ns`, the memory challenger's network round trip and its jitter,
which it cannot tell apart from the responder's time and so also grants the adversary.

Two clocks.  `UnsignedRounds` is the memory challenger's timed rounds, unsigned (`memory_challenger` signs them): it measures
each answer's wall-clock latency from the reveal and rejects any over the deadline.  `simulate` is the model's: the responder
answers each index with a `Prog` (Lean `SeqResponder.step`, state kept inside the responder), `oracle.run` counts the
answer's rounds, and any answer over the round budget or `work` queries is late.  The budget is the deadline over `call_ns`,
the fastest whole oracle call, unless `deadline_rounds` gives it directly.  `on_answer` is the timing hook for both.  The
conversion of wall-clock time into calls holds in the ideal-permutation model only: at the reference Feistel-SHAKE256 `Π₂` a
prover answers without completing a call, rebuilding a label from 1.53% of `C` in 481 Keccak-f (the red team's state
attack, `primitives.pi2_keccak_f`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from verity.ml.randomness import Key, derive

from .band import check_deployment
from .oracle import Cost, Prog, pure, run
from .primitives import Primitives
from .scheme import Encoding
from .verifier import MerkleTree, VerifierKey

CHALLENGE_DOMAIN = "verity/pous/audit-challenge/v1"

DEADLINE_NS = 500_000
"""Δ: the strict per-answer deadline, 0.5 ms (`docs/p3-gpu-measurements.md` §1)."""

RTT_ALLOWANCE_NS = 0
"""The default network round-trip allowance added to Δ: none, until the memory challenger's RTT is measured."""

CALL_NS = 242_000
"""The fastest whole oracle call: `P_T`'s Feistel-SHAKE256 floor at 64 KB, 242 µs; a `Π₂` call is at least 480 µs
(`docs/p3-instantiation.md` §3).  The ideal-permutation model's unit, not an adversary's floor at the reference
Feistel-SHAKE256, whose round functions resume from kept 200-byte states (`primitives.pi2_keccak_f`)."""

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
`D_late ≤ d`): at 241.5 µs per call the hard deadline needs `Δ_late + RTT < 13 × 241.5 µs ≈ 3.14` ms.  The wall-clock
reading holds in the ideal-permutation model only; at the reference Feistel-SHAKE256 `Π₂` a prover answers in about
24 µs from 1.53% of `C` (`primitives.pi2_keccak_f`)."""

AUDIT_CAP = 1505
"""The most call-times a graded audit may last, from its first challenge's send to its last answer's receipt
(`Pous.Rebuild.auditCap`, which `Pous.SecurityProofs.BackgroundRebuildExcluded` requires the memory challenger to
enforce): at 241.5 µs per call, `1,505 × 241.5 µs ≈ 363.5` ms.  Past it the memory challenger aborts the audit without
accepting, as for a missed hard deadline, and never retries it with fresh challenges.  What the cap excludes
(re-encoding a segment from `W`) is counted in whole `Π₂` calls, so it holds in the ideal-permutation model only: at the
reference Feistel-SHAKE256 `Π₂` a prover needs no re-encode, answering from 1.53% of `C` (`primitives.pi2_keccak_f`)."""


def _challenges(vk: VerifierKey) -> int:
    return getattr(vk.params, "challenges", CHALLENGES)


def challenge_key(source: bytes, vk: VerifierKey) -> Key:
    """The audit's key: `source` is the memory challenger's own secret, kept secret until the audit starts, never a
    public beacon or the prover's root. A public source makes every challenged index computable before the audit, so a
    prover holding little of `C` can rebuild only the challenged blocks within the deadline and pass."""
    return derive(source, CHALLENGE_DOMAIN, {"root": vk.root, "salt": vk.salt})


class Responder(Protocol):
    def answer(self, j: int, i: int) -> Prog[int]:
        """Round `j`'s answer to index `i`, block `i` as an `m`-bit label, as an oracle program."""
        ...


@dataclass(frozen=True, slots=True)
class HonestResponder:
    """Answers from `C`, with no queries; `path` gives a root-only memory challenger the audit path."""

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
class ChallengeReply:
    """One answer: its index, verdict, and either its oracle cost (`simulate`) or its latency (`UnsignedRounds`), with
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
    """The answers recorded, and whether the audit was `aborted` past its duration cap, which rejects it whatever its
    answers."""

    answers: tuple[ChallengeReply, ...]
    aborted: bool = False

    @property
    def accepted(self) -> bool:
        return not self.aborted and all(a.correct and a.on_time for a in self.answers)

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
             on_answer: Callable[[ChallengeReply], None] | None = None) -> AuditResult:
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
        a = ChallengeReply(j, i, correct, cost.depth <= budget and cost.work <= work, cost=cost)
        if on_answer is not None:
            on_answer(a)
        answers.append(a)
    return AuditResult(tuple(answers))


class UnsignedRounds:
    """The memory challenger's sequential reveal with a strict wall-clock deadline per answer, its rounds unsigned.

    For the band codec it refuses a deadline past the cliff (`band.check_deployment`: the effective `D` must not
    exceed `d`).  `challenge()` reveals the next index and starts its clock; `respond(block)` stops it, is on time within
    `deadline_ns + rtt_allowance_ns`, and checks the block against
    `vk`'s leaf digests, or with `path` against its root when `root_only`.  A second challenge before the answer, or
    more than `k`, is refused.

    `late_deadline_ns` (`Δ_late`, at least `Δ`) adds a hard deadline: it is refused when `Δ_late + RTT` fits more than
    `LATE_ROUNDS` calls of the parameters' `call_ns`, or more than the band's `d`.  An answer past `Δ_late + RTT` is
    rejected without checking its block.  Once that deadline has passed with no answer, `expire()` closes the challenge
    as rejected, and so does the next `challenge()` before it reveals another index.  Without it, the memory challenger
    waits for each answer however late it is.  The continuous audit runs on `continuous.ContinuousMemoryChallenger`,
    never on this class alone: `respond(block)` names no challenge, so a stale answer to an expired challenge would
    answer the next one.

    `audit_cap` (at most `AUDIT_CAP`) caps the whole audit's duration at `audit_cap` calls of `call_ns`, from the first
    challenge's send to the last answer's receipt (`ends_at`).  An answer received past it is rejected unchecked and
    aborts the audit, and so does a challenge closed past it by `expire()` or the next `challenge()`; one sent past it
    can be closed at once.  An aborted audit reveals no other index and never accepts, whatever its answers: it is
    rejected, never retried.
    """

    def __init__(self, vk: VerifierKey, key: Key, k: int | None = None, deadline_ns: int = DEADLINE_NS,
                 rtt_allowance_ns: int = RTT_ALLOWANCE_NS, clock: Callable[[], int] = time.perf_counter_ns,
                 on_answer: Callable[[ChallengeReply], None] | None = None, root_only: bool = False,
                 late_deadline_ns: int | None = None, audit_cap: int | None = None) -> None:
        if deadline_ns < 0 or rtt_allowance_ns < 0:
            raise ValueError("deadlines and allowances are nonnegative")
        check_deployment(vk.params, deadline_ns, rtt_allowance_ns)
        call_ns = getattr(vk.params, "call_ns", CALL_NS)
        if audit_cap is not None and not 1 <= audit_cap <= AUDIT_CAP:
            raise ValueError(f"an audit's cap is from 1 to {AUDIT_CAP} calls, not {audit_cap}")
        if late_deadline_ns is not None:
            if late_deadline_ns < deadline_ns:
                raise ValueError("the hard deadline is at least the deadline")
            late = deadline_rounds(late_deadline_ns, rtt_allowance_ns, call_ns)
            if late > LATE_ROUNDS:
                raise ValueError(f"Δ_late + RTT = {late_deadline_ns + rtt_allowance_ns} ns fits D_late = {late} "
                                 f"calls of {call_ns} ns, past the cap of {LATE_ROUNDS}")
            check_deployment(vk.params, late_deadline_ns, rtt_allowance_ns)
        self.vk, self.key, self.k, self.root_only = vk, key, _challenges(vk) if k is None else k, root_only
        self.deadline_ns, self.rtt_allowance_ns, self.late_deadline_ns = deadline_ns, rtt_allowance_ns, late_deadline_ns
        self.clock, self.on_answer, self.call_ns, self.audit_cap = clock, on_answer, call_ns, audit_cap
        self.answers: list[ChallengeReply] = []
        self.aborted = False
        self._open: tuple[int, int] | None = None
        self._first: int | None = None

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
        """The open challenge's send time on the memory challenger's clock, or None."""
        return None if self._open is None else self._open[1]

    @property
    def cap_ns(self) -> int | None:
        """`audit_cap` calls of `call_ns`, or None without a cap."""
        return None if self.audit_cap is None else self.audit_cap * self.call_ns

    @property
    def ends_at(self) -> int | None:
        """The first challenge's send time plus `cap_ns`: on the memory challenger's clock past it, the audit is
        aborted.  None without a cap, or before the first challenge."""
        return None if self.cap_ns is None or self._first is None else self._first + self.cap_ns

    @property
    def done(self) -> bool:
        """All `k` challenges are closed, or the audit is aborted."""
        return len(self.answers) == self.k or self.aborted

    @property
    def accepted(self) -> bool:
        return self.done and self.result().accepted

    def challenge(self) -> int:
        """Reveal the next index and start its clock.  A challenge open within its hard deadline and the audit's cap,
        or with neither, is refused.  One open past either is first closed as rejected, as by `expire()`, and that
        close stands even when the call then raises because it was the last of the `k` or aborted the audit.
        `continuous.run_continuous` closes such a challenge with `expire()` and calls this only with none open."""
        if self._open is not None and self._expire(self.clock()) is None:
            raise RuntimeError("the previous challenge is unanswered")
        if self.aborted:
            raise RuntimeError(f"the audit is aborted: it lasted past its cap of {self.audit_cap} calls")
        if self.done:
            raise RuntimeError(f"all {self.k} challenges are answered")
        i = self.key.uniform(self.vk.params.blocks, len(self.answers))
        self._open = (i, self.clock())
        if self._first is None:
            self._first = self._open[1]
        return i

    def respond(self, block: bytes, path: tuple[bytes, ...] | None = None, report: object = None) -> ChallengeReply:
        stop = self.clock()
        if self._open is None:
            raise RuntimeError("no challenge is open")
        i, start = self._open
        self._open = None
        elapsed = stop - start
        if self._past_cap(stop):
            self.aborted = True
        if self.aborted or (self.late_limit_ns is not None and elapsed > self.late_limit_ns):
            return self._record(ChallengeReply(len(self.answers), i, False, False, elapsed_ns=elapsed, report=report,
                                       rejected=True))
        correct = self.vk.check_path(i, block, path or ()) if self.root_only else self.vk.check(i, block)
        return self._record(ChallengeReply(len(self.answers), i, correct, elapsed <= self.limit_ns, elapsed_ns=elapsed,
                                   report=report))

    def expire(self) -> ChallengeReply:
        """Close the open challenge unanswered, once its hard deadline or the audit's cap has passed: rejected, and
        past the cap the audit is aborted."""
        if self._open is None:
            raise RuntimeError("no challenge is open")
        if self.late_limit_ns is None and self.cap_ns is None:
            raise RuntimeError("no hard deadline: the memory challenger waits for the answer")
        a = self._expire(self.clock())
        if a is None:
            cap = "" if self.cap_ns is None else " and the audit's cap"
            raise RuntimeError(f"the open challenge is within its hard deadline{cap}")
        return a

    def _past_cap(self, now: int) -> bool:
        return self.ends_at is not None and now > self.ends_at

    def _expire(self, now: int) -> ChallengeReply | None:
        i, start = self._open
        if self._past_cap(now):
            self.aborted = True
        elif self.late_limit_ns is None or now - start <= self.late_limit_ns:
            return None
        self._open = None
        return self._record(ChallengeReply(len(self.answers), i, False, False, elapsed_ns=now - start, rejected=True))

    def _record(self, a: ChallengeReply) -> ChallengeReply:
        self.answers.append(a)
        if self.on_answer is not None:
            self.on_answer(a)
        return a

    def result(self) -> AuditResult:
        return AuditResult(tuple(self.answers), self.aborted)


def run_timed(challenger: UnsignedRounds, respond: Callable[[int], bytes | tuple[bytes, object]],
              path: Callable[[int], tuple[bytes, ...]] | None = None) -> AuditResult:
    """Drive `challenger` against `respond(index)` until all `k` are answered.  `respond` returns the block, or
    `(block, report)` to keep the responder's own record in the `ChallengeReply`; `path(index)` serves a root-only
    memory challenger."""
    while not challenger.done:
        i = challenger.challenge()
        out = respond(i)
        block, report = out if isinstance(out, tuple) else (out, None)
        challenger.respond(block, path(i) if path is not None else None, report)
    return challenger.result()
