"""The timed audit with sequential reveal (Lean `seqAccepts`), strict: every answer correct and on time, and its
deadlines.

`simulate` draws `k` indices uniformly with replacement from `0..B-1` through `verity.ml.randomness`, keyed by the
memory challenger's own secret and bound to `vk`'s root (`challenge_key`), and reveals index `j` only once answer
`j - 1` is in.

The deadline is `Δ = deadline_ns` (0.5 ms: `docs/p3-gpu-measurements.md` found 300 µs fails about 1.5% of honest
132-answer audits on the host clock) plus `rtt_allowance_ns`, the memory challenger's network round trip and its jitter,
which it cannot tell apart from the responder's time and so also grants the adversary.

Two clocks.  The timed rounds on the memory challenger's own clock are `memory_challenger.MemoryChallenger`'s, which
signs them, and a stream of them is the continuous audit (`continuous`).  `simulate` is the model's: the responder
answers each index with a `Prog` (Lean `SeqResponder.step`, state kept inside the responder), `oracle.run` counts the
answer's rounds, and any answer over the round budget or `work` queries is late.  The budget is the deadline over `call_ns`,
the fastest whole oracle call, unless `deadline_rounds` gives it directly.  `on_answer` is its timing hook.  The
conversion of wall-clock time into calls holds in the ideal-permutation model only: at the reference Feistel-SHAKE256 `Π₂` a
prover answers without completing a call, rebuilding a label from 1.53% of `C` in 481 Keccak-f (the red team's state
attack, `primitives.pi2_keccak_f`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from verity.ml.randomness import Key, derive

from .oracle import Cost, Prog, pure, run
from .primitives import Primitives
from .scheme import Encoding
from .verifier import VerifierKey

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
    """Answers from `C`, with no queries."""

    encoding: Encoding

    def answer(self, j: int, i: int) -> Prog[int]:
        return pure(self.encoding.blocks[i])


@dataclass(frozen=True, slots=True)
class Answer:
    """One answer: its index, verdict, and either its oracle cost (`simulate`) or its latency on the memory challenger's
    clock (`continuous`), with the responder's own `report` (for example its device and server nanoseconds) beside it.
    `rejected` is an answer past the hard deadline, or none by then: its block is never checked, and it is neither
    correct nor on time."""

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

    answers: tuple[Answer, ...]
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

