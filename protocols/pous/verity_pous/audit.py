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
"""The adversary's fastest wide call: the Feistel-SHAKE256 floor at 64 KB, 242 µs (`docs/p3-instantiation.md` §3)."""

CHALLENGES = 132
"""`k` for 1% at §2c's point (`γ = 0.0183`)."""

CERTIFIED_ROUNDS = 28
"""The most oracle rounds per answer the band certificate covers when a level costs 3 calls (an edge from the oldest
parent: `Π(c_{v-12})`, `Π(pad)`, `P`): `⌊(D − 1)/3⌋ + 1 ≤ 10` (`docs/p3-instantiation.md` §1, `band_twoWay_220_12`)."""

WORK = 1 << 14
"""`Q`: queries per answer (§2c)."""


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
    """One answer: its index, verdict, and either its oracle cost (`simulate`) or its latency (`TimedVerifier`)."""

    j: int
    index: int
    correct: bool
    on_time: bool
    cost: Cost | None = None
    elapsed_ns: int | None = None


@dataclass(frozen=True, slots=True)
class AuditResult:
    answers: tuple[Answer, ...]

    @property
    def accepted(self) -> bool:
        return all(a.correct and a.on_time for a in self.answers)

    @property
    def rounds(self) -> tuple[int, ...]:
        return tuple(a.cost.depth for a in self.answers if a.cost is not None)


def deadline_rounds(deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS,
                    call_ns: int = CALL_NS) -> int:
    """The oracle rounds an adversary finishes within the deadline."""
    if deadline_ns < 0 or rtt_allowance_ns < 0 or call_ns <= 0:
        raise ValueError("deadlines and allowances are nonnegative, call times positive")
    return (deadline_ns + rtt_allowance_ns) // call_ns


def simulate(vk: VerifierKey, responder: Responder, prims: Primitives, key: Key, k: int = CHALLENGES,
             deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS, call_ns: int = CALL_NS,
             rounds: int | None = None, work: int = WORK,
             on_answer: Callable[[Answer], None] | None = None) -> AuditResult:
    """The audit in the oracle model: rounds are the clock, `rounds` per answer if given, else
    `deadline_rounds(deadline_ns, rtt_allowance_ns, call_ns)`."""
    budget = deadline_rounds(deadline_ns, rtt_allowance_ns, call_ns) if rounds is None else rounds
    m = vk.params.m
    answers = []
    for j in range(k):
        i = key.uniform(vk.params.blocks, j)
        value, cost = run(responder.answer(j, i), prims)
        correct = type(value) is int and 0 <= value < 1 << m and vk.check(i, value.to_bytes((m + 7) // 8, "little"))
        a = Answer(j, i, correct, cost.depth <= budget and cost.work <= work, cost=cost)
        if on_answer is not None:
            on_answer(a)
        answers.append(a)
    return AuditResult(tuple(answers))


class TimedVerifier:
    """Verifier-driven sequential reveal with a strict wall-clock deadline per answer.

    `challenge()` reveals the next index and starts its clock; `respond(block)` stops it, is on time within
    `deadline_ns + rtt_allowance_ns`, and checks the block against
    `vk`'s leaf digests, or with `path` against its root when `root_only`.  A second challenge before the answer, or
    more than `k`, is refused.
    """

    def __init__(self, vk: VerifierKey, key: Key, k: int = CHALLENGES, deadline_ns: int = DEADLINE_NS,
                 rtt_allowance_ns: int = RTT_ALLOWANCE_NS, clock: Callable[[], int] = time.perf_counter_ns,
                 on_answer: Callable[[Answer], None] | None = None, root_only: bool = False) -> None:
        if deadline_ns < 0 or rtt_allowance_ns < 0:
            raise ValueError("deadlines and allowances are nonnegative")
        self.vk, self.key, self.k, self.root_only = vk, key, k, root_only
        self.deadline_ns, self.rtt_allowance_ns = deadline_ns, rtt_allowance_ns
        self.clock, self.on_answer = clock, on_answer
        self.answers: list[Answer] = []
        self._open: tuple[int, int] | None = None

    @property
    def limit_ns(self) -> int:
        """Δ plus the round-trip allowance."""
        return self.deadline_ns + self.rtt_allowance_ns

    @property
    def done(self) -> bool:
        return len(self.answers) == self.k

    @property
    def accepted(self) -> bool:
        return self.done and AuditResult(tuple(self.answers)).accepted

    def challenge(self) -> int:
        if self._open is not None:
            raise RuntimeError("the previous challenge is unanswered")
        if self.done:
            raise RuntimeError(f"all {self.k} challenges are answered")
        i = self.key.uniform(self.vk.params.blocks, len(self.answers))
        self._open = (i, self.clock())
        return i

    def respond(self, block: bytes, path: tuple[bytes, ...] | None = None) -> Answer:
        stop = self.clock()
        if self._open is None:
            raise RuntimeError("no challenge is open")
        i, start = self._open
        self._open = None
        elapsed = stop - start
        correct = self.vk.check_path(i, block, path or ()) if self.root_only else self.vk.check(i, block)
        a = Answer(len(self.answers), i, correct, elapsed <= self.limit_ns, elapsed_ns=elapsed)
        self.answers.append(a)
        if self.on_answer is not None:
            self.on_answer(a)
        return a

    def result(self) -> AuditResult:
        return AuditResult(tuple(self.answers))


def run_timed(verifier: TimedVerifier, respond: Callable[[int], bytes],
              path: Callable[[int], tuple[bytes, ...]] | None = None) -> AuditResult:
    """Drive `verifier` against `respond(index) -> block` (and `path(index)` for a root-only verifier) until all `k`
    are answered."""
    while not verifier.done:
        i = verifier.challenge()
        block = respond(i)
        verifier.respond(block, path(i) if path is not None else None)
    return verifier.result()
