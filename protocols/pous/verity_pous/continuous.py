"""The continuous audit's verifier (Lean `PousContinuous.ContinuousAudit`, a draft): `n` challenges over time, each
answered in turn, accepted when at most `max_misses` of them miss.

The game (`lean/Pous/SecurityProofs/Continuous/Game.lean`) draws `n` uniform, independent indices and counts a miss for
every answer that is wrong or late.  It holds only for a protocol that keeps three requirements, and
`ContinuousVerifier` enforces each one on a single `audit.TimedVerifier` over the whole stream, with its clock and its
index draws:

1. **No pipelining.** Challenge `j + 1` is revealed only once answer `j` is in or its hard deadline has passed
   (`TimedVerifier.challenge`), and each deadline runs from the verifier's own clock reading when it reveals the index,
   never from a time the responder reports.  An answer names the challenge it answers; one for any other challenge
   than the open one (an earlier, closed one, or one not yet revealed) is refused.
2. **A hard deadline.** An answer past `Δ_late + RTT`, or none by then, is rejected and its block is never checked.
   A `Δ_late` past `D_late ≤ min(d, 12)` is refused.
3. **Every late answer counted.** Every answer past `Δ + RTT`, rejected or not, is a miss, and so is every wrong one
   (`AuditResult.misses`).  Every challenge closes with an `Answer` recorded, whether answered, rejected or expired.

`n` is fixed before the stream (the realized number of arrivals, from the verifier's own randomness), so the stream is
never stopped or extended on misses, though it may reject early once they pass `max_misses` (`exceeded`).  When each
challenge arrives is the caller's (`run_continuous`'s `wait`): the arrival times must be the verifier's own randomness,
independent of the setup, the indices and the answers.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from verity.randomness import Key

from .audit import DEADLINE_NS, RTT_ALLOWANCE_NS, Answer, AuditResult, TimedVerifier
from .verifier import VerifierKey


class ContinuousVerifier:
    """`n` challenges, each closed by its answer or its hard deadline before the next is revealed.

    `challenge()` returns `(j, index)`; `respond(j, block)` answers challenge `j` and refuses any other than the open
    one; `expire()` closes the open challenge as rejected once its hard deadline has passed.  The stream accepts when
    all `n` are closed with at most `max_misses` misses."""

    def __init__(self, vk: VerifierKey, key: Key, n: int, *, late_deadline_ns: int, max_misses: int = 0,
                 deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS,
                 clock: Callable[[], int] = time.perf_counter_ns, on_answer: Callable[[Answer], None] | None = None,
                 root_only: bool = False) -> None:
        if late_deadline_ns is None:
            raise ValueError("the continuous audit needs a hard deadline")
        if n < 1 or max_misses < 0:
            raise ValueError("a stream has at least one challenge, and a nonnegative miss allowance")
        self.timed = TimedVerifier(vk, key, n, deadline_ns, rtt_allowance_ns, clock, on_answer, root_only,
                                   late_deadline_ns=late_deadline_ns)
        self.max_misses = max_misses

    @property
    def n(self) -> int:
        return self.timed.k

    @property
    def limit_ns(self) -> int:
        """Δ plus the round-trip allowance: an answer after it is late, a miss."""
        return self.timed.limit_ns

    @property
    def late_limit_ns(self) -> int:
        """Δ_late plus the round-trip allowance: an answer after it is rejected."""
        return self.timed.late_limit_ns

    @property
    def answers(self) -> tuple[Answer, ...]:
        return tuple(self.timed.answers)

    @property
    def misses(self) -> int:
        return self.result().misses

    @property
    def done(self) -> bool:
        return self.timed.done

    @property
    def exceeded(self) -> bool:
        """The misses already pass `max_misses`, so the stream rejects whatever follows."""
        return self.misses > self.max_misses

    @property
    def accepted(self) -> bool:
        return self.done and not self.exceeded

    def challenge(self) -> tuple[int, int]:
        i = self.timed.challenge()
        return self.timed.pending, i

    def respond(self, j: int, block: bytes, path: tuple[bytes, ...] | None = None, report: object = None) -> Answer:
        if j != self.timed.pending:
            state = "closed" if j < len(self.timed.answers) else "not revealed"
            raise RuntimeError(f"challenge {j} is {state}: an answer counts only for the open challenge")
        return self.timed.respond(block, path, report)

    def expire(self) -> Answer:
        return self.timed.expire()

    def result(self) -> AuditResult:
        return self.timed.result()


def run_continuous(verifier: ContinuousVerifier, respond: Callable[[int], bytes | tuple[bytes, object]],
                   path: Callable[[int], tuple[bytes, ...]] | None = None,
                   wait: Callable[[int], None] | None = None) -> AuditResult:
    """Drive `verifier` against `respond(index)` through all `n` challenges.  `wait(j)` returns at challenge `j`'s
    arrival time on the verifier's own schedule; without it the challenges arrive back to back.  `respond` returns the
    block, or `(block, report)`; `path(index)` serves a root-only verifier."""
    while not verifier.done:
        if wait is not None:
            wait(len(verifier.timed.answers))
        j, i = verifier.challenge()
        out = respond(i)
        block, report = out if isinstance(out, tuple) else (out, None)
        verifier.respond(j, block, path(i) if path is not None else None, report)
    return verifier.result()


__all__ = ["ContinuousVerifier", "run_continuous"]
