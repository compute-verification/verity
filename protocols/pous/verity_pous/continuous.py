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
from dataclasses import dataclass

from verity.randomness import Key

from .audit import DEADLINE_NS, RTT_ALLOWANCE_NS, Answer, AuditResult, TimedVerifier
from .verifier import VerifierKey


@dataclass(frozen=True, slots=True)
class Reply:
    """An answer off the wire: the challenge `j` it names, its block, a root-only verifier's `path`, and the responder's
    own `report`."""

    j: int
    block: bytes
    path: tuple[bytes, ...] | None = None
    report: object = None


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
    def pending(self) -> int | None:
        """The open challenge's `j`, or None."""
        return self.timed.pending

    @property
    def closes_at(self) -> int | None:
        """The open challenge's send time plus `Δ_late + RTT`: on the verifier's clock past it, the challenge is
        rejected.  None with no challenge open."""
        sent = self.timed.sent_ns
        return None if sent is None else sent + self.late_limit_ns

    @property
    def overdue(self) -> bool:
        """The verifier's clock is past `closes_at`, so `expire()` closes the open challenge."""
        return self.closes_at is not None and self.timed.clock() > self.closes_at

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
        """Reveal the next challenge, `(j, index)`, by `TimedVerifier.challenge()`: an open challenge past its hard
        deadline is closed first, and stays closed when the call then raises because it was the last."""
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


def run_continuous(verifier: ContinuousVerifier, send: Callable[[int, int], None],
                   receive: Callable[[int], Reply | None], wait: Callable[[int], None] | None = None) -> AuditResult:
    """Drive `verifier` through all `n` challenges over a transport.  `wait(j)` returns at challenge `j`'s arrival time
    on the verifier's own schedule; without it each challenge is sent once the last one closes.  `send(j, index)` sends
    challenge `j`.  `receive(until)` returns the next `Reply` in, or None once the verifier's clock is past `until`, the
    open challenge's `closes_at`, with none.

    A reply naming the open challenge answers it; any other is dropped, its own challenge closed and counted already
    or not yet revealed.  Once the hard deadline has passed the driver closes the open challenge with `expire()`, the
    last one too, so a silent or slow responder never holds the stream open.  A transport that returns None before
    then fails closed: `expire()` refuses, and the audit stops with the challenge open."""
    while not verifier.done:
        if wait is not None:
            wait(len(verifier.answers))
        j, i = verifier.challenge()
        send(j, i)
        while verifier.pending == j:
            reply = receive(verifier.closes_at)
            if reply is not None and reply.j == j:
                verifier.respond(j, reply.block, reply.path, reply.report)
            elif reply is None or verifier.overdue:
                verifier.expire()
    return verifier.result()


__all__ = ["ContinuousVerifier", "Reply", "run_continuous"]
