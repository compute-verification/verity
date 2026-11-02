"""The continuous audit's memory challenger: `n` challenges over time, each answered in turn, accepted when at most `max_misses`
of them miss.

Its model is the memory challenger of `Security/Definitions/Pous/Model/Continuous.lean`, with the
guarantee `Pous.SecurityProofs.ContinuousAudit`.  That memory challenger sends `n` uniform, independent indices and
counts a miss for every round that is not a hit: wrong, late or rejected.  Three requirements are theorems about it
(`Security/Proofs/Pous/Continuous/Verifier.lean`), and `ContinuousMemoryChallenger` enforces each one on a single
`audit.UnsignedRounds` over the whole stream, with its clock and its index draws:

1. **No pipelining.** Challenge `j + 1` is revealed only once answer `j` is in or its hard deadline has passed
   (`UnsignedRounds.challenge`), and each deadline runs from the memory challenger's own clock reading when it reveals
   the index, never from a time the responder reports.  An answer names the challenge it answers; one for any other
   challenge than the open one (an earlier, closed one, or one not yet revealed) is refused.
2. **A hard deadline.** An answer past `Δ_late + RTT`, or none by then, is rejected and its block is never checked;
   `run_continuous` closes a challenge with none by then with `expire()`.  A `Δ_late` past `D_late ≤ min(d, 12)` is
   refused.  That cap is the graded audit's; `ContinuousAudit` holds at every `D_late`, so for it the cap is only
   conservative.
3. **Every late answer counted.** Every answer past `Δ + RTT`, rejected or not, is a miss, and so is every wrong one
   (`AuditResult.misses`).  Every challenge closes with a `ChallengeReply` recorded, whether answered, rejected or
   expired.

A graded audit's window also needs a cap on its whole duration (`Pous.SecurityProofs.BackgroundRebuildExcluded`):
with `audit_cap=audit.AUDIT_CAP` (1,505 calls of `call_ns`, ≈ 363.5 ms at 241.5 µs), a receipt past it rejects the
stream, which then reveals no other index, never accepts and is never retried.  The cap is off by default, because
the continuous audit's guarantee does not need it and its arrivals may spread past it.

Every count of calls here (`D`, `D_late`, the cap) converts wall-clock time at `call_ns` per whole `Π₂` call, which
holds in the ideal-permutation model only.  `ContinuousAudit` is sound in that model, but at the reference
Feistel-SHAKE256 `Π₂` a prover holding 1.53% of `C` answers every challenge in 481 Keccak-f, about 24 µs, so the
storage conclusion fails there (the red team's state attack, `primitives.pi2_keccak_f`).

`n` is fixed before the stream (the realized number of arrivals, from the memory challenger's own randomness), so the
stream is never stopped or extended on misses, though it may reject early once they pass `max_misses` (`exceeded`).
When each challenge arrives is the caller's (`run_continuous`'s `wait`): the arrival times must be the memory
challenger's own randomness, independent of the setup, the indices and the answers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from verity.ml.randomness import Key

from .audit import DEADLINE_NS, RTT_ALLOWANCE_NS, AuditResult, ChallengeReply, UnsignedRounds
from .verifier import VerifierKey


@dataclass(frozen=True, slots=True)
class Reply:
    """An answer off the wire: the challenge `j` it names, its block, a root-only memory challenger's `path`, and the responder's
    own `report`."""

    j: int
    block: bytes
    path: tuple[bytes, ...] | None = None
    report: object = None


class ContinuousMemoryChallenger:
    """`n` challenges, each closed by its answer or its hard deadline before the next is revealed.

    `challenge()` returns `(j, index)`; `respond(j, block)` answers challenge `j` and refuses any other than the open
    one; `expire()` closes the open challenge as rejected once its hard deadline has passed (`overdue`).  The stream
    accepts when all `n` are closed with at most `max_misses` misses.

    With `audit_cap` (`audit.AUDIT_CAP` for a graded audit's window), it accepts only within `audit_cap` calls of
    `call_ns` from the first send to the last receipt.  Past them it is `aborted`: rejected whatever its misses,
    and never retried with fresh challenges.  `Pous.SecurityProofs.BackgroundRebuildExcluded` requires that cap of a
    graded audit's window.  A stream whose arrivals are spread out runs without it:
    `Pous.SecurityProofs.ContinuousAudit` measures the memory at each send time and needs no cap, and its model's
    memory challenger has none.

    The continuous audit runs only on this class, never on a bare `UnsignedRounds` with a hard deadline.  That one's
    `respond(block)` names no challenge, so once a challenge expires, an answer to it still in flight is credited to
    the next one, a hit when both ask for the same block.  That block was on the wire, not in the memory the model
    measures at the send time, so the audit would credit storage the responder does not hold (a red-team finding)."""

    def __init__(self, vk: VerifierKey, key: Key, n: int, *, late_deadline_ns: int, max_misses: int = 0,
                 deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS,
                 clock: Callable[[], int] = time.perf_counter_ns,
                 on_answer: Callable[[ChallengeReply], None] | None = None, root_only: bool = False,
                 audit_cap: int | None = None) -> None:
        if late_deadline_ns is None:
            raise ValueError("the continuous audit needs a hard deadline")
        if n < 1 or max_misses < 0:
            raise ValueError("a stream has at least one challenge, and a nonnegative miss allowance")
        self.timed = UnsignedRounds(vk, key, n, deadline_ns, rtt_allowance_ns, clock, on_answer, root_only,
                                   late_deadline_ns=late_deadline_ns, audit_cap=audit_cap)
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
    def answers(self) -> tuple[ChallengeReply, ...]:
        return tuple(self.timed.answers)

    @property
    def pending(self) -> int | None:
        """The open challenge's `j`, or None."""
        return self.timed.pending

    @property
    def ends_at(self) -> int | None:
        """The first challenge's send time plus `audit_cap` calls: on the memory challenger's clock past it, the stream
        is aborted.  None without a cap, or before the first challenge."""
        return self.timed.ends_at

    @property
    def closes_at(self) -> int | None:
        """The open challenge's send time plus `Δ_late + RTT`, or `ends_at` if sooner: on the memory challenger's clock
        past it, the challenge is rejected.  None with no challenge open."""
        sent = self.timed.sent_ns
        if sent is None:
            return None
        ends = self.ends_at
        return sent + self.late_limit_ns if ends is None else min(sent + self.late_limit_ns, ends)

    @property
    def overdue(self) -> bool:
        """The memory challenger's clock is past `closes_at`, so `expire()` closes the open challenge."""
        return self.closes_at is not None and self.timed.clock() > self.closes_at

    @property
    def misses(self) -> int:
        return self.result().misses

    @property
    def done(self) -> bool:
        """All `n` challenges are closed, or the stream is aborted."""
        return self.timed.done

    @property
    def aborted(self) -> bool:
        """The stream lasted past its cap: it rejects, and reveals no other index."""
        return self.timed.aborted

    @property
    def exceeded(self) -> bool:
        """The misses already pass `max_misses`, so the stream rejects whatever follows."""
        return self.misses > self.max_misses

    @property
    def accepted(self) -> bool:
        return self.done and not self.aborted and not self.exceeded

    def challenge(self) -> tuple[int, int]:
        """Reveal the next challenge, `(j, index)`, by `UnsignedRounds.challenge()`: an open challenge past its hard
        deadline is closed first, and stays closed when the call then raises because it was the last."""
        i = self.timed.challenge()
        return self.timed.pending, i

    def respond(self, j: int, block: bytes, path: tuple[bytes, ...] | None = None,
                report: object = None) -> ChallengeReply:
        if j != self.timed.pending:
            state = "closed" if j < len(self.timed.answers) else "not revealed"
            raise RuntimeError(f"challenge {j} is {state}: an answer counts only for the open challenge")
        return self.timed.respond(block, path, report)

    def expire(self) -> ChallengeReply:
        return self.timed.expire()

    def result(self) -> AuditResult:
        return self.timed.result()


def run_continuous(challenger: ContinuousMemoryChallenger, send: Callable[[int, int], None],
                   receive: Callable[[int], Reply | None], wait: Callable[[int], None] | None = None) -> AuditResult:
    """Drive `challenger` through all `n` challenges over a transport.  `wait(j)` returns at challenge `j`'s arrival
    time on the memory challenger's own schedule; without it each challenge is sent once the last one closes.
    `send(j, index)` sends challenge `j`.  `receive(until)` returns the next `Reply` in, or None once the memory
    challenger's clock is past `until`, the open challenge's `closes_at`, with none.

    A reply naming the open challenge answers it; any other is dropped, its own challenge closed and counted already
    or not yet revealed.  Once the hard deadline has passed the driver closes the open challenge with `expire()`, the
    last one too, so a silent or slow responder never holds the stream open.  A transport that returns None before
    then fails closed: `expire()` refuses, and the audit stops with the challenge open.  With an `audit_cap`,
    `closes_at` is never past `ends_at`, and the driver stops once a challenge closed past it aborts the stream."""
    while not challenger.done:
        if wait is not None:
            wait(len(challenger.answers))
        j, i = challenger.challenge()
        send(j, i)
        while challenger.pending == j:
            reply = receive(challenger.closes_at)
            if reply is not None and reply.j == j:
                challenger.respond(j, reply.block, reply.path, reply.report)
            elif reply is None or challenger.overdue:
                challenger.expire()
    return challenger.result()


__all__ = ["ContinuousMemoryChallenger", "Reply", "run_continuous"]
