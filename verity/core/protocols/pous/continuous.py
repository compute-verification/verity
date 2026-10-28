"""The continuous audit's memory challenger: `n` challenges over time, each answered in turn, accepted when at most `max_misses`
of them miss.

Its model is the memory challenger of `Security/Definitions/Pous/Model/Continuous.lean`, with the
guarantee `Pous.SecurityProofs.ContinuousAudit`.  That memory challenger sends `n` uniform, independent indices and
counts a miss for every round that is not a hit: wrong, late or rejected.  Each challenge here is one round of
`memory_challenger.MemoryChallenger`, a session of its own under the auditor's id for it (`sessions[j]`), its index
drawn by that session's coin tree from the memory challenger's own secret, signed once the answer is in and judged by
`memory_challenger.verdict`: a stream goes on past a miss, where a session ends at its first late round.  A round is a
hit when its session's verdict accepts.  Three requirements are theorems about the model
(`Security/Proofs/Pous/Continuous/Verifier.lean`), and `ContinuousMemoryChallenger` enforces each one over the stream:

1. **No pipelining.** Challenge `j + 1` is revealed only once answer `j` is in or its hard deadline has passed, and
   each deadline runs from the memory challenger's own clock reading when it reveals the index (the session's send
   time), never from a time the responder reports.  An answer names the challenge it answers; one for any other
   challenge than the open one (an earlier, closed one, or one not yet revealed) is refused.
2. **A hard deadline.** An answer past `Δ_late + RTT`, or none by then, is rejected: its session is left unsigned and
   its block is never judged.  `run_continuous` closes a challenge with none by then with `expire()`.  A `Δ_late` past
   `D_late ≤ min(d, 12)` is refused.  That cap is the graded audit's; `ContinuousAudit` holds at every `D_late`, so for
   it the cap is only conservative.
3. **Every late answer counted.** Every answer past `Δ + RTT`, rejected or not, is a miss, and so is every wrong one
   (`AuditResult.misses`).  Every challenge closes with an `Answer` recorded, whether answered, rejected or expired.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from . import memory_challenger as mc
from .audit import AUDIT_CAP, CALL_NS, DEADLINE_NS, LATE_ROUNDS, RTT_ALLOWANCE_NS, Answer, AuditResult, deadline_rounds
from .band import check_deployment
from .codec import Geometry


@dataclass(frozen=True, slots=True)
class Reply:
    """An answer off the wire: the challenge `j` it names, its block, and the responder's own `report`."""

    j: int
    block: bytes
    report: object = None


class ContinuousMemoryChallenger:
    """`n = len(sessions)` challenges over the `C*` whose blocks' digests are `digests` (`memory_challenger.block_digest`
    of each, in label order) and whose setup root is `binding`, each closed by its answer or its hard deadline before the
    next is revealed.  Each is a one-round `memory_challenger.MemoryChallenger` session under `sessions[j]`, its coin
    drawn from `source` and its id recorded in `log` when it is revealed, and signed by `signer`; `trusted` (the
    signer's own key by default) and `schemes` (`memory_challenger.SCHEMES`) are what its verdict accepts, so a test
    signer's rounds all miss unless a test passes its own mapping.

    `challenge()` returns `(j, index)`; `respond(j, block)` answers challenge `j` and refuses any other than the open
    one; `expire()` closes the open challenge as rejected once its hard deadline has passed (`overdue`).  The stream
    accepts when all `n` are closed with at most `max_misses` misses.

    With `audit_cap` (`audit.AUDIT_CAP` for a graded audit's window), it accepts only within `audit_cap` calls of
    `call_ns` from the first send to the last receipt.  Past them it is `aborted`: rejected whatever its misses,
    and never retried with fresh challenges.  `Pous.SecurityProofs.BackgroundRebuildExcluded` requires that cap of a
    graded audit's window.  A stream whose arrivals are spread out runs without it:
    `Pous.SecurityProofs.ContinuousAudit` measures the memory at each send time and needs no cap, and its model's
    memory challenger has none.

    An answer names its challenge because a stale one must never count: once a challenge expires, an answer to it still
    in flight would otherwise be credited to the next one, a hit when both ask for the same block.  That block was on
    the wire, not in the memory the model measures at the send time, so the audit would credit storage the responder
    does not hold (a red-team finding)."""

    def __init__(self, params: Geometry, source: bytes, sessions: Sequence[bytes], signer: mc.Signer, *, log: mc.SessionLog,
                 digests: Sequence[bytes], binding: bytes, late_deadline_ns: int, max_misses: int = 0,
                 deadline_ns: int = DEADLINE_NS, rtt_allowance_ns: int = RTT_ALLOWANCE_NS,
                 clock: Callable[[], int] = time.perf_counter_ns, on_answer: Callable[[Answer], None] | None = None,
                 audit_cap: int | None = None, trusted: frozenset[bytes] | None = None,
                 schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> None:
        if late_deadline_ns is None:
            raise ValueError("the continuous audit needs a hard deadline")
        if not sessions or max_misses < 0:
            raise ValueError("a stream has at least one challenge, and a nonnegative miss allowance")
        if len(set(sessions)) != len(sessions) or any(len(s) != 32 for s in sessions):
            raise ValueError("each challenge has its own 32-byte session id")
        if deadline_ns < 0 or rtt_allowance_ns < 0:
            raise ValueError("deadlines and allowances are nonnegative")
        if len(digests) != params.blocks:
            raise ValueError(f"{len(digests)} block digests for a store of {params.blocks} blocks")
        call_ns = getattr(params, "call_ns", CALL_NS)
        if audit_cap is not None and not 1 <= audit_cap <= AUDIT_CAP:
            raise ValueError(f"an audit's cap is from 1 to {AUDIT_CAP} calls, not {audit_cap}")
        if late_deadline_ns < deadline_ns:
            raise ValueError("the hard deadline is at least the deadline")
        late = deadline_rounds(late_deadline_ns, rtt_allowance_ns, call_ns)
        if late > LATE_ROUNDS:
            raise ValueError(f"Δ_late + RTT = {late_deadline_ns + rtt_allowance_ns} ns fits D_late = {late} "
                             f"calls of {call_ns} ns, past the cap of {LATE_ROUNDS}")
        check_deployment(params, deadline_ns, rtt_allowance_ns)
        check_deployment(params, late_deadline_ns, rtt_allowance_ns)
        self.params, self.digests, self.signer, self.schemes = params, digests, signer, schemes
        self.trusted = frozenset({signer.public_key}) if trusted is None else trusted
        self.deadline_ns, self.rtt_allowance_ns, self.late_deadline_ns = deadline_ns, rtt_allowance_ns, late_deadline_ns
        self.clock, self.on_answer, self.call_ns, self.audit_cap, self.max_misses = clock, on_answer, call_ns, audit_cap, max_misses
        self._source, self._sessions, self._log, self._binding = source, tuple(sessions), log, binding
        self._resolution = time.clock_getres(time.CLOCK_MONOTONIC) * 1e9 if clock is time.perf_counter_ns else 1
        self._read_ns = 0
        self.rounds: list[mc.MemoryChallenger] = []
        """Each revealed challenge's session, drawn and recorded in `log` as it is revealed; it is signed (`signed`)
        once its answer is in within the hard deadline and the cap."""
        self._answers: list[Answer] = []
        self.aborted = False
        self._open: tuple[int, int, int] | None = None
        self._first: int | None = None

    def _read(self) -> int:
        """The clock as the sessions read it, the reading kept: a round's send and receipt are its session's own."""
        self._read_ns = self.clock()
        return self._read_ns

    @property
    def n(self) -> int:
        return len(self._sessions)

    @property
    def limit_ns(self) -> int:
        """Δ plus the round-trip allowance: an answer after it is late, a miss."""
        return self.deadline_ns + self.rtt_allowance_ns

    @property
    def late_limit_ns(self) -> int:
        """Δ_late plus the round-trip allowance: an answer after it is rejected."""
        return self.late_deadline_ns + self.rtt_allowance_ns

    @property
    def signed(self) -> tuple[mc.SignedSession, ...]:
        """The signed sessions, one for each challenge answered within its hard deadline and the cap, in order."""
        return tuple(r.signed for r in self.rounds if r.signed is not None)

    @property
    def answers(self) -> tuple[Answer, ...]:
        """The closed challenges' answers, in order."""
        return tuple(self._answers)

    @property
    def pending(self) -> int | None:
        """The open challenge's `j`, or None."""
        return None if self._open is None else self._open[0]

    @property
    def sent_ns(self) -> int | None:
        """The open challenge's send time on the memory challenger's clock, or None."""
        return None if self._open is None else self._open[2]

    @property
    def cap_ns(self) -> int | None:
        """`audit_cap` calls of `call_ns`, or None without a cap."""
        return None if self.audit_cap is None else self.audit_cap * self.call_ns

    @property
    def ends_at(self) -> int | None:
        """The first challenge's send time plus `cap_ns`: on the memory challenger's clock past it, the stream is
        aborted.  None without a cap, or before the first challenge."""
        return None if self.cap_ns is None or self._first is None else self._first + self.cap_ns

    @property
    def closes_at(self) -> int | None:
        """The open challenge's send time plus `Δ_late + RTT`, or `ends_at` if sooner: on the memory challenger's clock
        past it, the challenge is rejected.  None with no challenge open."""
        sent = self.sent_ns
        if sent is None:
            return None
        ends = self.ends_at
        return sent + self.late_limit_ns if ends is None else min(sent + self.late_limit_ns, ends)

    @property
    def overdue(self) -> bool:
        """The memory challenger's clock is past `closes_at`, so `expire()` closes the open challenge."""
        return self.closes_at is not None and self.clock() > self.closes_at

    @property
    def misses(self) -> int:
        return self.result().misses

    @property
    def done(self) -> bool:
        """All `n` challenges are closed, or the stream is aborted."""
        return len(self._answers) == self.n or self.aborted

    @property
    def exceeded(self) -> bool:
        """The misses already pass `max_misses`, so the stream rejects whatever follows."""
        return self.misses > self.max_misses

    @property
    def accepted(self) -> bool:
        return self.done and not self.aborted and not self.exceeded

    def challenge(self) -> tuple[int, int]:
        """Reveal the next challenge, `(j, index)`, its session's one round.  A challenge open within its hard deadline
        and the stream's cap is refused.  One open past either is first closed as rejected, as by `expire()`, and that
        close stands even when the call then raises because it was the last of the `n` or aborted the stream."""
        if self._open is not None and self._expire(self.clock()) is None:
            raise RuntimeError("the previous challenge is unanswered")
        if self.aborted:
            raise RuntimeError(f"the audit is aborted: it lasted past its cap of {self.audit_cap} calls")
        if self.done:
            raise RuntimeError(f"all {self.n} challenges are answered")
        j = len(self._answers)
        self.rounds.append(mc.MemoryChallenger(
            mc.CoinTree.draw(self._source, self._sessions[j], self.params.blocks, 1), self.signer, log=self._log,
            limit_ns=self.limit_ns, block_bytes=self.params.block_bytes, binding=self._binding, clock=self._read,
            resolution_ns=self._resolution))
        i = self.rounds[j].challenge()
        self._open = (j, i, self._read_ns)
        if self._first is None:
            self._first = self._read_ns
        return j, i

    def respond(self, j: int, block: bytes, report: object = None) -> Answer:
        """Challenge `j`'s answer, the whole block: its session reads the receipt time, then hashes it.  Within the hard
        deadline and the cap the session is signed and its verdict is the answer's (on time when it has no late round,
        correct when it has no wrong one and is otherwise accepted); past either it is rejected, never signed or
        judged."""
        if j != self.pending:
            state = "closed" if j < len(self._answers) else "not revealed"
            raise RuntimeError(f"challenge {j} is {state}: an answer counts only for the open challenge")
        session = self.rounds[j]
        r = session.receive(block)
        _, i, sent = self._open
        self._open = None
        if self._past_cap(r.receipt_ns):
            self.aborted = True
        if self.aborted or r.elapsed_ns > self.late_limit_ns:
            return self._record(Answer(j, i, False, False, elapsed_ns=r.elapsed_ns, report=report, rejected=True))
        signed = session.close()
        v = mc.verdict(signed, session.commitment, session.openings(), self._expected, trusted=self.trusted,
                       schemes=self.schemes)
        on_time = v.late == 0
        correct = v.wrong == 0 and (v.accepted or not on_time)
        return self._record(Answer(j, i, correct, on_time, elapsed_ns=r.elapsed_ns, report=report))

    def expire(self) -> Answer:
        """Close the open challenge unanswered, once its hard deadline or the stream's cap has passed: rejected, its
        session left unsigned, and past the cap the stream is aborted."""
        if self._open is None:
            raise RuntimeError("no challenge is open")
        a = self._expire(self.clock())
        if a is None:
            cap = "" if self.cap_ns is None else " and the audit's cap"
            raise RuntimeError(f"the open challenge is within its hard deadline{cap}")
        return a

    def result(self) -> AuditResult:
        return AuditResult(self.answers, self.aborted)

    def _expected(self, i: int) -> bytes:
        return self.digests[i]

    def _past_cap(self, now: int) -> bool:
        return self.ends_at is not None and now > self.ends_at

    def _expire(self, now: int) -> Answer | None:
        j, i, sent = self._open
        if self._past_cap(now):
            self.aborted = True
        elif now - sent <= self.late_limit_ns:
            return None
        self._open = None
        return self._record(Answer(j, i, False, False, elapsed_ns=now - sent, rejected=True))

    def _record(self, a: Answer) -> Answer:
        self._answers.append(a)
        if self.on_answer is not None:
            self.on_answer(a)
        return a


def run_continuous(challenger: ContinuousMemoryChallenger, send: Callable[[int, int], None],
                   receive: Callable[[int], Reply | None], wait: Callable[[int], None] | None = None) -> AuditResult:
    """Drive `challenger` through all `n` challenges over a transport.  `wait(j)` returns at challenge `j`'s arrival
    time on the memory challenger's own schedule; without it each challenge is sent once the last one closes.
    `send(j, index)` sends challenge `j`.  `receive(until)` returns the next `Reply` in, or None once the memory
    challenger's clock is past `until`, the open challenge's `closes_at`, with none.

    A reply naming the open challenge and holding a whole block answers it; any other is dropped: its own challenge
    closed and counted already or not yet revealed, or no answer at all.  Once the hard deadline has passed the driver
    closes the open challenge with `expire()`, the last one too, so a silent or slow responder never holds the stream
    open.  A transport that returns None before then fails closed: `expire()` refuses, and the audit stops with the
    challenge open.  With an `audit_cap`, `closes_at` is never past `ends_at`, and the driver stops once a challenge
    closed past it aborts the stream."""
    block_bytes = challenger.params.block_bytes
    while not challenger.done:
        if wait is not None:
            wait(len(challenger.answers))
        j, i = challenger.challenge()
        send(j, i)
        while challenger.pending == j:
            reply = receive(challenger.closes_at)
            if reply is not None and reply.j == j and len(reply.block) == block_bytes:
                challenger.respond(j, reply.block, reply.report)
            elif reply is None or challenger.overdue:
                challenger.expire()
    return challenger.result()


__all__ = ["ContinuousMemoryChallenger", "Reply", "run_continuous"]
