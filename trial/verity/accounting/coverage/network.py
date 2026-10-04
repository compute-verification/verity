"""Network coverage: what left the site, as the warden on its links saw it, against what the stream explains: the
outputs of its executions, which leave as responses. The headroom is everything else that left, plus what the timing
and the side channels could carry.

    warden      on every link out of the site: reports the bytes that left over the epoch, and the bits their timing
                could carry (its readings)
    verifier    takes the registration and the stream, audits the warden's report, and records the coverage

Stands in for `verity_network_warden`, whose warden shapes frames onto a grid of time slots and whose audit replays its
records against the logical schedule; here the audit only checks the timing channel against `max_timing_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, collect, receive, tell
from verity.primitives.registration import (
    Closed,
    Execution,
    Registration,
    built,
    digest,
)


@dataclass(frozen=True)
class Observed:
    bytes: int
    timing_bits: int


@dataclass(frozen=True)
class NetworkCoverage:
    """At most `bits` left the site over the epoch, of which the stream's outputs explain `covered`; at most `headroom`
    were anything else. Holds if the warden sees every link out, and the side channels carry at most `side_bits`."""

    stream: str
    bits: int
    covered: int
    headroom: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Network:
    max_timing_bits: int
    side_bits: int  # what the side channels the warden cannot see carry over an epoch: an assumption

    def verifier(self) -> Actor:
        registration = yield from receive(Registration, "prover")
        executions = yield from collect(Execution, Closed, "prover")
        observed = yield from receive(Observed, "warden")
        reasons = (
            (f"a timing channel of {observed.timing_bits} bits, over {self.max_timing_bits}",)
            if observed.timing_bits > self.max_timing_bits
            else ()
        )
        bits = 8 * observed.bytes + observed.timing_bits + self.side_bits
        explained = sum(len(_outputs(registration, e)) for e in executions)
        covered = 0 if reasons else min(explained, 8 * observed.bytes)
        return (Send(NetworkCoverage(digest(registration, executions), bits, covered, bits - covered, reasons)),)

    def warden(self, bytes_out: int, timing_bits: int) -> Actor:
        return tell(Send(Observed(bytes_out, timing_bits), "verifier"))


def _outputs(registration: Registration, execution: Execution) -> tuple[int, ...]:
    """The execution's output bits; none if its advice does not build."""
    try:
        return built(registration.builder, execution.advice)[0].outputs
    except ValueError:
        return ()
