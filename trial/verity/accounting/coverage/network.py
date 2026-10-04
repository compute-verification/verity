"""Network coverage: what left the site, as the warden on its links saw it, against what the stream explains: the
output bits of its executions, which leave as responses, where integrity says they are what the circuits computed. The
headroom is everything else that left, plus what the timing and the side channels could carry.

    warden      on every link out of the site: reports the bytes that left over the epoch, and the bits their timing
                could carry (its readings)
    verifier    reads integrity's profile and the warden's report, audits the report, and records the coverage

Stands in for `verity_network_warden`, whose warden shapes frames onto a grid of time slots and whose audit replays its
records against the logical schedule; here the audit only checks the timing channel against `max_timing_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.accounting.integrity.profile import IntegrityProfile
from verity.primitives.events import Actor, Send, gather, tell


@dataclass(frozen=True)
class Observed:
    bytes: int
    timing_bits: int


@dataclass(frozen=True)
class NetworkCoverage:
    """At most `bits` left the site over the epoch, of which the stream's outputs explain `covered`; at most `headroom`
    were anything else. Holds if the warden sees every link out and the side channels carry at most `side_bits`;
    except with integrity's probability `delta`."""

    stream: str
    bits: int
    covered: int
    headroom: int
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Network:
    max_timing_bits: int
    side_bits: int  # what the side channels the warden cannot see carry over an epoch: an assumption

    def verifier(self) -> Actor:
        got = yield from gather({IntegrityProfile: "verifier", Observed: "warden"})
        profile, observed = got[IntegrityProfile], got[Observed]
        reasons = (
            *(("integrity not established",) if not profile.accepted else ()),
            *(
                (f"a timing channel of {observed.timing_bits} bits, over {self.max_timing_bits}",)
                if observed.timing_bits > self.max_timing_bits
                else ()
            ),
        )
        bits = 8 * observed.bytes + observed.timing_bits + self.side_bits
        covered = 0 if reasons else min(profile.outputs.total - profile.outputs.wrong, 8 * observed.bytes)
        return (Send(NetworkCoverage(profile.stream, bits, covered, bits - covered, profile.delta, reasons)),)

    def warden(self, bytes_out: int, timing_bits: int) -> Actor:
        return tell(Send(Observed(bytes_out, timing_bits), "verifier"))
