"""Reconcile: what the protocols established against what the site could have done, per resource.

Accounting covers three resources: work (operations performed in a window), space (bits held at an instant) and
communication (bits that cross the site's boundary, through any channel, side channels too). Each resource's protocol
records a statement that needs no physics: `work`'s tries made between the salt and the shares, `space`'s blocks held
if they were answered in time, `communication`'s audited report of what left. Reconciling wraps each in the physical
facts it rests on: the census's facts per device type, the site's devices, the clock's window, how long rebuilding a
block takes, what the side channels carry. Per resource, a `Usage` puts side by side:
- capacity: what the site could have supplied, its devices times their facts;
- claimed: what the stream says went to the declared work;
- observed: what hardware at the site saw used, by anything (None where nothing looks);
- explained: the claim as far as verification backs it, a lower bound;
- unexplained = min(capacity, observed) - explained: where anything else could have run.

    verifier    reads the stream and integrity's profile, the statement of each resource it reconciles, the window
                `clock_from`'s clock read and the site `site_from` reported; records the `Balance`

Who reads the clock and counts the site is the product's wiring: the prover's word is an assumption, a reading by the
verifier's hardware a measurement. The window is the epoch's, and the other readings must fall in it: space's challenge
inside it, and the warden watching all of it. Work explains the tries less the share of the gates in units integrity
could not rule out being bad; communication, the output bits integrity checked, which leave as responses, up to what
the warden saw leave. Every output counts as a response, so outputs that stay as state (a KV cache) explain as many
bits of anything else.

The balance holds except with the sum of the probabilities of the statements it reads, each counted once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, gather
from verity.primitives.physical import Device, Site, Window
from verity.protocols.accounting.communication import Left
from verity.protocols.accounting.integrity import IntegrityProfile, Stream
from verity.protocols.accounting.space.pous import Held
from verity.protocols.accounting.work.pouw import Worked

STATEMENTS = {"work": Worked, "space": Held, "communication": Left}


@dataclass(frozen=True)
class Usage:
    """Of the `capacity` the site could have supplied of a resource over `window`, the stream claimed `claimed`; at most
    `observed` was used (None where nothing at the site looks), at least `explained` went to the declared work, and at
    most `unexplained` to anything else."""

    window: Window
    capacity: int
    claimed: int
    observed: int | None
    explained: int
    unexplained: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Balance:
    """The stream's usage of each resource; except with probability `delta`."""

    stream: str
    usage: Mapping[str, Usage]
    delta: float

    def __getitem__(self, resource: str) -> Usage:
        return self.usage[resource]


@dataclass(frozen=True)
class Reconcile:
    resources: tuple[str, ...]  # those the product accounts for
    census: Mapping[str, Device]  # device type -> its facts
    deadline: float = 0.0  # seconds: quicker than any device at the site rebuilds a block of space's encoding
    side_bits: int = 0  # what the side channels carry over an epoch: an assumption

    def verifier(self, clock_from: str, site_from: str) -> Actor:
        trusted = {Stream: "verifier", IntegrityProfile: "verifier", Window: clock_from, Site: site_from}
        got = yield from gather({**trusted, **{STATEMENTS[r]: "verifier" for r in self.resources}})
        how = {"work": self._work, "space": self._space, "communication": self._communication}
        usage = {r: how[r](got) for r in self.resources}
        delta = got[IntegrityProfile].delta + sum(got[k].delta for k in (Worked, Held) if k in got)
        return (Send(Balance(got[Stream].digest(), usage, delta)),)

    def _work(self, got: Mapping[type, object]) -> Usage:
        worked, profile, window = got[Worked], got[IntegrityProfile], got[Window]
        peak, reasons = got[Site].supply(self.census, "work")
        gates = profile.gates
        explained = worked.tries * (gates.total - gates.wrong) // max(gates.total, 1)
        reasons += _integrity(profile)
        return _usage(window, int(window.seconds * peak), gates.total, None, explained, worked.tries, reasons)

    def _space(self, got: Mapping[type, object]) -> Usage:
        held, window = got[Held], got[Window]
        memory, reasons = got[Site].supply(self.census, "memory")
        reasons += held.reasons
        if held.window.seconds > self.deadline:
            reasons += (f"answered in {held.window.seconds}s, over the deadline",)
        if not window.opened <= held.window.opened <= held.window.closed <= window.closed:
            reasons += ("challenged outside the window",)
        return _usage(held.window, int(memory), held.encoded, None, held.held, held.held, reasons)

    def _communication(self, got: Mapping[type, object]) -> Usage:
        left, profile, window = got[Left], got[IntegrityProfile], got[Window]
        egress, reasons = got[Site].supply(self.census, "egress")
        reasons += left.reasons + _integrity(profile)
        if left.window.opened > window.opened or left.window.closed < window.closed:
            reasons += ("the warden did not watch the whole window",)
        seen, outputs = 8 * left.bytes, profile.outputs
        observed = seen + left.timing_bits + self.side_bits
        explained = min(outputs.total - outputs.wrong, seen)
        capacity = int(left.window.seconds * egress)
        return _usage(left.window, capacity, outputs.total, observed, explained, seen, reasons)


def _integrity(profile: IntegrityProfile) -> tuple[str, ...]:
    return () if profile.accepted else ("integrity not established",)


def _usage(
    window: Window,
    capacity: int,
    claimed: int,
    observed: int | None,
    explained: int,
    reading: int,
    reasons: tuple[str, ...],
) -> Usage:
    """A resource's usage. A `reading` (the bound or the observation it is explained by) over what the site could supply
    means the site or the census is wrong; and with any reason, the stream explains nothing."""
    if reading > capacity:
        reasons += (f"{reading} used of a capacity of {capacity}",)
    explained = 0 if reasons else explained
    used = capacity if observed is None else min(capacity, observed)
    return Usage(window, capacity, claimed, observed, explained, used - explained, reasons)
