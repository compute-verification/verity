"""Compute accounting: how much of the hardware's capacity the verified work accounts for, and so how much is left where
other work could hide.

    verifier    reads integrity's profile and PoUW's bound, over one registration, and a capacity reading from
                `capacity_from`; records the account
    capacity    the reading, run by `capacity_from`: the prover's declaration (remote, an assumption) or a pinger's
                measurement (onsite)
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, gather, tell
from verity.protocols.integrity.profile import IntegrityProfile
from verity.protocols.pouw import UsefulWork


@dataclass(frozen=True)
class Capacity:
    work: int  # what the hardware could have done over the period, in PoUW's units


@dataclass(frozen=True)
class ComputeAccount:
    """Of `capacity`, as `capacity_from` reported it, at least `verified` went to the registered work, which integrity
    checked; at most `unaccounted` went anywhere else. Except with probability `delta`."""

    registration: str
    capacity: int
    capacity_from: str
    verified: int
    unaccounted: int
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Compute:
    capacity_from: str  # the principal the verifier takes capacity from

    def verifier(self) -> Actor:
        got = yield from gather({IntegrityProfile: "verifier", UsefulWork: "verifier", Capacity: self.capacity_from})
        profile, work, capacity = got[IntegrityProfile], got[UsefulWork], got[Capacity].work
        reasons = (("integrity not established",) if not profile.accepted else ()) + (
            ("integrity and PoUW are over different registrations",)
            if profile.registration != work.registration
            else ()
        )
        verified = (
            0 if reasons or not profile.units else work.work * (profile.units - profile.wrong_units) // profile.units
        )
        delta = profile.delta + work.delta
        account = ComputeAccount(
            profile.registration, capacity, self.capacity_from, verified, max(capacity - verified, 0), delta, reasons
        )
        return (Send(account),)

    def capacity(self, work: int) -> Actor:
        return tell(Send(Capacity(work), "verifier"))
