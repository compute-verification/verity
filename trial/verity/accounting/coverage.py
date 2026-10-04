"""Coverage: the account. Per resource it puts the other families side by side: the `capacity` the site could have
supplied, what the stream `claimed`, what was `observed` used (where something at the site looks), and what the stream
`explained`, its claim as far as verification backs it. What is left is the headroom, where anything else could have
run:

    headroom = min(capacity, observed) - explained

The stream explains compute by the work bound, less the share of the gates in units integrity could not rule out being
wrong: work counts for the registered computation only where integrity says that is what ran. It explains memory by the
space bound, and network by the output bits integrity checked, which leave as responses, up to what the warden saw
leave. Every output counts as a response, so outputs that stay as state (a KV cache) explain as many bits of anything
else.

    verifier    reads the site from `site_from`, and the statements its `resources` need; records the account

The account's window is the work's. It holds except with the probabilities of the statements it read, which add.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.accounting.capacity import Device, Site, capacity
from verity.accounting.explained.integrity.profile import IntegrityProfile
from verity.accounting.explained.space import SpaceBound
from verity.accounting.explained.work import Window, WorkBound
from verity.accounting.observed import Observed
from verity.primitives.events import Actor, Send, gather

NEEDS = {"compute": (IntegrityProfile,), "memory": (SpaceBound,), "network": (IntegrityProfile, Observed)}


@dataclass(frozen=True)
class Resource:
    """Of the `capacity` the site could have supplied, the stream claimed `claimed`; at most `observed` was used (None
    where nothing at the site looks), at least `explained` went to the registered work, and at most `headroom` to
    anything else."""

    capacity: int
    claimed: int
    observed: int | None
    explained: int
    headroom: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Account:
    """The site `site_from` reported, accounted per resource over the work's `window`; except with probability
    `delta`."""

    stream: str
    window: Window
    site_from: str
    resources: Mapping[str, Resource]
    delta: float


@dataclass(frozen=True)
class Coverage:
    census: Mapping[str, Device]  # device type -> its facts
    resources: tuple[str, ...]  # those the product accounts for

    def verifier(self, site_from: str) -> Actor:
        trusted = {kind: "verifier" for resource in self.resources for kind in NEEDS[resource]}
        got = yield from gather({**trusted, WorkBound: "verifier", Site: site_from})
        site, work, profile = got[Site], got[WorkBound], got.get(IntegrityProfile)
        counted = {device: n for device, n in site.devices.items() if isinstance(n, int) and n >= 0}
        caps = capacity(Site(counted), self.census, work.window.closed - work.window.opened)
        site_reasons = (
            *(f"{n!r} {device}s" for device, n in site.devices.items() if device not in counted),
            *(f"no census entry for {device}" for device in counted if device not in self.census),
        )
        checked = site_reasons
        if profile and not profile.accepted:
            checked += ("integrity not established",)
        if profile and profile.stream != work.stream:
            checked += ("integrity and work are over different streams",)
        resources = {}
        for name in self.resources:
            if name == "compute":
                gates = profile.gates
                explained = work.work * (gates.total - gates.wrong) // max(gates.total, 1)
                resources[name] = _resource(caps[name], gates.total, None, explained, checked, work.work)
            elif name == "memory":
                space = got[SpaceBound]
                reasons = site_reasons + (() if space.accepted else ("space not established",))
                held = int(space.bytes * space.rho)
                resources[name] = _resource(caps[name], space.bytes, None, held, reasons, held)
            else:
                seen, outputs = got[Observed], profile.outputs
                explained = min(outputs.total - outputs.wrong, seen.seen)
                observed = seen.seen + seen.unseen
                resources[name] = _resource(
                    caps[name], outputs.total, observed, explained, checked + seen.reasons, seen.seen
                )
        delta = work.delta + sum(getattr(got.get(kind), "delta", 0) for kind in (IntegrityProfile, SpaceBound))
        return (Send(Account(work.stream, work.window, site_from, resources, delta)),)


def _resource(
    capacity: int, claimed: int, observed: int | None, explained: int, reasons: tuple[str, ...], reading: int
) -> Resource:
    """A `reading` (the bound or the observation the resource is explained by) over what the site could supply means
    the site or the census is wrong; and with any reason, the stream explains nothing."""
    if reading > capacity:
        reasons += (f"{reading} used of a capacity of {capacity}",)
    explained = 0 if reasons else explained
    used = capacity if observed is None else min(capacity, observed)
    return Resource(capacity, claimed, observed, explained, used - explained, reasons)
