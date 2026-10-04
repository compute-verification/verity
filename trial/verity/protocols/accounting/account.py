"""The account: each resource's line, side by side.

Accounting covers three resources: work (operations performed in a window), space (bits held at an instant) and
communication (bits that cross the site's boundary, through any channel, side channels too). Each has a module that
establishes its line, a cryptographic bound wrapped in the physical facts it rests on. Per resource, a line puts side by
side:
- capacity: what the site could have supplied, its devices times their facts;
- claimed: what the stream says went to the declared work;
- observed: what hardware at the site saw used, by anything (None where nothing looks);
- explained: the claim as far as verification backs it, a lower bound;
- headroom = min(capacity, observed) - explained: where anything else could have run.

    verifier    reads the recorded stream and the line of each resource it accounts for; records the account

The account holds except with the sum of its lines' probabilities: a union bound, which counts a statement two lines
read (integrity's) twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive
from verity.primitives.physical import Window
from verity.protocols.accounting.integrity import Stream


@dataclass(frozen=True)
class Line:
    """Of the `capacity` the site could have supplied of `resource` over `window`, the stream claimed `claimed`; at most
    `observed` was used (None where nothing at the site looks), at least `explained` went to the declared work, and at
    most `headroom` to anything else; except with probability `delta`."""

    resource: str
    window: Window
    capacity: int
    claimed: int
    observed: int | None
    explained: int
    headroom: int
    delta: float
    reasons: tuple[str, ...]


def line(
    resource: str,
    window: Window,
    capacity: int,
    claimed: int,
    observed: int | None,
    explained: int,
    reading: int,
    delta: float,
    reasons: tuple[str, ...],
) -> Line:
    """A resource's line. A `reading` (the bound or the observation it is explained by) over what the site could supply
    means the site or the census is wrong; and with any reason, the stream explains nothing."""
    if reading > capacity:
        reasons += (f"{reading} used of a capacity of {capacity}",)
    explained = 0 if reasons else explained
    used = capacity if observed is None else min(capacity, observed)
    return Line(resource, window, capacity, claimed, observed, explained, used - explained, delta, reasons)


@dataclass(frozen=True)
class Account:
    """The stream's lines, by resource; except with probability `delta`."""

    stream: str
    lines: Mapping[str, Line]
    delta: float


@dataclass(frozen=True)
class Accounting:
    resources: tuple[str, ...]  # those the product accounts for

    def verifier(self) -> Actor:
        stream = yield from receive(Stream, "verifier")
        lines: dict[str, Line] = {}
        while not set(self.resources) <= lines.keys():
            got = yield from receive(Line, "verifier")
            lines.setdefault(got.resource, got)
        lines = {r: lines[r] for r in self.resources}
        return (Send(Account(stream.digest(), lines, sum(each.delta for each in lines.values()))),)
