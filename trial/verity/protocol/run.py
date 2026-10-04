"""Running the protocol: every party against every other, over one log.

A party's name places it in a trust domain: `verifier/integrity` and `verifier/proofs` are both the verifier, which the
auditor trusts; `prover/...` is the prover, which it does not. `run` starts each party, then delivers the log's events
in order, each to the parties it is addressed to, appending whatever they send in reply, until nothing is left to
deliver. It yields one domain's view as the log grows: every event a party in it sent or received. The verifier's view
is the protocol's output: its profiles are events in it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping

from verity.primitives.events import Event, Party


def run(parties: Mapping[str, Party], view: str) -> Iterator[Event]:
    live = dict(parties)
    log: deque[Event] = deque()
    for name in parties:
        log.extend(_advance(live, name, None))
    while log:
        event = log.popleft()
        if any(_within(name, view) for name in (event.sender, *event.to)):
            yield event
        for name in event.to:
            if name not in parties:
                raise KeyError(f"{event.sender} sent {type(event.body).__name__} to {name}, who is not a party")
            if name in live:
                log.extend(_advance(live, name, event))


def _within(name: str, domain: str) -> bool:
    return name == domain or name.startswith(domain + "/")


def _advance(live: dict[str, Party], name: str, event: Event | None) -> list[Event]:
    try:
        sends = live[name].send(event)
    except StopIteration as stop:
        sends = stop.value or ()
        del live[name]
    return [Event(name, s.to, s.body) for s in sends]
