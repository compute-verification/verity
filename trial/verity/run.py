"""Running actors against each other over one log.

An actor's name starts with its principal: `verifier/integrity` and `verifier/compute` are both the verifier, which the
auditor trusts; `prover/...` is the prover, which it does not. `run` starts each actor, then delivers the log's events
in order, each to the actors it is addressed to, appending whatever they send in reply, until nothing is left to
deliver. It yields one principal's view as the log grows: every event one of its actors sent or received. The
verifier's view is the output: what its actors establish are events in it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping

from verity.primitives.events import Actor, Event


def run(actors: Mapping[str, Actor], view: str) -> Iterator[Event]:
    live = dict(actors)
    log: deque[Event] = deque()
    for name in actors:
        log.extend(_advance(live, name, None))
    while log:
        event = log.popleft()
        if any(_of(name, view) for name in (event.sender, *event.to)):
            yield event
        for name in event.to:
            if name not in actors:
                raise KeyError(f"{event.sender} sent {type(event.body).__name__} to {name}, who is not an actor")
            if name in live:
                log.extend(_advance(live, name, event))


def _of(name: str, principal: str) -> bool:
    return name == principal or name.startswith(principal + "/")


def _advance(live: dict[str, Actor], name: str, event: Event | None) -> list[Event]:
    try:
        sends = live[name].send(event)
    except StopIteration as stop:
        sends = stop.value or ()
        del live[name]
    return [Event(name, s.to, s.body) for s in sends]
