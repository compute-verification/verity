"""Running principals' actors against each other over one log.

`run` starts every actor, then delivers the log's events in order, each to every actor of the principal it is sent to
(a record to its sender's own) but the one that sent it, appending whatever they send in reply, until nothing is left
to deliver. It yields one principal's view as the log grows: every event that principal sent or received. The
verifier's view is the output: what its actors establish are records in it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence

from verity.primitives.events import Actor, Event


def run(principals: Mapping[str, Sequence[Actor]], view: str) -> Iterator[Event]:
    live = {principal: list(actors) for principal, actors in principals.items()}
    log: deque[tuple[Event, Actor]] = deque()
    for principal, actors in principals.items():
        for actor in actors:
            log.extend(_advance(live, principal, actor, None))
    while log:
        event, origin = log.popleft()
        if view in (event.sender, event.to):
            yield event
        if event.to not in live:
            raise KeyError(
                f"{event.sender} sent {type(event.body).__name__} to {event.to}, who is not a principal here"
            )
        for actor in [a for a in live[event.to] if a is not origin]:
            log.extend(_advance(live, event.to, actor, event))


def _advance(
    live: dict[str, list[Actor]], principal: str, actor: Actor, event: Event | None
) -> list[tuple[Event, Actor]]:
    try:
        sends = actor.send(event)
    except StopIteration as stop:
        sends = stop.value or ()
        live[principal].remove(actor)
    return [(Event(principal, s.to or principal, s.body), actor) for s in sends]
