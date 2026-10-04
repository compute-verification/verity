"""Running principals' actors against each other over one log.

`run` starts every actor, then delivers the log's events in order: each into the mailbox of every actor of the
principal it is sent to (a record's, its sender's own) but the one that sent it. An actor is resumed with the earliest
event in its mailbox that it waits for, and whatever it sends is appended to the log, until nothing is left to deliver.
`run` yields one principal's view as the log grows: every event that principal sent or received. The verifier's view
is the output: what its actors establish are records in it.

An actor that raises (on a body no honest sender would send: a path left out, a nonce that is not an integer) ends
there, and its principal records that it `Raised`. What it would have established is missing, and whatever reads it
waits for it forever: a missing statement fails closed, so the actors need check only what could make them wrong, not
what could make them raise.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Event


@dataclass(frozen=True)
class Raised:
    actor: str  # its generator's name
    error: str


def run(principals: Mapping[str, Sequence[Actor]], view: str) -> Iterator[Event]:
    actors = [(principal, actor) for principal, group in principals.items() for actor in group]
    waiting: dict[Actor, Mapping[type, str]] = {}  # each live actor's kinds -> senders
    mailbox: dict[Actor, list[Event]] = {actor: [] for _, actor in actors}
    log: deque[tuple[Event, Actor]] = deque()
    for principal, actor in actors:
        log.extend(_resume(waiting, principal, actor, None))
    while log:
        event, origin = log.popleft()
        if view in (event.sender, event.to):
            yield event
        if event.to not in principals:
            raise KeyError(
                f"{event.sender} sent {type(event.body).__name__} to {event.to}, who is not a principal here"
            )
        for principal, actor in actors:
            if principal == event.to and actor is not origin and actor in waiting:
                mailbox[actor].append(event)
                log.extend(_deliver(waiting, principal, actor, mailbox[actor]))


def _deliver(
    waiting: dict[Actor, Mapping[type, str]], principal: str, actor: Actor, mailbox: list[Event]
) -> list[tuple[Event, Actor]]:
    """Resume `actor` with the earliest event in its mailbox it waits for, for as long as there is one."""
    out = []
    while actor in waiting:
        kinds = waiting[actor]
        i = next((i for i, e in enumerate(mailbox) if kinds.get(type(e.body)) == e.sender), None)
        if i is None:
            break
        out += _resume(waiting, principal, actor, mailbox.pop(i))
    return out


def _resume(
    waiting: dict[Actor, Mapping[type, str]], principal: str, actor: Actor, event: Event | None
) -> list[tuple[Event, Actor]]:
    try:
        wait = actor.send(event)
        waiting[actor], sends = wait.kinds, wait.sends
    except StopIteration as stop:
        waiting.pop(actor, None)
        sends = stop.value or ()
    except Exception as error:
        waiting.pop(actor, None)
        return [(Event(principal, principal, Raised(actor.__qualname__, f"{type(error).__name__}: {error}")), actor)]
    return [(Event(principal, s.to or principal, s.body), actor) for s in sends]
