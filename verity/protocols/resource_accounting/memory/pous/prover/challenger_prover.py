"""The prover's side of PoUS's memory-challenger session, on its node: `Prover`, a party of the deployment layer
(`verity.runtime`) over the messages `verity.protocols.resource_accounting.memory.pous.attester.challenger_session` declares, and its runs.

`answer(f, block_bytes, respond)` runs `Prover` over the connection `f` on the session's loop (`challenger_session.run`)
and `Frames`. `session(side, request, respond)` is one session with the challenger's side: `request` sent to the side
(`challenger_session.Side`), each index answered with `respond(index)`, and the side's report back. `PROTOCOL` declares
its one role and `start` how a deployment starts it; the challenger's is `challenger_party.PROTOCOL`'s.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

from verity.protocol import ProtocolDef, Start
from verity.party import Finish, Identity, Received, Send
from verity.channel import Message
from verity.protocols.resource_accounting.memory.pous.attester import memory_challenger as mc
from verity.protocols.resource_accounting.memory.pous.attester.challenger_session import (
    CHALLENGER,
    MANIFEST,
    ONSITE_GOAL,
    PROVER,
    Frames,
    Polled,
    Side,
    run,
)


class Prover:
    """The prover: each index answered with its whole block (`respond`, else its machine's `capabilities["store"]`)
    until the challenger ends the session; it finishes with the frames it answered, the warm-ups among them."""

    def __init__(self, respond: Callable[[int], bytes] | None = None) -> None:
        self.respond = respond

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.block_bytes, self.answered = params["block_bytes"], 0
        if self.respond is None:
            self.respond = identity.capabilities["store"]
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish(self.answered, m.fields["name"] if m.type == "Refused" else "closed")]
        if m.type == "End":
            return [Finish(self.answered)]
        block = self.respond(m.fields["index"])
        if len(block) != self.block_bytes:
            raise ValueError(f"an answer is the whole {self.block_bytes}-byte block, not {len(block)} bytes")
        self.answered += 1
        return [Send(CHALLENGER, Message("Block", body=block))]


def start(configuration: Mapping[str, Any], bound: Mapping[str, Any], state: Path | None) -> Start:
    """The prover's start on its node (`ProtocolDef.starts`): the session's request
    (`challenger_session.deployed_request`), and its store (`capabilities["store"]`), each index answered with its block
    of the served `C*` on the card (`verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store`)."""
    raise NotImplementedError(ONSITE_GOAL)


PROTOCOL = ProtocolDef(MANIFEST, {PROVER: Prover}, {PROVER: start})


def answer(f: BinaryIO, block_bytes: int, respond: Callable[[int], bytes]) -> int:
    """The prover's side over the connection `f`: `Prover`, each index answered with `respond(index)` until the
    challenger ends the session, the collector paused as on the challenger's side; the frames answered, the warm-ups
    among them."""
    with mc.collector_paused():
        done, _ = run(Prover(respond), {"block_bytes": block_bytes}, Identity(PROVER), f, Frames(), peer=CHALLENGER)
    return done.outcome


def session(side: Side, request: Mapping[str, Any], respond: Callable[[int], bytes]) -> dict[str, Any]:
    """One session with the challenger's side `side` starts: `request` sent, each index answered with `respond`, the
    side's report back, with `rounds`, the timed rounds the prover answered (the frames less the side's warm-ups)."""

    def drive(s: socket.socket) -> int:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return answer(Polled(s), request["block_bytes"], respond)

    out, rounds = side.talk(request, drive)
    return out | {"rounds": rounds - out.get("warmups", 0)}


__all__ = ["PROTOCOL", "Prover", "answer", "session", "start"]
