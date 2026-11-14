"""PoUB's session over PoUS's P2 store from the prover's side, on its node. The lines, the verifier's side and what
both read are `verity.protocols.accounting.poub.verifier.p2_parties`'s.

`Prover` is the prover's party (`PROTOCOL`): it sends the request (its params) and answers each message with its
`store` capability's (a `Peer`) `hold`, `run`, `open` and `end` until the answer, which it finishes with, and hands the
case to the store's `case`, if it has one. `prove(reader, writer, prover, request)` runs it over a channel and returns
the answer. It also reads a verifier that sends each challenge as the coin's whole opening and discloses no coins at
the end, the form before `Released`: the challenge is read as its released part (`released_of`), and `end` never comes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, BinaryIO, Protocol

from verity.channel import Message
from verity.party import Finish, Identity, Received, Send
from verity.protocol import ProtocolDef
from verity.protocols.accounting.poub.verifier import coins as C
from verity.protocols.accounting.poub.verifier import p2
from verity.protocols.accounting.poub.verifier import p2_parties as P2


class Peer(Protocol):
    """The prover's side as `Prover` drives it. One that also has `case(doc)` is handed the session's core case
    (`p2_case`), witness it keeps privately."""

    def hold(self, registration: Mapping[str, Any]) -> None:
        """The verifier's registration, before any coin: refuse a registration that isn't the store's."""

    def run(self, challenge: C.Released, t: int, i: int, previous: bytes) -> bytes:
        """Pass i of window t: one step's reads under the pass's seed, its receipts committed; the root."""

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        """Window t's sampled receipts `(pass, position)` opened."""

    def end(self, commitment: C.LiveCommitment, openings: Sequence[C.LiveOpening]) -> None:
        """The session's coin commitment and every released coin's opening, after its last root: refuse them unless
        every challenge this prover was sent opens against them (`coins.received_reason`)."""


def released_of(doc: Any) -> C.Released:
    """A run's challenge: a released coin, or the coin's whole opening (with its `path` and `salt`) read as its released
    part."""
    if isinstance(doc, dict) and "path" in doc and "salt" in doc:
        doc = {k: v for k, v in doc.items() if k not in ("path", "salt")}
    return C.Released.from_json(doc)


class Prover:
    """The prover on its node: its capabilities are its `store` (a `Peer`) and a progress `log`."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        c = identity.capabilities
        self.store: Peer = c["store"]
        self.log: Callable[[str], None] = c.get("log") or (lambda _m: None)
        return [Send(P2.VERIFIER, Message("Request", body=P2.line_of("request", dict(params))))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Closed", "Refused"):
            return [Finish(None, "closed")]
        (kind, v), = P2.message(m.body).items()
        out: list[Any] = []
        if kind == "hold":
            self.store.hold(v["registration"])
        elif kind == "run":
            root = self.store.run(released_of(v["challenge"]), v["t"], v["i"], bytes.fromhex(v["previous"]))
            out = [Send(P2.VERIFIER, Message("Root", body=P2.line_of("root", root.hex())))]
        elif kind == "open":
            opened = self.store.open(v["t"], [tuple(d) for d in v["draws"]])
            line = P2.line_of("openings", [P2.opening_json(o) for o in opened])
            out = [Send(P2.VERIFIER, Message("Openings", body=line))]
        elif kind == "end":
            self.store.end(C.LiveCommitment.from_json(v["commitment"]),
                           [C.LiveOpening.from_json(o) for o in v["openings"]])
        elif kind == "case":
            keep = getattr(self.store, "case", None)
            if keep is not None:
                keep(v)
        elif kind == "answer":
            return [Finish(v)]
        self.log(f"{kind} answered")
        return out


PROTOCOL = ProtocolDef(P2.MANIFEST, {P2.PROVER: Prover})


def prove(f_in: BinaryIO, f_out: BinaryIO, prover: Peer, request: Mapping[str, Any],
          log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """The prover's side of one session over the channel: `request` sent, then each message answered by `prover`
    until the verifier's answer, which it returns."""
    identity = Identity(P2.PROVER, capabilities={"store": prover, "log": log})
    return P2.drive(Prover(), dict(request), identity, f_in, f_out, peer=P2.VERIFIER).outcome


__all__ = ["PROTOCOL", "Peer", "Prover", "prove", "released_of"]
