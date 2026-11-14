"""PoUB's session over PoUS's P2 store from the prover's side, on its node. The lines, the verifier's side and what
both read are `verity.protocols.accounting.poub.verifier.p2_parties`'s.

`prove(reader, writer, prover)` sends the request and answers each message with `prover`'s (a `Peer`) `hold`, `run`,
`open` and `end` until the answer, which it returns, and hands the case to its `case`, if it has one. It also reads a
verifier that sends each challenge as the coin's whole opening and discloses no coins at the end, the form before
`Released`: the challenge is read as its released part (`released_of`), and `end` never comes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, BinaryIO, Protocol

from verity.protocols.accounting.poub.verifier import coins as C
from verity.protocols.accounting.poub.verifier import p2
from verity.protocols.accounting.poub.verifier.p2_parties import opening_json, recv, send


class Peer(Protocol):
    """The prover's side as `prove` drives it. One that also has `case(doc)` is handed the session's core case
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


def prove(f_in: BinaryIO, f_out: BinaryIO, prover: Peer, request: Mapping[str, Any],
          log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """The prover's side of one session over the channel: `request` sent, then each message answered by `prover`
    until the verifier's answer, which it returns."""
    send(f_out, {"request": dict(request)})
    while True:
        (kind, m), = recv(f_in).items()
        if kind == "hold":
            prover.hold(m["registration"])
        elif kind == "run":
            root = prover.run(released_of(m["challenge"]), m["t"], m["i"], bytes.fromhex(m["previous"]))
            send(f_out, {"root": root.hex()})
        elif kind == "open":
            send(f_out, {"openings": [opening_json(o) for o in prover.open(m["t"], [tuple(d) for d in m["draws"]])]})
        elif kind == "end":
            prover.end(C.LiveCommitment.from_json(m["commitment"]), [C.LiveOpening.from_json(o) for o in m["openings"]])
        elif kind == "case":
            keep = getattr(prover, "case", None)
            if keep is not None:
                keep(m)
        elif kind == "answer":
            return m
        else:
            raise ValueError(f"no message {kind!r}")
        log(f"{kind} answered")


__all__ = ["Peer", "prove", "released_of"]
