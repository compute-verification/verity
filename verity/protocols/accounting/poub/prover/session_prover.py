"""PoUB's session (`period.session`) from the prover's side: its party of the deployment layer (`verity.runtime`), on
its node. The messages, the manifest and the verifier's party are `verity.protocols.accounting.poub.verifier.session_parties`'s.

The developer's machine gives it its store (`capabilities["store"]`, a `period.Peer`: `period_prover.Local` is the
reference); it finishes on `End`, refused `coins` when a challenge it was sent doesn't open against the coins disclosed
(`period.Peer.end`). `PROTOCOL` declares its one role.
"""

from __future__ import annotations

import json
from typing import Any

from verity.party import Finish, Identity, Received, Send
from verity.protocol import ProtocolDef
from verity.channel import Message, Phase, Refused
from verity.protocols.accounting.poub.verifier import coins as C
from verity.protocols.accounting.poub.verifier import period as P
from verity.protocols.accounting.poub.verifier import wire as W
from verity.protocols.accounting.poub.verifier.session_parties import DRAW, MANIFEST, PROVER, VERIFIER


class Prover:
    """The prover: its store (`capabilities["store"]`, a `period.Peer`) registered, held to the registration and the
    session's coin commitment it is sent, and read for each pass and each window's openings."""

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        self.store, self.passes, self.phase = identity.capabilities["store"], 0, Phase("Hold")
        blocks, root = self.store.register()
        return [Send(VERIFIER, Message("Register", {"blocks": blocks}, root))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish({"passes": self.passes}, m.fields["name"] if m.type == "Refused" else "closed")]
        try:
            self.phase.check(m)
        except Refused as e:
            return [Finish({"passes": self.passes}, e.name)]
        self.phase.expect("Run", "Open", "End")
        if m.type == "Hold":
            self.store.hold(P.registration_of(m.fields["registration"]))
            return []
        t = m.fields.get("t")
        if m.type == "Run":
            try:
                challenge = C.Released.from_json(m.fields["challenge"])
            except ValueError:
                return [Finish({"passes": self.passes}, "coins")]
            root = self.store.run(challenge, t, m.fields["i"], m.body)
            self.passes += 1
            return [Send(VERIFIER, Message("Root", {"t": t, "i": m.fields["i"]}, root))]
        if m.type == "Open":
            sampled = [DRAW.unpack_from(m.body, k) for k in range(0, len(m.body), DRAW.size)]
            return [Send(VERIFIER, Message("Openings", {"t": t}, W.encode_openings(self.store.open(t, sampled))))]
        if m.body:
            try:
                doc = json.loads(m.body)
                self.store.end(C.LiveCommitment.from_json(doc["commitment"]),
                               [C.LiveOpening.from_json(o) for o in doc["openings"]])
            except (ValueError, KeyError, TypeError):
                return [Finish({"passes": self.passes}, "coins")]
        return [Finish({"passes": self.passes})]


PROTOCOL = ProtocolDef(MANIFEST, {PROVER: Prover})


__all__ = ["PROTOCOL", "Prover"]
