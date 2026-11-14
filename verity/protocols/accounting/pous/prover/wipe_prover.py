"""The streamed wipe's prover as a party of the deployment layer (`verity.runtime`), on its node: `Prover`, over the
messages `verity.protocols.accounting.pous.verifier.wipe_parties` declares.

Its machine gives it `store` (`capabilities["store"]`, the prover a `WipeParams` makes: `streamed_wipe.Prover` unless
given, `wipe_store.StoreProver` on a node). It finishes with the batches it answered, refused by name when the
challenger leaves (`closed`) or sends a frame the wire refuses, and refused `schema` on parameters `wipe_party.params_of`
refuses. `PROTOCOL` declares its one role.
"""

from __future__ import annotations

from typing import Any

from verity.party import Finish, Identity, Received, Send
from verity.protocol import ProtocolDef
from verity.channel import Message, Phase, Refused
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.pous.verifier import wipe_party as WP
from verity.protocols.accounting.pous.verifier.wipe_parties import CHALLENGER, MANIFEST, PROVER


class Prover:
    """The prover: it keeps the fill in its store (`capabilities["store"]`) and answers each batch from it."""

    def start(self, params: dict[str, Any], identity: Identity) -> list[Any]:
        self.answered = 0
        try:
            self.store = identity.capabilities.get("store", sw.Prover)(WP.params_of(params))
        except ValueError:
            return [Finish({"answered": 0}, "schema")]
        self.phase = Phase("Fill", "FillDone")
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish({"answered": self.answered}, m.fields["name"] if m.type == "Refused" else "closed")]
        try:
            self.phase.check(m)
        except Refused as e:
            return [Finish({"answered": self.answered}, e.name)]
        if m.type == "Fill":
            self.store.receive(m.fields["first"], m.body)
            return []
        if m.type == "FillDone":
            self.phase.expect("Read")
            return [Send(CHALLENGER, Message("FillAck"))]
        if m.type == "Read":
            self.phase.expect("Read", "End")
            self.answered += 1
            return [Send(CHALLENGER, Message("Answer", body=self.store.answer(m.fields["indices"])))]
        return [Finish({"answered": self.answered})]


PROTOCOL = ProtocolDef(MANIFEST, {PROVER: Prover})


__all__ = ["PROTOCOL", "Prover"]
