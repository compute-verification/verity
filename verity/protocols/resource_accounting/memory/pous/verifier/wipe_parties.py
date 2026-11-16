"""The streamed wipe as two parties of the deployment layer (`verity.runtime`), the protocol `wipe_party` speaks as a byte
stream, in messages: the memory attester, whose party is `verity/protocols/resource_accounting/memory/pous/attester/wipe_challenger.py`'s,
and the prover, whose party is `verity/protocols/resource_accounting/memory/pous/prover/wipe_prover.py`'s. This module holds what both
speak.

~~~text
memory-attester -> prover   Fill {first, count} + labels (private) ... FillDone; then per batch Read {indices} (coins); End
prover -> memory-attester   FillAck once it holds the fill; per batch Answer, 64 bytes an index (private)
~~~

The run's parameters are `wipe_party.request_of`'s request, which names no reads: the challenger chooses them, never the
prover. A deployment starts both parties from its configuration (`verity.protocol.ProtocolDef.starts`), where the wipe is
`COMPONENT`, each with that request (`deployed_request`): its fill and tiers are the node's card memory and its deadline
the card's calibration, which no configuration states, so until those are read it raises
`NotImplementedError(challenger_session.ONSITE_GOAL)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.protocol import Channel, Manifest, Role
from verity.channel import Body, Int, List, Schema
from verity.protocols.resource_accounting.memory.pous.attester.challenger_session import ONSITE_GOAL

from . import streamed_wipe as sw
from . import wipe_party as WP

ID = "verity/pous/streamed-wipe"
VERSION = 1
CHALLENGER, PROVER = "memory-attester", "prover"
#: the wipe's name in a deployment configuration: in its `protocols`, a `[[pin]]`'s `protocol` and a binding's table
COMPONENT = "wipe"
MAX_BATCH = 1 << 10
MANIFEST = Manifest(
    ID, VERSION, "verity.protocols.resource_accounting.memory.pous.verifier.wipe_party:request_of",
    parties=(Role(CHALLENGER, "auditor", needs=("clock", "coins", "signing-key")),
             Role(PROVER, "developer", per="server", needs=("store",))),
    channels=(Channel(CHALLENGER, PROVER, ("Fill", "FillDone", "Read", "End"), carries_coins=True),
              Channel(PROVER, CHALLENGER, ("FillAck", "Answer"))),
    messages=(Schema("Fill", 1, "private", {"first": Int(), "count": Int(1, WP.CHUNK)},
                     Body(max=WP.CHUNK * sw.LABEL_BYTES, unit=sw.LABEL_BYTES)),
              Schema("FillDone", 1, "structure"), Schema("FillAck", 1, "structure"),
              Schema("Read", 1, "coins", {"indices": List(Int(), MAX_BATCH)}),
              Schema("Answer", 1, "private", body=Body(max=MAX_BATCH * sw.LABEL_BYTES, unit=sw.LABEL_BYTES)),
              Schema("End", 1, "structure")))


def deployed_request(configuration: Mapping[str, Any], bound: Mapping[str, Any]) -> dict[str, Any]:
    """The wipe's request (`wipe_party.request_of`) as a deployment starts it: `node` the prover's node from its deploy
    bindings (`bound`), `labels` and `tiers` the node's card memory from its data sheet, and `limit_ns` the card's
    calibration, which no configuration states."""
    raise NotImplementedError(ONSITE_GOAL)


__all__ = ["CHALLENGER", "COMPONENT", "ID", "MANIFEST", "PROVER", "VERSION", "deployed_request"]
