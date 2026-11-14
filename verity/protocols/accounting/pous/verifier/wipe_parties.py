"""The streamed wipe as two parties of the deployment layer (`verity.runtime`), the protocol `wipe_party` speaks as a byte
stream, in messages: the memory challenger, whose party is `verity/protocols/accounting/pous/device/wipe_challenger.py`'s,
and the prover, whose party is `verity/protocols/accounting/pous/prover/wipe_prover.py`'s. This module holds what both
speak.

~~~text
memory-challenger -> prover   Fill {first, count} + labels (private) ... FillDone; then per batch Read {indices} (coins); End
prover -> memory-challenger   FillAck once it holds the fill; per batch Answer, 64 bytes an index (private)
~~~

The run's parameters are `wipe_party.request_of`'s request, which names no reads: the challenger chooses them, never the
prover.
"""

from __future__ import annotations

from verity.protocol import Channel, Manifest, Role
from verity.channel import Body, Int, List, Schema

from . import streamed_wipe as sw
from . import wipe_party as WP

ID = "verity/pous/streamed-wipe"
VERSION = 1
CHALLENGER, PROVER = "memory-challenger", "prover"
MAX_BATCH = 1 << 10
MANIFEST = Manifest(
    ID, VERSION, "verity.protocols.accounting.pous.verifier.wipe_party:request_of",
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


__all__ = ["CHALLENGER", "ID", "MANIFEST", "PROVER", "VERSION"]
