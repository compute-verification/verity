"""PoUW's messages with the workload: its own message types (`verity.channel.Schema`) and its channel from the workload
to the prover (`verity.protocol.Channel`), which also names a message sampled proofs declare
(`verity.protocols.proofs.spot.verifier.workload`, where a message that serves more than one protocol sits once).
`tests/test_workload_messages.py` holds the list, and where the tree writes and reads each message today.

~~~text
workload -> prover
Multiply        the workload's one call: run this matrix multiply under PoUW, over its operands
WindowProgram   the Program the workload declares for its execution phases' multiplies
Calls           a closed execution phase's record of its calls: their root, the session and the outcome
Window          sampled proofs': the closed execution phase

prover -> workload: none
~~~
"""

from __future__ import annotations

from verity.channel import Body, Schema
from verity.protocol import Channel
from verity.protocols.proofs.spot.verifier.workload import BLOCK, MAX_BODY, PROVER, WORKLOAD

MESSAGES = (
    Schema("Multiply", 1, "private", body=Body(max=MAX_BODY)),
    Schema("WindowProgram", 1, "structure", body=Body(max=MAX_BODY)),
    Schema("Calls", 1, "commitment", BLOCK, Body(max=MAX_BODY)),
)
CHANNELS = (Channel(WORKLOAD, PROVER, ("Multiply", "WindowProgram", "Calls", "Window")),)

__all__ = ["CHANNELS", "MESSAGES"]
