"""PoUS's messages with the workload: its own message types (`verity.channel.Schema`) and the two channels between the
workload and the prover (`verity.protocol.Channel`), which also name the messages sampled proofs declare (a message
that serves more than one protocol sits once, in sampled proofs' module of this name). The bandwidth audit reads the
same store, and its channels name these two. `tests/test_workload_messages.py` holds the list, and where the tree
writes and reads each message today.

~~~text
prover -> workload
Store      store these weights
Weights    sampled proofs': the weights' registration

workload -> prover
Stored     the store: its commitment, salt, root and segments, and what answers the memory challenger's rounds
Settings   sampled proofs': the workload's own statement of how it runs
~~~

This package imports no other protocol, so the two parties' names and the body's bound are stated here, and the test
holds them to sampled proofs' declaration.
"""

from __future__ import annotations

from verity.channel import Body, Schema
from verity.protocol import Channel

WORKLOAD, PROVER = "workload", "prover"
MAX_BODY = (1 << 32) - 1
MESSAGES = (
    Schema("Store", 1, "private", body=Body(max=MAX_BODY)),
    Schema("Stored", 1, "private", body=Body(max=MAX_BODY)),
)
CHANNELS = (
    Channel(PROVER, WORKLOAD, ("Store", "Weights")),
    Channel(WORKLOAD, PROVER, ("Stored", "Settings")),
)

__all__ = ["CHANNELS", "MAX_BODY", "MESSAGES", "PROVER", "WORKLOAD"]
