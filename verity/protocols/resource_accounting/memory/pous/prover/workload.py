"""PoUS's messages between the workload and the prover: its Setup, which asks the workload to store the weights, and the
store the workload hands back, which PoUB's bandwidth check reads too. The one list of every protocol's messages is
`tests/test_workload_messages.py`'s. The parties are SPOT's (`spot.prover.workload`), declared again here because
this place does not load SPOT's.

~~~text
prover   -> workload   PoUSSetup
workload -> prover     PoUSStore
~~~
"""

from __future__ import annotations

from verity.channel import Body, Schema
from verity.protocol import Channel, Role

WORKLOAD = "workload"
PROVER = "prover"
PARTIES = (Role(WORKLOAD, "developer"), Role(PROVER, "developer"))
#: A body's bound: the wire's u32 length.
BODY = Body(max=(1 << 32) - 1)

MESSAGES = (
    Schema("PoUSSetup", 1, "private", body=BODY),
    Schema("PoUSStore", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(PROVER, WORKLOAD, ("PoUSSetup",)),
    Channel(WORKLOAD, PROVER, ("PoUSStore",)),
)
