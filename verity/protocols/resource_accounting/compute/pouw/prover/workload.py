"""PoUW's messages between the workload and the prover, H1's and Pearl+'s: what PoUW's prover asks of the workload, and
what the workload hands it. The parties are SPOT's (`spot.prover.workload`), declared again here because this place
does not load SPOT's, and the one list of every protocol's messages is `tests/test_workload_messages.py`'s.

~~~text
prover   -> workload   PoUWScheme, H1State, PoUWMonitoringPeriod
workload -> prover     H1RunCommitment, H1Program, H1Record, H1Registration, PearlPass, PoUWValues
~~~
"""

from __future__ import annotations

from verity.channel import Body, Schema
from verity.protocol import Channel, Role

__all__ = ["CHANNELS", "MESSAGES", "PARTIES"]

WORKLOAD = "workload"
PROVER = "prover"
PARTIES = (Role(WORKLOAD, "developer"), Role(PROVER, "developer"))
#: A body's bound: the wire's u32 length.
BODY = Body(max=(1 << 32) - 1)

MESSAGES = (
    Schema("PoUWScheme", 1, "structure", body=BODY),
    Schema("H1State", 1, "structure", body=BODY),
    Schema("PoUWMonitoringPeriod", 1, "coins", body=BODY),
    Schema("H1RunCommitment", 1, "commitment", body=BODY),
    Schema("H1Program", 1, "structure", body=BODY),
    Schema("H1Record", 1, "commitment", body=BODY),
    Schema("H1Registration", 1, "commitment", body=BODY),
    Schema("PearlPass", 1, "private", body=BODY),
    Schema("PoUWValues", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(PROVER, WORKLOAD, ("PoUWScheme", "H1State", "PoUWMonitoringPeriod")),
    Channel(WORKLOAD, PROVER, ("H1RunCommitment", "H1Program", "H1Record", "H1Registration", "PearlPass", "PoUWValues")),
)
