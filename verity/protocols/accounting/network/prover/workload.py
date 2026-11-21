"""The network attester's messages between the workload and the prover: the committed rows the served-rows tie reads,
and the reported advice network attestation's prover computes from the workload's scheduling. The parties are SPOT's
(`spot.prover.workload`), declared again here because this place does not load SPOT's, and the one list of every
protocol's messages is `tests/test_workload_messages.py`'s.

~~~text
workload -> prover     CommittedRows, ReportedAdvice
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
    Schema("CommittedRows", 1, "private", body=BODY),
    Schema("ReportedAdvice", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(WORKLOAD, PROVER, ("CommittedRows", "ReportedAdvice")),
)
