"""The network certifier's messages between the workload and the prover: the committed rows the served-rows tie reads,
and the reported advice the certifier's prover computes from the workload's scheduling. The parties are SPOT's
(`spot.verifier.workload`), and the one list of every protocol's messages is `tests/test_workload_messages.py`'s.

~~~text
workload -> prover     CommittedRows, ReportedAdvice
~~~
"""

from __future__ import annotations

from verity.channel import Schema
from verity.protocol import Channel
from verity.protocols.proofs.spot.verifier.workload import BODY, PARTIES, PROVER, WORKLOAD

__all__ = ["CHANNELS", "MESSAGES", "PARTIES"]

MESSAGES = (
    Schema("CommittedRows", 1, "private", body=BODY),
    Schema("ReportedAdvice", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(WORKLOAD, PROVER, ("CommittedRows", "ReportedAdvice")),
)
