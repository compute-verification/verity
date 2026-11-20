"""PoUW's messages between the workload and the prover: what PoUW's prover asks of the workload, and what the workload
hands it. Pearl+ declares its own of them (`PEARL_PLUS.messages`). The parties are SPOT's (`spot.prover.workload`),
declared again here because this place does not load SPOT's, and the one list of every protocol's messages is
`tests/test_workload_messages.py`'s.

~~~text
prover   -> workload   PoUWScheme, PoUWMonitoringPeriod
workload -> prover     PearlPass
~~~
"""

from __future__ import annotations

from verity.channel import Body, Int, Json, Schema
from verity.protocol import Channel, Role

__all__ = ["CHANNELS", "MESSAGES", "PARTIES", "PEARL_CHANNELS", "PEARL_PARTIES", "PEARL_SALT"]

WORKLOAD = "workload"
PROVER = "prover"
PARTIES = (Role(WORKLOAD, "developer"), Role(PROVER, "developer"))
#: A body's bound: the wire's u32 length.
BODY = Body(max=(1 << 32) - 1)

MESSAGES = (
    Schema("PoUWScheme", 1, "structure", body=BODY),
    Schema("PoUWMonitoringPeriod", 1, "coins", body=BODY),
    Schema("PearlPass", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(PROVER, WORKLOAD, ("PoUWScheme", "PoUWMonitoringPeriod")),
    Channel(WORKLOAD, PROVER, ("PearlPass",)),
)

#: Pearl+'s own (`PEARL_PLUS.channel_schemas`, ``parties`` and ``channels``), stated here because this place may load the
#: wire's types: the challenger's salt coin for salt epoch k, opened after its root (`salt.epoch_open_root`;
#: `coins.CoinOpening`), the parties beside the verifier, and the salt coin's one channel
PEARL_SALT = Schema("PearlSalt", 1, "coins", {"epoch": Int(0, (1 << 64) - 1), "opening": Json(1 << 12)})
PEARL_PARTIES = (Role("challenger", "auditor", needs=("host",)), Role("workers", "developer", needs=("serve",)))
PEARL_CHANNELS = (Channel("challenger", "workers", ("PearlSalt",), carries_coins=True),)
