"""PoUW's messages between the workload and the prover, H1's and Pearl+'s: what PoUW's prover asks of the workload, and
what the workload hands it. H1 has no scheme declaration, so its messages live here alone; Pearl+ declares its own of
them (`PEARL_PLUS.messages`). The parties are SPOT's (`spot.prover.workload`), declared again here because this place
does not load SPOT's, and the one list of every protocol's messages is `tests/test_workload_messages.py`'s.

~~~text
prover   -> workload   PoUWScheme, H1State, PoUWMonitoringPeriod
workload -> prover     H1RunCommitment, H1Program, H1Record, H1Registration, PearlPass
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
    Schema("H1State", 1, "structure", body=BODY),
    Schema("PoUWMonitoringPeriod", 1, "coins", body=BODY),
    Schema("H1RunCommitment", 1, "commitment", body=BODY),
    Schema("H1Program", 1, "structure", body=BODY),
    Schema("H1Record", 1, "commitment", body=BODY),
    Schema("H1Registration", 1, "commitment", body=BODY),
    Schema("PearlPass", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(PROVER, WORKLOAD, ("PoUWScheme", "H1State", "PoUWMonitoringPeriod")),
    Channel(WORKLOAD, PROVER, ("H1RunCommitment", "H1Program", "H1Record", "H1Registration", "PearlPass")),
)

#: Pearl+'s own (`PEARL_PLUS.channel_schemas`, ``parties`` and ``channels``), stated here because this place may load the
#: wire's types: the challenger's salt coin for salt epoch k, opened after its root (`salt.epoch_open_root`;
#: `coins.CoinOpening`), the parties beside the verifier, and the salt coin's one channel
PEARL_SALT = Schema("PearlSalt", 1, "coins", {"epoch": Int(0, (1 << 64) - 1), "opening": Json(1 << 12)})
PEARL_PARTIES = (Role("challenger", "auditor", needs=("host",)), Role("workers", "developer", needs=("serve",)))
PEARL_CHANNELS = (Channel("challenger", "workers", ("PearlSalt",), carries_coins=True),)
