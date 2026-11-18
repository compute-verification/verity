"""PoUS's messages between the workload and the prover: none. Its Setup and the store the workload hands back are Python
calls inside the adapter's process (`verity_vllm.protocols.pous`), so no body crosses between two processes. The one
list of every protocol's messages is `tests/test_workload_messages.py`'s. The parties are SPOT's
(`spot.prover.workload`), declared again here because this place does not load SPOT's."""

from __future__ import annotations

from verity.channel import Schema
from verity.protocol import Channel, Role

WORKLOAD = "workload"
PROVER = "prover"
PARTIES = (Role(WORKLOAD, "developer"), Role(PROVER, "developer"))

MESSAGES: tuple[Schema, ...] = ()

CHANNELS: tuple[Channel, ...] = ()
