"""The bandwidth audit's messages with the workload: none of its own. It reads the store PoUS asks the workload for, so
its two channels between the workload and the prover (`verity.protocol.Channel`) name PoUS's messages
(`verity.protocols.accounting.pous.verifier.workload`). `tests/test_workload_messages.py` holds the list, and where the
tree writes and reads each message today.

~~~text
prover -> workload    Store, PoUS's: store these weights
workload -> prover    Stored, PoUS's: the store
~~~
"""

from __future__ import annotations

from verity.protocol import Channel
from verity.protocols.accounting.pous.verifier.workload import PROVER, WORKLOAD

MESSAGES = ()
CHANNELS = (Channel(PROVER, WORKLOAD, ("Store",)), Channel(WORKLOAD, PROVER, ("Stored",)))

__all__ = ["CHANNELS", "MESSAGES"]
