"""The network certifier's messages with the workload: none of its own. Each serves sampled proofs too, so sampled
proofs' module of this name declares it (`verity.protocols.proofs.spot.verifier.workload`), and the two channels here
between the workload and the prover (`verity.protocol.Channel`) name them. `tests/test_workload_messages.py` holds the
list, and where the tree writes and reads each message today.

~~~text
workload -> prover    Settings, Window, Io, Summary, Advice, Values
prover -> workload    Compiler
~~~
"""

from __future__ import annotations

from verity.protocol import Channel
from verity.protocols.proofs.spot.verifier.workload import PROVER, WORKLOAD

MESSAGES = ()
CHANNELS = (
    Channel(WORKLOAD, PROVER, ("Settings", "Window", "Io", "Summary", "Advice", "Values")),
    Channel(PROVER, WORKLOAD, ("Compiler",)),
)

__all__ = ["CHANNELS", "MESSAGES"]
